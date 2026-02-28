"""Qdrant vector storage for MetaWeave paper chunk embeddings.

各論文のチャンクを text-embedding-3-large でベクトル化し、
Qdrant コレクション "papers" に arxiv_id メタデータ付きで保存する。

環境変数
--------
QDRANT_HOST   Qdrant サービスのホスト名 (default: localhost)
QDRANT_PORT   Qdrant REST API ポート   (default: 6333)
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

logger = logging.getLogger(__name__)

_COLLECTION = "papers"
_VECTOR_DIM = 3072  # text-embedding-3-large の次元数


@lru_cache(maxsize=1)
def _qdrant() -> QdrantClient:
    """Qdrant クライアントのシングルトンを返す。"""
    host = os.environ.get("QDRANT_HOST", "localhost")
    port = int(os.environ.get("QDRANT_PORT", "6333"))
    logger.info("Connecting to Qdrant at %s:%d", host, port)
    return QdrantClient(host=host, port=port)


def _ensure_collection() -> None:
    """コレクション "papers" が存在しない場合は作成する。"""
    client = _qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if _COLLECTION not in existing:
        client.create_collection(
            collection_name=_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Qdrant collection '%s' created (dim=%d)", _COLLECTION, _VECTOR_DIM)


def embed_and_store(
    chunks: list[str],
    arxiv_id: str,
    openai_client: OpenAI,
    embedding_model: str,
) -> None:
    """チャンクリストを一括 Embedding して Qdrant に upsert する。

    Parameters
    ----------
    chunks:
        論文テキストを分割したチャンクのリスト。
    arxiv_id:
        論文の arXiv ID（メタデータとして各ベクトルに付与）。
    openai_client:
        使用する OpenAI クライアント。
    embedding_model:
        Embedding モデル名（例: "text-embedding-3-large"）。
    """
    if not chunks:
        return

    _ensure_collection()

    # OpenAI Embeddings API は最大 2048 入力を一度に受け付けるが、
    # テキストが大きい場合は分割して呼ぶ（ここでは 100 件ずつ）
    _BATCH = 100
    all_embeddings: list[list[float]] = []
    for i in range(0, len(chunks), _BATCH):
        batch = chunks[i : i + _BATCH]
        resp = openai_client.embeddings.create(
            model=embedding_model,
            input=batch,
        )
        all_embeddings.extend([e.embedding for e in resp.data])

    # 決定論的な整数 ID を生成（arxiv_id + チャンクインデックスのハッシュ）
    safe_id = arxiv_id.replace("/", "_").replace(".", "_")
    points: list[PointStruct] = []
    for i, vector in enumerate(all_embeddings):
        point_id = abs(hash(f"{safe_id}_{i}")) % (2**53)
        points.append(
            PointStruct(
                id=point_id,
                vector=vector,
                payload={
                    "arxiv_id": arxiv_id,
                    "chunk_index": i,
                    # Qdrant ペイロードには先頭 500 文字のみ保存（コスト削減）
                    "text": chunks[i][:500],
                },
            )
        )

    _qdrant().upsert(collection_name=_COLLECTION, points=points)
    logger.info(
        "Stored %d chunk embeddings for arxiv_id=%s in Qdrant collection '%s'",
        len(points),
        arxiv_id,
        _COLLECTION,
    )
