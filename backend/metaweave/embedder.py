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
from qdrant_client.models import Distance, FieldCondition, Filter, MatchValue, PointStruct, VectorParams

logger = logging.getLogger(__name__)

_COLLECTION = "papers"
_PATTERNS_COLLECTION = "patterns"
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


# ---------------------------------------------------------------------------
# Patterns collection: store and search abstraction patterns
# ---------------------------------------------------------------------------

def _ensure_patterns_collection() -> None:
    """コレクション "patterns" が存在しない場合は作成する。"""
    client = _qdrant()
    existing = {c.name for c in client.get_collections().collections}
    if _PATTERNS_COLLECTION not in existing:
        client.create_collection(
            collection_name=_PATTERNS_COLLECTION,
            vectors_config=VectorParams(size=_VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info(
            "Qdrant collection '%s' created (dim=%d)", _PATTERNS_COLLECTION, _VECTOR_DIM
        )


def embed_and_store_pattern(
    pattern_id: str,
    pattern_text: str,
    openai_client: OpenAI,
    embedding_model: str,
) -> None:
    """パターンのテキスト表現を Embedding して Qdrant patterns コレクションに保存する。

    Parameters
    ----------
    pattern_id:
        AbstractionPattern の pattern_id。
    pattern_text:
        パターンの説明文（name + description + structural_rules を結合したもの）。
    openai_client:
        使用する OpenAI クライアント。
    embedding_model:
        Embedding モデル名。
    """
    _ensure_patterns_collection()

    resp = openai_client.embeddings.create(model=embedding_model, input=[pattern_text])
    vector = resp.data[0].embedding

    point_id = abs(hash(f"pattern_{pattern_id}")) % (2**53)
    _qdrant().upsert(
        collection_name=_PATTERNS_COLLECTION,
        points=[
            PointStruct(
                id=point_id,
                vector=vector,
                payload={"pattern_id": pattern_id, "text": pattern_text[:500]},
            )
        ],
    )
    logger.info("Stored pattern embedding for pattern_id=%s", pattern_id)


def search_similar_papers(
    query_text: str,
    openai_client: OpenAI,
    embedding_model: str,
    top_k: int = 5,
    exclude_arxiv_id: str | None = None,
) -> list[dict]:
    """パターン（またはクエリテキスト）に類似する過去の論文チャンクを Qdrant から検索する。

    Returns
    -------
    list[dict]
        各要素は ``{"arxiv_id": str, "score": float, "text": str}``。
        同一 arxiv_id の重複は排除し、最高スコアのみを残す。
    """
    _ensure_collection()

    resp = openai_client.embeddings.create(model=embedding_model, input=[query_text])
    vector = resp.data[0].embedding

    # Qdrant 検索（top_k * 3 で多めに取得して重複排除）
    results = _qdrant().search(
        collection_name=_COLLECTION,
        query_vector=vector,
        limit=top_k * 3,
    )

    seen: dict[str, dict] = {}
    for hit in results:
        arxiv_id = hit.payload.get("arxiv_id", "")
        if not arxiv_id:
            continue
        if exclude_arxiv_id and arxiv_id == exclude_arxiv_id:
            continue
        if arxiv_id not in seen or hit.score > seen[arxiv_id]["score"]:
            seen[arxiv_id] = {
                "arxiv_id": arxiv_id,
                "score": hit.score,
                "text": hit.payload.get("text", ""),
            }

    # スコア降順でソートし、上位 top_k 件を返す
    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)
    return ranked[:top_k]
