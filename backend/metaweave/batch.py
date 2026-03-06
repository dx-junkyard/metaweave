"""Async batch evaluation for abstraction pattern matching.

新しい AbstractionPattern が登録されたとき、過去の論文群に対して
構造的同型性（Structural Isomorphism）を評価するバックグラウンドタスク。

処理フロー
----------
1. Qdrant でパターンのベクトルと類似度の高い過去の論文を上位 N 件取得
2. 候補論文の PaperStructure を MinIO からロード
3. LLM（Reasoning モデル）で構造的同型性を評価
4. 閾値以上の場合、Neo4j に MATCHES_PATTERN エッジを作成

Notes on Reasoning models
-------------------------
system ロールは使用不可。user ロールのみ。
temperature / max_tokens は指定しない。
"""

from __future__ import annotations

import json
import logging

from metaweave.db import get_driver
from metaweave.embedder import search_similar_papers
from metaweave.llm import get_client, get_settings
from metaweave.schema import AbstractionPattern, PaperStructure, PatternMatch

logger = logging.getLogger(__name__)

# 同型性評価の自信度スコア閾値
_CONFIDENCE_THRESHOLD = 0.6

# Qdrant から取得する候補論文数
_TOP_K = 5


def _build_pattern_query_text(pattern: AbstractionPattern) -> str:
    """パターンの検索クエリテキストを構築する。"""
    rules = "; ".join(pattern.structural_rules) if pattern.structural_rules else ""
    variables = ", ".join(pattern.variables_template) if pattern.variables_template else ""
    return (
        f"{pattern.name}. {pattern.description} "
        f"Variables: {variables}. Rules: {rules}"
    )


def _load_paper_structure(arxiv_id: str, storage_client) -> PaperStructure | None:
    """MinIO から PaperStructure を読み込む。見つからなければ None を返す。"""
    safe_id = arxiv_id.replace("/", "_")
    try:
        response = storage_client.get_object("extracted-structures", f"{safe_id}.json")
        data = response.read()
        response.close()
        response.release_conn()
        return PaperStructure.model_validate_json(data)
    except Exception:
        logger.warning("Could not load PaperStructure for %s", arxiv_id)
        return None


def _evaluate_isomorphism(
    pattern: AbstractionPattern,
    paper: PaperStructure,
) -> PatternMatch | None:
    """LLM を使ってパターンと論文の構造的同型性を評価する。

    Returns
    -------
    PatternMatch | None
        閾値以上の自信度の場合は PatternMatch を返す。閾値未満なら None。
    """
    client = get_client()
    settings = get_settings()

    prompt = (
        "あなたはメタ構造転写エンジンの同型性評価モジュールです。\n"
        "以下の「抽象化パターン」が、対象論文の構造と「構造的同型性（Structural Isomorphism）」\n"
        "を持つかどうかを評価してください。\n\n"
        "【評価基準】\n"
        "- パターンの変数（X, Y, Z等）が論文の具体的な概念に自然にマッピングできるか\n"
        "- パターンの構造ルール（変数間の関係）が論文の因果構造と一致するか\n"
        "- マッピングが論理的に整合しているか\n\n"
        f"--- 抽象化パターン ---\n{pattern.model_dump_json(indent=2)}\n\n"
        f"--- 対象論文の構造 ---\n{paper.model_dump_json(indent=2)}\n\n"
        "以下のJSONのみで回答してください:\n"
        "{\n"
        '  "is_isomorphic": true/false,\n'
        '  "confidence_score": 0.0〜1.0,\n'
        '  "mapping_explanation": "変数Xは論文の○○に対応し…"\n'
        "}"
    )

    try:
        resp = client.chat.completions.create(
            model=settings.analysis_model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = resp.choices[0].message.content or "{}"

        # JSON 抽出
        import re
        match = re.search(r"\{[\s\S]*\}", raw)
        if not match:
            logger.warning("Could not extract JSON from LLM response for pattern %s / paper %s",
                           pattern.pattern_id, paper.paper_id)
            return None

        result = json.loads(match.group())
        confidence = float(result.get("confidence_score", 0.0))

        if confidence < _CONFIDENCE_THRESHOLD:
            logger.info(
                "Pattern %s does not match paper %s (confidence=%.2f < %.2f)",
                pattern.pattern_id, paper.paper_id, confidence, _CONFIDENCE_THRESHOLD,
            )
            return None

        return PatternMatch(
            pattern_id=pattern.pattern_id,
            target_arxiv_id=paper.paper_id,
            mapping_explanation=result.get("mapping_explanation", ""),
            confidence_score=confidence,
        )

    except Exception:
        logger.exception(
            "Isomorphism evaluation failed for pattern %s / paper %s",
            pattern.pattern_id, paper.paper_id,
        )
        return None


def _save_match_to_neo4j(match: PatternMatch) -> None:
    """PatternMatch を Neo4j に保存する（MATCHES_PATTERN エッジ）。"""
    driver = get_driver()
    with driver.session() as session:
        session.run(
            """
            MERGE (p:Paper {arxiv_id: $target_arxiv_id})
            MERGE (ap:AbstractionPattern {pattern_id: $pattern_id})
            MERGE (p)-[r:MATCHES_PATTERN]->(ap)
            SET r.match_id = $match_id,
                r.mapping_explanation = $mapping_explanation,
                r.confidence_score = $confidence_score
            """,
            target_arxiv_id=match.target_arxiv_id,
            pattern_id=match.pattern_id,
            match_id=match.match_id,
            mapping_explanation=match.mapping_explanation,
            confidence_score=match.confidence_score,
        )
    logger.info(
        "Saved PatternMatch %s → %s (confidence=%.2f) to Neo4j",
        match.pattern_id, match.target_arxiv_id, match.confidence_score,
    )


def run_pattern_evaluation_task(
    pattern: AbstractionPattern,
    storage_client,
) -> list[PatternMatch]:
    """新しいパターンに対して、過去の論文群から構造的同型性を評価するバッチタスク。

    Parameters
    ----------
    pattern:
        評価対象の AbstractionPattern。
    storage_client:
        MinIO クライアント（PaperStructure のロードに使用）。

    Returns
    -------
    list[PatternMatch]
        閾値以上のマッチ結果のリスト。
    """
    logger.info("Starting pattern evaluation task for pattern_id=%s", pattern.pattern_id)

    client = get_client()
    settings = get_settings()

    # 1. パターンのテキスト表現で類似論文を検索
    query_text = _build_pattern_query_text(pattern)
    try:
        candidates = search_similar_papers(
            query_text=query_text,
            openai_client=client,
            embedding_model=settings.embedding_model,
            top_k=_TOP_K,
            exclude_arxiv_id=pattern.source_arxiv_id,
        )
    except Exception:
        logger.exception("Qdrant search failed for pattern %s", pattern.pattern_id)
        return []

    if not candidates:
        logger.info("No candidate papers found for pattern %s", pattern.pattern_id)
        return []

    logger.info(
        "Found %d candidate papers for pattern %s: %s",
        len(candidates),
        pattern.pattern_id,
        [c["arxiv_id"] for c in candidates],
    )

    # 2–4. 各候補論文に対して同型性を評価
    matches: list[PatternMatch] = []
    for candidate in candidates:
        arxiv_id = candidate["arxiv_id"]

        # 2. PaperStructure をロード
        paper = _load_paper_structure(arxiv_id, storage_client)
        if paper is None:
            continue

        # 3. LLM で同型性を評価
        match = _evaluate_isomorphism(pattern, paper)
        if match is None:
            continue

        # 4. Neo4j に保存
        try:
            _save_match_to_neo4j(match)
            matches.append(match)
        except Exception:
            logger.exception("Failed to save match to Neo4j for %s", arxiv_id)

    logger.info(
        "Pattern evaluation completed for pattern_id=%s: %d matches found",
        pattern.pattern_id, len(matches),
    )
    return matches
