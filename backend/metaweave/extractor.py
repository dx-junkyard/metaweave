"""PDF text extraction and hypothesis-driven sequential LLM structure extraction.

Architecture (仮説検証型チャンク解析):
1. Chunk:      Split text by paragraph/section boundaries into ~4 000-char chunks.
2. Hypothesis: Send first chunk (Abstract etc.) to LLM → initial draft.
3. Refine:     Process each subsequent chunk, updating accumulated state.
4. Finalize:   Produce final PaperStructure from the accumulated state.

Embedding runs concurrently in a background thread using ThreadPoolExecutor.

Notes on Reasoning models
--------------------------
o1 / o3-mini / gpt-5.2 等の reasoning モデルは以下の制約がある:
- ``system`` ロールは使用不可。``developer`` または ``user`` ロールのみ。
- ``temperature`` / ``max_tokens`` は非推奨のため指定しない。
  （必要なら ``max_completion_tokens`` を使う）
ここではすべてのプロンプトを ``user`` ロールで送信し、
temperature 等のパラメータは一切指定しないことで制約に対応している。
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import fitz  # PyMuPDF

from metaweave.llm import get_client, get_settings
from metaweave.schema import AbstractionPattern, MergeResult, PaperStructure

logger = logging.getLogger(__name__)

# チャンクサイズ設定
_CHUNK_SIZE = 4_000   # 1チャンクあたりの目標文字数
_MIN_CHUNK = 500      # これ以下はフラッシュしない最小文字数


# ---------------------------------------------------------------------------
# 内部データクラス
# ---------------------------------------------------------------------------

@dataclass
class _AnalysisState:
    """論文を逐次読み込む際に蓄積する分析状態。"""

    draft: dict[str, Any] = field(default_factory=dict)
    confirmed: list[str] = field(default_factory=list)
    revised: list[str] = field(default_factory=list)
    new_info: list[str] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    chunk_summaries: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public: PDF テキスト抽出
# ---------------------------------------------------------------------------

def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PyMuPDF を使って PDF バイナリからテキストを抽出する。"""
    parts: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Public: チャンク分割
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = _CHUNK_SIZE) -> list[str]:
    """段落・セクション区切りを考慮してテキストをチャンクに分割する。

    2つ以上の改行で段落を区切り、チャンクサイズを超えた場合はフラッシュする。
    1段落が chunk_size を超える場合はセンテンス単位でさらに分割する。
    """
    paragraphs = re.split(r"\n{2,}", text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_len = len(para)

        # チャンクサイズを超えた場合はフラッシュ
        if current_len + para_len > chunk_size and current_len >= _MIN_CHUNK:
            chunks.append("\n\n".join(current))
            current = []
            current_len = 0

        # 1段落が chunk_size を超える場合はセンテンス単位でさらに分割
        if para_len > chunk_size:
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub: list[str] = []
            sub_len = 0
            for sent in sentences:
                if sub_len + len(sent) > chunk_size and sub_len >= _MIN_CHUNK:
                    combined = ("\n\n".join(current) + "\n" if current else "") + " ".join(sub)
                    chunks.append(combined.strip())
                    current = []
                    current_len = 0
                    sub = []
                    sub_len = 0
                sub.append(sent)
                sub_len += len(sent)
            if sub:
                remainder = " ".join(sub)
                current.append(remainder)
                current_len += len(remainder)
        else:
            current.append(para)
            current_len += para_len

    if current:
        chunks.append("\n\n".join(current))

    # フォールバック: チャンクが生成されなかった場合
    return chunks or [text[:_CHUNK_SIZE]]


# ---------------------------------------------------------------------------
# Private: LLM ヘルパー
# ---------------------------------------------------------------------------

def _parse_json(raw: str) -> dict[str, Any]:
    """LLM レスポンスから最初の JSON オブジェクトを抽出する。"""
    match = re.search(r"\{[\s\S]*\}", raw)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {}


def _generate_hypothesis(first_chunk: str, paper_id: str) -> dict[str, Any]:
    """最初のチャンク（Abstract等）から初期仮説ドラフトを生成する。

    LLM に渡すプロンプトはコスト削減のため最小限にする。
    返却 JSON のキー: problem / hypothesis / methodology / contributions
    """
    client = get_client()
    settings = get_settings()

    prompt = (
        f"[paper_id: {paper_id}] 以下は論文の冒頭です。\n"
        "全体像の初期仮説ドラフトを簡潔に作成してください。\n"
        '以下のJSONのみで回答 (キー: "problem", "hypothesis", "methodology", "contributions"):\n\n'
        f"{first_chunk}"
    )

    resp = client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or "{}"
    result = _parse_json(raw)
    if not result:
        # JSON 抽出失敗時のフォールバック
        result = {
            "problem": raw[:300],
            "hypothesis": "",
            "methodology": "",
            "contributions": "",
        }
    logger.debug("Hypothesis draft generated for %s", paper_id)
    return result


def _refine_with_chunk(state: _AnalysisState, chunk: str, chunk_idx: int) -> None:
    """チャンクを読み込んで分析状態をインプレースで更新する。"""
    client = get_client()
    settings = get_settings()

    draft_str = (
        f"Problem: {state.draft.get('problem', '')[:200]}\n"
        f"Hypothesis: {state.draft.get('hypothesis', '')[:200]}"
    )
    confirmed_str = "; ".join(state.confirmed[-5:]) if state.confirmed else "none"

    prompt = (
        f"[Chunk {chunk_idx}]\n"
        f"Initial draft:\n{draft_str}\n"
        f"Confirmed so far: {confirmed_str}\n\n"
        f"New chunk:\n{chunk}\n\n"
        "Return JSON only "
        '(keys: "confirmed"[], "revised"[], "new_info"[], "pending"[], "summary" str):'
    )

    resp = client.chat.completions.create(
        model=settings.analysis_model,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.choices[0].message.content or "{}"
    update = _parse_json(raw)
    if update:
        state.confirmed.extend(update.get("confirmed") or [])
        state.revised.extend(update.get("revised") or [])
        state.new_info.extend(update.get("new_info") or [])
        if update.get("pending") is not None:
            state.pending = update["pending"]
        if update.get("summary"):
            state.chunk_summaries.append(update["summary"])
    logger.debug("Chunk %d refined for state", chunk_idx)


def _finalize_structure(state: _AnalysisState, paper_id: str) -> PaperStructure:
    """蓄積された分析状態から最終的な PaperStructure を生成する。"""
    client = get_client()
    settings = get_settings()

    def _bullets(items: list[str], limit: int = 8) -> str:
        return "\n".join(f"- {x}" for x in items[:limit]) or "none"

    state_str = (
        f"Confirmed findings:\n{_bullets(state.confirmed)}\n\n"
        f"Revised assumptions:\n{_bullets(state.revised)}\n\n"
        f"Unexpected new info:\n{_bullets(state.new_info)}\n\n"
        f"Chunk summaries:\n{_bullets(state.chunk_summaries, 12)}"
    )

    prompt = (
        f'paper_id="{paper_id}"\n\n'
        "Accumulated analysis state from sequential reading:\n"
        f"{state_str}\n\n"
        "Based on the above, extract the final paper structure.\n"
        "Use both Japanese and English in descriptions. "
        f'Set paper_id="{paper_id}".'
    )

    resp = client.beta.chat.completions.parse(
        model=settings.analysis_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=PaperStructure,
    )
    structure: PaperStructure = resp.choices[0].message.parsed
    return structure.model_copy(update={"paper_id": paper_id})


def _embed_and_store_chunks(chunks: list[str], paper_id: str) -> None:
    """チャンクを Embedding して Qdrant に保存する（ベストエフォート）。"""
    try:
        from metaweave.embedder import embed_and_store  # 循環 import 回避のため遅延

        embed_and_store(chunks, paper_id, get_client(), get_settings().embedding_model)
    except Exception:
        logger.warning(
            "Embedding/Qdrant storage failed for %s (continuing without embeddings)",
            paper_id,
            exc_info=True,
        )


# ---------------------------------------------------------------------------
# Public: メイン抽出関数
# ---------------------------------------------------------------------------

def extract_paper_structure(text: str, paper_id: str = "") -> PaperStructure:
    """仮説検証型の逐次処理で論文テキストから PaperStructure を抽出する。

    処理フロー:
    1. テキストをチャンク分割
    2. Embedding を ThreadPoolExecutor でバックグラウンド並行実行
    3. 最初のチャンクから初期仮説ドラフトを生成
    4. 2番目以降のチャンクで逐次精錬（状態を更新）
    5. 最終評価で PaperStructure を出力
    6. Embedding 完了を待機（最大 90 秒）
    """
    chunks = chunk_text(text)
    logger.info("paper=%s  total_chunks=%d", paper_id, len(chunks))

    # Embedding をバックグラウンドで並行実行
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    embed_future = executor.submit(_embed_and_store_chunks, chunks, paper_id)

    try:
        # Step 1: 初期仮説生成（最初のチャンク）
        state = _AnalysisState()
        state.draft = _generate_hypothesis(chunks[0], paper_id)
        logger.info("Hypothesis generated for %s", paper_id)

        # Step 2: 2番目以降のチャンクで逐次精錬
        for idx, chunk in enumerate(chunks[1:], start=1):
            try:
                _refine_with_chunk(state, chunk, idx)
                logger.info("Chunk %d/%d refined for %s", idx, len(chunks) - 1, paper_id)
            except Exception:
                logger.warning(
                    "Refinement failed for chunk %d of %s — skipping",
                    idx,
                    paper_id,
                    exc_info=True,
                )

        # Step 3: 最終評価
        logger.info("Finalizing structure for %s", paper_id)
        structure = _finalize_structure(state, paper_id)

    finally:
        # Embedding の完了を待つ（最大 90 秒、失敗しても続行）
        try:
            embed_future.result(timeout=90)
            logger.info("Embedding completed for %s", paper_id)
        except Exception:
            logger.warning("Embedding future failed for %s", paper_id, exc_info=True)
        executor.shutdown(wait=False)

    return structure


# ---------------------------------------------------------------------------
# Public: LLM提案評価・マージ関数 (Gateway層)
# ---------------------------------------------------------------------------

def evaluate_and_merge_proposals(
    base_structure: PaperStructure,
    proposed_structure: PaperStructure,
) -> MergeResult:
    """Reasoningモデルを使って正典構造とユーザー提案をマージ・評価する。

    方針: 「ジャンクの中の宝石」を最大限に拾い上げる。
    粗削りな提案であっても有用な洞察・補足・修正を積極的に取り込み、
    正典構造をより良いものに育てる。

    Parameters
    ----------
    base_structure:
        現在の正典 PaperStructure（マージのベースライン）。
    proposed_structure:
        ユーザーが提出した提案 PaperStructure。

    Returns
    -------
    MergeResult
        ``merged_structure`` (更新後の正典) と
        ``evaluation_reasoning`` (マージ方針・却下理由のテキスト) を含む。

    Notes on Reasoning models
    -------------------------
    gpt-5.2 等の Reasoning モデルは ``system`` ロールをサポートしないため、
    すべてのプロンプトを ``user`` ロールで送信する。
    temperature 等のパラメータも指定しない。
    """
    client = get_client()
    settings = get_settings()

    prompt = (
        "あなたは論文構造レビュアーです。\n"
        "以下に「現在の正典構造 (base)」と「ユーザー提案構造 (proposed)」を示します。\n\n"
        "【マージ方針】\n"
        "提案はジャンクを含む可能性がありますが、その中にある「宝石」（有用な洞察・"
        "補足・修正・新しい視点）を最大限に拾い上げてください。\n"
        "たとえ粗削りな提案であっても、正典構造をより正確・豊かにする部分があれば"
        "積極的に取り込んでください。\n"
        "一方、誤り・無関係・冗長な部分は正典から除外し、その理由を明記してください。\n\n"
        f"--- base_structure ---\n{base_structure.model_dump_json(indent=2)}\n\n"
        f"--- proposed_structure ---\n{proposed_structure.model_dump_json(indent=2)}\n\n"
        "上記をマージした最終構造 (merged_structure) と、"
        "マージした理由・却下した部分の理由 (evaluation_reasoning) を出力してください。\n"
        f"merged_structure の paper_id は \"{base_structure.paper_id}\" を引き継いでください。"
    )

    resp = client.beta.chat.completions.parse(
        model=settings.analysis_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=MergeResult,
    )
    result: MergeResult = resp.choices[0].message.parsed
    # paper_id は正典のものを必ず引き継ぐ
    return MergeResult(
        merged_structure=result.merged_structure.model_copy(
            update={"paper_id": base_structure.paper_id}
        ),
        evaluation_reasoning=result.evaluation_reasoning,
    )


# ---------------------------------------------------------------------------
# Public: 抽象化パターン抽出 (Public層)
# ---------------------------------------------------------------------------

def extract_abstraction_pattern(structure: PaperStructure) -> AbstractionPattern:
    """承認済み PaperStructure から汎用的な抽象化パターンを LLM で抽出する。

    Evo-DKD アプローチに基づき、具体的事象を変数（X, Y, Z 等）に置換して
    分野横断で適用可能な「問題解決の型」を生成する。

    Parameters
    ----------
    structure:
        パターン抽出対象の PaperStructure（review_status が approved 推奨）。

    Returns
    -------
    AbstractionPattern
        抽出された抽象化パターン。source_arxiv_id には元論文のIDがセットされる。
    """
    client = get_client()
    settings = get_settings()

    prompt = (
        "あなたはメタ構造転写エンジンの一部です。\n"
        "以下の論文構造データから、具体的な事象を「変数（X, Y, Z等）」に置き換え、\n"
        "分野横断で適用可能な汎用的な「問題解決の型（Abstraction Pattern）」を抽出してください。\n\n"
        "【ルール】\n"
        "- name: パターンの簡潔な名称（英語、20語以内）\n"
        "- description: パターンの説明（具体的なドメイン用語は使わず、抽象変数で記述）\n"
        "- variables_template: パターン内で使われる抽象変数のリスト（例: [\"X\", \"Y\", \"Z\"]）\n"
        "- structural_rules: 変数間の関係ルール（例: [\"X inhibits Y\", \"Y enables Z\"]）\n\n"
        f"--- 論文構造データ ---\n{structure.model_dump_json(indent=2)}\n\n"
        f'source_arxiv_id は "{structure.paper_id}" を設定してください。\n'
        "JSONスキーマに厳格に従って出力してください。"
    )

    resp = client.beta.chat.completions.parse(
        model=settings.analysis_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=AbstractionPattern,
    )
    pattern: AbstractionPattern = resp.choices[0].message.parsed
    return pattern.model_copy(update={"source_arxiv_id": structure.paper_id})
