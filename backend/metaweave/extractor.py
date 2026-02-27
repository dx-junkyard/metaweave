"""PDF text extraction and LLM-based structure extraction for MetaWeave.

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

import fitz  # PyMuPDF

from metaweave.llm import get_client, get_settings
from metaweave.schema import PaperStructure

# トークン上限超過を防ぐため先頭 _MAX_CHARS 文字に絞る（≒ 25k tokens）
_MAX_CHARS = 100_000


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """PyMuPDF を使って PDF バイナリからテキストを抽出する。"""
    parts: list[str] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            parts.append(page.get_text())
    return "\n".join(parts)


def extract_paper_structure(text: str, paper_id: str = "") -> PaperStructure:
    """OpenAI Structured Outputs を用いて論文テキストから PaperStructure を抽出する。

    Parameters
    ----------
    text:
        論文の全文テキスト。長すぎる場合は先頭 _MAX_CHARS 文字に切り捨てる。
    paper_id:
        抽出結果の paper_id フィールドに設定する値（arXiv ID など）。
    """
    client = get_client()
    settings = get_settings()

    truncated = text[:_MAX_CHARS]

    prompt = (
        "あなたは学術論文の構造を分析する専門家です。\n"
        "以下の論文テキストを詳細に読み、指定された JSON スキーマに厳密に従って"
        "問題構造を日本語と英語の両方で抽出してください。\n\n"
        f'paper_id フィールドには "{paper_id}" を設定してください。\n\n'
        "【論文テキスト】\n"
        f"{truncated}"
    )

    response = client.beta.chat.completions.parse(
        model=settings.analysis_model,
        messages=[{"role": "user", "content": prompt}],
        response_format=PaperStructure,
    )

    structure: PaperStructure = response.choices[0].message.parsed
    # LLM が別の値を設定した場合に備えて paper_id を確実に上書き
    return structure.model_copy(update={"paper_id": paper_id})
