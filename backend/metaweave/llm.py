"""OpenAI client factory for MetaWeave.

設定はすべて環境変数（.env）から読み込む。

環境変数
--------
OPENAI_API_KEY          OpenAI API キー（必須）
OPENAI_ANALYSIS_MODEL   文章解析・構造抽出に使う reasoning モデル名
                        デフォルト: gpt-4o
OPENAI_EMBEDDING_MODEL  Embedding に使うモデル名
                        デフォルト: text-embedding-3-large
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI

# プロジェクトルートの .env を読み込む（既に設定済みなら上書きしない）
load_dotenv()


@dataclass(frozen=True)
class LLMSettings:
    """環境変数から読み込んだ LLM 設定。"""

    api_key: str
    analysis_model: str
    embedding_model: str

    @classmethod
    def from_env(cls) -> "LLMSettings":
        api_key = os.environ.get("OPENAI_API_KEY", "")
        if not api_key or api_key.startswith("sk-your"):
            raise EnvironmentError(
                "OPENAI_API_KEY が設定されていません。"
                " .env ファイルに OPENAI_API_KEY=sk-... を追記してください。"
            )
        return cls(
            api_key=api_key,
            analysis_model=os.environ.get("OPENAI_ANALYSIS_MODEL", "gpt-4o"),
            embedding_model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large"),
        )


@lru_cache(maxsize=1)
def get_settings() -> LLMSettings:
    """LLMSettings のシングルトンを返す。"""
    return LLMSettings.from_env()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """OpenAI クライアントのシングルトンを返す。"""
    settings = get_settings()
    return OpenAI(api_key=settings.api_key)
