# MetaWeave v1 プロジェクトコンテキスト

## 🎯 プロジェクトの目的
「人類の問題構造マップ」を生成するメタ構造転写エンジン。
学問分野ごとの問題解決パターンを抽出し、異分野へ転写（Structural Isomorphism）することを目指す。

## 🛠 技術スタック
- バックエンド: FastAPI (Python 3.11+)
- フロントエンド: Streamlit
- データベース: Neo4j (グラフDB), Qdrant (ベクトルDB), MinIO (オブジェクトストレージ)
- LLM: OpenAI (gpt-5.2 等、Reasoningモデルを優先)

## ⚠️ 開発の原則と制約
1. **仮説検証型チャンク解析:** 論文抽出ロジック (`extractor.py`) は、最初のチャンクで仮説を立て、後続チャンクで状態を更新する逐次処理を厳守すること。
2. **Reasoningモデルの制約:** OpenAI API 呼び出し時、`system` ロールは使用不可。`developer` または `user` ロールのみを使用し、`temperature` や `max_tokens` の指定は避けること。
3. **Pydantic スキーマの正典化:** 抽出データの構造は必ず `backend/metaweave/schema.py` の Pydantic モデルを正典として扱うこと。
4. **セキュリティと権利関係:** `harvester.py` における商用出版社の検知・除外ロジックを常に維持すること。環境変数のハードコーディングは禁止。

