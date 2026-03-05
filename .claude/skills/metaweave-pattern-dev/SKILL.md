---
name: metaweave-pattern-dev
description: MetaWeaveのIssue 3（抽象化パターンの動的テンプレート化と定期再評価）の実装とテストを支援します。パターン抽出、Qdrant検索、バッチ処理のCLIテストを提供します。
allowed-tools: Bash(python3 *)
---

# MetaWeave Pattern Engine 開発支援スキル

このスキルは、MetaWeaveの「Public層（抽象化パターンのテンプレート化と再評価）」のロジック検証をCLI上で高速に行うためのものです。

## 1. パターン抽出テスト (`/extract-pattern <arxiv_id>`)
指定された `arxiv_id` の抽出済み構造（MinIO内の `extracted-structures`）を読み込み、`backend/metaweave/extractor.py` の `extract_abstraction_pattern` 関数を呼び出してテストします。
**実行時の指示:**
- 一時的なPythonスクリプトを生成し、対象論文から `AbstractionPattern` を抽出してください。
- 出力されたJSONがPydanticモデル（変数テンプレート、構造ルールなど）に厳格に従っているか、またEvo-DKDの要件（汎用化されているか）を満たしているか検証し、結果をターミナルに整形して出力してください。

## 2. FANNS ハイブリッド検索テスト (`/test-fanns-search <query_regex> <query_text>`)
Qdrantのペイロードに保存された `smiles_dsl` に対する正規表現フィルタと、ベクトル類似度検索を組み合わせたハイブリッド検索（FANNS）のテストを実行します。
**実行時の指示:**
- 一時スクリプトを作成し、まず `query_regex` を用いてQdrantの `smiles_dsl` ペイロードを事前フィルタリング（Pre-filtering）してください。
- その後、フィルタを通過したID（Candidate Set）に対して `query_text` のベクトル検索を行い、トップ5件を出力してください。構造的条件と意味的条件が両立できているか確認してください。

## 3. 再評価バッチの強制実行 (`/run-pattern-batch <pattern_id>`)
`backend/main.py` (または `batch.py`) の `_run_pattern_evaluation_task` を強制的に同期実行します。
**実行時の指示:**
- スクリプト経由で対象パターンのバッチ評価処理を走らせ、LLMが「過去の論文がこのパターンと同型（Isomorphic）であるか」をどう判断したか（`PatternMatch` の生成と confidence_score）のログを出力してください。
- Neo4jに `(:Paper)-[:MATCHES_PATTERN]->(:AbstractionPattern)` のエッジが正しく張られたかをCypherクエリで確認してください。

## 開発上の注意点（エージェント向け）
- OpenAI APIを使用するため、`.env` の読み込みを確実に行ってください。
- Qdrantのローカル接続（メモリモードまたはDockerコンテナ）の設定に注意し、接続エラー時は詳細なログを提示してください。
- 常に `backend/metaweave/schema.py` の正典モデルをインポートして型検証を行ってください。

