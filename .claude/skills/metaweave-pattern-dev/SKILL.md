---
name: metaweave-pattern-dev
description: MetaWeaveのIssue 3（抽象化パターンの動的テンプレート化と定期再評価）の実装とテストを支援します。パターン抽出、Qdrant検索、バッチ処理のCLIテストを提供します。
allowed-tools: Bash(python3 *)
---

# MetaWeave Pattern Engine 開発支援スキル

このスキルは、MetaWeave の「Public 層（抽象化パターンのテンプレート化と再評価）」のロジック検証を CLI 上で高速に行うためのものです。

## アーキテクチャ概要（パターン関連）

### パターンライフサイクル

```
PaperStructure (approved)
  → extract_abstraction_pattern() [extractor.py]
    → AbstractionPattern (Evo-DKD: 変数テンプレート化)
      → embed_and_store_pattern() [embedder.py] → Qdrant "patterns" コレクション
      → run_pattern_evaluation_task() [batch.py]
        → Qdrant で類似論文検索 → LLM で構造的同型性評価
        → Neo4j に (:Paper)-[:MATCHES_PATTERN]->(:AbstractionPattern) エッジ保存
      → generate_missing_link_suggestions() [llm.py]
        → 構造的空白を AI が自律検知 → arXiv 検索クエリを提案
```

### 関連モジュールと責務

| ファイル | パターン関連の責務 |
|---|---|
| `backend/metaweave/extractor.py` | `extract_abstraction_pattern()`: 承認済み `PaperStructure` から Evo-DKD アプローチで具体事象を変数（X, Y, Z）に置換し、分野横断の `AbstractionPattern` を LLM で抽出。 |
| `backend/metaweave/embedder.py` | `embed_and_store_pattern()`: パターンを Qdrant `patterns` コレクションに Embedding 保存。`search_similar_papers()`: パターンに類似する過去論文を検索。`search_fanns_hybrid()`: FANNS ハイブリッド検索。 |
| `backend/metaweave/batch.py` | `run_pattern_evaluation_task()`: 新パターン登録時の非同期バッチ。候補論文の `PaperStructure` を MinIO からロード→LLM で同型性評価→閾値 (0.6) 以上なら Neo4j にエッジ保存。 |
| `backend/metaweave/llm.py` | `generate_missing_link_suggestions()`: パターンの構造的空白（適用可能だが未カバーの異分野）を AI が検知し、3〜5 件の分野別 arXiv 検索キーワードを提案。 |
| `backend/metaweave/schema.py` | `AbstractionPattern`, `PatternMatch`, `MissingLinkSuggestion`, `FieldSuggestion` の Pydantic 正典スキーマ。 |
| `backend/metaweave/db.py` | Neo4j ドライバ管理。`(:Paper)-[:MATCHES_PATTERN]->(:AbstractionPattern)` エッジの永続化先。 |

---

## 検索基盤（FANNS ハイブリッド検索）

Qdrant の **Pre-filtering（`MatchText` による `smiles_dsl` ペイロードのテキストマッチ）とベクトル検索を組み合わせた真の構造検索**が `embedder.py: search_fanns_hybrid()` として実装完了しています。

**FANNS が Pre-filtering を採用する理由**:
意味（ベクトル）的に遠いが構造（SMILES DSL）が一致する異分野の論文を確実に発見するため。Post-filtering ではベクトル検索の時点で意味的に遠い論文が足切りされ、分野横断検索が成立しない。

**処理フロー** (`embedder.py: search_fanns_hybrid()`):
1. `query_text` を `text-embedding-3-large` でベクトル化
2. `query_dsl_regex` が指定されている場合、`MatchText` フィルタで `smiles_dsl` ペイロードを DB 側で事前絞り込み
3. フィルタ付きベクトル検索を実行（構造一致 → 意味的類似度ランキング）
4. 同一 `arxiv_id` の重複排除（最高スコア保持）→ 上位 `top_k` 件を返却

**API / UI**: FastAPI エンドポイントおよび Streamlit の Cross-Domain Search 画面から利用可能。

---

## パターン機能（Missing Link Suggestion）

登録済みパターンの **「構造的空白（適用可能な異分野）」を AI が自律検知**し、ユーザーに arXiv 検索クエリを提案する機能です。

**処理フロー** (`llm.py: generate_missing_link_suggestions()`):
1. パターンのメタデータ（name, description, structural_rules, variables_template）と既知の適用分野リストを LLM に入力
2. LLM が「このパターンが出現しうるが、まだカバーされていない学術分野」を 3〜5 件特定
3. 各分野について、パターンが出現する具体的理由と arXiv 検索キーワード（2〜4 個）を生成
4. 結果を `MissingLinkSuggestion`（`FieldSuggestion` のリスト）として返却

**関連スキーマ** (`schema.py`): `MissingLinkSuggestion`, `FieldSuggestion`

---

## 1. パターン抽出テスト (`/extract-pattern <arxiv_id>`)
指定された `arxiv_id` の抽出済み構造（MinIO 内の `extracted-structures`）を読み込み、`backend/metaweave/extractor.py` の `extract_abstraction_pattern` 関数を呼び出してテストします。
**実行時の指示:**
- 一時的な Python スクリプトを生成し、対象論文から `AbstractionPattern` を抽出してください。
- 出力された JSON が Pydantic モデル（変数テンプレート、構造ルールなど）に厳格に従っているか、また Evo-DKD の要件（汎用化されているか）を満たしているか検証し、結果をターミナルに整形して出力してください。

## 2. FANNS ハイブリッド検索テスト (`/test-fanns-search <query_regex> <query_text>`)
`backend/metaweave/embedder.py` の `search_fanns_hybrid()` を呼び出し、Qdrant の `smiles_dsl` ペイロードに対するテキストフィルタとベクトル類似度検索を組み合わせたハイブリッド検索（FANNS）のテストを実行します。
**実行時の指示:**
- 一時スクリプトを作成し、`search_fanns_hybrid(query_dsl_regex=query_regex, query_text=query_text)` を直接呼び出してください。
- トップ 5 件を出力し、構造的条件（SMILES DSL マッチ）と意味的条件（ベクトル類似度）が両立できているか確認してください。

## 3. 再評価バッチの強制実行 (`/run-pattern-batch <pattern_id>`)
`backend/metaweave/batch.py` の `run_pattern_evaluation_task` を強制的に同期実行します。
**実行時の指示:**
- スクリプト経由で対象パターンのバッチ評価処理を走らせ、LLM が「過去の論文がこのパターンと同型（Isomorphic）であるか」をどう判断したか（`PatternMatch` の生成と confidence_score）のログを出力してください。
- Neo4j に `(:Paper)-[:MATCHES_PATTERN]->(:AbstractionPattern)` のエッジが正しく張られたかを Cypher クエリで確認してください。

## 4. Missing Link Suggestion テスト (`/test-missing-link <pattern_id>`)
`backend/metaweave/llm.py` の `generate_missing_link_suggestions()` を呼び出し、指定パターンの構造的空白検知をテストします。
**実行時の指示:**
- 対象パターンの情報を MinIO / Qdrant からロードし、`generate_missing_link_suggestions()` に渡してください。
- 出力された `MissingLinkSuggestion` の各 `FieldSuggestion` について、提案された分野・キーワードが妥当かどうかを確認してください。

## 開発上の注意点（エージェント向け）
- OpenAI API を使用するため、`.env` の読み込みを確実に行ってください。
- Qdrant のローカル接続（メモリモードまたは Docker コンテナ）の設定に注意し、接続エラー時は詳細なログを提示してください。
- 常に `backend/metaweave/schema.py` の正典モデルをインポートして型検証を行ってください。
- Reasoning モデル（o1 / o3-mini / gpt-5.2）使用時は `system` ロール不可、`temperature` / `max_tokens` 指定不可の制約を遵守してください。
