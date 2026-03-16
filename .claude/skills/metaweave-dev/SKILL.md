---
name: metaweave-dev
description: MetaWeaveの論文構造抽出パイプラインの実行、検証、プロンプトの微調整を支援します。
allowed-tools: Bash(python3 *)
---

# MetaWeave 開発・検証エージェント

## アーキテクチャ概要

MetaWeave は「人類の問題構造マップ」を生成するメタ構造転写エンジンです。
学問分野ごとの問題解決パターンを抽出し、異分野へ転写（Structural Isomorphism）します。

### 技術スタック
- **バックエンド**: FastAPI (Python 3.11+)
- **フロントエンド**: Streamlit
- **グラフDB**: Neo4j（論文・パターン間の構造的同型性エッジを管理）
- **ベクトルDB**: Qdrant（Embedding 保存、FANNS ハイブリッド検索）
- **オブジェクトストレージ**: MinIO（PDF・抽出済み構造 JSON の永続化）
- **LLM**: OpenAI (gpt-5.2 等の Reasoning モデルを優先)
- **PDF 解析**: GROBID（TEI XML 経由の論理セクション抽出）

### コアモジュールと責務一覧

| ファイル | 責務 |
|---|---|
| `backend/metaweave/extractor.py` | PDF→GROBID→TEI XML→論理チャンク→仮説検証型 LLM 抽出→`PaperStructure` 生成。Diff ベースの提案マージ（Gateway 層）。抽象化パターン抽出（Public 層）。 |
| `backend/metaweave/chat.py` | RAG ベースのチャット応答。Qdrant でチャンクを類似度検索し、MinIO から `PaperStructure` をロードして LLM にコンテキストとして渡す。 |
| `backend/metaweave/embedder.py` | Qdrant への Embedding 保存・検索。`papers` / `patterns` コレクション管理。FANNS ハイブリッド検索（Pre-filtering + ベクトル検索）の実装。 |
| `backend/metaweave/batch.py` | パターン登録時の非同期バッチ評価。Qdrant で候補論文を検索→LLM で構造的同型性を評価→Neo4j に `MATCHES_PATTERN` エッジを保存。 |
| `backend/metaweave/llm.py` | OpenAI クライアントファクトリ (`get_client` / `get_settings`)。Missing Link Suggestion の LLM プロンプト生成 (`generate_missing_link_suggestions`)。 |
| `backend/metaweave/schema.py` | Pydantic 正典スキーマ。`PaperStructure`, `AbstractionPattern`, `PatternMatch`, `FieldDiff`, `MergeResult`, `MissingLinkSuggestion`, `FieldSuggestion` 等。 |
| `backend/metaweave/harvester.py` | arXiv API 検索・PDF ダウンロード。商用出版社の検知・除外ロジックを内蔵。 |
| `backend/metaweave/storage.py` | MinIO ラッパー (`StorageManager`)。バケット管理・PDF アップロード・署名付き URL 生成。 |
| `backend/metaweave/db.py` | Neo4j ドライバのシングルトン管理。 |
| `frontend/app.py` | Streamlit UI。論文検索・構造ビュー・チャット・Cross-Domain Search・パターン管理画面。 |

---

## 抽出エンジン（GROBID 統合）

従来の PyMuPDF による盲目的なテキスト分割から、**GROBID の TEI XML パースを利用した論理セクション単位の抽出**へ進化しました。

**処理フロー** (`extractor.py`):
1. PDF バイナリを GROBID API (`/api/processFulltextDocument`) に送信し TEI XML を取得
2. `parse_tei_to_logical_chunks()` で XML から `Abstract` / `Body` の各 `<div>` を論理チャンクとして抽出
3. **References / Acknowledgments は正規表現 (`_EXCLUDED_HEADINGS`) で除外**（ノイズ排除）
4. 8,000 文字超のセクションはセンテンス境界でフォールバック分割
5. GROBID 未起動時は PyMuPDF へ自動フォールバック

**仮説検証型の逐次精錬**:
1. 最初のチャンク（Abstract 等）から `_generate_hypothesis()` で初期仮説ドラフトを生成
2. 後続チャンクで `_refine_with_chunk()` により `_AnalysisState` を逐次更新（confirmed / revised / new_info / pending）
3. `_finalize_structure()` で蓄積された状態から最終 `PaperStructure` を Structured Output で生成

**Embedding の並行実行**: 抽出と並行して `ThreadPoolExecutor` で Qdrant への Embedding 保存を実行（最大 90 秒待機）。

---

## LLM Gateway（Diff ベースの査読）

ユーザーからの構造変更提案をレビューする際、全体を LLM に渡すのではなく **Python 側で `PaperStructure` の差分（Diff）を計算し、変更フィールドのみを LLM に評価・マージさせる**堅牢な設計です。

**処理フロー** (`extractor.py: evaluate_and_merge_proposals()`):
1. `compute_structure_diff()` で `base` と `proposed` の `PaperStructure` を再帰比較し `FieldDiff` リストを生成
2. 差分がなければ早期リターン（LLM 呼び出しなし）
3. 差分フィールドのみを LLM に送信 → 各フィールドごとに `accept` / `reject` を判定
4. `accept` された変更のみを `base` 構造にマージ → 未変更フィールドのハルシネーション破損を防止
5. 結果を `MergeResult`（`merged_structure` + `evaluation_reasoning`）として返却

**関連スキーマ** (`schema.py`): `FieldDiff`, `MergeResult`, `StructureProposal`

---

## Reasoning モデルの制約

OpenAI の o1 / o3-mini / gpt-5.2 等の Reasoning モデルには以下の制約があり、コード全体で遵守しています:
- `system` ロールは使用不可 → `user` ロールのみで送信
- `temperature` / `max_tokens` は指定しない（`max_completion_tokens` のみ許可）

---

## 1. 論文の強制抽出・再評価 (`/metaweave-dev extract <arxiv_id>`)
`backend/metaweave/extractor.py` のロジックを用いて構造抽出を実行し、AnalysisState の遷移ログを確認します。

## 2. 抽出プロンプトの最適化 (`/metaweave-dev tune-prompt`)
- 現在の抽出結果と原文を比較し、因果関係（CausalEdge）や制約条件の抽出精度を向上させるためのプロンプト改善案を提示します。
- **[New]** LLM が冗長な JSON ではなく、指定された `MetaWeave-SMILES` の DSL 構文（例: `[a:Agent:Toyota] -[causes:+]-> [r:Resource:Profit]`）を厳格に出力できているか、ダングリングエッジが発生していないか、また `OntologyType` が適切に分類されているかを重点的に検証します。

---

## 🛡️ 法的・アーキテクチャ上の絶対防衛線 (Legal & Architectural Guardrails)
MetaWeaveの設計・実装を行う際は、以下のデータライフサイクルと法的境界を「システムの憲法」として死守し、それに反するコードやスキーマを生成してはならない。

### 1. データライフサイクル (3 Zones + Gateway) の厳守
- **Private Zone (私有地):** ユーザーがAIと壁打ちし、TDM（情報解析）を行う安全地帯。未整理のジャンクや元論文のフルテキスト（表現）のキャッシュが許容されるが、外部共有は厳禁（RLSやテナント分離で保護すること）。
- **The Gateway (関所):** 抽出ロジック (`extractor.py`) とスキーマ (`schema.py`) によって、著者の「表現」を完全に焼き払い、200文字以内の「事実の概要」と純粋なDSLに不可逆変換（脱表現化）する。
- **Public Zone / GitHub (共有地):** Gatewayを通過したクリーンなDSLのみが保管され、Flash Teamの文脈接続に使われる。ライセンス汚染（CC BY-NC, ND等）はここで完全にDropすること。

### 2. 著作権と特許権の境界線のハンドリング
- **著作権（表現の保護）の徹底回避:** 論文の「事実（アイデア）」は人類の共有財産だが、「表現」は保護される。抽出データを外部出力するコードを書く際は、必ず「文字数制限（200文字以内ハードリミット）」と「脱表現化のLLMプロンプト」を強制的に実装せよ。
- **特許権（実施の保護）の分離:** 特許は技術の「実施」を制限するものであり、「構造の分析・公開」は制限しない。システム側で特許の有無をフィルタリングする実装は不要だが、外部公開用スキーマ (`PublicDSLExport` 等) には必ず「技術の実施権を保証しない」旨の免責事項（Disclaimer）をデフォルト値として含めること。


