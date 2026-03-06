# MetaWeave v1

**メタ構造転写エンジン** — 人類の問題構造マップを生成する知識グラフシステム

---

## ビジョン

あらゆる学問分野には、繰り返し登場する「問題解決の型」が存在します。

「共有地の悲劇」は生態学、経済学、ソフトウェア設計、クラウドリソース管理にまで通底する構造です。
「囚人のジレンマ」は価格競争、軍拡、セキュリティ投資の本質を同じ型で説明します。

MetaWeave は、学術論文から問題解決パターンを自動抽出し、それを **Structural Isomorphism（構造同型）** として異なる分野へ転写することで、分野を超えた洞察を生み出すシステムです。

最終的な目標は「**人類の問題構造マップ**」— 全分野の知恵が相互参照される巨大な知識グラフの構築です。

---

## システム構成

```
                ┌─────────────────────────────────────────┐
                │            Streamlit Frontend            │
                │  検索 / 抽出 / レビュー / チャット / パターン  │
                └───────────────────┬─────────────────────┘
                                    │ HTTP
                ┌───────────────────▼─────────────────────┐
                │              FastAPI Backend             │
                │  harvester │ extractor │ chat │ patterns │
                └───┬───────────┬──────────┬──────────────┘
                    │           │          │
          ┌─────────▼──┐ ┌──────▼───┐ ┌──▼────────┐
          │   MinIO    │ │  Qdrant  │ │   Neo4j   │
          │  (PDF/JSON)│ │ (ベクトル) │ │ (グラフDB) │
          └────────────┘ └──────────┘ └───────────┘
                                    │
                        ┌───────────▼───────────┐
                        │      OpenAI API        │
                        │  Reasoning / Embedding │
                        └───────────────────────┘
```

### コンポーネント一覧

| コンポーネント | 役割 |
|--------------|------|
| **Streamlit** (port 8501) | Web UI。論文の検索・取得・レビュー・チャット・パターン管理 |
| **FastAPI** (port 8000) | REST API サーバー。全ビジネスロジックを束ねる |
| **MinIO** (port 9000) | オブジェクトストレージ。PDF 原本と抽出済み JSON を保管 |
| **Qdrant** (port 6333) | ベクトル DB。論文チャンクとパターンの意味検索に使用 |
| **Neo4j** (port 7474) | グラフ DB。論文・ユーザー・パターン・提案の関係を保持 |
| **OpenAI API** | 仮説生成・構造抽出・パターン抽出・RAG チャットに使用 |

---

## 提供する機能

### 1. arXiv 論文ハーベスト

- キーワードで arXiv を検索し、結果をリスト表示
- 商用出版社（Elsevier, Springer, Nature, IEEE, Wiley 等）を自動検出し警告
- PDF を MinIO へダウンロード・保管

### 2. 仮説駆動型 構造抽出

論文テキストを逐次チャンク処理しながら、Reasoning モデルが段階的に構造を精緻化します。

```
チャンク[0] → 仮説生成
チャンク[1..N] → 確認 / 修正 / 新情報を蓄積しながら逐次更新
最終チャンク → PaperStructure として確定
（並行してチャンクを Qdrant へ埋め込み）
```

抽出される構造（`PaperStructure`）:

| フィールド | 内容 |
|-----------|------|
| `problem` | 背景・コア問題 |
| `hypothesis` | 仮説と根拠 |
| `methodology` | アプローチ・手法 |
| `constraints` | 前提条件・限界 |
| `abstract_structure` | 変数と因果グラフ（エッジ付き） |

### 3. 人間参加型レビュー & LLM マージゲートウェイ

- Web UI で抽出結果を編集し「変更を提案」できる
- 提案は Reasoning モデルが自動レビューし、有益な変更を正典にマージ
- マージ根拠と却下理由も記録（`evaluation_reasoning`）
- 提案履歴を UI から参照可能

### 4. 抽象化パターン抽出 & 同型評価

論文の承認済み構造から「問題解決の型（`AbstractionPattern`）」を生成します。

```python
AbstractionPattern(
    name="Resource Overuse Cascade",
    description="...",
    variables_template=["X", "Y", "Z"],
    structural_rules=["X consumes Y", "Y depletion inhibits Z"],
)
```

生成後はバックグラウンドで過去論文との **構造同型評価** を自動実行し、一致する論文に `MATCHES_PATTERN` エッジを追加します。

### 5. RAG チャット

- ユーザーが論文に対して自然言語で質問できる
- Qdrant でセマンティック検索 → MinIO から構造 JSON 取得 → LLM が回答生成
- チャット履歴を Neo4j に永続化

### 6. ユーザードラフト管理 & パターンプレビューワークフロー

正典データ（MinIO / Qdrant）を保護するため、ユーザーごとの「ドラフト層」を導入しています。

```
[編集] → 💾 Save (Neo4j ドラフト保存)
[ドラフト確認] → 🔄 Re-Extract (is_draft=True で Neo4j にのみ保存)
[パターン確認] → ✨ Pattern (プレビュー生成のみ、DB 書き込みなし)
[確定] → 🌍 Register (Neo4j + Qdrant へ正式登録 + バッチ評価)
[提案] → 💡 Propose (LLM Gateway 経由でレビュー → 正典マージ)
```

Validation View では論文ごとにバッジで状態を表示します：

| バッジ | 意味 |
|--------|------|
| `📝 draft` | ユーザーが編集中のドラフトが存在する |
| `🏛️ canonical` | MinIO の正典データを表示中 |

**新規 API エンドポイント：**

| メソッド | パス | 説明 |
|---------|------|------|
| `GET` | `/api/draft/{arxiv_id}` | ユーザーのドラフト取得（要認証） |
| `PUT` | `/api/draft/{arxiv_id}` | ドラフト保存（要認証）|
| `POST` | `/api/patterns/register` | パターンの正式登録（要認証） |

### 7. MetaWeave-SMILES DSL & FANNS 検索基盤

因果グラフを化学式 SMILES に倣ったテキスト DSL で表現する独自フォーマットを導入しました。

```
[a:Agent:Organization] -[cause:+]-> [r:Resource:Profit]
[r:Resource:Profit] -[inhibit:-]-> [e:Event:Collapse]
```

#### DSL 設計要素

| 要素 | 説明 |
|------|------|
| `[変数名:OntologyType:具体例]` | ノード表現。OntologyType は UFO-C / REA に準拠 |
| `-[関係:極性]->` | 有向エッジ。極性は `+`（正）/ `-`（負）|

**OntologyType（`OntologyType` enum）：**

| 値 | 意味 |
|----|------|
| `Agent` | 意図を持つ行為者 |
| `Resource` | 消費・転用される資源 |
| `Event` | 時間的な出来事 |
| `Purpose-oriented group` | 目的集団 |
| `Institutional Agent` | 制度的主体 |
| `Intentional Moment` | 意図・動機 |

`CausalEdge` モデルに `polarity` / `ontology_level` フィールドが追加され、`AbstractStructure` に `smiles_dsl` フィールドが追加されました。

**FANNS（Filtered Approximate Nearest Neighbor Search）** のプレースホルダーも `embedder.py` に整備済みで、`smiles_dsl` や変数情報を Qdrant ペイロードに付加し、将来的な構造フィルタ検索を可能にします。

### 8. Re-Extract 時の重複 Embedding スキップ

`POST /api/extract` で `is_draft=True` を指定（Re-Extract ボタン経由）の場合、Qdrant への Embedding 処理を自動的にスキップします。これにより正典論文の埋め込みベクトルが二重登録されることを防ぎます。

```python
# ExtractRequest の新フィールド
is_draft: bool = False        # True → Neo4j のみ保存
skip_embedding: bool = False  # True → Qdrant への埋め込みをスキップ
```

### 9. Foundation Pattern シード注入

コールドスタート問題を解消するため、人類が確立した基盤パターンを初期注入するスクリプトを提供します。

収録パターン（`backend/data/foundation_seeds.json`）:

- 共有地の悲劇 (Tragedy of the Commons)
- 囚人のジレンマ (Prisoner's Dilemma)
- 赤の女王仮説 (Red Queen Hypothesis)
- 競争排除則 (Competitive Exclusion Principle)
- 成長の限界／成功の限界 (Limits to Growth / Success to the Successful)
- 予期せぬ結果の法則 (Law of Unintended Consequences)
- 相転移 (Phase Transition)
- ボトルネック／制約理論 (Bottleneck / Theory of Constraints)

---

## ファイル構成

```
metaweave/
│
├── docker-compose.yml          # 全サービス起動定義
├── .env.example                # 環境変数テンプレート
├── CLAUDE.md                   # AI 開発ガイドライン
│
├── backend/
│   ├── main.py                 # FastAPI エントリーポイント・全エンドポイント定義
│   ├── Dockerfile
│   ├── requirements.txt
│   │
│   ├── data/
│   │   └── foundation_seeds.json   # 基盤パターンのシードデータ
│   │
│   ├── metaweave/              # コアライブラリ
│   │   ├── schema.py           # Pydantic モデル（正典スキーマ）+ OntologyType / SMILES DSL
│   │   ├── extractor.py        # 仮説駆動型論文構造抽出・SMILES DSL 生成
│   │   ├── harvester.py        # arXiv 検索・PDF 取得
│   │   ├── embedder.py         # Qdrant ベクトル登録・検索・FANNS プレースホルダー
│   │   ├── db.py               # Neo4j ドライバ
│   │   ├── llm.py              # OpenAI クライアント
│   │   ├── storage.py          # MinIO ラッパー
│   │   ├── chat.py             # RAG チャットロジック
│   │   └── batch.py            # パターン同型評価バッチ
│   │
│   └── scripts/
│       └── seed_patterns.py    # Foundation Pattern シード注入スクリプト
│
└── frontend/
    ├── app.py                  # Streamlit UI（ドラフト管理・SMILES DSL・パターンプレビュー対応）
    ├── Dockerfile
    └── requirements.txt
```

### 主要モジュールの責務

#### `schema.py` — 正典スキーマ

全データ構造の定義元。他モジュールはここを参照し、独自に型を定義しない。

| モデル | 用途 |
|--------|------|
| `PaperStructure` | 論文から抽出した構造（正典） |
| `AbstractStructure` | 変数・エッジ・SMILES DSL を含む抽象構造 |
| `CausalEdge` | 因果エッジ（`polarity`, `ontology_level` を含む） |
| `OntologyType` | UFO-C / REA に基づく上位オントロジー型 enum |
| `AbstractionPattern` | 抽象化された問題解決パターン |
| `PatternMatch` | パターンと論文の対応関係 |
| `StructureProposal` | ユーザーによる構造変更提案 |
| `MergeResult` | LLM によるマージ評価結果 |

#### `extractor.py` — 抽出エンジン

仮説検証型チャンク解析の実装。最初のチャンクで仮説を立て、後続チャンクで `_AnalysisState`（confirmed / revised / new_info / pending）を更新する逐次処理。最終的に `beta.chat.completions.parse` で `PaperStructure` を確定する。

#### `batch.py` — 同型評価バッチ

新パターン登録後にトリガーされる。Qdrant で類似論文を候補検索し、Reasoning モデルで構造同型性を評価、信頼スコア付きで Neo4j にマッチエッジを保存する。

---

## セットアップ

### 前提

- Docker / Docker Compose
- OpenAI API キー

### 1. リポジトリのクローン

```bash
git clone https://github.com/dx-junkyard/metaweave.git
cd metaweave
```

### 2. 環境変数の設定

```bash
cp .env.example .env
```

`.env` を編集して `OPENAI_API_KEY` を設定します。

```env
OPENAI_API_KEY=sk-...
OPENAI_ANALYSIS_MODEL=gpt-4o          # Reasoning モデルを推奨（gpt-5.2 等）
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
```

### 3. 起動

```bash
docker compose up -d
```

| サービス | URL |
|---------|-----|
| フロントエンド (Streamlit) | http://localhost:8501 |
| バックエンド (FastAPI) | http://localhost:8000 |
| API ドキュメント | http://localhost:8000/docs |
| MinIO コンソール | http://localhost:9001 |
| Neo4j ブラウザ | http://localhost:7474 |
| Qdrant ダッシュボード | http://localhost:6333/dashboard |

### 4. Foundation Pattern シード注入（初回のみ）

コールドスタート時に基盤パターンを DB へ登録します。

```bash
docker compose exec backend python -m scripts.seed_patterns
```

---

## 使用方法

### Web UI (Streamlit)

**① アカウント作成・ログイン**

`http://localhost:8501` にアクセスし、Register タブでアカウントを作成してログインします。

**② 論文の検索・取得**

「Harvester」ページで検索ワードを入力し、arXiv から論文を検索します。
目的の論文の「Fetch & Store」ボタンを押すと PDF をダウンロードし、バックグラウンドで構造抽出が始まります。

**③ 抽出結果のレビュー（ドラフトワークフロー）**

「Validation」ページで論文を選択すると、抽出された構造を確認・編集できます。
ヘッダーバッジで現在の状態（`📝 draft` / `🏛️ canonical`）を確認できます。

Structure Editor は 4 つのタブで構成されています：

| タブ | 内容 |
|------|------|
| 📄 Overview | Problem / Hypothesis / Methodology / Constraints |
| 🧬 SMILES DSL | MetaWeave-SMILES 形式の因果グラフ表現 |
| 🕸️ Graph Preview | 抽象変数とエッジのビジュアル |
| 🔧 Raw Variables & Edges | 変数リストとエッジ JSON の直接編集 |

編集後の操作：

- **💾 Save** → ユーザーの Neo4j ドラフトに保存（正典は変更しない）
- **🔄 Re-Extract** → `is_draft=True` で再抽出（Qdrant への二重登録をスキップ）
- **💡 Propose** → 現在のドラフトを LLM Gateway に送信してレビュー → 正典マージ
- **✅ Approve** → 論文を承認済みに設定

「📋 提案履歴」タブで過去の提案とその評価理由をタイムライン表示できます。

**④ パターン抽出 & 登録**

承認済みの論文で「✨ Pattern」ボタンを押すと、抽象化パターンの**プレビュー**が生成されます（DB への書き込みはありません）。
内容を確認・編集し「🌍 Register」ボタンを押すと正式に Neo4j + Qdrant へ登録され、バッチ同型評価が走ります。
「Pattern Library」ページで登録されたパターンを一覧確認できます。

**⑤ RAG チャット**

Validation ページの「💬 Chat」タブで論文に関する質問を自然言語で入力できます。

### API

```bash
# ヘルスチェック
curl http://localhost:8000/healthz

# arXiv 検索
curl "http://localhost:8000/api/search?query=transformer+attention&max_results=10"

# 抽出ジョブのステータス確認
curl "http://localhost:8000/api/extract-status/2301.00001"
```

詳細なエンドポイント仕様は http://localhost:8000/docs を参照してください。

---

## API エンドポイント一覧

### 論文管理

| メソッド | パス | 説明 |
|---------|------|------|
| `GET` | `/api/search` | arXiv 検索（`query`, `max_results`） |
| `POST` | `/api/fetch` | PDF ダウンロード & MinIO 保存 |
| `GET` | `/api/papers` | 保存済み論文一覧 |
| `GET` | `/api/presigned-url` | PDF 閲覧用プリサインド URL |

### 構造抽出

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/extract` | 非同期抽出ジョブ開始 |
| `GET` | `/api/extract-status/{arxiv_id}` | ジョブステータス確認 |
| `GET` | `/api/extract-result/{arxiv_id}` | 抽出済み構造取得 |
| `PUT` | `/api/extract-result/{arxiv_id}` | 構造の手動更新 |

### 提案・レビュー

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/propose-structure` | 構造変更提案の送信 |
| `GET` | `/api/proposals/{arxiv_id}` | 提案履歴の取得 |

### パターン

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/patterns/extract/{arxiv_id}` | パターン抽出（プレビューのみ、DB 書き込みなし） |
| `POST` | `/api/patterns/register` | パターンの正式登録（Neo4j + Qdrant + バッチ評価） |
| `GET` | `/api/patterns` | 全パターン一覧 |
| `GET` | `/api/papers/{arxiv_id}/patterns` | 論文にマッチするパターン一覧 |

### ドラフト管理

| メソッド | パス | 説明 |
|---------|------|------|
| `GET` | `/api/draft/{arxiv_id}` | ユーザーのドラフト取得（要認証） |
| `PUT` | `/api/draft/{arxiv_id}` | ドラフト保存（要認証） |

### チャット & 認証

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/chat` | RAG チャット（要認証） |
| `GET` | `/api/chat/history/{arxiv_id}` | チャット履歴取得（要認証） |
| `POST` | `/api/auth/register` | ユーザー登録 |
| `POST` | `/api/auth/login` | ログイン（JWT 取得） |
| `GET` | `/api/auth/me` | 認証ユーザー情報 |

---

## 開発ガイドライン

### Reasoning モデルの利用制約

OpenAI の Reasoning モデル（o1, o3, gpt-5.2 等）を使用する際は以下を守ること:

- `system` ロールは使用不可。`user` または `developer` ロールのみ使用する
- `temperature` / `max_tokens` は指定しない（`max_completion_tokens` のみ可）
- 構造出力には `client.beta.chat.completions.parse()` と Pydantic モデルを使用する

### スキーマの正典化

データ構造の追加・変更は必ず `backend/metaweave/schema.py` で行うこと。
他モジュールは独自に型を定義せず、`schema.py` のモデルを参照すること。

### セキュリティ

- 環境変数をコードにハードコーディングしない
- `harvester.py` の商用出版社フィルタリングロジックを維持すること
- 認証が必要なエンドポイントには `get_current_user` 依存を使用すること

---

## グラフデータモデル（Neo4j）

```
(User)-[:CHATTED_ABOUT]->(Paper)
(User)-[:PROPOSED]->(StructureProposal)
(User)-[:HAS_DRAFT {structure_json}]->(Paper)       # ドラフト管理
(Paper)-[:HAS_PATTERN]->(AbstractionPattern)
(Paper)-[:MATCHES_PATTERN {confidence_score, mapping_explanation}]->(AbstractionPattern)
```

ドラフトは `(User)-[:HAS_DRAFT]->(Paper)` のエッジプロパティとして保持され、正典（MinIO）を汚染しません。

---

## ライセンス

[LICENSE](./LICENSE) を参照してください。
