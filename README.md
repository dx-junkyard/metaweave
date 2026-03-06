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

### 6. Foundation Pattern シード注入

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
│   │   ├── schema.py           # Pydantic モデル（正典スキーマ）
│   │   ├── extractor.py        # 仮説駆動型論文構造抽出
│   │   ├── harvester.py        # arXiv 検索・PDF 取得
│   │   ├── embedder.py         # Qdrant ベクトル登録・検索
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
    ├── app.py                  # Streamlit UI
    ├── Dockerfile
    └── requirements.txt
```

### 主要モジュールの責務

#### `schema.py` — 正典スキーマ

全データ構造の定義元。他モジュールはここを参照し、独自に型を定義しない。

| モデル | 用途 |
|--------|------|
| `PaperStructure` | 論文から抽出した構造（正典） |
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

**③ 抽出結果のレビュー**

「Validation」ページで論文を選択すると、抽出された構造を確認・編集できます。
内容を修正し「変更を提案する」ボタンを押すと LLM がレビューし正典に反映します。
問題がなければ「✅ Approve」で承認します。

**④ パターン抽出**

承認済みの論文で「✨ Pattern」ボタンを押すと、抽象化パターンの抽出とバッチ同型評価が走ります。
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
| `POST` | `/api/patterns/extract/{arxiv_id}` | パターン抽出ジョブ開始 |
| `GET` | `/api/patterns` | 全パターン一覧 |
| `GET` | `/api/papers/{arxiv_id}/patterns` | 論文にマッチするパターン一覧 |

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
(Paper)-[:HAS_PATTERN]->(AbstractionPattern)
(Paper)-[:MATCHES_PATTERN {confidence_score, mapping_explanation}]->(AbstractionPattern)
```

---

## ライセンス

[LICENSE](./LICENSE) を参照してください。
