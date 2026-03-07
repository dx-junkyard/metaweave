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
                ┌──────────────────────────────────────────────────┐
                │                 Streamlit Frontend                │
                │  検索 / 抽出 / レビュー / チャット / パターン          │
                │  Cross-Domain Search / Missing Link Suggestion    │
                └───────────────────┬──────────────────────────────┘
                                    │ HTTP
                ┌───────────────────▼──────────────────────────────┐
                │                  FastAPI Backend                  │
                │  harvester │ extractor │ chat │ patterns │ FANNS  │
                └───┬──────────┬──────────┬───────────┬────────────┘
                    │          │          │           │
          ┌─────────▼──┐ ┌─────▼────┐ ┌──▼────────┐  │
          │   MinIO    │ │  Qdrant  │ │   Neo4j   │  │
          │  (PDF/JSON)│ │ (ベクトル) │ │ (グラフDB) │  │
          └────────────┘ └──────────┘ └───────────┘  │
                                                       │
                              ┌────────────────────────▼──┐
                              │         GROBID              │
                              │  PDF → TEI XML パーサー     │
                              └────────────────────────────┘
                                    │
                        ┌───────────▼───────────┐
                        │      OpenAI API        │
                        │  Reasoning / Embedding │
                        └───────────────────────┘
```

### コンポーネント一覧

| コンポーネント | 役割 |
|--------------|------|
| **Streamlit** (port 8501) | Web UI。論文の検索・取得・レビュー・チャット・パターン管理・Cross-Domain Search |
| **FastAPI** (port 8000) | REST API サーバー。全ビジネスロジックを束ねる |
| **MinIO** (port 9000) | オブジェクトストレージ。PDF 原本と抽出済み JSON を保管 |
| **Qdrant** (port 6333) | ベクトル DB。論文チャンクとパターンの意味検索・FANNS ハイブリッド検索に使用 |
| **Neo4j** (port 7474) | グラフ DB。論文・ユーザー・パターン・提案の関係を保持 |
| **GROBID** (port 8070) | PDF → TEI XML 変換。論理セクション単位の構造事前マッピングに使用 |
| **OpenAI API** | 仮説生成・構造抽出・パターン抽出・RAG チャットに使用 |

---

## 提供する機能

### 1. arXiv 論文ハーベスト

- キーワードで arXiv を検索し、結果をリスト表示
- 商用出版社（Elsevier, Springer, Nature, IEEE, Wiley 等）を自動検出し警告
- PDF を MinIO へダウンロード・保管

### 2. 仮説駆動型 構造抽出（GROBID による構造事前マッピング）

論文の抽出エンジンは、単純なテキスト分割から **GROBID を統合した構造事前マッピング** へと進化しました。

#### GROBID による論理セクション分割

PDF を GROBID API（`/api/processFulltextDocument`）に送信し、TEI XML から **論理セクション（Abstract, Introduction, Methods, Results 等）** を抽出します。これにより：

- **References / Acknowledgments などのノイズを完全除去** — 参考文献リストがチャンクに混入しない
- **セクション境界を意識した分割** — 機械的な文字数区切りではなく意味的に完結した単位で処理
- 抽出精度が大幅に向上

#### 逐次処理パイプライン

```
PDF → GROBID → TEI XML → 論理セクション一覧
                              │
            セクション[0] (Abstract 等) → 仮説生成
            セクション[1..N]            → 確認 / 修正 / 新情報を蓄積しながら逐次更新
            最終セクション               → PaperStructure として確定
            （並行してチャンクを Qdrant へ埋め込み）
```

Reasoning モデルが段階的に構造を精緻化しながら、最終的に `PaperStructure` を確定します。

抽出される構造（`PaperStructure`）:

| フィールド | 内容 |
|-----------|------|
| `problem` | 背景・コア問題 |
| `hypothesis` | 仮説と根拠 |
| `methodology` | アプローチ・手法 |
| `constraints` | 前提条件・限界 |
| `abstract_structure` | 変数と因果グラフ（エッジ + SMILES DSL） |

### 3. 人間参加型レビュー & LLM マージゲートウェイ（Diff ベース最適化）

ユーザーの構造提案をレビューする際、**Python 側で正典と提案の差分（Diff）を計算し、変更があったフィールドのみを LLM に評価させるアーキテクチャ** を採用しています。

#### Diff ベースレビューの利点

| 従来の方式 | Diff ベース方式 |
|-----------|----------------|
| JSON 全体（数千トークン）を LLM に送信 | 変更フィールドのみを送信（トークン大幅削減） |
| LLM のハルシネーションで未変更フィールドが破損するリスクあり | 未変更フィールドは LLM に一切触れさせないため破損ゼロ |
| 差分なしでも必ず LLM コールが発生 | 差分がなければ早期リターン（LLM コスト ゼロ） |

#### フロー

```
1. compute_structure_diff(base, proposed) → FieldDiff リスト
2. 差分が空 → 即時リターン（LLM 呼び出しなし）
3. 差分フィールドのみを LLM に送信 → accept / reject を判定
4. accept されたフィールドのみ base に適用 → MergeResult 生成
```

- 提案は Reasoning モデルが自動レビューし、有益な変更を正典にマージ（「ジャンクの中の宝石を拾い上げる」方針）
- マージ根拠と却下理由も記録（`evaluation_reasoning`）
- 提案履歴を UI から参照可能

### 4. 抽象化パターン抽出 & 同型評価 & Missing Link Suggestion

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

#### Missing Link Suggestion（構造的空白の検知）

パターンが確立されると、システムは「**そのパターンがまだ適用されていないが有効そうな異分野**」を自律的に検知します。

- LLM がパターンの構造変数と既存マッチ論文の分野を分析
- まだカバーされていない学術分野を特定し、arXiv 検索クエリ（キーワード）を自動生成
- UI からワンクリックで提案された分野の論文を検索可能

これにより、システム自身が「次に調べるべき分野」を提案し、パターンライブラリの成長を加速します。

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

#### FANNS（Filtered Approximate Nearest Neighbor Search）— 実装完了

FANNS ハイブリッド検索が稼働しています。Qdrant の **Pre-filtering（SMILES DSL の正規表現マッチ）** とベクトル検索を組み合わせ、構造的類似性と意味的類似性の両面から異分野論文を横断検索します。

```
入力: DSL 正規表現パターン + 自然言語クエリ
  ↓
1. Qdrant Pre-filter: smiles_dsl フィールドに正規表現マッチ（構造フィルタ）
2. Vector Search:     クエリ埋め込みで意味的類似論文を上位 K 件取得
  ↓
出力: 構造的にも意味的にも類似した論文リスト
```

#### Cross-Domain Search UI

Streamlit に **「Cross-Domain Search」** 画面が追加されました。

- ユーザーが自然言語でパターン構造を記述
- LLM が自動的に MetaWeave-SMILES DSL へ変換（`POST /api/search/nl-to-dsl`）
- 変換された DSL で FANNS ハイブリッド検索を実行
- 異分野の類似論文を横断的に発見可能

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
│   │   ├── schema.py           # Pydantic モデル（正典スキーマ）+ FieldDiff / SMILES DSL
│   │   ├── extractor.py        # GROBID 統合・仮説駆動型抽出・Diff ベースマージ
│   │   ├── harvester.py        # arXiv 検索・PDF 取得・商用出版社フィルタ
│   │   ├── embedder.py         # Qdrant ベクトル登録・FANNS ハイブリッド検索
│   │   ├── db.py               # Neo4j ドライバ
│   │   ├── llm.py              # OpenAI クライアント・Missing Link Suggestion
│   │   ├── storage.py          # MinIO ラッパー
│   │   ├── chat.py             # RAG チャットロジック
│   │   └── batch.py            # パターン同型評価バッチ
│   │
│   ├── tests/
│   │   └── test_diff_merge.py  # Diff 計算・マージロジックのユニットテスト
│   │
│   └── scripts/
│       └── seed_patterns.py    # Foundation Pattern シード注入スクリプト
│
└── frontend/
    ├── app.py                  # Streamlit UI（Cross-Domain Search / Missing Link Suggestion 対応）
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
| `FieldDiff` | Diff ベースレビューの単一フィールド差分（field_path / base_value / proposed_value） |
| `AbstractionPattern` | 抽象化された問題解決パターン |
| `PatternMatch` | パターンと論文の対応関係 |
| `StructureProposal` | ユーザーによる構造変更提案 |
| `MergeResult` | LLM によるマージ評価結果 |
| `MissingLinkSuggestion` | 構造的空白の検知結果（異分野 + 検索クエリ） |

#### `extractor.py` — 抽出エンジン

GROBID 統合による構造事前マッピングと、仮説検証型チャンク解析の実装。最初のセクションで仮説を立て、後続セクションで `_AnalysisState` を更新する逐次処理。最終的に `PaperStructure` を確定する。

Diff ベースマージ関数も実装：

- `compute_structure_diff(base, proposed)` — フィールドレベルの差分計算
- `evaluate_and_merge_proposals(base, proposed)` — 差分フィールドのみを LLM に送信し選択的マージ

#### `embedder.py` — FANNS 検索エンジン

- `search_fanns_hybrid(dsl_regex, query_text, top_k)` — DSL 正規表現 Pre-filtering + ベクトル検索のハイブリッド検索（実装完了）

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
| GROBID | http://localhost:8070 |

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
（抽出は GROBID による論理セクション分割を経て、Reasoning モデルが逐次処理します）

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
- **💡 Propose** → 現在のドラフトを LLM Gateway に送信。Diff ベースで差分フィールドのみレビューし正典にマージ
- **✅ Approve** → 論文を承認済みに設定

「📋 提案履歴」タブで過去の提案とその評価理由をタイムライン表示できます。

**④ パターン抽出 & 登録**

承認済みの論文で「✨ Pattern」ボタンを押すと、抽象化パターンの**プレビュー**が生成されます（DB への書き込みはありません）。
内容を確認・編集し「🌍 Register」ボタンを押すと正式に Neo4j + Qdrant へ登録され、バッチ同型評価が走ります。
「Pattern Library」ページで登録されたパターンを一覧確認できます。

**⑤ Missing Link Suggestion（構造的空白の探索）**

「Pattern Library」ページのパターン詳細から「🔍 Find Missing Links」を実行すると、そのパターンがまだ適用されていない有望な異分野が提案されます。
提案された分野の arXiv 検索クエリをワンクリックで「Harvester」に引き渡して新たな論文を探索できます。

**⑥ Cross-Domain Search（異分野横断検索）**

「Cross-Domain Search」ページで自然言語によるパターン構造の記述を入力します。

1. LLM が自然言語を MetaWeave-SMILES DSL へ自動変換
2. FANNS ハイブリッド検索（DSL 正規表現 Pre-filtering + ベクトル類似検索）を実行
3. 構造的にも意味的にも類似した論文を異分野横断で一覧表示

これにより、専門用語を知らなくても「この構造に似た問題を扱っている論文」を異分野から発見できます。

**⑦ RAG チャット**

Validation ページの「💬 Chat」タブで論文に関する質問を自然言語で入力できます。

### API

```bash
# ヘルスチェック
curl http://localhost:8000/healthz

# arXiv 検索
curl "http://localhost:8000/api/search?query=transformer+attention&max_results=10"

# 抽出ジョブのステータス確認
curl "http://localhost:8000/api/extract-status/2301.00001"

# FANNS ハイブリッド検索
curl -X POST http://localhost:8000/api/search/structure \
  -H "Content-Type: application/json" \
  -d '{"dsl_regex": ".*Agent.*causes.*Resource.*", "query_text": "tragedy of the commons", "top_k": 10}'

# Missing Link Suggestion
curl "http://localhost:8000/api/patterns/{pattern_id}/suggestions"
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
| `POST` | `/api/extract` | 非同期抽出ジョブ開始（GROBID 経由） |
| `GET` | `/api/extract-status/{arxiv_id}` | ジョブステータス確認 |
| `GET` | `/api/extract-result/{arxiv_id}` | 抽出済み構造取得 |
| `PUT` | `/api/extract-result/{arxiv_id}` | 構造の手動更新 |

### 提案・レビュー

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/propose-structure` | 構造変更提案の送信（Diff ベース LLM レビュー） |
| `GET` | `/api/proposals/{arxiv_id}` | 提案履歴の取得 |

### パターン

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/patterns/extract/{arxiv_id}` | パターン抽出（プレビューのみ、DB 書き込みなし） |
| `POST` | `/api/patterns/register` | パターンの正式登録（Neo4j + Qdrant + バッチ評価） |
| `GET` | `/api/patterns` | 全パターン一覧 |
| `GET` | `/api/papers/{arxiv_id}/patterns` | 論文にマッチするパターン一覧 |
| `GET` | `/api/patterns/{pattern_id}/suggestions` | Missing Link Suggestion（構造的空白の異分野 + 検索クエリ） |

### 検索（FANNS）

| メソッド | パス | 説明 |
|---------|------|------|
| `POST` | `/api/search/structure` | FANNS ハイブリッド検索（DSL 正規表現 + ベクトル類似） |
| `POST` | `/api/search/nl-to-dsl` | 自然言語 → MetaWeave-SMILES DSL 変換（Cross-Domain Search 用） |

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
