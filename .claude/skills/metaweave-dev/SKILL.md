---
name: metaweave-dev
description: MetaWeaveの論文構造抽出パイプラインの実行、検証、プロンプトの微調整を支援します。
allowed-tools: Bash(python3 *)
---

# MetaWeave 開発・検証エージェント

## 1. 論文の強制抽出・再評価 (`/metaweave-dev extract <arxiv_id>`)
`backend/metaweave/extractor.py` のロジックを用いて構造抽出を実行し、AnalysisStateの遷移ログを確認します。

## 2. 抽出プロンプトの最適化 (`/metaweave-dev tune-prompt`)
- 現在の抽出結果と原文を比較し、因果関係（CausalEdge）や制約条件の抽出精度を向上させるためのプロンプト改善案を提示します。
- **[New]** LLMが冗長なJSONではなく、指定された `MetaWeave-SMILES` のDSL構文（例: `[a:Agent:Toyota] -[causes:+]-> [r:Resource:Profit]`）を厳格に出力できているか、ダングリングエッジが発生していないか、また `OntologyType` が適切に分類されているかを重点的に検証します。

