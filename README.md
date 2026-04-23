# 資料科學與機器學習：期末研究提案

本專案為「資料科學與機器學習」課程之期末研究文件儲存庫。
歷經多次討論與收斂，最終確立之研究主題為：**新番動畫播出前熱度預測 (Pre-release Anime Popularity Prediction Based on Multimodal Features)**。

## 📁 專案目錄架構 (Project Structure)

```text
Anime_Prediction/
├── data/
│   ├── raw/                                        # AniList 原始資料集檔案 (csv/pkl + manifest)
│   ├── interim/                                    # 清理後中間資料 (可重建，不入版控)
│   ├── processed/                                  # 最終建模資料 (可重建，不入版控)
│   ├── eda/                                        # EDA 摘要輸出 (md/json)
│   └── archive_local/                              # 本機長期保存區 (不入版控)
├── docs/                                           # 文件目錄
│   ├── data_processing_for_paper.md               # 論文處理紀錄 (方法學說明)
│   ├── data_pipeline_handoff.md                   # 團隊交接指南
│   └── archive_proposal_versions/                 # 提案階段歷史文件歸檔
├── scripts/                                        # 資料流程腳本 (EDA/Cleaning/Outlier)
│   ├── run_baseline_eda.py
│   ├── run_decision_eda.py
│   ├── build_interim_dataset.py
│   ├── build_processed_dataset.py
│   ├── export_multimodal_inputs.py
│   └── generate_raw_manifest.py
├── fetch_data.py                                   # AniList GraphQL 抓取與匯出腳本
├── .github/                                        # GitHub 相關設定與 Skills
├── .gitignore                                      # Git 忽略檔案設定 (排除 agents/skills)
└── README.md                                       # 專案說明文件 (本檔案)
```

## Dataset Processing Workflow

### Quick Rebuild (Team Handoff Friendly)

```bash
python scripts/generate_raw_manifest.py && ^
python scripts/run_baseline_eda.py && ^
python scripts/run_decision_eda.py && ^
python scripts/build_interim_dataset.py && ^
python scripts/build_processed_dataset.py && ^
python scripts/export_multimodal_inputs.py && ^
python scripts/run_rq_eda.py && ^
python scripts/run_rq_eda_plots.py && ^
python scripts/run_holdout_unknown_diagnostic.py && ^
python scripts/run_column_lineage_report.py
```

### 1) Baseline EDA

```bash
python scripts/run_baseline_eda.py
```

輸出：
- `data/eda/baseline_eda_summary.json`
- `data/eda/baseline_eda_summary.md`

### 2) Decision EDA (Rule Recommendation Layer)

```bash
python scripts/run_decision_eda.py
```

輸出：
- `data/eda/decision_eda_summary.json`
- `data/eda/decision_eda_summary.md`

### 3) Build Interim Dataset

```bash
python scripts/build_interim_dataset.py
```

主要處理：
- 保留建模核心欄位
- 型別統一（數值欄位 coercion）
- 以 `id` 去重
- 缺值補值（如 `episodes`, `duration`, `averageScore`）

輸出：
- `data/interim/anilist_anime_data_interim_YYYYMMDD.csv`
- `data/interim/anilist_anime_data_interim_YYYYMMDD_meta.json`

### 4) Build Processed Dataset (Outlier Handling)

```bash
python scripts/build_processed_dataset.py
```

主要處理：
- 非負值約束（負值裁切為 0）
- 關鍵數值欄位 percentile clipping (P1-P99)
- 生成同季度 `popularity` 百分比分類標籤（`popularity_quarter_pct`, `popularity_quarter_bucket`）
- 生成 pre-release 時序切分欄位（`split_pre_release`: train/val/test/unknown）

輸出：
- `data/processed/anilist_anime_data_processed_v1.csv`
- `data/processed/anilist_anime_data_processed_v1_meta.json`
- `data/eda/outlier_handling_summary.json`
- `data/eda/outlier_handling_summary.md`
- `data/eda/target_engineering_summary.json`
- `data/eda/target_engineering_summary.md`

### 5) Freeze Raw Snapshot Metadata

```bash
python scripts/generate_raw_manifest.py
```

輸出：
- `data/raw/raw_manifest.json`

### 6) Export Multimodal Inputs (Feature Contract + Split Files)

```bash
python scripts/export_multimodal_inputs.py
```

輸出：
- `data/processed/anilist_anime_multimodal_input_v1.csv`
- `data/processed/anilist_anime_multimodal_input_train.csv`
- `data/processed/anilist_anime_multimodal_input_val.csv`
- `data/processed/anilist_anime_multimodal_input_test.csv`
- `data/processed/anilist_anime_multimodal_input_holdout_unknown.csv`
- `data/eda/multimodal_input_summary.json`
- `data/eda/multimodal_input_summary.md`

### 7) RQ-oriented EDA

```bash
python scripts/run_rq_eda.py
```

輸出：
- `data/eda/rq_eda_summary.json`
- `data/eda/rq_eda_summary.md`

### 8) RQ Figure Generation (Paper-ready)

```bash
python scripts/run_rq_eda_plots.py
```

輸出：
- `data/eda/figures/rq_snapshot_control.png`
- `data/eda/figures/rq_split_bucket_balance.png`
- `data/eda/figures/rq_multimodal_coverage_by_split.png`
- `data/eda/figures/rq_figure_notes.md`

### 9) Holdout Unknown Diagnostic

```bash
python scripts/run_holdout_unknown_diagnostic.py
```

輸出：
- `data/eda/holdout_unknown_diagnostic.json`
- `data/eda/holdout_unknown_diagnostic.md`

### 10) Column Lineage Report

```bash
python scripts/run_column_lineage_report.py
```

輸出：
- `data/eda/column_lineage_summary.json`
- `data/eda/column_lineage_summary.md`

## 檔名規範與格式政策

- Raw（canonical）：`anilist_anime_data_complete.pkl` + `anilist_anime_data_complete.csv`
- Interim：`anilist_anime_data_interim_YYYYMMDD.*`
- Processed：`anilist_anime_data_processed_v1.*`

## 版控策略

- `data/raw` 保留原始資料來源。
- `data/interim`、`data/processed` 大型可重建產物不納入版控。
- `data/eda` 保留輕量摘要（`*_summary.md`, `*_summary.json`）便於追蹤品質變化。
- `data/archive_local` 作為本機長期保存與版本紀錄區，不納入版控。

## 規則維護入口（給接手成員）

- 缺值處理與補值規則：`scripts/build_interim_dataset.py`（`MISSING_RULES`）
- 異常值閾值與 clipping 設定：`scripts/build_processed_dataset.py`（`CLIP_COLUMNS`）
- 分類標籤與時序切分策略：`scripts/build_processed_dataset.py`（`_add_popularity_quarter_target`, `_apply_pre_release_temporal_split`）
- 多模態輸入匯出與 split 分檔：`scripts/export_multimodal_inputs.py`
- 規則建議來源：`scripts/run_decision_eda.py` + `data/eda/decision_eda_summary.*`
- RQ 導向可行性與 snapshot 緩解證據：`scripts/run_rq_eda.py` + `data/eda/rq_eda_summary.*`
- 論文圖表輸出：`scripts/run_rq_eda_plots.py` + `data/eda/figures/*`
- holdout 風險診斷：`scripts/run_holdout_unknown_diagnostic.py` + `data/eda/holdout_unknown_diagnostic.*`
- 欄位血緣對照：`scripts/run_column_lineage_report.py` + `data/eda/column_lineage_summary.*`
- 規則版本追蹤：`data/interim/*_meta.json`、`data/processed/*_meta.json` 的 `rule_version`

## 論文寫作處理紀錄

- 請直接使用 `docs/data_processing_for_paper.md`。
- 內容包含：處理階段目的、規則定義、參數、target engineering 公式、時序切分協議、可重現證據與限制說明。

## 🎯 最終定案研究任務摘要

在動畫正式釋出第一集前 (冷啟動狀態下)，模型僅能取得**開播前的多模態資訊**與**表格元資料**。本研究以 **Popularity** 與 **Mean Score** 作為兩個核心目標，兩者皆採迴歸設定，用於支援平台與 IP 合作方的前置決策評估。現階段不再以「開播後第 7 天」作為固定目標定義，而是以同一資料快照下可用的 post-release 指標進行流程驗證與建模設計。

### 🔑 核心研究特徵 (Input Features)
- **文字 (Text)**：劇情大綱 (Synopsis)
- **圖片 (Image)**：主視覺圖 / 海報 (Cover Image)
- **影片 (Video/Audio)**：預告片 (PV) 的時空特徵與配樂節奏
- **表格 (Tabular)**：製作公司陣容、是否為續作、前作 IP 歷史影響力等 (作為 Control Variables)

### 🔬 消融實驗設計 (Ablation Study)
本研究設計三組漸進式的消融實驗，以量化拆解各模態的影響力：
1. **Baseline (Tabular + Text)**：僅依賴「IP/製作陣容 + 故事設定」，代表業界最基本的企劃文案評估水準。
2. **加入靜態視覺 (+Image)**：評估加入主視覺圖後，對觀眾吸引力的額外增益。
3. **完整模型 (+PV)**：檢驗動態視聽資訊 (畫風流暢度、分鏡、配樂節奏) 所帶來的最終極限增益。
