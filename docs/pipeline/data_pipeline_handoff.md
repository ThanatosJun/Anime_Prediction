# 資料流程交接指南（Data Pipeline Handoff）

本文件提供給需要「重跑」或「修改」AniList 資料流程的接手成員。

## 1) 端到端重建順序

請在專案根目錄執行：

```bash
python scripts/generate_raw_manifest.py
python scripts/run_baseline_eda.py
python scripts/run_decision_eda.py
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
python scripts/export_multimodal_inputs.py
python scripts/run_rq_eda.py
python scripts/run_rq_eda_plots.py
python scripts/run_holdout_unknown_diagnostic.py
python scripts/run_column_lineage_report.py
```

## 2) 主要輸出檔案說明

- `data/raw/raw_manifest.json`
  - 原始資料快照指紋（檔案大小 + sha256）。
- `data/eda/baseline_eda_summary.*`
  - 描述性資料品質摘要（缺值率、分佈、IQR 邊界）。
- `data/eda/decision_eda_summary.*`
  - 缺值與異常值策略建議。
- `data/interim/anilist_anime_data_interim_YYYYMMDD_meta.json`
  - 已套用清理規則版本與缺值政策。
- `data/processed/anilist_anime_data_processed_v1_meta.json`
  - 已套用異常值閾值、clip 設定與規則版本。
- `data/processed/anilist_anime_multimodal_input_v1.csv` 與 split 分檔
  - 多模態建模輸入主契約，包含 `train/val/test/holdout_unknown` 實體檔。
- `data/eda/multimodal_input_summary.*`
  - 特徵契約摘要與模態覆蓋率。
- `data/eda/target_engineering_summary.*`
  - quarter-normalized popularity 標籤分佈與 pre-release 時序切分摘要。
- `data/eda/rq_eda_summary.*`
  - 對應研究問題（RQ）的流程可行性證據（含 split 平衡、模態覆蓋、統計檢定）。
- `data/eda/figures/*.png`
  - 論文可用圖（snapshot control、split balance、multimodal coverage）。
- `data/eda/figures/rq_figure_notes.md`
  - 圖表解讀說明。
- `data/eda/holdout_unknown_diagnostic.*`
  - holdout unknown 組成與分佈落差診斷。
- `data/eda/column_lineage_summary.*`
  - Raw -> Interim -> Processed 欄位血緣（保留/刪除/新增）報告。

## 2.1) Processed 六份 CSV 對照

- `data/processed/anilist_anime_data_processed_v1.csv`
  - 角色：tabular 主資料（baseline 與診斷的工程契約）
  - 列/欄：`20324` / `38`
  - 用途：tabular 建模、完整特徵檢查、切分策略驗證
- `data/processed/anilist_anime_multimodal_input_v1.csv`
  - 角色：multimodal 主資料（text/image/trailer-ready 契約）
  - 列/欄：`20324` / `21`
  - 用途：多模態分支實驗單一入口
- `data/processed/anilist_anime_multimodal_input_train.csv`
  - 角色：train 分檔
  - 列/欄：`13376` / `21`
  - 用途：訓練直接載入，不需執行時再切 split
- `data/processed/anilist_anime_multimodal_input_val.csv`
  - 角色：validation 分檔
  - 列/欄：`2918` / `21`
  - 用途：調參與 early stopping
- `data/processed/anilist_anime_multimodal_input_test.csv`
  - 角色：test 分檔
  - 列/欄：`3087` / `21`
  - 用途：最終 hold-out 測試回報
- `data/processed/anilist_anime_multimodal_input_holdout_unknown.csv`
  - 角色：時序未知 holdout 分檔
  - 列/欄：`943` / `21`
  - 用途：風險診斷；不納入 train/val/test 擬合

## 2.2) 為什麼 `val` 與 `test` 筆數不相等

- 切分策略是 **chronological + quarter-block assignment**（`chronological_cumulative_rows`）。
- pipeline **不會**把同一個 quarter 拆到不同 split。
- 因為每季樣本數不同，`15% / 15%` 不會保證完全相等。
- 目前已知時間資料列：`19381`（`20324 - 943 holdout_unknown`）。
- 目前實際比例（已知時間資料）：
  - train：`13376 / 19381` = `69.0%`
  - val：`2918 / 19381` = `15.1%`
  - test：`3087 / 19381` = `15.9%`
- 此結果屬預期且較符合時序完整性（避免 quarter leakage）。

## 3) 規則修改入口

- 缺值處理政策：
  - `scripts/build_interim_dataset.py`
  - 調整 `MISSING_RULES` 與 `impute_missing_values()`。
- 異常值政策：
  - `scripts/build_processed_dataset.py`
  - 調整 `CLIP_COLUMNS` 與 `_clip_by_percentile()`。
- popularity 目標與時序切分政策：
  - `scripts/build_processed_dataset.py`
  - 調整 `_add_popularity_quarter_target()` 與 `_apply_pre_release_temporal_split()`。
  - 現行策略為 chronological + cumulative row ratio（目標 70/15/15）。
- 多模態輸出契約：
  - `scripts/export_multimodal_inputs.py`
  - 調整模態欄位選取與 split 檔案契約。
- 建議策略邏輯：
  - `scripts/run_decision_eda.py`
  - 調整 `_missing_strategy()` 與 `_outlier_strategy()`。
- RQ 證據層：
  - `scripts/run_rq_eda.py`
  - 調整 snapshot/retrieval/multimodal proxy 指標與統計檢定設定。
- Holdout 風險診斷：
  - `scripts/run_holdout_unknown_diagnostic.py`
  - 調整 temporal-missing 與 distribution-gap 檢查。
- 欄位血緣報告：
  - `scripts/run_column_lineage_report.py`
  - 調整分階段 keep/drop/add 解讀邏輯。

## 4) 常見更新情境

### A. 原始資料更新
1. 替換 `data/raw` 內原始檔（`pkl/csv`）。
2. 執行 `scripts/generate_raw_manifest.py`。
3. 執行完整重建流程。
4. 確認 decision/outlier 摘要是否如預期變動。

### B. 模型需要新增欄位
1. 在 `scripts/build_interim_dataset.py` 的 `KEEP_COLUMNS` 加入欄位。
2. 若為數值欄位，同步更新 `NUMERIC_COLUMNS`。
3. 重建並檢查缺值/異常值摘要。

### C. 覺得 outlier clipping 太激進
1. 調整 `CLIP_COLUMNS` 百分位數。
2. 重建 processed 資料。
3. 比對 `outlier_handling_summary.*` 前後邊界與筆數變化。

### D. 需要調整季度 popularity 定義
1. 修改 `_add_popularity_quarter_target()` 的分箱標籤或門檻。
2. 重建 processed 資料。
3. 檢查 `target_engineering_summary.*` 是否出現類別分佈漂移。

## 5) 交接前驗收清單

- Raw 指紋已存在且對應目前檔案（`raw_manifest.json`）。
- Decision 摘要已產出（`decision_eda_summary.json/.md`）。
- Interim metadata 含 `rule_version` 與 `applied_missing_rules`。
- Processed metadata 含 `rule_version` 與 `clip_config`。
- Processed metadata 含 `popularity_quarter_target` 與 `pre_release_split`。
- Processed metadata 含 `unknown_split_policy`。
- Multimodal 輸出檔案齊全（主檔 + split 分檔）。
- Multimodal 摘要已產出（`multimodal_input_summary.json/.md`）。
- RQ 摘要已產出（`rq_eda_summary.json/.md`）。
- RQ 圖與圖表說明已產出於 `data/eda/figures/`。
- Holdout 診斷已產出（`holdout_unknown_diagnostic.json/.md`）。
- 欄位血緣報告已產出（`column_lineage_summary.json/.md`）。
- 在乾淨 shell session 中可無人工介入重跑整條 pipeline。

