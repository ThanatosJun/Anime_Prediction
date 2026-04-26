# 腳本導覽（Scripts Index）

這份索引用於快速回答「這支腳本做什麼、什麼時候跑」。

## 1) 一次重建（建議順序）

在專案根目錄執行：

```bash
python scripts/pipeline/generate_raw_manifest.py
python scripts/eda/run_baseline_eda.py
python scripts/eda/run_decision_eda.py
python scripts/pipeline/build_interim_dataset.py
python scripts/pipeline/build_processed_dataset.py
python scripts/pipeline/export_multimodal_inputs.py
python scripts/eda/run_rq_eda.py
python scripts/eda/run_rq_eda_plots.py
python scripts/eda/run_holdout_unknown_diagnostic.py
python scripts/eda/run_column_lineage_report.py
```

## 2) 腳本功能地圖

- `pipeline/generate_raw_manifest.py`
  - 固定原始資料快照（檔案大小與雜湊），避免來源漂移。
- `eda/run_baseline_eda.py`
  - 產生初始資料品質與分佈摘要。
- `eda/run_decision_eda.py`
  - 產出缺值與異常值策略建議。
- `pipeline/build_interim_dataset.py`
  - 套用缺值規則，輸出 interim 版資料與 meta。
- `pipeline/build_processed_dataset.py`
  - 套用異常值處理、目標工程、時序切分。
- `pipeline/export_multimodal_inputs.py`
  - 輸出 multimodal 主檔與 train/val/test/holdout 檔案。
- `eda/run_rq_eda.py`
  - 產出對應研究問題（RQ）的可行性證據摘要。
- `eda/run_rq_eda_plots.py`
  - 產生論文可用圖表。
- `eda/run_holdout_unknown_diagnostic.py`
  - 檢查 unknown holdout 的組成與分佈落差風險。
- `eda/run_column_lineage_report.py`
  - 追蹤 Raw -> Interim -> Processed 欄位血緣。
- `eda/run_missing_value_report.py`
  - 產出最新缺值狀態文件（給進度回報與交接使用）。
- `eda/run_target_correlation_heatmaps.py`
  - 產出目標與特徵相關性熱圖（探索分析）。
- `external/transform_external_dataset.py`
  - 外部資料欄位轉換與映射。

## 3) 常見維護入口

- 缺值規則：`pipeline/build_interim_dataset.py`（`MISSING_RULES`）
- 異常值規則：`pipeline/build_processed_dataset.py`（`CLIP_COLUMNS`）
- 目標與時序切分：`pipeline/build_processed_dataset.py`

## 4) 維護建議

- 新增腳本時，檔名使用 `verb_object.py`（例如 `run_xxx.py`, `build_xxx.py`）。
- 每支腳本應至少包含：用途、主要輸入、主要輸出、是否可重入（idempotent）。
- 任何新輸出檔案，請同步更新 `docs/pipeline/data_pipeline_handoff.md` 與本索引。
