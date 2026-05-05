# 文件導覽（Docs Index）

這份索引用於快速交接，優先回答「先看哪份文件」。

## 1) 新成員第一次接手（建議閱讀順序）

1. `pipeline/data_pipeline_handoff.md`：資料流程全貌與重建順序。
2. `../scripts/README.md`：每支腳本用途、輸入輸出、常用指令。
3. `handoff/handoff_text_model.md` / `handoff/handoff_image_model.md` / `handoff/handoff_fusion_model.md`：分組交接與任務切分。

## 2) 依任務找文件

- **重跑資料流程 / 改規則**
  - `pipeline/data_pipeline_handoff.md`
  - `../scripts/README.md`
- **論文方法章與處理細節**
  - `pipeline/data_processing_for_paper.md`
- **簡報與專案進度**
  - `reports/missing_value_status_latest.md`
- **外部資料轉換**
  - `pipeline/external_dataset_transform_flow.md`
  - `pipeline/external_schema_mapping_example.json`

## 3) 分支模型交接文件

- 文字：`handoff/handoff_text_model.md`
- 圖片：`handoff/handoff_image_model.md`
- 融合：`handoff/handoff_fusion_model.md`

## 4) 歷史提案與歸檔

- `archive_proposal_versions/`：提案迭代與舊版文件（僅供追溯，不作為目前流程依據）。

## 5) 文件維護規範（建議）

- 優先使用中文撰寫；必要英文術語可括號補充。
- 檔名採 `snake_case`，避免中英文混用同義檔案。
- 新增流程文件時，請同步更新本索引，避免「有文件但找不到」。
