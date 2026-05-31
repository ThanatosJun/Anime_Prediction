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
- **Reference baseline / 文獻對照實驗**
  - `../reports/reference_baseline_runs.md`：第一入口，列出已跑 route、run 來源與目前結論。
  - `../reports/reference_baseline_results.csv`：可攜帶的數字結果表，取代本機 `.exp/` raw outputs。
  - `../reports/reference_baseline_paper_alignment_audit.md`：判斷 C1/C2/C3 是否能主張對齊論文，避免把 proxy 寫成 reproduction。
  - `../reports/reference_baseline_status.md`：詳細工作紀錄與歷史脈絡，只有需要追溯決策時再看。
- **論文方法章與處理細節**
  - `pipeline/data_processing_for_paper.md`
- **Baseline 目錄與 ablation 分工**
  - `baseline_directory_planning.md`：說明 `src/reference_baseline_branch/`、`src/ablation_branch/`、`src/experiment_common/` 的存在理由。
  - `baseline_reference_implementation_plan.md`：reference baseline 初始路線圖；目前進度以 `../reports/reference_baseline_runs.md` 為準。
  - `rq2_rag_ablation_plan.md`：RAG 有無與檢索策略 ablation 的設計，不作為 reference baseline 總覽。
- **簡報與專案進度**
  - `../reports/missing_value_status_latest.md`
  - `../reports/external_evaluation_summary.md`
- **外部資料轉換**
  - `pipeline/external_dataset_transform_flow.md`
  - `pipeline/external_schema_mapping_example.json`
  - `pipeline/external_evaluation_method.md`

## 3) 分支模型交接文件

- 文字：`handoff/handoff_text_model.md`
- 圖片：`handoff/handoff_image_model.md`
- 融合：`handoff/handoff_fusion_model.md`
- 外部評估 Adapter：`handoff/handoff_external_eval_adapter.md`

## 4) 歷史提案與歸檔

- `archive_proposal_versions/`：提案迭代與舊版文件（僅供追溯，不作為目前流程依據）。

## 5) 文件維護規範（建議）

- 優先使用中文撰寫；必要英文術語可括號補充。
- 檔名採 `snake_case`，避免中英文混用同義檔案。
- 新增流程文件時，請同步更新本索引，避免「有文件但找不到」。
