# 實驗回補進度與後續待辦

日期：2026-06-12

分支：`feature/experiment-followups`

本文件是老師回饋後，針對 Exp1、Exp2、Exp3 的實驗補強進度總覽。它的用途是協助後續論文、簡報與程式交接判斷哪些結果已經可作為目前正本，哪些仍屬於 future work 或主框架延伸工作。

## 目前權威結果檔案

修正版論文敘事與後續報告應優先使用下列檔案：

- `reports/experiments/sample_alignment/eval_sample_alignment_report_2026-06-11.md`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.csv`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.md`
- `reports/experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.csv`
- `reports/experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.md`
- `reports/experiments/sample_alignment/mal2025_overlap_label_sanity.csv`
- `reports/experiments/sample_alignment/mal2025_external_calibration_summary.csv`
- `reports/experiments/sample_alignment/mal2025_external_calibration_bins.csv`
- `reports/experiments/sample_alignment/mal2025_external_error_slices.csv`
- `reports/experiments/sample_alignment/mal2025_external_case_examples.csv`
- `reports/experiments/sample_alignment/mal2025_external_diagnostics.md`
- `reports/experiments/sample_alignment/followup_paired_bootstrap_tests.csv`
- `reports/experiments/sample_alignment/followup_external_paired_bootstrap_tests.csv`
- `reports/experiments/sample_alignment/followup_external_paired_bootstrap_tests.md`
- `reports/experiments/sample_alignment/followup_image_proxy_diagnostics.csv`
- `reports/experiments/sample_alignment/followup_experiment_statistics.md`
- `reports/experiments/sample_alignment/run22_artifact_manifest.md`
- `reports/paper/paper_user_sections_complete_draft_2026-06-01.md`

`reports/baselines/` 底下的舊 baseline 報告，以及 `reports/paper/` 底下較早的 handoff 文件，部分仍會提到舊版 `n=2,808` complete-case baseline 設定。這些檔案現在應視為歷史開發紀錄，不應再作為目前主表或正式敘事的主要依據。

## Exp1：Baseline Comparison

已完成：

- 已將代表性 F1/F2/C1/C2/C3 baselines 重新計算到與 CARMA 相同的完整 internal temporal test set：`n=3,087`。
- 已把 C1/C2 rows 補回 Exp1 internal comparison，避免前文提到 literature baselines 但主表不呈現。
- 已將論文敘事從舊版 `n=2,808` complete-case table 改成目前的 full-test aligned baselines。
- 已加入 CARMA-vs-baseline headline deltas 的 paired bootstrap diagnostics。

仍需注意：

1. 論文或報告若空間允許，可加入一張精簡 paired-bootstrap summary table。
   - 建議 rows：CARMA vs F2、CARMA vs C2-Recurrent、CARMA vs C3。
   - 建議 metrics：popularity `log_MAE` / Spearman，以及 meanScore MAE / Spearman。
   - 目的：說明哪些差距穩定，哪些其實接近 tie。
2. Exp1 結論必須分 target 寫，不能宣稱 CARMA 全面勝出。
   - `meanScore`：CARMA 在 MAE 與 10-point accuracy 上有最清楚優勢。
   - `popularity`：CARMA 具 competitive performance，但 F2/C2/C3 在部分 ranking 或 error metrics 仍很強。
3. Baseline role table 應保持精簡且角色明確。
   - F1：metadata-only strong floor。
   - F2：simple multimodal concatenation floor。
   - C1/C2：literature-adapted fusion proxies。
   - C3：retrieval reference baseline。
4. C1/C2 的 claim boundary 要寫清楚。
   - 可用：`literature-adapted`、`project-input proxy`、`C2-inspired`。
   - 不可寫成 exact reproduction of the original papers。

## Exp2：CARMA Ablation

已完成：

- 主要 ablation axes 已可用於目前專案敘事：remove retrieval、remove image、remove temporal trend。
- 目前解讀：移除 retrieval、image 或 temporal trend 都會增加 error，因此這三個 component 對 CARMA 都有貢獻。
- 已加入主要 ablation deltas 的 paired bootstrap diagnostics。

仍需注意：

1. 若後續要回應老師的 robustness 建議，應補 key ablation deltas 的 multi-seed mean/std。
   - 優先 deltas：full model vs remove retrieval、remove image、remove temporal trend。
2. Exp2 圖表與 captions 要確認 metric direction 一致。
   - `log_MAE` 與 MAE 都是 lower is better。
   - 避免寫成 error 上升代表改善。
3. 若時間允許，可補 `meanScore` calibration 或 error-distribution view，因為 internal R2 仍偏低。

## Exp3：MAL External Test

已完成：

- 已重新執行 Run22 在 MAL 2025 local-ready splits 上的 external inference。
- 已在相同 MAL rows 上重新計算 F1/F2/C1/C2/C3 baselines。
- 已加入 cover-derived YOLO diagnostic splits，並重跑 CARMA 與 F1/F2/C1/C2/C3。
- 已確認 image encoder artifact 存在於 `src_2/component_image/model-image/best/`。
- 已完成 MAL 2025 overlap label sanity check。
  - Popularity Spearman：`0.9836`。
  - Score Spearman：`0.9446`。
- 已加入 Run22 external calibration 與 prediction-quantile diagnostics。
- 已加入 cover-as-banner proxy splits，並重跑 CARMA 與 F1/F2/C1/C2/C3。
  - 這只是 diagnostic proxy；MAL 2025 仍沒有真正的 banner images。
- 已加入相同 MAL rows 上的 CARMA-vs-F2/C2/C3 external paired-bootstrap diagnostics。
- 已加入 external error-slice diagnostics，涵蓋 MAL popularity/score quantiles、release period、format、source 與 high-popularity tail。
- 已加入精簡的 external calibration / slice explanation，用來說明為何 Spearman 仍有參考價值，但 R2 可能很弱或為負。
- 已加入 external success/failure case examples，涵蓋 high-confidence ranking successes、high-popularity underestimation 與 large score errors。

仍需注意：

1. External conclusion 必須保守。
   - CARMA 在 external score MAE / 10-point accuracy 上最穩定。
   - Popularity 與 score ranking 是 mixed result；F2/C2/C3 在某些 splits 的 Spearman 可能比 CARMA 強。
2. 真正的 banner-like branch 仍是 future work。
   - 已完成的 cover-as-banner proxy 顯示 naive cover duplication 不能取代真正 banner information。

## 全域 Robustness 待辦，不屬於 Exp1 專屬工作

以下工作會提升整體證據品質，但不應被寫成 Exp1-only work：

1. 對 CARMA 與重要 baselines 補 multiple seeds。
   - 影響 Exp1 baseline comparison 與 Exp2 ablation reliability。
   - 建議範圍：Run22-style CARMA、F2、C2-Recurrent、C3-RAG-XGB。
   - 回報 key metrics 的 mean/std。
2. 有了 multi-seed runs 後，再做 seed-level significance tests。
   - 目前 paired bootstrap 只檢查 fixed prediction artifacts。
   - 它不能取代 seed-level uncertainty。

## 主框架剩餘工作，不屬於目前 Exp1/Exp3 cleanup

以下任務需要模型或架構層級工作，不應與目前 Exp1/Exp3 evaluation cleanup 混在同一輪：

1. 在同一個 CARMA architecture 裡補 strict CNN-vs-Swin backbone ablation。
   - 目前 artifacts 支援 image-source ablations 與 ResNet/CNN literature proxy diagnostics。
   - 但還沒有提供一個只替換 backbone 的 CARMA CNN-vs-Swin one-variable comparison。
2. 訓練或取得真正的 banner-like external branch。
   - cover-as-banner proxy 只是 missing-modality diagnostic。
   - 合格的分支需要真正 banner data，或一個明確訓練過的 banner imputation method。

## 論文與報告敘事決策

- Exp1 main internal comparison 應使用 `n=3,087`。
- 舊版 `n=2,808` complete-case baseline files 應視為 historical development artifacts，不是目前主要證據。
- Exp3 敘事應區分：
  - main no-YOLO MAL local-ready external exam，
  - cover-derived YOLO diagnostic，
  - cover-as-banner proxy diagnostic，
  - MAL 2025 overlap label sanity，
  - external calibration diagnostics，
  - remaining true-banner limitation。
