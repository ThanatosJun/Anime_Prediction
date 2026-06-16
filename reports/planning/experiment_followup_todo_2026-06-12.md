# 專案回饋分析與待辦清單

日期：2026-06-12

用途：本文件把老師對期末報告的回饋拆成可執行代辦，並標示每個代辦直接屬於 Exp1、Exp2、Exp3、主框架或文件。若某項工作只是統計檢定，會在說明中補充，但不作為主要分類，避免無法分工。

## 1. 老師回饋重點拆解

### 1.1 老師肯定的部分

這些不是待辦，而是目前應保留的專案優勢：

| 老師指出的 strengths | 專案對應內容 | 後續處理 |
|---|---|---|
| Leakage-aware temporal split | 使用 temporal split 避免 future leakage | 保留為 Methodology 核心敘事 |
| Retrieval restricted to earlier titles | RAG/retrieval 不看未來作品 | 保留，避免任何實驗破壞時間限制 |
| Multiple baseline families | F1/F2/C1/C2/C3 baseline families | 保留，但主表要樣本對齊 |
| Three ablation axes | retrieval、image、temporal trend ablation | 保留，但若補 multi-seed 要重跑 Exp2 |
| Genuine out-of-domain evaluation | MAL 2025 external test | 保留，並補 calibration/quantile 說明 |
| Three interpretability methods | attention、Captum、SHAP | 保留，檢查圖表標題與說明 |
| Strong literature | Swin、SimCLR、ULMFiT、RRF、SKAPP | 保留 citation 與 claim boundary |
| Honest weak points | low MeanScore R2、negative external R2 | 保留，但要補合理解釋 |

### 1.2 老師要求改善的部分

| 老師回饋 | 問題本質 | 對應待辦 |
|---|---|---|
| Report multiple seeds with mean/std and significance test | single-run robustness 不足 | T1a、T1b、T1c、T2a、T2b、T2c |
| Re-run baselines on identical `n=3,087` | Exp1 樣本數不公平 | T4 |
| Fix mislabeled Table 9 headers | 文件表格錯誤 | T5 |
| Add CNN-vs-Swin backbone ablation | 主框架缺一個 image backbone 控制實驗 | T6a、T6b |
| Proofread awkward phrasing | 文件語句品質問題 | T7 |
| Consider quantile/calibration analysis | external scale mismatch 解釋不足 | T8a、T8b、T8c、T8d |
| Q1: `n=3,087` vs `n=2,808` 是否公平 | Exp1 headline claim 不嚴謹 | T4、T9 |
| Q2: MeanScore 是否真的可預測 | 需要區分模型能力與 temporal/popularity prior | T8、T10a、T10b、T10c、T10d、T10e |
| Q3: ablation deltas 是否跨 seed 穩定 | Exp2 fixed-seed 結論穩定性不足 | T1b、T2b |

## 2. 待辦事項總表

| ID | 代辦事項 | 歸屬 | 狀態 | 會影響其他部分嗎 |
|---|---|---|---|---|
| T1a | 補 Exp1 CARMA/baseline multiple seeds | Exp1 | 部分完成（CARMA 7-seed 已跑；baseline 多 seed 未做） | 會影響 Exp1，可能更新主表數字 |
| T1b | 補 Exp2 ablation multiple seeds | Exp2 | 未做 | 會影響 Exp2 ablation 結論穩定性 |
| T1c | 補 Exp3 external multiple seeds | Exp3 | 可選 / 未做 | 會影響 external uncertainty 說法，成本較高 |
| T2a | 補 Exp1 CARMA-vs-baseline significance test | Exp1 | 未做 | 需等 T1a 完成，不直接改模型 |
| T2b | 補 Exp2 ablation delta significance test | Exp2 | 未做 | 需等 T1b 完成，不直接改模型 |
| T2c | 補 Exp3 external key-delta significance test | Exp3 | 可選 / 未做 | 需等 T1c 或 external multi-run 結果 |
| T4 | baseline 統一 `n=3,087` 重跑 | Exp1 | 已完成 | 已影響 Exp1 與論文敘事 |
| T5 | 修正 Table 9 headers | 文件 | 未確認 / 待修 | 不影響實驗，只影響文件可信度 |
| T6a | 補 CNN-vs-Swin diagnostic ablation | 主框架 / Exp2 | 未做 | 新增 Exp2/Framework diagnostic，不一定重跑全部 |
| T6b | 若 backbone 結論改變主框架，重跑相關主實驗 | 主框架 / Exp1 / Exp2 / Exp3 | 條件式 | 只有 T6a 顯示需更換主設定時才做 |
| T7 | 全文 proofread awkward phrasing | 文件 | 待做 | 不影響實驗，但影響提交品質 |
| T8a | 補 external calibration bins | Exp3 | 已完成 | 已影響 Exp3 scale mismatch 解釋 |
| T8b | 補 external error slices | Exp3 | 已完成 | 已影響 Exp3 tail-error 解釋 |
| T8c | 補 external case examples | Exp3 | 已完成 | 已支援 success/failure case 說明 |
| T8d | 補 internal MeanScore residual slicing | Exp2 / internal diagnostic | 未做 | 可強化 Q2，不改模型 |
| T9 | 修正 Exp1 headline claim | Exp1 / 文件 | 已完成主要修正 | 依賴 T4，避免過度宣稱 |
| T10a | 用現有診斷補 MeanScore 可預測性文字解釋 | 文件 / Exp2 / Exp3 | 已完成可用版本 | 影響 Discussion，不需重跑 |
| T10b | 補 temporal/popularity prior-only baseline | Exp1 | 未做 | 會新增 Exp1 diagnostic row |
| T10c | 補 remove-prior feature ablation | Exp2 | 未做 | 會新增 Exp2 diagnostic；若改輸入群組需重跑相關 ablation |
| T10d | 補 MeanScore residual slicing | Exp3 / internal diagnostic | 部分完成 | 可強化 Q2/Q3 解釋，不改模型 |
| T10e | 補 MeanScore multi-seed ablation | Exp2 | 未做 | 會影響 Exp2 robustness 結論 |

## 3. 各待辦細節與牽動範圍

### T1a：補 Exp1 CARMA/baseline multiple seeds

- 歸屬：Exp1。
- 性質：統計穩定性檢查。
- 狀態：部分完成。CARMA 主模型已跑 7 個 seed（Run22–28，seeds 42/43/44/45/247135/610172/796445，`src_2/rerun_seeds.py`，摘要 `runs/rerun_seeds_summary.json`），test 與 val 的 mean±std 已記於 `src_2/README.md` 的 Seed Robustness 段。**baseline（F2/C2/C3）多 seed 尚未做。**
- 目的：回應老師「single fixed seed 不足」的問題。
- 建議對象：Run22-style CARMA、F2、C2-Recurrent、C3-RAG-XGB。
- 產出：key metrics 的 mean/std。
- 牽動範圍：
  - 會影響 Exp1 主表，因為原本主表是 fixed seed 結果。
  - 若 CARMA 或 baseline 排名變動，論文結論要跟著改。
  - 不一定需要改主框架程式，但需要固定實驗腳本與輸出格式。

### T1b：補 Exp2 ablation multiple seeds

- 歸屬：Exp2。
- 性質：統計穩定性檢查。
- 目的：回應 Q3，確認 retrieval/image/temporal trend 的 ablation deltas 不是 seed noise。
- 建議對象：full model vs remove retrieval、remove image、remove temporal trend。
- 牽動範圍：
  - 會影響 Exp2。
  - 若 ablation 差距跨 seed 不穩，Exp2 結論要改成更保守。
  - 不會直接影響 Exp3，除非因此改了主框架 checkpoint。

### T1c：補 Exp3 external multiple seeds

- 歸屬：Exp3。
- 性質：外部測試 seed uncertainty，可選。
- 目的：若要更嚴格檢查 external ranking/error 是否跨 seed 穩定，可以補外部 multi-seed。
- 牽動範圍：
  - 會影響 Exp3 uncertainty 說法。
  - 成本較高，因為需要多個 CARMA checkpoints 或多 seed external predictions。
  - 目前不是老師 Q1/Q3 的最低必要回應。

### T2a：補 Exp1 CARMA-vs-baseline significance test

- 歸屬：Exp1。
- 性質：統計檢定。
- 目的：判斷 CARMA-vs-baseline key deltas 是否跨 seed 穩定。
- 前提：必須先完成 T1a，否則沒有 seed-level 分布可檢定。
- 牽動範圍：
  - 會影響 Exp1 的可信度說法。
  - 不會直接改模型，也不會改資料切分。

### T2b：補 Exp2 ablation delta significance test

- 歸屬：Exp2。
- 性質：統計檢定。
- 目的：判斷 Exp2 ablation deltas 是否跨 seed 穩定。
- 前提：必須先完成 T1b。
- 牽動範圍：
  - 會影響 Exp2 的可信度說法。
  - 不會直接改模型，也不會改資料切分。

### T2c：補 Exp3 external key-delta significance test

- 歸屬：Exp3。
- 性質：統計檢定，可選。
- 目的：若補了 external multi-seed，就可檢定 external CARMA-vs-baseline key deltas。
- 前提：需要 T1c 或等價的 external multi-run 結果。
- 牽動範圍：
  - 只影響 Exp3 uncertainty 說法。
  - 不改主框架。

### T4：baseline 統一 `n=3,087` 重跑

- 歸屬：Exp1。
- 目的：回應 Q1，避免 CARMA 用 full test、baseline 用 common subset 的不公平比較。
- 狀態：已完成。
- 目前結果用途：
  - Exp1 主比較應使用 `n=3,087` full temporal test。
  - 舊 `n=2,808` complete-case baseline 只能當歷史紀錄。
- 牽動範圍：
  - 已影響 Exp1 表格與論文敘事。
  - 不影響 Exp3，因為 Exp3 有自己的 MAL rows 對齊。

### T5：修正 Table 9 headers

- 歸屬：文件。
- 目的：修正老師點名的表格標題錯誤。
- 狀態：待確認最終論文版本是否已修。
- 牽動範圍：
  - 不影響任何實驗。
  - 但若不修，會直接影響文件可信度。

### T6a：補 CNN-vs-Swin diagnostic ablation

- 歸屬：主框架 / Exp2。
- 目的：回應老師指出 missing CNN-vs-Swin backbone ablation。
- 關鍵限制：
  - 這不是單純拿 C1/C2 或 ResNet proxy 比較就能回答。
  - 合格做法是在同一個 CARMA architecture 中，只替換 image backbone，其他設定不變。
- 牽動範圍：
  - 會新增主框架 ablation。
  - 若結果被納入正文，可能需要更新 Methodology、Exp2、Future Work。
  - 若只作 diagnostic ablation，則不需要重跑所有主實驗。

### T6b：若 backbone 結論改變主框架，重跑相關主實驗

- 歸屬：主框架 / Exp1 / Exp2 / Exp3。
- 目的：如果 T6a 顯示應更換主 image backbone，必須重新確認主框架結果。
- 觸發條件：
  - 只有當 T6a 的結果導致主框架設定改變時才需要。
- 牽動範圍：
  - 可能需要重跑 Exp1。
  - 可能需要重跑 Exp2。
  - 可能需要重跑 Exp3。
  - 若 T6a 只是補 diagnostic，不改主設定，則不需要做 T6b。

### T7：全文 proofread awkward phrasing

- 歸屬：文件。
- 目的：修正老師點名的語句，例如 `A temporal design to light decrease concept drift`。
- 牽動範圍：
  - 不影響實驗。
  - 可能影響 Introduction、Methodology、Experiments、Conclusion 的文字一致性。

### T8a：補 external calibration bins

- 歸屬：Exp3。
- 目的：回應老師對 external scale mismatch 的疑問。
- 狀態：已完成。
- 產出：
  - `mal2025_external_calibration_summary.csv`
  - `mal2025_external_calibration_bins.csv`
- 牽動範圍：
  - 已影響 Exp3 解釋。
  - 不改模型，不需要重跑 Exp1/Exp2。

### T8b：補 external error slices

- 歸屬：Exp3。
- 目的：分析 external errors 在 popularity/score quantiles、release period、format、source、tail samples 的分布。
- 狀態：已完成。
- 產出：
  - `mal2025_external_error_slices.csv`
- 牽動範圍：
  - 已影響 Exp3 tail-error 解釋。
  - 不改模型，不需要重跑 Exp1/Exp2。

### T8c：補 external case examples

- 歸屬：Exp3。
- 目的：提供 success/failure cases，輔助解釋 external ranking successes、high-popularity underestimation 與 large score errors。
- 狀態：已完成。
- 產出：
  - `mal2025_external_case_examples.csv`
  - `mal2025_external_diagnostics.md`
- 牽動範圍：
  - 已影響 Exp3 解釋。
  - 不改模型，不需要重跑 Exp1/Exp2。

### T8d：補 internal MeanScore residual slicing

- 歸屬：Exp2 / internal diagnostic。
- 目的：補 internal test 上的 MeanScore residual 分層，和 external slices 搭配回答 Q2。
- 狀態：未做。
- 牽動範圍：
  - 可強化 Discussion 與 Exp2 解釋。
  - 不改主框架。

### T9：修正 Exp1 headline claim

- 歸屬：Exp1 / 文件。
- 目的：避免 `lowest reported error` 被老師質疑樣本不公平。
- 狀態：已完成主要修正。
- 目前可用說法：
  - CARMA 在 meanScore MAE 與 10-point accuracy 上有最清楚優勢。
  - Popularity 是 competitive，不是所有指標全面最佳。
- 牽動範圍：
  - 已依賴 T4 的 `n=3,087` 重跑結果。
  - 不需改主框架。

### T10a：用現有診斷補 MeanScore 可預測性文字解釋

- 歸屬：文件 / Exp2 / Exp3。
- 目的：用既有實驗回答 Q2：「MeanScore R2 低，模型到底是在預測分數，還是在吃 temporal/popularity prior？」
- 狀態：已完成可用版本。
- 目前可用證據：
  - Exp2 ablation 可說明 retrieval/image/temporal trend 各自貢獻。
  - External calibration/error slices 可說明 R2 弱與 scale mismatch、tail errors 有關。
  - SHAP/Captum/attention 可輔助說明模型不是只靠單一 temporal prior。
- 牽動範圍：
  - 主要影響 Discussion。
  - 不需要重跑任何實驗。

### T10b：補 temporal/popularity prior-only baseline

- 歸屬：Exp1。
- 目的：建立只使用 temporal/popularity prior 的 baseline，回答 MeanScore 是否只是靠 prior。
- 狀態：未做。
- 牽動範圍：
  - 會新增 Exp1 diagnostic row。
  - 不需要改 CARMA 主框架。

### T10c：補 remove-prior feature ablation

- 歸屬：Exp2。
- 目的：移除 popularity-like、sequel、prior performance features，檢查 MeanScore 是否仍可預測。
- 狀態：未做。
- 牽動範圍：
  - 會新增 Exp2 diagnostic。
  - 若現有 pipeline 沒有 feature group 開關，需補資料/輸入遮罩腳本。
  - 不一定改模型架構，但可能要重跑相關 ablation。

### T10d：補 MeanScore residual slicing

- 歸屬：Exp3 / internal diagnostic。
- 目的：按年份、popularity quantile、source、format 分析 MeanScore residual，說明模型錯在哪些區間。
- 狀態：部分完成。
- 牽動範圍：
  - External error slices 已完成一部分。
  - 若要完整回答 Q2，可補 internal test residual slicing。
  - 不需要改主框架。

### T10e：補 MeanScore multi-seed ablation

- 歸屬：Exp2。
- 目的：確認 MeanScore component contribution 是否跨 seed 穩定。
- 狀態：未做。
- 牽動範圍：
  - 會影響 Exp2 robustness 結論。
  - 不直接影響 Exp3，除非因此改主 checkpoint。

## 4. 優先順序

| 優先級 | 代辦 | 原因 |
|---|---|---|
| P0 | T5：修 Table 9 headers | 文件錯誤最容易被扣分，且成本低 |
| P0 | T7：proofread awkward phrasing | 成本低，直接改善可讀性 |
| P1 | T1a：Exp1 multiple seeds | 老師明確點名，影響 Exp1 robustness |
| P1 | T1b：Exp2 ablation multiple seeds | 直接回應 Q3 |
| P1 | T2a/T2b：seed-level significance tests | 需等 T1a/T1b 後才能做 |
| P2 | T10b：prior-only baseline | 回應 Q2，屬 Exp1 diagnostic |
| P2 | T10c：remove-prior feature ablation | 回應 Q2，屬 Exp2 diagnostic |
| P2 | T10d / T8d：MeanScore residual slicing | 回應 Q2，屬 Exp3/internal diagnostic |
| P2 | T6a：CNN-vs-Swin diagnostic ablation | 屬主框架延伸，成本較高 |

## 5. 目前不要再用的舊依據

- 舊 `n=2,808` complete-case baseline 不作為 Exp1 主表。
- 舊 paper handoff 若與目前 `n=3,087` 結果衝突，以 sample alignment reports 為準。
- C1/C2 不作 exact reproduction 宣稱。
- Cover-as-banner proxy 不作 true banner image 宣稱。

## 6. 查數字時看哪裡

- `reports/experiments/sample_alignment/eval_sample_alignment_report_2026-06-11.md`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.csv`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.md`
- `reports/experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.csv`
- `reports/experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.md`
- `reports/experiments/sample_alignment/mal2025_external_diagnostics.md`
- `reports/experiments/sample_alignment/mal2025_external_error_slices.csv`
- `reports/experiments/sample_alignment/mal2025_external_case_examples.csv`
- `reports/experiments/sample_alignment/followup_paired_bootstrap_tests.csv`
- `reports/experiments/sample_alignment/followup_external_paired_bootstrap_tests.csv`
- `reports/experiments/sample_alignment/followup_experiment_statistics.md`
- `reports/experiments/sample_alignment/run22_artifact_manifest.md`
- `reports/paper/paper_user_sections_complete_draft_2026-06-01.md`
