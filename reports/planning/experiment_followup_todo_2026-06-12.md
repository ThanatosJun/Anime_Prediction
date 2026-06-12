# Experiment Follow-up TODO

Date: 2026-06-12

Branch: `feature/experiment-followups`

## Current Authoritative Result Files

Use these files for the revised paper narrative and future reporting:

- `reports/experiments/sample_alignment/eval_sample_alignment_report_2026-06-11.md`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.csv`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.md`
- `reports/experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.csv`
- `reports/experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.md`
- `reports/experiments/sample_alignment/mal2025_overlap_label_sanity.csv`
- `reports/experiments/sample_alignment/mal2025_external_calibration_summary.csv`
- `reports/experiments/sample_alignment/mal2025_external_calibration_bins.csv`
- `reports/experiments/sample_alignment/mal2025_external_diagnostics.md`
- `reports/experiments/sample_alignment/followup_paired_bootstrap_tests.csv`
- `reports/experiments/sample_alignment/followup_external_paired_bootstrap_tests.csv`
- `reports/experiments/sample_alignment/followup_external_paired_bootstrap_tests.md`
- `reports/experiments/sample_alignment/followup_image_proxy_diagnostics.csv`
- `reports/experiments/sample_alignment/followup_experiment_statistics.md`
- `reports/experiments/sample_alignment/run22_artifact_manifest.md`
- `reports/paper/paper_user_sections_complete_draft_2026-06-01.md`

Older baseline reports under `reports/baselines/` and older paper handoff files
under `reports/paper/` are historical artifacts. They may still mention the old
`n=2,808` complete-case baseline setting and should not be used as the current
main-table narrative.

## Exp1: Baseline Comparison

Completed:

- Recomputed representative F1/F2/C1/C2/C3 baselines on the same full internal
  temporal test set as CARMA: `n=3,087`.
- Added C1/C2 rows back into the internal Exp1 comparison.
- Updated the paper narrative away from the old `n=2,808` complete-case table.
- Added paired bootstrap diagnostics for CARMA-vs-baseline headline deltas.

Remaining:

1. Add a compact paired-bootstrap summary table for the paper/report.
   - Suggested rows: CARMA vs F2, CARMA vs C2-Recurrent, CARMA vs C3.
   - Suggested metrics: popularity `log_MAE`/Spearman and meanScore MAE/Spearman.
   - Purpose: show which differences are stable and which are effectively tied.
2. Keep the Exp1 conclusion target-specific.
   - `meanScore`: CARMA has the clearest MAE and 10-point accuracy advantage.
   - `popularity`: CARMA is competitive, but F2/C2/C3 remain strong on some
     ranking/error metrics.
   - Avoid the claim that CARMA is the best on every popularity metric.
3. Keep the baseline role table compact and explicit.
   - F1: metadata-only strong floor.
   - F2: simple multimodal concatenation floor.
   - C1/C2: literature-adapted fusion proxies.
   - C3: retrieval reference baseline.
4. Keep C1/C2 claim boundaries explicit.
   - Use `literature-adapted`, `project-input proxy`, or `C2-inspired`.
   - Do not describe them as exact reproductions of the original papers.

## Exp2: CARMA Ablation

Completed:

- Main ablation axes are already available in the project narrative:
  retrieval, image, and temporal trend removal.
- Current interpretation: removing retrieval, image, or temporal trend increases
  error, so these components contribute to CARMA.
- Added paired bootstrap diagnostics for the main ablation deltas.

Remaining:

1. Add multi-seed mean/std for the key ablation deltas.
   - Priority deltas: full model vs remove retrieval, remove image, remove
     temporal trend.
2. Confirm Exp2 charts and captions use the same target/metric direction.
   - `log_MAE` and `MAE`: lower is better.
   - Avoid wording that implies higher error is an improvement.
3. If time permits, add a small calibration/error-distribution view for
   `meanScore`, because internal R2 remains modest.

## Exp3: MAL External Test

Completed:

- Recomputed Run22 external inference on MAL 2025 local-ready splits.
- Recomputed F1/F2/C1/C2/C3 baselines on the same MAL rows.
- Added cover-derived YOLO diagnostic splits and reran CARMA plus F1/F2/C1/C2/C3.
- Confirmed the image encoder artifact exists under
  `src_2/component_image/model-image/best/`.
- Added MAL 2025 overlap label sanity check.
  - Popularity Spearman: `0.9836`.
  - Score Spearman: `0.9446`.
- Added Run22 external calibration and prediction-quantile diagnostics.
- Added cover-as-banner proxy splits and reran CARMA plus F1/F2/C1/C2/C3.
  - This is a diagnostic proxy only; MAL 2025 still has no true banner images.
- Added external paired-bootstrap diagnostics for CARMA-vs-F2/C2/C3 on the
  same MAL rows.

Remaining:

1. Add external error slicing.
   - Suggested slices: MAL popularity quantiles, MAL score quantiles, release
     year, format, source, and high-popularity tail.
   - Purpose: identify where the external transfer degrades rather than only
     reporting aggregate metrics.
2. Add a compact calibration/quantile table or figure.
   - Use existing prediction-quantile bins.
   - Explain why Spearman can remain useful while external R2 is weak or
     negative.
3. Optionally add a small success/failure case table.
   - Include high-confidence ranking successes.
   - Include high-popularity underestimation examples.
   - Include large score-error examples.
4. Keep the external conclusion conservative.
   - CARMA is strongest for external score MAE/10-point accuracy.
   - Popularity and score ranking are mixed; F2/C2/C3 can be stronger on
     Spearman depending on the split.
5. A true banner-like branch remains future work.
   - The completed cover-as-banner proxy shows that naive cover duplication is
     not a substitute for real banner information.

## Global Robustness, Not Exp1-specific

These tasks improve overall evidence quality but should not be described as
Exp1-only work:

1. Run multiple seeds for CARMA and the most important baselines.
   - Affects Exp1 baseline comparison and Exp2 ablation reliability.
   - Suggested scope: Run22-style CARMA, F2, C2-Recurrent, C3-RAG-XGB.
   - Report mean/std for key metrics.
2. Add seed-level significance tests after multi-seed runs exist.
   - The current paired bootstrap checks fixed prediction artifacts only.
   - It does not replace seed-level uncertainty.

## Main Framework Remaining Work, Out of Scope for Exp1/Exp3 Cleanup

These tasks require model or architecture work and should not be mixed with the
current Exp1/Exp3 evaluation-cleanup track:

1. Add a strict CNN-vs-Swin backbone ablation inside the same CARMA architecture.
   - Existing artifacts support image-source ablations and ResNet/CNN
     literature proxy diagnostics.
   - They do not provide a one-variable CARMA CNN-vs-Swin replacement.
2. Train or acquire a true banner-like external branch.
   - The cover-as-banner proxy is only a missing-modality diagnostic.
   - A valid branch would need real banner data or a deliberately trained banner
     imputation method.

## Report Context Decisions

- Current paper/report narrative should use `n=3,087` for Exp1 main internal
  comparisons.
- The old `n=2,808` complete-case baseline files should be treated as historical
  development artifacts, not current main evidence.
- Exp3 should distinguish:
  - main no-YOLO MAL local-ready external exam,
  - cover-derived YOLO diagnostic,
  - cover-as-banner proxy diagnostic,
  - MAL 2025 overlap label sanity,
  - external calibration diagnostics,
  - remaining true-banner limitation.
