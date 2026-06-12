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

1. Run multiple seeds for CARMA and the most important baselines.
   - Suggested scope: Run22-style CARMA, F2, C2-Recurrent, C3-RAG-XGB.
   - Report mean/std for key metrics.
2. Add strict CNN-vs-Swin backbone ablation if the image-encoder contribution needs a
   stronger response to review feedback.
3. Keep C1/C2 claim boundaries explicit.
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

Remaining:

1. Keep the external conclusion conservative.
   - CARMA is strongest for external score MAE/10-point accuracy.
   - Popularity and score ranking are mixed; F2/C2/C3 can be stronger on
     Spearman depending on the split.
2. A true banner-like branch remains future work.
   - The completed cover-as-banner proxy shows that naive cover duplication is
     not a substitute for real banner information.

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
