# Reports Directory

This directory stores project reports, diagnostics, paper-facing assets, and experiment summaries.

## Layout

- `paper/`: paper-facing tables, answers, and section drafts.
- `baselines/`: reference baseline runs, metrics, reproduction notes, and alignment audits.
- `external/`: external dataset evaluation summaries.
- `diagnostics/`: focused diagnostic reports, including C3/SKAPP and source-faithful checks.
- `experiments/`: focused experiment comparisons that are not part of the main baseline archive.
- `eda/`: target correlation summaries and related EDA artifacts.
- `text_branch/`: text branch findings and metrics.
- `planning/`: meeting briefs, cleanup plans, and coordination notes.
- `figures/`: generated report figures. This path is kept stable because scripts write outputs here.

## Current Paper-facing Sources

For the revised Exp1-Exp3 narrative, use the sample-alignment reports as the
current source of truth:

- `experiments/sample_alignment/eval_sample_alignment_report_2026-06-11.md`
- `experiments/sample_alignment/carma_tensor_aligned_metrics.csv`
- `experiments/sample_alignment/carma_tensor_aligned_metrics.md`
- `experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.csv`
- `experiments/sample_alignment/mal2025_yolo_diagnostic_metrics.md`
- `experiments/sample_alignment/mal2025_external_diagnostics.md`
- `experiments/sample_alignment/mal2025_overlap_label_sanity.csv`
- `experiments/sample_alignment/mal2025_external_calibration_summary.csv`
- `experiments/sample_alignment/mal2025_external_calibration_bins.csv`
- `experiments/sample_alignment/followup_paired_bootstrap_tests.csv`
- `experiments/sample_alignment/followup_image_proxy_diagnostics.csv`
- `experiments/sample_alignment/followup_experiment_statistics.md`
- `experiments/sample_alignment/run22_artifact_manifest.md`
- `paper/paper_user_sections_complete_draft_2026-06-01.md`
- `planning/experiment_followup_todo_2026-06-12.md`

Older baseline handoff files may describe the former `n=2,808` complete-case
setting. Treat those files as historical development artifacts, not as the
current main-table evidence. The current Exp1 comparison uses the full internal
temporal test set (`n=3,087`) for both CARMA and representative baselines.
The current Exp3 diagnostics distinguish no-YOLO, cover-derived YOLO, and
cover-as-banner proxy MAL 2025 variants; the proxy variant is not a true banner
evaluation.

## Notes

Use stable relative paths when referencing report files from `docs/`, scripts, or other reports. If a report is moved, update the references in the same commit.
