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

## Notes

Use stable relative paths when referencing report files from `docs/`, scripts, or other reports. If a report is moved, update the references in the same commit.
