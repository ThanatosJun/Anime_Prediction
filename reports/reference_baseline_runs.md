# Reference Baseline Runs

Updated: 2026-05-12

This file records the tracked summary of reference baseline runs. Raw `.exp/`
outputs are local experiment artifacts and are intentionally not committed.
Use `reports/reference_baseline_results.csv` as the portable result table.

## Tracked Result Table

```text
reports/reference_baseline_results.csv
```

## Source Runs

| Run | Scope | Status | Notes |
|---|---|---|---|
| `.exp/baseline/results/05` | `F0-Mean`, `F0-Ridge-Meta`, `F1-RF-Meta`, `F1-GB-Meta` | local raw output | Source for completed lowest-reference and metadata-only classical results. |
| `.exp/baseline/results/14` | `F2-XGB-Concat` | local raw output | Source for completed feature-concat XGBoost results with real text/image embeddings. |

## Completed Routes

| Plan route | Baseline IDs | Completion status |
|---|---|---|
| `0. Lowest Reference / lowest floor` | `F0-Mean`, `F0-Ridge-Meta` | done |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation |

## Artifact Policy

`.exp/` is ignored because it can contain large or frequently changing
experiment outputs, predictions, and logs. Reportable numbers should be copied
into tracked files under `reports/`.
