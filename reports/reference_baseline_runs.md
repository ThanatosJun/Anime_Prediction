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
| `.exp/baseline/results/15` | `C1-Armenta-MLP` | local raw output | Source for first-pass anime-domain deep fusion adaptation results. |
| `.exp/baseline/results/16` | `T2-XGB-TextEmb` | local raw output | Source for completed text-embedding-only XGBoost results. |
| `.exp/baseline/results/17` | `I1-XGB-ImageEmb` | local raw output | Source for completed image-embedding-only XGBoost results. |

## Completed Routes

| Plan route | Baseline IDs | Completion status |
|---|---|---|
| `0. Lowest Reference / lowest floor` | `F0-Mean`, `F0-Ridge-Meta` | done |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | done as adaptation |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | done as adaptation |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP` | first-pass done as adaptation |

## C1 vs F2 Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-MLP` test R2 | Current interpretation |
|---|---:|---:|---|
| `popularity` | 0.5194 | -0.9811 | The first-pass MLP fusion head underperforms the feature-concat XGBoost floor. |
| `meanScore` | 0.0193 | -0.1173 | The first-pass MLP fusion head also underperforms on score regression. |

## Single-Modality Snapshot

| Target | `T2-XGB-TextEmb` test R2 | `I1-XGB-ImageEmb` test R2 | `F2-XGB-Concat` test R2 | Current interpretation |
|---|---:|---:|---:|---|
| `popularity` | -0.0152 | 0.0158 | 0.5194 | Text/image embeddings alone have ranking signal, but the strong result comes from the combined metadata + embedding setup. |
| `meanScore` | -0.3846 | -0.1559 | 0.0193 | Single-modality embeddings are weak for score regression; F2 remains the least weak multimodal classical reference. |

## Artifact Policy

`.exp/` is ignored because it can contain large or frequently changing
experiment outputs, predictions, and logs. Reportable numbers should be copied
into tracked files under `reports/`.
