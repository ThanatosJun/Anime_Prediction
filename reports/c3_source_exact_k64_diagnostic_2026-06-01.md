# C3 Source-Exact K64 Diagnostic 2026-06-01

This note records the first completed `c3_source_exact_pipeline.py` K64 diagnostic runs.

## Run

- Popularity run directory: `.exp/baseline/results/v2_source_exact_c3_pop_urgent_k64`
- meanScore run directory: `.exp/baseline/results/v2_source_exact_c3_urgent_k64_meanscore`
- Pipeline: `source_exact_staged`
- Targets: `popularity`, `meanScore`
- Dataset tensors: `.exp/baseline/skapp_full/dataset_v2`
- Retrieval tensor shape:
  - train: `(13321, 64, 768)`
  - val: `(2918, 64, 768)`
  - test: `(3087, 64, 768)`
- Important caveat: this is a `top_k=64` urgent run, not the SKAPP source default `top_k=500`.

## Popularity Test Metrics

| metric | value |
|---|---:|
| MAE | 99140.0794 |
| RMSE | 143347.7822 |
| raw R2 | -15.0432 |
| Spearman rho | 0.3170 |
| Pearson r | 0.2336 |
| log_MAE | 3.4361 |
| log_R2 | -2.1272 |
| factor_acc_2x | 0.0901 |

## Popularity Prediction Diagnostics

The output is highly saturated after clipping to the train-set model-space range.

| statistic | target | prediction |
|---|---:|---:|
| count | 3087 | 3087 |
| mean | 15182.4362 | 106573.8756 |
| std | 35794.4438 | 113188.8699 |
| min | 25.0000 | 25.0000 |
| 50% | 918.0000 | 10952.5850 |
| 75% | 10119.5000 | 231528.9800 |
| max | 231528.9000 | 231528.9800 |

Most frequent predictions:

| prediction | count |
|---:|---:|
| 231528.980000 | 1366 |
| 24.999998 | 1001 |

## meanScore Test Metrics

| metric | value |
|---|---:|
| MAE | 19.8518 |
| RMSE | 24.5982 |
| raw R2 | -4.2271 |
| Spearman rho | 0.1155 |
| Pearson r | 0.0850 |
| acc_within_10pt | 0.3061 |

## meanScore Prediction Diagnostics

The meanScore run shows the same boundary-saturation pattern as the popularity run.

| statistic | target | prediction |
|---|---:|---:|
| count | 3087 | 3087 |
| mean | 65.4302 | 63.4586 |
| std | 10.7608 | 22.9694 |
| min | 27.0000 | 27.0000 |
| max | 85.0000 | 85.0000 |

Most frequent predictions:

| prediction | count |
|---:|---:|
| 85.000000 | 905 |
| 27.000000 | 733 |

## Interpretation

These runs should not be promoted as the final C3 external baseline. They are useful diagnostic results showing that a more source-faithful staged SKAPP pipeline is currently unstable under the anime-domain mapping. The failure mode is not merely low accuracy: predictions collapse toward clipping boundaries for a large fraction of the test set.

Recommended paper positioning:

- Keep `C3-RAG-Selective-XGB` and `C3-ProjectInputSKAPPProxy-XGB` as the current C3 performance references.
- Mention `C3-SourceExact-Staged-K64` as an ongoing source-faithful diagnostic, not as a completed main-table external baseline.
- Future work should rerun the source-faithful path with `top_k=500`, calibration checks, and possibly a target-space/loss redesign before treating it as a final SKAPP reproduction.
