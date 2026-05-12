# Text Embedding Model Performance Report

Date: 2026-05-08
Project: Anime_Prediction (text branch)

## 1. Scope
This report summarizes performance for six text embedding models using:
- Full split embedding generation (`train`, `val`, `test`)
- Same downstream regressor for all models (`Ridge`, alpha=1.0)
- Quality metrics: `MAE`, `RMSE`, `Spearman`
- Targets: `popularity`, `meanScore`

The ranking is based on validation metrics (selection view), with test metrics shown as holdout behavior.

## 2. Data Sources
- Quality ranking CSV: `reports/text_branch_quality_ranked_full.csv`
- Quality markdown summary: `reports/text_branch_quality_ranking_full.md`
- Speed/throughput CSV: `reports/text_embedding_experiment_compare_full.csv`
- Per-model quality JSONs: `reports/text_branch_metrics_*_full.json`

## 3. Overall Ranking (Quality)
Ranking key:
- `selection_rank_avg`: average rank across validation metrics
- Lower rank is better

| Overall Rank | Model Key | Model Name | Selection Rank Avg | Test Rank Avg |
| --- | --- | --- | ---: | ---: |
| 1 | `e5_base` | `intfloat/e5-base-v2` | 1.00 | 1.67 |
| 2 | `e5_small` | `intfloat/e5-small-v2` | 2.17 | 1.83 |
| 3 | `bge_base` | `BAAI/bge-base-en-v1.5` | 3.33 | 3.83 |
| 4 | `multilingual_minilm` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | 4.17 | 4.50 |
| 5 | `minilm_l6` | `sentence-transformers/all-MiniLM-L6-v2` | 5.17 | 4.50 |
| 6 | `bge_small` | `BAAI/bge-small-en-v1.5` | 5.17 | 4.67 |

## 4. Test-Set Quality Snapshot
| Model Key | popularity RMSE | popularity Spearman | meanScore RMSE | meanScore Spearman |
| --- | ---: | ---: | ---: | ---: |
| `e5_base` | 32060.13 | 0.6172 | 13.1309 | 0.2525 |
| `e5_small` | 33020.09 | 0.5967 | 13.0786 | 0.2406 |
| `bge_base` | 33511.68 | 0.5565 | 13.1882 | 0.2278 |
| `multilingual_minilm` | 33910.52 | 0.5379 | 13.1132 | 0.2054 |
| `minilm_l6` | 34055.32 | 0.5408 | 13.1228 | 0.2152 |
| `bge_small` | 33791.43 | 0.5616 | 13.2509 | 0.2209 |

## 5. Runtime / Throughput (Full Run)
| Model Key | Total Duration (s) | Rows / Second |
| --- | ---: | ---: |
| `minilm_l6` | 198.71 | 91.73 |
| `multilingual_minilm` | 293.63 | 62.08 |
| `bge_small` | 365.35 | 49.89 |
| `e5_small` | 372.38 | 48.95 |
| `e5_base` | 1032.97 | 17.65 |
| `bge_base` | 1059.90 | 17.20 |

## 6. Metric Interpretation
### popularity Spearman
`popularity` Spearman measures ranking quality rather than exact numeric accuracy. It answers the question: when one anime is truly more popular than another, does the model also rank it higher? Higher is better.

For this project, this is one of the most meaningful metrics because popularity is often more useful as an ordering signal than as an exact predicted number. A model with stronger popularity Spearman is better at preserving the relative popularity structure across titles.

### meanScore RMSE
`meanScore` RMSE measures numeric prediction error for score values, while penalizing large mistakes more heavily than small ones. Lower is better.

This metric matters when the goal is to predict final score values as accurately as possible. If a model has lower meanScore RMSE, it is making fewer severe score prediction errors.

### meanScore Spearman
`meanScore` Spearman measures ranking quality for score rather than exact numeric closeness. It asks whether the model tends to place higher-scored anime above lower-scored anime. Higher is better.

This is useful when relative quality ordering matters more than exact score prediction. A model may have imperfect numeric predictions but still be valuable if it preserves the score ranking well.

### How to read these metrics together
- Strong `popularity` Spearman means the text encoder is good at capturing audience interest, hype, or discoverability.
- Lower `meanScore` RMSE means the model is better at predicting exact final score values.
- Strong `meanScore` Spearman means the model is better at separating relatively stronger versus weaker titles even when exact values are imperfect.

In the current experiments, `popularity` Spearman is consistently stronger than `meanScore` Spearman. This suggests anime descriptions contain more information about expected audience popularity than about final review score.

## 7. Interpretation
### Best pure quality
`e5_base` is the strongest overall quality model.
- Best ranking on validation aggregate
- Best test `popularity` RMSE and Spearman
- Best test `meanScore` Spearman

### Best quality/speed compromise
`e5_small` is the best compromise.
- Second-best quality
- Roughly 3x faster than `e5_base` in this pipeline
- Good candidate for iterative experiments and ablations

### Fastest baseline
`minilm_l6` remains the fastest by a wide margin.
- Useful when rapid iteration is more important than top-end quality
- Quality is materially lower than `e5_base`/`e5_small`

## 8. Recommendation
Primary recommendation:
1. Use `e5_base` for final quality-focused runs and reporting.

Practical engineering recommendation:
1. Use `e5_small` during development loops.
2. Switch to `e5_base` for final artifacts and paper-quality benchmarks.

## 9. Findings
### Model choice meaningfully changes downstream quality
The text encoder is not a cosmetic implementation detail. Under the same Ridge baseline, different embedding models produced materially different `RMSE` and `Spearman` values, especially for `popularity`. This means the semantic structure captured by the embedding model directly affects downstream predictive usefulness.

### Better semantic encoders help more on `popularity` than on `meanScore`
Across all experiments, the strongest gains appeared on the `popularity` target. The best-performing models improved both ranking quality (`Spearman`) and absolute error (`RMSE`) more clearly for `popularity` than for `meanScore`. This suggests description text carries stronger information about audience attention, hype, or discoverability than about final score quality.

### The `e5` family is the best fit for this task
Both `e5_base` and `e5_small` outperformed the other families overall. This indicates that, for the current anime-description regression setup, the `e5` embedding space aligns better with the downstream prediction task than the tested `bge` or MiniLM variants.

### Larger models help, but with diminishing returns per unit of compute
`e5_base` delivered the best overall quality, but its runtime cost was much higher than the smaller models. `e5_small` preserved most of the quality benefit while being substantially faster, making it the best practical compromise for repeated experiments.

### Fast models are useful, but mainly for iteration speed
`minilm_l6` remains valuable because it is extremely fast and easy to use for smoke tests, debugging, and pipeline validation. However, its lower downstream quality means it should be treated as a development baseline rather than the preferred final encoder.

### Multilingual support is not automatically an advantage here
`multilingual_minilm` was faster than the large base models, but it did not outperform the best English-focused encoders. This suggests multilingual capacity is not the main bottleneck for the current dataset, at least under the present evaluation setup.

### Practical project takeaway
For this project, the clean operating strategy is:
1. Use `e5_small` during development, ablations, and fusion iteration.
2. Use `e5_base` when generating final text artifacts and reporting best text-branch results.

## 10. Caveats
- Rankings are based on a text-only Ridge baseline. Fusion models may change relative ordering.
- Spearman and error metrics are target-dependent; if your final objective weights `popularity` more heavily, use weighted ranking for final selection.
- Current results were generated on CPU; GPU acceleration can change runtime but not quality ordering.
