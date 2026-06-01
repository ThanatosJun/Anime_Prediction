# Baseline Table Answer 2026-06-01

This note answers the paper-writing questions about baseline sample counts, main-table row selection, C2 claim boundary, and C3/Exp2 separation.

## A. Sample Count And Common Subset

The `n=3,087` rows are the full V2 test split from `data/fussion/fusion_meta_clean_test_v2.csv`.

The `n=2,808` rows are the strict multimodal common subset:

```text
metadata ids ∩ project text embedding ids ∩ project image embedding ids
```

For RAG rows, the pipeline additionally intersects RAG feature ids, but the current RAG feature parquet files have full test coverage, so they do not reduce the test set further.

Coverage on the V2 test split:

| artifact | test rows | missing from metadata test ids |
|---|---:|---:|
| metadata | 3,087 | 0 |
| project text embedding | 2,808 | 279 |
| project image embedding | 3,087 | 0 |
| RAG none/selective/skapp_proxy/skapp_graph | 3,087 | 0 |
| GPT-2 text embedding | 3,087 | 0 |
| ResNet-50 image embedding | 3,087 | 0 |

Therefore the 279 excluded rows are caused by missing project text embeddings, not missing image artifacts, high-resolution image artifacts, or RAG features.

The selected high-resolution multimodal rows share the exact same 2,808 test IDs for both `popularity` and `meanScore`: `F2-XGB-Concat`, `C1-Armenta-ProjectInputProxy`, `C2-ProjectInputCrossAttention`, `C2-ProjectInputRecurrentFusion`, `C3-RAG-Selective-XGB`, and `C3-ProjectInputSKAPPProxy-XGB`.

This is pipeline-guaranteed, not accidental. `BaselineFeatureStore._resolve_ids()` intersects split IDs with every required embedding/RAG artifact in the configured feature set before building the feature matrix.

`F1-RF-Meta` can run on 3,087 rows because it only requires metadata. It has now also been recomputed on the 2,808 common subset for fair main-table comparison.

## B. Main-Table Selection Rule

The main paper should not include every baseline row. The main table should include representative rows that answer distinct comparison questions:

| baseline | main-table role | reason |
|---|---|---|
| `F1-RF-Meta` | metadata-only strong baseline | Tests whether multimodal methods improve beyond strong structured metadata. |
| `F2-XGB-Concat` | simple multimodal fusion | Tests whether deep/reference fusion improves beyond early concatenation. |
| `C1-Armenta-ProjectInputProxy` | C1 representative | Same 2,808 common subset and high-res project-input setting; best for fair table comparison. |
| `C2-ProjectInputRecurrentFusion` | primary C2 representative | Keeps the C2-inspired cross-modal plus recurrent-fusion idea most completely. |
| `C2-ProjectInputCrossAttention` | optional secondary C2 row | Useful if table space allows, because it isolates the cross-attention component and is stronger on meanScore. |
| `C3-RAG-Selective-XGB` | selective retrieval baseline | Represents the selective retrieval strategy. |
| `C3-ProjectInputSKAPPProxy-XGB` | SKAPP-inspired performance row | Represents the strongest project-input SKAPP-style aggregate proxy, especially for meanScore. |

Recommended placement:

- Main table: selected rows above, with `C2-ProjectInputCrossAttention` optional depending on table space.
- Appendix: all baseline rows, including `F0`, `F1-GB`, text-only, image-only, CTNN-Lite, reconstruction variants, graph proxy, and source-exact diagnostics.
- Development/diagnostic paragraph or appendix table: `C3-SourceExact-Staged-K64`.

## C. C1/C2/C3 Version Choice

### C1

Use `C1-Armenta-ProjectInputProxy` in the main table.

Reason: it is on the same 2,808 high-res project-input common subset as F2/C2/C3, so it is the fairest row for main-table comparison. It should be described as a project-input proxy / literature-adapted reference, not exact reproduction.

Keep `C1-Armenta-ProjectInputReconstruction`, `C1-Armenta-ProjectInputProxy-ResNet50`, and `C1-Armenta-Figure2Reconstruction` in appendix or development notes. They are useful for completeness discussion, but not the cleanest main-table row.

### C2

Primary main-table row: `C2-ProjectInputRecurrentFusion`.

Reason: it is the most complete project-input C2-inspired fusion row among the high-res common-subset variants, retaining cross-modal attention plus recurrent fusion.

Optional secondary row: `C2-ProjectInputCrossAttention`.

Reason: it isolates cross-modal attention and performs better on meanScore, so it is useful if the paper can afford one extra C2 row.

Keep `C2-ProjectInputCTNNReconstruction` and `C2-ProjectInputCTNNDualVisualReconstruction` in appendix or future-work discussion unless a fresh high-res/common-subset version is selected later.

### C3

Use two C3 rows if table space allows:

- `C3-RAG-Selective-XGB`: selective retrieval strategy reference.
- `C3-ProjectInputSKAPPProxy-XGB`: SKAPP-inspired aggregate proxy and current strongest C3 meanScore row.

Do not use `C3-SourceExact-Staged-K64` as a main-table performance row. It belongs in diagnostic discussion or appendix because both targets show boundary saturation.

## D. C2 Claim Boundary

C2 is acceptable in the main paper only as a literature-adapted reference baseline.

Recommended naming:

- `C2-adapted`
- `C2 project-input proxy`
- `C2-inspired cross-modal/recurrent fusion`

Avoid:

- `C2 reproduction`
- `exact C2 reproduction`
- claims that this is the original paper's model performance on anime data

What C2 preserves:

- multimodal text-image fusion motivation
- cross-modal interaction between textual and visual representations
- recurrent/sequence-style fusion idea in `C2-ProjectInputRecurrentFusion`
- unified regression evaluation on `popularity` and `meanScore`

What C2 does not preserve:

- original movie box-office task
- original movie reviews/posters data distribution
- original target and split
- exact original encoders and training environment
- exact source-code reproduction unless a later branch produces and verifies one

Allowed claim:

> We implement a C2-inspired, literature-adapted cross-modal fusion baseline on the same anime pre-release project inputs.

Not allowed:

> We fully reproduce the original box-office revenue prediction model, or measure the original model's true performance on our task.

## E. C3 And Exp2 Boundary

Exp1 C3 rows are reference baselines. They compare external retrieval-augmented baseline families against other baseline families.

Exp2 should be reserved for the proposed framework's RAG component ablation, such as proposed No-RAG, metadata-only RAG, text-only RAG, and hybrid RAG.

Therefore, `none/sparse/dense/hybrid/selective` under C3 should not be written as the final Exp2 unless the experiment is explicitly about C3 reference-family ablation. In the current paper structure, they are baseline-family rows for Exp1 and appendix analysis.

`C3-SourceExact-Staged-K64` should be positioned as source-faithful diagnostic. It has both targets now, but both are unstable:

- `popularity`: `log_R2=-2.1272`, `factor_acc_2x=0.0901`, raw `R2=-15.0432`
- `meanScore`: `MAE=19.8518`, `acc_within_10pt=0.3061`, `R2=-4.2271`

Likely failure causes:

- `top_k=64` is an urgent reduced setting, not SKAPP's `top_k=500`.
- Target scaling/clipping is brittle under anime target distribution.
- The staged SKAPP/RRCP loss and prediction-space assumptions do not transfer cleanly.
- Original SKAPP task/data distribution differs from anime `popularity` and `meanScore`.

## F. Metrics And Authoritative Files

Main-table metrics:

- `popularity`: `Spearman_rho`, `log_MAE`, `log_R2`, `factor_acc_2x`
- `meanScore`: `Spearman_rho`, `MAE`, `R2`, `acc_within_10pt`

All selected rows can be recomputed from `test_predictions.csv`.

For `popularity`, raw `MAE` and raw `R2` should be appendix/supporting metrics because popularity is long-tailed and raw-scale metrics are dominated by extreme hits. Main text should prioritize log-space metrics and Spearman.

The interpretation `log_MAE ≈ 0.89` means a typical multiplicative error of approximately `exp(0.89) ≈ 2.43x`. This is acceptable as an intuitive explanation, but it should be phrased as an approximate geometric-scale interpretation, not exact per-sample multiplicative error.

Authoritative paper table file generated for main-table drafting:

```text
reports/paper_baseline_main_table_2026-06-01.csv
```

Supporting aggregate file:

```text
reports/reference_baseline_metrics_extended_2026-06-01.csv
```

The older files `reports/reference_baseline_v2_results.csv` and `reports/reference_baseline_v2_highres_results.csv` remain run-level source summaries, but the paper table should use the new main-table CSV to avoid mixing full-test and common-subset rows by hand.

High-res results should be used for the main multimodal table where available. In this run, high-res changes the image feature dimensionality and values, but not the selected 2,808 multimodal test IDs. The common-subset reduction is caused by project text embedding coverage, not high-res image coverage.
