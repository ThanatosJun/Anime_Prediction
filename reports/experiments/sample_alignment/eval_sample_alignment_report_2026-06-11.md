# Evaluation Sample Alignment Report

Date: 2026-06-11

Branch: `feature/eval-sample-alignment`; follow-up diagnostics added on
`feature/experiment-followups`

## Purpose

This follow-up addresses the presentation feedback that CARMA and reference
baselines were not always evaluated on identical sample sets. The priority is to
answer two questions:

1. Internal alignment: do representative baselines still support the CARMA
   claims when evaluated on the full temporal test set (`n=3,087`)?
2. External alignment: how do CARMA and baselines compare on the exact same MAL
   2025 rows?

## Main Reproduction Commands

```bash
python scripts/experiments/run_carma_tensor_aligned_baselines.py
```

This script flattens the actual tensors returned by
`src_2.fussion_training.dataset.AnimeDataset`, including:

- `meta_feat`
- `text_emb`
- `image_emb` and `image_mask`
- `rag_meta`, `rag_text`, `rag_image`, and `rag_mask`

No text, image, or RAG embeddings are regenerated. Existing embedding artifacts
from the CARMA pipeline are reused directly.

Required external artifact:

- GitHub Release: `https://github.com/ThanatosJun/Anime_Prediction/releases/tag/A7`
- Asset: `final_project.zip`
- SHA256: `4e537ff84978e29ea9fcfbee18bdc8e993a2ccbb0f101c62246e5beb99e20ee9`
- Restore path: extract the ZIP at the repository root so it creates
  `final_project/`, including `final_project/runs/22`.

Main outputs:

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
- `reports/experiments/sample_alignment/carma_tensor_predictions/`
- `data/external_transformed/run22_mal2025_popularity_local_ready_metrics.json`
- `data/external_transformed/run22_mal2025_dual_local_ready_metrics.json`
- `data/external_transformed/run22_mal2025_popularity_local_ready_yolo_metrics.json`
- `data/external_transformed/run22_mal2025_dual_local_ready_yolo_metrics.json`
- `data/external_transformed/run22_mal2025_popularity_local_ready_yolo_coverbanner_metrics.json`
- `data/external_transformed/run22_mal2025_dual_local_ready_yolo_coverbanner_metrics.json`

Run22 external inference can be reproduced with:

```bash
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready
```

The cover-derived YOLO diagnostic can be reproduced with:

```bash
python scripts/external/build_mal2025_yolo_image_embeddings.py --model-path src_2/component_image/model-image/best --splits mal2025_popularity_local_ready mal2025_dual_local_ready --suffix yolo --batch-size 64
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready_yolo --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready_yolo
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready_yolo --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready_yolo
```

The cover-as-banner proxy diagnostic can be reproduced with:

```bash
python scripts/external/build_mal2025_cover_banner_proxy.py
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready_yolo_coverbanner --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready_yolo_coverbanner
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready_yolo_coverbanner --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready_yolo_coverbanner
python scripts/experiments/run_carma_tensor_aligned_baselines.py --splits mal2025_popularity_local_ready_yolo_coverbanner mal2025_dual_local_ready_yolo_coverbanner --targets popularity meanScore
```

External label sanity and calibration diagnostics can be reproduced with:

```bash
python scripts/external/analyze_mal2025_external_diagnostics.py
```

Follow-up paired bootstrap and image-proxy diagnostics can be reproduced with:

```bash
python scripts/experiments/analyze_followup_experiment_statistics.py --n-boot 500
```

The YOLO diagnostic requires the image encoder artifact under
`src_2/component_image/model-image/best/`, including both `config.json` and
`model.safetensors`. The local artifact used here has SHA256
`86bd98d5dbf8022765b67b0d69c3a78ebe3d69b051c05ef87e156e310f1bc020`.

Legacy diagnostic outputs from earlier adapters are still kept, but should not
be used as the main paper comparison:

- `sample_aligned_baseline_metrics.csv`
- `external_sample_aligned_metrics.csv`

## Internal Full Test Alignment

All baseline rows below use the same internal temporal test split as CARMA:
`n=3,087`. The baseline features are built from CARMA's own tensor inputs, not
from the older baseline-only artifacts.

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman | Score MAE | Score 10pt acc | Score Spearman |
|---|---:|---:|---:|---:|---:|---:|
| F1-RF-Meta-CARMATensor | 0.9731 | 0.4655 | 0.8326 | 8.8547 | 0.6281 | 0.5171 |
| F2-XGB-Concat-CARMATensor | 0.8799 | 0.4885 | 0.8578 | 8.6542 | 0.6450 | 0.5591 |
| C1-Armenta-CARMATensor | 0.9210 | 0.4704 | 0.8520 | 8.1840 | 0.6728 | 0.5215 |
| C2-CrossAttention-CARMATensor | 0.9115 | 0.4739 | 0.8537 | 8.4217 | 0.6673 | 0.5354 |
| C2-RecurrentFusion-CARMATensor | 0.8915 | 0.4817 | 0.8614 | 9.2659 | 0.6132 | 0.5294 |
| C3-RAG-XGB-CARMATensor | 0.8947 | 0.4917 | 0.8520 | 8.7747 | 0.6323 | 0.5432 |
| CARMA Run22 (reported) | 0.8823 | 0.4943 | 0.8520 | 7.5911 | 0.7104 | 0.5424 |

Interpretation:

- The sample-size criticism was valid for the old Exp1 table. This table removes
  that mismatch for the representative F1/F2/C1/C2/C3 baseline families shown
  in the paper by evaluating them on the same full temporal test set as CARMA.
- CARMA remains clearly strongest on meanScore error and 10-point accuracy.
- Popularity is mixed. F2 has the lowest log_MAE (`0.8799` vs CARMA `0.8823`),
  C2-RecurrentFusion has the highest Spearman (`0.8614`), while CARMA has the
  best 2x accuracy and remains competitive in Spearman.
- The paper should avoid a universal claim that CARMA wins every popularity
  metric. A safer claim is: CARMA provides the strongest score prediction and
  competitive popularity prediction under the full temporal test setting.

Artifact note:

- Run22 checkpoints and per-row internal predictions are supplied by the
  release asset `final_project.zip` under `final_project/runs/22`. The main
  repository's `src_2/runs` directory still only contains Run02, so reproduction
  commands explicitly pass `--run-dir final_project/runs`.

### Paired Bootstrap Follow-up

The paired bootstrap diagnostic uses existing per-row prediction files and
resamples the same test ids with replacement. It is not a substitute for
multi-seed training, but it answers whether the observed fixed-seed deltas are
stable on the evaluated sample.

Selected results:

| Comparison | Target | Metric | Delta in favor of CARMA | 95% CI | p |
|---|---|---|---:|---:|---:|
| CARMA vs F2 | popularity | log_MAE | -0.0024 | [-0.0248, 0.0183] | 0.772 |
| CARMA vs F2 | meanScore | MAE | 1.0631 | [0.8995, 1.2552] | 0.000 |
| CARMA vs C3 | popularity | Spearman | 0.0000 | [-0.0074, 0.0071] | 0.952 |
| CARMA full vs remove retrieval | popularity | log_MAE | 0.0650 | [0.0429, 0.0839] | 0.000 |
| CARMA full vs remove image | meanScore | MAE | 1.0440 | [0.8929, 1.1862] | 0.000 |
| CARMA full vs remove temporal trend | meanScore | MAE | 0.2966 | [0.2054, 0.3941] | 0.000 |

Interpretation:

- CARMA's meanScore MAE advantage over F2/C2/C3 is stable under paired
  resampling.
- Popularity is genuinely mixed. CARMA is competitive, but F2/C2/C3 can match
  or exceed it on some popularity ranking/error metrics.
- The main ablation deltas for retrieval, image, and temporal trend are stable
  for error metrics, though some Spearman deltas are smaller or mixed.

## External MAL Alignment

External rows use the same MAL 2025 local-ready records for CARMA and baselines:

- Pop-only set: `n=3,765`
- Dual-target set: `n=1,202`

### MAL 2025 Overlap Label Sanity

The original paper draft used MAL July as a label-check source. To avoid the
criticism that the label-check dataset differs from the external exam dataset,
the same check was repeated on MAL 2025 overlap rows:

| Target | n | Spearman | Pearson | Calibration slope | Error |
|---|---:|---:|---:|---:|---:|
| popularity | 13,740 | 0.9836 | 0.9842 | 0.9351 | log_MAE 1.2657 |
| meanScore | 13,740 | 0.9446 | 0.9419 | 1.1643 | MAE 3.9705 |

Interpretation:

- MAL 2025 `members` is strongly rank-aligned with AniList `popularity` on
  overlap rows.
- MAL 2025 `score * 10` is strongly rank-aligned with AniList `meanScore`.
- This supports using MAL 2025 MAL-only rows as the external answer source.

### MAL Pop-only (`n=3,765`)

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman |
|---|---:|---:|---:|
| CARMA Run02 | 1.0120 | 0.4656 | 0.4709 |
| CARMA Run22 | 1.0359 | 0.4234 | 0.4998 |
| F1-RF-Meta-CARMATensor | 1.0015 | 0.4776 | 0.4572 |
| F2-XGB-Concat-CARMATensor | 1.0294 | 0.5017 | 0.5240 |
| C1-Armenta-CARMATensor | 1.1704 | 0.4345 | 0.2798 |
| C2-CrossAttention-CARMATensor | 1.0769 | 0.4677 | 0.3576 |
| C2-RecurrentFusion-CARMATensor | 1.2692 | 0.3073 | 0.0766 |
| C3-RAG-XGB-CARMATensor | 1.0383 | 0.4497 | 0.4392 |

Interpretation:

- On MAL pop-only, Run22 improves CARMA's ranking over Run02
  (`0.4998` vs `0.4709`) but has slightly worse log_MAE.
- F2 remains the strongest row by Spearman and 2x accuracy, while F1 has the
  lowest log_MAE. Therefore, MAL pop-only should be described as useful
  cross-platform signal, not as a clean CARMA win across all metrics.
- C1/C2 do not improve the no-YOLO pop-only setting, suggesting that the
  literature-adapted fusion proxies are sensitive to the external visual
  feature gap.

### MAL Dual-target (`n=1,202`)

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman | Score MAE | Score 10pt acc | Score Spearman |
|---|---:|---:|---:|---:|---:|---:|
| CARMA Run02 | 1.3910 | 0.3344 | 0.5495 | 7.5086 | 0.7488 | 0.6079 |
| CARMA Run22 | 1.1707 | 0.4260 | 0.5647 | 6.3363 | 0.7945 | 0.5770 |
| F1-RF-Meta-CARMATensor | 1.4387 | 0.3236 | 0.5144 | 9.1210 | 0.6106 | 0.5629 |
| F2-XGB-Concat-CARMATensor | 1.6172 | 0.2787 | 0.5789 | 9.7600 | 0.5649 | 0.6061 |
| C1-Armenta-CARMATensor | 1.9602 | 0.1298 | 0.4572 | 9.9122 | 0.5316 | 0.5826 |
| C2-CrossAttention-CARMATensor | 1.7178 | 0.2047 | 0.5607 | 8.3423 | 0.6622 | 0.5491 |
| C2-RecurrentFusion-CARMATensor | 1.8289 | 0.1814 | 0.4585 | 9.2142 | 0.5865 | 0.6113 |
| C3-RAG-XGB-CARMATensor | 1.5637 | 0.2704 | 0.6211 | 9.3699 | 0.5932 | 0.5989 |

Interpretation:

- Run22 is substantially better than Run02 on the dual subset for absolute
  error: popularity log_MAE improves from `1.3910` to `1.1707`, and score MAE
  improves from `7.5086` to `6.3363`.
- Run22 has the best popularity log_MAE, popularity 2x accuracy, score MAE, and
  score 10-point accuracy among the shown rows.
- Ranking remains mixed: C3 has the highest popularity Spearman, and F2 has the
  highest score Spearman among the original no-YOLO baselines. C2-Recurrent is
  close on score Spearman but weaker on absolute error.
- MeanScore external `R2` remains negative even for Run22 (`-0.3253`), so the
  claim should emphasize ranking and absolute error rather than calibrated
  variance explanation.

### Cover-derived YOLO Diagnostic

The original MAL local-ready splits include cover embeddings but have no banner
image and no YOLO branch:

- `mal2025_popularity_local_ready`: cover `3,765`, banner `0`, YOLO `0`
- `mal2025_dual_local_ready`: cover `1,202`, banner `0`, YOLO `0`

To test whether the missing YOLO branch explains the external drop, a diagnostic
adapter was added. It runs the existing anime person/face detector on each MAL
cover image, embeds the detected crops with the Swin image encoder, fills only
the `yolo_*` columns, and keeps banner missing. The generated splits are:

- `mal2025_popularity_local_ready_yolo`: cover `3,765`, banner `0`, YOLO `3,765`
- `mal2025_dual_local_ready_yolo`: cover `1,202`, banner `0`, YOLO `1,202`

Pop-only YOLO-filled rows:

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman |
|---|---:|---:|---:|
| CARMA Run22 | 1.0143 | 0.4356 | 0.5213 |
| F1-RF-Meta-CARMATensor | 1.0015 | 0.4776 | 0.4572 |
| F2-XGB-Concat-CARMATensor | 0.9932 | 0.5147 | 0.5297 |
| C1-Armenta-CARMATensor | 1.0240 | 0.4133 | 0.4329 |
| C2-CrossAttention-CARMATensor | 0.9562 | 0.4624 | 0.4961 |
| C2-RecurrentFusion-CARMATensor | 1.0549 | 0.3777 | 0.3963 |
| C3-RAG-XGB-CARMATensor | 0.9974 | 0.4869 | 0.5097 |

Dual-target YOLO-filled rows:

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman | Score MAE | Score 10pt acc | Score Spearman |
|---|---:|---:|---:|---:|---:|---:|
| CARMA Run22 | 1.2001 | 0.4060 | 0.6073 | 6.4919 | 0.7829 | 0.5999 |
| F1-RF-Meta-CARMATensor | 1.4387 | 0.3236 | 0.5144 | 9.1210 | 0.6106 | 0.5629 |
| F2-XGB-Concat-CARMATensor | 1.5784 | 0.2704 | 0.6379 | 9.9334 | 0.5424 | 0.6449 |
| C1-Armenta-CARMATensor | 1.3690 | 0.2937 | 0.5781 | 8.2518 | 0.6631 | 0.5607 |
| C2-CrossAttention-CARMATensor | 1.2208 | 0.3735 | 0.6687 | 8.0652 | 0.6714 | 0.5762 |
| C2-RecurrentFusion-CARMATensor | 1.3365 | 0.3111 | 0.6338 | 8.2210 | 0.6656 | 0.6459 |
| C3-RAG-XGB-CARMATensor | 1.5278 | 0.2629 | 0.6591 | 9.6667 | 0.5616 | 0.6186 |

Interpretation:

- Adding cover-derived YOLO improves CARMA ranking transfer: popularity
  Spearman rises from `0.4998` to `0.5213` on pop-only and from `0.5647` to
  `0.6073` on the dual split; score Spearman also rises from `0.5770` to
  `0.5999`.
- The improvement is not universal. CARMA's dual-target log_MAE, 2x accuracy,
  score MAE, and score 10-point accuracy slightly worsen after YOLO filling.
  Therefore the missing YOLO branch contributes to external ranking degradation,
  but it does not fully explain calibration or absolute-error degradation.
- YOLO filling also helps several reference baselines, especially C2 and F2 on
  ranking metrics. In the dual YOLO-filled split, C2-CrossAttention has the best
  popularity Spearman (`0.6687`), while CARMA keeps the best score MAE
  (`6.4919`) and score 10-point accuracy (`0.7829`).
- Since banner remains unavailable, the main paper should present this as a
  robustness/diagnostic result rather than replacing the original MAL external
  table.

### Cover-as-banner Proxy Diagnostic

MAL 2025 still has no true AniList-style banner images. To test whether the
empty banner slot itself explains external degradation, a second diagnostic
copies each cover embedding into the banner embedding slot after YOLO has been
filled. This creates:

- `mal2025_popularity_local_ready_yolo_coverbanner`: cover `3,765`, banner
  proxy `3,765`, YOLO `3,765`
- `mal2025_dual_local_ready_yolo_coverbanner`: cover `1,202`, banner proxy
  `1,202`, YOLO `1,202`

This is a proxy, not a true banner evaluation. It should not be used to claim
that MAL has equivalent banner information.

Selected cover-as-banner rows:

| Model | Split | Target | Main metric |
|---|---|---|---:|
| CARMA Run22 | pop-only | popularity | Spearman 0.5166 |
| F2-XGB-Concat-CARMATensor | pop-only | popularity | Spearman 0.5172 |
| C3-RAG-XGB-CARMATensor | pop-only | popularity | Spearman 0.5050 |
| CARMA Run22 | dual | popularity | Spearman 0.5955 |
| C2-CrossAttention-CARMATensor | dual | popularity | Spearman 0.6637 |
| C3-RAG-XGB-CARMATensor | dual | popularity | Spearman 0.6343 |
| CARMA Run22 | dual | meanScore | MAE 6.4098 |
| C1-Armenta-CARMATensor | dual | meanScore | MAE 6.9666 |
| C3-RAG-XGB-CARMATensor | dual | meanScore | MAE 7.3660 |

Interpretation:

- Copying cover into the banner slot does not improve CARMA beyond the
  cover-derived YOLO condition. Pop-only Spearman changes from `0.5213` to
  `0.5166`; dual popularity Spearman changes from `0.6073` to `0.5955`; score
  Spearman changes from `0.5999` to `0.5921`.
- The remaining visual limitation is therefore not solved by a naive
  cover-as-banner proxy. A true banner-like external branch would need a real
  banner source or a deliberately trained banner imputation method.
- CARMA remains best on dual external score MAE among the cover-as-banner rows,
  while C2/C3 baselines can be stronger on popularity ranking.

### External Calibration Diagnostics

Prediction-quantile calibration bins were added for Run22. The bins are grouped
by predicted value, not by ground truth, so they show whether higher model
predictions correspond to higher observed MAL outcomes.

Summary:

| Source | Target | n | Spearman | R2 | Calibration slope | Error |
|---|---|---:|---:|---:|---:|---:|
| pop_only:no_yolo | popularity | 3,765 | 0.4998 | 0.2979 | 1.0435 | log_MAE 1.0359 |
| dual:no_yolo | popularity | 1,202 | 0.5647 | -0.1188 | 0.8464 | log_MAE 1.1707 |
| dual:no_yolo | meanScore | 1,202 | 0.5770 | -0.3253 | 0.5890 | MAE 6.3363 |
| pop_only:cover_yolo | popularity | 3,765 | 0.5213 | 0.3173 | 1.1622 | log_MAE 1.0143 |
| dual:cover_yolo | popularity | 1,202 | 0.6073 | -0.1613 | 0.9310 | log_MAE 1.2001 |
| dual:cover_yolo | meanScore | 1,202 | 0.5999 | -0.3573 | 0.6367 | MAE 6.4919 |
| pop_only:cover_yolo_coverbanner_proxy | popularity | 3,765 | 0.5166 | 0.3149 | 1.1295 | log_MAE 1.0174 |
| dual:cover_yolo_coverbanner_proxy | popularity | 1,202 | 0.5955 | -0.1433 | 0.9057 | log_MAE 1.1880 |
| dual:cover_yolo_coverbanner_proxy | meanScore | 1,202 | 0.5921 | -0.3374 | 0.6197 | MAE 6.4098 |

Interpretation:

- Actual MAL outcomes are generally monotonic across prediction quantiles, so
  the external predictions retain useful ranking signal.
- The highest predicted popularity quantiles are still under-scaled relative to
  observed MAL members. This explains why ranking metrics can remain useful
  while external R2 is weak or negative.
- For meanScore, the model underpredicts low-to-mid MAL score bins more than
  high-score bins. This is another calibration issue rather than a complete
  failure of ranking transfer.

## Answer to Presentation Feedback

For Q1, the clean internal comparison now exists on `n=3,087`. It weakens the
headline "lowest reported popularity error" claim because F2 is slightly better
on popularity log_MAE. It strengthens the more precise claim that CARMA is best
for meanScore error and remains competitive for popularity.

For Q2, the aligned results support a conservative answer: meanScore is
partially predictable from pre-release signals, especially internally and on
external MAE. Run22 improves MAL dual-target score MAE to `6.3363`, but its
external `R2` is still negative, so cross-platform score scale is not well
calibrated. Evidence against a pure temporal/popularity prior comes from the
ablations and interpretability results: removing image/RAG/temporal components
worsens error, and visual plus retrieved context features receive substantial
attribution. The final claim should be "useful ranking and absolute error
prediction", not "well-calibrated score variance explanation".

## Remaining Work

High priority if the paper is revised again:

1. Add multi-seed mean/std for the key CARMA-vs-baseline and ablation deltas.

Lower priority:

1. Strict CNN-vs-Swin backbone ablation inside the same CARMA architecture.
   Existing artifacts support image-source ablations and ResNet/CNN literature
   proxy diagnostics, but not a one-variable CARMA CNN-vs-Swin replacement.
