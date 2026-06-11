# Evaluation Sample Alignment Report

Date: 2026-06-11

Branch: `feature/eval-sample-alignment`

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

Main outputs:

- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.csv`
- `reports/experiments/sample_alignment/carma_tensor_aligned_metrics.md`
- `reports/experiments/sample_alignment/carma_tensor_predictions/`
- `data/external_transformed/run22_mal2025_popularity_local_ready_metrics.json`
- `data/external_transformed/run22_mal2025_dual_local_ready_metrics.json`

Run22 external inference can be reproduced with:

```bash
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready
```

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
| C3-RAG-XGB-CARMATensor | 0.8947 | 0.4917 | 0.8520 | 8.7747 | 0.6323 | 0.5432 |
| CARMA Run22 (reported) | 0.8823 | 0.4943 | 0.8520 | 7.5911 | 0.7104 | 0.5424 |

Interpretation:

- The original `n=3,087` vs `n=2,808` criticism was valid for the old Exp1
  table. This table removes that sample-size mismatch for the shown baselines.
- CARMA remains clearly strongest on meanScore error and 10-point accuracy.
- Popularity is mixed. F2 has the lowest log_MAE (`0.8799` vs CARMA `0.8823`),
  while CARMA has the best 2x accuracy and remains competitive in Spearman.
- The paper should avoid a universal claim that CARMA wins every popularity
  metric. A safer claim is: CARMA provides the strongest score prediction and
  competitive popularity prediction under the full temporal test setting.

Artifact note:

- Run22 checkpoints and per-row internal predictions were supplied under
  `final_project/runs/22`. The main repository's `src_2/runs` directory still
  only contains Run02, so reproduction commands explicitly pass
  `--run-dir final_project/runs`.

## External MAL Alignment

External rows use the same MAL 2025 local-ready records for CARMA and baselines:

- Pop-only set: `n=3,765`
- Dual-target set: `n=1,202`

### MAL Pop-only (`n=3,765`)

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman |
|---|---:|---:|---:|
| CARMA Run02 | 1.0120 | 0.4656 | 0.4709 |
| CARMA Run22 | 1.0359 | 0.4234 | 0.4998 |
| F1-RF-Meta-CARMATensor | 1.0015 | 0.4776 | 0.4572 |
| F2-XGB-Concat-CARMATensor | 1.0294 | 0.5017 | 0.5240 |
| C3-RAG-XGB-CARMATensor | 1.0383 | 0.4497 | 0.4392 |

Interpretation:

- On MAL pop-only, Run22 improves CARMA's ranking over Run02
  (`0.4998` vs `0.4709`) but has slightly worse log_MAE.
- F2 remains the strongest row by Spearman and 2x accuracy, while F1 has the
  lowest log_MAE. Therefore, MAL pop-only should be described as useful
  cross-platform signal, not as a clean CARMA win across all metrics.

### MAL Dual-target (`n=1,202`)

| Model | Pop log_MAE | Pop 2x acc | Pop Spearman | Score MAE | Score 10pt acc | Score Spearman |
|---|---:|---:|---:|---:|---:|---:|
| CARMA Run02 | 1.3910 | 0.3344 | 0.5495 | 7.5086 | 0.7488 | 0.6079 |
| CARMA Run22 | 1.1707 | 0.4260 | 0.5647 | 6.3363 | 0.7945 | 0.5770 |
| F1-RF-Meta-CARMATensor | 1.4387 | 0.3236 | 0.5144 | 9.1210 | 0.6106 | 0.5629 |
| F2-XGB-Concat-CARMATensor | 1.6172 | 0.2787 | 0.5789 | 9.7600 | 0.5649 | 0.6061 |
| C3-RAG-XGB-CARMATensor | 1.5637 | 0.2704 | 0.6211 | 9.3699 | 0.5932 | 0.5989 |

Interpretation:

- Run22 is substantially better than Run02 on the dual subset for absolute
  error: popularity log_MAE improves from `1.3910` to `1.1707`, and score MAE
  improves from `7.5086` to `6.3363`.
- Run22 has the best popularity log_MAE, popularity 2x accuracy, score MAE, and
  score 10-point accuracy among the shown rows.
- Ranking remains mixed: C3 has the highest popularity Spearman, and F2 has the
  highest score Spearman. This keeps the external conclusion conservative.
- MeanScore external `R2` remains negative even for Run22 (`-0.3253`), so the
  claim should emphasize ranking and absolute error rather than calibrated
  variance explanation.

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
2. Optionally recompute the old `n=2,808` strict common-subset CARMA comparison
   from `final_project/runs/22/*/pred_test.csv` if the paper needs an appendix
   robustness table.

Lower priority:

1. CNN-vs-Swin backbone ablation.
2. External calibration or quantile analysis for MAL scale mismatch.
