# RQ Figure Notes

This note explains what each figure demonstrates, how to interpret it, and what caveats should be reported in the paper.

## Figure 1: Snapshot Control
- File: `data/eda/figures/rq_snapshot_control.png`
- Question addressed: does quarter-normalized popularity reduce snapshot time bias?
- Metric: absolute correlation `|corr(release_year, target)|` before vs after normalization.
- Core result: absolute correlation reduction = `0.10616242573276485`.
- Statistical support: bootstrap CI95 for reduction = [`0.0885708825502303`, `0.12215550704813904`], mean=`0.1064216371334734`.
- Interpretation: normalization consistently weakens year-driven bias in the target.
- Caveat: this is a bias-mitigation proxy, not a guarantee that all temporal confounding is removed.

## Figure 2: Split Bucket Balance
- File: `data/eda/figures/rq_split_bucket_balance.png`
- Question addressed: are class proportions stable across `train/val/test`?
- Bucket definition:
  - `cold_0_25`: within-quarter percentile in `[0, 25%]`
  - `warm_25_50`: within-quarter percentile in `(25%, 50%]`
  - `hot_50_75`: within-quarter percentile in `(50%, 75%]`
  - `top_75_100`: within-quarter percentile in `(75%, 100%]`
- Statistical support: permutation test (max TVD), stat=`0.005934000000495085`, p=`0.9925187032418953`.
- Interpretation: no meaningful distribution shift is detected across model splits.
- Caveat: near-equal bars are expected because buckets are built from within-quarter percentile ranking.

## Figure 3: Multimodal Coverage by Split
- File: `data/eda/figures/rq_multimodal_coverage_by_split.png`
- Question addressed: does modality availability differ across splits (RQ2 data-risk profile)?
- Coverage ratio definition: proportion of rows with non-null source field in each split.
- Core observation:
  - `cover_available` is fully available across splits (ratio = `1.0`).
  - text, banner, and trailer availability vary by split.

### Statistical Support for Coverage Gaps
- `text_available` permutation: stat=`0.05221452763084111`, p=`0.0024937655860349127`.
- `cover_available` permutation: stat=`0.0`, p=`1.0`.
- `banner_available` permutation: stat=`0.19077861057091605`, p=`0.0024937655860349127`.
- `trailer_available` permutation: stat=`0.5551406538919965`, p=`0.0024937655860349127`.

### Interpretation for Paper Writing
- Significant modality imbalance exists for text/banner/trailer; this should be treated as an experimental risk factor.
- Report split-wise modality coverage when presenting RQ2 model results.
- Use ablation and missing-modality handling strategy to avoid overstating multimodal gains.
