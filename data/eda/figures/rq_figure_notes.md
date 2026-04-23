# RQ Figure Notes

## Figure 1: Snapshot Control
- File: `data/eda/figures/rq_snapshot_control.png`
- Meaning: compare absolute correlation between release year and target before/after quarter normalization.
- Observed reduction: `0.10616242573276485`
- Statistical support: bootstrap CI95 for reduction = [`0.0885708825502303`, `0.12215550704813904`], mean=`0.1064216371334734`.

## Figure 2: Split Bucket Balance
- File: `data/eda/figures/rq_split_bucket_balance.png`
- Meaning: verify whether popularity bucket classes remain balanced across train/val/test.
- Statistical support: permutation test (max TVD) stat=`0.005934000000495085`, p=`0.9925187032418953`.

## Figure 3: Multimodal Coverage by Split
- File: `data/eda/figures/rq_multimodal_coverage_by_split.png`
- Meaning: show text/image/trailer availability mismatch between splits for RQ2 risk discussion.

### Statistical Support for Coverage Gaps
- `text_available` permutation: stat=`0.05221452763084111`, p=`0.0024937655860349127`.
- `cover_available` permutation: stat=`0.0`, p=`1.0`.
- `banner_available` permutation: stat=`0.19077861057091605`, p=`0.0024937655860349127`.
- `trailer_available` permutation: stat=`0.5551406538919965`, p=`0.0024937655860349127`.
