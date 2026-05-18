# Optimization Log

Tracks each optimization experiment against the baseline (minilm_l6, no marketing cleanup).

---

## Baseline — MiniLM-L6 (no preprocessing changes)

Model: `sentence-transformers/all-MiniLM-L6-v2`  
Preprocessing: lowercase, URL removal, whitespace normalization  
Regressor: Ridge α=1.0

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 20462.82 | 42125.67 | 0.5509 |
| popularity | test | 17946.53 | 34055.32 | 0.5408 |
| meanScore | val | 9.8100 | 11.9300 | 0.2886 |
| meanScore | test | 10.9400 | 13.1200 | 0.2152 |

---

## Experiment 01 — Marketing Fluff Removal

**Date:** 2026-05-18  
**Branch:** Text  
**Change:** Added `remove_marketing=True` to `TextPreprocessor`. Strips the following before encoding:
- HTML tags (`<br>`, `<i>`, etc.)
- `(Source: AniList)` / `(Written by MAL Rewrite)` attribution tags
- Streaming platform credit sentences (Crunchyroll, Funimation, Netflix, etc.)
- "Based on the manga/light novel by…" sentences
- Blu-ray/DVD release notes

Model and regressor unchanged.

### Results

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 20500.88 | 42163.61 | 0.5436 |
| popularity | test | 18070.80 | 34160.66 | 0.5310 |
| meanScore | val | 9.8865 | 12.0282 | 0.2845 |
| meanScore | test | 11.1576 | 13.3720 | 0.2002 |

### Delta vs Baseline (positive = worse)

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | +38.06 | +37.94 | −0.0073 |
| popularity | test | +124.27 | +105.34 | −0.0098 |
| meanScore | val | +0.0765 | +0.0982 | −0.0041 |
| meanScore | test | +0.2176 | +0.2520 | −0.0150 |

### Verdict: ❌ Marginal regression across all metrics

**Analysis:** The preprocessing made results slightly worse. Likely reasons:

1. **"Based on the manga/light novel"** phrases are actually useful signal — manga/LN adaptations have systematically different popularity patterns than original anime, so removing this acts as ablating a genre feature.
2. **Streaming credits as proxy signal** — "Streaming on Crunchyroll" implies a licensing deal, which correlates with production budget and therefore popularity. Removing it discards this indirect feature.
3. **MiniLM is robust to noise** — at 384 dims the model already dilutes short boilerplate; the gain from removing it is smaller than the loss of the above signals.

**Next steps:**
- Re-run on `e5_base` (best model) to see if the effect is model-dependent
- Try removing only the strict source attribution tags (`(Source: AniList)`) and keeping "Based on…" and streaming lines
- Consider making each pattern individually toggleable in config

---

## Baseline — e5_base (no LSA, no marketing cleanup)

Model: `intfloat/e5-base-v2`  
Preprocessing: lowercase, URL removal, whitespace normalization  
Regressor: Ridge α=1.0

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 20136.47 | 40474.55 | 0.6080 |
| popularity | test | 17411.99 | 32060.13 | 0.6172 |
| meanScore | val | 9.5503 | 11.6862 | 0.3494 |
| meanScore | test | 10.8129 | 13.1309 | 0.2525 |

> This is the reference point for Experiments 02 and 03.

---

## Experiment 02 — TF-IDF + LSA (128 dims) appended to e5_base

**Date:** 2026-05-18  
**Change:** `--tfidf-components 128`. Fit TfidfVectorizer (unigrams+bigrams, sublinear_tf, min_df=2) on train text, reduced with TruncatedSVD to 128 dims, L2-normalised, concatenated to 384 e5_base dims → 512 total features.

### Results

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 20722.22 | 41245.93 | 0.5597 |
| popularity | test | 18370.17 | 33544.16 | 0.5717 |
| meanScore | val | 9.1480 | 11.2032 | 0.3177 |
| meanScore | test | 10.1720 | 12.3372 | 0.2446 |

### Delta vs e5_base baseline (positive = worse)

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | +585.75 | +771.38 | −0.0483 |
| popularity | test | +958.18 | +1484.03 | −0.0455 |
| meanScore | val | **−0.40** | **−0.48** | −0.0317 |
| meanScore | test | **−0.64** | **−0.79** | −0.0079 |

### Verdict: ❌ Popularity worse; meanScore MAE/RMSE marginally better

---

## Experiment 03 — TF-IDF + LSA (64 dims) appended to e5_base

**Date:** 2026-05-18  
**Change:** Same as Exp 02 but `--tfidf-components 64` → 448 total features.

### Results

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 20663.72 | 41627.74 | 0.5483 |
| popularity | test | 18217.33 | 33650.23 | 0.5648 |
| meanScore | val | 9.3392 | 11.3811 | 0.3194 |
| meanScore | test | 10.4720 | 12.6358 | 0.2456 |

### Delta vs e5_base baseline (positive = worse)

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | +527.25 | +1153.19 | −0.0597 |
| popularity | test | +805.34 | +1590.10 | −0.0524 |
| meanScore | val | **−0.21** | **−0.31** | −0.0300 |
| meanScore | test | **−0.34** | **−0.50** | −0.0069 |

### Verdict: ❌ Worse than LSA-128 on popularity; similar meanScore pattern

---

## Cross-experiment Summary

| Experiment | Model | Features | popularity test Spearman | meanScore test Spearman | popularity test RMSE | meanScore test RMSE |
|---|---|---:|---:|---:|---:|---:|
| Baseline (MiniLM) | minilm_l6 | 384 | 0.5408 | 0.2152 | 34055.32 | 13.1200 |
| Exp 01 (MiniLM + cleanup) | minilm_l6 | 384 | 0.5310 | 0.2002 | 34160.66 | 13.3720 |
| **Baseline (e5_base)** | **e5_base** | **768** | **0.6172** | **0.2525** | **32060.13** | **13.1309** |
| Exp 02 (e5_base + LSA-128) | e5_base | 512 | 0.5717 | 0.2446 | 33544.16 | 12.3372 |
| Exp 03 (e5_base + LSA-64) | e5_base | 448 | 0.5648 | 0.2456 | 33650.23 | 12.6358 |

## Analysis — Why LSA Hurt Popularity

1. **Redundancy with e5_base** — e5_base (768 dims) already captures most semantic structure that TF-IDF would extract. Adding LSA dims brings diminishing information but more parameters for Ridge to overfit.
2. **Single alpha can't balance two spaces** — Ridge uses one α for all 512/448 features. The 128 LSA dims have a different scale and density from the 768 dense dims; the regularizer can't treat them separately.
3. **meanScore MAE/RMSE improved slightly** — score prediction is more keyword-sensitive (genre tags like "mecha", "sports" carry direct score priors) so sparse features add small signal there, just not enough to lift Spearman.

**Recommended next steps for Suggestion 2:**
- Try a much smaller LSA component (32 dims) to reduce redundancy
- Apply separate Ridge alphas per feature group (stack two Ridge models and blend)
- Test BM25 instead of TF-IDF — better term weighting for short texts

