# Optimization Log

Tracks each optimization experiment for the text branch of the Anime Prediction project.  
Primary metric: **popularity test Spearman** (higher = better).  
Reference model for all experiments from Exp 02 onwards: **e5_base** (0.6172).

---

## Reference Baselines

| Model | Dim | popularity val Spearman | popularity test Spearman | meanScore val Spearman | meanScore test Spearman | popularity test RMSE |
|---|---:|---:|---:|---:|---:|---:|
| MiniLM-L6 (`all-MiniLM-L6-v2`) | 384 | 0.5509 | 0.5408 | 0.2886 | 0.2152 | 34055.32 |
| **e5_base** (`intfloat/e5-base-v2`) | **768** | **0.6080** | **0.6172** | **0.3494** | **0.2525** | **32060.13** |

> MiniLM was the initial model. e5_base replaced it after showing clear improvement and is the **active reference** for all experiments from Exp 02 onwards.  
> Both use Ridge α=1.0, preprocessing: lowercase + URL removal + whitespace normalization.

---

## Cross-experiment Summary

| # | Experiment | Model | Dim | pop. test ρ | score test ρ | pop. test RMSE | Verdict |
|---|---|---|---:|---:|---:|---:|---:|
| — | Baseline (MiniLM) | minilm_l6 | 384 | 0.5408 | 0.2152 | 34055.32 | reference |
| — | **Baseline (e5_base)** | **e5_base** | **768** | **0.6172** | **0.2525** | **32060.13** | **reference** |
| 01 | Marketing cleanup | minilm_l6 | 384 | 0.5310 | 0.2002 | 34160.66 | ❌ |
| 02 | e5_base + LSA-128 | e5_base | 512 | 0.5717 | 0.2446 | 33544.16 | ❌ |
| 03 | e5_base + LSA-64 | e5_base | 448 | 0.5648 | 0.2456 | 33650.23 | ❌ |
| 04 | Fine-tune top-2 layers (A1) | e5_base ft | 768 | 0.5928 | 0.2702 | 31864.62 | ❌ |
| 05 | Fine-tune top-3 layers (A2) | e5_base ft | 768 | 0.5929 | 0.2775 | 33402.72 | ❌ |
| 06 | Frozen encoder + proj-384 (B1) | e5_base + Dense | 384 | 0.5774 | 0.2495 | 32510.21 | ❌ |
| 07 | Unfreeze top-2 + proj-384 (B2) | e5_base ft + Dense | 384 | 0.5912 | 0.3016 | 31557.36 | ❌ |

---

## Experiments

### Exp 01 — Marketing Fluff Removal (MiniLM)

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

### Exp 02 — TF-IDF + LSA (128 dims) appended to e5_base

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

### Exp 03 — TF-IDF + LSA (64 dims) appended to e5_base

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

### Exp 04 — Layer-wise Fine-tuning (A1, top-2 layers unfrozen)

**Date:** 2026-05-19  
**Change:** Added supervised fine-tuning step for `intfloat/e5-base-v2`:
- Freeze all encoder layers, unfreeze top 2 layers only (`layers 10-11` of 12)
- Discriminative LRs: head = `1e-4`, top layers = `1e-5`
- Early stopping on val Spearman (popularity)
- Saved best encoder as SentenceTransformer artifact and re-ran standard embedding pipeline + Ridge eval

### Fine-tune stage (direct validation)

From `reports/finetune_A1.json`:
- best epoch: 2
- best val Spearman (popularity): **0.6637**
- trainable encoder params: 14,175,744 / 109,482,240 (12.9%)

### A1 downstream run (first attempt, preprocessing mismatch)

Issue found:
- fine-tune used `remove_marketing=False`
- embedding export used config default `remove_marketing=True`
- this caused train/infer preprocessing mismatch and weakened transfer

Results from `reports/text_branch_metrics_A1.json`:

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 19657.66 | 37445.95 | 0.5515 |
| popularity | test | 18212.81 | 31737.45 | 0.5779 |
| meanScore | val | 8.8342 | 10.9576 | 0.4008 |
| meanScore | test | 10.2091 | 12.5089 | 0.2683 |

### A1 downstream run (corrected parity: `remove_marketing=False`)

Re-exported embeddings with CLI override to match fine-tune preprocessing (`reports/text_embedding_pipeline_summary_A1_rmfalse.json`).

Results from `reports/text_branch_metrics_A1_rmfalse.json`:

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 19592.98 | 37125.43 | 0.5670 |
| popularity | test | 18366.28 | 31864.62 | 0.5928 |
| meanScore | val | 8.8262 | 10.9547 | 0.4048 |
| meanScore | test | 10.1864 | 12.4757 | 0.2702 |

### Delta vs e5_base baseline (corrected parity run)

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | **−543.49** | **−3349.12** | −0.0410 |
| popularity | test | +954.28 | **−195.51** | −0.0244 |
| meanScore | val | **−0.7240** | **−0.7315** | +0.0554 |
| meanScore | test | **−0.6265** | **−0.6552** | +0.0177 |

### Promotion gate decision (A1)

Required:
1. Validation Spearman improves vs e5_base baseline (0.6080)
2. Test popularity MAE and RMSE do not regress

Observed (A1 corrected parity):
- val popularity Spearman = 0.5670 (**fails gate 1**)
- test popularity RMSE = 31864.62 (improves)
- test popularity MAE = 18366.28 (regresses; **fails gate 2**)

### Verdict: ❌ Do not promote A1

Although A1 improves RMSE and most meanScore metrics, it does not satisfy the popularity gate criteria.

---

### Exp 05 — Layer-wise Fine-tuning (A2, top-3 layers unfrozen)

**Date:** 2026-05-20  
**Change:** Same setup as A1, but unfreeze top 3 layers (`layers 9-11` of 12).
- Model: `intfloat/e5-base-v2`
- Discriminative LRs: head = `1e-4`, top layers = `1e-5`
- Early stopping on val Spearman (popularity)
- Export/eval run with preprocessing parity (`remove_marketing=False`)

### Fine-tune stage (direct validation)

From `reports/finetune_A2.json`:
- best epoch: 2
- best val Spearman (popularity): **0.6812**
- trainable encoder params: 21,263,616 / 109,482,240 (19.4%)

### A2 downstream run (corrected parity: `remove_marketing=False`)

From `reports/text_branch_metrics_A2_rmfalse.json`:

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 19066.77 | 36433.57 | 0.5618 |
| popularity | test | 19152.92 | 33402.72 | 0.5929 |
| meanScore | val | 8.7295 | 10.8816 | 0.4088 |
| meanScore | test | 10.0796 | 12.3881 | 0.2775 |

### Delta vs e5_base baseline

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | **−1069.70** | **−4040.98** | −0.0462 |
| popularity | test | +1740.93 | +1342.59 | −0.0243 |
| meanScore | val | **−0.8208** | **−0.8046** | +0.0594 |
| meanScore | test | **−0.7332** | **−0.7428** | +0.0250 |

### Promotion gate decision (A2)

Required:
1. Validation Spearman improves vs e5_base baseline (0.6080)
2. Test popularity MAE and RMSE do not regress

Observed (A2):
- val popularity Spearman = 0.5618 (**fails gate 1**)
- test popularity RMSE = 33402.72 (regresses)
- test popularity MAE = 19152.92 (regresses; **fails gate 2**)

### Verdict: ❌ Do not promote A2

A2 further improves meanScore metrics, but regresses popularity ranking and absolute error vs baseline.

---

### Exp 06 — Linear Bottleneck Projection (B1, frozen encoder + proj-384)

**Date:** 2026-05-21  
**Change:** Added a trainable `nn.Linear(768, 384)` projection between the encoder pool output and the regression head. Encoder fully frozen (`--unfreeze-layers 0`). Only the projection (768→384) and regression head were trained via backprop on the popularity/meanScore MSE loss.

- Model: `intfloat/e5-base-v2` (fully frozen)
- Projection: `Linear(768, 384)`, no activation, LR = `1e-4` (same as head)
- Saved best encoder as SentenceTransformer with `Dense` module appended (output dim = 384)
- Early stopping on val Spearman (popularity), patience = 3

### Fine-tune stage (direct validation)

From `reports/finetune_B1.json`:
- best epoch: 3
- best val Spearman (popularity): **0.6474**
- trainable encoder params: 0 / 109,482,240 (0%) — only projection + head

### B1 downstream run

From `reports/text_branch_metrics_B1.json`:

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 20954.96 | 40344.00 | 0.5851 |
| popularity | test | 18329.48 | 32510.21 | 0.5774 |
| meanScore | val | 9.3387 | 11.5081 | 0.3503 |
| meanScore | test | 10.6279 | 12.9161 | 0.2495 |

### Delta vs e5_base baseline

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | +818.49 | −130.55 | −0.0229 |
| popularity | test | +917.49 | +450.08 | −0.0398 |
| meanScore | val | **−0.2117** | **−0.1781** | +0.0009 |
| meanScore | test | **−0.1849** | **−0.2148** | −0.0030 |

### Promotion gate decision (B1)

Required:
1. Validation Spearman improves vs e5_base baseline (0.6080)
2. Test popularity MAE and RMSE do not regress

Observed (B1):
- val popularity Spearman = 0.5851 (**fails gate 1**)
- test popularity MAE = 18329.48 > 17411.99 (regresses; **fails gate 2**)
- test popularity RMSE = 32510.21 > 32060.13 (regresses; **fails gate 2**)

### Verdict: ❌ Do not promote B1

The frozen encoder means the 768-dim representations are not adapted for task-specific compression. The projection must blindly compress a general-purpose embedding space into 384 dims using only the regression loss, without the encoder being able to reorganise its output to make that compression lossless. This causes information loss rather than information distillation.

**Note:** B1 still beats the MiniLM-L6 baseline (popularity test Spearman 0.5774 > 0.5408) and slightly improves meanScore MAE/RMSE vs e5_base — the projection does learn something, but not enough to offset the forced dimensionality reduction on a frozen encoder.

---

### Exp 07 — Unfreeze Top-2 Layers + proj-384 (B2)

**Date:** 2026-05-25  
**Change:** Same Linear(768→384) projection as B1, but encoder top-2 transformer layers unfrozen (`--unfreeze-layers 2 --projection-dim 384`). This allows the encoder to reorganise its output space to be more compressible, addressing B1's root cause.

- Model: `intfloat/e5-base-v2` (top-2 layers unfrozen, LR = `1e-5`)
- Projection: `Linear(768, 384)`, no activation, LR = `1e-4`
- Epochs: 10 max, early stopping patience = 3

### B2 downstream run

From `reports/text_branch_metrics_B2.json`:

| Target | Split | MAE | RMSE | Spearman |
|---|---|---:|---:|---:|
| popularity | val | 18817.94 | 36005.44 | 0.5780 |
| popularity | test | 17833.82 | 31557.36 | 0.5912 |
| meanScore | val | 8.6897 | 10.8513 | 0.4104 |
| meanScore | test | 9.8315 | 12.0528 | 0.3016 |

### Delta vs e5_base baseline

| Target | Split | ΔMAE | ΔRMSE | ΔSpearman |
|---|---|---:|---:|---:|
| popularity | val | −1318.53 | −4469.11 | −0.0300 |
| popularity | test | +421.83 | **−502.77** | −0.0260 |
| meanScore | val | **−0.8606** | **−0.8350** | **+0.0610** |
| meanScore | test | **−0.9814** | **−1.0781** | **+0.0491** |

### Delta vs B1 (frozen encoder + proj-384)

| Target | Split | ΔSpearman | ΔRMSE |
|---|---|---:|---:|
| popularity | test | +0.0138 | −952.85 |
| meanScore | test | **+0.0521** | **−0.8633** |

### Promotion gate decision (B2)

Required:
1. Validation Spearman improves vs e5_base baseline (0.6080)
2. Test popularity MAE and RMSE do not regress

Observed (B2):
- val popularity Spearman = 0.5780 (**fails gate 1**)
- test popularity MAE = 17833.82 > 17411.99 (regresses; **fails gate 2**)
- test popularity RMSE = 31557.36 < 32060.13 (**passes** — RMSE improved)

### Verdict: ❌ Do not promote B2

Unfreezing did help vs B1 (popularity RMSE below baseline; meanScore significantly improved), confirming the hypothesis. However, popularity Spearman still regresses vs the frozen e5_base baseline. The MSE training objective doesn't directly optimise rank order, and compressing to 384 dims appears to flatten the ranking signal regardless of whether the encoder is frozen or not.

**Notable:** B2 is the first experiment to beat the e5_base baseline on popularity **RMSE** (31557 < 32060) while also achieving the best meanScore Spearman of all experiments (0.3016 vs 0.2525 baseline). If the downstream fusion model is sensitive to absolute error rather than ranking, B2 embeddings may still be valuable.

---

## Analysis

### Why LSA Hurt Popularity (Exp 02–03)

1. **Redundancy with e5_base** — e5_base already captures most semantic structure that TF-IDF extracts. Adding LSA dims brings diminishing information but more parameters for Ridge to overfit.
2. **Single alpha can't balance two spaces** — Ridge uses one α for all features. The LSA dims have a different scale/density from the dense dims; the regularizer can't treat them separately.
3. **meanScore improved slightly** — score prediction is more keyword-sensitive (genre tags like "mecha", "sports" carry direct priors), so sparse features add small signal there.

### Why Fine-tuning Didn't Beat Baseline (Exp 04–05)

Fine-tuning improved val MSE loss and meanScore metrics, but the popularity val Spearman dropped vs the frozen e5_base baseline. The encoder may be over-adapting to the regression signal in a way that reorganises the embedding space at a cost to ranking ability.

### Why Frozen Projection Failed (Exp 06)

The projection must compress a general-purpose embedding space without the encoder being able to reorganise its output. This causes information loss rather than distillation.

### Why Unfrozen Projection Improved RMSE but Not Spearman (Exp 07)

Unfreezing allowed the encoder to restructure its output space to be more compressible, which cut absolute error (RMSE) below the baseline. But the MSE loss only optimises for value closeness, not rank order. Popularity is a highly skewed distribution — a few titles have massive popularity. Ranking Spearman is sensitive to relative ordering across that skew, which the MSE objective doesn't directly preserve when compressing to 384 dims.

