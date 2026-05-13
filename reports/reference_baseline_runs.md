# Reference Baseline Runs

Updated: 2026-05-12

This file records the tracked summary of reference baseline runs. Raw `.exp/`
outputs are local experiment artifacts and are intentionally not committed.
Use `reports/reference_baseline_results.csv` as the portable result table.
Use `reports/reference_baseline_paper_alignment_audit.md` for the claim
boundary between implemented adaptations and paper framework reproductions.

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
| `.exp/baseline/results/18` | `C2-CTNN-Lite` | local raw output | Source for completed lightweight cross-modal transformer fusion results. |
| `.exp/baseline/results/19` | `C1-Armenta-ProxyBranchMLP` | local raw output | Source for branch-wise anime-domain multimodal MLP proxy adaptation results. |
| `.exp/baseline/results/21` | `C3-RAG-None-XGB` | local raw output | Source for no-retrieval control under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/22` | `C3-RAG-Sparse-XGB` | local raw output | Source for sparse metadata retrieval under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/23` | `C3-RAG-Dense-XGB` | local raw output | Source for dense semantic retrieval under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/24` | `C3-RAG-Hybrid-XGB` | local raw output | Source for hybrid sparse+dense retrieval under the SKAPP-inspired C3 route. |

## Completed Routes

| Plan route | Baseline IDs | Completion status |
|---|---|---|
| `0. Lowest Reference / lowest floor` | `F0-Mean`, `F0-Ridge-Meta` | done |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | done as adaptation |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | done as adaptation |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP`, `C1-Armenta-ProxyBranchMLP` | first-pass and branch-wise proxy done as adaptations |
| `2.2 Cross-modal Transformer Fusion` | `C2-CTNN-Lite` | done as adaptation |
| `2.3 Retrieval / RAG Competitive Baseline` | `C3-RAG-None-XGB`, `C3-RAG-Sparse-XGB`, `C3-RAG-Dense-XGB`, `C3-RAG-Hybrid-XGB` | first-pass SKAPP-inspired retrieval baselines done |

## C1 vs F2 Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-MLP` test R2 | Current interpretation |
|---|---:|---:|---|
| `popularity` | 0.5194 | -0.9811 | The first-pass MLP fusion head underperforms the feature-concat XGBoost floor. |
| `meanScore` | 0.0193 | -0.1173 | The first-pass MLP fusion head also underperforms on score regression. |

Paper alignment caveat: `C1-Armenta-MLP` is a loose adaptation. It does not
reproduce the paper's GPT-2 synopsis branch, GPT-2 character-description
branch, ResNet-50 character-portrait branch, or exact PyTorch MLP design.

## C1 Branch Proxy Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-ProxyBranchMLP` test R2 | Current interpretation |
|---|---:|---:|---|
| `popularity` | 0.5194 | 0.2600 | Branch-wise neural fusion is much stronger than the flat C1 MLP but still trails XGBoost concat. |
| `meanScore` | 0.0193 | 0.0398 | Branch-wise neural fusion slightly improves R2 over F2, though rank correlation is lower. |

Paper alignment caveat: `C1-Armenta-ProxyBranchMLP` is closer to the paper's
branch-fusion pattern, but it remains a proxy because current artifacts lack
main-character descriptions and main-character portraits.

## Single-Modality Snapshot

| Target | `T2-XGB-TextEmb` test R2 | `I1-XGB-ImageEmb` test R2 | `F2-XGB-Concat` test R2 | Current interpretation |
|---|---:|---:|---:|---|
| `popularity` | -0.0152 | 0.0158 | 0.5194 | Text/image embeddings alone have ranking signal, but the strong result comes from the combined metadata + embedding setup. |
| `meanScore` | -0.3846 | -0.1559 | 0.0193 | Single-modality embeddings are weak for score regression; F2 remains the least weak multimodal classical reference. |

## C2 Snapshot

| Target | `C2-CTNN-Lite` test R2 | `C2-CTNN-Lite` test Spearman | Current interpretation |
|---|---:|---:|---|
| `popularity` | 0.1716 | 0.7410 | Lightweight text-image transformer fusion improves over single-modality embedding baselines but still trails metadata+embedding XGBoost. |
| `meanScore` | -0.2602 | 0.3107 | The text-image transformer route remains weak for score regression in this first adaptation. |

Paper alignment caveat: `C2-CTNN-Lite` is not a CTNN reproduction. It omits
the full poster/review feature extraction, recurrent fusion component,
metadata-related movie factors, and box-office classification setup.

## C3 Retrieval Snapshot

These rows use full metadata + text + image features and only vary the RAG
feature source. Retrieval is generated offline from train-set knowledge only,
with temporal filtering so retrieved items are earlier than the query period.

| Target | `C3-RAG-None-XGB` test R2 | `C3-RAG-Sparse-XGB` test R2 | `C3-RAG-Dense-XGB` test R2 | `C3-RAG-Hybrid-XGB` test R2 | Current interpretation |
|---|---:|---:|---:|---:|---|
| `popularity` | 0.5064 | 0.5725 | 0.5084 | 0.4828 | Sparse metadata retrieval improves over the no-RAG control and beats the F2 feature-concat reference on R2. Dense semantic retrieval is close to no-RAG, while hybrid RRF does not improve over sparse. |
| `meanScore` | 0.0132 | 0.0730 | 0.0464 | 0.0307 | Sparse metadata retrieval is strongest in this first pass. Dense and hybrid both improve over no-RAG on rank correlation, but hybrid does not improve R2. |

Paper alignment caveat: these are SKAPP-inspired retrieval baselines, not SKAPP
reproduction. They do not implement RRCP selection, VL-GNN contextual learning,
or RRCP-Attention fusion.

## Artifact Policy

`.exp/` is ignored because it can contain large or frequently changing
experiment outputs, predictions, and logs. Reportable numbers should be copied
into tracked files under `reports/`.
