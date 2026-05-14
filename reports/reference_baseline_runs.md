# Reference Baseline Runs

Updated: 2026-05-14

本文件是 reference baseline 的閱讀入口與 run ledger。Raw `.exp/`
outputs 是本機實驗產物，不提交；可報告的數字統一整理到
`reports/reference_baseline_results.csv`。

## 文件用途與閱讀順序

| 文件 | 必須存在的理由 | 何時閱讀 |
|---|---|---|
| `reports/reference_baseline_runs.md` | Reference baseline 入口；整理已跑哪些 route、每個 run 的來源、目前結論。 | 組員第一次接手或要快速同步時先看。 |
| `reports/reference_baseline_results.csv` | 可攜帶的數字結果表；取代 ignored `.exp/` raw outputs。 | 要畫表、比較 R2/MAE/Spearman、寫實驗結果時看。 |
| `reports/reference_baseline_paper_alignment_audit.md` | 定義 C1/C2/C3 是否對齊論文、哪些話可以寫、哪些不能寫。 | 要寫論文/報告 claim，或決定下一條 reference route 時看。 |
| `reports/reference_baseline_status.md` | 詳細工作紀錄；保留實作脈絡、指令、歷史解讀。 | 需要追溯「為什麼這樣做」或 debug 舊決策時看，不作為第一入口。 |
| `docs/baseline_reference_implementation_plan.md` | 初始 reference baseline roadmap；說明 route 0、1、2 的設計來源。 | 要理解原始規劃時看；目前狀態以本文件與 status 為準。 |
| `docs/baseline_directory_planning.md` | 說明 `reference_baseline_branch`、`ablation_branch`、`experiment_common` 的目錄分工。 | 新增 baseline 程式或調整目錄時看。 |
| `docs/reference_baselines/f2_feature_concat_plan.md` | F2 feature-concat route 的實作規劃與輸入契約。 | 只在維護 F2 或解釋 no-RAG multimodal floor 時看。 |
| `docs/rq2_rag_ablation_plan.md` | RQ2/RAG ablation 的問題定義與實驗設計。 | 只在處理 EXP2/RAG ablation，不是 reference overview 時看。 |
| `reports/reference_baseline_weekly_sync_2026-05-12.md` | dated snapshot；保留當日同步紀錄。 | 僅供歷史追溯；不要當作最新狀態。 |

目前主線判準只有兩條：第一，baseline 輸入要對齊本專案主框架；第二，在該輸入限制下，模型方法要盡量還原原論文設計。

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
| `.exp/baseline/results/27` | `C2-ProjectInputCrossAttention` | local raw output | Source for completed project-input cross-attention fusion results. |
| `.exp/baseline/results/28` | `C2-ProjectInputRecurrentFusion` | local raw output | Source for completed project-input recurrent fusion results. |
| `.exp/baseline/results/19` | `C1-Armenta-ProxyBranchMLP` | local raw output | Source for branch-wise anime-domain multimodal MLP proxy adaptation results. |
| `.exp/baseline/results/21` | `C3-RAG-None-XGB` | local raw output | Source for no-retrieval control under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/22` | `C3-RAG-Sparse-XGB` | local raw output | Source for sparse metadata retrieval under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/23` | `C3-RAG-Dense-XGB` | local raw output | Source for dense semantic retrieval under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/24` | `C3-RAG-Hybrid-XGB` | local raw output | Source for hybrid sparse+dense retrieval under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/25` | `C3-RAG-Selective-XGB` | local raw output | Source for selective sparse retrieval under the SKAPP-inspired C3 route. |
| `.exp/baseline/results/26` | `C1-Armenta-ProjectInputProxy` | local raw output | Source for project-input Armenta-shaped C1 proxy results. |
| `.exp/baseline/results/30` | `C1-Armenta-ProjectInputProxy-ResNet50` | local raw output | Source for project-input Armenta-shaped C1 proxy with ImageNet ResNet-50 cover/banner features. |

## Completed Routes

| Plan route | Baseline IDs | Completion status |
|---|---|---|
| `0. Lowest Reference / lowest floor` | `F0-Mean`, `F0-Ridge-Meta` | done |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | done as adaptation |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | done as adaptation |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP`, `C1-Armenta-ProxyBranchMLP`, `C1-Armenta-ProjectInputProxy`, `C1-Armenta-ProjectInputProxy-ResNet50` | first-pass, branch-wise proxy, project-input Armenta-shaped proxy, and ResNet-50 visual-encoder proxy done as adaptations |
| `2.2 Cross-modal Transformer Fusion` | `C2-CTNN-Lite`, `C2-ProjectInputCrossAttention`, `C2-ProjectInputRecurrentFusion` | first-pass, project-input cross-attention, and recurrent-fusion proxies done as adaptations |
| `2.3 Retrieval / RAG Competitive Baseline` | `C3-RAG-None-XGB`, `C3-RAG-Sparse-XGB`, `C3-RAG-Dense-XGB`, `C3-RAG-Hybrid-XGB`, `C3-RAG-Selective-XGB` | first-pass SKAPP-inspired retrieval baselines done, including simple contribution filtering |

## C1 vs F2 Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-MLP` test R2 | Current interpretation |
|---|---:|---:|---|
| `popularity` | 0.5194 | -0.9811 | The first-pass MLP fusion head underperforms the feature-concat XGBoost floor. |
| `meanScore` | 0.0193 | -0.1173 | The first-pass MLP fusion head also underperforms on score regression. |

Paper alignment caveat: `C1-Armenta-MLP` is a loose adaptation. It does not
reproduce the paper's GPT-2 synopsis branch, GPT-2 character-description
branch, ResNet-50 character-portrait branch, or exact PyTorch MLP design.

## C1 Branch Proxy Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-ProxyBranchMLP` test R2 | `C1-Armenta-ProjectInputProxy` test R2 | `C1-Armenta-ProjectInputProxy-ResNet50` test R2 | Current interpretation |
|---|---:|---:|---:|---:|---|
| `popularity` | 0.5194 | 0.2600 | 0.3819 | 0.3817 | ResNet-50 cover/banner features make the visual encoder closer to Armenta, but do not materially improve popularity R2 over the existing project-input proxy. |
| `meanScore` | 0.0193 | 0.0398 | -0.0678 | -0.0633 | ResNet-50 slightly improves score R2/MAE versus ProjectInputProxy, but remains below the earlier branch proxy and F2 concat on score. |

Paper alignment caveat: `C1-Armenta-ProjectInputProxy-ResNet50` improves the
visual encoder alignment by using ImageNet ResNet-50 avg-pool features from
project cover/banner images. It still remains a project-input proxy because it
uses project synopsis embeddings and metadata rather than GPT-2 synopsis /
character-description branches, main-character portraits, and the original
split/target formulation.

## Single-Modality Snapshot

| Target | `T2-XGB-TextEmb` test R2 | `I1-XGB-ImageEmb` test R2 | `F2-XGB-Concat` test R2 | Current interpretation |
|---|---:|---:|---:|---|
| `popularity` | -0.0152 | 0.0158 | 0.5194 | Text/image embeddings alone have ranking signal, but the strong result comes from the combined metadata + embedding setup. |
| `meanScore` | -0.3846 | -0.1559 | 0.0193 | Single-modality embeddings are weak for score regression; F2 remains the least weak multimodal classical reference. |

## C2 Snapshot

| Target | `C2-CTNN-Lite` test R2 | `C2-ProjectInputCrossAttention` test R2 | `C2-ProjectInputRecurrentFusion` test R2 | Recurrent test Spearman | Current interpretation |
|---|---:|---:|---:|---:|---|
| `popularity` | 0.1716 | 0.3704 | 0.3545 | 0.8498 | Recurrent fusion improves rank correlation but does not beat CrossAttention by R2; both stronger proxies still trail `F2-XGB-Concat` and C3 selective retrieval. |
| `meanScore` | -0.2602 | 0.0597 | 0.0670 | 0.4723 | Recurrent fusion slightly improves score R2/Spearman over CrossAttention and turns C2 into a positive score-regression baseline. |

Paper alignment caveat: neither C2 row is a CTNN reproduction.
`C2-ProjectInputCrossAttention` and `C2-ProjectInputRecurrentFusion` are the
stronger project-input proxies because they keep this project's metadata,
synopsis/text, and cover/banner inputs while adding explicit bidirectional
text-image cross-attention, metadata-conditioned fusion, and in the recurrent
row a GRU token-fusion step.

## C3 Retrieval Snapshot

These rows use full metadata + text + image features and only vary the RAG
feature source. Retrieval is generated offline from train-set knowledge only,
with temporal filtering so retrieved items are earlier than the query period.

| Target | `C3-RAG-None-XGB` test R2 | `C3-RAG-Sparse-XGB` test R2 | `C3-RAG-Dense-XGB` test R2 | `C3-RAG-Hybrid-XGB` test R2 | `C3-RAG-Selective-XGB` test R2 | Current interpretation |
|---|---:|---:|---:|---:|---:|---|
| `popularity` | 0.5064 | 0.5725 | 0.5084 | 0.4828 | 0.5775 | Selective sparse retrieval is currently strongest by R2 and Spearman. It improves slightly over plain sparse retrieval, while dense and hybrid do not. |
| `meanScore` | 0.0132 | 0.0730 | 0.0464 | 0.0307 | 0.0905 | Selective sparse retrieval is also strongest by R2. Hybrid has competitive rank correlation, but the simple contribution filter gives the best regression result. |

Paper alignment caveat: these are SKAPP-inspired retrieval baselines, not SKAPP
reproduction. `C3-RAG-Selective-XGB` is only a deterministic median-threshold
contribution proxy; it does not implement SKAPP RRCP, VL-GNN contextual
learning, or RRCP-Attention fusion. The next useful C3 step is
`C3-ProjectInputSKAPPProxy`: keep project historical-anime retrieval first,
then move the model closer to SKAPP via learned contribution scoring and
retrieved-set graph/attention fusion.

## Artifact Policy

`.exp/` is ignored because it can contain large or frequently changing
experiment outputs, predictions, and logs. Reportable numbers should be copied
into tracked files under `reports/`.
