# Reference Baseline Runs

Updated: 2026-05-14

本文件是 reference baseline 的閱讀入口與 run ledger。Raw `.exp/`
outputs 是本機實驗產物，不提交；可報告的數字統一整理到
`reports/baselines/reference_baseline_results.csv`。

## 文件用途與閱讀順序

| 文件 | 必須存在的理由 | 何時閱讀 |
|---|---|---|
| `reports/baselines/reference_baseline_runs.md` | Reference baseline 入口；整理已跑哪些 route、每個 run 的來源、目前結論。 | 組員第一次接手或要快速同步時先看。 |
| `reports/baselines/reference_baseline_results.csv` | 可攜帶的數字結果表；取代 ignored `.exp/` raw outputs。 | 要畫表、比較 R2/MAE/Spearman、寫實驗結果時看。 |
| `reports/baselines/reference_baseline_paper_alignment_audit.md` | 定義 C1/C2/C3 是否對齊論文、哪些話可以寫、哪些不能寫。 | 要寫論文/報告 claim，或決定下一條 reference route 時看。 |
| `reports/baselines/reference_baseline_status.md` | 詳細工作紀錄；保留實作脈絡、指令、歷史解讀。 | 需要追溯「為什麼這樣做」或 debug 舊決策時看，不作為第一入口。 |
| `docs/baseline_reference_implementation_plan.md` | 初始 reference baseline roadmap；說明 route 0、1、2 的設計來源。 | 要理解原始規劃時看；目前狀態以本文件與 status 為準。 |
| `docs/baseline_directory_planning.md` | 說明 `reference_baseline_branch`、`ablation_branch`、`experiment_common` 的目錄分工。 | 新增 baseline 程式或調整目錄時看。 |
| `docs/reference_baselines/f2_feature_concat_plan.md` | F2 feature-concat route 的實作規劃與輸入契約。 | 只在維護 F2 或解釋 no-RAG multimodal floor 時看。 |
| `docs/rq2_rag_ablation_plan.md` | RQ2/RAG ablation 的問題定義與實驗設計。 | 只在處理 EXP2/RAG ablation，不是 reference overview 時看。 |
| `reports/baselines/reference_baseline_weekly_sync_2026-05-12.md` | dated snapshot；保留當日同步紀錄。 | 僅供歷史追溯；不要當作最新狀態。 |

目前主線判準只有兩條：第一，baseline 輸入要對齊本專案主框架；第二，在該輸入限制下，模型方法要盡量還原原論文設計。

## Tracked Result Table

```text
reports/baselines/reference_baseline_results.csv
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
| `.exp/baseline/results/33` | `C3-ProjectInputSKAPPProxy-XGB` | local raw output | Source for learned contribution filtering and attention-weighted C3 SKAPP-style proxy results. |
| `.exp/baseline/results/34` | `C3-ProjectInputSKAPPGraphProxy` | local raw output | Source for retrieved-set tensor, RRCP-mask, graph/attention SKAPP-style proxy results. |
| `.exp/baseline/results/35` | `C3-ProjectInputSKAPPFull` | local raw output | Source for structure-complete project-input SKAPP run with all-items model, single/dissembled model, RRCP_silver, and final RRCP/CXMI-style prediction. |
| `.exp/baseline/results/36` | `C1-Armenta-ProjectInputReconstruction` | local raw output | Source for GPT-2 synopsis + ResNet-50 cover/banner project-input Armenta reconstruction results. |
| `.exp/baseline/results/37` | `C2-ProjectInputCTNNReconstruction` | local raw output | Source for GPT-2/ResNet-50 project-input CTNN reconstruction results. |
| `.exp/baseline/results/38` | `C1-Armenta-Figure2Reconstruction` | local raw output | Source for character-description + portrait Figure 2 side reconstruction results. |
| `.exp/baseline/results/39` | `C2-ProjectInputCTNNDualVisualReconstruction` | local raw output | Source for GPT-2 + ResNet-50 + project-image dual-visual CTNN diagnostic results. |

## Completed Routes

| Plan route | Baseline IDs | Completion status |
|---|---|---|
| `0. Lowest Reference / lowest floor` | `F0-Mean`, `F0-Ridge-Meta` | done |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | done as adaptation |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | done as adaptation |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP`, `C1-Armenta-ProxyBranchMLP`, `C1-Armenta-ProjectInputProxy`, `C1-Armenta-ProjectInputProxy-ResNet50`, `C1-Armenta-ProjectInputReconstruction`, `C1-Armenta-Figure2Reconstruction` | first-pass/proxy rows, structure-complete project-input reconstruction, and Figure 2 side reconstruction done |
| `2.2 Cross-modal Transformer Fusion` | `C2-CTNN-Lite`, `C2-ProjectInputCrossAttention`, `C2-ProjectInputRecurrentFusion`, `C2-ProjectInputCTNNReconstruction`, `C2-ProjectInputCTNNDualVisualReconstruction` | first-pass/proxy rows, structure-complete project-input CTNN reconstruction, and dual-visual diagnostic done |
| `2.3 Retrieval / RAG Competitive Baseline` | `C3-RAG-None-XGB`, `C3-RAG-Sparse-XGB`, `C3-RAG-Dense-XGB`, `C3-RAG-Hybrid-XGB`, `C3-RAG-Selective-XGB`, `C3-ProjectInputSKAPPProxy-XGB`, `C3-ProjectInputSKAPPGraphProxy`, `C3-ProjectInputSKAPPFull` | first-pass retrieval baselines, aggregate proxy, graph proxy, and first structure-complete SKAPP reconstruction run done |

## C1 vs F2 Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-MLP` test R2 | Current interpretation |
|---|---:|---:|---|
| `popularity` | 0.5194 | -0.9811 | The first-pass MLP fusion head underperforms the feature-concat XGBoost floor. |
| `meanScore` | 0.0193 | -0.1173 | The first-pass MLP fusion head also underperforms on score regression. |

Paper alignment caveat: `C1-Armenta-MLP` is a loose adaptation. It does not
reproduce the paper's GPT-2 synopsis branch, GPT-2 character-description
branch, ResNet-50 character-portrait branch, or exact PyTorch MLP design.

## C1 Branch Proxy Snapshot

| Target | `F2-XGB-Concat` test R2 | `C1-Armenta-ProxyBranchMLP` test R2 | `C1-Armenta-ProjectInputProxy` test R2 | `C1-Armenta-ProjectInputProxy-ResNet50` test R2 | `C1-Armenta-ProjectInputReconstruction` test R2 | `C1-Armenta-Figure2Reconstruction` test R2 | Current interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `popularity` | 0.5194 | 0.2600 | 0.3819 | 0.3817 | 0.2898 | 0.3556 | GPT-2 + ResNet-50 project-input reconstruction improves paper alignment but not performance; the Figure 2 side reconstruction is closer to the original character-centric architecture but is not the project-input main row. |
| `meanScore` | 0.0193 | 0.0398 | -0.0678 | -0.0633 | -0.1096 | -0.2172 | Structure completion and Figure 2 side reconstruction do not improve score generalization; these rows are valuable for alignment, not as the strongest C1 model. |

Paper alignment update: `C1-Armenta-ProjectInputReconstruction` now uses
GPT-2 pooled synopsis embeddings and ImageNet ResNet-50 cover/banner features,
then keeps the Armenta-shaped synopsis branch, project-context MLP, and Big MLP.
It is structure-complete under the project input contract, but still not exact
paper reproduction because character descriptions/portraits and the original
split/target formulation are intentionally not used.

Figure 2 side update: `C1-Armenta-Figure2Reconstruction` now uses GPT-2
synopsis embeddings, GPT-2 main-character description/name embeddings, ResNet-50
main-character portrait features, the source-shaped character MLP, and the
Armenta Big MLP. It is closer to the paper's Figure 2 architecture, but is not
the main project-input comparison row because it uses character-specific inputs
instead of the project cover/banner contract and raw character coverage is
incomplete.

## Single-Modality Snapshot

| Target | `T2-XGB-TextEmb` test R2 | `I1-XGB-ImageEmb` test R2 | `F2-XGB-Concat` test R2 | Current interpretation |
|---|---:|---:|---:|---|
| `popularity` | -0.0152 | 0.0158 | 0.5194 | Text/image embeddings alone have ranking signal, but the strong result comes from the combined metadata + embedding setup. |
| `meanScore` | -0.3846 | -0.1559 | 0.0193 | Single-modality embeddings are weak for score regression; F2 remains the least weak multimodal classical reference. |

## C2 Snapshot

| Target | `C2-CTNN-Lite` test R2 | `C2-ProjectInputCrossAttention` test R2 | `C2-ProjectInputRecurrentFusion` test R2 | `C2-ProjectInputCTNNReconstruction` test R2 | `C2-ProjectInputCTNNDualVisualReconstruction` test R2 | Best C2 Spearman | Current interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| `popularity` | 0.1716 | 0.3704 | 0.3545 | 0.4608 | 0.4421 | 0.8491 | The single-visual structure-complete CTNN row remains best by R2, while the dual-visual diagnostic slightly improves rank correlation and log_MAE. |
| `meanScore` | -0.2602 | 0.0597 | 0.0670 | 0.0696 | -0.0720 | 0.5310 | Dual visual improves Spearman but hurts R2/MAE, so it is useful as source-alignment evidence rather than the strongest score row. |

Paper alignment update: `C2-ProjectInputCTNNReconstruction` keeps this
project's metadata/synopsis/cover/banner input contract, but restores the
paper's major CTNN stages: text and visual transformer encoders, bidirectional
cross-modal attention, GRU recurrent fusion, and metadata factor gating. It is
still not exact CTNN reproduction because it does not use movie reviews, movie
posters, box-office classes, or the original movie dataset.

Dual-visual update: `C2-ProjectInputCTNNDualVisualReconstruction` adds the
project's existing image embeddings as a ViT-like visual semantic stream beside
ResNet-50 cover/banner features. This mirrors the paper's ResNet50+ViT poster
feature idea more closely, but it did not beat the single-visual reconstruction
on R2. It should be kept as a source-alignment diagnostic, not as the primary C2
performance row.

## C1/C2 Structure-Complete Runs

2026-05-19 新增並執行：

```bash
python -m src.reference_baseline_branch.build_gpt2_text_embeddings --splits train val test --batch-size 16 --device auto --local-files-only
python -m src.reference_baseline_branch.build_c1_character_features --splits train val test --batch-size 32 --portrait-batch-size 32 --download-workers 24 --max-characters 5 --device auto --local-files-only
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-ProjectInputReconstruction --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCTNNReconstruction --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-Figure2Reconstruction --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCTNNDualVisualReconstruction --include-disabled
```

Artifacts:

```text
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_{train,val,test}.parquet
.exp/baseline/c1_character_features/c1_character_features_{train,val,test}.parquet
.exp/baseline/results/36
.exp/baseline/results/37
.exp/baseline/results/38
.exp/baseline/results/39
```

| Baseline | Target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C1-Armenta-ProjectInputReconstruction` | popularity | 10719.7513 | 0.2898 | 0.8192 | 1.0563 |
| `C1-Armenta-ProjectInputReconstruction` | meanScore | 9.0250 | -0.1096 | 0.4666 |  |
| `C2-ProjectInputCTNNReconstruction` | popularity | 10151.2161 | 0.4608 | 0.8471 | 0.9981 |
| `C2-ProjectInputCTNNReconstruction` | meanScore | 8.1751 | 0.0696 | 0.5247 |  |
| `C1-Armenta-Figure2Reconstruction` | popularity | 11878.6328 | 0.3556 | 0.7823 | 1.1688 |
| `C1-Armenta-Figure2Reconstruction` | meanScore | 9.7747 | -0.2172 | 0.3824 |  |
| `C2-ProjectInputCTNNDualVisualReconstruction` | popularity | 10214.6356 | 0.4421 | 0.8491 | 0.9399 |
| `C2-ProjectInputCTNNDualVisualReconstruction` | meanScore | 8.8957 | -0.0720 | 0.5310 |  |

Character artifact coverage:

| split | rows | has character description | has portrait URL | encoded portrait |
|---|---:|---:|---:|---:|
| train | 9583 | 4755 | 5620 | 4984 |
| val | 2918 | 1415 | 1921 | 1718 |
| test | 3087 | 1578 | 2193 | 1931 |

## C3 Retrieval Snapshot

These rows use full metadata + text + image features and only vary the RAG
feature source. Retrieval is generated offline from train-set knowledge only,
with temporal filtering so retrieved items are earlier than the query period.

| Target | `C3-RAG-None-XGB` test R2 | `C3-RAG-Selective-XGB` test R2 | `C3-ProjectInputSKAPPProxy-XGB` test R2 | `C3-ProjectInputSKAPPGraphProxy` test R2 | `C3-ProjectInputSKAPPFull` test R2 | Current interpretation |
|---|---:|---:|---:|---:|---:|---|
| `popularity` | 0.5064 | 0.5775 | 0.5170 | 0.4404 | -0.4927 | Full reconstruction now runs through all SKAPP stages, but is not tuned and currently underperforms badly; selective sparse remains the performance row. |
| `meanScore` | 0.0132 | 0.0905 | 0.0744 | 0.0690 | -0.2385 | The structure-complete row is architecturally more valuable than the proxies, but its current optimization/generalization is not acceptable yet. |

Paper alignment caveat: these are SKAPP-inspired retrieval baselines, not SKAPP
reproduction. `C3-ProjectInputSKAPPFull` is the first row that runs the actual
SKAPP stage structure under the project input contract: SKAPP-style tensor
dataset, all-items model, single/dissembled model, RRCP_silver generation,
thresholded final RRCP prediction, GraphLearner-style fusion, and RRCP/CXMI-style
feature weighting. Its first run is performance-poor, so it should be treated as
the reconstruction target that needs tuning/debugging, while `C3-RAG-Selective-XGB`
remains the strongest C3 performance row.

## Artifact Policy

`.exp/` is ignored because it can contain large or frequently changing
experiment outputs, predictions, and logs. Reportable numbers should be copied
into tracked files under `reports/`.
