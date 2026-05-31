# Reference Baseline Handoff - 2026-05-19

本文件交接給下一個 agent，用於延續 C1 / C2 / C3 reference baseline 的
完整復現工作。請先讀完本文件，再看：

- `reports/reference_baseline_reconstruction_taskboard_2026-05-19.md`
- `reports/reference_baseline_reproduction_commands_2026-05-19.md`
- `reports/reference_baseline_paper_alignment_audit.md`
- `reports/reference_baseline_status.md`
- `reports/reference_baseline_runs.md`
- `reports/reference_baseline_results.csv`

## Current Branch And Git State

目前分支：

```text
feature/baseline-armenta-fusion
```

最近已提交 commit：

```text
79b9c1e docs(reference-baselines): record C1 and C2 proxy outcomes
bd7c01b feat(reference-baselines): add project-input neural proxy variants
435d797 feat(image): add fetch coverage summary utility
2ef1580 docs(reference-baselines): clarify mainline criteria
171db78 feat(reference-baselines): add Armenta project-input proxy
```

重要：目前有大量未提交變更。不要假設工作樹乾淨；不要 revert 使用者或其他
agent 的改動。

本輪 reference-baseline 相關未提交變更包含：

```text
.gitignore
reports/reference_baseline_paper_alignment_audit.md
reports/reference_baseline_results.csv
reports/reference_baseline_runs.md
reports/reference_baseline_status.md
src/experiment_common/features.py
src/reference_baseline_branch/build_c3_rag_features.py
src/reference_baseline_branch/configs/reference_baselines.yaml
src/reference_baseline_branch/build_gpt2_text_embeddings.py
src/reference_baseline_branch/sklearn_models.py
src/reference_baseline_branch/run_c3_skapp_full.py
```

另有 unrelated dirty / untracked files，例如：

```text
data/fetch_log.csv
src/fussion_branch/RAG/rag_builder.py
src/fussion_branch/RAG/rag_query.py
src/fussion_branch/README.md
src/fussion_branch/configs/fusion_config.yaml
src/fussion_branch/utilities/summarize_experiments.py
AGENTS.md
docs/refer/
many EDA/report/script files
```

除非使用者明確要求，請不要處理這些 unrelated files。

## Repository Baseline Layout

目前 reference baseline 主要放在：

```text
src/reference_baseline_branch/
src/experiment_common/
reports/
docs/
baseline_refer/
```

`baseline_refer/` 已加入 `.gitignore`，因為它放外部原始碼，不應提交。

已知外部參考原始碼：

```text
baseline_refer/skapp-main/
baseline_refer/Popularity-Prediction-in-Anime-with-Deep-Learning-main/
```

本地論文 PDF：

```text
docs/refer/Anime popularity prediction before huge investments a multimodal approach using deep learning.pdf
docs/refer/Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers.pdf
docs/refer/Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation.pdf
```

## Project Input Contract

使用者已明確要求：要讓 baseline 對比有價值，不能只做 motivation-level
proxy。正確標準是：

1. baseline 的輸入要盡量對齊本專案主框架輸入。
2. 在輸入限制下，模型構造和訓練流程要盡量完整重做原論文。
3. 若 domain/input 不同，應清楚標註替換點，但不可省略原論文核心構造。

本專案主輸入大致是：

```text
metadata
synopsis/text embedding
cover/banner image embedding
optional project retrieval context
```

不可為了 exact paper reproduction 直接把任務改成：

```text
C1 character-only task
C2 movie box-office task
C3 social-media UGC popularity task
```

但是若原論文核心構造依賴某種 branch/module，應在 project input 上重做等價結構。

## Current Completion Status

### C1 - Armenta-Segura & Sidorov 2025

目前狀態：已有第一版 structure-complete project-input reconstruction，並已完成 Figure 2 character-centric side reconstruction。

已完成並跑過：

```text
C1-Armenta-MLP
C1-Armenta-ProxyBranchMLP
C1-Armenta-ProjectInputProxy
C1-Armenta-ProjectInputProxy-ResNet50
C1-Armenta-ProjectInputReconstruction
C1-Armenta-Figure2Reconstruction
```

目前 C1 主線對齊版是：

```text
C1-Armenta-ProjectInputReconstruction
```

它已做：

```text
project metadata + GPT-2 synopsis embeddings
project cover/banner ResNet-50 features
synopsis branch
project-context MLP
Armenta-shaped Big MLP
```

主線仍缺或不可完全主張：

```text
main-character description / portrait branch is not used in the project-input mainline
Figure 2 side reconstruction uses character-specific inputs, so it is not the project-input baseline
raw character/portrait coverage is incomplete
paper target/split formulation documentation
```

Run 36：

```text
popularity test_MAE=10719.7513 test_R2=0.2898 Spearman=0.8192 log_MAE=1.0563
meanScore  test_MAE=9.0250    test_R2=-0.1096 Spearman=0.4666
```

C1 可寫成 structure-complete project-input reconstruction，但不能寫成 exact
reproduction。`C1-Armenta-Figure2Reconstruction` 可寫成 non-mainline Figure 2
side reconstruction，但不能寫成 project-input mainline。

Run 38：

```text
popularity test_MAE=11878.6328 test_R2=0.3556 Spearman=0.7823 log_MAE=1.1688
meanScore  test_MAE=9.7747    test_R2=-0.2172 Spearman=0.3824
```

Character artifact coverage：

```text
train rows=9583 description=4755 portrait_url=5620 encoded_portrait=4984
val   rows=2918 description=1415 portrait_url=1921 encoded_portrait=1718
test  rows=3087 description=1578 portrait_url=2193 encoded_portrait=1931
```

下一步建議：

1. 若要守住 project input，下一步是 debug/tune run 36，而不是回到舊 proxy。
2. 若要研究原 Figure 2 架構本身，可 tune run 38，但要保持 side-analysis 標籤。

### C2 - Madongo, Tang & Hassan 2023 CTNN

目前狀態：已有第一版 structure-complete project-input CTNN reconstruction。

已完成並跑過：

```text
C2-CTNN-Lite
C2-ProjectInputCrossAttention
C2-ProjectInputRecurrentFusion
C2-ProjectInputCTNNReconstruction
C2-ProjectInputCTNNDualVisualReconstruction
```

目前 C2 主線對齊版是：

```text
C2-ProjectInputCTNNReconstruction
```

Dual-visual diagnostic row：

```text
C2-ProjectInputCTNNDualVisualReconstruction
```

它已做：

```text
project metadata + GPT-2 synopsis embeddings + ResNet-50 cover/banner features
text transformer encoder
visual transformer encoder over cover/banner tokens
explicit bidirectional text-image cross-attention
GRU recurrent token fusion
metadata factor gate
```

但仍缺：

```text
movie-paper CTNN exact architecture audit
poster/review transformer feature extraction equivalent
paper recurrent fusion exact implementation
paper metadata/factors exact handling equivalent
paper target/class/range formulation adaptation
full CTNN training procedure and hyperparameters
```

Run 37：

```text
popularity test_MAE=10151.2161 test_R2=0.4608 Spearman=0.8471 log_MAE=0.9981
meanScore  test_MAE=8.1751    test_R2=0.0696 Spearman=0.5247
```

Run 39：

```text
popularity test_MAE=10214.6356 test_R2=0.4421 Spearman=0.8491 log_MAE=0.9399
meanScore  test_MAE=8.8957    test_R2=-0.0720 Spearman=0.5310
```

C2 可寫成 structure-complete project-input CTNN reconstruction；仍不能寫成
exact CTNN reproduction。

Run 39 比 run 37 更接近原文 ResNet50 + ViT poster feature 設計，因為它把
project image embeddings 當作 ViT-like visual semantic tokens 加到 ResNet-50
cover/banner tokens 旁邊。但它不是新的主線，因為 popularity R2 與 meanScore
R2 都低於 run 37；它主要用來證明 dual-visual 對齊方向已被檢查。

下一步建議：

1. 若要繼續 C2，優先 tune run 37，而不是改以 run 39 作主線。
2. 若要進一步貼近原論文，只能考慮 source-training diagnostic，例如 SGD / small batch；但 BCE/class target 會改變本專案 regression contract，需另列旁支。
3. 不要再只微調 `C2-CTNN-Lite`；那只是 first-pass baseline。

### C3 - Xu et al. 2025 SKAPP

目前狀態：已有 structure-complete first run，但性能很差，尚未可作 final
valuable comparison。

已完成並跑過：

```text
C3-RAG-None-XGB
C3-RAG-Sparse-XGB
C3-RAG-Dense-XGB
C3-RAG-Hybrid-XGB
C3-RAG-Selective-XGB
C3-ProjectInputSKAPPProxy-XGB
C3-ProjectInputSKAPPGraphProxy
C3-ProjectInputSKAPPFull
```

`C3-ProjectInputSKAPPFull` 是第一條真正跑過 SKAPP 多階段構造的路線。

它已做：

```text
SKAPP-style tensor dataset
all-items model
single/dissembled model
RRCP_silver = abs(Predict - without) - abs(Predict - with)
threshold final RRCP model
GraphLearner-style cosine graph + graph convolution
RRCP/CXMI-style weighting
final prediction
```

Run 35 結果：

```text
C3-ProjectInputSKAPPFull popularity:
  test_MAE = 14668.1228
  test_R2 = -0.4927
  test_Spearman = 0.6985
  test_log_MAE = 1.2983

C3-ProjectInputSKAPPFull meanScore:
  test_MAE = 9.8063
  test_R2 = -0.2385
  test_Spearman = 0.3657
```

這代表完整構造跑通，但訓練/正規化/架構細節未調好。不要把 run 35 當 final
performance。

目前 C3 performance row 仍是：

```text
C3-RAG-Selective-XGB
```

但它只是 performance / ablation row，不是 SKAPP reproduction。

下一步建議：

1. Debug `C3-ProjectInputSKAPPFull`。
2. 優先檢查 all-items model、single/dissembled model 是否過擬合或 target scale
   不一致。
3. 檢查 RRCP_silver distribution 是否合理。
4. 對齊 SKAPP 原始碼 `graph_variable_length.py` 和 `graph_attention.py` 細節。
5. 再做正式 run，才可判斷 full reconstruction 的對比價值。

## Important Local Artifacts

`.exp/` 被 ignore，不會進 git。重要 raw outputs：

```text
.exp/baseline/results/15   C1-Armenta-MLP
.exp/baseline/results/18   C2-CTNN-Lite
.exp/baseline/results/19   C1-Armenta-ProxyBranchMLP
.exp/baseline/results/21   C3-RAG-None-XGB
.exp/baseline/results/22   C3-RAG-Sparse-XGB
.exp/baseline/results/23   C3-RAG-Dense-XGB
.exp/baseline/results/24   C3-RAG-Hybrid-XGB
.exp/baseline/results/25   C3-RAG-Selective-XGB
.exp/baseline/results/26   C1-Armenta-ProjectInputProxy
.exp/baseline/results/27   C2-ProjectInputCrossAttention
.exp/baseline/results/28   C2-ProjectInputRecurrentFusion
.exp/baseline/results/30   C1-Armenta-ProjectInputProxy-ResNet50
.exp/baseline/results/33   C3-ProjectInputSKAPPProxy-XGB
.exp/baseline/results/34   C3-ProjectInputSKAPPGraphProxy
.exp/baseline/results/35   C3-ProjectInputSKAPPFull
.exp/baseline/results/36   C1-Armenta-ProjectInputReconstruction
.exp/baseline/results/37   C2-ProjectInputCTNNReconstruction
.exp/baseline/results/38   C1-Armenta-Figure2Reconstruction
.exp/baseline/results/39   C2-ProjectInputCTNNDualVisualReconstruction
```

Important C1/C2 generated artifacts:

```text
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_train.parquet
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_val.parquet
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_test.parquet
.exp/baseline/c1_character_features/c1_character_features_train.parquet
.exp/baseline/c1_character_features/c1_character_features_val.parquet
.exp/baseline/c1_character_features/c1_character_features_test.parquet
.exp/baseline/c1_character_images/
```

Important C3 generated artifacts:

```text
.exp/baseline/rag_features/skapp_proxy/
.exp/baseline/rag_features/skapp_graph_proxy/
.exp/baseline/skapp_full/dataset/
.exp/baseline/results/35/rrcp_silver_popularity_train.npz
.exp/baseline/results/35/rrcp_silver_popularity_val.npz
.exp/baseline/results/35/rrcp_silver_popularity_test.npz
.exp/baseline/results/35/rrcp_silver_meanScore_train.npz
.exp/baseline/results/35/rrcp_silver_meanScore_val.npz
.exp/baseline/results/35/rrcp_silver_meanScore_test.npz
```

## Known Validation Commands

已通過：

```bash
python -m py_compile src/experiment_common/features.py src/reference_baseline_branch/build_c3_rag_features.py src/reference_baseline_branch/sklearn_models.py src/reference_baseline_branch/run_c3_skapp_full.py
python -c "import yaml; yaml.safe_load(open('src/reference_baseline_branch/configs/reference_baselines.yaml', encoding='utf-8')); print('yaml ok')"
git diff --check -- src/reference_baseline_branch/build_c3_rag_features.py src/reference_baseline_branch/run_c3_skapp_full.py src/reference_baseline_branch/configs/reference_baselines.yaml reports/reference_baseline_results.csv reports/reference_baseline_runs.md reports/reference_baseline_status.md reports/reference_baseline_paper_alignment_audit.md
```

PowerShell profile 會噴 `Microsoft.WinGet.CommandNotFound` 的訊息，這是環境
雜訊，不是專案錯誤。

## Caution For Next Agent

1. 不要把 `baseline_refer/` 或 `.exp/` 加進 commit。
2. 不要把 proxy 說成 reproduction。
3. 不要回頭只調 XGBoost aggregate proxy 來假裝 C3 完整。
4. 若要 commit，請依使用者規則分批，用英文 Conventional Commits，body 用
   numbered bullets。
5. 若遇到 `.git/index.lock` 或 git 寫入權限問題，使用 escalation。
6. 工作樹有 unrelated dirty files，請只 stage 本任務相關檔案。
