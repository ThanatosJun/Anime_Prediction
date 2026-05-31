# Reference Baseline Reconstruction Taskboard - 2026-05-19

本 taskboard 是給下一個 agent 的執行清單。最高原則：baseline 對比要有價值，
就必須完整重做原論文主要構造。既有 proxy 可保留作 ablation / milestone，
但不能取代 final reconstruction。

## Priority 0 - Protect Current Work

Status: pending for next agent

1. 先讀 `reports/reference_baseline_handoff_2026-05-19.md`。
2. 跑 `git status --short`，確認目前未提交檔案。
3. 不要 revert unrelated dirty files。
4. 不要 commit `.exp/`、`baseline_refer/`、`data/fetch_log.csv`。
5. 若要 commit，分批：
   - code changes
   - config changes
   - report/documentation changes

## Priority 1 - C3 SKAPPFull Debug And Stabilization

Current status: structure-complete first run exists, but performance is poor.

Primary files:

```text
src/reference_baseline_branch/run_c3_skapp_full.py
src/reference_baseline_branch/build_c3_rag_features.py
src/reference_baseline_branch/configs/reference_baselines.yaml
reports/reference_baseline_results.csv
reports/reference_baseline_runs.md
reports/reference_baseline_status.md
reports/reference_baseline_paper_alignment_audit.md
baseline_refer/skapp-main/src/
```

Current run:

```text
.exp/baseline/results/35
```

Current symptoms:

```text
popularity test_R2 = -0.4927
popularity test_log_MAE = 1.2983
meanScore test_R2 = -0.2385
```

Do not discard this row. It is the first structure-complete project-input SKAPP
pipeline. Debug it.

### C3 Debug Checklist

1. Verify dataset tensor correctness

   Check:

   ```text
   .exp/baseline/skapp_full/dataset/train.npz
   .exp/baseline/skapp_full/dataset/val.npz
   .exp/baseline/skapp_full/dataset/test.npz
   ```

   Validate:

   ```text
   ids align with meta split
   query_text / query_image nonzero coverage
   retrieved_text / retrieved_image nonzero coverage
   retrieved_label scaling per target
   retrieved_mask mean and empty rows
   retrieval temporal constraint
   ```

2. Compare `run_c3_skapp_full.py` to SKAPP source

   Source files:

   ```text
   baseline_refer/skapp-main/src/RRCP/predict_model.py
   baseline_refer/skapp-main/src/RRCP/RRCP.py
   baseline_refer/skapp-main/src/RRCP_prediction_variable_lenth.py
   baseline_refer/skapp-main/src/graph_attention.py
   baseline_refer/skapp-main/src/graph_variable_length.py
   ```

   Differences to inspect:

   ```text
   hidden_dim 768 in source vs d_model 128 in run 35
   MultiheadAttention behavior in all-items model
   label_embedding_linear dimensions
   graph_norm_ours implementation
   mask indexing in preprocess_data
   RRCP threshold behavior
   CXMI normalization when all weights are zero/negative
   ```

3. Inspect all-items model quality

   Add diagnostics:

   ```text
   all-items train/val/test prediction metrics before RRCP_silver
   single/dissembled train/val/test metrics
   without-retrieval vs with-retrieval prediction distributions
   ```

   Rationale: if all-items model is weak, RRCP_silver becomes noisy.

4. Inspect RRCP_silver

   Already observed in run 35:

   ```text
   popularity train mean = 0.01533, positive ratio = 0.52734
   popularity val mean = 0.01594, positive ratio = 0.52632
   popularity test mean = 0.01254, positive ratio = 0.51931
   meanScore train mean = -0.00236, positive ratio = 0.49060
   meanScore val mean = 0.00550, positive ratio = 0.50703
   meanScore test mean = 0.00410, positive ratio = 0.50855
   ```

   Next diagnostics:

   ```text
   RRCP_silver quantiles
   number of selected items after threshold
   selected item target similarity
   final prediction error by selected_count
   final prediction error by popularity quintile
   ```

5. Improve C3Full carefully

   Candidate changes:

   ```text
   try d_model=256 or 768 if GPU memory permits
   lower learning rate for final model
   train all-items and single models longer than final model
   save best states and diagnostics per stage
   test threshold_of_rrcp values: 0, small positive quantile, top-m
   compare top_k 10 vs 20 before attempting 500
   add dropout / layernorm if overfitting
   ```

   Do not jump directly to `retrieval_num=500`; source uses 500 for UGC
   benchmark, but project compute/data constraints differ.

## Priority 2 - C1 Full Reconstruction

Current status: project-input reconstruction and Figure 2 side reconstruction are both done.

Primary reference:

```text
docs/refer/Anime popularity prediction before huge investments a multimodal approach using deep learning.pdf
baseline_refer/Popularity-Prediction-in-Anime-with-Deep-Learning-main/
```

Existing project rows:

```text
C1-Armenta-MLP
C1-Armenta-ProxyBranchMLP
C1-Armenta-ProjectInputProxy
C1-Armenta-ProjectInputProxy-ResNet50
C1-Armenta-ProjectInputReconstruction
C1-Armenta-Figure2Reconstruction
```

Best current C1 alignment row:

```text
C1-Armenta-ProjectInputReconstruction
```

Current run:

```text
.exp/baseline/results/36
.exp/baseline/results/38
```

Completed C1 reconstruction work:

1. `C1-Armenta-ProjectInputReconstruction` keeps the project input contract and uses GPT-2 synopsis + ResNet-50 cover/banner features.
2. `C1-Armenta-Figure2Reconstruction` uses GPT-2 synopsis, GPT-2 main-character description/name, ResNet-50 main-character portraits, source-shaped character MLP, and Big MLP.
3. Character artifacts are saved under `.exp/baseline/c1_character_features/` and use zero-filled vectors when raw character/portrait data is missing.

Remaining work if C1 needs another quality pass:

1. Tune `C1-Armenta-ProjectInputReconstruction` and `C1-Armenta-Figure2Reconstruction` hyperparameters.
2. Decide whether to increase main-character portrait count or keep `max_characters=5`.
3. Re-check Table 8 / Table 9 details if the paper or source reveals exact dropout/activation differences not already mirrored.
4. Keep branch mapping documented:

   ```text
   mainline: paper synopsis GPT-2 branch -> project synopsis GPT-2 artifact
   mainline: paper character branch -> project metadata + cover/banner ResNet-50 context MLP
   side: paper character description GPT-2 branch -> raw AniList main-character description/name artifact
   side: paper portrait ResNet-50 branch -> raw AniList main-character portrait artifact
   ```

5. Run both targets again and update tracked reports if tuning changes results.

Implementation suggestion:

```text
src/reference_baseline_branch/build_c1_character_features.py
src/reference_baseline_branch/sklearn_models.py
```

Current Figure 2 reconstruction is implemented inside `sklearn_models.py` as a
torch-backed sklearn-compatible regressor. Only split it into a dedicated runner
if future work requires end-to-end encoder fine-tuning.

## Priority 3 - C2 Full Reconstruction

Current status: structure-complete project-input CTNN reconstruction and dual-visual diagnostic done.

Primary reference:

```text
docs/refer/Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers.pdf
```

Existing project rows:

```text
C2-CTNN-Lite
C2-ProjectInputCrossAttention
C2-ProjectInputRecurrentFusion
C2-ProjectInputCTNNReconstruction
C2-ProjectInputCTNNDualVisualReconstruction
```

Best current C2 alignment row:

```text
C2-ProjectInputCTNNReconstruction
```

Current run:

```text
.exp/baseline/results/37
.exp/baseline/results/39
```

Completed C2 reconstruction work:

1. `C2-ProjectInputCTNNReconstruction` restores the main CTNN stack under project inputs:

   ```text
   GPT-2 synopsis features
   ResNet-50 cover/banner features
   modality transformer encoders
   bidirectional cross-modal attention
   recurrent fusion
   metadata factor gate
   ```

2. `C2-ProjectInputCTNNDualVisualReconstruction` checks the paper's ResNet50 + ViT poster feature idea:

   ```text
   ResNet50 poster stream -> anime cover/banner ResNet-50 features
   ViT poster stream -> project image embeddings as ViT-like visual semantic tokens
   ```

Remaining work if C2 must move closer to the original CTNN:

1. Tune `C2-ProjectInputCTNNReconstruction`; it remains the C2 mainline by R2.
2. Keep `C2-ProjectInputCTNNDualVisualReconstruction` as source-alignment diagnostic unless later tuning makes it clearly better.
3. If needed, add a separate source-training diagnostic for SGD / small batch, but do not mix BCE/classification target into the main regression row.
4. Do not pursue exact movie-review/poster reproduction unless the project explicitly adds a movie-style benchmark.

Current implementation:

```text
src/reference_baseline_branch/sklearn_models.py
src/reference_baseline_branch/configs/reference_baselines.yaml
```

## Priority 4 - Documentation Hygiene

Current docs have grown organically. Keep them coherent.

Important tracked docs:

```text
docs/baseline_reference_implementation_plan.md
docs/baseline_directory_planning.md
reports/reference_baseline_handoff_2026-05-19.md
reports/reference_baseline_reconstruction_taskboard_2026-05-19.md
reports/reference_baseline_reproduction_commands_2026-05-19.md
reports/reference_baseline_paper_alignment_audit.md
reports/reference_baseline_status.md
reports/reference_baseline_runs.md
reports/reference_baseline_results.csv
```

Documentation rules:

1. Keep `reference_baseline_results.csv` as compact metrics table.
2. Keep `reference_baseline_runs.md` as run/artifact index and snapshots.
3. Keep `reference_baseline_paper_alignment_audit.md` as claim boundary and
   paper-to-project mapping.
4. Keep `reference_baseline_status.md` as chronological status log.
5. Use this handoff/taskboard only for agent continuity; update or supersede it
   when a new major milestone is reached.

## Definition Of Done For Final Baseline Comparison

C1 done means:

```text
full Armenta branch architecture is rebuilt under project input contract
input substitutions are explicit
both targets run
results recorded
claim boundary written
```

C2 done means:

```text
full CTNN architecture is rebuilt under project input contract
poster/review substitutions are explicit
cross-modal transformer and recurrent fusion match paper structure as much as possible
both targets run
results recorded
claim boundary written
```

C3 done means:

```text
SKAPPFull is stable enough for comparison
all-items/single/final stage diagnostics are available
RRCP_silver is not degenerate
final performance and failure modes are documented
both targets run
results recorded
claim boundary written
```
