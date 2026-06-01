# Reference Baseline Reproduction Commands - 2026-05-19

本文件是下一個 agent 接手 C1 / C2 / C3 baseline 復現時的指令索引。
背景與判斷請先看：

- `reports/baselines/reference_baseline_handoff_2026-05-19.md`
- `reports/baselines/reference_baseline_reconstruction_taskboard_2026-05-19.md`
- `reports/baselines/reference_baseline_paper_alignment_audit.md`
- `reports/baselines/reference_baseline_status.md`
- `reports/baselines/reference_baseline_runs.md`
- `reports/baselines/reference_baseline_results.csv`

## Common Sanity Checks

確認分支與工作樹：

```powershell
git branch --show-current
git status --short
```

列出目前 reference baseline 程式：

```powershell
Get-ChildItem src/reference_baseline_branch
Get-ChildItem src/reference_baseline_branch/configs
```

檢查 config 能被 YAML parser 讀取：

```powershell
python -c "import yaml; yaml.safe_load(open('src/reference_baseline_branch/configs/reference_baselines.yaml', encoding='utf-8')); print('yaml ok')"
```

檢查目前核心 Python 檔案語法：

```powershell
python -m py_compile src/reference_baseline_branch/build_c1_character_features.py src/reference_baseline_branch/build_gpt2_text_embeddings.py src/reference_baseline_branch/run_c3_skapp_full.py src/reference_baseline_branch/build_c3_rag_features.py src/reference_baseline_branch/sklearn_models.py src/experiment_common/features.py
```

檢查 reportable 結果：

```powershell
python -c "import pandas as pd; df=pd.read_csv('reports/baselines/reference_baseline_results.csv'); print(df.tail(20).to_string(index=False))"
```

注意：

- `.exp/` 是 ignored local output，不要 commit。
- `baseline_refer/` 是外部原始碼參考，不要 commit。
- `data/fetch_log.csv` 是 local image fetch log，除非使用者明確要求，否則不要 commit。
- PowerShell profile 的 `Microsoft.WinGet.CommandNotFound` warning 是環境雜訊，不代表指令失敗。

## Important Artifact Paths

Local artifacts:

```text
.exp/baseline/results/
.exp/baseline/text_features/gpt2/
.exp/baseline/c1_character_features/
.exp/baseline/c1_character_images/
.exp/baseline/rag_features/
.exp/baseline/skapp_full/dataset/
```

Tracked reports:

```text
reports/baselines/reference_baseline_results.csv
reports/baselines/reference_baseline_runs.md
reports/baselines/reference_baseline_status.md
reports/baselines/reference_baseline_paper_alignment_audit.md
reports/baselines/reference_baseline_handoff_2026-05-19.md
reports/baselines/reference_baseline_reconstruction_taskboard_2026-05-19.md
reports/baselines/reference_baseline_reproduction_commands_2026-05-19.md
```

External source reference:

```text
baseline_refer/skapp-main/
```

## C1 Reconstruction Runs

C1 目前已有 project-input structure-complete reconstruction。主線 row 是：

```text
C1-Armenta-ProjectInputReconstruction
```

查 C1 目前所有紀錄：

```powershell
python -c "import pandas as pd; df=pd.read_csv('reports/baselines/reference_baseline_results.csv'); print(df[df['baseline_id'].str.startswith('C1', na=False)].to_string(index=False))"
```

查 C1 文件狀態：

```powershell
Select-String -Path reports/baselines/reference_baseline_status.md -Pattern "C1"
Select-String -Path reports/baselines/reference_baseline_paper_alignment_audit.md -Pattern "C1"
```

重建 GPT-2 synopsis artifact：

```powershell
python -m src.reference_baseline_branch.build_gpt2_text_embeddings --splits train val test --batch-size 16 --device auto --local-files-only
```

重跑 C1 reconstruction：

```powershell
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-ProjectInputReconstruction --include-disabled
```

建立 C1 Figure 2 character artifact：

```powershell
python -m src.reference_baseline_branch.build_c1_character_features --splits train val test --batch-size 32 --portrait-batch-size 32 --download-workers 24 --max-characters 5 --device auto --local-files-only
```

重跑 C1 Figure 2 side reconstruction：

```powershell
python -m src.reference_baseline_branch.run_reference_baselines --baseline C1-Armenta-Figure2Reconstruction --include-disabled
```

目前 artifact/run：

```text
.exp/baseline/text_features/gpt2/gpt2_text_embeddings_{train,val,test}.parquet
.exp/baseline/c1_character_features/c1_character_features_{train,val,test}.parquet
.exp/baseline/results/36
.exp/baseline/results/38
```

不要把 `C1-Armenta-Figure2Reconstruction` 當主線 final；它和目前專案
cover/banner 輸入契約不完全對標，只能作 character-centric side analysis。

## C2 Reconstruction Runs

C2 目前已有 project-input CTNN structure-complete reconstruction。主線 row 是：

```text
C2-ProjectInputCTNNReconstruction
```

Dual-visual diagnostic row：

```text
C2-ProjectInputCTNNDualVisualReconstruction
```

查 C2 目前所有紀錄：

```powershell
python -c "import pandas as pd; df=pd.read_csv('reports/baselines/reference_baseline_results.csv'); print(df[df['baseline_id'].str.startswith('C2', na=False)].to_string(index=False))"
```

查 C2 文件狀態：

```powershell
Select-String -Path reports/baselines/reference_baseline_status.md -Pattern "C2"
Select-String -Path reports/baselines/reference_baseline_paper_alignment_audit.md -Pattern "C2"
```

重跑 C2 reconstruction：

```powershell
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCTNNReconstruction --include-disabled
```

重跑 C2 dual-visual diagnostic：

```powershell
python -m src.reference_baseline_branch.run_reference_baselines --baseline C2-ProjectInputCTNNDualVisualReconstruction --include-disabled
```

目前 artifact/run：

```text
.exp/baseline/results/37
.exp/baseline/results/39
```

`C2-CTNN-Lite`、`C2-ProjectInputCrossAttention`、`C2-ProjectInputRecurrentFusion`
只保留為 milestone/proxy rows。若要再逼近原論文，應在
`C2-ProjectInputCTNNReconstruction` 上補更精確的 training diagnostics。
`C2-ProjectInputCTNNDualVisualReconstruction` 已檢查 ResNet50 + ViT-like
dual visual stream，但目前不取代主線。

## C3 RAG Feature Builds

C3 feature build command：

```powershell
python -m src.reference_baseline_branch.build_c3_rag_features --modes none sparse dense hybrid selective skapp_proxy skapp_graph_proxy --top-k 10
```

只重建 graph proxy：

```powershell
python -m src.reference_baseline_branch.build_c3_rag_features --modes skapp_graph_proxy --top-k 10
```

檢查 C3 feature artifact：

```powershell
Get-ChildItem .exp/baseline/rag_features
```

檢查 `skapp_graph_proxy` 產物欄位：

```powershell
python -c "import pandas as pd; p='.exp/baseline/rag_features/skapp_graph_proxy_top10_train.parquet'; df=pd.read_parquet(p); print(df.shape); print([c for c in df.columns if c.startswith('retrieved_') or c.startswith('graph_')][:50])"
```

## C3 Existing Best-Performance Rows

查 C3 所有結果：

```powershell
python -c "import pandas as pd; df=pd.read_csv('reports/baselines/reference_baseline_results.csv'); print(df[df['baseline_id'].str.startswith('C3', na=False)].to_string(index=False))"
```

查目前 C3 selective / SKAPP 相關結果：

```powershell
python -c "import pandas as pd; df=pd.read_csv('reports/baselines/reference_baseline_results.csv'); m=df['baseline_id'].str.contains('SKAPP|Selective|RAG', case=False, na=False); print(df[m].to_string(index=False))"
```

目前性能上仍最強的 C3 row 是：

```text
C3-RAG-Selective-XGB
```

但它不是 SKAPP 完整構造，只能作為 strong selective retrieval baseline。

## C3 SKAPPFull Smoke Run

用途：確認 pipeline 能完整跑完，不作為正式結果。

```powershell
python -m src.reference_baseline_branch.run_c3_skapp_full --target popularity --top-k 10 --d-model 64 --batch-size 256 --max-epochs 2 --patience 1 --device auto --run-id c3_skapp_full_smoke
```

預期輸出：

```text
.exp/baseline/results/c3_skapp_full_smoke/
.exp/baseline/skapp_full/dataset/
```

檢查 smoke 結果：

```powershell
Get-ChildItem .exp/baseline/results/c3_skapp_full_smoke
```

## C3 SKAPPFull Formal Run

目前已跑過的 first structure-complete run：

```powershell
python -m src.reference_baseline_branch.run_c3_skapp_full --top-k 10 --d-model 128 --batch-size 256 --max-epochs 20 --patience 5 --device auto --run-id 35
```

預期輸出：

```text
.exp/baseline/results/35/
.exp/baseline/results/35/c3_skapp_full_metrics.csv
.exp/baseline/results/35/rrcp_silver_summary.json
```

查 run 35 metrics：

```powershell
python -c "import pandas as pd; p='.exp/baseline/results/35/c3_skapp_full_metrics.csv'; df=pd.read_csv(p); print(df.to_string(index=False))"
```

查 RRCP silver summary：

```powershell
python -c "import json; p='.exp/baseline/results/35/rrcp_silver_summary.json'; print(json.dumps(json.load(open(p, encoding='utf-8')), ensure_ascii=False, indent=2))"
```

目前 run 35 症狀：

```text
popularity test_MAE ~= 14668.1228
popularity test_R2 ~= -0.4927
popularity test_log_MAE ~= 1.2983
meanScore test_MAE ~= 9.8063
meanScore test_R2 ~= -0.2385
```

這代表 C3 SKAPPFull 的構造已經能跑完，但尚未穩定或調好。下一步不是刪掉它，
而是 debug dataset tensor、label scaling、retrieval mask、CXMI/RRCP objective 與
SKAPP 原始碼差距。

## C3 SKAPPFull Dataset Debug Commands

檢查 dataset npz keys / shapes：

```powershell
python -c "import numpy as np; p='.exp/baseline/skapp_full/dataset/train.npz'; d=np.load(p, allow_pickle=True); print(d.files); [print(k, d[k].shape, d[k].dtype) for k in d.files]"
```

檢查 retrieved mask 覆蓋：

```powershell
python -c "import numpy as np; d=np.load('.exp/baseline/skapp_full/dataset/train.npz', allow_pickle=True); m=d['retrieved_mask']; print(m.shape, m.mean(), m.sum(axis=1).min(), m.sum(axis=1).max())"
```

檢查 query / retrieved embeddings 是否大量為零：

```powershell
python -c "import numpy as np; d=np.load('.exp/baseline/skapp_full/dataset/train.npz', allow_pickle=True); keys=['query_text','query_image','retrieved_text','retrieved_image']; [print(k, d[k].shape, float(np.mean(np.abs(d[k])>0))) for k in keys]"
```

檢查 label scale：

```powershell
python -c "import numpy as np; d=np.load('.exp/baseline/skapp_full/dataset/train.npz', allow_pickle=True); [print(k, float(d[k].min()), float(d[k].mean()), float(d[k].max())) for k in d.files if 'label' in k or 'target' in k]"
```

## Compare Against SKAPP Source

SKAPP source root：

```text
baseline_refer/skapp-main/
```

列出 source：

```powershell
Get-ChildItem baseline_refer/skapp-main -Recurse -File | Select-Object -First 80 FullName
```

搜尋 RRCP / CXMI / dissembled：

```powershell
rg -n "RRCP|CXMI|dissembl|retriev|without|with" baseline_refer/skapp-main
```

搜尋模型類別：

```powershell
rg -n "class .*Model|nn\\.Module|Transformer|Attention|GRU|LSTM" baseline_refer/skapp-main
```

對照本專案 C3 full implementation：

```powershell
rg -n "RRCP|CXMI|dissembled|retrieved|mask|without|with" src/reference_baseline_branch/run_c3_skapp_full.py
```

## Report Update Pattern

每次正式跑完，至少更新：

```text
reports/baselines/reference_baseline_results.csv
reports/baselines/reference_baseline_runs.md
reports/baselines/reference_baseline_status.md
reports/baselines/reference_baseline_paper_alignment_audit.md
```

建議紀錄格式：

```text
1. baseline_id
2. route / paper
3. run_id
4. command
5. input contract
6. paper-aligned parts
7. remaining deviations
8. metrics
9. whether this is final, proxy, smoke, or diagnostic
```

`reports/baselines/reference_baseline_results.csv` 必須保存可比較 metrics。`.exp/` 裡的 raw
artifact 可以保留在本機，但不應作為唯一紀錄。

## Diff And Commit Preparation

只檢查 baseline 相關差異：

```powershell
git diff -- .gitignore reports/baselines/reference_baseline_paper_alignment_audit.md reports/baselines/reference_baseline_results.csv reports/baselines/reference_baseline_runs.md reports/baselines/reference_baseline_status.md reports/baselines/reference_baseline_handoff_2026-05-19.md reports/baselines/reference_baseline_reconstruction_taskboard_2026-05-19.md reports/baselines/reference_baseline_reproduction_commands_2026-05-19.md src/experiment_common/features.py src/reference_baseline_branch/build_c1_character_features.py src/reference_baseline_branch/build_gpt2_text_embeddings.py src/reference_baseline_branch/build_c3_rag_features.py src/reference_baseline_branch/configs/reference_baselines.yaml src/reference_baseline_branch/sklearn_models.py src/reference_baseline_branch/run_c3_skapp_full.py
```

空白錯誤檢查：

```powershell
git diff --check -- .gitignore reports/baselines/reference_baseline_paper_alignment_audit.md reports/baselines/reference_baseline_results.csv reports/baselines/reference_baseline_runs.md reports/baselines/reference_baseline_status.md reports/baselines/reference_baseline_handoff_2026-05-19.md reports/baselines/reference_baseline_reconstruction_taskboard_2026-05-19.md reports/baselines/reference_baseline_reproduction_commands_2026-05-19.md src/experiment_common/features.py src/reference_baseline_branch/build_c1_character_features.py src/reference_baseline_branch/build_gpt2_text_embeddings.py src/reference_baseline_branch/build_c3_rag_features.py src/reference_baseline_branch/configs/reference_baselines.yaml src/reference_baseline_branch/sklearn_models.py src/reference_baseline_branch/run_c3_skapp_full.py
```

若使用者要求 commit，建議分批：

```text
1. feat(reference-baselines): add SKAPP-style full C3 reconstruction pipeline
2. feat(reference-baselines): add C1 Figure 2 side reconstruction artifacts
3. feat(reference-baselines): add graph-aware SKAPP proxy features
4. docs(reference-baselines): add reproduction handoff and taskboard
5. docs(reference-baselines): record C1/C2/C3 reconstruction diagnostics
```

Commit body 請用英文 Conventional Commits，並用 numbered list 說明內容。
