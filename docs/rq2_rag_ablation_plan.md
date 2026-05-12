# RQ2 RAG 對比實驗設計

本文件對齊 RQ2：在相同 fusion backbone 下，加入 RAG 特徵是否對 `popularity`、`meanScore` 雙目標回歸帶來改善，以及哪種 retrieval 策略最有效。

## 參考論文對齊

參考文獻：`docs/refer/Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation.pdf`

該文的 SKAPP 核心觀念可拆成三層：

| 論文設計 | 對本專案的可落地版本 |
|---|---|
| non-retrieval baseline | `none`：產生 schema-compatible no-RAG features |
| vanilla semantic retrieval | `dense`：使用 text embedding 做語意檢索 |
| meta retriever | `sparse`：使用 genre / studio / voice_actor / source；`hybrid`：sparse + dense RRF |
| selective refiner / RRCP | 本輪先不做，列為 RQ2 延伸或 RQ3 robustness |

目前先做 RQ2 的最小完整對比，不直接複刻 VL-GNN 或 RRCP-Attention，避免讓 retrieval 策略和 backbone 架構同時改變。

更完整的 C3/SKAPP「可主張範圍」已記錄於：

```text
reports/reference_baseline_paper_alignment_audit.md
```

本文件中的 `none/sparse/dense/hybrid` 只回答「檢索特徵是否帶來增益」，不是 SKAPP 復現。若要把 C3 往更接近 SKAPP 推進，至少還需要：

1. top-k 檢索集合聚合，而不是只把 top-1 結果壓成幾個 RAG 欄位。
2. RRCP-like contribution scoring/filtering，用來排除對 query 預測有害或無貢獻的檢索項目。
3. 檢索集合的 graph/attention fusion module，而不是只把 RAG 摘要欄位接到現有 FusionMLP。

## 實驗組

固定資料切分、text/image embedding、FusionMLP 架構、training hyperparameters，只改 RAG feature 來源。

| mode | 說明 | 輸出位置 |
|---|---|---|
| `none` | 無檢索；RAG 數值欄位填 train fallback，multi-hot 為空，`rag_found=False` | `src/fussion_branch/RAG/return/none/` |
| `sparse` | metadata sparse retrieval | `src/fussion_branch/RAG/return/sparse/` |
| `dense` | text embedding semantic retrieval | `src/fussion_branch/RAG/return/dense/` |
| `hybrid` | sparse + dense RRF fusion | `src/fussion_branch/RAG/return/hybrid/` |

## 執行方式

先確認 Qdrant 已啟動，且 text embeddings 已產生：

```bash
docker start qdrant
python -m src.fussion_branch.run_text_embedding
```

產生四組 RAG features：

```bash
python -m src.fussion_branch.run_rag_ablation
```

若 Qdrant collection 已建立，只想重跑 query：

```bash
python -m src.fussion_branch.run_rag_ablation --skip-build
```

訓練 RQ2 對比：

```bash
python -m src.fussion_branch.run_rq2_rag_experiments --modes none sparse dense hybrid
```

若要訓練後直接做 test evaluation：

```bash
python -m src.fussion_branch.run_rq2_rag_experiments --modes none sparse dense hybrid --evaluate-test
```

## 結果彙整

每個 mode 會使用獨立 run_id 與獨立 MetaEncoder：

```text
.exp/fussion/results/{run_id}/{target}/
.exp/fussion/meta_encoder_{dataset}_{run_id}_{mode}.json
```

總表輸出：

```text
.exp/fussion/experiments_summary.csv
```

新增欄位：

| 欄位 | 說明 |
|---|---|
| `rag_mode` | `none` / `sparse` / `dense` / `hybrid` |
| `rag_features_dir` | 該 run 使用的 RAG feature 目錄 |

## 報告主表建議

以 target 分兩張表呈現：

| target | mode | test Spearman rho | test MAE | test R2 | test log_MAE |
|---|---|---:|---:|---:|---:|
| popularity | none | | | | |
| popularity | sparse | | | | |
| popularity | dense | | | | |
| popularity | hybrid | | | | |

`meanScore` 沒有 `log_MAE`，保留 `Spearman rho`、`MAE`、`R2` 即可。

主要判讀：
- `sparse/dense/hybrid` 相對 `none` 的增益回答「RAG 是否有效」。
- `sparse` vs `dense` 回答 metadata retrieval 與 semantic retrieval 的差異。
- `hybrid` 是否最佳，用來檢查多訊號 retrieval 是否比單一 retrieval 穩定。

## 目前 first-pass reference 結果

2026-05-12 已在 `src/reference_baseline_branch` 先完成不依賴 Qdrant/Docker 的 C3 reference baseline：

```bash
python -m src.reference_baseline_branch.build_c3_rag_features --modes none sparse dense hybrid --top-k 10
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-None-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Sparse-XGB --include-disabled
python -m src.reference_baseline_branch.run_reference_baselines --baseline C3-RAG-Dense-XGB --include-disabled
```

已完成 `none/sparse/dense`。`hybrid` 目前 train/val artifacts 已產生，但 test artifact 生成 timeout，暫不列入正式表。

| target | mode | test Spearman rho | test MAE | test R2 | test log_MAE |
|---|---|---:|---:|---:|---:|
| popularity | none | 0.8583 | 9664.2004 | 0.5064 | 0.9013 |
| popularity | sparse | 0.8722 | 9736.1037 | 0.5725 | 0.9429 |
| popularity | dense | 0.8584 | 9704.8621 | 0.5084 | 0.9016 |
| meanScore | none | 0.5307 | 8.3647 | 0.0132 | |
| meanScore | sparse | 0.5384 | 8.1703 | 0.0730 | |
| meanScore | dense | 0.5382 | 8.2445 | 0.0464 | |

目前判讀：

- `sparse` metadata retrieval 是 first-pass 最強 C3 設定。
- `dense` semantic retrieval 對 popularity 幾乎接近 no-RAG，但對 meanScore 有小幅增益。
- 這些結果仍是 SKAPP-inspired retrieval baselines，不是 SKAPP reproduction。
