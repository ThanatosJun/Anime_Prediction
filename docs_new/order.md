# v2 建置順序

架構參考：`docs_new/` 下的各說明文件

---

## ✅ 已完成

### Step 0：Component 建置

| 元件 | 路徑 | 說明 | 狀態 |
|------|------|------|------|
| Text Encoder | `src_2/component_text/` | e5-base-v2（768-dim），TextEmbedder + CLI | ✅ |
| Image Encoder | `src_2/component_image/` | Swin-B fine-tuned（1024-dim），ImageEmbedder + YOLO（both: face+person） | ✅ |
| RAG 基礎建設 | `src_2/RAG/` | sparse_encoder, rag_builder, rag_query, run_build_embeddings, rag_config | ✅ |

### Step 1：RAG Embeddings 生成

```bash
python src_2/RAG/run_build_embeddings.py
```

| 輸出 | 說明 | 狀態 |
|------|------|------|
| `src_2/embedding/text/text_embeddings_{split}.parquet` | e5-base-v2 768-dim | ✅ train / val / test / holdout |
| `src_2/embedding/image_rag/image_embeddings_train.parquet` | Swin-B 1024-dim（no YOLO，RAG 用） | ✅ train only |

### Step 2：YOLO Crop 圖片

```bash
python src_2/component_image/run_yolo_crop.py --splits train val test holdout_unknown
```

| 輸出 | 狀態 |
|------|------|
| `src_2/data/image/crops/{split}/{id}_crop_{n}.jpg` | ✅ train / val / test / holdout |

### Step 3：Fusion Image Embeddings（yolo + cover + banner）

```bash
python src_2/component_image/run_swin_embedding.py --splits train val test holdout_unknown
```

| 輸出 | 說明 | 狀態 |
|------|------|------|
| `src_2/embedding/image/image_embeddings_{split}.parquet` | id, yolo_0…1023, cover_0…1023, banner_0…1023, has_* | ✅ train / val / test / holdout |

### Step 4：建立 Qdrant Collection

```bash
bash src_2/RAG/start_qdrant.sh
python src_2/RAG/rag_builder.py
```

| 輸出 | 狀態 |
|------|------|
| `src_2/RAG/qdrant_storage/`（collection: `anime_rag_v2`） | ✅ |
| `src_2/RAG/sparse_encoder.json` | ✅ |

### Step 5：查詢 RAG Features

```bash
python src_2/RAG/rag_query.py --splits train val test holdout_unknown
```

| 輸出 | 說明 | 狀態 |
|------|------|------|
| `src_2/RAG/return/rag_features_{split}.parquet` | id, rag_popularity, rag_score, rag_release_year, rag_episodes, rag_found, retrieved_ids … | ✅ train / val / test / holdout |

### Step 6：Fusion Model 元件

| 元件 | 路徑 | 說明 | 狀態 |
|------|------|------|------|
| MetaEncoder v2 | `src_2/fussion_training/meta_encoder.py` | 56-dim（移除舊版 RAG 10 dims），`fit(meta_df)` / `transform(meta_df)` | ✅ |
| Cross Attention | `src_2/fussion_training/cross_attention.py` | Q[batch,1,128] × KV[batch,15,128] → [batch,128]，4 heads | ✅ |

### Step 7：Fusion Model v2 訓練

架構：

```
Image  [batch, 3, 1024] → ImageProjection (Shared + Gate) → [batch, 128] ─┐
Text   [batch, 768]     → TextProjection                  → [batch, 128] ─┤
Meta   [batch, 56]      → MetaProjection                  → [batch, 128] ─┤─ concat → MLP → [1]
                          MetaProjection → Q [batch,1,128] ─┐              │
RAG retrieved items                                          │              │
  [batch,5,10]  → Rag_meta Proj  ─┐                        │              │
  [batch,5,768] → Rag_text Proj  ─┼─ KV [batch,15,128] → Cross Attention ─┘
  [batch,5,1024]→ Rag_image Proj ─┘
  concat_dim: use_rag=True → 512（128×4）; use_rag=False → 384（128×3）
```

設計決策：
- **Loss**：popularity → HuberLoss（delta=1.0）；meanScore → Log-Cosh Loss
- **Optimizer**：SAM (rho=0.05, pure wrapper) + AdamW；LR scheduler：ReduceLROnPlateau（factor=0.5, patience=3）
- **AMP**：autocast（float16 forward，float32 gradient）；不用 GradScaler（與 SAM 兩步更新不相容）
- **DataLoader**：`num_workers=min(4, cpu_count())`，`persistent_workers=True`
- **train_separately**：true → 同一 script 兩遍循環，各 target 獨立模型
- **Checkpoint**：每 epoch 若 val_loss 改善則儲存 `best_model.pt` + `target_scaler.json`

執行：

```bash
python src_2/train.py                    # 跑全部 targets
python src_2/train.py --target popularity
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/best_model.pt` | 最佳 val loss checkpoint |
| `src_2/runs/{run_id}/{target}/target_scaler.json` | 正規化參數 |
| `src_2/runs/{run_id}/{target}/history.json` | 訓練曲線（含 run_id / notes） |
| `src_2/runs/{run_id}/{target}/final_metrics.json` | train + val 全量 metrics（含 run_id / notes） |

建置清單：
- [x] `src_2/fussion_training/meta_encoder.py`（56-dim）
- [x] `src_2/fussion_training/cross_attention.py`（RAGCrossAttention）
- [x] `src_2/fussion_training/dataset.py`（AnimeDataset，輸出 image[3,1024] + mask）
- [x] `src_2/fussion_training/model.py`（FusionModel v2，image/text/meta dim 從 config 讀取）
- [x] `src_2/fussion_configs.yaml`（image_mode / use_rag / targets / mixed_precision / notes）
- [x] `src_2/train.py`（訓練主程式）
- [x] `src_2/evaluate.py`（評估）

Code Review 修正紀錄（已套用）：

| 檔案 | 問題 | 修正 |
|------|------|------|
| `train.py` | SAM `group["rho"]` KeyError（AdamW param_groups 無此 key） | 改存 `self.rho` |
| `train.py` | SAM 繼承 `torch.optim.Optimizer`，新版 PyTorch `state` property setter 衝突 | 改為 pure wrapper class，`e_w` 存在 `self._e_w` dict |
| `train.py` | SAM `second_step` 的 `@torch.no_grad()` 包住 `base_opt.step()`，新版 PyTorch 報錯 | `second_step` 移除 decorator，手動 `with torch.no_grad()` 包權重還原 |
| `train.py` | `train_separately: false` 未實作但 config 存在 | 加 `NotImplementedError` guard |
| `train.py` | GradScaler + SAM 雙 `unscale_()` → RuntimeError | 移除 GradScaler，只用 `autocast`（gradient 仍 float32） |
| `train.py` | YAML `lr: 1e-3` 被解析為字串 | 改為 `0.001` 明確小數 |
| `dataset.py` | RAG 無命中 `rag_mask` 全 True → MultiheadAttention NaN | 強制 `rag_mask[0] = False` |
| `dataset.py` | `_load_emb_parquet` / `_load_image_emb_stack` 用 `iterrows()` 載入 → 極慢 | 改為向量化 `to_numpy()` + `.copy()` |
| `dataset.py` | `meta_encoder.transform()` 逐 sample 呼叫（N 次）→ 載入卡頓 | 改為一次批次 `transform(meta_df)`，再切分 dict |
| `dataset.py` | `_build_rag_meta_lookup` 重複 parse studios/genres → 慢 | 預先計算 train_lookup 所有欄位，`itertuples()` 取代 `iterrows()`，加 tqdm |
| `dataset.py` | `denormalize_target` 無 clip → 極端預測值 expm1 overflow | 加 `np.clip(y_norm, -5, 5)` |
| `evaluate.py` | `spearmanr().statistic`（scipy < 1.7 不支援） | 改用 `.correlation` |
| `evaluate.py` | metrics：RMSE 不適合 skewed 分佈 | 改為 `log_R2`（popularity）/ `R2`（meanScore）+ `log_MAE`（popularity only） |
| `evaluate.py` | metrics 欄位順序不一致 | 統一順序：spearman_rho → R2/log_R2 → MAE → log_MAE |
| `evaluate.py` | test metrics 存為獨立 `eval_test.json` | 改為 merge 進 `final_metrics.json`，單一檔案含 train/val/test |
| `dataset.py` | AMP float16 output → `denormalize_target` 中 `expm1` overflow → Infinity | 加 `np.asarray(y_norm, dtype=np.float64)` 強制轉型 |
| `model.py` | `ImageProjection` gate 為 soft-sum → 缺失模態時輸出量級縮小，sample 間不一致 | 改為 soft-average：`(proj * gate / gate_sum).sum(dim=1)` |
| `train.py` | `LogCoshLoss` 的 `1e-12` 在 `cosh()` 內部，大 diff 時 `cosh` 仍 overflow | 改為數值穩定公式：`\|x\| + log1p(exp(-2\|x\|)) - log2` |
| `train.py` / `evaluate.py` | `trend_head._active` 未在 `_compute_final_metrics` 與 `evaluate.py` 設定，trend_head 啟用時模型架構不一致，`load_state_dict` 報錯 | 新增 `make_model_config(config, target)` helper（在 `model.py`），統一所有建構 FusionModel 的呼叫點 |
| `train.py` | `_eval_epoch` 回傳 normalized space 的 MAE，`history.json` 的 `val_mae` 難以直覺判讀 | 新增 `target_scaler` 參數，傳入後回傳原始 scale MAE |
| `dataset.py` | train split 時 text embeddings 載入兩次（`text_map` 與 `rag_text_map` 同一份檔案） | `split == "train"` 時 `rag_text_map = self.text_map`，共用 dict |

### Step 8：評估與推論

```bash
# test set 評估（兩個 target 一起）
python src_2/evaluate.py --split test

# holdout 推論（無標籤，只輸出預測）
python src_2/evaluate.py --split holdout_unknown
```

| 輸出 | 說明 |
|------|------|
| `src_2/runs/{run_id}/{target}/final_metrics.json` | train / val / test 完整 metrics 合併於此（spearman_rho / R2 or log_R2 / MAE / log_MAE） |
| `src_2/runs/{run_id}/{target}/pred_{split}.csv` | id, pred, target（原始 scale） |

- [x] `src_2/evaluate.py`（spearman_rho / R2 or log_R2 / MAE / log_MAE，results merge 進 final_metrics.json）
- [x] Test set 評估完成（final_metrics.json 含 train / val / test）
- [x] Holdout 推論（pred_holdout_unknown.csv）

### Step 9：可解釋性分析

```bash
# RAG attention heatmap
python src_2/explain/rag_heatmap.py --target popularity --n 5

# Captum（modality 貢獻）+ SHAP（meta feature 貢獻）
python src_2/explain/feature_attr.py --target popularity --n 20
```

| 輸出 | 說明 |
|------|------|
| `runs/{run_id}/explain/{target}/rag/{id}_attn.png` | Cross Attention heatmap：x = retrieved anime，y = meta/text/image modality |
| `runs/{run_id}/explain/{target}/feature/captum_modality.csv` | 各 sample 的 modality 歸一化重要性 |
| `runs/{run_id}/explain/{target}/feature/captum_modality.png` | 平均 modality 重要性長條圖 |
| `runs/{run_id}/explain/{target}/feature/shap_values.npy` | raw SHAP values [n, 56] |
| `runs/{run_id}/explain/{target}/feature/shap_summary.png` | top-k meta feature 重要性（release_year, genre_Action 等） |

建置清單：
- [x] `src_2/fussion_training/cross_attention.py`（`return_attn=True` 回傳 `[batch, top_k, 3]` weights）
- [x] `src_2/fussion_training/model.py`（`forward(batch, return_attn=True)`）
- [x] `src_2/explain/rag_heatmap.py`（RAG attention heatmap）
- [x] `src_2/explain/feature_attr.py`（Captum IG + SHAP DeepExplainer）
- [x] `src_2/requirements.txt`

前置安裝（Step 9）：
```bash
pip install captum shap
```

---

## ⏳ 待完成

### Step 10：整個 Pipeline 最佳效果

從 hp_search（Run04~09）找出最佳超參數組合，以 test set 驗證最終效果。

| Target | 最佳 run（val） | 最佳設定 |
|--------|----------------|---------|
| popularity | hp07 | dropout=0.3, wd=1e-3, batch=512 |
| meanScore | hp08 | dropout=0.5, wd=5e-4, batch=512 |

- [ ] 對 hp07（popularity）執行 `python src_2/evaluate.py --split test --target popularity --config <hp07_config>`
- [ ] 對 hp08（meanScore）執行 `python src_2/evaluate.py --split test --target meanScore --config <hp08_config>`
- [ ] 更新 `src_2/README.md` 實驗記錄（test metrics）

---

### ✅ Step 11：RAG 影響消融實驗

控制變因：只改 `use_rag`，其餘超參數固定（= Run07：dropout=0.3, wd=1e-3, batch=512, TrendHead on）。

```bash
python src_2/ablation.py   # 一次跑完 Step 11 + 12（run_id: abl_rag_off / abl_img_cover / abl_img_cover_banner）
```

- [x] `abl_rag_off`（use_rag=false，純 MLP）
- [x] 與 full model（Run07）對比，test set

| 比較（test） | popularity log_MAE | popularity spearman | meanScore MAE | meanScore spearman |
|------|-----|-----|-----|-----|
| full（RAG on） | **0.8904** | **0.8498** | 8.1776 | **0.5397** |
| abl_rag_off | 0.9279 | 0.8385 | **7.6716** | 0.5093 |
| Δ（RAG 貢獻） | **-0.0375 ✅** | **+0.0113 ✅** | +0.506 ❌ | **+0.0304 ✅** |

> **結論**：RAG 對 **popularity 明確有益**（log_MAE -0.038，spearman +0.011）。對 meanScore：spearman 上升（排名更準）但 MAE 反而變差——RAG 帶入更多近鄰資訊放大了 distribution shift 下的數值偏移，與既有觀察一致。

---

### ✅ Step 12：模態消融實驗（Cross Attention & Image）

**12a. Cross Attention（RAG）消融**：見 Step 11。

**12b. Image 模態消融**（test set，與 full model = Run07 對照）：

| 設定（test） | pop log_MAE | pop facc_2x | pop spearman | score MAE | score spearman |
|------|-----|-----|-----|-----|-----|
| cover only | 1.1041 | 0.3965 | 0.8405 | 8.3641 | 0.5130 |
| cover + banner | 0.8922 | 0.4836 | 0.8500 | 7.5995 | 0.5377 |
| cover + banner + yolo（full / Run07） | **0.8904** | **0.4856** | 0.8498 | 8.1776 | 0.5397 |

> **結論**：
> - **banner 貢獻最大**：popularity log_MAE 從 cover only 的 1.1041 → cover+banner 的 0.8922（-0.21），是最關鍵的視覺模態。
> - **YOLO crop 邊際貢獻幾乎為零**：cover+banner（0.8922）≈ full（0.8904），加入 yolo 對 popularity 改善 <0.002。meanScore 甚至略退（7.5995→8.1776）。
> - **建議**：若要精簡 pipeline，可考慮移除 YOLO crop（省去 YOLO 偵測 + 額外 Swin embedding 計算），對 popularity 幾乎無損。

**12c. Multimodal 分支消融（重訓版，架構真的移除分支）**：

```bash
python src_2/ablation_multimodal.py
```

實作：`model.py` 新增 `modalities: {image, text, meta}` flag（向後相容，未指定時全開）。single-modality 組關閉 RAG + TrendHead，回答「單一模態各自上限」；`abl_no_image` 保留 RAG + TrendHead，對照 Run07。

popularity（test，log_MAE 越低越好）：

| 設定 | log_MAE | facc_2x | spearman |
|------|---------|---------|----------|
| full（img+txt+meta+rag, Run07） | **0.8904** | **0.4856** | **0.8498** |
| only_meta | 0.9507 | 0.4561 | 0.8322 |
| no_image（txt+meta+rag） | 1.0520 | 0.4270 | 0.8287 |
| only_text | 1.2634 | 0.3602 | 0.7056 |
| only_image | 1.3458 | 0.3337 | 0.7230 |

meanScore（test，MAE 越低越好）：

| 設定 | MAE | spearman |
|------|-----|----------|
| only_meta | **8.1420** | 0.5065 |
| full（Run07） | 8.1776 | **0.5397** |
| only_image | 8.4261 | 0.3948 |
| no_image | 8.4805 | 0.4998 |
| only_text | 10.2581 | 0.2164 |

> **結論**：
> - **meta 是最強的單一模態**：only_meta 在兩個 target 都是單模態最佳（pop log_MAE 0.9507、score MAE 8.142），甚至 meanScore MAE 略勝 full model。metadata（前作、studio/VA TE、format 等）攜帶最多預測訊號。
> - **text / image 單獨都很弱**：only_text（pop log_MAE 1.26）、only_image（1.35）遠差於 only_meta，說明視覺與描述需要與 metadata 結合才有效。
> - **多模態互補性確立**：full model（0.8904）明顯優於任何單模態，證明四個分支的資訊互補，融合架構有實質價值。
> - meanScore 的 spearman 仍以 full 最高（0.5397），排名能力靠多模態；但 MAE 受 distribution shift 主導，only_meta 數值誤差略低。

---

### ✅ Step 13：可解釋性分析（SHAP / Captum / Attention Heatmap）

對最佳 run（Run07）執行。執行時用 run_id=07 的 config。

```bash
pip install captum shap

# RAG Cross Attention heatmap（x=retrieved anime, y=modality）
python src_2/explain/rag_heatmap.py --config <run07.yaml> --target popularity --n 5
python src_2/explain/rag_heatmap.py --config <run07.yaml> --target meanScore  --n 5

# Captum（modality 貢獻）+ SHAP（meta feature 貢獻）
python src_2/explain/feature_attr.py --config <run07.yaml> --target popularity --n 20 --background 50
python src_2/explain/feature_attr.py --config <run07.yaml> --target meanScore  --n 20 --background 50
```

修正紀錄（執行時發現）：

| 檔案 | 問題 | 修正 |
|------|------|------|
| `rag_heatmap.py` / `feature_attr.py` | `FusionModel(config)` 未套 `make_model_config`，TrendHead checkpoint 載入失敗 | 改為 `FusionModel(make_model_config(config, target))` |
| `feature_attr.py` | Captum IG 把 batch 擴成 n_steps，但固定的 image/rag mask 仍 batch=1 → shape 不符 | `FusionWrapper.forward` 將 mask `expand` 對齊動態 batch |
| `feature_attr.py` | SHAP `MetaOnlyWrapper` 固定的 image/rag tensor batch=1，背景樣本 batch=N → MultiheadAttention shape 錯 | forward 內 `expand` 所有固定 tensor 對齊 |
| `feature_attr.py` | SHAP `DeepExplainer`（DeepLIFT）對 attention/LayerNorm additivity 檢查失敗（max diff ~1.0） | 改用 `GradientExplainer`（expected gradients） |
| `rag_heatmap.py` | id→title lookup 只建在 train_df，query 動畫（val/test split）標題顯示為 ID 數字 | 同時建在 `ds.meta_df`（romaji 優先、缺則 english），query 名稱正確顯示；retrieved 截斷 12→18、query 20→45 |

**Captum 模態重要性（normalized |IG|，越大越重要）：**

| 模態 | popularity | meanScore |
|------|-----------|-----------|
| **rag_image** | **0.407** | **0.375** |
| image_yolo | 0.206 | 0.263 |
| image_banner | 0.153 | 0.144 |
| image_cover | 0.089 | 0.095 |
| meta | 0.066 | 0.076 |
| rag_meta | 0.047 | 0.031 |
| text | 0.027 | 0.012 |
| rag_text | 0.004 | 0.004 |

> RAG 的 **image 模態貢獻最大**（兩個 target 皆 ~0.4），印證 Cross Attention 主要透過「相似動畫的視覺特徵」傳遞資訊；rag_text 貢獻極小。視覺資訊（rag_image + 三路 image = ~0.85）整體主導預測。

**SHAP top meta features：**

| 排名 | popularity | meanScore |
|------|-----------|-----------|
| 1 | prequel_meanScore_mean | va_te_score（聲優評分 TE） |
| 2 | va_te_pop（聲優人氣 TE） | prequel_meanScore_mean |
| 3 | va_te_score | studio_te_score（工作室評分 TE） |
| 4 | prequel_popularity_mean | va_te_pop |
| 5 | studio_te_score | studio_te_pop |

> **前作（prequel）與 target encoding（聲優 / 工作室）是兩個 target 最重要的 meta 特徵**。popularity 偏重前作人氣與聲優人氣；meanScore 偏重聲優 / 工作室的歷史評分。符合直覺：續作承襲前作熱度，評分則與製作團隊水準高度相關。

- [x] 安裝依賴：`pip install captum shap`
- [x] rag_heatmap.py（popularity + meanScore）
- [x] feature_attr.py（popularity + meanScore）
- [x] 分析：rag_image 模態貢獻最大；prequel + TE 特徵最重要

---

### Step 14：推論 Pipeline

給定一部新動畫（封面圖 + metadata + 描述），即時走完完整推論流程：

- [ ] YOLO crop（人物／臉部）
- [ ] Swin-B embedding（cover + banner + yolo）
- [ ] e5-base-v2 text embedding
- [ ] RAG query（Qdrant）
- [ ] FusionModel inference → popularity / meanScore 預測

---

## 資料流總覽

```
src_2/data/dataset/fusion_meta_clean_{split}_v2.csv
    │
    ├─ description ──→ TextEmbedder（e5-base-v2）──→ embedding/text/
    │
    ├─ coverImage ──→ YOLO crop ──→ Swin-B ──→ embedding/image/（yolo/cover/banner）
    │
    ├─ coverImage ──→ Swin-B（no YOLO）──→ embedding/image_rag/（RAG knowledge base）
    │
    ├─ metadata（56 cols）──→ MetaEncoder v2 ──→ [batch, 56]
    │
    └─ RAG query ──→ Qdrant ──→ retrieved_ids ──→ RAG/return/rag_features_{split}.parquet
```
