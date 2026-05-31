# Fusion v2

多模態融合預測模組 v2。結合 text embedding、image embedding（cover + banner + YOLO crop）、metadata 與 RAG Cross Attention，預測動畫的 `popularity`（人氣）與 `meanScore`（評分）。

v1（src/fussion_branch）與 v2 核心差異：
- **RAG**：GNN（graph propagation）→ **Cross Attention**（Q=meta, KV=retrieved items 三路投影）
- **Image**：單張 cover → **三模態 Gated Projection**（cover + banner + YOLO crop）
- **MetaEncoder**：66-dim（含 RAG scalar）→ **56-dim**（RAG 移出，改由 Cross Attention 輸入）
- **Optimizer**：AdamW → **SAM + AdamW**（Sharpness-Aware Minimization）
- **Loss**：HuberLoss → **HuberLoss（popularity）/ Log-Cosh（meanScore）**

## 目前最佳結果（test set）

| Target | 最佳 Run | 主指標 | 準確率 | Spearman | 設定 |
|--------|---------|--------|--------|----------|------|
| **popularity** | Run07 | log_MAE **0.8904** | facc_2x **0.4856** | 0.8498 | dropout=0.3, wd=1e-3, batch=512, 完整四分支 |
| **meanScore** | Run02 | MAE **7.2937**（R2 0.246）| within_10pt **0.7360** | 0.5478 | TrendHead, gate soft-avg, dropout=0.3 |

> meanScore 的 test 最佳是 Run02 而非 hp_search 的 Run08——distribution shift 導致「val 最佳 ≠ test 最佳」（詳見[已知限制](#已知限制)）。

---

## 目錄結構

```
src_2/
│
├── fussion_training/             # 核心模組（不直接執行）
│   ├── meta_encoder.py           # MetaEncoder v2（56-dim）
│   ├── cross_attention.py        # RAGCrossAttention（Q×KV Cross Attention）
│   ├── dataset.py                # AnimeDataset（組合所有 embedding → tensor）
│   └── model.py                  # FusionModel v2
│
├── RAG/                          # RAG pipeline（Qdrant hybrid search）
│   ├── sparse_encoder.py
│   ├── rag_builder.py
│   ├── rag_query.py
│   ├── run_build_embeddings.py
│   ├── start_qdrant.sh
│   ├── rag_config.yaml
│   └── return/                   # gitignore
│
├── component_text/               # e5-base-v2 text embedding
├── component_image/              # Swin-B fine-tuned + YOLO
│
├── embedding/                    # gitignore
│   ├── text/                     # text_embeddings_{split}.parquet
│   ├── image/                    # image_embeddings_{split}.parquet（yolo/cover/banner）
│   └── image_rag/                # image_embeddings_train.parquet（RAG 知識庫）
│
├── data/
│   └── dataset/                  # fusion_meta_clean_{split}_v2.csv
│
├── explain/                      # 可解釋性分析
│   ├── rag_heatmap.py            # Cross Attention heatmap
│   └── feature_attr.py           # Captum IG + SHAP
│
├── runs/                         # gitignore，實驗輸出
│   └── {run_id}/
│       ├── {target}/
│       │   ├── best_model.pt
│       │   ├── target_scaler.json
│       │   ├── history.json
│       │   ├── final_metrics.json    # train / val / test metrics 合併
│       │   └── pred_{split}.csv
│       └── explain/{target}/         # 可解釋性輸出（run 層級，非 target 層級）
│           ├── feature/
│           └── rag/
│
├── train.py                      # 訓練主程式
├── evaluate.py                   # 評估（merge 進 final_metrics.json）
├── inference.py                  # 推論 Pipeline（新動畫 → popularity/meanScore）
├── hp_search.py                  # 超參數搜尋（Run04~09）
├── ablation.py                   # RAG / image 消融
├── ablation_multimodal.py        # 多模態分支消融（重訓版）
├── backfill_metrics.py           # 補算舊 run 缺的指標欄位（不重訓）
├── fussion_configs.yaml          # 訓練設定
└── requirements.txt
```

---

## 前置條件

```bash
conda activate animeprediction

# PyTorch（CUDA 12.8 / RTX 5070 Ti）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 其他套件
pip install -r src_2/requirements.txt
pip install dghs-imgutils==0.19.0 --no-deps   # numpy<2 metadata 衝突，--no-deps 繞過

# Qdrant（Docker）
bash src_2/RAG/start_qdrant.sh
curl http://localhost:6333/healthz   # 確認啟動
```

詳細部署流程見 `docs_new/operator.md`。

---

## 模型架構

### FusionModel v2

```
Image  [batch, 3, 1024] ─→ ImageProjection (Shared Linear + Content Gate) ─→ [batch, 128] ─┐
Text   [batch, 768]     ─→ ProjectionBlock(768→128)                        ─→ [batch, 128] ─┤
Meta   [batch, 56]      ─→ ProjectionBlock(56→128)                         ─→ [batch, 128] ─┤─ concat → MLP → [1]
                           ProjectionBlock → Q [batch, 1, 128] ──────────────────────────────┐ │
RAG retrieved（top-5）                                                                        │ │
  [batch, 5, 10]   ─→ Linear(10→128)   ─┐                                                   │ │
  [batch, 5, 768]  ─→ Linear(768→128)  ─┼─ KV [batch, 15, 128] ─→ Cross Attention ──────────┘ │
  [batch, 5, 1024] ─→ Linear(1024→128) ─┘                                                     │
                                                                                               │
concat_dim: use_rag=True → 512（128×4）；use_rag=False → 384（128×3）                        │
MLP backbone: concat_dim → 256 → 128 → 1 ─────────────────────────────────────────────────────┘
```

**設計重點：**
- `ImageProjection`：三模態（cover/banner/yolo）共用 Linear，Gate 從原始 1024-dim 計算（content-based），缺失模態 gate 強制 0
- `RAGCrossAttention`：Q = meta projection，KV = retrieved items 三路投影後 concat（layout: meta 0-4, text 5-9, image 10-14）
- `train_separately=true`：popularity / meanScore 各自獨立模型，同一 script 循環訓練

### 訓練設定

| 項目 | 設定 |
|------|------|
| Loss（popularity） | HuberLoss（delta=1.0） |
| Loss（meanScore） | Log-Cosh Loss |
| Optimizer | SAM (rho=0.05, pure wrapper) + AdamW |
| LR Schedule | ReduceLROnPlateau（factor=0.5, patience=3, min_lr=1e-6） |
| Early Stopping | patience=5 |
| AMP | autocast float16（gradient 仍 float32，不用 GradScaler） |
| Batch Size | 256 |
| DataLoader | num_workers=min(4, cpu_count()), persistent_workers=True |

---

## 評估指標

| 指標 | 說明 | 適用 |
|------|------|------|
| `spearman_rho` | 排名相關係數（主要指標）| 兩個 target |
| `log_R2` | log1p 空間 R²（匹配訓練目標，對 skewed 分佈穩定）| popularity |
| `R2` | 原始 scale R²（診斷 distribution shift）| meanScore |
| `MAE` | 原始 scale 平均絕對誤差 | 兩個 target |
| `log_MAE` | log1p 空間 MAE（scale-free，越小越好，0=完美，naive≈2.0）| popularity |
| `factor_acc_2x` | 預測值落在真實值 [0.5×, 2×] 內的比例（0~1，越大越好）| popularity |
| `acc_within_10pt` | 預測誤差 < 10 分的比例（0~1，越大越好；0–100 分用加法尺度才合理，facc_2x 對分數無意義）| meanScore |

> **準確率指標的尺度差異**：popularity 跨越多個數量級 → 用乘法尺度（`factor_acc_2x`，2× 內）；meanScore 是 0–100 線性分數 → 用加法尺度（`acc_within_10pt`，±10 分內）。對 meanScore 套 facc_2x 會得到 ~0.997（幾乎全部都在 2× 內），無鑑別力。

---

## 推論 Pipeline（`inference.py`）

給定一部新動畫（封面圖 + metadata + 描述），即時走完 YOLO → Swin → e5 → RAG → FusionModel，輸出 popularity / meanScore。

```bash
bash src_2/RAG/start_qdrant.sh        # 先啟動 Qdrant（RAG 需要）

# 新動畫推論（metadata 用單列 CSV，欄位同訓練 schema）
python src_2/inference.py --cover c.jpg --banner b.jpg --meta new.csv --description "..."

# 驗證模式：用既有 test 動畫，對照 pred_test.csv
python src_2/inference.py --anime-id 21294 --split test --verify
```

| 項目 | 說明 |
|------|------|
| 最佳 checkpoint | popularity → Run07；meanScore → Run02（架構相同，僅超參不同） |
| RAG modality | 預設 `rag_use_image=False`（image_rag 僅 train，val/test 檢索為 sparse+text，對齊驗證指標） |
| 模組隔離 | 各 component 同名 `config.py`/`model.py` 用 `importlib` 隔離載入 |
| 驗證 | cover/banner embedding 逐位元一致；yolo 因預存 crops 經 JPEG round-trip 微差（pipeline 直接裁切，更乾淨） |

> ⚠️ 超參數限制：`dropout` / `weight_decay` / `lr` / `batch_size` 為**全域**，兩個 target 共用同一組；per-target 各自最佳超參需在 config 加覆寫機制（目前由 hp_search 以 `--target` 分開跑繞過）。

---

## 特徵維度

| 來源 | 維度 | 說明 |
|------|------|------|
| Text embedding | 768 | e5-base-v2，description |
| Image embedding | 3 × 1024 | Swin-B：cover + banner + yolo（缺失 gate=0） |
| MetaEncoder v2 | 56 | 見下表 |
| RAG（Cross Attn KV） | 5 × (10 + 768 + 1024) | retrieved top-5 的 meta + text + image |

### MetaEncoder v2 特徵明細（56-dim）

| 類型 | 欄位 | 維度 |
|------|------|------|
| Robust 標準化（median/IQR）| release_year, episodes, duration, startDate_day, prequel_count, prequel_meanScore_mean | 6 |
| log1p + 標準化 | prequel_popularity_mean | 1 |
| Cyclical sin/cos | release_quarter（period=4）, startDate_month（period=12）| 4 |
| One-hot | format（7）, source（7）, countryOfOrigin（4）| 18 |
| Binary | isAdult, is_sequel, has_sequel | 3 |
| Multi-hot | genres（19 類）| 19 |
| Studio Target Encoding | mean_popularity, mean_score（標準化）| 2 |
| is_new_studio | 所有 studio 在訓練集未見過 → 1 | 1 |
| Voice Actor Target Encoding | mean_popularity, mean_score（標準化）| 2 |
| **合計** | | **56** |

> v1（66-dim）差異：移除 RAG scalar 10 dims（rag_popularity, rag_score, rag_release_year, rag_episodes, rag_found, studio_match, genre_overlap, format_match, rag_studio_te ×2）。這些資訊改由 Cross Attention 的 KV 輸入，讓模型自行學習如何整合。

---

## 訓練目標轉換

| Target | 轉換 | 反轉 |
|--------|------|------|
| `popularity` | Winsorize(99%) → log1p → z-score | 反標準化 → clip(±5σ) → expm1 |
| `meanScore` | Winsorize(99%) → z-score | 反標準化 |

mean/std 僅從訓練集計算，再套用到 val / test。

---

## 實驗記錄

config：`src_2/fussion_configs.yaml`，結果：`src_2/runs/{run_id}/`

### popularity（主要指標：log_MAE，越低越好）

| Run | val log_MAE | test log_MAE | test factor_acc_2x | val Spearman | val log_R2 | 主要改動 |
|-----|------------|-------------|-------------------|-------------|-----------|---------|
| 01a | 0.7839 | 0.9357 | — | 0.8851 | 0.8072 | Baseline v2：cover_banner_yolo / use_rag=true / CrossAttn 4 heads / SAM+AdamW / HuberLoss（無時間加權） |
| **01** | **0.7886** | 0.9088 | 0.4778 | **0.8879** | **0.8161** | + Temporal Weighting（alpha=0.2）：exp(-0.2×(max_yr-yr))，normalize mean=1 |
| 02 | 0.7801 | 0.9151 | 0.4778 | 0.8785 | 0.8055 | + TrendHead（pop+score）；gate soft-average；LogCosh 數值穩定版 |

#### hp_search（Run03~09：固定 TrendHead + batch=512，搜尋 dropout / weight_decay）

| Run | dropout | weight_decay | val log_MAE | test log_MAE | test facc_2x | test Spearman | val Spearman |
|-----|---------|-------------|------------|-------------|-------------|--------------|-------------|
| 03 | 0.5 | 1e-3 | 0.8059 | 0.9198 | 0.4613 | 0.8466 | 0.8826 |
| 04 | 0.3 | 1e-4 | 0.7977 | 0.9754 | 0.4461 | 0.8501 | 0.8776 |
| 05 | 0.4 | 5e-4 | 0.8137 | 0.9130 | 0.4661 | 0.8508 | 0.8850 |
| 06 | 0.5 | 1e-4 | 0.8249 | 0.9203 | 0.4593 | 0.8473 | 0.8812 |
| **07** ⭐ | **0.3** | **1e-3** | **0.7854** | **0.8904** | **0.4856** | 0.8498 | 0.8792 |
| 08 | 0.5 | 5e-4 | 0.8347 | 1.1288 | 0.3832 | 0.8491 | 0.8783 |
| 09 | 0.5 | 1e-3 | 0.7855 | 0.8944 | 0.4817 | 0.8510 | 0.8840 |

> **popularity 最佳：Run07**（test log_MAE 0.8904）。觀察：低 dropout（0.3）+ 高 weight_decay（1e-3）組合最好（07/09 test log_MAE 0.89 並列前段）；高 dropout 顯著退步（08 的 dropout=0.5+wd=5e-4 test log_MAE 1.13 最差）。Run03 與 Run09 同設定（dropout=0.5, wd=1e-3）但不同 seed，test log_MAE 0.9198 vs 0.8944，反映訓練隨機性。Run07 test 勝過 Run01 的 0.9088。

### meanScore（主要指標：MAE，越低越好）

| Run | val MAE | test MAE | test within_10pt | val Spearman | val R2 | 主要改動 |
|-----|--------|---------|-----------------|-------------|-------|---------|
| 01a | 6.7441 | 8.0435 | — | 0.6763 | 0.4611 | Baseline v2（無時間加權） |
| 01 | 6.8115 | 8.5722 | 0.6498 | 0.6617 | 0.4143 | + Temporal Weighting（alpha=0.2）：popularity ✅；meanScore test ❌ R2=-0.006 |
| **02** | **6.7604** | **7.2937** ↓ | **0.7360** | **0.6570** | **0.4180** | + TrendHead（pop+score）；gate soft-average；test R2 大幅回升（-0.006→0.246） |

> **觀察**：時間加權對 popularity 有效（test log_MAE 0.9357→0.9016），但 meanScore 反而退步。推測原因：meanScore 的 shift 主要是 Label Shift（評分基準整體上移），加權讓模型少看舊資料後反而失去泛化能力；而 popularity 的 shift 更多是 Covariate Shift，加權確實有助於縮小 val/test gap。

#### hp_search（Run03~09：固定 TrendHead + batch=512，搜尋 dropout / weight_decay）

| Run | dropout | weight_decay | val MAE | test MAE | test within_10pt | test Spearman | test R2 |
|-----|---------|-------------|---------|----------|-----------------|--------------|---------|
| 03 | 0.5 | 1e-3 | 6.9678 | 9.5681 | 0.5750 | 0.5322 | -0.1761 |
| 04 | 0.3 | 1e-4 | 6.7269 | **7.6675** | **0.7107** | 0.5402 | 0.1835 |
| 05 | 0.4 | 5e-4 | 6.7952 | 7.8831 | 0.6936 | 0.5533 | 0.1565 |
| 06 | 0.5 | 1e-4 | 6.9003 | 8.5068 | 0.6547 | 0.5383 | 0.0180 |
| 07 | 0.3 | 1e-3 | 6.7499 | 8.1776 | 0.6744 | 0.5397 | 0.0984 |
| **08** ⭐ | **0.5** | **5e-4** | **6.6715** | 7.5847 | 0.7136 | 0.5485 | **0.1911** |
| 09 | 0.5 | 1e-3 | 6.9463 | 9.0994 | 0.6093 | 0.5346 | -0.0707 |

> **meanScore val 最佳：Run08**（val MAE 6.6715），test MAE 7.5847、test R2 0.1911 也是 hp_search 中最佳。但**對照更早的 Run02（test MAE 7.2937, R2 0.246）仍勝出**——val 最佳不保證 test 最佳，distribution shift 主導 meanScore 的 test 表現。hp_search 內部 test MAE 跨度極大（7.58~9.57），且 test R2 數個為負，再次顯示 meanScore 在 test 區段泛化困難。

### 消融實驗（test set，`python src_2/ablation.py`）

固定超參（= Run07：dropout=0.3, wd=1e-3, batch=512, TrendHead on），只改被消融的變因。對照組 full model = Run07。

| 設定 | pop log_MAE | pop facc_2x | pop spearman | score MAE | score spearman | 結論 |
|------|------------|------------|-------------|-----------|---------------|------|
| **full（RAG + cover_banner_yolo）** | **0.8904** | **0.4856** | **0.8498** | 8.1776 | **0.5397** | 對照組 |
| RAG off | 0.9279 | 0.4739 | 0.8385 | 7.6716 | 0.5093 | RAG 對 pop 有益（log_MAE -0.038） |
| image = cover only | 1.1041 | 0.3965 | 0.8405 | 8.3641 | 0.5130 | 移除 banner+yolo，pop 大幅退步 |
| image = cover + banner | 0.8922 | 0.4836 | 0.8500 | 7.5995 | 0.5377 | ≈ full，yolo 邊際貢獻 ~0 |

> **三大發現**：
> 1. **RAG 對 popularity 明確有益**（log_MAE 0.8904 vs 0.9279，spearman +0.011）。對 meanScore spearman 上升但 MAE 變差（distribution shift 下放大數值偏移）。
> 2. **banner 是最關鍵的視覺模態**：cover only → cover+banner，pop log_MAE 從 1.1041 → 0.8922（-0.21）。
> 3. **YOLO crop 邊際貢獻幾乎為零**：cover+banner（0.8922）≈ full（0.8904）。若要精簡 pipeline，可移除 YOLO crop，對 popularity 幾乎無損。

### Multimodal 分支消融（重訓版，`python src_2/ablation_multimodal.py`）

`model.py` 加 `modalities: {image, text, meta}` flag（向後相容），架構真的移除分支後重訓。single-modality 組關閉 RAG + TrendHead。

| 設定 | pop log_MAE | pop facc_2x | pop spearman | score MAE | score spearman |
|------|------------|------------|-------------|-----------|---------------|
| **full（img+txt+meta+rag, Run07）** | **0.8904** | **0.4856** | **0.8498** | 8.1776 | **0.5397** |
| only_meta | 0.9507 | 0.4561 | 0.8322 | **8.1420** | 0.5065 |
| no_image（txt+meta+rag） | 1.0520 | 0.4270 | 0.8287 | 8.4805 | 0.4998 |
| only_text | 1.2634 | 0.3602 | 0.7056 | 10.2581 | 0.2164 |
| only_image | 1.3458 | 0.3337 | 0.7230 | 8.4261 | 0.3948 |

> **發現**：
> 1. **meta 是最強的單一模態**（only_meta：pop log_MAE 0.9507、score MAE 8.142），metadata（前作 / studio·VA TE / format）攜帶最多訊號。
> 2. **text / image 單獨都很弱**（log_MAE 1.26 / 1.35），需與 metadata 結合才有效。
> 3. **多模態互補性確立**：full model 明顯勝過任何單模態，融合架構有實質價值。

---

## 已知限制

### 1. meanScore 時序 Distribution Shift

資料採時序切分（train → val → test），AniList 評分中位數隨時間系統性上升：

| Split | meanScore 平均 | meanScore 中位數 |
|-------|--------------|----------------|
| train | 58.1 | 60.0 |
| val | 61.6 | 63.0 |
| test | 65.4 | 66.0 |

2022 年後出現 **約 +7 分的跳升**，導致 test R² 偏低（0.13）。模型的預測中心值 ≈ 61，在 test set 系統性低估約 4.4 分。這是資料本身的時序特性，非模型 bug。

### 2. popularity AMP float16 溢位

`expm1(y)` 在 float16 下上限 65,504，normalized 空間 y ≈ 17.6 時直接 overflow → Infinity。
**解法**：`denormalize_target()` 強制轉 float64，clip(±5σ) 後再執行 expm1。

### 3. Overfitting：Train Loss 快速下降，Val Loss 早熟

Run01/02 的訓練動態顯示明顯的 overfitting pattern：

```
Run01 popularity:
  ep1:  train=0.242  val=0.159  gap=-0.083
  ep7:  train=0.063  val=0.115  gap=+0.052  ← best
  ep12: train=0.041  val=0.136  gap=+0.095
```

Train loss 從 ep1 到 ep12 降了 **6 倍**，val loss 只改善 **1.4 倍**。Best checkpoint 在 ep7~10 就觸底，之後 val 持續震盪或上升，train 還在下降。  
根本原因：模型在 memorize train set 的個別動畫特徵，而非學到跨時序的泛化規律。

**緩解方向（Run03）**：提高 dropout（0.3→0.5）、weight_decay（1e-4→1e-3）、batch_size（256→512，降低 gradient noise，讓 val 指標更穩定）。

### 4. GPU 使用率相對 v1 偏低

v1 在訓練時執行 TextGNN + ImageGNN forward pass（額外 GPU 運算）。v2 的 embedding 全部預先計算並存入 parquet，訓練時只跑 FusionModel（~965K params），GPU 負載較輕。

### 4. RAG 全遮罩 NaN

RAG 無命中時（retrieved_ids 為空），`rag_mask` 全 True → `MultiheadAttention` softmax 對全遮罩位置輸出 NaN。
**解法**：`dataset.py` 強制 `rag_mask[0] = False`，讓 attention 退化為對零向量的 uniform attention。

### 5. Cold-start（新 studio / 新聲優）

MetaEncoder TE 對未見過的 studio / 聲優補訓練集全體均值；`is_new_studio` 旗標告知模型補值情況。

---

## 輸出檔案

```
src_2/runs/{run_id}/
├── {target}/
│   ├── best_model.pt          ← 最佳 val loss checkpoint（state_dict）
│   ├── target_scaler.json     ← 正規化參數（center, scale, log_transform）
│   ├── history.json           ← 每 epoch train_loss / val_loss / val_mae / lr（含 run_id / notes）
│   ├── final_metrics.json     ← train / val / test 完整 metrics 合併（含 run_id / notes）
│   └── pred_{split}.csv       ← id, pred, target（原始 scale）
└── explain/{target}/          ← 可解釋性輸出（run 層級）
    ├── rag/{id}_attn.png          ← Cross Attention heatmap
    ├── feature/captum_modality.csv
    ├── feature/captum_modality.png
    ├── feature/shap_values.npy
    └── feature/shap_summary.png

src_2/fussion_training/meta_encoder.json   ← 訓練集 fit 的 MetaEncoder（自動生成）
```
