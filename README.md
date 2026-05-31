# Image Pipeline — image-process 分支

本分支為「新番動畫播出前熱度預測」專案的**影像處理分支**，負責從動漫封面圖（coverImage）與橫幅圖（bannerImage）中提取視覺特徵向量，供下游多模態融合模型使用。

---

## 檔案結構

```
project_root/
├── src/
│   ├── config.py              # 讀取 yaml 設定（load_config, load_yolo_config）
│   ├── model.py               # 載入 Swin Transformer、取得 embedding
│   ├── loss.py                # InfoNCE loss
│   └── YOLO.py                # detect_person / detect_faces wrapper
├── util/
│   ├── getImage.py            # 爬蟲、圖片下載
│   ├── image_process.py       # transform pipeline、ResizeWithPad
│   ├── dataset.py             # Dataset class、yolo_collate_fn、get_dataloader
│   ├── train.py               # 訓練、驗證、存檔、TensorBoard
│   ├── yolo_for_image.py      # YOLO 偵測 + 裁切 + canvas 輸出
│   └── predictor.py           # 批次 inference → parquet
├── output.py                  # 對外推論介面（ImageEmbedder class）
├── main.py                    # 主程式入口
├── image_process_config.yaml  # Swin 訓練與推論設定
├── yolo_config.yaml           # YOLO 偵測設定
├── docs/
│   ├── relatedWork.md         # Related Work（結構化版）
│   ├── methodnology.md        # Methodology（結構化版）
│   ├── paper.md               # 正式論文草稿（Related Work + Methodology）
│   └── pappers/               # 參考論文與筆記
├── gen/                       # YOLO 裁切圖與 canvas 輸出
└── results/
    ├── {run_id}/
    │   ├── best/              # 最佳模型權重（HuggingFace 格式）
    │   ├── checkpoint/        # 訓練 checkpoint（含 optimizer 狀態）
    │   └── logs/              # TensorBoard log
```

---

## 執行順序

```
python util/getImage.py        # 下載圖片、記錄 fetch_log.csv
python util/yolo_for_image.py  # YOLO 偵測人物/臉部、裁切、儲存 canvas
python main.py                 # Swin Transformer 自監督訓練
python util/predictor.py       # 批次推論，產生 image_embeddings.parquet
```

---

## 設定檔說明

### `image_process_config.yaml`

| 區塊 | 說明 |
|------|------|
| `data` | 輸入 CSV 路徑、圖像欄位、image_size |
| `model` | 骨幹網路名稱、stage 模式、fusion_embed_mode |
| `training` | batch_size、learning_rate、epochs、warmup、scheduler |
| `inference` | run_id、model_path、embedding_path、file_kind、use_yolo |

### `yolo_config.yaml`

| 區塊 | 說明 |
|------|------|
| `yolo` | use（是否啟用 YOLO） |
| `detect_mode` | `person` / `face` / `both` |
| `model` + `detection` | 人物偵測模型參數（level、conf、iou、max_detections） |
| `face_model` + `face_detection` | 臉部偵測模型參數 |

---

## YOLO 偵測流程

1. 圖像 upscale 至 640px（提升小圖偵測率）
2. 依 `detect_mode` 執行人物偵測（`deepghs/anime_person_detection`）、臉部偵測（`deepghs/anime_face_detection`）或兩者
3. 依信心分數排序，取前 `max_detections` 個裁切區域
4. 每個裁切區域經 ResizeWithPad(224) 後送入 transform pipeline
5. 無偵測時 fallback 使用整張圖像

`yolo.use` 參數可在 `yolo_config.yaml` 設定，亦可由 `image_process_config.yaml` 的 `inference.use_yolo` 覆蓋。

---

## 模型訓練

使用 **Swin Transformer**（`microsoft/swin-base-patch4-window7-224`）作為視覺骨幹網路，以 **InfoNCE loss** 進行自監督對比式學習。

- **正樣本對**：同一張圖的 original view 與 augmented view
- **負樣本**：batch 內其他圖片的 original embedding（batch size=64，每張圖有 63 個負樣本）
- **學習率排程**：Linear Warmup + Cosine Annealing
- **溫度參數 τ**：0.07

訓練指標透過 TensorBoard 追蹤，啟動指令：

```bash
conda run -n anime_prediction tensorboard --logdir results/{run_id}/logs
```

---

## 輸出格式

輸出儲存於 `data/processed/image_embeddings.parquet`，以 AniList 動畫 ID（`idx`）作為索引。

**`stage = false`（預設）**

| 欄位 | 維度 | 說明 |
|------|------|------|
| `idx` | — | AniList 動畫 ID |
| `coverImage_emb` | 1024 | 封面圖 pooler_output |
| `bannerImage_emb` | 1024 | 橫幅圖 pooler_output |

**`stage = true`**

| 欄位 | 維度 | 語義 |
|------|------|------|
| `coverImage_emb_s0` | 128 | 局部紋理、線條筆觸 |
| `coverImage_emb_s1` | 256 | 色塊分布、局部結構 |
| `coverImage_emb_s2` | 512 | 人物部位、光影風格 |
| `coverImage_emb_s3` | 1024 | 整體語義、畫風流派 |

bannerImage 欄位命名方式相同（`bannerImage_emb_s0~s3`）。

下游模型讀取方式：

```python
import numpy as np, pandas as pd

df = pd.read_parquet("data/processed/image_embeddings.parquet")
cover = np.array(df["coverImage_emb"].tolist())   # (N, 1024)
banner = np.array(df["bannerImage_emb"].tolist())  # (N, 1024)
```

---
