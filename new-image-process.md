# image-process branch 更新內容

> 與 `dev` 的差異，基於 `git diff dev...image-process`

---

## 1. 新增：YOLO 角色偵測（`src/YOLO.py` + `yolo_config.yaml`）

新增獨立的 YOLO 封裝模組，透過 `imgutils` 庫進行偵測。

**`src/YOLO.py`**
- `detect_person()`: 偵測動畫人物，回傳 `[(bbox, 'person', confidence), ...]`
- `detect_faces()`: 偵測動畫臉部，回傳 `[(bbox, 'face', confidence), ...]`

**`yolo_config.yaml`**（新增設定檔）
```yaml
detect_mode: face     # person | face | both
model:                # person 偵測模型（deepghs/anime_person_detection）
face_model:           # face 偵測模型（deepghs/anime_face_detection）
detection:            # conf_threshold, iou_threshold, max_detections=5
face_detection:       # 同上，門檻針對臉部微調
yolo:
  use: true           # 全域開關
```

偵測流程：對輸入圖片先 upscale 到 640px（提升小圖偵測率），再依 `detect_mode` 偵測，最多保留 top-N 個框，無結果時 fallback 整張圖。

---

## 2. 新增：Stage Embedding（`src/model.py`）

Swin-B 有 4 個 stage，新增函式直接提取各 stage 的特徵：

```
Stage 0: (B, 128, 56, 56)  → 局部紋理、線條筆觸
Stage 1: (B, 256, 28, 28)  → 色塊分布、局部結構
Stage 2: (B, 512, 14, 14)  → 人物部位、光影風格
Stage 3: (B,1024,  7,  7)  → 整體語義、畫風流派
```

新增函式與 class：
- `get_stage_embeddings(model, pixel_values)` → `[(B,128), (B,256), (B,512), (B,1024)]`，對每個 stage 做 spatial mean pool
- `StageProjector(project_dim)` → 將各 stage 投影到相同維度（e.g. 256），方便 downstream 使用

設定開關（`image_process_config.yaml`）：
```yaml
model:
  stage: false       # true → 輸出 4 個 stage vector；false → pooler_output (1024-dim)
  projection: false
  project_dim: 256
```

---

## 3. 重構：`output.py` — 支援 YOLO + Stage + CLI 入口

`ImageEmbedder` 全面支援 YOLO crop 和 stage embedding：

| 功能 | 舊版 | 新版 |
|------|------|------|
| 單張 embed | `pooler_output` only | YOLO crop → mean pool；或 stage embedding |
| batch embed | 自行 stack tensor | 改用 `embed()` 逐張處理（支援可變 crop 數） |
| 儲存格式 | `coverImage_emb`, `bannerImage_emb` 單欄 | stage 模式下拆成 `_s0~s3` 四欄 |

新增 `main()` CLI 入口（`python output.py`）：
- `file_kind: file` → 讀 CSV，批次處理整個資料集
- `file_kind: image` → `python output.py <image_path>`，處理單張圖片

新增 `inference` config section（與 `output` section 並列）：
```yaml
inference:
  file_kind: file    # file | image
  use_yolo: true     # 覆蓋 yolo_config.yaml 的設定
```

---

## 4. 更新：`util/dataset.py` — YOLO 訓練 pipeline

- `AnimeImageDataset` 新增 `use_yolo` 參數
- YOLO 開啟時，`__getitem__` 回傳 `(crops_tensor, aug_tensor, idx)`，其中 `crops_tensor` shape 為 `(N_crops, 3, 224, 224)`（N 可變）
- 新增 `yolo_collate_fn`：處理 batch 內 crop 數不同的情況（回傳 List 而非 stack tensor）
- `get_dataloader()` 新增 `use_yolo` 參數，自動切換 `collate_fn`

---

## 5. 更新：`util/predictor.py` — Stage + YOLO 推論

- `predict_one_col()` 新增 `use_stage` 參數
  - YOLO 路徑：對每個 sample 的多個 crops 各自 forward，再 mean pool
  - Stage 路徑：回傳 `{idx: [Tensor(128,), Tensor(256,), Tensor(512,), Tensor(1024,)]}`
- `merge_embeddings()` stage 模式下儲存為 `coverImage_emb_s0~s3` / `bannerImage_emb_s0~s3`
- `predict()` 新增 `use_yolo` 參數（注意：banner 圖永遠不開 YOLO，只有 cover 開）

---

## 6. 更新：`util/train.py` — YOLO 訓練支援 + Bug fix

- 新增 `_pool_embeddings(model, samples, device, no_grad)` 統一處理兩種輸入：
  - `Tensor(B, 3, 224, 224)` → 直接 forward
  - `List[Tensor(N_i, 3, 224, 224)]` → 逐樣本 forward + mean pool → stack 成 `(B, 1024)`
- `_forward_orig` / `_forward_aug` 改用 `_pool_embeddings` 取代原本的直接呼叫
- 訓練/驗證/測試 DataLoader 全部加入 `use_yolo` 參數
- **Bug fix**：修正 val/test 迴圈中 `enable_grad` 可能覆蓋外層 `no_grad` context 的問題

---

## 7. 工具腳本：`util/yolo_for_image.py`

Debug 用視覺化工具，可對指定範圍的圖片跑 YOLO 偵測並將原圖 + crops 拼成 canvas 儲存，方便肉眼確認偵測品質。

---

## 總結：主要新能力

| 能力 | 說明 |
|------|------|
| **人物/臉部偵測 crop** | 先裁切角色區域再 embed，比整張圖更 focused |
| **Stage embedding** | 可取 Swin 4 個 stage 的中間特徵，而非只用最終 pooler_output |
| **Face detection 模式** | 除 person 外新增 `face` / `both` 模式 |
| **彈性 CLI** | `output.py` 可直接當 script 跑（單張圖或整批 CSV） |
