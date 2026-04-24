# 工作流程/方法說明

## 目標說明
本次專案，最主要的目標為 ```新番動畫播出前熱度預測 (Pre-release Anime Popularity Prediction Based on Multimodal Features)```。但在這份文件中，我們需要**專注**的內容為影像處理。如何將影像穩定的轉成一個向量，是本次文件的主要任務。

## 需要用到的內容

- 根據資料前處理的文件(```docs/handleoff_image_model.md```)中，已有說明要使用的檔案名稱和欄位，以下為參考使用的檔案位置和欄位:
    - 檔案位置: `data/processed/anilist_anime_multimodal_input_v1.csv`
    - 圖像欄位:
        - `coverImage_medium`（主圖）
        - `bannerImage`（輔助圖）
        - `trailer_thumbnail`（目前不打算使用）

- 將圖像轉換成向量的模型:
    - swin transformer (來源: Hugging Face `transformers`，模型：`microsoft/swin-base-patch4-window7-224`)
    - **其他內容將會手動補充**

- 網路爬蟲:
    - 根據給定的url抓取，設計成一個模組，隨時抓取


## 影像處理的架構和位置
>[! 該架構會隨日後需要擴充和調整而改變，以下為目前的初步規劃]

```
project_root/
├── config.yaml        # 設定模型、圖片處理參數、檔案輸入輸出路徑等
├── utils/             # 存放 function 和 class 的程式碼
├── main.py            # 主程式
└── results/
    ├── 01/
    │   ├── checkpoint/    # 第 1 次微調的模型權重
    │   └── best/          # 第 1 次微調中表現最好的權重
    ├── 02/
    │   ├── checkpoint/    # 第 2 次微調的模型權重
    │   └── best/          # 第 2 次微調中表現最好的權重
    └── .../

>[! 照片的部分會先存在 data/image下]
```


## 使用到的工具
>[! 以下工具需隨時更新，根據實際使用的工具和版本進行調整]

- numpy
- pandas
- torch (這部份如果需要，請先用 ```nvidia-smi``` 確認 版本和 GPU 的狀況， 並根據需要創建conda 環境) 
    >[! 任何的AI工具在這部分皆須詢問是否需要安裝conda 和確認 GPU 的狀況，並且在安裝前先確認版本和相容性，避免安裝後出現問題。]
- transformers
- requests (用於爬蟲)
- tqdm (用於顯示模型進度)


## 工作流程
>[! 以下工作流程需隨時更新，根據實際使用的流程和需求進行調整]

### 執行順序
```
getImage.py → train.py → predictor.py → output.py（日後使用）
```

---

### getImage.py
```
image_process_config.yaml
    ↓
load_config()
    ↓
read CSV → filter_by_ratio(df, ratio)
    ↓
for each row × each col (coverImage_medium, bannerImage):
    fetch_one(url, save_path)
        ├── 成功 → 存 {idx}_{col}.jpg → log_result(success)
        └── 失敗 → log_result(error)，跳過
```

---

### train.py
```
load_config() → load_model(config)
    ↓
建立 train / val / test DataLoader（各自傳入對應 df、image_col）
    ↓
┌──────────────── __getitem__ ────────────────┐
│ load_image({idx}_{col}.jpg)                 │
│     ↓                                       │
│ ResizeWithPad(224)   ← 共用前置步驟         │
│     ↓               ↓                       │
│ transform_orig      transform_aug           │
│ ToTensor            RandomResizedCrop(224)  │
│ Normalize           RandomCrop(224,pad=22)  │
│                     ColorJitter    p=0.8    │
│                     GaussianBlur   p=0.5    │
│                     RandomHFlip    p=0.5    │
│                     RandomGrayscale p=0.2   │
│                     ToTensor                │
│                     Normalize               │
│     ↓               ↓                       │
│     (orig_tensor, aug_tensor, idx)          │
└─────────────────────────────────────────────┘
    ↓
─── Training loop ───────────────────────────────────────────
每 epoch：
    _forward_orig(orig) → torch.no_grad() → orig_emb（anchor）
    _forward_aug(aug)   → grad            → aug_emb
    infonce_loss(aug_emb, orig_emb, tau)  → loss
    backprop → optimizer.step()

每 val_interval epoch：
    validate() → _val_step → infonce_loss（no backprop）→ avg val_loss
    log_metrics(train_loss, val_loss, cosine_sim, lr) → TensorBoard

─── 模型儲存 ────────────────────────────────────────────────
    val_loss 改善時
        → save_best(model, path)
        → model.save_pretrained()
        → results/{run_id}/best/        ← HuggingFace 格式，可直接 from_pretrained 載入

    每 checkpoint_interval epoch
        → save_checkpoint(model, optimizer, epoch, path)
        → results/{run_id}/checkpoint/epoch_{N}.pt  ← 含 optimizer 狀態，可續訓

─── Test ────────────────────────────────────────────────────
    evaluate_similarity() → avg cosine_similarity(orig_emb, aug_emb)
    結束 → close_writer()
```

---

### predictor.py（test set）
```
load model from results/{run_id}/best/
    ↓
建立 test DataLoader（coverImage_medium, image_col）
    ↓
__getitem__: load_image → ResizeWithPad → transform_orig → orig_tensor
    ↓
predict_one_col(model, loader) → {idx: embedding}  ← cover_embs
    ↓
建立 test DataLoader（bannerImage, image_col）
    ↓
predict_one_col(model, loader) → {idx: embedding}  ← banner_embs
    ↓
merge_embeddings(cover_embs, banner_embs)
    → outer join，缺失填 None
    → DataFrame: [idx, coverImage_emb, bannerImage_emb]
    ↓
save_embeddings() → data/processed/image_embeddings.parquet
```

---

### output.py（日後使用）
```
ImageEmbedder(model_path, config)
    → load model from results/{run_id}/best/
    → transform_orig（ResizeWithPad → ToTensor → Normalize）
    → model.eval()

embed(image_path)
    → load_image → ResizeWithPad → transform_orig → forward → (768,) numpy

embed_batch(image_paths)
    → 逐張 load → stack batch → forward → (N, 768) numpy

embed_url(url)
    → 下載到 tempfile → embed() → (768,) numpy
```

---

### InfoNCE loss 說明
- `L = -log( exp(sim(aug, orig) / τ) / Σ exp(sim(aug, all_orig) / τ) )`
- `sim()` 為 cosine similarity；`τ`（temperature）預設 0.07
- batch 內其他圖片的 original embedding 自動作為 negative（batch size 64，每張圖有 63 個 negative）
- 防止 representation collapse：加入 negative 後模型必須區分不同圖片

---

### step4：輸出 embedding
- 儲存為**單一 parquet 檔**：`data/processed/image_embeddings.parquet`
- 欄位：`idx`、`coverImage_emb`（768 維）、`bannerImage_emb`（768 維）
- 讀取：`np.array(df["coverImage_emb"].tolist())` → shape `(N, 768)`
- 可直接用 `idx` 和原本 CSV merge

## 各 Function 設計簡介和說明

### 檔案結構

```
project_root/
├── src/
│   ├── config.py        # 讀取 yaml 設定
│   ├── model.py         # 載入 Swin Transformer、取得 embedding
│   └── loss.py          # InfoNCE loss
├── util/
│   ├── getImage.py      # 爬蟲、圖片下載
│   ├── image_process.py # transform pipeline、ResizeWithPad
│   ├── dataset.py       # Dataset class、DataLoader
│   ├── train.py         # 訓練、驗證、存檔、TensorBoard
│   └── predictor.py     # 批次 inference → parquet
├── output.py            # 對外推論介面（ImageEmbedder class）
└── main.py
```

---

### src/config.py
- `load_config(config_path)` → 讀取 yaml，回傳 dict

### src/model.py
- `load_model(config)` → 從 HuggingFace 載入 SwinModel（pretrained），回傳 model（無 classifier head）
- `get_embedding(model, pixel_values)` → forward pass，取 pooler_output，回傳 shape `(B, 768)` tensor

### src/loss.py
- `infonce_loss(aug_emb, orig_emb, tau)` → 計算 cosine similarity matrix，套用 InfoNCE 公式，回傳 scalar loss

---

### util/getImage.py
- `make_image_dir(image_dir)` → 確認資料夾存在，不存在則建立
- `fetch_one(url, save_path, timeout)` → 抓單張圖片存檔，成功回傳 True，失敗回傳 False
- `log_result(log_path, idx, col, url, status)` → 寫一行到 fetch_log.csv
- `filter_by_ratio(df, ratio, seed=42)` → 隨機取 ratio 比例的列，回傳篩選後 df
- `getImage(config)` → 讀 CSV → `filter_by_ratio()` → `fetch_one()` for each row/col
    - 命名規則：`{idx}_{col}.jpg`（e.g. `123_coverImage_medium.jpg`）

### util/image_process.py

**ResizeWithPad（class）**：保留完整畫面，不裁切
```
scale = target_size / max(w, h)   ← 以最長邊為基準
new_w, new_h = int(w*scale), int(h*scale)
img.resize((new_w, new_h))        ← 等比縮放
pad 左右 or 上下至 target_size    ← 補黑邊
```

- `get_transform_original(image_size)` → Compose: ToTensor → Normalize(ImageNet)（ResizeWithPad 已在 __getitem__ 完成）
- `get_transform_aug(config)` → 呼叫所有 `_make_*` helper，組成最終 Compose
    - `_make_crop(config)` → RandomResizedCrop(scale=[0.8,1.0], ratio=[0.85,1.18])，p=1.0
    - `_make_random_crop(config)` → RandomCrop(224, padding=int(224 * max_crop_ratio))，p=0.3
    - `_make_color_jitter(config)` → RandomApply([ColorJitter(...)], p=0.8)
    - `_make_gaussian_blur(config)` → RandomApply([GaussianBlur(...)], p=0.5)
    - `_make_flip(config)` → RandomHorizontalFlip(p=0.5)
    - `_make_grayscale(config)` → RandomGrayscale(p=0.2)
- `load_image(path)` → 開啟圖片，轉 RGB PIL Image，失敗回傳 None

### util/dataset.py
- `class AnimeImageDataset(Dataset)`
    - `__init__(self, df, image_dir, image_col, transform_orig, transform_aug)`
    - `__len__(self)` → 回傳資料筆數
    - `__getitem__(self, i)` → `load_image()` → `ResizeWithPad(224)`（共用前置）→ `transform_orig` + `transform_aug` → 回傳 `(orig_tensor, aug_tensor, idx)`
- `get_dataloader(dataset, batch_size, shuffle)` → 建立 DataLoader，回傳

### util/train.py

單步 forward：
- `_forward_orig(model, pixel_values, device)` → `torch.no_grad()` forward，回傳 orig embedding
- `_forward_aug(model, pixel_values, device)` → 正常 forward（更新梯度），回傳 aug embedding

單步 train / val：
- `_train_step(model, orig, aug, optimizer, loss_fn, device)` → forward × 2 → loss → backprop → optimizer.step()，回傳 scalar loss
- `_val_step(model, orig, aug, loss_fn, device)` → `torch.no_grad()`，forward × 2 → loss，不 backprop，回傳 scalar loss

epoch 迴圈：
- `train_one_epoch(model, loader, optimizer, loss_fn, device)` → 迭代 loader，呼叫 `_train_step`，回傳平均 loss
- `validate(model, loader, loss_fn, device)` → 迭代 loader，呼叫 `_val_step`，回傳平均 val loss

評估：
- `_compute_cosine_similarity(emb_a, emb_b)` → 計算 cosine similarity，回傳 scalar
- `evaluate_similarity(model, loader, device)` → 迭代 test loader，回傳平均 cosine similarity

TensorBoard：
- `init_writer(log_dir)` → 建立 SummaryWriter，回傳 writer
- `log_metrics(writer, metrics, epoch)` → 寫入一組 metrics dict（train_loss, val_loss, cosine_sim, lr）
- `close_writer(writer)` → 關閉 writer

存檔（位置與時機）：

| 時機 | 函式 | 儲存位置 |
|------|------|----------|
| 每 `checkpoint_interval` epoch | `save_checkpoint(model, optimizer, epoch, path)` | `results/{run_id}/checkpoint/epoch_{N}.pt` |
| val loss 改善時 | `save_best(model, path)` | `results/{run_id}/best/`（HuggingFace 格式） |

主流程：
- `train(config)` → 組裝所有模組，執行完整訓練迴圈
    - 每 `val_interval` epoch：`train_one_epoch` → `validate` → `log_metrics`
    - val loss 改善 → `save_best`
    - 每 `checkpoint_interval` epoch → `save_checkpoint`
    - 結束時 → `close_writer`

### util/predictor.py
- `predict_one_col(model, loader, device)` → `torch.no_grad()` inference，回傳 `{idx: embedding}` dict
- `merge_embeddings(cover_embs, banner_embs)` → 合併成 DataFrame（欄位：`idx`、`coverImage_emb`、`bannerImage_emb`）
- `save_embeddings(df, path)` → 儲存為 parquet
- `predict(model, config)` → `predict_one_col`（cover + banner）→ `merge_embeddings` → `save_embeddings`

---

### output.py
對外推論介面，模型只載入一次，之後可連續調用。

- `class ImageEmbedder`
    - `__init__(self, model_path, config)` → 載入模型、建立 transform、`model.eval()`
    - `embed(self, image_path)` → 單張圖片路徑 → 回傳 shape `(768,)` numpy array
    - `embed_batch(self, image_paths)` → 多張圖片路徑 → 回傳 shape `(N, 768)` numpy array
    - `embed_url(self, url)` → 從 URL 下載圖片 → 回傳 shape `(768,)` numpy array

| | predictor.py | output.py |
|---|---|---|
| 輸入 | DataLoader（整批） | 圖片路徑 / URL（彈性） |
| 輸出 | parquet 檔案 | numpy array |
| 用途 | 一次性產生全部 embedding | 隨時調用、外部接口 |

