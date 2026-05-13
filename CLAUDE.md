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
    backprop → optimizer.step() → scheduler.step()

    scheduler（Warmup + Cosine Annealing）：
        epoch 1 ~ warmup_epochs   : lr 從 0 線性升到 learning_rate
        epoch warmup_epochs ~ end : lr cosine decay 到 ~0

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
    - `idx` 為 **AniList 動畫 ID**（來自 CSV 的 `id` 欄位），**不是** DataFrame 的 row index
- 讀取：`np.array(df["coverImage_emb"].tolist())` → shape `(N, 768)`
- 可直接用 `idx` 和原本 CSV 以 `id` 欄位 merge

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
    - 建立 scheduler：`LinearLR`（warmup）→ `CosineAnnealingLR`（cosine decay），以 `SequentialLR` 串接
    - 每 epoch：`train_one_epoch` → `scheduler.step()`
    - 每 `val_interval` epoch：`validate` → `evaluate_similarity`（val set）→ `log_metrics`
    - val loss 改善 → `save_best`
    - 每 `checkpoint_interval` epoch → `save_checkpoint`
    - 訓練結束：`evaluate_similarity`（test set）→ `log_metrics` → `close_writer`

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

---

## TensorBoard 使用說明

TensorBoard log 儲存於 `results/{run_id}/logs/`（例如 `results/01/logs/`）。

### 啟動指令

```bash
# 啟動特定 run（例如 run_id = 01）
conda run -n anime_prediction tensorboard --logdir results/01/logs

# 啟動並比較所有 run
conda run -n anime_prediction tensorboard --logdir results

# 指定 port（預設 6006，若被佔用可改其他 port）
conda run -n anime_prediction tensorboard --logdir results/01/logs --port 6007

# 文字 Embedding
python -m src.text_branch.run_text_embedding_pipeline

# 圖片 Embedding（需要 data/image/ 內有圖片，以及 src/fussion_branch/model/best/ 的 checkpoint）
python -m src.image_branch.run_fetch       # 下載封面圖片
python util/split_images_by_split.py       # 將圖片依 split 分入子資料夾
python -m src.image_branch.run_train       # 微調 Swin Transformer
python -m src.image_branch.run_predict     # 推論 → .exp/image_embedding/image_embeddings.parquet

# Fusion RAG Pipeline
python -m src.fussion_branch.run_rag       # 建立 Qdrant + 查詢所有 split

# Fusion MLP 訓練
python -m src.fussion_branch.run_train                     # 訓練兩個目標
python -m src.fussion_branch.run_train --target popularity # 只訓練 popularity
python -m src.fussion_branch.run_train --target meanScore  # 只訓練 meanScore

# SHAP feature importance 分析（訓練完成後）
python -m src.fussion_branch.run_shap --target popularity
python -m src.fussion_branch.run_shap --target meanScore

# 快速測試（1000 筆 smoke test）
python test_train.py
```

啟動後開啟瀏覽器：[http://localhost:6006](http://localhost:6006)

### 架構

```
src/
  text_branch/        all-MiniLM-L6-v2，輸出 artifacts/text_embeddings_{split}.parquet（384 維）
  image_branch/       Swin-base 對比學習微調，輸出 .exp/image_embedding/image_embeddings.parquet（1024 維）
  fussion_branch/     RAG（Qdrant sparse）+ MLP 融合模型
    text_components/  text_branch 元件本地副本（TextPreprocessor、EmbeddingGenerator）
    image_components/ image_branch 元件本地副本（image_process、configs）

data/
  interim/            清洗後中間層 CSV
  processed/          特徵工程 CSV + 各 split CSV（anilist_anime_data_processed_v1_{split}.csv）
  fussion/
    fusion_meta_{train,val,test,holdout_unknown}.csv        — 33 欄（原始，未清理）
    fusion_meta_clean_{train,val,test,holdout_unknown}.csv  — 24 欄（模型可用，見下方說明）
  image/
    train_image/          train split 的封面 + banner 圖片
    validation_image/     val split 的封面 + banner 圖片
    test_image/           test split 的封面 + banner 圖片
    holdout_unknow_image/ holdout_unknown 的封面 + banner 圖片

artifacts/
  text_embeddings_{split}.parquet   欄位：id, emb_0..emb_383, popularity, meanScore, split（text_branch 產出）

src/fussion_branch/RAG/             （RAG 中間產物）
  sparse_encoder.json               genre+studio 稀疏向量詞表（1483 tokens：19 genres + 1464 studios）
  rag_features_{split}.parquet      欄位：id, rag_title_romaji, rag_popularity, rag_score, rag_release_year, rag_studios, rag_found

.exp/
  image_embedding/image_embeddings.parquet   平坦格式 id + img_0..img_1023（1024 維 Swin-base）
  fussion/
    meta_encoder.json               已擬合的 MetaEncoder 詞表與 scaler
    results/{run_id}/{target}/      訓練 checkpoint、metrics、target_scaler.json

qdrant_db/            本地 Qdrant 向量資料庫（已 gitignore，執行 run_rag 自動重建）
results/              image branch checkpoint（已 gitignore）
.exp/                 所有 branch 的推論/訓練產出（已 gitignore）
```

---

## fusion_meta_clean 欄位（24 欄）

| 類別 | 欄位 | 編碼方式 |
|------|------|---------|
| 識別 | `id`, `title_romaji`, `title_english`, `title_native` | 不編碼（join / 顯示用） |
| 識別 | `description` | 不編碼（text embedding 原始文字，null 約 4–10%） |
| 動畫規格 | `format`（7 類） | one-hot |
| 動畫規格 | `episodes`, `duration` | 標準化 |
| 時間 | `release_year` | 標準化 |
| 時間 | `release_quarter`（4）、`startDate_month`（12） | cyclical sin/cos |
| 時間 | `startDate_day` | 標準化 |
| 內容屬性 | `source`（7 類，含 UNKNOWN_SOURCE）、`countryOfOrigin`（4 類） | one-hot |
| 內容屬性 | `isAdult` | binary 0/1 |
| 標籤 | `genres`（19 類） | multi-hot |
| 標籤 | `studios` | Target Encoding（log1p_popularity, score）|
| 續集 | `is_sequel`、`has_sequel` | binary 0/1 |
| 續集 | `prequel_count` | 標準化 |
| 續集 | `prequel_popularity_mean` | log1p + 標準化 |
| 續集 | `prequel_meanScore_mean` | 標準化 |
| 目標 | `popularity` | log1p + 標準化（訓練目標） |
| 目標 | `meanScore` | 標準化（訓練目標） |

**已移除欄位：** `type`（zero variance）、`season` / `seasonYear` / `startDate_year`（重複時間資訊）、`release_date` / `release_quarter_key`（重複）、`popularity_quarter_pct` / `popularity_quarter_bucket`（target leakage）、`is_source_missing`（冗餘，`source='UNKNOWN_SOURCE'` 由 one-hot 自然處理）

**MetaEncoder 編碼說明：**
- `studios`：Target Encoding（log1p_popularity + score），取該動畫所有 studio 的平均，z-score 標準化
- `voice_actor_names`：同上，缺失時以訓練集全體均值補值（標準化後 ≈ 0）
- `rag_popularity`：log1p 後標準化（與訓練目標單位一致）
- RAG overlap：`studio_match`（binary）、`genre_overlap`（Jaccard）、`format_match`（binary）
- MetaEncoder 只在訓練集 fit，transform 套用於所有 split

---

## Fusion 模型

```
輸入：text(384) + image(1024，缺失時補零) + meta_rag(65) = 1473 維

架構：
  text_proj  (384→128, Linear→LN→GELU) → × α_text  ┐
  image_proj (1024→256, Linear→LN→GELU) → × α_image ┤→ concat(448) → backbone → head
  meta_proj  (65→64, Linear→LN→GELU)   → × α_meta  ┘

  Modality Gate（各自獨立）：
    α = softmax([Linear(128→1)(t), Linear(256→1)(img), Linear(64→1)(m)])

  backbone: Dropout → [Linear→LN→GELU→Dropout] × 3（256→128→64）
  head: Linear(64→1)

輸出：scalar（每個 target 獨立）
```

### 評估指標

| 指標 | 說明 | 適用 |
|------|------|------|
| MAE / RMSE | 原始尺度誤差 | 兩個 target |
| Spearman ρ | 排名相關性 | 兩個 target |
| MAPE | 百分比誤差（scale-free） | popularity |
| log_MAE | log 空間誤差（對應訓練目標） | popularity |
| bucket_accuracy | 訓練集四分位分類準確率 | popularity |

---

## 程式碼慣例

- 所有模組從專案根目錄以 `python -m src.<branch>.run_*` 執行
- import 使用 `from src.<branch>.<module> import ...`，不使用相對 import
- Config 為 YAML 檔，放在 `src/<branch>/configs/`
- Embedding 輸出使用 Parquet 格式；元資料使用 CSV
- `id`（AniList 整數）為所有資料集的 join key
- CSV 中 `studios` 欄位為 JSON 字串 — 用 `json.loads()` 解析，不要用 `ast.literal_eval()`（含 JSON boolean）
- CSV 中 `genres` 欄位為 Python list 字串 — 用 `ast.literal_eval()` 解析

---

## 資料規則

- **時間序 split**：train = 較早季度，val/test = 較晚季度（時序切分，非隨機）
- **holdout_unknown**（943 筆，`release_quarter` 缺失）不參與任何模型訓練與評估
- **Pre-release 限制**：只允許播出前已知的特徵；排除 `averageScore`、`favourites`、`trending`、`status`
- **RAG leakage 防護**：Qdrant 只收錄 training set；Python 端 post-filter 移除 `release_year >= target`
- **Target leakage**：`popularity_quarter_pct` 和 `popularity_quarter_bucket` 絕對不能作為模型輸入特徵

---

## 常見操作

**新增 fusion metadata 欄位：**
1. 修改 source processed CSV 或 `fusion_data_exploration.ipynb`
2. 透過 notebook 或 script 重新產生 `fusion_meta_clean_{split}.csv`
3. 重新執行 `python -m src.fussion_branch.run_rag` 以更新 Qdrant payload 欄位

**重建文字 Embedding：**
```bash
python -m src.text_branch.run_text_embedding_pipeline
```
覆寫 `artifacts/text_embeddings_{train,val,test}.parquet`。

**產生圖片 Embedding：**
```bash
python -m src.image_branch.run_predict
```
輸出 `.exp/image_embedding/image_embeddings.parquet`（平坦格式 `id + img_0..img_1023`）。
若檔案不存在，FusionDataset 自動補零向量（1024 維），訓練可正常進行。

**重置 Qdrant collection：**
每次執行 `run_rag.py` 都會自動重建 `qdrant_db/`，直接刪除資料夾或重跑腳本即可。

---

## 注意事項

- 執行訓練指令前，必須確認 GPU 可用（`torch.cuda.is_available()`）
- 不得將後製特徵（`averageScore`、`favourites`、`trending`）加入 fusion_meta — 會造成 data leakage
- `rag_features` 中的 `rag_title_romaji` 僅供可解釋性使用 — 不得輸入 MLP
- `rag_features` 中的 `rag_studios` 為 JSON 字串；multi-hot encoding 在模型訓練時進行
- Image embedding 維度為 **1024**（Swin-base `pooler_output`）；不是 768（那是 Swin-tiny）
- `startDate_month` 補值：使用 season 對應月份（SPRING→4、SUMMER→7、FALL→10、WINTER→1）
- holdout_unknown 的 `startDate_month` / `release_quarter` / `season` 全部為 null，無法補值

---

## 注意事項（MetaEncoder）

- MetaEncoder 有改動時，必須刪除舊的 `.exp/fussion/meta_encoder.json` 讓訓練重新 fit
- `rag_features` 中的 `rag_studios` 為 JSON 字串；overlap scalar 與 TE 在 MetaEncoder.transform() 計算
- `rag_features` 中的 `rag_title_romaji` 僅供可解釋性使用，不得輸入 MLP
