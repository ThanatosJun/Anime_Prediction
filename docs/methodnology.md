# Methodology

## 影像特徵提取流程

本研究的影像處理流程分為四個階段，依序執行：資料收集、人物偵測與裁切、對比式學習訓練、以及特徵向量輸出。整體架構如下：

```
getImage.py → yolo_for_image.py → train.py → predictor.py
```

---

## 一、資料收集

資料來源為 `data/processed/anilist_anime_multimodal_input_v1.csv`，使用其中兩個圖像欄位：

- `coverImage_medium`：動漫主封面圖
- `bannerImage`：橫幅輔助圖

透過 `getImage.py` 根據 CSV 中的圖像 URL 逐張下載並儲存至本機快取（`data/image/`），命名規則為 `{anime_id}_{col}.jpg`（例如 `12345_coverImage_medium.jpg`）。每張圖片的下載結果（成功／失敗）皆記錄於 `fetch_log.csv`，供後續流程過濾使用。若 URL 無效或下載失敗，該筆資料會被跳過，不影響整體流程。

---

## 二、前處理：人物偵測與裁切

直接對完整圖像提取特徵，模型容易受背景雜訊干擾而學到偽相關（spurious correlations），如 Related Work 第三節所述。因此，在送入特徵提取模型前，先執行人物偵測與裁切作為前處理步驟。

`yolo_for_image.py` 讀取 `fetch_log.csv` 中下載成功的圖像，流程如下：

1. 將圖像 upscale 至 640px（提升小圖的偵測率）
2. 使用 `dghs-imgutils`（基於 `deepghs/anime_person_detection`）偵測圖中的動漫人物
3. 依信心分數排序，裁切置信度最高的前 N 個人物區域
4. 若未偵測到任何人物，則 fallback 使用整張圖像

裁切結果儲存至 `gen/` 資料夾，同時輸出包含原圖與裁切結果的視覺化 canvas，供人工確認偵測品質。

---

## 三、模型訓練：對比式學習

### 3.1 模型架構

以 Hugging Face `transformers` 載入預訓練的 Swin Transformer（`microsoft/swin-base-patch4-window7-224`）作為視覺 encoder，不附加分類頭，直接取 `pooler_output` 作為 1024 維的圖像嵌入向量。

### 3.2 資料集與資料增強

資料集依 `split_pre_release_effective` 欄位切分為訓練、驗證、測試三組。每張圖像在讀取後先經過 `ResizeWithPad(224)` 統一尺寸（等比縮放後補黑邊，保留完整畫面），再分別套用兩條 transform pipeline：

- **Original（anchor）**：ToTensor → Normalize（ImageNet 均值與標準差）
- **Augmented（positive view）**：隨機裁剪調整大小（p=1.0）、隨機裁切（p=0.3）、顏色扭曲（p=0.8）、高斯模糊（p=0.5）、水平翻轉（p=0.5）、隨機灰階（p=0.2）、ToTensor → Normalize

兩個視角形成一組正樣本對，batch 內其他圖片的 original embedding 自動作為負樣本。

### 3.3 損失函數：InfoNCE

採用 InfoNCE loss 作為訓練目標：

$$L = -\log \frac{\exp(\text{sim}(\mathbf{z}_{\text{aug}}, \mathbf{z}_{\text{orig}}) / \tau)}{\sum_{k=1}^{N} \exp(\text{sim}(\mathbf{z}_{\text{aug}}, \mathbf{z}_k) / \tau)}$$

其中 $\text{sim}(\cdot)$ 為 cosine similarity，溫度參數 $\tau = 0.07$。batch size 設為 64，每張圖片有 63 個負樣本，促使模型學會區分不同圖片的視覺語義，防止 representation collapse。

### 3.4 訓練排程

學習率採用兩階段排程：

1. **Warmup 階段**（前 N epochs）：學習率從 0 線性上升至目標值
2. **Cosine Annealing 階段**：學習率餘弦衰減至趨近 0

### 3.5 模型儲存

| 時機 | 儲存位置 | 格式 |
|------|----------|------|
| val loss 改善時 | `results/{run_id}/best/` | HuggingFace 格式，可直接 `from_pretrained` 載入 |
| 每 N epochs | `results/{run_id}/checkpoint/epoch_{N}.pt` | 含 optimizer 狀態，可續訓 |

驗證指標除 val loss 外，亦記錄 val set 上 original 與 augmented embedding 的平均 cosine similarity，以監測表徵學習的收斂情況。所有訓練指標透過 TensorBoard 視覺化追蹤。

---

## 四、特徵向量輸出

訓練完成後，以 `predictor.py` 對全資料集（test split）進行批次推論，分別對 `coverImage_medium` 與 `bannerImage` 產生 1024 維的嵌入向量，合併後儲存為：

```
data/processed/image_embeddings.parquet
```

輸出欄位：

| 欄位 | 維度 | 說明 |
|------|------|------|
| `idx` | — | AniList 動畫 ID（與原始 CSV 的 `id` 欄位對應） |
| `coverImage_emb` | 1024 | 封面圖嵌入向量 |
| `bannerImage_emb` | 1024 | 橫幅圖嵌入向量 |

下游模型可直接以 `idx` 與原始 CSV 透過 `id` 欄位進行 merge，讀取方式如下：

```python
import numpy as np
import pandas as pd

df = pd.read_parquet("data/processed/image_embeddings.parquet")
cover_emb = np.array(df["coverImage_emb"].tolist())  # shape: (N, 1024)
banner_emb = np.array(df["bannerImage_emb"].tolist()) # shape: (N, 1024)
```
