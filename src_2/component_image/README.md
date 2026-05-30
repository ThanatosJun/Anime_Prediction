# component_image

動畫封面圖片的 **推論 pipeline**：讀取本地圖片 → YOLO 偵測角色/臉部 → Swin-B 產生 embedding → 儲存為 parquet。

---

## 檔案一覽

| 檔案 | 用途 |
|------|------|
| `output.py` | **主入口**。`ImageEmbedder` class + `main()` CLI |
| `model.py` | Swin-B 模型載入與 embedding 提取 |
| `YOLO.py` | 動畫人物 / 臉部偵測（imgutils 封裝） |
| `image_process.py` | 圖片預處理：Resize + Padding + Normalize |
| `config.py` | 讀取設定檔 |
| `image_encoder_config.yaml` | **所有設定合一**（路徑、模型、推論模式、YOLO 參數） |

模型權重放在：`src_2/model-image/`

---

## 各檔說明

### `output.py` — 主入口

核心 class `ImageEmbedder`，負責整個推論流程：

```
圖片路徑
  → load_image()           讀圖
  → _get_yolo_crops()      YOLO 偵測，裁出角色/臉部區域（可關閉）
  → _preprocess()          ResizeWithPad(224) + Normalize
  → _forward()             Swin-B forward → pooler_output (1024-dim)
                           或 stage embeddings [128, 256, 512, 1024]
  → 回傳 numpy array
```

主要方法：

| 方法 | 說明 |
|------|------|
| `embed(image_path)` | 單張圖片 → embedding |
| `embed_batch(paths)` | 多張圖片 → embedding list |
| `embed_url(url)` | 從 URL 下載後 embed |
| `embed_dataframe(df, image_dir, col)` | 對整個 DataFrame 批次 embed |
| `save_embeddings(cover, banner, path)` | 將 cover/banner embedding 存成 parquet |

**CLI 使用方式：**
```bash
# 批次處理整個 CSV
python output.py           # image_encoder_config.yaml 中 file_kind: file

# 單張圖片
python output.py path/to/image.jpg   # file_kind: image
```

---

### `model.py` — Swin-B 模型

三個可用功能：

| 函式/Class | 輸入 | 輸出 | 說明 |
|-----------|------|------|------|
| `load_model(config)` | config dict | SwinModel | 從 HuggingFace 載入預訓練模型 |
| `get_embedding(model, pixel_values)` | `(B,3,224,224)` | `(B,1024)` | 最終 pooler_output |
| `get_stage_embeddings(model, pixel_values)` | `(B,3,224,224)` | `[(B,128),(B,256),(B,512),(B,1024)]` | 4 個中間層特徵 |
| `StageProjector(project_dim)` | stage embeddings | `[(B,D)×4]` | 將 4 個 stage 投影到相同維度 |

Swin-B 的 4 個 stage 各自捕捉不同粒度的特徵：
- Stage 0 (128-dim)：線條、局部紋理
- Stage 1 (256-dim)：色塊、局部結構
- Stage 2 (512-dim)：人物部位、光影
- Stage 3 (1024-dim)：整體語義、畫風

---

### `YOLO.py` — 偵測封裝

包裝 `imgutils` 的偵測函式，支援兩種模式：

| 函式 | 模型來源 | 說明 |
|------|---------|------|
| `detect_person(image, ...)` | `deepghs/anime_person_detection` | 偵測動畫人物，回傳 `[(bbox, 'person', confidence), ...]` |
| `detect_faces(image, ...)` | `deepghs/anime_face_detection` | 偵測動畫臉部，回傳 `[(bbox, 'face', confidence), ...]` |

偵測流程（`_get_yolo_crops` 中）：
1. 小圖先 upscale 到 640px（提升偵測率）
2. 依信心分數排序，保留前 N 個框
3. 無偵測結果時 fallback 整張圖

---

### `image_process.py` — 圖片預處理

推論專用，不含 augmentation：

| 功能 | 說明 |
|------|------|
| `ResizeWithPad(size)` | 等比縮放後補黑邊至正方形，保持長寬比 |
| `load_image(path)` | 讀取圖片並轉 RGB，失敗回傳 `None` |
| `get_transform_original(image_size)` | ToTensor + ImageNet Normalize |

---

### `config.py` — 設定載入

```python
from config import load_config, load_yolo_config

config      = load_config()           # 讀整份 image_encoder_config.yaml
yolo_config = load_yolo_config()      # 讀 yolo_detection 區段
```

---

### `image_encoder_config.yaml` — 設定檔

主要需要調整的項目：

```yaml
data:
  image_dir: src_2/data/image/train_image   # 圖片所在目錄（依 split 選擇）

model:
  stage: false      # false → pooler_output (1024-dim)
                    # true  → 4 個 stage embeddings

inference:
  file_kind: file   # file(批次 CSV) | image(單張)
  use_yolo: true    # 是否啟用 YOLO 偵測裁切

yolo_detection:
  detect_mode: face  # person | face | both
```

---

## 快速使用範例

```python
from config import load_config
from output import ImageEmbedder

config   = load_config('image_encoder_config.yaml')
embedder = ImageEmbedder(model_path='src_2/model-image/01/best', config=config)

# 單張圖片
emb = embedder.embed('src_2/data/image/train_image/100003_coverImage_medium.jpg')
# emb: np.ndarray (1024,)

# 整個 DataFrame 批次
import pandas as pd
df = pd.read_csv('data/processed/train.csv')
cover_embs = embedder.embed_dataframe(df, 'src_2/data/image/train_image', 'coverImage_medium')
# cover_embs: {anime_id: [float × 1024], ...}
```
