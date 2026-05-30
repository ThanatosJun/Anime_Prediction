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

---

## ⏳ 待完成

### Step 7：Fusion Model v2 訓練

架構：

```
Image  [batch, 1024] → Image Projection  → [batch, 128] ─┐
Text   [batch, 768]  → Text Projection   → [batch, 128] ─┤
Meta   [batch, 56]   → MetaData Proj     → [batch, 128] ─┤─ concat(512) → MLP → [1]
                       MetaData Proj     → Q [batch,1,128] ─┐
RAG retrieved items                                          │
  [batch,5,10]  → Rag_meta Proj  ─┐                        │
  [batch,5,768] → Rag_text Proj  ─┼─ KV [batch,15,128] → Cross Attention ─┘
  [batch,5,1024]→ Rag_image Proj ─┘
```

建置清單：
- [x] `src_2/fussion_training/meta_encoder.py`（56-dim）
- [x] `src_2/fussion_training/cross_attention.py`（RAGCrossAttention）
- [ ] `src_2/fussion_training/dataset.py`（載入 text/image/meta/RAG embeddings）
- [ ] `src_2/fussion_training/model.py`（FusionMLP v2 + Cross Attention）
- [ ] `src_2/fussion_training/train.py`（訓練主程式）
- [ ] `src_2/fussion_training/evaluate.py`（評估）
- [ ] `src_2/fussion_configs.yaml`（訓練設定）

### Step 8：評估與推論

- [ ] 對 test / holdout_unknown 評估
- [ ] 推論 pipeline（即時 YOLO + Swin + RAG query + FusionMLP）

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
