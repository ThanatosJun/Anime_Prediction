# RAG Embedding Exchange：舊 → 新

| | 舊（src/fussion_branch/RAG） | 新（src_2/RAG） |
|--|--|--|
| Text model | `all-MiniLM-L6-v2` | `intfloat/e5-base-v2` |
| Text dim | **384** | **768** |
| Image model | Swin-B（fine-tuned） | Swin-B（fine-tuned） |
| Image dim | 1024 | 1024（不變） |
| Collection name | `anime_rag` | `anime_rag_v2` |
| Meta data 來源 | `data/fussion/` | `src_2/data/dataset/` |
| Embedding 儲存 | `src/fussion_branch/embedding/` | `src_2/embedding/` |
| RAG return 位置 | `src/fussion_branch/RAG/return/` | `src_2/RAG/return/` |
| top_k_ids 用途 | GNN 訓練時 embedding lookup | Cross Attention（5 個 retrieved） |

---

## 檔案對應

| 舊檔案 | 新檔案 | 變更 |
|--------|--------|------|
| `sparse_encoder.py` | `sparse_encoder.py` | 無變更（直接複製） |
| `rag_builder.py` | `rag_builder.py` | 路徑更新，import 改為相對 |
| `rag_query.py` | `rag_query.py` | 路徑更新，import 改為相對 |
| — | `run_build_embeddings.py` | **新增**：一次生成 text + image embeddings |
| `rag_config.yaml`（在 configs/） | `rag_config.yaml`（在 RAG/） | 路徑和維度更新 |

---

## Embedding 格式

### Text（`emb_` 前綴，格式不變）

| | 舊 | 新 |
|--|--|--|
| 欄位 | `id, emb_000 … emb_383` | `id, emb_000 … emb_767` |
| 維度 | 384 | 768 |
| 產生方式 | `src/text_branch/` | `src_2/component_text/` |

### Image（`img_` 前綴，格式不變）

| | 舊 | 新 |
|--|--|--|
| 欄位 | `id, img_0 … img_1023` | `id, img_0 … img_1023` |
| 維度 | 1024 | 1024 |
| 圖片來源 | `coverImage_medium` | `coverImage_extraLarge` |

---

## 執行順序

```bash
# Step 1：生成 embeddings（text + image，所有 splits）
cd src_2/RAG
python run_build_embeddings.py --splits train val test holdout_unknown

# 或分開執行
python run_build_embeddings.py --splits train --modality text
python run_build_embeddings.py --splits train --modality image

# Step 2：建立 Qdrant collection（僅 training set）
python rag_builder.py

# Step 3：查詢所有 splits
python rag_query.py --splits train val test holdout_unknown
```

---

## 注意事項

1. **text_emb_dim 必須與模型一致**：`rag_config.yaml` 中 `text_emb_dim: 768`，若更換模型需同步修改
2. **collection_name 已更名**：`anime_rag` → `anime_rag_v2`，避免與舊 collection 衝突
3. **image 圖片來源升級**：舊版使用 `coverImage_medium`（低解析度），新版使用 `coverImage_extraLarge`
4. **top_k_ids 用途改變**：舊版 top-5 IDs 供 GNN 訓練時做 embedding lookup；新版 top-5 直接在 Cross Attention 中使用（`[batch, 5, 768]` text + `[batch, 5, 1024]` image）
