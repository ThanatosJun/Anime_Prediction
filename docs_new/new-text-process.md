# Text Branch 更新說明（origin/Text）

> 對應 branch：`remotes/origin/Text`
> 參考基線：`intfloat/e5-base-v2`，popularity test Spearman = **0.6172**

---

## 更新檔案一覽

| 檔案 | 類型 | 說明 |
|------|------|------|
| `text_preprocessor.py` | 修改 | 新增 `remove_marketing` 選項 |
| `baseline_model.py` | 修改 | 新增 TF-IDF+LSA append、實驗追蹤欄位 |
| `run_text_embedding_pipeline.py` | 修改 | 新增 model preset 系統、`--finetuned-model-path` |
| `configs/embedding_config.yaml` | 修改 | 新增 6 種模型 preset、切換基礎模型為 e5-base |
| `finetune_encoder.py` | **新增** | Encoder fine-tune pipeline |
| `build_quality_ranking_report.py` | **新增** | 實驗比較排名報告產生器 |
| `optimization.md` | **新增** | 4 種優化策略說明 |
| `optimization_result.md` | **新增** | 7 次實驗完整記錄 |

---

## 1. `text_preprocessor.py` — 新增行銷文字清除

新增 `remove_marketing: bool = True` 參數，加入 5 條 regex 規則：

| 規則 | 說明 | 範例 |
|------|------|------|
| `_SOURCE_TAG_RE` | 來源標記 | `(Source: AniList)`, `(Written by MAL Rewrite)` |
| `_HTML_TAG_RE` | HTML 標籤 | `<br>`, `<i>` |
| `_PLATFORM_RE` | 串流平台廣告 | "Streaming on Crunchyroll…" |
| `_BASED_ON_RE` | 原作資訊 | "Based on the manga by…" |
| `_DISC_RE` | 光碟發行備注 | "Blu-ray includes extra scenes." |

> **Exp 01 結論**：marketing cleanup 反而讓結果略差（-0.0098 Spearman）。
> 原因：「Based on the manga」等資訊本身是有效的流行度預測訊號，不應移除。
> 後續所有實驗均使用 `remove_marketing=False`。

---

## 2. `configs/embedding_config.yaml` — 多模型 Preset

新增 `embedding.experiments` 區段，支援 6 種模型快速切換：

```yaml
embedding:
  experiments:
    active_model_key: minilm_l6   # 切換此 key 即可換模型
    models:
      minilm_l6:    sentence-transformers/all-MiniLM-L6-v2   # 384-dim（舊預設）
      bge_small:    BAAI/bge-small-en-v1.5
      e5_small:     intfloat/e5-small-v2
      bge_base:     BAAI/bge-base-en-v1.5
      e5_base:      intfloat/e5-base-v2                      # 768-dim（實驗基線）
      multilingual_minilm: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

基線模型比較：

| 模型 | 維度 | pop. test ρ | meanScore test ρ |
|------|-----:|---:|---:|
| all-MiniLM-L6-v2 | 384 | 0.5408 | 0.2152 |
| **intfloat/e5-base-v2** | **768** | **0.6172** | **0.2525** |

e5-base 明顯優於 MiniLM，成為後續所有實驗的參考基線。

---

## 3. `run_text_embedding_pipeline.py` — Preset 解析與 Fine-tune 路徑

- 新增 `_resolve_embedding_runtime_cfg()`：解析 `active_model_key` 並套用對應模型設定
- 新增 `--finetuned-model-path`：指定 fine-tune 後的 SentenceTransformer artifact
- 新增 `--comparison-csv-name`：每次 run 的結果累積到同一 CSV，方便橫向比較

---

## 4. `baseline_model.py` — TF-IDF+LSA 與實驗追蹤

- 新增 `--tfidf-components N`：在 dense embedding 後 concat N 維 LSA 特徵（TF-IDF → TruncatedSVD）
- 新增 `--experiment-name`、`--embedding-model-key` 等追蹤欄位
- 新增 `_build_compare_row()`：將每次 run 結果匯整成單列，寫入比較 CSV

---

## 5. `finetune_encoder.py` — Encoder Fine-tune Pipeline（新增）

**策略（Exp 04–07 所使用）**：

```
intfloat/e5-base-v2
  → 凍結全部層
  → 解凍 top-N transformer blocks（--unfreeze-layers）
  → 加上 regression head（可選：Linear bottleneck projection）
  → 以 MSE loss 訓練，early stopping on val Spearman
  → 儲存 encoder 為 SentenceTransformer artifact（head 丟棄）
```

主要功能：

| 元件 | 說明 |
|------|------|
| `_freeze_all()` / `_unfreeze_top_n_layers()` | 層凍結控制 |
| `_EncoderWithHead` | Encoder + mean-pool + optional projection + regression head |
| `_build_param_groups()` | 差分學習率：head=`1e-4`，unfrozen layers=`1e-5` |
| `_save_as_sentence_transformer()` | 儲存為 ST 格式（含可選 Dense projection module） |

使用方式：

```bash
# fine-tune（top-2 layers，Run A1）
python -m src.text_branch.finetune_encoder --unfreeze-layers 2 --run-id A1

# 搭配 Linear projection（768→384，Run B2）
python -m src.text_branch.finetune_encoder --unfreeze-layers 2 --projection-dim 384 --run-id B2

# 重新產生 embedding
python -m src.text_branch.run_text_embedding_pipeline \
    --finetuned-model-path artifacts/finetuned_encoder_B2 \
    --output-prefix text_embeddings_B2
```

---

## 6. `build_quality_ranking_report.py` — 實驗排名報告（新增）

讀取 `text_branch_quality_compare.csv`，依各指標排名並輸出 Markdown 報表。

```bash
python -m src.text_branch.build_quality_ranking_report
# → reports/text_branch_quality_ranked.csv
# → reports/text_branch_quality_ranking.md
```

---

## 實驗結果總覽（7 次，均未超越基線）

| # | 實驗 | pop. test ρ | vs 基線 | 結論 |
|---|------|---:|---:|------|
| — | **e5_base 基線** | **0.6172** | — | 參考點 |
| 01 | Marketing cleanup | 0.5310 | −0.0098 | ❌ 訊號損失 |
| 02 | e5_base + LSA-128 | 0.5717 | −0.0455 | ❌ |
| 03 | e5_base + LSA-64 | 0.5648 | −0.0524 | ❌ |
| 04 | Fine-tune top-2 (A1) | 0.5928 | −0.0244 | ❌ |
| 05 | Fine-tune top-3 (A2) | 0.5929 | −0.0243 | ❌ |
| 06 | Frozen + proj-384 (B1) | 0.5774 | −0.0398 | ❌ |
| 07 | Unfreeze top-2 + proj-384 (B2) | 0.5912 | −0.0260 | ❌ |

**B2 特別說明**：pop. RMSE 31557 < 基線 32060（唯一超越基線的指標），meanScore Spearman 0.3016 為所有實驗最佳。但 pop. Spearman 未達晉升門檻，故不採用。

**結論：frozen `intfloat/e5-base-v2`（768-dim）維持最佳 ranking 能力，為目前 Text branch 推薦 embedding model。**

---

## 接入 FusionMLP 的建議

目前 dev 的 FusionMLP 使用 MiniLM 384-dim text embedding（`text_proj: 128`）。
若要切換為 e5-base 768-dim：

1. 修改 `embedding_config.yaml`：`active_model_key: e5_base`
2. 重跑 `run_text_embedding_pipeline.py` 產生新 parquet
3. 修改 `fusion_config.yaml`：`text_proj: 128`（768→128，維持不變）或加大至 `256`（768→256）
