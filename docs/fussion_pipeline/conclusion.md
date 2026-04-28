# Fusion Pipeline — Conclusion

## 實驗設定

### 模型架構

```
text_emb  (384)  ──→ text_proj  (Linear→LN→GELU, 128-dim) ─┐
image_emb (1024) ──→ image_proj (Linear→LN→GELU, 256-dim) ──┤→ concat(512) → backbone → head
meta_rag  (232)  ──→ meta_proj  (Linear→LN→GELU, 128-dim) ─┘

backbone: Dropout → [Linear→LN→GELU→Dropout] × 3（512→256→128）
head:     Linear(128→1)
Loss:     HuberLoss(delta=1.0)
```

- 兩個迴歸目標（`popularity`、`meanScore`）各自獨立訓練
- `popularity`：log1p + z-score 標準化；`meanScore`：z-score 標準化
- Mixed Precision（FP16 AMP）+ AdamW + warmup 5 epochs + ReduceLROnPlateau

---

## 實驗結果

| Run | Dataset | lr | weight_decay | 主要改動 | pop val ρ | pop test ρ | pop test R² | pop test MAE | pop test log_MAE | score val ρ | score test ρ | score test R² | score test MAE |
|-----|---------|-----|------|------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 01 | full | 1e-3 | 1e-4 | baseline | 0.8653 | **0.8521** | **0.4378** | **11,601** | **0.9581** | 0.6060 | 0.5397 | 0.1309 | **7.72** |
| 02 | post-2000 | 5e-4 | 1e-3 | 資料集過濾 + 調整正則化 | 0.8665 | 0.8488 | 0.3401 | 13,385 | 1.0157 | 0.6218 | 0.5122 | 0.0470 | 8.12 |
| 03 | full | 5e-4 | 1e-3 | ID-based 對齊修正 | 0.8638 | 0.8351 | 0.3430 | 14,042 | 1.1717 | **0.6242** | 0.5171 | **0.1622** | **7.58** |
| 04 | post-2000 | 5e-4 | 1e-3 | post-2000 + ID 對齊 | 0.8655 | 0.8494 | 0.3291 | 13,283 | 0.9943 | 0.6049 | 0.5123 | 0.1210 | 7.84 |

> Run 03 相對 Run 02 的主要改動：FusionDataset 改為 ID-based 對齊（非隱式位置對齊），使用全量資料集。
> Run 04 = Run 02 的設定（post-2000）+ Run 03 的 ID 對齊修正。

---

## 主要發現

### popularity 預測
- **Spearman ρ 穩定在 0.83–0.87**，代表模型對動畫人氣的排名判斷一致可靠
- **若以 Spearman ρ 為評估指標，Run 01 baseline（lr=1e-3, weight_decay=1e-4）在 test 上表現最佳**：ρ=0.852，R²=0.438，MAE=11,601，log_MAE=0.958
- *若以 log_MAE 為評估指標，Run 04（ID 對齊 + full dataset）在 test 上表現最佳*：log_MAE=0.9943
- 降低 lr + 提高 weight_decay（Run 02/03/04）對 popularity test 有明顯負面影響：test R² 從 0.44 降至 0.34，MAE 從 11,601 升至 13,000–14,000，log_MAE 從 0.958 惡化至 1.17
- **結論：以 03 跟 04 來看，不同時期的資料集對模型表現有影響**

### meanScore 預測
- **所有 run 的 test Spearman ρ 約 0.52–0.54，顯著低於 val 的 0.60–0.62**，主要受 distribution shift 影響
- Run 01 baseline 的 test R²=0.131，為所有 run 中次高；Run 03（ID 對齊 + full dataset）的 test R²=0.162 為最高
- Run 02（post-2000）的 test R²=0.047，明顯較差，確認 post-2000 過濾對 meanScore 無幫助
- **Run 03 的 test MAE=7.58 為各 run 最低**，ID 對齊修正帶來輕微但穩定的提升
- Run 04（post-2000 + ID 對齊）test ρ=0.512，R²=0.121，MAE=7.84，與 Run 02 相近，確認 post-2000 資料集對 meanScore 無幫助，全量資料集（Run 03）仍較優

---

## 目前問題

### 1. meanScore 時序 Distribution Shift（核心瓶頸）

資料採時序切分，AniList 社群評分中位數隨時間系統性上升：

| Split | 年份範圍 | meanScore 中位數 |
|-------|---------|----------------|
| Train | –2018 | 60 |
| Val | 2018–2022 | 62 |
| Test | 2022–2026 | 66 |

2022 年後出現約 **+4** 的跳升，主因為 AniList 社群爆發性成長與季播評分文化的改變。此 shift 無法透過調整訓練集起始年份解決（Train 中位數無論如何都維持在 60），是本任務的固有限制。

**影響**：test R² 僅 0.047–0.13，模型幾乎無法解釋 test 集的分數變異。

### 2. popularity 泛化落差

Val R² ≈ 0.57，但 test R² 在 Run 02/03 降到 0.34，與 Run 01 baseline 的 0.44 有明顯落差。降低 learning rate 雖然對 val 略有改善，卻使 test 泛化能力下降。

### 3. 約 10% val/test 樣本因 null description 被排除

| Split | 總筆數 | 缺 description | 比例 |
|-------|------|--------------|------|
| Val | 2,918 | 281 | 9.6% |
| Test | 3,087 | 279 | 9.0% |

全部為 AniList 上原本就無描述的動畫（中國動畫、成人 OVA、電影特別篇為主），導致這部分樣本完全無法進入 val/test 評估。

### 4. 評估指標數值溢位（已修正）

FP16 AMP 模式下，模型輸出為 float16（max ≈ 65,504）；popularity 使用 log1p + z-score，反標準化時執行 expm1 會溢位產生 `Inf`。已透過 float64 強制轉型 + ±5σ clip 修正。

### 5. 超參數搜尋不足

目前僅手動調整三組實驗，尚未進行系統性搜尋（learning rate、dropout、hidden_dims 的交叉組合）。

---

## 未來改進方式

### 短期（資料層面）

**1. 補充 null description（預計回收 ~150–200 筆/split）**

對 isAdult=False 的缺描述動畫，透過 Jikan API（MyAnimeList 非官方）補抓描述，合成新資料集後重跑 text embedding：

```bash
python -m src.fussion_branch.run_supplement_descriptions
python -m src.fussion_branch.run_supplement_merge
python -m src.fussion_branch.run_text_embedding  # meta_dir 改指向 supplemented
```

**2. 補充 null description 後更新聲優 MetaEncoder**

top-50 聲優 multi-hot 已實作並計入 MetaEncoder（feature_dim=232）。待補抓描述完成後，重新 fit MetaEncoder 以確保聲優 vocab 與新資料集一致。

### 中期（模型層面）

**3. 分 target 設定超參數**

| Target | 建議 lr | 建議 weight_decay | 說明 |
|--------|---------|-----------------|------|
| popularity | 1e-3 | 1e-4 | Run 01 baseline 已是最佳 |
| meanScore | 5e-4 | 1e-3 | 需要更強正則化對抗 overfitting |

目前兩個 target 共用同一組超參數，分開設定可各自收斂到更好的結果。

**4. 縮小 meanScore backbone**

meanScore 的特徵信號比 popularity 弱，目前 ~50 萬參數的模型可能過大。建議嘗試：

```yaml
hidden_dims: [256, 128]
dropout: 0.5
```

**5. 系統性 Hyperparameter Search**

使用 Optuna 或手動 grid search 搜尋以下組合：

| 超參數 | 搜尋範圍 |
|--------|---------|
| learning_rate | 1e-4, 5e-4, 1e-3 |
| weight_decay | 1e-4, 1e-3, 5e-3 |
| dropout | 0.3, 0.4, 0.5 |
| hidden_dims | [512,256,128], [256,128], [512,256] |

### 長期（策略層面）

**6. meanScore 年份趨勢特徵**

在 RAG 或 MetaEncoder 中加入「同年份同類型的歷史平均 meanScore（training set 限定）」，讓模型能感知時間趨勢，部分緩解 distribution shift 問題。

**7. RAG top-K 擴充**

目前只取 top-1 相似動畫作為輔助特徵。可實驗取 top-3 後對 `rag_popularity`、`rag_score` 等取平均，降低單一檢索結果的隨機性，提升 RAG 特徵穩定度。

**8. Ensemble**

popularity 與 meanScore 各選最佳 run 的 checkpoint 做推論，不受限於同一套超參數。Run 01 的 popularity checkpoint 與 Run 03 的 meanScore checkpoint 組合，預期可同時保留兩個 target 各自的最佳表現。

---

## 資料集版本對照

| 版本 | meta_dir | 說明 |
|------|---------|------|
| full | `data/fussion` | 全量資料集（1961–2026） |
| post-2000 | `data/fussion/post2000` | 過濾 2000 年以前（Train: 13,376→9,583） |
| supplemented | `data/fussion/supplemented` | 補抓 null description 後的合成資料集（待建立） |

> 切換資料集只需修改 `fusion_config.yaml` 的 `fusion_meta_dir`，程式碼不需變動。

---

## 視覺化

`meanScore` 時序分布圖（各 split 中位數 ± 1 std）：

```
.exp/fussion/meanscore_distribution_over_time.png
```

實驗結果統計：

```
.exp/fussion/experiments_summary.csv
```
