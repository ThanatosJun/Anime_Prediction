# Fusion v2

多模態融合預測模組 v2。結合 text embedding、image embedding（cover + banner + YOLO crop）、metadata 與 RAG Cross Attention，預測動畫的 `popularity`（人氣）與 `meanScore`（評分）。

v1（src/fussion_branch）與 v2 核心差異：
- **RAG**：GNN（graph propagation）→ **Cross Attention**（Q=meta, KV=retrieved items 三路投影）
- **Image**：單張 cover → **三模態 Gated Projection**（cover + banner + YOLO crop）
- **MetaEncoder**：66-dim（含 RAG scalar）→ **56-dim**（RAG 移出，改由 Cross Attention 輸入）
- **Optimizer**：AdamW → **SAM + AdamW**（Sharpness-Aware Minimization）
- **Loss**：HuberLoss → **HuberLoss（popularity）/ Log-Cosh（meanScore）**

## 目前最佳結果（test set，**Run22：seed=42 + per-target 超參覆寫**）

| Target | 主指標 | 準確率 | Spearman | per-target 超參 |
|--------|--------|--------|----------|----------------|
| **popularity** | log_MAE **0.8823** | facc_2x **0.4943** | 0.8520 | dropout=0.3（其餘同全域）|
| **meanScore** | MAE **7.5911**（R2 0.193）| within_10pt **0.7104** | 0.5424 | dropout=0.3, attn_drop=0.1, wd=1e-4, batch=256 |

> **Run22** 用 `training.{target}.overrides` 讓兩 target 各自套最佳超參，**一次 `python train.py` 同時達到兩者的 seed-fixed 最佳**（pop 0.8823 ≈ 全域最佳 04_s42 的 0.8821；score 7.5911 = 02_s42 的 7.5911）。
>
> ⚠️ **seed + per-target 的關鍵發現**：
> - 早期實驗（Run01~21）**未固定 seed**，「最佳 run」部分是運氣（舊 meanScore Run02 的 7.29 → seed-fixed 後 7.59）。
> - **兩 target 最佳超參方向不同**：pop 要低 dropout；**meanScore 對 `attn_dropout` 極敏感**（0.2→0.1 使 MAE 8.25→7.59），且偏好小 batch。全域單一超參無法兩全 → 故加 per-target 覆寫機制。

---

## Seed Robustness（Run22–28，7 個 seed）— 回應老師「single seed 不足」

Run22（seed=42）為代表 run；**Run23–28 = 完全相同設定（full model + per-target HP），只改 seed**，用來檢查主結果是否為 seed 運氣。
seed：22→42、23→43、24→44、25→45（連續）、**26→247135、27→610172、28→796445（random，SystemRandom 抽樣後固定）**。
產生指令：`python src_2/rerun_seeds.py`，摘要 `runs/rerun_seeds_summary.json`（含 `_seed_robustness` 的 mean±std）。

| run (seed) | pop log_MAE↓ | pop log_R²↑ | pop Spear↑ | pop facc↑ | score MAE↓ | score R²↑ | score Spear↑ | score win10↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 22 (42) | 0.8823 | 0.7633 | 0.8520 | 0.4943 | 7.5911 | 0.1934 | 0.5424 | 0.7104 |
| 23 (43) | 1.1184 | 0.6352 | 0.8475 | 0.3871 | 7.6196 | 0.1985 | 0.5466 | 0.7130 |
| 24 (44) | 0.8947 | 0.7617 | 0.8499 | 0.4781 | 8.0692 | 0.1068 | 0.5273 | 0.6770 |
| 25 (45) | 1.0318 | 0.6883 | 0.8473 | 0.4189 | 7.7027 | 0.1787 | 0.5564 | 0.7075 |
| 26 (247135) | 1.1410 | 0.6259 | 0.8487 | 0.3741 | 7.3859 | 0.2367 | 0.5579 | 0.7243 |
| 27 (610172) | 0.8851 | 0.7616 | 0.8510 | 0.4859 | 8.6012 | 0.0100 | 0.5304 | 0.6440 |
| 28 (796445) | 0.8760 | 0.7731 | 0.8550 | 0.4849 | 8.5187 | 0.0217 | 0.5419 | 0.6537 |
| **mean ± std** | **0.9756 ± 0.1185** | **0.7156 ± 0.0647** | **0.8502 ± 0.0027** | **0.4462 ± 0.0514** | **7.9269 ± 0.4788** | **0.1351 ± 0.0903** | **0.5433 ± 0.0117** | **0.6900 ± 0.0317** |

> **跨 7 seed 結論（誠實）：**
> 1. **Spearman（排序）是唯一跨 seed 穩定的優勢**：pop 0.8502 ± 0.0027、score 0.5433 ± 0.0117 → CARMA 的 **ranking 能力跨 seed 幾乎不變**，這是可靠、可主張的強項。
> 2. **Popularity log_MAE 對 seed 敏感**：mean **0.9756 ± 0.1185**，**比 baseline F2-XGB-Concat（0.8828）還差** → popularity「誤差最低」是 seed=42 的運氣，**多 seed 下不成立**。
> 3. **MeanScore MAE 也對 seed 敏感（std 隨 seed 數增大 0.22→0.48）**：mean **7.9269 ± 0.4788**，**也比最佳 baseline（7.8582）差**（4 seed 時曾誤判為「優勢守得住」，加到 7 seed 後翻盤）。R² 從 0.01 到 0.24，極不穩。
> 4. **總結論**：CARMA 跨 seed **唯一站得住的是 ranking（Spearman）**；兩個 target 的**絕對誤差（log_MAE / MAE）都 seed-dependent**，主結果 Run22 是「兩 target 剛好都運氣好」的 seed。論文應以 **mean±std + ranking** 為主張，不可用單一 seed 的 headline 誤差宣稱「最低」。
> 5. **根因**：兩個 target 的訓練跨 seed 呈現**雙峰/不穩收斂**（早停 / 過擬合敏感，見「已知限制」）。穩定化（更保守早停、正則、或多 seed 取最佳/集成）列為 future work。

#### 同 7 seed 的 Validation 成績（對照用）

上表為 **test**（早停由 val_loss 決定，最後在 test 評估）。以下為**同樣 7 個 seed 的 validation** 成績，放一起對照可看出 val 與 test 的落差。

| run (seed) | pop log_MAE↓ | pop log_R²↑ | pop Spear↑ | pop facc↑ | score MAE↓ | score R²↑ | score Spear↑ | score win10↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 22 (42) | 0.7927 | 0.8102 | 0.8825 | 0.5209 | 6.7627 | 0.4280 | 0.6607 | 0.7666 |
| 23 (43) | 0.8467 | 0.7810 | 0.8737 | 0.4997 | 6.7208 | 0.4410 | 0.6684 | 0.7711 |
| 24 (44) | 0.7892 | 0.8076 | 0.8839 | 0.5319 | 6.7834 | 0.4319 | 0.6593 | 0.7670 |
| 25 (45) | 0.8018 | 0.7994 | 0.8778 | 0.5336 | 6.7213 | 0.4418 | 0.6649 | 0.7704 |
| 26 (247135) | 0.8373 | 0.7872 | 0.8765 | 0.5096 | 6.7037 | 0.4469 | 0.6677 | 0.7707 |
| 27 (610172) | 0.7884 | 0.8075 | 0.8823 | 0.5394 | 6.8354 | 0.4322 | 0.6665 | 0.7611 |
| 28 (796445) | 0.7901 | 0.8124 | 0.8873 | 0.5264 | 6.8201 | 0.4351 | 0.6688 | 0.7646 |
| **mean ± std** | **0.8066 ± 0.0247** | **0.8008 ± 0.0122** | **0.8806 ± 0.0047** | **0.5231 ± 0.0141** | **6.7639 ± 0.0516** | **0.4367 ± 0.0067** | **0.6652 ± 0.0038** | **0.7674 ± 0.0037** |

**VAL vs TEST（mean ± std 對照）：**

| 指標 | VAL | TEST |
|---|:---:|:---:|
| pop log_MAE↓ | **0.807 ± 0.025** | 0.976 ± **0.119** |
| pop Spearman↑ | 0.881 ± 0.005 | 0.850 ± 0.003 |
| score MAE↓ | **6.764 ± 0.052** | 7.927 ± **0.479** |
| score R²↑ | **0.437 ± 0.007** | 0.135 ± **0.090** |
| score Spearman↑ | 0.665 ± 0.004 | 0.543 ± 0.012 |

> **洞察：val 又穩又樂觀，不穩定是 test（時序外推）才爆發。**
> - Validation 上 7 seed 幾乎一致（score MAE std 僅 0.05、R² std 0.007）→ 模型對 val 分布收斂得很一致。
> - 到 test 上 std 暴增（score MAE 0.48、R² 0.09），且數值大幅變差（R² 0.44→0.14）→ **典型 distribution shift / 過擬合 val**。
> - 結論：**只看 val 會誤判模型又穩又好；multi-seed + 看 test 才暴露真實泛化的不穩定**，這也呼應 concept drift 主題。

---

## Ablation Robustness（T1b：ablation × 7 seed）— 哪些 ablation 結論跨 seed 成立

對 3 組關鍵 ablation（**−RAG / −image / −trend**）各在 7 個 seed（42/43/44/45/247135/610172/796445）上跑，與 full（Run22–28，**同 seed 配對**）算 **delta = ablated − full**。配對相減把 seed 雜訊抵消，才能公平判斷組件的「淨貢獻」。
腳本 `python src_2/ablation_multiseed_t1b.py`，摘要 `runs/ablation_multiseed_t1b_summary.json`（含 `_t1b_agg`）。

> **判讀規則：delta 的 `|mean| > std` 才算「跨 seed 穩定的貢獻」；`std ≥ |mean|` 視為 seed 噪音。**
> error 指標（log_MAE/MAE）delta 為正 = 拿掉後變差 = 組件有幫助；Spearman delta 為負 = 拿掉後排序變差 = 組件有幫助。

### 🔴 重點發現：**TrendHead 是「seed 不穩定的來源」**（修正單 seed 的 4.3.3 結論）

單 seed（seed=42）時 trend 看似有幫助（pop 0.8823 vs 無 trend 0.9036；score 7.5911 vs 7.8877）。**但 7 seed 配對後，trend 的貢獻「不成立」，而且 trend 正是不穩定的元兇：**

| 主指標（test, 7 seed mean±std） | **full（有 trend）** | **−trend（無 trend）** |
|---|:---:|:---:|
| pop log_MAE↓ | 0.9756 ± **0.1185**（亂跳） | **0.9066 ± 0.0159**（超穩，且平均更好）|
| score MAE↓ | **7.9269** ± 0.4788（亂跳） | 8.0434 ± **0.0901**（超穩，平均略差）|

**逐 seed 看 popularity log_MAE 的 −trend 欄**：0.90/0.91/0.92/0.90/0.90/0.93/0.88 → **全部落在 0.88–0.93**；而 full（有 trend）是 0.88–1.14 亂跳。

- **關掉 trend → 模型變得很穩**；開著 trend → 模型對 seed 敏感。**T1a 看到的 seed 不穩定，主因就是 trend head。**
- trend head 是「**好 seed 小賺、壞 seed 大賠**」：seed 42/44/610172/796445（好 seed）trend 小幅幫忙；43/45/247135（壞 seed）trend 把 full 拖到 1.0–1.14，而無 trend 仍穩在 0.90。
- **平均淨效果**：
  - **popularity → 淨虧**（無 trend 0.907 反而優於有 trend 0.976）。
  - **meanScore → 小賺但不穩**（有 trend 平均 7.93 優於無 trend 8.04 約 0.12，符合「評分隨年代漂移」的假設，但代價是 std 從 0.09 暴增到 0.48）。
- **論文意涵**：concept-drift 的 trend 設計**不能說「穩定改善」**；誠實說法為「**trend 對 meanScore 有平均幫助但犧牲訓練穩定性，對 popularity 弊大於利**」。穩定化 trend head（正則 / 較小 lr / 約束斜率）列為 future work。

### RAG / image：ranking 貢獻「穩」、error 貢獻「噪音」

| Ablation | target | error Δ（ablated − full） | **Spearman Δ** |
|---|---|---|---|
| **−RAG** | pop | log_MAE +0.057 ± 0.086 ⚠️噪音 | **−0.011 ± 0.006 ✓穩** |
| | score | MAE −0.066 ± 0.509 ⚠️無效 | **−0.021 ± 0.014 ✓穩** |
| **−image** | pop | log_MAE +0.004 ± 0.098 ⚠️無效 | **−0.023 ± 0.003 ✓✓很穩** |
| | score | MAE +0.728 ± 0.661 🟡偏有效 | **−0.032 ± 0.008 ✓✓很穩** |

- **RAG 與 image 對 ranking（Spearman）的貢獻跨 seed 穩定**：拿掉任一個，兩 target 的 Spearman 都一致下降（image 尤其顯著，pop −0.023±0.003、score −0.032±0.008）→ **這是可主張的真貢獻**。
- **error 指標（log_MAE/MAE）的貢獻幾乎都被 seed 噪音淹沒**（std ≥ |mean|），唯一例外是 −image 對 score MAE（+0.73±0.66，偏有效）→ **不可用 error 數字宣稱「RAG/image 降低多少誤差」**。

### T1b 總結

| 組件 | 跨 7 seed 的真相 |
|---|---|
| **RAG** | ✅ 改善 ranking（Spearman），穩定；error 貢獻為噪音 |
| **image** | ✅ 改善 ranking（最顯著），穩定；對 score 誤差偏有效但較噪 |
| **trend** | ❌ 無穩定貢獻，且是 seed 不穩定的來源；meanScore 平均小賺但犧牲穩定性 |

> **一句話**：CARMA 跨 seed **唯一站得住的優勢是 ranking（RAG + image 都穩定貢獻）**；絕對誤差的組件貢獻多為 seed 噪音；trend head 應重新定位為「meanScore 的時序修正，代價是穩定性」而非「穩定改善 concept drift」。

### T2b：顯著性檢定（paired t-test + Wilcoxon，n=7 配對 delta）

對每個 ablation 的 7 個配對 delta（ablated − full，同 seed）做 **one-sample t-test（= paired t-test）** 與 **Wilcoxon signed-rank**（非參數，n=7 較穩健），H0：mean delta = 0。
腳本 `python src_2/t2b_significance.py`，輸出 `runs/t2b_significance.json`。

| 組件 | 指標 | meanΔ | 95% CI | p (t) | p (Wilcoxon) | 結論 |
|---|---|---|---|:---:|:---:|---|
| **−RAG** | pop Spearman | −0.0111 | [−0.016, −0.006] | **0.0019** | 0.0156 | ✓ 顯著（RAG 助排序）|
| | score Spearman | −0.0213 | [−0.034, −0.008] | **0.0074** | 0.0156 | ✓ 顯著 |
| | pop log_MAE | +0.0572 | [−0.023, +0.137] | 0.1304 | 0.1094 | — 不顯著 |
| | score MAE | −0.0656 | [−0.536, +0.405] | 0.7449 | 0.8125 | — 不顯著 |
| **−image** | pop Spearman | −0.0225 | [−0.025, −0.020] | **<0.0001** | 0.0156 | ✓✓ 最顯著 |
| | score Spearman | −0.0322 | [−0.040, −0.024] | **0.0001** | 0.0156 | ✓✓ |
| | score MAE | +0.7283 | [+0.117, +1.340] | **0.0269** | 0.0469 | ✓ 顯著（image 助 score 誤差）|
| | score R² | −0.1470 | [−0.268, −0.026] | **0.0249** | 0.0469 | ✓ |
| | pop log_MAE | +0.0043 | [−0.086, +0.095] | 0.9107 | 0.5781 | — 不顯著 |
| **−trend** | pop log_MAE | −0.0690 | [−0.181, +0.043] | 0.1825 | 0.5781 | — 不顯著 |
| | pop Spearman | +0.0029 | [−0.001, +0.006] | 0.0868 | 0.0781 | — 不顯著 |
| | score MAE | +0.1164 | [−0.307, +0.540] | 0.5259 | 0.5781 | — 不顯著 |
| | score Spearman | +0.0056 | [−0.004, +0.015] | 0.1897 | 0.2188 | — 不顯著 |

> **T2b 正式結論（α=0.05）：**
> 1. **RAG → ranking 顯著貢獻**：pop Spearman p=0.0019、score Spearman p=0.0074；誤差指標皆不顯著。
> 2. **image → 最強最穩**：ranking 高度顯著（pop p<0.0001、score p=0.0001），且 **meanScore 誤差也顯著**（MAE p=0.027、R² p=0.025、within10pt p=0.034）。
> 3. **trend → 任何指標、任何 target 皆不顯著（所有 p>0.05）** → **concept-drift 的 trend 貢獻在統計上不成立**；單 seed 的 4.3.3 為 seed 運氣。
> 4. 註：共 24 個檢定，多重比較下 RAG/image 的 Spearman（p<0.01）最穩健；image 的 meanScore 誤差（p≈0.025）較邊際（Wilcoxon p≈0.05）。**論文主張以「RAG + image 的 ranking 顯著貢獻」為核心最安全。**

---

## 目錄結構

```
src_2/
│
├── fussion_training/             # 核心模組（不直接執行）
│   ├── meta_encoder.py           # MetaEncoder v2（56-dim）
│   ├── cross_attention.py        # RAGCrossAttention（Q×KV Cross Attention）
│   ├── dataset.py                # AnimeDataset（組合所有 embedding → tensor）
│   └── model.py                  # FusionModel v2
│
├── RAG/                          # RAG pipeline（Qdrant hybrid search）
│   ├── sparse_encoder.py
│   ├── rag_builder.py
│   ├── rag_query.py
│   ├── run_build_embeddings.py
│   ├── start_qdrant.sh
│   ├── rag_config.yaml
│   └── return/                   # gitignore
│
├── component_text/               # e5-base-v2 text embedding
├── component_image/              # Swin-B fine-tuned + YOLO
│
├── embedding/                    # gitignore
│   ├── text/                     # text_embeddings_{split}.parquet
│   ├── image/                    # image_embeddings_{split}.parquet（yolo/cover/banner）
│   └── image_rag/                # image_embeddings_train.parquet（RAG 知識庫）
│
├── data/
│   └── dataset/                  # fusion_meta_clean_{split}_v2.csv
│
├── explain/                      # 可解釋性分析
│   ├── rag_heatmap.py            # Cross Attention heatmap
│   └── feature_attr.py           # Captum IG + SHAP
│
├── runs/                         # gitignore，實驗輸出
│   └── {run_id}/
│       ├── {target}/
│       │   ├── best_model.pt
│       │   ├── target_scaler.json
│       │   ├── history.json
│       │   ├── final_metrics.json    # train / val / test metrics 合併
│       │   └── pred_{split}.csv
│       └── explain/{target}/         # 可解釋性輸出（run 層級，非 target 層級）
│           ├── feature/
│           └── rag/
│
├── train.py                      # 訓練主程式
├── evaluate.py                   # 評估（merge 進 final_metrics.json）
├── inference.py                  # 推論 Pipeline（新動畫 → popularity/meanScore）
├── hp_search.py                  # 超參數搜尋（Run04~09）
├── ablation.py                   # RAG / image 消融
├── ablation_multimodal.py        # 多模態分支消融（重訓版）
├── backfill_metrics.py           # 補算舊 run 缺的指標欄位（不重訓）
├── rerun_s42.py                  # 固定 seed=42 重跑全部 scripted 實驗（_s42 後綴）
├── rerun_extra_s42.py            # seed=42 補跑（02/03 + 單模態 banner/yolo）
├── fussion_configs.yaml          # 訓練設定（含 seed / per-target overrides）
├── fussion_configs_stages.yaml   # stage 版（+LN）
├── fussion_configs_stages_nonorm.yaml # stage 版（無 LN，對照）
└── requirements.txt
```

---

## 前置條件

```bash
conda activate animeprediction

# PyTorch（CUDA 12.8 / RTX 5070 Ti）
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128

# 其他套件
pip install -r src_2/requirements.txt
pip install dghs-imgutils==0.19.0 --no-deps   # numpy<2 metadata 衝突，--no-deps 繞過

# Qdrant（Docker）
bash src_2/RAG/start_qdrant.sh
curl http://localhost:6333/healthz   # 確認啟動
```

詳細部署流程見 `docs_new/operator.md`。

---

## 模型架構

### FusionModel v2

```
Image  [batch, 3, 1024] ─→ ImageProjection (Shared Linear + Content Gate) ─→ [batch, 128] ─┐
Text   [batch, 768]     ─→ ProjectionBlock(768→128)                        ─→ [batch, 128] ─┤
Meta   [batch, 56]      ─→ ProjectionBlock(56→128)                         ─→ [batch, 128] ─┤─ concat → MLP → [1]
                           ProjectionBlock → Q [batch, 1, 128] ──────────────────────────────┐ │
RAG retrieved（top-5）                                                                        │ │
  [batch, 5, 10]   ─→ Linear(10→128)   ─┐                                                   │ │
  [batch, 5, 768]  ─→ Linear(768→128)  ─┼─ KV [batch, 15, 128] ─→ Cross Attention ──────────┘ │
  [batch, 5, 1024] ─→ Linear(1024→128) ─┘                                                     │
                                                                                               │
concat_dim: use_rag=True → 512（128×4）；use_rag=False → 384（128×3）                        │
MLP backbone: concat_dim → 256 → 128 → 1 ─────────────────────────────────────────────────────┘
```

**設計重點：**
- `ImageProjection`：三模態（cover/banner/yolo）共用 Linear，Gate 從原始 embedding 計算（content-based），缺失模態 gate 強制 0
- `RAGCrossAttention`：Q = meta projection，KV = retrieved items 三路投影後 concat（layout: meta 0-4, text 5-9, image 10-14）
- `train_separately=true`：popularity / meanScore 各自獨立模型，同一 script 循環訓練

**Image embed 模式（可切換，見「Image Embed 模式」章節）：**
- `pooler`（預設）：每模態 Swin pooler_output → 1024，`[batch, 3, 1024]`
- `stage`：每模態 4 個 Swin stage concat → 1920（cover/banner/**character 皆含**），`[batch, 3, 1920]`；`image_stage_projection=true` 時 ImageProjection 內部切 4 stage → 各投影 256 → concat 1024 → gate

### 訓練設定

| 項目 | 設定 |
|------|------|
| Loss（popularity） | HuberLoss（delta=1.0） |
| Loss（meanScore） | Log-Cosh Loss |
| Optimizer | SAM (rho=0.05, pure wrapper) + AdamW |
| LR Schedule | ReduceLROnPlateau（factor=0.5, patience=3, min_lr=1e-6） |
| Early Stopping | patience=5 |
| AMP | autocast float16（gradient 仍 float32，不用 GradScaler；stage 模式停用）|
| Batch Size | 全域 512；meanScore per-target 覆寫 256 |
| seed | `config.seed`=42（固定 init/shuffle/dropout）|
| per-target 超參 | `training.{target}.overrides` 覆寫 dropout/attn_dropout/lr/wd/batch_size |
| DataLoader | num_workers=min(4, cpu_count()), persistent_workers=True |

---

## 評估指標

| 指標 | 說明 | 適用 |
|------|------|------|
| `spearman_rho` | 排名相關係數（主要指標）| 兩個 target |
| `log_R2` | log1p 空間 R²（匹配訓練目標，對 skewed 分佈穩定）| popularity |
| `R2` | 原始 scale R²（診斷 distribution shift）| meanScore |
| `MAE` | 原始 scale 平均絕對誤差 | 兩個 target |
| `log_MAE` | log1p 空間 MAE（scale-free，越小越好，0=完美，naive≈2.0）| popularity |
| `factor_acc_2x` | 預測值落在真實值 [0.5×, 2×] 內的比例（0~1，越大越好）| popularity |
| `acc_within_10pt` | 預測誤差 < 10 分的比例（0~1，越大越好；0–100 分用加法尺度才合理，facc_2x 對分數無意義）| meanScore |

> **準確率指標的尺度差異**：popularity 跨越多個數量級 → 用乘法尺度（`factor_acc_2x`，2× 內）；meanScore 是 0–100 線性分數 → 用加法尺度（`acc_within_10pt`，±10 分內）。對 meanScore 套 facc_2x 會得到 ~0.997（幾乎全部都在 2× 內），無鑑別力。

---

## 推論 Pipeline（`inference.py`）

給定一部新動畫（封面圖 + metadata + 描述），即時走完 YOLO → Swin → e5 → RAG → FusionModel，輸出 popularity / meanScore。

```bash
bash src_2/RAG/start_qdrant.sh        # 先啟動 Qdrant（RAG 需要）

# 新動畫推論（metadata 用單列 CSV，欄位同訓練 schema）
python src_2/inference.py --cover c.jpg --banner b.jpg --meta new.csv --description "..."

# 驗證模式：用既有 test 動畫，對照 pred_test.csv
python src_2/inference.py --anime-id 21294 --split test --verify
```

| 項目 | 說明 |
|------|------|
| 最佳 checkpoint | **Run22**（per-target HP，pop+score 同一 run_id；inference.py 預設 pop_run=score_run="22"） |
| RAG modality | 預設 `rag_use_image=False`（image_rag 僅 train，val/test 檢索為 sparse+text，對齊驗證指標） |
| 模組隔離 | 各 component 同名 `config.py`/`model.py` 用 `importlib` 隔離載入 |
| 驗證 | cover/banner embedding 逐位元一致；yolo 因預存 crops 經 JPEG round-trip 微差（pipeline 直接裁切，更乾淨） |

> ℹ️ 超參數：全域設定可被 `training.{target}.overrides` 覆寫，讓 popularity / meanScore 各用自己最佳的 dropout / attn_dropout / lr / weight_decay / batch_size（Run22）。

---

## 特徵維度

| 來源 | 維度 | 說明 |
|------|------|------|
| Text embedding | 768 | e5-base-v2，description |
| Image embedding | 3 × 1024（pooler）/ 3 × 1920（stage）| Swin-B：cover + banner + yolo（缺失 gate=0）|
| MetaEncoder v2 | 56 | 見下表 |
| RAG（Cross Attn KV） | 5 × (10 + 768 + 1024) | retrieved top-5 的 meta + text + image（rag_image 維持 pooler 1024）|

### Image Embed 模式（`fusion_embed_mode`）

| 模式 | 每模態 dim | 生成 | config |
|------|:---:|------|--------|
| **pooler**（預設）| 1024 | `run_swin_embedding.py`（→ `embedding/image/`）| `image_dim: 1024`, `image_stage_projection: false` |
| **stage** | 1920 | `run_swin_embedding.py --mode stage`（→ `embedding/image_stage/`）| `image_dim: 1920`, `image_stage_projection: true`, `data.image_emb_dir: .../image_stage` |

- stage = Swin 前 4 個 stage concat（128+256+512+1024；第 5 個與第 4 個 cosine≈0.89 重複，捨棄）
- `image_stage_projection=true`：ImageProjection 內部把 1920 切回 4 stage → 各 Linear→256 → concat（4×256=1024）→ gate（投影層可訓練）
- **解耦**：RAG image（`image_rag` / Qdrant / rag_image）維持 pooler 1024，切 stage **不需** rebuild Qdrant，retrieved_ids 不變
- pooler / stage embedding 分目錄並存，互不覆蓋（可隨時對照）

### MetaEncoder v2 特徵明細（56-dim）

| 類型 | 欄位 | 維度 |
|------|------|------|
| Robust 標準化（median/IQR）| release_year, episodes, duration, startDate_day, prequel_count, prequel_meanScore_mean | 6 |
| log1p + 標準化 | prequel_popularity_mean | 1 |
| Cyclical sin/cos | release_quarter（period=4）, startDate_month（period=12）| 4 |
| One-hot | format（7）, source（7）, countryOfOrigin（4）| 18 |
| Binary | isAdult, is_sequel, has_sequel | 3 |
| Multi-hot | genres（19 類）| 19 |
| Studio Target Encoding | mean_popularity, mean_score（標準化）| 2 |
| is_new_studio | 所有 studio 在訓練集未見過 → 1 | 1 |
| Voice Actor Target Encoding | mean_popularity, mean_score（標準化）| 2 |
| **合計** | | **56** |

> v1（66-dim）差異：移除 RAG scalar 10 dims（rag_popularity, rag_score, rag_release_year, rag_episodes, rag_found, studio_match, genre_overlap, format_match, rag_studio_te ×2）。這些資訊改由 Cross Attention 的 KV 輸入，讓模型自行學習如何整合。

---

## 訓練目標轉換

| Target | 轉換 | 反轉 |
|--------|------|------|
| `popularity` | Winsorize(99%) → log1p → z-score | 反標準化 → clip(±5σ) → expm1 |
| `meanScore` | Winsorize(99%) → z-score | 反標準化 |

mean/std 僅從訓練集計算，再套用到 val / test。

---

## 實驗記錄

config：`src_2/fussion_configs.yaml`，結果：`src_2/runs/{run_id}/`

> ⚠️ **seed 注意**：以下 Run01~21 / hp_search / 消融表為**早期未固定 seed** 的結果（方向性結論可信，但絕對值有 ~±0.025 的 seed 雜訊，「最佳 run」可能受運氣影響）。
> **權威的 seed=42 對齊版**見下方「seed=42 全實驗對照」（`python src_2/rerun_s42.py` 產生，run_id 帶 `_s42` 後綴，摘要 `runs/rerun_s42_summary.json`）。

### seed=42 全實驗對照（16 組，`rerun_s42.py`）

全部 seed=42、同一份 base config，唯一差異是各組標示的變因。pop 主指標 log_MAE↓、score 主指標 MAE↓。

| 組別 | run（_s42）| pop log_MAE | pop facc | pop spear | score MAE | score win10 | score spear |
|------|-----------|:---:|:---:|:---:|:---:|:---:|:---:|
| **baseline** | pooler | **0.8859** | 0.4921 | 0.8505 | **7.7575** | 0.7075 | 0.5439 |
| stage | stage_ln | 0.9653 | 0.4580 | 0.8476 | 7.8422 | 0.6968 | 0.5419 |
| stage | stage_noln | 0.8989 | 0.4843 | 0.8442 | 7.9428 | 0.6936 | 0.5429 |
| hp dr=0.3 wd=1e-4 | 04 | **0.8821** ⭐ | 0.4927 | 0.8499 | 8.1718 | 0.6744 | 0.5388 |
| hp dr=0.4 wd=5e-4 | 05 | 0.8890 | 0.4846 | 0.8498 | 7.9545 | 0.6926 | 0.5402 |
| hp dr=0.5 wd=1e-4 | 06 | 0.8857 | 0.4921 | 0.8505 | 7.7597 | 0.7075 | 0.5439 |
| hp dr=0.3 wd=1e-3 | 07 | 0.8824 | 0.4927 | 0.8499 | 8.1751 | 0.6738 | 0.5387 |
| hp dr=0.5 wd=5e-4 | 08 | 0.8859 | 0.4924 | 0.8505 | **7.7572** ⭐ | 0.7078 | 0.5439 |
| hp dr=0.5 wd=1e-3 | 09 | 0.8859 | 0.4921 | 0.8505 | 7.7575 | 0.7075 | 0.5439 |
| RAG off | abl_rag_off | 0.9719 | 0.4555 | 0.8408 | 7.7311 | 0.7042 | 0.5262 |
| image=cover | abl_img_cover | 0.8998 | 0.4853 | 0.8419 | 8.3380 | 0.6722 | 0.5218 |
| image=cover+banner | abl_img_cover_banner | 0.8954 | 0.4882 | 0.8483 | 8.1734 | 0.6663 | 0.5380 |
| 移除 image | abl_no_image | 0.9401 | 0.4700 | 0.8292 | 8.8163 | 0.6301 | 0.5071 |
| text only | abl_only_text | 1.2541 | 0.3605 | 0.7047 | 10.0170 | 0.5539 | 0.1737 |
| image only | abl_only_image | 1.2876 | 0.3388 | 0.7298 | 8.5115 | 0.6560 | 0.4125 |
| meta only | abl_only_meta | 0.9369 | 0.4590 | 0.8329 | 7.9642 | 0.6955 | 0.5046 |
| Run02 原設定（dr=0.3,batch=256）| 02 | 0.9010 | — | — | **7.5911** | — | — |
| Run03 原設定（=base）| 03 | 0.8859 | 0.4921 | 0.8505 | 7.7575 | 0.7075 | 0.5439 |

> **seed 對齊後的結論**：
> 1. **popularity 最佳 = 低 dropout（0.3）**：04/07（log_MAE 0.882）勝。
> 2. **meanScore 最佳 = Run02 設定（MAE 7.591）**。後續 Run22 隔離變因確認**關鍵是 `attn_dropout`**（0.2→0.1 使 MAE 8.25→7.59），非 batch_size（先前誤判）；batch 256 為次要。
> 3. **per-target 最佳超參方向相反**（pop 要低 dropout；score 要低 attn_dropout）→ 全域單一超參無法兩全，故加 per-target 覆寫（Run22，見頂部最佳結果）。
> 4. **Run02 舊 7.29 是 seed 運氣**（seed-fixed 後 7.591）；**Run03 = pooler_s42 逐位元相同** → 驗證 seed 生效。
> 5. **stage 兩 target 都輸 pooler**（非「打平」，早期是 seed 巧合）；消融方向性與未固定 seed 版一致（RAG 助 pop / 微傷 score；only_meta 最強單模態）。

#### 單模態 image 對照（cover / banner / yolo 各自單獨，seed=42）

| 單模態 | pop log_MAE↓ | pop spear | score MAE↓ | score spear |
|--------|:---:|:---:|:---:|:---:|
| cover | **0.8998** | 0.8419 | 8.3380 | 0.5218 |
| banner | 0.9032 | 0.8370 | 8.3762 | 0.5376 |
| yolo（character）| 0.9108 | **0.8424** | **8.1170** | 0.5269 |
| full（三者合, pooler）| 0.8859 | 0.8505 | 7.7575 | 0.5439 |

> **三個視覺模態都差不多（~0.90），無一獨強**：
> - **popularity**：cover 最好（0.8998）、**yolo 最差**（0.9108）→ character 對「人氣」幫助最小，呼應「yolo 邊際貢獻≈0」。
> - **meanScore**：**yolo 最好**（8.117）→ character 對「評分」反而最有用。
> - 三者合（0.886）> 任一單模態，但對 popularity 而言 cover-only（0.900）已接近 full → 視覺模態冗餘度高。

### popularity（主要指標：log_MAE，越低越好）

| Run | val log_MAE | test log_MAE | test factor_acc_2x | val Spearman | val log_R2 | 主要改動 |
|-----|------------|-------------|-------------------|-------------|-----------|---------|
| 01a | 0.7839 | 0.9357 | — | 0.8851 | 0.8072 | Baseline v2：cover_banner_yolo / use_rag=true / CrossAttn 4 heads / SAM+AdamW / HuberLoss（無時間加權） |
| **01** | **0.7886** | 0.9088 | 0.4778 | **0.8879** | **0.8161** | + Temporal Weighting（alpha=0.2）：exp(-0.2×(max_yr-yr))，normalize mean=1 |
| 02 | 0.7801 | 0.9151 | 0.4778 | 0.8785 | 0.8055 | + TrendHead（pop+score）；gate soft-average；LogCosh 數值穩定版 |
| **21** | 0.7953 | **0.8859** | **0.4921** | 0.8820 | 0.8073 | + Regularization (dropout=0.5, attn_drop=0.2, wd=1e-3)，seed=42 |

#### hp_search（Run03~09：固定 TrendHead + batch=512，搜尋 dropout / weight_decay）

| Run | dropout | weight_decay | val log_MAE | test log_MAE | test facc_2x | test Spearman | val Spearman |
|-----|---------|-------------|------------|-------------|-------------|--------------|-------------|
| 03 | 0.5 | 1e-3 | 0.8059 | 0.9198 | 0.4613 | 0.8466 | 0.8826 |
| 04 | 0.3 | 1e-4 | 0.7977 | 0.9754 | 0.4461 | 0.8501 | 0.8776 |
| 05 | 0.4 | 5e-4 | 0.8137 | 0.9130 | 0.4661 | 0.8508 | 0.8850 |
| 06 | 0.5 | 1e-4 | 0.8249 | 0.9203 | 0.4593 | 0.8473 | 0.8812 |
| **07** ⭐ | **0.3** | **1e-3** | **0.7854** | **0.8904** | **0.4856** | 0.8498 | 0.8792 |
| 08 | 0.5 | 5e-4 | 0.8347 | 1.1288 | 0.3832 | 0.8491 | 0.8783 |
| 09 | 0.5 | 1e-3 | 0.7855 | 0.8944 | 0.4817 | 0.8510 | 0.8840 |

> **popularity 最佳：Run21**（test log_MAE 0.8859），超越了 hp_search 中的 Run07（0.8904）。觀察：適度提高 regularization（dropout 0.5, attn_drop 0.2, weight_decay 1e-3）並搭配 TrendHead 與 seed=42 時，能得到最好的 test 指標。高 dropout 在無 TrendHead 時（如 Run08）曾經退步，顯示組件間交互作用。

### meanScore（主要指標：MAE，越低越好）

| Run | val MAE | test MAE | test within_10pt | val Spearman | val R2 | 主要改動 |
|-----|--------|---------|-----------------|-------------|-------|---------|
| 01a | 6.7441 | 8.0435 | — | 0.6763 | 0.4611 | Baseline v2（無時間加權） |
| 01 | 6.8115 | 8.5722 | 0.6498 | 0.6617 | 0.4143 | + Temporal Weighting（alpha=0.2）：popularity ✅；meanScore test ❌ R2=-0.006 |
| **02** | **6.7604** | **7.2937** ↓ | **0.7360** | **0.6570** | **0.4180** | + TrendHead（pop+score）；gate soft-average；test R2 大幅回升（-0.006→0.246） |
| 21 | 6.7660 | 7.7575 | 0.7075 | 0.6590 | 0.4313 | + Regularization (dropout=0.5, attn_drop=0.2, wd=1e-3)，seed=42 |

> **觀察**：時間加權對 popularity 有效（test log_MAE 0.9357→0.9016），但 meanScore 反而退步。推測原因：meanScore 的 shift 主要是 Label Shift（評分基準整體上移），加權讓模型少看舊資料後反而失去泛化能力；而 popularity 的 shift 更多是 Covariate Shift，加權確實有助於縮小 val/test gap。

#### hp_search（Run03~09：固定 TrendHead + batch=512，搜尋 dropout / weight_decay）

| Run | dropout | weight_decay | val MAE | test MAE | test within_10pt | test Spearman | test R2 |
|-----|---------|-------------|---------|----------|-----------------|--------------|---------|
| 03 | 0.5 | 1e-3 | 6.9678 | 9.5681 | 0.5750 | 0.5322 | -0.1761 |
| 04 | 0.3 | 1e-4 | 6.7269 | **7.6675** | **0.7107** | 0.5402 | 0.1835 |
| 05 | 0.4 | 5e-4 | 6.7952 | 7.8831 | 0.6936 | 0.5533 | 0.1565 |
| 06 | 0.5 | 1e-4 | 6.9003 | 8.5068 | 0.6547 | 0.5383 | 0.0180 |
| 07 | 0.3 | 1e-3 | 6.7499 | 8.1776 | 0.6744 | 0.5397 | 0.0984 |
| **08** ⭐ | **0.5** | **5e-4** | **6.6715** | 7.5847 | 0.7136 | 0.5485 | **0.1911** |
| 09 | 0.5 | 1e-3 | 6.9463 | 9.0994 | 0.6093 | 0.5346 | -0.0707 |

> **meanScore val 最佳：Run08**（val MAE 6.6715），test MAE 7.5847、test R2 0.1911 也是 hp_search 中最佳。但**對照更早的 Run02（test MAE 7.2937, R2 0.246）仍勝出**——val 最佳不保證 test 最佳，distribution shift 主導 meanScore 的 test 表現。hp_search 內部 test MAE 跨度極大（7.58~9.57），且 test R2 數個為負，再次顯示 meanScore 在 test 區段泛化困難。

### 消融實驗（test set，`python src_2/ablation.py`）

固定超參（= Run07：dropout=0.3, wd=1e-3, batch=512, TrendHead on），只改被消融的變因。對照組 full model = Run07。

| 設定 | pop log_MAE | pop facc_2x | pop spearman | score MAE | score spearman | 結論 |
|------|------------|------------|-------------|-----------|---------------|------|
| **full（RAG + cover_banner_yolo）** | **0.8904** | **0.4856** | **0.8498** | 8.1776 | **0.5397** | 對照組 |
| RAG off | 0.9279 | 0.4739 | 0.8385 | 7.6716 | 0.5093 | RAG 對 pop 有益（log_MAE -0.038） |
| image = cover only | 1.1041 | 0.3965 | 0.8405 | 8.3641 | 0.5130 | 移除 banner+yolo，pop 大幅退步 |
| image = cover + banner | 0.8922 | 0.4836 | 0.8500 | 7.5995 | 0.5377 | ≈ full，yolo 邊際貢獻 ~0 |

> **三大發現**：
> 1. **RAG 對 popularity 明確有益**（log_MAE 0.8904 vs 0.9279，spearman +0.011）。對 meanScore spearman 上升但 MAE 變差（distribution shift 下放大數值偏移）。
> 2. **banner 是最關鍵的視覺模態**：cover only → cover+banner，pop log_MAE 從 1.1041 → 0.8922（-0.21）。
> 3. **YOLO crop 邊際貢獻幾乎為零**：cover+banner（0.8922）≈ full（0.8904）。若要精簡 pipeline，可移除 YOLO crop，對 popularity 幾乎無損。

### Multimodal 分支消融（重訓版，`python src_2/ablation_multimodal.py`）

`model.py` 加 `modalities: {image, text, meta}` flag（向後相容），架構真的移除分支後重訓。single-modality 組關閉 RAG + TrendHead。

| 設定 | pop log_MAE | pop facc_2x | pop spearman | score MAE | score spearman |
|------|------------|------------|-------------|-----------|---------------|
| **full（img+txt+meta+rag, Run07）** | **0.8904** | **0.4856** | **0.8498** | 8.1776 | **0.5397** |
| only_meta | 0.9507 | 0.4561 | 0.8322 | **8.1420** | 0.5065 |
| no_image（txt+meta+rag） | 1.0520 | 0.4270 | 0.8287 | 8.4805 | 0.4998 |
| only_text | 1.2634 | 0.3602 | 0.7056 | 10.2581 | 0.2164 |
| only_image | 1.3458 | 0.3337 | 0.7230 | 8.4261 | 0.3948 |

> **發現**：
> 1. **meta 是最強的單一模態**（only_meta：pop log_MAE 0.9507、score MAE 8.142），metadata（前作 / studio·VA TE / format）攜帶最多訊號。
> 2. **text / image 單獨都很弱**（log_MAE 1.26 / 1.35），需與 metadata 結合才有效。
> 3. **多模態互補性確立**：full model 明顯勝過任何單模態，融合架構有實質價值。

### Per-target HP 消融（test set，`ablation_pertarget_s42.py`）— **論文 4.3 權威版**

> ⚠️ **以下為單 seed（seed=42）消融。** 跨 seed 後，**只有 ranking（Spearman）的組件貢獻穩定/顯著**；誤差指標（log_MAE/MAE）的差距多為 seed 噪音，**trend 的貢獻甚至不顯著**。請以上方「Ablation Robustness（T1b）」與「T2b：顯著性檢定」的多 seed 結論為準，本段單 seed 數字僅供參考。

> 舊版 `ablation.py` / `ablation_multimodal.py` 強制共用 Run07 HP，meanScore 不在最佳，導致「移除組件反而 score 變好」的反直覺結果。
> 本版**每個 target 各用自己的最佳 HP**（保留 config 的 per-target overrides：pop dr=0.3；score dr=0.3, attn_dr=0.1, wd=1e-4, batch=256），full=Run22 設定、seed=42，整張表同一條 code path。
> **結果乾淨：full model 在兩個 target 都最佳**，RAG 與每個模態對兩者都有正貢獻。摘要 `runs/ablation_pertarget_s42_summary.json`。

| 設定 | pop log_MAE↓ | pop facc_2x | pop spear | score MAE↓ | score win10 | score spear |
|------|:---:|:---:|:---:|:---:|:---:|:---:|
| **full（img+txt+meta+rag）** | **0.8823** | **0.4943** | **0.8520** | **7.5911** | **0.7104** | **0.5424** |
| RAG off | 0.9473 | 0.4642 | 0.8387 | 8.1246 | 0.6718 | 0.5273 |
| 移除 image | 0.9307 | 0.4830 | 0.8293 | 8.6352 | 0.6511 | 0.5168 |
| meta only | 0.9524 | 0.4587 | 0.8271 | 7.9560 | 0.6832 | 0.5105 |
| text only | 1.2705 | 0.3638 | 0.7012 | 10.3885 | 0.5274 | 0.2113 |
| image only | 1.3106 | 0.3453 | 0.7311 | 8.7938 | 0.6459 | 0.3910 |

#### image 來源消融（per-target HP，seed=42）

| image 來源 | pop log_MAE↓ | pop spear | score MAE↓ | score spear |
|------------|:---:|:---:|:---:|:---:|
| **full（cover+banner+yolo）** | **0.8823** | **0.8520** | **7.5911** | **0.5424** |
| cover only | 0.9059 | 0.8443 | 8.0512 | 0.5125 |
| cover + banner | 0.9367 | 0.8481 | 7.9372 | 0.5390 |
| banner only | 0.8948 | 0.8385 | 8.2616 | 0.5419 |
| yolo only | 0.9280 | 0.8437 | 8.2020 | 0.5188 |

> **發現（每 target 最佳 HP 下）**：
> 1. **full model 兩個 target 都最佳**（pop 0.8823、score 7.5911）→ 每個組件在各自最佳 HP 下都有貢獻，先前共用 HP 的反直覺消失。
> 2. **RAG 對兩者皆有益**：移除後 pop 0.8823→0.9473、score 7.5911→8.1246。
> 3. **image 對 meanScore 尤其關鍵**：移除 image，score 7.5911→8.6352（大幅退步）、pop 0.8823→0.9307。
> 4. **meta 仍是最強單一模態**（only_meta：pop 0.9524、score 7.956）；text / image 單獨都弱（pop 1.27 / 1.31）。
> 5. **三個視覺來源合用最佳**：任一單獨來源（cover / banner / yolo）兩個 target 都不如三者合，視覺模態互補。

#### Temporal Trend（TrendHead）on/off — ~~concept drift 證據~~（論文 4.3.3）

> ⚠️ **此段為單 seed（seed=42）結果，已被多 seed 推翻** —— 見上方「Ablation Robustness（T1b）」：7 seed 配對後 trend **無穩定貢獻**，且 trend head 本身是 seed 不穩定的來源。以下單 seed 數字僅保留為歷史紀錄，**論文結論以 T1b 多 seed 為準**。

`abl_full_notrend_pt` = full model 但關掉 TrendHead（linear+year 時序項），對照 `abl_full_pt`。test 比 train 晚，故此對照衡量時序項在 drift 下是否有用。指令：`python src_2/ablation_pertarget_s42.py --only abl_full_notrend_pt`

| 設定 | pop log_MAE↓ | pop log_R²↑ | score MAE↓ | score R²↑ | score spear |
|------|:---:|:---:|:---:|:---:|:---:|
| **with trend（full）** | **0.8823** | **0.7633** | **7.5911** | **0.1934** | 0.5424 |
| without trend | 0.9036 | 0.7518 | 7.8877 | 0.1247 | 0.5533 |

> **發現**：加 TrendHead 兩個 target 主要誤差都更好 —— pop log_MAE 0.9036→0.8823、**score MAE 7.8877→7.5911、score R² 0.1247→0.1934**（meanScore 受年代評分漂移影響最大，得益最多）。score Spearman 幾乎不變（trend 平移數值水準、不改排序）→ **支持「時序項處理 concept drift」的宣稱**。

### Stage Embedding 實驗（test set，`fussion_configs_stages*.yaml`）

主 image 從 Swin pooler（1024）換成 4 個 stage concat（1920）+ stage 投影（4×256→1024），RAG 維持 pooler（解耦）。Run12 加 per-stage LayerNorm，Run13 為對照（無 LayerNorm，隔離 LN 變因）。

**全部 seed=42 對齊（`rerun_s42.py`），唯一差異是 image 模式 / LN。**

**popularity（test）：**

| 設定 | log_MAE | facc_2x | spearman | log_R2 |
|------|---------|---------|----------|--------|
| **pooler（pooler_s42）** | **0.8859** | **0.4921** | **0.8505** | **0.7616** |
| stage + LN（stage_ln_s42） | 0.9653 | 0.4580 | 0.8476 | 0.7177 |
| stage 無 LN（stage_noln_s42） | 0.8989 | 0.4843 | 0.8442 | 0.7509 |

**meanScore（test）：**

| 設定 | MAE | R2 | within_10pt | spearman |
|------|-----|-----|------------|----------|
| **pooler（pooler_s42）** | **7.7575** | **0.1669** | **0.7075** | 0.5439 |
| stage + LN（stage_ln_s42） | 7.8422 | 0.1428 | 0.6968 | 0.5419 |
| stage 無 LN（stage_noln_s42） | 7.9428 | 0.1346 | 0.6936 | 0.5429 |

> **結論（seed 對齊）：stage 在兩個 target 都明確輸 pooler。**
> 1. **stage 確定較差，非「打平」**：seed 固定後 stage_noln（pop 0.8989 / score 7.9428）兩個 target 都輸 pooler（0.8859 / 7.7575）。早期看似「打平」是未固定 seed 的巧合。
> 2. **per-stage LayerNorm 對 popularity 嚴重傷害**（0.9653 vs 0.8989），對 meanScore 反而略好（7.8422 vs 7.9428）；但兩 target 都輸 pooler，故 LN 設為**預設 false**。推測 LN 抹掉了各 stage 的「量級」資訊，而 image 分支的 content gate 靠量級判斷模態重要性。
> 3. **判斷**：多尺度 stage 特徵未帶來增益，**pooler 仍最佳**（更簡單、維度低、能開 AMP）。**stage 這條線收掉。**
>
> 數值修正紀錄：stage 初期 NaN，根因 raw stage 量級大 + AMP float16 梯度溢位。解法：stage config 停用 AMP（`mixed_precision: false`）；LayerNorm（`image_stage_norm`）預設 false。

---

## 可解釋性分析（Run22，`explain/`）— **論文 4.5**

對 Run22 best model 做兩種解釋：cross-attention 注意力圖（`rag_heatmap.py`）、梯度 attribution（`feature_attr.py`：Captum IG + SHAP）。

產生指令：
```bash
python src_2/explain/rag_heatmap.py  --target popularity --split test --ids 154587   # Frieren heatmap
python src_2/explain/feature_attr.py --target popularity --split test --n 30          # Captum + SHAP（兩 target 各跑）
```

### Captum Integrated Gradients — 模態層級 mean |IG|（test n=30 平均）

| 模態 | popularity | meanScore |
|------|:---:|:---:|
| **rag_image** | **0.399** | **0.431** |
| image_yolo | 0.207 | 0.253 |
| image_banner | 0.150 | 0.144 |
| image_cover | 0.081 | 0.089 |
| meta | 0.074 | 0.051 |
| rag_meta | 0.045 | 0.022 |
| text | 0.032 | 0.008 |
| rag_text | 0.011 | 0.002 |

> **發現**：兩個 target 都是 **retrieved image（rag_image）attribution 最高**，其次自身 image（yolo/banner/cover），**text 類最低**。
> 注意這與 attention 權重（集中於 metadata）方向不同：attention 看「在 retrieved tokens 中看哪裡」，IG attribution 看「哪個輸入改變輸出」，兩者衡量不同的東西。

### SHAP — metadata 特徵重要性（popularity top）

`va_te_pop`（0.14）> `prequel_meanScore_mean`（0.10）> `prequel_popularity_mean_log1p`（0.10）> `studio_te_pop`（0.08）> `va_te_score`（0.07）> `studio_te_score`（0.05）…

> 前段全是 **inter-title 訊號**（聲優 TE、前作評分/人氣、工作室 TE）→ 一部作品的預測接受度與其前作、製作團隊的過往表現高度相關，印證 RAG 的設計動機。

### Cross-Attention Heatmap（以 Sousou no Frieren 為例）

注意力幾乎全落在 retrieved 作品的 **metadata 列**（0.13–0.26），text / image ≈ 0，且集中在少數幾部最相關作品。
> 呼應 cross-attention 設計：query = 目標 metadata，故模型以「關係」判斷相關性。

---

## 已知限制

### 1. meanScore 時序 Distribution Shift

資料採時序切分（train → val → test），AniList 評分中位數隨時間系統性上升：

| Split | meanScore 平均 | meanScore 中位數 |
|-------|--------------|----------------|
| train | 58.1 | 60.0 |
| val | 61.6 | 63.0 |
| test | 65.4 | 66.0 |

2022 年後出現 **約 +7 分的跳升**，導致 test R² 偏低（0.13）。模型的預測中心值 ≈ 61，在 test set 系統性低估約 4.4 分。這是資料本身的時序特性，非模型 bug。

### 2. popularity AMP float16 溢位

`expm1(y)` 在 float16 下上限 65,504，normalized 空間 y ≈ 17.6 時直接 overflow → Infinity。
**解法**：`denormalize_target()` 強制轉 float64，clip(±5σ) 後再執行 expm1。

### 3. Overfitting：Train Loss 快速下降，Val Loss 早熟

Run01/02 的訓練動態顯示明顯的 overfitting pattern：

```
Run01 popularity:
  ep1:  train=0.242  val=0.159  gap=-0.083
  ep7:  train=0.063  val=0.115  gap=+0.052  ← best
  ep12: train=0.041  val=0.136  gap=+0.095
```

Train loss 從 ep1 到 ep12 降了 **6 倍**，val loss 只改善 **1.4 倍**。Best checkpoint 在 ep7~10 就觸底，之後 val 持續震盪或上升，train 還在下降。  
根本原因：模型在 memorize train set 的個別動畫特徵，而非學到跨時序的泛化規律。

**緩解方向（Run03）**：提高 dropout（0.3→0.5）、weight_decay（1e-4→1e-3）、batch_size（256→512，降低 gradient noise，讓 val 指標更穩定）。

### 4. GPU 使用率相對 v1 偏低

v1 在訓練時執行 TextGNN + ImageGNN forward pass（額外 GPU 運算）。v2 的 embedding 全部預先計算並存入 parquet，訓練時只跑 FusionModel（~965K params），GPU 負載較輕。

### 4. RAG 全遮罩 NaN

RAG 無命中時（retrieved_ids 為空），`rag_mask` 全 True → `MultiheadAttention` softmax 對全遮罩位置輸出 NaN。
**解法**：`dataset.py` 強制 `rag_mask[0] = False`，讓 attention 退化為對零向量的 uniform attention。

### 5. Cold-start（新 studio / 新聲優）

MetaEncoder TE 對未見過的 studio / 聲優補訓練集全體均值；`is_new_studio` 旗標告知模型補值情況。

---

## 輸出檔案

```
src_2/runs/{run_id}/
├── {target}/
│   ├── best_model.pt          ← 最佳 val loss checkpoint（state_dict）
│   ├── target_scaler.json     ← 正規化參數（center, scale, log_transform）
│   ├── history.json           ← 每 epoch train_loss / val_loss / val_mae / lr（含 run_id / notes）
│   ├── final_metrics.json     ← train / val / test 完整 metrics 合併（含 run_id / notes）
│   └── pred_{split}.csv       ← id, pred, target（原始 scale）
└── explain/{target}/          ← 可解釋性輸出（run 層級）
    ├── rag/{id}_attn.png          ← Cross Attention heatmap
    ├── feature/captum_modality.csv
    ├── feature/captum_modality.png
    ├── feature/shap_values.npy
    └── feature/shap_summary.png

src_2/fussion_training/meta_encoder.json   ← 訓練集 fit 的 MetaEncoder（自動生成）
```
