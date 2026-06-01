# Reference Baseline 本週進度統整

更新日期：2026-05-12

本文件依據既有追蹤檔進行統一口徑盤點，目的為支援本週週會與短報告敘述。範圍限於 `reference baseline`，不含本專案內部 ablation。

## 依據來源

- `docs/baseline_reference_implementation_plan.md`
- `reports/baselines/reference_baseline_status.md`
- `reports/baselines/reference_baseline_runs.md`
- `reports/baselines/reference_baseline_results.csv`

## 一、目前進度（已完成 vs 未完成）

### 已完成路線

| 路線 | Baseline IDs | 目前狀態 |
|---|---|---|
| `0. Lowest Reference` | `F0-Mean`, `F0-Ridge-Meta` | done |
| `1.1 Metadata-only Classical ML` | `F1-RF-Meta`, `F1-GB-Meta` | done as adaptation |
| `1.2 Feature-concat Classical ML` | `F2-XGB-Concat` | done as adaptation |
| `1.3 Text-only Baseline` | `T2-XGB-TextEmb` | done as adaptation |
| `1.4 Image-only Baseline` | `I1-XGB-ImageEmb` | done as adaptation |
| `2.1 Anime Domain Deep Fusion` | `C1-Armenta-MLP`, `C1-Armenta-ProxyBranchMLP` | first-pass / proxy done as adaptation |
| `2.2 Cross-modal Transformer Fusion` | `C2-CTNN-Lite` | done as adaptation |

### 尚未完成路線

| 路線 | Baseline IDs | 缺口 |
|---|---|---|
| `2.3 Retrieval / RAG Competitive Baseline` | 未建立 C3 baseline row | 尚未完成 `SKAPP-inspired reference baseline` 的獨立 runner 與結果列 |

## 二、當前困難（本週需明講）

### 1) 效能排序困難：F2 仍是最穩參考地板

以 test R2 觀察，`F2-XGB-Concat` 目前仍是多數比較中的穩定強基準：

- `popularity`: `F2 = 0.5194`（高於 `C1-Armenta-ProxyBranchMLP = 0.2600`、`C2-CTNN-Lite = 0.1716`）
- `meanScore`: `F2 = 0.0193`（與 `C1-Armenta-ProxyBranchMLP = 0.0398` 接近）

含義：現階段「metadata + text/image embedding 的 classical concat」仍是主要 performance floor。

### 2) target 困難：`meanScore` 訊號偏弱

多數模型在 `meanScore` 的 test R2 接近 0 或為負值（如 `C2-CTNN-Lite = -0.2602`、`T2-XGB-TextEmb = -0.3846`），顯示此 target 在現有 pre-release 特徵下可預測性偏低，且模型間差距較不穩定。

### 3) 宣稱邊界困難：目前多為 adaptation，不是 reproduction

- `C1-Armenta-*`：缺少原文角色描述/角色肖像分支資產，屬 proxy adaptation。
- `C2-CTNN-Lite`：屬 lightweight cross-modal adaptation，非完整 CTNN 重現。
- `C3` 尚未建立：在三篇 anchor paper map 上仍未 fully closed。

### 4) 資料覆蓋困難：多模態交集樣本下降

embedding strict intersection 後，樣本由 metadata 全量下降：

- train：`9583 -> 9205`
- val：`2918 -> 2637`
- test：`3087 -> 2808`

含義：跨模態比較雖維持一致 input 規格，但有效樣本縮減，會影響模型上限與比較解讀。

## 三、本週預計到達點（執行版）

### 最低保證交付（must-have）

1. 建立並跑通 `C3-RAG-Minimal`（SKAPP-inspired baseline 的最小可報告版本）。
2. 將 `C3` 結果加入 `reports/baselines/reference_baseline_results.csv`。
3. 同步更新：
   - `reports/baselines/reference_baseline_status.md`
   - `reports/baselines/reference_baseline_runs.md`
4. 完成後可宣告：reference baseline map 已覆蓋 `0 ~ 2.3` 全路線（以 adaptation/reproduction 等級如實標註）。

### 可選加值（should-have, 視剩餘工時）

1. 對 `C1-Armenta-ProxyBranchMLP`、`C2-CTNN-Lite` 各做一輪低成本調參（不改大架構）：
   - 例如學習率、hidden size、dropout、early stopping 條件。
2. 補一段固定模板解讀（`popularity` vs `meanScore`），統一簡報與文件說法。

### 風險與停損

- 若本週無法完成 C3：先凍結為「Foundation + C1/C2 adaptation 已完成」版本，並把 C3 列為下週第一優先。
- 任何結果敘述都維持 claim boundary：adaptation 不寫成 paper reproduction。

## 週會可直接使用的三句話

1. 目前 reference baseline 已完成 Foundation、Single-modality，以及 Competitive 的 C1/C2 adaptation，主體比較已可成立。
2. 現階段最穩定的 performance floor 仍是 `F2-XGB-Concat`，顯示 metadata + embeddings 的 classical concat 在現資料條件下最可靠。
3. 本週關鍵任務是補齊 `C3-RAG-Minimal` 與對應報表更新，完成三篇 anchor route 的 coverage closure。
