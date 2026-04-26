# 交接文件：Fusion Model 組

本文件提供融合模型組在現有資料流程上的直接接手指南。

## 1) 任務定位

- 任務：整合 text/image/tabular 分支，建立最終雙目標回歸模型。
- 目標：
  - `popularity`（回歸）
  - `meanScore`（回歸）
- 關聯研究問題：
  - RQ1：加入 relation / retrieval 類訊號是否提升效能
  - RQ2：多模態融合是否優於單分支

## 2) 主要輸入資料與分支特徵

- 基礎資料：
  - `data/processed/anilist_anime_data_processed_v1.csv`（tabular 母表）
  - `data/processed/anilist_anime_multimodal_input_v1.csv`（multimodal 母表）
- split 欄位：`split_pre_release_effective`
- 新增 relation 特徵（已落地）：
  - `is_sequel`, `has_sequel`, `prequel_count`
  - `prequel_popularity_mean`, `prequel_meanScore_mean`

## 3) 融合策略建議（第一版）

- Baseline 1：Tabular-only（XGBoost / LightGBM / MLP）
- Baseline 2：Text-only / Image-only（由各分支組提供 embedding）
- Fusion 版本：
  - Early fusion：concat 各分支 embedding + tabular 特徵
  - Late fusion：分支預測再做加權/stacking
- 輸出層：
  - 共享 backbone + 雙 head 回歸（popularity / meanScore）

## 4) 交付與輸出規格

- 產出建議：
  - `reports/fusion_metrics.json`
  - `reports/ablation_results.json`
  - `reports/error_analysis_by_split.json`
- 必回報：
  - `MAE`, `RMSE`, `Spearman`（雙目標）
  - 與單分支 baseline 的增益
  - 按 split 的誤差分佈

## 5) 代辦清單（接手後順序）

1. 固定基準切分，先完成 tabular-only baseline。
2. 接入 text/image embedding，完成 early fusion baseline。
3. 加入 relation 特徵與 ablation（含/不含 relation）。
4. 做 SHAP 與特徵重要度解釋。
5. 輸出最終可報告版本（圖表與摘要）。

## 6) 風險與注意事項

- 必須維持目前 split 協議，避免 leakage。
- 模態缺失需明確策略（mask、drop、impute），不能隱式忽略。
- 若結果與預期差距大，先做子群分析（sequel/non-sequel、has_text、has_cover）。
