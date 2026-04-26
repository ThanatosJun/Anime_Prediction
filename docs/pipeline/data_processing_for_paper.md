# 論文寫作用資料處理紀錄

本文件以論文可直接引用的格式，記錄目前資料處理流程與研究範圍決策，重點包含可重現性、設計理由、建模影響與研究邊界。

## 0) 核心目標與範圍定義

- **主要目標：** 使用 **開播前（pre-release）** 可取得的多模態訊號（圖片、文字描述、metadata）預測開播後指標（`popularity`、`meanScore`）。
- **目標協議（目前決策）：** 兩個主要目標皆採迴歸任務；目前範圍不設定固定 `Day7` 目標。
- **Pre-release 定義：** 作品正式播出/發行/上架前的資訊狀態。
- **目前專案階段：** 完成資料 pipeline 與證據層（尚非最終模型結果報告）。

## 0.1) 領域評估與最終選擇

團隊曾評估多個候選領域，最終選定 **Anime**。

- **A. Anime（採用）**
  - 本專案主資料來源：AniList（快照 `20,324` 筆）。
  - 主目標：`popularity` 與 `meanScore`。
  - 主要風險：`popularity` 為累積型指標，可能產生 snapshot bias。
- **B. Airbnb 定價（未採用）**
  - 價格波動高度受事件/日期驅動。
  - 靜態快照特徵不足以支撐穩定預測。
- **C. 其他領域（未採用）**
  - Movie/Game：下載/採用標籤蒐集難度高，且仍有快照漂移問題。
  - YouTube：公開指標較標準化，新增可用預測訊號空間有限。

## 0.2) 預計建模架構（研究設計層）

下游模型規劃為融合式架構：

- **Image branch：** Swin Transformer（或 ResNet-50 baseline）處理封面與視覺特徵。
- **Text branch：** Transformer 編碼器（如 GPT-2 family 風格 embedding 管線）處理劇情語義。
- **Metadata branch：** 結構化特徵（genres、episodes、studio、relation、cast 等）。
- **Retrieval augmentation（對應 RQ1）：** 以 relation/company/sequel 連結作為檢索上下文。

> 註：本文件同時記錄「已落地」與「下一層實驗設計」。上列架構屬研究設計目標；目前 repository 主要完成資料與證據準備。

## 1) 資料集範圍與快照控制

- **領域：** 動畫 pre-release 預測。
- **原始快照來源：** `data/raw/anilist_anime_data_complete.pkl`、`data/raw/anilist_anime_data_complete.csv`。
- **快照指紋：** `data/raw/raw_manifest.json`（sha256 + 檔案大小）。
- **動機：** 避免上游資料更新造成無聲漂移（silent drift）。

## 2) 處理階段與產物

### Stage A：Baseline EDA（`scripts/run_baseline_eda.py`）
- **目的：** 在規則決策前完成描述性盤點。
- **輸出：** `data/eda/baseline_eda_summary.json/.md`
- **主要指標：** 缺值率、數值分佈、IQR 異常值邊界。

### Stage B：Decision EDA（`scripts/run_decision_eda.py`）
- **目的：** 把描述性統計轉為可執行規則。
- **輸出：** `data/eda/decision_eda_summary.json/.md`
- **決策訊號：**
  - 依缺值比例建議 `drop/fill/keep`
  - 依異常值比例建議 `clip/winsorize/retain`
  - 目標欄位相關性輪廓
  - 群組影響摘要（例如 `format` -> `popularity`）

### Stage C：Interim Dataset（`scripts/build_interim_dataset.py`）
- **輸出：** `data/interim/anilist_anime_data_interim_YYYYMMDD.csv` + metadata json
- **目前規則版本：** `decision_eda_v2_relation_features`
- **操作：**
  - 僅保留建模關聯欄位
  - 統一數值 dtype
  - 以 `id` 去重
  - 以明確政策映射（`MISSING_RULES`）補值

### Stage D：Processed Dataset（`scripts/build_processed_dataset.py`）
- **輸出：** `data/processed/anilist_anime_data_processed_v1.csv` + metadata json
- **目前規則版本：** `decision_eda_v3`
- **操作：**
  - 關鍵數值欄位非負約束
  - 依 `CLIP_COLUMNS` 做百分位裁切
  - quarter-normalized popularity 目標工程
  - 時序 pre-release 切分（`train/val/test/unknown`）
  - unknown 政策：`unknown` 轉為 `holdout_unknown`，不納入模型 split

### Stage E：Multimodal Input Export（`scripts/export_multimodal_inputs.py`）
- **目的：** 保留 text/image/trailer 欄位並維持 split 對齊。
- **輸出：**
  - `data/processed/anilist_anime_multimodal_input_v1.csv`
  - `data/processed/anilist_anime_multimodal_input_{train|val|test|holdout_unknown}.csv`
  - `data/eda/multimodal_input_summary.json/.md`
- **目前追蹤證據：**
  - feature contract（join key、目標欄位、模態原始欄位）
  - 模態可用旗標與比例
  - 實體 split 筆數

### Stage F：RQ-oriented EDA（`scripts/run_rq_eda.py`）
- **目的：** 產出對應 RQ 的論文證據層。
- **輸出：** `data/eda/rq_eda_summary.json/.md`
- **目前追蹤證據：**
  - snapshot 緩解代理指標：`corr(release_year, popularity_raw)` 與 `corr(release_year, popularity_quarter_pct)` 比較
  - quarter normalization 前後相關性絕對值下降幅度
  - RQ1 可行性代理（metadata/relation 覆蓋與有效 split 分佈）
  - 各 split 的 popularity class balance
  - RQ2 可行性代理（text/image/trailer 覆蓋）
  - 各 split 的 multimodal 覆蓋
  - 統計檢定層：
    - split bucket balance 的 permutation test
    - split 間 multimodal 覆蓋差異的 permutation tests
    - snapshot 相關性下降的 bootstrap CI

### Stage G：RQ 圖表產生（`scripts/run_rq_eda_plots.py`）
- **目的：** 把 RQ 指標直接轉成論文圖表。
- **輸出：** `data/eda/figures/*.png` + `data/eda/figures/rq_figure_notes.md`
- **目前圖表：**
  - snapshot bias 代理圖（quarter normalization 前後絕對相關）
  - 各 split 的 popularity bucket 平衡
  - 各 split 的 multimodal 覆蓋

### Stage H：Holdout Unknown 診斷（`scripts/run_holdout_unknown_diagnostic.py`）
- **目的：** 量化被排除之 temporal-unknown 樣本風險。
- **輸出：** `data/eda/holdout_unknown_diagnostic.json/.md`
- **目前追蹤證據：**
  - holdout 規模與全體比例
  - 時序欄位缺值輪廓
  - 與模型樣本母體在關鍵欄位上的分佈落差

### Stage I：欄位血緣報告（`scripts/run_column_lineage_report.py`）
- **目的：** 明確記錄 raw -> interim -> processed -> multimodal 欄位變換證據。
- **輸出：** `data/eda/column_lineage_summary.json/.md`
- **目前追蹤證據：**
  - 各階段欄位數量
  - 各階段 keep/drop/add 集合
  - 衍生欄位與轉換函式來源映射
  - multimodal 回補欄位與可用旗標的推導理由

## 3) 目前版本明確規則

### 3.1 缺值規則（interim）
- `episodes`：format 中位數，回退全域中位數
- `duration`：format 中位數，回退全域中位數
- `averageScore`：先以 `meanScore` 回補，再回退全域中位數
- `seasonYear`：以 `startDate_year` 回補
- `title_english`：以 `title_romaji` 回補

### 3.2 異常值規則（processed）
- `episodes`：P1-P99
- `duration`：P1-P99
- `averageScore`：P0.5-P99.5
- `meanScore`：P0.5-P99.5
- `popularity`：P1-P99
- `favourites`：P1-P99
- `trending`：P1-P95

## 4) 用於快照緩解的 popularity 目標工程

### 4.1 quarter key 建構
- `release_year` 由 `seasonYear` 建立（回退 `startDate_year`）。
- `release_quarter` 由以下規則建立：
  - `season` 映射（`WINTER=1`, `SPRING=2`, `SUMMER=3`, `FALL=4`）
  - 若 `season` 缺值，回退 `startDate_month`。
- 合併為 `release_quarter_key`（例如 `2021Q3`）。

### 4.2 相對熱度目標（輔助特徵 / 診斷層）
- 在同 quarter 內計算百分位排名：
  - `popularity_quarter_pct = rank(popularity within release_quarter_key, pct=True)`
- 分箱定義：
  - `cold_0_25`：0-25%
  - `warm_25_50`：25-50%
  - `hot_50_75`：50-75%
  - `top_75_100`：75-100%

### 4.3 設計理由
- 原始 popularity 屬累積型，存在時間偏差（snapshot issue）。
- 同 quarter 的相對排名更符合 pre-release 場景比較需求。

## 5) Pre-release 時序切分協議

- 建立 quarter index：`quarter_index = release_year * 10 + release_quarter`
- quarter 群組依時間排序
- 切分點以 **累積列數**（不是 quarter 數）決定：
  - train 目標：70%
  - val 目標：15%
  - test 目標：15%
- 依 quarter index 將每筆樣本映射到 `split_pre_release`
- quarter 資訊缺失樣本標記為 `unknown`
- unknown 政策：
  - 在 `split_pre_release_effective` 中 `unknown -> holdout_unknown`
  - `holdout_unknown` 不納入 train/val/test 模型擬合

## 6) 可重現性證據

- 原始快照完整性：`data/raw/raw_manifest.json`
- 規則版本追蹤：
  - interim metadata：`rule_version`、`applied_missing_rules`
  - processed metadata：`rule_version`、`clip_config`、`popularity_quarter_target`、`pre_release_split`
- 決策與證據輸出：
  - `data/eda/multimodal_input_summary.*`
  - `data/eda/decision_eda_summary.*`
  - `data/eda/target_engineering_summary.*`
  - `data/eda/outlier_handling_summary.*`
  - `data/eda/rq_eda_summary.*`
  - `data/eda/holdout_unknown_diagnostic.*`
  - `data/eda/column_lineage_summary.*`

## 7) 目前限制與論文註記

- 目前切分仍含 temporal 欄位缺失造成的 `unknown` bucket。
- 相對熱度標籤可降低 snapshot bias，但無法完全消除生命週期效應。
- 目前 pipeline 聚焦 tabular 前處理；多模態 embedding 管線屬下一階段。
- 論文建議同時回報：
  - normalization 前（raw）指標行為
  - target engineering 後（quarter-normalized）指標行為

## 7.1) RQ 對應與評估規劃

- **RQ1：** retrieval-based augmentation 是否提升 `popularity` 與 `meanScore` 回歸表現。
- **RQ2：** transformer-based 圖像語義是否優於簡單 tag-style 特徵。
- **可解釋性規劃：**
  - 用 SHAP 分析 metadata 影響
  - 用 ablation（`tabular+text`、`+image`、`+retrieval`）驗證增益

## 7.2) 範圍邊界與既有疑慮回應

- 本階段僅做 **pre-release** 預測準備，不含即時 post-release 社群擴散訊號。
- snapshot 控制由 quarter-relative popularity 工程與時序切分協議處理。
- relation/studio/cast 與 multimodal 可用性已保留為 EDA 與 lineage 證據。
- `popularity_quarter_pct` 與 `popularity_quarter_bucket` 屬輔助診斷與控制特徵，不是最終 target 定義本體。

## 7.3) 已採納之下一階段 TODO

- [completed] relation-based IP 歷史特徵：
  - 在 interim 契約加入結構特徵（`is_sequel`、`has_sequel`、`prequel_count`）
  - 加入前作表現代理（`prequel_popularity_mean`、`prequel_meanScore_mean`）
- [pending] studio 能量特徵：
  - 以時間安全視窗推導 studio 歷史強度
- [pending] 多模態 embedding bridge：
  - 增加 image/text/video 資產下載與 embedding 萃取階段
- [pending] 時變 tag 熱度：
  - 建立 genre/tag 市場週期的時間趨勢訊號

## 8) 最小可重跑指令

```bash
python scripts/generate_raw_manifest.py
python scripts/run_baseline_eda.py
python scripts/run_decision_eda.py
python scripts/build_interim_dataset.py
python scripts/build_processed_dataset.py
python scripts/export_multimodal_inputs.py
python scripts/run_rq_eda.py
python scripts/run_rq_eda_plots.py
python scripts/run_holdout_unknown_diagnostic.py
python scripts/run_column_lineage_report.py
```
