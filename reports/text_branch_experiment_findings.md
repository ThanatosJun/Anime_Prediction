# 文本分支實驗結果

日期：2026-04-25
負責人：Text branch
範圍：面向 popularity 與 meanScore 回歸任務的預發布文本嵌入與純文本基線

## 1. 目標
基於動漫簡介（description）生成可復用的文本嵌入產物，用於後續與圖像及其他模態進行融合。

## 2. 流水線概覽
輸入來源檔案：
- data/processed/anilist_anime_multimodal_input_train.csv
- data/processed/anilist_anime_multimodal_input_val.csv
- data/processed/anilist_anime_multimodal_input_test.csv

文本處理與嵌入：
- 文本欄位：description
- 清洗：轉小寫、移除 URL、空白規範化、最小長度 10、最大長度 512
- 模型：sentence-transformers/all-MiniLM-L6-v2
- 嵌入維度：384
- 設備：CPU
- 批次大小：16
- 隨機種子：42

產出檔案：
- artifacts/text_embeddings_train.parquet
- artifacts/text_embeddings_val.parquet
- artifacts/text_embeddings_test.parquet

配套報告：
- reports/text_embedding_pipeline_summary.json
- reports/text_branch_metrics.json

## 3. 清洗後的資料保留率
- Train：12783 / 13376 已編碼（95.57%）
- Val：2637 / 2918 已編碼（90.37%）
- Test：2808 / 3087 已編碼（90.96%）

解讀：
- 大多數樣本列包含可用文本。
- 少量樣本因清洗後文本缺失或無效而被捨棄。
- 融合階段必須明確處理缺失文本的樣本列。

## 4. 基線模型與指標
基線模型：
- Ridge 回歸，alpha = 1.0
- 僅使用文本嵌入，並針對每個目標分別訓練

目標：popularity
- 驗證集：MAE 20462.82，RMSE 42125.67，Spearman 0.5509
- 測試集：MAE 17946.53，RMSE 34055.32，Spearman 0.5408

目標：meanScore
- 驗證集：MAE 9.81，RMSE 11.93，Spearman 0.2886
- 測試集：MAE 10.94，RMSE 13.12，Spearman 0.2152

解讀：
- 文本嵌入對 popularity 提供了有意義的排序訊號。
- 對 meanScore 也存在純文本訊號，但相對較弱。
- 這支持在多模態融合中使用文本特徵，尤其是針對 popularity。

## 5. 劃分完整性與洩漏檢查
各劃分間 ID 重疊檢查：
- train vs val：0
- train vs test：0
- val vs test：0

結論：
- 在嵌入產物中未檢測到跨劃分 ID 洩漏。

## 6. 面向融合團隊同學的交接約定
在融合流水線中將以下欄位作為文本特徵：
- emb_000 到 emb_383（384 維向量）

將以下欄位用於關聯與監督：
- id 作為關聯鍵
- split 欄位用於與資料劃分對齊的使用方式
- 在需要時將 popularity 與 meanScore 作為目標參考

建議的融合實務：
- 按檔案嚴格保持資料劃分邊界（train 配 train，val 配 val，test 配 test）
- 為缺失文本樣本制定明確策略（取交集、插補，或缺失模態處理）

## 7. 可重現性快照
基線報告使用的套件版本：
- python 3.13.5
- numpy 2.1.3
- pandas 2.2.3
- scipy 1.15.3
- scikit-learn 1.6.1
- pyarrow 19.0.0

使用指令：
- C:/Users/User/anaconda3/python.exe src/text_branch/run_text_embedding_pipeline.py --splits train val test
- C:/Users/User/anaconda3/python.exe src/text_branch/baseline_model.py

## 8. 推薦的下一步（可選）
再執行一個額外的文本基線（例如 MLPRegressor）並與 Ridge 進行對比，以為論文報告提供更強的文本分支基準。
