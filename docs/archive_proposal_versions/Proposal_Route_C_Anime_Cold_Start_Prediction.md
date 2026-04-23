# 期末研究提案：基於多模態特徵之新番動畫首播熱度預測
(Cold-Start Anime Popularity Prediction based on Multimodal Features)

## 📌 目前決策統整 (進度紀錄)

### 1. 核心研究任務
* **任務定義**：新番動畫熱度預測 (Popularity Prediction)。
* **預測目標 ($Y$)**：預測開播後的「觀看熱度 / 追蹤人數」(例如 MyAnimeList 的 Members 數量、平台播放量等單調遞增數值)。**排除預測「評分」**，以避免開播初期的極端評價偏誤 (Early-bird bias) 與歷史資料取得困難。

### 2. 特徵工程與實驗設計 ($X$)
* **輸入特徵**：
  * **多模態特徵 (Multimodal)**：純文字 (Text, 劇情大綱)、靜態圖片 (Image, 主視覺圖)、動態影片 (Video/PV)。
  * **表格型元資料 (Tabular Metadata)**：製作公司等級、是否為續作等。
* **實驗設計 (範圍限制)**：採用**「特徵控制法 (Feature-level Control)」**。保留資料集的多樣性 (不剔除大作或續作)，強制將 IP 知名度、製作公司等外部因素作為 Baseline (控制變數)。
* **核心假設**：在已知 IP 熱度的情況下，加入多模態特徵 (特別是 PV 捕捉到的畫風、氛圍、劇情節奏、音樂與最終成品品質) 能顯著降低預測誤差。

### 3. 研究關鍵字 (Keywords)
1. `Multimodal Representation Learning` (多模態表徵學習)
2. `Feature Fusion` (特徵融合)
3. `Ablation Study` (消融實驗)
4. `New Anime Popularity Prediction` (新番預測)

---

## 📝 提案四大主軸 (撰寫區)

### 1. 研究背景與動機 (Overview & Motivation)
> **[狀態：討論與草稿構思中]**
> * What is it about? (輸入/輸出的具體定義)
> * Why it is important? (對平台方、代理商的商業或學術價值)
> * Why current methods are not enough? (現有只看大IP、製作團隊的盲點，多模態能帶來什麼突破)

### 2. 文獻回顧與研究現況 (Literature Review & SOTA)
> **[狀態：尚未開始]**

### 3. 研究設計與資料集 (Proposed Study & Datasets)
> **[狀態：尚未開始]**

### 4. 預期成果、挑戰與限制 (Outcomes, Challenges & Limitations)
> **[狀態：尚未開始]**