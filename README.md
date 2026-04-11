# 資料科學與機器學習：期末研究提案

本專案為「資料科學與機器學習」課程之期末研究文件儲存庫。
歷經多次討論與收斂，最終確立之研究主題為：**新番動畫播出前熱度預測 (Pre-release Anime Popularity Prediction Based on Multimodal Features)**。

## 📁 專案目錄架構 (Project Structure)

```text
Anime_Prediction/
├── docs/                                           # 提案相關文件目錄
│   ├── A-anime_popularity_prediction_proposal.md   # 🌟 最終定案版：期末研究提案 (主檔)
│   ├── B-Proposal_Anime_Multimodal_Recommendation.md # 歷史提案 (推薦系統方向草案)
│   ├── Proposal_Route_C_Anime_Cold_Start_Prediction.md # 提案 C (路線決策脈絡紀錄檔)
│   ├── Proposal_Route_A_Biometric_Vulnerability.md # 提案 A (生物辨識漏洞相關草案)
│   └── Proposal_Route_B_Social_Media_Deepfake.md   # 提案 B (社群媒體 Deepfake 相關草案)
├── .github/                                        # GitHub 相關設定與 Skills
├── .gitignore                                      # Git 忽略檔案設定 (排除 agents/skills)
└── README.md                                       # 專案說明文件 (本檔案)
```

## 🎯 最終定案研究任務摘要

在動畫正式釋出第一集前 (冷啟動狀態下)，模型僅能取得**開播前的多模態資訊**與**表格元資料**。本研究旨在建立雙軌模型，分別預測開播後第 7 天的 **人氣熱度 (Popularity)** 與 **評價分數 (Score)**，藉此量化分析不同模態 (特別是 PV 影片) 對觀眾預期心理的影響力，並協助串流平台進行版權採購評估與行銷預算分配。

### 🔑 核心研究特徵 (Input Features)
- **文字 (Text)**：劇情大綱 (Synopsis)
- **圖片 (Image)**：主視覺圖 / 海報 (Cover Image)
- **影片 (Video/Audio)**：預告片 (PV) 的時空特徵與配樂節奏
- **表格 (Tabular)**：製作公司陣容、是否為續作、前作 IP 歷史影響力等 (作為 Control Variables)

### 🔬 消融實驗設計 (Ablation Study)
本研究設計三組漸進式的消融實驗，以量化拆解各模態的影響力：
1. **Baseline (Tabular + Text)**：僅依賴「IP/製作陣容 + 故事設定」，代表業界最基本的企劃文案評估水準。
2. **加入靜態視覺 (+Image)**：評估加入主視覺圖後，對觀眾吸引力的額外增益。
3. **完整模型 (+PV)**：檢驗動態視聽資訊 (畫風流暢度、分鏡、配樂節奏) 所帶來的最終極限增益。
