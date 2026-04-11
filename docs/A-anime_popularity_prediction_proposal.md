# 期末研究提案：新番動畫首播熱度預測
**(Anime Popularity Prediction Based on Multimodal Features)**

> 📅 討論紀錄整理版｜based on 與 Claude 的完整討論

---

## 📌 核心決策定案

### 1. 研究任務定義

- **任務名稱**：新番動畫播出前熱度預測（Pre-release Anime Popularity Prediction）
- **注意**：不使用「Cold-Start」這個詞，因為那是推薦系統術語，與本研究定位不符
- **研究問題核心**：在動畫正式播出前，僅憑播出前可取得的多模態資訊（PV、封面圖、劇情大綱、元資料），能否有效預測首播後的熱度？

---

### 2. 預測目標（Y）— 兩個獨立模型

| 模型 | 預測目標 | 資料來源 | 處理方式 | 任務類型 |
|---|---|---|---|---|
| **模型 A** | 首周 popularity 數量 | AniList MediaTrends Day7 的 popularity | log(Y+1) 處理長尾 | 回歸 |
| **模型 B** | 首周平均評分 | AniList MediaTrends Day7 的 averageScore | 直接回歸 | 回歸 |

#### 為什麼選 Popularity（模型 A）而非純 Score？

- **Popularity** 反映「有多少人追蹤」，直接對應業界 IP 合作的商業決策需求
- **Score** 反映「觀眾喜愛程度」，對內容創作端（製作公司、編劇）更有價值
- 兩者相關係數約 0.4~0.6，**不可互相替代**
- 現有論文（Armenta-Segura & Sidorov, 2025, PeerJ）預測 MAL score，本研究改預測 popularity，填補研究空缺

#### 為什麼同時做 Score（模型 B）？

- 播出前不知道 score，因此預測 score 也是合理的研究目標
- 業界 IP 合作需要雙重判斷：popularity 高代表觸及人數廣，score 高代表品質有保證
- 兩個模型共用同一份前處理資料，X 完全相同，只有 Y 不同

#### 為什麼不做 Multi-task Learning？

- 可能產生 **Negative Transfer**：兩個任務的梯度方向互相干擾，導致雙輸
- 實驗複雜度倍增（loss weight 調整、task-specific adapter 等）
- 獨立訓練更乾淨，各自的 Ablation Study 更清晰
- Multi-task Learning 列為 Future Work

#### 為什麼用回歸而非分類？

- Popularity 增量嚴重右偏（長尾分佈），採用 **log(Y+1) 轉換**後接近常態，回歸可行
- 分類（熱門/普通/冷門）雖解決長尾，但會損失「熱門區間內的人數差距」資訊
- 業界需要具體的量級估計，不只是「熱門/不熱門」的標籤
- log 轉換是 popularity 預測的標準做法，有充分文獻支持

---

### 3. 輸入特徵（X）

所有特徵均為**播出前可取得**的資訊。

| 類別 | 特徵內容 | 來源 | 備註 |
|---|---|---|---|
| **文字（Text）** | 劇情大綱（Synopsis） | AniList / MAL | 使用預訓練語言模型編碼 |
| **圖片（Image）** | 主視覺封面圖（Cover Image） | AniList | 使用預訓練視覺模型編碼 |
| **影片（Video/PV）** | 宣傳影片影像 + 音訊特徵 | YouTube | 自行抽取，用預訓練模型 |
| **表格元資料（Tabular）** | 製作公司等級、是否為續作、原作類型、季度競爭強度 | AniList / MAL | 結構化特徵 |
| **IP 歷史特徵（Tabular）** | 前作 AniList popularity（最終靜態值）、前作 score | AniList API | 無前作則填 0 或 null |

#### 關於 IP 知名度特徵的設計說明

- IP 知名度（前作 popularity）與 Y 的相關性極高，但**不能忽略**，因為它是真實影響首周熱度的重要因素
- 設計上將其放入 Tabular 特徵，讓 Ablation Study 可以測試「拿掉 IP 特徵後，多模態特徵單獨的貢獻」
- 這對**原創動畫預測**（無前作資料）的情境格外重要

---

### 4. 資料集設計

#### 主要資料來源

| 資料 | 來源 | 取得方式 |
|---|---|---|
| 首周 popularity 絕對值 | AniList MediaTrends API | 自行建立（GraphQL 抓取 Day7 數據） |
| 首周 averageScore | AniList MediaTrends API | 同上（抓取 Day7 數據） |
| 封面圖、大綱 | AniList API | 同上 |
| 前作 popularity、score 等靜態資料 | AniList API | 同上 |
| PV 影片 | YouTube | 自行下載並抽取特徵 |

#### 時間範圍

- **起點**：2019 年春季番（4 月）
- **原因**：AniList MediaTrends 功能於 2019 年 3 月 19 日（API v2.5.0）推出，春季番是第一個有完整首日資料的季度
- **終點**：現在（2026 年）
| 預計涵蓋 | 約 28 季、1,400+ 部 TV 動畫（資料限制於開播後 Day7 的數量） |

#### 資料代表性說明（Limitation）

- AniList 用戶以英語系西方觀眾為主，**不直接代表台灣、日本、東南亞市場**
- 台灣（巴哈姆特動畫瘋）、日本（NicoNico）、中國（Bilibili）等亞洲平台均無公開時間序 API
- 此為本研究明確的 Limitation，未來工作可探索亞洲平台資料的取得

---

### 5. 實驗設計（Ablation Study）

兩個模型（Popularity、Score）各自獨立進行以下三組消融實驗：

| 實驗組 | 輸入特徵 | 目的 |
|---|---|---|
| **Baseline** | Tabular + 文字（大綱） | 代表現有業界基本的「IP + 故事劇情」評估水準 |
| **加入靜態主視覺 (+Image)** | Tabular + 文字 + 封面圖 | 測試靜態主視覺圖（視覺設計）帶來的額外貢獻 |
| **完整模型 (+PV)** | Tabular + 文字 + 封面圖 + PV（影片+音訊） | 測試動態 PV 特徵的極限增益 |

**核心假設**：在已知 IP 熱度的情況下，加入多模態特徵（特別是 PV）能顯著降低預測誤差。

---

### 6. 評估指標

#### 模型 A（Popularity 回歸）

| 指標 | 說明 |
|---|---|
| MAE | 主要指標，直觀反映平均預測誤差（log 空間） |
| RMSE | 對大誤差更敏感 |
| MAPE | 相對誤差，適合跨規模比較（小眾 vs 熱門） |
| Pearson / Spearman | 與 PeerJ 論文直接可比 |

#### 模型 B（Score 回歸）

| 指標 | 說明 |
|---|---|
| MAE | 主要指標 |
| RMSE | 輔助指標 |
| R² | 解釋變異量，與 PeerJ 論文直接可比（他們 R²=0.142） |

---

### 7. 業界應用框架（雙重檢定）

兩個模型的預測結果可結合為一個決策矩陣：

```
                    Score 預測
                  低          高
              ┌──────────┬──────────┐
Popularity 高 │  謹慎合作  │ 強烈建議  │
              ├──────────┼──────────┤
Popularity 低 │ 不建議合作 │ 利基市場  │
              └──────────┴──────────┘
```

- **高 Popularity + 高 Score**：大眾爆紅 + 品質保證 → 強烈建議 IP 合作
- **高 Popularity + 低 Score**：話題性強但品質存疑 → 謹慎合作，避免品牌形象受損
- **低 Popularity + 高 Score**：小眾精品 → 利基市場，針對特定客群
- **低 Popularity + 低 Score**：不建議合作

---

### 8. 與 PeerJ 論文的差異定位

| 面向 | Armenta-Segura & Sidorov (2025) | 本研究 |
|---|---|---|
| 預測目標 Y | MAL score（靜態評分） | AniList 首周 popularity 數量 + 首周 averageScore |
| 模態 | 文字 + 圖片 | 文字 + 圖片 + **PV（影片 + 音訊）** |
| 模型架構 | GPT-2 + ResNet-50 | 待定（更現代的預訓練模型） |
| 時間序 | 無 | 本研究改為僅抓取 Day7 數據 |
| 資料集 | MAL 靜態快照 | AniList 首周資料（自建） |
| 研究問題 | 能否用有限特徵預測評分？ | 多模態特徵能否預測首播熱度？ |

---

## 📝 提案四大主軸（待撰寫）

### 1. 研究背景與動機

- **任務與問題定義 (What is it about?)**：
  本研究定義了一項「新番動畫播出前熱度預測 (Pre-release Anime Popularity Prediction)」任務。在動畫正式開播前 (無歷史觀看數據的冷啟動狀態下)，模型僅依賴播出前釋出的多模態資訊 (劇本大綱 Text、主視覺圖 Image、宣傳預告片 Video) 以及結構化的 IP 歷史/元資料 (Tabular)，預測動畫首播後第 7 天的熱度 (Popularity) 與評分 (Score)。

- **應用重要性與商業價值 (Why it is important?)**：
  精準的「播出前熱度預測」對動畫產業鏈具有決定性的經濟價值，至少包含以下五個面向：
  1. **串流平台版權採買**：協助平台方 (如巴哈姆特、Crunchyroll) 評估獨家播映權的溢價空間，優化採購預算 ROI。
  2. **伺服器與頻寬配置**：透過預測首週流量，平台能提前調度運算資源，避免熱門大作首播時導致系統崩潰。
  3. **品牌廣告精準投放**：幫助廣告商衡量置入性行銷或片頭尾廣告的曝光潛力，制定合理的廣告定價策略。
  4. **周邊商品量產備貨**：授權商品製造商可依據熱度預測，調整初期首刷產品的備貨量以降低庫存風險。
  5. **行銷資源動態分配**：製作委員會能根據預測結果檢視事前宣傳成效，及早加碼行銷預算或轉換行銷媒體策略。

- **現有方法不足與本研究之突破 (Why current methods are not good enough?)**：
  傳統的發行商多依賴經驗法則，或僅將「原作銷量 (IP 知名度)」、「製作團隊陣容 (如 Ufotable、MAPPA)」作為唯二的評估指標。然而，這種方式在面對「原創動畫」或「由二線工作室改編之黑馬」時往往失效，且容易忽略實際視覺呈現。
  本研究的核心突破在於**引入動態的 PV (Video) 多模態特徵**與**靜態圖文進行融合 (Multimodal Fusion)**。相比於純文字與靜態圖片，PV 能提供觀眾決定「追番」的關鍵動態資訊：包含畫風流暢度、分鏡節奏、音樂氛圍與最終畫面品質。本研究欲透過消融實驗 (Ablation Study)，證明在控制了 IP 知名度等元資料後，多模態特徵能顯著彌補現有靜態評估機制的盲點。

### 2. 文獻回顧與研究現況

本研究之開展建立在以下幾個關鍵學術基礎上：

- **2.1 動畫領域之靜態多模態預測**：
  近期研究開始探索多模態特徵在動漫領域的預測潛力。最為核心的對標文獻為 Armenta-Segura & Sidorov (2025, PeerJ)，該研究使用 Text (GPT-2) 與 Image (ResNet-50) 預測 MAL 的最終靜態評分 (Score)。
  *侷限性與突破*：該研究證實了圖文特徵的有效性，但忽略了作畫與配樂等動態影音元素。且其預測目標為長期累積靜態評分，無法滿足產業對於「開播前（冷啟動）首週熱度爆發力」的商業預測需求。本研究引入 PV 影片模態並預測首週 Popularity 以填補此一實務空白。

- **動態影像（Video/PV）於影視預測之潛力**：
  在真人電影與短影音領域，SOTA 研究（如 Liu et al., 2023, IEEE TMM）已證實，透過 3D-CNN 或 VideoMAE 萃取預告片 (Trailer/PV) 的動態視覺美學與音軌情感波動，能大幅提升電影首週末票房的預測準確率。
  *侷限性*：動漫的 PV 剪輯邏輯（如 OP/ED 試聽、特定作畫高光展示）與真人電影存在顯著差異，且現有文獻缺乏針對動漫 PV 進行深度特徵萃取的專門評估。

- **本研究之切入點 (Research Gap)**：
  現有研究要麼專注於動漫的靜態圖文預測，要麼針對真人影視的動預告片分析。本研究精準填補了此一多模態在動漫預測的空白，引入 **PV 預告片的高維時空特徵 (Spatio-temporal features)**，並結合嚴謹的 **表格控制變數 (Tabular Control Variables)**，建立一專為「動漫首週熱度與評分」設計的雙軌冷啟動基準模型 (Dual-track Baseline)。
  *回應「PV 詐欺（宣傳包裝過度）」疑慮*：本研究藉由建立「獨立雙預測模型」——分別預測 Popularity（吸引力規模）與 Score（品質評價），以解耦 PV 之不同特質。若 PV 剪輯華麗帶來話題性，將反映於 Popularity 預測模型對該特徵之高敏感度；若其實際品質能如實兌現口碑，則將在 Score 預測模型中被驗證。透過消融比對此雙軌預測結果，可有效釐清「行銷吸引力」與「真實作畫品質」對觀眾決策之分離效益。

### 3. 研究設計與資料集

**3.1 資料集與預處理 (Dataset & Preprocessing)**
本研究鎖定 2019 年春季至 2026 年的 TV 動畫，基於 Kaggle 之開源資料集 `calebmwelsh/anilist-anime-dataset`，獲取靜態的播前元資料 (如 `description`, `coverImage`, `trailer_id`, `studios`, 與前作 `popularity`)。為彌補其缺乏時序資料之限制並預測動態爆發力，本研究將自行開發爬蟲，透過 AniList GraphQL API 獲取開播首週 (Day 7) 的真實 `popularity` 與 `averageScore` 作為目標變數 $Y$。
針對嚴重的右偏分佈問題，對 Day 7 `popularity` 進行 $\log(Y+1)$ 轉換，確保模型訓練的穩定性。

**3.2 多模態特徵萃取 (Multimodal Feature Extraction)**
> **[狀態：待後續文獻調查與實驗環境決策後定案]**
> 針對以下模態，需根據運算資源 (GPU VRAM) 與文獻支持度進行具體的預訓練模型 (Pre-trained Models) 選型調查：
> *   **Text (文字大綱)**：要使用輕量的 `BERT` / `RoBERTa`，還是更大更通用的 `GPT` 系列？
> *   **Image (主視覺圖)**：要使用傳統的 `ResNet-50/101`，還是視覺 Transformer (`ViT`)？
> *   **Video (YouTube PV)**：(最困難且核心) 傳統方法是抽取單張影格用 ResNet 算平均 (喪失時間資訊)；現代方法則使用 3D 卷積萃取時空特徵 (如 `I3D`, `SlowFast`)，或基於 Transformer 的影片模型 (如 `VideoMAE`)。應選擇哪一種？
> *   **Tabular Encoder**: 將製作公司等級、IP 歷史熱度等控制變數進行 Embedding。

**3.3 預測模型與消融實驗設計 (Ablation Study Framework)**
建立雙軌回歸模型，分別以 MSE (Mean Squared Error) 訓練預測 Day 7 $\log(\text{Popularity})$ 與 Score。為驗證 PV 動態特徵的有效性，實驗分為三組：
1.  **Baseline (Tabular + Text)**: 輸入控制變數與劇情大綱 (代表業界最基本的「IP/製作陣容 + 故事設定」評估水準)。
2.  **加入靜態視覺 (+Image)**: Tabular + Text + Image (評估加入靜態主視覺圖後，對觀眾吸引力的增益)。
3.  **Full Model (+PV)**: Tabular + Text + Image + Video/Audio PV (檢驗動態視聽資訊帶來的最終極限增益)。

**3.4 評估指標 (Evaluation Metrics)**
針對「開播前熱度 (Popularity)」與「評分 (Score)」雙軌預測的迴歸任務，本研究採用以下指標來評估模型表現與消融實驗的成效：
*   **MAE (Mean Absolute Error)**：直觀反映預測值與真實值之間的絕對誤差大小，特別適合用來評估 Score（常見於 1~10 或 1~100）的預測準確度。
*   **RMSE (Root Mean Squared Error)**：對預測偏差較大的極端值（Outliers）更敏感，可檢驗模型是否在少數「出圈爆紅」或「翻車墜海」的極端作品中產生嚴重誤判。
*   **MAPE (Mean Absolute Percentage Error)**：衡量預測誤差的相對比例。由於 Popularity（觀看/追蹤人數）跨距極大，即使經過 $\log$ 轉換，使用 MAPE 仍有助於衡量模型在不同量級作品上的等效預測能力。
*   **$R^2$ (決定係數)**：衡量模型所能解釋的變異比例。用於對比 Baseline 與 Full Model，可以清晰量化「加入 PV 或圖片資訊後，對整體預測變異的解釋力提升了多少 %」。

### 4. 預期成果與挑戰限制

- **預期成果與實務貢獻 (Expected Outcomes)**：
  1. **提升冷啟動預測之精確度 (Accuracy Improvement)**：透過引入高維度的 PV 動態特徵（Spatio-temporal Features），本研究預期能顯著降低僅依賴 Tabular 與靜態圖文模型的預測誤差 (MAE / RMSE)，實現動漫首播熱度的高精準測。
  2. **消融實驗之量化驗證 (Ablation Study Validation)**：本研究設計了嚴謹的三組消融實驗（Baseline 基礎大綱、+Image 靜態與 +PV 動態），其預期成果能將各模態的貢獻「量化拆解」。不僅能直接證明加入影像特徵的階段性有效性，更能向業界明確展示「動態行銷宣傳 (PV)」與「先天 IP 暨劇情設定 (Tabular+Text)」在驅動初期熱度上的權重差異，為未來的行銷資源分配提供科學化數據支持。
  3. **實務決策框架 (Business Decision Matrix)**：基於 Popularity 與 Score 的雙軌預測結果，產出如同前述之「大眾爆紅 vs. 利基精品」的決策矩陣，直接賦能串流平台的版權採購策略。

- **技術挑戰 (Technical Challenges)**：
  1. **異質多模態融合 (Heterogeneous Multimodal Fusion)**：將低維度的 Tabular (如製作公司) 與極高維度的 Video (如 3D-CNN 萃取之時空特徵) 進行有效融合是極大的神經網路設計挑戰。若 Fusion 策略 (如 Early vs. Late fusion、Attention-based fusion) 調校不當，高維度特徵極易淹沒低維度的關鍵控制變數，導致模型退化。
  2. **影像資料工程與運算瓶頸**：大規模自 YouTube 爬取並預處理上千部動畫之 PV 影片，以及使用預訓練模型萃取幀級序列特徵，對硬體 (GPU VRAM) 與儲存空間將是一大考驗。

- **研究限制 (Scope Limitations)**：
  1. **區域受眾偏差 (Geographic & Cultural Bias)**：本研究之資料集以 AniList 為核心，其受眾組成以歐美西方觀眾為主。因此，模型的預測結果可能無法完全代表日本本土 (NicoNico) 或台灣 (巴哈姆特動畫瘋) 市場的真實喜好。
  2. **獨立預測之侷限**：受限於本階段對「負遷移 (Negative Transfer)」風險之控管，研究暫以兩獨立模型分別預測 Popularity 與 Score。未來工作 (Future Work) 可進一步探討 Multi-task Learning (多任務學習) 架構，測試其在同時預測雙目標時的模型效能。

---

## 🔑 關鍵字（Keywords）

1. `Multimodal Representation Learning`（多模態表徵學習）
2. `Feature Fusion`（特徵融合）
3. `Ablation Study`（消融實驗）
4. `Anime Pre-release Popularity Prediction`（動漫播出前熱度預測）
5. `IP Coordination Strategy`（IP 合作策略輔助）

---

## 📚 參考文獻與必讀清單 (Reference Checklist)

### 一、直接相關：動漫評分與人氣預測
- 🔴 **必讀**: Armenta-Segura, E., & Sidorov, G. (2025). *Anime popularity prediction before huge investments: a multimodal approach using deep learning*. PeerJ Computer Science.
- 🔴 **必讀**: *Unveiling Anime Preferences: A Data-driven Analysis using MyAnimeList API*.
- 🔴 **必讀**: Liu et al. (2025). 動漫評分預測相關研究 (arXiv:2501.01422).
- 🔴 **必讀**: Ryu et al. (2024). 播出前基於 PV 之預測研究 (JIIS).

### 二、PV / Trailer 影像與音訊特徵提取
- 🟡 **建議**: *Movie Trailer Deep Features* (JAIT V15N6, 2024).
- 🟡 **建議**: BIIC Lab NTHU. *Trailer + Reactor Expressions* (2024).

### 三、多模態融合架構 (Multimodal Fusion)
- 🔴 **必讀**: *Bag of Tricks Multimodal AutoML* (arXiv:2412.16243, 2024).
- 🟡 **建議**: *Cross-modal Transformer, Box Office* (ACM CTNN, 2024).

### 四、串流平台內容人氣預測
- 🟡 **建議**: *AI-based Popularity Prediction of TV Shows* (IJRASET, 2026).
- 🟡 **建議**: *Streaming Popularity with NLP* (PMC, 2021).

---

*最後更新：2026-04-11*
