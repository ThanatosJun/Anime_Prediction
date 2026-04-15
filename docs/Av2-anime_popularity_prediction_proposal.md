# 期末研究提案 Av2：新番動畫播前熱度與評分預測之基準研究（不含 PV 模態）
**(A Baseline Study of Pre-release Anime Popularity and Score Prediction without Trailer Modality)**

> 本版本以文字、封面影像與結構化元資料為主要輸入，旨在建立可重現且具實務可解釋性的播前預測基準。

---

## Introduction

在動畫產業的實務流程中，產品端（例如品牌方、授權商品商、通路平台與行銷團隊）對 IP 合作的評估與談判，通常發生於作品正式播出之前的宣傳與授權規劃階段。此階段的核心決策並非「作品已經紅了是否跟進」，而是「在資訊有限條件下，是否提前投入資源布局合作」。因此，若缺乏可操作的播前量化工具，合作決策往往高度依賴經驗法則或單一先驗訊號（如原作知名度、製作公司聲量），容易造成高估或低估，進而影響授權成本、行銷配置與產品上市時機。

從學術研究脈絡觀察，現有 SOTA 文獻雖已證實多模態特徵對內容熱度或評價預測具有潛力，但仍存在三項缺口。第一，時間節點仍有落差：部分研究採用評論或播後互動等訊號，與播前決策所需的資訊可得性並不一致；另有研究即使使用播前資訊，仍主要對整體成效進行單軸預測，尚未直接對齊「播前合作決策」的雙指標需求。[A-2][A-1] 第二，任務設定多為單一目標：部分研究採二元成功判定，另一些研究則屬單一主目標建模，尚未同時處理觸及規模（popularity）與內容品質（score）之雙目標判讀。[A-2][A-3][A-1] 第三，IP 合作研究多偏向策略分析與質性歸納，主要提供產業機制與成功因素分析，仍缺少可直接支援作品級合作評估的量化預測框架。[C-1][C-2][C-3] 基於上述缺口，本研究將問題明確界定為播前雙目標回歸任務，並以可重現的 Text + Image + Tabular 基準模型作為方法起點。[A-1][A-7]

本研究的核心目的在於：在動畫正式上線前，建立可解釋且可驗證的預測機制，以提前辨識「具高效益產品 IP 合作機會」之候選作品。換言之，本研究不僅關注模型精度本身，更強調模型輸出如何轉化為合作決策訊號，協助產業在有限時間與資源下進行更具依據的前置配置。

### Introduction 所參考之 研究（Citation Mapping）

| Introduction 核心主張 | 對應 研究 | 對應理由（本研究如何引用） |
|---|---|---|
| 播前特徵可用於人氣/評分預測，但仍有可預測性上限 | Armenta-Segura, E., & Sidorov, G. (2025). *Anime popularity prediction before huge investments: a multimodal approach using deep learning*. PeerJ Computer Science. | 直接支撐「Anime 領域播前多模態預測可行」，同時說明關聯能力仍有限，合理化本研究需建立更聚焦於決策場景的基準。 |
| 現有研究常以單一模態或單一任務為主，跨模態與可重現設計仍需系統化 | *Prediction of Anime Series' Success using Sentiment Analysis and Deep Learning* (WiDSTaif 2021); *Anime Success Prediction Based on Synopsis Using Traditional Classifiers* (2023); *Bag of Tricks for Multimodal AutoML with Image, Text, and Tabular Data* (arXiv 2024). | 前兩者作為 Anime 領域文本向基線（偏單模態/單任務），後者作為多模態融合與消融方法論參考，支撐 Av2 採 Text + Image + Tabular 的可重現路線。 |
| 人氣形成受外部脈絡影響，且強先驗可能造成偏誤 | *Model Can Be Subtle: Two Important Mechanisms for Social Media Popularity Prediction* (ACM TOMM 2025). | 支撐 Introduction 中對「模型可能過度依賴先驗訊號」的警示，並作為後續偏誤控制與子群評估設計依據。 |
| IP 合作研究多為策略與質性分析，缺少可直接支援播前決策的量化框架 | *Anime IP in China: A Comprehensive Review of Academic Endeavors* (ADDT 2024, EAI); *Research on the Success Elements of Animation IP: Creativity, Marketing, and Globalization* (Scientific and Social Research, 2025); *The Growth Impact of Japanese Animation IP and its Related By-industries on Its Economy* (ICEMED 2025, Atlantis Press). | 三篇共同支撐「產業價值論述充足、量化決策工具不足」之研究缺口，對應本研究提出播前雙目標回歸框架的動機。 |

### Introduction 所參考之研究（標籤對應清單）

[標籤對應文獻]
- [A-1] Armenta-Segura, E., & Sidorov, G. (2025). Anime popularity prediction before huge investments: a multimodal approach using deep learning. PeerJ Computer Science.
- [A-2] Prediction of Anime Series' Success using Sentiment Analysis and Deep Learning (WiDSTaif 2021).
- [A-3] Anime Success Prediction Based on Synopsis Using Traditional Classifiers (2023).
- [A-7] Bag of Tricks for Multimodal AutoML with Image, Text, and Tabular Data (arXiv 2024).
- [C-1] Anime IP in China: A Comprehensive Review of Academic Endeavors (ADDT 2024, EAI).
- [C-2] Research on the Success Elements of Animation IP: Creativity, Marketing, and Globalization (Scientific and Social Research, 2025).
- [C-3] The Growth Impact of Japanese Animation IP and its Related By-industries on Its Economy (ICEMED 2025, Atlantis Press).

[三項缺口對應標籤]
- 缺口一（時間節點不一致）：[A-2][A-1]
- 缺口二（單一目標限制）：[A-2][A-3][A-1]
- 缺口三（IP 合作量化不足）：[C-1][C-2][C-3]
- 方法起點（可重現基準）：[A-1][A-7]

註：完整摘要與限制分析請對照 All_Papers_Summary 的對應編號段落。
> 註：上述文獻之完整清單與分組，請對照本文件「參考文獻與必讀清單（Av2）」段落。

---

## 核心研究設定（Av2）

### 1. 研究任務定義

- **任務名稱**：新番動畫播出前熱度預測（Pre-release Anime Popularity Prediction）
- **注意**：不使用「Cold-Start」一詞（避免與推薦系統任務混淆）
- **研究問題核心**：在動畫正式播出前，僅憑播前可取得的文字、封面圖與結構化元資料，能否有效預測首播後第 7 天熱度與評分？
- **Av2 差異**：本版本暫不納入 PV，以降低資料工程負擔與特徵不穩定風險，優先建立可驗證之基準模型

---

### 2. 預測目標（Y）— 兩個獨立模型

| 模型 | 預測目標 | 資料來源 | 處理方式 | 任務類型 |
|---|---|---|---|---|
| **模型 A** | 首周 popularity 數量 | AniList MediaTrends Day7 的 popularity | log(Y+1) 處理長尾 | 回歸 |
| **模型 B** | 首周平均評分 | AniList MediaTrends Day7 的 averageScore | 直接回歸 | 回歸 |

#### 選擇 Popularity（模型 A）而非僅預測 Score 之理由

- **Popularity** 反映受眾觸及規模，與 IP 合作與資源配置決策具有直接對應性
- **Score** 反映內容滿意度，對製作端內容策略具有補充價值
- 兩者雖具相關性，惟在決策層面不可互為替代

#### 同時預測 Score（模型 B）之必要性

- 播前決策不僅關注觸及規模，亦需評估內容品質風險
- 雙目標設計可形成更完整之合作決策矩陣

#### 暫不採用 Multi-task Learning 之理由

- 降低任務間負遷移（Negative Transfer）之風險
- 控制訓練與超參數搜尋複雜度，優先確保單任務可重現性
- Multi-task Learning 規劃於後續延伸研究

#### 採用回歸而非分類之理由

- 可保留熱門作品間之量級差異資訊
- 較符合實務端對連續數值預測之需求
- 對 popularity 採用 $\log(Y+1)$ 轉換可緩解長尾分布問題

---

### 3. 輸入特徵（X）

所有特徵均為**播出前可取得**。

| 類別 | 特徵內容 | 來源 | 備註 |
|---|---|---|---|
| **文字（Text）** | 劇情大綱（Synopsis） | AniList / MAL | 預訓練語言模型編碼 |
| **圖片（Image）** | 主視覺封面圖（Cover Image） | AniList | 預訓練視覺模型編碼 |
| **表格元資料（Tabular）** | 製作公司等級、是否續作、原作類型、季度競爭強度 | AniList / MAL | 結構化特徵 |
| **IP 歷史特徵（Tabular）** | 前作 popularity（最終靜態值）、前作 score | AniList API | 無前作以缺失值策略處理 |

#### IP 知名度特徵之處理原則

- IP 特徵與目標變數相關性高，故不可省略
- 需透過消融實驗檢驗其與內容模態之相對貢獻
- 針對原創動畫（無前作）需額外檢驗模型穩健性

---

### 4. 資料集設計

#### 主要資料來源

| 資料 | 來源 | 取得方式 |
|---|---|---|
| 首周 popularity | AniList MediaTrends API | GraphQL 抓取 Day7 |
| 首周 averageScore | AniList MediaTrends API | GraphQL 抓取 Day7 |
| 封面圖、大綱 | AniList API | API 擷取 |
| 前作 popularity、score 等 | AniList API | API 擷取 |

#### 時間範圍

- **起點**：2019 年春季番（MediaTrends 可用後第一個完整季度）
- **終點**：2026 年
- **預計規模**：約 28 季、1,400+ 部 TV 動畫（以可取得 Day7 樣本為準）

#### 資料代表性限制

- AniList 受眾偏歐美，不完全代表台灣、日本、中國市場
- 亞洲平台高品質時間序資料開放度有限

---

### 5. 實驗設計（Ablation Study, Av2）

兩個模型（Popularity、Score）各自進行以下三組實驗：

| 實驗組 | 輸入特徵 | 目的 |
|---|---|---|
| **E1: Tabular Only** | Tabular（含 IP 歷史） | 建立可解釋之結構化基線 |
| **E2: +Text** | Tabular + Text | 檢驗文本語意特徵之增益 |
| **E3: +Image（Av2 Full）** | Tabular + Text + Image | 檢驗視覺特徵對誤差降低之邊際效果 |

**核心假設（Av2）**：在控制 IP 先驗訊號後，加入 Text 與 Image 可顯著提升首周 popularity 與 score 之預測效能。

---

### 6. 評估指標

#### 模型 A（Popularity 回歸）

| 指標 | 說明 |
|---|---|
| MAE | 主要指標 |
| RMSE | 檢驗大誤差敏感性 |
| MAPE | 比較不同量級作品之相對誤差 |
| Pearson / Spearman | 評估線性與排序關聯 |

#### 模型 B（Score 回歸）

| 指標 | 說明 |
|---|---|
| MAE | 主要指標 |
| RMSE | 輔助指標 |
| R² | 解釋變異量 |

---

### 7. 實務應用框架（雙重檢定）

```
                    Score 預測
                  低          高
              ┌──────────┬──────────┐
Popularity 高 │  謹慎合作  │ 強烈建議  │
              ├──────────┼──────────┤
Popularity 低 │ 不建議合作 │ 利基市場  │
              └──────────┴──────────┘
```

- **高 Popularity + 高 Score**：具高觸及與高品質，適合作為優先合作標的
- **高 Popularity + 低 Score**：具話題擴散潛力，惟需搭配品質風險控管
- **低 Popularity + 高 Score**：屬小眾高口碑型內容，適合利基市場策略
- **低 Popularity + 低 Score**：不建議進行高成本合作投入

---

### 8. 與既有文獻差異定位（Av2）

| 面向 | 既有代表研究（例：PeerJ 2025） | 本研究 Av2 |
|---|---|---|
| 預測目標 | 多為最終靜態評分或單一目標 | Day7 popularity + Day7 score 雙目標 |
| 模態 | 常見 Text + Image 或任務特化組合 | Text + Image + Tabular（不含 PV） |
| 實驗重點 | 著重模型效果 | 著重可重現基線 + 模態邊際貢獻 |
| 決策輸出 | 多為學術指標 | 連結 IP 合作決策矩陣 |

---

## 提案四大主軸（Av2 版）

### 1. 研究背景與動機（Overview & Motivation）

本研究聚焦於播前決策場景：在動畫上線前，平台與合作方即需完成採購、宣傳與資源配置等關鍵決策。現有 IP 合作研究多偏向策略討論，尚缺可直接支援決策之量化預測工具；另一方面，既有人氣預測研究亦未必對齊 IP 合作情境。基於此，Av2 採用 Text+Image+Tabular 架構，以建立具可重現性與可解釋性的基準模型，作為後續擴展至更高複雜度模態（如 PV）之研究起點。

### 2. 文獻回顧與研究現況（Literature Review & SOTA）

- 動畫預測文獻已顯示 Text/Image 對評分或成功機率具解釋力
- 社群與短影音文獻指出，多模態特徵與外部脈絡可提升人氣預測效能
- IP 合作與商業化研究雖已揭示跨媒體、行銷與衍生商業等成功因素，惟多缺乏量化預測框架
- **Research Gap（Av2）**：目前仍缺乏可直接支援「Anime IP 合作」場景之播前雙目標（popularity + score）量化模型

### 3. 研究設計與資料集（Proposed Study & Datasets）

- 資料：AniList Day7 popularity/score 與播前可取得之文字、影像與結構化特徵
- 模型：兩個獨立回歸模型（Popularity / Score）
- 實驗：E1/E2/E3 消融（Tabular → +Text → +Image）
- 目的：量化各模態邊際貢獻，建立可重現基線並支援實務決策情境

### 4. 預期成果、挑戰與限制（Outcomes, Challenges & Limitations）

#### 預期成果
1. 在 Day7 popularity/score 任務中，E3（Tabular+Text+Image）相對 E1/E2 顯著降低 MAE/RMSE。
2. 建立可操作之 IP 合作決策矩陣，將模型輸出轉換為決策可解釋訊號。
3. 建立可重現資料與評估流程，作為 Av3（含 PV）擴充基礎。

#### 技術挑戰
1. 強先驗偏誤：IP 與製作公司特徵可能掩蓋內容模態訊號。
2. 長尾與不平衡：熱門樣本稀少，極端值易影響訓練穩定性。
3. 跨平台泛化：不同社群與地區受眾偏好差異可能導致 domain shift。

#### 研究限制
1. Av2 不使用 PV，可能遺失動態視聽訊號。
2. 以 AniList 為主，市場代表性受限。
3. 暫不處理多任務共同訓練與因果推論。

---

## 關鍵字（Keywords）

1. `Multimodal Representation Learning`（多模態表徵學習）
2. `Feature Fusion`（特徵融合）
3. `Ablation Study`（消融實驗）
4. `Anime Pre-release Popularity Prediction`（動漫播出前熱度預測）
5. `IP Coordination Strategy`（IP 合作策略輔助）

---

## 參考文獻與必讀清單（Av2）

### 一、直接相關：動漫評分與人氣預測
- Armenta-Segura, E., & Sidorov, G. (2025). Anime popularity prediction before huge investments: a multimodal approach using deep learning. PeerJ Computer Science.
- Prediction of Anime Series' Success using Sentiment Analysis and Deep Learning (WiDSTaif 2021).
- Anime Success Prediction Based on Synopsis Using Traditional Classifiers.

### 二、多模態融合與方法參考
- Bag of Tricks for Multimodal AutoML with Image, Text, and Tabular Data (arXiv 2024).
- Model Can Be Subtle: Two Important Mechanisms for Social Media Popularity Prediction (ACM TOMM 2025).

### 三、IP 合作與產業影響（研究缺口支撐）
- Anime IP in China: A Comprehensive Review of Academic Endeavors (ADDT 2024, EAI).
- Research on the Success Elements of Animation IP: Creativity, Marketing, and Globalization (SSR 2025).
- The Growth Impact of Japanese Animation IP and its Related By-industries on Its Economy (ICEMED 2025).

---

*最後更新：2026-04-15*