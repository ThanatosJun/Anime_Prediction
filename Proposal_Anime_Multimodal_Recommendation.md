# 期末研究提案：基於多模態特徵的新番動畫冷啟動推薦系統研究

**研究關鍵字 (Keywords)**：`Anime Recommendation`, `Multimodal`, `Item-Cold-Start`, `Content-based Filtering`, `Representation Learning`

---

## 一、 研究背景與動機 (Overview & Motivation)

在現代影音串流平台中，精準推薦尚未開播的新動畫（Item-Cold-Start，物品冷啟動）具有極高的商業價值。然而，傳統高度依賴使用者歷史互動紀錄的協同過濾（Collaborative Filtering）無法對無歷史數據的新項目運作。目前業界解決新番冷啟動的方案，多半退而求其次，僅使用「純文字標籤與摘要（Text Genres/Tags & Synopsis）」進行內容過濾（Content-based Filtering）。

然而，純文字標籤存在嚴重的**資訊同質化（Information Homogeneity）盲點**。以《刀劍神域》與《JOJO的奇妙冒險》為例，兩者在純文字標籤上高度重疊（皆為動作、冒險、奇幻），但在視覺呈現上卻有天壤之別——前者為大眾向精緻柔和的光影風格，後者則具備強烈美漫色彩與粗獷狂放的線條。這證明了現有推薦系統若僅依賴「純文字標籤」，在描繪視覺與氛圍特徵時將面臨嚴重盲點。

為了解決此痛點，本系統的**輸入 (Input)** 將設定為使用者的歷史互動序列，以及候選新番的多模態特徵（包含文本、靜態視覺、動態影像與音訊）；**輸出 (Output)** 則是目標使用者對特定候選新番的偏好機率值，並據此生成個人化推薦排序清單。我們期望透過引入「動態宣傳片（PV）」等多模態特徵，精準捕捉受眾在開播前的觀看意圖。

---

## 二、 文獻回顧與研究現況 (Literature Review & SOTA)

在多模態推薦系統中，不僅需要考量單一模態特徵的萃取品質，更需要強大的推薦演算法（Recommendation Algorithms）將這些異質特徵與使用者偏好進行對齊。以下盤點現有兩大領域的 SOTA 方法及應用於「動漫領域」時面臨的挑戰：

### 1. 推薦系統模型現況 (Recommendation Models)
解決推論問題的核心在於如何將 Item 特徵與 User 互動進行有效融合。
* **傳統協同過濾的侷限**：如矩陣分解（Matrix Factorization）或近期主流的 LightGCN 等 Graph-based 方法，雖能精準捕捉高密度互動資料，但其架構本質上不支援內容特徵（Content Features）的直接輸入，面對 Item-Cold-Start 時會完全失效。
* **內容注入與多模態推薦 (Feature Injection & Multimodal RecSys)**：為解決冷啟動，學界轉向能整合多模態屬性的模型。例如 **VBPR (Visual Bayesian Personalized Ranking)** 能將視覺特徵融入偏好排序；近期如 **BM3** 或 **LATTICE** 等模型，則利用對比學習（Contrastive Learning）與 Cross-Attention 機制，讓異質特徵（如圖文）在同一潛在空間中學習對齊。然而，上述模型主要針對真實世界圖片或短影音設計，鮮少涉及長篇動漫 PV 的處理。

### 2. 動漫領域之多模態特徵工程現況 (Multimodal Feature Engineering)
單一模態的萃取品質決定了推薦模型預測效能的上限。以下盤點五大維度的方法及挑戰：

1. **文字特徵處理 (Text)**：
   * **現況**：使用預訓練語言模型（如 BERT, Sentence-BERT）將故事摘要轉為稠密向量。
   * **動畫領域侷限**：通用模型難以理解動漫專屬術語（ACG Jargon，如「傲嬌」、「異世界轉生」），未經領域微調容易產生語義流失。
2. **圖像特徵處理 (Image)**：
   * **現況**：從 CNN（ResNet）轉向 Vision Transformer (ViT)，捕捉全局空間關係。
   * **動畫領域侷限**：動畫封面常有大量排版字體。本研究擬採用專為動漫設計的預訓練模型（如 **DeepDanbooru** 或 **AnimeCLIP**），以精準提取 2D 插畫風格特徵。
3. **音訊特徵處理 (Audio)**：
   * **現況**：將音訊轉為梅爾頻譜圖後，使用 AST 或 VGGish 模型分類聲音張力。
   * **動畫領域侷限**：PV 聲音極度複雜，充滿強烈 BGM、爆炸音效與聲優台詞的混合疊加，難以單獨拆解評估。
4. **影像特徵處理 (Video/Frames)**：
   * **現況**：依賴 SlowFast (3D-CNN) 或 VideoMAE 捕捉光流與動態。
   * **動畫領域侷限 (域偏移 Domain Shift)**：2D 動畫特有的一拍三（Low-FPS，約 8幀）及非物理寫實光影，會導致依賴連續光流的通用模型效能嚴重崩跌。
5. **特徵整合方式 (Multimodal Fusion)**：
   * **現況**：從早期的串接融合（Concatenation）轉向最強的交叉注意力機制（Cross-Attention，如 ALBEF）。

**本研究切入點**：現有如 BM3 或 LATTICE 等多模態推薦系統雖解決了部分圖文推薦問題，但針對「處理 2D 低幀率動態影像與混雜音訊（Video & Audio），並將其注入推薦模型」的冷啟動研究極度稀缺。本研究將填補此一文獻缺口。

---

## 三、 研究設計與資料集 (Proposed Study & Datasets)

本研究放棄端到端（End-to-End）系統部署，將運算資源全數集中於**「多模態表徵學習與特徵工程」**，旨在驗證各項模態對預測準確度的貢獻。

### 1. 資料來源與跨模態對齊
* **互動數據與文本標籤**：採用 Kaggle 開源的 MyAnimeList (MAL) 資料集（篩選具備足夠基礎觀看紀錄的活躍用戶）。
* **PV 影片來源與對齊 (Alignment)**：透過 Jikan API 抓取 MAL 動畫 ID 對應的官方 YouTube 預告片（PV），達成 100% 準確的跨模態 ID 對齊，免除模糊比對的資料污染。

### 2. 用戶偏好建模 (User Modeling)
本研究專注於 Item 側的冷啟動問題，因此對於活躍用戶，將不另外訓練獨立且複雜的 User Embedding 塔。
* **具體作法**：採用 MAL 資料集中用戶歷史的高分或 `completed` 清單，透過對其觀賞過的 Item Embeddings 進行 Average Pooling 或簡單的 Attention 聚合，藉此建立該用戶的 **User Profile Vector**。此舉能有效控制系統複雜度，將驗證焦點鎖定在 Item 多模態特徵的萃取品質上。

### 3. 真實標籤定義 (Ground Truth)
為避免特徵（開播前 PV）與標籤（開播後完食率 `Completed`）產生時間軸錯位（Temporal Misalignment / Data Leakage），本研究將目標變數 $Y$ 設定為**「預測使用者是否將該動畫加入想看清單（Plan to Watch）」**。
* **邏輯防禦 (PV 詐欺免疫)**：此舉不僅精準對應 PV 作為「開播前行銷特徵」的商業首要任務（引起興趣），更因為預測的是「開播前的美好預期」，從而徹底免疫了「開播後作畫崩壞 / 劇情爛尾 / PV詐欺」所帶來的標籤雜訊（Noisy Labels）。

### 4. 消融實驗設計 (Ablation Study)
在保有強大且穩定的文字基底上，測試疊加不同視覺與聽覺模態的增量價值（Incremental Value）：
* **組合 1 (Baseline)**：動漫分類標籤 + 故事摘要 `(Tags + Synopsis)`
* **組合 2 (靜態視覺對照)**：組合 1 + 動畫封面圖 `(Text + Image)`
* **組合 3a (純動態視覺)**：組合 1 + 宣傳影片畫面 `(Text + Video_Visual)`
* **組合 3b (本研究之 Proposed SOTA)**：組合 1 + 完整宣傳影片 `(Text + Video_Visual + Video_Audio)`

---

## 四、 預期成果、挑戰與限制 (Expected Outcomes, Challenges & Limitations)

### 1. 預期成果與貢獻
1. **量化特徵有效性**：透過消融實驗，量化證明在支援內容特徵輸入的推薦模型（如 VBPR 或導入 Feature Injection 的演算法）架構下，加入「PV 的動態影像與聲音特徵」相較於「傳統純文字」與「靜態封面圖」，能為冷啟動推薦帶來多少 NDCG@K 或 Hit Ratio 排序指標的實質提升。尤其是組合 3a 與 3b 的對比，將能精確剝離出「音訊（Audio）」在動漫預告片中的獨立預測貢獻。
2. **建立開源對齊資料集**：總結出一套有效應對 2D 動畫 PV（低幀率視覺、音效混雜）的特徵降維與萃取管線（Pipeline），並建構、開源一份對齊 MAL 互動紀錄與 YouTube PV 的多模態動漫推薦資料集。

### 2. 技術挑戰與研究限制
1. **音訊特徵的語義對齊挑戰**：考量到動畫 PV 常將背景音樂（BGM）與聲優台詞混雜，本研究將不進行高耗能的音軌分離（如 Demucs），而是直接採用 CLAP（Contrastive Language-Audio Pretraining）或這類基於整體氛圍音頻的預訓練模型，將「整體音樂熱血度與聽覺張力」作為單一特徵。這在實作上是設計選擇，但在區分「單一台詞情緒」時仍存在侷限。
2. **多模態對齊與融合難題**：文字（靜態）、封面（空間）、影片（空間時序）與聲音（頻譜）的向量維度與特質迥異，如何設計合適的交叉注意力（Cross-Attention）機制使其在潛在空間完美融合而不互為雜訊，為實作最大門檻。
3. **時間戳記的不可考與資料限制 (Scope Limitations)**：本研究基於「開播前意圖假設」，假定 `Plan to Watch` 狀態反映了使用者在開播前受 PV 吸引的觀看意圖。惟 MAL 資料集並未提供將動畫加入該清單的確切時間戳記（無法驗證使用者是否是在開播後才加入）。此時間軸假設無法被資料直接驗證，為本研究之先天限制之一。
