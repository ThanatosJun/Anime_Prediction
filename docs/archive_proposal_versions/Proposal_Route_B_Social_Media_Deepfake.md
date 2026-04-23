# 期末研究提案摘要（路線 B：社群媒體深偽檢測與假訊息防制）

## 主題（Title）
**"When Both Sides Lie in the Wild: Vulnerability Analysis of Multimodal Deepfake Detection on Social Media"**
（當雙方都在野外說謊：社群媒體環境下多模態深偽檢測之脆弱性分析）

## 1. 研究關鍵字 (Keywords)
* **Media Forensics**（多媒體鑑識）
* **Audio-Visual Deepfake Detection**（影音深偽偵測）
* **Misinformation / Disinformation**（假訊息防制）
* **In-the-Wild Forgery**（野外真實偽造 / 非受控環境偽造）
* **Half-Truth Manipulation**（半真半假操作）

## 2. 背景介紹 (Background)
**【深偽技術武器化與假訊息風暴】**
隨著生成式 AI 門檻降低，多模態深偽（Deepfake）技術已成為假訊息傳播的強大武器。從變造名人影片推銷虛擬貨幣引發的金融詐騙，到選舉期間利用政治人物深偽影音進行的認知作戰，這些流竄於社群平台的惡意內容嚴重破壞了社會信任。

**【半真半假攻擊與多模態結合】**
近期的攻擊手法已從單純的「全臉替換」進化為「聯合攻擊（Joint Spoofing）」與「半真半假（Half-Truth）」操作。例如：擷取一段真實的演講影片，利用語音轉換（Voice Conversion）生成虛假聲明，再輔以唇音同步（Lipsync）技術微調嘴型。這種結合真實背景與局部偽造的高級手法極難被一般群眾肉眼識破。

**【現有檢測模型的社群媒體水土不服（研究動機）】**
現有的單模態深偽檢測模型（如專注臉部的 CNN 或專注聲音的 SSL 模型）在實驗室的高畫質無損資料集上表現優異。然而，當這些偽造影音被上傳至 X、TikTok 或 Instagram 等社群平台後，經過多重演算法壓縮、降度轉檔與背景噪音渲染，微小的 AI 生成痕跡（Artifacts）會被徹底抹除。這些模型面對「雙方都在說謊」且「被社群平台破壞痕跡」的影音時，準確率往往雪崩式下滑。本研究旨在剖析現有模型在高度降級（Degraded）的社群平台環境中的脆弱性，並探討跨模態一致性之防禦潛力。

## 3. 研究困境 (Research Challenges)
1. **社群平台的強烈破壞性（Severe Degradation in the Wild）：** 影片在社群媒體上流傳時會經歷嚴重的破壞性壓縮與失真，這些自然產生的雜訊會偽裝或覆蓋掉 Deepfake 模型的生成瑕疵，導致依賴微觀紋理特徵的檢測器失效。
2. **半真半假的聯合攻擊（Half-Truth Manipulations）：** 當影片大部分畫面（包含光影、背景）都是真實的，只有嘴唇與聲音被局部竄改，這種高相似度的偽造會讓許多依賴全域特徵（Global features）的模型判斷為真，極大化了檢測難度。
3. **對未知生成演算法的泛化能力（Cross-dataset Generalization）：** 偽造技術日新月異，防禦模型在 A 資料集（特定生成演算法）訓練後，一旦在社群平台上遇到未知的 B 演算法生成的影片，往往無法成功檢測。

## 4. 資料集與收集方式 (Datasets & Data Collection)
為契合社群媒體真實環境，本研究將直接採用包含極高噪音與多樣生成演算法的最新開源基準：
* **核心多模態檢測源（In-the-wild Spoofing Source）：** 
  * 優先採用 **Deepfake-Eval-2024**（目前最大的多模態真實 Deepfake benchmark，涵蓋 44 小時影片與 56.5 小時音訊），該資料集特色在於全數由真實社群媒體流通收集，完美對應 In-the-wild 情境（自 HuggingFace 取得）。
  * 輔以 **MultiFF (Inclusion 2024)**：提供高達 80+ 種生成演算法的多樣性，強攻模型對未知演算法的泛化能力。
* **真實語音與局部偽造對照（Half-Truth Verification）：** 使用 **In-the-Wild** 資料集（包含真實名人聲音與網路流通偽造版本），並引入 ACL 2025 的 **SpeechFake**（支援各類語音合成與編碼器）作為單一維度的壓力測試。
* **資料實驗處理：** 無須如身分驗證般刻意去對齊時間偏移，而是直接針對社群媒體環境設定不同的「壓縮率（Compression Rate）與噪音比（SNR）」介入，測試模型在不同降級程度下的性能衰退曲線。