# Paper Draft Sections

---

## 一、相關研究

### 1. 多模態融合

多模態融合（multimodal fusion）常用於需要同時理解文字、視覺與結構化資訊的預測任務。對於動畫作品而言，播出前可取得的訊號並不只限於作品名稱或基本 metadata；文字簡介提供劇情與主題線索，封面或宣傳圖呈現視覺風格，格式、集數、類型、製作公司與關聯作品則提供作品定位與製作脈絡。單一模態通常只能捕捉其中一部分資訊，因此多模態融合的核心價值，在於整合互補訊號，使模型能同時考慮作品內容、視覺吸引力與產業背景 [J1], [J2]。

既有 popularity prediction 研究中，常見方法包含 early fusion 與 learned fusion。Early fusion 會先取得文字、圖片與 metadata 的特徵，再直接串接後交給傳統機器學習模型或多層感知機進行預測；此方法實作穩定，適合作為 strong baseline。Learned fusion 則進一步透過 branch-wise projection、MLP、cross-modal attention 或 recurrent fusion 等機制，讓模型學習不同模態之間的交互關係。本文在 baseline 設計中同時保留這兩類方向：feature concatenation 用來檢查簡單多模態組合已能達到的性能，C1/C2 類 source-inspired fusion baselines 則用來檢查外部多模態架構在 anime pre-release prediction 任務中的參考價值 [J1], [J2]。

本文並不宣稱完整復現外部文獻模型。由於本研究使用的是 AniList anime snapshot、pre-release temporal split，以及 `popularity` / `meanScore` 雙目標回歸，與許多原始論文的資料來源、輸入模態、任務設定與切分方式不同，因此相關 baseline 被定位為 project-input adaptation、proxy 或 source-inspired reconstruction。這樣的定位可以保留文獻方法的比較價值，同時避免將不同資料條件下的改寫版本誤寫為 exact reproduction。

### 2. 檢索增強（Retrieval Augmentation）

檢索增強（retrieval augmentation）的核心想法是，單一樣本本身的文字、圖片與 metadata 不一定足以完整預測其未來表現，因此可以檢索相似的歷史樣本作為額外 context。對於動畫播出前預測而言，這個想法尤其合理：若一部新作在題材、製作公司、聲優、系列關係、文字語意或視覺風格上接近過去作品，這些歷史作品的受眾反應可能提供關於人氣與評分的先驗訊號 [J3]。

Retrieval augmentation 可依檢索依據分成 metadata-based retrieval 與 semantic retrieval。前者使用 genre、studio、source、voice actor 或 relation 等結構化欄位尋找相似作品，優點是可解釋性較高，也較容易控制時間洩漏；後者則使用文字或多模態 embedding 搜尋語意相近作品，能捕捉標籤欄位無法完整描述的內容相似性。兩者也可結合成 hybrid retrieval，以同時使用結構化與語意相似度。

本文的 retrieval 設計受到 selective retrieval knowledge augmentation 類方法啟發，但不將其寫成完整 SKAPP reproduction。原方法主要針對 social-media UGC popularity prediction，包含使用者生成內容、平台互動與特定 graph/attention 模組；而本文任務是 pre-release anime prediction，輸入契約與 target formulation 皆不同。因此，本文在 related work 中將 retrieval augmentation 定位為補充歷史上下文的研究方向；retrieval mechanisms 的詳細消融分析則於 Exp2 中呈現 [J3]。

---

## 二、研究方法

### 1. 資料集與資料切分

#### 1.1 資料集適用性與選用依據

本研究以 AniList 動畫資料快照作為主要資料集，目標是在作品正式播出前，利用當時可取得的文字描述、圖片與結構化 metadata，預測其後續的 `popularity` 與 `meanScore`。選用 AniList 的主要原因，是該資料同時包含兩項目標標籤，並保留較完整的發布前多模態欄位，能支撐本文的 pre-release anime prediction 任務 [J4]。

除 AniList 外，本文也評估多個外部資料來源，但不同資料集在本研究中的角色並不相同。Anime Offline Database 主要用於建立 AniList ID 與 MAL ID 的跨平台對照，並作為缺失播出季度與年份的補值診斷來源 [J5]；MAL July 2025 具有 `members` 與 `score`，可用於檢查 MAL 標籤是否能合理對應 AniList 的人氣與評分，但因缺少 image URL，不作為完整多模態外部考卷 [J6]；MyAnimeList scraped data 亦作為外部資料適用性比較來源之一 [J7]；MAL 2025 同時包含 MAL ID、會員數、評分、cover image URL、文字描述與 metadata，因此被選為正式外部測試來源 [J8]。Largest MAL User Dataset 則因包含作品上架後的使用者評分紀錄，具有明顯 label leakage 風險，本文僅將其列為未來研究素材 [J9]。

| Dataset | Role | Reason |
|---|---|---|
| AniList | Main data | Full fields |
| AODB | ID bridge | Crosswalk |
| MAL July | Label check | No image |
| MAL scraped | Candidate | Partial |
| MAL 2025 | External exam | Cover-ready |
| MAL Users | Future work | Leakage risk |

#### 1.2 發布前時間切分方式

由於 `popularity` 是累積型平台指標，會受到作品播出時間、資料快照時間與平台使用者規模變化影響，本文不採用隨機切分作為主要評估設定，而是建立 pre-release temporal split。此設計的目的，是讓模型使用較早播出作品作為訓練資料，並在較晚播出的作品上進行驗證與測試，更接近「在作品播出前預測未來表現」的應用情境。

實作上，資料處理流程先由 `seasonYear` 與 `season` 推導 `release_year` 與 `release_quarter`；若 `season` 不完整，則以 `startDate_month` 回推季度。接著將年份與季度合併為 `release_quarter_key`，依時間順序排序後，以累積樣本數接近 70% / 15% / 15% 的比例切分為 train、validation 與 test。最終模型切分包含 13,376 筆 train、2,918 筆 validation 與 3,087 筆 test。

| Split | Rows | Role |
|---|---:|---|
| Train | 13,376 | Model fit |
| Val | 2,918 | Tuning |
| Test | 3,087 | Internal exam |
| Holdout | 943 | Diagnostic |

#### 1.3 累積型指標與時間偏誤

`popularity` 並不是單純的作品品質分數，而是平台上的累積關注程度。較早播出的作品通常有更長時間累積使用者互動，因此若直接使用隨機切分，模型可能學到與年代或資料快照相關的偏差，而不是作品本身在播出前可觀察到的吸引力。Temporal split 雖無法完全消除 popularity 的生命週期效應，但能降低未來資料流入訓練階段的風險，並使評估設定更接近實際預測情境。

#### 1.4 保留組與資料完整性診斷

對於無法形成完整 `release_quarter_key` 的樣本，本文不將其強行放入 train、validation 或 test，而是標記為 `holdout_unknown`。該集合共 943 筆，約佔整體資料的 4.64%。診斷結果顯示，`holdout_unknown` 並非隨機缺漏：其作品型態、人氣、評分與圖片／預告片覆蓋率皆與正式模型樣本存在差異。因此，本文將其定位為資料完整性診斷集合，而非正式模型訓練或測試集合。

後續檢查顯示，Anime Offline Database 可補回其中 789 筆樣本的 release year 與 quarter，但由於這些值來自外部補值，本文主實驗仍維持原始 temporal split，不將補值後資料回灌至正式 train、validation 或 test。補值版本可作為未來 robustness check，用於觀察模型在原本 temporal 欄位缺失樣本上的表現 [J5]。

### 2. 評估指標與比較設定

本研究同時預測 `popularity` 與 `meanScore`，但兩個目標的尺度與意義不同，因此採用不同的主要評估指標。`popularity` 是長尾且累積性的數值，訓練與主要誤差分析以 `log1p` 空間為主；本文使用 Spearman correlation 衡量排序能力，使用 `log_MAE` 衡量對數空間誤差，使用 `log_R2` 衡量對數空間解釋能力，並以 `factor_acc_2x` 檢查預測是否落在真實值 0.5 倍至 2 倍的可接受區間內。原始尺度 MAE 仍保留為輔助參考，但不作為唯一判斷依據，因為其容易受到極端高人氣作品影響。

從直覺上看，`log_MAE` 可以近似解讀為乘法尺度上的平均誤差。例如 `log_MAE` 約為 0.89 時，對應到原始尺度約 `exp(0.89) = 2.43` 倍的幾何平均偏差；這也能與 `factor_acc_2x` 共同解讀，後者表示有多少比例的樣本落在真實人氣 0.5 倍至 2 倍的範圍內。

`meanScore` 則是 0 到 100 的線性分數，因此所有主要指標皆在原始尺度下計算。本文使用 Spearman correlation 檢查評分排序是否合理，使用 MAE 衡量平均分數偏差，使用 R2 檢查模型是否能解釋分數變異，並使用 `acc_within_10pt` 表示預測誤差在 10 分以內的比例。這組指標能同時反映排序能力、絕對誤差與實際可接受範圍。

為了避免模型因可用輸入欄位不同而產生不公平比較，Exp1 將代表性 baselines 與 CARMA 統一評估於相同的 3,087 筆 internal temporal test set。這些 baselines 直接使用 CARMA pipeline 輸出的 tensor inputs，包括 metadata、text embedding、image embedding 與 retrieval tensors，因此不重新產生 embedding，也不更換資料處理設定。

### 3. 研究問題

本研究的核心目的並非僅建立單一最高分模型，而是系統性檢驗動畫作品在正式播出前可取得的資訊，是否能有效預測其播出後的人氣與評分表現。由於本文同時涉及結構化 metadata、文字描述、圖片特徵、retrieval context，以及跨平台外部驗證，因此研究問題分為三個層次。

**研究問題一：不同 baseline 與多模態建模策略在 anime pre-release prediction 中能提供多強的參考基準？**

此問題對應 Exp1: Baseline Effect Comparison。本文建立 classical machine learning、feature concatenation、source-inspired fusion baseline 與 retrieval-oriented baseline，用於檢查 metadata-only、文字／圖片特徵、多模態串接，以及文獻啟發式模型在 `popularity` 與 `meanScore` 預測上的表現。此設計的重點是建立合理比較座標，避免後續模型只與過弱 baseline 比較。

**研究問題二：Retrieval augmentation 是否能為動畫人氣與評分預測提供額外訊號？**

此問題對應 Exp2: RAG ablation。本文在此將其定位為研究問題：若模型能參考相似歷史作品或關聯上下文，是否能補充單一作品本身 metadata、文字與圖片之外的資訊。需要注意的是，Exp1 中的 `C3` 系列是將 retrieval features 作為外部文獻對齊的靜態 reference baseline；Exp2 則針對主框架中的 retrieval component 進行消融、參數敏感度與架構演進分析。因此，Exp1 的 C3 baseline 不應直接等同於 Exp2 的完整 RAG ablation。

**研究問題三：以 AniList 訓練出的模型是否能泛化到 MAL-only 外部資料集？**

此問題對應 Exp3: Out Dataset Result。若模型只在 AniList 內部 test split 上表現良好，仍可能只是適應單一平台的資料分佈。為檢查跨平台泛化能力，本文建立 MAL-only external exam，透過 AniList/MAL ID crosswalk 排除可對回內部 AniList universe 的作品，並使用 MAL `members` 作為外部 popularity proxy、MAL `score * 10` 作為外部 score answer。此實驗主要檢查模型是否能在未出現在內部資料流程中的 MAL 作品上保留有效排序能力，而不是宣稱能完全校準 MAL 平台的絕對數值尺度。

| RQ | Focus | Experiment |
|---|---|---|
| RQ1 | Baseline strength | Exp1 |
| RQ2 | RAG gain | Exp2 |
| RQ3 | MAL generalization | Exp3 |

---

## 三、實驗

### 1. 實驗一：內部基準與主框架比較

本實驗用來回答 RQ1：在相同的 AniList temporal split 下，本研究的主框架是否能比只使用 metadata、簡單多模態串接、文獻改編融合模型與 retrieval baseline 提供更穩定的預測能力。換言之，Exp1 的重點不是建立一張包含所有嘗試模型的排行榜，而是用少量可解釋的參考系統，確認 proposed framework 的改善究竟來自 metadata、本身模態訊號、多模態融合，或歷史相似作品的補充資訊。

為確保 reference baselines 的比較公平，Exp1 將代表性 baseline rows 與 CARMA 統一放在完整 internal temporal test set `n=3,087` 上比較。這批 baseline features 不是使用獨立的 baseline-only artifacts，而是直接展平 `AnimeDataset` 回傳給 CARMA 的實際 tensor inputs；因此比較樣本、文字 embedding、圖片 embedding、retrieval tensors 與 temporal split 均與主框架一致。

本文將 Exp1 的比較系統整理為五種角色。`F1-RF-Meta-CARMATensor` 是 metadata-only 強基準，用來確認播出前結構化資訊本身能提供多少預測力；`F2-XGB-Concat-CARMATensor` 是 simple fusion floor，用來檢查將 metadata、文字、圖片與可用 tensor features 直接串接是否已足以帶來多模態增益；C1/C2 是 literature-adapted fusion baselines，用來回應 related work 中的 anime-domain multimodal MLP 與 cross-modal / recurrent fusion 參考方向，但僅作為 project-input proxy，而非原論文 exact reproduction [J1], [J2]；`C3-RAG-XGB-CARMATensor` 是 retrieval reference，用來檢查相似歷史作品是否能補充當前作品的播出前資訊 [J3]；`FusionModel v2 Run22` 則是主框架代表模型。

| Comparison | Models | Input | Purpose |
|---|---|---|---|
| Metadata baseline | F1 | Metadata | Strong floor |
| Simple fusion | F2 | Meta+Text+Image | Fusion floor |
| Literature-adapted | C1/C2 | Multimodal | Diagnostic ref |
| Retrieval baseline | C3 | Retrieved context | Retrieval ref |
| Proposed framework | Run22 | Full framework | Representative run |

在評估指標上，Exp1 依 target 性質採用不同主指標。對 `popularity` 而言，由於其分布具有長尾與累積特性，本文同時觀察 `log_MAE`、`log_R2`、`factor_acc_2x` 與 Spearman；其中 `log_MAE` 約可解讀為幾何尺度誤差，例如 `log_MAE=0.89` 約對應到 `e^0.89=2.43` 倍的平均誤差尺度，能與 `factor_acc_2x` 共同解讀。對 `meanScore` 而言，本文觀察 MAE、R2、`acc_within_10pt` 與 Spearman，因為評分是 0 到 100 的線性尺度，MAE 可直接解讀為平均偏離幾分。

`popularity` 的代表性結果顯示，強基準不只來自深度模型。下表所有 rows 皆於相同的 `n=3,087` internal temporal test set 上評估，CARMA row 使用 Run22 seed=42 checkpoint。metadata-only `F1-RF-Meta-CARMATensor` 已取得 Spearman `0.8326`，表示播出年份、格式、來源、類型與製作相關資訊本身已包含大量排序訊號。`F2-XGB-Concat-CARMATensor` 在 `log_MAE` 上略優於 Run22 (`0.8799` vs. `0.8823`)，代表簡單 tensor 串接仍是很強的 popularity error baseline；C2-RecurrentFusion 則取得最高 Spearman (`0.8614`)。Run22 在 2x accuracy 上最高，且 ranking 表現與 retrieval baseline 接近。因此，本文不將 popularity 寫成 CARMA 全指標勝出，而是解讀為「competitive popularity prediction」。

| Method | log_MAE | 2x_acc | Spearman |
|---|---:|---:|---:|
| F1-RF-Meta-CARMATensor | 0.9731 | 0.4655 | 0.8326 |
| F2-XGB-Concat-CARMATensor | 0.8799 | 0.4885 | 0.8578 |
| C1-Armenta-CARMATensor | 0.9210 | 0.4704 | 0.8520 |
| C2-CrossAttention-CARMATensor | 0.9115 | 0.4739 | 0.8537 |
| C2-RecurrentFusion-CARMATensor | 0.8915 | 0.4817 | 0.8614 |
| C3-RAG-XGB-CARMATensor | 0.8947 | 0.4917 | 0.8520 |
| FusionModel v2 Run22 | 0.8823 | 0.4943 | 0.8520 |

對 `meanScore` 而言，整體 R2 明顯低於 `popularity`，表示分數較難由播出前資訊穩定預測；因此正文表格保留 MAE、10 分內準確率與 Spearman，將 R2 作為診斷指標。下表同樣使用 `n=3,087` internal temporal test set。Run22 在 seed=42 下取得最低 MAE (`7.5911`) 與最高 10 分內準確率 (`0.7104`)，但此結果仍應視為代表性 run 的表現，而非跨所有 seeds 的最低誤差保證；Spearman 最高者則為 F2。因此，本文將 meanScore 結論限定為「Run22 代表 run 有良好 absolute-error 表現」，而非所有指標皆勝出。

| Method | MAE | 10pt_acc | Spearman |
|---|---:|---:|---:|
| F1-RF-Meta-CARMATensor | 8.8547 | 0.6281 | 0.5171 |
| F2-XGB-Concat-CARMATensor | 8.6542 | 0.6450 | 0.5591 |
| C1-Armenta-CARMATensor | 8.1840 | 0.6728 | 0.5215 |
| C2-CrossAttention-CARMATensor | 8.4217 | 0.6673 | 0.5354 |
| C2-RecurrentFusion-CARMATensor | 9.2659 | 0.6132 | 0.5294 |
| C3-RAG-XGB-CARMATensor | 8.7747 | 0.6323 | 0.5432 |
| FusionModel v2 Run22 | 7.5911 | 0.7104 | 0.5424 |

`C3-SourceExact-K64` 屬於 source-faithful diagnostic run，且使用完整 test set `n=3,087`，因此不與 main performance rows 直接排名。其結果顯示，直接將 SKAPP/RRCP staged pipeline 轉換到 anime pre-release prediction 時，仍會受到 target calibration、retrieval size、loss design 與 target-space 設計影響。

主框架代表模型 `FusionModel v2 Run22` 使用 fixed seed 與 per-target hyperparameter overrides，讓 `popularity` 與 `meanScore` 分別套用較適合的 dropout、attention dropout、weight decay 與 batch size。由於 Exp1 採用 full-test aligned baselines，因此可直接回應樣本數不一致的疑慮：CARMA 並非 popularity 全面勝出，但在完整 internal test set 上維持 competitive ranking performance，並在代表性 seed 下提供最好的 meanScore absolute-error 表現。由於本文主表仍以代表性 run 呈現，seed sensitivity 將作為限制說明；因此 Exp1 的主張限定為「CARMA is competitive on popularity ranking and strong on representative meanScore error」，而非宣稱所有指標或所有 seeds 皆穩定勝出。

### 2. 實驗三：外部資料集測試

本實驗用於檢查模型是否能從內部 AniList temporal test split 推廣至 MyAnimeList（MAL）來源的外部資料。由於動畫名稱容易受到續作、劇場版、OVA、別名與翻譯差異影響，本研究的外部資料對齊只使用穩定 ID，不使用 title matching 或模糊名稱比對。具體而言，本研究先利用 Anime Offline Database 建立 AniList ID 與 MAL ID 的 crosswalk，再排除所有可對回內部 AniList universe 的 MAL rows，以確保外部考卷中的樣本不是模型在內部資料流程中已見過的作品 [J5]。

| Dataset | Role | Reason |
|---|---|---|
| AODB | ID bridge | Crosswalk |
| MAL July | Label check | No image |
| MAL scraped | Candidate | Partial |
| MAL 2025 | External exam | Cover-ready |
| MAL Users | Future work | Leakage risk |

在資料來源篩選上，MAL July 2025 具有 `members` 與 `score`，因此一開始可用來檢查 MAL label 是否能合理對應 AniList 的 `popularity` 與 `meanScore` [J6]。為避免 label check 與正式外部考卷來自不同資料集，本文進一步在 MAL 2025 overlap rows 上重做同一項檢查：AniList `popularity` 與 MAL 2025 `members` 的 Spearman correlation 為 `0.9836`，AniList `meanScore` 與 MAL 2025 `score x 10` 的 Spearman correlation 為 `0.9446`。此結果顯示 MAL 2025 的 `members` 與 `score` 本身適合作為外部答案來源；MAL July 則僅保留為早期資料適用性檢查，不作為完整多模態外部主考卷。

正式外部測試採用 MAL 2025。該資料集提供 MAL ID、`members`、`score`、description、metadata 與 cover image URL，可透過 adapter 轉換為模型可讀的外部 split [J8]。下載 cover 圖片並移除本機圖片缺失樣本後，得到兩份 local-ready external exam：`mal2025_popularity_local_ready` 共 `3,765` 筆，用於評估 MAL `members`；`mal2025_dual_local_ready` 共 `1,202` 筆，可同時評估 MAL `members` 與 `score x 10`。MAL 2025 具備 cover image，因此可作為外部多模態測試來源；但第一版 adapter 並未重建 AniList 內部資料才有的 banner branch 與 YOLO branch，因此這兩個 branch 在主外部推論中仍視為 unavailable modality。

| Exam | Rows | Target | Label |
|---|---:|---|---|
| Pop-only | 3,765 | Popularity | Members |
| Dual | 1,202 | Popularity | Members |
| Dual | 1,202 | Score | Score x10 |

外部推論使用 Run22 checkpoint 作為主框架代表，並將 F1/F2/C1/C2/C3 baselines 重算於相同 MAL rows，以避免外部測試再次出現樣本集合不一致。結果顯示，模型在 MAL-only rows 上仍保有一定程度的排序轉移能力，但跨平台表現並非單一模型全指標勝出。在 `3,765` 筆 popularity-only 外部考卷上，Run22 的 Spearman 為 `0.4998`，高於 Run02 舊結果但低於 F2；在 `1,202` 筆 dual-target 外部考卷上，Run22 的 popularity log_MAE (`1.1707`) 與 score MAE (`6.3363`) 皆為表中最佳，但 popularity Spearman 最高者為 C3，score Spearman 最高者為 C2-Recurrent。

| Exam | Model | Rows | Pop log_MAE | Pop Spearman | Score MAE | Score Spearman |
|---|---|---:|---:|---:|---:|---:|
| Pop-only | F1-RF-Meta | 3,765 | 1.0015 | 0.4572 | - | - |
| Pop-only | F2-XGB-Concat | 3,765 | 1.0294 | 0.5240 | - | - |
| Pop-only | C1-Armenta | 3,765 | 1.1704 | 0.2798 | - | - |
| Pop-only | C2-CrossAttention | 3,765 | 1.0769 | 0.3576 | - | - |
| Pop-only | C2-RecurrentFusion | 3,765 | 1.2692 | 0.0766 | - | - |
| Pop-only | C3-RAG-XGB | 3,765 | 1.0383 | 0.4392 | - | - |
| Pop-only | CARMA Run22 | 3,765 | 1.0359 | 0.4998 | - | - |
| Dual | F1-RF-Meta | 1,202 | 1.4387 | 0.5144 | 9.1210 | 0.5629 |
| Dual | F2-XGB-Concat | 1,202 | 1.6172 | 0.5789 | 9.7600 | 0.6061 |
| Dual | C1-Armenta | 1,202 | 1.9602 | 0.4572 | 9.9122 | 0.5826 |
| Dual | C2-CrossAttention | 1,202 | 1.7178 | 0.5607 | 8.3423 | 0.5491 |
| Dual | C2-RecurrentFusion | 1,202 | 1.8289 | 0.4585 | 9.2142 | 0.6113 |
| Dual | C3-RAG-XGB | 1,202 | 1.5637 | 0.6211 | 9.3699 | 0.5989 |
| Dual | CARMA Run22 | 1,202 | 1.1707 | 0.5647 | 6.3363 | 0.5770 |

外部 paired bootstrap diagnostic 進一步確認上述解讀。以 dual no-YOLO split 為例，CARMA 相對 F2 的 score MAE 改善為 `+3.4237`，95% CI 為 `[3.1828, 3.7018]`；相對 C3 的 popularity log_MAE 改善為 `+0.3929`，95% CI 為 `[0.3568, 0.4270]`。但 ranking 指標仍呈現混合結果：同一 split 中 C3 的 popularity Spearman 穩定高於 CARMA (`delta=-0.0564`, 95% CI `[-0.0901, -0.0216]`)；在 cover-derived YOLO split 中，F2 的 score Spearman 也穩定高於 CARMA (`delta=-0.0450`, 95% CI `[-0.0774, -0.0124]`)。因此外部測試最穩定的主張是 CARMA 在 external score absolute error 上有優勢，而不是在所有跨平台 ranking 指標上勝出。

不過，外部結果也顯示模型的數值校準仍有限。特別是 dual-target split 中，即使 Run22 將 meanScore MAE 從 Run02 的 `7.5086` 改善至 `6.3363`，meanScore R2 仍為負值 (`-0.3253`)；popularity log R2 也為 `-0.1188`。進一步的 prediction-quantile calibration 分析顯示，較高預測分位通常對應較高的實際 MAL members 與 score，代表排序訊號仍存在；但高人氣分位的實際 MAL members 明顯大於模型預測，顯示模型低估 MAL 平台上高人氣作品的累積尺度。error-slice diagnostics 進一步確認此問題主要集中在外部尾端：dual no-YOLO split 中最高人氣分位的 popularity `log_MAE` 為 `2.6768`，平均 MAL members 約為模型預測的 `17.40` 倍；cover-derived YOLO split 的 top 10% members slice 中，平均 MAL members 約為預測的 `30.42` 倍。meanScore 也呈現類似 top-end compression：dual no-YOLO split 的最高 MAL score 分位 MAE 為 `10.4623`，平均 signed error 為 `-10.4035`，代表模型對高分作品仍有明顯低估。這代表模型能提供有用的排序與絕對誤差預測，但無法直接把 AniList 的絕對數值尺度完整轉移到 MAL 平台。這是合理限制，因為 AniList `popularity` 與 MAL `members` 雖然都反映人氣，但兩者屬於不同平台的累積 count scale。因此，本研究將外部測試的主要解讀放在 Spearman ranking transfer、log-scale error 與 score MAE，而不是原始尺度 R2。

同時，外部 Spearman correlation 也低於內部 temporal test split，表示退化不只來自絕對尺度校準。為檢查 missing visual branch 的影響，本研究進一步建立 cover-derived YOLO diagnostic：從 MAL cover image 偵測人物與臉部 crop，補入原本為 zero-filled 的 YOLO branch，但仍保留 banner branch missing。補入 YOLO 後，CARMA 的 pop-only popularity Spearman 從 `0.4998` 提升至 `0.5213`；dual-target popularity Spearman 從 `0.5647` 提升至 `0.6073`，score Spearman 也從 `0.5770` 提升至 `0.5999`。然而 dual-target popularity log_MAE (`1.1707` → `1.2001`) 與 score MAE (`6.3363` → `6.4919`) 略微變差。相同 YOLO-filled split 上，F2/C2/C3 的 ranking 也明顯受益；例如 dual-target C2-CrossAttention 的 popularity Spearman 達 `0.6687`，C2-Recurrent 的 score Spearman 達 `0.6459`。因此，YOLO 缺失確實會影響外部排序能力，但無法單獨解釋所有外部退化；CARMA 在 YOLO-filled external setting 下仍保有最佳 score MAE 與 10 分內準確率，但不是所有 ranking 指標最佳。

由於 MAL 2025 沒有真正的 AniList-style banner image，本研究另設計 cover-as-banner proxy diagnostic：將 cover embedding 複製到 banner slot，以檢查「banner slot 全缺失」是否可由簡單 cover proxy 緩解。結果顯示此 proxy 並未穩定優於 cover-derived YOLO 條件：CARMA 的 pop-only Spearman 從 `0.5213` 變為 `0.5166`，dual popularity Spearman 從 `0.6073` 變為 `0.5955`，score Spearman 從 `0.5999` 變為 `0.5921`；dual score MAE 則為 `6.4098`，仍優於所有 cover-as-banner baseline rows。因此，naive cover-as-banner 不能取代真 banner branch。剩餘落差更可能來自真實 banner 資訊缺失、AniList 與 MyAnimeList 使用者社群差異、收錄時間差異，以及兩平台人氣累積尺度不同。

---

## 四、未來工作

未來工作可沿著三個方向延伸。第一，MAL 2025 local-ready external split 可作為固定外部考卷，用於比較後續模型架構、checkpoint 或圖片處理 pipeline 的跨平台泛化能力；本研究已納入 cover-derived YOLO diagnostic、cover-as-banner proxy diagnostic、calibration diagnostics、error-slice diagnostics 與 case examples，但 naive cover-as-banner 不能取代真 banner 資訊，因此後續若要縮小外部 adapter 與內部 AniList pipeline 的模態落差，應取得真實 banner-like source、訓練明確的 banner imputation method，或加入 explicit calibration correction [J8]。第二，Largest MAL User Dataset 雖然具有大量使用者層級紀錄，但必須單獨建立嚴格的時間切分，只能使用動畫上映前可觀察的使用者行為；若直接使用觀看後評分或完成後互動紀錄，將造成 label leakage [J9]。第三，`holdout_unknown` 中可由 Anime Offline Database 補回 season/year 的樣本，可在未來形成 `holdout_recovered` robustness check，但不應回灌到本文正式 temporal split [J5]。

對 reference baseline 而言，未來亦可持續提升外部文獻方法與原始設計之間的對齊程度。C1 可進一步補足角色描述與角色肖像資料 [J1]；C2 可補足更接近原文的 BERT text encoder 與 ViT visual stream [J2]；C3 目前已有 `popularity` 與 `meanScore` 的 source-faithful K64 diagnostic，但仍需在 SKAPP/RRCP pipeline 上完成 `top_k=500`、校準檢查與 loss/target-space 設計修正 [J3]。此外，本研究已針對主框架與主要 ablation 進行 7-seed robustness 檢查，顯示 RAG 與 image 對 ranking 的貢獻較穩定，但 error 指標與 trend branch 對 seed 較敏感；後續若要進一步強化統計證據，可延伸至 baseline families 的 multiple-seed evaluation 與 seed-level significance tests。主框架層面則可加入 strict CNN-vs-Swin backbone ablation，並重新設計 trend head，例如加入獨立 learning rate、斜率正則，或避免 `release_year` 同時由 metadata 與 trend branch 重複注入。這些方向屬於後續擴充，而非本文核心實驗結論的必要前提。

---

## References

[J1] Jesús Armenta-Segura, Grigori Sidorov. (2025). Anime popularity prediction before huge investments: a multimodal approach using deep learning. PeerJ Computer Science, 11, e2715. https://doi.org/10.7717/peerj-cs.2715

[J2] Canaan Tinotenda Madongo, Zhongjun Tang, Jahanzeb Hassan. (2023). Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers. Proceedings of the 2023 6th International Conference on Artificial Intelligence and Pattern Recognition (AIPR 2023). ACM. https://doi.org/10.1145/3641584.3641796

[J3] Xovee Xu, Yifan Zhang, Fan Zhou, Jingkuan Song. (2025). Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation. Proceedings of the AAAI Conference on Artificial Intelligence.

[J4] Caleb M. Welsh. (n.d.). AniList Anime Dataset. Kaggle. https://www.kaggle.com/datasets/calebmwelsh/anilist-anime-dataset/data

[J5] Abhishek Gupta. (n.d.). Anime Offline Database. Kaggle. https://www.kaggle.com/datasets/abhishekgupta56447/anime-offline-database

[J6] wiltheman. (n.d.). Anime Data Set for ML, Version 1. Kaggle. https://www.kaggle.com/datasets/wiltheman/anime-data-set-for-ml/versions/1

[J7] Hamza Ashfaque. (n.d.). MyAnimeList Scraped Data. Kaggle. https://www.kaggle.com/datasets/hamzaashfaque1999/myanimelist-scraped-data

[J8] Syahrul Apriansyah. (n.d.). MyAnimeList 2025. Kaggle. https://www.kaggle.com/datasets/syahrulapriansyah2/myanimelist-2025/discussion?sort=hotness

[J9] dousitekounarundarou. (n.d.). Largest MyAnimeList User Dataset You'll Ever Find. Kaggle. https://www.kaggle.com/datasets/dousitekounarundarou/largest-myanimelist-user-dataset-youll-ever-find?resource=download

## Project Evidence

- `docs/pipeline/data_processing_for_paper.md`
- `docs/pipeline/external_evaluation_method.md`
- `docs/paper_baseline_sections_draft.md`
- `reports/external/external_evaluation_summary.md`
- `reports/baselines/reference_baseline_metrics_extended_2026-06-01.md`
- `reports/diagnostics/c3_source_exact_k64_diagnostic_2026-06-01.md`
- `reports/experiments/sample_alignment/eval_sample_alignment_report_2026-06-11.md`
- `reports/experiments/sample_alignment/run22_artifact_manifest.md`
