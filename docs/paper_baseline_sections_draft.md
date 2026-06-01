# Baseline 與實驗段落草稿

本文件整理本研究中與 baseline、reference baseline、research questions、Exp1 與 Exp3 相關的論文草稿。此版本用於後續整合進正式報告；目前已填入 baseline 與 C3 source-faithful diagnostic 的可用結果，主框架與 out-dataset 結果仍需由對應實驗補齊。

## 2. Related Works

### 2.1 Multimodal Fusion

多模態融合（multimodal fusion）近年被廣泛應用於媒體內容理解、人氣預測與推薦系統等任務中，其核心目標在於整合不同來源的資訊，例如文字敘述、視覺內容與結構化後設資料，以補足單一模態在表徵能力上的限制。最直接的作法是 early fusion，即先分別取得各模態的特徵向量，再將其串接後輸入傳統機器學習模型或多層感知機進行預測。此方法實作簡單且可作為強基準線，但其限制在於不同模態的重要性被隱含地交由下游模型學習，較難明確控制各模態的貢獻。

相較之下，branch-wise fusion 會先為不同模態建立各自的投影分支，使文字、圖像與後設資料能在進入共同預測層之前先完成模態內部的表徵轉換。此類設計能較清楚地保留不同模態的結構差異，也便於分析各模態對最終預測的影響。例如，C1 所參考的動畫人氣預測研究使用動畫簡介、角色描述與角色圖像資訊建立多輸入深度神經網路；C2 所參考的電影票房預測研究則將評論文字與海報視覺特徵納入多模態預測架構。由於上述研究的任務、資料來源與可用欄位皆與本研究的播出前動畫人氣預測不完全相同，本研究並不將其視為 exact reproduction，而是將其核心融合架構轉化為 literature-adapted baseline，用以比較不同外部多模態設計在本專案統一輸入設定下的表現。

此外，attention 或 gating 機制也常被用於多模態融合，以動態調整不同模態在不同樣本上的相對權重。相較於固定串接或固定分支融合，attention/gating 更適合處理模態品質不一致的情境，例如某些作品的文字描述較完整，而某些作品的圖像資訊較具代表性。這類方法也與本研究的主框架設計相關，因為動畫在播出前可取得的資訊品質並不穩定，模型若能根據樣本特性調整文字、圖像與後設資料的影響力，理論上能取得更具彈性的預測效果。

### 2.2 Retrieval-Augmented Popularity Prediction

除了直接使用目標作品本身的多模態資訊外，檢索增強方法（retrieval augmentation）近年也被用於補充外部知識或歷史相似案例。此類方法的基本假設是，目標樣本的表現可能與過去相似樣本具有關聯；在人氣預測任務中，與目標動畫具有相似題材、製作公司、視覺風格或文字語意的既有作品，其歷史人氣與評分可以作為額外參照。相較於只依賴單一作品本身的描述與圖像，retrieval augmentation 能引入資料集中既有作品的分布資訊，進而提升模型對冷啟動或播出前樣本的判斷能力。

然而，檢索結果並不必然帶來正面效果。若檢索依據僅仰賴表面相似度，模型可能取得語意相近但人氣模式差異很大的樣本，反而引入噪音。因此，檢索策略本身需要被檢驗與消融。常見作法包括 sparse retrieval，例如基於關鍵詞、類別或後設資料重疊進行檢索；dense retrieval，例如基於文字或圖像 embedding 的向量相似度；以及 hybrid retrieval，結合 sparse 與 dense 訊號以取得較平衡的相似樣本。C3 所參考的社群媒體人氣預測研究即屬於 retrieval-based multimodal popularity prediction 的相關方向，其重點不僅在於加入檢索結果，也在於透過選擇性檢索與後續融合機制降低不相關樣本帶來的干擾。本研究同樣不將 C3 視為完整原始資料與原始流程的 exact reproduction，而是將其作為 literature-adapted baseline，用於檢驗 retrieval augmentation 在播出前動畫人氣預測任務中的可遷移性。

上述相關研究共同指出，多模態資訊與檢索增強皆可能提升人氣預測表現，但其有效性高度依賴任務設定、輸入模態與資料可得性。因此，本研究的 baseline 設計採取分層比較：首先以平均值、線性模型與傳統機器學習模型建立基本參照；接著以文字、圖像與簡單串接模型分析單模態與 early fusion 的效果；最後將 C1、C2、C3 轉化為 literature-adapted baselines，以比較既有多模態融合與檢索增強架構在統一動畫資料集、相同切分方式與相同評估指標下的表現。

## 3. Methodology

### 3.1 Dataset and Split

本研究聚焦於新番動畫播出前之熱度與評分預測任務，目標是在作品尚未正式播出或尚未累積大量觀眾回饋前，根據可於播出前取得的多模態資訊，預測其未來的 `popularity` 與 `meanScore`。相較於一般播出後推薦或評分預測任務，本研究刻意避免使用播出後才會產生的互動資訊，例如實際觀看人數變化、評論內容、使用者評分分布或社群討論量等，以維持 pre-release prediction 的任務設定。資料來源以 AniList 動畫資料為主，並整合動畫的結構化 metadata、文字描述、影像資訊，以及由相似作品檢索而來的輔助特徵，作為後續 baseline 與主框架的共同輸入基礎。

為了確保不同方法之間的比較具有一致性，本研究要求所有 baseline 與 proposed framework 盡可能對齊相同的 dataset、target variables、data split 與 evaluation metrics。具體而言，所有模型皆以相同的訓練集、驗證集與測試集切分進行實驗，並分別對 `popularity` 與 `meanScore` 兩個目標進行預測。評估指標則包含 MAE、RMSE、R² 與 Spearman's rank correlation，以同時觀察模型在絕對誤差、解釋能力與排序能力上的表現。此設計的目的並非比較各方法在原始論文資料集上的表現，而是將不同 baseline 的核心建模概念映射至同一個 anime pre-release prediction 任務中，進而檢驗其在本研究情境下是否仍具有參考價值。

然而，由於不同 baseline 所需的特徵來源並不完全相同，部分模型可能需要額外的文字 embedding、影像 embedding、RAG 特徵或文獻方法對應的重建特徵。因此，在實際實驗中可能出現部分樣本缺少特定模態或外部 artifact 的情況。為了避免誤將資料覆蓋率差異解讀為模型能力差異，本研究在結果比較時區分 available-case comparison 與 common-subset comparison。available-case comparison 代表各模型在其可取得完整輸入特徵的樣本上進行評估，能反映該方法在現有資料條件下的實際可用表現；common-subset comparison 則限制所有被比較方法使用共同可用的樣本集合，以提升不同模型之間的公平性。本文在討論 baseline 與主框架結果差異時，會明確標註比較設定，並將特徵缺失、樣本數差異與 artifact coverage 視為解釋實驗結果時的重要限制。

### 3.2 Evaluation Metrics

本研究同時預測 `popularity` 與 `meanScore`，但兩個目標的尺度與解讀方式不同，因此採用不同的主要評估指標。對於 `popularity`，由於其分布具有明顯長尾特性，訓練與主要誤差評估以 `log1p` 空間為主；對於 `meanScore`，由於其本身為 0 到 100 的線性分數，所有主要指標皆在原始尺度下計算。

對 `popularity` 而言，本文以 `spearman_rho`、`log_MAE`、`log_R2` 與 `factor_acc_2x` 作為主要評估指標，並以原始尺度 `MAE` 作為輔助參考。`spearman_rho` 衡量模型是否能正確排序作品人氣；`log_MAE` 衡量模型在對數空間下的平均絕對誤差；`log_R2` 衡量模型在對數空間中解釋人氣變異的能力；`factor_acc_2x` 則表示預測值落在真實值 0.5 倍至 2 倍範圍內的比例。其定義如下，其中 $\hat{y}$ 為預測值、$y$ 為真實值、$n$ 為樣本數：

$$
\log\_MAE = \frac{1}{n}\sum_i |\log_{1p}(\hat{y_i}) - \log_{1p}(y_i)|
$$

$$
\log\_R^2 = 1 - \frac{\sum_i(\log_{1p}(y_i) - \log_{1p}(\hat{y_i}))^2}{\sum_i(\log_{1p}(y_i) - \overline{\log_{1p}(y)})^2}
$$

$$
\text{factor\_acc\_2x} = \frac{1}{n}\sum_i \mathbf{1}\left(|\log_{1p}(\hat{y_i}) - \log_{1p}(y_i)| < \log 2\right)
$$

對 `meanScore` 而言，本文以 `spearman_rho`、`MAE`、`R2` 與 `acc_within_10pt` 作為主要評估指標。`MAE` 表示平均預測分數偏離真實分數的絕對距離；`R2` 用於檢查模型是否能解釋分數變異；`acc_within_10pt` 表示預測誤差在 10 分以內的樣本比例：

$$
MAE = \frac{1}{n}\sum_i |\hat{y_i} - y_i|
$$

$$
R^2 = 1 - \frac{\sum_i(y_i - \hat{y_i})^2}{\sum_i(y_i - \bar{y})^2}
$$

$$
\text{acc\_within\_10pt} = \frac{1}{n}\sum_i \mathbf{1}\left(|\hat{y_i} - y_i| < 10\right)
$$

需要注意的是，早期 baseline result CSV 已輸出 `MAE`、`R2`、`Spearman_rho` 與 popularity 的 `log_MAE`；本草稿中的 `log_R2`、`factor_acc_2x` 與 `acc_within_10pt` 是依照現有 prediction 檔重新計算所得，正式結果表應以更新後的 evaluation pipeline 重新產生。

### 3.3 Research Questions

本研究的實驗設計圍繞三個研究問題展開，分別對應多模態貢獻、檢索增益，以及模型在資料分布變化下的穩健性。這三個問題共同構成本文對 pre-release anime popularity and meanScore prediction 的主要分析架構：首先確認不同模態與 baseline 方法本身是否具有預測能力，其次檢驗 RAG 是否能提供額外資訊增益，最後觀察模型在 out-dataset 或分布轉移情境下是否仍具泛化能力。

**RQ1: How do different baseline families and modality combinations contribute to pre-release anime prediction?** 此問題對應 Exp1: Baseline Effect Comparison。Exp1 的目的在於建立由弱到強的比較基準，包含 common sanity baselines、metadata-based classical baselines、single-modality baselines、simple multimodal fusion baselines，以及 literature-adapted external baselines。透過比較 metadata、text、image 與其組合，本研究檢驗哪些資訊來源對 `popularity` 與 `meanScore` 預測最具貢獻，並進一步分析 proposed framework 相較於簡單 concat、傳統機器學習方法與外部文獻改寫 baseline 是否具有實質改善。

**RQ2: Does retrieval-augmented information improve prediction performance beyond non-RAG multimodal features?** 此問題對應 Exp2: RAG Ablation。由於動畫作品的熱度可能與歷史相似作品具有關聯，本研究進一步檢驗 RAG 特徵是否能在既有 metadata、text 與 image representations 之外提供額外訊號。Exp2 比較 No-RAG、metadata-only RAG、text-only RAG 與 hybrid RAG 等設定，藉此分析模型改善究竟來自檢索本身，還是來自特定檢索策略與模態組合。同時，此實驗也有助於辨識 RAG 是否可能引入噪音，特別是在語意相似但 popularity 分布不一致的情況下。

**RQ3: How robust is the proposed framework under dataset shift or out-dataset evaluation?** 此問題對應 Exp3: Out Dataset Result。前兩項實驗主要評估模型在同一資料分布下的表現，但若模型高度依賴 AniList 特定資料分布、年份區間或平台特徵，則其在不同資料來源或不同時間區間上的效果可能下降。因此，Exp3 著重於 out-dataset 或 distribution shift 情境下的泛化能力，並觀察 baseline 與 proposed framework 在外部資料條件下的性能衰退程度。此實驗不僅用於檢驗模型是否具備穩健性，也可協助分析目前方法仍受限於資料來源、特徵覆蓋率或檢索品質的部分。

## 4. Experiments

### 4.1 Exp1: Baseline Effect Comparison

本實驗旨在建立一組由弱到強、由單一模態到多模態、由一般控制組到外部文獻改寫模型的 baseline 比較架構，以評估本研究提出之動畫播出前人氣預測框架是否確實帶來額外效益。不同於僅將所有模型結果排列成排行榜，本實驗更重視各類 baseline 在研究問題中的角色：哪些結果代表任務本身的最低參照，哪些結果反映 metadata、文字與圖像模態的個別貢獻，哪些結果則可作為既有多模態或檢索式方法的外部參照。

本研究將 baseline 分為四類。第一類為 common sanity baseline，包含 `F0-Mean` 與 `F0-Ridge-Meta`。`F0-Mean` 為最基礎的無資訊回歸基線，僅以訓練集目標值平均作為預測結果，用以確認後續模型是否至少優於簡單平均預測。`F0-Ridge-Meta` 則使用 metadata 特徵建立線性回歸基線，用以檢查任務是否可由線性關係初步解釋。此類模型並非外部文獻復現，而是回歸任務中常見的 sanity check，可作為判斷模型是否具備基本預測能力的最低標準。

第二類為 metadata classical baseline，包含 `F1-RF-Meta` 與 `F1-GB-Meta`。此類模型僅使用播出前可取得之結構化 metadata，例如播出年份、集數、作品格式、來源類型、類型標籤、製作公司與相關歷史統計特徵等。`F1-RF-Meta` 採用 Random Forest，`F1-GB-Meta` 採用 Gradient Boosting，兩者皆屬於傳統機器學習方法。此類 baseline 的目的，是估計在不使用文字描述、圖像特徵與檢索增強資訊的情況下，僅依賴 metadata 能達到的預測上限。因此，若後續多模態模型未能明顯超越此類模型，即表示新增模態可能未有效提供額外資訊，或融合方式仍需改進。

第三類為 modality and simple fusion baseline，包含 `T2-XGB-TextEmb`、`I1-XGB-ImageEmb` 與 `F2-XGB-Concat`。`T2-XGB-TextEmb` 僅使用文字 embedding，評估動畫簡介與文字語意對人氣及評分預測的貢獻；`I1-XGB-ImageEmb` 僅使用圖像 embedding，評估封面、橫幅或角色相關視覺訊號的預測能力；`F2-XGB-Concat` 則將 metadata、文字 embedding 與圖像 embedding 直接串接後輸入 XGBoost，作為簡單多模態融合基線。此類 baseline 的功能是提供模態消融的控制參照，而非宣稱為外部 SOTA 方法。透過此類設計，本研究能分辨模型效能提升究竟來自單一強模態、簡單特徵串接，或來自更進一步的多模態融合與檢索增強設計。

第四類為 literature-adapted baseline，即 `C1`、`C2` 與 `C3` 三組外部文獻改寫模型。`C1` 參考動畫人氣預測相關研究中以 GPT-2、ResNet-50 與 MLP 進行多輸入融合的設計；`C2` 參考電影票房預測研究中結合文字評論、電影海報與 transformer-based fusion 的方法；`C3` 則參考多模態社群媒體人氣預測中 selective retrieval knowledge augmentation 的設計。需要特別說明的是，這三類方法在本研究中皆屬於 adapted 或 proxy reproduction，而非 exact reproduction。其原因在於原始論文的資料集、任務定義與輸入模態並不完全等同於本研究的動畫播出前預測任務。例如，C1 原始設計使用動畫簡介、主要角色描述與角色肖像；C2 原始設計面向電影票房，使用電影評論與電影海報；C3 原始設計則面向社群媒體貼文，包含使用者生成內容、圖片與檢索式知識增強。本研究為了確保比較公平，將其核心建模思想映射至相同的 AniList 動畫資料、相同 target、相同 split 與相同評估指標。因此，C1/C2/C3 的結果應被解讀為外部方法在本研究任務下的架構性參照，而不是原論文完整資料環境下的復現分數。

在結果分析上，本實驗不僅比較各模型的 MAE、R² 與 Spearman correlation，也將其放回對應研究問題中解讀。首先，`F0-Mean` 與 `F0-Ridge-Meta` 用於確認預測任務是否能超越無資訊與簡單線性基線。其次，`F1-RF-Meta` 與 `F1-GB-Meta` 用於衡量 metadata 本身的強度；若 metadata-only 模型已取得高表現，代表動畫播出前的結構化資訊對 `popularity` 與 `meanScore` 具有高度解釋力。第三，`T2-XGB-TextEmb`、`I1-XGB-ImageEmb` 與 `F2-XGB-Concat` 用於檢驗文字、圖像與簡單多模態串接是否提供額外增益。最後，C1/C2/C3 則用於觀察外部多模態或檢索式架構在本研究資料設定下是否仍具競爭力。

因此，本研究在比較 baseline 與 proposed framework 時，並不將結果簡化為單一排名，而是採取分層比較。若 proposed framework 僅優於 `F0`，表示其具備基本預測能力；若能進一步優於 metadata classical baseline，表示多模態資訊確實帶來超越結構化資料的增益；若能優於 `F2-XGB-Concat`，則表示模型的融合機制不只是簡單特徵串接；若能優於 C1/C2/C3 adapted baselines，則代表本研究針對動畫播出前預測所設計的多模態與檢索增強框架，在統一資料設定下較既有外部方法改寫版本更適合此任務。相反地，若某些 baseline 在特定指標上優於 proposed framework，該結果亦具有研究價值，因為它可指出目前框架仍可能受到模態噪音、embedding 品質、RAG 檢索誤差或融合策略不足等因素限制。

#### 4.1.1 Current Available Baseline Results

正式文件中建議將 Exp1 結果分成兩種表格。第一張表呈現 baseline family 的定位；第二組表格呈現目前已完成的 baseline 結果。以下數值使用既有結果檔與 prediction 檔重新整理；其中 `v2` 代表 `reports/reference_baseline_v2_results.csv`，`highres` 代表 `reports/reference_baseline_v2_highres_results.csv`。由於部分模型的可用樣本數不同，以下屬於 available-case comparison，正式主表仍建議另補 common-subset comparison。

| Group | Method | Metadata | Text | Image | RAG | Literature support / adaptation | Role |
|---|---|---|---|---|---|---|---|
| Sanity | `F0-Mean` | No | No | No | No | No | Lowest-reference predictor |
| Sanity | `F0-Ridge-Meta` | Yes | No | No | No | No | Linear metadata baseline |
| Metadata classical | `F1-RF-Meta` | Yes | No | No | No | Lo & Syu-style adapted route | Metadata-only classical ML |
| Metadata classical | `F1-GB-Meta` | Yes | No | No | No | Classical ML extension | Gradient boosting extension |
| Modality control | `T2-XGB-TextEmb` | No | Yes | No | No | No | Text-only baseline |
| Modality control | `I1-XGB-ImageEmb` | No | No | Yes | No | No | Image-only baseline |
| Simple fusion | `F2-XGB-Concat` | Yes | Yes | Yes | No | Visual-textual fusion literature | Early-fusion multimodal baseline |
| Literature-adapted | `C1` | Depends | Depends | Depends | No | External reference adapted | Anime multimodal MLP reference |
| Literature-adapted | `C2` | Depends | Depends | Depends | No | External reference adapted | Cross-modal transformer fusion reference |
| Literature-adapted | `C3` | Yes | Yes | Yes | Yes | External reference inspired/adapted | Retrieval-augmented reference |

Popularity baseline results:

| Method | Source | n_test | MAE | log_MAE | log_R2 | factor_acc_2x | Spearman | raw R2 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `F0-Mean` | v2 | 3087 | 14935.0718 | 1.9884 | -0.0000 | 0.1950 | 0.0000 | -0.1479 |
| `F0-Ridge-Meta` | v2 | 3087 | 12802.2336 | 1.0551 | 0.6855 | 0.3962 | 0.8007 | -0.4020 |
| `F1-RF-Meta` | v2 | 3087 | 8551.7168 | 0.8938 | 0.7554 | 0.4891 | 0.8420 | 0.5865 |
| `F1-GB-Meta` | v2 | 3087 | 9006.6811 | 0.8958 | 0.7568 | 0.4911 | 0.8303 | 0.5004 |
| `T2-XGB-TextEmb` | v2 | 2808 | 15203.9965 | 1.5077 | 0.3920 | 0.2817 | 0.6433 | -0.0621 |
| `I1-XGB-ImageEmb` | highres | 3087 | 12100.8365 | 1.3590 | 0.4804 | 0.3058 | 0.7257 | 0.2096 |
| `F2-XGB-Concat` | highres | 2808 | 9539.7047 | 0.8828 | 0.7760 | 0.4708 | 0.8650 | 0.5515 |
| `C1-Armenta-ProjectInputProxy` | highres | 2808 | 11095.5952 | 0.9538 | 0.7310 | 0.4626 | 0.8418 | 0.4332 |
| `C1-Armenta-ProjectInputReconstruction` | v2 | 3087 | 10501.5398 | 1.0244 | 0.6953 | 0.4137 | 0.8149 | 0.3963 |
| `C2-ProjectInputCrossAttention` | highres | 2808 | 11193.8393 | 0.9236 | 0.7525 | 0.4601 | 0.8647 | 0.4478 |
| `C2-ProjectInputRecurrentFusion` | highres | 2808 | 10469.8080 | 0.9151 | 0.7607 | 0.4605 | 0.8673 | 0.4748 |
| `C2-ProjectInputCTNNReconstruction` | v2 | 3087 | 10448.2886 | 0.9725 | 0.7280 | 0.4321 | 0.8481 | 0.4189 |
| `C3-RAG-Selective-XGB` | highres | 2808 | 9256.1195 | 0.9266 | 0.7537 | 0.4665 | 0.8719 | 0.6182 |
| `C3-ProjectInputSKAPPProxy-XGB` | highres | 2808 | 10075.2543 | 0.9363 | 0.7521 | 0.4548 | 0.8633 | 0.5430 |
| `C3-ProjectInputSKAPPGraphProxy` | highres | 2808 | 11254.5741 | 0.9305 | 0.7575 | 0.4494 | 0.8737 | 0.3862 |
| `C3-SourceExact-Staged-K64` | source-exact diagnostic | 3087 | 99140.0794 | 3.4361 | -2.1272 | 0.0901 | 0.3170 | -15.0432 |

meanScore baseline results:

| Method | Source | n_test | MAE | acc_within_10pt | Spearman | R2 |
|---|---|---:|---:|---:|---:|---:|
| `F0-Mean` | v2 | 3087 | 10.9115 | 0.5083 | 0.0000 | -0.4631 |
| `F0-Ridge-Meta` | v2 | 3087 | 9.2029 | 0.6054 | 0.4913 | -0.1266 |
| `F1-RF-Meta` | v2 | 3087 | 8.0179 | 0.6761 | 0.5759 | 0.1111 |
| `F1-GB-Meta` | v2 | 3087 | 8.8758 | 0.6213 | 0.5265 | -0.0518 |
| `T2-XGB-TextEmb` | v2 | 2808 | 10.6671 | 0.5082 | 0.2262 | -0.4773 |
| `I1-XGB-ImageEmb` | highres | 3087 | 8.6345 | 0.6391 | 0.4180 | -0.0103 |
| `F2-XGB-Concat` | highres | 2808 | 8.2031 | 0.6556 | 0.5530 | 0.0562 |
| `C1-Armenta-ProjectInputProxy` | highres | 2808 | 8.4901 | 0.6503 | 0.4808 | -0.0187 |
| `C1-Armenta-ProjectInputReconstruction` | v2 | 3087 | 10.5367 | 0.5507 | 0.4447 | -0.4982 |
| `C2-ProjectInputCrossAttention` | highres | 2808 | 8.0630 | 0.6863 | 0.5044 | 0.0586 |
| `C2-ProjectInputRecurrentFusion` | highres | 2808 | 8.3908 | 0.6720 | 0.4895 | 0.0037 |
| `C2-ProjectInputCTNNReconstruction` | v2 | 3087 | 8.3066 | 0.6631 | 0.5269 | 0.0541 |
| `C3-RAG-Selective-XGB` | highres | 2808 | 8.0901 | 0.6667 | 0.5561 | 0.0884 |
| `C3-ProjectInputSKAPPProxy-XGB` | highres | 2808 | 7.8582 | 0.6912 | 0.5634 | 0.1274 |
| `C3-ProjectInputSKAPPGraphProxy` | highres | 2808 | 8.1218 | 0.6806 | 0.5169 | 0.0671 |
| `C3-SourceExact-Staged-K64` | source-exact diagnostic | 3087 | 19.8518 | 0.3061 | 0.1155 | -4.2271 |

目前結果顯示，`popularity` 的強基準並不只來自深度模型：`F1-RF-Meta` 已有相當強的 metadata-only 表現，代表播出前結構化資訊本身包含大量訊號。`F2-XGB-Concat` 在 `log_R2` 上表現最佳，說明簡單 early fusion 已具競爭力；但 `C3-RAG-Selective-XGB` 在原始尺度 `MAE/raw R2` 與 Spearman 上較突出，顯示 selective retrieval 對人氣排序與高人氣樣本的解釋有幫助。對 `meanScore` 而言，整體 R2 仍偏低，最佳結果為 `C3-ProjectInputSKAPPProxy-XGB`，表示 retrieved aggregate 對分數預測有一定幫助，但 `meanScore` 相較 `popularity` 更難由播出前資訊穩定預測。

此外，`C3-SourceExact-Staged-K64` 是 `c3_source_exact_pipeline.py` 完成的第一批 source-faithful staged diagnostic run，已涵蓋 `popularity` 與 `meanScore`，但使用 `top_k=64` urgent setting，而非 SKAPP source 預設的 `top_k=500`。其結果明顯不穩定：`popularity` 的 `test_log_R2=-2.1272`、`factor_acc_2x=0.0901`、`raw R2=-15.0432`；`meanScore` 的 `MAE=19.8518`、`acc_within_10pt=0.3061`、`R2=-4.2271`。兩個目標皆出現大量 prediction 被 clip 到 train-set 上下界的現象，其中 popularity 常落在 24.999998 或 231528.98，meanScore 則大量落在 27 或 85。此結果應作為 source-faithful 路線的 diagnostic finding，而不是主表中的最終 C3 external baseline。它的價值在於指出直接搬移 staged SKAPP/RRCP pipeline 到 anime-domain mapping 仍需要 target calibration、retrieval size、loss design 與重新訓練設定調整。

與 proposed framework 的差距需等主框架最終結果固定後再計算；正式文件不應先放空值表格。建議在主框架結果確認後，以 `proposed - baseline` 的形式補一張差距表，並優先比較 `F1-RF-Meta`、`F2-XGB-Concat`、C1/C2/C3 代表 row。這張表的目的不是重新排行，而是回答 proposed framework 是否真的超越 metadata-only、simple fusion 與 literature-adapted references。

### 4.3 Exp3: Out-Dataset Result

在 Exp1 與 Exp2 中，本研究分別檢驗了不同 baseline 與主框架在同一資料集切分下的預測效果，以及 RAG 機制在固定資料環境中的增益。然而，若僅以 in-dataset 的測試結果評估模型，可能無法充分反映模型面對資料分布改變時的穩健性。因此，Exp3 的目的並非單純追求 out-dataset 上的最高分，而是進一步檢驗各方法在 dataset shift 情境下的泛化能力。具體而言，本實驗關注當測試資料來源、時間分布或資料特徵與訓練資料存在差異時，baseline 與本研究主框架的表現是否仍能維持穩定。

本研究在 Exp3 中應同時比較 reference baseline、傳統機器學習 baseline，以及主框架在 out-dataset 測試集上的表現。相較於僅呈現 out-dataset 的絕對指標，本研究更重視模型從 in-dataset 到 out-dataset 的效能下降幅度。若某一方法在原測試集上表現良好，但在 out-dataset 下產生明顯退化，則代表該方法可能高度依賴原資料集中的分布特徵；反之，若模型在 out-dataset 下仍能維持相對穩定的誤差與相關性指標，則可說明其具有較佳的 robustness 與 cross-dataset generalization。

因此，Exp3 的分析重點應包含兩個層次。第一，應比較各方法在 out-dataset 上的 MAE、R² 與 Spearman correlation 等指標，以觀察不同方法在分布轉移後的絕對預測能力。第二，應計算各方法相對於 in-dataset 結果的 degradation，例如 MAE 上升幅度、R² 下降幅度與排序相關性的變化。透過此設計，本研究能進一步判斷主框架的優勢是否僅存在於原始資料分布內，或是在面對不同資料條件時仍能保有穩定的預測能力。

此外，reference baseline 在 Exp3 中亦具有重要意義。C1、C2 與 C3 分別代表既有多模態融合與檢索增強方法在本研究任務中的 adapted baseline。若這些外部參考方法在 out-dataset 下退化幅度較大，則可說明直接遷移既有架構至播出前動畫人氣預測任務時仍存在限制；若主框架相較之下具有較小的 degradation，則可進一步支持本研究所設計之 metadata、text、image 與 RAG 融合流程具有較佳的任務適應性。相反地，若部分 reference baseline 在 out-dataset 上較為穩定，也應被視為重要觀察，代表其架構中可能存在值得主框架後續吸收的設計。

#### 4.3.1 Reporting Plan

Exp3 的正式結果表應避免只列 out-dataset 絕對分數，而應同時列出 in-dataset 與 out-dataset 的差異。對 `popularity` 應至少比較 `log_MAE`、`log_R2`、`factor_acc_2x` 與 `spearman_rho` 的變化；對 `meanScore` 應至少比較 `MAE`、`R2`、`acc_within_10pt` 與 `spearman_rho` 的變化。若 out-dataset 結果尚未產出，本文不應填入空值表格，而應在結果產出後再加入 degradation table。

## 5. Future Works

### 5.1 Reference Baseline Completeness

雖然本研究已將 C1、C2 與 C3 作為外部文獻參考 baseline 納入比較，但目前實作仍屬於 literature-adapted reproduction，而非對原論文所有元件的 exact reproduction。這主要是因為三篇參考文獻的原始任務、資料來源與可用欄位皆與本研究的播出前動畫人氣預測設定不同。為了使 baseline 能與主框架在相同資料集、相同切分方式與相同評估指標下比較，本研究優先保留其核心架構思想，並將輸入對齊至本專案的 metadata、text、image 與 RAG 設定。未來若要進一步提升 reference baseline 的完整度，仍可針對各路線補足更接近原論文的資料與模型元件。

對於 C1 所參考的動畫人氣預測研究，未來可進一步補齊 main character descriptions 與 main character portraits。原論文的核心輸入並非一般動畫封面或作品後設資料，而是以 anime synopsis、主要角色描述與角色繪像作為多輸入深度模型的主要訊號。因此，若後續能從資料源中穩定取得主要角色文字描述，並建立角色圖像的擷取與清理流程，C1 baseline 將能更接近原論文的 character-centric multimodal design。這將有助於釐清角色層級資訊是否能為播出前動畫人氣預測提供額外貢獻。

對於 C2 所參考的電影票房預測研究，未來可補足更接近原始設計的 BERT text encoder 與 ViT visual stream。原論文針對電影評論文字使用 BERT 進行語意編碼，並同時使用 ResNet-50 與 Vision Transformer 擷取電影海報的視覺特徵。目前本研究的 C2 adaptation 受限於動畫資料集並不存在完全對應的 movie reviews 與 movie posters，因此以動畫描述文字與專案圖像特徵作為替代。未來若能為動畫作品建立更接近 review-style 或 pre-release discussion 的文字來源，並加入 ViT-based cover/banner visual encoder，將能更完整地檢驗 C2 類型 cross-modal transformer fusion 架構在動畫人氣預測上的可遷移性。

對於 C3 所參考的 SKAPP retrieval-based popularity prediction 研究，未來工作應進一步完成 source-faithful SKAPP full pipeline。C3 的價值不僅在於加入 RAG 特徵，而在於其 selective retrieval、retrieved candidate filtering 與後續融合機制。現階段本研究已完成 project-input RAG proxy 與若干 SKAPP-inspired baseline，並已完成 `c3_source_exact_pipeline.py` 在 `popularity` 與 `meanScore` 上的 source-faithful staged diagnostic run。然而，該 run 使用 `top_k=64` 而非原始 SKAPP 設定的 `top_k=500`，且兩個目標皆顯示 prediction 明顯飽和於 clipping boundary。因此，本研究不將其寫成已完成的最終 C3 external baseline，而是將其視為後續補強 reference baseline 完整度的重要診斷結果。後續仍需完成 `top_k=500`、校準檢查與 loss/target-space 設計修正後，才能更嚴格地評估 source-faithful SKAPP pipeline 在本任務上的可遷移性。

整體而言，未來 reference baseline 的改進方向並非單純增加更多模型，而是提高外部文獻方法與原始設計之間的對齊程度。透過補齊 C1 的角色層級資料、C2 的 BERT 與 ViT 雙路徑設計，以及 C3 的 source-faithful SKAPP retrieval pipeline，後續研究將能更嚴格地區分「本研究主框架的實際貢獻」與「外部方法在資料轉換後的適應能力」。這也有助於使 baseline comparison 不僅作為效能表格，而是成為分析不同多模態與檢索增強設計在播出前動畫人氣預測任務中適用性的依據。

## Integration Notes

- Exp1 的核心不是排行 baseline，而是依序回答 sanity、metadata strength、modality contribution、simple fusion、external-adapted reference comparison。
- C1/C2/C3 必須使用 `literature-adapted`、`project-input proxy`、`source-faithful diagnostic` 等字眼，不應寫成 exact reproduction。
- C3 source-faithful SKAPP full pipeline 目前已有 `top_k=64` popularity 與 meanScore diagnostic run，但尚未形成最終 baseline；正式文件應標註為 diagnostic/ongoing，而不是 completed external baseline。
- 若最終表格要比較 baseline 與主框架，應優先補齊 common-subset evaluation，或在表格註明 available-case comparison 與各模型 `n_test`。
