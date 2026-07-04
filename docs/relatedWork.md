# Related Work

## 影像特徵提取與視覺表徵學習

### 問題背景

在預測動漫播出前熱度的多模態系統中，影像特徵的品質直接影響最終預測效能。動漫封面圖（cover image）是觀眾對一部作品的第一印象，蘊含畫風、人物設計、色彩配置等豐富視覺資訊。然而，如何從這些圖像中提取出有意義的語義特徵，面臨兩個核心挑戰：第一，動漫圖像風格差異極大，從精緻寫實到誇張抽象皆有，需要能同時理解局部細節與全局語義的特徵提取器；第二，這些圖像缺乏視覺層面的人工標注，無法直接採用監督式學習。

---

## 一、視覺 Backbone 的選擇：Swin Transformer

### 1.1 CNN 的局限

傳統卷積神經網路（CNN）以固定大小的卷積核提取局部特徵，雖然在影像識別任務中表現穩定，但固定感受野的限制使其難以捕捉長距離的空間依賴關係。對於動漫圖像而言，局部紋理（如線條筆觸、色塊分布）與全局語義（如整體畫風、構圖風格）同樣重要，CNN 難以在兩者之間取得平衡。

### 1.2 Swin Transformer 的優勢

Liu et al.（2021）提出的 **Swin Transformer** 採用階層式滑動視窗注意力機制（shifted window attention），有效解決了上述問題。與傳統 Vision Transformer（ViT）的二次方計算複雜度相比，Swin 的視窗設計將複雜度降至線性，同時保留了 CNN 的多尺度階層特性。

Swin Transformer 原生輸出四個 stage 的特徵圖，維度分別為 128、256、512、1024，對應從局部到全局不同的抽象層次：

| Stage | 維度 | 捕捉的視覺資訊 |
|-------|------|----------------|
| 0 | 128 | 局部紋理、線條筆觸 |
| 1 | 256 | 色塊分布、局部結構 |
| 2 | 512 | 人物部位、光影風格 |
| 3 | 1024 | 整體語義、畫風流派 |

此多尺度輸出天然契合動漫圖像的理解需求，因此本研究選用 `microsoft/swin-base-patch4-window7-224` 作為視覺特徵提取的 backbone。

> 參考文獻：Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *ICCV 2021*.

---

## 二、自監督訓練策略：對比式學習

### 2.1 為何無法使用監督式學習

動漫封面圖缺乏視覺語義層面的人工標注（如畫風類別、視覺品質分數），無法直接以監督式方式訓練特徵提取器。若僅使用 ImageNet 預訓練權重，模型未必能有效理解動漫圖像的特有視覺風格。

### 2.2 對比式學習的核心機制

Chen et al.（2020）提出的 **SimCLR** 框架奠定了現代對比式學習的基礎：對同一張圖片施加不同的隨機資料增強（隨機裁剪、顏色扭曲、高斯模糊），產生兩個視角作為正樣本對，並以 batch 內其他圖片作為負樣本，透過 NT-Xent 損失函數（即 InfoNCE 的變體）訓練模型拉近正樣本、推遠負樣本。

### 2.3 實驗效益

SimCLR 的實驗結果驗證了對比式學習的強大表達能力：

- **線性評估（Linear Evaluation）**：僅用線性分類器，在 ImageNet 達到 **76.5% top-1 準確率**，追平使用完整標籤的監督式 ResNet-50
- **半監督學習**：僅用 1% 標注資料微調，達到 **85.8% top-5 準確率**，超越使用 100% 標籤的 AlexNet
- **遷移學習**：在 12 個資料集中 5 個超越監督式預訓練基準

### 2.4 本研究的應用

本研究以 **InfoNCE loss** 作為訓練目標，延續 SimCLR 的核心設計：在無標注的動漫圖像上，透過資料增強讓模型學習對色彩、裁切、模糊等變換保持不變性，同時保有區分不同圖片的判別能力，進而從動漫封面圖中提取出有意義的視覺表徵。

> 參考文獻：Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *ICML 2020*.

---

## 三、前處理雜訊抑制：先裁切再嵌入

### 3.1 完整圖像的偽相關問題

動漫封面圖通常包含複雜的背景元素（場景、特效、文字等），若直接對完整圖像進行特徵提取，模型可能學到與人物無關的背景特徵，形成**偽相關（spurious correlations）**，使 embedding 反映的是背景風格而非人物設計，降低特徵的泛化能力。

### 3.2 裁切前處理的實驗依據

Beery et al. 在「To Crop or Not to Crop」中，系統性比較了直接對完整圖像分類與先偵測裁切再分類兩種方法。實驗結果顯示：

- 在大規模、類別不平衡的資料集上，加入物件偵測與裁切的流程使 **Macro-average F1 提升約 25%**
- 裁切強制分類器專注於主體，有效消除背景雜訊與無關資訊的干擾
- 在所有評估指標中，包含裁切的模型表現均優於直接使用完整圖像的方法

### 3.3 本研究的應用

本研究借鑑此概念，在特徵提取前先以 YOLO-based 動漫人物偵測模型（`dghs-imgutils`，基於 `deepghs/anime_person_detection`）對圖像進行人物偵測與裁切，強制後續的 Swin Transformer 專注於人物本身的視覺特徵。當圖像未偵測到人物時，則 fallback 使用整張圖像，確保流程的穩健性。

> 參考文獻：Beery, S., et al. To Crop or Not to Crop: Comparing Whole-Image and Detection-Based Inference Methods for Camera Trap Classification.
