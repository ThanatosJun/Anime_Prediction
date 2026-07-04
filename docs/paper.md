# 3. 相關研究（Related Work）

影像視覺特徵的提取在多模態預測任務中扮演至關重要的角色。本研究以動漫封面圖像（cover image）與橫幅圖像（banner image）作為核心視覺輸入，此類圖像蘊含畫風風格、人物設計與色彩配置等豐富的語義資訊，對觀眾形成作品第一印象具有決定性影響。然而，從動漫圖像中提取具有判別性的視覺表徵面臨兩項根本性挑戰：其一，現有大規模視覺預訓練模型（如以 ImageNet 訓練的模型）主要以真實世界的自然圖像作為學習來源，動漫圖像所具有的平塗色彩、誇張比例與輪廓線條等視覺特性，與自然圖像的資料分佈存在顯著落差，直接套用通用預訓練權重難以有效捕捉動漫圖像的特有語義；其二，此類圖像缺乏視覺層面的人工標注，無法直接採用監督式學習進行特徵提取器的領域適應訓練。本節針對上述挑戰，就視覺骨幹網路的選用、自監督訓練策略的設計，以及影像前處理中的主體擷取等相關工作進行梳理。

### 3.1 視覺骨幹網路：Swin Transformer
在視覺骨幹網路的選用上，傳統卷積神經網路（Convolutional Neural Network, CNN）長期以來主導電腦視覺領域，其以固定大小的卷積核逐層提取局部特徵的設計在影像分類任務中表現穩健。然而，固定感受野的架構本質限制了 CNN 捕捉長距離空間依賴關係的能力，對於需要同時解析局部筆觸細節與整體畫風語義的動漫圖像而言，此一局限尤為顯著。Liu et al. 提出的 Swin Transformer [1] 透過引入階層式滑動視窗自注意力機制（shifted window self-attention），有效兼顧了局部特徵提取的精細性與全局上下文的感知能力。在計算效率方面，Swin Transformer 採用的視窗注意力設計使計算複雜度由傳統 Vision Transformer（ViT）的二次方降至線性，進一步支援了對高解析度圖像的高效處理。尤其值得關注的是，Swin Transformer 的架構設計使其在前向傳播過程中自然輸出四個不同尺度的特徵圖，維度分別為 128、256、512 與 1024，由淺至深依次對應影像從局部紋理到整體語義的多層次表達，此特性與動漫圖像多尺度視覺理解的需求高度契合。

### 3.2 自監督表徵學習：對比式學習
在自監督訓練策略的設計上，由於動漫封面圖缺乏視覺語義層面的人工標注資料，監督式學習所需的標注成本難以承受，而直接沿用 ImageNet 預訓練權重亦未必能有效捕捉動漫圖像的特有視覺風格分佈。對比式學習（contrastive learning）作為近年來自監督表徵學習的主流範式，提供了一條在無標注情境下習得判別性視覺表徵的可行路徑。Chen et al. 提出的 SimCLR 框架 [2] 對任意輸入圖像施加不同的隨機資料增強操作——包括隨機裁剪、顏色扭曲與高斯模糊——以生成兩個語義等價但外觀相異的視圖，構成正樣本對；同一批次中的其他圖像則作為負樣本。透過帶有溫度參數的正規化交叉熵損失函數（NT-Xent），模型被訓練在特徵空間中使正樣本對的距離最小化、負樣本對的距離最大化，從而習得對語義無關變換保持不變性的視覺表徵。其損失函數定義如下：

$$\mathcal{L} = -\log \frac{\exp\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_j) / \tau\right)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp\left(\text{sim}(\mathbf{z}_i, \mathbf{z}_k) / \tau\right)}$$

其中 $\text{sim}(\mathbf{u}, \mathbf{v}) = \mathbf{u}^\top \mathbf{v} / (\|\mathbf{u}\| \|\mathbf{v}\|)$ 為 cosine similarity，$\tau$ 為溫度參數，$N$ 為批次大小，$\mathbf{1}_{[k \neq i]}$ 為指示函數，表示排除自身以外的所有樣本作為負樣本。SimCLR 的大規模實驗結果驗證了此訓練範式的有效性：在僅使用線性分類器的線性評估協議下，模型在 ImageNet 資料集上達到 76.5% 的 top-1 分類準確率，與使用完整標籤訓練的監督式 ResNet-50 相當；在半監督學習設定下，模型僅以 1% 的標注樣本進行微調，即可達到 85.8% 的 top-5 準確率，超越使用全量標籤訓練的 AlexNet，充分展現了對比式學習在無標注情境下習得通用視覺表徵的潛力。基於上述優勢，本研究採用 InfoNCE loss 作為訓練目標，對 Swin Transformer 進行自監督領域適應，具體訓練設計詳見第 4 節。

### 3.3 影像前處理：主體擷取與雜訊抑制
在影像前處理中的主體擷取方面，動漫作品中人物角色的設計風格——包括造型、服裝與表情——是影響觀眾視覺印象與作品吸引力的核心要素。相較之下，背景元素雖豐富畫面層次，卻非預測作品熱度的關鍵視覺信號。然而，動漫封面圖通常包含複雜的背景元素，包括場景環境、視覺特效與文字標題等。研究表明，當模型直接以完整圖像作為輸入時，容易習得與目標語義無關的背景線索，形成偽相關（spurious correlations），導致所提取的視覺表徵反映背景風格而非核心語義，進而損及特徵的跨域泛化能力。Beery et al. 於 "To Crop or Not to Crop" [3] 一文中，針對大規模生態相機影像分類任務，系統性比較了直接以完整圖像進行推論，以及先透過物件偵測器識別主體、裁切後再進行分類的兩種推論策略。實驗結果顯示，在類別分布嚴重不均衡的資料集上，兩階段裁切流程使宏觀平均 F1 分數（Macro-average F1）提升約 25%；作者亦指出，裁切操作透過強制模型聚焦於主體視覺特徵、排除背景干擾，有效緩解了偽相關問題，並在跨場景的泛化評估中取得顯著優勢。上述發現表明，在特徵提取前先行進行主體擷取，是提升視覺表徵品質與模型泛化能力的有效手段。本研究將此思路應用於動漫圖像的前處理流程，在送入視覺骨幹網路前，先以 YOLO-based 動漫人物偵測模型對封面圖進行人物定位與裁切，使後續特徵提取過程得以專注於人物本身的視覺語義，而非受背景雜訊左右。

---

# 4. 研究方法（Methodology）

本研究的影像特徵提取流程由四個依序執行的階段構成：資料收集、影像前處理、自監督表徵學習訓練，以及特徵向量輸出。整體設計目標在於從無標注的動漫封面圖像與橫幅圖像中，提取具有判別性的視覺表徵，以供下游熱度預測模型使用。

### 4.1 資料收集

本研究的圖像資料來源為 AniList 動漫資料庫，透過 API 取得的結構化資料中，每筆記錄包含兩類視覺素材的 URL：封面圖像（coverImage）與橫幅圖像（bannerImage）。資料收集階段依據 URL 逐張下載圖像並儲存於本機快取，以避免後續處理時重複發送網路請求。每張圖像以對應的 AniList 動畫 ID 與圖像類別命名（格式為 `{id}_{col}.jpg`），確保圖像與原始資料的一對一對應關係。下載過程中，所有請求的成功與失敗狀態均記錄於日誌檔案，供後續流程進行資料過濾與品質控管。

### 4.2 影像前處理：人物偵測與裁切

如第 3.3 節所述，動漫封面圖像中的背景元素可能引發偽相關問題，損及特徵提取的品質。為此，本研究在送入視覺骨幹網路前，先對圖像進行人物偵測與裁切作為前處理步驟。

此前處理步驟由設定參數 `yolo.use` 控制。當 `yolo.use = true` 時，對每張圖像首先進行上採樣（upscale）至 640 像素，以提升小尺寸圖像中目標的偵測率；隨後依 `detect_mode` 設定執行對應的偵測策略：`person` 模式僅偵測人物、`face` 模式僅偵測臉部、`both` 模式則同時執行兩者並合併結果。偵測結果依信心分數由高至低排序，取前 `max_detections` 個邊界框分別裁切；每個裁切區域隨後經 ResizeWithPad(224) 等比縮放並補零邊至 224×224 後，再送入 transform pipeline 進行標準化，最終依序通過 Swin Transformer 提取特徵，並以 mean pooling 合併為單一嵌入向量。若未偵測到任何目標，則以整張圖像作為 fallback 輸入。當 `yolo.use = false` 時，此階段略過，直接以完整圖像送入後續特徵提取流程。

### 4.3 自監督表徵學習

**骨幹網路。** 本研究採用 Swin Transformer [1]（`microsoft/swin-base-patch4-window7-224`）作為視覺特徵提取的骨幹網路，並取其 `pooler_output` 作為圖像的全局表徵，輸出維度為 1024。

**資料增強策略。** 對每張圖像首先施以 ResizeWithPad 操作，等比縮放至 224×224 並以零值填補邊緣，以保留圖像的完整畫面比例。此後，每張圖像分別經由兩條 transform pipeline 進行處理，生成一組訓練樣本對：原始視圖（original view）僅執行標準化（Normalize）以 ImageNet 的均值與標準差進行正規化；增強視圖（augmented view）則依序套用隨機大小裁剪（RandomResizedCrop，p=1.0）、隨機裁切（RandomCrop，p=0.3）、顏色扭曲（ColorJitter，p=0.8）、高斯模糊（GaussianBlur，p=0.5）、水平翻轉（RandomHorizontalFlip，p=0.5）及隨機灰階（RandomGrayscale，p=0.2）等增強操作。兩個視圖構成正樣本對，批次內其餘圖像的原始視圖則自動作為負樣本。

**訓練目標。** 本研究以 InfoNCE loss（即 NT-Xent）作為訓練目標，目標函數定義如第 3.2 節所示。模型在訓練過程中以原始視圖的 embedding 作為 anchor，使增強視圖的 embedding 在特徵空間中向其靠攏，同時遠離批次內其他圖像的表徵，促使模型習得對視覺變換保持不變性、對不同圖像保有判別性的視覺表徵。

**學習率排程。** 訓練採用兩階段學習率排程：前期執行線性 warmup，學習率由 0 線性上升至目標值；warmup 結束後切換為 Cosine Annealing，學習率餘弦衰減至趨近 0。此設計有效避免訓練初期因學習率過高導致的不穩定，並在後期以漸進收斂的方式提升表徵品質。

**模型儲存。** 訓練過程中，當驗證集損失出現改善時，模型以 HuggingFace 格式儲存為當前最佳權重；此外，每隔固定 epoch 數額外儲存含 optimizer 狀態的完整 checkpoint，支援訓練中斷後的續訓需求。

### 4.4 特徵向量輸出

訓練完成後，以最佳權重對全資料集進行批次推論，分別對封面圖像與橫幅圖像提取視覺嵌入向量。輸出模式由設定參數 `stage` 控制，支援兩種格式：

當 `stage = false` 時，取模型的 `pooler_output` 作為圖像的全局表徵，每張圖像輸出 1024 維的嵌入向量；輸出欄位為 `coverImage_emb`（1024 維）與 `bannerImage_emb`（1024 維）。

當 `stage = true` 時，提取 Swin Transformer 四個階段的中間特徵圖（`reshaped_hidden_states` 前四層，第五層與第四層高度重複故略去），各自經全局平均池化（Global Average Pooling）後，輸出四組不同尺度的嵌入向量，維度分別為 128、256、512 與 1024，依序對應從局部紋理到整體語義的多層次視覺表達；輸出欄位為 `coverImage_emb_s0` 至 `coverImage_emb_s3`（bannerImage 同）。

推論階段同樣受 `yolo.use` 參數控制。當 `yolo.use = true` 時，每張圖像先經 YOLO 偵測產生最多 `max_detections` 個裁切區域，各裁切區域分別通過 Swin Transformer 取得特徵向量後，以 mean pooling 合併為該圖像的單一嵌入向量；當 `yolo.use = false` 時，完整圖像直接送入模型取得嵌入向量。`yolo.use` 的開關不影響輸出的維度與欄位結構，差異僅在特徵計算路徑；輸出格式的變化由 `stage` 參數決定。

兩類圖像的 embedding 依 AniList 動畫 ID 進行對齊後合併，儲存為 Parquet 格式（`data/processed/image_embeddings.parquet`），供下游融合模型直接讀取使用。

### 4.5 實作細節（Implementation Details）

**資料參數**

| 參數 | 值 |
|------|----|
| 輸入圖像欄位 | `coverImage_medium`、`bannerImage` |
| 輸入解析度（`image_size`） | 224 |

**骨幹模型參數**

| 參數 | 值 | 說明 |
|------|----|----|
| `name` | `microsoft/swin-base-patch4-window7-224` | 骨幹網路 |
| `pretrained` | true | 使用預訓練權重 |
| `stage` | false | false = `pooler_output`（1024 維）；true = 各 stage 分開（s0–s3） |
| `fusion_embed_mode` | pooler | pooler = 1024 維；stage = 1920 維（128+256+512+1024 concat） |

**訓練參數**

| 參數 | 值 |
|------|----|
| `batch_size` | 64 |
| 溫度參數（τ） | 0.07 |
| `device` | cuda |
| 學習率排程 | Linear Warmup + Cosine Annealing |

**YOLO 偵測參數**

| 參數 | 值 | 說明 |
|------|----|----|
| `detect_mode` | both | `person` / `face` / `both` |
| `yolo.use` | true | 是否啟用 YOLO 前處理 |
| 人物偵測 `repo_id` | `deepghs/anime_person_detection` | level=m、version=v1.1 |
| 人物偵測 `conf_threshold` | 0.2 | |
| 人物偵測 `iou_threshold` | 0.8 | |
| 人物偵測 `max_detections` | 5 | |
| 人物偵測 `min_bbox_ratio` | 0.05 | |
| 臉部偵測 `repo_id` | `deepghs/anime_face_detection` | level=s、version=v1.4 |
| 臉部偵測 `conf_threshold` | 0.15 | |
| 臉部偵測 `iou_threshold` | 0.9 | |
| 臉部偵測 `max_detections` | 5 | |
| 臉部偵測 `min_bbox_ratio` | 0.02 | |

**推論參數**

| 參數 | 值 | 說明 |
|------|----|----|
| `model_path` | `src_2/component_image/model-image/best` | 訓練好的 Swin checkpoint |
| `embedding_path` | `data/processed/image_embeddings.parquet` | 輸出路徑 |
| `file_kind` | file | `file`（CSV 批次）/ `image`（單張） |
| `use_yolo` | true | 覆蓋 `yolo_detection.yolo.use` |

---

## References

[1] Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., Lin, S., & Guo, B. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 10012–10022.

[2] Chen, T., Kornblith, S., Norouzi, M., & Hinton, G. (2020). A Simple Framework for Contrastive Learning of Visual Representations. *Proceedings of the 37th International Conference on Machine Learning (ICML)*, PMLR 119, 1597–1607.

[3] Beery, S., Cole, E., Parker, J., Perona, P., & Winner, K. To Crop or Not to Crop: Comparing Whole-Image and Detection-Based Inference Methods for Camera Trap Classification. *arXiv preprint*.
