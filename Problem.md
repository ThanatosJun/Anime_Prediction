1. Loss function 是使用哪一個
    - 可以參考 Poisson Regression Loss
    - 於 deep learning 最後一層：Softplus: $f(x) = \ln(1 + e^x)$
    -> loss function 最終選用 Huber Loss
2. 可能可以使用的指標：
    - MAPE, Mean Absolute Percentage Error
    - Spearman's rho
3. 可以修改 deep learning 的 layers
    - 既然「冷門番」和「霸權番」的爆紅邏輯不同，我們不要讓同一個全連接層（Dense Layer）去預測所有人氣。
4. Distrbution shift 的問題
    - 可能需要一些 domain adaptation 的方法
    - 例如：在訓練時加入一些 regularization term，讓模型對於 distribution shift 更加 robust

===
1. 是否加入 MSE 作為論文 Anime popularity prediction before huge investments: a multimodal approach using deep learning 的對比，論文是用 log MSE 計算的
2. 資料集問題
    - 是否需要補足 description 的資料，因為有缺失就沒有該筆 text embedding
3. optimizer 的選擇
    - AdamW 是目前的選擇，但是否需要嘗試其他 optimizer，例如：RAdam, Ranger 等等

===
1. 資料遷移問題：popularity 跟 meanscore 的分布 shift 很大，可能需要一些 domain adaptation 的方法。
    - 或許嘗試使用一個模型預測眾數 popularity 跟 meanscore。讓我們可以根據年份預測模型。


# text（全部 splits）+ image（train only）
# Step 1：RAG embeddings（text 全 splits + image train only）
python src_2/RAG/run_build_embeddings.py

# Step 2：YOLO crop
python src_2/component_image/run_yolo_crop.py --splits train val test holdout_unknown

# Step 3：Fusion image embeddings（yolo + cover + banner）
python src_2/component_image/run_swin_embedding.py --splits train val test holdout_unknown

# Step 4：建立 Qdrant collection
python src_2/RAG/rag_builder.py

# Step 5：查詢 RAG
python src_2/RAG/rag_query.py --splits train val test holdout_unknown

LoRA (Low-Rank Adaptation)

實驗結果的可解釋性：
1. RAG： attetion heatmap
2. Fusion Model Captumn and SHAP
完成驗證後，幫我生成一份操作手冊，讓新的開發者可以快速部署這份系統，包含以下內容：
1. 每個步驟的指令
2. 每個步驟的輸出說明

要完成的三項主要實驗，外加一個分析：
1. 整個 pipeline 的最佳效果
2. RAG 加入的影響
3. 外部資料的比對 (這個實驗先跳過)
4. SHAP , Captum , attention heatmap 的可解釋性分析

接下來我們要完成的三項主要實驗，外加一個分析，幫我加入 order.md 當中：
1. 整個 pipeline 的最佳效果
2. RAG 加入的影響
3. 消融實驗 (attention, image)這兩者是否加入 MLP 輸入的消融
4. SHAP , Captum , attention heatmap 的可解釋性分析

id 154587
English:Sousou no Frieren

id 170068

Bleach
id 1686
id 2889
id 4835
id 100719
id 8247
id 116674
1686 2889 4835 100719 8247 116674