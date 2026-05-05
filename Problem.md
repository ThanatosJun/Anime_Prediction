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