# Reference Baseline 論文對齊稽核

更新日期：2026-05-19

本文件記錄目前已實作或規劃中的 reference baseline，是否真的對齊錨定論文中的架構。核心目的不是幫模型背書，而是把「可主張的程度」寫清楚，避免把 adaptation 或 proxy 誤寫成 reproduction。

## 共同判準

C1、C2、C3 一律只用兩個條件判斷是否能當主線 reference baseline：

1. 輸入必須對齊本專案主框架：baseline 要使用本專案設計的主輸入契約，也就是 metadata、synopsis/text embedding、cover/banner image，以及必要時的 project retrieval context。若改用本專案主框架沒有的輸入，例如 C1 character portraits、C2 movie reviews、C3 social-media UGC graph/context，就不能作為主線 comparison row。
2. 模型方法要在上述輸入限制下盡量還原原論文：輸入固定為本專案主框架後，fusion shape、attention/retrieval/refinement module、MLP depth、encoder choice 等才往原論文靠近。可主張的程度取決於還原到哪一層；目前多數只能寫成 project-input proxy/adaptation，不能寫成原論文 reproduction。

2026-05-19 補充：若 baseline 要作為主要論文對比，不能只做到
motivation-level proxy。可接受的最低標準應改為
structure-complete reconstruction：在不破壞本專案輸入契約與 temporal
evaluation 的前提下，必須把原論文的主要構造、訓練階段、mask/attention
邏輯完整重做。現有 proxy rows 只能作為開發里程碑或 ablation 參照，不能當
final reference row。

## 總覽

| Baseline | 對應論文路線 | 目前對齊程度 | 報告寫法 |
|---|---|---|---|
| `C1-Armenta-MLP` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | 寬鬆 adaptation | 只能作為 anime-domain multimodal MLP 的 first-pass adaptation，不能稱框架重現。 |
| `C1-Armenta-ProxyBranchMLP` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | project-aligned proxy adaptation | 代表本專案早期 C1 branch-fusion route；三個 project feature groups 平行分支後融合，但還不是 Armenta 的 character/context MLP + Big MLP 形狀。 |
| `C1-Armenta-ProjectInputProxy` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | stronger project-input proxy adaptation | 目前較適合代表 C1 主線：保留本專案 metadata / synopsis-text / cover-banner image artifacts，同時改成 synopsis branch + project-context MLP + Big MLP。仍不可稱 Figure 2 或 GPT-2/ResNet-50 復現。 |
| `C1-Armenta-ProjectInputProxy-ResNet50` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | stronger project-input proxy adaptation | 目前 C1 中 visual encoder 最接近原論文的一版：保留本專案 metadata / synopsis-text / cover-banner inputs，並用 ImageNet ResNet-50 抽 cover/banner features；仍不可稱 Figure 2、GPT-2 或 character-branch 復現。 |
| `C1-Armenta-ProjectInputReconstruction` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | structure-complete project-input reconstruction | 已補 GPT-2 synopsis embeddings、ResNet-50 cover/banner features、project-context MLP 與 Armenta Big MLP；因仍以 project metadata/cover/banner 取代 character descriptions/portraits，不可稱 exact reproduction。 |
| `C1-Armenta-Figure2Reconstruction` | Armenta-Segura & Sidorov 2025 Figure 2 | figure2 side reconstruction | 已使用 GPT-2 synopsis、GPT-2 main-character description/name、ResNet-50 main-character portrait、character MLP 與 Big MLP；因輸入轉向 character-specific artifacts，不符合本專案 cover/banner 主輸入契約，因此只能作旁支分析，不作主線 baseline。 |
| `C2-CTNN-Lite` | Madongo, Tang & Hassan 2023 CTNN | first-pass project-input adaptation | 只能稱 lightweight cross-modal transformer fusion adaptation；它對齊本專案 text/image inputs，但還沒有 explicit cross-attention、metadata fusion、recurrent-style fusion，不能稱 CTNN reproduction。 |
| `C2-ProjectInputCrossAttention` | Madongo, Tang & Hassan 2023 CTNN | stronger project-input proxy adaptation | 目前較適合代表 C2 主線：先滿足本專案 synopsis/cover-banner/metadata inputs，再用 explicit bidirectional text-image cross-attention 與 metadata-conditioned fusion 往 CTNN 靠近；改用 movie reviews 的版本不列主線，因為違反條件 1。 |
| `C2-ProjectInputRecurrentFusion` | Madongo, Tang & Hassan 2023 CTNN | stronger project-input proxy adaptation | 在 CrossAttention 之上補 GRU recurrent fusion，覆蓋原文 recurrent-fusion 動機；結果上 meanScore 略優、popularity Spearman 略優，但 popularity R2 低於 CrossAttention。 |
| `C2-ProjectInputCTNNReconstruction` | Madongo, Tang & Hassan 2023 CTNN | structure-complete project-input reconstruction | 已補 GPT-2 synopsis branch、ResNet-50 visual branch、modality transformer encoders、bidirectional cross-modal attention、GRU recurrent fusion 與 metadata factor gate；因不使用 movie reviews/posters/box-office classes，不可稱 exact CTNN reproduction。 |
| `C2-ProjectInputCTNNDualVisualReconstruction` | Madongo, Tang & Hassan 2023 CTNN | structure-complete project-input dual-visual diagnostic | 已在 CTNN reconstruction 上補 project image embedding 作為 ViT-like visual semantic stream，與 ResNet-50 cover/banner features 一起形成雙視覺來源；因不是 MovieNet-finetuned ViT poster features，且性能不全面優於主線，只能作 source-alignment diagnostic。 |
| `C3-RAG-*` | Xu et al. 2025 SKAPP | project-aligned retrieval baseline | 目前完成 `none/sparse/dense/hybrid/selective` retrieval baselines；`selective` 可作本專案 RAG 主線，但只能稱 SKAPP-inspired/project retrieval proxy，不是 RRCP/VL-GNN/RRCP-Attention reproduction。 |
| `C3-ProjectInputSKAPPProxy-XGB` | Xu et al. 2025 SKAPP | stronger project-input proxy adaptation | 已保留 project historical-anime retrieval，並補上 train-only learned contribution scoring、learned filtering、attention-weighted retrieved context aggregation；仍不是 SKAPP reproduction，因為沒有 VL-GNN 或 RRCP-Attention。 |
| `C3-ProjectInputSKAPPGraphProxy` | Xu et al. 2025 SKAPP | intermediate architecture proxy | 已補 retrieved text/image/label/contribution tensors、RRCP-style mask、learned graph adjacency、contribution-aware attention head；但 RRCP_silver、all-items/dissembled training、GraphLearner 細節、RRCP/CXMI attention 仍未完整重做，因此只能當中間版本。 |
| `C3-ProjectInputSKAPPFull` | Xu et al. 2025 SKAPP | required next reconstruction target | 目標是完整重做 SKAPP 的 retrieval dataset、all-items RRCP model、dissembled model、RRCP_silver generation、threshold variable-length filtering、GraphLearner、RRCP/CXMI attention prediction head；只替換 domain-specific UGC input 為本專案 anime project inputs。 |

## 復現可行性矩陣

| 路線 | 若要 exact reproduction 需要什麼 | 目前專案已有什麼 | 主要限制 | 實務上下一層可做什麼 |
|---|---|---|---|---|
| `C1` Armenta anime deep model | MAL-style dataset、synopsis、main-character descriptions、main-character portraits、GPT-2 text branches、ResNet-50 portrait branch、character MLP、Big MLP，以及原論文 split/target 設定。 | AniList processed metadata、GPT-2 synopsis embeddings、cover/banner raw images、cover/banner ResNet-50 features、`voice_actor_names` cast metadata、raw AniList character JSON、main-character description/name GPT-2 artifacts、main-character portrait ResNet-50 artifacts、flat MLP、branch-wise proxy MLP、project-input Armenta-shaped proxy MLP、structure-complete project-input reconstruction、Figure 2 side reconstruction。 | character-description / character-portrait branch 已能做旁支 reconstruction，但不能納入主線；target/split 與原論文不同。Figure 2 旁支更貼近論文，但偏離本專案主框架的 cover/banner image setup。 | 主線仍用 `C1-Armenta-ProjectInputReconstruction` 作為目前 C1 對齊版：先守住本專案輸入契約，再還原 GPT-2 synopsis、ResNet-50 visual encoder、project-context MLP 與 Big MLP。`C1-Armenta-Figure2Reconstruction` 保留為 side analysis。 |
| `C2` CTNN box-office model | movie poster + movie review dataset、ResNet50 + ViT poster features、BERT review features、cross-modal attention transformer、recurrent fusion、metadata factors，以及 box-office class/range target。 | anime metadata、GPT-2 synopsis embeddings、anime cover/banner ResNet-50 features、project image embeddings as ViT-like visual proxy、two-token TransformerEncoder fusion、project-input cross-attention fusion、project-input recurrent fusion、structure-complete CTNN reconstruction、dual-visual diagnostic、regression targets。 | domain、inputs、feature extractors、target formulation 都不同；目前 pre-release anime dataset 沒有 movie review 等價訊號。若硬改成 movie-review CTNN，會偏離本專案輸入契約。 | 主線仍用 `C2-ProjectInputCTNNReconstruction` 作為目前 C2 對齊版；`C2-ProjectInputCTNNDualVisualReconstruction` 保留為 ResNet50+ViT-like source-alignment diagnostic。 |
| `C3` SKAPP retrieval model | UGC knowledge base、multimodal/meta retriever、top-k retrieval、RRCP selective refiner、VL-GNN contextual learning、RRCP-Attention prediction network，以及 social-media popularity targets。 | Offline train-set knowledge base、`none/sparse/dense/hybrid/selective/skapp_proxy/skapp_graph_proxy` artifacts、metadata sparse retrieval、text dense retrieval、hybrid RRF retrieval、train-only learned contribution scorer、retrieved text/image/label/contribution tensors、RRCP-style mask、learned graph adjacency、contribution-aware attention head。 | RRCP_silver、all-items/dissembled training、VL-GNN、RRCP/CXMI attention 仍未完整重做；anime release metadata 也缺少 SKAPP 使用的 user/social context。若硬改成 social-media UGC SKAPP，會換成另一個任務。 | `C3-ProjectInputSKAPPFull` 必須成為下一個 reconstruction target；既有 C3 rows 只能當 ablation 或 development milestones。 |

## C1：Armenta-Segura & Sidorov 2025

已核對的本地 PDF：

```text
docs/refer/Anime popularity prediction before huge investments a multimodal approach using deep learning.pdf
```

原論文關鍵元件：

1. Synopsis branch：GPT-2 pretrained model，輸出 768 維。
2. Main character description branch：GPT-2 pretrained model。
3. Main character portrait branch：ResNet-50 pretrained model。
4. Character portrait 與 character description embeddings 會先 concat，再透過 character MLP 形成 unified 768-dimensional character embedding。
5. Character output 再與 synopsis GPT-2 embeddings concat，送入 Big MLP 做 regression。
6. 實驗包含 full model、synopsis only、portraits only、descriptions only、traditional benchmark。
7. 原論文 target 是 MAL weighted average score，split 也依 shared main characters 設計，不是本專案 temporal split。

目前專案 baseline 可分成兩層：

```text
C1-Armenta-MLP = metadata + project text embeddings + project image embeddings -> sklearn MLPRegressor
```

較強 proxy baseline：

```text
C1-Armenta-ProxyBranchMLP =
    metadata proxy branch MLP
  + text embedding branch MLP
  + image embedding branch MLP
  -> fusion MLP -> regression
```

其中 `C1-Armenta-ProxyBranchMLP` 應視為早期 project-aligned branch-fusion baseline：它保留本專案實際使用的 metadata、synopsis/text embedding、cover/banner image embedding 與 cast/actor metadata 訊號，但三個分支平行融合，沒有明確模仿原論文的 context/character MLP + Big MLP 兩階段形狀。

更貼近 Armenta fusion shape 的 project-input proxy：

```text
C1-Armenta-ProjectInputProxy =
    project synopsis/text embedding -> synopsis branch -> 768-dim synopsis vector
  + project metadata + cover/banner image embedding -> project-context MLP -> 768-dim context vector
  -> concat -> Big MLP -> regression
```

這條線仍使用 project embeddings，不是重新抽 GPT-2 / ResNet-50，因此不能稱 encoder reproduction。不過它比 `C1-Armenta-ProxyBranchMLP` 更貼近原論文「主文字分支 + context/character MLP + Big MLP」的融合形狀，也比較不會偏離本專案的 project input contract。

visual encoder 更貼近原論文的 project-input proxy：

```text
C1-Armenta-ProjectInputProxy-ResNet50 =
    project synopsis/text embedding -> synopsis branch -> 768-dim synopsis vector
  + project cover/banner raw images -> ImageNet ResNet-50 avg-pool features
  + project metadata + ResNet-50 cover/banner features -> project-context MLP -> 768-dim context vector
  -> concat -> Big MLP -> regression
```

這條線把 image branch 從本專案既有 image embeddings 換成 ImageNet ResNet-50 features，是目前 C1 中 visual encoder 對齊最好的主線版本。但它仍不是 Figure 2 reproduction，因為影像輸入是 cover/banner 而不是 main-character portraits，文字 branch 仍不是 GPT-2 synopsis / character-description branches。

非主線旁支 reconstruction：

```text
C1-Armenta-Figure2Reconstruction =
    GPT-2 synopsis branch
  + GPT-2 main-character description/name branch
  + ResNet-50 main-character portrait branch
  -> character MLP -> Big MLP -> regression
```

`C1-Armenta-Figure2Reconstruction` 不列為目前主線 baseline。它已經比較接近原論文 Figure 2，但會使用 character-specific artifacts，與本專案主框架的 cover/banner image setup 不同型；除非研究問題明確需要 character-only/character-centric side analysis，否則不可用它取代 project-input C1 主線。

差異：

1. 使用 project metadata，但 metadata 並不是原論文三輸入 neural architecture 的核心 branch。
2. 文字 branch 使用 precomputed project text embeddings，不是 end-to-end GPT-2。
3. ResNet-50 版使用 anime cover/banner raw images，不是 main-character portraits。
4. `voice_actor_names` 是 cast metadata，能作為 character/cast proxy，但不能取代 main-character descriptions 或 portraits。
5. 沒有把 main-character descriptions 作為獨立 branch。
6. 使用本專案 temporal split 與 `popularity` / `meanScore` targets，不是原論文 split 與 MAL score target。

目前 character artifact 檢查：

1. `data/raw/anilist_anime_data_complete.csv` 有 character JSON，內含 role、character name、image URL、description。
2. 已新增 `.exp/baseline/c1_character_features/c1_character_features_{train,val,test}.parquet`，欄位為 `id + char_gpt2_000..767 + char_resnet_000..048`。
3. coverage：train 9583 rows，其中 description 4755、portrait URL 5620、成功 portrait encoding 4984；val 2918 rows，其中 1415/1921/1718；test 3087 rows，其中 1578/2193/1931。
4. 缺 character artifact 的 row 使用 zero-filled character features，因此結果可跑完整 split，但不可宣稱 character coverage 與原論文一致。

Proxy 改善點：

```text
C1-Armenta-ProxyBranchMLP 比 flat C1 更接近原論文 branch-fusion 思路，因為它先分別處理各 input group，再做 final fusion。C1-Armenta-ProjectInputProxy 則進一步把融合形狀改成 synopsis branch + project-context MLP + Big MLP。C1-Armenta-ProjectInputReconstruction 補上 GPT-2 synopsis 與 ResNet-50 cover/banner features，因此是目前 C1 project-input 主線；C1-Armenta-Figure2Reconstruction 更接近原 Figure 2，但因輸入改成 character descriptions/portraits，只能作 side analysis。性能上不能宣稱這些 C1 rows 優於 F2/C3。
```

剩餘限制：

1. metadata branch 是本專案 proxy，不是原論文 input branch。
2. `C1-Armenta-ProjectInputReconstruction` 已使用 GPT-2 synopsis artifact，但不是 end-to-end fine-tuning。
3. project-input 主線 image branch 使用 cover/banner ResNet-50 features，不是 main-character portraits。
4. `C1-Armenta-Figure2Reconstruction` 雖然補上 character description/portrait branch，但不符合主框架輸入契約，且 raw character coverage 不完整。

可允許的寫法：

```text
C1-Armenta-MLP 是受 Armenta-Segura & Sidorov 2025 啟發的寬鬆 anime-domain multimodal MLP adaptation。
C1-Armenta-ProxyBranchMLP 是受 Armenta-Segura & Sidorov 2025 啟發、較接近 branch-wise fusion 的 project-aligned proxy adaptation。
C1-Armenta-ProjectInputReconstruction 是 structure-complete project-input Armenta reconstruction；它保留 project input contract，並補上 GPT-2 synopsis、ResNet-50 cover/banner、project-context MLP 與 Big MLP。
C1-Armenta-Figure2Reconstruction 是 non-mainline Figure 2 side reconstruction；它更接近原論文 character-centric Figure 2，但不可當成本專案 cover/banner 主框架 baseline 的替代品。
```

不可寫：

```text
We reproduce the Armenta-Segura & Sidorov 2025 framework.
```

## C2：Madongo, Tang & Hassan 2023

已核對的本地 PDF：

```text
docs/refer/Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers.pdf
```

原論文關鍵元件：

1. Inputs 是 movie posters 與 movie reviews。
2. CTNN 被定義為 end-to-end cross-modal transformer-based neural network。
3. 文字側使用 fine-tuned 12-layer BERT 取得 movie review features。
4. 視覺側使用 ResNet50 與 ViT poster features。
5. 方法包含 cross-modal attention transformer fusion。
6. 另外描述 recurrent fusion，以及 movie metadata-related factors，例如 cast/crew influence、director influence、release-date influence。
7. prediction 被簡化為 box-office class/range prediction，並回報 RMSE 與 APHR。

目前專案 baseline：

```text
C2-CTNN-Lite = project text embedding + project image embedding
             -> projection into two modality tokens
             -> lightweight TransformerEncoder
             -> pooled regression head
```

依照兩條主線判準，C2 應先守住本專案 synopsis/text、cover/banner image、metadata 的輸入契約，再把模型往 CTNN 的 cross-modal attention 與 metadata fusion 靠近。原論文的 reviews、box-office classes、movie metadata factors 都不是本專案的 pre-release anime input contract；若硬做 movie-review 版本，會更像另一個 movie benchmark，而不是我們框架的公平比較。

目前 C2 主線定位：

```text
C2-CTNN-Lite =
    project synopsis/text embedding
  + project cover/banner image embedding
  -> two-token TransformerEncoder
  -> regression head
```

這條線有價值，但只是 first-pass。目前已完成的較強 project-aligned proxy 是：

```text
C2-ProjectInputCrossAttention =
    project synopsis/text embedding branch
  + project cover/banner image embedding branch
  + explicit bidirectional text-image cross-attention blocks
  + metadata-conditioned modality-gated fusion
  -> regression head
```

這樣同時滿足條件 1 的 project input contract，並在條件 2 下比 `C2-CTNN-Lite` 更接近 CTNN 的跨模態 transformer 動機。

補齊 recurrent-fusion 動機的 project-aligned proxy：

```text
C2-ProjectInputRecurrentFusion =
    project synopsis/text embedding branch
  + project cover/banner image embedding branch
  + explicit bidirectional text-image cross-attention blocks
  + metadata token
  + GRU recurrent fusion over [text, image, metadata] tokens
  -> regression head
```

這條線補上原論文提到的 recurrent fusion 動機，但仍維持本專案輸入契約。

目前最接近主要構造的 project-input reconstruction：

```text
C2-ProjectInputCTNNReconstruction =
    metadata + GPT-2 synopsis embeddings + ResNet-50 cover/banner features
  + modality transformer encoders
  + bidirectional text-image cross-modal attention
  + GRU recurrent fusion
  + metadata factor gate
  -> regression head
```

本輪新增的 dual-visual diagnostic：

```text
C2-ProjectInputCTNNDualVisualReconstruction =
    metadata + GPT-2 synopsis embeddings
  + ResNet-50 cover/banner features
  + project image embeddings as ViT-like visual semantic tokens
  + modality transformer encoders
  + bidirectional text-image cross-modal attention
  + GRU recurrent fusion
  + metadata factor gate
  -> regression head
```

這條線更接近原文 ResNet50 + ViT poster feature design，但 project image
embeddings 只是 ViT-like proxy，不是 MovieNet-finetuned ViT。run 39 顯示它
對 popularity Spearman 與 log_MAE 有幫助，但 R2 低於單視覺 reconstruction；
meanScore 則只有 Spearman 改善，R2/MAE 變差。因此它是對齊診斷 row，不是 C2 主線。

差異：

1. 使用 precomputed anime embeddings，不是原論文 poster/review deep feature extraction pipeline。
2. `C2-ProjectInputCTNNReconstruction` 已補完整 CTNN major stages；`C2-ProjectInputCTNNDualVisualReconstruction` 也補上 ResNet50 + ViT-like dual visual stream。
3. 沒有 movie metadata factors，也沒有原論文 classification/range target structure。
4. 使用 anime synopsis/cover/banner 與 regression targets，不是 movie reviews/posters 與 weekend box-office classes。
5. 本專案目前沒有 review-equivalent pre-release text source；synopsis/descriptions 不是 audience 或 critic reviews。
6. 原論文訓練提到 SGD、小 batch、binary cross-entropy；本專案維持 scaled regression + MSE/AdamW，否則會把 target formulation 改成另一個分類任務。

如果要做更接近的 C2 proxy：

1. 已完成 explicit cross-attention module，不只是在兩個 modality tokens 上做 self-attention。
2. 已完成 metadata-conditioned fusion weighting step，並補上 GRU recurrent-fusion proxy。
3. 已完成 GPT-2/ResNet-50 project-input reconstruction。
4. 已完成 dual-visual diagnostic，檢查 ResNet50 + ViT-like 視覺雙流是否有幫助。
5. target 仍維持 project regression，報告必須標成 transfer/adaptation，因為 domain 與 target 仍然不同。
6. 不建議追求 movie-review/poster exact reproduction，除非另外建立 movie-style benchmark；這已超出目前 anime project 範圍，且不適合拿來當本專案主線比較。

可允許的寫法：

```text
C2-CTNN-Lite 是受 Madongo, Tang & Hassan 2023 啟發的 lightweight cross-modal transformer fusion adaptation。
C2-ProjectInputCrossAttention 應寫成 project-input CTNN-style cross-attention proxy，而不是 CTNN reproduction。
C2-ProjectInputRecurrentFusion 應寫成 project-input CTNN-style recurrent-fusion proxy，而不是 CTNN reproduction。
C2-ProjectInputCTNNReconstruction 應寫成 structure-complete project-input CTNN reconstruction。
C2-ProjectInputCTNNDualVisualReconstruction 應寫成 dual-visual project-input diagnostic，用來檢查 ResNet50 + ViT-like 視覺來源。
```

不可寫：

```text
We reproduce CTNN.
```

## C3：Xu et al. 2025 SKAPP

已核對的本地 PDF：

```text
docs/refer/Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation.pdf
```

已核對的原始碼：

```text
baseline_refer/skapp-main/
baseline_refer/skapp-main/src/preprocess/ICIP/3_retrieval.py
baseline_refer/skapp-main/src/dataset.py
baseline_refer/skapp-main/src/RRCP/RRCP.py
baseline_refer/skapp-main/src/RRCP/predict_model.py
baseline_refer/skapp-main/src/RRCP_prediction_variable_lenth.py
baseline_refer/skapp-main/src/graph_attention.py
baseline_refer/skapp-main/src/graph_variable_length.py
```

原論文關鍵元件：

1. Meta retriever：使用 multimodal UGC semantics 加上 metadata/context。
2. 從 UGC knowledge base 做 top-k retrieval。
3. Selective refiner：使用 Relative Retrieval Contribution to Prediction (RRCP) 減少 noisy retrieved examples。
4. Vision-language GNNs：用於 query 與 selected UGC 的 contextual learning。
5. RRCP-Attention-based multimodal fusion 與 final prediction network。

原始碼核對後的實作流程：

1. `preprocess/*/3_retrieval.py` 先建立 retrieval pool，對每個 query 取大量 top-k retrieved items；ICIP/SMPD/Instagram config 皆以 `retrieval_num: 500` 作為主設定。
2. 每筆資料保留 query visual/text features、retrieved visual features、retrieved textual features、retrieved labels，而不是只保留 aggregate statistics。
3. `RRCP/train_all_item.py` 訓練 all-items prediction model；`RRCP/train_single_item.py` 訓練 single-item / dissembled model。
4. `RRCP/RRCP.py` 用 all-items model 和 dissembled model 產生 `RRCP_silver`。`RRCP_silver` 的核心是比較「without retrieval」與「with one retrieved item」對 prediction error 的改善。
5. `RRCP_prediction_variable_lenth.py` 依 `RRCP > threshold` 產生 binary mask，若全部被濾掉則保留第一個 retrieved item。
6. `graph_attention.py` / `graph_variable_length.py` 對 query + selected retrieved visual/text features 建 graph，做 GraphConvolution，再用 RRCP/CXMI 權重聚合 retrieved context 與 label embedding 做 final prediction。

目前專案 retrieval implementation：

```text
src/reference_baseline_branch/build_c3_rag_features.py
src/reference_baseline_branch/run_reference_baselines.py
src/fussion_branch/run_rag_ablation.py
src/fussion_branch/RAG/rag_query.py
```

目前支援模式：

| Mode | 專案行為 | 與 SKAPP 的關係 |
|---|---|---|
| `none` | 產生 schema-compatible no-retrieval features | 只是 non-retrieval control |
| `sparse` | 用 genre/studio/voice actor/source 做 metadata sparse retrieval，目前已跑通 reference baseline | partial meta retrieval proxy |
| `dense` | text embedding semantic retrieval，目前已跑通 reference baseline | vanilla semantic retrieval proxy |
| `hybrid` | sparse + dense RRF retrieval，目前已跑通 reference baseline | 較強 retrieval proxy，但仍不是 selective retrieval |
| `selective` | 從 sparse top-k retrieved candidates 中，用候選分數中位數作為 deterministic contribution threshold，過濾低分候選後再聚合 | RRCP-style filtering motivation 的 simple proxy；不是 RRCP reproduction |
| `skapp_proxy` | sparse+dense hybrid retrieval，使用 train-only query-candidate pairs 學 contribution scorer，對 top-k candidates 做 learned filtering，再用 softmax attention 權重聚合 retrieved context | 比 `selective` 更接近 SKAPP selective retrieval / contribution-aware context refinement；仍不是 RRCP/VL-GNN/RRCP-Attention reproduction |

依照兩條主線判準，C3 應先守住本專案 query anime + historical anime retrieval 的輸入契約，再把模型往 SKAPP 的 selective retrieval、contribution scoring、context fusion 靠近。SKAPP 原文的 UGC knowledge base、user/post context、social diffusion traces 與本專案 anime pre-release metadata 不同；若硬做 social-media UGC 版本，會換掉任務本身。

目前 C3 主線定位：

```text
C3-RAG-Selective-XGB =
    project metadata + text embedding + image embedding
  + temporally valid historical anime retrieval features
  + deterministic contribution proxy filtering
  -> XGBoost regression
```

這條線可作本專案 RAG route 的 strongest performance reference row。更接近 SKAPP selective retrieval motivation 的 project-aligned proxy 是：

```text
C3-ProjectInputSKAPPProxy-XGB =
    project query anime features
  + top-k retrieved historical anime set
  + train-only learned contribution scoring / filtering
  + attention-weighted retrieved-context aggregation
  -> XGBoost prediction head
```

這樣同時滿足條件 1 的 project retrieval input contract，並在條件 2 下比 deterministic aggregate RAG features 更接近 SKAPP 的 selective retrieval 與 context fusion 動機。不過它仍不是 SKAPP reproduction，因為 attention 是 aggregate feature weighting，不是原始碼中的 GraphLearner / RRCP_prediction neural module。

對照原始碼後，已完成的下一層版本不再只增加 aggregate RAG 欄位，而是建立 tensor artifact：

```text
C3-ProjectInputSKAPPGraphProxy =
    query text embedding + query image embedding
  + retrieved text embedding tensor [N, K, D_text]
  + retrieved image embedding tensor [N, K, D_image]
  + retrieved label tensor [N, K]
  + RRCP-like contribution tensor [N, K]
  -> RRCP threshold mask
  -> GraphLearner-style text/text and image/text graph fusion
  -> RRCP-weighted prediction head
```

這比 `C3-ProjectInputSKAPPProxy-XGB` 更接近 SKAPP source code，但仍保留本專案 temporal retrieval restriction，不能照原始碼把 validation 直接併入 retrieval pool，否則會違反本專案 pre-release evaluation contract。

第一版結果：

| baseline_id | target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|---:|
| `C3-ProjectInputSKAPPGraphProxy` | popularity | 11501.8681 | 0.4404 | 0.8561 | 1.0245 |
| `C3-ProjectInputSKAPPGraphProxy` | meanScore | 8.1448 | 0.0690 | 0.4973 |  |

### C3 gap triage

| SKAPP source component | 目前本專案狀態 | 決策 | 理由 |
|---|---|---|---|
| Meta retriever / retrieval pool | 已有 sparse/dense/hybrid retrieval；但本專案維持 train-only + temporal filtering，不採原始碼 train+valid retrieval pool。 | 繼續保留本專案版本 | 這是主輸入契約與 evaluation contract；不能為了貼近 source code 引入 validation leakage。 |
| Top-k retrieved set | `skapp_graph_proxy` 已輸出固定 top-k retrieved-set tensor columns。 | 已完成第一版，後續可強化 | 目前用 fixed top-k padding，不是 SKAPP 原始 variable-length/full retrieval_num setup。 |
| Retrieval size `retrieval_num=500` | 目前正式 C3 用 `top_k=10`。 | 暫不照搬 | 原始碼的 500 來自 UGC benchmark；本專案資料量、CPU/GPU 時間與 temporal filtering 不同。可在 GraphProxy 穩定後做 top-k sensitivity，但不是下一步核心。 |
| Retrieved visual/text features | `skapp_graph_proxy` 已保存每個 query 的 retrieved text/image embedding tensor columns。 | 已完成第一版，後續可強化 | 目前使用 project text/image embeddings，不是 SKAPP 原始 social-media visual/text encoders。 |
| Retrieved labels | `skapp_graph_proxy` 已保存 retrieved popularity/score label tensor columns。 | 已完成第一版，後續可強化 | 原始碼的 label embedding target 不同；本專案先用 popularity + meanScore 代理 retrieved labels。 |
| RRCP_silver | `skapp_graph_proxy` 已有 per-item GradientBoosting contribution proxy。 | 分階段逼近 | 尚未做 SKAPP all-items + single-item/dissembled model 的 RRCP_silver pretraining。 |
| RRCP threshold mask | `skapp_graph_proxy` 已輸出 selected mask，neural proxy 以 mask 做 graph/attention filtering。 | 已完成第一版，後續可強化 | 目前 threshold 在 feature builder 端完成，模型端使用 mask；尚未完全還原 source code threshold behavior。 |
| GraphLearner / VL-GNN | `C3-ProjectInputSKAPPGraphProxy` 已加入 learned graph adjacency over query + retrieved tokens。 | 已完成第一版，後續可強化 | 這是 GraphLearner-style proxy，不是逐行移植 SKAPP `graph_variable_length.py`。 |
| RRCP-Attention prediction network | `C3-ProjectInputSKAPPGraphProxy` 已加入 contribution-aware attention head。 | 已完成第一版，後續可強化 | 尚未還原 SKAPP RRCP/CXMI 權重與完整 prediction network。 |
| Social-media UGC context | 本專案沒有 user/post dynamics、friends、platform interactions、social diffusion traces。 | 明確棄用 | 這會換掉研究任務，不符合 pre-release anime popularity prediction。 |
| 原始碼不修改直接搬入 | 未採用。 | 明確棄用 | 原始碼綁 social UGC schema、CUDA device、train+valid retrieval pool；直接搬入會破壞本專案資料契約與可重現性。 |

### C3 structure-complete reconstruction checklist

`C3-ProjectInputSKAPPFull` 必須完成以下構造後，才可作為主要對比：

| 階段 | 必須重做的 SKAPP 構造 | 本專案替換方式 | 完成狀態 |
|---|---|---|---|
| Retrieval dataset | `merged_text_vec`, `cls_vec`, `retrieved_visual_feature_embedding_cls`, `retrieved_textual_feature_embedding`, `retrieved_label_list`, `label` | query text embedding、query image embedding、retrieved text embedding、retrieved image embedding、retrieved popularity/score label、target popularity/score | 已完成第一版：`.exp/baseline/skapp_full/dataset/{train,val,test}.npz`。 |
| All-items model | `RRCP/predict_model.py` 的 `RRCP_Model`，吃 top-N retrieved items 直接預測 label | project text/image/retrieved tensors + label embedding；train-only temporal retrieval | 已完成第一版：`run_c3_skapp_full.py` 的 `_SKAPPAllItemsModel`。 |
| Dissembled/single-item model | 用單一 retrieved item 與 query self replacement 比較有/無 retrieval 的誤差 | 對每個 retrieved anime 產生 single-item prediction；保留 query self replacement baseline | 已完成第一版：`_SKAPPSingleItemModel` 與 `_make_disassembled_data()`。 |
| RRCP_silver | `abs(Predict - without) - abs(Predict - with)` per retrieved item | 用 all-items prediction 作 pseudo target，再比較 single-item with/without prediction | 已完成第一版：run 35 已輸出 `rrcp_silver_{target}_{split}.npz`。 |
| Threshold variable-length filtering | `RRCP > threshold`，全被濾掉則保留第一個 retrieved item | 嚴格重做 source behavior，不只使用 median mask | 已完成第一版：`_SKAPPFinalRRCPModel` 中以 RRCP threshold 產生 selected mask。 |
| GraphLearner | `graph_variable_length.py` 的 cosine edge + graph norm + TT/IT GCN fusion | 用 project text/image hidden dim adapter 後重做 GraphConvolution/GraphLearner | 已完成第一版：source-shaped cosine graph + normalized graph convolution。 |
| RRCP/CXMI attention head | `graph_attention.py` 中用 RRCP/CXMI normalize 後加權 packed features，再 concat label embedding | 用 RRCP_silver 權重加權 text/image graph outputs，concat retrieved label embedding 後回歸 | 已完成第一版：RRCP/CXMI-style weighting 已在 final model 中執行。 |
| Final training/evaluation | `RRCP_prediction_variable_lenth.py` final model | popularity 與 meanScore 各自訓練/評估，保留 temporal split | 已完成第一版：run 35 已產生兩個 target 的 validation/test results。 |

Run 35 第一版結果：

| target | test_MAE | test_R2 | test_Spearman_rho | test_log_MAE |
|---|---:|---:|---:|---:|
| `popularity` | 14668.1228 | -0.4927 | 0.6985 | 1.2983 |
| `meanScore` | 9.8063 | -0.2385 | 0.3657 |  |

目前結論：完整構造已可執行，但尚未調好。下一步應 debug all-items /
single-item training、target/label scaling、RRCP_silver distribution、以及
GraphLearner normalization，而不是再回到 aggregate XGBoost proxy。

### C3 baseline row decisions

| Baseline row | 決策 | 用途 | 不再追的原因或限制 |
|---|---|---|---|
| `C3-RAG-None-XGB` | 保留 | no-RAG control，衡量 retrieval 是否有增益 | 不作為 SKAPP proxy。 |
| `C3-RAG-Sparse-XGB` | 保留 | metadata retrieval 的強 simple row | 已被 selective row 小幅超過；不再作主線。 |
| `C3-RAG-Dense-XGB` | 保留但不延伸 | semantic retrieval comparison | 對 popularity 幾乎接近 no-RAG，現階段不值得優先 tune。 |
| `C3-RAG-Hybrid-XGB` | 保留但不延伸 | sparse+dense RRF comparison | performance 低於 sparse/selective；不再把「混合檢索必然更好」當假設。 |
| `C3-RAG-Selective-XGB` | 保留為 strongest performance row | C3 performance 主線；報告最佳 RAG reference performance 時使用 | 只是一個 deterministic contribution proxy，不可說成 SKAPP reproduction。 |
| `C3-ProjectInputSKAPPProxy-XGB` | 保留為 closest aggregate SKAPP-style proxy | C3 方法對齊主線；展示 learned contribution + attention-weighted context | performance 不如 `C3-RAG-Selective-XGB`，且仍是 aggregate feature proxy。 |
| `C3-ProjectInputSKAPPGraphProxy` | 保留並後續強化 | closest architecture proxy；source-code-aligned retrieved-set tensor + RRCP-mask graph/attention neural proxy | 第一版 performance 低於 selective row；仍不可宣稱完整 SKAPP reproduction。 |

差異：

1. `C3-ProjectInputSKAPPGraphProxy` 已把 selected retrieved set tensor 送進 neural contextual module，但採 fixed top-k padding，不是原始碼 variable-length retrieval pipeline。
2. 已建立 retrieved text/image/label/contribution proxy fields，但 `RRCP_silver` 仍是 GradientBoosting contribution proxy，不是 all-items + single-item/dissembled model 預訓練結果。
3. 已有 GraphLearner-style learned adjacency，但不是逐行移植 SKAPP `graph_variable_length.py`。
4. 已有 contribution-aware attention head，但尚未還原 SKAPP RRCP/CXMI 權重與完整 RRCP-Attention prediction network。
5. anime release metadata 缺少 SKAPP 用到的 social-media UGC contexts，例如 user/post dynamics、friends、platform interactions、social diffusion traces。

目前或近期可允許的寫法：

```text
C3-RAG-Minimal / C3-RAG-Selective 是用於 anime popularity prediction 的 SKAPP-inspired retrieval baseline。
C3-RAG-Selective 可寫成 simple RRCP-style contribution-filtering proxy，但不可稱 RRCP reproduction。
C3-ProjectInputSKAPPProxy-XGB 可寫成 project-input SKAPP-style learned contribution and attention-weighted context proxy，而不是 SKAPP reproduction。
C3-ProjectInputSKAPPGraphProxy 可寫成 project-input SKAPP architecture proxy，因為它包含 retrieved-set tensors、RRCP-style mask、graph adjacency、contribution-aware attention head。
```

若要比目前 `C3-ProjectInputSKAPPGraphProxy` 更強的 SKAPP-style claim，至少還需要：

1. 改用更接近 SKAPP 的 RRCP_silver 產生流程，例如 all-items model + single-item/dissembled contribution estimation。
2. 讓 retrieved candidates 使用更接近原始碼的 threshold behavior 與 variable-length handling。
3. 更完整還原 `graph_variable_length.py` 與 RRCP/CXMI attention prediction head。
4. 報告仍需說明這是從 social-media UGC popularity 到 anime release popularity 的 domain transfer。

沒有上述模組時不可寫：

```text
We reproduce SKAPP.
```

## 報告規則

目前 C1/C2 主線 rows 應寫成 project-input reconstruction/adaptation/proxy；`C1-Armenta-Figure2Reconstruction` 是 character-centric Figure 2 side reconstruction，不是 project-input mainline；`C2-ProjectInputCTNNDualVisualReconstruction` 是 ResNet50+ViT-like source-alignment diagnostic，不是 C2 主線替代品。C3 目前有三個定位：`C3-RAG-Selective-XGB` 是 strongest performance row；`C3-ProjectInputSKAPPProxy-XGB` 是 aggregate SKAPP-style proxy；`C3-ProjectInputSKAPPGraphProxy` 是 closest architecture proxy；`C3-ProjectInputSKAPPFull` 是 first structure-complete reconstruction run，但 performance 仍需 debug。C3 rows 仍不可寫成 SKAPP exact reproduction；`F2-XGB-Concat` 仍可作為 no-RAG multimodal classical floor，不是 reproduced neural framework。
