# Reference Baseline 論文對齊稽核

更新日期：2026-05-14

本文件記錄目前已實作或規劃中的 reference baseline，是否真的對齊錨定論文中的架構。核心目的不是幫模型背書，而是把「可主張的程度」寫清楚，避免把 adaptation 或 proxy 誤寫成 reproduction。

## 共同判準

C1、C2、C3 一律只用兩個條件判斷是否能當主線 reference baseline：

1. 輸入必須對齊本專案主框架：baseline 要使用本專案設計的主輸入契約，也就是 metadata、synopsis/text embedding、cover/banner image，以及必要時的 project retrieval context。若改用本專案主框架沒有的輸入，例如 C1 character portraits、C2 movie reviews、C3 social-media UGC graph/context，就不能作為主線 comparison row。
2. 模型方法要在上述輸入限制下盡量還原原論文：輸入固定為本專案主框架後，fusion shape、attention/retrieval/refinement module、MLP depth、encoder choice 等才往原論文靠近。可主張的程度取決於還原到哪一層；目前多數只能寫成 project-input proxy/adaptation，不能寫成原論文 reproduction。

## 總覽

| Baseline | 對應論文路線 | 目前對齊程度 | 報告寫法 |
|---|---|---|---|
| `C1-Armenta-MLP` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | 寬鬆 adaptation | 只能作為 anime-domain multimodal MLP 的 first-pass adaptation，不能稱框架重現。 |
| `C1-Armenta-ProxyBranchMLP` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | project-aligned proxy adaptation | 代表本專案早期 C1 branch-fusion route；三個 project feature groups 平行分支後融合，但還不是 Armenta 的 character/context MLP + Big MLP 形狀。 |
| `C1-Armenta-ProjectInputProxy` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | stronger project-input proxy adaptation | 目前較適合代表 C1 主線：保留本專案 metadata / synopsis-text / cover-banner image artifacts，同時改成 synopsis branch + project-context MLP + Big MLP。仍不可稱 Figure 2 或 GPT-2/ResNet-50 復現。 |
| `C1-Armenta-ProjectInputProxy-ResNet50` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | stronger project-input proxy adaptation | 目前 C1 中 visual encoder 最接近原論文的一版：保留本專案 metadata / synopsis-text / cover-banner inputs，並用 ImageNet ResNet-50 抽 cover/banner features；仍不可稱 Figure 2、GPT-2 或 character-branch 復現。 |
| `C1-Armenta-Figure2Proxy` | Armenta-Segura & Sidorov 2025 Figure 2 | 非主線旁支分析 | 因為輸入會轉向 main-character descriptions / portraits，不符合本專案 cover/banner 主輸入契約；除非另有分析需求，否則不作為主線 baseline。 |
| `C2-CTNN-Lite` | Madongo, Tang & Hassan 2023 CTNN | first-pass project-input adaptation | 只能稱 lightweight cross-modal transformer fusion adaptation；它對齊本專案 text/image inputs，但還沒有 explicit cross-attention、metadata fusion、recurrent-style fusion，不能稱 CTNN reproduction。 |
| `C2-ProjectInputCrossAttention` | Madongo, Tang & Hassan 2023 CTNN | stronger project-input proxy adaptation | 目前較適合代表 C2 主線：先滿足本專案 synopsis/cover-banner/metadata inputs，再用 explicit bidirectional text-image cross-attention 與 metadata-conditioned fusion 往 CTNN 靠近；改用 movie reviews 的版本不列主線，因為違反條件 1。 |
| `C2-ProjectInputRecurrentFusion` | Madongo, Tang & Hassan 2023 CTNN | stronger project-input proxy adaptation | 在 CrossAttention 之上補 GRU recurrent fusion，覆蓋原文 recurrent-fusion 動機；結果上 meanScore 略優、popularity Spearman 略優，但 popularity R2 低於 CrossAttention。 |
| `C3-RAG-*` | Xu et al. 2025 SKAPP | project-aligned retrieval baseline | 目前完成 `none/sparse/dense/hybrid/selective` retrieval baselines；`selective` 可作本專案 RAG 主線，但只能稱 SKAPP-inspired/project retrieval proxy，不是 RRCP/VL-GNN/RRCP-Attention reproduction。 |
| `C3-ProjectInputSKAPPProxy` | Xu et al. 2025 SKAPP | planned stronger project-input proxy | 若要繼續 C3，應保留 project historical-anime retrieval，再把模型往 learned contribution scoring + retrieved-set graph/attention fusion 靠近；改用 social-media UGC 的版本不列主線，因為違反條件 1。 |

## 復現可行性矩陣

| 路線 | 若要 exact reproduction 需要什麼 | 目前專案已有什麼 | 主要限制 | 實務上下一層可做什麼 |
|---|---|---|---|---|
| `C1` Armenta anime deep model | MAL-style dataset、synopsis、main-character descriptions、main-character portraits、GPT-2 text branches、ResNet-50 portrait branch、character MLP、Big MLP，以及原論文 split/target 設定。 | AniList processed metadata、project text embeddings、cover/banner raw images、cover/banner ResNet-50 features、`voice_actor_names` cast metadata、raw AniList character JSON 覆蓋多數 split IDs、flat MLP、branch-wise proxy MLP、project-input Armenta-shaped proxy MLP。 | 目前主線 artifacts 仍沒有 character-description 或 character-portrait branches；target/split 與原論文不同。若硬做 Figure 2，會更貼近論文，但會偏離本專案主框架的 cover/banner image setup。 | 主線使用 `C1-Armenta-ProjectInputProxy-ResNet50` 作為對齊版：先守住本專案輸入契約，再盡量還原 Armenta 的 synopsis branch + context/character MLP + Big MLP 形狀，並用 ResNet-50 補上 visual encoder 對齊。`C1-Armenta-Figure2Proxy` 不列主線。 |
| `C2` CTNN box-office model | movie poster + movie review dataset、poster/review transformer feature extraction、cross-modal attention transformer、recurrent fusion、metadata factors，以及 box-office class/range target。 | anime metadata、anime text embeddings、anime cover/banner image embeddings、two-token TransformerEncoder fusion、project-input cross-attention fusion、project-input recurrent fusion、regression targets。 | domain、inputs、feature extractors、fusion architecture、target formulation 都不同；目前 pre-release anime dataset 沒有 movie review 等價訊號。若硬改成 movie-review CTNN，會偏離本專案輸入契約。 | `C2-CTNN-Lite` 保留為 first-pass；`C2-ProjectInputCrossAttention` 與 `C2-ProjectInputRecurrentFusion` 作為目前較強 C2 主線 proxies。 |
| `C3` SKAPP retrieval model | UGC knowledge base、multimodal/meta retriever、top-k retrieval、RRCP selective refiner、VL-GNN contextual learning、RRCP-Attention prediction network，以及 social-media popularity targets。 | Offline train-set knowledge base、`none/sparse/dense/hybrid/selective` RAG feature artifacts、metadata sparse retrieval、text embedding dense retrieval、hybrid RRF retrieval、top-k aggregate RAG features、median-threshold contribution proxy、XGBoost fusion。 | 目前沒有 learned RRCP contribution scoring、沒有 selected retrieved-set graph、沒有 VL-GNN、沒有 RRCP-Attention。anime release metadata 也缺少 SKAPP 使用的 user/social context。若硬改成 social-media UGC SKAPP，會換成另一個任務。 | `C3-RAG-Selective-XGB` 先作 project-aligned RAG 主線；若要更接近 SKAPP，做 `C3-ProjectInputSKAPPProxy`：在 project historical-anime retrieval 上加入 learned contribution scoring + retrieved-set graph/attention fusion。 |

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

非主線旁支分析：

```text
C1-Armenta-Figure2Proxy =
    synopsis text embedding branch
  + main-character description embedding branch
  + main-character portrait embedding branch
  -> character MLP -> Big MLP -> regression
```

`C1-Armenta-Figure2Proxy` 不列為目前主線 baseline。它比較接近原論文 Figure 2，但會使用 character-specific artifacts，與本專案主框架的 cover/banner image setup 不同型；除非研究問題明確需要 character-only/character-centric side analysis，否則不應投入為優先工作。

差異：

1. 使用 project metadata，但 metadata 並不是原論文三輸入 neural architecture 的核心 branch。
2. 文字 branch 使用 precomputed project text embeddings，不是 end-to-end GPT-2。
3. ResNet-50 版使用 anime cover/banner raw images，不是 main-character portraits。
4. `voice_actor_names` 是 cast metadata，能作為 character/cast proxy，但不能取代 main-character descriptions 或 portraits。
5. 沒有把 main-character descriptions 作為獨立 branch。
6. 使用本專案 temporal split 與 `popularity` / `meanScore` targets，不是原論文 split 與 MAL score target。

目前資料檢查：

1. `data/raw/anilist_anime_data_complete.csv` 有 character JSON，內含 role、character name、image URL、description。
2. 目前 processed/fusion artifacts 在 embedding 生成前已經丟掉 character-level inputs。
3. raw coverage 對目前 split 不是完整覆蓋：validation/test 有覆蓋，但 train 缺部分 raw IDs。
4. 若要更貼近 Figure 2，需要先建立 main-character descriptions 與 portrait URLs 的 artifact stage，再談 embedding/model；但這不是目前 C1 對齊的必要條件。

Proxy 改善點：

```text
C1-Armenta-ProxyBranchMLP 比 flat C1 更接近原論文 branch-fusion 思路，因為它先分別處理各 input group，再做 final fusion。C1-Armenta-ProjectInputProxy 則進一步把融合形狀改成 synopsis branch + project-context MLP + Big MLP。C1-Armenta-ProjectInputProxy-ResNet50 再把 visual encoder 往原論文 ResNet-50 靠近，因此是目前 C1 對齊版主線；性能上則與 ProjectInputProxy 接近，不能因此宣稱優於 F2/C3。
```

剩餘限制：

1. metadata branch 是本專案 proxy，不是原論文 input branch。
2. text branch 使用 project embeddings，不是 end-to-end GPT-2 synopsis branch。
3. image branch 使用 cover/banner ResNet-50 features，不是 main-character portraits。
4. 仍然沒有 separate main-character description branch。

可允許的寫法：

```text
C1-Armenta-MLP 是受 Armenta-Segura & Sidorov 2025 啟發的寬鬆 anime-domain multimodal MLP adaptation。
C1-Armenta-ProxyBranchMLP 是受 Armenta-Segura & Sidorov 2025 啟發、較接近 branch-wise fusion 的 project-aligned proxy adaptation。
C1-Armenta-ProjectInputProxy-ResNet50 是 project-input Armenta-style proxy with ResNet-50 visual features；可說 visual encoder 比既有 project-image-embedding 版本更接近原論文。
C1-Armenta-Figure2Proxy 若未來實作，應寫成非主線 character-centric side analysis，而不是本專案主框架 baseline 的替代品。
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
3. 使用 BERT/ViT 類 transformer encoder 來處理 textual reviews 與 visual poster features。
4. 方法包含 cross-modal attention transformer fusion。
5. 另外描述 recurrent fusion，以及 movie metadata-related factors，例如 cast/crew influence、director influence、release-date influence。
6. prediction 被簡化為 box-office class/range prediction，並回報 RMSE 與 APHR。

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

差異：

1. 使用 precomputed anime embeddings，不是原論文 poster/review deep feature extraction pipeline。
2. `C2-CTNN-Lite` 使用 two-token TransformerEncoder，不是完整 CTNN 的 cross-modal attention + recurrent fusion；`C2-ProjectInputCrossAttention` 已補 explicit cross-attention；`C2-ProjectInputRecurrentFusion` 已補 recurrent token fusion proxy。
3. 沒有 movie metadata factors，也沒有原論文 classification/range target structure。
4. 使用 anime description/image embeddings 與 regression targets，不是 movie reviews/posters 與 weekend box-office classes。
5. 本專案目前沒有 review-equivalent pre-release text source；synopsis/descriptions 不是 audience 或 critic reviews。

如果要做更接近的 C2 proxy：

1. 已完成 explicit cross-attention module，不只是在兩個 modality tokens 上做 self-attention。
2. 已完成 metadata-conditioned fusion weighting step，並補上 GRU recurrent-fusion proxy。
3. target 仍維持 project regression，報告必須標成 transfer/adaptation，因為 domain 與 target 仍然不同。
4. 不建議追求 movie-review/poster exact reproduction，除非另外建立 movie-style benchmark；這已超出目前 anime project 範圍，且不適合拿來當本專案主線比較。

可允許的寫法：

```text
C2-CTNN-Lite 是受 Madongo, Tang & Hassan 2023 啟發的 lightweight cross-modal transformer fusion adaptation。
C2-ProjectInputCrossAttention 應寫成 project-input CTNN-style cross-attention proxy，而不是 CTNN reproduction。
C2-ProjectInputRecurrentFusion 應寫成 project-input CTNN-style recurrent-fusion proxy，而不是 CTNN reproduction。
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

原論文關鍵元件：

1. Meta retriever：使用 multimodal UGC semantics 加上 metadata/context。
2. 從 UGC knowledge base 做 top-k retrieval。
3. Selective refiner：使用 Relative Retrieval Contribution to Prediction (RRCP) 減少 noisy retrieved examples。
4. Vision-language GNNs：用於 query 與 selected UGC 的 contextual learning。
5. RRCP-Attention-based multimodal fusion 與 final prediction network。

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

依照兩條主線判準，C3 應先守住本專案 query anime + historical anime retrieval 的輸入契約，再把模型往 SKAPP 的 selective retrieval、contribution scoring、context fusion 靠近。SKAPP 原文的 UGC knowledge base、user/post context、social diffusion traces 與本專案 anime pre-release metadata 不同；若硬做 social-media UGC 版本，會換掉任務本身。

目前 C3 主線定位：

```text
C3-RAG-Selective-XGB =
    project metadata + text embedding + image embedding
  + temporally valid historical anime retrieval features
  + deterministic contribution proxy filtering
  -> XGBoost regression
```

這條線可作本專案 RAG route 的 strongest reference row。若要繼續 C3，下一個有意義的 project-aligned proxy 應是：

```text
C3-ProjectInputSKAPPProxy =
    project query anime features
  + top-k retrieved historical anime set
  + learned contribution scoring / filtering
  + retrieved-set graph or attention fusion
  -> prediction head
```

這樣同時滿足條件 1 的 project retrieval input contract，並在條件 2 下比目前 aggregate RAG features 更接近 SKAPP 的 selective retrieval 與 context fusion 動機。

差異：

1. 目前 retrieval 只保留 aggregate RAG features，不會把 selected retrieved set 送進 neural contextual module。
2. 目前 `selective` 只有 deterministic score-threshold proxy，沒有 learned RRCP score 估計 retrieved item 對 query prediction 是否有幫助。
3. 沒有真正 selective refiner 依 learned contribution 過濾 noisy retrieved examples。
4. 沒有 VL-GNN，也沒有 query/retrieved multimodal nodes 的 graph construction。
5. 沒有 RRCP-Attention prediction network。
6. anime release metadata 缺少 SKAPP 用到的 social-media UGC contexts，例如 user/post dynamics、friends、platform interactions、social diffusion traces。

目前或近期可允許的寫法：

```text
C3-RAG-Minimal / C3-RAG-Selective 是用於 anime popularity prediction 的 SKAPP-inspired retrieval baseline。
C3-RAG-Selective 可寫成 simple RRCP-style contribution-filtering proxy，但不可稱 RRCP reproduction。
C3-ProjectInputSKAPPProxy 若未來實作，應寫成 project-input SKAPP-style retrieval/context-fusion proxy，而不是 SKAPP reproduction。
```

若要更強 SKAPP-style claim，至少需要：

1. Retrieval 必須成為 prediction input，不只是 precomputed top-1 summary fields。
2. Retrieved candidates 必須用 contribution-like criterion 評分、過濾或加權。
3. Retrieved items 的 visual/textual context 必須透過 graph 或 attention fusion module 聚合。
4. 報告仍需說明這是從 social-media UGC popularity 到 anime release popularity 的 domain transfer。

沒有上述模組時不可寫：

```text
We reproduce SKAPP.
```

## 報告規則

目前所有 C1/C2 rows 都應寫成 project-input adaptations/proxies。現有 C3/RAG work 必須寫成 SKAPP-inspired/project-input retrieval proxy，除非實作 learned RRCP-style selection 與 graph/attention fusion。這些 baseline 是有價值的 reference coordinates；目前最強的 RAG reference row 是 `C3-RAG-Selective-XGB`，而 `F2-XGB-Concat` 仍可作為 no-RAG multimodal classical floor，不是 reproduced neural framework。
