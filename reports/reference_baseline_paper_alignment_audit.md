# Reference Baseline 論文對齊稽核

更新日期：2026-05-12

本文件記錄目前已實作或規劃中的 reference baseline，是否真的對齊錨定論文中的架構。核心目的不是幫模型背書，而是把「可主張的程度」寫清楚，避免把 adaptation 或 proxy 誤寫成 reproduction。

## 總覽

| Baseline | 對應論文路線 | 目前對齊程度 | 報告寫法 |
|---|---|---|---|
| `C1-Armenta-MLP` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | 寬鬆 adaptation | 只能作為 anime-domain multimodal MLP 的 first-pass adaptation，不能稱框架重現。 |
| `C1-Armenta-ProxyBranchMLP` | Armenta-Segura & Sidorov 2025 anime multimodal deep model | 較強 proxy adaptation | 可優先用來代表 Armenta 路線，但必須清楚標註缺少 main-character description / portrait artifacts。 |
| `C2-CTNN-Lite` | Madongo, Tang & Hassan 2023 CTNN | partial adaptation | 只能稱 lightweight cross-modal transformer fusion adaptation，不能稱 CTNN reproduction。 |
| future `C3-*` | Xu et al. 2025 SKAPP | reference runner 尚未實作 | 目前 `none/sparse/dense/hybrid` retrieval 只能稱 SKAPP-inspired。若要更強 claim，至少需要 RRCP-style selection 與 graph/attention fusion。 |

## 復現可行性矩陣

| 路線 | 若要 exact reproduction 需要什麼 | 目前專案已有什麼 | 主要限制 | 實務上下一層可做什麼 |
|---|---|---|---|---|
| `C1` Armenta anime deep model | MAL-style dataset、synopsis、main-character descriptions、main-character portraits、GPT-2 text branches、ResNet-50 portrait branch、character MLP、Big MLP，以及原論文 split/target 設定。 | AniList processed metadata、project text embeddings、cover/banner image embeddings、raw AniList character JSON 覆蓋多數 split IDs、flat MLP 與 branch-wise proxy MLP。 | 目前 embedding artifacts 沒有 character-description 或 character-portrait branches；raw train coverage 不完整；target/split 與原論文不同。 | 建立 `C1-Armenta-Figure2Proxy`：從 raw character inputs 產生新 artifacts，之後用 strict subset 或補抓缺失 raw IDs 後重跑。 |
| `C2` CTNN box-office model | movie poster + movie review dataset、poster/review transformer feature extraction、cross-modal attention transformer、recurrent fusion、metadata factors，以及 box-office class/range target。 | anime text embeddings、anime cover/banner image embeddings、two-token TransformerEncoder fusion、regression targets。 | domain、inputs、feature extractors、fusion architecture、target formulation 都不同；目前 pre-release anime dataset 沒有 movie review 等價訊號。 | 保留 `C2-CTNN-Lite`；若需要更強 proxy，可加 cross-attention blocks 與 metadata branch，但仍只能稱 CTNN-inspired/adapted。 |
| `C3` SKAPP retrieval model | UGC knowledge base、multimodal/meta retriever、top-k retrieval、RRCP selective refiner、VL-GNN contextual learning、RRCP-Attention prediction network，以及 social-media popularity targets。 | Qdrant-based `none/sparse/dense/hybrid` retrieval features、metadata sparse encoder、text embedding dense retrieval、RRF hybrid retrieval、FusionMLP input features。 | 目前 retrieval 只輸出 top-1 summary features；沒有 RRCP contribution scoring、沒有 selected retrieved-set graph、沒有 VL-GNN、沒有 RRCP-Attention。anime release metadata 也缺少 SKAPP 使用的 user/social context。 | 先做 `C3-SKAPP-Inspired`：top-k aggregate retrieval + contribution filter + fixed fusion backbone。只有完成 RRCP + graph/attention 後才保留 SKAPP reproduction 的可能性。 |

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

目前專案 baseline：

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

差異：

1. 使用 project metadata，但 metadata 並不是原論文三輸入 neural architecture 的核心 branch。
2. 使用 precomputed project text/image embeddings，不是 end-to-end GPT-2 與 ResNet-50 branches。
3. 使用 anime cover/banner image embeddings，不是 main-character portraits。
4. 沒有把 main-character descriptions 作為獨立 branch。
5. 使用本專案 temporal split 與 `popularity` / `meanScore` targets，不是原論文 split 與 MAL score target。

目前資料檢查：

1. `data/raw/anilist_anime_data_complete.csv` 有 character JSON，內含 role、character name、image URL、description。
2. 目前 processed/fusion artifacts 在 embedding 生成前已經丟掉 character-level inputs。
3. raw coverage 對目前 split 不是完整覆蓋：validation/test 有覆蓋，但 train 缺部分 raw IDs。
4. 若要更貼近 C1，需要先建立 main-character descriptions 與 portrait URLs 的 artifact stage，再談 embedding/model。

Proxy 改善點：

```text
C1-Armenta-ProxyBranchMLP 比 flat C1 更接近原論文 branch-fusion 思路，因為它先分別處理各 input group，再做 final fusion。
```

剩餘限制：

1. metadata branch 是本專案 proxy，不是原論文 input branch。
2. text branch 使用 project embeddings，不是 end-to-end GPT-2 synopsis branch。
3. image branch 使用 cover/banner embeddings，不是 main-character portraits。
4. 仍然沒有 separate main-character description branch。

可允許的寫法：

```text
C1-Armenta-MLP 是受 Armenta-Segura & Sidorov 2025 啟發的寬鬆 anime-domain multimodal MLP adaptation。
C1-Armenta-ProxyBranchMLP 是受 Armenta-Segura & Sidorov 2025 啟發、較接近 branch-wise fusion 的 proxy adaptation。
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

差異：

1. 使用 precomputed anime embeddings，不是原論文 poster/review deep feature extraction pipeline。
2. 使用 two-token TransformerEncoder，不是完整 CTNN 的 cross-modal attention + recurrent fusion。
3. 沒有 movie metadata factors，也沒有原論文 classification/range target structure。
4. 使用 anime description/image embeddings 與 regression targets，不是 movie reviews/posters 與 weekend box-office classes。
5. 本專案目前沒有 review-equivalent pre-release text source；synopsis/descriptions 不是 audience 或 critic reviews。

如果要做更接近的 C2 proxy：

1. 加 explicit cross-attention module，不只是在兩個 modality tokens 上做 self-attention。
2. 加 metadata branch 與 fusion weighting step，對齊 CTNN/recurrent-fusion 的動機。
3. target 仍可維持 project regression，但報告必須標成 transfer/adaptation，因為 domain 與 target 仍然不同。
4. 不建議追求 exact paper reproduction，除非另外建立 movie-style review/poster benchmark；這已超出目前 anime project 範圍。

可允許的寫法：

```text
C2-CTNN-Lite 是受 Madongo, Tang & Hassan 2023 啟發的 lightweight cross-modal transformer fusion adaptation。
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
src/fussion_branch/run_rag_ablation.py
src/fussion_branch/RAG/rag_query.py
```

目前支援模式：

| Mode | 專案行為 | 與 SKAPP 的關係 |
|---|---|---|
| `none` | 產生 schema-compatible no-retrieval features | 只是 non-retrieval control |
| `sparse` | 用 genre/studio/voice actor/source 做 metadata sparse retrieval | partial meta retrieval proxy |
| `dense` | text embedding semantic retrieval | vanilla semantic retrieval proxy |
| `hybrid` | sparse + dense RRF retrieval | 較強 retrieval proxy，但仍不是 selective retrieval |

差異：

1. 目前 retrieval 只保留 summary RAG features，主要來自 top retrieved item，不會把 selected retrieved set 送進 neural contextual module。
2. 沒有 RRCP score 估計 retrieved item 對 query prediction 是否有幫助。
3. 沒有 selective refiner 依 contribution 過濾 noisy retrieved examples。
4. 沒有 VL-GNN，也沒有 query/retrieved multimodal nodes 的 graph construction。
5. 沒有 RRCP-Attention prediction network。
6. anime release metadata 缺少 SKAPP 用到的 social-media UGC contexts，例如 user/post dynamics、friends、platform interactions、social diffusion traces。

目前或近期可允許的寫法：

```text
C3-RAG-Minimal / C3-RAG-Selective 是用於 anime popularity prediction 的 SKAPP-inspired retrieval baseline。
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

目前所有 C1/C2 rows 都應寫成 adaptations。現有 C3/RAG work 必須寫成 SKAPP-inspired，除非實作 RRCP-style selection 與 graph/attention fusion。這些 baseline 是有價值的 reference coordinates，但目前完成度最高、最穩定的 empirical floor 仍是 `F2-XGB-Concat`，不是 reproduced neural framework。
