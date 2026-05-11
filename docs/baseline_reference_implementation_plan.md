# Baseline Reference and Implementation Plan

本文件是給後續實作 AI / 組員使用的 baseline 交接文件。請先讀完「概念釐清」再實作，避免把 baseline、ablation、RAG 重現混在一起。

## 0. 核心定位

本專案主題是：

> Pre-release Anime Popularity Prediction Based on Multimodal Features

也就是在動畫播出前，使用可於播出前取得的 metadata、文字、圖片與可能的 retrieval features，預測：

- `popularity`
- `meanScore`

本文件負責的是「建立可引用文獻支撐的 baseline 座標系」，不是替其他組員完成 text/image/RAG/fusion ablation。

## 1. 必須先釐清的概念

### 1.1 Baseline 不是 Ablation

Baseline 是外部參考點，用來回答：

> 我們的方法至少要和哪些既有方法或合理強基準比較，才有說服力？

Ablation 是內部消融，用來回答：

> 我們自己的模型裡，metadata、text、image、RAG 各模態或模組各自貢獻多少？

因此：

- `metadata only`
- `metadata + text`
- `metadata + image`
- `metadata + text + image`
- `RAG on/off`

這些主要是本專案內部 ablation / RQ 實驗，不是本文獻 baseline 工作的全部。

### 1.2 Baseline 可以分成兩層

Baseline 不只是一種。這裡分為：

| 層級 | 名稱 | 目的 |
|---|---|---|
| Foundation baseline | 基礎地板 | 確認簡單或傳統方法能做到哪裡 |
| Competitive baseline | 強競爭基準 | 和同題型或近似 SOTA 方法比較 |

不能只做 SOTA baseline，否則無法回答「是否只是 metadata 已經夠強」。
也不能只做簡單 baseline，否則無法說服評審我們和現有強方法相比如何。

### 1.3 不要誤解 SKAPP

`Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation` 不是普通的 RAG baseline。它的核心是：

- meta retriever
- selective refiner
- RRCP
- vision-language GNN
- RRCP-Attention

若只是做 `none / sparse / dense / hybrid`，那只能稱為 SKAPP-inspired retrieval baseline，不是重現 SKAPP。

## 2. Baseline 總覽與重要性排序

### 2.0 Baseline Branch Map

```text
Baseline 實驗總圖
目的：建立「我們的方法要比誰好」的外部比較座標
不是 RQ ablation，不是證明我們內部哪個模態有用

Baseline System
├── 0. Lowest Reference / 最低地板
│   ├── 測什麼
│   │   ├── Mean predictor
│   │   └── Linear / Ridge regression
│   ├── 用什麼文獻
│   │   └── 不需要特定文獻，屬於通用 baseline
│   └── 證明什麼
│       └── 我們的任務至少不是連平均值或線性模型都贏不了
│
├── 1. Foundation Baseline / 傳統 ML 地板
│   ├── 1.1 Metadata-only Classical ML
│   │   ├── 測什麼
│   │   │   └── metadata → RF / XGBoost / LightGBM → popularity, meanScore
│   │   ├── 用什麼文獻
│   │   │   └── Lo & Syu 2023
│   │   │       "Analyzing drama metadata through machine learning..."
│   │   └── 證明什麼
│   │       └── pre-release metadata 本身能提供多少預測力
│   │
│   ├── 1.2 Feature-concat Classical ML
│   │   ├── 測什麼
│   │   │   └── metadata + text_emb + image_emb → XGBoost / LightGBM
│   │   ├── 用什麼文獻
│   │   │   ├── Chen et al. 2019
│   │   │   │   "Social Media Popularity Prediction Based on Visual-Textual Features with XGBoost"
│   │   │   └── Jeong et al. 2024
│   │   │       "Enhancing Social Media Post Popularity Prediction with Visual Content"
│   │   └── 證明什麼
│   │       └── 不用 deep fusion，只把多模態特徵 concat 給強傳統模型能做到哪裡
│   │
│   ├── 1.3 Text-only Baseline
│   │   ├── 測什麼
│   │   │   ├── description / synopsis → TF-IDF + classical regressor
│   │   │   └── description embedding → Ridge / XGBoost / MLP
│   │   ├── 用什麼文獻
│   │   │   └── "Anime Success Prediction Based on Synopsis Using Traditional Classifiers"
│   │   └── 證明什麼
│   │       └── 播出前文字 synopsis / description 單獨有多少預測力
│   │
│   └── 1.4 Image-only Baseline
│       ├── 測什麼
│       │   ├── cover image embedding → regressor
│       │   └── cover + banner embedding → regressor
│       ├── 用什麼文獻
│       │   ├── Zhou, Zhang & Yi 2019
│       │   │   "Predicting movie box-office revenues using deep neural networks"
│       │   └── Rengkung & Mandala 2025
│       │       "Investigating the Impact of Movie Poster Clustering on Box Office Prediction"
│       │       注意：方法細節需全文確認
│       └── 證明什麼
│           └── poster / cover visual signal 單獨能否預測娛樂作品表現
│
└── 2. Competitive Baseline / 強競爭基準
    ├── 2.1 Anime Domain Deep Fusion
    │   ├── 測什麼
    │   │   └── metadata + text_emb + image_emb → DNN / MLP fusion
    │   ├── 用什麼文獻
    │   │   └── Armenta-Segura & Sidorov 2025
    │   │       "Anime popularity prediction before huge investments"
    │   └── 證明什麼
    │       └── 我們的方法是否至少能對齊同領域 anime multimodal deep baseline
    │
    ├── 2.2 Cross-modal Transformer Fusion
    │   ├── 測什麼
    │   │   └── text_emb + image_emb → cross-attention / transformer fusion → regression
    │   ├── 用什麼文獻
    │   │   └── Madongo, Tang & Hassan 2023
    │   │       "Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers"
    │   └── 證明什麼
    │       └── 我們的 fusion 是否能和 poster + text transformer 類方法比較
    │
    └── 2.3 Retrieval / RAG Competitive Baseline
        ├── 測什麼
        │   ├── no retrieval
        │   ├── vanilla semantic retrieval
        │   ├── metadata-aware retrieval
        │   └── selective retrieval, if implemented
        ├── 用什麼文獻
        │   └── Xu et al. 2025
        │       "Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation"
        └── 證明什麼
            ├── retrieval 是否能補足 target anime 本身資訊不足
            └── 若沒有 RRCP / VL-GNN / RRCP-Attention，只能稱 SKAPP-inspired，不能稱重現 SKAPP
```

```text
最簡版理解

我們要比的東西
├── 最低地板：Mean / Linear
├── 傳統強地板：metadata-only RF/XGBoost
├── 傳統多模態地板：metadata + text + image concat → XGBoost
├── 單模態地板
│   ├── text-only
│   └── image-only
└── 強競爭基準
    ├── anime deep fusion
    ├── poster-text transformer fusion
    └── retrieval / RAG fusion
```

```text
原始三篇
├── Anime popularity before huge investments
│   └── 用在：2.1 Anime Domain Deep Fusion
│       證明：同領域 multimodal deep baseline
│
├── Box-office Revenue Prediction with Transformers
│   └── 用在：2.2 Cross-modal Transformer Fusion
│       證明：poster/image + text transformer fusion baseline
│
└── SKAPP
    └── 用在：2.3 Retrieval / RAG Competitive Baseline
        證明：retrieval augmentation 是合理的強競爭 baseline
```

```text
補充文獻
├── Lo & Syu 2023
│   └── metadata-only classical ML
│
├── Chen et al. 2019 + Jeong et al. 2024
│   └── feature-concat XGBoost / RF / LightGBM
│
├── Anime synopsis traditional classifiers
│   └── text-only baseline
│
└── Zhou et al. 2019 + Rengkung & Mandala 2025
    └── image-only poster/cover baseline
```

一句話版：

> Foundation baseline 證明「簡單模型和傳統 ML 到哪裡」；Competitive baseline 證明「我們和同領域/強方法相比如何」；原始三篇負責 Competitive baseline；後補文獻負責 Foundation baseline 和單模態 baseline。

### 2.1 Original Three Anchor Papers

本次 baseline 設計最初是從三篇指定論文出發。它們沒有被取代，而是作為 competitive baseline 的三個主軸；後續補找的 XGBoost / RF / synopsis / poster 文獻，是為了補足 foundation baseline 與單模態 baseline 的立基點。

| Anchor paper | 在本專案中的定位 | 不應誤用為 |
|---|---|---|
| `Anime popularity prediction before huge investments: a multimodal approach using deep learning` | **Domain deep fusion baseline**。同領域、同 pre-investment / pre-release 精神，是整體 multimodal anime baseline 的主引用。 | 不要把它拆成所有 text/image/classical baseline 的唯一來源。 |
| `Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers` | **Cross-modal transformer fusion baseline**。提供 poster + review/text 的 transformer fusion 類比，適合作為較強的跨模態 fusion 參考。 | 不要當 text-only 主 baseline，因為 movie reviews 可能有 post-release leakage，和 anime description 不完全等價。 |
| `Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation` | **Retrieval / RAG competitive baseline**。提供 retrieval-augmented popularity prediction 的強方法參考。 | 不要把普通 `none/sparse/dense/hybrid` 查詢稱為完整 SKAPP reproduction；除非實作 RRCP、VL-GNN、RRCP-Attention。 |

因此，本專案 baseline 文獻分工是：

| 類型 | 主引用 |
|---|---|
| Domain anime fusion | 原始三篇之一：Anime huge investments |
| Cross-modal transformer fusion | 原始三篇之一：Box-office CTNN |
| Retrieval / RAG fusion | 原始三篇之一：SKAPP |
| Metadata classical ML | 補充文獻：Lo & Syu 2023 |
| Feature-concat XGBoost | 補充文獻：Chen et al. 2019 / Jeong et al. 2024 |
| Text-only anime synopsis | 補充文獻：Anime synopsis traditional classifiers |
| Image-only poster/cover | 補充文獻：Zhou et al. 2019 等 |

### 2.2 Full Baseline Map

以下是建議實作與比較的完整 baseline map。

| 優先級 | Baseline 類型 | 層級 | 是否必做 | 對應目的 |
|---:|---|---|---|---|
| 1 | Mean / Linear / Ridge | Foundation L0 | 必做 | 最低參考點 |
| 2 | Metadata classical ML | Foundation L1 | 必做 | 檢查 metadata 本身預測力 |
| 3 | Feature-concat classical ML | Foundation L2 | 必做 | 檢查不用 deep fusion，直接 concat embeddings 給 XGBoost/LightGBM 能做到哪裡 |
| 4 | Text-only baseline | Foundation / single-modality | 建議做 | 檢查 synopsis/description 本身預測力 |
| 5 | Image-only baseline | Foundation / single-modality | 建議做 | 檢查 cover/banner 視覺訊號本身預測力 |
| 6 | Domain deep fusion baseline | Competitive C1 | 必做 | 和同領域 anime multimodal deep baseline 比較 |
| 7 | Cross-modal transformer fusion | Competitive C2 | 可選/建議 | 和 poster-review transformer 類方法比較 |
| 8 | Retrieval / RAG baseline | Competitive C3 | 視時間 | 和 retrieval-augmented popularity prediction 比較 |

如果時間只能做三類，建議優先：

1. Metadata classical ML
2. Feature-concat classical ML
3. Domain deep fusion baseline

## 3. 參考文獻清單與用途

### 3.1 Main Chapter A: Anime Domain Deep Fusion Baseline

本章節必須以原始三篇之一為主引用：

**Armenta-Segura & Sidorov (2025), "Anime popularity prediction before huge investments: a multimodal approach using deep learning"**

- Link: [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12190294/)
- DOI: `10.7717/peerj-cs.2715`
- Task: anime popularity prediction before major investment
- Features: anime-related pre-investment multimodal features
- Model: GPT-2 + ResNet-50 + deep neural network architecture
- Metrics reported in abstract: MSE, R2, Pearson, Spearman

這篇是本專案最重要的 domain baseline，因為它同時滿足：

- anime domain
- pre-investment / pre-release spirit
- text + image + deep fusion
- popularity prediction

Implementation target:

| Paper component | Project implementation |
|---|---|
| GPT-2 text branch | text embedding from title + description |
| ResNet-50 image branch | image embedding from `coverImage_medium`; optional `bannerImage` |
| multimodal DNN | MLP fusion head |
| anime popularity | `popularity`; optionally also `meanScore` |

Baseline variants under this anchor:

| Baseline ID | Input | Model | Why |
|---|---|---|---|
| `C1-AnimeFusion` | metadata + text_emb + image_emb | DNN / MLP fusion | direct domain competitive baseline |
| `T2-AnimeTextBranch` | text_emb only | MLP or Ridge/XGBoost head | branch-level adaptation of GPT-2 text branch |
| `I1-AnimeImageBranch` | image_emb only | MLP or Ridge/XGBoost head | branch-level adaptation of ResNet-50 image branch |

Important caveat:

Do not claim numerical comparability with the paper. The dataset and target definition differ. Cite it as an architectural and domain baseline.

### 3.2 Main Chapter B: Cross-Modal Transformer Fusion Baseline

本章節必須以原始三篇之一為主引用：

**Madongo, Tang & Hassan (2023), "Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers"**

- Local PDF: `docs/refer/Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers.pdf`
- Link: [ACM DOI](https://dl.acm.org/doi/10.1145/3641584.3641796)
- DOI: `10.1145/3641584.3641796`
- Task: box-office revenue prediction
- Features: movie posters and reviews
- Model: Cross-modal Transformer-based Neural Network (CTNN)

這篇用來支撐「poster/image + text 的 cross-modal transformer fusion」這條 competitive baseline。

Implementation target:

| Paper component | Project implementation |
|---|---|
| movie poster | anime cover/banner image |
| movie review | anime description; note leakage caveat |
| CTNN | cross-modal transformer or lightweight cross-attention fusion |
| box-office revenue | `popularity` / `meanScore` |

Baseline variants under this anchor:

| Baseline ID | Input | Model | Why |
|---|---|---|---|
| `C2-CTNN` | text_emb + image_emb | transformer / cross-attention fusion | competitive multimodal fusion baseline |
| `T3-BoxOfficeTextAnalog` | description text only | transformer text branch + regressor | optional branch baseline if implementation cost is low |
| `I3-BoxOfficePosterAnalog` | cover/banner image only | image branch + regressor | optional branch baseline if implementation cost is low |

Important caveat:

Movie reviews may be post-release, while anime descriptions are pre-release. Therefore this paper should not be the main text-only reference. It is primarily a cross-modal fusion reference.

### 3.3 Main Chapter C: Retrieval / RAG Competitive Baseline

本章節必須以原始三篇之一為主引用：

**Xu et al. (2025), "Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation"**

- Link: [AAAI page](https://ojs.aaai.org/index.php/AAAI/article/view/32078)
- Local PDF: `docs/refer/Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation.pdf`
- Task: multimodal social media popularity prediction
- Core method: SKAPP
- Components:
  - meta retriever
  - selective refiner
  - Relative Retrieval Contribution to Prediction (RRCP)
  - vision-language GNN
  - RRCP-Attention

這篇用來支撐 retrieval-augmented popularity prediction，不是普通 feature ablation。

Implementation target:

| Paper component | Project implementation |
|---|---|
| UGC | anime item |
| UGC visual/text/metadata | anime image/text/metadata |
| related UGC retrieval | retrieve related anime from training set |
| selective refiner / RRCP | optional contribution-based filter |
| VL-GNN / RRCP-Attention | optional advanced fusion module |

Baseline variants under this anchor:

| Baseline ID | Input | Model | What can be claimed |
|---|---|---|---|
| `C3-RAG-Minimal` | metadata/text/image + retrieved features | fusion regressor | SKAPP-inspired retrieval baseline |
| `C3-RAG-Selective` | top-k retrieval + contribution filter | fusion regressor | closer SKAPP adaptation |
| `C3-SKAPP-Repro` | RRCP + VL-GNN + RRCP-Attention | full retrieval model | only this can be called SKAPP-style reproduction |

Important caveat:

Unless RRCP, VL-GNN, and RRCP-Attention are implemented, do not write "we reproduce SKAPP." Use "SKAPP-inspired retrieval baseline."

### 3.4 Supplemental Foundation: Metadata Classical ML Baseline

主引用：

**Lo & Syu (2023), "Analyzing drama metadata through machine learning to gain insights into social information dissemination patterns"**

- Link: [PLOS ONE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0288932)
- DOI: `10.1371/journal.pone.0288932`
- Task: Japanese prime-time TV drama rating classification
- Features: pre-broadcast metadata, such as broadcast year, season, station, day, time, genre, screenwriter, original/sequel, cast; optional poster facial features
- Models: Naive Bayes, ANN, SVM, Random Forest
- Verified claim: RF improves from 75.80% to 77.10% after adding poster facial features, according to the public abstract / article metadata

How to use in this project:

| Paper component | Project equivalent |
|---|---|
| drama pre-broadcast metadata | anime pre-release metadata |
| rating high/low classification | `popularity` / `meanScore` regression |
| Random Forest | RF baseline; XGBoost/LightGBM can be added as stronger tabular variants |
| optional poster face feature | optional image feature baseline, not pure metadata |

Implementation:

- Input: metadata only
- Models:
  - Mean predictor
  - Linear / Ridge
  - Random Forest Regressor
  - XGBoost Regressor or LightGBM Regressor if available
- Targets:
  - `popularity`, preferably `log1p(popularity)` during training
  - `meanScore`
- Metrics:
  - MAE
  - RMSE or MSE
  - R2
  - Spearman rho
  - `log_MAE` for popularity

Important caveat:

This paper supports "pre-broadcast entertainment metadata + classical ML." It does not directly support "XGBoost regression", because the paper itself uses RF/SVM/ANN/NB and classification.

### 3.5 Supplemental Foundation: Feature-Concat Classical ML Baseline

主引用：

**Chen et al. (2019), "Social Media Popularity Prediction Based on Visual-Textual Features with XGBoost"**

- Link: [ACM DOI](https://dl.acm.org/doi/10.1145/3343031.3356072)
- DOI: `10.1145/3343031.3356072`
- Task: social media popularity prediction
- Features: visual-textual features plus metadata, fused into XGBoost
- Model: XGBoost regression
- Verified high-level claim: visual-textual feature extraction, feature fusion, and XGBoost are central to the method

Supplementary modern reference:

**Jeong et al. (2024), "Enhancing Social Media Post Popularity Prediction with Visual Content"**

- Link: [arXiv](https://arxiv.org/abs/2405.02367)
- DOI: `10.48550/arXiv.2405.02367`
- Journal version DOI reported in metadata: `10.1007/s42952-024-00270-7`
- Task: Instagram post like-count prediction
- Features: image labels / representative colors from Google Cloud Vision API, user metadata, tags
- Models: LMM, SVR, MLP, Random Forest, XGBoost
- Verified claim: compares XGBoost/RF/SVR/MLP and studies visual-content contribution

How to use in this project:

| Paper component | Project equivalent |
|---|---|
| visual-textual features | image embeddings + text embeddings |
| user / post metadata | anime pre-release metadata |
| XGBoost final regressor | XGBoost/LightGBM on concatenated features |

Implementation:

- Input variants:
  - metadata only
  - metadata + text embedding
  - metadata + image embedding
  - metadata + text embedding + image embedding
- Model:
  - XGBoost Regressor or LightGBM Regressor
  - RF can be included as a simpler tree baseline
- This is not a neural fusion model. It is a feature-concat classical ML baseline.

Important caveat:

Chen et al. 2019 should be cited for feature-concat XGBoost. Do not overclaim that it uses the same encoders as our project. It is the fusion pattern, not exact feature extractor, that is being borrowed.

### 3.6 Supplemental Single-Modality: Text-Only Baseline

Recommended reference:

**"Anime Success Prediction Based on Synopsis Using Traditional Classifiers"**

- Link: [PDF](https://www.rcs.cic.ipn.mx/2023_152_9/Anime%20Success%20Prediction%20Based%20on%20Synopsis%20Using%20Traditional%20Classifiers.pdf)
- Task: anime success prediction from synopsis
- Features: synopsis text
- Models: traditional classifiers such as SVM / Naive Bayes / Logistic Regression, based on the retrieved summary

How to use in this project:

| Paper component | Project equivalent |
|---|---|
| anime synopsis | AniList `description` / title + description |
| traditional classifiers | TF-IDF + Ridge/SVR/RF/XGBoost or embedding + regressor |
| success classification | `popularity` / `meanScore` regression |

Implementation options:

Option A, classical text baseline:

- TF-IDF on cleaned description
- Ridge / SVR / Random Forest / XGBoost

Option B, embedding text baseline:

- Use existing text embedding pipeline
- Feed text embedding into Ridge / MLP / XGBoost

Important caveat:

If the implementation uses sentence-transformer embeddings instead of n-gram features, describe it as a project-adapted text baseline inspired by synopsis-only anime success prediction, not as an exact reproduction.

### 3.7 Supplemental Single-Modality: Image-Only / Poster-Cover Baseline

Stable primary reference:

**Zhou, Zhang & Yi (2019), "Predicting movie box-office revenues using deep neural networks"**

- Link: [ResearchGate metadata](https://www.researchgate.net/publication/318831837_Predicting_movie_box-office_revenues_using_deep_neural_networks)
- DOI: `10.1007/s00521-017-3162-x`
- Journal: Neural Computing and Applications
- Task: pre-release movie box-office revenue prediction
- Features: movie poster CNN features plus movie-related data
- Verified high-level claim: CNN is used to extract movie poster features; multimodal DNN combines poster features and movie-related data

Modern candidate, requires full-text confirmation:

**Rengkung & Mandala (2025), "Investigating the Impact of Movie Poster Clustering on Box Office Prediction"**

- Link: [ResearchGate metadata](https://www.researchgate.net/publication/394626010_Investigating_the_Impact_of_Movie_Poster_Clustering_on_Box_Office_Prediction)
- DOI: `10.1109/IAICT65714.2025.11100483`
- Venue: 2025 IEEE International Conference on Industry 4.0, Artificial Intelligence, and Communications Technology
- Current verified status: title, authors, year, DOI, venue are verified from public metadata
- Not fully verified from public metadata: VGG16, K-means clusters, LightGBM/XGBoost/RF/SVR details, R2 value

How to use in this project:

| Paper component | Project equivalent |
|---|---|
| movie poster | anime `coverImage_medium`; optionally `bannerImage` |
| CNN poster feature | Swin / ResNet / CLIP image embedding |
| movie-related data | anime metadata |
| box-office revenue | `popularity` / `meanScore` |

Implementation:

Image-only baseline:

- image embedding only
- Ridge / MLP / XGBoost regressor

Image + metadata baseline:

- metadata + image embedding
- XGBoost/LightGBM or shallow MLP

Important caveats:

- Do not cite Rengkung & Mandala 2025 for specific implementation details unless the full paper is obtained and checked.
- Do not write "Qu et al. 2017" for DOI `10.1007/s00521-017-3162-x`; the correct authors are Zhou, Zhang & Yi.

## 4. Recommended Experiment Matrix

### 4.1 Foundation Baselines

| ID | Name | Input | Model | Reference |
|---|---|---|---|---|
| F0-Mean | Mean predictor | none | train target mean | common baseline |
| F0-Linear | Linear/Ridge | metadata | Linear/Ridge regression | common baseline |
| F1-RF-Meta | Metadata RF | metadata | Random Forest | Lo & Syu 2023 |
| F1-XGB-Meta | Metadata XGBoost | metadata | XGBoost/LightGBM | Lo & Syu 2023 + classical ML adaptation |
| F2-XGB-Concat | Feature-concat XGBoost | metadata + text_emb + image_emb | XGBoost/LightGBM | Chen et al. 2019; Jeong et al. 2024 |

### 4.2 Single-Modality Baselines

| ID | Name | Input | Model | Reference |
|---|---|---|---|---|
| T1-TFIDF | Text TF-IDF baseline | title + description | TF-IDF + Ridge/SVR/XGBoost | Anime synopsis paper |
| T2-Emb | Text embedding baseline | text embedding | Ridge/MLP/XGBoost | project adaptation |
| I1-Emb | Image embedding baseline | cover image embedding | Ridge/MLP/XGBoost | Zhou et al. 2019 |
| I2-CoverBanner | Cover + banner image baseline | cover + banner embeddings | concat + regressor | Zhou et al. 2019 adaptation |

### 4.3 Competitive Baselines

| ID | Name | Input | Model | Reference |
|---|---|---|---|---|
| C1-AnimeFusion | Domain deep fusion | text + image + metadata | GPT-2/ResNet-style MLP adaptation | Armenta-Segura & Sidorov 2025 |
| C2-CTNN | Cross-modal transformer fusion | text + image | transformer fusion | Madongo et al. 2023 |
| C3-RAG | SKAPP-inspired retrieval | text + image + metadata + retrieved features | retrieval-augmented fusion | Xu et al. 2025 |

## 5. Feature and Target Rules

### 5.1 Allowed Pre-release Inputs

Use only fields available before release:

- title fields
- description
- format
- source
- countryOfOrigin
- season / seasonYear
- release year / quarter
- episodes / duration if available pre-release
- genres
- studios
- coverImage_medium
- bannerImage
- trailer thumbnail only if the project later decides to include it
- relation-derived prequel features only if constructed without leakage

Avoid post-release leakage:

- favourites
- trending
- averageScore if used as post-release signal
- user ratings collected after release
- real review text unless explicitly available before release
- social media volume after release

### 5.2 Targets

Primary targets:

- `popularity`
- `meanScore`

Recommended target transforms:

- `popularity`: train on `log1p(popularity)` or standardized log popularity
- `meanScore`: direct z-score or raw-scale regression depending pipeline

### 5.3 Metrics

Use the same metrics for every baseline:

- MAE
- RMSE or MSE
- R2
- Spearman rho
- Pearson r if desired
- log_MAE for popularity

Primary report metrics:

- popularity: Spearman rho, MAE, log_MAE, R2
- meanScore: Spearman rho, MAE, R2

## 6. Fairness and Reporting Requirements

For all baselines:

1. Use the same train / val / test split.
2. Fit scalers, encoders, vocabularies, and imputers on train only.
3. Do not tune on test.
4. Keep target transforms consistent across models.
5. Report missing-modality handling.
6. Report whether `bannerImage` is used.
7. Report whether RAG retrieves only from train set and earlier release periods.

## 7. What Is Already Collected vs. Still Risky

Collected and usable:

| Purpose | Reference | Status |
|---|---|---|
| Metadata classical ML | Lo & Syu 2023 | usable |
| Feature-concat XGBoost | Chen et al. 2019 | usable |
| Modern visual-content classical ML support | Jeong et al. 2024 | usable |
| Text-only anime synopsis | Anime synopsis traditional classifiers | usable, but details should be checked from PDF |
| Domain anime deep fusion | Armenta-Segura & Sidorov 2025 | usable |
| Cross-modal transformer fusion | Madongo et al. 2023 | usable |
| RAG competitive baseline | Xu et al. 2025 SKAPP | usable |
| Poster CNN baseline | Zhou et al. 2019 | usable |

Risky or requiring full-text confirmation:

| Reference | Issue |
|---|---|
| Rengkung & Mandala 2025 | metadata verified, but method details such as VGG16/K-means/LGBM/XGBoost/R2 need full text |
| Any claim saying "Qu et al. 2017" for DOI `10.1007/s00521-017-3162-x` | incorrect author attribution |

## 8. Suggested Implementation Order

1. Implement `F0-Mean` and `F0-Linear`.
2. Implement `F1-RF-Meta` and `F1-XGB-Meta`.
3. Implement `F2-XGB-Concat` using existing text/image embeddings.
4. Implement `T1/T2` text-only baselines.
5. Implement `I1/I2` image-only baselines.
6. Implement `C1-AnimeFusion`.
7. Only if time permits, implement `C2-CTNN` and `C3-RAG`.

## 9. Minimal Deliverables

The implementation AI should produce:

1. A baseline config file listing enabled baselines and feature sets.
2. A training script that can run one baseline by name.
3. A result table with one row per baseline and target.
4. A markdown summary mapping every result row to its reference paper.
5. A leakage checklist confirming only pre-release features were used.

Recommended output table columns:

| baseline_id | target | feature_set | model | reference | val_MAE | val_R2 | val_Spearman | test_MAE | test_R2 | test_Spearman | notes |
|---|---|---|---|---|---:|---:|---:|---:|---:|---:|---|

## 10. Final Interpretation Guide

When results are available:

- If deep fusion only beats Mean/Linear but not XGBoost, the model architecture is not justified.
- If XGBoost concat is close to deep fusion, then embeddings are useful but deep fusion may be overkill.
- If text-only is strong, description carries major signal.
- If image-only is weak but image improves fusion, image works as complementary signal.
- If RAG helps only popularity but not meanScore, retrieval may capture market attention more than quality.
- If all pre-release baselines have low R2, this supports the limitation that pre-release features have a natural prediction ceiling.
