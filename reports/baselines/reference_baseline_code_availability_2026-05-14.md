# Reference Baseline 論文程式碼可用性盤點

更新日期：2026-05-14

目的：整理 baseline 參考論文是否有公開程式碼，方便對照本專案目前的 reproduction / adaptation 位置。

## 路線總覽表（0 ~ 2.3）

| baseline 路線 | 論文 | 是否找到 GitHub | 連結 | 判讀 |
|---|---|---|---|---|
| 0 (Lowest Reference) | 通用 baseline（Mean / Linear / Ridge） | 不適用 | （不需） | `F0` 是通用統計/線性地板，不是特定論文復現，故沒有「官方 repo」問題。 |
| 1.1 (Metadata-only Classical ML) | Lo & Syu (2023), *Analyzing drama metadata through machine learning...* | 未見公開 GitHub | （暫無） | 文章頁可見資料可用性敘述，但未看到明確 GitHub 連結。 |
| 1.2 (Feature-concat Classical ML) | Chen et al. (2019), *Social Media Popularity Prediction Based on Visual-Textual Features with XGBoost* | 未確認官方 repo | （暫無） | 目前找到的是 challenge 或衍生實作，不足以判定為該 paper 官方程式碼。 |
| 1.2 (Feature-concat Classical ML, supplementary) | Jeong et al. (2024), *Enhancing Social Media Post Popularity Prediction with Visual Content* | 未見官方 GitHub | （暫無） | 目前可找到 arXiv/期刊頁與摘要，未看到作者提供 code repository。 |
| 1.3 (Text-only Baseline) | *Anime Success Prediction Based on Synopsis Using Traditional Classifiers* | 未確認官方 repo | （暫無） | 有社群專題 repo 類似題目，但未能確認為原文作者官方釋出。 |
| 1.4 (Image-only Baseline) | Zhou, Zhang & Yi (2019), *Predicting movie box-office revenues using deep neural networks* | 未見官方 GitHub | （暫無） | 目前未找到可直接對應該 DOI 的作者官方程式碼。 |
| 1.4 (Image-only Baseline, supplementary) | Rengkung & Mandala (2025), *Investigating the Impact of Movie Poster Clustering on Box Office Prediction* | 未確認官方 repo | （暫無） | 目前有論文/中繼頁與相近主題 repo，但尚無可確認的官方 code link。 |
| C1 | Armenta-Segura & Sidorov (2025), *Anime popularity prediction before huge investments* | 有（高可信） | <https://github.com/JesusASmx/Popularity-Prediction-in-Anime-with-Deep-Learning> | repo 與作者帳號 `JesusASmx` 一致，內容主題對應該文，建議視為「官方/作者釋出」。 |
| C2 | Madongo, Tang & Hassan (2023), *Box-office Revenue Prediction by Mining Deep Features from Movie Posters and Reviews Using Transformers* | 未確認官方 repo | （暫無） | 目前僅找到主題相近的第三方實作，未找到可明確對應該篇 CTNN 論文的作者官方程式碼。 |
| C3 | Xu et al. (2025), *Improving Multimodal Social Media Popularity Prediction via Selective Retrieval Knowledge Augmentation (SKAPP)* | 有（中高可信） | <https://github.com/Xovee/skapp> | 搜尋結果顯示為 SKAPP 實作；且標示為 `YifanZhang-git/SKAPP` 相關 fork，建議作為可參考實作。 |

## 你剛問的重點：0 ~ 1.4 快速結論

- `0`：不需對應論文官方程式碼（屬通用 baseline）。
- `1.1`：目前未找到 Lo & Syu 官方 GitHub。
- `1.2`：目前未找到 Chen 2019 / Jeong 2024 官方 GitHub。
- `1.3`：目前未找到 synopsis paper 官方 GitHub。
- `1.4`：目前未找到 Zhou 2019 / Rengkung 2025 官方 GitHub。

## 對比你目前復現狀況的建議

1. **優先對比 C1 與 C3**：這兩條目前最有機會做「論文方法細節 vs 專案 proxy」的可操作比對。
2. **C2 暫時以論文描述比對**：在未找到官方 repo 前，維持你目前 `CTNN-style project-input proxy` 的 claim 邊界最安全。
3. **Foundation 類 (F1/F2/T/I) 以「文獻路線」為主**：無官方碼時，重點放在方法精神對齊，不做「實作等價」主張。
4. **報告建議固定一句**：`official implementation availability varies across references; therefore, several baselines are implemented as project-aligned adaptations/proxies.`

## 後續可補強（若要更嚴謹）

- 逐篇再做一次「paper PDF / supplementary」人工核對，確認是否有遺漏的 code link。
- 對已找到 repo 的 C1/C3，補一頁 `implementation-gap checklist`（輸入、encoder、fusion、target、split）做逐項打勾。
