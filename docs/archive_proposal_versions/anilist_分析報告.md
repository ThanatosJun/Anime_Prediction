# AniList Anime 資料分析報告

- **資料檔路徑**：`C:/Users/g1014308/Desktop/archive_4/anilist_anime_data_complete.csv`
- **分析日期**：2026-04-23

## 1) 資料概覽

- 列數（rows）：**20,324**
- 欄數（columns）：**62**
- 記憶體估計（含字串深度）：**438,719,959 bytes（約 418.40 MB）**

## 2) 全欄位清單與 dtype

| 欄位                     | dtype   |   非缺失數 |   缺失數 |
|:-------------------------|:--------|-----------:|---------:|
| Unnamed: 0               | int64   |      20324 |        0 |
| id                       | int64   |      20324 |        0 |
| idMal                    | float64 |      19325 |      999 |
| title_romaji             | object  |      20324 |        0 |
| title_english            | object  |       9701 |    10623 |
| title_native             | object  |      20157 |      167 |
| title_userPreferred      | object  |      20324 |        0 |
| type                     | object  |      20324 |        0 |
| format                   | object  |      20323 |        1 |
| status                   | object  |      20324 |        0 |
| description              | object  |      19085 |     1239 |
| startDate_year           | int64   |      20324 |        0 |
| startDate_month          | float64 |      19380 |      944 |
| startDate_day            | float64 |      18919 |     1405 |
| endDate_year             | float64 |      20074 |      250 |
| endDate_month            | float64 |      19068 |     1256 |
| endDate_day              | float64 |      18603 |     1721 |
| season                   | object  |      13965 |     6359 |
| seasonYear               | float64 |      13965 |     6359 |
| seasonInt                | float64 |      14164 |     6160 |
| episodes                 | float64 |      20163 |      161 |
| duration                 | float64 |      20137 |      187 |
| chapters                 | float64 |          0 |    20324 |
| volumes                  | float64 |          0 |    20324 |
| countryOfOrigin          | object  |      20324 |        0 |
| isLicensed               | bool    |      20324 |        0 |
| source                   | object  |      18004 |     2320 |
| hashtag                  | object  |       4415 |    15909 |
| trailer_id               | object  |       7490 |    12834 |
| trailer_site             | object  |       7490 |    12834 |
| trailer_thumbnail        | object  |       7490 |    12834 |
| updatedAt                | int64   |      20324 |        0 |
| coverImage_extraLarge    | object  |      20324 |        0 |
| coverImage_large         | object  |      20324 |        0 |
| coverImage_medium        | object  |      20324 |        0 |
| coverImage_color         | object  |      18573 |     1751 |
| bannerImage              | object  |       7349 |    12975 |
| genres                   | object  |      20324 |        0 |
| synonyms                 | object  |      20324 |        0 |
| tags                     | object  |      20324 |        0 |
| averageScore             | float64 |      16069 |     4255 |
| meanScore                | int64   |      20324 |        0 |
| popularity               | int64   |      20324 |        0 |
| favourites               | int64   |      20324 |        0 |
| trending                 | int64   |      20324 |        0 |
| rankings                 | object  |      20324 |        0 |
| isFavourite              | bool    |      20324 |        0 |
| isAdult                  | bool    |      20324 |        0 |
| isLocked                 | bool    |      20324 |        0 |
| siteUrl                  | object  |      20324 |        0 |
| externalLinks            | object  |      20324 |        0 |
| streamingEpisodes        | object  |      20324 |        0 |
| relations                | object  |      20324 |        0 |
| characters               | object  |      20324 |        0 |
| staff                    | object  |      20324 |        0 |
| studios                  | object  |      20324 |        0 |
| nextAiringEpisode        | object  |         30 |    20294 |
| airingSchedule           | object  |      20324 |        0 |
| recommendations          | object  |      20324 |        0 |
| reviews                  | object  |      20324 |        0 |
| stats_scoreDistribution  | object  |      20324 |        0 |
| stats_statusDistribution | object  |      20324 |        0 |

## 3) 缺失率前20欄

| 欄位              |   缺失數 | 缺失率   |
|:------------------|---------:|:---------|
| chapters          |    20324 | 100.00%  |
| volumes           |    20324 | 100.00%  |
| nextAiringEpisode |    20294 | 99.85%   |
| hashtag           |    15909 | 78.28%   |
| bannerImage       |    12975 | 63.84%   |
| trailer_id        |    12834 | 63.15%   |
| trailer_site      |    12834 | 63.15%   |
| trailer_thumbnail |    12834 | 63.15%   |
| title_english     |    10623 | 52.27%   |
| season            |     6359 | 31.29%   |
| seasonYear        |     6359 | 31.29%   |
| seasonInt         |     6160 | 30.31%   |
| averageScore      |     4255 | 20.94%   |
| source            |     2320 | 11.42%   |
| coverImage_color  |     1751 | 8.62%    |
| endDate_day       |     1721 | 8.47%    |
| startDate_day     |     1405 | 6.91%    |
| endDate_month     |     1256 | 6.18%    |
| description       |     1239 | 6.10%    |
| idMal             |      999 | 4.92%    |

## 4) 目標相關欄位描述統計與分位數

### averageScore
| 指標   | 數值    |
|:-------|:--------|
| count  | 16,069  |
| mean   | 59.9732 |
| std    | 11.2141 |
| min    | 16.0000 |
| 1%     | 35.0000 |
| 5%     | 41.0000 |
| 25%    | 52.0000 |
| 50%    | 61.0000 |
| 75%    | 68.0000 |
| 95%    | 78.0000 |
| 99%    | 84.0000 |
| max    | 91.0000 |

### meanScore
| 指標   | 數值    |
|:-------|:--------|
| count  | 20,324  |
| mean   | 58.9918 |
| std    | 12.5799 |
| min    | 10.0000 |
| 1%     | 30.0000 |
| 5%     | 37.0000 |
| 25%    | 50.0000 |
| 50%    | 61.0000 |
| 75%    | 68.0000 |
| 95%    | 78.0000 |
| 99%    | 83.0000 |
| max    | 91.0000 |

### popularity
| 指標   | 數值        |
|:-------|:------------|
| count  | 20,324      |
| mean   | 13845.8367  |
| std    | 48171.0846  |
| min    | 9.0000      |
| 1%     | 25.0000     |
| 5%     | 50.0000     |
| 25%    | 234.0000    |
| 50%    | 1029.0000   |
| 75%    | 5557.2500   |
| 95%    | 67951.9000  |
| 99%    | 231528.9000 |
| max    | 977273.0000 |

### favourites
| 指標   | 數值       |
|:-------|:-----------|
| count  | 20,324     |
| mean   | 379.1933   |
| std    | 2388.9191  |
| min    | 0.0000     |
| 1%     | 0.0000     |
| 5%     | 0.0000     |
| 25%    | 1.0000     |
| 50%    | 10.0000    |
| 75%    | 64.0000    |
| 95%    | 1290.5500  |
| 99%    | 7852.5500  |
| max    | 99684.0000 |

### trending
| 指標   | 數值     |
|:-------|:---------|
| count  | 20,324   |
| mean   | 0.2779   |
| std    | 1.8789   |
| min    | 0.0000   |
| 1%     | 0.0000   |
| 5%     | 0.0000   |
| 25%    | 0.0000   |
| 50%    | 0.0000   |
| 75%    | 0.0000   |
| 95%    | 1.0000   |
| 99%    | 6.0000   |
| max    | 145.0000 |

### episodes
| 指標   | 數值      |
|:-------|:----------|
| count  | 20,163    |
| mean   | 12.0048   |
| std    | 38.3364   |
| min    | 1.0000    |
| 1%     | 1.0000    |
| 5%     | 1.0000    |
| 25%    | 1.0000    |
| 50%    | 2.0000    |
| 75%    | 12.0000   |
| 95%    | 50.0000   |
| 99%    | 104.0000  |
| max    | 1818.0000 |

### duration
| 指標   | 數值     |
|:-------|:---------|
| count  | 20,137   |
| mean   | 24.0364  |
| std    | 25.4144  |
| min    | 1.0000   |
| 1%     | 1.0000   |
| 5%     | 2.0000   |
| 25%    | 5.0000   |
| 50%    | 23.0000  |
| 75%    | 25.0000  |
| 95%    | 90.0000  |
| 99%    | 115.0000 |
| max    | 168.0000 |

## 5) 類別欄位分布（前10）

### type
| 值    |   筆數 | 占比    |
|:------|-------:|:--------|
| ANIME |  20324 | 100.00% |

### status
| 值        |   筆數 | 占比   |
|:----------|-------:|:-------|
| FINISHED  |  20116 | 98.98% |
| RELEASING |    194 | 0.95%  |
| CANCELLED |     14 | 0.07%  |

### source
| 值           |   筆數 | 占比   |
|:-------------|-------:|:-------|
| ORIGINAL     |   7313 | 35.98% |
| MANGA        |   5229 | 25.73% |
| OTHER        |   2390 | 11.76% |
| <NA>         |   2320 | 11.42% |
| VIDEO_GAME   |   1067 | 5.25%  |
| VISUAL_NOVEL |   1020 | 5.02%  |
| LIGHT_NOVEL  |    985 | 4.85%  |

### format
| 值       |   筆數 | 占比   |
|:---------|-------:|:-------|
| TV       |   4559 | 22.43% |
| OVA      |   3770 | 18.55% |
| MOVIE    |   3242 | 15.95% |
| ONA      |   3029 | 14.90% |
| MUSIC    |   2654 | 13.06% |
| SPECIAL  |   1770 | 8.71%  |
| TV_SHORT |   1299 | 6.39%  |
| <NA>     |      1 | 0.00%  |

### season
| 值     |   筆數 | 占比   |
|:-------|-------:|:-------|
| <NA>   |   6359 | 31.29% |
| FALL   |   3940 | 19.39% |
| SPRING |   3838 | 18.88% |
| SUMMER |   3279 | 16.13% |
| WINTER |   2908 | 14.31% |

### rating
- 欄位不存在。

### countryOfOrigin
| 值   |   筆數 | 占比   |
|:-----|-------:|:-------|
| JP   |  18256 | 89.82% |
| CN   |   1676 | 8.25%  |
| KR   |    364 | 1.79%  |
| TW   |     28 | 0.14%  |

## 6) genres 欄位解析

- 空陣列（或無法解析為有效類別）比例：**13.14%**
| genre         |   出現次數 |
|:--------------|-----------:|
| Comedy        |       6777 |
| Action        |       5006 |
| Fantasy       |       4488 |
| Adventure     |       3743 |
| Drama         |       3172 |
| Sci-Fi        |       2899 |
| Slice of Life |       2707 |
| Romance       |       2620 |
| Supernatural  |       1767 |
| Hentai        |       1605 |
| Mecha         |       1132 |
| Ecchi         |        962 |
| Mystery       |        912 |
| Sports        |        842 |
| Music         |        833 |
| Psychological |        780 |
| Horror        |        538 |
| Mahou Shoujo  |        456 |
| Thriller      |        226 |

## 7) 時間欄位與資料品質

- startDate_* 欄位：startDate_day, startDate_month, startDate_year
- endDate_* 欄位：endDate_day, endDate_month, endDate_year
- startDate_year 範圍：1940 ~ 2025（非缺失）
- endDate_year 範圍：1940 ~ 2026（非缺失）
- seasonYear 範圍：1940 ~ 2026（非缺失）
- 非法月份筆數（不在 1~12）：startDate_month: 0； endDate_month: 0
- 非法日期筆數（不在 1~31）：startDate_day: 0； endDate_day: 0
- `end_before_start` 筆數：**8**

## 8) 唯一性與重複

- 重複列數（全欄位完全相同）：**0**
- `id`：非缺失 20,324、唯一值 20,324、非缺失重複值數 0
- `idMal`：非缺失 19,325、唯一值 19,305、非缺失重複值數 20

## 9) 給「上映後人氣與品質預測」的可執行資料清理與特徵工程清單

- 統一缺失值表示（空字串、Unknown、[]、0 的語意需分流），並保留缺失指示欄位（missing indicators）。
- 對 averageScore、meanScore、popularity、favourites、trending 做 log1p 候選轉換以降低長尾影響。
- 以 IQR 或分位數截尾（winsorization）處理極端值，避免少數爆量作品主導模型。
- 時間特徵工程：由 start/end 日期建構上映年份、季度、是否完結、上映至今月數。
- 類別欄位（type/status/source/format/season/rating/countryOfOrigin）採 One-Hot 或 Target Encoding（需交叉驗證防洩漏）。
- genres 解析後做 Multi-hot 編碼；可新增 genre 數量、主 genre、稀有 genre 指標。
- 長度特徵：episodes、duration 建立總時長（episodes*duration）與分箱特徵。
- 建立互動特徵：如 format×season、source×rating、country×genre。
- 切分策略採時間切分（time-based split）以模擬真實預測場景，避免未來資訊外洩。
- 評估指標建議同時使用 MAE/RMSE（迴歸）與 Spearman（排序相關）衡量熱門度/品質預測效果。

## 10) 結論摘要

1. 資料集規模為 20,324 列、62 欄，記憶體約 418.40 MB。
2. 缺失最嚴重欄位為 `chapters`，缺失率 100.00%。
3. genres 欄位可解析為多標籤特徵，適合做 multi-hot 與主題聚合特徵。
4. id 非缺失重複值數為 0，可用於主鍵一致性檢核。
