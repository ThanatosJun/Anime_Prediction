# Decision EDA Summary

- Generated at (UTC): `2026-04-23T07:29:41.287713+00:00`
- Rows: `20324`
- Columns: `61`

## Missing Value Policy Recommendations

- `chapters`: missing=100.0000%, recommend=`drop`
- `volumes`: missing=100.0000%, recommend=`drop`
- `nextAiringEpisode`: missing=99.8524%, recommend=`drop`
- `hashtag`: missing=78.1391%, recommend=`fill`
- `bannerImage`: missing=63.8408%, recommend=`fill`
- `trailer_id`: missing=63.1470%, recommend=`fill`
- `trailer_site`: missing=63.1470%, recommend=`fill`
- `trailer_thumbnail`: missing=63.1470%, recommend=`fill`
- `title_english`: missing=52.2683%, recommend=`fill`
- `season`: missing=31.2881%, recommend=`fill`
- `seasonYear`: missing=31.2881%, recommend=`fill`
- `seasonInt`: missing=30.3090%, recommend=`fill`
- `averageScore`: missing=20.9358%, recommend=`fill`
- `source`: missing=11.4151%, recommend=`keep`
- `coverImage_color`: missing=8.6154%, recommend=`keep`
- `endDate_day`: missing=8.4678%, recommend=`keep`
- `startDate_day`: missing=6.9130%, recommend=`keep`
- `endDate_month`: missing=6.1799%, recommend=`keep`
- `description`: missing=5.8404%, recommend=`keep`
- `idMal`: missing=4.9154%, recommend=`keep`

## Outlier Policy Recommendations

- `episodes`: outlier_ratio=8.6892%, bounds=[-15.5, 28.5], recommend=`winsorize_p1_p99`
- `duration`: outlier_ratio=10.1703%, bounds=[-25.0, 55.0], recommend=`clip_p1_p99`
- `averageScore`: outlier_ratio=0.1494%, bounds=[28.0, 92.0], recommend=`retain`
- `meanScore`: outlier_ratio=0.1230%, bounds=[23.0, 95.0], recommend=`retain`
- `popularity`: outlier_ratio=16.5371%, bounds=[-7750.875, 13542.125], recommend=`clip_p1_p99`
- `favourites`: outlier_ratio=16.5273%, bounds=[-93.5, 158.5], recommend=`clip_p1_p99`
- `trending`: outlier_ratio=9.5798%, bounds=[0.0, 0.0], recommend=`winsorize_p1_p99`

## Correlation Profile (to targets)

- `episodes`: popularity=0.05003406693105933, averageScore=0.09770218767609681
- `duration`: popularity=0.08411294863366227, averageScore=0.260145664380285
- `meanScore`: popularity=0.34013060234800113, averageScore=0.9713205270890001
- `favourites`: popularity=0.8635627643781054, averageScore=0.2906763463101651
- `trending`: popularity=0.6562765211323012, averageScore=0.245707362264052
- `seasonYear`: popularity=0.1732473372070323, averageScore=0.22880583502974464

## Format Impact on Popularity (Top 10 by count)

- `TV`: count=4559, median=12197.0, mean=44981.46325948673
- `OVA`: count=3770, median=1473.0, mean=4722.749336870026
- `MOVIE`: count=3242, median=652.5, mean=9639.949413942011
- `ONA`: count=3029, median=475.0, mean=4427.3251898316275
- `MUSIC`: count=2654, median=156.0, mean=466.8036925395629
- `SPECIAL`: count=1770, median=1289.5, mean=4139.299435028249
- `TV_SHORT`: count=1299, median=387.0, mean=4079.107775211701
- `None`: count=1, median=227.0, mean=227.0
