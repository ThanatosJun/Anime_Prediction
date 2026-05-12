# Text Branch Quality Ranking

Ranking policy:
- Selection rank uses validation metrics only (lower MAE/RMSE is better, higher Spearman is better).
- Test rank is reported separately as a holdout view.

## Leaderboard

| Rank | Experiment | Model Key | Val Rank Avg | Test Rank Avg | popularity Test Spearman | meanScore Test Spearman | popularity Test RMSE | meanScore Test RMSE |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | e5_base | e5_base | 1.00 | 1.67 | 0.6172 | 0.2525 | 32060.1275 | 13.1309 |
| 2 | e5_small | e5_small | 2.17 | 1.83 | 0.5967 | 0.2406 | 33020.0928 | 13.0786 |
| 3 | bge_base | bge_base | 3.33 | 3.83 | 0.5565 | 0.2278 | 33511.6782 | 13.1882 |
| 4 | multilingual_minilm | multilingual_minilm | 4.17 | 4.50 | 0.5379 | 0.2054 | 33910.5157 | 13.1132 |
| 5 | minilm_l6 | minilm_l6 | 5.17 | 4.50 | 0.5408 | 0.2152 | 34055.3225 | 13.1228 |
| 6 | bge_small | bge_small | 5.17 | 4.67 | 0.5616 | 0.2209 | 33791.4261 | 13.2509 |

## Metric Columns Used

- popularity_val_MAE
- popularity_val_RMSE
- popularity_val_Spearman
- popularity_test_MAE
- popularity_test_RMSE
- popularity_test_Spearman
- meanScore_val_MAE
- meanScore_val_RMSE
- meanScore_val_Spearman
- meanScore_test_MAE
- meanScore_test_RMSE
- meanScore_test_Spearman
