# External MAL sample-aligned evaluation

These baselines use CARMA-input artifacts so they can be evaluated on the exact MAL local-ready rows.
They are not exact replacements for the older paper reference baselines that used 384-d text and 1024-d image artifacts.

| exam                           | model                    | target     |    n | log_MAE   | factor_acc_2x   |   Spearman_rho |       MAE | acc_within_10pt   | R2      |
|:-------------------------------|:-------------------------|:-----------|-----:|:----------|:----------------|---------------:|----------:|:------------------|:--------|
| mal2025_popularity_local_ready | CARMA-Run02              | popularity | 3765 | 1.012     | 0.4656          |         0.4709 | 3518.33   |                   |         |
| mal2025_dual_local_ready       | CARMA-Run02              | popularity | 1202 | 1.391     | 0.3344          |         0.5495 | 7750.94   |                   |         |
| mal2025_dual_local_ready       | CARMA-Run02              | meanScore  | 1202 |           |                 |         0.6079 |    7.5086 | 0.7488            | -1.0659 |
| mal2025_popularity_local_ready | F1-RF-Meta               | popularity | 3765 | 1.2148    | 0.3503          |         0.2971 | 3690.26   | nan               | 0.032   |
| mal2025_popularity_local_ready | F2-XGB-Concat-CARMAInput | popularity | 3765 | 1.144     | 0.341           |         0.4298 | 3592.53   | nan               | -0.0113 |
| mal2025_popularity_local_ready | C3-RAG-XGB-CARMAInput    | popularity | 3765 | 1.1689    | 0.3588          |         0.3153 | 3598.65   | nan               | 0.0045  |
| mal2025_dual_local_ready       | F1-RF-Meta               | popularity | 1202 | 1.0134    | 0.4567          |         0.626  | 7320.34   | nan               | 0.0059  |
| mal2025_dual_local_ready       | F1-RF-Meta               | meanScore  | 1202 | nan       | nan             |         0.6339 |    6.6473 | 0.787             | -0.3838 |
| mal2025_dual_local_ready       | F2-XGB-Concat-CARMAInput | popularity | 1202 | 1.0714    | 0.5033          |         0.6395 | 7682.18   | nan               | -0.0615 |
| mal2025_dual_local_ready       | F2-XGB-Concat-CARMAInput | meanScore  | 1202 | nan       | nan             |         0.6565 |    6.7287 | 0.7696            | -0.4132 |
| mal2025_dual_local_ready       | C3-RAG-XGB-CARMAInput    | popularity | 1202 | 1.0706    | 0.4742          |         0.5966 | 7575.48   | nan               | -0.043  |
| mal2025_dual_local_ready       | C3-RAG-XGB-CARMAInput    | meanScore  | 1202 | nan       | nan             |         0.6295 |    6.5991 | 0.7887            | -0.3957 |
