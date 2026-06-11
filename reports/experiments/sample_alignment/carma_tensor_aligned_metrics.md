# CARMA Tensor-aligned Baseline Evaluation

Baselines in this table flatten the actual tensors returned by `src_2.fussion_training.dataset.AnimeDataset`.
No text/image/RAG embeddings are regenerated.

| split                          | model                     | target     |    n | log_MAE   | factor_acc_2x   |   Spearman_rho |       MAE | acc_within_10pt   | R2      |
|:-------------------------------|:--------------------------|:-----------|-----:|:----------|:----------------|---------------:|----------:|:------------------|:--------|
| mal2025_popularity_local_ready | CARMA-Run02               | popularity | 3765 | 1.012     | 0.4656          |         0.4709 | 3518.33   |                   |         |
| mal2025_dual_local_ready       | CARMA-Run02               | popularity | 1202 | 1.391     | 0.3344          |         0.5495 | 7750.94   |                   |         |
| mal2025_dual_local_ready       | CARMA-Run02               | meanScore  | 1202 |           |                 |         0.6079 |    7.5086 | 0.7488            | -1.0659 |
| mal2025_popularity_local_ready | CARMA-Run22               | popularity | 3765 | 1.0359    | 0.4234          |         0.4998 | 3588.06   |                   |         |
| mal2025_dual_local_ready       | CARMA-Run22               | popularity | 1202 | 1.1707    | 0.426           |         0.5647 | 7732.83   |                   |         |
| mal2025_dual_local_ready       | CARMA-Run22               | meanScore  | 1202 |           |                 |         0.577  |    6.3363 | 0.7945            | -0.3253 |
| test                           | F1-RF-Meta-CARMATensor    | popularity | 3087 | 0.9731    | 0.4655          |         0.8326 | 9728.65   |                   | 0.5181  |
| mal2025_popularity_local_ready | F1-RF-Meta-CARMATensor    | popularity | 3765 | 1.0015    | 0.4776          |         0.4572 | 3558.3    |                   | -0.0258 |
| mal2025_dual_local_ready       | F1-RF-Meta-CARMATensor    | popularity | 1202 | 1.4387    | 0.3236          |         0.5144 | 7916.62   |                   | -0.0794 |
| test                           | F1-RF-Meta-CARMATensor    | meanScore  | 3087 |           |                 |         0.5171 |    8.8547 | 0.6281            | -0.0438 |
| mal2025_dual_local_ready       | F1-RF-Meta-CARMATensor    | meanScore  | 1202 |           |                 |         0.5629 |    9.121  | 0.6106            | -1.3687 |
| test                           | F2-XGB-Concat-CARMATensor | popularity | 3087 | 0.8799    | 0.4885          |         0.8578 | 8872.96   |                   | 0.6032  |
| mal2025_popularity_local_ready | F2-XGB-Concat-CARMATensor | popularity | 3765 | 1.0294    | 0.5017          |         0.524  | 3598.18   |                   | -0.0335 |
| mal2025_dual_local_ready       | F2-XGB-Concat-CARMATensor | popularity | 1202 | 1.6172    | 0.2787          |         0.5789 | 8061.09   |                   | -0.0864 |
| test                           | F2-XGB-Concat-CARMATensor | meanScore  | 3087 |           |                 |         0.5591 |    8.6542 | 0.645             | 0.0142  |
| mal2025_dual_local_ready       | F2-XGB-Concat-CARMATensor | meanScore  | 1202 |           |                 |         0.6061 |    9.76   | 0.5649            | -1.5992 |
| test                           | C3-RAG-XGB-CARMATensor    | popularity | 3087 | 0.8947    | 0.4917          |         0.852  | 9032.59   |                   | 0.5908  |
| mal2025_popularity_local_ready | C3-RAG-XGB-CARMATensor    | popularity | 3765 | 1.0383    | 0.4497          |         0.4392 | 3586.92   |                   | -0.0296 |
| mal2025_dual_local_ready       | C3-RAG-XGB-CARMATensor    | popularity | 1202 | 1.5637    | 0.2704          |         0.6211 | 8012.37   |                   | -0.0822 |
| test                           | C3-RAG-XGB-CARMATensor    | meanScore  | 3087 |           |                 |         0.5432 |    8.7747 | 0.6323            | -0.0107 |
| mal2025_dual_local_ready       | C3-RAG-XGB-CARMATensor    | meanScore  | 1202 |           |                 |         0.5989 |    9.3699 | 0.5932            | -1.43   |
