# CARMA Tensor-aligned Baseline Evaluation

Baselines in this table flatten the actual tensors returned by `src_2.fussion_training.dataset.AnimeDataset`.
No text/image/RAG embeddings are regenerated.

| split                                           | model                          | target     |    n | log_MAE   | factor_acc_2x   |   Spearman_rho |       MAE | acc_within_10pt   | R2      |
|:------------------------------------------------|:-------------------------------|:-----------|-----:|:----------|:----------------|---------------:|----------:|:------------------|:--------|
| test                                            | F1-RF-Meta-CARMATensor         | popularity | 3087 | 0.9731    | 0.4655          |         0.8326 | 9728.65   |                   | 0.5181  |
| test                                            | F1-RF-Meta-CARMATensor         | meanScore  | 3087 |           |                 |         0.5171 |    8.8547 | 0.6281            | -0.0438 |
| test                                            | F2-XGB-Concat-CARMATensor      | popularity | 3087 | 0.8799    | 0.4885          |         0.8578 | 8872.96   |                   | 0.6032  |
| test                                            | F2-XGB-Concat-CARMATensor      | meanScore  | 3087 |           |                 |         0.5591 |    8.6542 | 0.645             | 0.0142  |
| test                                            | C1-Armenta-CARMATensor         | popularity | 3087 | 0.921     | 0.4704          |         0.852  | 9439.97   |                   | 0.4949  |
| test                                            | C1-Armenta-CARMATensor         | meanScore  | 3087 |           |                 |         0.5215 |    8.184  | 0.6728            | 0.0711  |
| test                                            | C2-CrossAttention-CARMATensor  | popularity | 3087 | 0.9115    | 0.4739          |         0.8537 | 9903.44   |                   | 0.5174  |
| test                                            | C2-CrossAttention-CARMATensor  | meanScore  | 3087 |           |                 |         0.5354 |    8.4217 | 0.6673            | 0.0286  |
| test                                            | C2-RecurrentFusion-CARMATensor | popularity | 3087 | 0.8915    | 0.4817          |         0.8614 | 9056.83   |                   | 0.5829  |
| test                                            | C2-RecurrentFusion-CARMATensor | meanScore  | 3087 |           |                 |         0.5294 |    9.2659 | 0.6132            | -0.1395 |
| test                                            | C3-RAG-XGB-CARMATensor         | popularity | 3087 | 0.8947    | 0.4917          |         0.852  | 9032.59   |                   | 0.5908  |
| test                                            | C3-RAG-XGB-CARMATensor         | meanScore  | 3087 |           |                 |         0.5432 |    8.7747 | 0.6323            | -0.0107 |
| mal2025_popularity_local_ready                  | F1-RF-Meta-CARMATensor         | popularity | 3765 | 1.0015    | 0.4776          |         0.4572 | 3558.3    |                   | -0.0258 |
| mal2025_dual_local_ready                        | F1-RF-Meta-CARMATensor         | popularity | 1202 | 1.4387    | 0.3236          |         0.5144 | 7916.62   |                   | -0.0794 |
| mal2025_popularity_local_ready_yolo             | F1-RF-Meta-CARMATensor         | popularity | 3765 | 1.0015    | 0.4776          |         0.4572 | 3558.3    |                   | -0.0258 |
| mal2025_dual_local_ready_yolo                   | F1-RF-Meta-CARMATensor         | popularity | 1202 | 1.4387    | 0.3236          |         0.5144 | 7916.62   |                   | -0.0794 |
| mal2025_dual_local_ready                        | F1-RF-Meta-CARMATensor         | meanScore  | 1202 |           |                 |         0.5629 |    9.121  | 0.6106            | -1.3687 |
| mal2025_dual_local_ready_yolo                   | F1-RF-Meta-CARMATensor         | meanScore  | 1202 |           |                 |         0.5629 |    9.121  | 0.6106            | -1.3687 |
| mal2025_popularity_local_ready                  | F2-XGB-Concat-CARMATensor      | popularity | 3765 | 1.0294    | 0.5017          |         0.524  | 3598.18   |                   | -0.0335 |
| mal2025_dual_local_ready                        | F2-XGB-Concat-CARMATensor      | popularity | 1202 | 1.6172    | 0.2787          |         0.5789 | 8061.09   |                   | -0.0864 |
| mal2025_popularity_local_ready_yolo             | F2-XGB-Concat-CARMATensor      | popularity | 3765 | 0.9932    | 0.5147          |         0.5297 | 3573.22   |                   | -0.0288 |
| mal2025_dual_local_ready_yolo                   | F2-XGB-Concat-CARMATensor      | popularity | 1202 | 1.5784    | 0.2704          |         0.6379 | 8013.82   |                   | -0.0814 |
| mal2025_dual_local_ready                        | F2-XGB-Concat-CARMATensor      | meanScore  | 1202 |           |                 |         0.6061 |    9.76   | 0.5649            | -1.5992 |
| mal2025_dual_local_ready_yolo                   | F2-XGB-Concat-CARMATensor      | meanScore  | 1202 |           |                 |         0.6449 |    9.9334 | 0.5424            | -1.6242 |
| mal2025_popularity_local_ready                  | C1-Armenta-CARMATensor         | popularity | 3765 | 1.1704    | 0.4345          |         0.2798 | 3571.29   |                   | -0.0103 |
| mal2025_dual_local_ready                        | C1-Armenta-CARMATensor         | popularity | 1202 | 1.9602    | 0.1298          |         0.4572 | 8015.96   |                   | -0.0595 |
| mal2025_popularity_local_ready_yolo             | C1-Armenta-CARMATensor         | popularity | 3765 | 1.024     | 0.4133          |         0.4329 | 3482.73   |                   | 0.0198  |
| mal2025_dual_local_ready_yolo                   | C1-Armenta-CARMATensor         | popularity | 1202 | 1.369     | 0.2937          |         0.5781 | 7691.62   |                   | -0.0292 |
| mal2025_dual_local_ready                        | C1-Armenta-CARMATensor         | meanScore  | 1202 |           |                 |         0.5826 |    9.9122 | 0.5316            | -1.7643 |
| mal2025_dual_local_ready_yolo                   | C1-Armenta-CARMATensor         | meanScore  | 1202 |           |                 |         0.5607 |    8.2518 | 0.6631            | -1.0714 |
| mal2025_popularity_local_ready                  | C2-CrossAttention-CARMATensor  | popularity | 3765 | 1.0769    | 0.4677          |         0.3576 | 3601.28   |                   | -0.03   |
| mal2025_dual_local_ready                        | C2-CrossAttention-CARMATensor  | popularity | 1202 | 1.7178    | 0.2047          |         0.5607 | 8057.35   |                   | -0.0818 |
| mal2025_popularity_local_ready_yolo             | C2-CrossAttention-CARMATensor  | popularity | 3765 | 0.9562    | 0.4624          |         0.4961 | 3471.65   |                   | 0.0042  |
| mal2025_dual_local_ready_yolo                   | C2-CrossAttention-CARMATensor  | popularity | 1202 | 1.2208    | 0.3735          |         0.6687 | 7646.56   |                   | -0.0446 |
| mal2025_dual_local_ready                        | C2-CrossAttention-CARMATensor  | meanScore  | 1202 |           |                 |         0.5491 |    8.3423 | 0.6622            | -1.0387 |
| mal2025_dual_local_ready_yolo                   | C2-CrossAttention-CARMATensor  | meanScore  | 1202 |           |                 |         0.5762 |    8.0652 | 0.6714            | -0.9664 |
| mal2025_popularity_local_ready                  | C2-RecurrentFusion-CARMATensor | popularity | 3765 | 1.2692    | 0.3073          |         0.0766 | 3632.64   |                   | -0.0221 |
| mal2025_dual_local_ready                        | C2-RecurrentFusion-CARMATensor | popularity | 1202 | 1.8289    | 0.1814          |         0.4585 | 8037.26   |                   | -0.0731 |
| mal2025_popularity_local_ready_yolo             | C2-RecurrentFusion-CARMATensor | popularity | 3765 | 1.0549    | 0.3777          |         0.3963 | 3505.4    |                   | 0.007   |
| mal2025_dual_local_ready_yolo                   | C2-RecurrentFusion-CARMATensor | popularity | 1202 | 1.3365    | 0.3111          |         0.6338 | 7707.47   |                   | -0.0432 |
| mal2025_dual_local_ready                        | C2-RecurrentFusion-CARMATensor | meanScore  | 1202 |           |                 |         0.6113 |    9.2142 | 0.5865            | -1.4073 |
| mal2025_dual_local_ready_yolo                   | C2-RecurrentFusion-CARMATensor | meanScore  | 1202 |           |                 |         0.6459 |    8.221  | 0.6656            | -1.013  |
| mal2025_popularity_local_ready                  | C3-RAG-XGB-CARMATensor         | popularity | 3765 | 1.0383    | 0.4497          |         0.4392 | 3586.92   |                   | -0.0296 |
| mal2025_dual_local_ready                        | C3-RAG-XGB-CARMATensor         | popularity | 1202 | 1.5637    | 0.2704          |         0.6211 | 8012.37   |                   | -0.0822 |
| mal2025_popularity_local_ready_yolo             | C3-RAG-XGB-CARMATensor         | popularity | 3765 | 0.9974    | 0.4869          |         0.5097 | 3563.79   |                   | -0.0261 |
| mal2025_dual_local_ready_yolo                   | C3-RAG-XGB-CARMATensor         | popularity | 1202 | 1.5278    | 0.2629          |         0.6591 | 7974.52   |                   | -0.0784 |
| mal2025_dual_local_ready                        | C3-RAG-XGB-CARMATensor         | meanScore  | 1202 |           |                 |         0.5989 |    9.3699 | 0.5932            | -1.43   |
| mal2025_dual_local_ready_yolo                   | C3-RAG-XGB-CARMATensor         | meanScore  | 1202 |           |                 |         0.6186 |    9.6667 | 0.5616            | -1.5387 |
| mal2025_popularity_local_ready_yolo_coverbanner | F1-RF-Meta-CARMATensor         | popularity | 3765 | 1.0015    | 0.4776          |         0.4572 | 3558.3    |                   | -0.0258 |
| mal2025_dual_local_ready_yolo_coverbanner       | F1-RF-Meta-CARMATensor         | popularity | 1202 | 1.4387    | 0.3236          |         0.5144 | 7916.62   |                   | -0.0794 |
| mal2025_dual_local_ready_yolo_coverbanner       | F1-RF-Meta-CARMATensor         | meanScore  | 1202 |           |                 |         0.5629 |    9.121  | 0.6106            | -1.3687 |
| mal2025_popularity_local_ready_yolo_coverbanner | F2-XGB-Concat-CARMATensor      | popularity | 3765 | 0.9586    | 0.4834          |         0.5172 | 3526.01   |                   | -0.0188 |
| mal2025_dual_local_ready_yolo_coverbanner       | F2-XGB-Concat-CARMATensor      | popularity | 1202 | 1.3277    | 0.3669          |         0.607  | 7831.2    |                   | -0.0703 |
| mal2025_dual_local_ready_yolo_coverbanner       | F2-XGB-Concat-CARMATensor      | meanScore  | 1202 |           |                 |         0.6275 |    7.3707 | 0.7413            | -0.6609 |
| mal2025_popularity_local_ready_yolo_coverbanner | C1-Armenta-CARMATensor         | popularity | 3765 | 1.1816    | 0.328           |         0.347  | 3603.17   |                   | 0.0587  |
| mal2025_dual_local_ready_yolo_coverbanner       | C1-Armenta-CARMATensor         | popularity | 1202 | 1.1786    | 0.3735          |         0.51   | 7535.17   |                   | 0.0216  |
| mal2025_dual_local_ready_yolo_coverbanner       | C1-Armenta-CARMATensor         | meanScore  | 1202 |           |                 |         0.5535 |    6.9666 | 0.7654            | -0.5426 |
| mal2025_popularity_local_ready_yolo_coverbanner | C2-CrossAttention-CARMATensor  | popularity | 3765 | 1.0008    | 0.4181          |         0.4608 | 3463.9    |                   | 0.021   |
| mal2025_dual_local_ready_yolo_coverbanner       | C2-CrossAttention-CARMATensor  | popularity | 1202 | 1.1343    | 0.3894          |         0.6637 | 7531.74   |                   | -0.026  |
| mal2025_dual_local_ready_yolo_coverbanner       | C2-CrossAttention-CARMATensor  | meanScore  | 1202 |           |                 |         0.5711 |    7.1641 | 0.7429            | -0.5995 |
| mal2025_popularity_local_ready_yolo_coverbanner | C2-RecurrentFusion-CARMATensor | popularity | 3765 | 1.1163    | 0.3461          |         0.3846 | 3533.78   |                   | 0.028   |
| mal2025_dual_local_ready_yolo_coverbanner       | C2-RecurrentFusion-CARMATensor | popularity | 1202 | 1.1803    | 0.3777          |         0.5911 | 7571.26   |                   | -0.0168 |
| mal2025_dual_local_ready_yolo_coverbanner       | C2-RecurrentFusion-CARMATensor | meanScore  | 1202 |           |                 |         0.6465 |    7.3125 | 0.7396            | -0.647  |
| mal2025_popularity_local_ready_yolo_coverbanner | C3-RAG-XGB-CARMATensor         | popularity | 3765 | 0.9726    | 0.4515          |         0.505  | 3518.29   |                   | -0.0155 |
| mal2025_dual_local_ready_yolo_coverbanner       | C3-RAG-XGB-CARMATensor         | popularity | 1202 | 1.2893    | 0.3844          |         0.6343 | 7796.24   |                   | -0.0668 |
| mal2025_popularity_local_ready                  | CARMA-Run02                    | popularity | 3765 | 1.012     | 0.4656          |         0.4709 | 3518.33   |                   |         |
| mal2025_dual_local_ready                        | CARMA-Run02                    | popularity | 1202 | 1.391     | 0.3344          |         0.5495 | 7750.94   |                   |         |
| mal2025_dual_local_ready                        | CARMA-Run02                    | meanScore  | 1202 |           |                 |         0.6079 |    7.5086 | 0.7488            | -1.0659 |
| mal2025_popularity_local_ready                  | CARMA-Run22                    | popularity | 3765 | 1.0359    | 0.4234          |         0.4998 | 3588.06   |                   |         |
| mal2025_dual_local_ready                        | CARMA-Run22                    | popularity | 1202 | 1.1707    | 0.426           |         0.5647 | 7732.83   |                   |         |
| mal2025_dual_local_ready                        | CARMA-Run22                    | meanScore  | 1202 |           |                 |         0.577  |    6.3363 | 0.7945            | -0.3253 |
| mal2025_popularity_local_ready_yolo             | CARMA-Run22                    | popularity | 3765 | 1.0143    | 0.4356          |         0.5213 | 3587.32   |                   |         |
| mal2025_dual_local_ready_yolo                   | CARMA-Run22                    | popularity | 1202 | 1.2001    | 0.406           |         0.6073 | 7796.84   |                   |         |
| mal2025_dual_local_ready_yolo                   | CARMA-Run22                    | meanScore  | 1202 |           |                 |         0.5999 |    6.4919 | 0.7829            | -0.3573 |
| mal2025_popularity_local_ready_yolo_coverbanner | CARMA-Run22                    | popularity | 3765 | 1.0174    | 0.4353          |         0.5166 | 3584.24   |                   |         |
| mal2025_dual_local_ready_yolo_coverbanner       | CARMA-Run22                    | popularity | 1202 | 1.188     | 0.4143          |         0.5955 | 7773.6    |                   |         |
| mal2025_dual_local_ready_yolo_coverbanner       | CARMA-Run22                    | meanScore  | 1202 |           |                 |         0.5921 |    6.4098 | 0.7879            | -0.3374 |
| mal2025_dual_local_ready_yolo_coverbanner       | C3-RAG-XGB-CARMATensor         | meanScore  | 1202 |           |                 |         0.6109 |    7.366  | 0.7371            | -0.6635 |
