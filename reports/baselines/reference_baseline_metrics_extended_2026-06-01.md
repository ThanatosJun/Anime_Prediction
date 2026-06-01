# Extended Reference Baseline Metrics 2026-06-01

Generated from existing `test_predictions.csv` files. `test_R2_raw` is original-scale R2; `test_log_R2` is only for popularity.


## popularity

| baseline_id                           | source     |   n_test |   test_MAE |   test_log_MAE |   test_log_R2 |   test_factor_acc_2x |   test_Spearman_rho |   test_R2_raw |
|:--------------------------------------|:-----------|---------:|-----------:|---------------:|--------------:|---------------------:|--------------------:|--------------:|
| F0-Mean                               | v2         |     3087 | 14935.0718 |         1.9884 |       -0.0000 |               0.1950 |              0.0000 |       -0.1479 |
| F0-Ridge-Meta                         | v2         |     3087 | 12802.2336 |         1.0551 |        0.6855 |               0.3962 |              0.8007 |       -0.4020 |
| F1-RF-Meta                            | v2         |     3087 |  8551.7168 |         0.8938 |        0.7554 |               0.4891 |              0.8420 |        0.5865 |
| F1-GB-Meta                            | v2         |     3087 |  9006.6811 |         0.8958 |        0.7568 |               0.4911 |              0.8303 |        0.5004 |
| T2-XGB-TextEmb                        | v2         |     2808 | 15203.9965 |         1.5077 |        0.3920 |               0.2817 |              0.6433 |       -0.0621 |
| I1-XGB-ImageEmb                       | v2_highres |     3087 | 12100.8365 |         1.3590 |        0.4804 |               0.3058 |              0.7257 |        0.2096 |
| F2-XGB-Concat                         | v2_highres |     2808 |  9539.7047 |         0.8828 |        0.7760 |               0.4708 |              0.8650 |        0.5515 |
| C1-Armenta-ProjectInputProxy          | v2_highres |     2808 | 11095.5952 |         0.9538 |        0.7310 |               0.4626 |              0.8418 |        0.4332 |
| C1-Armenta-ProjectInputReconstruction | v2         |     3087 | 10501.5398 |         1.0244 |        0.6953 |               0.4137 |              0.8149 |        0.3963 |
| C2-ProjectInputCrossAttention         | v2_highres |     2808 | 11193.8393 |         0.9236 |        0.7525 |               0.4601 |              0.8647 |        0.4478 |
| C2-ProjectInputRecurrentFusion        | v2_highres |     2808 | 10469.8080 |         0.9151 |        0.7607 |               0.4605 |              0.8673 |        0.4748 |
| C2-ProjectInputCTNNReconstruction     | v2         |     3087 | 10448.2886 |         0.9725 |        0.7280 |               0.4321 |              0.8481 |        0.4189 |
| C3-RAG-Selective-XGB                  | v2_highres |     2808 |  9256.1195 |         0.9266 |        0.7537 |               0.4665 |              0.8719 |        0.6182 |
| C3-ProjectInputSKAPPProxy-XGB         | v2_highres |     2808 | 10075.2543 |         0.9363 |        0.7521 |               0.4548 |              0.8633 |        0.5430 |
| C3-ProjectInputSKAPPGraphProxy        | v2_highres |     2808 | 11254.5741 |         0.9305 |        0.7575 |               0.4494 |              0.8737 |        0.3862 |
| C3-SourceExact-Staged-K64             | c3_source_exact_k64 | 3087 | 99140.0794 | 3.4361 | -2.1272 | 0.0901 | 0.3170 | -15.0432 |



## meanScore

| baseline_id                           | source     |   n_test |   test_MAE |   test_acc_within_10pt |   test_Spearman_rho |   test_R2_raw |
|:--------------------------------------|:-----------|---------:|-----------:|-----------------------:|--------------------:|--------------:|
| F0-Mean                               | v2         |     3087 |    10.9115 |                 0.5083 |              0.0000 |       -0.4631 |
| F0-Ridge-Meta                         | v2         |     3087 |     9.2029 |                 0.6054 |              0.4913 |       -0.1266 |
| F1-RF-Meta                            | v2         |     3087 |     8.0179 |                 0.6761 |              0.5759 |        0.1111 |
| F1-GB-Meta                            | v2         |     3087 |     8.8758 |                 0.6213 |              0.5265 |       -0.0518 |
| T2-XGB-TextEmb                        | v2         |     2808 |    10.6671 |                 0.5082 |              0.2262 |       -0.4773 |
| I1-XGB-ImageEmb                       | v2_highres |     3087 |     8.6345 |                 0.6391 |              0.4180 |       -0.0103 |
| F2-XGB-Concat                         | v2_highres |     2808 |     8.2031 |                 0.6556 |              0.5530 |        0.0562 |
| C1-Armenta-ProjectInputProxy          | v2_highres |     2808 |     8.4901 |                 0.6503 |              0.4808 |       -0.0187 |
| C1-Armenta-ProjectInputReconstruction | v2         |     3087 |    10.5367 |                 0.5507 |              0.4447 |       -0.4982 |
| C2-ProjectInputCrossAttention         | v2_highres |     2808 |     8.0630 |                 0.6863 |              0.5044 |        0.0586 |
| C2-ProjectInputRecurrentFusion        | v2_highres |     2808 |     8.3908 |                 0.6720 |              0.4895 |        0.0037 |
| C2-ProjectInputCTNNReconstruction     | v2         |     3087 |     8.3066 |                 0.6631 |              0.5269 |        0.0541 |
| C3-RAG-Selective-XGB                  | v2_highres |     2808 |     8.0901 |                 0.6667 |              0.5561 |        0.0884 |
| C3-ProjectInputSKAPPProxy-XGB         | v2_highres |     2808 |     7.8582 |                 0.6912 |              0.5634 |        0.1274 |
| C3-ProjectInputSKAPPGraphProxy        | v2_highres |     2808 |     8.1218 |                 0.6806 |              0.5169 |        0.0671 |
| C3-SourceExact-Staged-K64             | c3_source_exact_k64 | 3087 | 19.8518 | 0.3061 | 0.1155 | -4.2271 |

