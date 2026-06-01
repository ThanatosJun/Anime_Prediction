# Reference Baseline V2 Results

Comparison contract: all rows here use `data/fussion/fusion_meta_clean_{split}_v2.csv` through `data.meta_suffix: "_v2"`. Historical post2000/non-V2 runs should not be used in the formal V2 comparison table.

## Completed V2 Runs

| baseline_id                           | target     | model                             | feature_set                               |   n_train |   n_val |   n_test |   n_features |   test_MAE |   test_R2 |   test_Spearman_rho | status   | run_dir                        |
|:--------------------------------------|:-----------|:----------------------------------|:------------------------------------------|----------:|--------:|---------:|-------------:|-----------:|----------:|--------------------:|:---------|:-------------------------------|
| F0-Mean                               | popularity | mean                              | none                                      |     13321 |    2918 |     3087 |            0 | 14935.1    |   -0.1479 |              0      | ok       | .exp/baseline/results/v2_01    |
| F0-Mean                               | meanScore  | mean                              | none                                      |     13321 |    2918 |     3087 |            0 |    10.9115 |   -0.4631 |              0      | ok       | .exp/baseline/results/v2_01    |
| F0-Ridge-Meta                         | popularity | ridge                             | metadata                                  |     13321 |    2918 |     3087 |          151 | 12802.2    |   -0.402  |              0.8007 | ok       | .exp/baseline/results/v2_01    |
| F0-Ridge-Meta                         | meanScore  | ridge                             | metadata                                  |     13321 |    2918 |     3087 |          151 |     9.2029 |   -0.1266 |              0.4913 | ok       | .exp/baseline/results/v2_01    |
| F1-RF-Meta                            | popularity | random_forest                     | metadata                                  |     13321 |    2918 |     3087 |          151 |  8551.72   |    0.5865 |              0.842  | ok       | .exp/baseline/results/v2_01    |
| F1-RF-Meta                            | meanScore  | random_forest                     | metadata                                  |     13321 |    2918 |     3087 |          151 |     8.0179 |    0.1111 |              0.5759 | ok       | .exp/baseline/results/v2_01    |
| F1-GB-Meta                            | popularity | gradient_boosting                 | metadata                                  |     13321 |    2918 |     3087 |          151 |  9006.68   |    0.5004 |              0.8303 | ok       | .exp/baseline/results/v2_01    |
| F1-GB-Meta                            | meanScore  | gradient_boosting                 | metadata                                  |     13321 |    2918 |     3087 |          151 |     8.8758 |   -0.0518 |              0.5265 | ok       | .exp/baseline/results/v2_01    |
| F2-XGB-Concat                         | popularity | xgboost                           | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 |  9688.3    |    0.5108 |              0.8579 | ok       | .exp/baseline/results/v2_01    |
| F2-XGB-Concat                         | meanScore  | xgboost                           | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 |     8.5473 |   -0.0231 |              0.5102 | ok       | .exp/baseline/results/v2_01    |
| T2-XGB-TextEmb                        | popularity | xgboost                           | text_embedding                            |     12729 |    2637 |     2808 |          384 | 15204      |   -0.0621 |              0.6433 | ok       | .exp/baseline/results/v2_01    |
| T2-XGB-TextEmb                        | meanScore  | xgboost                           | text_embedding                            |     12729 |    2637 |     2808 |          384 |    10.6671 |   -0.4773 |              0.2262 | ok       | .exp/baseline/results/v2_01    |
| I1-XGB-ImageEmb                       | popularity | xgboost                           | image_embedding                           |     13321 |    2918 |     3087 |         1024 | 13928.2    |   -0.0039 |              0.582  | ok       | .exp/baseline/results/v2_01    |
| I1-XGB-ImageEmb                       | meanScore  | xgboost                           | image_embedding                           |     13321 |    2918 |     3087 |         1024 |     9.4161 |   -0.1603 |              0.2826 | ok       | .exp/baseline/results/v2_01    |
| C1-Armenta-ProjectInputProxy          | popularity | armenta_project_input_mlp         | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 | 11672.2    |    0.3794 |              0.8287 | ok       | .exp/baseline/results/v2_01_2  |
| C1-Armenta-ProjectInputProxy          | meanScore  | armenta_project_input_mlp         | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 |     9.2523 |   -0.1983 |              0.4307 | ok       | .exp/baseline/results/v2_01_2  |
| C2-ProjectInputCrossAttention         | popularity | project_input_cross_attention     | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 | 11044.7    |    0.4165 |              0.8473 | ok       | .exp/baseline/results/v2_01_3  |
| C2-ProjectInputCrossAttention         | meanScore  | project_input_cross_attention     | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 |     8.4384 |    0.0087 |              0.4837 | ok       | .exp/baseline/results/v2_01_3  |
| C2-ProjectInputRecurrentFusion        | popularity | project_input_recurrent_fusion    | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 | 11302.3    |    0.409  |              0.8479 | ok       | .exp/baseline/results/v2_01_4  |
| C2-ProjectInputRecurrentFusion        | meanScore  | project_input_recurrent_fusion    | metadata_text_image                       |     12729 |    2637 |     2808 |         1559 |     8.9897 |   -0.1324 |              0.4373 | ok       | .exp/baseline/results/v2_01_4  |
| C2-CTNN-Lite                          | popularity | cross_modal_transformer           | text_image                                |     12729 |    2637 |     2808 |         1408 | 14129.8    |    0.1798 |              0.7301 | ok       | .exp/baseline/results/v2_01_5  |
| C2-CTNN-Lite                          | meanScore  | cross_modal_transformer           | text_image                                |     12729 |    2637 |     2808 |         1408 |     9.5143 |   -0.2506 |              0.303  | ok       | .exp/baseline/results/v2_01_5  |
| C3-RAG-None-XGB                       | popularity | xgboost                           | metadata_text_image_rag_none              |     12729 |    2637 |     2808 |         1567 |  9584.36   |    0.5224 |              0.8584 | ok       | .exp/baseline/results/v2_01_6  |
| C3-RAG-None-XGB                       | meanScore  | xgboost                           | metadata_text_image_rag_none              |     12729 |    2637 |     2808 |         1567 |     8.4819 |   -0.0047 |              0.5175 | ok       | .exp/baseline/results/v2_01_6  |
| C3-RAG-Sparse-XGB                     | popularity | xgboost                           | metadata_text_image_rag_sparse            |     12729 |    2637 |     2808 |         1643 |  9564.72   |    0.584  |              0.8699 | ok       | .exp/baseline/results/v2_01_7  |
| C3-RAG-Sparse-XGB                     | meanScore  | xgboost                           | metadata_text_image_rag_sparse            |     12729 |    2637 |     2808 |         1643 |     8.3676 |    0.0317 |              0.5193 | ok       | .exp/baseline/results/v2_01_7  |
| C3-RAG-Dense-XGB                      | popularity | xgboost                           | metadata_text_image_rag_dense             |     12729 |    2637 |     2808 |         1643 | 10069      |    0.4742 |              0.8517 | ok       | .exp/baseline/results/v2_01_8  |
| C3-RAG-Dense-XGB                      | meanScore  | xgboost                           | metadata_text_image_rag_dense             |     12729 |    2637 |     2808 |         1643 |     8.4275 |    0.0034 |              0.5172 | ok       | .exp/baseline/results/v2_01_8  |
| C3-RAG-Hybrid-XGB                     | popularity | xgboost                           | metadata_text_image_rag_hybrid            |     12729 |    2637 |     2808 |         1643 | 10298.5    |    0.464  |              0.8487 | ok       | .exp/baseline/results/v2_01_9  |
| C3-RAG-Hybrid-XGB                     | meanScore  | xgboost                           | metadata_text_image_rag_hybrid            |     12729 |    2637 |     2808 |         1643 |     8.4433 |    0.01   |              0.5367 | ok       | .exp/baseline/results/v2_01_9  |
| C3-RAG-Selective-XGB                  | popularity | xgboost                           | metadata_text_image_rag_selective         |     12729 |    2637 |     2808 |         1643 |  9520.22   |    0.5901 |              0.8719 | ok       | .exp/baseline/results/v2_01_10 |
| C3-RAG-Selective-XGB                  | meanScore  | xgboost                           | metadata_text_image_rag_selective         |     12729 |    2637 |     2808 |         1643 |     8.309  |    0.0418 |              0.5234 | ok       | .exp/baseline/results/v2_01_10 |
| C3-ProjectInputSKAPPProxy-XGB         | popularity | xgboost                           | metadata_text_image_rag_skapp_proxy       |     12729 |    2637 |     2808 |         1652 | 10121.8    |    0.5174 |              0.8563 | ok       | .exp/baseline/results/v2_01_11 |
| C3-ProjectInputSKAPPProxy-XGB         | meanScore  | xgboost                           | metadata_text_image_rag_skapp_proxy       |     12729 |    2637 |     2808 |         1652 |     8.263  |    0.0472 |              0.5217 | ok       | .exp/baseline/results/v2_01_11 |
| C1-Armenta-ProjectInputReconstruction | popularity | armenta_project_input_mlp         | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 | 10501.5    |    0.3963 |              0.8149 | ok       | .exp/baseline/results/v2_01_12 |
| C1-Armenta-ProjectInputReconstruction | meanScore  | armenta_project_input_mlp         | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 |    10.5367 |   -0.4982 |              0.4447 | ok       | .exp/baseline/results/v2_01_12 |
| C2-ProjectInputCTNNReconstruction     | popularity | project_input_ctnn_reconstruction | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 | 10448.3    |    0.4189 |              0.8481 | ok       | .exp/baseline/results/v2_01_13 |
| C2-ProjectInputCTNNReconstruction     | meanScore  | project_input_ctnn_reconstruction | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 |     8.3066 |    0.0541 |              0.5269 | ok       | .exp/baseline/results/v2_01_13 |
| C3-ProjectInputSKAPPGraphProxy        | popularity | project_input_skapp_graph_proxy   | metadata_text_image_rag_skapp_graph_proxy |     12729 |    2637 |     2808 |        15695 | 11512      |    0.4046 |              0.8563 | ok       | .exp/baseline/results/v2_01_14 |
| C3-ProjectInputSKAPPGraphProxy        | meanScore  | project_input_skapp_graph_proxy   | metadata_text_image_rag_skapp_graph_proxy |     12729 |    2637 |     2808 |        15695 |     8.5741 |   -0.0355 |              0.4719 | ok       | .exp/baseline/results/v2_01_14 |

## Completed Baseline IDs

- `C1-Armenta-ProjectInputProxy`
- `C1-Armenta-ProjectInputReconstruction`
- `C2-CTNN-Lite`
- `C2-ProjectInputCTNNReconstruction`
- `C2-ProjectInputCrossAttention`
- `C2-ProjectInputRecurrentFusion`
- `C3-ProjectInputSKAPPGraphProxy`
- `C3-ProjectInputSKAPPProxy-XGB`
- `C3-RAG-Dense-XGB`
- `C3-RAG-Hybrid-XGB`
- `C3-RAG-None-XGB`
- `C3-RAG-Selective-XGB`
- `C3-RAG-Sparse-XGB`
- `F0-Mean`
- `F0-Ridge-Meta`
- `F1-GB-Meta`
- `F1-RF-Meta`
- `F2-XGB-Concat`
- `I1-XGB-ImageEmb`
- `T2-XGB-TextEmb`

## External Baseline Mainline Readiness

| baseline_id                           | target     | model                             | feature_set                               |   n_train |   n_val |   n_test |   n_features |   test_MAE |   test_R2 |   test_Spearman_rho | status   | run_dir                        |
|:--------------------------------------|:-----------|:----------------------------------|:------------------------------------------|----------:|--------:|---------:|-------------:|-----------:|----------:|--------------------:|:---------|:-------------------------------|
| C3-RAG-Selective-XGB                  | popularity | xgboost                           | metadata_text_image_rag_selective         |     12729 |    2637 |     2808 |         1643 |  9520.22   |    0.5901 |              0.8719 | ok       | .exp/baseline/results/v2_01_10 |
| C3-RAG-Selective-XGB                  | meanScore  | xgboost                           | metadata_text_image_rag_selective         |     12729 |    2637 |     2808 |         1643 |     8.309  |    0.0418 |              0.5234 | ok       | .exp/baseline/results/v2_01_10 |
| C3-ProjectInputSKAPPProxy-XGB         | popularity | xgboost                           | metadata_text_image_rag_skapp_proxy       |     12729 |    2637 |     2808 |         1652 | 10121.8    |    0.5174 |              0.8563 | ok       | .exp/baseline/results/v2_01_11 |
| C3-ProjectInputSKAPPProxy-XGB         | meanScore  | xgboost                           | metadata_text_image_rag_skapp_proxy       |     12729 |    2637 |     2808 |         1652 |     8.263  |    0.0472 |              0.5217 | ok       | .exp/baseline/results/v2_01_11 |
| C1-Armenta-ProjectInputReconstruction | popularity | armenta_project_input_mlp         | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 | 10501.5    |    0.3963 |              0.8149 | ok       | .exp/baseline/results/v2_01_12 |
| C1-Armenta-ProjectInputReconstruction | meanScore  | armenta_project_input_mlp         | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 |    10.5367 |   -0.4982 |              0.4447 | ok       | .exp/baseline/results/v2_01_12 |
| C2-ProjectInputCTNNReconstruction     | popularity | project_input_ctnn_reconstruction | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 | 10448.3    |    0.4189 |              0.8481 | ok       | .exp/baseline/results/v2_01_13 |
| C2-ProjectInputCTNNReconstruction     | meanScore  | project_input_ctnn_reconstruction | metadata_gpt2_resnet50_image              |     13321 |    2918 |     3087 |         5017 |     8.3066 |    0.0541 |              0.5269 | ok       | .exp/baseline/results/v2_01_13 |
| C3-ProjectInputSKAPPGraphProxy        | popularity | project_input_skapp_graph_proxy   | metadata_text_image_rag_skapp_graph_proxy |     12729 |    2637 |     2808 |        15695 | 11512      |    0.4046 |              0.8563 | ok       | .exp/baseline/results/v2_01_14 |
| C3-ProjectInputSKAPPGraphProxy        | meanScore  | project_input_skapp_graph_proxy   | metadata_text_image_rag_skapp_graph_proxy |     12729 |    2637 |     2808 |        15695 |     8.5741 |   -0.0355 |              0.4719 | ok       | .exp/baseline/results/v2_01_14 |

## Still Pending Or Optional

| baseline_id                                 | reason                                                                                                                |
|:--------------------------------------------|:----------------------------------------------------------------------------------------------------------------------|
| C1-Armenta-Figure2Reconstruction            | optional side reconstruction; needs V2 character description/portrait artifacts and is not the project-input mainline |
| C2-ProjectInputCTNNDualVisualReconstruction | optional diagnostic; needs decision on whether to use project Swin/src_2 image stream for dual-visual comparison      |
| C3-ProjectInputSKAPPFull                    | separate full SKAPP-style runner; V2 not rerun and previous full run remains a diagnostic with weak performance       |

## Artifact Locations

- Combined CSV: `reports/baselines/reference_baseline_v2_results.csv`
- Per-run raw outputs: `.exp/baseline/results/v2_01*`
- V2 GPT-2 synopsis features: `.exp/baseline/text_features/gpt2_v2`
- V2 ResNet-50 cover/banner features: `.exp/baseline/image_features/resnet50_v2`
- V2 RAG features: `.exp/baseline/rag_features_v2`
