# Sample-aligned baseline evaluation

Policies:

- `strict_common`: uses only rows with all requested artifacts.
- `zero_fallback_full`: keeps metadata split rows and zero-fills missing artifact vectors.

| policy             | baseline_id          | target     |   n_test |   test_log_MAE |   test_factor_acc_2x |   test_Spearman_rho |   test_MAE |   test_acc_within_10pt |   test_R2 |
|:-------------------|:---------------------|:-----------|---------:|---------------:|---------------------:|--------------------:|-----------:|-----------------------:|----------:|
| strict_common      | F1-RF-Meta           | popularity |     3087 |         0.8938 |               0.4895 |              0.842  |  8547.47   |               nan      |    0.5876 |
| strict_common      | F1-RF-Meta           | meanScore  |     3087 |       nan      |             nan      |              0.5776 |     8.004  |                 0.6783 |    0.114  |
| strict_common      | F2-XGB-Concat        | popularity |     2808 |         0.9021 |               0.4772 |              0.8579 |  9688.3    |               nan      |    0.5108 |
| strict_common      | F2-XGB-Concat        | meanScore  |     2808 |       nan      |             nan      |              0.5102 |     8.5473 |                 0.6375 |   -0.0231 |
| strict_common      | C3-RAG-Selective-XGB | popularity |     2808 |         0.9399 |               0.4594 |              0.8727 |  9517.1    |               nan      |    0.5838 |
| strict_common      | C3-RAG-Selective-XGB | meanScore  |     2808 |       nan      |             nan      |              0.5243 |     8.3048 |                 0.656  |    0.0426 |
| zero_fallback_full | F1-RF-Meta           | popularity |     3087 |         0.8938 |               0.4895 |              0.842  |  8547.47   |               nan      |    0.5876 |
| zero_fallback_full | F1-RF-Meta           | meanScore  |     3087 |       nan      |             nan      |              0.5776 |     8.004  |                 0.6783 |    0.114  |
| zero_fallback_full | F2-XGB-Concat        | popularity |     3087 |         0.8807 |               0.4862 |              0.8465 |  8803.09   |               nan      |    0.5236 |
| zero_fallback_full | F2-XGB-Concat        | meanScore  |     3087 |       nan      |             nan      |              0.5453 |     8.6076 |                 0.6339 |    0.0086 |
| zero_fallback_full | C3-RAG-Selective-XGB | popularity |     3087 |         0.9226 |               0.4697 |              0.8605 |  8802.87   |               nan      |    0.5815 |
| zero_fallback_full | C3-RAG-Selective-XGB | meanScore  |     3087 |       nan      |             nan      |              0.5405 |     8.5021 |                 0.6417 |    0.0427 |
