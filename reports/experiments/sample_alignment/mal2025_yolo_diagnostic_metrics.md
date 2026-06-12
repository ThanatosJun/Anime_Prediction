# MAL 2025 Cover-derived YOLO Diagnostic Metrics

Pop-only YOLO-filled rows:

| Model | n | pop_log_mae | pop_factor_acc_2x | pop_spearman |
|---|---:|---:|---:|---:|
| CARMA-Run22 | 3765 | 1.0143 | 0.4356 | 0.5213 |
| F1-RF-Meta-CARMATensor | 3765 | 1.0015 | 0.4776 | 0.4572 |
| F2-XGB-Concat-CARMATensor | 3765 | 0.9932 | 0.5147 | 0.5297 |
| C1-Armenta-CARMATensor | 3765 | 1.0240 | 0.4133 | 0.4329 |
| C2-CrossAttention-CARMATensor | 3765 | 0.9562 | 0.4624 | 0.4961 |
| C2-RecurrentFusion-CARMATensor | 3765 | 1.0549 | 0.3777 | 0.3963 |
| C3-RAG-XGB-CARMATensor | 3765 | 0.9974 | 0.4869 | 0.5097 |

Dual-target YOLO-filled rows:

| Model | n | pop_log_mae | pop_factor_acc_2x | pop_spearman | score_mae | score_acc_within_10pt | score_spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| CARMA-Run22 | 1202 | 1.2001 | 0.4060 | 0.6073 | 6.4919 | 0.7829 | 0.5999 |
| F1-RF-Meta-CARMATensor | 1202 | 1.4387 | 0.3236 | 0.5144 | 9.1210 | 0.6106 | 0.5629 |
| F2-XGB-Concat-CARMATensor | 1202 | 1.5784 | 0.2704 | 0.6379 | 9.9334 | 0.5424 | 0.6449 |
| C1-Armenta-CARMATensor | 1202 | 1.3690 | 0.2937 | 0.5781 | 8.2518 | 0.6631 | 0.5607 |
| C2-CrossAttention-CARMATensor | 1202 | 1.2208 | 0.3735 | 0.6687 | 8.0652 | 0.6714 | 0.5762 |
| C2-RecurrentFusion-CARMATensor | 1202 | 1.3365 | 0.3111 | 0.6338 | 8.2210 | 0.6656 | 0.6459 |
| C3-RAG-XGB-CARMATensor | 1202 | 1.5278 | 0.2629 | 0.6591 | 9.6667 | 0.5616 | 0.6186 |

Notes:

- `cover_yolo` fills the `yolo_*` branch from MAL cover-image crops only.
- Banner remains unavailable for both variants.
- Ranking metrics improve for CARMA and several baselines after adding YOLO,
  but CARMA's dual-target absolute-error metrics slightly worsen.
