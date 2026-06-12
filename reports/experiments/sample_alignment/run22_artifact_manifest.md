# Run22 Artifact Manifest

This repository keeps large model artifacts out of Git. To reproduce the Run22
external MAL evaluation, restore the final project artifact from the course
release first.

## Release Asset

- Release: https://github.com/ThanatosJun/Anime_Prediction/releases/tag/A7
- Asset: `final_project.zip`
- SHA256: `4e537ff84978e29ea9fcfbee18bdc8e993a2ccbb0f101c62246e5beb99e20ee9`
- Restore location: repository root
- Expected restored directory: `final_project/`

After extraction, these files must exist:

| Path | Purpose |
|---|---|
| `final_project/runs/22/popularity/best_model.pt` | Run22 popularity checkpoint |
| `final_project/runs/22/popularity/target_scaler.json` | Popularity target scaler |
| `final_project/runs/22/popularity/pred_test.csv` | Internal test predictions |
| `final_project/runs/22/meanScore/best_model.pt` | Run22 meanScore checkpoint |
| `final_project/runs/22/meanScore/target_scaler.json` | meanScore target scaler |
| `final_project/runs/22/meanScore/pred_test.csv` | Internal test predictions |
| `src_2/component_image/model-image/best/config.json` | Swin image encoder config |
| `src_2/component_image/model-image/best/model.safetensors` | Swin image encoder weights for YOLO diagnostic |

## Key File Checksums

| Path | SHA256 |
|---|---|
| `final_project/runs/22/popularity/best_model.pt` | `cc1543ee785f4c535eb8dac2ffe99281b34d9ba4e0c065fb3b8651467898a538` |
| `final_project/runs/22/popularity/target_scaler.json` | `741c6027999a686a8147d0d761d64e924f6f2940faa230c1bd421376406e26f5` |
| `final_project/runs/22/popularity/pred_test.csv` | `2f8e94d8b19349c68f4f5067d5cb93f2fb8a41b0da5088f98abd755720b45ee8` |
| `final_project/runs/22/meanScore/best_model.pt` | `956a3d75aa0d234443753a81affe51e4674c03c265955129de5fc32b490e55ec` |
| `final_project/runs/22/meanScore/target_scaler.json` | `05a93cb02a401412743b54e8e9b8ef4048931014c4bd80eb40b69b348bfec339` |
| `final_project/runs/22/meanScore/pred_test.csv` | `2304039a8d7da915a28004a47484bb4e89d44679fa6224daaf0bc743079610a1` |
| `src_2/component_image/model-image/best/model.safetensors` | `86bd98d5dbf8022765b67b0d69c3a78ebe3d69b051c05ef87e156e310f1bc020` |

## Reproduction Commands

```bash
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready
python scripts/external/build_mal2025_yolo_image_embeddings.py --splits mal2025_popularity_local_ready mal2025_dual_local_ready --suffix yolo --batch-size 64
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready_yolo --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready_yolo
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready_yolo --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready_yolo
python scripts/external/build_mal2025_cover_banner_proxy.py
python scripts/external/run_external_inference.py --split mal2025_popularity_local_ready_yolo_coverbanner --targets popularity --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_popularity_local_ready_yolo_coverbanner
python scripts/external/run_external_inference.py --split mal2025_dual_local_ready_yolo_coverbanner --targets popularity meanScore --run-id 22 --run-dir final_project/runs --output-prefix run22_mal2025_dual_local_ready_yolo_coverbanner
python scripts/experiments/run_carma_tensor_aligned_baselines.py
python scripts/external/analyze_mal2025_external_diagnostics.py
python scripts/experiments/analyze_followup_experiment_statistics.py --n-boot 500
```
