# Reference Baseline Branch

Literature-facing baselines used as external comparison points.

Run:

```bash
python -m src.reference_baseline_branch.run_reference_baselines
```

Narrow runs:

```bash
python -m src.reference_baseline_branch.run_reference_baselines --baseline F0-Mean
python -m src.reference_baseline_branch.run_reference_baselines --baseline F1-RF-Meta --target popularity
```

Config:

- `configs/reference_baselines.yaml`

Outputs:

- `.exp/baseline/results/{run_id}/baseline_results.csv`
- `.exp/baseline/results/{run_id}/baseline_summary.md`
- `.exp/baseline/results/{run_id}/predictions/`

