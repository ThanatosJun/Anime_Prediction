# Git Cleanup Plan - 2026-05-21

Scope: baseline-only cleanup.

This plan organizes the current baseline-related changes into commit batches. Each batch is a research story, and each story is described with four artifact classes:

```text
source code
raw results
result analysis
synthesis
```

Important boundary:

```text
Out of scope for this cleanup:
- fusion/RAG experiment preparation under src/fussion_branch/
- EDA scripts and feature-filter reports
- presentation materials
- PDFs and local reference files
- personal/tooling notes such as AGENTS.md, karpathy-guidelines.mdc, docs/cursor_.md
```

## Batch 0 - Repository Hygiene And Cleanup Policy

Purpose: record local artifact policy before staging baseline work.

| Class | Files | Commit? | Reason |
|---|---|---|---|
| source code | `.gitignore` | yes | Keeps generated/local artifacts out of future commits. |
| raw results | `data/fetch_log.csv` | no | Local image download log; should not be committed. |
| synthesis | `reports/git_cleanup_plan_2026-05-21.md` | yes | Documents the baseline-only cleanup and commit boundary. |

## Batch 1 - Reference Baseline Source Code

Purpose: commit the source needed to reproduce C1, C2, and C3 reference baseline runs.

| Class | Files | Commit? | Reason |
|---|---|---|---|
| source code | `src/experiment_common/features.py` | yes | Shared feature loading and alignment helpers. |
| source code | `src/reference_baseline_branch/build_c1_character_features.py` | yes | C1 Figure 2 character feature builder. |
| source code | `src/reference_baseline_branch/build_gpt2_text_embeddings.py` | yes | GPT-2 text embedding builder used by C1/C2. |
| source code | `src/reference_baseline_branch/build_c3_rag_features.py` | yes | C3 RAG/SKAPP feature builder. |
| source code | `src/reference_baseline_branch/run_c3_skapp_full.py` | yes | C3 SKAPPFull diagnostics and regularized runner. |
| source code | `src/reference_baseline_branch/sklearn_models.py` | yes | C1/C2/C3 reconstruction model implementations. |
| source code | `src/reference_baseline_branch/configs/reference_baselines.yaml` | yes | Reference baseline registry and run configuration. |
| raw results | `.exp/baseline/**` | no | Local, reproducible run outputs. |

Practical note:

```text
sklearn_models.py and reference_baselines.yaml contain C1, C2, and C3 changes in the same files.
To avoid fragile patch staging, commit the reference baseline source as one batch.
```

## Batch 2 - Reference Baseline Curated Results

Purpose: commit the result layer that lets teammates inspect the scores without checking local `.exp` folders.

| Class | Files | Commit? | Reason |
|---|---|---|---|
| result analysis | `reports/reference_baseline_results.csv` | yes | Consolidated C1/C2/C3 score table. |
| result analysis | `reports/reference_baseline_runs.md` | yes | Maps result rows to local run IDs and provenance. |
| raw results | `.exp/baseline/results/18,25,33,34,35,36,37,38,39,40,41,42` | no | Bulky/local raw artifacts; summarized by the CSV and run index. |

## Batch 3 - Reference Baseline Analysis

Purpose: commit the reasoning behind the baseline interpretation and stopping decisions.

| Class | Files | Commit? | Reason |
|---|---|---|---|
| result analysis | `reports/reference_baseline_paper_alignment_audit.md` | yes | Explains C1/C2/C3 paper-alignment boundaries. |
| result analysis | `reports/c3_skappfull_negative_r2_diagnosis_2026-05-20.md` | yes | Records why C3 SKAPPFull underperforms and why it should stop. |

## Batch 4 - Reference Baseline Synthesis

Purpose: commit the teammate-facing summaries and handoff material.

| Class | Files | Commit? | Reason |
|---|---|---|---|
| result analysis | `reports/reference_baseline_status.md` | yes | Main status record for completed reference baselines. |
| synthesis | `reports/sota_reconstruction_summary_2026-05-19.md` | yes | Final SOTA/reference reconstruction summary. |
| synthesis | `reports/meeting_brief_2026-05-19_exp_reference_alignment.md` | yes | Meeting-readable baseline/EXP alignment brief. |
| synthesis | `reports/reference_baseline_handoff_2026-05-19.md` | yes | Handoff for future agents/teammates. |
| synthesis | `reports/reference_baseline_reconstruction_taskboard_2026-05-19.md` | yes | Follow-up taskboard and stop criteria. |
| synthesis | `reports/reference_baseline_reproduction_commands_2026-05-19.md` | yes | Commands needed to reproduce the curated results. |
| synthesis | `reports/reference_baseline_code_availability_2026-05-14.md` | no | Older note superseded by audit/status. |
| synthesis | `reports/reference_baseline_weekly_sync_2026-05-12.md` | no | Older weekly note; not needed for this baseline commit set. |

## Out-Of-Scope Dirty Files

Leave these uncommitted in the baseline cleanup:

```text
src/fussion_branch/**
scripts/eda/**
reports/feature_filter_*
reports/target_correlation_*
reports/trending_clip_*
reports/figures/**
docs/reports/a4_*
reports/a4_*
docs/*.pdf
docs/refer/**
docs/research_notes/**
docs/rq_experiment_realignment.md
AGENTS.md
karpathy-guidelines.mdc
docs/cursor_.md
```

They may be useful, but they belong to separate cleanup decisions and should not be mixed into the baseline commits.

