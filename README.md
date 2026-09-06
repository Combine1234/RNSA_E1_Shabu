# RSNA Knee Abnormality Detection — Kaggle pipeline and handoff

**เริ่มบนเครื่องใหม่:** [คู่มือย่อภาษาไทย](docs/START_HERE_TH.md).

**Resume here:** [Detailed Codex handoff](docs/CODEX_HANDOFF.md), then
[execution evidence](docs/implementation_status.md). The first submission
completed with **Public AUC 0.891**; it did **not** reach the Top 100 target.
Rank 1,826 is a historical September 5, 2026 snapshot, not a current rank.
Completed Kaggle artifacts should be reused, not automatically retrained.

This repository implements a Kaggle-first, offline-safe pipeline for the 2026
RSNA Knee Abnormality Detection competition. Raw competition data stays on
Kaggle. Local execution is limited to fast unit and synthetic smoke tests.

The implementation has gated promotion stages:

1. decode three real studies and verify the current competition schema;
2. ensemble three CC0 public report-derived scores after per-target rank
   normalization and prevalence calibration while preserving confidence
   and `UNK`/not-addressed masks; audit (and reject unless promoted) an optional
   Apache-2.0 Qwen2.5-1.5B candidate, then create report-hash-safe folds;
3. build a deterministic six-slot, geometry-sorted 2.5D pixel cache;
4. train cross-fitted 224 px DINOv2-Small and stride-safe 320 px CoAtNet
   study-level MIL models;
5. retain a blend only if it clears pseudo-label macro OOF and paired gold
   bootstrap gates (otherwise use the best single model);
6. benchmark full-model inference and repeatability before creating a
   submission file.

Do not download the competition corpus, pixel caches or model weights to the
workstation. Competition data and derived pixels must not be committed or
published. The Git repository contains source, configuration and documentation;
private Kaggle artifacts and account access are not transferred by cloning it.

## Repository layout

- `src/rsna_knee/`: reusable label, DICOM, cache, model, training, inference,
  blend, and validation code.
- `scripts/`: small command-line entrypoints used by Kaggle jobs.
- `kaggle/`: Kaggle script-notebook entrypoints and metadata templates.
- `configs/`: pinned experiment defaults.
- `tests/`: dependency-light tests that do not need real medical data.

## Kaggle job order

```text
00_smoke -> 01_labels (passes gold gate)
         \-> 01b_open_llm (rejected) -> 01c_labels_plus_open_llm -> 02_cache_0 + 02_cache_1
                                                   |-> 03_train_dino (five OOF folds)
                                                   \-> 04_train_coatnet (five OOF folds)
02b_raptor_preflight ------------------------------/
models -> 05_select_blend -> 05_benchmark -> 06_submission
```

This is the reconstruction order, not a request to rerun completed jobs or the
rejected LLM experiment. Labels/cache use CPU; training uses Kaggle GPUs.
The final job must use GPU T4 x2, internet
off, finish under 9 hours, and create `/kaggle/working/submission.csv`.

## Local smoke test

```bash
PYTHONPATH=src pytest
```

The Kaggle deployment runbook is in `kaggle/README.md`.

## Storage boundary

Raw DICOM, pixel caches and large checkpoints must remain on Kaggle. Only small
JSON/CSV/log diagnostics may be retrieved into an automatically selected writable
directory outside Git. An SSD is optional. Prefer a user-owned application-data
directory for persistent audits and the OS temporary directory for disposable
files; check space and record the path locally. The original disk root was:

```text
/media/monkey/PortableSSD/Healtcare/rsna-knee/
```

Do not require or recreate that mount path on a new machine. Never commit study identifiers,
reports, row-level labels/predictions or authentication files, even if the repo
is made private later. A second competition submission needs new user approval.

The upload bundler rejects DICOM, checkpoints, NumPy caches and Parquet data.
Each generated Kaggle script also carries a compressed source/config fallback
so a transient private-dataset mount failure cannot invalidate a long job; that
fallback contains no competition data or model weights.
