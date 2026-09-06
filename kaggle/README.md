# Kaggle deployment runbook

For continuation, read [the handoff](../docs/CODEX_HANDOFF.md) first. The first
submission is already complete (Public AUC 0.891, below the target). The steps
below describe reconstruction, not instructions to rerun completed jobs.
Reuse existing private outputs and verify their saved versions before launching
anything. A kernel push starts remote compute; this handoff alone does not
authorize another training run or competition submission.

The jobs are scripts rather than interactive notebooks so every saved version is
reproducible. Replace `__KAGGLE_USERNAME__` in metadata templates, then attach
the listed dependencies in the Kaggle editor if they are not resolved by the
API.

Reconstruction order:

1. Publish this repository (excluding `.git`) as a **private** Kaggle Dataset
   named `rsna-knee-code`.
2. Run `00_smoke`; its three-study DICOM/schema audit must pass.
3. Run `01_labels/kernel.py` with the competition and the three CC0 public label
   datasets attached. The current rank-normalized audit passes at 0.89241; do
   not use the excluded Barun source to inflate it.
4. The optional historical experiment ran `01b_open_llm` on T4 x2 with Apache-2.0
   Qwen2.5-1.5B, then run CPU job `01c_labels_plus_open_llm`. The current Qwen
   candidate is rejected (2.45% parse failures and lower gold AUC); the selected
   labels are the clean three-source public ensemble. Do not rerun the rejected
   candidate by default; reuse the completed `01c` selected-label artifact.
5. Run `02_cache_0` and `02_cache_1` independently without GPU. Each writes one
   memory-mappable pixel-cache half below the 20 GB output limit.
   Training is blocked if either half has fewer than 98% studies with at least
   one decodable slot.
6. Run CPU-only `02b_raptor_preflight` to record the exact CC0 checkpoint hash
   and state-dict compatibility before starting Model B.
7. Attach both cache outputs and the label output to `03_train_dino` and
   `04_train_coatnet`; run each with T4 x2 and internet enabled only for fetching
   the public timm initialization when a local checkpoint is not attached.
8. Run `05_select_blend` on CPU, then `05_benchmark` on T4 x2 after blend output
   is complete. The latter tests
   132 stratified-by-series-count studies, chooses three-pass or center-only
   TTA, and repeats four studies.
9. Attach model, blend, benchmark, code, and competition outputs to
   `06_submission`; use T4 x2, internet off, Save & Run All.
10. Inspect `inference_summary.json`, validate `submission.csv`, and request the
   user's final approval before pressing Kaggle **Submit**.

The visible three-study test is only a schema smoke test. The hidden-test
projection must remain below 7.5 hours, leaving at least 1.5 hours of the
competition's 9-hour submission allowance unused.

## Prepare an API upload bundle

Use the isolated local CLI only after Kaggle authentication is configured:

```bash
.venv-kaggle/bin/python scripts/prepare_kaggle_bundle.py \
  --username YOUR_KAGGLE_USERNAME \
  --output /tmp/rsna-knee-kaggle-bundle
```

Each GPU metadata file pins `machine_shape` to `NvidiaTeslaT4`. Pushing a
kernel creates/runs a saved version; it does **not** press the competition
Submit button. Never use `kaggle competitions submit` in this workflow—the
last action remains a deliberate user confirmation in the Kaggle UI.

## Hard gates

- Label training cannot begin unless `label_audit.json` reports gold macro AUC
  at or above 0.89.
- All five folds of both model families must exist and have matching OOF truth
  and gold masks.
- A blend needs at least +0.002 pseudo-label macro OOF and a positive lower
  bound for the paired gold bootstrap delta; otherwise it becomes the best
  single model automatically.
- The benchmark must pass both projected runtime and deterministic-repeat tests.
- A study with no recoverable slot is forced to exactly 0.5 for all targets
  after rank fusion.
