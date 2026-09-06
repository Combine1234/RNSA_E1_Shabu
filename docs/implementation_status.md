# Implementation status

Updated: 2026-09-05 (Asia/Bangkok)

Handoff clarification added 2026-09-06: see [CODEX_HANDOFF.md](CODEX_HANDOFF.md)
for migration instructions and validation limitations. The execution results
below are historical; no new Kaggle runs were started for the handoff.

## Implemented

- Current competition schema and three-study DICOM smoke job.
- Six acquisition slots selected by physical span, geometry-based slice order,
  laterality canonicalisation, 150 mm crop and robust per-volume intensity
  scaling.
- Eleven adjacent center slices cached as `uint8`; training consumes one
  three-adjacent-slice RGB view per slot and uses deterministic slice-offset
  TTA at inference.
- Three-CC0-source, percentile-rank-normalized robust median report-label
  ensemble with prevalence calibration and explicit source
  confidence, verdict/not-addressed masks, expert override and Synovitis-only
  Effusion imputation.
- Gold-copy leakage detection excludes contaminated sources from audit. The
  rank-normalized, prevalence-calibrated public ensemble scores 0.89241 macro
  AUC on the 58 expert studies (bootstrap 95% CI 0.85606-0.92462), passing the
  0.89 label-quality gate.
- The pinned Apache-2.0 Qwen2.5-1.5B candidate was evaluated and rejected: its
  parse-failure rate was 2.45% (gate: at most 2%) and adding it reduced gold
  macro AUC by 0.01337 (paired bootstrap 95% CI -0.02528 to -0.00112). The final
  teacher therefore remains the three-public-source ensemble.
- Five deterministic folds grouped by normalized report hash, balanced for
  pseudo labels, gold-study membership and gold positives.
- 224 px DINOv2-Small and 320 px CoAtNet study MIL models with target-specific slot attention,
  differential learning rates, EMA, soft BCE, rare-target weighting and early
  stopping. Training uses shuffled contiguous 64-study blocks so memory-mapped
  cache reads remain local while study order still changes each epoch.
- Raptor weights compatibility preflight and explicit timm fallback.
- Per-target rank fusion plus macro OOF and paired gold-bootstrap promotion.
- Two-GPU, decode-once inference; corrupted/missing slot handling; exact 0.5
  fallback only for wholly unrecoverable studies.
- Runtime projection, automatic TTA reduction, repeatability check, schema
  validation, artifact SHA-256 manifests and a raw-data-free upload bundler.

## Remote execution status

- `rsna-knee-00-smoke`: complete. Three real studies decoded successfully; all
  produced `[6, 11, 336, 336]` caches with 5/6 usable slots.
- `rsna-knee-01-labels`: complete and passed the 0.89 gold-AUC gate.
- `rsna-knee-01a-llm-probe`: complete; prompt/parser probe failed on 1/96 rows.
- `rsna-knee-01b-open-llm-labels`: complete; rejected by the final label audit.
- `rsna-knee-01c-labels-plus-open-llm`: complete; selected only the three clean
  public sources and produced balanced folds of 881-882 studies.
- `rsna-knee-02b-raptor-preflight`: complete. The selected SWA checkpoint loads
  and forwards successfully at 320 px; 336 px is incompatible with its learned
  relative-position grid.
- `rsna-knee-02a-orientation-audit`: complete over all 24,371 series.
  Patient-centre laterality agreed with explicit DICOM laterality on 99.08% of
  axial, 98.55% of coronal and 98.12% of sagittal series, while covering nearly
  all series missing the explicit tag. The cache now trusts the explicit tag
  first, uses patient-centre position as fallback, canonicalizes coronal/axial
  horizontally and reverses right-knee sagittal slice order to a consistent
  lateral-to-medial direction.
- `rsna-knee-02-cache-0` and `rsna-knee-02-cache-1`: complete with audited
  laterality and slice-order normalization. Coverage is 2,204/2,204 and
  2,203/2,203 respectively, with zero wholly undecodable studies reported in
  either shard. This does not prove every slot/slice decoded successfully.
- `rsna-knee-03-train-dino`: version 3 completed all five folds on Kaggle T4
  x2 with the corrected token pooling and block-local sampler. The audited OOF
  set contains 4,407 unique studies in the expected 881-882-study folds, all
  predictions are finite (range 0.09864-0.99538), and macro AUC is 0.85822 on
  pseudo labels and 0.85921 on the 58 expert studies. The five
  87,205,911-byte checkpoints have distinct SHA-256 hashes. Only the small OOF,
  history, manifest and log files were downloaded for this audit.
- `rsna-knee-04-train-coatnet`: version 1 was killed by the worker after long
  random memory-map reads. Version 2 still failed the runtime gate: it had not
  reached batch 250 after more than 12 minutes, projecting over 80 minutes per
  epoch, so its verified run was cancelled to preserve free GPU quota. Model B
  now uses the same audited frozen Raptor encoder to extract a center-triplet
  feature table once, followed by five inexpensive fold-specific attention
  heads. The saved checkpoints retain the standard study-model schema for
  unchanged submission inference. Version 3 completed all five folds with
  4,407 unique OOF studies, finite predictions, pseudo-label macro AUC 0.86421
  and gold-label macro AUC 0.85383. All five 296,017,351-byte checkpoints are
  recorded with distinct SHA-256 hashes; only the small OOF, history, manifest
  and log files were downloaded for the audit.
- `rsna-knee-05-select-blend`: complete. The rank-fused candidate improved
  pseudo-label macro AUC from the best single model's 0.86421 to 0.87821, but
  its paired gold bootstrap delta versus the gold-best DINO model was 0.01633
  with 95% CI -0.00076 to 0.03457. Because the lower bound did not exceed zero,
  the mandatory gold gate rejected the ensemble and selected CoAtNet/Raptor
  alone for every target.
- `rsna-knee-05-benchmark`: complete and passed. On 132 studies with all three
  deterministic TTA offsets, inference took 208.02 seconds after model setup;
  the safety-adjusted projection for 1,322 hidden-test studies is 2,742.18
  seconds (0.762 hours), safely below the 7.5-hour promotion limit.
  Repeatability was exact across two runs (maximum absolute difference 0.0,
  tolerance 1e-5). The benchmark and final notebook read the gated blend
  artifact first and load only model families with non-zero final weights.
- `rsna-knee-06-submission`: version 1 completed successfully on the three
  visible test studies using the five gated CoAtNet/Raptor folds and all three
  deterministic TTA offsets. Model inference took 27.71 seconds after Kaggle
  mounted the inputs. `submission.csv` has exactly 3 unique study IDs followed
  by the 12 official targets in order; all 36 probabilities are finite and in
  [0, 1], every target varies, and no study required the all-data-missing 0.5
  fallback. After explicit user approval, Kaggle accepted this notebook/output
  as the first competition submission with description `First gated
  submission: 5-fold frozen Raptor/CoAtNet with 3-offset TTA`. Its private
  hidden-test notebook rerun completed successfully. Kaggle submission
  `56035531` received Public AUC **0.891** at 2026-09-05 23:00 Asia/Bangkok,
  placing team `E1_Shabu` at public rank **1,826** in the 16:01 UTC leaderboard
  snapshot. Rank 100 remained 0.940. The first-submission target of 0.945 / Top
  100 was therefore not met, despite the execution, schema and runtime gates
  all passing. No duplicate submission was created.

The pre-submit target was 0.945 (`max(0.945, cutoff + 0.003)`), based on the
historical 0.940 cutoff. It was not achieved. A future target requires a fresh
leaderboard check and new submission approval.

No competition DICOM has been downloaded locally. Only small audit files are
stored outside Git on the Portable SSD or in temporary directories. Completed
cache/model artifacts should be reused. The submission monitor was paused after
the score arrived; do not automatically resubmit or restart the pipeline.
