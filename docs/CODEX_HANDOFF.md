# RSNA Knee — detailed handoff for another Codex machine

Prepared 2026-09-06, Asia/Bangkok. This is an operational handoff, not a claim
that the Top 100 objective has been achieved. อ่านไฟล์นี้ก่อนเริ่มงานบนเครื่องใหม่:
งานส่งครั้งแรกเสร็จแล้ว แต่คะแนนยังไม่ถึงเป้าหมาย ต้องตรวจคุณภาพโมเดลและวิธี
validation ก่อนทดลองต่อ ไม่ต้องสร้าง cache หรือ train ทุกอย่างใหม่ทันที

## 1. Outcome and exact resume point

| Item | Recorded result |
| --- | --- |
| Competition | `rsna-knee-abnormality-detection` |
| Kaggle owner of private pipeline | `fronktja` |
| Team at first submission | `E1_Shabu` |
| Submitted notebook | `fronktja/rsna-knee-06-submission`, Version 1 |
| Notebook script version / run ID | `347459760` |
| Output selected for scoring | `/kaggle/working/submission.csv` |
| Competition submission reference | `56035531` |
| Submitted at | 2026-09-05 15:07:56.207 UTC (22:07:56 Bangkok) |
| Status observed | COMPLETE at the 16:00 UTC check |
| Public AUC displayed | **0.891** |
| Historical rank | **1,826**, leaderboard export at 2026-09-05 16:01:11 UTC |
| Rank 100 in that snapshot | 0.940 |
| Target at submission | 0.945 and rank better than 100 |
| Outcome | Target not met; execution succeeded |

Description used: `First gated submission: 5-fold frozen Raptor/CoAtNet with 3-offset TTA`.
Completion time above is the time it was observed, not an exact worker finish
timestamp. No private score was available. Do not reuse the historical rank as
a live rank or infer a precise unrounded score from the three-decimal display.

The final model used **CoAtNet/Raptor only**, five fold-specific heads with
one shared frozen pretrained encoder architecture. DINO was trained, but the
blend was rejected by the recorded bootstrap gate and received zero final
weight on all targets. The code still attaches both model notebooks as inputs;
it loads only families with non-zero final weights.

Links:

- [Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [Submissions](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/submissions)
- [Submitted source version](https://www.kaggle.com/code/fronktja/rsna-knee-06-submission?scriptVersionId=347459760)
- [Leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)

Access to private notebooks requires the authorized Kaggle account. A Git clone
does not transfer ownership, access rights, checkpoints or competition data.

## 2. Scope, authority and storage

The original objective was a first submission above Top 100, using free
resources and little local compute. The first attempt is over and cannot be
retroactively made successful. Improving a subsequent submission is possible,
but requires evidence and further work; do not guarantee its score.

Heavy compute belongs on Kaggle. The working budget was approximately 24 of
30 free GPU hours per week, reserving six hours for validation/retries. These
are planning assumptions from 2026-09-05: verify the current quota in the account
before launching jobs. Previous failed/aborted runs also consumed resources;
there is no reliable remaining-quota counter committed here. The target
accelerator was T4 x2 (`NvidiaTeslaT4` in metadata), not paid infrastructure.

Original workstation source root: `/home/monkey/Documents/Healtcare`.
Original external-disk root: `/media/monkey/PortableSSD/Healtcare/rsna-knee/`.
**User update on 2026-09-06: the new machine may have no SSD attached; the agent
may choose suitable local storage automatically without asking for that mount.**
No raw DICOM, pixel-cache NPY/NPZ, or large checkpoints are to be downloaded
locally. The storage permission applies to source/configuration and small
JSON/CSV/log audits, not to moving heavy data/compute off Kaggle.

For persistent small audits, choose a writable, user-owned application-data
directory outside the Git checkout, for example:

- Linux: `$XDG_DATA_HOME/rsna-knee/audits`, or
  `$HOME/.local/share/rsna-knee/audits` if XDG_DATA_HOME is unset.
- macOS: `$HOME/Library/Application Support/rsna-knee/audits`.
- Windows: `%LOCALAPPDATA%\\rsna-knee\\audits`.

Resolve the actual OS/user paths rather than copying these strings blindly.
Check free space and writability, keep medical audit files out of shared or
cloud-synced folders, and restrict directory access to the current user where
supported. If the preferred directory is unavailable, choose another suitable
local user-owned directory outside Git; do not delete unrelated files to make
room. Record the resolved location in a local note outside Git and tell the user
which directory was selected. No need to ask merely because an SSD is absent.
Use the OS temporary directory for disposable source bundles/logs. Temporary
files may disappear after reboot; keep resume-critical small evidence in the
persistent audit directory. Do not create the old `/media/...` path to pretend
an external drive is mounted. Large artifacts remain on Kaggle in all cases.

The user explicitly authorized the first competition Submit, which has already
happened. That approval does not authorize additional submissions. Current
handoff work does not authorize new training, a second submission, changing
Kaggle team membership, publishing private model outputs or accepting new terms
on the user's behalf. Preparing and reviewing a new candidate is distinct from
submitting it.

The local heartbeat `rsna-knee-daily-gate-check` was paused after the score was
observed. It is a desktop-thread automation, not a Kaggle job and not part of
the cloned repository. Old task ID: `01a06d5d-e5f3-71b2-b473-00b8e828cc95`.
Already launched Kaggle saved runs continue when the workstation shuts down;
local Codex orchestration should not be assumed to run while the machine is off.

## 3. Repository map and sources of truth

| Path | Responsibility |
| --- | --- |
| `AGENTS.md` | Short continuation constraints |
| `docs/CODEX_HANDOFF.md` | This detailed resume guide |
| `docs/implementation_status.md` | Historical executed stages and score |
| `docs/report-source.md` | Earlier research snapshot and source/licensing ledger |
| `assets/public_assets.yaml` | Public assets, excluded sources, declared licenses |
| `configs/base.yaml` | Experiment configuration actually consumed by entrypoints |
| `kaggle/*/kernel.py` | Remote entrypoint for each stage |
| `kaggle/*/kernel-metadata.template.json` | Dependency attachments and GPU/internet settings |
| `kaggle_support.py` | Input discovery, embedded-code bootstrap and gate checks |
| `src/rsna_knee/labels.py`, `folds.py` | Label aggregation, masks and grouped splitting |
| `src/rsna_knee/dicom.py`, `cache.py` | DICOM decoding/orientation and cache access |
| `src/rsna_knee/dataset.py` | Triplet selection, augmentation, mmap-aware sampling |
| `src/rsna_knee/models.py`, `train.py` | Study-level models and both training paths |
| `src/rsna_knee/blend.py`, `metrics.py` | Rank fusion, AUC and bootstrap selection |
| `src/rsna_knee/infer.py`, `validate.py` | GPU inference, timing, repeatability and schema |
| `scripts/prepare_kaggle_bundle.py` | Source-only deployment bundle generator |
| `scripts/*.py` | Individual command-line helpers; inspect `--help` before use |
| `tests/` | Synthetic/dependency-light regression tests |

Prefer the actual source, saved Kaggle versions and artifacts over old chat
claims. This Git snapshot is not byte-identical to an old uploaded bundle:
documentation changes also affect the embedded archive. The submitted Version 1
is immutable evidence of what actually ran. Keep it available.

## 4. Start on a new machine

Clone the user-provided Git remote and run all commands from its root. Do not
copy `.venv-kaggle`, `~/.kaggle`, browser cookies, OAuth codes or API tokens from
the original workstation. Authenticate locally through Kaggle's own flow.

Repository: [Combine1234/RNSA_E1_Shabu](https://github.com/Combine1234/RNSA_E1_Shabu).
It was public when this handoff was prepared; the owner plans to make it private.
No medical data, weights or credentials are included, regardless of visibility.

```bash
git clone https://github.com/Combine1234/RNSA_E1_Shabu.git
cd RNSA_E1_Shabu
```

Minimal local setup (avoid installing torch just to read status/build bundles):

```bash
python3 -m venv .venv-kaggle
.venv-kaggle/bin/python -m pip install kaggle numpy pandas PyYAML pytest
.venv-kaggle/bin/kaggle auth login
PYTHONPATH=src .venv-kaggle/bin/python -m pytest -q
.venv-kaggle/bin/kaggle competitions submissions -c rsna-knee-abnormality-detection --page-size 10
.venv-kaggle/bin/kaggle kernels status fronktja/rsna-knee-06-submission
```

The last recorded test run was **22 passing tests**. These tests do not prove
CUDA behavior, decoded anatomy correctness, missing-data coverage in the hidden
test, or leaderboard quality. Do not run local training as an environment test.
`pyproject.toml` gives broad supported ranges, not a reproducible lockfile.

Observed DINO artifact environment: Python 3.12.13, torch 2.10.0+cu128,
timm 1.0.26, numpy 2.0.2, pandas 2.3.3, pydicom 3.0.2. This is a recorded remote
environment, not an instruction to install CUDA packages on the workstation.
Future Kaggle container changes must be tested and recorded explicitly.

If CLI permissions fail for a private notebook, first check the owner/slug and
account. On the original machine, OAuth credentials existed but private access
was denied after restart; `kaggle auth login --force` and an authorized browser
consent flow restored access. Do not print access tokens or paste them into chat.
An agent without access needs the user to sign in locally; a public Git repo
does not solve private Kaggle access. Use the configured browser skill if
controlling the in-app browser; do not assume old tab objects survive migration.

## 5. Remote job inventory: reuse completed work

Every slug below is under `https://www.kaggle.com/code/fronktja/`.
The code dataset is `fronktja/rsna-knee-code` (last recorded published version 9).
Versions not listed here should be resolved on Kaggle rather than guessed.

| Local folder | Remote slug | Last recorded result |
| --- | --- | --- |
| `00_smoke` | `rsna-knee-00-smoke` | COMPLETE; three studies decoded |
| `01_labels` | `rsna-knee-01-labels` | COMPLETE; public-label audit passed |
| `01a_llm_probe` | `rsna-knee-01a-llm-probe` | COMPLETE; 1/96 parse failures |
| `01b_open_llm` | `rsna-knee-01b-open-llm-labels` | COMPLETE; candidate rejected |
| `01c_labels_plus_llm` | `rsna-knee-01c-labels-plus-open-llm` | COMPLETE; selected public labels/folds |
| `02a_orientation_audit` | `rsna-knee-02a-orientation-audit` | COMPLETE; all 24,371 series |
| `02_cache_0` | `rsna-knee-02-cache-0` | COMPLETE; 2,204/2,204 valid studies |
| `02_cache_1` | `rsna-knee-02-cache-1` | COMPLETE; 2,203/2,203 valid studies |
| `02b_raptor_preflight` | `rsna-knee-02b-raptor-preflight` | COMPLETE; SWA compatible at 320 px |
| `03_train_dino` | `rsna-knee-03-train-dino` | Version 3 COMPLETE; five folds |
| `04_train_coatnet` | `rsna-knee-04-train-coatnet` | Version 3 COMPLETE; five frozen-encoder heads |
| `05_select_blend` | `rsna-knee-05-select-blend` | Version 1 COMPLETE; ensemble rejected |
| `05_benchmark` | `rsna-knee-05-benchmark` | Version 1 COMPLETE; runtime/repeatability pass |
| `06_submission` | `rsna-knee-06-submission` | Version 1 COMPLETE; submitted once |

Recorded run IDs: DINO v3 `347398474`, CoAtNet v3 `347407114`, benchmark v1
`347442485`, final v1 `347459760`. Obsolete DINO v2 `347381535` and CoAtNet v2
`347402943` were cancelled. Do not resume them as if they were current jobs.

Training consumes the **01c** label notebook even though its name mentions the
rejected LLM. That artifact contains the selected public-only labels and final
fold assignment. Attaching 01 and 01c simultaneously can make strict
`one_file()` discovery ambiguous. Exact dependencies are in metadata templates.
The benchmark depends on the completed blend notebook to choose active families.

The two cache manifests are `cache_*_of_02.json`. Cache pixels are
`*_images.npy`, with associated validity/ID files on Kaggle. Do not fetch these
NPY files locally. Each training output has `fold_0.pt` through `fold_4.pt`,
`oof_fold_0.csv` through `oof_fold_4.csv`, histories, and `artifact_manifest.json`.
Other important small artifacts: `label_audit.json`, `raptor_preflight.json`,
`blend_weights.json`, `benchmark.json`, `inference_summary.json`.

## 6. Retrieve evidence without downloading weights or pixels

First inspect file names with `kaggle kernels files OWNER/SLUG`. The CLI displayed
implausibly small file sizes for some outputs, so do not trust those sizes as a
safe download filter. Checkpoint sizes recorded in manifests were 87,205,911
bytes per DINO fold and 296,017,351 bytes per CoAtNet fold.

Safe example for a selected small output:

```bash
audit_dir=$(mktemp -d /tmp/rsna-audit.XXXXXX)
.venv-kaggle/bin/kaggle kernels output fronktja/rsna-knee-05-benchmark \
  -p "$audit_dir" --file-pattern '^benchmark\.json$'
```

The CLI also downloaded the notebook log during these requests. Inspect log
size and contents before retaining it; do not put row-level data or credentials
into Git. For model verification use the allowlist
`^(artifact_manifest\.json|history_fold_[0-4]\.json|oof_fold_[0-4]\.csv)$`.
OOF files are small enough for local numerical audits but contain study IDs;
keep them outside the repository. Never invoke unfiltered `kernels output` on
cache/training jobs. Never invoke competition data downloads for this handoff.

To recover the precise submission version, use the saved Kaggle Version 1
source. CLI help for `kernels pull` describes source-only retrieval; do not
confuse it with `kernels output`. Metadata attachments without explicit version
pins can resolve newer upstream notebook outputs in future runs. Preserve or
record source versions and manifest hashes before modifying any upstream slug.

## 7. Data, labels and fold conventions

Historical dataset description: approximately 569.76 GB, 4,407 training studies,
58 expert-labelled studies, multilingual reports for weak supervision. Mounted
competition files include `train.csv`, `train_series.csv`, `test.csv`,
`test_series.csv`, `train_series/` and `test_series/`. Inference uses images and
series metadata and does not require `Report`.

Submission columns, in exact order:

```text
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
```

Selected public label inputs:

- `pilkwang/rsna-knee-llm-labels`
- `lixin73/rsna-knee-llm-report-labels-sol56`
- `stevenleehans/rsna-knee-llm-report-labels`

These were recorded as CC0 sources. Audit source versions/licenses again before
adding or replacing assets. Rank-normalize each target, apply prevalence
calibration and robust median while keeping confidence/addressed information.
Expert labels override teacher probabilities. Gold training loss multiplier is
4.0. Unknown probability is 0.5, not automatically a negative. Effusion informs
Synovitis only when Synovitis is unaddressed, with configured weight 0.55.

The public-only teacher achieved gold macro AUC 0.8924062551, bootstrap 95% CI
0.856058–0.924621, passing the 0.89 gate. The audit is performed without copying
expert labels into the teacher predictions used for that comparison.

Optional Qwen2.5-1.5B-Instruct used revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (Apache-2.0), batches of four per GPU,
1,536 input tokens and at most 40 new tokens. It failed parsing on 108/4,407
studies (2.45%, allowed 2%). Including it reduced gold AUC to approximately
0.87904, delta -0.01337 with CI -0.02528 to -0.00112. It is excluded; no paid API
was used. Do not repeat that full labeling experiment unchanged.

Folds use seed 20260905 and normalized-report hashes (casefold + whitespace
normalization; SHA-256 truncated to 20 characters). Five fold sizes:
881, 882, 881, 882, 881. Gold membership and positives were included in balancing.
Group hashes must not cross train/validation boundaries. This only addresses
identical normalized reports; it does not prove patient-level independence or
absence of semantically duplicated reports.

In OOF CSVs, `pred__TARGET` is the prediction, `true__TARGET` is the soft/binary
truth used for evaluation, `weight__TARGET` is loss/audit weight, and
**`gold__TARGET` is a membership mask, not a ground-truth value**. Compute gold
AUC from `true__TARGET` on rows whose `gold__TARGET` is true. Do not infer gold
membership from a weight threshold. Production blend code binarizes truth with
`> 0.5`; a previous hand calculation used `>= 0.5`, yielding DINO 0.85821859
instead of the blend artifact's 0.85821199. Use the production convention for
comparisons and explicitly handle unknown labels.

## 8. DICOM preprocessing and model inputs

Six fixed slots in order: sagittal fluid-sensitive, sagittal non-fluid,
coronal fluid-sensitive, coronal non-fluid, axial fluid-sensitive, axial
non-fluid. Slot metadata uses `Sagittal/Coronal/Axial` and fluid flag 1/0.
Series selection uses physical span. Slice order uses orientation and position
geometry rather than lexical filename order. Crop is 150 mm; robust intensity
normalization uses 0.5th and 99.5th percentiles with edge fraction 0.02.

Cache shape per study is `[6, 11, 336, 336]` uint8 plus validity masks. The
models consume three adjacent slices as channels, one triplet per slot. This
is a limited 2.5D representation, not full-volume MRI analysis. DINO resizes to
224 px; Raptor resizes to 320 px. Inference uses offsets `[-2, 0, 2]` within the
cached-style 11-slice window. No horizontal-flip TTA is used.

The orientation audit covered 24,371 series. Patient-centre laterality agreed
with explicit tags on 99.08% axial, 98.55% coronal and 98.12% sagittal series.
The policy trusts explicit tags first, uses patient-centre fallback, applies
coronal/axial horizontal canonicalization and reverses right-knee sagittal
slice order. Agreement with tags is not a radiologist's independent anatomical
audit; residual errors remain possible.

Cache coverage was 100% of studies with at least one usable slot in both halves,
not a guarantee that all slots or all slices decode. Coverage gate is 98% per
half. Inference tracks wholly unrecoverable studies; after fusion those studies
receive exactly 0.5 on all targets. The three-study dry run had none. Synthetic
decoder/missing-data tests do not establish coverage of every transfer syntax
or corrupt-file condition in the hidden set.

## 9. Models actually trained

### Model A: DINOv2-Small

Backbone `vit_small_patch14_dinov2.lvd142m`, 224 px, last six blocks trainable,
target-specific attention over slots. Token pooling was corrected for the ViT;
do not substitute CNN average-pooling arguments without testing. Five folds,
maximum 12 epochs, minimum 4, early-stop patience 3, batch size 2, block shuffle
size 64. AdamW head LR 3e-4, backbone LR 3e-6, weight decay 0.05; soft BCE with
confidence/gold/rare-positive weights and EMA.

Training augmentation uses random slice offset -2 through 2, scale 0.94–1.06,
intensity gain 0.9–1.1 and bias -0.04–0.04. Validation uses centre triplets.
Best epochs were 12, 11, 12, 8 and 10 for folds 0–4. Gold macro OOF was
0.85921225; pseudo macro OOF in blend evaluation was 0.85821199.

### Model B: frozen Raptor/CoAtNet

Backbone name `coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k`, **actual input 320 px**.
Initial weights came from `dreaddevelopment/raptor-knee-widedense`, not the
similarly named MaxSpan dataset. Selected SWA SHA-256:
`d8bb0f8751b4bb65750257869ddc4c7a3c0cdfc6e62596fc68193919406c53eb`.
The preflight passed at 320; 336 was incompatible with the learned relative
position grid. This is why the original 336 px model plan was changed.

Full fine-tuning was abandoned under the free-compute budget: v1 was killed
after slow random mmap reads; v2 remained too slow even with block sampling
(over 12 minutes before batch 250, projected over 80 minutes per epoch).
Version 3 calls `train_frozen_feature_folds`, not `train_fold`.

It freezes the encoder, extracts centre-triplet features once over all studies
with batch size 8, and trains five separately reset attention heads. Head batch
size 256, max 40 epochs, minimum 12, patience 6, LR 0.001, weight decay 0.05,
EMA decay 0.995. Persisted model config has `last_blocks_trainable=0` even though
the YAML still contains legacy full-finetune fields such as -1 and 10 epochs.
The entrypoint/function determines which fields actually apply.

Checkpoints contain full standard study-model state, including the encoder,
so each is approximately 296 MB even though the encoder is shared in concept.
Pseudo macro OOF 0.86420806; gold macro OOF 0.85382608. Features are generated
with `training=False`, so Model B does not inherit Model A's image augmentation.
It retains target-specific slot attention, not the original public Raptor head.

### Public references and exclusions

`prvsiyan/head-and-shoulders-knees-and-toes`, recorded v38 / reported 0.936,
was a research reference, **not a reproduced 0.936 baseline in this project**.
Raptor WideDense was recorded at reported public 0.924, CC0. DINO/timm fallback
and Qwen were recorded Apache-2.0. The local LICENSE covers local code, not every
external checkpoint. See `assets/public_assets.yaml` and the source ledger.

RadImageNet was excluded over licensing compatibility questions. The Barun
label dataset was excluded due to unclear `other` licensing and suspected
copying of expert values. Do not silently reinstate either one.

## 10. Blend decision and interpretation

Per-target search uses percentile ranks and a 0.05 weight grid. It allows at
most 0.01 target gold-AUC drop, requires 0.002 target pseudo gain, then applies
an overall macro gate and paired study-bootstrap gate (2,000 repeats).

| Aggregate diagnostic | Recorded value |
| --- | --- |
| Best single pseudo macro (CoAtNet) | 0.86420806 |
| Best single gold macro (DINO) | 0.85921225 |
| Candidate blend pseudo macro | 0.87821483 |
| Gold delta vs gold-best single | +0.01632504 |
| Delta 95% interval | -0.00075637 to +0.03456916 |
| Macro gate | pass |
| Gold lower-bound > 0 gate | fail |
| Final family weights for every target | `[0.0, 1.0]` (DINO, CoAtNet) |

In `blend_weights.json`, individual target `promoted` flags describe the
preliminary target search. The final `promotion.ensemble_promoted` and final
`weights` are authoritative. A target can still say `promoted: true` even when
the macro bootstrap rejected the ensemble and replaced all weights. Do not
accidentally deploy the diagnostic candidate weights.

Rank fusion maps three distinct scores to 0, 0.5 and 1 in the visible three-row
file. That is expected from percentile ranking and does not mean the network
emitted three hard classes. Rankings preserve single-model AUC but are not
calibrated probabilities. On hidden rerun the ranks are recomputed over the
hidden test cohort. Preserve the whole-cohort ranking if batching inference.

## 11. Runtime, final validation and honest limits

Benchmark used 132 training studies sampled across series-count distribution,
five selected CoAtNet folds, three offsets. Inference elapsed 208.018 seconds,
model/setup overhead 33.840 seconds. Projection was
`overhead + sample_seconds / 132 * 1322 * 1.3 = 2742.177 seconds`, or 0.761716 h.
Four studies were run twice; maximum raw prediction difference 0.0 at tolerance
1e-5. This is repeatability evidence for that environment/subset, not a promise
of bitwise identical results on any GPU/container.

Dry-run inference on three test studies took 27.70775 seconds. Entire saved
notebook runtime was 1,636.6 seconds (27m17s), including substantial startup/input
discovery/model I/O. The first printed stage appeared at 1,256.9 seconds. The
old chat's claim that this was precisely mount time was an inference, not a
measured decomposition. Recursive input globs may also be expensive and should
be investigated. The benchmark projection did not account for every pre-infer
operation in the final entrypoint; it is not a wall-clock ETA or SLA.

Validation checked ID order against mounted test metadata, exactly 12 ordered
targets, finite [0,1] values and no constant targets for larger test sets.
Dry run: 3 unique IDs, all targets varied, zero all-missing fallback. Hidden
rerun completed and scored. There is no committed hidden prediction file or
hidden per-study decode audit. Completion does not justify claiming zero
corrupt studies throughout the hidden test.

## 12. Validation risks to resolve before pursuing higher scores

These are source-review findings and hypotheses, not an established explanation
of why the public score is 0.891:

1. Both training paths choose best epochs using
   `0.7 * pseudo_auc + 0.3 * gold_auc` on validation folds. The same 58 expert
   studies are therefore used for checkpoint selection and later ensemble
   selection. OOF predictions are held out from that fold's gradient updates,
   but the gold result is **not an untouched final test**. Bootstrap on fixed
   predictions does not account for this selection process or weight search.
2. Raptor's upstream training-fold exposure has not been established here.
   Five new cross-fitted heads do not make a pretrained encoder independent of
   the validation studies if that encoder previously trained on them. Audit
   upstream provenance before treating Model B OOF as leakage-free.
3. Centre-triplet representation and centre-only feature extraction may miss
   lesions outside the small slab. TTA offsets widen the slab slightly but do
   not cover a full volume. Compare slice/series coverage on Kaggle.
4. Public Raptor quality does not automatically transfer to our new head,
   preprocessing, orientation, normalization or labels. A frozen encoder plus
   newly trained head is materially different from the public solution.
5. Current OOF selection uses centre views; submitted inference averages five
   heads and three offsets. TTA's quality benefit was not separately established
   with matched OOF evaluation. Timing/repeatability are not accuracy tests.
6. The public anchor's reported 0.936 was never reproduced here. Do not claim
   that our pipeline beats it because engineering gates passed.
7. Two cache coverage manifests alone do not prove unique IDs, full union,
   no overlap, matching geometry versions or exact agreement with label rows.
   The existing 4,407-row OOF audit supports coverage; future cache changes need
   explicit ID/set/version checks as well as the fraction gate.
8. Initial source/license records and dynamic timm identifiers do not fully
   pin every model download. Preserve artifact hashes and actual initialization
   source, including any fallback, before claiming exact reproducibility.

Do not simply loosen the failed bootstrap gate or search until a seed gives a
positive interval. Record any planned validation redesign before evaluating
new candidates. A clean nested/held-out protocol and a provenance audit are
more informative than repeatedly tuning on the tiny expert set.

## 13. Recommended next work, after reviewing this handoff

First verify existing submissions and completed artifact versions. Recover
only small manifests, OOF/history files and configuration needed for analysis.
Check per-target failures, fold alignment and checkpoint initialization source.
Do not rerun old cache/LLM jobs just to reconstruct context.

Then design an evaluation protocol addressing section 12. Reproduce one
licensed public model with its intended preprocessing on Kaggle, if accessible,
and compare it on an appropriate local validation protocol. It is an experiment,
not an authorized automatic leaderboard submission. Prioritize modest changes
that isolate a cause: slice coverage, feature/head compatibility, label quality,
and TTA matching. Keep a ledger of GPU time and stop unproductive full-finetune
runs rather than consuming all free quota.

For every candidate record input version IDs/hashes, source commit, actual
container versions, training mode, fold IDs, pseudo/gold per-target metrics,
selection policy, timing and remaining uncertainty. The original gates remain
the historical policy until explicitly revised; do not relabel old results as
passing a new policy retroactively.

## 14. Deploy an authorized candidate (not a default restart command)

Build a fresh source-only bundle in an empty temporary directory:

```bash
bundle_dir=$(mktemp -d /tmp/rsna-source-bundle.XXXXXX)
.venv-kaggle/bin/python scripts/prepare_kaggle_bundle.py \
  --username fronktja --output "$bundle_dir"
```

The bundle includes `dataset/`, `kernels/`, and `bundle_manifest.json`. Templates
replace `__KAGGLE_USERNAME__`; all Kaggle kernels are private by default. Code
is embedded as a compressed source-only fallback in each generated script;
`bootstrap_repo()` prefers this embedded source when present. Push generated
scripts rather than raw entrypoints, which assume a particular support mount.

Only when an experiment is authorized and prerequisites pass:

```bash
.venv-kaggle/bin/kaggle kernels push -p "$bundle_dir/kernels/STAGE_FOLDER"
```

Replace `STAGE_FOLDER` deliberately. This starts a saved run; it is not a dry
upload. Metadata templates are under the old owner; changing only the output
owner on another account does not grant access to the old private inputs.
Use existing authorized account/input access or arrange access with the user.

The code dataset can be versioned from the bundle's `dataset/` when needed;
inspect `kaggle datasets version --help` and existing privacy before doing so.
Do not publish the entire working directory. The bundler rejects major data/
weight suffixes, but it is not a general secret scanner. Review generated
contents and dependency versions before upload.

Competition Submit is a separate action. Future explicit approval should name
the candidate version/file. On the Kaggle Output tab select **submission.csv**
before opening Submit to Competition; selecting `inference_summary.json` opens
an invalid-file dialog. Confirm notebook version and output in the dialog.
After one Submit, inspect the submissions list before any retry, to avoid
duplicates. Do not upload the visible three-row CSV as an ordinary static-file
prediction: this competition privately reruns the selected notebook on hidden
test data.

## 15. Copy/paste instruction for Codex on the other machine

> Read AGENTS.md, docs/CODEX_HANDOFF.md and docs/implementation_status.md.
> Continue from the completed first submission 56035531 (Public AUC 0.891,
> historical rank 1,826), whose Top 100 goal was not met. Inspect existing Kaggle
> artifacts before rerunning anything. Keep heavy compute on free Kaggle and
> keep raw images/caches/weights off the workstation and Git. Audit validation
> and Raptor pretraining provenance before trusting OOF comparisons. Prepare
> concrete experiments and preserve the first submitted version as evidence.
> Do not create another competition submission without explicit user approval.

This handoff intentionally contains source/configuration references and aggregate
results rather than credentials or row-level competition data. Authenticate on
the new machine and resolve private artifacts on Kaggle to continue.
