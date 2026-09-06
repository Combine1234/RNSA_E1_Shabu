# Instructions for continuing this project

Read `docs/CODEX_HANDOFF.md` first, then `docs/implementation_status.md` and
the actual source files involved in the requested work. The handoff describes
the completed first submission, its limitations, and how to resume on another
machine. `docs/report-source.md` is an earlier research snapshot, not live status.

- First submission `56035531` completed with Public AUC 0.891. It did **not**
  achieve the intended Top 100 result. Do not describe schema/runtime success
  as score success. Rank 1,826 was a historical 2026-09-05 snapshot.
- Use free Kaggle resources for image processing, feature extraction, training,
  and inference. Keep workstation work to source/config/docs and short tests.
- Do not download raw DICOM, pixel NPY/NPZ caches, or large model artifacts to
  the workstation. Read small JSON/CSV/log outputs using an explicit allowlist
  into an automatically selected writable directory outside Git. An SSD is
  optional: the user explicitly authorized choosing local storage on the new
  machine. Prefer a user-owned application-data directory for persistent small
  audits, or the OS temporary directory for disposable files. Check free space,
  restrict access to the current user where supported, and record the resolved
  path locally. Do not download all kernel outputs. The original SSD path
  `/media/monkey/PortableSSD/Healtcare/rsna-knee/` is historical, not required.
- The Git repository may be public. Never commit credentials, reports, study
  identifiers, row-level labels/predictions, medical images, caches or weights.
  Making a repository private later does not undo prior exposure.
- Inspect current Kaggle status and artifact versions before launching jobs.
  Reuse completed caches/models; do not automatically rebuild the full pipeline.
  A kernel push launches a remote run and can consume free GPU quota.
- Preserve the recorded promotion gates. Before future experiments, review the
  validation limitations in the handoff, particularly gold-based checkpoint
  selection and unknown upstream Raptor training-fold provenance.
- User approval covered the first submission only. Do not submit a second
  prediction, change team settings, or publish private Kaggle artifacts without
  further authorization. The former monitoring automation was paused after the
  first score arrived; do not assume it exists on the new machine.

Use relative repository paths in code and runbooks. Do not copy the original
machine's virtual environment or authentication files to a new machine.
