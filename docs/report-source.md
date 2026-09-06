# RSNA knee submission implementation evidence

Historical pre-training snapshot, retained for provenance. The status below is
superseded by [implementation_status.md](implementation_status.md) and
[CODEX_HANDOFF.md](CODEX_HANDOFF.md). The first submission subsequently completed
with Public AUC 0.891; caches and training are no longer awaiting those gates.

Audience: project owner  
Date: 2026-09-05 (Asia/Bangkok)  
Scope: a free, Kaggle-first first-submission pipeline targeting a public score
above the live top-100 cutoff, without storing raw DICOM on the workstation.

## Direct answer

The implementation is ready through label selection and architecture
preflight. The selected teacher is the robust ensemble of three public label
sets, which scored 0.892406 macro AUC on the 58 expert-labelled studies. The
open Qwen candidate was correctly rejected. Both CPU cache jobs are running;
training remains blocked until their manifests show at least 98% study
coverage. No submission action is automated.

## Evidence and decisions

- Competition files are consumed only through Kaggle's mounted competition
  source. No raw DICOM is included in the local or private code bundle.
- The public notebook reported at 0.936 is retained only as a reproducibility
  anchor, not assumed to clear the current target: [Head and shoulders, knees
  and toes](https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes).
- The selected labels come from [Pilkwang](https://www.kaggle.com/datasets/pilkwang/rsna-knee-llm-labels),
  [Lixin73](https://www.kaggle.com/datasets/lixin73/rsna-knee-llm-report-labels-sol56),
  and [Steven Lee Hans](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels).
  Their rank-normalized, prevalence-calibrated median achieved 0.892406 gold
  macro AUC (bootstrap 95% CI 0.856058-0.924621).
- The pinned [Qwen2.5-1.5B-Instruct model](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
  is Apache-2.0. Its run produced 108/4,407 parse failures (2.45%) and reduced
  gold macro AUC by 0.013367 (paired bootstrap 95% CI -0.025282 to -0.001121),
  so it is excluded.
- Model A uses the Apache-2.0 [timm DINOv2-Small checkpoint](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m).
- Model B uses the CC0 [Raptor WideDense weights](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).
  The selected SWA file passed strict state loading and a synthetic forward at
  320 px; its SHA-256 is
  `d8bb0f8751b4bb65750257869ddc4c7a3c0cdfc6e62596fc68193919406c53eb`.
  A 336 px forward was rejected because the learned relative-position grid is
  incompatible. The Apache-2.0 [timm CoAtNet checkpoint](https://huggingface.co/timm/coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k)
  is the explicit fallback.

## Remaining gates

1. Both cache manifests must report at least 98% studies with a recoverable
   slot.
2. Each model family must produce five complete OOF folds.
3. The blend must beat the best single model by at least 0.002 pseudo-label
   macro AUC and have a positive lower bound on paired gold bootstrap delta;
   otherwise the best single family is used.
4. The 132-study runtime benchmark must project at most 7.5 hours and pass the
   repeated-inference tolerance test.
5. The final file must pass exact schema, ID order, finite range, variation and
   unrecoverable-study fallback checks. Competition submission requires an
   explicit user confirmation.

## Limitations

The 58-study expert set yields wide uncertainty and cannot guarantee a public
leaderboard score. Public leaderboard rank also changes over time. Candidate
selection therefore relies on fixed OOF gates and uses the public leaderboard
only for the final pre-submit target check.

## Claim-to-source ledger

- Kaggle competition/data/rules/leaderboard: RSNA and Kaggle, accessed
  2026-09-05, <https://www.kaggle.com/competitions/rsna-knee-abnormality-detection>.
- Public anchor: prvsiyan, notebook version 38, accessed 2026-09-05,
  <https://www.kaggle.com/code/prvsiyan/head-and-shoulders-knees-and-toes>.
- Label sources: Pilkwang, Lixin73, Steven Lee Hans, Kaggle datasets, accessed
  2026-09-05; links are provided above.
- DINOv2 and CoAtNet license/model cards: timm on Hugging Face, accessed
  2026-09-05; links are provided above.
- Raptor license/checkpoint: dreaddevelopment, Kaggle dataset, accessed
  2026-09-05; link is provided above.
- Qwen license/model revision: Qwen on Hugging Face, accessed 2026-09-05; link
  is provided above. Run metrics are first-party artifacts from private Kaggle
  notebooks `rsna-knee-01b-open-llm-labels` and
  `rsna-knee-01c-labels-plus-open-llm`.
