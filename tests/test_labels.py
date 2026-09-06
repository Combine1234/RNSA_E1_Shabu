import numpy as np

from rsna_knee.constants import TARGETS
from rsna_knee.labels import (
    mask_suspected_gold_leakage,
    normalize_report,
    report_hash,
    robust_label_ensemble,
    target_column_candidates,
)
from rsna_knee.llm_labeler import parse_codes


def test_report_hash_normalizes_case_and_whitespace():
    assert report_hash("  ACL   Tear\n") == report_hash("acl tear")
    assert normalize_report(None) == ""


def test_gold_overrides_pseudo_and_unknown_is_not_negative():
    sources = np.full((3, 2, len(TARGETS)), 0.5)
    sources[:, 0, 0] = [0.8, 0.9, 1.0]
    gold = np.full((2, len(TARGETS)), np.nan)
    gold[0, 0] = 0.0
    result = robust_label_ensemble(sources, gold)
    assert result.probabilities[0, 0] == 0.0
    assert result.gold[0, 0]
    assert result.confidence[1, 0] == 0.0
    assert not result.addressed[1, 0]


def test_synovitis_only_imputes_when_effusion_addressed():
    sources = np.full((3, 2, len(TARGETS)), 0.5)
    effusion = TARGETS.index("Effusion")
    synovitis = TARGETS.index("Synovitis")
    sources[:, 0, effusion] = [0.8, 0.9, 1.0]
    result = robust_label_ensemble(sources)
    assert result.addressed[0, synovitis]
    assert result.probabilities[0, synovitis] > 0.5
    assert not result.addressed[1, synovitis]


def test_public_label_column_aliases_exclude_confidence():
    columns = ["pseudo_ACL", "ACL_confidence", "ACL_status", "MCL"]
    assert target_column_candidates(columns, "ACL") == ["pseudo_ACL"]
    assert target_column_candidates(columns, "MCL") == ["MCL"]


def test_source_confidence_stays_separate_from_probability():
    sources = np.full((3, 1, len(TARGETS)), 0.5)
    sources[:, 0, 0] = 0.8
    addressed = np.zeros_like(sources, dtype=bool)
    addressed[:, 0, 0] = True
    confidence = np.zeros_like(sources)
    confidence[:, 0, 0] = 0.2
    result = robust_label_ensemble(
        sources,
        source_confidence=confidence,
        source_addressed=addressed,
    )
    assert np.isclose(result.probabilities[0, 0], 0.8)
    assert np.isclose(result.confidence[0, 0], 0.2)


def test_gold_copy_is_removed_from_audit_only():
    gold = np.array([[0.0, 1.0], [1.0, 0.0]])
    sources = np.stack([gold, 1.0 - gold])
    confidence = np.ones_like(sources)
    addressed = np.ones_like(sources, dtype=bool)
    audit_confidence, audit_addressed, diagnostics = mask_suspected_gold_leakage(
        sources, confidence, addressed, gold
    )
    assert diagnostics[0]["suspected_gold_copy"]
    assert not diagnostics[1]["suspected_gold_copy"]
    assert not audit_addressed[0].any()
    assert addressed[0].all()
    assert audit_confidence[0].sum() == 0


def test_open_llm_code_parser_requires_twelve_labels():
    assert parse_codes("Y,N,U,Y,N,U,Y,N,U,Y,N,U") == list("YNUYNUYNUYNU")
    assert parse_codes("YNYYNNYNNYYYUYNYYY") == list("YNYYNNYNNYYY")


def test_rank_normalized_ensemble_preserves_prevalence_and_order():
    values = np.array([
        [[0.1] * 12, [0.2] * 12, [0.8] * 12, [0.9] * 12],
        [[0.0] * 12, [0.4] * 12, [0.6] * 12, [1.0] * 12],
        [[0.2] * 12, [0.3] * 12, [0.7] * 12, [0.8] * 12],
    ])
    raw = robust_label_ensemble(values)
    ranked = robust_label_ensemble(values, rank_normalize_sources=True)
    assert np.all(np.diff(ranked.probabilities[:, 0]) > 0)
    assert np.isclose(ranked.probabilities[:, 0].mean(), raw.probabilities[:, 0].mean(), atol=1e-5)
    assert parse_codes("Y,N,U") is None
