"""One test per defect the predecessor shipped, stated as an invariant.

These are not hypothetical. Each was found by audit or by a live failure,
executed, and confirmed to change a reported number before it was fixed. A
rewrite that dropped them would reintroduce them, so they are carried across as
properties of the new code rather than as ported code.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

import fixtures
from mag import evidence, gates, metrics, rules, stats
from mag.errors import BadInput
from mag.schema import PARASITE_CLASSES, UNINFECTED, content_hash


# --------------------------------------------------------------------------- #
# Ties decided by spelling
# --------------------------------------------------------------------------- #

def test_argmax_ties_use_the_declared_order_not_the_alphabet():
    """Alphabetical order decided 45% of one rule's calls; reversing only that
    component moved its speciation accuracy from 0.819 to 0.897."""
    ev = {"P_falciparum": 0.999, "P_malariae": 0.999, "P_vivax": 0.4}
    cls, ties = rules.argmax_class(ev)
    assert cls == "P_falciparum"
    assert ties == 2, "a tie was resolved without being counted"


def test_the_tie_order_covers_every_class():
    assert set(rules.TIE_ORDER) == set(PARASITE_CLASSES)


def test_the_trivial_baseline_does_not_depend_on_row_order():
    a = pd.Series({"p1": "x", "p2": "y"})
    b = pd.Series({"p1": "y", "p2": "x"})
    assert (rules.trivial(["p1"], a).iloc[0] == rules.trivial(["p1"], b).iloc[0])


# --------------------------------------------------------------------------- #
# Saturation
# --------------------------------------------------------------------------- #

def test_the_screening_score_does_not_saturate():
    """1 - exp(-t) is exactly 1.0 in float64 for t >= 37, and the median
    patient here carries t near 64: 116 of 170 collapsed onto one value."""
    scores = [rules.screening_score({"P_falciparum": t})
              for t in (0.0, 5.0, 37.0, 64.0, 115.0, 400.0)]
    assert len(set(scores)) == len(scores)
    assert all(np.diff(scores) > 0) and max(scores) < 1.0


def _view(**by_class):
    v = evidence.PatientView("p", n_fields=25)
    v.scores = {c: np.zeros(0) for c in PARASITE_CLASSES}
    for c, arr in by_class.items():
        v.scores[c] = np.asarray(arr, dtype=float)
    return v


def test_noisy_or_discriminates_where_the_direct_product_cannot():
    """Log space extends the range over which the accumulation is informative.

    It does not abolish saturation, and claiming it would be false: 1 - 0.05^400
    is 1.0 at any precision. What log space buys is that the ordering survives
    into the tail, where a running product has already underflowed to zero and
    reported every patient as exactly 1."""
    ev = rules.RULES["R7"].evidence
    a = ev(_view(P_falciparum=np.full(20, 0.6)), {})["P_falciparum"]
    b = ev(_view(P_falciparum=np.full(40, 0.6)), {})["P_falciparum"]
    assert 0.0 < a < b < 1.0, "the tail of the accumulation is not resolved"


def test_true_saturation_is_reported_as_a_tie_rather_than_resolved_silently():
    """Once two classes both reach 1.0 the argmax is arbitrary. It is broken by
    the declared order and the tie is counted, which is the only honest option
    left at that point."""
    ev = rules.RULES["R7"].evidence(
        _view(P_falciparum=np.full(400, 0.95), P_vivax=np.full(400, 0.95)), {})
    assert ev["P_falciparum"] == ev["P_vivax"] == 1.0
    assert all(np.isfinite(list(ev.values())))
    cls, ties = rules.argmax_class(ev)
    assert cls == "P_falciparum" and ties == 2


# --------------------------------------------------------------------------- #
# Two co-primary components at the same level
# --------------------------------------------------------------------------- #

def test_the_two_components_can_share_a_level():
    """Pairing a two-sided superiority test with a one-sided non-inferiority
    test put them at different levels: 5 wins and 0 losses then failed the
    primary endpoint at p = 0.0625 though the one-sided p was 0.031."""
    base = [False] * 5 + [True] * 49
    cand = [True] * 5 + [True] * 49
    two = stats.mcnemar_exact(base, cand)
    one = stats.mcnemar_exact(base, cand, alternative="greater")
    assert two.p == pytest.approx(0.0625)
    assert one.p == pytest.approx(0.03125)
    assert one.p <= 0.05 < two.p


def test_an_unknown_alternative_is_refused():
    with pytest.raises(BadInput):
        stats.mcnemar_exact([True], [True], alternative="less")


def test_a_fitted_arm_marks_its_p_value_invalid():
    r = stats.mcnemar_exact([False, True], [True, True], arms_are_fixed=False)
    assert r.valid is False and "exchangeable" in r.reason


def test_an_intersection_union_fails_if_any_component_fails():
    iu = stats.intersection_union({"a": True, "b": False})
    assert iu["reject"] is False and iu["failed"] == ["b"]


# --------------------------------------------------------------------------- #
# Missing predictions, degenerate resamples
# --------------------------------------------------------------------------- #

def test_a_patient_without_a_prediction_is_refused_not_scored():
    """NaN != 'uninfected' is True, so a missing patient used to be counted as
    a false positive on the study's primary endpoint."""
    truth = pd.Series({"a": UNINFECTED, "b": "P_falciparum", "c": UNINFECTED})
    with pytest.raises(BadInput):
        metrics.screening(truth, pd.Series({"a": UNINFECTED, "b": "P_falciparum"}))


def test_every_patient_gets_a_view_even_with_no_detections(tmp_path):
    patients = fixtures.patient_table(n_fields=(3, 5))
    det = fixtures.detections(patients.head(3), "ck")
    views = evidence.build_views(det, patients, "raw")
    assert len(views) == len(patients)
    silent = views[patients["patient_id"].iloc[-1]]
    assert silent.n_boxes("P_falciparum") == 0


def test_a_bootstrap_over_too_few_units_refuses():
    r = stats.bootstrap([1.0, 2.0], lambda u: float(np.mean(u)))
    assert r.method == "too-few-units" and not np.isfinite(r.lo)


def test_a_float32_nan_statistic_is_counted_as_degenerate():
    r = stats.bootstrap(list(range(50)), lambda u: np.float32("nan"), n_boot=200)
    assert r.n_degenerate == 200


def test_no_band_is_claimed_from_zero_plasmodium_patients():
    t = pd.Series({"a": UNINFECTED})
    got = metrics.band_referenced_accuracy(t, t)
    assert got["band"] is None and got["unambiguous"] is False


# --------------------------------------------------------------------------- #
# Portability
# --------------------------------------------------------------------------- #

def test_the_content_hash_does_not_depend_on_the_operating_system(monkeypatch):
    """pandas resolves an unset line terminator to os.linesep even when
    returning a string, so the same table hashed differently on Windows and
    Linux and every cache failed its own integrity check on the other."""
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    posix = content_hash(df)
    monkeypatch.setattr(os, "linesep", "\r\n")
    assert content_hash(df) == posix


def test_a_tampered_cache_is_refused(tmp_path):
    from mag import cache as cache_mod
    from mag.errors import IntegrityError

    cache, _ = fixtures.write_cache(tmp_path, checkpoints=("rtdetr",))
    det, _ = cache_mod.load(cache, "rtdetr")
    det.iloc[0, det.columns.get_loc("score")] = 0.123456
    det.to_parquet(cache / "detections_rtdetr.parquet", index=False)
    with pytest.raises(IntegrityError):
        cache_mod.load(cache, "rtdetr", verify=True)


def test_a_write_leaves_no_temporary_file(tmp_path):
    cache, _ = fixtures.write_cache(tmp_path, checkpoints=("rtdetr",))
    assert not list(cache.glob("*.tmp"))


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def test_a_nearly_empty_pass_is_refused_as_a_measurement():
    """A released checkpoint returned two boxes across 4,508 fields. Reported
    as a detector property it would have read as 'YOLOv5 fails at every rule'."""
    man = type("M", (), {"checkpoint": "yolov5x", "conf_floor": 0.001,
                         "n_detections": 2, "n_images": 2})()
    g = gates.yield_gate(man, 4508)
    assert g.passed is False and "4,508" in g.message


def test_a_full_pass_is_accepted():
    man = type("M", (), {"checkpoint": "rtdetr", "conf_floor": 0.001,
                         "n_detections": 1_352_400, "n_images": 4508})()
    assert gates.yield_gate(man, 4508).passed


def test_sparsity_is_only_suspicious_below_a_low_floor():
    man = type("M", (), {"checkpoint": "x", "conf_floor": 0.25,
                         "n_detections": 2, "n_images": 2})()
    assert gates.yield_gate(man, 4508).passed


def test_the_design_gate_refuses_a_screening_equivalent_candidate():
    truth = pd.Series({"a": UNINFECTED, "b": "P_falciparum"})
    base = pd.Series({"a": "P_falciparum", "b": "P_falciparum"})
    cand = pd.Series({"a": "P_vivax", "b": "P_vivax"})     # same D1 calls
    g = gates.design_gate(truth, base, cand, checkpoint="ck", rule="R4")
    assert g.passed is False and "screening-equivalent" in g.message


def test_the_design_gate_refuses_a_baseline_with_no_headroom():
    # differs on D1 for patient c, so it is NOT screening-equivalent; the
    # baseline is nonetheless perfect on the negatives, so nothing is winnable
    truth = pd.Series({"a": UNINFECTED, "b": "P_falciparum", "c": "P_vivax"})
    base = pd.Series({"a": UNINFECTED, "b": "P_falciparum", "c": "P_vivax"})
    cand = pd.Series({"a": UNINFECTED, "b": "P_falciparum", "c": UNINFECTED})
    g = gates.design_gate(truth, base, cand, checkpoint="ck", rule="R4")
    assert g.passed is False and "headroom" in g.message


def test_identifiability_fails_when_every_negative_is_one_centre():
    patients = fixtures.patient_table()
    g = gates.identifiability(patients)
    assert g.passed is False and "collinear" in g.message


def test_the_confirmatory_checkpoint_is_declared_not_selected():
    """Protocol v3 §8.3. The previous version chose the arm with the most
    baseline headroom, and that inflates the size of the test that follows --
    headroom is b plus the both-wrong count, so choosing the largest headroom
    chooses in part the largest b. v3 declares the arm in advance."""
    heads = {"rtdetr": 3, "yolov8x": 14, "yolov5x": 8}
    discs = {"rtdetr": 2, "yolov8x": 9, "yolov5x": 5}
    chosen, rule, detail = gates.select_checkpoint(heads, discs)
    assert rule == "declared"
    assert chosen == gates.DECLARED_CHECKPOINT == "rtdetr"
    assert chosen != detail["by_max_headroom"], (
        "this fixture is only informative if the declared arm is NOT the one "
        "headroom would have chosen")
    assert detail["primary_is_pre_registered"] is True
    assert "descriptive feasibility" in detail["caveat"]


def test_a_declared_checkpoint_that_did_not_survive_is_refused():
    """Substituting another arm is selection by a different name."""
    from mag.errors import BadInput

    with pytest.raises(BadInput) as e:
        gates.select_checkpoint({"a": 14, "b": 8}, {"a": 3, "b": 11})
    assert "declares this checkpoint in advance" in e.value.remedy


def test_the_level_preserving_criterion_is_carried_beside_it():
    heads = {"rtdetr": 3, "yolov8x": 14}
    discs = {"rtdetr": 2, "yolov8x": 9}
    _, _, detail = gates.select_checkpoint(heads, discs)
    assert detail["by_max_discordance"] == "yolov8x"
    assert detail["agree_with_discordance"] is False


def test_selecting_by_headroom_inflates_the_size_and_by_discordance_does_not():
    """The claim this package used to make about its own selection rule was
    false, and this is the test that says so.

    Headroom is b plus the both-wrong count, so selecting the largest headroom
    selects in part the largest b -- the statistic's own numerator. Selecting
    on the discordant-pair count cannot, because the exact test conditions on
    that count and is Binomial(n, 1/2) given it.
    """
    from mag import simulate

    r = simulate.selection_comparison(54, [0.5, 0.5, 0.5], reps=4000, seed=1)
    assert r["headroom"]["size_after_selection"] > 0.05
    assert r["discordance"]["size_after_selection"] <= 0.05
    assert r["fixed"]["size_after_selection"] <= 0.05
