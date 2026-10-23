"""One test per defect found in the adversarial audit of this package.

Four independent reviews were run against the finished code — one on the
numerical modules, one on the statistics, one on the data handling, one on the
wiring — with a brief to find real bugs and confirm each by running it. Between
them they found about fifty. Every one is fixed, and every one is pinned here,
because a defect that was found once and not pinned is a defect that comes back
the next time the surrounding code is touched.

Each test names the defect it prevents. Where the reviewers measured the damage,
the measurement is in the docstring: a number is harder to argue away than a
description.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import fixtures
from mag import (cache as cache_mod, calibrate, diagnostics, evidence, figures,
                 gates, power, provenance, rules, schema, search, selective,
                 simulate, stats, study)
from mag.errors import BadInput, IntegrityError, ProvenanceError
from mag.schema import DETECTION_COLUMNS, UNINFECTED


def _det(rows):
    return pd.DataFrame(rows, columns=list(DETECTION_COLUMNS))


def _pat(rows):
    return pd.DataFrame([{"patient_id": p, "centre": "Montpellier",
                          "truth": t, "n_images": n} for p, t, n in rows])


# --------------------------------------------------------------------------- #
# Data handling
# --------------------------------------------------------------------------- #

def test_suppression_never_runs_across_patients():
    """The worst finding of the audit. A field is a (patient, image) pair, and
    the suppression group was keyed on image_id alone -- so per-patient frame
    numbering (000.jpg, 001.jpg) made one patient's boxes delete another's.

    On a fixture re-keyed that way, 28% of detections under `matched` and 38%
    under `source` were wrongly removed, and a patient whose only detection lost
    a tie became uninfected. It also turned "one field" into "every patient's
    copy of frame 003", whose O(n^2) IoU matrix exhausted memory.
    """
    d = _det([("A", "IMG1", "ck", "P_vivax", 0.9, 0.5, 0.5, 0.1, 0.1),
              ("B", "IMG1", "ck", "P_vivax", 0.8, 0.5, 0.5, 0.1, 0.1)])
    for condition in ("matched", "source"):
        kept = evidence.apply_condition(d, condition)
        assert len(kept) == 2, (
            f"{condition}: two patients' identical boxes are two fields, not one")
        assert set(kept["patient_id"]) == {"A", "B"}


def test_the_box_census_counts_fields_not_image_names():
    d = _det([("A", "IMG1", "ck", "RBC", 0.9, 0.5, 0.5, 0.1, 0.1),
              ("B", "IMG1", "ck", "RBC", 0.8, 0.2, 0.2, 0.1, 0.1)])
    census = diagnostics.box_census({"ck": d}, 2)
    assert census.iloc[0]["per_field_sd"] == 0.0
    assert bool(census.iloc[0]["fixed_budget"]) is True


def test_a_join_that_matches_nothing_is_refused_not_analysed():
    """A zero-padded identifier read back as an integer matches nothing, and the
    analysis then produced a complete, plausible set of results: every patient
    with no detections, the trivial baseline as the answer, no warning."""
    patients = _pat([("P0001", UNINFECTED, 2), ("P0002", "P_vivax", 2)])
    d = _det([("1", "i0", "ck", "P_vivax", 0.9, 0.5, 0.5, 0.1, 0.1)])
    with pytest.raises(BadInput) as e:
        evidence.build_views(d, patients, "raw")
    assert "zero-padded" in e.value.remedy


def test_partial_drops_are_counted_and_travel_to_the_reader():
    patients = _pat([("A", "P_vivax", 2)])
    d = _det([("A", "i0", "ck", "P_vivax", 0.9, 0.5, 0.5, 0.1, 0.1),
              ("ZZ", "i0", "ck", "P_vivax", 0.9, 0.5, 0.5, 0.1, 0.1)])
    rep = {}
    evidence.build_views(d, patients, "raw", report=rep)
    assert rep["n_dropped"] == 1
    assert rep["dropped_share"] == pytest.approx(0.5)


def test_a_detection_with_no_field_cannot_land_in_another_patients_arrays():
    """`pd.factorize` codes a missing value as -1, which is a valid index into
    the row above -- so a detection with no image_id was filed under the
    previous patient's last field, confidence and all."""
    with pytest.raises(BadInput) as e:
        schema.validate_detections(_det([
            ("A", "i0", "ck", "P_vivax", 0.9, 0.5, 0.5, 0.1, 0.1),
            ("A", None, "ck", "P_ovale", 0.95, 0.5, 0.5, 0.1, 0.1)]))
    assert "image_id" in e.value.problem


def test_a_box_of_non_positive_extent_is_refused():
    """Its area clips to zero, so it neither suppresses nor is suppressed: it
    passes straight through the condition whose purpose is to make two detector
    families comparable."""
    with pytest.raises(BadInput) as e:
        schema.validate_detections(_det([
            ("A", "i0", "ck", "P_vivax", 0.9, 0.5, 0.5, -0.1, 0.1)]))
    assert "suppression" in e.value.remedy


def test_a_patient_with_no_reference_diagnosis_is_refused():
    """Depending only on the dtype pandas chose, `truth != UNINFECTED` either
    dropped them from the specificity denominator or scored them as a false
    negative. Two different wrong answers for one input."""
    with pytest.raises(BadInput) as e:
        schema.validate_patients(_pat([("A", UNINFECTED, 2), ("B", None, 2)]))
    assert "reference diagnosis" in e.value.problem


def test_a_negative_field_count_is_refused():
    with pytest.raises(BadInput):
        schema.validate_patients(_pat([("A", UNINFECTED, -5)]))


def test_the_content_hash_ignores_a_widening_but_not_a_narrowing():
    """An integer column read back as a float, or an object column read back as
    pandas "string", must hash the same -- those are lossless. A NARROWING to
    float32 is not: 0.1 in float32 is a different number, and pretending
    otherwise would make the check unable to see real corruption. What is fixed
    is the diagnosis: `cache.load` names dtype narrowing as a likely cause
    instead of telling the operator their table was tampered with."""
    d = _det([("A", "i0", "ck", "P_vivax", 0.1, 0.5, 0.5, 0.1, 0.1),
              ("A", "i1", "ck", "P_ovale", 0.2, 0.4, 0.4, 0.1, 0.1)])
    widened = d.copy()
    widened["patient_id"] = widened["patient_id"].astype("string")
    widened["score"] = widened["score"].astype("float64")
    assert schema.content_hash(d) == schema.content_hash(widened)

    narrowed = d.copy()
    for c in ("score", "x", "y", "w", "h"):
        narrowed[c] = narrowed[c].astype("float32")
    assert schema.content_hash(d) != schema.content_hash(narrowed), (
        "a lossy narrowing is real corruption and must be visible")


# --------------------------------------------------------------------------- #
# The rules
# --------------------------------------------------------------------------- #

def test_r5_carries_two_thresholds_with_two_names():
    """`theta` stood for both the per-field vote threshold and the share-of-
    fields screening threshold, in different units, so the sweep varied both at
    once and the threshold profile described nothing."""
    from mag.schema import ALL_CLASSES

    v = evidence.PatientView(
        "P", 3, scores={c: np.zeros(0) for c in ALL_CLASSES},
        per_field_max={c: (np.full(3, 0.6) if c == "P_falciparum"
                           else np.zeros(3)) for c in ALL_CLASSES},
        per_field_count={})
    assert rules.RULES["R5"].apply(v, {"theta": 0.7})["label"] == "P_falciparum"
    assert rules.RULES["R5"].apply(v, {"vote_tau": 0.7})["label"] == UNINFECTED


def test_the_density_fallback_is_visible():
    """It changes the unit of the statistic while rho stays put: the same
    patient scores 0.0025 with an RBC channel and 0.1 without, and is called
    uninfected or falciparum accordingly. The docstring said "loudly"; the code
    said nothing."""
    from mag.schema import ALL_CLASSES

    scores = {c: np.zeros(0) for c in ALL_CLASSES}
    scores["P_falciparum"] = np.full(3, 0.9)
    v = evidence.PatientView("P", 30, scores=scores, per_field_max={},
                             per_field_count={})
    assert rules.RULES["R6"].apply(v)["fallback"] is True
    scores["RBC"] = np.full(1200, 0.9)
    v2 = evidence.PatientView("P", 30, scores=scores, per_field_max={},
                              per_field_count={})
    assert rules.RULES["R6"].apply(v2)["fallback"] is False


def test_non_finite_evidence_does_not_crash_the_argmax():
    """`nan >= best - 1e-12` is False for every class, so one NaN emptied the
    tie list and `min()` raised "arg is an empty sequence" from inside a rule."""
    assert rules.argmax_class({"P_vivax": float("nan")}) == (UNINFECTED, 0)
    assert rules.argmax_class({}) == (UNINFECTED, 0)


def test_the_sweep_does_not_count_one_rule_twice():
    """R6 and R6b differ only in a default the grid overrides, so every R6b row
    was a byte-identical duplicate -- and `n_configurations` and `n_reproducing`
    are the two numbers the sweep asks a reader to weigh."""
    patients = fixtures.patient_table(n_fields=(2, 4), seed=21)
    det = fixtures.detections(patients, "ck", seed=21)
    truth = patients.set_index("patient_id")["truth"]
    sw = search.sweep(det, patients, truth, conditions=("matched",),
                      rule_ids=["R6", "R6b"])
    assert set(sw.table["rule"]) == {"R6"}
    assert sw.aliases == {"R6": ("R6b",)}


# --------------------------------------------------------------------------- #
# Calibration and selective prediction
# --------------------------------------------------------------------------- #

def test_a_tie_group_is_never_split_across_calibration_bins():
    """A perfectly calibrated predictor reported an ECE of 0.36, and the answer
    moved with the frame's row order: cross-fitted isotonic on 170 patients
    swung from 0.050 to 0.070 across permutations, across the finite-sample
    null, so the headline calibration claim flipped on row order alone."""
    p = np.full(10, 0.30)
    y = np.array([1., 1., 1., 0., 0., 0., 0., 0., 0., 0.])
    assert calibrate.ece_equal_mass(p, y, 5)[0] == pytest.approx(0.0, abs=1e-12)
    for perm in ([2, 0, 1, 3, 4, 5, 6, 7, 8, 9], list(range(9, -1, -1))):
        assert calibrate.ece_equal_mass(p[perm], y[perm], 5)[0] == \
            pytest.approx(0.0, abs=1e-12)


def test_temperature_never_silently_returns_the_identity():
    """A same-sign bracket means the optimum is at or beyond an endpoint, not
    at T = 1. Returning 1.0 handed back the identity map: three uninfected
    patients kept at 0.9, 0.8, 0.7 when the optimum wanted 0.5. On an
    uninformative score, 51% of folds took that branch."""
    s = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    y = np.array([0., 0., 0., 1., 1., 1.])
    t = calibrate.temperature(s, y)
    assert t != 1.0
    calibrated = calibrate.apply_temperature(t, s)
    assert np.all(np.abs(calibrated - 0.5) < 0.05), (
        "an anti-correlated fold should be shrunk to the base rate")


def test_a_calibration_slope_is_nan_when_it_is_undefined():
    """`platt`'s neutral fallback is (1, 0), which is a defensible identity for
    a calibrator and the worst possible value for a slope: a = 1 is the
    definition of perfect calibration."""
    a, b = calibrate.slope_intercept(np.array([0.1, 0.4, 0.9]), np.ones(3))
    assert np.isnan(a) and np.isnan(b)


def test_fewer_than_two_folds_is_refused():
    """k = 0 divided by zero and returned one fold; k = 1 trained on every
    patient it scored. Both wrote results while the report still claimed each
    probability came from a map fitted without that patient."""
    for k in (0, 1):
        with pytest.raises(BadInput):
            calibrate.folds(10, k)


def test_coverage_selection_does_not_depend_on_row_order():
    """A stable sort on confidence alone let the frame's order decide which of
    the forty patients at exactly zero was retained: the risk at 90% coverage
    moved from 0.163 to 0.196 across permutations of one dataset -- either an
    improvement on the full-cohort 0.176 or worse than not abstaining at all."""
    idx = [f"P{i:03d}" for i in range(100)]
    conf = pd.Series([0.0] * 60 + list(np.linspace(0.1, 1.0, 40)), index=idx)
    pred = pd.Series(["a"] * 100, index=idx)
    base, _ = selective.apply_coverage(pred, conf, 0.9)
    rng = np.random.default_rng(0)
    for _ in range(5):
        perm = list(rng.permutation(idx))
        shuffled, _ = selective.apply_coverage(pred[perm], conf[perm], 0.9)
        assert (shuffled.reindex(idx) == base).all()


def test_a_coverage_threshold_that_cannot_reproduce_itself_says_so():
    """Applying the reported threshold to new patients reproduced the exact
    failure the module was written to fix: all eight retained, not four."""
    pred = pd.Series(list("abcdefgh"))
    _, meta = selective.apply_coverage(pred, pd.Series([0.0] * 8), 0.5)
    assert meta["threshold_reproduces_coverage"] is False
    assert meta["n_at_threshold"] == 8


def test_an_out_of_range_coverage_target_is_refused():
    pred = pd.Series(list("abcd"))
    for target in (1.5, -0.1):
        with pytest.raises(BadInput):
            selective.apply_coverage(pred, pd.Series([0.5] * 4), target)


def test_a_patient_with_no_prediction_is_refused_by_the_selective_curve():
    """Reindexing put NaN there, `t != pg` was True, and the patient was
    counted as misclassified -- the predecessor bug the schema module exists
    to catch, reappearing three modules downstream."""
    truth = pd.Series(["uninfected", "P_falciparum", "uninfected"],
                      index=["p1", "p2", "p3"])
    pred = pd.Series(["uninfected", "P_falciparum"], index=["p1", "p2"])
    conf = pd.Series([1.0, 1.0, 1.0], index=truth.index)
    with pytest.raises(BadInput):
        selective.risk_coverage(pred, truth, conf)


def test_the_aurc_is_the_mean_risk_over_the_whole_domain():
    """The trapezoid over the realised grid covers [1/n, 1], not [0, 1], and
    dropped the first slice -- short by risk[0]/n, up to a fifth of a small
    AURC, appearing or vanishing according to whether the single most confident
    patient happened to be right."""
    idx = [f"P{i}" for i in range(5)]
    truth = pd.Series(["x"] * 5, index=idx)
    pred = pd.Series(["y", "x", "x", "x", "x"], index=idx)   # top-ranked wrong
    conf = pd.Series([0.9, 0.8, 0.7, 0.6, 0.5], index=idx)
    rc = selective.risk_coverage(pred, truth, conf)
    assert rc.aurc == pytest.approx(float(np.mean(rc.risk)))
    # The trapezoid over the realised grid runs from 1/n to 1, so it halves the
    # two endpoints and drops the [0, 1/n] slice entirely: it is short by
    # (risk[0] + risk[-1]) / 2n. Here that is 0.12 on an AURC of 0.457 -- a
    # quarter of the value, and it appears or vanishes according to whether the
    # single most confident patient happened to be right.
    cov = np.asarray(rc.coverage)
    risk = np.asarray(rc.risk)
    trapezoid = float(np.trapezoid(risk, cov))
    n = len(risk)
    assert rc.aurc - trapezoid == pytest.approx((risk[0] + risk[-1]) / (2 * n),
                                                abs=1e-12)


def test_confidence_is_a_posterior_margin_not_a_raw_one():
    """On raw evidence a near-tie carrying 100 and 90 units outranked a clean
    single-species call carrying 10. Since kappa decides who is referred, the
    two definitions refer different patients."""
    cols = list(rules.PARASITE_CLASSES)
    ev = pd.DataFrame([[10, 0, 0, 0, 0, 0], [100, 90, 0, 0, 0, 0]],
                      index=["clean", "near_tie"], columns=cols, dtype=float)
    k = selective.confidence(ev)
    assert k["clean"] > k["near_tie"]
    assert (k <= 1.0).all(), "a posterior margin times u is bounded by 1"


def test_the_two_screening_scores_are_the_same_function():
    """`selective.confidence` recomputes `rules.screening_score` inline. Two
    copies of one formula drift; this is what holds them together."""
    cols = list(rules.PARASITE_CLASSES)
    rows = [[3.0, 1.0] + [0.0] * 4, [0.0] * 6, [40.0] + [0.0] * 5]
    ev = pd.DataFrame(rows, index=["a", "b", "c"], columns=cols)
    got = selective.confidence(ev)
    for name, row in zip(ev.index, rows):
        d = dict(zip(cols, row))
        total = sum(max(v, 0.0) for v in d.values())
        post = sorted((v / total if total else 0.0) for v in d.values())[::-1]
        expected = (post[0] - post[1]) * rules.screening_score(d)
        assert got[name] == pytest.approx(expected)


def test_the_null_reference_depends_on_the_predictor():
    tight = calibrate.null_reference(np.full(170, 0.3), reps=200, seed=0)
    bimodal = calibrate.null_reference(
        np.r_[np.full(85, 0.02), np.full(85, 0.98)], reps=200, seed=0)
    assert bimodal < tight


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #

def test_the_non_inferiority_limit_holds_its_level():
    """The percentile bootstrap does not, at this endpoint's parameters. At the
    non-inferiority null with n = 116 its type I error was 0.063 to 0.076
    against a nominal 0.05, and because this is half of an intersection-union
    co-primary the whole endpoint inherited that level."""
    rng = np.random.default_rng(0)
    n, margin = 116, 0.05
    p_loss, p_gain = 0.06, 0.01           # true delta = -0.05 = -margin
    p_both = 0.966 - p_loss
    probs = [p_both, p_loss, p_gain, 1 - p_both - p_loss - p_gain]
    base_of = np.array([True, True, False, False])
    cand_of = np.array([True, False, True, False])
    rejected = 0
    reps = 1200
    for r in range(reps):
        d = rng.choice(4, size=n, p=probs)
        rejected += stats.noninferiority(
            base_of[d], cand_of[d], margin, seed=r,
            with_bootstrap=False)["non_inferior"]
    size = rejected / reps
    assert size <= 0.05 + 2 * np.sqrt(0.05 * 0.95 / reps), (
        f"type I error {size:.4f} exceeds the nominal 0.05")


def test_tango_agrees_with_its_own_score_function():
    for b, c, n in ((0, 6, 116), (5, 5, 116), (20, 2, 54), (0, 0, 30)):
        lo = stats.tango_lower(b, c, n)
        if lo <= -1.0:
            continue
        assert stats.tango_score(b, c, n, lo) == pytest.approx(
            stats._z(2 * stats.ALPHA), abs=1e-4)


def test_the_vectorised_tango_matches_the_scalar_one():
    b = np.array([0, 5, 20, 0]); c = np.array([6, 5, 2, 0]); n = 116
    many = stats.tango_lower_many(b, c, n)
    for i in range(len(b)):
        assert many[i] == pytest.approx(stats.tango_lower(int(b[i]), int(c[i]), n),
                                        abs=1e-6)


def test_an_empty_intersection_union_does_not_reject():
    """`all([])` is True, so an IU test with no components rejected."""
    with pytest.raises(BadInput):
        stats.intersection_union({})


def test_p_values_cannot_be_passed_as_intersection_union_components():
    """Evaluated by truthiness the sense inverts exactly: p = 0.9 would
    "reject" and p = 0.0 would not."""
    with pytest.raises(BadInput) as e:
        stats.intersection_union({"a": 0.9, "b": 0.04})
    assert "p-value" in e.value.remedy


def test_a_count_larger_than_its_denominator_is_refused():
    with pytest.raises(BadInput):
        stats.wilson(5, 3)


def test_the_power_calculation_does_not_overflow_at_a_low_discordance():
    """`comb(2048, 1024)` is about 10**614; multiplying it by a float raised
    OverflowError, and a *descriptive* power calculation then destroyed a run
    whose gates, factorial and exact test had all already been computed."""
    assert np.isfinite(power.power_mcnemar(1536, 0.02, 0.75))
    assert power.required_n_mcnemar(0.02, 0.75)["n"] is not None


def test_required_n_returns_the_smallest_n_that_works():
    """The power curve is not monotone in n -- `critical_b` jumps -- so a plain
    bisection settled above the true minimum, by up to 13%."""
    for pi, psi in ((1.0, 0.70), (1.0, 0.75)):
        n = power.required_n_mcnemar(pi, psi)["n"]
        assert power.power_mcnemar(n, pi, psi) >= 0.80
        assert all(power.power_mcnemar(k, pi, psi) < 0.80 for k in range(1, n))


def test_a_permutation_p_value_is_never_exactly_zero():
    """The identity permutation is a valid permutation and is never drawn, so
    the plain proportion had an attainable minimum of zero -- and p = 0.0000 is
    a claim no permutation test can make."""
    rng = np.random.default_rng(1)
    m = (rng.random((150, 10)) < 0.5).astype(float)
    m[:, 0] = 1.0
    r = simulate.winners_curse(pd.DataFrame(m, columns=list("abcdefghij")),
                               reps=199, seed=0)
    assert r.p_value >= 1 / (199 + 1)


def test_level_preserved_uses_the_monte_carlo_bound():
    """`A or B` where A implies B is just B, so the Monte-Carlo guard never
    fired: a run whose upper 95% bound was 0.0714 reported its level as
    preserved -- on the one boolean this package uses to defend its own
    selection rule."""
    r = simulate.selection_null(54, [0.5, 0.5, 0.5], reps=400, seed=3,
                                criterion="headroom")
    if r["size_after_selection"] <= 0.05 < r["upper_95"]:
        assert r["level_preserved"] is False
        assert "not resolved" in r["level_verdict"] or \
            r["level_verdict"] == "not preserved"


# --------------------------------------------------------------------------- #
# Gates
# --------------------------------------------------------------------------- #

def test_the_reconstruction_gate_checks_its_denominator():
    """135 of 1,000 patients -- 13.5% -- passed a gate defined against a
    published 135 of 170, because both have the same numerator."""
    assert gates.reconstruction_gate(135, 170).passed is True
    bad = gates.reconstruction_gate(135, 1000)
    assert bad.passed is False
    assert "not comparable" in bad.message


def test_headroom_refuses_a_missing_prediction():
    """NaN != "uninfected" is True, so an unscored patient became a baseline
    false positive -- inventing headroom that does not exist and bypassing the
    "no win is available" refusal it feeds."""
    truth = pd.Series([UNINFECTED] * 3, index=["a", "b", "c"])
    partial = pd.Series([UNINFECTED, UNINFECTED], index=["a", "b"])
    with pytest.raises(BadInput):
        gates.headroom(truth, partial)


def test_the_design_gate_counts_discordance_among_the_uninfected():
    """The specificity McNemar it protects conditions on that subset; counting
    over all patients reported a number for a different test."""
    truth = pd.Series([UNINFECTED, UNINFECTED, UNINFECTED, "P_vivax",
                       "P_vivax", "P_vivax"], index=list("abcdef"))
    base = pd.Series([UNINFECTED, "P_vivax", "P_vivax", "P_vivax",
                      "P_vivax", "P_vivax"], index=list("abcdef"))
    cand = pd.Series([UNINFECTED, UNINFECTED, "P_vivax", UNINFECTED,
                      "P_vivax", "P_vivax"], index=list("abcdef"))
    g = gates.design_gate(truth, base, cand, checkpoint="ck", rule="R4")
    assert g.detail["discordant"] == 1
    assert g.detail["discordant_all_patients"] == 2
    assert "uninfected" in g.detail["discordant_denominator"]


def test_max_discordance_without_counts_is_refused_not_substituted():
    """`chosen or by_headroom` silently substituted the max-HEADROOM arm -- the
    criterion documented as not level-preserving -- while returning
    "max_discordance" as the criterion used."""
    with pytest.raises(BadInput):
        gates.select_checkpoint({"a": 9, "b": 2}, None,
                                criterion="max_discordance")


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def test_the_error_taxonomy_reads_the_detections_it_describes():
    """`view` was a parameter that the body never referenced, so every false
    positive was labelled "many low confidence" from the label pair alone --
    including a patient whose call rested on one box at 0.99."""
    from mag.schema import ALL_CLASSES

    def view(n, score):
        sc = {c: np.zeros(0) for c in ALL_CLASSES}
        sc["P_falciparum"] = np.full(n, score)
        return evidence.PatientView("P", 1, scores=sc, per_field_max={},
                                    per_field_count={})

    assert diagnostics._category(UNINFECTED, "P_falciparum", view(1, 0.99),
                                 "P_falciparum") == "few_high_confidence"
    assert diagnostics._category(UNINFECTED, "P_falciparum", view(40, 0.05),
                                 "P_falciparum") == "many_low_confidence"


def test_an_all_uninfected_cohort_does_not_crash_the_error_table():
    """A bare `.idxmax()` raised "attempt to get argmax of an empty sequence",
    which is not a MagError and reached the operator as a traceback."""
    patients = _pat([("a", UNINFECTED, 1), ("b", UNINFECTED, 1)])
    truth = patients.set_index("patient_id")["truth"]
    pred = pd.Series([UNINFECTED, "P_vivax"], index=["a", "b"])
    from mag.schema import ALL_CLASSES

    views = {p: evidence.PatientView(p, 1,
                                     scores={c: np.zeros(0) for c in ALL_CLASSES},
                                     per_field_max={}, per_field_count={})
             for p in ("a", "b")}
    tbl = diagnostics.error_table(truth, pred, patients, views, "R0")
    assert len(tbl) == 2
    assert (tbl.loc[tbl["pred"] == "P_vivax", "category"] != "").all()


def test_the_domain_contrast_refuses_a_missing_prediction():
    patients = _pat([("a", "P_vivax", 1), ("b", "P_ovale", 1)])
    truth = patients.set_index("patient_id")["truth"]
    full = pd.Series(["P_vivax", "P_ovale"], index=["a", "b"])
    partial = pd.Series(["P_vivax"], index=["a"])
    with pytest.raises(BadInput):
        diagnostics.domain_contrast(truth, full, partial, patients)


def test_an_empty_checkpoint_does_not_fabricate_a_census():
    """`100.0 * (1 - 0/max(0,1))` reported that nothing was suppressed as
    100%, and a 1e-9 floor made the headline ratio 999,999,999.9999999."""
    empty = _det([])
    one = _det([("A", "i0", "ck", "RBC", 0.9, 0.5, 0.5, 0.1, 0.1)])
    census = diagnostics.box_census({"a": empty, "b": one}, 1)
    assert np.isnan(census.loc[census.checkpoint == "a", "suppressed_pct"].iloc[0])
    assert np.isinf(census.attrs["ratio_raw"])


def test_the_field_curve_samples_the_fields_a_patient_actually_has():
    """Only fields carrying a surviving detection appeared in the sampler's
    universe, so a patient declaring 40 fields of which 3 held a box offered 3
    to sample from: "examine 25 fields" returned 3 and the curve was flat by
    construction."""
    patients = fixtures.patient_table(n_fields=(30, 31), seed=31)
    det = fixtures.detections(patients, "ck", seed=31)
    # keep one field per patient: the declared count stays 30
    keep = det.groupby("patient_id", observed=True)["image_id"].transform("first")
    sparse = det[det["image_id"] == keep]
    index = diagnostics._index_by_field(
        sparse, patients["patient_id"].astype(str).tolist(),
        {str(r.patient_id): int(r.n_images) for r in patients.itertuples(index=False)})
    any_pid = patients["patient_id"].iloc[0]
    assert len(index[str(any_pid)]) == int(patients["n_images"].iloc[0]), (
        "the universe is the declared field count, padded with empty fields")


def test_a_subsampled_view_reports_the_fields_examined():
    """`n_fields` was the number that carried a detection, not the number
    examined, so R5 -- whose screen is a share of fields -- read 1.0 where
    `build_views` read 0.2: an opposite call on the same data."""
    from mag.schema import ALL_CLASSES

    v = diagnostics._view_from_fields("P", [{"P_falciparum": np.array([0.9])}],
                                      ALL_CLASSES, n_fields=10)
    assert v.n_fields == 10


# --------------------------------------------------------------------------- #
# The cache and the chain of custody
# --------------------------------------------------------------------------- #

def test_a_csv_cache_verifies_against_its_own_manifest(tmp_path):
    """Pandas' default float parser is off by one unit in the last place on
    many values, and for some of them that changes the tenth significant digit
    -- the precision the content hash pins. Every CSV-fallback cache failed its
    own integrity check on the next load, and the refusal blamed tampering."""
    patients = fixtures.patient_table(n_fields=(2, 3), seed=41)
    det = fixtures.detections(patients, "ck", seed=41)[list(DETECTION_COLUMNS)]
    root = tmp_path / "c"
    root.mkdir()
    det.to_csv(root / "detections_ck.csv", index=False, lineterminator="\n")
    (root / "manifest_ck.json").write_text(json.dumps({
        "checkpoint": "ck", "checkpoint_sha256": "a" * 64, "conf_floor": 0.001,
        "nms": "none", "iou": None, "imgsz": 640,
        "n_images": int(det["image_id"].nunique()), "n_detections": len(det),
        "content_sha256": schema.content_hash(det, list(DETECTION_COLUMNS))}))
    got, _ = cache_mod.load(root, "ck", verify=True)
    assert len(got) == len(det)


def test_a_truncated_table_is_named_rather_than_traced(tmp_path):
    patients = fixtures.patient_table(n_fields=(2, 3), seed=42)
    det = fixtures.detections(patients, "ck", seed=42)[list(DETECTION_COLUMNS)]
    root = tmp_path / "c"
    root.mkdir()
    p = root / "detections_ck.parquet"
    det.to_parquet(p, index=False)
    p.write_bytes(p.read_bytes()[: len(p.read_bytes()) // 2])
    (root / "manifest_ck.json").write_text(json.dumps({
        "checkpoint": "ck", "checkpoint_sha256": "a" * 64, "conf_floor": 0.001,
        "nms": "none", "iou": None, "imgsz": 640, "n_images": 1,
        "n_detections": 1, "content_sha256": "x" * 64}))
    with pytest.raises(IntegrityError) as e:
        cache_mod.load(root, "ck")
    assert "truncated or corrupt" in e.value.remedy


def test_a_torn_manifest_is_named_rather_than_traced(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "manifest_ck.json").write_text('{"checkpoint": "ck", "conf_flo')
    with pytest.raises(IntegrityError):
        cache_mod.Manifest.load(root / "manifest_ck.json")


def test_a_manifest_missing_required_fields_is_named_rather_than_traced(tmp_path):
    root = tmp_path / "c"
    root.mkdir()
    (root / "manifest_ck.json").write_text('{"checkpoint": "ck"}')
    with pytest.raises(IntegrityError) as e:
        cache_mod.Manifest.load(root / "manifest_ck.json")
    assert "missing required fields" in e.value.problem


def test_a_stray_copy_beside_the_cache_does_not_abort_the_analysis(tmp_path):
    """`detections_rtdetr.backup-2024.parquet` was announced as a checkpoint
    called `rtdetr.backup-2024`, whose missing manifest then aborted the run."""
    cache, _ = fixtures.write_cache(tmp_path, checkpoints=("rtdetr",),
                                    n_fields=(2, 3), seed=43)
    src = cache / "detections_rtdetr.parquet"
    (cache / "detections_rtdetr.backup-2024.parquet").write_bytes(src.read_bytes())
    assert cache_mod.available(cache) == ["rtdetr"]
    assert cache_mod.orphans(cache)["table_without_manifest"] == \
        ["rtdetr.backup-2024"]


def test_a_manifest_with_no_checkpoint_hash_is_refused(tmp_path):
    """`elif sha and ...` skipped the cross-check when the manifest's hash was
    empty AND recorded no reason, so the cache was declared usable."""
    cache, pcsv = fixtures.write_cache(tmp_path, checkpoints=("ck",),
                                       provenanced=True, n_fields=(2, 3), seed=44)
    man = json.loads((cache / "manifest_ck.json").read_text())
    man["checkpoint_sha256"] = ""
    (cache / "manifest_ck.json").write_text(json.dumps(man))
    _, m = cache_mod.load(cache, "ck", verify=False)
    prov = provenance.inspect(cache, "set", {"ck": m}, "hash", 170, "matched")
    assert not prov.usable
    assert any("no checkpoint hash" in r for r in prov.reasons)


def test_empty_shard_stubs_do_not_count_as_an_inference_pass(tmp_path):
    """`touch anything.parquet` satisfied the one artefact only a real pass
    leaves. Three stubs beside a 170-patient table were indistinguishable from
    a forgery, because shards were counted by filename."""
    cache, pcsv = fixtures.write_cache(tmp_path, checkpoints=("ck",),
                                       provenanced=True, n_fields=(2, 3), seed=45)
    shards = sorted((cache / "_shards" / "ck").glob("*.parquet"))
    for q in shards:
        q.write_bytes(b"")
    _, m = cache_mod.load(cache, "ck", verify=False)
    prov = provenance.inspect(cache, "set", {"ck": m}, "hash", 170, "matched")
    assert not prov.usable
    assert any("non-empty inference shards" in r for r in prov.reasons)

    for q in shards[:3]:
        q.write_bytes(b"\x00shard")
    prov = provenance.inspect(cache, "set", {"ck": m}, "hash", 170, "matched")
    assert not prov.usable, "3 shards for 170 patients is not a pass"


def test_the_provenance_line_names_the_patient_table_and_the_tables(tmp_path):
    """Two runs on DIFFERENT patient tables produced identical lines -- 23
    patients relabelled, a different oracle bound -- and `figures.check`, which
    exists to catch exactly that, passed."""
    cache, _ = fixtures.write_cache(tmp_path, checkpoints=("ck",),
                                    provenanced=True, n_fields=(2, 3), seed=46)
    _, m = cache_mod.load(cache, "ck", verify=False)
    a = provenance.inspect(cache, "set", {"ck": m}, "hash_A", 170, "matched")
    b = provenance.inspect(cache, "set", {"ck": m}, "hash_B", 170, "matched")
    assert a.line() != b.line()
    assert m.content_sha256[:8] in a.line()


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #

def test_the_exchangeability_flag_is_derived_from_its_arguments():
    """It read two module constants, so it was `True` for every possible call
    -- a literal with extra steps, on the one guard that keeps an invalid
    p-value off the page."""
    truth = pd.Series([UNINFECTED] * 4 + ["P_vivax"] * 4,
                      index=[f"p{i}" for i in range(8)])
    a = pd.Series([UNINFECTED] * 8, index=truth.index)
    b = pd.Series([UNINFECTED] * 3 + ["P_vivax"] * 5, index=truth.index)
    fixed = study._confirm(truth, a, b, checkpoint="ck", seed=0)
    assert fixed["arms_are_fixed"] is True
    fitted = study._confirm(truth, a, b, checkpoint="ck", seed=0,
                            candidate_rule="R9iso")
    assert fitted["arms_are_fixed"] is False
    assert fitted["mcnemar"]["valid"] is False
    assert "not exchangeable" in fitted["mcnemar"]["reason"]


def test_a_figure_that_raises_is_a_failure_not_an_absence(tmp_path):
    """`except Exception` put a coding bug and a legitimately absent analysis
    in one list, and `mag all` went on to write the manuscript and return 0."""
    s = study.Study()
    s.factorial = pd.DataFrame([{"checkpoint": "a", "rule": "R0",
                                 "family": "published", "fitted": False,
                                 "accuracy": float("nan")}])
    s.meta = {"condition": "matched", "seed": 0}
    stamp = figures.write_all(s, tmp_path)
    assert "failed" in stamp
    assert any("factorial_heatmap" in f for f in stamp["failed"])
    with pytest.raises(ProvenanceError):
        figures.check(tmp_path, s)


def test_figures_from_a_different_seed_are_refused(tmp_path):
    class Prov:
        def line(self):
            return "one run"

    s = study.Study()
    s.factorial = pd.DataFrame([{"checkpoint": "a", "rule": "R0",
                                 "family": "published", "fitted": False,
                                 "accuracy": 0.5}])
    s.meta = {"condition": "matched", "seed": 0}
    s.provenance = Prov()
    figures.write_all(s, tmp_path)
    s.meta["seed"] = 99
    with pytest.raises(ProvenanceError) as e:
        figures.check(tmp_path, s)
    assert "different seeds" in e.value.problem


def test_the_command_line_refuses_arguments_that_disable_cross_fitting(tmp_path):
    from mag import cli

    cache, pcsv = fixtures.write_cache(tmp_path, checkpoints=("ck",),
                                       provenanced=True, n_fields=(2, 3), seed=47)
    for flag, value in (("--folds", "1"), ("--seed", "-1"), ("--reps", "0")):
        rc = cli.main(["run", "--cache", str(cache), "--patients", str(pcsv),
                       "--out", str(tmp_path / "o"), flag, value,
                       "--force-reconstruction"])
        assert rc == 3, f"{flag} {value} was accepted"
