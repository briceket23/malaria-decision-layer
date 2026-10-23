"""Calibration and the fitted rules.

Every test here is either a mathematical property that must hold, or a defect
that was found by running the code and is now pinned. The docstrings say which.
"""

from __future__ import annotations

import numpy as np
import pytest

from mag import calibrate, rules


# --------------------------------------------------------------------------- #
# The three maps
# --------------------------------------------------------------------------- #

def test_isotonic_is_non_decreasing():
    rng = np.random.default_rng(0)
    x = rng.random(200)
    y = (rng.random(200) < x).astype(float)
    kx, ky = calibrate.pav(x, y)
    assert np.all(np.diff(ky) >= -1e-12)
    assert np.all(np.diff(kx) > 0), "knots must be strictly increasing"


def test_isotonic_expands_by_knot_span_not_by_observation_weight():
    """A confirmed defect, and the reason the blocks carry three fields.

    With ties in x, a block's observation weight and the number of knots it
    covers are different numbers. Expanding by the former scrambles the map on
    exactly the tied scores this study is full of -- it moved the isotonic ICI
    from 0.036 to 0.40 -- and leaves it well formed enough to go unnoticed.
    """
    x = np.array([0.1, 0.1, 0.1, 0.1, 0.2, 0.9])
    y = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
    kx, ky = calibrate.pav(x, y)
    assert len(kx) == len(ky) == 3, "one knot per distinct x"
    assert ky[0] == pytest.approx(0.25)
    assert np.all(np.diff(ky) >= -1e-12)


def test_isotonic_recovers_a_monotone_distortion():
    rng = np.random.default_rng(1)
    p = rng.random(2000)
    y = (rng.random(2000) < p).astype(float)
    squashed = p ** 3                                  # monotone, badly scaled
    kx, ky = calibrate.pav(squashed, y)
    fitted = calibrate.apply_pav(kx, ky, squashed)
    assert calibrate.brier(fitted, y) < calibrate.brier(squashed, y)


def test_platt_does_not_diverge_on_separable_data():
    """A confirmed defect. Undamped Newton returned a slope of -3.1e9.

    Perfect separation has no finite maximum likelihood, so an undamped solver
    walks the coefficients toward infinity and returns whatever it reached when
    the loop ran out. The backtracking line search stops instead.
    """
    s = np.concatenate([np.linspace(0.0, 0.4, 50), np.linspace(0.6, 1.0, 50)])
    y = np.concatenate([np.zeros(50), np.ones(50)])
    a, b = calibrate.platt(s, y)
    assert np.isfinite(a) and np.isfinite(b)
    assert abs(a) < 1e4, f"slope {a} means the solver diverged"
    p = calibrate.apply_platt(a, b, s)
    assert p[:50].mean() < 0.5 < p[50:].mean()


def test_platt_never_increases_the_negative_log_likelihood():
    rng = np.random.default_rng(2)
    s = rng.random(300)
    y = (rng.random(300) < s).astype(float)
    a, b = calibrate.platt(s, y)
    assert calibrate._nll(a, b, s, y) <= calibrate._nll(1.0, 0.0, s, y) + 1e-9


def test_platt_returns_the_identity_when_one_class_is_absent():
    s = np.linspace(0, 1, 20)
    assert calibrate.platt(s, np.ones(20)) == (1.0, 0.0)


def test_temperature_is_positive_and_moves_the_right_way():
    """Overconfident scores need T > 1; timid scores need T < 1."""
    rng = np.random.default_rng(3)
    y = (rng.random(500) < 0.5).astype(float)
    logit = rng.normal(0, 1, 500) + 2.0 * (y - 0.5)
    over = 1.0 / (1.0 + np.exp(-4.0 * logit))          # too extreme
    timid = 1.0 / (1.0 + np.exp(-0.25 * logit))        # too flat
    assert calibrate.temperature(over, y) > 1.0
    assert calibrate.temperature(timid, y) < 1.0


def test_a_perfectly_calibrated_score_is_left_alone():
    rng = np.random.default_rng(4)
    p = rng.random(4000)
    y = (rng.random(4000) < p).astype(float)
    assert calibrate.temperature(p, y) == pytest.approx(1.0, abs=0.15)
    a, _ = calibrate.slope_intercept(p, y)
    assert a == pytest.approx(1.0, abs=0.15)


# --------------------------------------------------------------------------- #
# Cross-fitting
# --------------------------------------------------------------------------- #

def test_a_patient_is_never_scored_by_a_map_that_saw_it():
    """The property the whole module exists for, checked directly.

    A patient's fold is flipped and the calibrated value must not move if the
    map really was fitted without it -- so instead the check is structural: two
    runs whose only difference is one patient's LABEL must agree on every
    patient assigned to a different fold from that one.
    """
    rng = np.random.default_rng(5)
    s = rng.random(120)
    y = (rng.random(120) < 0.6).astype(float)
    assign = calibrate.folds(120, 10, 0)
    victim = 0
    y2 = y.copy()
    y2[victim] = 1.0 - y2[victim]
    a = calibrate.crossfit(s, y, "isotonic", k=10, seed=0)
    b = calibrate.crossfit(s, y2, "isotonic", k=10, seed=0)
    same_fold = assign == assign[victim]
    # every patient in the victim's own fold is scored by a map fitted WITHOUT
    # the victim, so flipping the victim's label cannot move any of them
    assert np.allclose(a[same_fold], b[same_fold]), (
        "a patient's own fold must not see its label")
    # and every other fold's map does train on the victim, so at least one of
    # them must move -- otherwise the cross-fitting is not happening at all
    assert not np.allclose(a[~same_fold], b[~same_fold]), (
        "changing one patient's label must move the maps fitted on it")


def test_leave_one_out_is_available_and_is_the_documented_default_of_nothing():
    n = 40
    assert np.array_equal(calibrate.folds(n, n, 0), np.arange(n))
    assert np.array_equal(calibrate.folds(n, n + 5, 0), np.arange(n))
    assert calibrate.DEFAULT_FOLDS == 10


def test_folds_are_deterministic_and_balanced():
    a = calibrate.folds(170, 10, seed=7)
    b = calibrate.folds(170, 10, seed=7)
    assert np.array_equal(a, b)
    counts = np.bincount(a)
    assert counts.max() - counts.min() <= 1


def test_a_degenerate_fold_falls_back_to_the_identity_and_is_counted():
    s = np.linspace(0, 1, 20)
    y = np.ones(20)                                   # one class only
    cf = calibrate.crossfit_detail(s, y, "platt", k=5, seed=0)
    assert cf.n_folds_fitted == 0
    assert cf.n_folds_degenerate == 5
    assert cf.degenerate
    assert np.allclose(cf.calibrated, s), "identity is the honest fallback"


def test_an_unknown_calibrator_is_refused_with_a_remedy():
    from mag.errors import BadInput

    with pytest.raises(BadInput) as e:
        calibrate.crossfit(np.zeros(5), np.zeros(5), "magic")
    assert "isotonic" in e.value.remedy


def test_the_threshold_comes_from_the_training_folds_too():
    """Fitting the map out of fold and picking the threshold on everybody moves
    the leak rather than removing it, so both are cross-fitted."""
    rng = np.random.default_rng(6)
    s = rng.random(100)
    y = (rng.random(100) < s).astype(float)
    cf = calibrate.crossfit_detail(s, y, "platt", k=5, seed=0,
                                   target_sensitivity=0.9)
    assert cf.thresholds.shape == s.shape
    # patients in different folds may carry different thresholds; patients in
    # the same fold must carry the same one
    for f in np.unique(cf.assign):
        assert len(np.unique(cf.thresholds[cf.assign == f])) == 1


def test_the_sensitivity_threshold_meets_its_target():
    p = np.linspace(0, 1, 101)
    y = (p > 0.4).astype(float)
    t = calibrate.sensitivity_threshold(p, y, 0.95)
    assert float((p[y > 0.5] >= t).mean()) >= 0.95


def test_an_unreachable_sensitivity_target_returns_zero_not_a_guess():
    assert calibrate.sensitivity_threshold(np.zeros(0), np.zeros(0), 0.95) == 0.0


# --------------------------------------------------------------------------- #
# Measures of a probability
# --------------------------------------------------------------------------- #

def test_equal_mass_binning_not_equal_width():
    """A confirmed defect in the predecessor. With scores piled near one, an
    equal-width binning puts nearly everybody in the last bin and reports a
    calibration error of nearly zero for a badly calibrated predictor."""
    p = np.concatenate([np.full(190, 0.99), np.linspace(0.0, 0.5, 10)])
    y = np.concatenate([np.zeros(190), np.ones(10)])   # confidently wrong
    ece, detail = calibrate.ece_equal_mass(p, y, bins=5)
    assert ece > 0.5, "an equal-width binning would have hidden this"
    assert len({d["n"] for d in detail}) <= 2, "bins hold equal counts"


def test_the_null_reference_is_not_zero_at_a_finite_sample():
    rng = np.random.default_rng(0)
    p = rng.beta(2, 1, 170)
    assert calibrate.null_reference(p, bins=5, reps=100, seed=0) > 0.0


def test_the_null_reference_describes_the_predictor_it_is_a_reference_for():
    """A confirmed defect. It drew p from Uniform(0, 1) and rescaled toward the
    observed prevalence, which at 0.68 pinned 27% of draws at exactly 1.0 and
    missed the target by five points -- and reported the same number for a
    tight predictor and a bimodal one whose true nulls differ by 70%."""
    tight = np.full(170, 0.30)
    bimodal = np.r_[np.full(85, 0.02), np.full(85, 0.98)]
    a = calibrate.null_reference(tight, bins=5, reps=200, seed=0)
    b = calibrate.null_reference(bimodal, bins=5, reps=200, seed=0)
    assert a > 0 and b > 0
    assert a != b, "the reference must depend on the predictor's own spread"
    # a confident, well-separated predictor has less room to look miscalibrated
    assert b < a


def test_ici_is_zero_for_a_calibrated_predictor_and_large_for_a_wrong_one():
    rng = np.random.default_rng(8)
    p = rng.random(3000)
    y = (rng.random(3000) < p).astype(float)
    assert calibrate.ici(p, y) < 0.06
    assert calibrate.ici(np.clip(p ** 4, 0, 1), y) > 0.06


def test_report_is_finite_on_a_degenerate_input():
    r = calibrate.report(np.full(10, 0.5), np.ones(10))
    assert np.isfinite(r.brier) and np.isfinite(r.ece)
    assert np.isnan(r.ici), "one class: no isotonic smoother is defined"


# --------------------------------------------------------------------------- #
# The fitted rules
# --------------------------------------------------------------------------- #

def test_no_fitted_rule_is_in_the_fixed_registry():
    """The single property that keeps an invalid p-value off the page."""
    assert not (set(rules.FITTED) & set(rules.RULES))
    assert not (set(rules.FITTED) & rules.FIXED_RULES)
    for rid in rules.fitted_ids():
        assert rid not in rules.rule_ids()


def test_every_fitted_rule_says_what_it_was_fitted_on():
    for rid in rules.fitted_ids():
        d = rules.describe(rid)
        assert d["fitted"] is True
        assert d["fitted_on"] != "nothing"
        assert "cross-fitted" in d["fitted_on"]


def test_a_fitted_prediction_carries_the_flag_that_blocks_an_exact_test():
    import fixtures
    from mag import evidence

    patients = fixtures.patient_table(n_fields=(3, 5), seed=1)
    det = fixtures.detections(patients, "rtdetr", seed=1)
    views = evidence.build_views(det, patients, "matched")
    order = patients["patient_id"].tolist()
    truth = patients.set_index("patient_id")["truth"]
    for rid in rules.fitted_ids():
        _, p, _, detail = rules.fit_predict(views, order, truth, rid, k=5)
        assert detail["exact_test_valid"] is False
        assert detail["why_not"]
        assert np.all((p >= 0.0) & (p <= 1.0))


def test_a_fitted_rule_predicts_every_patient():
    import fixtures
    from mag import evidence

    patients = fixtures.patient_table(n_fields=(3, 5), seed=2)
    det = fixtures.detections(patients, "rtdetr", seed=2)
    views = evidence.build_views(det, patients, "matched")
    order = patients["patient_id"].tolist()
    truth = patients.set_index("patient_id")["truth"]
    pred, _, _, _ = rules.fit_predict(views, order, truth, "R9iso", k=5)
    assert list(pred.index) == order
    assert not pred.isna().any()


def test_the_staged_rule_never_calls_a_patient_stage_two_would_not():
    """Stage two both confirms and speciates, so a patient the staged rule
    calls infected must carry exactly the label the confirming rule gives it.

    That is the checkable content of "screen then confirm". A staged rule that
    could return a label its own confirmation step disagreed with would be a
    third rule wearing two names.
    """
    import fixtures
    from mag import evidence
    from mag.schema import UNINFECTED

    patients = fixtures.patient_table(n_fields=(3, 5), seed=3)
    det = fixtures.detections(patients, "rtdetr", seed=3)
    views = evidence.build_views(det, patients, "matched")
    order = patients["patient_id"].tolist()
    truth = patients.set_index("patient_id")["truth"]
    staged, _, _, detail = rules.fit_predict(views, order, truth, "R13staged", k=5)
    confirm, _, _, _ = rules.predict(views, order, detail["confirm_rule"])

    assert detail["confirm_rule"] == "R4"
    infected = staged != UNINFECTED
    assert infected.any(), "a fixture where nothing is called leaves this untested"
    assert (staged[infected] == confirm[infected]).all(), (
        "an infected call must be the confirming rule's own label")
    # and the staged rule is a strict narrowing of its confirmation step: it can
    # withhold a call the confirming rule would make, never add one
    assert not ((confirm == UNINFECTED) & infected).any()
