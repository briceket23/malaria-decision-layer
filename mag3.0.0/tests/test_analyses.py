"""Selective prediction, diagnostics, power, the nulls, the sweep, the figures.

The analyses the reviewers asked for. Each one is checked against a property it
must have, and several are checked against the specific way an earlier version
got it wrong -- those tests name the defect in their docstring, because a test
whose reason for existing is not written down is a test that gets deleted the
first time it is inconvenient.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import fixtures
from mag import (diagnostics, evidence, figures, power, rules, search,
                 selective, simulate)
from mag.schema import UNINFECTED


@pytest.fixture(scope="module")
def cohort():
    patients = fixtures.patient_table(n_fields=(6, 10), seed=11)
    det = fixtures.detections(patients, "rtdetr", seed=11)
    views = evidence.build_views(det, patients, "matched")
    order = patients["patient_id"].tolist()
    truth = patients.set_index("patient_id")["truth"]
    return patients, det, views, order, truth


# --------------------------------------------------------------------------- #
# Selective prediction
# --------------------------------------------------------------------------- #

def test_coverage_is_taken_by_rank_not_by_quantile_value():
    """A confirmed defect. Forty of 170 patients sat at exactly zero
    confidence, so every quantile threshold retained the whole cohort and the
    reported risk at 90% coverage was the full-cohort risk with "0 patients
    referred" printed beside it."""
    pred = pd.Series(["a"] * 100)
    conf = pd.Series([0.0] * 60 + list(np.linspace(0.1, 1.0, 40)))
    kept, meta = selective.apply_coverage(pred, conf, 0.9)
    assert meta["n_referred"] == 10
    assert (kept == selective.ABSTAIN).sum() == 10
    assert meta["coverage"] == pytest.approx(0.9)


def test_every_coverage_target_refers_the_right_number():
    pred = pd.Series(["a"] * 170)
    conf = pd.Series(np.zeros(170))                 # every value identical
    for target in selective.COVERAGE_TARGETS:
        _, meta = selective.apply_coverage(pred, conf, target)
        assert meta["n_referred"] == 170 - int(np.floor(target * 170 + 1e-9))


def test_risk_falls_as_coverage_falls_when_confidence_is_informative():
    idx = [f"P{i}" for i in range(200)]
    rng = np.random.default_rng(0)
    conf = pd.Series(rng.random(200), index=idx)
    truth = pd.Series(["x"] * 200, index=idx)
    wrong = conf < 0.3                              # the least confident err
    pred = pd.Series(np.where(wrong, "y", "x"), index=idx)
    rc = selective.risk_coverage(pred, truth, conf)
    assert rc.operating["0.8"]["risk"] <= rc.operating["0.95"]["risk"]
    assert 0.0 <= rc.aurc <= 1.0


def test_a_confidence_that_knows_nothing_gives_a_flat_curve():
    """The negative control. Under a rule that calls everyone infected, the
    dominant error arrives with high confidence and no threshold separates it,
    which is the case the triage argument has to survive."""
    idx = [f"P{i}" for i in range(200)]
    rng = np.random.default_rng(1)
    conf = pd.Series(rng.random(200), index=idx)
    truth = pd.Series(["x"] * 200, index=idx)
    pred = pd.Series(np.where(rng.random(200) < 0.3, "y", "x"), index=idx)
    rc = selective.risk_coverage(pred, truth, conf)
    assert abs(rc.operating["0.8"]["risk"] - rc.operating["0.95"]["risk"]) < 0.15


def test_confidence_is_zero_when_there_is_no_evidence(cohort):
    _, _, _, _, _ = cohort
    ev = pd.DataFrame(0.0, index=["P1", "P2"],
                      columns=list(rules.PARASITE_CLASSES))
    assert (selective.confidence(ev) == 0.0).all()


def test_confidence_rises_with_both_margin_and_evidence():
    cols = list(rules.PARASITE_CLASSES)
    def row(vals):
        d = dict(zip(cols, vals + [0.0] * (len(cols) - len(vals))))
        return d
    ev = pd.DataFrame([row([1.0, 0.9]), row([1.0, 0.0]), row([10.0, 0.0])],
                      index=["close", "clear", "clear_and_plentiful"])
    k = selective.confidence(ev)
    assert k["close"] < k["clear"] < k["clear_and_plentiful"]


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def test_the_box_census_detects_a_fixed_query_budget():
    """RT-DETR emits a fixed number of queries per field, the same on a
    parasite-dense smear as on a clean one. That is not a detection count and
    the census has to say so."""
    rows = []
    for img in range(20):
        for i in range(7):                          # exactly seven, always
            rows.append(("P1", f"i{img}", "rtdetr", "RBC", 0.5, 0.1, 0.1, 0.05, 0.05))
    det = pd.DataFrame(rows, columns=["patient_id", "image_id", "checkpoint",
                                      "cls", "score", "x", "y", "w", "h"])
    census = diagnostics.box_census({"rtdetr": det}, 20)
    assert bool(census.iloc[0]["fixed_budget"]) is True
    assert census.iloc[0]["per_field_sd"] == 0.0


def test_the_box_census_reports_the_ratio_before_and_after_suppression(cohort):
    patients, det, _, _, _ = cohort
    census = diagnostics.box_census({"a": det, "b": det}, int(patients["n_images"].sum()))
    assert census.attrs["ratio_raw"] == pytest.approx(1.0)
    assert (census["matched_boxes"] <= census["raw_boxes"]).all()
    assert (census["suppressed_pct"] > 0).all(), (
        "the fixture emits duplicate boxes; if none is suppressed the matched "
        "path is not being exercised at all")


def test_fragility_counts_the_boxes_that_would_change_a_call(cohort):
    _, _, views, order, _ = cohort
    vals = [diagnostics.fragility(views[str(p)], "R0") for p in order[:20]]
    finite = [v for v in vals if np.isfinite(v)]
    assert finite, "no patient had a computable fragility"
    assert all(v >= 1 for v in finite)


def test_fragility_says_infinite_rather_than_n_when_nothing_flips():
    """The distinction matters. Returning n conflates "survives everything"
    with "flips only after all n are gone", and those are opposite readings."""
    from mag.schema import ALL_CLASSES

    empty = evidence.PatientView(
        patient_id="P", n_fields=1,
        scores={c: np.zeros(0) for c in ALL_CLASSES},
        per_field_max={}, per_field_count={})
    assert np.isnan(diagnostics.fragility(empty, "R0")), (
        "no detections of the chosen class: the question is not defined")

    # R1 calls the patient by its single strongest box; remove that box and the
    # rule has nothing left, so this one does flip and flips at k = 1
    scores = {c: np.zeros(0) for c in ALL_CLASSES}
    scores["P_falciparum"] = np.array([0.9])
    one = evidence.PatientView(patient_id="P", n_fields=1, scores=scores,
                               per_field_max={}, per_field_count={})
    assert diagnostics.fragility(one, "R1") == 1.0

    # the case the return value exists for. R4 requires two detections above
    # 0.5; this patient has five, all at 0.2, so it is already called
    # uninfected and removing them one at a time never changes that. The old
    # code returned 5 here, indistinguishable from a call that survived four
    # removals and fell on the fifth.
    scores2 = {c: np.zeros(0) for c in ALL_CLASSES}
    scores2["P_falciparum"] = np.full(5, 0.2)
    immovable = evidence.PatientView(patient_id="P", n_fields=1, scores=scores2,
                                     per_field_max={}, per_field_count={})
    assert rules.RULES["R4"].apply(immovable)["label"] == "uninfected"
    assert np.isinf(diagnostics.fragility(immovable, "R4"))


def test_fragility_refuses_a_rule_whose_evidence_it_cannot_rebuild():
    """R5 votes per field. Fragility removes detections without knowing which
    field each came from, so it cannot rebuild the per-field maxima and would
    silently score the rule against stale ones."""
    from mag.errors import BadInput
    from mag.schema import ALL_CLASSES

    v = evidence.PatientView(patient_id="P", n_fields=1,
                             scores={c: np.zeros(1) for c in ALL_CLASSES},
                             per_field_max={}, per_field_count={})
    with pytest.raises(BadInput) as e:
        diagnostics.fragility(v, "R5")
    assert "per-field" in e.value.remedy
    assert "R5" not in diagnostics.SCORE_ONLY_RULES


def test_the_error_table_covers_every_patient_and_names_a_reason(cohort):
    patients, _, views, order, truth = cohort
    pred, _, _, _ = rules.predict(views, order, "R0")
    tbl = diagnostics.error_table(truth.reindex(order), pred, patients, views, "R0")
    assert len(tbl) == len(order)
    assert set(tbl.columns) >= {"truth", "pred", "correct", "category",
                                "fragility", "domain"}
    wrong = tbl[~tbl["correct"]]
    assert (wrong["category"] != "").all(), "every error must carry a reason"
    assert (tbl[tbl["correct"]]["category"] == "").all()


def test_the_domain_contrast_refuses_to_touch_specificity(cohort):
    patients, _, views, order, truth = cohort
    a, _, _, _ = rules.predict(views, order, "R0")
    b, _, _, _ = rules.predict(views, order, "R4")
    dd = diagnostics.domain_contrast(truth.reindex(order), a, b, patients)
    assert "speciation only" in dd["note"]
    assert dd["n_internal"] + dd["n_external"] == int((truth != UNINFECTED).sum())


def test_the_field_curve_refuses_a_denominator_it_cannot_defend(cohort):
    patients, det, _, order, truth = cohort
    out = diagnostics.field_curve(det, patients, truth, ["R0"],
                                  condition="matched", min_fields=999, reps=2)
    assert not out["by_rule"]
    assert "too few patients" in out["note"]


def test_the_field_curve_holds_its_patients_fixed_across_the_curve(cohort):
    patients, det, _, order, truth = cohort
    out = diagnostics.field_curve(det, patients, truth, ["R0", "R4"],
                                  condition="matched", min_fields=6,
                                  sizes=(2, 4), reps=3, seed=0)
    assert set(out["by_rule"]) == {"R0", "R4"}
    for curve in out["by_rule"].values():
        for k, v in curve.items():
            # a tolerance, because with identical replicates np.quantile and
            # np.mean can disagree in the last bit and that is not a finding
            assert v["lo"] <= v["mean"] + 1e-12
            assert v["mean"] <= v["hi"] + 1e-12
    assert out["n_eligible"] > 0


def test_the_field_curve_is_deterministic_under_a_seed(cohort):
    patients, det, _, _, truth = cohort
    kw = dict(condition="matched", min_fields=6, sizes=(3,), reps=3, seed=4)
    a = diagnostics.field_curve(det, patients, truth, ["R0"], **kw)
    b = diagnostics.field_curve(det, patients, truth, ["R0"], **kw)
    assert a["by_rule"] == b["by_rule"]


# --------------------------------------------------------------------------- #
# Power
# --------------------------------------------------------------------------- #

def test_below_five_discordant_pairs_no_split_can_reject():
    """Not an approximation: 2^-n > 0.05 for n <= 4. A design whose expected
    discordance is four pairs cannot produce a significant result by any
    behaviour of the candidate."""
    for n in range(0, 5):
        assert power.critical_b(n) is None
    assert power.critical_b(5) == 5
    assert power.critical_b(6) == 6


def test_the_critical_value_matches_the_test_it_is_supposed_to_describe():
    from mag.stats import mcnemar_exact

    for n in range(1, 30):
        crit = power.critical_b(n)
        if crit is None:
            continue
        base = np.zeros(n, dtype=bool)
        cand = np.zeros(n, dtype=bool)
        cand[:crit] = True                          # b = crit, c = n - crit
        base[crit:] = True
        assert mcnemar_exact(base, cand, alternative="greater").p <= 0.05
        if crit > 0:
            cand2 = cand.copy()
            cand2[crit - 1] = False
            base2 = base.copy()
            base2[crit - 1] = True
            assert mcnemar_exact(base2, cand2, alternative="greater").p > 0.05


def test_power_rises_with_the_win_ratio_and_with_the_sample():
    a = power.power_mcnemar(60, 0.4, 0.6)
    b = power.power_mcnemar(60, 0.4, 0.9)
    c = power.power_mcnemar(200, 0.4, 0.6)
    assert a < b and a < c
    assert 0.0 <= a <= 1.0


def test_an_unreachable_target_is_reported_as_unreachable():
    out = power.detectable_psi(20, 0.10)             # about two discordant pairs
    assert out["achievable"] is False
    assert "even a candidate that wins every discordant pair" in out["note"]


def test_the_feasible_discordance_excludes_impossible_cohorts():
    """At a baseline accuracy of 0.966 the largest possible disagreement at
    zero true difference is about 0.068, so a power calculation assuming 10%
    is computing with a cohort that cannot exist."""
    lo, hi = power.feasible_discordance(0.966, 0.0)
    assert hi == pytest.approx(0.068, abs=1e-9)
    assert np.isnan(power.power_noninferiority(100, 0.966, 0.0, discordance=0.10))
    assert np.isfinite(power.power_noninferiority(100, 0.966, 0.0,
                                                  discordance=0.05, reps=200))


def test_both_non_inferiority_limits_are_reported_and_tango_governs():
    """The percentile bootstrap does not hold its level on this endpoint --
    0.063 to 0.076 at the null against a nominal 0.05 -- so Tango's score limit
    governs. Both are reported, because an endpoint whose verdict depends on
    which limit was used is one a reader should see twice."""
    b = np.array([1.0] * 112 + [0.0] * 4)
    c = np.array([1.0] * 108 + [0.0] * 8)
    out = power.method_agreement(b, c, margin=0.05)
    assert out["governs"] == "tango"
    assert np.isfinite(out["tango_lower"]) and np.isfinite(out["bootstrap_lower"])
    assert out["same_verdict"] is True
    assert "type I error" in out["note"] and "not used" in out["note"]


def test_no_observed_power_is_ever_reported():
    """Observed power is a monotone function of the p-value. Reporting it would
    restate the test result as though it explained it."""
    src = (__import__("pathlib").Path(power.__file__)).read_text()
    assert "observed power" in src.lower()
    assert not any(name.startswith("observed_power") for name in dir(power))


# --------------------------------------------------------------------------- #
# The nulls
# --------------------------------------------------------------------------- #

def test_the_winners_curse_null_preserves_each_patients_difficulty():
    rng = np.random.default_rng(0)
    difficulty = rng.beta(5, 2, 120)
    m = pd.DataFrame((rng.random((120, 20)) < difficulty[:, None]).astype(float),
                     columns=[f"c{i}" for i in range(20)])
    r = simulate.winners_curse(m, reps=200, seed=0)
    assert r.null_mean > 0, "the maximum of 20 correlated draws is not the mean"
    assert 0.0 <= r.p_value <= 1.0
    assert r.detail["shrunk_best"] < r.detail["best_accuracy"]


def test_the_winners_curse_finds_a_real_effect_when_there_is_one():
    rng = np.random.default_rng(1)
    m = (rng.random((150, 10)) < 0.5).astype(float)
    m[:, 0] = (rng.random(150) < 0.95).astype(float)      # one genuinely better
    r = simulate.winners_curse(pd.DataFrame(m, columns=list("abcdefghij")),
                               reps=300, seed=0)
    assert r.p_value < 0.05
    assert r.detail["best_cell"] == "a"


def test_the_crossfit_null_measures_against_the_best_constant_predictor():
    """A confirmed defect. Measured against `s >= 0.5`, a calibrator handed a
    meaningless score learns the base rate, predicts the majority class for
    everybody, and appears to gain eighteen points. That is not optimism."""
    out = simulate.crossfit_null(n=120, prevalence=0.68, reps=25,
                                 k_values=(10,), seed=0)
    assert out["k10"]["mean_apparent_gain"] < 0.05
    assert "constant predictor" in out["k10"]["reference"]


def test_selecting_on_headroom_inflates_the_size():
    """The claim this package used to make about its own selection rule.
    Headroom is b plus the both-wrong count, so choosing the largest headroom
    chooses in part the largest b -- the statistic's own numerator."""
    r = simulate.selection_null(54, [0.5, 0.5, 0.5], reps=4000, seed=2,
                                criterion="headroom")
    assert r["size_after_selection"] > 0.05
    assert r["level_preserved"] is False


def test_selecting_on_discordance_does_not():
    """The exact test conditions on the discordant-pair count, so a rule that
    is a function of those counts alone leaves the conditional distribution
    untouched. This is a theorem; the simulation is the check."""
    r = simulate.selection_null(54, [0.5, 0.5, 0.5], reps=4000, seed=2,
                                criterion="discordance")
    assert r["size_after_selection"] <= 0.05


def test_a_degenerate_selection_simulation_says_so_rather_than_passing():
    r = simulate.selection_comparison(54, [1.0, 1.0, 1.0], reps=300, seed=0)
    assert r["degenerate"] is True
    assert "says nothing about selection" in r["verdict"]


def test_a_deterministic_selection_is_reported_as_such():
    r = simulate.selection_null(54, [0.9, 0.2, 0.1], reps=300, seed=0)
    assert r["selection_was_effectively_deterministic"] is True


# --------------------------------------------------------------------------- #
# The post hoc sweep
# --------------------------------------------------------------------------- #

def test_the_sweep_declares_itself_post_hoc_everywhere(cohort):
    patients, det, _, _, truth = cohort
    sw = search.sweep(det, patients, truth, conditions=("matched",),
                      rule_ids=["R0", "R4"])
    assert sw.post_hoc is True
    assert "post hoc" in sw.note
    out = search.summarise(sw)
    assert out["post_hoc"] is True
    assert "post hoc" in out["sentence"]


def test_the_sweep_covers_the_declared_grid(cohort):
    patients, det, _, _, truth = cohort
    sw = search.sweep(det, patients, truth, conditions=("matched", "raw"),
                      rule_ids=["R0", "R4"])
    expected = (len(search.GRID["theta"]) + len(search.GRID["m"])) * 2
    assert len(sw.table) == expected
    assert sw.table["distance_to_target"].is_monotonic_increasing


def test_the_sweep_reports_the_split_behind_the_total(cohort):
    """Hitting 135 of 170 is not the same as reproducing the result: a
    configuration can reach the total with a completely different sensitivity
    and specificity."""
    patients, det, _, _, truth = cohort
    sw = search.sweep(det, patients, truth, conditions=("matched",), tolerance=200)
    out = search.summarise(sw, published_sensitivity=0.99,
                           published_specificity=0.60)
    assert out["any_reproduces"] is True
    agree = out["agreement_at_target"]
    assert agree["sensitivity_range"][0] <= agree["sensitivity_range"][1]
    assert "n_matching_published_split" in agree


def test_the_threshold_profile_is_ordered_by_threshold(cohort):
    patients, det, _, _, truth = cohort
    sw = search.sweep(det, patients, truth, conditions=("matched",),
                      rule_ids=["R4"])
    prof = search.profile(sw, "R4", "matched")
    assert prof["theta"].is_monotonic_increasing
    assert len(prof) == len(search.GRID["m"])


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

def test_a_figure_with_no_data_is_skipped_and_named_never_drawn(tmp_path):
    from mag.study import Study

    s = Study()
    s.factorial = pd.DataFrame([{"checkpoint": "a", "rule": "R0",
                                 "family": "published", "fitted": False,
                                 "accuracy": 0.5}])
    s.meta = {"condition": "matched"}
    stamp = figures.write_all(s, tmp_path)
    assert "factorial_heatmap" in stamp["figures"]
    assert any("apportionment" in why for why in stamp["skipped"])
    assert (tmp_path / figures.STAMP).exists()


def test_the_manuscript_build_refuses_figures_from_another_run(tmp_path):
    """A confirmed defect. `analyse` wrote the numbers and a separate script
    drew the figures, hours apart against different caches; a reviewer found a
    figure whose bars disagreed with the table above them."""
    from mag.errors import ProvenanceError
    from mag.study import Study

    class FakeProv:
        def line(self):
            return "run A"

    s = Study()
    s.factorial = pd.DataFrame([{"checkpoint": "a", "rule": "R0",
                                 "family": "published", "fitted": False,
                                 "accuracy": 0.5}])
    s.meta = {"condition": "matched"}
    s.provenance = FakeProv()
    figures.write_all(s, tmp_path)

    class OtherProv:
        def line(self):
            return "run B"

    s.provenance = OtherProv()
    with pytest.raises(ProvenanceError):
        figures.check(tmp_path, s)


def test_a_missing_stamp_is_refused_rather_than_assumed(tmp_path):
    from mag.errors import ProvenanceError
    from mag.study import Study

    with pytest.raises(ProvenanceError) as e:
        figures.check(tmp_path, Study())
    assert "mag figures" in e.value.remedy
