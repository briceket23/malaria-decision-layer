"""The five analyses protocol v3 added, tested against known answers.

Each module here exists because an external review asked for something the
previous version did not do. A module written in response to a review is
exactly the kind that gets written once, read once and never checked, so each
one is exercised here against a case whose answer is known in advance --
simulated with a planted effect, or hand-computed, or checked against an
independent implementation written in the test itself.

Three of these tests deliberately construct a situation the real data is in:
complete separation, a single-centre negative arm, and a cohort prevalence far
above any screening prevalence. Those are not edge cases for this study. They
are the study.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import fixtures
from mag import attribute, audit, estimand, predictive, stress
from mag.errors import BadInput, IntegrityError
from mag.schema import UNINFECTED


# --------------------------------------------------------------------------- #
# §4  Estimands
# --------------------------------------------------------------------------- #

def test_specificity_is_declared_unidentifiable_with_a_reason():
    """Not "underpowered". The distinction is the point of the module.

    A reader who is told specificity is imprecise will ask for a larger sample.
    A reader who is told it is collinear with centre knows no sample fixes it.
    """
    e = estimand.SPECIFICITY
    assert e.identifiable is False
    assert e.why_not, "an unidentifiable estimand must say why"
    assert "rank deficiency" in e.why_not
    assert "no sample size fixes it" in e.why_not
    # and the other three are identifiable, so the flag means something
    assert all(estimand.ESTIMANDS[k].identifiable
               for k in estimand.ESTIMANDS if k != e.name)


def test_every_estimand_carries_a_scope_that_reaches_a_sentence():
    for name, e in estimand.ESTIMANDS.items():
        assert e.scope.strip(), f"{name} has no scope"
        s = e.qualify("the rule reached 91%")
        assert e.scope in s and s.endswith(".")
        assert "91%" in s


def test_scope_of_refuses_to_invent_a_scope_for_an_unknown_metric():
    assert estimand.scope_of("specificity") == estimand.SPECIFICITY.scope
    assert estimand.scope_of("f1_score") == ""


def test_the_cohort_check_notices_when_the_written_scope_stops_being_true():
    """The scopes are written before the run. That is only safe if a cohort
    that no longer matches them is detected rather than described by stale
    prose."""
    p = fixtures.patient_table(seed=0)
    ok = estimand.check_cohort(p)
    assert ok["specificity_scope_still_holds"] is True
    assert ok["negative_centres"] == [estimand.NEGATIVE_CENTRE]

    # move one uninfected patient to a second centre: the scope is now wrong
    q = p.copy()
    idx = q.index[q["truth"] == UNINFECTED][0]
    q.loc[idx, "centre"] = "Rouen"
    bad = estimand.check_cohort(q)
    assert bad["specificity_scope_still_holds"] is False
    assert "must be rebuilt" in bad["note"]
    assert len(bad["negative_centres"]) == 2


def test_the_declaration_is_machine_readable_and_complete():
    d = estimand.declaration()
    assert set(d) == set(estimand.ESTIMANDS)
    for name, body in d.items():
        assert set(body) >= {"target", "scope", "identifiable", "why_not"}
        assert body["target"].strip() and body["scope"].strip()
    assert json.loads(json.dumps(d)) == d


# --------------------------------------------------------------------------- #
# §9  Attribution
# --------------------------------------------------------------------------- #

def _planted(n_patients=140, rule_effects=(0.0, 1.2, -0.9),
             ck_effects=(0.0, 0.15), seed=7):
    """A factorial with effects put in by hand, so the fit has a right answer.

    Patient random intercepts are included, because the whole reason the model
    is clustered is that the same patient appears in every cell.
    """
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, 0.8, n_patients)
    rows = []
    for i in range(n_patients):
        for ri, a in enumerate(rule_effects):
            for ci, b in enumerate(ck_effects):
                eta = 0.4 + a + b + u[i]
                p = 1.0 / (1.0 + np.exp(-eta))
                rows.append({"patient": f"P{i:04d}", "rule": f"R{ri}",
                             "checkpoint": f"C{ci}",
                             "y": int(rng.random() < p)})
    df = pd.DataFrame(rows)
    df["rule"] = df["rule"].astype("category")
    df["checkpoint"] = df["checkpoint"].astype("category")
    return df


def test_the_interaction_is_estimable_which_the_previous_version_denied():
    """v2 claimed the interaction "is arithmetically the residual and is not
    separately estimable". A review rejected that and was right: every patient
    contributes an outcome in every cell. If it were inestimable the fit below
    could not produce a test of it."""
    a = attribute.fit(_planted())
    assert a.interaction, "no interaction was estimated"
    assert np.isfinite(a.interaction["p"])
    assert a.interaction["n_terms"] == 2, "two rules beyond reference x one ckpt"
    # nothing was simulated, so the test should not reject at 5%
    assert a.interaction["p"] > 0.05


def test_a_planted_interaction_is_detected():
    """The complement of the test above: a null result is only evidence if the
    same machinery finds a real effect."""
    rng = np.random.default_rng(3)
    rows = []
    for i in range(160):
        u = rng.normal(0, 0.6)
        for ri in range(2):
            for ci in range(2):
                eta = 0.2 + 1.0 * ri + 0.1 * ci + 2.2 * (ri and ci) + u
                rows.append({"patient": f"P{i:04d}", "rule": f"R{ri}",
                             "checkpoint": f"C{ci}",
                             "y": int(rng.random() < 1 / (1 + np.exp(-eta)))})
    df = pd.DataFrame(rows)
    df["rule"] = df["rule"].astype("category")
    df["checkpoint"] = df["checkpoint"].astype("category")
    assert attribute.fit(df).interaction["p"] < 0.01


def test_two_independent_implementations_are_reported_as_agreeing():
    """statsmodels against the hand-rolled sandwich. The number that matters is
    not either estimate but the distance between them: a disagreement is
    reported rather than resolved silently in favour of one."""
    a = attribute.fit(_planted())
    assert a.agreement["checked"] is True
    assert a.agreement["agree"] is True
    assert a.agreement["max_abs_coefficient_difference"] < attribute.AGREEMENT_TOL
    assert "independently written" in a.agreement["note"]


def test_the_contrast_recovers_a_planted_ordering():
    """Rules were given a 2.1 log-odds spread and checkpoints 0.15. The
    contrast is on the probability scale, so it is not that number -- but it
    must be positive, and its interval must exclude zero."""
    a = attribute.fit(_planted())
    c = a.contrast
    assert c["estimate"] > 0
    assert c["lo"] > 0, "a planted rule effect should exclude zero"
    assert c["lo"] < c["estimate"] < c["hi"]


def test_a_null_contrast_does_not_exclude_zero():
    """The same estimator on data with no rule effect at all. Without this the
    test above only shows the estimator returns something positive."""
    a = attribute.fit(_planted(rule_effects=(0.0, 0.0, 0.0),
                               ck_effects=(0.0, 0.0), seed=11))
    assert a.contrast["lo"] <= 0 <= a.contrast["hi"]


def test_complete_separation_is_detected_and_the_log_odds_is_withheld():
    """The real data is in this state: R0 has specificity 0.000, so its level
    never varies and its log-odds effect is unbounded. An earlier version fitted
    anyway and the two implementations disagreed by 4e8 -- a number that would
    have been printed."""
    df = _planted(n_patients=120, seed=5)
    df.loc[df["rule"] == "R2", "y"] = 0          # R2 is now always wrong

    a = attribute.fit(df)
    assert a.separation["any"] is True
    assert any(r["level"] == "R2" for r in a.separation["rules"])

    # the log-odds contrast is refused, and says why
    assert a.contrast_logodds.get("reported") is False
    assert "separated" in a.contrast_logodds.get("why", "")

    # the probability-scale contrast survives, because it is not fitted
    assert np.isfinite(a.contrast["estimate"])
    assert a.contrast["lo"] < a.contrast["hi"]


def test_separation_is_not_claimed_when_there_is_none():
    a = attribute.fit(_planted())
    assert a.separation["any"] is False
    assert a.separation["n_separated_levels"] == 0
    assert a.contrast_logodds.get("reported") is True


def test_the_link_overflow_is_recorded_rather_than_swallowed():
    """Separation overflows the inverse logit. That warning is silenced at the
    fit -- but an overflow with NO separated level would mean something else,
    so which case occurred is kept."""
    df = _planted(n_patients=120, seed=5)
    df.loc[df["rule"] == "R2", "y"] = 0
    a = attribute.fit(df)
    assert set(a.numerics) >= {"link_overflowed", "expected", "note"}
    assert a.numerics["expected"] is True

    clean = attribute.fit(_planted())
    assert clean.numerics["link_overflowed"] is False
    assert clean.numerics["note"] == "no overflow"


def test_an_incomplete_factorial_is_refused_not_fitted_on_what_is_there():
    truth = pd.Series({"P1": UNINFECTED, "P2": "P_falciparum"})
    with pytest.raises(BadInput) as e:
        attribute.long_frame({("c1", "R0"): pd.Series({"P1": UNINFECTED,
                                                       "P2": UNINFECTED})},
                             truth, ["P1", "P2"], rules=["R0", "R4"],
                             checkpoints=["c1"])
    assert "complete factorial" in str(e.value.remedy)


def test_the_long_frame_is_one_row_per_cell_per_patient():
    order = ["P1", "P2", "P3"]
    truth = pd.Series({"P1": UNINFECTED, "P2": "P_falciparum",
                       "P3": "P_vivax"})
    preds = {(ck, r): truth.copy()
             for ck in ("c1", "c2") for r in ("R0", "R4")}
    df = attribute.long_frame(preds, truth, order, rules=["R0", "R4"],
                              checkpoints=["c1", "c2"])
    assert len(df) == 3 * 2 * 2
    assert df["y"].sum() == len(df), "identical predictions are all correct"
    assert set(df.columns) == {"patient", "rule", "checkpoint", "y"}


# --------------------------------------------------------------------------- #
# §13  The field-sampling stress test
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    from mag import cache as cache_mod

    tmp = tmp_path_factory.mktemp("stressfx")
    cache, pcsv = fixtures.write_cache(tmp, checkpoints=("rtdetr",),
                                       provenanced=True, n_fields=(6, 10),
                                       seed=23)
    patients = pd.read_csv(pcsv)
    det, _ = cache_mod.load(cache, "rtdetr", verify=True)
    truth = patients.set_index(patients["patient_id"].astype(str))["truth"]
    return det, patients, truth


def test_dilution_does_not_touch_uninfected_patients(loaded):
    """Their fields are already parasite-free; diluting them would mean
    nothing, and a specificity that moved with pi would be an artefact of the
    procedure rather than a finding."""
    det, patients, truth = loaded
    s = stress.run(det, patients, truth, ["R0", "R4"], reps=3, seed=1)
    if s.note:
        pytest.skip(s.note)
    for rid in ("R0", "R4"):
        specs = {v["specificity"] for v in s.by_rule[rid].values()}
        assert len(specs) == 1, f"{rid} specificity moved under dilution"


def test_dilution_lowers_sensitivity_monotonically_or_says_it_did_not(loaded):
    """Removing parasite-bearing fields cannot make an infected patient easier
    to detect. This is the test that would catch a dilution that silently
    kept the fields it claimed to replace."""
    det, patients, truth = loaded
    s = stress.run(det, patients, truth, ["R4"], reps=6, seed=2)
    if s.note:
        pytest.skip(s.note)
    by_rate = [(float(k), v["sensitivity"]) for k, v in s.by_rule["R4"].items()]
    by_rate.sort(reverse=True)
    sens = [v for _, v in by_rate]
    assert sens[0] >= sens[-1] - 1e-12, "dilution raised sensitivity"


def test_the_field_count_is_held_constant_so_a_normalised_rule_is_comparable():
    """R6 divides by fields examined. If dilution changed the denominator, its
    curve would be measuring the denominator."""
    from mag.diagnostics import _view_from_fields
    from mag.schema import ALL_CLASSES

    own = [{"P_falciparum": np.array([0.9, 0.8])} for _ in range(10)]
    donors = [{"blood_cell": np.array([0.7])} for _ in range(10)]
    mixed = own[:3] + donors[:7]
    v = _view_from_fields("P1", mixed, ALL_CLASSES, n_fields=10)
    assert v.n_fields == 10, "the denominator moved with the dilution"
    # and the evidence really did thin out: 3 of 10 fields still carry a parasite
    assert int((v.per_field_count["P_falciparum"] > 0).sum()) == 3


def test_a_pool_too_small_to_dilute_with_refuses_rather_than_reusing_one_field(loaded):
    det, patients, truth = loaded
    tiny = patients.copy()
    # leave a single uninfected patient, so the donor pool is a handful of fields
    neg = tiny.index[tiny["truth"] == UNINFECTED]
    tiny = tiny.drop(neg[1:])
    sub = det[det["patient_id"].astype(str).isin(
        set(tiny["patient_id"].astype(str)))]
    t = tiny.set_index(tiny["patient_id"].astype(str))["truth"]
    s = stress.run(sub, tiny, t, ["R4"], reps=2, seed=0)
    assert s.by_rule == {}
    assert "Not computed" in s.note
    assert str(stress.MIN_DONOR_FIELDS) in s.note


def test_the_donor_centre_mismatch_is_counted_and_never_hidden(loaded):
    """Every uninfected patient comes from one centre, so most infected
    patients are diluted across a domain shift. The size of that problem is
    reported beside the result rather than in a caveat."""
    det, patients, truth = loaded
    s = stress.run(det, patients, truth, ["R0", "R4"], reps=2, seed=0)
    if s.note:
        pytest.skip(s.note)
    dm = s.donor_mismatch
    assert dm["n_with_no_same_centre_donor"] > 0
    assert 0 < dm["share"] <= 1
    assert str(dm["n_with_no_same_centre_donor"]) in s.sentence
    assert "not an estimate of screening performance" in s.sentence


def test_kendall_tau_is_tie_corrected():
    """Several rules reach identical sensitivity here. A tau-a would report an
    ordering change where there is only a tie."""
    a = np.array([1.0, 1.0, 2.0])
    tau, n_pairs, disc = stress._kendall(a, a)
    assert tau == pytest.approx(1.0)
    assert disc == 0
    rev = np.array([3.0, 2.0, 1.0])
    tau_rev, _, _ = stress._kendall(np.array([1.0, 2.0, 3.0]), rev)
    assert tau_rev == pytest.approx(-1.0)


def test_kendall_agrees_with_an_established_implementation():
    """Written out here rather than imported, so scipy is a test-time luxury
    and never a dependency of the package. It is the check that found the
    tie-correction bug: the previous denominator ADDED the tied pairs where
    tau-b subtracts them, returning 0.67 for two identical rankings with one
    tie and pulling every stability verdict toward "unstable"."""
    kendalltau = pytest.importorskip("scipy.stats").kendalltau

    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        n = int(rng.integers(3, 9))
        # a small integer alphabet, so ties are frequent -- which is the case
        # the rules are actually in
        a = rng.integers(0, 4, n).astype(float)
        b = rng.integers(0, 4, n).astype(float)
        tau, _, _ = stress._kendall(a, b)
        ref = kendalltau(a, b, variant="b").statistic
        assert np.isfinite(tau) == np.isfinite(ref), (a, b)
        if np.isfinite(tau):
            worst = max(worst, abs(tau - ref))
    assert worst < 1e-12, f"worst disagreement with scipy was {worst:.2e}"


def test_an_inverted_ordering_is_named_as_such_not_softened():
    per_rule = {"R0": {1.0: {"sensitivity": 0.9}, 0.1: {"sensitivity": 0.1}},
                "R4": {1.0: {"sensitivity": 0.2}, 0.1: {"sensitivity": 0.8}}}
    o = stress._ordering(per_rule, (1.0, 0.1))
    assert o["inverted"] is True
    assert o["stable"] is False
    assert o["top_rule_changes"] is True
    s = stress.Stress(ordering=o, donor_mismatch={"n_infected": 10,
                                                  "n_with_no_same_centre_donor": 3})
    assert "INVERTED" in stress._sentence(s)


# --------------------------------------------------------------------------- #
# §14  Predictive values
# --------------------------------------------------------------------------- #

def test_ppv_and_npv_match_bayes_by_hand():
    pos = np.array([True] * 90 + [False] * 10)      # sensitivity 0.90
    neg = np.array([True] * 95 + [False] * 5)       # specificity 0.95
    out = predictive.at_prevalences(pos, neg, prevalences=(0.01,), n_boot=200)
    want = (0.90 * 0.01) / (0.90 * 0.01 + 0.05 * 0.99)
    assert out.at["0.01"]["ppv"] == pytest.approx(want)
    want_npv = (0.95 * 0.99) / (0.95 * 0.99 + 0.10 * 0.01)
    assert out.at["0.01"]["npv"] == pytest.approx(want_npv)


def test_no_predictive_value_is_reported_without_the_prevalence_it_assumed():
    pos = np.array([True] * 90 + [False] * 10)
    neg = np.array([True] * 95 + [False] * 5)
    out = predictive.at_prevalences(pos, neg, n_boot=200)
    for key, row in out.at.items():
        assert "prevalence" in row and np.isfinite(row["prevalence"])
        assert row["assumed"].strip(), f"{key} carries no assumption"


def test_the_cohort_figure_is_labelled_as_a_referral_centre_quantity():
    """68% prevalence produces a PPV around 0.97. Quoted bare it reads as a
    screening PPV and is an order of magnitude too high for one."""
    pos = np.array([True] * 116)
    neg = np.array([True] * 40 + [False] * 14)
    out = predictive.at_prevalences(pos, neg, n_boot=200)
    row = out.at["cohort"]
    assert row["ppv"] > out.at["0.01"]["ppv"]
    assert "referral-centre" in row["assumed"]
    assert "not transportable" in row["assumed"].lower()
    assert "do not transport between settings" in out.sentence
    assert out.scope == estimand.SPECIFICITY.scope


def test_the_interval_is_asymmetric_where_the_specificity_arm_is_small():
    """54 uninfected patients. A delta-method interval is symmetric and would
    put the bound in the wrong place; the bootstrap one need not be."""
    rng = np.random.default_rng(0)
    pos = rng.random(116) < 0.97
    neg = rng.random(54) < 0.74
    out = predictive.at_prevalences(pos, neg, prevalences=(0.01,), n_boot=800,
                                    seed=1)
    r = out.at["0.01"]
    lo, hi = r["ppv_ci"]
    assert lo < r["ppv"] < hi
    left, right = r["ppv"] - lo, hi - r["ppv"]
    assert abs(left - right) > 1e-6, "an exactly symmetric interval is suspect"


def test_an_empty_arm_refuses_rather_than_dividing_by_zero():
    out = predictive.at_prevalences(np.array([], dtype=bool),
                                    np.array([True, False]))
    assert out.at == {}
    assert "one arm is empty" in out.note


# --------------------------------------------------------------------------- #
# §19  The build audit
# --------------------------------------------------------------------------- #

def _tex(**macros) -> str:
    return "\n".join(f"\\defresult{{{k}}}{{{v}}}" for k, v in macros.items())


def test_an_empty_macro_is_a_failure_not_a_blank_in_the_pdf(tmp_path):
    p = tmp_path / "results.tex"
    p.write_text(_tex(primaryP="0.031", brierRaw=""))
    found = audit.check_macros(p)
    assert any(f.check == "macros" and "empty" in f.message for f in found)


def test_a_manuscript_calling_an_undefined_macro_is_caught(tmp_path):
    tex = tmp_path / "results.tex"
    tex.write_text(_tex(primaryP="0.031"))
    ms = tmp_path / "paper.tex"
    ms.write_text(r"We found \R{primaryP} and \R{neverDefined}.")
    found = audit.check_macros(tex, ms)
    fails = [f for f in found if f.severity == "fail"]
    assert fails and "neverDefined" in fails[0].message
    assert "read as data" in fails[0].message


def test_a_typed_verdict_is_refused_even_when_it_is_the_right_one(tmp_path):
    """The previous build's abstract said the conjunction was met while its
    conclusion said the opposite. Both were typed. This check does not ask
    which was correct."""
    ms = tmp_path / "paper.tex"
    ms.write_text("The co-primary endpoint was met.\n"
                  "Sensitivity was non-inferior.\n")
    found = audit.check_no_typed_verdict(ms)
    assert len(found) == 2
    assert all(f.severity == "fail" for f in found)
    assert all("generated macro" in f.message for f in found)


def test_retired_vocabulary_is_caught(tmp_path):
    ms = tmp_path / "paper.tex"
    ms.write_text("The pipeline showed a failed reproduction of the result.")
    found = audit.check_no_typed_verdict(ms)
    assert found and "retired vocabulary" in found[0].message


def test_a_commented_out_verdict_is_not_a_finding(tmp_path):
    ms = tmp_path / "paper.tex"
    ms.write_text("% the co-primary endpoint was met\nNothing here.\n")
    assert audit.check_no_typed_verdict(ms) == []


def test_a_rate_that_does_not_follow_from_its_counts_is_caught():
    good = pd.DataFrame([{"checkpoint": "c", "rule": "R0", "tp": 90, "fn": 10,
                          "tn": 40, "fp": 10, "sensitivity": 0.9,
                          "specificity": 0.8}])
    assert audit.check_arithmetic(good) == []
    bad = good.copy()
    bad.loc[0, "sensitivity"] = 0.95
    found = audit.check_arithmetic(bad)
    assert found and "but its counts give" in found[0].message


def test_a_brier_printed_to_three_decimals_is_refused(tmp_path):
    """0.000 and 0.0004 are different findings and the same string. The
    previous build printed the first."""
    p = tmp_path / "results.tex"
    p.write_text(_tex(brierRaw="0.000", eceMatched="0.0123"))
    found = audit.check_precision(p)
    assert len(found) == 1
    assert "brierRaw" in found[0].message
    assert "indistinguishable" in found[0].message


def test_a_number_that_disagrees_between_tex_and_json_is_caught(tmp_path):
    tex = tmp_path / "results.tex"
    res = tmp_path / "results.json"
    payload = {"confirmatory": {"mcnemar": {"b": 12, "c": 3}},
               "reconstruction": {"n_correct": 96, "n_total": 170,
                                  "published": 135},
               "attribution": {"n_patients": 170}}
    res.write_text(json.dumps(payload))
    tex.write_text(_tex(primaryB="12", primaryC="3", reconN="97",
                        reconTotal="170", reconPublished="135",
                        attrNPatients="170"))
    found = audit.check_cross_artefact(tex, res)
    assert len(found) == 1
    assert "reconN is 97" in found[0].message and "96" in found[0].message


def test_a_pair_that_cannot_be_compared_is_a_failure_not_a_silent_pass(tmp_path):
    """The first version of this check still read results.json's "reproduction"
    key after it had been renamed to "reconstruction", and skipped in silence.
    It reported success while comparing nothing."""
    tex = tmp_path / "results.tex"
    res = tmp_path / "results.json"
    res.write_text(json.dumps({"reproduction": {"n_correct": 96}}))
    tex.write_text(_tex(primaryB="12"))
    found = audit.check_cross_artefact(tex, res)
    assert found
    assert any("cannot be compared" in f.message for f in found)
    assert all(f.severity == "fail" for f in found)


def test_missing_declarations_are_named_and_never_supplied(tmp_path):
    found = audit.check_declarations(None)
    assert len(found) == 1 and found[0].severity == "fail"
    assert "will not invent them" in found[0].message
    assert set(found[0].detail["required"]) == set(audit.REQUIRED_DECLARATIONS)

    partial = {k: "given" for k in audit.REQUIRED_DECLARATIONS}
    partial["funding"] = ""
    found = audit.check_declarations(partial)
    assert found and "funding" in found[0].message


def test_a_declaration_still_holding_a_placeholder_is_not_a_declaration():
    decl = {k: "given" for k in audit.REQUIRED_DECLARATIONS}
    decl["osf_registration"] = "[OSF DOI]"
    decl["funding"] = "TBD"
    found = audit.check_declarations(decl)
    msgs = " ".join(f.message for f in found)
    assert "placeholder" in msgs
    assert "osf_registration" in msgs and "funding" in msgs


def test_figures_from_another_run_are_caught(tmp_path):
    from mag import figures as figmod

    d = tmp_path / "figures"
    d.mkdir()
    (d / figmod.STAMP).write_text(json.dumps(
        {"provenance": "run-A", "seed": 0, "figures": ["f1"], "failed": []}))
    assert audit.check_figures(d, "run-A", 0) == []
    found = audit.check_figures(d, "run-B", 0)
    assert found and "different runs" in found[0].message
    found = audit.check_figures(d, "run-A", 7)
    assert found and "different seed" in found[0].message


def test_the_report_raises_with_every_failure_named(tmp_path):
    rep = audit.AuditReport(
        findings=[audit.Finding("macros", "fail", "one is empty"),
                  audit.Finding("macros", "warn", "one is unused"),
                  audit.Finding("precision", "fail", "brier is 0.000")],
        n_checks=4)
    assert rep.passed is False
    assert len(rep.failed) == 2
    with pytest.raises(IntegrityError) as e:
        rep.raise_if_failed()
    assert "one is empty" in str(e.value) and "brier is 0.000" in str(e.value)
    assert "one is unused" not in str(e.value), "a warning is not a blocker"
    assert "there is no flag" in str(e.value.remedy)


def test_a_clean_report_passes_and_says_how_many_checks_ran():
    rep = audit.AuditReport(findings=[audit.Finding("m", "warn", "x")],
                            n_checks=6)
    assert rep.passed is True
    rep.raise_if_failed()
    assert "6 checks, all passed" in rep.summary()


# --------------------------------------------------------------------------- #
# The audit, wired to the command line
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def finished(tmp_path_factory):
    """A real results directory, produced by `mag all`."""
    from mag import cli

    tmp = tmp_path_factory.mktemp("audited")
    cache, pcsv = fixtures.write_cache(tmp, checkpoints=("rtdetr", "yolov8x"),
                                       provenanced=True, n_fields=(5, 8),
                                       seed=17)
    out = tmp / "out"
    rc = cli.main(["all", "--cache", str(cache), "--patients", str(pcsv),
                   "--out", str(out), "--depth", "core", "--min-fields", "5",
                   "--reps", "3", "--force-reconstruction"])
    assert rc == 0
    return out


def test_mag_audit_fails_a_run_with_no_declarations(finished, capsys):
    from mag import cli

    rc = cli.main(["audit", "--out", str(finished)])
    assert rc == 5
    out = capsys.readouterr().out
    assert "FAILED" in out
    assert "will not invent them" in out


def test_mag_audit_passes_once_the_author_has_declared(finished, tmp_path,
                                                       capsys):
    from mag import cli

    decl = tmp_path / "declarations.json"
    decl.write_text(json.dumps(
        {k: f"supplied by the author: {k}" for k in audit.REQUIRED_DECLARATIONS}))
    rc = cli.main(["audit", "--out", str(finished), "--declarations",
                   str(decl), "--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    assert "all passed" in out
    payload = json.loads((finished / "audit.json").read_text())
    assert payload["passed"] is True and payload["n_checks"] > 0


def test_mag_audit_says_where_to_look_when_the_declarations_file_is_absent(
        finished, tmp_path):
    from mag import cli

    with pytest.raises(BadInput) as e:
        cli.cmd_audit(type("A", (), {
            "out": str(finished), "manuscript": None,
            "declarations": str(tmp_path / "nope.json"), "json": False})())
    assert "will not guess" in str(e.value.remedy)


def test_the_cross_artefact_check_is_live_against_a_real_run(finished):
    """Not a mock. If a future rename silently disconnects results.tex from
    results.json, this is the test that notices."""
    found = audit.check_cross_artefact(finished / "results.tex",
                                       finished / "results.json")
    assert [f for f in found if "cannot be compared" in f.message] == []
    assert found == []
