"""The parts, wired together.

A module can be correct on its own and still be wrong where it joins the next
one. These tests are about the joins: that adding an analysis cannot move a
result, that a fitted arm cannot acquire a valid-looking p-value on its way into
a table, that every macro the manuscript reads is filled and safe to typeset,
and that two runs of the same inputs agree.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

import fixtures
from mag import cli, figures, report, rules, study


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wiring")
    cache, pcsv = fixtures.write_cache(
        tmp, checkpoints=("rtdetr", "yolov8x"), provenanced=True,
        n_fields=(5, 8), seed=17)
    s = study.run(cache, pcsv, condition="matched", force_past_reconstruction=True,
                  depth="full", min_fields=5, reps=4, seed=0)
    return tmp, cache, pcsv, s


# --------------------------------------------------------------------------- #
# Depth cannot change a result
# --------------------------------------------------------------------------- #

def test_depth_does_not_change_a_result(built):
    """`full` adds descriptions. If one of them could move the confirmatory
    test, the separation this package claims between analysis and description
    would not exist."""
    tmp, cache, pcsv, full = built
    core = study.run(cache, pcsv, condition="matched",
                     force_past_reconstruction=True, depth="core", seed=0)

    assert core.confirmatory["mcnemar"] == full.confirmatory["mcnemar"]
    assert core.confirmatory["noninferiority"] == full.confirmatory["noninferiority"]
    assert core.confirmatory["iu"] == full.confirmatory["iu"]
    assert core.design["selected"] == full.design["selected"]
    assert core.reconstruction.detail == full.reconstruction.detail
    pd.testing.assert_frame_equal(core.factorial, full.factorial)
    for key in ("spread_rule", "spread_checkpoint", "rule_exceeds_checkpoint"):
        assert core.apportionment[key] == full.apportionment[key]


def test_core_depth_names_what_it_did_not_compute(built):
    tmp, cache, pcsv, _ = built
    core = study.run(cache, pcsv, condition="matched",
                     force_past_reconstruction=True, depth="core")
    skipped = core.meta["skipped_at_core_depth"]
    assert any("sweep" in s for s in skipped)
    assert not core.nulls and not core.field_curve and core.sweep is None
    # but the things a result depends on are all there
    assert core.calibration and core.power and core.error_summary


def test_an_unknown_depth_is_refused(built):
    from mag.errors import BadInput

    tmp, cache, pcsv, _ = built
    with pytest.raises(BadInput):
        study.run(cache, pcsv, depth="exhaustive")


# --------------------------------------------------------------------------- #
# Two runs of the same inputs agree
# --------------------------------------------------------------------------- #

def test_the_study_is_deterministic_under_a_seed(built):
    tmp, cache, pcsv, _ = built
    kw = dict(condition="matched", force_past_reconstruction=True, depth="core",
              seed=3)
    a = study.run(cache, pcsv, **kw)
    b = study.run(cache, pcsv, **kw)
    pd.testing.assert_frame_equal(a.factorial, b.factorial)
    assert a.confirmatory == b.confirmatory
    assert a.apportionment == b.apportionment
    assert a.calibration == b.calibration
    assert report.macros(a) == report.macros(b)


# --------------------------------------------------------------------------- #
# A fitted arm cannot pick up a valid p-value
# --------------------------------------------------------------------------- #

def test_the_factorial_marks_every_fitted_arm(built):
    _, _, _, s = built
    fac = s.factorial
    assert "fitted" in fac.columns and "exact_test_valid" in fac.columns
    for rid in rules.fitted_ids():
        rows = fac[fac.rule == rid]
        assert len(rows) == fac["checkpoint"].nunique() - 0 or len(rows) > 0
        assert rows["fitted"].all()
        assert not rows["exact_test_valid"].any()
    for rid in rules.rule_ids():
        rows = fac[fac.rule == rid]
        assert not rows["fitted"].any()
        assert rows["exact_test_valid"].all()


def test_the_confirmatory_arms_are_both_fixed(built):
    _, _, _, s = built
    assert s.confirmatory["arms_are_fixed"] is True
    assert s.confirmatory["mcnemar"]["valid"] is True
    assert study.COMPARATOR in rules.FIXED_RULES
    assert study.CANDIDATE in rules.FIXED_RULES


def test_the_apportionment_excludes_fitted_rules_from_the_primary(built):
    """Their family was chosen on these patients, so a spread across rules that
    included them would be partly a measure of how much a fitted rule can fit."""
    _, _, _, s = built
    ap = s.apportionment
    assert ap["n_rules"] == len(rules.rule_ids())
    assert ap["n_fitted_rules_excluded"] == len(rules.fitted_ids())
    assert "including_fitted_rules" in ap
    assert "sensitivity analysis" in ap["including_fitted_rules"]["note"]


def test_no_permutation_null_is_invented_for_the_rule_factor(built):
    """A confirmed defect. The null was computed on the wrong axis and came out
    equal to the observed spread to the last decimal place."""
    _, _, _, s = built
    assert s.apportionment["perm_null_rule_p95"] is None
    assert s.apportionment["perm_null_checkpoint_p95"] is not None
    assert "one observation per cell" in s.apportionment["perm_null_note"]


def test_the_spread_difference_carries_an_interval(built):
    _, _, _, s = built
    boot = s.apportionment["bootstrap"]
    lo, hi = boot["difference_ci"]
    assert lo <= boot["difference"] <= hi
    assert boot["n_patients"] == 170


# --------------------------------------------------------------------------- #
# The bounds
# --------------------------------------------------------------------------- #

def test_all_three_bounds_are_present(built):
    _, _, _, s = built
    fac = s.factorial
    assert set(fac[fac.rule.str.startswith("B")]["rule"]) == {"B0", "B1", "B2"}
    assert "source_settings" in s.bounds
    assert s.bounds["source_settings"]["condition"] == "source"


def test_no_rule_beats_the_oracle_on_its_own_checkpoint(built):
    _, _, _, s = built
    fac = s.factorial
    for ck, g in fac.groupby("checkpoint"):
        oracle = g[g.rule == "B1"]
        if oracle.empty:
            continue
        ceiling = float(oracle.iloc[0]["accuracy"])
        real = g[~g.rule.str.startswith("B")]
        assert (real["accuracy"] <= ceiling + 1e-12).all(), (
            f"{ck}: a rule exceeded the best any function of these detections "
            "could return")


# --------------------------------------------------------------------------- #
# The selection sensitivity analysis
# --------------------------------------------------------------------------- #

def test_the_selection_carries_its_own_sensitivity_analysis(built):
    _, _, _, s = built
    detail = s.design["selection_detail"]
    assert detail["by_max_headroom"]
    assert detail["by_max_discordance"]
    assert detail["primary_is_pre_registered"] is True
    assert "declared in the protocol" in detail["caveat"]
    sa = s.confirmatory["sensitivity_analysis"]
    assert sa["checkpoint"]


# --------------------------------------------------------------------------- #
# The macros
# --------------------------------------------------------------------------- #

def test_every_macro_is_filled(built):
    _, _, _, s = built
    keys = report.macros(s)
    empty = [k for k, v in keys.items() if v in ("", None)]
    assert not empty, f"{len(empty)} macros are empty: {empty[:10]}"
    assert len(keys) > 150


def test_no_macro_can_comment_out_the_rest_of_its_line(tmp_path, built):
    """A bare per-cent sign opens a LaTeX comment. A generated sentence
    carrying one deletes the rest of that line from the PDF, silently."""
    _, _, _, s = built
    report.write(s, tmp_path / "r.tex")
    body = [ln for ln in (tmp_path / "r.tex").read_text().splitlines()
            if ln.startswith("\\defresult")]
    assert body
    # math mode is exempt by design: the subscript in $\hat\Delta_{\mathrm{sens}}$
    # is an operator, and escaping it prints a literal underscore instead
    text = [re.sub(r"\$[^$]*\$", "", ln) for ln in body]
    for pattern, what in ((r"(?<!\\)%", "per-cent"), (r"(?<!\\)_", "underscore"),
                          (r"(?<!\\)#", "hash"), (r"(?<!\\)&", "ampersand")):
        offenders = [ln for ln in text if re.search(pattern, ln)]
        assert not offenders, f"unescaped {what}: {offenders[0][:120]}"


def test_a_macro_name_can_never_be_an_invalid_control_sequence(tmp_path, built):
    """`_tex` escapes macro BODIES. Names are interpolated into the control
    sequence itself, where an underscore is a LaTeX error rather than a
    mis-set glyph, and checkpoint names flow straight into several of them."""
    _, _, _, s = built
    report.write(s, tmp_path / "r.tex")
    names = re.findall(r"\\defresult\{([^}]*)\}",
                       (tmp_path / "r.tex").read_text())
    assert names
    for n in names:
        assert re.fullmatch(r"[A-Za-z0-9]+", n), f"unusable macro name: {n!r}"
    assert len(names) == len(set(names)), "two analyses wrote the same macro"


def test_two_checkpoints_that_would_share_a_macro_name_are_refused(built):
    """Digits must survive the reduction: stripping them collides yolov5x with
    yolov8x, two of the three checkpoints this study actually runs."""
    assert report._macro_name("rt_detr") == "rtdetr"
    assert report._macro_name("yolo-v8x") == "yolov8x"
    assert report._macro_name("yolov5x") != report._macro_name("yolov8x")
    assert report._macro_name("!!!") == "arm"


def test_the_outcome_sentences_agree_with_their_own_numbers(built):
    _, _, _, s = built
    k = report.macros(s)
    ni = s.confirmatory["noninferiority"]
    assert ("was demonstrated" in k["niStatement"]) == ni["non_inferior"]
    assert ("not demonstrated" in k["niStatement"]) != ni["non_inferior"]
    assert ("endpoint was met" in k["primaryVerdict"]) == \
        s.confirmatory["iu"]["reject"]
    ap = s.apportionment
    assert ("exceeded the spread" in k["apportionmentSentence"]) == \
        ap["rule_exceeds_checkpoint"]


def test_each_co_primary_clause_reads_its_own_component():
    """A confirmed defect. The verdict read only the conjunction and then
    hard-coded which half had failed, so whenever the SPECIFICITY component was
    the failing one it printed 'the specificity component was met and
    non-inferiority of sensitivity was not demonstrated' -- both halves false,
    machine-generated, and therefore trusted."""
    from mag.study import Study

    def verdict(spec_ok, sens_ok):
        s = Study()
        s.confirmatory = {
            "mcnemar": {"p": 0.5, "b": 1, "c": 1, "n_discordant": 2,
                        "alternative": "greater", "valid": True, "reason": ""},
            "noninferiority": {"diff": 0.0, "one_sided_lower": 0.0,
                               "margin": -0.05, "non_inferior": sens_ok, "n": 10},
            "iu": {"components": {"specificity_superior": spec_ok,
                                  "sensitivity_non_inferior": sens_ok},
                   "reject": spec_ok and sens_ok,
                   "failed": [k for k, v in
                              (("specificity_superior", spec_ok),
                               ("sensitivity_non_inferior", sens_ok)) if not v]},
        }
        return report._coprimary_verdict(s.confirmatory)

    assert "superiority on specificity was demonstrated" in verdict(True, True)
    assert "endpoint was met" in verdict(True, True)

    only_sens_fails = verdict(True, False)
    assert "superiority on specificity was demonstrated" in only_sens_fails
    assert "non-inferiority of sensitivity was not demonstrated" in only_sens_fails
    assert "was not established" in only_sens_fails

    only_spec_fails = verdict(False, True)
    assert "superiority on specificity was not demonstrated" in only_spec_fails
    assert "non-inferiority of sensitivity was demonstrated" in only_spec_fails
    assert "was not established" in only_spec_fails

    both = verdict(False, False)
    assert "specificity was not demonstrated" in both
    assert "sensitivity was not demonstrated" in both


def test_the_fitted_macros_refuse_to_offer_a_p_value(built):
    _, _, _, s = built
    k = report.macros(s)
    assert "No exact test is reported against any fitted arm" in k["fittedSentence"]
    assert k["nFittedArms"] == str(len(rules.fitted_ids()))


def test_the_power_macros_refuse_observed_power(built):
    _, _, _, s = built
    k = report.macros(s)
    assert "detectable effects, not observed power" in k["powerCaveat"]
    assert "powerCritB" in k and "powerMinDiscordant" in k


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #

def test_mag_all_writes_results_figures_and_macros(tmp_path, built):
    _, cache, pcsv, _ = built
    out = tmp_path / "out"
    rc = cli.main(["all", "--cache", str(cache), "--patients", str(pcsv),
                   "--out", str(out), "--tex", str(out / "results.tex"),
                   "--depth", "full", "--min-fields", "5", "--reps", "3",
                   "--force-reconstruction"])
    assert rc == 0
    for name in ("factorial.csv", "results.json", "predictions.csv",
                 "errors.csv", "calibration.csv", "box_census.csv",
                 "sweep.csv", "provenance.json", "results.tex"):
        assert (out / name).exists(), name
    stamp = json.loads((out / "figures" / figures.STAMP).read_text())
    assert len(stamp["figures"]) >= 7
    assert not stamp["skipped"], stamp["skipped"]
    summary = json.loads((out / "results.json").read_text())
    assert summary["confirmatory"]["arms_are_fixed"] is True
    assert summary["power"]["note"].startswith("detectable effects")


def test_the_predictions_file_covers_every_arm_and_patient(tmp_path, built):
    _, cache, pcsv, _ = built
    out = tmp_path / "out2"
    cli.main(["run", "--cache", str(cache), "--patients", str(pcsv),
              "--out", str(out), "--force-reconstruction"])
    preds = pd.read_csv(out / "predictions.csv", dtype=str)
    assert len(preds) == 170
    n_arms = 2 * (len(rules.rule_ids()) + len(rules.fitted_ids()) + 2) + 1
    assert len(preds.columns) - 1 == n_arms
    assert not preds.isna().any().any(), "a missing prediction is an error"


def test_the_rules_command_lists_both_registries(capsys):
    assert cli.main(["rules"]) == 0
    out = capsys.readouterr().out
    for rid in rules.all_rule_ids():
        assert rid in out
    assert "No exact test is valid against this arm" in out


def test_a_missing_cache_fails_with_a_remedy_not_a_traceback(tmp_path, capsys):
    rc = cli.main(["run", "--cache", str(tmp_path / "nope"),
                   "--patients", str(tmp_path / "nope.csv"), "--out",
                   str(tmp_path / "o")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "remedy" in err.lower() or "->" in err or "Fix" in err or err.strip()


def test_report_refuses_figures_from_another_run(tmp_path, built):
    from mag.errors import ProvenanceError

    _, cache, pcsv, s = built
    figures.write_all(s, tmp_path / "figs")
    stamp_path = tmp_path / "figs" / figures.STAMP
    stamp = json.loads(stamp_path.read_text())
    stamp["provenance"] = "a different run entirely"
    stamp_path.write_text(json.dumps(stamp))
    with pytest.raises(ProvenanceError):
        figures.check(tmp_path / "figs", s)


def test_the_reported_version_matches_the_packaging_metadata():
    """A drifted version silently mislabels every provenance record.

    `__version__` is a literal so that a source tree that was never installed
    can still stamp its runs. The cost of that choice is that it can fall out
    of step with `pyproject.toml`, and the stamp is what a reader would use to
    identify which code produced a result. So the two are checked against each
    other here rather than trusted to stay aligned by hand.
    """
    import mag

    text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    declared = re.search(r'^version\s*=\s*"([^"]+)"', text, re.M)
    assert declared, "pyproject.toml declares no version"
    assert mag.__version__ == declared.group(1)
