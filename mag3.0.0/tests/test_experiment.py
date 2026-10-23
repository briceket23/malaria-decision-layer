"""The experiment end to end, and the properties of its output.

These run on a fixture cache, which is why every one of them either forces past
the reproduction gate or asserts that the gate fired. A fixture cannot
reproduce a published number, and a test that pretended otherwise would be
asserting something about the fixture rather than about the code.
"""

from __future__ import annotations

import json

import pytest

import fixtures
from mag import cli, report, rules, study
from mag.errors import GateRefusal


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("study")
    cache, pcsv = fixtures.write_cache(tmp, provenanced=True, seed=5,
                                       n_fields=(4, 7))
    s = study.run(cache, pcsv, condition="matched", seed=5,
                  force_past_reconstruction=True)
    return s, cache, pcsv


def test_the_factorial_covers_every_rule_on_every_surviving_arm(built):
    s, _, _ = built
    arms = set(s.factorial["checkpoint"])
    for ck in arms:
        got = set(s.factorial[s.factorial.checkpoint == ck]["rule"])
        assert set(rules.rule_ids()) <= got
        assert {"B0", "B1"} <= got, "the bounds are missing"


def test_the_oracle_is_never_beaten_by_a_rule(built):
    """P4. If a rule outscores the ceiling, the ceiling is defined wrongly."""
    s, _, _ = built
    for ck, g in s.factorial.groupby("checkpoint"):
        oracle = float(g[g.rule == "B1"]["n_correct"].iloc[0])
        real = g[~g.rule.str.startswith("B")]["n_correct"].max()
        assert real <= oracle, f"{ck}: a rule beat the oracle"


def test_the_reproduction_gate_stops_a_run_by_default(built):
    _, cache, pcsv = built
    with pytest.raises(GateRefusal) as ei:
        study.run(cache, pcsv, condition="matched", seed=5)
    assert "reconstruction gate" in str(ei.value)
    assert ei.value.remedy, "the refusal offers no way forward"


def test_forcing_past_the_gate_is_recorded_not_forgotten(built):
    s, _, _ = built
    assert s.meta["forced_past_reconstruction"] is True


def test_the_confirmatory_family_holds_exactly_one_comparison(built):
    s, _, _ = built
    assert len(s.design["family"]) == 1
    c = s.confirmatory
    assert c["rule"] == study.CANDIDATE
    assert c["comparator"] == study.COMPARATOR
    assert c["arms_are_fixed"] is True, "a fitted arm reached the exact test"


def test_both_co_primary_components_are_one_sided_in_the_same_direction(built):
    s, _, _ = built
    assert s.confirmatory["mcnemar"]["alternative"] == "greater"
    ni = s.confirmatory["noninferiority"]
    assert ni["margin"] < 0 and "lower confidence limit" in ni["criterion"]
    assert ni["method"] == "tango", (
        "the percentile bootstrap does not hold its level on this endpoint")


def test_the_conjunction_is_computed_once_and_agrees_with_its_parts(built):
    s, _, _ = built
    c = s.confirmatory
    comps = c["iu"]["components"]
    assert c["iu"]["reject"] == all(comps.values())
    assert comps["sensitivity_non_inferior"] == c["noninferiority"]["non_inferior"]


def test_the_apportionment_reports_both_summaries_and_a_null(built):
    s, _, _ = built
    ap = s.apportionment
    for key in ("spread_rule", "spread_checkpoint", "pairwise_rule_median",
                "pairwise_checkpoint_median", "perm_null_rule_p95"):
        assert key in ap and ap[key] == ap[key]


def test_the_condition_changes_the_answer_and_is_recorded(built):
    """Post-processing is a standardising condition, not a free choice: raw and
    matched do not agree, so a checkpoint claim has to name which one it is."""
    _, cache, pcsv = built
    raw = study.run(cache, pcsv, condition="raw", seed=5,
                    force_past_reconstruction=True)
    matched = study.run(cache, pcsv, condition="matched", seed=5,
                        force_past_reconstruction=True)
    assert raw.meta["condition"] == "raw"
    assert "post-processing raw" in raw.provenance.line()
    a = raw.factorial.set_index(["checkpoint", "rule"])["n_correct"]
    b = matched.factorial.set_index(["checkpoint", "rule"])["n_correct"]
    assert not a.equals(b), "suppression changed nothing; check the IoU path"


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def test_the_outcome_sentences_agree_with_the_numbers(built):
    """The manuscript prints these instead of typing a direction. If they can
    disagree with the run, the paper can contradict its own tables -- which is
    exactly what happened to the predecessor."""
    s, _, _ = built
    m = report.macros(s)
    reject = s.confirmatory["iu"]["reject"]
    assert (m["iuReject"] == "1") == reject
    assert ("was met" in m["primaryVerdict"]) == reject
    ni = s.confirmatory["noninferiority"]["non_inferior"]
    assert (m["niMet"] == "1") == ni
    assert ("not demonstrated" in m["niStatement"]) == (not ni)


def test_the_report_names_the_excluded_arms_rather_than_dropping_them(tmp_path):
    cache, pcsv = fixtures.write_cache(tmp_path, provenanced=True, seed=2)
    # blank one checkpoint's table so the yield gate refuses it
    import mag.cache as cache_mod
    det, _ = cache_mod.load(cache, "yolov8x")
    cache_mod.save(det.head(1), cache, "yolov8x", checkpoint_sha=f"{1:064d}",
                   conf_floor=0.001, imgsz=640)
    s = study.run(cache, pcsv, condition="matched", force_past_reconstruction=True)
    assert "yolov8x" in s.excluded
    m = report.macros(s)
    assert m["nArmsExcluded"] == "1" and "YOLOv8x" in m["armsExcluded"]
    assert m["armsExcludedWhy"] != "no checkpoint was excluded"


def test_identifiability_travels_into_the_report(built):
    s, _, _ = built
    m = report.macros(s)
    assert m["identifiable"] == "0"
    assert "collinear" in m["identifiabilityNote"]


# --------------------------------------------------------------------------- #
# The command line
# --------------------------------------------------------------------------- #

def test_the_cli_runs_and_writes_what_it_says(built, tmp_path, capsys):
    _, cache, pcsv = built
    rc = cli.main(["run", "--cache", str(cache), "--patients", str(pcsv),
                   "--condition", "matched", "--force-reconstruction",
                   "--out", str(tmp_path / "out")])
    assert rc == 0
    out = capsys.readouterr().out
    assert "confirmatory" in out and "intersection-union" in out
    res = json.loads((tmp_path / "out" / "results.json").read_text())
    assert res["provenance"] and res["confirmatory"]["mcnemar"]["alternative"] == "greater"
    assert (tmp_path / "out" / "factorial.csv").exists()
    assert (tmp_path / "out" / "provenance.json").exists()


def test_the_cli_reports_a_refusal_without_a_traceback(tmp_path, capsys):
    cache, pcsv = fixtures.write_cache(tmp_path, provenanced=True, seed=3)
    rc = cli.main(["run", "--cache", str(cache), "--patients", str(pcsv),
                   "--out", str(tmp_path / "out")])
    assert rc == 3
    err = capsys.readouterr().err
    assert "GateRefusal" in err and "What to do:" in err
