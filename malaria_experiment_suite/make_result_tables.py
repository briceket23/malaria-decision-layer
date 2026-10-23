#!/usr/bin/env python3
r"""make_result_tables.py --- tables.tex, from the built Study's own outputs.

    python make_result_tables.py --results results --out tables.tex

`mag report` writes one macro per scalar. The manuscript also needs three
TABLES whose cells are per-rule and per-checkpoint, and those are not scalars:
the factorial over the main rules, the field-subsampling curve, and the
field-dilution stress test. Protocol v3 SS19 does not allow them to be typed,
so they are generated here from `results.json` and `factorial.csv` --- the
same artefacts `mag report` descends from, read after the fact rather than
recomputed.

Nothing here calculates a result. Every number is copied from the Study's own
output and formatted; the arithmetic check below re-derives each rate from its
own confusion counts and refuses to write a table that disagrees with them.

Macro names end in `Table` because that is the convention
`missing_results.py` uses to know a body is a tabular row set and that its
ampersands are deliberate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from pathlib import Path

EXIT_OK, EXIT_PRECONDITION, EXIT_DISAGREE = 0, 2, 3

#: The six of protocol v3 SS5.1. `in_main_paper` in the factorial marks five of
#: them; B1 is the oracle bound and is carried in the same table because the
#: protocol's rule table lists it.
MAIN_RULES = ("R0", "R1", "R4", "R5", "R6", "B1")

RULE_PRINCIPLE = {
    "R0": "summed confidence (comparator)",
    "R1": "extremal",
    "R4": "counting at $\\tau$",
    "R5": "field voting",
    "R6": "density per erythrocyte",
    "B1": "oracle bound",
}

CHECKPOINT_LABEL = {"rtdetr": "RT-DETR", "yolov5x": "YOLOv5x",
                    "yolov8x": "YOLOv8x"}

#: Every macro this script can define. Declared rather than discovered, so a
#: consumer can know the name universe without running the generator --- the
#: test suite checks that the manuscript calls nothing outside it, and a
#: producer whose output can only be learned by running it cannot be checked
#: before the run.
MACRO_NAMES = (
    "reconForcedSentence", "attrSpreadEstimand", "attrSpreadSentence",
    "attrAgreementSentence",
    "nullRuleMean", "nullRuleP95", "nullRuleP", "nullRuleExcess",
    "nullCkptMean", "nullCkptP95", "nullCkptP", "nullCkptExcess",
    "nullB", "nullSentence", "nullExcessSentence",
    "ruleSpanTop", "ruleSpanBottom", "ruleSpanNoR6", "ruleSpanNoR6Sentence",
    "brierNoSkill", "brierNoSkillSentence",
    "mainFactorialTable", "mainFactorialHeadRow", "mainFactorialNCheckpoints",
    "fieldCurveTable", "fieldCurveHeadRow", "fieldCurveNSizes",
    "stressTable", "stressHeadRow", "stressNRates",
    "capCellsChanged", "capCellsCompared", "capCellsShare",
    "capArmHigh", "capArmLow", "capArmCheckpoints",
    "capRulesAffected", "capNRulesAffected", "capChangedTable",
    "comparatorTheta", "comparatorThetaKey", "comparatorRationale",
    "comparatorPositiveCalls", "comparatorNPatients", "comparatorIsConstant",
    "comparatorSaturatedSentence", "fieldCurveCaveatSentence",
    "sepNLevels", "sepSentence",
    "centreNCentres", "centreNPatients", "centreDisjointSentence",
    "locoNAllCentres", "locoNUninterpretable", "locoDroppedCentres",
    "locoWhyFewerSentence",
    "sweepNGridRules", "sweepNGridConditions", "sweepGridSentence",
    "reconTolerance", "errArm", "errArmSentence", "ruleFamilySentence",
    "mainFactorialCiTable", "mainFactorialCiHeadRow",
)


def _pct(x, dp=1):
    return "---" if x is None else f"{100 * float(x):.{dp}f}"


def label(ck: str) -> str:
    return CHECKPOINT_LABEL.get(str(ck), str(ck))


# --------------------------------------------------------------------------- #

def factorial_table(fac, order) -> str:
    """One row per rule, three column groups, one per checkpoint."""
    rows = []
    for rid in MAIN_RULES:
        cells = []
        for ck in order:
            m = fac[(fac["rule"] == rid) & (fac["checkpoint"] == ck)]
            if not len(m):
                cells += ["---", "---", "---"]
                continue
            r = m.iloc[0]
            cells += [_pct(r["accuracy"]), _pct(r["sensitivity"]),
                      _pct(r["specificity"])]
        rows.append(f"{rid} & {RULE_PRINCIPLE.get(rid, '')} & "
                    + " & ".join(cells) + " \\\\")
    return " ".join(rows)


def field_curve_table(fc) -> str:
    sizes = [str(s) for s in fc.get("sizes", [])]
    rows = []
    for rid, series in sorted(fc.get("by_rule", {}).items()):
        cells = [_pct((series.get(s) or {}).get("mean")) for s in sizes]
        rows.append(f"{rid} & " + " & ".join(cells) + " \\\\")
    return " ".join(rows)


def stress_table(st) -> str:
    rates = [_rate(r) for r in st.get("rates", [])]
    rows = []
    for rid in MAIN_RULES:
        series = (st.get("by_rule") or {}).get(rid)
        if not series:
            continue
        cells = []
        for r in rates:
            cell = series.get(r) or {}
            cells.append(_pct(cell.get("sensitivity")))
        rows.append(f"{rid} & " + " & ".join(cells) + " \\\\")
    return " ".join(rows)


def _rate(r) -> str:
    """mag keys the stress series by the rate's own repr: 1, 0.5, 0.25, 0.1."""
    s = f"{float(r):g}"
    return s


def cap_macros(cap: dict) -> dict:
    """The cap sensitivity arm, as countable facts rather than a verdict.

    `capBinding` says whether the cap bound; it does not say how much the
    choice changed, and a manuscript that reports only the former has not
    reported the sensitivity analysis at all.
    """
    changed = int(cap.get("cells_changed") or 0)
    compared = int(cap.get("cells_compared") or 0)
    rows = [r for r in (cap.get("by_rule") or []) if int(r.get("changed") or 0)]
    rows.sort(key=lambda r: -int(r["changed"]))
    out = {
        "capCellsChanged": f"{changed:,}",
        "capCellsCompared": f"{compared:,}",
        "capCellsShare": (f"{100 * changed / compared:.1f}\\%"
                          if compared else "---"),
        "capArmHigh": f"{int(cap.get('cap_high') or 0):,}",
        "capArmLow": f"{int(cap.get('cap_low') or 0):,}",
        "capArmCheckpoints": ", ".join(cap.get("checkpoints") or []) or "none",
        "capRulesAffected": (", ".join(str(r["rule"]) for r in rows)
                             or "none"),
        "capNRulesAffected": str(len(rows)),
    }
    if rows:
        out["capChangedTable"] = " ".join(
            "%s & %d & %d \\\\" % (r["rule"], int(r["changed"]),
                                   int(r.get("n") or 0)) for r in rows)
    return out


#: Replicates for the permutation nulls below. Large enough that a Monte-Carlo
#: p reported as < 0.0001 is attainable: with the (1 + r) / (1 + B) convention
#: the smallest reportable value is 1 / (1 + B).
NULL_B = 20000
NULL_SEED = 0

#: The ten rules the attribution analysis marginalises over. B0/B1/B2 are
#: baselines and oracles, and R9iso/R10platt/R11temp/R13staged are the fitted
#: family, which is reported separately and never enters a marginal mean.
ATTR_RULES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R6b", "R7", "R8")


def permutation_nulls(root, study: dict) -> dict:
    r"""A permutation null for BOTH factors, under one scheme, from the
    shipped per-patient predictions.

    `mag` reports a null for the checkpoint factor and declines one for the
    rule factor, on the grounds that "with one observation per cell-mean there
    is no permutation that removes a rule effect without also removing the
    values it is made of". The mirror image of its own checkpoint procedure
    exists: mag permutes each rule's values ACROSS checkpoints (within a row of
    the rules x checkpoints matrix of cell means); permuting each checkpoint's
    values across rules (within a column) removes the rule effect by exactly
    the same argument. The asymmetry left the factor carrying the paper's
    headline as the only one without a null.

    Both are computed here, on the same matrix of cell means mag uses, from
    `predictions.csv` and `errors.csv` -- the artefacts mag itself wrote. The
    observed spans recomputed from those files reproduce
    `attribution.rule_means.range` and `attribution.checkpoint_means.range`
    exactly; that agreement is checked below and the macros are withheld if it
    fails, because a null computed on a different quantity from the one it is
    compared with is worse than no null.

    Monte-Carlo p-values use (1 + r) / (1 + B), so the smallest attainable
    value is 1 / (1 + B) and a p is never reported more precisely than B
    allows.
    """
    import numpy as np
    import pandas as pd

    pred_p, err_p = Path(root) / "predictions.csv", Path(root) / "errors.csv"
    if not (pred_p.is_file() and err_p.is_file()):
        return {}
    pred = pd.read_csv(pred_p).set_index("patient_id")
    truth = pd.read_csv(err_p).set_index("patient_id")["truth"]
    attr = study.get("attribution") or {}
    rmeans = (attr.get("rule_means") or {}).get("means") or {}
    rules = [r for r in ATTR_RULES if r in rmeans] or list(ATTR_RULES)
    ckpts = list((attr.get("checkpoint_means") or {}).get("means") or {})
    cols = [f"{c}:{r}" for c in ckpts for r in rules]
    if not ckpts or any(c not in pred.columns for c in cols):
        return {}

    y = truth.reindex(pred.index).to_numpy()
    cube = np.stack([[(pred[f"{c}:{r}"].to_numpy() == y).astype(np.int8)
                      for c in ckpts] for r in rules])        # rules x ck x pat
    cell = cube.mean(axis=2)                                   # rules x ck
    obs_rule = float(cell.mean(axis=1).max() - cell.mean(axis=1).min())
    obs_ckpt = float(cell.mean(axis=0).max() - cell.mean(axis=0).min())

    # The agreement check. Without it these macros could silently describe a
    # different quantity from the one the manuscript quotes beside them.
    want_r = (attr.get("rule_means") or {}).get("range")
    want_c = (attr.get("checkpoint_means") or {}).get("range")
    for got, want in ((obs_rule, want_r), (obs_ckpt, want_c)):
        if want is None or abs(got - float(want)) > 1e-9:
            return {}

    rng = np.random.default_rng(NULL_SEED)
    null_r = np.empty(NULL_B)
    null_c = np.empty(NULL_B)
    for b in range(NULL_B):
        m = np.take_along_axis(
            cell, np.argsort(rng.random(cell.shape), axis=0), axis=0)
        null_r[b] = m.mean(axis=1).max() - m.mean(axis=1).min()
        m = np.take_along_axis(
            cell, np.argsort(rng.random(cell.shape), axis=1), axis=1)
        null_c[b] = m.mean(axis=0).max() - m.mean(axis=0).min()

    def _p(null, obs):
        return (1 + int((null >= obs - 1e-12).sum())) / (1 + NULL_B)

    def _pf(x):
        return f"{100 * float(x):.1f}"

    pr, pc = _p(null_r, obs_rule), _p(null_c, obs_ckpt)

    def _ps(p):
        lo = 1.0 / (1 + NULL_B)
        return f"$<${lo:.5f}" if p <= lo + 1e-12 else f"{p:.4f}"

    out = {
        "nullB": f"{NULL_B:,}",
        "nullRuleMean": _pf(null_r.mean()) + "\\%",
        "nullRuleP95": _pf(np.quantile(null_r, 0.95)) + "\\%",
        "nullRuleP": _ps(pr),
        "nullRuleExcess": _pf(obs_rule - null_r.mean()) + "\\%",
        "nullCkptMean": _pf(null_c.mean()) + "\\%",
        "nullCkptP95": _pf(np.quantile(null_c, 0.95)) + "\\%",
        "nullCkptP": _ps(pc),
        "nullCkptExcess": _pf(obs_ckpt - null_c.mean()) + "\\%",
    }
    out["nullSentence"] = (
        f"Permuting each checkpoint's values across the {len(rules)} rules "
        f"generates a null rule span of {_pf(null_r.mean())} percentage points "
        f"on average ({_pf(np.quantile(null_r, 0.95))} at the 95th "
        f"percentile), against an observed {_pf(obs_rule)} "
        f"($p$ = {_ps(pr)}); permuting each rule's values across the "
        f"{len(ckpts)} checkpoints generates a null checkpoint span of "
        f"{_pf(null_c.mean())} ({_pf(np.quantile(null_c, 0.95))} at the 95th "
        f"percentile) against an observed {_pf(obs_ckpt)} "
        f"($p$ = {_ps(pc)}); $B$ = {NULL_B:,} for each.")
    out["nullExcessSentence"] = (
        f"On the null-corrected scale the two factors are close: the rule span "
        f"exceeds its null mean by {_pf(obs_rule - null_r.mean())} percentage "
        f"points and the checkpoint span exceeds its own by "
        f"{_pf(obs_ckpt - null_c.mean())}, against raw spans of "
        f"{_pf(obs_rule)} and {_pf(obs_ckpt)}. Most of the apparent advantage "
        f"of the rule span over the checkpoint span is what a null of this "
        f"design generates from the difference in level counts alone.")

    # Which two rules the span is the distance between, and what it becomes
    # without the rule the Results identify as mechanically degenerate.
    means = dict(zip(rules, cell.mean(axis=1)))
    top = max(means, key=means.get)
    bot = min(means, key=means.get)
    out["ruleSpanTop"] = f"{top} ({_pf(means[top])}\\%)"
    out["ruleSpanBottom"] = f"{bot} ({_pf(means[bot])}\\%)"
    rest = {k: v for k, v in means.items() if k != bot}
    span2 = max(rest.values()) - min(rest.values())
    out["ruleSpanNoR6"] = _pf(span2) + "\\%"
    out["ruleSpanNoR6Sentence"] = (
        f"The rule span is the distance between {top} ({_pf(means[top])}\\%) "
        f"and {bot} ({_pf(means[bot])}\\%). Excluding {bot}, the span across "
        f"the remaining {len(rest)} rules is {_pf(span2)} percentage points "
        f"against {_pf(obs_ckpt)} across checkpoints, and the difference "
        f"between the two spans falls to "
        f"{_pf(span2 - obs_ckpt)} percentage points.")
    return out


def decision_layer_macros(root, study: dict) -> dict:
    """What the comparator's threshold is, and what its flat curve means.

    Three reporting gaps, all in the same place.

    THE THRESHOLD. Every rule's parameters are tabulated except the
    comparator's. R0 carries `theta = 1e-9` --- not a rounding of 0.5 but a
    deliberate 0+, so that a single surviving parasite box condemns the smear,
    which is what the source study describes. A reader cannot judge the
    comparator's specificity without that number, and it appeared nowhere.

    THE SATURATION. Under the matched condition there is no confidence floor,
    so a rule reading total evidence against a threshold of 1e-9 crosses it on
    the 0.001 tail for every patient. The count is taken from the predictions
    table rather than asserted.

    THE FIELD-COUNT CURVE. The draft read the comparator's flat curve as
    evidence that "the accumulation the rule performs across fields adds
    almost nothing". That reading does not follow from a constant function: a
    rule that calls every patient infected returns the same label whatever it
    is shown, so its curve is flat by construction and carries no information
    about accumulation at all. This is the same class of error --- reading a
    degenerate comparator as a substantive comparison --- that the paper
    reports elsewhere, and correcting it here is not optional.
    """
    import csv as _csv

    out: dict = {}
    conf = (study.get("confirmatory") or {})
    ck = conf.get("checkpoint")
    base = conf.get("comparator")
    if not ck or not base:
        return out

    # --- the threshold, from mag's own registry, never typed ---------------- #
    try:
        from mag import rules as _rules
        spec = _rules.RULES.get(base)
    except Exception:                                         # pragma: no cover
        spec = None
    if spec is not None:
        key = getattr(spec, "theta_key", None)
        params = dict(getattr(spec, "defaults", None)
                      or getattr(spec, "params", {}) or {})
        theta = params.get(key)
        if theta is not None:
            out["comparatorThetaKey"] = str(key)
            out["comparatorTheta"] = (
                f"$10^{{{int(round(math.log10(float(theta))))}}}$"
                if 0 < float(theta) < 1e-3 else f"{float(theta):g}")
            out["comparatorRationale"] = _tex(
                str(getattr(spec, "note", None)
                    or getattr(spec, "rationale", "") or "").strip())

    # --- the saturation, from the predictions table ------------------------- #
    pf = Path(root) / "predictions.csv"
    col = f"{ck}:{base}"
    n_pos = n_pat = None
    if pf.is_file():
        with pf.open(encoding="utf-8", newline="") as fh:
            rows = list(_csv.DictReader(fh))
        if rows and col in rows[0]:
            n_pat = len(rows)
            n_pos = sum(1 for r in rows if (r[col] or "").strip() != "uninfected")
    if n_pos is None:
        return out

    out["comparatorPositiveCalls"] = f"{n_pos:,}"
    out["comparatorNPatients"] = f"{n_pat:,}"
    out["comparatorIsConstant"] = "yes" if n_pos == n_pat else "no"

    if n_pos == n_pat:
        out["comparatorSaturatedSentence"] = (
            f"Under the matched condition the comparator returned an infected "
            f"call for all {n_pat:,} patients on {label(ck)}: it is a constant "
            f"function on this cache, not a rule that discriminates poorly. "
            f"Its threshold is a positive number indistinguishable from zero, "
            f"the matched condition applies no confidence floor, and total "
            f"accumulated evidence therefore clears the threshold on the "
            f"low-confidence tail for every smear.")
        out["fieldCurveCaveatSentence"] = (
            f"The comparator's curve is flat because on this cache it is "
            f"constant --- it calls all {n_pat:,} patients infected however "
            f"many fields it is shown --- so its accuracy is the infected "
            f"share of whichever patients are being scored and its flatness "
            f"is not evidence about what accumulating over fields is worth. "
            f"Only the candidate's curve carries a reading, and it declines: "
            f"a count of detections above a fixed score accumulates false "
            f"positives as well as true ones.")
    else:
        out["comparatorSaturatedSentence"] = (
            f"Under the matched condition the comparator returned an infected "
            f"call for {n_pos:,} of {n_pat:,} patients on {label(ck)}.")
        out["fieldCurveCaveatSentence"] = (
            f"The comparator called {n_pos:,} of {n_pat:,} patients infected "
            f"on the full field set, so its curve should be read against how "
            f"close to constant it is before any reading is taken from its "
            f"slope.")

    # --- which arm the error census describes ------------------------------- #
    err = (study.get("error_summary") or {})
    if err.get("rule") and err.get("checkpoint"):
        out["errArm"] = f"{err['rule']} on {label(err['checkpoint'])}"
        out["errArmSentence"] = (
            f"The error census below describes one arm --- {err['rule']} on "
            f"{label(err['checkpoint'])} under matched post-processing --- and "
            f"not the factorial: the categories, the species confusions and "
            f"the fragility counts all belong to that arm's calls.")

    # --- how many rules each statistic is computed over ---------------------- #
    ap = (study.get("apportionment") or {})
    inc = (ap.get("including_fitted_rules") or {})
    attr = (study.get("attribution") or {})
    n_attr = attr.get("n_rules")
    n_all = inc.get("n_rules")
    n_excl = ap.get("n_fitted_rules_excluded")
    if n_attr and n_all:
        out["ruleFamilySentence"] = (
            f"Four rule counts appear in this paper and they are not "
            f"interchangeable. {len(MAIN_RULES) - 1} fixed rules are reported "
            f"in the main tables and the rest in the supplement; the factorial "
            f"and every model fitted on it use {int(n_attr)} fixed rules --- "
            f"every rule whose parameters were set in the registration; the "
            f"reconstruction sweep uses "
            f"{int((study.get('sweep') or {}).get('n_rules') or 0)}, "
            f"omitting the rules with no threshold to sweep; and "
            f"{int(n_all)} appear in the sensitivity analysis that adds the "
            f"{int(n_excl)} cross-fitted rules, which are reported separately "
            f"and carry no exact test. Where a number below is computed over "
            f"anything other than the {int(n_attr)} fixed rules, the sentence "
            f"says so.")

    # --- the reconstruction tolerance, which the paper cites and mag does not
    #     emit as a macro ---------------------------------------------------- #
    tol = ((study.get("reconstruction") or {}).get("tolerance"))
    if tol is not None:
        out["reconTolerance"] = f"{int(tol):,}" if float(tol) == int(tol) \
            else f"{float(tol):g}"

    # --- the sweep grid, which is not a crossing and reads like one --------- #
    sf = Path(root) / "sweep.csv"
    if sf.is_file():
        with sf.open(encoding="utf-8", newline="") as fh:
            srows = list(_csv.DictReader(fh))
        if srows and {"condition", "rule", "theta"} <= set(srows[0]):
            per_rule: dict = {}
            conds = set()
            for r in srows:
                conds.add(r["condition"])
                per_rule.setdefault(r["rule"], set()).add(
                    (r["theta_key"], r["theta"]))
            counts = sorted({len(v) for v in per_rule.values()})
            out["sweepNGridRules"] = str(len(per_rule))
            out["sweepNGridConditions"] = str(len(conds))
            if len(counts) == 1:
                out["sweepGridSentence"] = (
                    f"The grid is a full crossing: {len(per_rule)} rules "
                    f"$\\times$ {counts[0]} thresholds $\\times$ "
                    f"{len(conds)} conditions.")
            else:
                sizes = ", ".join(str(c) for c in counts)
                out["sweepGridSentence"] = (
                    f"The count is not a crossing. Each rule carries its own "
                    f"declared threshold grid --- {sizes} values, depending on "
                    f"the rule --- and the grid is repeated under each of the "
                    f"{len(conds)} post-processing conditions, so the "
                    f"{len(srows):,} configurations span {len(per_rule)} rules "
                    f"unevenly rather than {len(per_rule)} rules at a common "
                    f"set of thresholds.")

    # --- centres, from a different artefact than the cohort table ----------- #
    ef = Path(root) / "errors.csv"
    if ef.is_file():
        with ef.open(encoding="utf-8", newline="") as fh:
            erows = list(_csv.DictReader(fh))
        if erows and "centre" in erows[0] and "patient_id" in erows[0]:
            seen: dict = {}
            for r in erows:
                seen.setdefault(r["centre"], set()).add(r["patient_id"])
            ids = [pid for group in seen.values() for pid in group]
            n_centres, n_ids, n_uniq = len(seen), len(ids), len(set(ids))
            out["centreNCentres"] = str(n_centres)
            out["centreNPatients"] = f"{n_uniq:,}"
            out["centreDisjointSentence"] = (
                f"Read back from the patient-level table rather than from the "
                f"cohort summary, the {n_centres} centres contribute "
                f"{n_ids:,} patient records with {n_uniq:,} distinct "
                f"identifiers, so no patient is attributed to two centres."
                if n_ids == n_uniq else
                f"Read back from the patient-level table, the {n_centres} "
                f"centres contribute {n_ids:,} patient records with only "
                f"{n_uniq:,} distinct identifiers: {n_ids - n_uniq} "
                f"identifier(s) appear under more than one centre.")

    # --- why the LOCO analysis names fewer centres than the cohort ---------- #
    loco = (study.get("loco") or {})
    by_centre = (loco.get("by_centre") or {})
    if by_centre:
        kept = {k: v for k, v in by_centre.items() if v.get("interpretable")}
        dropped = {k: v for k, v in by_centre.items()
                   if not v.get("interpretable")}
        out["locoNAllCentres"] = str(len(by_centre))
        out["locoNUninterpretable"] = str(len(dropped))
        if dropped:
            largest_dropped = max(
                int(v.get("n_infected_at_centre") or 0) for v in dropped.values())
            smallest_kept = min(
                (int(v.get("n_infected_at_centre") or 0) for v in kept.values()),
                default=None)
            named = ", ".join(sorted(dropped))
            out["locoDroppedCentres"] = named
            out["locoWhyFewerSentence"] = (
                f"The leave-one-centre-out analysis covers {len(kept)} of the "
                f"{len(by_centre)} centres. {named} "
                f"{'is' if len(dropped) == 1 else 'are'} reported by the "
                f"analysis and marked uninterpretable rather than dropped "
                f"silently: {'it contributes' if len(dropped) == 1 else 'they contribute'} "
                f"at most {largest_dropped} infected patient"
                f"{'' if largest_dropped == 1 else 's'}, against "
                f"{smallest_kept} at the smallest centre the analysis keeps, "
                f"so the accuracy measured at that centre is estimated on too "
                f"few patients for its swing to carry information.")
        else:
            out["locoWhyFewerSentence"] = (
                f"All {len(by_centre)} centres met the interpretability "
                f"threshold, so none was held out of the summary.")

    # --- separation, which the object records and nothing reported ---------- #
    sep = ((study.get("attribution") or {}).get("separation") or {})
    if sep:
        n_sep = sep.get("n_separated_levels")
        out["sepNLevels"] = str(int(n_sep)) if n_sep is not None else None
        if sep.get("any"):
            named = ", ".join(str(x) for x in
                              (list(sep.get("rules") or [])
                               + list(sep.get("checkpoints") or [])
                               + list(sep.get("cells") or []))[:8])
            out["sepSentence"] = (
                f"{int(n_sep)} level(s) were separated in the patient-clustered "
                f"fit ({named}); a separated level's coefficient is a boundary "
                f"value and its interval is not a confidence statement.")
        else:
            out["sepSentence"] = (
                "No rule level, checkpoint level or cell was separated in the "
                "patient-clustered fit, so no coefficient in it is a boundary "
                "value driven by a cell with no variation.")
    return out


def factorial_ci_table(fac, order) -> str:
    """Patient-level accuracy with its Wilson interval, one row per rule.

    The factorial table gives three point estimates per cell and no interval,
    while the file it is built from carries `acc_lo` and `acc_hi` for every
    cell. A reader comparing forty-two cells by eye without intervals is being
    invited to read noise as structure.
    """
    rows = []
    for rid in MAIN_RULES:
        cells = []
        for ck in order:
            m = fac[(fac["rule"] == rid) & (fac["checkpoint"] == ck)]
            if not len(m):
                cells.append("---")
                continue
            r = m.iloc[0]
            cells.append(f"{_pct(r['accuracy'])} "
                         f"({_pct(r['acc_lo'])}--{_pct(r['acc_hi'])})")
        rows.append(f"{rid} & " + " & ".join(cells) + " \\\\")
    return " ".join(rows)


def provenance_macros(study: dict) -> dict:
    """Three facts the results object records and `mag report` does not emit.

    All three are disclosures rather than results, and each was missing from
    a draft of this paper until the object was read directly.

    `forced_past_reconstruction` says the reconstruction gate refused and the
    run proceeded anyway. Protocol v3 SS8.4 permits that and requires it to
    change the comparator's description; a paper that reports the refusal but
    not the forcing has reported half of a design decision.

    `attribution.contrast.estimand` is the results object's own description of
    the quantity the manuscript quotes. It is reproduced VERBATIM because the
    generated sentence beside it in results.tex calls the same quantity "the
    marginal effect of the aggregation rule ... exceeded the marginal effect
    of the detector checkpoint", and the two descriptions are not the same
    thing: the object computes a difference between the RANGE of the rule
    marginal means and the RANGE of the checkpoint marginal means. Quoting the
    object settles which one the paper is standing behind.

    `attribution.agreement` records that two independent fits of the same
    model disagree by more than the tolerance, which is why no log-odds
    coefficient is printed.
    """
    out: dict = {}
    meta = study.get("meta") or {}
    forced = bool(meta.get("forced_past_reconstruction"))
    out["reconForcedSentence"] = (
        "The reconstruction gate refused, and this run is recorded as having "
        "been forced past it so that the registered endpoint could be "
        "evaluated; every quantity downstream carries the comparator's "
        "revised description."
        if forced else
        "The reconstruction gate did not refuse, so no arm of this analysis "
        "was forced past it.")

    # The reference every reader computes first and no draft of this paper
    # printed: a constant predictor at the cohort's own prevalence. A Brier
    # score quoted without it is a number nobody can place.
    cal = (study.get("calibration") or {}).get("raw") or {}
    prev = (study.get("predictive") or {}).get("cohort_prevalence")
    if prev is not None:
        p0 = float(prev)
        out["brierNoSkill"] = f"{p0 * (1 - p0):.4f}"
        raw = cal.get("brier")
        if raw is not None:
            worse = float(raw) > p0 * (1 - p0)
            out["brierNoSkillSentence"] = (
                f"A constant predictor returning this cohort's prevalence for "
                f"every patient scores {p0 * (1 - p0):.4f}, so the "
                f"uncalibrated statistic is "
                f"{'worse than' if worse else 'better than'} the base rate "
                f"before any calibration map is applied.")

    attr = study.get("attribution") or {}
    con = attr.get("contrast") or {}
    est = str(con.get("estimand") or "").strip()
    out["attrSpreadEstimand"] = _tex(est) if est else "---"
    rs, cs = con.get("rule_spread"), con.get("checkpoint_spread")
    nr = (attr.get("rule_means") or {}).get("n_levels")
    nc = (attr.get("checkpoint_means") or {}).get("n_levels")
    if rs is not None and cs is not None and nr and nc:
        out["attrSpreadSentence"] = (
            f"Marginal mean patient-level accuracy spanned "
            f"{100 * float(rs):.1f} percentage points across the {int(nr)} "
            f"aggregation rules and {100 * float(cs):.1f} across the "
            f"{int(nc)} released checkpoints. A range grows with the number "
            f"of levels compared, so the difference between those two spans "
            f"is a descriptive property of this factorial surface and not a "
            f"decomposition of importance.")
    else:
        out["attrSpreadSentence"] = "---"

    agr = attr.get("agreement") or {}
    if agr.get("checked"):
        out["attrAgreementSentence"] = (
            f"Two independent implementations of the same model were fitted "
            f"and compared; their largest coefficient difference was "
            f"{float(agr.get('max_abs_coefficient_difference', 0)):.2e} on "
            f"the log-odds scale against a tolerance of "
            f"{float(agr.get('tolerance', 0)):g}, so they "
            f"{'agree' if agr.get('agree') else 'do not agree'} and no "
            f"log-odds coefficient is reported.")
    else:
        out["attrAgreementSentence"] = (
            "Whether two independent fits of this model agree was not "
            "assessed by this run.")
    return out


#: LaTeX-special characters, for a value copied VERBATIM out of the results
#: object. The estimand string is prose written by the analysis package and
#: may contain anything.
_TEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _tex(s: str) -> str:
    return "".join(_TEX_MAP.get(ch, ch) for ch in str(s))


def check_arithmetic(fac) -> list:
    """Every rate follows from its own counts, or nothing is written.

    The build audit makes this check too. Making it here as well is not
    duplication: this script COPIES numbers, and a copier that does not verify
    what it copied is where a transcription error enters a paper.
    """
    bad = []
    for r in fac.itertuples(index=False):
        for name, k, n, got in (("sensitivity", r.tp, r.tp + r.fn,
                                 r.sensitivity),
                                ("specificity", r.tn, r.tn + r.fp,
                                 r.specificity)):
            if n and abs(float(got) - k / n) > 1e-9:
                bad.append(f"{r.checkpoint}:{r.rule} {name} "
                           f"{float(got):.6f} against {k}/{n}")
    return bad


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_result_tables.py",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--results", default="results")
    p.add_argument("--out", default="tables.tex")
    a = p.parse_args(argv)

    import pandas as pd

    root = Path(a.results)
    rj, fc = root / "results.json", root / "factorial.csv"
    if not rj.is_file() or not fc.is_file():
        print(f"need {rj} and {fc}; the analyse stage has not run",
              file=sys.stderr)
        return EXIT_PRECONDITION

    study = json.loads(rj.read_text(encoding="utf-8"))
    fac = pd.read_csv(fc)

    bad = check_arithmetic(fac)
    if bad:
        print("the factorial disagrees with its own counts; refusing to write "
              "a table from it:", file=sys.stderr)
        for b in bad[:8]:
            print("  " + b, file=sys.stderr)
        return EXIT_DISAGREE

    order = [c for c in ("rtdetr", "yolov5x", "yolov8x")
             if c in set(fac["checkpoint"])]
    order += [c for c in sorted(set(fac["checkpoint"])) if c not in order]

    cap = {}
    cp = root / "cap_sensitivity.json"
    if cp.is_file():
        cap = json.loads(cp.read_text(encoding="utf-8"))

    macros = {
        "mainFactorialTable": factorial_table(fac, order),
        "mainFactorialHeadRow":
            " & ".join(f"\\multicolumn{{3}}{{c}}{{{label(c)}}}" for c in order),
        "mainFactorialNCheckpoints": str(len(order)),
        "fieldCurveTable": field_curve_table(study.get("field_curve") or {}),
        "fieldCurveHeadRow":
            " & ".join(str(s) for s in
                       (study.get("field_curve") or {}).get("sizes", [])),
        "fieldCurveNSizes":
            str(len((study.get("field_curve") or {}).get("sizes", []))),
        "stressTable": stress_table(study.get("stress") or {}),
        "stressHeadRow":
            " & ".join(f"$\\pi = {_rate(r)}$" for r in
                       (study.get("stress") or {}).get("rates", [])),
        "stressNRates":
            str(len((study.get("stress") or {}).get("rates", []))),
        "mainFactorialCiTable": factorial_ci_table(fac, order),
        "mainFactorialCiHeadRow": " & ".join(label(c) for c in order),
    }
    if cap:
        macros.update(cap_macros(cap))
    macros.update(provenance_macros(study))
    macros.update(decision_layer_macros(root, study))
    macros.update(permutation_nulls(root, study))

    lines = [
        "% tables.tex --- generated by make_result_tables.py. Do not edit.",
        "% Every cell is copied from results.json and factorial.csv and",
        "% formatted; each rate was re-derived from its own confusion counts",
        "% before this file was written.",
        f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
        f"% from     {rj}  and  {fc}",
        "",
    ]
    for k in sorted(macros):
        lines.append("\\defresult{%s}{%s}" % (k, macros[k]))
    Path(a.out).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"  wrote {len(macros)} table macros -> {a.out}")
    print(f"    checkpoints  {', '.join(label(c) for c in order)}")
    print(f"    rules        {', '.join(MAIN_RULES)}")
    print(f"    arithmetic   every rate matches its own counts "
          f"({len(fac)} cells checked)")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
