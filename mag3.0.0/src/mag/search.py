"""How far is the published operating point from anything we can reproduce?

Declared post hoc, in this docstring, in the returned object, and in the macro
the manuscript prints. Nothing here is an estimate of anything: every
configuration it reports was chosen because of how it scored, so its score is a
maximum over a search and not a measurement.

The question it answers is narrow and worth answering. The baseline reproduces
96 of 170 against a published 135, and a reader is entitled to ask whether some
other reasonable choice of rule, threshold and post-processing would have landed
on 135 -- in which case the discrepancy is a matter of unreported settings -- or
whether nothing in a wide sweep gets there, in which case it is not.

There is a second question, and it is the more informative one. A configuration
can hit 135 of 170 with a completely different sensitivity and specificity split
from the published one. `agreement_at_target` reports the split of every
configuration that hits the total, so that "reproduced the headline number" is
not mistaken for "reproduced the result".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .evidence import CONDITIONS, build_views
from .metrics import overall, screening
from .rules import RULES, predict
from .schema import GROUPING_5

#: The thresholds swept for each rule, on the rule's own theta key. Declared
#: here so that the grid is visible rather than buried in a loop, and so that a
#: reader can see it is a coarse sweep of plausible values and not a fine
#: optimisation.
GRID = {
    "theta": (1e-9, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90),
    "m": (1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0),
    "rho": (1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.05, 0.10, 0.25),
}


@dataclass(frozen=True)
class Sweep:
    table: pd.DataFrame
    target: int
    n_total: int
    tolerance: int
    #: rules dropped because the grid makes them identical to one already here
    aliases: dict = None
    post_hoc: bool = True
    note: str = ("a post hoc characterisation of the reproduction gap; every "
                 "configuration here was chosen by its score, so no accuracy in "
                 "this table estimates the accuracy of anything")


def sweep(detections, patients, truth, *, conditions=CONDITIONS,
          rule_ids=None, target: int = 135, tolerance: int = 5) -> Sweep:
    """Score every (condition, rule, threshold) in the declared grid.

    One row per configuration. The views are built once per condition rather
    than once per configuration, which is the difference between a sweep that
    runs in seconds and one that rebuilds the same suppression a hundred times.
    """
    order = patients["patient_id"].tolist()
    ids = list(rule_ids) if rule_ids else list(RULES)
    # R6 and R6b share their evidence, their screen and their threshold key,
    # and differ only in a DEFAULT that the grid then overrides -- so every R6b
    # row was a byte-identical duplicate of an R6 row. `n_configurations` and
    # `n_reproducing`, the two numbers the sweep's sentence asks a reader to
    # weigh, counted that rule twice.
    signature, deduped, aliases = set(), [], {}
    for rid in ids:
        r = RULES[rid]
        key = (r.evidence, r.screen, r.theta_key,
               tuple(sorted((k, v) for k, v in r.defaults.items()
                            if k != r.theta_key)))
        if key in signature:
            aliases.setdefault(
                next(o for o in deduped
                     if (RULES[o].evidence, RULES[o].screen, RULES[o].theta_key,
                         tuple(sorted((k, v) for k, v in RULES[o].defaults.items()
                                      if k != RULES[o].theta_key))) == key),
                []).append(rid)
            continue
        signature.add(key)
        deduped.append(rid)
    ids = deduped
    rows = []
    for condition in conditions:
        views = build_views(detections, patients, condition)
        for rid in ids:
            rule = RULES[rid]
            for theta in GRID.get(rule.theta_key, GRID["theta"]):
                pred, _, _, ties = predict(views, order, rid,
                                           {rule.theta_key: theta})
                ov = overall(truth, pred, GROUPING_5)
                sc = screening(truth, pred)
                rows.append({
                    "condition": condition, "rule": rid,
                    "theta_key": rule.theta_key, "theta": float(theta),
                    "n_correct": int(ov["n_correct"]), "n_total": int(ov["n"]),
                    "accuracy": float(ov["accuracy"]),
                    "sensitivity": float(sc["sensitivity"]),
                    "specificity": float(sc["specificity"]),
                    "tp": sc["tp"], "tn": sc["tn"], "fp": sc["fp"], "fn": sc["fn"],
                    "argmax_ties": int((ties > 1).sum()),
                    "distance_to_target": abs(int(ov["n_correct"]) - int(target)),
                })
    table = pd.DataFrame(rows).sort_values(
        ["distance_to_target", "condition", "rule", "theta"],
        kind="mergesort").reset_index(drop=True)
    return Sweep(table=table, target=int(target),
                 n_total=int(len(order)), tolerance=int(tolerance),
                 aliases={k: tuple(v) for k, v in aliases.items()})


def summarise(s: Sweep, *, published_sensitivity: float | None = None,
              published_specificity: float | None = None) -> dict:
    """What the sweep found, and what it does not license anyone to say."""
    t = s.table
    if t.empty:
        return {"n_configurations": 0,
                "note": "the grid produced no configurations"}
    hits = t[t["distance_to_target"] <= s.tolerance]
    best = t.iloc[0]
    out = {
        "n_configurations": int(len(t)),
        "n_conditions": int(t["condition"].nunique()),
        "n_rules": int(t["rule"].nunique()),
        "target": s.target, "tolerance": s.tolerance, "n_total": s.n_total,
        "any_reproduces": bool(len(hits) > 0),
        "n_reproducing": int(len(hits)),
        "closest": {"condition": str(best["condition"]), "rule": str(best["rule"]),
                    "theta_key": str(best["theta_key"]),
                    "theta": float(best["theta"]),
                    "n_correct": int(best["n_correct"]),
                    "gap": int(best["distance_to_target"]),
                    "sensitivity": float(best["sensitivity"]),
                    "specificity": float(best["specificity"])},
        "range_n_correct": (int(t["n_correct"].min()), int(t["n_correct"].max())),
        "post_hoc": True, "note": s.note,
    }
    if len(hits):
        out["agreement_at_target"] = {
            "sensitivity_range": (float(hits["sensitivity"].min()),
                                  float(hits["sensitivity"].max())),
            "specificity_range": (float(hits["specificity"].min()),
                                  float(hits["specificity"].max())),
            "rules": sorted(set(hits["rule"].astype(str))),
            "conditions": sorted(set(hits["condition"].astype(str))),
        }
        if (published_sensitivity is not None
                and published_specificity is not None):
            near = hits[(np.abs(hits["sensitivity"] - published_sensitivity) <= 0.05)
                        & (np.abs(hits["specificity"] - published_specificity) <= 0.05)]
            out["agreement_at_target"]["n_matching_published_split"] = int(len(near))
            out["agreement_at_target"]["matching_published_split"] = bool(len(near))
    out["sentence"] = (
        (f"across {len(t)} configurations spanning {t['rule'].nunique()} rules, "
         f"{t['condition'].nunique()} post-processing conditions and the "
         f"declared threshold grid, {len(hits)} landed within {s.tolerance} of "
         f"the published {s.target}/{s.n_total}"
         + (f"; the closest was {best['rule']} at "
            f"{best['theta_key']}={best['theta']:g} under {best['condition']}, "
            f"scoring {int(best['n_correct'])}/{s.n_total}"
            if not len(hits) else ""))
        + ("; this is a post hoc sweep and the configurations it names were "
           "selected by their scores, so none of these accuracies estimates "
           "out-of-sample performance"))
    return out


def profile(s: Sweep, rule: str, condition: str) -> pd.DataFrame:
    """One rule's threshold profile: the shape behind a single reported number.

    Useful in one specific way. If a rule's accuracy is flat across two orders
    of magnitude of its threshold, the threshold was never the interesting
    choice; if it swings twenty points, then quoting one value without the
    profile hides most of what the rule does.
    """
    t = s.table
    sub = t[(t["rule"] == rule) & (t["condition"] == condition)]
    return sub.sort_values("theta", kind="mergesort").reset_index(drop=True)
