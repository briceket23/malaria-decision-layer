"""Endpoints. Two decisions, computed separately and never conflated.

D1 is whether the patient is infected. D2 is which species, and it is always
evaluated on the fixed set of truly infected patients regardless of what D1
predicted, so its denominator is identical across rules and the paired
structure survives.

A patient without a prediction is an error, not a false positive. `NaN !=
"uninfected"` is True, so reindexing a short prediction series used to score a
patient the rule never saw as a false positive on the study's primary endpoint.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import PARASITE_CLASSES, UNINFECTED, regroup
from .stats import wilson


def _aligned(pred: pd.Series, truth: pd.Series) -> pd.Series:
    from .errors import BadInput

    p = pred.reindex(truth.index)
    if p.isna().any():
        missing = p.index[p.isna()].tolist()
        raise BadInput(
            f"{len(missing)} patients have no prediction (first: {missing[:3]})",
            remedy=("A missing prediction cannot be scored. Build the views "
                    "from the patient table so that every patient gets one, "
                    "including patients with no detections."))
    return p


def screening(truth: pd.Series, pred: pd.Series) -> dict:
    t = truth != UNINFECTED
    p = _aligned(pred, truth) != UNINFECTED
    tp = int((t & p).sum())
    tn = int((~t & ~p).sum())
    fp = int((~t & p).sum())
    fn = int((t & ~p).sum())
    sens = wilson(tp, tp + fn)
    spec = wilson(tn, tn + fp)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "n_pos": tp + fn, "n_neg": tn + fp,
            "sensitivity": sens.point, "sensitivity_ci": (sens.lo, sens.hi),
            "specificity": spec.point, "specificity_ci": (spec.lo, spec.hi),
            "correct_pos": (t & p)[t].tolist(),
            "correct_neg": (~t & ~p)[~t].tolist()}


def speciation(truth: pd.Series, pred: pd.Series, grouping=None) -> dict:
    mask = truth != UNINFECTED
    t = truth[mask].map(lambda x: regroup(x, grouping))
    p = _aligned(pred, truth)[mask].map(lambda x: regroup(x, grouping))
    correct = (t == p)
    ci = wilson(int(correct.sum()), int(len(correct)))
    return {"accuracy": ci.point, "ci": (ci.lo, ci.hi), "n": int(len(correct)),
            "correct": correct.tolist(), "index": list(t.index)}


def overall(truth: pd.Series, pred: pd.Series, grouping=None) -> dict:
    t = truth.map(lambda x: regroup(x, grouping))
    p = _aligned(pred, truth).map(lambda x: regroup(x, grouping))
    correct = (t == p)
    ci = wilson(int(correct.sum()), len(correct))
    per_class, recalls = {}, []
    for c in sorted(set(t)):
        m = t == c
        k, n = int((p[m] == c).sum()), int(m.sum())
        r = wilson(k, n)
        per_class[c] = {"k": k, "n": n, "recall": r.point, "ci": (r.lo, r.hi),
                        "interpretable": n >= 10}
        recalls.append(r.point)
    return {"accuracy": ci.point, "ci": (ci.lo, ci.hi), "n": len(correct),
            "n_correct": int(correct.sum()),
            "balanced_accuracy": float(np.mean(recalls)) if recalls else float("nan"),
            "per_class": per_class, "correct": correct.tolist()}


def confusion(truth: pd.Series, pred: pd.Series, grouping=None) -> pd.DataFrame:
    t = truth.map(lambda x: regroup(x, grouping))
    p = _aligned(pred, truth).map(lambda x: regroup(x, grouping))
    labels = sorted(set(t) | set(p))
    return pd.crosstab(t, p).reindex(index=labels, columns=labels, fill_value=0)


# --------------------------------------------------------------------------- #
# A descriptive band scale. Not a competence level.
# --------------------------------------------------------------------------- #

BANDS = ((0.90, 1), (0.80, 2), (0.70, 3), (0.00, 4))


def band(p: float) -> int:
    for thr, level in BANDS:
        if p >= thr:
            return level
    return 4


def band_referenced_accuracy(truth: pd.Series, pred: pd.Series) -> dict:
    """Species accuracy on Plasmodium patients, against band boundaries.

    The boundaries come from an external competence assessment, but nothing
    here is that assessment: it is a 55-slide panel with mixed-species slides,
    a time limit, and three scored components of which the awarded level is the
    lowest. This dataset has none of those, and every rule here returns one
    label, so all would score zero on the panel's mixed slides. A band is
    claimed only when the whole interval lies inside it.
    """
    plasmodium = tuple(c for c in PARASITE_CLASSES if c.startswith("P_"))
    mask = truth.isin(plasmodium)
    t, p = truth[mask], _aligned(pred, truth)[mask]
    k, n = int((t == p).sum()), int(mask.sum())
    if n == 0:
        return {"accuracy": float("nan"), "k": 0, "n": 0, "band": None,
                "span": (None, None), "unambiguous": False,
                "caveat": "no Plasmodium patients; no band is defined"}
    ci = wilson(k, n)
    lo_band, hi_band = band(ci.hi), band(ci.lo)
    return {"accuracy": ci.point, "ci": (ci.lo, ci.hi), "k": k, "n": n,
            "band": band(ci.point), "span": (lo_band, hi_band),
            "unambiguous": lo_band == hi_band,
            "caveat": "band-referenced accuracy; not a competence level"}
