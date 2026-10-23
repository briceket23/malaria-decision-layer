"""The rule registry: the manuscript's (E, g, theta) triple, as data.

A rule is an evidence functional E that turns a patient's detections into one
number per class, a screening statistic g that turns that vector into a single
score, and a threshold theta. The patient is called uninfected when g < theta,
and otherwise carries the argmax class of E. Writing the family this way is
what makes the propositions checkable: scale invariance and screening
equivalence are statements about E and g, not about fourteen functions that
happen to be adjacent in a file.

Four of the definitions here exist in the exact form they do because the
obvious form was wrong, and each cost a confirmed defect in the predecessor:

  * argmax ties are broken by a declared order, not by class name. Alphabetical
    order decided 45% of one rule's calls; reversing only that component moved
    its speciation accuracy from 0.819 to 0.897 and its band from 3 to 2. The
    order is declared, and the number of ties is reported alongside the result.
  * noisy-OR accumulates in log space. In direct form it saturates to 1.0 after
    a few dozen fields and every patient becomes indistinguishable.
  * the screening score is t/(1+t), not 1 - exp(-t). The latter is exactly 1.0
    in float64 for t >= 37, and the median patient here carries t near 64, so
    116 of 170 patients collapsed onto a single value.
  * the trivial baseline breaks its own ties deterministically, so that it does
    not depend on the order rows arrive in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .evidence import PatientView
from .schema import PARASITE_CLASSES, UNINFECTED

#: Declared tie order for argmax over classes. Prevalence in the cohort, most
#: common first; ties are rare and this makes them reproducible rather than
#: alphabetical. Reported, never silent.
TIE_ORDER = ("P_falciparum", "P_vivax", "P_ovale", "P_malariae",
             "Babesia_divergens", "T_brucei")

assert set(TIE_ORDER) == set(PARASITE_CLASSES)

_TIE_RANK = {c: i for i, c in enumerate(TIE_ORDER)}


def argmax_class(ev: dict) -> tuple[str, int]:
    """(winning class, number of classes tied at the top).

    The tie count travels with the answer so that a result which depended on
    the declared order can be recognised as such.
    """
    # non-finite evidence is dropped rather than compared. `nan >= best - 1e-12`
    # is False for every class, so a single NaN emptied the tie list and the
    # `min()` below raised "arg is an empty sequence" -- from a rule, mid-run.
    ev = {c: float(v) for c, v in (ev or {}).items() if np.isfinite(v)}
    if not ev:
        return UNINFECTED, 0
    best = max(ev.values())
    tied = [c for c, v in ev.items() if v >= best - 1e-12]
    if not tied:
        return UNINFECTED, 0
    if len(tied) == 1:
        return tied[0], 1
    return min(tied, key=lambda c: _TIE_RANK.get(c, len(_TIE_RANK))), len(tied)


def screening_score(ev: dict) -> float:
    """Total parasite evidence, squashed into [0, 1) without saturating."""
    t = float(sum(max(v, 0.0) for v in ev.values()))
    return t / (1.0 + t)


# --------------------------------------------------------------------------- #
# Evidence functionals
# --------------------------------------------------------------------------- #

def _ev_sum(v: PatientView, p: dict) -> dict:
    return {c: v.total(c) for c in PARASITE_CLASSES}


def _ev_max(v: PatientView, p: dict) -> dict:
    return {c: (float(v.scores[c].max()) if v.n_boxes(c) else 0.0)
            for c in PARASITE_CLASSES}


def _ev_mean(v: PatientView, p: dict) -> dict:
    return {c: (float(v.scores[c].mean()) if v.n_boxes(c) else 0.0)
            for c in PARASITE_CLASSES}


def _ev_topk(v: PatientView, p: dict) -> dict:
    # k = 0 makes an empty slice, whose mean is NaN with two RuntimeWarnings,
    # and NaN evidence then broke the argmax two calls later
    k = max(int(p.get("k", 5)), 1)
    out = {}
    for c in PARASITE_CLASSES:
        s = v.scores[c]
        out[c] = float(np.sort(s)[::-1][:k].mean()) if s.size else 0.0
    return out


def _ev_count(v: PatientView, p: dict) -> dict:
    """How many detections clear a fixed confidence tau. The counting rule."""
    tau = float(p.get("tau", 0.5))
    return {c: float((v.scores[c] >= tau).sum()) for c in PARASITE_CLASSES}


def _ev_field_votes(v: PatientView, p: dict) -> dict:
    """One vote per field, cast for that field's strongest parasite class.

    The per-field confidence a field must reach is `vote_tau`, NOT `theta`.
    R5 declares `theta_key="theta"` for its screening threshold -- the share of
    fields that must vote -- so reading the same key here made one parameter
    stand for two quantities in two different units. The sweep then varied both
    at once and `search.profile` was a threshold profile of nothing.
    """
    tau = float(p.get("vote_tau", 0.5))
    votes = {c: 0.0 for c in PARASITE_CLASSES}
    # `.get`, and the length taken from the longest: a view whose per-field
    # arrays are absent or ragged raised KeyError or returned all-zero votes,
    # and the guard inspected only the first class
    per_field = [np.asarray(v.per_field_max.get(c, np.zeros(0)), dtype=float)
                 for c in PARASITE_CLASSES]
    width = max((a.size for a in per_field), default=0)
    if width == 0:
        return votes
    per_field = [a if a.size == width else
                 np.pad(a, (0, width - a.size)) for a in per_field]
    stacked = np.vstack(per_field)                       # (classes, fields)
    for j in range(stacked.shape[1]):
        col = stacked[:, j]
        if col.max() < tau:
            continue
        winners = {PARASITE_CLASSES[i]: float(col[i]) for i in range(len(col))}
        c, _ = argmax_class(winners)
        if c != UNINFECTED:
            votes[c] += 1.0
    return votes


def _ev_density(v: PatientView, p: dict) -> dict:
    """Parasites per erythrocyte, or per field when no erythrocyte was found.

    The fallback changes the UNIT of the statistic while rho stays where it
    was. A patient with three parasite boxes among 1,200 erythrocytes over 30
    fields has a density of 0.0025 and is called uninfected at rho = 0.05; the
    same patient with an empty RBC channel -- a loader that missed the class,
    a checkpoint without it -- has a density of 0.1 and is called falciparum.
    Nothing downstream could tell, because the docstring said "loudly" and the
    code said nothing at all.

    It is now recorded on the view, `Rule.apply` carries it out, and
    `study` counts the patients it affected.
    """
    rbc = v.n_boxes("RBC")
    if rbc > 0:
        denom, fallback = float(rbc), False
    else:
        denom, fallback = float(max(v.n_fields, 1)), True
    out = {c: v.n_boxes(c) / denom for c in PARASITE_CLASSES}
    p["_density_fallback"] = fallback
    return out


def _ev_noisy_or(v: PatientView, p: dict) -> dict:
    """1 - prod(1 - s), computed as -expm1(sum(log1p(-s))).

    The direct product saturates: with a few hundred detections above 0.9 every
    class reaches 1.0 in float64 and the argmax becomes a coin flip.
    """
    out = {}
    for c in PARASITE_CLASSES:
        s = np.clip(v.scores[c], 0.0, 1.0 - 1e-12)
        out[c] = float(-np.expm1(np.log1p(-s).sum())) if s.size else 0.0
    return out


def _ev_lse(v: PatientView, p: dict) -> dict:
    """Log-sum-exp with a fixed sharpness: between max and sum."""
    beta = float(p.get("beta", 10.0))
    out = {}
    for c in PARASITE_CLASSES:
        s = v.scores[c]
        if not s.size:
            out[c] = 0.0
            continue
        m = float(s.max())
        out[c] = float(m + np.log(np.exp(beta * (s - m)).sum()) / beta)
    return out


# --------------------------------------------------------------------------- #
# Screening statistics
# --------------------------------------------------------------------------- #

def _g_total(ev: dict, v: PatientView, p: dict) -> float:
    return float(sum(ev.values()))


def _g_max(ev: dict, v: PatientView, p: dict) -> float:
    return float(max(ev.values())) if ev else 0.0


def _g_vote_share(ev: dict, v: PatientView, p: dict) -> float:
    n = max(v.n_fields, 1)
    return float(sum(ev.values()) / n)


# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Rule:
    rid: str
    family: str
    evidence: Callable[[PatientView, dict], dict]
    screen: Callable[[dict, PatientView, dict], float]
    theta_key: str
    defaults: dict = field(default_factory=dict)
    note: str = ""

    def apply(self, v: PatientView, params: dict | None = None) -> dict:
        p = {**self.defaults, **(params or {})}
        ev = self.evidence(v, p)
        g = self.screen(ev, v, p)
        theta = float(p.get(self.theta_key, 0.0))
        if g < theta or all(x <= 0 for x in ev.values()):
            label, ties = UNINFECTED, 0
        else:
            label, ties = argmax_class(ev)
        return {"label": label, "evidence": ev, "screen": g, "theta": theta,
                "ties": ties, "confidence": screening_score(ev),
                # set by any functional that had to change its own denominator
                "fallback": bool(p.get("_density_fallback", False))}


RULES: dict[str, Rule] = {
    "R0": Rule("R0", "published", _ev_sum, _g_total, "theta", {"theta": 1e-9},
               "Sum of confidences, then argmax. theta at 0+ means one "
               "surviving parasite box condemns the smear, which is what the "
               "source study describes."),
    "R1": Rule("R1", "extremal", _ev_max, _g_max, "theta", {"theta": 0.50},
               "The single most confident detection decides."),
    "R2": Rule("R2", "central", _ev_mean, _g_max, "theta", {"theta": 0.50},
               "Mean confidence: robust to one lucky box, hostile to sparse "
               "true positives."),
    "R3": Rule("R3", "trimmed", _ev_topk, _g_max, "theta",
               {"theta": 0.30, "k": 5},
               "Mean of the k strongest detections."),
    "R4": Rule("R4", "counting", _ev_count, _g_max, "m",
               {"tau": 0.50, "m": 2.0},
               "At least m detections above tau. The rule the confirmatory "
               "comparison names."),
    "R5": Rule("R5", "voting", _ev_field_votes, _g_vote_share, "theta",
               {"theta": 0.50, "vote_tau": 0.50},
               "One vote per field, cast when that field's strongest parasite "
               "box clears vote_tau; the share of fields voting must then "
               "clear theta. Two thresholds, two names -- they were one key "
               "and the sweep varied both at once."),
    "R6": Rule("R6", "density", _ev_density, _g_max, "rho", {"rho": 0.05},
               "Parasitaemia-like: parasites per erythrocyte."),
    "R6b": Rule("R6b", "density", _ev_density, _g_max, "rho", {"rho": 0.001},
                "The same rule at a screening-sensitive threshold."),
    "R7": Rule("R7", "noisy-OR", _ev_noisy_or, _g_max, "theta", {"theta": 0.50},
               "Independent-evidence accumulation, in log space."),
    "R8": Rule("R8", "log-sum-exp", _ev_lse, _g_max, "theta",
               {"theta": 0.50, "beta": 10.0},
               "Between the max and the sum, with fixed sharpness."),
}

#: Rules whose parameters are fixed in advance. The exact McNemar test is valid
#: only against these; a rule fitted on the patients being tested is not
#: exchangeable with the baseline under the null.
FIXED_RULES = frozenset(RULES)

#: The six rules the MAIN paper reports. Protocol v3 §5.1.
#:
#: The previous version carried fourteen. On 170 patients that is a framework
#: competing with its own finding, and several of the fourteen were
#: near-duplicates of each other. These six span the decision principles rather
#: than filling a table; the rest are still computed -- the factorial needs
#: them -- and are reported in the supplement.
MAIN_RULES = ("R0", "R1", "R4", "R5", "R6")

#: Why each parameter holds the value it does, and whether it was registered.
#: Protocol v3 §5.3. None was fitted to this dataset: that is what makes the
#: confirmatory test valid, and it is why this table exists rather than being
#: left for a reviewer to ask about.
PARAMETER_RATIONALE = {
    ("R4", "tau"): ("0.50", "the source study's own reported operating point", True),
    ("R4", "m"): ("2", "minimum evidence: one box is a single-detection artefact", True),
    ("R3", "k"): ("5", "top-k robustness, an order of magnitude below the "
                       "median per-patient box count", True),
    ("R6", "rho"): ("0.05", "parasitaemia threshold in routine reporting", True),
    ("R6b", "rho"): ("0.001", "a screening-sensitive threshold, declared as a "
                              "sensitivity analysis", True),
    ("R8", "beta"): ("10", "log-sum-exp sharpness placing it between the max "
                           "and the sum", True),
    ("R5", "vote_tau"): ("0.50", "matches tau so the two counting-family rules "
                                 "share a per-detection floor", True),
}

#: What each rule's threshold MEANS, and what it depends on. Protocol v3 §5.2.
#:
#: Fixed parameters make rules pre-registered without making them comparable:
#: their thresholds live on different scales and depend on different things. A
#: reader who is not told this will assume each rule was given its best shot.
#: Columns: input quantity, what the threshold means, and three dependencies.
COMPARABILITY = {
    "R0":  ("sum of confidences", "total evidence", False, True, False),
    "R1":  ("maximum confidence", "one detection's score", False, False, False),
    "R2":  ("mean confidence", "average detection quality", False, True, False),
    "R3":  ("mean of the k strongest", "average of the top k", False, True, False),
    "R4":  ("count of detections >= tau", "a number of detections", False, True, False),
    "R5":  ("share of fields voting", "a fraction of fields examined", True, False, False),
    "R6":  ("parasites per erythrocyte", "parasitaemia", False, True, True),
    "R6b": ("parasites per erythrocyte", "parasitaemia", False, True, True),
    "R7":  ("noisy-OR accumulation", "an accumulated probability", False, True, False),
    "R8":  ("log-sum-exp of confidences", "a soft maximum", False, True, False),
}
COMPARABILITY_COLUMNS = ("input_quantity", "threshold_means",
                         "depends_on_field_count", "depends_on_duplicate_boxes",
                         "depends_on_rbc_detection")


def comparability_table():
    """The §5.2 table, derived from the registry rather than typed."""
    import pandas as pd

    rows = []
    for rid in all_rule_ids():
        fitted = rid in FITTED
        base = COMPARABILITY.get(rid)
        if fitted:
            base = ("calibrated screening probability", "a probability",
                    False, True, False)
        if base is None:
            continue
        rows.append({
            "rule": rid,
            "in_main_paper": rid in MAIN_RULES,
            "fitted": fitted,
            **dict(zip(COMPARABILITY_COLUMNS, base)),
            "can_abstain": True,          # every rule abstains via kappa
            "yields_a_probability": bool(fitted),
        })
    return pd.DataFrame(rows)


def parameter_table():
    """The §5.3 table, with the pre-registration status of every parameter."""
    import pandas as pd

    return pd.DataFrame([
        {"rule": rid, "parameter": param, "value": value,
         "rationale": why, "pre_registered": reg}
        for (rid, param), (value, why, reg) in PARAMETER_RATIONALE.items()
    ]).sort_values(["rule", "parameter"], kind="mergesort").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Fitted rules. Deliberately a separate registry.
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FittedRule:
    """A rule with a component estimated from the cohort being analysed.

    Kept out of `RULES` on purpose. Everything downstream derives "may I compute
    an exact p-value here?" from membership of `FIXED_RULES`, so a fitted rule
    cannot acquire a valid-looking p-value by being added to the wrong dict.

    The calibration acts on the SCREENING statistic only. It is a monotone map
    on one number, so it can move the infected/uninfected boundary and cannot
    reorder the classes; the species call is the same argmax as the fixed rule
    it is built on. Saying so here is what stops a reader from believing that
    calibration improved speciation, which it cannot.
    """

    rid: str
    family: str
    evidence: Callable[[PatientView, dict], dict]
    calibrator: str                     # isotonic | platt | temperature | none
    theta: float = 0.5
    theta_mode: str = "fixed"           # fixed | train_sensitivity
    target_sensitivity: float = 0.95
    confirm: str | None = None          # a fixed rule id used as stage two
    defaults: dict = field(default_factory=dict)
    note: str = ""

    @property
    def fitted_on(self) -> str:
        parts = []
        if self.calibrator != "none":
            article = "an" if self.calibrator[0] in "aeiou" else "a"
            parts.append(f"{article} {self.calibrator} map on the screening "
                         "statistic")
        if self.theta_mode == "train_sensitivity":
            parts.append(f"a threshold set for {self.target_sensitivity:.0%} "
                         "sensitivity")
        return (" and ".join(parts) + ", both cross-fitted by fold"
                if parts else "nothing")


FITTED: dict[str, FittedRule] = {
    "R9iso": FittedRule(
        "R9iso", "fitted", _ev_sum, "isotonic", theta=0.5,
        note="R0's evidence, screening decided on an isotonically calibrated "
             "probability at 0.5. Isotonic can reproduce any monotone "
             "distortion and is the most flexible of the three, which is also "
             "why it is the one most able to overfit a fold."),
    "R10platt": FittedRule(
        "R10platt", "fitted", _ev_sum, "platt", theta=0.5,
        note="Two parameters on the logit. Least flexible, most stable, and "
             "the one to prefer when the three disagree at this sample size."),
    "R11temp": FittedRule(
        "R11temp", "fitted", _ev_sum, "temperature", theta=0.5,
        note="One parameter. It can fix overconfidence and cannot fix a wrong "
             "prevalence, so a large residual intercept after temperature "
             "scaling is diagnostic rather than a failure of the method."),
    "R13staged": FittedRule(
        "R13staged", "staged", _ev_sum, "platt", theta_mode="train_sensitivity",
        target_sensitivity=0.95, confirm="R4",
        note="Screen then confirm, which is how a laboratory actually works. "
             "Stage one flags a patient whose calibrated probability clears a "
             "threshold set on the TRAINING folds for 95% sensitivity; stage "
             "two asks the counting rule to confirm, and a patient it does not "
             "confirm returns to uninfected. Both the map and the threshold "
             "are cross-fitted, because fitting the map out of fold and then "
             "choosing the threshold on everybody moves the leak rather than "
             "removing it."),
}

assert not (set(FITTED) & set(RULES)), "a rule id cannot be both fixed and fitted"


def rule_ids() -> tuple[str, ...]:
    """The fixed rules, in registry order. The factorial's rows."""
    return tuple(RULES)


def fitted_ids() -> tuple[str, ...]:
    return tuple(FITTED)


def all_rule_ids() -> tuple[str, ...]:
    return tuple(RULES) + tuple(FITTED)


def describe(rid: str) -> dict:
    """One row of documentation for any rule id, fixed or fitted."""
    if rid in RULES:
        r = RULES[rid]
        return {"rule": rid, "family": r.family, "fitted": False,
                "fitted_on": "nothing", "theta_key": r.theta_key,
                "defaults": dict(r.defaults), "note": r.note}
    r = FITTED[rid]
    return {"rule": rid, "family": r.family, "fitted": True,
            "fitted_on": r.fitted_on, "theta_key": "theta",
            "defaults": {"theta": r.theta, "theta_mode": r.theta_mode,
                         "calibrator": r.calibrator, "confirm": r.confirm},
            "note": r.note}


def predict(views: dict, order, rid: str, params: dict | None = None):
    """Apply one rule to every patient, in `order`.

    Returns (labels, evidence, screen, ties) as plain dicts keyed by patient.
    """
    import pandas as pd

    r = RULES[rid]
    labels, evs, screens, ties = {}, {}, {}, {}
    for pid in order:
        out = r.apply(views[str(pid)], params)
        labels[pid] = out["label"]
        evs[pid] = out["evidence"]
        screens[pid] = out["screen"]
        ties[pid] = out["ties"]
    return (pd.Series(labels, name=rid), pd.DataFrame(evs).T,
            pd.Series(screens, name=f"{rid}_g"), pd.Series(ties, name=f"{rid}_ties"))


def fit_predict(views: dict, order, truths, rid: str, *, k: int = 10,
                seed: int = 0):
    """Apply a fitted rule, cross-fitted, and say what was fitted on what.

    Returns (labels, calibrated, ties, detail). `detail` carries the fold count,
    how many folds degenerated, the thresholds actually used, and the flag that
    stops anything downstream computing an exact p-value against this arm.
    """
    import pandas as pd

    from . import calibrate

    r = FITTED[rid]
    ids = [str(p) for p in order]
    evs = []
    raw = np.empty(len(ids), dtype=float)
    for i, pid in enumerate(ids):
        ev = r.evidence(views[pid], dict(r.defaults))
        evs.append(ev)
        raw[i] = screening_score(ev)
    y = np.asarray([0.0 if str(truths[p]) == UNINFECTED else 1.0 for p in order])

    cf = calibrate.crossfit_detail(raw, y, r.calibrator, k=k, seed=seed,
                                   target_sensitivity=r.target_sensitivity)
    p = cf.calibrated
    theta = (cf.thresholds if r.theta_mode == "train_sensitivity"
             else np.full(len(ids), float(r.theta)))

    labels, ties, stage2 = {}, {}, 0
    for i, pid in enumerate(order):
        ev = evs[i]
        if p[i] < theta[i] or all(x <= 0 for x in ev.values()):
            labels[pid], ties[pid] = UNINFECTED, 0
            continue
        if r.confirm is not None:
            # stage two both confirms and speciates: the rule that decided the
            # patient is infected is the one that says of what, rather than one
            # rule confirming and a different functional naming the species
            out = RULES[r.confirm].apply(views[str(pid)])
            if out["label"] == UNINFECTED:
                labels[pid], ties[pid] = UNINFECTED, 0
                stage2 += 1
                continue
            labels[pid], ties[pid] = out["label"], out["ties"]
            continue
        labels[pid], ties[pid] = argmax_class(ev)

    detail = {
        "rule": rid, "fitted": True, "calibrator": r.calibrator,
        "fitted_on": r.fitted_on, "k": cf.k,
        "folds_fitted": cf.n_folds_fitted,
        "folds_degenerate": cf.n_folds_degenerate,
        "theta_mode": r.theta_mode,
        "theta_min": float(np.min(theta)), "theta_max": float(np.max(theta)),
        "theta_spread": float(np.max(theta) - np.min(theta)),
        "confirm_rule": r.confirm,
        "n_returned_by_stage_two": int(stage2),
        "exact_test_valid": False,
        "why_not": ("a component of this rule was estimated from the patients "
                    "it is scored on, so its discordant pairs are not "
                    "exchangeable with the baseline's under the null"),
    }
    if cf.n_folds_degenerate and not cf.n_folds_fitted:
        detail["warning"] = ("every fold degenerated to the identity map; this "
                             "arm is R0 under another name and should be read "
                             "as such")
    return (pd.Series(labels, name=rid), pd.Series(p, index=list(order),
                                                   name=f"{rid}_p"),
            pd.Series(ties, name=f"{rid}_ties"), detail)


# --------------------------------------------------------------------------- #
# Bounds. Not rules: they answer what a rule could have done.
# --------------------------------------------------------------------------- #

def trivial(order, truths, mode: str = "majority"):
    """B0. The prior alone, with a deterministic tie break."""
    import pandas as pd

    counts = truths.value_counts()
    top = counts.max()
    winner = min([c for c, n in counts.items() if n == top], key=str)
    label = winner if mode == "majority" else UNINFECTED
    return pd.Series({pid: label for pid in order}, name="B0")


def oracle(views: dict, order, truths):
    r"""B1. The best label any function of the cached detections could return.

    Defined by cases so that it is computable for every configuration,
    including the empty one:

        O(i) = uninfected                     if y_i is uninfected, or the
                                              cache for i is empty
             = y_i                            if the detector emitted at least
                                              one box of the true class
             = argmax over emitted classes    otherwise, by summed confidence
             = uninfected                     if no parasite class was emitted

    It uses the true label and is therefore not a rule; it is the ceiling on
    what aggregation can be worth on this cache, and the gap between it and the
    baseline is the only quantity here that answers the study's question
    without depending on which rules were tried.
    """
    import pandas as pd

    out = {}
    for pid in order:
        v = views[str(pid)]
        truth = truths[pid]
        emitted = {c: v.total(c) for c in PARASITE_CLASSES if v.n_boxes(c)}
        if truth == UNINFECTED or not emitted:
            out[pid] = UNINFECTED
        elif truth in emitted:
            out[pid] = truth
        else:
            out[pid], _ = argmax_class(emitted)
    return pd.Series(out, name="B1")
