"""Four refusals, evaluated before anything they could contaminate.

A gate is not a check that prints a warning. It stops the analysis, and none of
the four can be waved through by naming a later stage: `study.run` evaluates
them in order and raises `GateRefusal`, which the command line reports and does
not catch.

  yield          A checkpoint that produced almost nothing did not detect
                 badly; it was not driven correctly. Declared post hoc: it was
                 written after a released checkpoint returned two boxes across
                 4,508 fields.
  reconstruction If the comparator does not reconstruct the published figure,
                 every downstream comparison is against our own reconstruction
                 of the published rule and must say so. Protocol v3 §3 retires
                 the word 'reproduction' here: it licenses the reading that the
                 original paper was wrong, which this design cannot support.
  design         A confirmatory comparison whose candidate is screening-
                 equivalent to the baseline, or whose baseline has no headroom,
                 cannot produce a non-null result. Spending alpha on it is
                 spending alpha on arithmetic.
  identifiability If every uninfected patient comes from one centre, specificity
                 and domain are perfectly collinear and no estimator separates
                 them. This is rank deficiency, not low power.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .errors import GateRefusal
from .schema import UNINFECTED

#: At a 0.001 floor every working detector in this study fires on essentially
#: every field: a smear is dense with erythrocytes and each checkpoint carries
#: an RBC class. These thresholds are deliberately far from any defensible
#: detector behaviour, because the gate is meant to catch a harness fault and
#: not to grade detectors.
YIELD_MIN_FIELD_COVERAGE = 0.5
YIELD_MIN_BOXES_PER_FIELD = 1.0
YIELD_APPLIES_BELOW_FLOOR = 0.05

PUBLISHED_N = 135
PUBLISHED_TOTAL = 170
REPRO_TOLERANCE = 5


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: dict = field(default_factory=dict)
    message: str = ""


def yield_gate(manifest, n_fields_expected: int) -> GateResult:
    """Is this cache a measurement at all?"""
    floor = float(getattr(manifest, "conf_floor", 0.0) or 0.0)
    n_det = int(getattr(manifest, "n_detections", 0) or 0)
    n_img = int(getattr(manifest, "n_images", 0) or 0)
    name = str(getattr(manifest, "checkpoint", "?"))
    detail = {"conf_floor": floor, "n_detections": n_det, "n_images": n_img,
              "n_fields_expected": int(n_fields_expected)}
    if floor > YIELD_APPLIES_BELOW_FLOOR or n_fields_expected <= 0:
        return GateResult("yield", True, detail,
                          "not applicable above a 0.05 confidence floor")
    coverage = n_img / n_fields_expected
    per_field = n_det / max(n_fields_expected, 1)
    detail.update(coverage=coverage, boxes_per_field=per_field)
    if coverage < YIELD_MIN_FIELD_COVERAGE:
        return GateResult("yield", False, detail,
                          f"{name}: {n_img:,} of {n_fields_expected:,} fields "
                          f"({coverage:.2%}) produced any box at a floor of {floor}")
    if per_field < YIELD_MIN_BOXES_PER_FIELD:
        return GateResult("yield", False, detail,
                          f"{name}: {n_det:,} boxes over {n_fields_expected:,} "
                          f"fields is under one per field at a floor of {floor}")
    return GateResult("yield", True, detail, "")


def reconstruction_gate(n_correct: int, n_total: int,
                      published: int = PUBLISHED_N,
                      tolerance: int = REPRO_TOLERANCE,
                      published_total: int = PUBLISHED_TOTAL) -> GateResult:
    """Does the baseline land on the published figure?

    The denominator is checked, not merely stored. Comparing raw counts alone
    -- which this did -- let a baseline scoring 135 of 1,000 patients (13.5%)
    pass a gate defined against a published 135 of 170 (79.4%), because both
    have the same numerator. `PUBLISHED_TOTAL` was declared and never read.
    """
    delta = int(n_correct) - int(published)
    if int(n_total) != int(published_total):
        return GateResult(
            "reconstruction", False,
            {"n_correct": int(n_correct), "n_total": int(n_total),
             "published": int(published), "published_total": int(published_total),
             "delta": delta, "tolerance": tolerance, "direction": "incomparable"},
            f"the comparator was scored on {n_total} patients against a published "
            f"figure over {published_total}; the two counts are not comparable")
    ok = abs(delta) <= tolerance
    where = "above" if delta > 0 else "below" if delta < 0 else "equal to"
    return GateResult(
        "reconstruction", ok,
        {"n_correct": int(n_correct), "n_total": int(n_total),
         "published": int(published), "published_total": int(published_total),
         "delta": delta, "tolerance": tolerance, "direction": where},
        "" if ok else (f"the comparator scores {n_correct}/{n_total}, {abs(delta)} "
                       f"{where} the published {published}"))


def screening_equivalent(pred_a: pd.Series, pred_b: pd.Series) -> bool:
    """P2. Two rules that make identical infected/uninfected calls."""
    from .metrics import _aligned

    a = (pred_a != UNINFECTED).to_numpy()
    b = (_aligned(pred_b, pred_a) != UNINFECTED).to_numpy()
    return bool(np.array_equal(a, b))


def headroom(truth: pd.Series, baseline: pd.Series) -> int:
    """P3. How many uninfected patients the baseline gets wrong.

    An upper bound on the wins available on the specificity endpoint. Zero
    headroom means no candidate can win, whatever it does.

    Routed through `metrics._aligned`, which refuses a missing prediction.
    A bare `reindex` puts NaN there, `NaN != "uninfected"` is True, and an
    unscored patient became a baseline false positive -- inventing headroom
    that does not exist and bypassing the very refusal this function feeds.
    """
    from .metrics import _aligned

    neg = truth == UNINFECTED
    called_positive = _aligned(baseline, truth)[neg] != UNINFECTED
    return int(called_positive.sum())


def design_gate(truth: pd.Series, baseline: pd.Series,
                candidate: pd.Series, *, checkpoint: str,
                rule: str) -> GateResult:
    from .metrics import _aligned

    base = _aligned(baseline, truth)
    cand = _aligned(candidate, truth)
    h = headroom(truth, base)
    equiv = screening_equivalent(base, cand)
    # Discordance among the UNINFECTED. The specificity McNemar this gate
    # protects conditions on exactly that subset; counting over all patients
    # reported a number for a different test.
    neg = truth == UNINFECTED
    disc_neg = int((((base[neg] != UNINFECTED)) !=
                    ((cand[neg] != UNINFECTED))).sum())
    disc_all = int(((base != UNINFECTED) != (cand != UNINFECTED)).sum())
    detail = {"checkpoint": checkpoint, "rule": rule, "headroom": h,
              "screening_equivalent": equiv, "discordant": disc_neg,
              "discordant_all_patients": disc_all,
              "discordant_denominator": "uninfected patients only"}
    if equiv:
        return GateResult("design", False, detail,
                          f"{checkpoint}:{rule} is screening-equivalent to the "
                          "baseline: the comparison cannot produce a non-null "
                          "result on this endpoint")
    if h == 0:
        return GateResult("design", False, detail,
                          f"{checkpoint}:{rule} has no headroom: the baseline "
                          "already calls every uninfected patient correctly, so "
                          "no win is available")
    return GateResult("design", True, detail, "")


def identifiability(patients: pd.DataFrame) -> GateResult:
    """P5. Can specificity be separated from centre at all?

    This one does not stop the analysis. It changes what may be claimed from
    it, so it is computed early and its verdict travels into the report.
    """
    neg = patients[patients["truth"] == UNINFECTED]
    centres = sorted(set(neg["centre"].dropna().astype(str)))
    ok = len(centres) > 1
    return GateResult(
        "identifiability", ok,
        {"n_uninfected": int(len(neg)), "centres": centres},
        "" if ok else (f"all {len(neg)} uninfected patients come from "
                       f"{centres[0] if centres else 'one centre'}, so "
                       "specificity is perfectly collinear with centre; no "
                       "estimator separates them and no sample size fixes it"))


def refuse(result: GateResult, remedy: str) -> None:
    if not result.passed:
        raise GateRefusal(f"[{result.name} gate] {result.message}",
                          remedy=remedy, **result.detail)


#: The pre-registered criterion. Kept as the primary despite the defect
#: documented below, because changing a primary after the data are in hand is
#: the practice this whole re-analysis exists to criticise.
#: Protocol v3 §8.3. The confirmatory checkpoint is DECLARED, not selected.
#:
#: v2 selected it by maximum baseline headroom, on the argument that headroom
#: reads the baseline only and so cannot see which way the discordant pairs
#: fell. An external reviewer and this package's own null simulation rejected
#: that argument from opposite directions, and the simulation settles it:
#:
#:     headroom = b + (patients both rules get wrong)
#:
#: where b is the numerator of the test statistic. Selecting the largest
#: headroom selects, in part, the largest b; simulated at three comparable arms
#: the size rises to about 6.5% against a nominal 5%.
#:
#: RT-DETR is declared because it is the architecture the source study reports
#: -- a fact independent of any outcome in this dataset.
DECLARED_CHECKPOINT = "rtdetr"

PRIMARY_SELECTION = "declared"

SELECTION_CRITERIA = ("declared", "max_headroom", "max_discordance")


def select_checkpoint(headrooms: dict, discordances: dict | None = None,
                      criterion: str = PRIMARY_SELECTION,
                      declared: str = DECLARED_CHECKPOINT) -> tuple:
    """The confirmatory checkpoint, by a stated rule rather than by name.

    Returns (chosen, criterion, detail).

    **max_headroom** picks the checkpoint on which the published rule leaves the
    most room to improve. It was pre-registered on the argument that it reads
    the baseline only, never a candidate, and therefore cannot see which way the
    discordant pairs fell. *That argument is wrong.* Headroom is the number of
    uninfected patients the baseline calls infected, and

        headroom = b + (patients both rules get wrong)

    where b is the count the test statistic is built from, so selecting the
    largest headroom selects in part the largest b. `simulate.selection_null`
    puts the inflation at roughly 6.5% against a nominal 5% when three arms have
    comparable headroom. This is recorded rather than quietly repaired: the
    criterion stays primary, the inflation is reported as a limitation, and
    `max_discordance` is carried beside it as a pre-specified sensitivity
    analysis.

    **max_discordance** picks the checkpoint with the most discordant pairs. The
    exact McNemar test conditions on n = b + c, and conditional on n the
    statistic is Binomial(n, 1/2) whatever else is true of the arm, so a rule
    that is a function of the n's alone leaves the conditional distribution
    untouched and preserves the level exactly. Simulation agrees.

    When the two criteria choose the same checkpoint the distinction is moot for
    this study, and `detail["agree"]` says so in one boolean.
    """
    from .errors import BadInput

    if criterion not in SELECTION_CRITERIA:
        raise BadInput(f"unknown selection criterion {criterion!r}",
                       remedy=f"Use one of: {', '.join(SELECTION_CRITERIA)}.")

    def pick(scores: dict) -> str:
        if not scores:
            return ""
        best = max(scores.values())
        return sorted(k for k, v in scores.items() if v == best)[0]

    by_headroom = pick(headrooms)
    by_discordance = pick(discordances or {})
    if criterion == "declared":
        if declared not in (headrooms or {}):
            raise BadInput(
                f"the declared confirmatory checkpoint {declared!r} is not "
                "among the arms that survived the yield gate",
                remedy=("The protocol declares this checkpoint in advance. If "
                        "it did not survive, say so and report no confirmatory "
                        "test -- do not substitute another arm, which is "
                        "selection by a different name."),
                available=sorted(headrooms or {}))
        return declared, criterion, {
            "declared": declared,
            "by_max_headroom": by_headroom,
            "by_max_discordance": by_discordance or None,
            "headrooms": {str(k): int(v) for k, v in (headrooms or {}).items()},
            "discordances": ({str(k): int(v) for k, v in discordances.items()}
                             if discordances else None),
            "agree_with_headroom": bool(declared == by_headroom),
            "agree_with_discordance": bool(by_discordance
                                           and declared == by_discordance),
            "primary_is_pre_registered": True,
            "caveat": ("the checkpoint is declared in the protocol, not "
                       "selected from these data. Headroom is reported as a "
                       "descriptive feasibility statement only; selecting on "
                       "it inflates the size of the test that follows (see "
                       "simulate.selection_null)."),
        }
    if criterion == "max_discordance" and not by_discordance:
        # `chosen or by_headroom` silently substituted the max-HEADROOM arm --
        # the criterion documented above as not level-preserving -- while
        # returning "max_discordance" as the criterion used.
        raise BadInput(
            "max_discordance was requested but no discordance counts were given",
            remedy=("Pass the per-checkpoint discordant-pair counts, or ask "
                    "for max_headroom explicitly. Falling back would report "
                    "one criterion and apply another."))
    chosen = by_headroom if criterion == "max_headroom" else by_discordance
    detail = {
        "by_max_headroom": by_headroom,
        "by_max_discordance": by_discordance or None,
        "headrooms": {str(k): int(v) for k, v in (headrooms or {}).items()},
        "discordances": ({str(k): int(v) for k, v in discordances.items()}
                         if discordances else None),
        "agree": bool(by_discordance and by_headroom == by_discordance),
        "primary_is_pre_registered": criterion == PRIMARY_SELECTION,
        "caveat": ("max_headroom is not level-preserving; see "
                   "simulate.selection_null. It remains primary because it was "
                   "pre-registered, and max_discordance is reported beside it."),
    }
    if not by_discordance:
        detail["agree"] = None
        detail["caveat"] += (" Discordance counts were not supplied, so the "
                             "sensitivity analysis could not be evaluated.")
    return chosen or by_headroom, criterion, detail
