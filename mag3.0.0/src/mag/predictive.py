"""Predictive values, at prevalences a reader chooses rather than inherits.

Protocol v3 §14. A short module that exists to stop one specific misreading.

Sensitivity and specificity are properties of the rule on the patients it saw.
Predictive values are not: they depend on prevalence, and this cohort is 68%
infected because a referral centre assembled it. A positive predictive value
computed at 68% and quoted without that number attached will be read as a
screening PPV, and it is roughly an order of magnitude too high for one.

So the module refuses to report a predictive value without the prevalence it
assumed, and reports several -- because the honest answer to "what is the PPV?"
in a paper like this is a small table and not a number.

Intervals come from the screening pair's own uncertainty, propagated by a
patient-level bootstrap rather than by a delta-method approximation: at 54
uninfected patients the specificity interval is wide and asymmetric, and a
symmetric approximation to it puts the PPV bound in the wrong place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Prevalences reported. Chosen to span the settings a reader might have in
#: mind and declared in the protocol, not fitted to anything: routine screening
#: in a non-endemic setting, a symptomatic-traveller clinic, and an endemic
#: outpatient department.
TARGET_PREVALENCES = (0.01, 0.05, 0.20)


@dataclass
class Predictive:
    at: dict = field(default_factory=dict)
    cohort_prevalence: float = float("nan")
    sensitivity: float = float("nan")
    specificity: float = float("nan")
    scope: str = ""
    note: str = ""
    sentence: str = ""


def _ppv(sens: float, spec: float, prev: float) -> float:
    num = sens * prev
    den = num + (1.0 - spec) * (1.0 - prev)
    return float(num / den) if den > 0 else float("nan")


def _npv(sens: float, spec: float, prev: float) -> float:
    num = spec * (1.0 - prev)
    den = num + (1.0 - sens) * prev
    return float(num / den) if den > 0 else float("nan")


def at_prevalences(correct_pos, correct_neg, *,
                   prevalences=TARGET_PREVALENCES, n_boot: int = 2000,
                   alpha: float = 0.05, seed: int = 0) -> Predictive:
    """PPV and NPV at each declared prevalence, with bootstrap intervals.

    `correct_pos` and `correct_neg` are the per-patient correctness vectors the
    screening endpoint was computed from -- passed rather than summarised,
    because a bootstrap needs the patients and not the proportions.
    """
    from .estimand import SPECIFICITY

    pos = np.asarray(list(correct_pos), dtype=bool)
    neg = np.asarray(list(correct_neg), dtype=bool)
    out = Predictive()
    if pos.size == 0 or neg.size == 0:
        out.note = ("predictive values need both infected and uninfected "
                    "patients; one arm is empty")
        return out

    sens, spec = float(pos.mean()), float(neg.mean())
    out.sensitivity, out.specificity = sens, spec
    out.cohort_prevalence = float(pos.size / (pos.size + neg.size))
    # The scope of the SPECIFICITY estimand travels with every predictive value,
    # because a PPV is a function of specificity and inherits its conditioning.
    out.scope = SPECIFICITY.scope

    rng = np.random.default_rng(seed)
    boot_sens = np.empty(n_boot)
    boot_spec = np.empty(n_boot)
    for b in range(n_boot):
        boot_sens[b] = pos[rng.integers(0, pos.size, pos.size)].mean()
        boot_spec[b] = neg[rng.integers(0, neg.size, neg.size)].mean()

    for prev in prevalences:
        p = float(prev)
        ppv_b = np.array([_ppv(s, k, p) for s, k in zip(boot_sens, boot_spec)])
        npv_b = np.array([_npv(s, k, p) for s, k in zip(boot_sens, boot_spec)])
        ppv_b = ppv_b[np.isfinite(ppv_b)]
        npv_b = npv_b[np.isfinite(npv_b)]
        out.at[f"{p:g}"] = {
            "prevalence": p,
            "ppv": _ppv(sens, spec, p),
            "ppv_ci": (float(np.quantile(ppv_b, alpha / 2)),
                       float(np.quantile(ppv_b, 1 - alpha / 2)))
            if ppv_b.size > 1 else (float("nan"), float("nan")),
            "npv": _npv(sens, spec, p),
            "npv_ci": (float(np.quantile(npv_b, alpha / 2)),
                       float(np.quantile(npv_b, 1 - alpha / 2)))
            if npv_b.size > 1 else (float("nan"), float("nan")),
            "assumed": (f"computed at an assumed prevalence of {100 * p:g}%, "
                        "not the prevalence of this cohort"),
        }

    # The cohort's own prevalence is reported too, and labelled, because a
    # reader will look for it and its absence invites the reconstruction that
    # this module exists to prevent.
    out.at["cohort"] = {
        "prevalence": out.cohort_prevalence,
        "ppv": _ppv(sens, spec, out.cohort_prevalence),
        "npv": _npv(sens, spec, out.cohort_prevalence),
        "ppv_ci": (float("nan"), float("nan")),
        "npv_ci": (float("nan"), float("nan")),
        "assumed": ("the prevalence of this referral-centre cohort "
                    f"({100 * out.cohort_prevalence:.1f}%). Not a screening "
                    "prevalence and not transportable to one."),
    }

    lo = min(float(p) for p in prevalences)
    out.sentence = (
        f"At an assumed prevalence of {100 * lo:g}%, the positive predictive "
        f"value was {100 * out.at[f'{lo:g}']['ppv']:.1f}% and the negative "
        f"predictive value {100 * out.at[f'{lo:g}']['npv']:.1f}%; at this "
        f"cohort's own {100 * out.cohort_prevalence:.1f}% prevalence the "
        f"positive predictive value was "
        f"{100 * out.at['cohort']['ppv']:.1f}%. Predictive values do not "
        "transport between settings, and the cohort figure is a referral-centre "
        "quantity. " + out.scope.capitalize() + ".")
    return out
