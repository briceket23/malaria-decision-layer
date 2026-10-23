"""What this design could have detected, which is not the same as what it did.

Reviewers asked for a power calculation. The one thing this module will not
produce is *observed* power -- the power to detect the effect that was in fact
observed. That quantity is a strictly decreasing function of the p-value, so it
adds no information to the test result and licenses the wrong conclusion in both
directions: a large p-value always yields low observed power, and reporting it
as "the study was underpowered" recycles the same non-finding as an explanation
for itself.

What is informative, and what is computed here, is the *detectable effect*: with
the sample size and the discordance rate this study actually realised, how large
would the true effect have had to be for the pre-specified procedure to find it
with 80% probability? That question is answerable before the data and its answer
does not change after them.

Both co-primary components are treated separately, because they have different
sample sizes and different tests:

  * specificity, by exact McNemar over the discordant pairs among the 54
    uninfected patients. The exact test is used here as it is used in the
    analysis, not a normal approximation to it, because at these counts the
    difference decides the answer: with four discordant pairs no split
    whatsoever reaches a one-sided 0.05, and a normal approximation cheerfully
    reports otherwise;
  * sensitivity, by the paired non-inferiority procedure over the infected
    patients, simulated.

The specificity arithmetic is exact and closed. The sensitivity arithmetic is
simulated, because the analysis's lower bound comes from a bootstrap and a power
calculation that models a different test is a power calculation for a different
study. `method_agreement` reports both limits side by side.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

ALPHA = 0.05
TARGET_POWER = 0.80


# --------------------------------------------------------------------------- #
# Exact McNemar: the critical value, then the power
# --------------------------------------------------------------------------- #

def critical_b(n_discordant: int, alpha: float = ALPHA) -> int | None:
    """Smallest b for which a one-sided exact McNemar rejects, or None.

    None is a real answer and the reason this function exists. Below FIVE
    discordant pairs there is no b at all that reaches a one-sided 0.05: even a
    clean sweep, every discordant pair favouring the candidate, gives
    P = 2^-n > 0.05 for n <= 4. At n = 5 a clean sweep gives exactly 0.03125,
    which does reject. So a design whose expected discordance is four pairs
    cannot produce a significant result by any behaviour of the candidate, and
    that is a fact about the design rather than about the detector.
    """
    n = int(n_discordant)
    if n <= 0:
        return None
    return _critical_b(n, float(alpha))


@lru_cache(maxsize=8192)
def _critical_b(n: int, alpha: float) -> int | None:
    """The cached, vectorised body of `critical_b`.

    The obvious form recomputes an O(n) sum of exact integer binomials for
    every candidate b, which is O(n^2) per call in Python. `required_n_mcnemar`
    walks n up to a few thousand and calls this for every discordance count at
    every candidate n, and a single power calculation on a low-discordance
    cohort took sixteen minutes. The whole upper tail is one cumulative sum
    here, and the result is cached because it depends on nothing but (n, alpha).
    """
    k = np.arange(n + 1)
    lf = _log_factorial(n)
    log_pmf = lf[n] - lf[k] - lf[n - k] - n * np.log(2.0)
    sf = np.cumsum(np.exp(log_pmf)[::-1])[::-1]       # sf[b] = P(B >= b)
    hit = np.flatnonzero(sf <= alpha)
    return int(hit[0]) if hit.size else None


@lru_cache(maxsize=8)
def _log_factorial(n: int) -> np.ndarray:
    """log(k!) for k = 0..n, as one cumulative sum.

    A table rather than `scipy.special.gammaln`, so that the inference in this
    package depends on numpy alone. For integer arguments it is exact to
    floating point and it is computed once per distinct n; `gammaln` would be a
    whole extra dependency for a function two lines of numpy can build.
    """
    out = np.zeros(int(n) + 1, dtype=float)
    if n > 0:
        out[1:] = np.cumsum(np.log(np.arange(1, int(n) + 1, dtype=float)))
    return out


def power_mcnemar(n_patients: int, pi_discordant: float, psi: float, *,
                  alpha: float = ALPHA) -> float:
    """Exact power of the one-sided exact McNemar test.

    Both stages are modelled, which matters at this size. The number of
    discordant pairs is itself random -- Binomial(n_patients, pi_discordant) --
    and conditional on it the number favouring the candidate is
    Binomial(n_disc, psi). Treating the discordance count as fixed at its
    expectation overstates power here, because the mass below the critical
    discordance contributes exactly zero and a point calculation misses it.
    """
    n = int(n_patients)
    if n <= 0 or not (0.0 <= pi_discordant <= 1.0) or not (0.0 <= psi <= 1.0):
        return float("nan")
    pi = float(pi_discordant)
    m_all = np.arange(n + 1)
    pm = _binom_pmf_array(n, m_all, pi)

    # Only the discordance counts that carry mass are visited. At n = 2,048 and
    # pi = 0.02 the distribution lives on m in [0, 105] and the other 1,943
    # values contribute less than 1e-12 between them, so walking all of them is
    # twenty times the work for an answer that agrees to the twelfth decimal.
    keep = np.flatnonzero(pm > MASS_FLOOR)
    if keep.size == 0:
        keep = np.array([int(round(n * pi))])
    total = 0.0
    for m in keep:
        m = int(m)
        crit = critical_b(m, alpha)
        if crit is None:
            continue                      # this discordance cannot reject at all
        b = np.arange(crit, m + 1)
        total += float(pm[m]) * float(_binom_pmf_array(m, b, psi).sum())
    return float(total)


#: Discordance counts carrying less probability than this are skipped. Chosen
#: far below any digit ever reported, and the sum of everything skipped is
#: bounded by (n + 1) times this.
MASS_FLOOR = 1e-15


def _binom_pmf_array(n: int, k, p: float) -> np.ndarray:
    """Binomial pmf over an array of k, in log space.

    Log space is not a nicety. `comb(2048, 1024)` is an exact Python int of
    about 10**614, and multiplying it by a float raises OverflowError -- so any
    cohort with a discordance rate below about 2.5% crashed the entire run,
    after every gate, the factorial and the exact test had already been
    computed, with a traceback and no remedy, from a *descriptive* calculation.
    """
    k = np.asarray(k)
    if p <= 0.0:
        return np.where(k == 0, 1.0, 0.0).astype(float)
    if p >= 1.0:
        return np.where(k == n, 1.0, 0.0).astype(float)
    lf = _log_factorial(n)
    log_c = lf[n] - lf[k] - lf[n - k]
    return np.exp(log_c + k * np.log(p) + (n - k) * np.log1p(-p))


def detectable_psi(n_patients: int, pi_discordant: float, *,
                   alpha: float = ALPHA, power: float = TARGET_POWER) -> dict:
    """The win ratio among discordant pairs this design could detect.

    psi is the probability that a discordant pair favours the candidate. It is
    reported alongside the odds form psi/(1-psi), which is the sentence a reader
    wants: "the candidate would have had to correct k uninfected patients for
    every one it broke".
    """
    lo, hi = 0.5, 1.0
    if power_mcnemar(n_patients, pi_discordant, hi, alpha=alpha) < power:
        return {"psi": float("nan"), "odds": float("nan"),
                "achievable": False, "max_power": float(
                    power_mcnemar(n_patients, pi_discordant, 1.0, alpha=alpha)),
                "note": ("no value of psi reaches the target power at this "
                         "sample size and discordance rate: even a candidate "
                         "that wins every discordant pair would not reject "
                         "with the required probability")}
    for _ in range(60):
        mid = (lo + hi) / 2
        if power_mcnemar(n_patients, pi_discordant, mid, alpha=alpha) < power:
            lo = mid
        else:
            hi = mid
    psi = (lo + hi) / 2
    return {"psi": float(psi),
            "odds": float(psi / max(1 - psi, 1e-12)),
            "achievable": True,
            "max_power": float(power_mcnemar(n_patients, pi_discordant, 1.0,
                                             alpha=alpha)),
            "note": (f"with {n_patients} patients and a discordance rate of "
                     f"{pi_discordant:.3f}, the candidate would need to win "
                     f"about {psi / max(1 - psi, 1e-12):.1f} discordant pairs "
                     "for every one it loses")}


def required_n_mcnemar(pi_discordant: float, psi: float, *, alpha: float = ALPHA,
                       power: float = TARGET_POWER, cap: int = 5000) -> dict:
    """How many patients a future study would need at this effect."""
    if not (0 < pi_discordant <= 1) or psi <= 0.5:
        return {"n": None, "note": "no effect to power against (psi <= 0.5)"}
    lo, hi = 1, 16
    while hi <= cap and power_mcnemar(hi, pi_discordant, psi, alpha=alpha) < power:
        lo, hi = hi, hi * 2
    if hi > cap:
        return {"n": None, "cap": cap,
                "note": f"more than {cap:,} patients; not a feasible design"}
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if power_mcnemar(mid, pi_discordant, psi, alpha=alpha) < power:
            lo = mid
        else:
            hi = mid
    # `power_mcnemar` is NOT monotone in n: `critical_b` jumps (critical_b(6)
    # is 6 but critical_b(8) is 7), so the bisection above can settle above the
    # true minimum -- by up to 13% at some settings. A downward walk that stops
    # at the first failure is not enough either, because the dips are local: it
    # must look past them. A bounded window is scanned instead, which is cheap
    # and covers gaps far wider than any observed.
    best = int(hi)
    window = max(1, best - 64)
    for candidate in range(window, best):
        if power_mcnemar(candidate, pi_discordant, psi, alpha=alpha) >= power:
            best = candidate
            break
    return {"n": best, "power_at_n": float(
        power_mcnemar(best, pi_discordant, psi, alpha=alpha)),
        "note": ("the smallest n meeting the target; the power curve is not "
                 "monotone in n because the exact test's critical value jumps")}


# --------------------------------------------------------------------------- #
# Non-inferiority, simulated against the procedure the analysis runs
# --------------------------------------------------------------------------- #

def method_agreement(base, cand, *, margin: float, alpha: float = ALPHA,
                     seed: int = 0) -> dict:
    """Do Tango's limit and the percentile bootstrap agree on this data?

    The analysis reports Tango's limit because the bootstrap does not hold its
    level on this endpoint. Both are computed and compared, because an endpoint
    whose verdict depends on which limit was used is one a reader is entitled
    to see twice rather than once.
    """
    from .stats import noninferiority

    b = np.asarray(list(base), dtype=float) > 0.5
    c = np.asarray(list(cand), dtype=float) > 0.5
    out = noninferiority(b, c, margin, alpha=alpha, seed=seed)
    gap = abs(out["tango_lower"] - out["bootstrap_lower"])
    return {"tango_lower": out["tango_lower"],
            "bootstrap_lower": out["bootstrap_lower"],
            "abs_difference": float(gap),
            "as_share_of_margin": float(gap / abs(margin)) if margin else float("nan"),
            "same_verdict": bool(out["methods_agree"]),
            "governs": "tango",
            "note": ("the percentile bootstrap's type I error at this "
                     "endpoint's null was 0.063 to 0.076 against a nominal "
                     "0.05, so it is reported and not used")}


def feasible_discordance(p_base: float, delta: float) -> tuple:
    """The discordance rates compatible with a baseline accuracy and a delta.

    The four cells of a paired 2x2 are not free. Given the baseline's accuracy
    and the true difference, a discordance rate that would require a negative
    "both wrong" or "both right" cell describes no possible cohort. This bites
    exactly where this study sits: at a baseline sensitivity near 0.97 the
    largest possible disagreement at zero true difference is about 0.07, so a
    power calculation casually assuming 10% disagreement is computing with a
    cohort that cannot exist.
    """
    lo = abs(float(delta))
    hi = min(2.0 * p_base + delta, 2.0 * (1.0 - p_base) - delta)
    return float(lo), float(max(hi, lo))


def power_noninferiority(n: int, p_base: float, delta: float, *,
                         margin: float = 0.05, discordance: float = 0.10,
                         alpha: float = ALPHA, reps: int = 4000,
                         seed: int = 0) -> float:
    """Probability the NI component concludes non-inferiority.

    The pair structure is generated explicitly rather than as two independent
    arms: `discordance` is the share of patients on whom the two rules disagree,
    and `delta` the true difference in the candidate's favour. Independent arms
    would overstate the variance of the difference by a factor that depends on
    the correlation, and the whole point of a paired design is that the
    correlation is high.

    The limit computed per replicate is Tango's -- the same one the analysis
    reports. An earlier version substituted a normal approximation because a
    2,000-draw bootstrap per replicate was unaffordable; Tango's limit is
    closed-form up to a bisection, so the substitution is no longer needed and
    this simulates the procedure that actually runs.
    """
    from .stats import tango_lower_many

    n = int(n)
    if n <= 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    # solve the 2x2: p_bc (base right, cand wrong) and p_cb (base wrong, cand
    # right) satisfy p_cb - p_bc = delta and p_cb + p_bc = discordance
    p_cb = (discordance + delta) / 2.0
    p_bc = (discordance - delta) / 2.0
    if p_cb < 0 or p_bc < 0:
        return float("nan")
    p_both = p_base - p_bc                        # base right and cand right
    p_neither = 1.0 - p_both - p_bc - p_cb
    if min(p_both, p_neither) < 0:
        return float("nan")
    probs = np.array([p_both, p_bc, p_cb, p_neither], dtype=float)
    probs = probs / probs.sum()
    # cell 1 is base right / cand wrong (a loss, counted in c); cell 2 is the
    # reverse (a gain, counted in b). Drawing the counts straight from a
    # multinomial is equivalent to drawing n patients and tallying them, and
    # avoids materialising a reps x n array.
    counts = rng.multinomial(n, probs, size=reps)
    c_counts = counts[:, 1]
    b_counts = counts[:, 2]
    lower = tango_lower_many(b_counts, c_counts, n, alpha)
    return float((lower > -abs(margin)).mean())


def detectable_delta(n: int, p_base: float, *, margin: float = 0.05,
                     discordance: float = 0.10, alpha: float = ALPHA,
                     power: float = TARGET_POWER, reps: int = 4000,
                     seed: int = 0) -> dict:
    """The largest true deficit at which non-inferiority is still concluded.

    Read it as the design's resolution: a candidate truly worse than this by
    more than the quoted amount would usually fail the component, and one worse
    by less would usually pass it. If the answer is close to zero the margin is
    doing no work -- only an exactly-as-good candidate would pass -- and if it
    is close to the margin the test is nearly powerless to distinguish the two.
    """
    clipped = [False]

    def pw_at(delta: float) -> float:
        d_lo, d_hi = feasible_discordance(p_base, delta)
        d = min(max(discordance, d_lo), d_hi)
        if abs(d - discordance) > 1e-12:
            clipped[0] = True
        return power_noninferiority(n, p_base, delta, margin=margin,
                                    discordance=d, alpha=alpha, reps=reps,
                                    seed=seed)

    lo, hi = -abs(margin), 0.0
    top = pw_at(hi)
    if not np.isfinite(top) or top < power:
        return {"delta": float("nan"), "achievable": False,
                "power_at_zero": float(top),
                "discordance_was_clipped": bool(clipped[0]),
                "note": ("even a candidate exactly as good as the baseline "
                         "would not demonstrate non-inferiority with the "
                         "target probability at this sample size")}
    for _ in range(24):
        mid = (lo + hi) / 2
        p_mid = pw_at(mid)
        if not np.isfinite(p_mid) or p_mid < power:
            lo = mid
        else:
            hi = mid
    return {"delta": float((lo + hi) / 2), "achievable": True,
            "margin": -abs(margin), "power_at_zero": float(top),
            "discordance_was_clipped": bool(clipped[0]),
            "note": (f"a true deficit larger than {abs((lo + hi) / 2) * 100:.1f} "
                     "percentage points would usually fail this component"
                     + (" (the assumed disagreement rate was reduced to the "
                        "largest one compatible with the baseline accuracy)"
                        if clipped[0] else ""))}


# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class PowerReport:
    specificity: dict
    sensitivity: dict
    agreement: dict
    note: str


def analyse(confirmatory: dict, base_pos, cand_pos, n_uninfected: int, *,
            alpha: float = ALPHA, power: float = TARGET_POWER, seed: int = 0,
            reps: int = 4000) -> PowerReport:
    """The full calculation, from a `Study.confirmatory` block and its pairs.

    `base_pos` and `cand_pos` are the two arms' per-patient correctness on the
    infected patients -- the same vectors the non-inferiority component was
    computed from. They are passed rather than recovered from summary
    statistics, because the paired discordance cannot be recovered from two
    marginal proportions and assuming a value for it is the single easiest way
    to produce a confidently wrong power calculation.

    Everything is driven by what the study realised -- its patient counts, its
    discordance rate -- so the answer describes this design and not a textbook
    one.
    """
    mc = confirmatory["mcnemar"]
    ni = confirmatory["noninferiority"]
    n_neg = int(n_uninfected)
    n_pos = int(ni.get("n") or 0)
    n_disc = int(mc["n_discordant"])
    pi_disc = n_disc / n_neg if n_neg else float("nan")
    b = np.asarray(list(base_pos), dtype=float)
    c = np.asarray(list(cand_pos), dtype=float)

    spec = {
        "n_uninfected": n_neg, "n_discordant": n_disc,
        "discordance_rate": pi_disc,
        "critical_b_at_realised_discordance": critical_b(n_disc, alpha),
        "min_discordant_pairs_that_can_reject": next(
            (m for m in range(1, 200) if critical_b(m, alpha) is not None), None),
        "power_if_candidate_wins_every_pair": power_mcnemar(
            n_neg, pi_disc, 1.0, alpha=alpha) if n_neg else float("nan"),
    }
    if n_neg and np.isfinite(pi_disc):
        spec["detectable"] = detectable_psi(n_neg, pi_disc, alpha=alpha,
                                            power=power)
        spec["required_n_for_3_to_1"] = required_n_mcnemar(
            pi_disc, 0.75, alpha=alpha, power=power)
    crit = spec["critical_b_at_realised_discordance"]
    spec["sentence"] = (
        f"with {n_disc} discordant pairs among {n_neg} uninfected patients, a "
        + (f"one-sided exact test rejects only if at least {crit} of them "
           "favour the candidate" if crit is not None else
           "one-sided exact test cannot reject at any split, so this component "
           "could not have produced a positive result whatever the candidate "
           "did"))

    sens = {"n_infected": n_pos, "margin": ni["margin"],
            "observed_diff": ni["diff"]}
    if n_pos > 1 and b.size == c.size and b.size > 1:
        p_base = float(b.mean())
        disc = float((b != c).mean())
        if disc <= 0:
            # the two rules agreed on every infected patient: the paired
            # difference is identically zero and no simulation is meaningful
            sens["discordance_rate"] = 0.0
            sens["sentence"] = (
                "the two rules made identical calls on every infected patient, "
                "so the sensitivity difference is exactly zero by construction "
                "and its detectable effect is not defined")
            return PowerReport(specificity=spec, sensitivity=sens,
                               agreement=method_agreement(
                                   b, c, margin=abs(ni["margin"]), alpha=alpha,
                                   seed=seed),
                               note=_NOTE)
        sens["baseline_sensitivity"] = p_base
        sens["discordance_rate"] = disc
        d_lo, d_hi = feasible_discordance(p_base, 0.0)
        sens["feasible_discordance_at_zero"] = (d_lo, d_hi)
        sens["power_at_zero_true_difference"] = power_noninferiority(
            n_pos, p_base, 0.0, margin=abs(ni["margin"]),
            discordance=min(max(disc, d_lo), d_hi),
            alpha=alpha, reps=reps, seed=seed)
        sens["detectable"] = detectable_delta(
            n_pos, p_base, margin=abs(ni["margin"]), discordance=disc,
            alpha=alpha, power=power, reps=reps, seed=seed)
        sens["sentence"] = (
            f"with {n_pos} infected patients and a {abs(ni['margin']) * 100:.0f} "
            "percentage point margin, the component concludes non-inferiority "
            f"with probability {sens['power_at_zero_true_difference']:.2f} "
            "when the two rules are in truth equally sensitive")

    agree = (method_agreement(b, c, margin=abs(ni["margin"]), alpha=alpha,
                                 seed=seed)
             if b.size == c.size and b.size > 1 else {})
    return PowerReport(specificity=spec, sensitivity=sens, agreement=agree,
                       note=_NOTE)


_NOTE = ("detectable effects, not observed power. Observed power is a monotone "
         "function of the p-value and would restate the test result as if it "
         "were an explanation for it.")
