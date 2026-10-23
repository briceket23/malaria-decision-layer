"""Inference. Exact where it can be, and refusing where it cannot.

Three things here were wrong in the predecessor and are pinned by tests:

  * the two co-primary components were tested at different levels. Superiority
    used a two-sided exact McNemar and non-inferiority a one-sided bound, so
    5 wins against 0 losses failed the primary endpoint at p = 0.0625 while the
    one-sided p was 0.031. Both components of an intersection-union pair must
    be one-sided in the same direction, and `mcnemar_exact` takes the
    alternative explicitly rather than defaulting to a convention;
  * `arms_are_fixed` is DERIVED from the rule registry by the caller, never
    passed as a literal. Passing True by hand is how an invalid p-value for a
    fitted rule gets printed with no warning;
  * a bootstrap over fewer units than it has percentiles returns a refusal
    rather than an interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import comb, sqrt

import numpy as np

MIN_BOOTSTRAP_UNITS = 5
ALPHA = 0.05


@dataclass(frozen=True)
class McNemar:
    b: int                 # baseline wrong, candidate right
    c: int                 # baseline right, candidate wrong
    p: float
    n_discordant: int
    alternative: str
    valid: bool
    reason: str = ""


@dataclass(frozen=True)
class Interval:
    point: float
    lo: float
    hi: float
    method: str
    n_degenerate: int = 0


def _binom_tail_greater(b: int, n: int) -> float:
    """P(B >= b) for B ~ Binomial(n, 1/2). Exact, no normal approximation."""
    if n == 0:
        return 1.0
    return sum(comb(n, k) for k in range(b, n + 1)) / (2 ** n)


def mcnemar_exact(baseline, candidate, *, alternative: str = "two-sided",
                  arms_are_fixed: bool = True) -> McNemar:
    """Exact McNemar on paired binary correctness.

    `alternative='greater'` tests that the candidate corrects more than it
    breaks, which is the direction an intersection-union component needs.
    """
    from .errors import BadInput

    if alternative not in ("two-sided", "greater"):
        raise BadInput(f"unknown alternative {alternative!r}",
                       remedy="Use 'two-sided' or 'greater'.")
    base = np.asarray(list(baseline), dtype=bool)
    cand = np.asarray(list(candidate), dtype=bool)
    if base.shape != cand.shape:
        raise BadInput(
            f"the two arms cover different patients ({base.size} and {cand.size})",
            remedy="A paired test needs the same patients in the same order.")
    b = int((~base & cand).sum())
    c = int((base & ~cand).sum())
    n = b + c
    if alternative == "greater":
        p = _binom_tail_greater(b, n)
    else:
        p = min(1.0, 2.0 * _binom_tail_greater(max(b, c), n))
    reason = ""
    if not arms_are_fixed:
        reason = ("one arm's parameters were fitted on these patients, so the "
                  "discordant pairs are not exchangeable under the null and "
                  "this p-value is not valid")
    return McNemar(b=b, c=c, p=float(p), n_discordant=n,
                   alternative=alternative, valid=arms_are_fixed, reason=reason)


def paired_table(baseline, candidate) -> dict:
    """The full paired 2x2, which is the inferential object.

    Protocol v3 §10.2. The previous version reported only b and c, so a reader
    could not reconstruct the result: without a and d the discordant counts
    float free of the cohort they came from. All four cells, their total, and
    the sentence a Results section should lead with.
    """
    base = np.asarray(list(baseline), dtype=bool)
    cand = np.asarray(list(candidate), dtype=bool)
    if base.shape != cand.shape:
        from .errors import BadInput

        raise BadInput(
            f"the two arms cover different patients ({base.size} and {cand.size})",
            remedy="A paired table needs the same patients in the same order.")
    a = int((base & cand).sum())          # both correct
    b = int((~base & cand).sum())         # candidate corrects the comparator
    c = int((base & ~cand).sum())         # candidate breaks what worked
    d = int((~base & ~cand).sum())        # both wrong
    return {
        "both_correct": a, "candidate_corrects": b,
        "candidate_breaks": c, "both_wrong": d,
        "n": a + b + c + d, "n_discordant": b + c,
        "comparator_correct": a + c, "candidate_correct": a + b,
        "sentence": (
            f"the candidate corrected {b} of the comparator's errors and "
            f"introduced {c}, over {a + b + c + d} patients, giving "
            f"{b + c} discordant pairs"),
    }


def wilson(k: int, n: int, alpha: float = ALPHA) -> Interval:
    """Wilson score interval. Behaves at 0 and n, where Wald does not."""
    from .errors import BadInput

    if n <= 0:
        return Interval(float("nan"), float("nan"), float("nan"), "undefined")
    if k < 0 or k > n:
        raise BadInput(f"a count of {k} out of {n}",
                       remedy="A proportion needs 0 <= k <= n. Check which "
                              "denominator this count belongs to.")
    z = 1.959963984540054 if abs(alpha - 0.05) < 1e-9 else _z(alpha)
    phat = k / n
    denom = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / denom
    half = z * sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / denom
    return Interval(phat, max(0.0, centre - half), min(1.0, centre + half),
                    "wilson")


def _z(alpha: float) -> float:
    from math import erf
    lo, hi = 0.0, 10.0
    target = 1 - alpha / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def bootstrap(units, statistic, *, n_boot: int = 2000, alpha: float = ALPHA,
              seed: int = 0) -> Interval:
    """Percentile bootstrap over patients, counting degenerate resamples.

    Resamples in which the statistic is undefined are counted and reported
    rather than dropped in silence: with two three-patient classes, a recall
    can be undefined in a tenth of resamples, and an interval computed from the
    survivors is an interval for a different estimand.
    """
    u = list(units)
    if len(u) < MIN_BOOTSTRAP_UNITS:
        return Interval(float("nan"), float("nan"), float("nan"),
                        "too-few-units")
    rng = np.random.default_rng(seed)
    point = _finite(statistic(u))
    vals, degenerate = [], 0
    idx = np.arange(len(u))
    for _ in range(n_boot):
        pick = rng.integers(0, len(u), size=len(u))
        try:
            v = _finite(statistic([u[i] for i in idx[pick]]))
        except (TypeError, ValueError):
            # a statistic that raises on a degenerate resample is a degenerate
            # resample; one that raises on every resample is a broken
            # statistic, and the count below is what distinguishes them
            v = None
        if v is None:
            degenerate += 1
        else:
            vals.append(v)
    if len(vals) < 2:
        return Interval(point if point is not None else float("nan"),
                        float("nan"), float("nan"), "degenerate", degenerate)
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return Interval(point if point is not None else float("nan"),
                    float(lo), float(hi), "percentile", degenerate)


def _finite(x):
    """None unless x is a finite float. float32 NaN is not caught by `!= x`."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def tango_score(b: int, c: int, n: int, delta0: float) -> float:
    r"""Tango's score statistic for a paired difference of proportions.

    Tests H0: pi_b - pi_c = delta0 on the paired 2x2, where `b` is the count of
    patients the candidate gets right and the baseline gets wrong and `c` is the
    reverse. The constrained maximum likelihood estimate of pi_c under the null
    has a closed form, which is what makes this a score test rather than another
    resampling procedure:

        B    = b + c - delta0 * (2n - b + c)
        t~   = [B + sqrt(B^2 + 8 n c delta0 (1 - delta0))] / (4n)
        z    = (b - c - n delta0) / sqrt(n (2 t~ + delta0 (1 - delta0)))

    The alternative this package first tried was to condition on the number of
    discordant pairs and put an exact Clopper-Pearson bound on the win ratio.
    That is exact for delta = 0 and WRONG here: the estimand is
    (2 psi - 1) * pi_discordant, and substituting the observed m/n for
    pi_discordant treats an estimate as known. The interval came out too narrow
    exactly when m was small, and its type I error at the non-inferiority null
    was 0.50 against a nominal 0.05.
    """
    if n <= 0:
        return float("nan")
    d = float(delta0)
    B = b + c - d * (2 * n - b + c)
    disc = B * B + 8.0 * n * c * d * (1.0 - d)
    t = (B + sqrt(max(disc, 0.0))) / (4.0 * n)
    # the constrained estimate is a probability, and so is pi_c + delta0
    t = min(max(t, max(0.0, -d)), max((1.0 - abs(d)) / 2.0, 0.0))
    var = n * (2.0 * t + d * (1.0 - d))
    if var <= 0:
        return float("inf") if (b - c - n * d) > 0 else float("-inf")
    return float((b - c - n * d) / sqrt(var))


def tango_lower(b: int, c: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided lower confidence limit on pi_b - pi_c, by Tango's method.

    The limit is the delta0 at which the score statistic falls to the normal
    critical value. `tango_score` is decreasing in delta0, so a bisection on
    [-1, point estimate] finds it without an optimiser.
    """
    if n <= 0:
        return float("nan")
    z_crit = _z(2 * alpha)                       # one-sided: 1.645 at alpha=0.05
    point = (b - c) / n
    if tango_score(b, c, n, -1.0) < z_crit:
        return -1.0
    lo, hi = -1.0, point
    for _ in range(80):
        mid = (lo + hi) / 2
        if tango_score(b, c, n, mid) >= z_crit:
            lo = mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def tango_lower_many(b, c, n: int, alpha: float = ALPHA) -> np.ndarray:
    """`tango_lower` over many (b, c) pairs at once.

    Vectorised because the power calculation runs it thousands of times, and a
    power calculation that models a different test from the one the analysis
    runs is a power calculation for a different study. Same bisection, same
    critical value, one array at a time.
    """
    b = np.asarray(b, dtype=float)
    c = np.asarray(c, dtype=float)
    if n <= 0:
        return np.full(b.shape, np.nan)
    z_crit = _z(2 * alpha)

    def score(d):
        B = b + c - d * (2 * n - b + c)
        disc = np.maximum(B * B + 8.0 * n * c * d * (1.0 - d), 0.0)
        t = (B + np.sqrt(disc)) / (4.0 * n)
        t = np.clip(t, np.maximum(0.0, -d), np.maximum((1.0 - np.abs(d)) / 2.0, 0.0))
        var = n * (2.0 * t + d * (1.0 - d))
        num = b - c - n * d
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(var > 0, num / np.sqrt(np.maximum(var, 1e-300)),
                           np.where(num > 0, np.inf, -np.inf))
        return out

    lo = np.full(b.shape, -1.0)
    hi = (b - c) / n
    unbounded = score(lo) < z_crit          # the limit is at or below -1
    for _ in range(50):
        mid = (lo + hi) / 2
        take = score(mid) >= z_crit
        lo = np.where(take, mid, lo)
        hi = np.where(take, hi, mid)
    return np.where(unbounded, -1.0, (lo + hi) / 2)


def noninferiority(baseline, candidate, margin: float, *,
                   alpha: float = ALPHA, seed: int = 0,
                   method: str = "tango", with_bootstrap: bool = True) -> dict:
    r"""One-sided non-inferiority on a paired proportion difference.

    The endpoint is stated in full rather than as a verdict word: the point
    estimate, the one-sided lower bound, the margin, and the criterion the
    bound has to satisfy.

    **The default bound is Tango's score interval.** The percentile bootstrap
    that used to supply it is only first-order accurate, and on a paired binary
    endpoint at this study's parameters -- a baseline sensitivity near 0.97, so
    per-patient differences that are almost always 0 or -1 -- it does not hold
    its level. Simulated at the non-inferiority null with n = 116, its type I
    error was 0.063 to 0.076 against a nominal 0.05, and its one-sided coverage
    away from the boundary fell to 0.81. Since this component is half of an
    intersection-union co-primary, the whole endpoint inherited that level.

    Tango's method is the standard one for this endpoint and is what a
    statistical reviewer will expect to see. It inverts a score test whose
    nuisance parameter is profiled out in closed form, so it needs no
    resampling and no optimiser; `tango_score` carries the algebra.

    `method="bootstrap"` returns the old percentile bound. It is kept because
    the two are reported side by side: an endpoint whose verdict depends on
    which of them was used is an endpoint a reader is entitled to see twice.
    `with_bootstrap=False` skips it, for callers running this inside a
    simulation where 2,000 resamples per replicate is the whole cost.
    """
    from .errors import BadInput

    if method not in ("tango", "bootstrap"):
        raise BadInput(f"unknown non-inferiority method {method!r}",
                       remedy="Use 'tango' (score-based, the default) or "
                              "'bootstrap' (percentile, first-order).")
    base = np.asarray(list(baseline), dtype=bool)
    cand = np.asarray(list(candidate), dtype=bool)
    if base.shape != cand.shape:
        raise BadInput("non-inferiority needs the same patients in both arms",
                       remedy="Align the two prediction series first.")
    n = int(base.size)
    diff = float(cand.mean() - base.mean()) if n else float("nan")
    b = int((~base & cand).sum())
    c = int((base & ~cand).sum())
    m = b + c

    if with_bootstrap or method == "bootstrap":
        pairs = list(zip(base.tolist(), cand.tolist()))
        boot = bootstrap(pairs,
                         lambda u: float(np.mean([y for _, y in u]) -
                                         np.mean([x for x, _ in u])),
                         n_boot=2000, alpha=2 * alpha, seed=seed)
        boot_lo, boot_degenerate = boot.lo, boot.n_degenerate
    else:
        boot_lo, boot_degenerate = float("nan"), 0

    tango = tango_lower(b, c, n, alpha) if n else float("nan")
    lower = tango if method == "tango" else boot_lo
    out = {
        "diff": diff, "one_sided_lower": lower, "margin": -abs(margin),
        "non_inferior": bool(np.isfinite(lower) and lower > -abs(margin)),
        "criterion": ("one-sided lower confidence limit on the paired "
                      "difference, by Tango's score method, > -margin"),
        "method": method, "n": n, "b": b, "c": c, "n_discordant": m,
        "tango_lower": tango,
        "bootstrap_lower": boot_lo,
        "bootstrap_degenerate": boot_degenerate,
        "methods_agree": bool(
            np.isfinite(boot_lo) and
            (tango > -abs(margin)) == (boot_lo > -abs(margin))),
    }
    out["note"] = (
        "Tango's score limit governs; the percentile bootstrap is reported "
        "beside it because it does not hold its level on this endpoint"
        + ("" if out["methods_agree"] or not np.isfinite(boot_lo) else
           " -- and here the two methods disagree on the verdict, which is "
           "reported rather than resolved by choosing the more convenient"))
    return out


def intersection_union(components: dict) -> dict:
    """An IU test rejects only if every component rejects.

    Kept as its own function so that the conjunction is computed once and read
    everywhere. A paper that decides this in prose can contradict its own
    tables, and one did.
    """
    from .errors import BadInput

    if not components:
        # `all([])` is True, so an IU test with no components rejected.
        raise BadInput("an intersection-union test with no components",
                       remedy="Name the components the endpoint conjoins. "
                              "An empty conjunction is vacuously true and "
                              "would reject unconditionally.")
    bad = {k: v for k, v in components.items() if not isinstance(v, (bool, np.bool_))}
    if bad:
        # a caller passing p-values instead of verdicts inverts the sense
        # exactly: p = 0.9 would "reject" and p = 0.0 would not
        raise BadInput(
            f"non-boolean IU components: {', '.join(sorted(bad))}",
            remedy=("Each component must be the verdict of its own test, not "
                    "its p-value. A p-value evaluated by truthiness inverts "
                    "the conjunction."),
            got={k: type(v).__name__ for k, v in bad.items()})
    ok = all(bool(v) for v in components.values())
    failed = sorted(k for k, v in components.items() if not v)
    return {"components": {k: bool(v) for k, v in components.items()},
            "reject": ok, "failed": failed}
