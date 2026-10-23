"""Calibration, and the rules that are fitted rather than fixed.

Three maps, all cross-fitted, none of them depending on scikit-learn: isotonic
regression by pool-adjacent-violators, Platt scaling by Newton's method on a
one-feature logistic, and temperature scaling by bisection on the derivative of
the negative log likelihood. Fifty lines of numpy each, no dependency, and the
arithmetic is visible rather than delegated.

Two things here are not conveniences and must not be simplified away.

**Cross-fitting.** A map fitted on the patients it then scores reports its own
training accuracy. Folds are assigned deterministically from a seed, and a
patient's score always comes from a map fitted without that patient. Leave-one-
out is available and is *not* the default: with 170 patients the 169-patient
training sets overlap in 168 patients, so every patient is scored by
approximately the same map and the selection effect survives almost intact.
K-fold at K=10 makes the overlap visible in the fold count rather than hiding
it, and `simulate.crossfit_null` measures what either choice costs.

**Fitted rules are not exchangeable with the baseline.** Their parameters come
from the patients being tested, so an exact McNemar test against them is not
valid. `rules.FIXED_RULES` excludes them and `study` derives the
`arms_are_fixed` flag from that set rather than accepting a literal, because
passing `True` by hand is exactly how an invalid p-value gets printed with no
warning.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

CALIBRATORS = ("isotonic", "platt", "temperature")
DEFAULT_FOLDS = 10


# --------------------------------------------------------------------------- #
# The three maps
# --------------------------------------------------------------------------- #

def pav(x: np.ndarray, y: np.ndarray):
    """Isotonic regression by pool-adjacent-violators.

    Returns (knots_x, knots_y) defining a non-decreasing step function. Ties in
    x are pooled, so the map is well defined when many patients share a score.
    """
    order = np.argsort(x, kind="mergesort")
    xs, ys = np.asarray(x, float)[order], np.asarray(y, float)[order]
    # pool exact ties in x first: an isotonic fit cannot separate them
    ux, inv = np.unique(xs, return_inverse=True)
    sums = np.bincount(inv, weights=ys)
    counts = np.bincount(inv).astype(float)
    values = sums / counts

    # PAVA over the unique x, as a stack. Each block carries three things: its
    # pooled value, the observation weight behind it, and how many knots of the
    # unique grid it spans. The third is what the expansion below needs, and an
    # earlier version expanded by the observation weight instead -- which is
    # the same number only when every x is distinct, so the map came out
    # scrambled precisely on the tied scores this study is full of.
    #
    # The stack form is the textbook linear-time one. The equivalent written
    # with an index and `blocks[i:i+2] = [...]` re-copies the list on every
    # merge, which makes it quadratic; that mattered because this runs tens of
    # thousands of times inside the null simulations.
    blocks: list = []
    for v, c in zip(values, counts):
        blk = [float(v), float(c), 1]
        while blocks and blocks[-1][0] > blk[0] + 1e-15:
            v0, w0, k0 = blocks.pop()
            w = w0 + blk[1]
            blk = [(v0 * w0 + blk[0] * blk[1]) / w, w, k0 + blk[2]]
        blocks.append(blk)
    out_y, at = np.empty(len(ux)), 0
    for value, _, span in blocks:
        out_y[at:at + span] = value
        at += span
    assert at == len(ux), "the isotonic blocks did not cover the knot grid"
    return ux, np.clip(out_y, 0.0, 1.0)


def apply_pav(knots_x: np.ndarray, knots_y: np.ndarray, x) -> np.ndarray:
    """Evaluate a fitted isotonic map, flat outside its support."""
    x = np.atleast_1d(np.asarray(x, float))
    idx = np.searchsorted(knots_x, x, side="right") - 1
    idx = np.clip(idx, 0, len(knots_y) - 1)
    return knots_y[idx]


def _nll(a: float, b: float, s: np.ndarray, y: np.ndarray) -> float:
    z = np.clip(a * s + b, -60, 60)
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def platt(scores: np.ndarray, labels: np.ndarray, iters: int = 100):
    """Logistic a*s + b, by damped Newton. Returns (a, b).

    Damping is not decoration. Undamped Newton on separable or near-separable
    data walks the coefficients to infinity, and the first version of this
    function returned a slope of -3.1e9 on a perfectly ordinary overconfident
    score. Each step is halved until the negative log likelihood actually
    decreases, and the fit stops rather than diverging when it cannot.
    """
    s = np.asarray(scores, float)
    y = np.asarray(labels, float)
    if len(s) == 0 or len(np.unique(y)) < 2:
        return 1.0, 0.0
    # scale to unit variance so the Hessian is well conditioned, then undo it
    mu, sd = float(s.mean()), float(s.std()) or 1.0
    z = (s - mu) / sd
    a, b = 0.0, float(np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6)))
    cur = _nll(a, b, z, y)
    for _ in range(iters):
        eta = np.clip(a * z + b, -60, 60)
        p = 1.0 / (1.0 + np.exp(-eta))
        g = np.array([np.sum((p - y) * z), np.sum(p - y)])
        if np.max(np.abs(g)) < 1e-8:
            break                        # already at the optimum; Newton on a
            #                              zero gradient is thirty wasted
            #                              backtracking steps, and this solver
            #                              runs tens of thousands of times
        w = p * (1 - p) + 1e-9
        h = np.array([[np.sum(w * z * z), np.sum(w * z)],
                      [np.sum(w * z), np.sum(w)]])
        try:
            step = np.linalg.solve(h + 1e-6 * np.eye(2), g)
        except np.linalg.LinAlgError:
            break
        t = 1.0
        for _ in range(30):                      # backtrack until it improves
            na, nb = a - t * step[0], b - t * step[1]
            nxt = _nll(na, nb, z, y)
            if nxt <= cur + 1e-12:
                break
            t /= 2.0
        else:
            break
        if abs(cur - nxt) < 1e-12 and np.max(np.abs(t * step)) < 1e-10:
            a, b, cur = na, nb, nxt
            break
        a, b, cur = na, nb, nxt
    return float(a / sd), float(b - a * mu / sd)


def apply_platt(a: float, b: float, x) -> np.ndarray:
    z = a * np.atleast_1d(np.asarray(x, float)) + b
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def temperature(scores: np.ndarray, labels: np.ndarray) -> float:
    """One scalar T > 0 dividing the logit. Bisection on the NLL derivative.

    Bisection rather than a gradient step because the objective is convex in
    1/T and a fixed-step optimiser silently fails to converge on a flat region,
    which is the kind of failure that shows up as a strange calibration slope
    three modules later.
    """
    s = np.clip(np.asarray(scores, float), 1e-6, 1 - 1e-6)
    y = np.asarray(labels, float)
    logit = np.log(s / (1 - s))
    neg_logit = -logit                              # hoisted: it is a constant

    def dnll(t: float) -> float:
        p = 1.0 / (1.0 + np.exp(-np.clip(logit / t, -60, 60)))
        return float(np.sum((p - y) * neg_logit)) / (t * t)

    lo, hi = 1e-3, 1e3
    f_lo, f_hi = dnll(lo), dnll(hi)
    if f_lo * f_hi > 0:
        # The derivative has the same sign at both ends: the optimum is at or
        # beyond an endpoint, NOT at T = 1. Returning 1.0 here handed back the
        # identity map -- so an anti-correlated fold kept three uninfected
        # patients at p = 0.9, 0.8, 0.7 when the optimum wanted everything
        # shrunk to 0.5. On an uninformative score with 10-fold cross-fitting,
        # 51% of folds took this branch, which is exactly the regime
        # `simulate.crossfit_null` exists to measure.
        return float(lo if _nll_t(lo, logit, y) < _nll_t(hi, logit, y) else hi)
    # 60 geometric halvings take the bracket [1e-3, 1e3] down to a relative
    # width of about 1e-11, which is far below anything that could change a
    # reported figure. The previous 200 were not more accurate, only slower:
    # they were 140 iterations below the resolution of a float64 ratio.
    #
    # The endpoint value is carried rather than recomputed. Evaluating dnll(lo)
    # afresh every iteration doubled the cost of the whole calibration module
    # for no gain, and this function was 43% of a full run.
    for _ in range(60):
        mid = np.sqrt(lo * hi)
        f_mid = dnll(mid)
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return float(np.sqrt(lo * hi))


def _nll_t(t: float, logit: np.ndarray, y: np.ndarray) -> float:
    """Negative log likelihood of a temperature-scaled score."""
    z = np.clip(logit / max(t, 1e-12), -60, 60)
    return float(np.sum(np.logaddexp(0.0, z) - y * z))


def apply_temperature(t: float, x) -> np.ndarray:
    s = np.clip(np.atleast_1d(np.asarray(x, float)), 1e-6, 1 - 1e-6)
    z = np.log(s / (1 - s)) / max(t, 1e-6)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# --------------------------------------------------------------------------- #
# Cross-fitting
# --------------------------------------------------------------------------- #

def folds(n: int, k: int = DEFAULT_FOLDS, seed: int = 0) -> np.ndarray:
    """Deterministic fold assignment. k >= n means leave-one-out."""
    from .errors import BadInput

    if k < 2:
        # k = 0 divided by zero and returned a single fold; k = 1 gives a
        # training set that is the whole cohort. Either way every patient is
        # scored by a map that saw it, which is the one thing this module
        # exists to prevent -- and neither raised.
        raise BadInput(f"{k} cross-fitting folds",
                       remedy=("Use at least 2. A single fold trains on every "
                               "patient it then scores, so nothing is "
                               "cross-fitted and the reported accuracy is a "
                               "training accuracy."))
    if k >= n:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return rng.permutation(np.arange(n) % k)


def sensitivity_threshold(p: np.ndarray, y: np.ndarray, target: float) -> float:
    """The largest threshold at which sensitivity is still at least `target`.

    Largest rather than smallest: among the thresholds that meet the sensitivity
    requirement this is the one that refers fewest uninfected patients, so a
    staged rule built on it is the most specific rule that still satisfies its
    own stated screening constraint. Returns 0.0 when the target cannot be met,
    which calls everyone infected and is the honest behaviour for a screen.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    pos = p[y > 0.5]
    n = pos.size
    if n == 0:
        return 0.0
    # Sensitivity at threshold t is mean(pos >= t): a step function whose steps
    # sit at the positives' own scores, so only those need be considered. The
    # whole curve is one searchsorted rather than one pass per candidate --
    # written as a loop this was 21% of a full run, because it is called once
    # per fold per calibrator per replicate of the null simulations.
    ordered = np.sort(pos)
    candidates = np.unique(ordered)
    at_or_above = n - np.searchsorted(ordered, candidates, side="left")
    ok = np.flatnonzero(at_or_above / n >= target)
    return float(candidates[ok[-1]]) if ok.size else 0.0


@dataclass(frozen=True)
class CrossFit:
    """What a cross-fitted calibration produced, and how."""

    calibrated: np.ndarray
    thresholds: np.ndarray          # per patient, from the folds that trained it
    assign: np.ndarray
    kind: str
    k: int
    n_folds_fitted: int
    n_folds_degenerate: int
    target_sensitivity: float
    params: tuple = ()

    @property
    def degenerate(self) -> bool:
        return self.n_folds_degenerate > 0


def crossfit_detail(scores, labels, kind: str, *, k: int = DEFAULT_FOLDS,
                    seed: int = 0, target_sensitivity: float = 0.95,
                    with_thresholds: bool = True) -> CrossFit:
    """Cross-fitted calibration, with everything a caller might need to declare.

    Each patient's calibrated score comes from a map fitted on the other folds,
    and so does the threshold reported beside it. Both matter: a rule that fits
    its map out of fold but picks its operating threshold on the full cohort has
    only moved the leak, and the resulting sensitivity is still optimistic.

    Folds whose training half holds a single class cannot be calibrated. They
    fall back to the identity map and are COUNTED, because a calibration whose
    folds mostly degenerated is an identity map wearing a name.
    """
    from .errors import BadInput

    if kind not in CALIBRATORS and kind != "none":
        raise BadInput(f"unknown calibrator {kind!r}",
                       remedy=f"Use one of: {', '.join(CALIBRATORS)}, or 'none'.")
    s = np.asarray(scores, float)
    y = np.asarray(labels, float)
    if s.shape != y.shape:
        raise BadInput(f"{s.size} scores against {y.size} labels",
                       remedy="Calibration is paired; align them first.")
    assign = folds(len(s), k, seed)
    out = np.empty(len(s), float)
    thr = np.empty(len(s), float)
    fitted = degenerate = 0
    params = []
    for f in np.unique(assign):
        train, test = assign != f, assign == f
        if train.sum() < 3 or len(np.unique(y[train])) < 2:
            out[test] = s[test]
            thr[test] = (sensitivity_threshold(s[train], y[train],
                                               target_sensitivity)
                         if with_thresholds and train.any() else 0.0)
            degenerate += 1
            continue
        # the training half is mapped only when a threshold has to be chosen on
        # it; skipping that halves the work for callers that want the
        # calibrated scores alone
        if kind == "none":
            test_p = s[test]
            train_p = s[train] if with_thresholds else None
            params.append(("none",))
        elif kind == "isotonic":
            kx, ky = pav(s[train], y[train])
            test_p = apply_pav(kx, ky, s[test])
            train_p = apply_pav(kx, ky, s[train]) if with_thresholds else None
            params.append(("isotonic", int(len(kx))))
        elif kind == "platt":
            a, b = platt(s[train], y[train])
            test_p = apply_platt(a, b, s[test])
            train_p = apply_platt(a, b, s[train]) if with_thresholds else None
            params.append(("platt", float(a), float(b)))
        else:
            t = temperature(s[train], y[train])
            test_p = apply_temperature(t, s[test])
            train_p = (apply_temperature(t, s[train]) if with_thresholds
                       else None)
            params.append(("temperature", float(t)))
        out[test] = test_p
        thr[test] = (sensitivity_threshold(train_p, y[train], target_sensitivity)
                     if with_thresholds else 0.0)
        fitted += 1
    return CrossFit(calibrated=np.clip(out, 0.0, 1.0), thresholds=thr,
                    assign=assign, kind=kind, k=int(k), n_folds_fitted=fitted,
                    n_folds_degenerate=degenerate,
                    target_sensitivity=float(target_sensitivity),
                    params=tuple(params))


def crossfit(scores: np.ndarray, labels: np.ndarray, kind: str, *,
             k: int = DEFAULT_FOLDS, seed: int = 0) -> np.ndarray:
    """Calibrated scores, each from a map fitted without that patient."""
    return crossfit_detail(scores, labels, kind, k=k, seed=seed,
                           with_thresholds=False).calibrated


# --------------------------------------------------------------------------- #
# How good is a probability?
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CalibrationReport:
    brier: float
    ece: float
    bins: int
    slope: float
    intercept: float
    ici: float
    n: int
    null_reference: float
    detail: tuple = ()


def brier(p: np.ndarray, y: np.ndarray) -> float:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    return float(np.mean((p - y) ** 2))


def ece_equal_mass(p: np.ndarray, y: np.ndarray, bins: int = 5):
    """Expected calibration error over equal-mass bins.

    Equal mass rather than equal width: with scores that pile up near one, an
    equal-width binning puts almost every patient in the last bin and reports a
    calibration error of nearly zero for a predictor that is badly calibrated.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    n = len(p)
    if n == 0:
        return float("nan"), ()

    # A TIE GROUP IS NEVER SPLIT. Patients sharing a score must share a bin:
    # cutting them apart by row position made the answer depend on the order
    # the frame happened to be in, and gave a perfectly calibrated predictor a
    # large non-zero error. Ten patients all at p = 0.30 of whom exactly three
    # are positive scored 0.36 rather than 0.
    #
    # This is not a corner case here. Isotonic calibration is a step function:
    # cross-fitted on 170 patients it yields about 40 distinct values with tie
    # groups of a dozen or more against a bin size of 34, and permuting the
    # rows moved the reported ECE from 0.050 to 0.070 -- across the finite-
    # sample null, so the headline calibration claim flipped on row order alone.
    values, counts = np.unique(p, return_counts=True)
    target = n / max(min(bins, len(values)), 1)
    groups, current, filled = [], [], 0
    for value, count in zip(values, counts):
        current.append(value)
        filled += count
        if filled >= target - 1e-9 and len(groups) < min(bins, len(values)) - 1:
            groups.append((list(current), filled))
            current, filled = [], 0
    if current or not groups:
        groups.append((list(current), filled))

    err, detail = 0.0, []
    for members, size in groups:
        if size == 0:
            continue
        mask = np.isin(p, members)
        conf, acc = float(p[mask].mean()), float(y[mask].mean())
        err += mask.sum() / n * abs(conf - acc)
        detail.append({"n": int(mask.sum()), "conf": conf, "acc": acc})
    return float(err), tuple(detail)


def slope_intercept(p: np.ndarray, y: np.ndarray):
    r"""Logistic recalibration slope and intercept of the log-odds.

    The Cox calibration slope: fit logit(y) ~ a * logit(p) + b, where a = 1 is
    perfect. **a < 1 means the scores are too extreme** -- their spread is wider
    than the outcomes justify, which is the classic signature of a model fitted
    on the data it is scored on. **a > 1 means they are too timid**, compressed
    toward the base rate.

    The direction is written out because it was reversed in this docstring for
    two revisions while the code was right, which is precisely the kind of error
    that survives review and then appears in a manuscript.

    The fit is the same Newton solver used for Platt scaling, run on the logit.
    """
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    # `platt` returns the identity (1, 0) when one class is absent, which is a
    # defensible neutral map for a CALIBRATOR and the worst possible value for
    # a calibration SLOPE: a = 1 is this docstring's own definition of perfect.
    # `ici` guards the same case and returns nan; this did not.
    if len(p) < 3 or len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    logit = np.log(p / (1 - p))
    a, b = platt(logit, y)
    return float(a), float(b)


def ici(p: np.ndarray, y: np.ndarray) -> float:
    """Integrated calibration index: mean |smoothed observed - predicted|.

    Smoothing is the isotonic fit, which needs no bandwidth choice and cannot
    be tuned into a flattering answer.
    """
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    if len(p) < 3 or len(np.unique(y)) < 2:
        return float("nan")
    kx, ky = pav(p, y)
    return float(np.mean(np.abs(apply_pav(kx, ky, p) - p)))


def null_reference(p: np.ndarray, bins: int = 5, reps: int = 400,
                   seed: int = 0) -> float:
    """The ECE a PERFECTLY calibrated predictor with these scores would show.

    An expected calibration error is not zero for a good predictor on a finite
    sample, so the observed value is meaningless without this. Reported beside
    it everywhere.

    The reference resamples the predictor's OWN probabilities and draws
    outcomes from them: `y ~ Bernoulli(p)`. That is the only construction that
    answers the question, because the finite-sample ECE of a perfect predictor
    depends on how its probabilities are spread. The version this replaces drew
    p from Uniform(0, 1) and then rescaled toward the observed prevalence and
    clipped -- which at this cohort's 0.68 pinned 27% of the draws at exactly
    1.0, missed the target prevalence by five points, and reported the same
    number for a tight predictor and a bimodal one that differ by 70%.
    """
    p = np.asarray(p, float)
    n = len(p)
    if n == 0:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = np.empty(reps, dtype=float)
    for r in range(reps):
        draw = (rng.uniform(0, 1, n) < p).astype(float)
        vals[r], _ = ece_equal_mass(p, draw, bins)
    return float(vals.mean())


def report(p: np.ndarray, y: np.ndarray, *, bins: int = 5,
           seed: int = 0) -> CalibrationReport:
    e, detail = ece_equal_mass(p, y, bins)
    a, b = slope_intercept(p, y)
    return CalibrationReport(brier=brier(p, y), ece=e, bins=bins, slope=a,
                             intercept=b, ici=ici(p, y), n=int(len(p)),
                             null_reference=null_reference(p, bins, seed=seed),
                             detail=detail)


# --------------------------------------------------------------------------- #
# Speciation calibration, kept separate from infection calibration
# --------------------------------------------------------------------------- #
#
# Protocol v3 §11. The previous version reported one set of calibration numbers
# and let a reader decide whether they described the infected/uninfected call or
# the six-way species call. They are different questions on different
# denominators, and pooling them produces a number that answers neither.
#
# 11.A is the binary infection analysis and is everything above this line.
# 11.B is here: the class posterior among the TRULY INFECTED patients only, so
# the denominator matches the speciation endpoint and the screening decision
# cannot contaminate it.


def multiclass_brier(P: np.ndarray, y_onehot: np.ndarray) -> float:
    r"""The multiclass Brier score, by the definition the paper states.

        BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    This is the sum-over-classes form, whose range is [0, 2] rather than
    [0, 1] -- there are two published conventions and quoting a value without
    the formula is how a reader compares it against the wrong one. The formula
    goes in the Methods; this docstring is what the Methods is generated from.
    """
    P = np.asarray(P, dtype=float)
    Y = np.asarray(y_onehot, dtype=float)
    if P.shape != Y.shape:
        from .errors import BadInput

        raise BadInput(f"{P.shape} probabilities against {Y.shape} labels",
                       remedy="One row per patient, one column per class.")
    if P.size == 0:
        return float("nan")
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


@dataclass(frozen=True)
class SpeciationCalibration:
    n: int
    n_classes: int
    brier: float
    brier_formula: str
    per_class: tuple
    top_label_accuracy: float
    mean_top_probability: float
    probabilities_sum_to_one: bool
    uninfected_mass: str
    note: str


def speciation_report(evidence, truth, classes, *, bins: int = 5,
                      seed: int = 0) -> SpeciationCalibration:
    """Calibration of the species posterior, among truly infected patients.

    `evidence` is the per-class evidence table; the posterior is that evidence
    normalised to sum to one over the parasite classes. A patient with no
    parasite evidence at all has no posterior -- there is nothing to normalise
    -- and is reported in the note rather than given a uniform one, because a
    uniform posterior is a claim of equal support and this is a claim of no
    support.
    """
    import pandas as pd

    ev = pd.DataFrame(evidence)
    cols = [c for c in classes if c in ev.columns]
    idx = [i for i in truth.index if str(truth[i]) in set(classes)]
    if not idx or not cols:
        return SpeciationCalibration(
            0, len(cols), float("nan"),
            "BS = mean_i sum_k (p_ik - y_ik)^2", (), float("nan"), float("nan"),
            False, "not applicable",
            "no infected patients or no parasite classes present")

    E = ev.loc[idx, cols].to_numpy(dtype=float).clip(min=0.0)
    total = E.sum(axis=1)
    has = total > 0
    P = np.zeros_like(E)
    P[has] = E[has] / total[has, None]

    Y = np.zeros_like(E)
    for r, i in enumerate(idx):
        Y[r, cols.index(str(truth[i]))] = 1.0

    scored = has                      # only patients with a posterior
    bs = multiclass_brier(P[scored], Y[scored])
    per_class = []
    for j, c in enumerate(cols):
        m = Y[scored][:, j] > 0.5
        n_c = int(m.sum())
        per_class.append({
            "class": c, "n": n_c,
            "mean_probability_when_true": float(P[scored][m, j].mean())
            if n_c else float("nan"),
            "mean_probability_when_false": float(P[scored][~m, j].mean())
            if (~m).any() else float("nan"),
            "interpretable": n_c >= 10,
        })
    top = P[scored].argmax(axis=1)
    truth_idx = Y[scored].argmax(axis=1)
    return SpeciationCalibration(
        n=int(scored.sum()), n_classes=len(cols), brier=bs,
        brier_formula="BS = mean_i sum_k (p_ik - y_ik)^2, range [0, 2]",
        per_class=tuple(per_class),
        top_label_accuracy=float((top == truth_idx).mean()) if scored.any()
        else float("nan"),
        mean_top_probability=float(P[scored].max(axis=1).mean())
        if scored.any() else float("nan"),
        probabilities_sum_to_one=True,
        uninfected_mass=("none: the posterior is over the parasite classes "
                         "only, conditional on infection, so no mass is "
                         "assigned to uninfected"),
        note=(f"{int((~has).sum())} of {len(idx)} infected patients carried no "
              "parasite evidence and therefore no posterior; they are excluded "
              "from this analysis and counted here rather than given a uniform "
              "posterior, which would assert equal support where there is "
              "none"),
    )
