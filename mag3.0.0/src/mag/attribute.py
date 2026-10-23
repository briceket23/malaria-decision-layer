"""Rule against checkpoint, as a model rather than as a pair of ranges.

Protocol v3 §9. This replaces the weakest analysis in the previous version and
retracts one of its claims.

**What was wrong.** The previous version compared a range across fourteen rules
with a range across two or three checkpoints and called the difference an
apportionment. A range grows with the number of levels; the two factors have
wildly different level counts; the comparison is outlier-sensitive. It then
claimed the interaction "is arithmetically the residual and is not separately
estimable."

**That claim is false, and an external review was right to reject it.** Every
patient contributes an outcome under EVERY rule-checkpoint cell. The design is a
complete factorial with one repeated patient-level observation per cell per
patient -- 170 of them -- not one observation per cell. What the previous
version actually lacked was independent residual error *at the cell-mean level*,
which is a much narrower statement and does not prevent estimating anything.

**What replaces it.** A patient-clustered model on the binary patient-level
outcome:

    logit P(Y_irc = 1) = mu + alpha_r + beta_c + (alpha beta)_rc + u_i

fitted as a GEE with an exchangeable working correlation and robust sandwich
standard errors clustered on patient. Reported: the marginal rule effect, the
marginal checkpoint effect, the interaction with a test, and the contrast
between the two marginals -- which is the paper's claim and is pre-declared as
the estimand.

**Why statsmodels.** It is the first dependency outside numpy and pandas that
this package carries, and the choice is deliberate. A reviewer challenging the
attribution analysis should be able to check it against a named implementation;
"we wrote our own GEE" invites a question the paper does not need. The
no-dependency argument is strong for a Brier score and weak for a GEE.

**And why there is also a hand-rolled estimator below.** `sandwich_check` fits
the same marginal model by Fisher scoring with a cluster-robust covariance
written out in forty lines, and `fit` reports the agreement between the two. It
is not a fallback and it is not there to be trusted instead: it is there so that
a disagreement between two independent implementations is visible rather than
absent. If they disagree, the analysis says so and neither is reported.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .errors import BadInput

#: Agreement tolerance between statsmodels and the hand-rolled estimator, on
#: the log-odds scale. Chosen far below any difference that could change a
#: sentence: the effects this study reports are tenths, not thousandths.
AGREEMENT_TOL = 1e-3

#: Bootstrap replicates for the probability-scale contrast, which is the
#: PRIMARY estimand. See `fit` for why it is not the log-odds one.
CONTRAST_BOOT = 2000


@dataclass
class Attribution:
    """The factorial, modelled. Every field is reported; none is a verdict."""

    n_patients: int = 0
    n_rules: int = 0
    n_checkpoints: int = 0
    #: marginal effects on the log-odds scale, with clustered intervals
    rule_effect: dict = field(default_factory=dict)
    checkpoint_effect: dict = field(default_factory=dict)
    #: the PRIMARY estimand: rule marginal spread minus checkpoint marginal
    #: spread, on the probability scale. Always defined, separation or not.
    contrast: dict = field(default_factory=dict)
    #: the same contrast on the log-odds scale, when no level is separated
    contrast_logodds: dict = field(default_factory=dict)
    interaction: dict = field(default_factory=dict)
    #: per-level marginal means, for the figure
    rule_means: dict = field(default_factory=dict)
    checkpoint_means: dict = field(default_factory=dict)
    agreement: dict = field(default_factory=dict)
    separation: dict = field(default_factory=dict)
    #: whether the inverse link overflowed while fitting, and
    #: whether a separated level explains it
    numerics: dict = field(default_factory=dict)
    converged: bool = False
    note: str = ""
    sentence: str = ""


# --------------------------------------------------------------------------- #
# The design matrix
# --------------------------------------------------------------------------- #

def long_frame(predictions: dict, truth: pd.Series, order, *, rules,
               checkpoints, grouping=None) -> pd.DataFrame:
    """One row per (patient, rule, checkpoint): the model's input.

    This is the frame the previous version never built, and building it is what
    makes the interaction estimable. `Y` is patient-level correctness on the
    same grouping the factorial's accuracy column uses -- a model fitted on a
    different grouping would answer a different question from the table it
    qualifies.
    """
    from .metrics import _aligned
    from .schema import regroup

    t = truth.reindex(order).map(lambda x: regroup(x, grouping))
    rows = []
    for ck in checkpoints:
        for rid in rules:
            pred = predictions.get((ck, rid))
            if pred is None:
                raise BadInput(
                    f"no predictions for {ck}:{rid}",
                    remedy=("The model needs a complete factorial. Either "
                            "supply every cell or restrict `rules` and "
                            "`checkpoints` to the cells that exist."))
            p = _aligned(pred, truth).reindex(order).map(
                lambda x: regroup(x, grouping))
            rows.append(pd.DataFrame({
                "patient": [str(i) for i in order],
                "rule": rid,
                "checkpoint": ck,
                "y": (p.to_numpy() == t.to_numpy()).astype(int),
            }))
    out = pd.concat(rows, ignore_index=True)
    out["rule"] = out["rule"].astype("category")
    out["checkpoint"] = out["checkpoint"].astype("category")
    return out


def _design(df: pd.DataFrame, *, interaction: bool):
    """Treatment-coded design matrix, with the first level of each factor held
    as the reference. Returned with its column names so that a coefficient can
    be traced to the level it belongs to."""
    rules = list(pd.Categorical(df["rule"]).categories)
    cks = list(pd.Categorical(df["checkpoint"]).categories)
    cols = {"(intercept)": np.ones(len(df))}
    for r in rules[1:]:
        cols[f"rule[{r}]"] = (df["rule"] == r).to_numpy(dtype=float)
    for c in cks[1:]:
        cols[f"ckpt[{c}]"] = (df["checkpoint"] == c).to_numpy(dtype=float)
    if interaction:
        for r in rules[1:]:
            for c in cks[1:]:
                cols[f"rule[{r}]:ckpt[{c}]"] = (
                    (df["rule"] == r).to_numpy(dtype=float)
                    * (df["checkpoint"] == c).to_numpy(dtype=float))
    names = list(cols)
    return np.column_stack([cols[n] for n in names]), names, rules, cks


# --------------------------------------------------------------------------- #
# The hand-rolled cross-check
# --------------------------------------------------------------------------- #

def sandwich_check(X: np.ndarray, y: np.ndarray, cluster: np.ndarray,
                   *, iters: int = 60, tol: float = 1e-10) -> dict:
    """Logistic regression with a cluster-robust covariance, written out.

    Fisher scoring for the point estimate, then the standard sandwich

        V = (X'WX)^-1 [ sum_g (X_g' e_g)(X_g' e_g)' ] (X'WX)^-1

    summed over clusters g. This is the independence-working-correlation GEE,
    which has the same point estimate as an exchangeable one for a saturated
    mean model and a covariance that should agree closely. It exists to be
    compared against statsmodels, not to replace it -- two implementations that
    agree are evidence; one implementation is an assertion.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, p = X.shape
    beta = np.zeros(p)
    beta[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))

    converged = False
    for _ in range(iters):
        eta = np.clip(X @ beta, -30, 30)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(mu * (1 - mu), 1e-10, None)
        z = eta + (y - mu) / w
        # weighted least squares step, ridged so a separated cell cannot make
        # the information matrix singular and end the fit with a traceback
        XtW = X.T * w
        try:
            new = np.linalg.solve(XtW @ X + 1e-8 * np.eye(p), XtW @ z)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(new - beta)) < tol:
            beta = new
            converged = True
            break
        beta = new

    eta = np.clip(X @ beta, -30, 30)
    mu = 1.0 / (1.0 + np.exp(-eta))
    w = np.clip(mu * (1 - mu), 1e-10, None)
    bread = np.linalg.pinv((X.T * w) @ X)
    resid = y - mu
    meat = np.zeros((p, p))
    for g in np.unique(cluster):
        m = cluster == g
        s = X[m].T @ resid[m]
        meat += np.outer(s, s)
    cov = bread @ meat @ bread
    return {"beta": beta, "cov": cov, "converged": bool(converged),
            "n_clusters": int(len(np.unique(cluster)))}


# --------------------------------------------------------------------------- #
# The analysis
# --------------------------------------------------------------------------- #

def detect_separation(df: pd.DataFrame) -> dict:
    """Levels whose outcome never varies -- the logistic model's blind spot.

    A rule that is right about every patient, or wrong about every patient, has
    a log-odds effect of plus or minus infinity. That is not a numerical
    problem to be worked around; it is the data, and a solver that returns a
    large finite number instead has returned an artefact of where its iteration
    stopped.

    **This is not a corner case here.** The reconstructed published rule calls
    every uninfected patient infected, so on the screening endpoint it is
    perfectly separated by construction, and any factorial containing it will
    contain a separated level. The first version of this module fitted anyway
    and the two implementations disagreed by 4e8 -- which is how the condition
    was found.
    """
    out = {"rules": [], "checkpoints": [], "cells": []}
    for factor, key in (("rule", "rules"), ("checkpoint", "checkpoints")):
        for lvl, g in df.groupby(factor, observed=True):
            m = float(g["y"].mean())
            if m in (0.0, 1.0):
                out[key].append({"level": str(lvl), "outcome": m,
                                 "n": int(len(g))})
    for (r, c), g in df.groupby(["rule", "checkpoint"], observed=True):
        m = float(g["y"].mean())
        if m in (0.0, 1.0):
            out["cells"].append({"rule": str(r), "checkpoint": str(c),
                                 "outcome": m})
    out["any"] = bool(out["rules"] or out["checkpoints"])
    out["n_separated_levels"] = len(out["rules"]) + len(out["checkpoints"])
    out["note"] = (
        "a level whose outcome never varies has an unbounded log-odds effect. "
        "Where this occurs the log-odds contrast is not reported and the "
        "probability-scale contrast, which remains defined, governs."
        if out["any"] else "no level was separated")
    return out


def probability_contrast(df: pd.DataFrame, *, n_boot: int = CONTRAST_BOOT,
                         alpha: float = 0.05, seed: int = 0) -> dict:
    """The PRIMARY estimand, on the probability scale, clustered on patient.

    The difference between the rule marginal-mean spread and the checkpoint
    marginal-mean spread, with a bootstrap over PATIENTS -- so the interval
    respects the fact that one patient contributes an outcome to every cell.

    This is not the naive range comparison an external review rightly rejected.
    That comparison was unmodelled and uninterval'd; this one resamples the
    unit of observation and reports the difference of the two spreads as a
    single estimand with its own interval. It is the primary because it is
    always defined: a separated level makes a log-odds effect infinite but
    leaves a marginal mean perfectly finite, and the reconstructed published
    rule is separated by construction.
    """
    pivot = (df.pivot_table(index="patient", columns=["rule", "checkpoint"],
                            values="y", observed=True)
             .sort_index(axis=1))
    rules_lv = list(dict.fromkeys(c[0] for c in pivot.columns))
    cks_lv = list(dict.fromkeys(c[1] for c in pivot.columns))
    arr = pivot.to_numpy(dtype=float)
    n_pat = arr.shape[0]
    grid = arr.reshape(n_pat, len(rules_lv), len(cks_lv))

    def spreads(a):
        m = a.mean(axis=0)
        r, c = m.mean(axis=1), m.mean(axis=0)
        return float(r.max() - r.min()), float(c.max() - c.min())

    obs_r, obs_c = spreads(grid)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        sr, sc = spreads(grid[rng.integers(0, n_pat, n_pat)])
        diffs[b] = sr - sc
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    return {
        "scale": "probability (patient-level accuracy)",
        "rule_spread": obs_r, "checkpoint_spread": obs_c,
        "estimate": float(obs_r - obs_c),
        "lo": float(lo), "hi": float(hi),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "n_boot": int(n_boot), "n_patients": int(n_pat),
        "estimand": ("the difference between the rule marginal-mean spread and "
                     "the checkpoint marginal-mean spread in patient-level "
                     "accuracy, bootstrapped over patients; pre-declared"),
    }


def _marginal_spread(df: pd.DataFrame, factor: str) -> dict:
    """Observed marginal means per level of a factor, and their range.

    Descriptive, and labelled as such wherever it is printed. It is here so the
    figure and the model come from one object, not so it can carry a claim.
    """
    means = df.groupby(factor, observed=True)["y"].mean()
    return {"means": {str(k): float(v) for k, v in means.items()},
            "range": float(means.max() - means.min()),
            "n_levels": int(len(means))}


def fit(df: pd.DataFrame, *, alpha: float = 0.05) -> Attribution:
    """Fit the factorial model and report every quantity §9 requires."""
    out = Attribution(
        n_patients=int(df["patient"].nunique()),
        n_rules=int(df["rule"].nunique()),
        n_checkpoints=int(df["checkpoint"].nunique()),
        rule_means=_marginal_spread(df, "rule"),
        checkpoint_means=_marginal_spread(df, "checkpoint"),
    )
    if out.n_rules < 2 or out.n_checkpoints < 2:
        out.note = (f"{out.n_rules} rules and {out.n_checkpoints} checkpoints: "
                    "a two-factor model needs at least two levels of each. No "
                    "model was fitted and no attribution is reported.")
        return out

    out.separation = detect_separation(df)
    # The primary estimand, computed first and unconditionally: it is defined
    # whether or not the log-odds model is.
    out.contrast = probability_contrast(df)

    try:
        from statsmodels.genmod.cov_struct import Exchangeable
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE
    except ImportError as exc:                                  # pragma: no cover
        raise BadInput(
            "statsmodels is required for the factorial attribution",
            remedy=("Install it: `pip install statsmodels`. The attribution "
                    "analysis is the paper's second contribution and is fitted "
                    "with a named implementation deliberately, so that a "
                    "reviewer can check it against one they recognise."),
        ) from exc

    groups = pd.factorize(df["patient"], sort=True)[0]
    y = df["y"].to_numpy(dtype=float)

    # --- the interaction test: full against additive -----------------------
    X_full, names_full, rules, cks = _design(df, interaction=True)
    X_add, names_add, _, _ = _design(df, interaction=False)

    # Under separation the linear predictor diverges and the inverse link
    # overflows in `exp`. That is arithmetic reporting the separation we
    # already detected, not a defect, and numpy's warning would otherwise be
    # promoted to an error by this package's own filter. It is silenced here,
    # at the two calls that can raise it and nowhere else, and the fact that it
    # fired is kept rather than swallowed -- a run that overflows without a
    # separated level would be a real finding.
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"), \
            warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        full = GEE(y, X_full, groups=groups, family=Binomial(),
                   cov_struct=Exchangeable()).fit()
        add = GEE(y, X_add, groups=groups, family=Binomial(),
                  cov_struct=Exchangeable()).fit()
    overflowed = any(issubclass(w.category, RuntimeWarning) and
                     "overflow" in str(w.message) for w in caught)
    out.converged = bool(getattr(full, "converged", True))
    out.numerics = {
        "link_overflowed": bool(overflowed),
        "expected": bool(out.separation.get("any")),
        "note": ("the inverse logit overflowed while fitting, which is what a "
                 "separated level does to a linear predictor; the probability-"
                 "scale contrast is unaffected because it is not fitted"
                 if overflowed and out.separation.get("any") else
                 "the inverse logit overflowed although no level is fully "
                 "separated; treat the log-odds fit as unstable and read the "
                 "probability-scale contrast instead"
                 if overflowed else "no overflow"),
    }

    inter_idx = [i for i, n in enumerate(names_full) if ":" in n]
    if inter_idx:
        b = np.asarray(full.params)[inter_idx]
        V = np.asarray(full.cov_params())[np.ix_(inter_idx, inter_idx)]
        try:
            stat = float(b @ np.linalg.pinv(V) @ b)
            dfree = len(inter_idx)
            # upper tail of a chi-square with dfree degrees of freedom, by the
            # regularised gamma function, so no scipy is needed for a p-value
            p_inter = _chi2_sf(stat, dfree)
        except np.linalg.LinAlgError:                            # pragma: no cover
            stat, dfree, p_inter = float("nan"), len(inter_idx), float("nan")
        out.interaction = {
            "n_terms": int(dfree), "wald_chi2": stat, "p": p_inter,
            "detected": bool(np.isfinite(p_inter) and p_inter <= alpha),
            "note": ("a Wald test on the rule x checkpoint terms, clustered on "
                     "patient. This test is what the previous version said "
                     "could not be computed."),
        }

    # --- marginal effects, from the ADDITIVE model -------------------------
    # Marginal, therefore additive: with an interaction present, a single
    # "rule effect" is an average over checkpoints, and the interaction test
    # above is what tells a reader whether that average conceals anything.
    z = _z_two_sided(alpha)
    params = np.asarray(add.params)
    se = np.sqrt(np.diag(np.asarray(add.cov_params())))

    def levels(prefix: str, ref: str) -> dict:
        idx = [i for i, n in enumerate(names_add) if n.startswith(prefix)]
        got = {}
        for i in idx:
            lvl = names_add[i][len(prefix):-1]
            got[lvl] = {"beta": float(params[i]), "se": float(se[i]),
                        "lo": float(params[i] - z * se[i]),
                        "hi": float(params[i] + z * se[i])}
        got[ref] = {"beta": 0.0, "se": 0.0, "lo": 0.0, "hi": 0.0,
                    "reference": True}
        return got

    rule_lv = levels("rule[", rules[0])
    ck_lv = levels("ckpt[", cks[0])

    def spread(lv: dict) -> dict:
        betas = np.array([v["beta"] for v in lv.values()])
        hi_name = max(lv, key=lambda k: lv[k]["beta"])
        lo_name = min(lv, key=lambda k: lv[k]["beta"])
        return {"levels": lv, "max_level": hi_name, "min_level": lo_name,
                "spread_logodds": float(betas.max() - betas.min())}

    out.rule_effect = spread(rule_lv)
    out.checkpoint_effect = spread(ck_lv)

    # --- the same contrast on the log-odds scale, where it is defined ------
    # Reported only when no level is separated. A separated level's coefficient
    # is unbounded, and a linear contrast that includes one is a statement
    # about where the solver stopped iterating.
    contrast_vec = np.zeros(len(names_add))
    for name, sign in ((f"rule[{out.rule_effect['max_level']}]", +1.0),
                       (f"rule[{out.rule_effect['min_level']}]", -1.0),
                       (f"ckpt[{out.checkpoint_effect['max_level']}]", -1.0),
                       (f"ckpt[{out.checkpoint_effect['min_level']}]", +1.0)):
        if name in names_add:
            contrast_vec[names_add.index(name)] += sign
    cov_add = np.asarray(add.cov_params())
    est = float(contrast_vec @ params)
    var = float(contrast_vec @ cov_add @ contrast_vec)
    sd = float(np.sqrt(max(var, 0.0)))
    if out.separation["any"]:
        out.contrast_logodds = {
            "reported": False,
            "why": ("levels are separated ("
                    + ", ".join(d["level"] for d in
                                out.separation["rules"] + out.separation["checkpoints"])
                    + "), so their log-odds effects are unbounded and a "
                      "contrast containing one is not defined"),
        }
    else:
        out.contrast_logodds = {
            "reported": True,
            "estimate_logodds": est, "se": sd,
            "lo": est - z * sd, "hi": est + z * sd,
            "excludes_zero": bool((est - z * sd) > 0 or (est + z * sd) < 0),
        }

    # --- the two implementations, compared ---------------------------------
    # Skipped under separation: both fits are then chasing an infinity and a
    # disagreement between them says nothing about either.
    if out.separation["any"]:
        out.agreement = {
            "checked": False,
            "why": ("levels are separated, so both fits diverge and their "
                    "difference is not informative about either"),
        }
        out.sentence = _sentence(out)
        return out
    hand = sandwich_check(X_add, y, groups)
    gap = float(np.max(np.abs(hand["beta"] - params))) if len(params) else 0.0
    out.agreement = {
        "checked": True,
        "max_abs_coefficient_difference": gap,
        "tolerance": AGREEMENT_TOL,
        "agree": bool(gap <= AGREEMENT_TOL),
        "hand_rolled_converged": hand["converged"],
        "n_clusters": hand["n_clusters"],
        "note": ("statsmodels GEE against an independently written Fisher "
                 "scoring fit with a cluster-robust sandwich. Two "
                 "implementations that agree are evidence; one is an "
                 "assertion."),
    }
    if not out.agreement["agree"]:
        out.contrast_logodds = {
            "reported": False,
            "why": (f"the two implementations disagree by {gap:.2e} on the "
                    "log-odds scale, which exceeds the tolerance; a number two "
                    "independent fits cannot reproduce is not a number"),
        }
        out.note = out.contrast_logodds["why"]

    out.sentence = _sentence(out)
    return out


def _sentence(a: Attribution) -> str:
    """The §9.4 required form, generated. Never typed."""
    c = a.contrast
    inter = a.interaction or {}
    lo = getattr(a, "contrast_logodds", {}) or {}

    head = (
        "Within the levels surviving the yield gate and under matched "
        "post-processing, the marginal effect of the aggregation rule on "
        "patient-level accuracy exceeded the marginal effect of the detector "
        f"checkpoint by {100 * c['estimate']:.1f} percentage points "
        f"(95% CI {100 * c['lo']:.1f} to {100 * c['hi']:.1f}, bootstrapped "
        f"over {c['n_patients']} patients, who each contribute an outcome to "
        "every cell).")

    if lo.get("reported"):
        head += (f" On the log-odds scale the same contrast was "
                 f"{lo['estimate_logodds']:.3f} (95% CI {lo['lo']:.3f} to "
                 f"{lo['hi']:.3f}), from a patient-clustered generalised "
                 "estimating equation.")
    elif a.separation.get("any"):
        head += (" The log-odds form of this contrast is not reported: "
                 + lo.get("why", "a level is separated") + ".")

    if inter:
        head += (
            " A rule-by-checkpoint interaction was "
            + ("detected" if inter.get("detected") else "not detected")
            + f" (Wald chi-square {inter['wald_chi2']:.2f} on "
              f"{inter['n_terms']} degrees of freedom, p = {inter['p']:.4f}), "
              "so the marginal effects above are averages over the other "
              "factor.")

    return head + (
        " These are marginal effects with an interval, not variance "
        "components, and the descriptive ranges reported beside them are "
        "ranges over factors with different numbers of levels.")


# --------------------------------------------------------------------------- #
# Small numerics, so the module needs no scipy
# --------------------------------------------------------------------------- #

def _z_two_sided(alpha: float) -> float:
    from math import erf, sqrt
    lo, hi = 0.0, 10.0
    target = 1 - alpha / 2
    for _ in range(200):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2.0))) < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _chi2_sf(x: float, k: int) -> float:
    """Upper tail of a chi-square, by the regularised incomplete gamma.

    Series below the transition point, continued fraction above it -- the
    standard split. Written out so the p-value for the interaction test does
    not pull in a dependency for one number.
    """
    from math import exp, lgamma, log

    if not np.isfinite(x) or x <= 0:
        return 1.0
    a, xx = k / 2.0, x / 2.0
    if xx < a + 1.0:
        term = 1.0 / a
        total = term
        n = a
        for _ in range(500):
            n += 1.0
            term *= xx / n
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        lower = total * exp(-xx + a * log(xx) - lgamma(a))
        return float(min(max(1.0 - lower, 0.0), 1.0))
    # Lentz's algorithm for the continued fraction
    tiny = 1e-300
    b = xx + 1.0 - a
    c = 1.0 / tiny
    d = 1.0 / b
    h = d
    for i in range(1, 500):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    upper = exp(-xx + a * log(xx) - lgamma(a)) * h
    return float(min(max(upper, 0.0), 1.0))
