"""Three nulls, because three places in this design could manufacture a result.

Every one of them is a step the analysis genuinely takes, and each is simulated
rather than argued about. An argument that a selection step "should not matter
much" is not checkable; a rejection rate under a null where nothing is true is.

  winners_curse        the reported best cell of the factorial is the maximum of
                       thirty correlated estimates, chosen on the same patients
                       it is reported on. How much does the maximum exceed the
                       truth when there is no truth to exceed?
  crossfit_null        a fitted rule scored by cross-fitting still chose its
                       family on these patients. How much apparent gain survives
                       when the score carries no signal at all -- and what does
                       leave-one-out cost against K-fold?
  selection_null       the confirmatory checkpoint is chosen by maximum
                       headroom. Does choosing it move the size of the exact
                       test that follows?

The first uses no parametric model. It permutes cell labels within each patient,
which preserves exactly how hard that patient was for every cell while
destroying any real difference between cells.

Note what that null assumes: within-row exchangeability, which is STRONGER than
"no cell is better than any other". Where several cells are near-duplicates --
two rules that happen to agree, one detector entered twice -- the permutation
treats them as independent draws and the null gap comes out too wide. On a
matrix holding two distinct detectors duplicated fifteen times each, the null
mean gap was 0.081 against an observed 0.008: the null behaved like thirty
independent cells when the data held two. It errs conservative, but it IMPOSES
an exchangeable structure rather than reading one off the data, and a reader
comparing thirty correlated cells is entitled to know that.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .stats import mcnemar_exact

ALPHA = 0.05


@dataclass(frozen=True)
class NullResult:
    name: str
    observed: float
    null_mean: float
    null_p95: float
    p_value: float
    reps: int
    detail: dict


# --------------------------------------------------------------------------- #
# 1. The winner's curse over the factorial
# --------------------------------------------------------------------------- #

def winners_curse(correct_by_cell: pd.DataFrame, *, reps: int = 2000,
                  seed: int = 0) -> NullResult:
    """How high does the best of many correlated cells sit by chance?

    `correct_by_cell` is patients x cells of 0/1. Each replicate permutes the
    cell labels independently within every patient's row. That leaves every
    patient's total number of correct cells untouched -- an easy patient stays
    easy, a hard one stays hard -- and removes any systematic difference between
    cells. The maximum cell accuracy under that null is the reference the
    observed maximum has to beat.

    The quantity to report is the optimism: observed maximum minus the mean cell
    accuracy, against the same difference under the null. A best cell that
    exceeds the mean by no more than chance does not identify a better rule; it
    identifies the largest of thirty draws.
    """
    m = np.asarray(correct_by_cell.to_numpy(), dtype=float)
    n_pat, n_cell = m.shape
    if n_pat < 2 or n_cell < 2:
        return NullResult("winners_curse", float("nan"), float("nan"),
                          float("nan"), float("nan"), 0,
                          {"note": "needs at least two patients and two cells"})
    obs_means = m.mean(axis=0)
    observed = float(obs_means.max() - obs_means.mean())

    rng = np.random.default_rng(seed)
    null = np.empty(reps, dtype=float)
    for r in range(reps):
        # one independent permutation per row, vectorised: sort a matrix of
        # random keys along the cell axis and use the resulting order
        order = np.argsort(rng.random((n_pat, n_cell)), axis=1)
        shuffled = np.take_along_axis(m, order, axis=1)
        means = shuffled.mean(axis=0)
        null[r] = means.max() - means.mean()
    # (1 + count) / (reps + 1). The identity permutation is a valid
    # permutation and is never drawn, so the plain proportion has an attainable
    # minimum of exactly zero -- and a p-value of 0.0000 printed in a
    # manuscript is a claim no permutation test can make.
    p = float((1 + int((null >= observed).sum())) / (reps + 1))
    return NullResult(
        "winners_curse", observed, float(null.mean()),
        float(np.quantile(null, 0.95)), p, reps,
        {"n_patients": int(n_pat), "n_cells": int(n_cell),
         "best_cell": str(correct_by_cell.columns[int(np.argmax(obs_means))]),
         "best_accuracy": float(obs_means.max()),
         "mean_accuracy": float(obs_means.mean()),
         "expected_optimism": float(null.mean()),
         "shrunk_best": float(obs_means.max() - null.mean()),
         "sentence": (
             f"the best of {n_cell} cells stands {100 * observed:.1f} "
             "percentage points above the mean cell; under a null in which no "
             "cell is better than any other, and with each patient's difficulty "
             f"held fixed, that gap is {100 * null.mean():.1f} points on "
             f"average and exceeds the observed gap in {100 * p:.1f}% of "
             "replicates")})


# --------------------------------------------------------------------------- #
# 2. What cross-fitting does and does not remove
# --------------------------------------------------------------------------- #

def crossfit_null(n: int = 170, prevalence: float = 0.68, *, reps: int = 400,
                  k_values=(10, None), seed: int = 0) -> dict:
    """Apparent gain from calibration when the score carries no signal.

    The score is drawn independently of the label, so the true gain is zero and
    anything the procedure reports is optimism. `k=None` means leave-one-out.

    The reference matters as much as the procedure. It is the best CONSTANT
    predictor -- call everyone by the majority class -- and not `s >= 0.5`. A
    calibrator handed a meaningless score learns the base rate and then predicts
    the majority class for everybody, which at a prevalence of 0.68 looks like
    an eighteen point gain over a coin flip and is not a gain at all; the first
    version of this function reported exactly that and called it optimism.
    Measured against the constant predictor, the same behaviour scores zero and
    what remains is the overfitting this null exists to measure.

    The comparison is the point. With 170 patients, leave-one-out trains on 169
    of them, and two patients' training sets overlap in 168 -- so every patient
    is scored by very nearly the same map, and the selection effect that
    cross-fitting is supposed to remove survives almost intact. K-fold at K=10
    makes the overlap visible in the fold count instead of hiding it. This
    function measures the difference rather than asserting it.
    """
    from . import calibrate

    out = {}
    for k in k_values:
        label = "loo" if k is None else f"k{k}"
        # Leave-one-out fits n maps per replicate against K for K-fold, so at
        # n = 170 it is seventeen times the work for the same estimate. It is a
        # comparison arm, not a headline, and it gets fewer replicates -- which
        # is recorded per arm rather than left for a reader to assume.
        arm_reps = max(reps // 4, 25) if k is None else reps
        gains = np.empty(arm_reps, dtype=float)
        rng = np.random.default_rng(seed)
        for r in range(arm_reps):
            y = (rng.random(n) < prevalence).astype(float)
            s = rng.random(n)                      # independent of y by design
            kk = n if k is None else k
            rate = float(y.mean())
            base = max(rate, 1.0 - rate)           # the best constant predictor
            best = base
            for kind in calibrate.CALIBRATORS:
                p = calibrate.crossfit(s, y, kind, k=kk, seed=r)
                acc = float(((p >= 0.5).astype(float) == y).mean())
                best = max(best, acc)
            gains[r] = best - base
        out[label] = {
            "k": ("leave-one-out" if k is None else int(k)),
            "mean_apparent_gain": float(gains.mean()),
            "p95_apparent_gain": float(np.quantile(gains, 0.95)),
            "max_apparent_gain": float(gains.max()),
            "reps": int(arm_reps), "n": int(n),
            "monte_carlo_se": float(gains.std(ddof=1) / np.sqrt(arm_reps))
            if arm_reps > 1 else float("nan"),
            "reference": "the best constant predictor at the realised prevalence",
            "note": ("the true gain is zero by construction; what is reported "
                     "here is how much the best of three cross-fitted "
                     "calibrators appears to gain on pure noise, and is the "
                     "amount by which an observed fitted-rule gain must be "
                     "discounted before it is called an improvement")}
    if "k10" in out and "loo" in out:
        gap = float(out["loo"]["mean_apparent_gain"] -
                    out["k10"]["mean_apparent_gain"])
        se = float(np.hypot(out["loo"].get("monte_carlo_se") or 0.0,
                            out["k10"].get("monte_carlo_se") or 0.0))
        out["loo_penalty"] = gap
        out["loo_penalty_se"] = se
        # a difference inside the Monte Carlo error is not a difference, and
        # saying so here stops a two-decimal-place gap becoming an argument
        out["loo_penalty_resolved"] = bool(abs(gap) > 2 * se)
        out["verdict"] = (
            ("leave-one-out left more optimism than 10-fold" if gap > 0 else
             "10-fold left more optimism than leave-one-out at this n")
            if abs(gap) > 2 * se else
            "the two fold counts left indistinguishable optimism at this "
            "number of replicates")
    return out


# --------------------------------------------------------------------------- #
# 3. Choosing the checkpoint before testing on it
# --------------------------------------------------------------------------- #

def selection_null(n_negative: int, headroom_rates, *, n_checkpoints: int = 3,
                   discordance: float = 0.30, reps: int = 4000,
                   alpha: float = ALPHA, seed: int = 0,
                   criterion: str = "headroom") -> dict:
    """Size of the exact test after the checkpoint is chosen, not before it.

    Under the null the candidate is exchangeable with the baseline: each
    discordant pair falls either way with probability one half. The question is
    whether choosing which checkpoint to test on, after seeing the data, moves
    the size of the test that follows.

    The pre-registered rule chose by maximum baseline headroom, and it was
    defended in this package's own gate module on the grounds that headroom
    reads the baseline only and therefore cannot see which way the discordant
    pairs fell. **That argument is wrong, and this function is what showed it.**
    Headroom is the number of uninfected patients the baseline calls infected,
    which is exactly

        headroom = b + (patients both rules call wrong)

    where b is the count the test's own statistic is built from. Selecting the
    largest headroom therefore selects, in part, the largest b. At three arms
    with equal headroom rates the simulated size rises to roughly 6.5% against a
    nominal 5%: an inflation, not the promised conservatism.

    `criterion="discordance"` is the repair, and it is repaired by construction
    rather than by tuning. The exact McNemar test conditions on the number of
    discordant pairs n = b + c, and conditional on n the statistic is
    Binomial(n, 1/2) whatever else is true of the arm. A selection rule that is
    a function of the n's alone therefore leaves that conditional distribution
    untouched and preserves the level exactly, for any number of arms.

    `criterion="fixed"` always tests the first arm and is the no-selection
    reference every run is compared against.

    Two caveats travel with the result and are reported in the returned dict.
    The inflation requires the arms to be close enough that the selection is
    genuinely random; where one arm's headroom dominates, the selection is
    deterministic and nothing is spent, which is reported as
    `selection_was_effectively_deterministic`. And a preserved level is not the
    same as an unchanged test -- `conservatism_spent` records how much of the
    exact test's usual slack the selection consumed.
    """
    from .errors import BadInput

    if criterion not in ("headroom", "discordance", "fixed"):
        raise BadInput(f"unknown selection criterion {criterion!r}",
                       remedy="Use 'headroom', 'discordance' or 'fixed'.")
    rng = np.random.default_rng(seed)
    rates = list(headroom_rates)
    if not rates:
        rates = [0.5] * n_checkpoints
    while len(rates) < n_checkpoints:
        rates.append(rates[-1])
    rates = rates[:n_checkpoints]

    # Each patient falls into one of four cells. The two discordant cells carry
    # equal probability, which is what "exchangeable under the null" means and
    # is the whole content of the null being simulated. The both-wrong cell then
    # absorbs whatever the arm's headroom requires.
    #
    # Not every (headroom, discordance) pair describes a possible cohort: an arm
    # whose baseline is wrong 90% of the time cannot also have 30% of patients
    # in a discordant cell split evenly. The discordance is clipped per arm to
    # the largest feasible value and the clipping is reported, because silently
    # simulating an impossible cohort would give a level that means nothing.
    per_arm = []
    clipped = False
    for rate in rates:
        pd_arm = min(discordance, 2.0 * min(rate, 1.0 - rate))
        clipped = clipped or abs(pd_arm - discordance) > 1e-12
        w = rate - pd_arm / 2.0                       # concordant, both wrong
        both_right = 1.0 - pd_arm - w
        probs = np.array([pd_arm / 2, pd_arm / 2, max(w, 0.0),
                          max(both_right, 0.0)], dtype=float)
        per_arm.append((probs / probs.sum(), pd_arm))
    #                      base wrong/cand right, base right/cand wrong,
    #                      both wrong,            both right
    base_of = np.array([False, True, False, True])
    cand_of = np.array([True, False, False, True])

    rejected = np.zeros(reps, dtype=bool)
    rejected_fixed = np.zeros(reps, dtype=bool)
    chosen_idx = np.zeros(reps, dtype=int)
    for r in range(reps):
        arms = []
        for probs, _ in per_arm:
            draw = rng.choice(4, size=n_negative, p=probs)
            b = base_of[draw]
            c = cand_of[draw]
            arms.append((int((~b).sum()), b, c, int((b != c).sum())))
        if criterion == "headroom":
            pick = int(np.argmax([a[0] for a in arms]))
        elif criterion == "discordance":
            pick = int(np.argmax([a[3] for a in arms]))
        else:
            pick = 0
        chosen_idx[r] = pick
        _, b, c, _ = arms[pick]
        rejected[r] = mcnemar_exact(b, c, alternative="greater").p <= alpha
        _, b0, c0, _ = arms[0]
        rejected_fixed[r] = mcnemar_exact(b0, c0, alternative="greater").p <= alpha

    size = float(rejected.mean())
    size_fixed = float(rejected_fixed.mean())
    se = float(np.sqrt(size * (1 - size) / reps))
    return {
        "criterion": criterion,
        "size_after_selection": size,
        "size_without_selection": size_fixed,
        "monte_carlo_se": se,
        "upper_95": size + 1.96 * se,
        "alpha": alpha, "reps": int(reps),
        "n_negative": int(n_negative), "n_checkpoints": int(n_checkpoints),
        "headroom_rates": rates, "discordance": float(discordance),
        "discordance_per_arm": [float(d) for _, d in per_arm],
        "discordance_was_clipped": bool(clipped),
        "chosen_distribution": {int(i): int((chosen_idx == i).sum())
                                for i in range(n_checkpoints)},
        "selection_was_effectively_deterministic": bool(
            max((chosen_idx == i).mean() for i in range(n_checkpoints)) > 0.99),
        # `A or B` where A implies B is just B. Written that way, the
        # Monte-Carlo guard never fired and a run whose upper 95% bound was
        # 0.0714 reported its level as preserved -- on the one boolean this
        # package uses to defend its own selection rule.
        "level_preserved": bool(size + 1.96 * se <= alpha),
        "level_verdict": ("preserved" if size + 1.96 * se <= alpha else
                          "not preserved" if size - 1.96 * se > alpha else
                          "not resolved at this number of replicates"),
        "conservatism_spent": float(size - size_fixed),
        "sentence": (
            f"selecting the checkpoint by {criterion} and then testing on it "
            f"rejected in {100 * size:.2f}% of replicates against a nominal "
            f"{100 * alpha:.0f}%, compared with {100 * size_fixed:.2f}% for a "
            "checkpoint fixed in advance; the level is "
            + ("preserved" if size + 1.96 * se <= alpha else
               "NOT preserved" if size - 1.96 * se > alpha else
               "not resolved at this number of replicates")
            + f" and {100 * (size - size_fixed):+.2f} percentage points of the "
              "exact test's usual conservatism is spent")}


def selection_comparison(n_negative: int, headroom_rates, **kw) -> dict:
    """All three selection criteria at one set of arms, side by side.

    This is the table the design section should carry. A rule that inflates the
    size is not a matter of opinion once the three sizes sit in one row.
    """
    out = {}
    for criterion in ("fixed", "headroom", "discordance"):
        r = selection_null(n_negative, headroom_rates, criterion=criterion, **kw)
        out[criterion] = {k: r[k] for k in
                          ("size_after_selection", "monte_carlo_se", "upper_95",
                           "level_preserved", "level_verdict",
                           "conservatism_spent",
                           "selection_was_effectively_deterministic")}
    alpha = kw.get("alpha", ALPHA)
    sizes = {k: out[k]["size_after_selection"] for k in
             ("fixed", "headroom", "discordance")}
    if max(sizes.values()) <= 0.0:
        out["verdict"] = (
            "no replicate rejected under any criterion: at these headroom and "
            "discordance rates the arms produce too few discordant pairs for "
            "the exact test to reach any level, so this simulation says nothing "
            "about selection and must not be quoted as evidence that it is safe")
        out["degenerate"] = True
        return out
    out["degenerate"] = False
    worst = max(sizes, key=lambda k: sizes[k])
    out["verdict"] = (
        f"at these arms the {worst} criterion had the largest size "
        f"({100 * sizes[worst]:.2f}% against a nominal {100 * alpha:.0f}%); "
        + ("selection by discordance held its level, as the conditional "
           "argument requires"
           if out["discordance"]["upper_95"] <= alpha else
           "even selection by discordance exceeded its nominal level, which "
           "should not happen and means this simulation, not the test, needs "
           "checking"))
    return out
