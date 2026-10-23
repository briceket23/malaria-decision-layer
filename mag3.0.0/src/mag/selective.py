"""Abstention: what a rule is worth when it may decline to answer.

The deployment case for these systems is triage — clear the easy slides so a
scarce microscopist reads the hard ones — and that case rests entirely on the
confidence being informative about the errors. It is not automatic. Under a
rule that calls every patient infected, the dominant error arrives with high
confidence, and no threshold on that confidence separates it out.

The confidence is defined once, identically for every rule, so that a
difference between two risk-coverage curves is a difference between the rules
and not between two ways of deciding what to refer:

    kappa_i = (pi_(1) - pi_(2)) * u_i

where pi_(1) >= pi_(2) are the two largest entries of the patient's class
posterior -- the evidence normalised to sum to one -- and u_i is the screening
score. The product is deliberate. A patient
is referred either because the two leading species are close, which the first
factor detects, or because there is little evidence of infection at all, which
the second detects. Both are reasons a microscopist should look, and a
statistic using only the posterior would treat a confidently-argued call on
almost no evidence as safe.

One implementation note that was a confirmed defect in the predecessor: the
threshold is taken by RANK, not by quantile value. Forty of 170 patients sat at
exactly zero confidence, so every quantile threshold retained the whole cohort
and the reported risk at 90% coverage was the full-cohort risk with "0 patients
referred" printed beside it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .schema import PARASITE_CLASSES

ABSTAIN = "abstain"
COVERAGE_TARGETS = (0.95, 0.90, 0.80)


@dataclass(frozen=True)
class RiskCoverage:
    coverage: tuple
    risk: tuple
    aurc: float
    operating: dict


def confidence(evidence: pd.DataFrame) -> pd.Series:
    """kappa for every patient, from the class-evidence table."""
    cols = [c for c in PARASITE_CLASSES if c in evidence.columns]
    ev = evidence[cols].to_numpy(dtype=float)
    if ev.size == 0:
        return pd.Series(0.0, index=evidence.index)
    total = ev.clip(min=0).sum(axis=1)
    # The margin is taken on the POSTERIOR -- the evidence normalised to sum to
    # one -- not on the raw evidence. On raw evidence a near-tie between two
    # species carrying 100 and 90 units outranks a clean single-species call
    # carrying 10, because the raw margin is 10 against 9; on the posterior the
    # clean call wins 0.91 to 0.05, which is the ordering the docstring above
    # describes and the one a microscopist would want. Since kappa decides who
    # is referred, the two definitions refer different patients.
    with np.errstate(divide="ignore", invalid="ignore"):
        post = np.where(total[:, None] > 0, ev.clip(min=0) / np.where(
            total[:, None] > 0, total[:, None], 1.0), 0.0)
    part = np.sort(post, axis=1)[:, ::-1]
    margin = part[:, 0] - (part[:, 1] if part.shape[1] > 1 else 0.0)
    # the same t/(1+t) that `rules.screening_score` computes, vectorised over
    # the frame. `test_the_two_screening_scores_are_the_same_function` holds
    # them together, because two copies of one formula drift.
    u = total / (1.0 + total)
    return pd.Series(margin * u, index=evidence.index)


def apply_coverage(pred: pd.Series, conf: pd.Series, target: float):
    """Keep the `target` fraction with the highest confidence; abstain on rest.

    Ranking rather than thresholding: with ties at zero -- which this study has
    by the dozen -- a quantile threshold retains everyone and the coverage
    control silently does nothing.
    """
    from .errors import BadInput

    if not 0.0 <= float(target) <= 1.0:
        raise BadInput(f"a coverage target of {target}",
                       remedy="Coverage is a share of the cohort: 0 to 1.")
    n = len(pred)
    keep_n = int(np.floor(target * n + 1e-9))
    c = conf.reindex(pred.index)
    if n and c.isna().any():
        raise BadInput(f"{int(c.isna().sum())} patients have no confidence",
                       remedy="Every patient needs a confidence before any of "
                              "them can be ranked against the others.")
    values = c.to_numpy(dtype=float)
    # Ties are broken on the patient identifier, not on row position. A stable
    # sort by confidence alone let the frame's own order decide which of the
    # forty patients at exactly zero was retained, and the reported risk at 90%
    # coverage then moved from 0.163 to 0.196 across permutations of the same
    # data -- either an improvement on the full-cohort 0.176 or worse than not
    # abstaining at all, depending on nothing.
    ident = np.asarray([str(i) for i in pred.index])
    order = np.lexsort((ident, -values))
    keep_positions = np.zeros(n, dtype=bool)
    keep_positions[order[:keep_n]] = True
    out = pred.copy()
    out[~keep_positions] = ABSTAIN
    if keep_n <= 0:
        thr, reproducible = float("nan"), True
    else:
        thr = float(values[order[keep_n - 1]])
        # a threshold that does not reproduce this coverage is not a threshold
        # anyone can apply to new patients, and saying so is the point
        reproducible = bool((values >= thr).sum() == keep_n)
    return out, {"coverage": keep_n / n if n else float("nan"),
                 "n_referred": int(n - keep_n), "threshold": thr,
                 "threshold_reproduces_coverage": reproducible,
                 "n_at_threshold": int((values == thr).sum()) if keep_n > 0 else 0}


def risk_coverage(pred: pd.Series, truth: pd.Series, conf: pd.Series,
                  grouping=None) -> RiskCoverage:
    """Risk against coverage, plus the area under it and three operating points."""
    from .schema import regroup

    from .metrics import _aligned

    idx = truth.index
    # a patient with no prediction is an error, not a wrong answer. Reindexing
    # put NaN there, `t != pg` was True, and the patient was silently counted
    # as misclassified -- the predecessor bug this package's schema module
    # exists to catch.
    p = _aligned(pred, truth)
    c = _aligned(conf, truth).to_numpy(dtype=float)
    t = truth.map(lambda x: regroup(x, grouping)).to_numpy()
    pg = p.map(lambda x: regroup(x, grouping)).to_numpy()
    wrong = (t != pg).astype(float)

    ident = np.asarray([str(i) for i in idx])
    order = np.lexsort((ident, -c))          # ties broken on the identifier
    wrong_sorted = wrong[order]
    n = len(idx)
    cum_wrong = np.cumsum(wrong_sorted)
    k = np.arange(1, n + 1)
    coverage = k / n
    risk = cum_wrong / k
    # The mean risk over the coverage grid: the standard AURC. Integrating the
    # trapezoid over the realised grid instead covers [1/n, 1] rather than
    # [0, 1] and drops the first slice, so the answer was short by risk[0]/n --
    # up to a fifth of a small AURC, appearing or vanishing according to
    # whether the single most confident patient happened to be right.
    aurc = float(risk.mean()) if n >= 1 else float("nan")

    operating = {}
    for target in COVERAGE_TARGETS:
        kept, meta = apply_coverage(p, conf, target)
        keep_mask = (kept != ABSTAIN).to_numpy()
        r = float(wrong[keep_mask].mean()) if keep_mask.any() else float("nan")
        operating[f"{target:g}"] = {**meta, "risk": r}
    return RiskCoverage(tuple(coverage.tolist()), tuple(risk.tolist()), aurc,
                        operating)
