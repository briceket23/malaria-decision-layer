"""The descriptive analyses the reviewers asked for, in one place.

Four things live here because they are all answers to "what is actually going
on", rather than endpoints:

  * the box census, which shows that two detector families are not comparable
    until identical suppression has been applied;
  * the error taxonomy and fragility, which say what kind of mistake a rule
    makes and how few detections it would take to change the call;
  * the internal/external contrast, which is the only domain comparison this
    dataset supports, and only on speciation;
  * the field-count curve, which asks how many fields a rule needs.

None of them carries a p-value. They are reported as descriptions, and the
Discussion is restricted in advance to the oracle gap and the apportionment, so
that a striking cell in a table cannot become a finding after the fact.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evidence import apply_condition
from .rules import RULES
from .schema import (GROUPING_5, INTERNAL_CENTRES, PARASITE_CLASSES,
                     UNINFECTED, regroup)


# --------------------------------------------------------------------------- #
# 1. The box census
# --------------------------------------------------------------------------- #

def box_census(detections: dict, n_fields: int) -> pd.DataFrame:
    """Boxes per field per checkpoint, raw against matched.

    The quantity of interest is the ratio of the largest per-field count to the
    smallest. If it is large in the raw condition and near one after identical
    suppression, then a checkpoint effect present only in the raw condition
    belongs to post-processing and not to the detector.

    On the released weights the raw condition is not a detection count at all
    for RT-DETR: it emits a fixed number of queries per field, the same on a
    parasite-dense smear as on a clean one.
    """
    rows = []
    for name, det in detections.items():
        matched = apply_condition(det, "matched")
        raw_per_field = len(det) / max(n_fields, 1)
        matched_per_field = len(matched) / max(n_fields, 1)
        # by (patient, field), not by image name. Per-patient frame numbering
        # makes "IMG_003" several different fields, and counting them as one
        # made `per_field_sd` and `fixed_budget` -- the two things this census
        # exists to report -- describe several patients' boxes at once.
        per_field_counts = det.groupby(["patient_id", "image_id"],
                                       observed=True).size()
        rows.append({
            "checkpoint": name,
            "raw_boxes": int(len(det)),
            "raw_per_field": raw_per_field,
            "matched_boxes": int(len(matched)),
            "matched_per_field": matched_per_field,
            # a checkpoint that emitted nothing suppressed nothing; the
            # arithmetic below reported 100% for it, which is a confident
            # number for an undefined quantity
            "suppressed_pct": (100.0 * (1 - len(matched) / len(det))
                               if len(det) else float("nan")),
            "median_score_raw": float(det["score"].median()) if len(det) else float("nan"),
            "median_score_matched": float(matched["score"].median()) if len(matched) else float("nan"),
            # a fixed query budget shows up here as zero dispersion
            "per_field_sd": float(per_field_counts.std()) if len(per_field_counts) > 1 else 0.0,
            "fixed_budget": bool(len(per_field_counts) > 1
                                 and per_field_counts.nunique() == 1),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        # a floor of 1e-9 on the denominator turned "one checkpoint emitted
        # nothing" into a ratio of 999,999,999.9999999 -- the census's headline
        # quantity, fabricated
        def ratio(col: str) -> float:
            lo, hi = float(out[col].min()), float(out[col].max())
            return hi / lo if lo > 0 else (float("inf") if hi > 0 else float("nan"))
        out.attrs["ratio_raw"] = ratio("raw_per_field")
        out.attrs["ratio_matched"] = ratio("matched_per_field")
    return out


# --------------------------------------------------------------------------- #
# 2. Error taxonomy and fragility
# --------------------------------------------------------------------------- #

#: A false positive is called "many low confidence" only when it really is
#: many and really is low. The version this replaces took the parameter `view`
#: and never referenced it, so every false positive got that label from the
#: label pair alone -- including a patient whose call rested on one box at 0.99.
FP_MANY = 5
FP_LOW = 0.60


def _category(truth: str, pred: str, view, majority: str) -> str:
    if truth == pred:
        return ""
    if truth == UNINFECTED:
        scores = np.asarray(view.scores.get(pred, np.zeros(0)), dtype=float) \
            if view is not None else np.zeros(0)
        n, top = int(scores.size), (float(scores.max()) if scores.size else 0.0)
        if n >= FP_MANY and top < FP_LOW:
            return "many_low_confidence"
        if n < FP_MANY and top >= FP_LOW:
            return "few_high_confidence"
        if n >= FP_MANY:
            return "many_with_a_confident_one"
        return "few_low_confidence"
    if pred == UNINFECTED:
        return "missed_entirely"
    if {truth, pred} == {"P_ovale", "P_vivax"}:
        return "ovale_vivax"
    if majority is not None and pred == majority:
        return "majority_capture"
    return "minority_suppressed"


#: Rules whose evidence functional reads only `PatientView.scores`. Fragility
#: removes detections without knowing which field each came from, so it cannot
#: rebuild the per-field structure that a voting rule reads; running it against
#: one would silently score the rule against stale per-field maxima. The list is
#: checked rather than assumed, and a rule outside it is refused by name.
SCORE_ONLY_RULES = frozenset({"R0", "R1", "R2", "R3", "R4", "R6", "R6b",
                              "R7", "R8"})


def fragility(view, rule_id: str, params: dict | None = None) -> float:
    """How many detections would have to be removed to change the call.

    Removed from the class the rule chose, strongest first. A call that flips
    after one box is not the same result as one that survives four hundred, and
    a table of accuracies cannot tell them apart.

    Returns `inf` when no number of removals changes the call, and `nan` when
    the question is not defined because the chosen class has no detections. An
    earlier version returned n in the first case, which conflates "survives
    everything" with "flips only once all n are gone" -- and those are opposite
    readings of the same patient.
    """
    from .errors import BadInput

    if rule_id not in SCORE_ONLY_RULES:
        raise BadInput(
            f"fragility is not defined for {rule_id!r}",
            remedy=("This rule's evidence reads the per-field structure, which "
                    "cannot be rebuilt after removing detections without "
                    "knowing which field each came from. Report fragility for "
                    f"one of: {', '.join(sorted(SCORE_ONLY_RULES))}."),
            score_only=sorted(SCORE_ONLY_RULES))

    rule = RULES[rule_id]
    base = rule.apply(view, params)["label"]
    ordered = {}
    for c in PARASITE_CLASSES:
        s = np.asarray(view.scores.get(c, np.zeros(0)), dtype=float)
        ordered[c] = np.sort(s)[::-1]
    target = base if base != UNINFECTED else max(
        PARASITE_CLASSES, key=lambda c: ordered[c].size)
    n = ordered[target].size
    if n == 0:
        return float("nan")

    # one shallow view, rebound per step: the rule only reads `scores`, so
    # swapping that one dict is enough and copying the whole view per removal
    # dominated the runtime of the error table
    import copy

    trimmed = copy.copy(view)
    scores = dict(view.scores)
    trimmed.scores = scores
    for k in range(1, n + 1):
        scores[target] = ordered[target][k:]
        if rule.apply(trimmed, params)["label"] != base:
            return float(k)
    return float("inf")


def error_table(truth: pd.Series, pred: pd.Series, patients: pd.DataFrame,
                views: dict, rule_id: str, *, with_fragility: bool = True
                ) -> pd.DataFrame:
    """One row per patient: what was called, what it should have been, why."""
    from .metrics import _aligned

    infected = truth[truth != UNINFECTED]
    # a cohort with no infected patient has no majority species. The bare
    # `.idxmax()` raised `ValueError: attempt to get argmax of an empty
    # sequence`, which is not a MagError and so reached the operator as a
    # traceback rather than as a refusal with a remedy.
    majority = infected.value_counts().idxmax() if len(infected) else None
    pred = _aligned(pred, truth)
    centre = patients.set_index("patient_id")["centre"]
    rows = []
    for pid in truth.index:
        t, p = str(truth[pid]), str(pred[pid])
        rows.append({
            "patient_id": pid,
            "centre": str(centre.get(pid, "")),
            "domain": ("internal" if str(centre.get(pid, "")) in INTERNAL_CENTRES
                       else "external"),
            "truth": t, "pred": p, "correct": t == p,
            "category": _category(t, p, views[str(pid)], majority),
            "fragility": (fragility(views[str(pid)], rule_id)
                          if with_fragility else float("nan")),
        })
    return pd.DataFrame(rows)


def leave_one_centre_out(truth: pd.Series, pred: pd.Series,
                         patients: pd.DataFrame, *, min_infected: int = 10,
                         grouping=None) -> dict:
    """Speciation accuracy with each centre held out in turn.

    Protocol v3 §15. Speciation only, and for the same reason the domain
    contrast is speciation only: every uninfected patient comes from one
    centre, so a screening endpoint computed with that centre held out has no
    negatives at all.

    Centres with fewer than `min_infected` infected patients are reported but
    marked uninterpretable rather than dropped, because a reader who sees five
    centres in the cohort and four in the table will wonder what happened to
    the fifth.
    """
    from .metrics import _aligned

    p = _aligned(pred, truth)
    centre = patients.set_index("patient_id")["centre"].astype(str)
    infected = truth[truth != UNINFECTED]
    t = infected.map(lambda x: regroup(x, grouping))
    pg = p.reindex(infected.index).map(lambda x: regroup(x, grouping))
    dom = pd.Series([str(centre.get(i, "")) for i in infected.index],
                    index=infected.index)

    overall_acc = float((t == pg).mean()) if len(t) else float("nan")
    rows = {}
    for c in sorted(set(dom)):
        held_in = dom != c
        n_in = int(held_in.sum())
        n_out = int((~held_in).sum())
        rows[c] = {
            "n_infected_at_centre": n_out,
            "n_remaining": n_in,
            "accuracy_without_this_centre":
                float((t[held_in] == pg[held_in]).mean()) if n_in else float("nan"),
            "accuracy_at_this_centre":
                float((t[~held_in] == pg[~held_in]).mean()) if n_out else float("nan"),
            "interpretable": bool(n_out >= min_infected),
        }
    swings = [abs(v["accuracy_without_this_centre"] - overall_acc)
              for v in rows.values()
              if v["interpretable"] and np.isfinite(v["accuracy_without_this_centre"])]
    return {
        "overall_speciation_accuracy": overall_acc,
        "by_centre": rows,
        "max_swing": float(max(swings)) if swings else float("nan"),
        "n_interpretable_centres": int(sum(v["interpretable"] for v in rows.values())),
        "endpoint": "speciation only",
        "note": ("specificity cannot appear in a leave-one-centre-out analysis: "
                 "every uninfected patient comes from one centre, so holding "
                 "that centre out leaves no negatives to score"),
    }


def error_summary(errors: pd.DataFrame) -> dict:
    wrong = errors[~errors["correct"]]
    if wrong.empty:
        return {"n_errors": 0}
    cat = wrong["category"].value_counts()
    pred = wrong["pred"].value_counts()
    finite = errors.loc[np.isfinite(errors["fragility"]), "fragility"]
    fp = wrong[wrong["truth"] == UNINFECTED]
    fp_frag = fp.loc[np.isfinite(fp["fragility"]), "fragility"]
    return {
        "n_errors": int(len(wrong)),
        "top_category": str(cat.index[0]),
        "top_category_share": float(cat.iloc[0] / cat.sum()),
        "categories": {str(k): int(v) for k, v in cat.items()},
        "top_predicted": str(pred.index[0]),
        "top_predicted_share": float(pred.iloc[0] / pred.sum()),
        "median_fragility": float(finite.median()) if len(finite) else float("nan"),
        "median_fragility_false_positive":
            float(fp_frag.median()) if len(fp_frag) else float("nan"),
        "share_flipped_by_two_boxes":
            float((fp_frag <= 2).mean()) if len(fp_frag) else float("nan"),
        "by_centre": {str(k): int(v) for k, v in
                      wrong.groupby("centre").size().items()},
        # Protocol v3 §16. A category share of 100% does NOT mean the detector
        # is perfect, and an external reviewer was right that the previous
        # version invited exactly that reading. It means that under THIS
        # decision rule every observed patient-level error arrived through that
        # pathway -- a statement about the rule, not about the detector.
        "share_caveat": (
            f"{100 * float(cat.iloc[0] / cat.sum()):.0f}% of errors fell into "
            f"the '{str(cat.index[0]).replace('_', ' ')}' category. This "
            "describes the decision rule, not the detector: it says every "
            "observed patient-level error arrived through that pathway under "
            "this rule, not that no other failure mode exists in the "
            "detections."),
    }


# --------------------------------------------------------------------------- #
# 3. Internal against external centres
# --------------------------------------------------------------------------- #

def domain_contrast(truth: pd.Series, baseline: pd.Series, candidate: pd.Series,
                    patients: pd.DataFrame, *, seed: int = 0) -> dict:
    """Difference in differences, on SPECIATION only.

    Specificity cannot appear here. Every uninfected patient in this dataset
    comes from one centre, so a domain contrast on the screening endpoint would
    be comparing that centre with the empty set. The gate in
    `gates.identifiability` says the same thing before any of this runs.
    """
    from .metrics import _aligned
    from .stats import bootstrap

    # a patient with no prediction is refused, not scored wrong. `reindex`
    # inserted NaN, `regroup(nan)` returned nan, and the comparison silently
    # counted the patient as an error in both arms.
    baseline = _aligned(baseline, truth)
    candidate = _aligned(candidate, truth)
    centre = patients.set_index("patient_id")["centre"]
    dom = truth.index.map(lambda p: "internal"
                          if str(centre.get(p, "")) in INTERNAL_CENTRES
                          else "external")
    infected = (truth != UNINFECTED).to_numpy()
    t = truth.map(lambda x: regroup(x, GROUPING_5)).to_numpy()
    b = baseline.map(lambda x: regroup(x, GROUPING_5)).to_numpy()
    c = candidate.map(lambda x: regroup(x, GROUPING_5)).to_numpy()
    dom = np.asarray(dom)

    units = [(dom[i], t[i] == b[i], t[i] == c[i])
             for i in range(len(t)) if infected[i]]

    def did(sample):
        out = {}
        for d in ("internal", "external"):
            rows = [(x, y) for dd, x, y in sample if dd == d]
            if not rows:
                return float("nan")
            out[d] = (np.mean([y for _, y in rows]) -
                      np.mean([x for x, _ in rows]))
        return float(out["internal"] - out["external"])

    ci = bootstrap(units, did, seed=seed)
    return {"difference_in_differences": ci.point, "ci": (ci.lo, ci.hi),
            "n_internal": int(sum(1 for d, _, _ in units if d == "internal")),
            "n_external": int(sum(1 for d, _, _ in units if d == "external")),
            "n_degenerate": ci.n_degenerate,
            "note": ("speciation only: specificity is not estimable by domain "
                     "because every uninfected smear is internal")}


# --------------------------------------------------------------------------- #
# 4. How many fields does a rule need?
# --------------------------------------------------------------------------- #

MIN_FIELD_CURVE_PATIENTS = 20


def _index_by_field(work: pd.DataFrame, ids, declared: dict | None = None) -> dict:
    """patient -> list of per-field {class: score array}, built once.

    The obvious implementation of the curve below masks the whole detection
    table once per replicate. At 100 replicates, five sizes and two rules that
    is a thousand passes over a million rows, and on the real cache it does not
    finish. Everything the rules need is per patient, per field, per class, so
    it is extracted once here and the replicates then concatenate small arrays.
    """
    wanted = set(str(i) for i in ids)
    out: dict[str, list] = {pid: [] for pid in wanted}
    for (pid, img), g in work.groupby(["patient_id", "image_id"], sort=True,
                                      observed=True):
        p = str(pid)
        if p not in wanted:
            continue
        by_cls: dict[str, np.ndarray] = {}
        for cls, gg in g.groupby("cls", sort=False, observed=True):
            by_cls[str(cls)] = gg["score"].to_numpy(dtype=float)
        out[p].append(by_cls)
    # Pad to the DECLARED field count with empty fields. Only fields carrying a
    # surviving detection appear above, so a patient declaring 40 fields of
    # which 3 held a box offered a universe of 3 to sample from: "examine 25
    # fields" returned 3, and the curve was flat by construction. Under the
    # `source` condition, where a 0.25 floor empties many fields, that is most
    # of the cohort.
    if declared:
        for pid, n in declared.items():
            p = str(pid)
            if p in out:
                out[p].extend({} for _ in range(max(int(n) - len(out[p]), 0)))
    return out


def _view_from_fields(pid: str, fields: list, classes, n_fields: int | None = None
                      ) -> "object":
    """A PatientView over exactly the fields handed in.

    `n_fields` is the number EXAMINED, which is the curve's x-axis and is not
    the same as the number that carried a detection. `build_views` sets
    `n_fields` to the declared field count; taking the length of the non-empty
    fields instead made R5 -- whose screening statistic is a share of fields --
    read 1.0 where `build_views` read 0.2, an opposite call on the same data.
    """
    from .evidence import PatientView

    scores, fmax, fcount = {}, {}, {}
    n = len(fields) if n_fields is None else int(n_fields)
    for c in classes:
        parts = [f[c] for f in fields if c in f]
        scores[c] = (np.concatenate(parts) if parts else np.zeros(0))
        fmax[c] = np.array([float(f[c].max()) if c in f and f[c].size else 0.0
                            for f in fields], dtype=float)
        fcount[c] = np.array([int(f[c].size) if c in f else 0 for f in fields],
                             dtype=int)
    return PatientView(patient_id=pid, n_fields=n, scores=scores,
                       per_field_max=fmax, per_field_count=fcount)


def field_curve(det: pd.DataFrame, patients: pd.DataFrame, truth: pd.Series,
                rule_ids, *, condition: str, sizes=(5, 10, 15, 20, 25),
                reps: int = 50, seed: int = 0, min_fields: int = 25) -> dict:
    """Accuracy against the number of fields examined, per rule.

    Restricted to patients with at least `min_fields` so that the same patients
    appear at every point on the curve; a curve whose denominator changes with
    x describes nobody. Reported per rule, because a curve computed only for
    the baseline says nothing about whether other rules plateau earlier.

    Every rule sees the same subsample within a replicate, so a difference
    between two curves is a difference between the rules rather than between
    two draws of fields.
    """
    from .schema import ALL_CLASSES

    eligible = patients[pd.to_numeric(patients["n_images"]) >= min_fields]
    ids = eligible["patient_id"].astype(str).tolist()
    out = {"n_eligible": int(len(eligible)),
           "n_excluded": int(len(patients) - len(eligible)),
           "min_fields_required": int(min_fields), "reps": int(reps),
           "sizes": [int(k) for k in sizes], "condition": str(condition),
           "by_rule": {}}
    if len(eligible) < MIN_FIELD_CURVE_PATIENTS:
        out["note"] = (f"only {len(eligible)} patients contribute {min_fields} "
                       "fields or more; the curve would describe too few "
                       "patients to mean anything and was not computed")
        return out

    work = apply_condition(det, condition)
    declared = {str(r.patient_id): int(r.n_images)
                for r in eligible.itertuples(index=False)}
    by_patient = _index_by_field(work, ids, declared)
    truth_g = {pid: regroup(str(truth[pid]), GROUPING_5) for pid in ids}
    rule_ids = list(rule_ids)
    rng = np.random.default_rng(seed)

    acc = {rid: {int(k): [] for k in sizes} for rid in rule_ids}
    for k in sizes:
        for _ in range(reps):
            views = {}
            for pid in ids:
                fields = by_patient.get(pid) or []
                if len(fields) > k:
                    pick = rng.choice(len(fields), size=int(k), replace=False)
                    chosen = [fields[i] for i in pick]
                else:
                    chosen = fields
                views[pid] = _view_from_fields(pid, chosen, ALL_CLASSES,
                                               n_fields=len(chosen))
            for rid in rule_ids:
                rule = RULES[rid]
                correct = sum(
                    regroup(rule.apply(views[pid])["label"], GROUPING_5)
                    == truth_g[pid] for pid in ids)
                acc[rid][int(k)].append(correct / len(ids))

    for rid in rule_ids:
        out["by_rule"][rid] = {
            k: {"mean": float(np.mean(v)),
                "lo": float(np.quantile(v, 0.025)),
                "hi": float(np.quantile(v, 0.975))}
            for k, v in acc[rid].items()}
    return out
