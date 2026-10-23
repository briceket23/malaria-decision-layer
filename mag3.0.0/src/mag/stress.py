"""Do the rules keep their order when the fields stop being enriched?

Protocol v3 §13. Three independent reviews asked for one additional experiment
between them, and this is it -- worth more than any additional rule.

**The problem.** Every field from an infected patient in this archive was
selected to contain at least one parasite. That is not a mild optimism. It is a
sampling regime screening does not provide, and the rules respond to it
differently by construction: an accumulating rule sums evidence without bound
over fields that all carry some, while a counting rule needs a threshold number
of confident detections and a field-normalised rule divides by the fields
examined. The concern is therefore NOT that every performance figure is
inflated. It is that **the ORDERING of the rules may be an artefact of the
enrichment**, and an ordering is what the paper's second contribution rests on.

**The design.** Hold the detector cache fixed -- nothing is re-inferred and no
detection is invented. For each infected patient, rebuild the smear at a
declared positive-field rate pi by keeping a pi-share of that patient's own
fields and replacing the remainder with parasite-free fields drawn from
uninfected patients. The field COUNT is held constant so that a rule normalised
by fields examined is comparable across pi. Uninfected patients are untouched:
their fields are already parasite-free, and diluting them would mean nothing.

**What is reported.** Each rule's screening sensitivity and specificity against
pi, and the question the analysis exists to answer: whether the ordering of the
rules is stable across pi, by Kendall's tau against the ordering at pi = 1.

**What it is not.** A sensitivity analysis under a stated resampling model, not
an estimate of screening performance. The negative fields come from a DIFFERENT
patient, so this measures the rules' response to evidence dilution and not the
full screening problem -- a real screening smear's negative fields come from the
same slide, the same stain and the same camera as its positive ones.

One consequence of this archive's structure has to be said out loud. Every
uninfected patient comes from one centre, so for an infected patient from
anywhere else the donor fields necessarily come from a different centre and the
dilution carries a domain shift with it. `donor_mismatch` counts exactly how
many patients that affects, and it is reported beside every number here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .evidence import apply_condition
from .schema import ALL_CLASSES, UNINFECTED

#: The positive-field rates swept. 1.0 is the archive as released, and is the
#: reference the ordering is compared against.
RATES = (1.0, 0.5, 0.25, 0.1)

#: Below this many donor fields the dilution cannot be built without reusing
#: the same field many times, which would make the result a statement about one
#: field rather than about dilution.
MIN_DONOR_FIELDS = 20


@dataclass
class Stress:
    rates: tuple = ()
    #: rule -> rate -> {sensitivity, specificity, accuracy, ...}
    by_rule: dict = field(default_factory=dict)
    ordering: dict = field(default_factory=dict)
    donor_mismatch: dict = field(default_factory=dict)
    n_donor_fields: int = 0
    reps: int = 0
    note: str = ""
    sentence: str = ""


def _donor_pool(work: pd.DataFrame, patients: pd.DataFrame) -> dict:
    """Parasite-free fields, indexed by the centre that supplied them.

    A field is a donor only if it belongs to an uninfected patient. Their
    detections are kept exactly as the detector produced them -- including any
    spurious parasite boxes, which is the point: a rule that survives dilution
    only because the donor fields are pristine has not survived anything.
    """
    negatives = patients[patients["truth"] == UNINFECTED]
    keep = set(negatives["patient_id"].astype(str))
    centre_of = dict(zip(negatives["patient_id"].astype(str),
                         negatives["centre"].astype(str)))
    pool: dict = {}
    for (pid, img), g in work.groupby(["patient_id", "image_id"], sort=True,
                                      observed=True):
        p = str(pid)
        if p not in keep:
            continue
        by_cls = {str(c): gg["score"].to_numpy(dtype=float)
                  for c, gg in g.groupby("cls", sort=False, observed=True)}
        pool.setdefault(centre_of.get(p, ""), []).append(by_cls)
    return pool


def run(det: pd.DataFrame, patients: pd.DataFrame, truth: pd.Series, rule_ids,
        *, condition: str = "matched", rates=RATES, reps: int = 30,
        seed: int = 0, grouping=None) -> Stress:
    """Sweep the positive-field rate and report whether the ordering holds."""
    from .diagnostics import _index_by_field, _view_from_fields
    from .metrics import screening
    from .rules import RULES

    out = Stress(rates=tuple(float(r) for r in rates), reps=int(reps))
    work = apply_condition(det, condition)
    ids = patients["patient_id"].astype(str).tolist()
    declared = {str(r.patient_id): int(r.n_images)
                for r in patients.itertuples(index=False)}
    fields = _index_by_field(work, ids, declared)

    pool = _donor_pool(work, patients)
    n_donor = sum(len(v) for v in pool.values())
    out.n_donor_fields = n_donor
    if n_donor < MIN_DONOR_FIELDS:
        out.note = (f"only {n_donor} parasite-free donor fields are available; "
                    f"below {MIN_DONOR_FIELDS} the dilution would reuse the "
                    "same field repeatedly and would describe that field "
                    "rather than dilution. Not computed.")
        return out

    centre_of = dict(zip(patients["patient_id"].astype(str),
                         patients["centre"].astype(str)))
    infected = [p for p in ids if str(truth[p]) != UNINFECTED]
    mismatched = [p for p in infected if centre_of.get(p, "") not in pool]
    out.donor_mismatch = {
        "n_infected": len(infected),
        "n_with_no_same_centre_donor": len(mismatched),
        "share": len(mismatched) / max(len(infected), 1),
        "donor_centres": sorted(pool),
        "note": ("every uninfected patient in this archive comes from one "
                 "centre, so an infected patient from anywhere else is diluted "
                 "with fields from a different centre and the dilution carries "
                 "a domain shift. This count is the size of that problem."),
    }

    rng = np.random.default_rng(seed)
    per_rule: dict = {rid: {} for rid in rule_ids}

    for pi in out.rates:
        acc: dict = {rid: [] for rid in rule_ids}
        for _ in range(reps if pi < 1.0 else 1):
            views = {}
            for pid in ids:
                own = fields.get(pid) or []
                n = len(own)
                if str(truth[pid]) == UNINFECTED or n == 0 or pi >= 1.0:
                    views[pid] = _view_from_fields(pid, own, ALL_CLASSES,
                                                   n_fields=max(n, 1))
                    continue
                keep_n = int(round(pi * n))
                donors = pool.get(centre_of.get(pid, "")) or [
                    f for v in pool.values() for f in v]
                pick_own = rng.choice(n, size=keep_n, replace=False) if keep_n else []
                pick_don = rng.choice(len(donors), size=n - keep_n,
                                      replace=len(donors) < (n - keep_n))
                chosen = ([own[i] for i in pick_own]
                          + [donors[j] for j in pick_don])
                views[pid] = _view_from_fields(pid, chosen, ALL_CLASSES,
                                               n_fields=n)
            for rid in rule_ids:
                rule = RULES[rid]
                pred = pd.Series({p: rule.apply(views[p])["label"] for p in ids})
                sc = screening(truth.reindex(ids), pred)
                acc[rid].append((sc["sensitivity"], sc["specificity"]))

        for rid in rule_ids:
            arr = np.asarray(acc[rid], dtype=float)
            per_rule[rid][pi] = {
                "sensitivity": float(arr[:, 0].mean()),
                "sensitivity_sd": float(arr[:, 0].std()) if len(arr) > 1 else 0.0,
                "specificity": float(arr[:, 1].mean()),
                "specificity_sd": float(arr[:, 1].std()) if len(arr) > 1 else 0.0,
                "reps": int(len(arr)),
            }

    out.by_rule = {rid: {f"{k:g}": v for k, v in d.items()}
                   for rid, d in per_rule.items()}
    out.ordering = _ordering(per_rule, out.rates)
    out.sentence = _sentence(out)
    return out


def _ordering(per_rule: dict, rates) -> dict:
    """Is the ranking of the rules the same after dilution as before it?

    Kendall's tau between the ordering at each rate and the ordering at the
    reference rate, on sensitivity -- the endpoint dilution actually attacks.
    A tau near 1 means the paper's ordering claim survives; a tau near or below
    zero means the ordering was a property of the enrichment.
    """
    ref = max(rates)
    rules = sorted(per_rule)
    base = np.array([per_rule[r][ref]["sensitivity"] for r in rules])
    out = {"reference_rate": float(ref), "rules": rules, "by_rate": {}}
    for pi in rates:
        here = np.array([per_rule[r][pi]["sensitivity"] for r in rules])
        tau, n_pairs, discordant = _kendall(base, here)
        out["by_rate"][f"{pi:g}"] = {
            "kendall_tau": tau, "n_pairs": n_pairs,
            "n_discordant_pairs": discordant,
            "top_rule": rules[int(np.argmax(here))],
        }
    taus = [v["kendall_tau"] for k, v in out["by_rate"].items()
            if float(k) < ref and np.isfinite(v["kendall_tau"])]
    out["min_tau"] = float(min(taus)) if taus else float("nan")
    out["stable"] = bool(taus and min(taus) >= 0.8)
    out["inverted"] = bool(taus and min(taus) < 0.0)
    tops = {v["top_rule"] for v in out["by_rate"].values()}
    out["top_rule_changes"] = bool(len(tops) > 1)
    out["top_rules_seen"] = sorted(tops)
    return out


def _kendall(a: np.ndarray, b: np.ndarray) -> tuple:
    """Kendall's tau-b between two rankings, written out.

    Ties are common here -- several rules reach the same sensitivity -- and a
    tau-a would report an ordering change that is really a tie, so the tie
    correction is not optional.
    """
    n = len(a)
    conc = disc = tie_a_only = tie_b_only = tie_both = 0
    for i in range(n):
        for j in range(i + 1, n):
            da, db = a[i] - a[j], b[i] - b[j]
            if da == 0 and db == 0:
                tie_both += 1
            elif da == 0:
                tie_a_only += 1
            elif db == 0:
                tie_b_only += 1
            elif da * db > 0:
                conc += 1
            else:
                disc += 1
    # tau-b = (C - D) / sqrt((n0 - n1)(n0 - n2)), where n0 is every pair, n1
    # the pairs tied in a and n2 the pairs tied in b. The first version of this
    # function wrote `conc + disc + tie_a` for the first factor, which is
    # n0 - n1 only when there are no ties at all: with ties it ADDS them where
    # the definition subtracts, inflating the denominator and pulling tau
    # toward zero. On two identical rankings containing one tie it returned
    # 0.67 instead of 1.0 -- and since `stable` is decided at tau >= 0.8, that
    # would have reported a stable ordering as unstable. Ties are the normal
    # case here, several rules reaching the same sensitivity, so the error was
    # not rare.
    n0 = n * (n - 1) // 2
    n1 = tie_a_only + tie_both
    n2 = tie_b_only + tie_both
    denom = np.sqrt(float(n0 - n1) * float(n0 - n2))
    tau = float((conc - disc) / denom) if denom > 0 else float("nan")
    return tau, int(conc + disc), int(disc)


def _sentence(s: Stress) -> str:
    o = s.ordering
    if not o:
        return s.note
    lowest = min(float(k) for k in o["by_rate"])
    verdict = (
        "the ordering of the rules was preserved" if o["stable"] else
        "the ordering of the rules INVERTED" if o["inverted"] else
        "the ordering of the rules was only partly preserved")
    return (
        f"Diluting each infected patient's fields with parasite-free fields "
        f"down to a positive-field rate of {lowest:g}, holding the field count "
        f"and the detector cache fixed, {verdict} "
        f"(minimum Kendall tau {o['min_tau']:.2f} against the undiluted "
        f"ordering)"
        + (f"; the best-performing rule changed across rates "
           f"({', '.join(o['top_rules_seen'])})" if o["top_rule_changes"]
           else "; the best-performing rule was unchanged")
        + ". This is a sensitivity analysis under a stated resampling model, "
          "not an estimate of screening performance: the donor fields come "
          "from a different patient, and for "
        + f"{s.donor_mismatch.get('n_with_no_same_centre_donor', 0)} of "
        + f"{s.donor_mismatch.get('n_infected', 0)} infected patients from a "
          "different centre.")
