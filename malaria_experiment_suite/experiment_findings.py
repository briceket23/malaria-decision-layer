#!/usr/bin/env python3
"""experiment_findings.py --- what EDA and verification establish, as code.

Twelve things are now known about this archive and these checkpoints. Each was
measured, not assumed, and each constrains what the experiment may claim. Left
as prose in a chat log they decay: the next person runs the pipeline, gets
numbers, and writes them up without knowing that specificity is not
identifiable or that one checkpoint's cache is truncated.

So each is a `Finding` here, with

  * `check(eda, verify)` --- re-derives the fact from the two JSON artefacts,
    reading **only through `exp_schema`**. No key name is spelled in this file.
    That is not tidiness: both bugs this suite has had were a consumer looking
    for a key a producer never wrote, and the schema plus one test in
    `test_experiment.py` makes that a failing test instead of an UNKNOWN on
    someone's archive.
  * `severity` --- how loudly it is reported:
        BLOCK    the run must not proceed as if this were settled
        DECLARE  the manuscript MUST state it
        CAVEAT   the manuscript SHOULD state it
    None of them stops the run. A check that halts the pipeline gets disabled;
    a check that reports gets read.
  * `manuscript` --- the sentence the paper owes the reader. This is what turns
    a diagnostic into a finished section.

`ledger()` produces the whole table. `run_experiment.py` prints it, writes it
to `results/findings.csv`, and carries on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_schema as S                                        # noqa: E402
from exp_schema import EDA, VERIFY, MISSING                   # noqa: E402

BLOCK, DECLARE, CAVEAT = "BLOCK", "DECLARE", "CAVEAT"
SEVERITY_ORDER = {BLOCK: 0, DECLARE: 1, CAVEAT: 2}

#: `gates.DECLARED_CHECKPOINT`. Named because two findings turn on the
#: pre-registered primary being this particular architecture.
DECLARED_CHECKPOINT = "rtdetr"

#: A patient-level truth needs more than a couple of smears behind it before a
#: per-class sensitivity means anything. Three is not a rate.
MIN_SMEARS_FOR_CLASS_CLAIM = 5

#: How far measured retention may sit from a pure crop's prediction before the
#: crop stops being an adequate model. Declared, not discovered: roughly the
#: slack that box CLIPPING at the new boundary accounts for on its own.
CROP_MODEL_TOLERANCE = 0.03

#: Below this, two checkpoints' confidence distributions are close enough that
#: a rule reading raw scores is not mostly reading a calibration convention.
SCORE_CONVENTION_RATIO = 2.0


@dataclass
class Outcome:
    """What a check found. `fired` means the condition is present."""
    fired: bool
    detail: str = ""
    data: dict = field(default_factory=dict)
    unknown: bool = False        # the input needed was not in the JSON


def _miss(what: str) -> Outcome:
    return Outcome(False, f"not checked: {what} absent", unknown=True)


@dataclass
class Finding:
    key: str
    severity: str
    title: str
    check: Callable
    manuscript: str
    where: str = ""
    reads: tuple = ()            # the schema Keys this check consults

    def evaluate(self, eda: dict, verify: dict) -> dict:
        try:
            o = self.check(eda or {}, verify or {})
        except Exception as exc:                              # noqa: BLE001
            o = Outcome(False, f"check raised {type(exc).__name__}: {exc}",
                        unknown=True)
        return {"key": self.key, "severity": self.severity, "title": self.title,
                "fired": bool(o.fired), "unknown": bool(o.unknown),
                "detail": o.detail, "data": o.data,
                "manuscript": self.manuscript, "where": self.where}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

NAMED_RATIOS = ((16 / 9, "16:9"), (4 / 3, "4:3"), (3 / 2, "3:2"),
                (5 / 4, "5:4"), (1.0, "1:1"))


def name_ratio(k) -> str:
    """`1.778` -> `16:9`. The file stores the number; humans read the name."""
    try:
        v = float(k)
    except (TypeError, ValueError):
        return str(k)
    for target, nm in NAMED_RATIOS:
        if abs(v - target) < 0.02:
            return nm
    return f"{v:.3f}"


def _num(x, d=None):
    return d if x is MISSING or x is None else float(x)


# --------------------------------------------------------------------------- #
# The checks
# --------------------------------------------------------------------------- #

def chk_cap_saturation(eda, verify) -> Outcome:
    hits = dict(VERIFY.SMOKE_FIELDS_AT_CAP.each(verify))
    if not hits:
        return _miss("verify.json smoke block")
    fired = {k: v for k, v in hits.items() if v}
    if not fired:
        return Outcome(False, "no checkpoint reached the detection cap")
    n_img = dict(VERIFY.SMOKE_N_IMAGES.each(verify))
    caps = dict(VERIFY.OP_MAX_DET.each(verify))
    sug = dict(VERIFY.PROBE_SUGGESTED.each(verify))
    thr = dict(VERIFY.SMOKE_CENSORING.each(verify))
    parts = [f"{k} {v}/{n_img.get(k, '?')} fields at max_det="
             f"{caps.get(k, '?')}" for k, v in sorted(fired.items())]
    return Outcome(True, "; ".join(parts),
                   {"fields_at_cap": fired, "n_images": n_img,
                    "max_det": caps, "suggested_max_det": sug,
                    "censoring_threshold": thr})


def chk_density_bias(eda, verify) -> Outcome:
    # "no probe ran" means two different things and they must not collapse:
    # no checkpoint saturated (a genuine clear) or verify.json is absent (not
    # checked). Distinguishing them is the whole point of the UNKNOWN state.
    if not VERIFY.SMOKE_FIELDS_AT_CAP.names(verify):
        return _miss("verify.json smoke block")
    bias = dict(VERIFY.PROBE_DENSITY_BIAS.each(verify))
    if not bias:
        return Outcome(False, "no checkpoint saturated the cap, so no probe "
                              "ran and there is no density bias to report")
    worst = max(bias, key=lambda k: abs(bias[k]))
    lost_rbc = dict(VERIFY.PROBE_LOST_RBC.each(verify)).get(worst, "?")
    lost_par = dict(VERIFY.PROBE_LOST_PAR.each(verify)).get(worst, "?")
    material = dict(VERIFY.PROBE_MATERIAL.each(verify)).get(worst)
    return Outcome(abs(bias[worst]) > 0,
                   f"{worst}: R6/R6b density {bias[worst]:+.1%} on capped "
                   f"fields ({lost_rbc} RBC lost from the denominator, "
                   f"{lost_par} parasite from the numerator)"
                   + ("  --- material" if material else ""),
                   {"density_bias": bias, "worst": worst,
                    "material": material,
                    "discarded": dict(VERIFY.PROBE_DISCARDED.each(verify)),
                    "discarded_share":
                        dict(VERIFY.PROBE_DISCARDED_SHARE.each(verify))})


def chk_identifiability(eda, verify) -> Outcome:
    sep = EDA.IDENT_SEPARABLE.get(eda)
    centres = EDA.IDENT_CENTRES.get(eda)
    if sep is MISSING and centres is MISSING:
        return _miss("eda.json identifiability block")
    centres = [] if centres is MISSING else list(centres)
    if sep is True or len(centres) > 1:
        return Outcome(False, f"uninfected span {len(centres)} centres "
                              f"({', '.join(centres)})")
    n = EDA.IDENT_N_UNINFECTED.get(eda, "?")
    return Outcome(True,
                   f"all {n} uninfected smears come from "
                   f"{centres[0] if centres else 'one centre'}; specificity "
                   f"and site are perfectly collinear",
                   {"n_uninfected": n, "centres": centres,
                    "prevalence": _num(EDA.IDENT_PREVALENCE.get(eda))})


def chk_enrichment(eda, verify) -> Outcome:
    share = _num(EDA.LOAD_ENRICHMENT.get(eda))
    if share is None:
        return _miss("eda.json parasite_load enrichment")
    return Outcome(share > 0.5,
                   f"{share:.1%} of an infected patient's fields carry a "
                   f"parasite box",
                   {"share": share,
                    "median_per_smear": _num(EDA.LOAD_MEDIAN.get(eda)),
                    "over_smears": EDA.LOAD_COVERAGE.get(eda, None),
                    "of_total": EDA.LOAD_TOTAL.get(eda, None)})


def chk_rising_class(eda, verify) -> Outcome:
    rising = EDA.TF_RISING.get(eda)
    if rising is MISSING:
        return _miss("eda.json classes_rising_after_transform")
    if not rising:
        return Outcome(False, "no class gains boxes after the transform")
    return Outcome(True,
                   "; ".join(f"{k} {v.get('base')}->{v.get('zoom')} "
                             f"({v.get('kept'):.3f}x)"
                             for k, v in sorted(rising.items())),
                   {"classes": rising})


def chk_crop_model(eda, verify) -> Outcome:
    """Does a pure 4:3 -> 16:9 crop actually explain the transform?

    The geometry gives a PREDICTION --- a crop keeping the central 75% of the
    height removes every box with |y - 0.5| > 0.375, so retention should be
    one minus the margin share, per kind. `transform_retention` gives the
    MEASURED retention. Two independent numbers about the same operation.

    Clipping is ONE-DIRECTIONAL and that is what makes this decidable. The
    margin test is on box CENTRES, so a box whose centre falls outside the
    boundary may still be clipped and kept --- which can only push measured
    retention ABOVE the prediction, never below. A single positive gap is
    therefore consistent with clipping and proves nothing. Gaps of OPPOSITE
    sign cannot both be clipping.
    """
    pp = _num(EDA.GEO_PRED_PAR.get(eda))
    po = _num(EDA.GEO_PRED_OTH.get(eda))
    mp = _num(EDA.TF_PARASITE_KEPT.get(eda))
    mo = _num(EDA.TF_OTHER_KEPT.get(eda))
    if pp is None or mp is None:
        return _miss("predicted or measured retention")

    gaps = {"parasite": mp - pp}
    bits = [f"parasite: crop predicts {pp:.1%}, measured {mp:.1%} "
            f"({gaps['parasite']:+.1%})"]
    if po is not None and mo is not None:
        gaps["other"] = mo - po
        bits.append(f"blood: crop predicts {po:.1%}, measured {mo:.1%} "
                    f"({gaps['other']:+.1%})")

    signs = {k: (g > 0) for k, g in gaps.items() if abs(g) > 1e-9}
    opposite = len(set(signs.values())) > 1
    worst = max(gaps, key=lambda k: abs(gaps[k]))
    return Outcome(opposite or abs(gaps[worst]) > CROP_MODEL_TOLERANCE,
                   "; ".join(bits)
                   + ("  --- OPPOSITE SIGNS: clipping cannot explain both"
                      if opposite else ""),
                   {"gaps": gaps, "tolerance": CROP_MODEL_TOLERANCE,
                    "worst": worst, "opposite_signs": opposite})


def chk_mixed_aspect(eda, verify) -> Outcome:
    ratios = EDA.IMG_RATIOS.get(eda)
    if ratios is MISSING:
        return _miss("eda.json images block")
    if len(ratios) < 2:
        return Outcome(False, "one aspect ratio in the sample: "
                              + (", ".join(name_ratio(k) for k in ratios)
                                 or "none recorded"))
    return Outcome(True,
                   f"{len(ratios)} distinct aspect ratios: "
                   + ", ".join(f"{name_ratio(k)} ({v})" for k, v in
                               sorted(ratios.items(), key=lambda kv: -kv[1])),
                   {"aspect_ratios": ratios,
                    "sampled": EDA.IMG_SAMPLED.get(eda, None),
                    "sizes": EDA.IMG_SIZES.get(eda, None)})


def chk_rare_classes(eda, verify) -> Outcome:
    tbl = EDA.COHORT_TABLE.get(eda)
    if tbl is MISSING:
        return _miss("eda.json cohort_table")
    totals: dict = {}
    for centre, row in tbl.items():
        if str(centre).lower() == "total":
            continue
        for cls, n in (row or {}).items():
            if str(cls).lower() in ("total", "uninfected"):
                continue
            totals[cls] = totals.get(cls, 0) + int(n or 0)
    thin = {k: v for k, v in totals.items()
            if 0 < v < MIN_SMEARS_FOR_CLASS_CLAIM}
    if not thin:
        return Outcome(False, "every present class has at least "
                              f"{MIN_SMEARS_FOR_CLASS_CLAIM} smears")
    return Outcome(True,
                   ", ".join(f"{k} {v} smear{'s' if v != 1 else ''}"
                             for k, v in sorted(thin.items(),
                                                key=lambda kv: kv[1])),
                   {"thin": thin, "threshold": MIN_SMEARS_FOR_CLASS_CLAIM,
                    "totals": totals})


def chk_fixed_budget(eda, verify) -> Outcome:
    lo = dict(VERIFY.SMOKE_PER_FIELD_MIN.each(verify))
    hi = dict(VERIFY.SMOKE_PER_FIELD_MAX.each(verify))
    caps = dict(VERIFY.OP_MAX_DET.each(verify))
    if not lo:
        return _miss("verify.json per-field counts")
    # a constant that EQUALS the cap is saturation, not a query budget
    fixed = {k: lo[k] for k in lo
             if lo[k] and lo[k] == hi.get(k) and lo[k] != caps.get(k)}
    if not fixed:
        return Outcome(False, "no checkpoint emitted a constant per-field "
                              "count below the cap")
    n_img = dict(VERIFY.SMOKE_N_IMAGES.each(verify))
    bits = [f"{k} exactly {v} boxes on all {n_img.get(k, '?')} fields"
            + (" (the PRE-REGISTERED primary)" if k == DECLARED_CHECKPOINT
               else "")
            for k, v in sorted(fixed.items())]
    return Outcome(True, "; ".join(bits),
                   {"fixed": fixed,
                    "declared_is_fixed": DECLARED_CHECKPOINT in fixed})


def chk_score_convention(eda, verify) -> Outcome:
    med = {k: v for k, v in VERIFY.SMOKE_SCORE_MEDIAN.each(verify)
           if v is not None}
    if len(med) < 2:
        return _miss("two or more score medians")
    lo, hi = min(med.values()), max(med.values())
    if lo <= 0 or hi / lo < SCORE_CONVENTION_RATIO:
        return Outcome(False, "score medians agree within "
                              f"{SCORE_CONVENTION_RATIO:g}x: "
                              + ", ".join(f"{k} {v:.4f}"
                                          for k, v in sorted(med.items())))
    return Outcome(True,
                   f"median confidence differs by {hi / lo:.1f}x across "
                   f"checkpoints: "
                   + ", ".join(f"{k} {v:.4f}" for k, v in sorted(med.items())),
                   {"score_median": med, "ratio": hi / lo})


def chk_imbalance(eda, verify) -> Outcome:
    per = EDA.LAB_PER_CLASS.get(eda)
    if per is MISSING:
        return _miss("eda.json labels_base per_class")
    counts = {k: int(v) for k, v in per.items() if int(v or 0) > 0}
    if len(counts) < 2:
        return _miss("per-class annotation counts")
    top, bot = max(counts, key=counts.get), min(counts, key=counts.get)
    ratio = counts[top] / counts[bot]
    return Outcome(ratio > 100,
                   f"{ratio:,.1f}:1 ({top} {counts[top]:,} to "
                   f"{bot} {counts[bot]:,})",
                   {"ratio": ratio, "max_class": top, "min_class": bot,
                    "counts": counts})


def chk_geometry(eda, verify) -> Outcome:
    p = _num(EDA.GEO_PAR_RADIUS.get(eda))
    b = _num(EDA.GEO_OTH_RADIUS.get(eda))
    if p is None or b is None:
        return _miss("eda.json centre_distance radii")
    pm = _num(EDA.GEO_PAR_MARGIN.get(eda), 0.0)
    om = _num(EDA.GEO_OTH_MARGIN.get(eda), 0.0)
    return Outcome(p < b,
                   f"parasites sit nearer the field centre: median radius "
                   f"{p:.4f} vs {b:.4f}; {pm:.1%} vs {om:.1%} in the crop "
                   f"margin",
                   {"parasite_median_radius": p, "other_median_radius": b,
                    "parasite_margin_share": pm, "other_margin_share": om})


def chk_partition_overlap(eda, verify) -> Outcome:
    """A smear in two partitions makes the test estimate in-sample."""
    probs = EDA.PART_PROBLEMS.get(eda)
    if probs is MISSING:
        return _miss("eda.json partitions block")
    if not probs:
        return Outcome(False, "every partition pair is disjoint at the smear "
                              "level")
    return Outcome(True, "; ".join(probs), {"problems": list(probs)})


def chk_operating_point(eda, verify) -> Outcome:
    """Was the smoke test run at mag's operating point, or ultralytics'?

    A verification run at `iou=0.7, max_det=300` measures a different detector
    from the one the pass will cache, and reports plausible-looking numbers
    while doing it. This is the check on the checker.
    """
    iou = dict(VERIFY.OP_IOU.each(verify))
    md = dict(VERIFY.OP_MAX_DET.each(verify))
    mag_cap = VERIFY.MAG_MAX_DET.get(verify)
    if not iou or mag_cap is MISSING:
        return _miss("verify.json operating_point / mag_max_det")
    bad = {k: (v, md.get(k)) for k, v in iou.items()
           if abs(float(v) - 1.0) > 1e-9 or md.get(k) != mag_cap}
    if not bad:
        return Outcome(False, f"every checkpoint was driven at iou=1.0, "
                              f"max_det={mag_cap:,}, matching mag")
    return Outcome(True,
                   "; ".join(f"{k}: iou={v[0]}, max_det={v[1]} against mag's "
                             f"iou=1.0, max_det={mag_cap}"
                             for k, v in sorted(bad.items())),
                   {"mismatched": bad, "mag_max_det": mag_cap})


# --------------------------------------------------------------------------- #

FINDINGS = (
    Finding(
        "cap_saturation", BLOCK,
        "a checkpoint saturates the detection cap",
        chk_cap_saturation,
        "State mag.detect.MAX_DET as a declared property of the detection "
        "stage, give its value, and report that the pass ran at a cap no "
        "field reached. If any field did reach it, the cache for that "
        "checkpoint is censored and the counting and density rules read a "
        "truncated numerator.",
        "Methods / detection",
        reads=(VERIFY.SMOKE_FIELDS_AT_CAP, VERIFY.SMOKE_N_IMAGES,
               VERIFY.OP_MAX_DET, VERIFY.PROBE_SUGGESTED,
               VERIFY.SMOKE_CENSORING)),
    Finding(
        "density_bias", DECLARE,
        "the cap moves R6/R6b density",
        chk_density_bias,
        "Report the measured density bias and the sensitivity analysis that "
        "re-ran the saturating checkpoint at a non-binding cap. Density is "
        "parasite boxes over RBC boxes with no score filter on either, so a "
        "score-ranked truncation moves the ratio rather than scaling it.",
        "Results / sensitivity",
        reads=(VERIFY.PROBE_DENSITY_BIAS, VERIFY.PROBE_LOST_RBC,
               VERIFY.PROBE_LOST_PAR, VERIFY.PROBE_MATERIAL,
               VERIFY.PROBE_DISCARDED, VERIFY.PROBE_DISCARDED_SHARE)),
    Finding(
        "identifiability", BLOCK,
        "specificity is not separable from centre (Proposition 5)",
        chk_identifiability,
        "State plainly that every uninfected smear comes from a single "
        "centre, so specificity and site are perfectly collinear. This is "
        "RANK DEFICIENCY, not low power: no sample size fixes it and no "
        "estimator recovers the separated effects. Report no "
        "domain-stratified specificity, in any rotation of the design.",
        "Methods / estimands; Limitations",
        reads=(EDA.IDENT_SEPARABLE, EDA.IDENT_CENTRES, EDA.IDENT_N_UNINFECTED,
               EDA.IDENT_PREVALENCE)),
    Finding(
        "partition_overlap", BLOCK,
        "a smear appears in two partitions",
        chk_partition_overlap,
        "If this fires, the test estimate is in-sample and nothing downstream "
        "can detect it --- not a gate, not an interval, not a simulation. "
        "Name the smears and either exclude them or report the whole "
        "evaluation as contaminated.",
        "Methods / data",
        reads=(EDA.PART_PROBLEMS,)),
    Finding(
        "operating_point", BLOCK,
        "verification did not run at mag's operating point",
        chk_operating_point,
        "If this fires the verification measured a different detector from "
        "the one the pass caches, and its per-field counts are not evidence "
        "about the run. Re-run the check before citing any of its numbers.",
        "Methods / detection",
        reads=(VERIFY.OP_IOU, VERIFY.OP_MAX_DET, VERIFY.MAG_MAX_DET)),
    Finding(
        "enrichment", DECLARE,
        "fields are parasite-enriched",
        chk_enrichment,
        "State the share of infected patients' fields carrying a parasite box "
        "and that every accuracy reported here is therefore an optimistic "
        "upper bound. The bias is UNEVEN across the rule grid --- counting "
        "and density rules see an inflated, low-variance signal that would "
        "sit near zero under random field sampling --- so the RANKING of "
        "rules is at risk, not only the levels. Cite the dilution stress "
        "analysis.",
        "Limitations; Results / stress",
        reads=(EDA.LOAD_ENRICHMENT, EDA.LOAD_MEDIAN, EDA.LOAD_COVERAGE,
               EDA.LOAD_TOTAL)),
    Finding(
        "rising_class", DECLARE,
        "a class gains boxes after the transform",
        chk_rising_class,
        "A crop removes boxes; it cannot add one. Report that these smears "
        "were RE-ANNOTATED rather than merely re-cropped, and state it as a "
        "property of the released label trees --- not as a discrepancy "
        "between the source publication and its data.",
        "Data / transform",
        reads=(EDA.TF_RISING,)),
    Finding(
        "crop_model", DECLARE,
        "a pure crop does not account for the measured retention",
        chk_crop_model,
        "The geometry predicts retention under a 4:3 -> 16:9 crop; the label "
        "trees give the retention that actually occurred. Report both and the "
        "gap per kind. State the clipping caveat --- the margin test is on "
        "box centres, so a box straddling the boundary may be clipped and "
        "kept, which inflates measured retention on its own. That effect runs "
        "in ONE direction, so a single positive gap settles nothing. Gaps of "
        "opposite sign do: clipping cannot simultaneously keep boxes the crop "
        "would remove and remove boxes it would keep. Where they are "
        "opposite, say so --- it is the conclusion the rising class forces, "
        "reached independently, from the whole annotation set rather than one "
        "species.",
        "Data / transform",
        reads=(EDA.GEO_PRED_PAR, EDA.GEO_PRED_OTH, EDA.TF_PARASITE_KEPT,
               EDA.TF_OTHER_KEPT)),
    Finding(
        "mixed_aspect", CAVEAT,
        "the archive carries more than one aspect ratio",
        chk_mixed_aspect,
        "Report the distinct image dimensions found. A single geometric "
        "transform is unlikely to describe all contributing centres, which "
        "bounds how far the crop analysis generalises.",
        "Data / transform",
        reads=(EDA.IMG_RATIOS, EDA.IMG_SAMPLED, EDA.IMG_SIZES)),
    Finding(
        "rare_classes", BLOCK,
        "some species are carried by too few smears for a per-class claim",
        chk_rare_classes,
        "Name the classes with fewer smears than the pre-specified minimum "
        "and report no per-class sensitivity for them. A rate over three "
        "patients is not a rate; presenting one invites a reader to compare "
        "it with the others as though it were.",
        "Results / speciation; Limitations",
        reads=(EDA.COHORT_TABLE,)),
    Finding(
        "fixed_budget", DECLARE,
        "a checkpoint emits a query budget, not a detection count",
        chk_fixed_budget,
        "Report that this checkpoint returns the same number of boxes on a "
        "parasite-dense field and a clean one, because a detection "
        "transformer's decoder emits a fixed query budget. Any statistic that "
        "counts, averages or thresholds boxes measures that output convention "
        "before it measures the detector. Where the pre-registered primary "
        "checkpoint is the one with the fixed budget, say so.",
        "Methods / detectors; Results / box census",
        reads=(VERIFY.SMOKE_PER_FIELD_MIN, VERIFY.SMOKE_PER_FIELD_MAX,
               VERIFY.OP_MAX_DET, VERIFY.SMOKE_N_IMAGES)),
    Finding(
        "score_convention", DECLARE,
        "confidence distributions are not comparable across checkpoints",
        chk_score_convention,
        "Report the median confidence per checkpoint. Rules that read raw "
        "scores as evidence are reading a calibration convention; this is the "
        "argument for the fitted rule family and for restricting checkpoint "
        "comparisons to the matched condition.",
        "Methods / conditions; Results / calibration",
        reads=(VERIFY.SMOKE_SCORE_MEDIAN,)),
    Finding(
        "imbalance", CAVEAT,
        "extreme class imbalance in the annotations",
        chk_imbalance,
        "Report the imbalance ratio. It bounds what a per-class metric can "
        "resolve and motivates reporting patient-level rather than box-level "
        "performance.",
        "Data / annotations",
        reads=(EDA.LAB_PER_CLASS,)),
    Finding(
        "geometry", CAVEAT,
        "parasites sit nearer the field centre",
        chk_geometry,
        "Report the radial distributions. They support the manuscript's "
        "explanation for differential retention under the crop directly, "
        "rather than leaving it inferred from two totals.",
        "Data / transform",
        reads=(EDA.GEO_PAR_RADIUS, EDA.GEO_OTH_RADIUS, EDA.GEO_PAR_MARGIN,
               EDA.GEO_OTH_MARGIN)),
)


def ledger(eda: dict, verify: dict) -> list:
    """Every finding, evaluated against the two JSON artefacts."""
    rows = [f.evaluate(eda, verify) for f in FINDINGS]
    rows.sort(key=lambda r: (not r["fired"], SEVERITY_ORDER[r["severity"]],
                             r["key"]))
    return rows


def fired(rows: list) -> list:
    return [r for r in rows if r["fired"]]


def blocking(rows: list) -> list:
    return [r for r in rows if r["fired"] and r["severity"] == BLOCK]


def unknown(rows: list) -> list:
    """Rows whose input was missing --- neither fired nor cleared."""
    return [r for r in rows if r["unknown"]]


def keys_read() -> set:
    """Every schema path any check consults. The drift test uses this."""
    return {k.path for f in FINDINGS for k in f.reads}


def main() -> int:
    """`python experiment_findings.py` prints what each finding reads."""
    for f in FINDINGS:
        print(f"{f.severity:<8} {f.key}")
        print(f"         {f.title}")
        for k in f.reads:
            print(f"           reads  {k.path}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
