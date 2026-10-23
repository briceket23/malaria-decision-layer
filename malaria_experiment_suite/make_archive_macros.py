#!/usr/bin/env python3
r"""make_archive_macros.py --- archive.tex, from the EDA and verify layers.

    python make_archive_macros.py --eda eda --verify verify --out archive.tex

`mag report` writes `results.tex`: every macro that descends from the built
Study --- the factorial, the endpoint, calibration, attribution. It cannot
write the ones that describe the ARCHIVE and the CHECKPOINTS, because those
are measured before any Study exists, by `eda_dataset.py` and
`verify_models.py`.

Those numbers were being typed into the manuscript by hand. Protocol v3 SS19
says no number in the paper is hand-typed, and the reason is concrete: the
reviewed v2 build carried demo-generator figures that three reviewers read as
data. A number typed once is a number that cannot be re-checked.

So this writes the second macro file. `main.tex` inputs both, and every
archive-verification quantity in the manuscript is then traceable to a JSON
key in a run the reader can reproduce.

Macro names are prefixed `arc` so they cannot collide with mag's namespace,
and every one is read through `exp_schema`, so a producer that renames a key
breaks the test suite rather than silently emptying a slot in the PDF.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_schema as S                                        # noqa: E402
from exp_schema import EDA, VERIFY, MISSING                   # noqa: E402

EXIT_OK, EXIT_PRECONDITION = 0, 2

#: The species carried by so few patients that a per-class recall moves in
#: steps a reader will over-read. Protocol v3 SS15 requires them named.
THIN_SPECIES_MAX = 5

PARASITE_CLASSES = ("P_falciparum", "P_ovale", "P_malariae", "P_vivax",
                    "Babesia_divergens", "T_brucei")


def _v(x, default=None):
    return default if x is MISSING or x is None else x


def _pct(x, dp=1):
    x = _v(x)
    return None if x is None else f"{100 * float(x):.{dp}f}\\%"


def _n(x):
    x = _v(x)
    return None if x is None else f"{int(x):,}"


def _f(x, dp=4):
    x = _v(x)
    return None if x is None else f"{float(x):.{dp}f}"


#: LaTeX-special characters, for values copied VERBATIM out of the JSON.
#: They must not be applied to the generated sentences and tabular bodies in
#: this file, which contain deliberate LaTeX. mag's own transform verdict
#: contains a literal "75%", which comments out the rest of the line and made
#: the first build die with `File ended while scanning use of \defresult`.
_TEX_MAP = {
    "\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
    "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
    "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
}


def _lit(x):
    """A string out of the JSON, safe to place in a macro body.

    ONE pass over the characters, not a sequence of str.replace calls. A
    sequence cannot be ordered correctly: replacing the backslash first
    introduces braces that the later brace rule then escapes, and replacing
    the braces first leaves the backslash rule to escape its own output.
    Mapping each character once has neither problem.
    """
    x = _v(x)
    if x is None:
        return None
    return "".join(_TEX_MAP.get(ch, ch) for ch in str(x))


def _join(names) -> str:
    """Join for prose. Takes values that are ALREADY LaTeX --- the output of
    _species(), a formatted count --- and must not escape them."""
    names = [str(n) for n in names]
    if not names:
        return "none"
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def _joinlit(names) -> str:
    """Join names taken VERBATIM from the archive --- centres, checkpoints,
    image sizes. A centre name is archive data and may hold anything, so each
    is escaped before it becomes part of a macro body.

    Kept separate from _join because _join receives values that are already
    LaTeX: escaping those turned `\emph{B.\,divergens}` into the literal
    text `extbackslash{}emph{B. extbackslash{},divergens}` in the first
    build."""
    return _join([_lit(n) for n in names])


def _species(name: str) -> str:
    """`P_falciparum` -> `\\emph{P.\\,falciparum}`, for prose and tables."""
    pretty = {"P_falciparum": r"\emph{P.\,falciparum}",
              "P_ovale": r"\emph{P.\,ovale}",
              "P_malariae": r"\emph{P.\,malariae}",
              "P_vivax": r"\emph{P.\,vivax}",
              "Babesia_divergens": r"\emph{B.\,divergens}",
              "T_brucei": r"\emph{T.\,brucei}",
              "uninfected": "uninfected", "RBC": "RBC", "WBC": "WBC",
              "platelet": "platelet"}
    return pretty.get(name, name.replace("_", r"\_"))


# --------------------------------------------------------------------------- #
# the cohort and the archive
# --------------------------------------------------------------------------- #

def from_eda(e: dict) -> dict:
    out: dict = {}
    if not e:
        return out

    out["arcMag"] = _lit(EDA.MAG_VERSION.get(e))
    out["arcEdaGenerated"] = _lit(EDA.GENERATED.get(e))
    out["arcManifestRows"] = _n(EDA.MANIFEST_ROWS.get(e))

    parts = dict(EDA.PART_SMEARS.each(e))
    imgs = dict(EDA.PART_IMAGES.each(e))
    for st in ("train", "validation", "test"):
        out[f"arcSmears{st.capitalize()}"] = _n(parts.get(st))
        out[f"arcImages{st.capitalize()}"] = _n(imgs.get(st))

    # --- identifiability (P5), the estimand-defining fact ------------------- #
    out["arcNPatients"] = _n(EDA.IDENT_N_PATIENTS.get(e))
    out["arcNInfected"] = _n(EDA.IDENT_N_INFECTED.get(e))
    out["arcNUninfected"] = _n(EDA.IDENT_N_UNINFECTED.get(e))
    out["arcPrevalence"] = _pct(EDA.IDENT_PREVALENCE.get(e))
    centres = _v(EDA.IDENT_CENTRES.get(e), [])
    out["arcUninfectedCentres"] = _joinlit(centres)
    out["arcNUninfectedCentres"] = str(len(centres))
    sep = _v(EDA.IDENT_SEPARABLE.get(e))
    out["arcSpecSeparable"] = "1" if sep else "0"
    # The sentence a specificity claim must carry. Generated, so the scope
    # cannot drift from the measurement that forces it.
    if sep is False and centres:
        out["arcIdentifiabilitySentence"] = (
            f"All {_n(EDA.IDENT_N_UNINFECTED.get(e))} uninfected patients in "
            f"this archive come from a single centre ({_joinlit(centres)}), so "
            f"screening specificity and centre are perfectly collinear. This "
            f"is rank deficiency rather than low power: no estimator "
            f"separates the two and no sample size repairs it. Every "
            f"specificity quantity below is therefore an estimate for the "
            f"{_joinlit(centres)}-negative subset of this archive, and is "
            f"reported with that scope attached.")
        out["arcSpecificityScope"] = (
            f"among uninfected patients of the {_joinlit(centres)} cohort of "
            f"this archive")
    else:
        out["arcIdentifiabilitySentence"] = (
            f"Uninfected patients come from {_joinlit(centres)}, so "
            f"specificity is not collinear with centre in this archive."
            if sep else
            "Whether specificity is separable from centre was not assessed.")
        out["arcSpecificityScope"] = "among uninfected patients of this archive"

    # --- the leakage check, and the participant flow (STARD item 19) --------- #
    #
    # `partitions.problems` is the SMEAR-LEVEL DISJOINTNESS result, not a count
    # of refused smears: eda_dataset.survey_partitions checks every pair of
    # released partitions for a shared smear, because a smear in both train and
    # test makes the whole test estimate in-sample and nothing downstream can
    # detect it. An earlier version of this file labelled that list "smears
    # refused by a manifest check", which is the same class of mistake this
    # paper is about --- a number reported as something it is not.
    #
    # The three manifest refusals of protocol SS22.1 (an unmapped centre or
    # species, a smear carrying two of either, a listed image the archive does
    # not hold) are hard exits rather than per-smear exclusions: the ingest
    # stage either refuses the archive or proceeds with all of it. There is
    # therefore no partial-exclusion row to report, and the flow says so rather
    # than printing a zero that would imply one.
    sets = [str(x) for x in (_v(EDA.SETS.get(e), []) or [])]
    problems = _v(EDA.PART_PROBLEMS.get(e), []) or []
    out["arcNSets"] = str(len(sets))
    out["arcSetsChecked"] = _joinlit(sets) if sets else "none"
    out["arcNSetPairs"] = str(len(sets) * (len(sets) - 1) // 2)
    out["arcNPartitionOverlaps"] = _n(len(problems))
    if not sets:
        out["arcLeakageSentence"] = (
            "Whether the released partitions share a smear was not assessed by "
            "this run, so the test estimate cannot be shown to be "
            "out-of-sample.")
    elif problems:
        out["arcLeakageSentence"] = (
            f"{len(problems)} of the "
            f"{len(sets) * (len(sets) - 1) // 2} partition pairs share at "
            f"least one smear ({_joinlit(sorted({str(p) for p in problems}))}), "
            f"so the test estimate is in part in-sample and every quantity "
            f"below inherits that.")
    else:
        out["arcLeakageSentence"] = (
            f"The {len(sets)} released partitions ({_joinlit(sets)}) were "
            f"checked pairwise at smear level, all "
            f"{len(sets) * (len(sets) - 1) // 2} pairs of them, and no smear "
            f"appears in more than one. The test estimate is out-of-sample "
            f"with respect to the partition the archive declares, which is the "
            f"one leakage this design could neither detect downstream nor "
            f"repair.")

    _listed = parts.get("test")
    _flow = [
        ("fields listed in the manifest, all partitions",
         _n(EDA.MANIFEST_ROWS.get(e))),
        ("smears in the test partition", _n(_listed)),
        ("fields in the test partition", _n(imgs.get("test"))),
        ("partition pairs sharing a smear",
         f"{_n(len(problems))} of {len(sets) * (len(sets) - 1) // 2}"),
        ("patients analysed, every rule and every checkpoint",
         _n(EDA.IDENT_N_PATIENTS.get(e))),
        ("\\quad infected by the reference standard",
         _n(EDA.IDENT_N_INFECTED.get(e))),
        ("\\quad uninfected by the reference standard",
         _n(EDA.IDENT_N_UNINFECTED.get(e))),
    ]
    out["arcFlowTable"] = " ".join(f"{what} & {n} \\\\" for what, n in _flow)

    cohort = _v(EDA.COHORT_TABLE.get(e), {}) or {}
    tot = cohort.get("Total", {}) if isinstance(cohort, dict) else {}
    thin = sorted(k for k, n in tot.items()
                  if k not in ("Total", "uninfected")
                  and isinstance(n, (int, float)) and 0 < n <= THIN_SPECIES_MAX)
    out["arcThinSpecies"] = _join([_species(t) for t in thin]) if thin \
        else "none"
    out["arcThinSpeciesN"] = str(len(thin))
    # Every sentence slot is defined in BOTH directions. A conditional macro
    # leaves a red box in the PDF when the condition does not hold, which
    # reads as a missing analysis rather than as the finding it is: that the
    # archive does not have that property. The test suite pins this.
    out["arcThinSpeciesSentence"] = (
        f"No species is carried by {THIN_SPECIES_MAX} patients or fewer, so "
        f"no per-class claim is withheld on that ground.")
    if thin:
        counts = ", ".join(f"{_species(t)} {int(tot[t])}" for t in thin)
        out["arcThinSpeciesSentence"] = (
            f"{len(thin)} species are carried by {THIN_SPECIES_MAX} patients "
            f"or fewer ({counts}). A per-class recall over so few patients "
            f"moves in steps of tens of percentage points, so per-class "
            f"claims for these species are not made and a sensitivity "
            f"analysis excluding them is reported.")
    out["arcCentres"] = _joinlit([c for c in cohort if c != "Total"])
    out["arcNCentres"] = str(len([c for c in cohort if c != "Total"]))
    out["arcCohortTable"] = _cohort_tabular(cohort)
    out.update(_centre_twins(cohort))

    # --- annotations -------------------------------------------------------- #
    out["arcLabelBoxes"] = _n(EDA.LAB_BOXES.get(e))
    out["arcLabelFields"] = _n(EDA.LAB_FIELDS.get(e))
    out["arcLabelPerField"] = _f(EDA.LAB_PER_FIELD.get(e), 2)
    per = _v(EDA.LAB_PER_CLASS.get(e), {}) or {}
    if per:
        tot_boxes = sum(per.values())
        top = max(per, key=per.get)
        low = min(per, key=per.get)
        out["arcClassImbalance"] = f"{per[top] / max(per[low], 1):,.1f}:1"
        out["arcImbalanceExtremes"] = f"{_species(top)} to {_species(low)}"
        out["arcRbcShare"] = _pct(per.get("RBC", 0) / max(tot_boxes, 1))
        out["arcClassTable"] = _class_tabular(per, tot_boxes)

    # --- enrichment, which conditions the sensitivity estimand -------------- #
    enr = EDA.LOAD_ENRICHMENT.get(e)
    out["arcEnrichment"] = _pct(enr)
    out["arcEnrichmentSentence"] = ("Field enrichment among infected patients "
                                    "was not measured.")
    out["arcLoadMedian"] = _f(EDA.LOAD_MEDIAN.get(e), 0)
    out["arcLoadMin"] = _f(EDA.LOAD_MIN.get(e), 0)
    out["arcLoadMax"] = _f(EDA.LOAD_MAX.get(e), 0)
    out["arcLoadCoverage"] = _n(EDA.LOAD_COVERAGE.get(e))
    if _v(enr) is not None:
        out["arcEnrichmentSentence"] = (
            f"{_pct(enr)} of the fields of an infected patient carry at least "
            f"one annotated parasite (median {_f(EDA.LOAD_MEDIAN.get(e), 0)} "
            f"parasite boxes per smear, range {_f(EDA.LOAD_MIN.get(e), 0)} to "
            f"{_f(EDA.LOAD_MAX.get(e), 0)}). Fields were selected for "
            f"annotation, not sampled from a slide, so every rule here is "
            f"evaluated under field sampling far more favourable than "
            f"screening. This is part of the sensitivity estimand and not a "
            f"limitation disclosed afterwards.")

    # --- geometry and the transformation ------------------------------------ #
    out["arcParasiteRadius"] = _f(EDA.GEO_PAR_RADIUS.get(e))
    out["arcOtherRadius"] = _f(EDA.GEO_OTH_RADIUS.get(e))
    out["arcParasiteMargin"] = _pct(EDA.GEO_PAR_MARGIN.get(e))
    out["arcOtherMargin"] = _pct(EDA.GEO_OTH_MARGIN.get(e))
    out["arcPredParasite"] = _pct(EDA.GEO_PRED_PAR.get(e))
    out["arcPredOther"] = _pct(EDA.GEO_PRED_OTH.get(e))
    out["arcKeptParasite"] = _pct(EDA.TF_PARASITE_KEPT.get(e))
    out["arcKeptOther"] = _pct(EDA.TF_OTHER_KEPT.get(e))
    out["arcKeptAll"] = _pct(EDA.TF_ALL_KEPT.get(e))
    nearer = _v(EDA.GEO_NEARER.get(e))
    out["arcParasitesNearerCentre"] = "1" if nearer else "0"

    out["arcCropModelSentence"] = (
        "The crop model was not fitted: the archive did not supply both a "
        "predicted and a measured retention.")
    out["arcCropGapsOpposite"] = "0"
    gp, gm = EDA.GEO_PRED_PAR.get(e), EDA.TF_PARASITE_KEPT.get(e)
    go, gn = EDA.GEO_PRED_OTH.get(e), EDA.TF_OTHER_KEPT.get(e)
    if all(_v(x) is not None for x in (gp, gm, go, gn)):
        d_par, d_oth = float(gm) - float(gp), float(gn) - float(go)
        out["arcCropGapParasite"] = f"{100 * d_par:+.1f}\\%"
        out["arcCropGapOther"] = f"{100 * d_oth:+.1f}\\%"
        opposite = (d_par > 0) != (d_oth > 0)
        out["arcCropGapsOpposite"] = "1" if opposite else "0"
        # Clipping is one-directional. A box whose centre survives the crop
        # but whose extent does not is kept and truncated, which can only
        # push measured retention ABOVE the centre-based prediction. Gaps of
        # opposite sign therefore cannot both be clipping.
        out["arcCropModelSentence"] = (
            f"A crop of the field margins predicts {_pct(gp)} retention for "
            f"parasite boxes against {_pct(go)} for blood-component boxes, "
            f"from the box centres alone. Measured retention was "
            f"{_pct(gm)} and {_pct(gn)}, gaps of "
            f"{100 * d_par:+.1f} and {100 * d_oth:+.1f} percentage points."
            + (" The gaps have opposite signs. Clipping a box that straddles "
               "the new boundary can only raise measured retention above a "
               "centre-based prediction, never lower it, so a single positive "
               "gap would be unremarkable; two gaps of opposite sign cannot "
               "both arise from clipping, and the transformation is therefore "
               "not fully described by a crop."
               if opposite else
               " Both gaps lie on the side a centre-based prediction admits "
               "under clipping, and are consistent with a crop."))

    rising = _v(EDA.TF_RISING.get(e), {}) or {}
    out["arcRisingClasses"] = _join([_species(k) for k in rising]) \
        if rising else "none"
    out["arcNRisingClasses"] = str(len(rising))
    out["arcRisingSentence"] = (
        "No annotated class gained boxes between the two label trees, which "
        "is what a purely subtractive transformation predicts.")
    if rising:
        bits = ", ".join(
            f"{_species(k)} {int(x['base'])} to {int(x['zoom'])} "
            f"({float(x['kept']):.3f}$\\times$)"
            for k, x in sorted(rising.items()))
        out["arcRisingSentence"] = (
            f"{len(rising)} annotated class gained boxes between the two "
            f"label trees ({bits}). A crop removes boxes and cannot add one, "
            f"so these smears were re-annotated rather than only re-cropped. "
            f"This is a property of the released label trees; it is not a "
            f"discrepancy between the source publication and its data, and "
            f"is not reported as one."
            if len(rising) == 1 else
            f"{len(rising)} annotated classes gained boxes between the two "
            f"label trees ({bits}). A crop removes boxes and cannot add one, "
            f"so those smears were re-annotated rather than only re-cropped. "
            f"This is a property of the released label trees, not a "
            f"discrepancy between the source publication and its data.")
    out["arcTransformVerdict"] = _lit(EDA.TF_VERDICT.get(e)) or "not assessed"

    ratios = _v(EDA.IMG_RATIOS.get(e), {}) or {}
    sizes = _v(EDA.IMG_SIZES.get(e), {}) or {}
    out["arcNAspectRatios"] = str(len(ratios))
    out["arcAspectSentence"] = "Field dimensions were not sampled."
    out["arcImageSizes"] = _joinlit([f"{k} ({v:,})" for k, v in
                                  sorted(sizes.items(), key=lambda kv: -kv[1])])
    if len(ratios) > 1:
        out["arcAspectSentence"] = (
            f"The sampled fields carry {len(ratios)} distinct aspect ratios "
            f"({_joinlit(sorted(ratios))}), so a single geometric transformation "
            f"is unlikely to describe every contributing centre.")
    elif ratios:
        out["arcAspectSentence"] = (
            f"Every sampled field carries one aspect ratio "
            f"({_joinlit(sorted(ratios))}).")
    return out


def _centre_twins(cohort: dict) -> dict:
    """Centres whose species vectors are identical, and whether that is real.

    Two centres with the same count in every one of seven classes is either a
    balanced sampling design or a duplicated block in the manifest, and a
    reader cannot tell which from the table. The arithmetic settles it here:
    if the per-centre totals sum to the archive's own smear count, no smear is
    counted twice and the centres are disjoint by construction. This is
    reported in the caption rather than left for a reviewer to notice.
    """
    out: dict = {}
    rows = {k: v for k, v in (cohort or {}).items()
            if k != "Total" and isinstance(v, dict)}
    if len(rows) < 2:
        return out
    cols = [c for c in ("uninfected", "P_falciparum", "P_ovale", "P_malariae",
                        "P_vivax", "Babesia_divergens", "T_brucei")
            if any(c in r for r in rows.values())]
    sig: dict = {}
    for name, r in rows.items():
        sig.setdefault(tuple(int(r.get(c, 0) or 0) for c in cols),
                       []).append(name)
    twins = [names for names in sig.values() if len(names) > 1]

    declared = int(((cohort.get("Total") or {}).get("Total") or 0))
    summed = sum(int(r.get("Total", 0) or 0) for r in rows.values())
    out["arcCohortSums"] = "yes" if declared and summed == declared else "no"
    out["arcCohortSummed"] = _n(summed)
    out["arcNCentreTwinGroups"] = str(len(twins))

    if not twins:
        out["arcCentreTwinSentence"] = (
            "No two centres contribute the same count in every species class.")
        return out

    named = "; ".join(_joinlit(sorted(g)) for g in twins)
    n_each = twins[0] and int(rows[twins[0][0]].get("Total", 0) or 0)
    if declared and summed == declared:
        out["arcCentreTwinSentence"] = (
            f"{named} contribute an identical count in every species class "
            f"({_n(n_each)} patients each). The counts are genuine rather than "
            f"a duplicated manifest block: the per-centre totals sum to "
            f"{_n(summed)}, the archive's own smear count, so no smear is "
            f"counted under two centres. The agreement reflects a balanced "
            f"referral mix at those two centres and is reported here so that "
            f"it need not be read as an error.")
    else:
        out["arcCentreTwinSentence"] = (
            f"{named} contribute an identical count in every species class "
            f"({_n(n_each)} patients each), and the per-centre totals sum to "
            f"{_n(summed)} against a declared {_n(declared)}. The discrepancy "
            f"is reported and not resolved here.")
    return out


def _cohort_tabular(cohort: dict) -> str:
    """The cohort cross-tabulation, as a LaTeX tabular body."""
    if not cohort:
        return "\\multicolumn{2}{l}{not measured}"
    cols = [c for c in ("uninfected", "P_falciparum", "P_ovale", "P_malariae",
                        "P_vivax", "Babesia_divergens", "T_brucei")
            if any(row.get(c) for row in cohort.values())]
    rows = [c for c in cohort if c != "Total"]
    head = "centre & " + " & ".join(_species(c) for c in cols) + " & total \\\\"
    body = []
    for r in rows:
        cells = [str(int(cohort[r].get(c, 0) or 0)) for c in cols]
        body.append(f"{r} & " + " & ".join(cells)
                    + f" & {int(cohort[r].get('Total', 0) or 0)} \\\\")
    t = cohort.get("Total", {})
    body.append(r"\midrule total & "
                + " & ".join(str(int(t.get(c, 0) or 0)) for c in cols)
                + f" & {int(t.get('Total', 0) or 0)} \\\\")
    return head + r" \midrule " + " ".join(body)


def _class_tabular(per: dict, total: int) -> str:
    rows = sorted(per.items(), key=lambda kv: -kv[1])
    return " ".join(f"{_species(k)} & {v:,} & {100 * v / max(total, 1):.1f}\\% \\\\"
                    for k, v in rows)


# --------------------------------------------------------------------------- #
# the checkpoints
# --------------------------------------------------------------------------- #

def from_verify(v: dict) -> dict:
    out: dict = {}
    if not v:
        return out
    out["arcVerifyGenerated"] = _lit(VERIFY.GENERATED.get(v))
    out["arcMagMaxDet"] = _n(VERIFY.MAG_MAX_DET.get(v))
    out["arcVerifyVerdict"] = _lit(VERIFY.VERDICT.get(v))
    gpu = _v(VERIFY.GPU.get(v), {}) or {}
    out["arcGpu"] = _lit(gpu.get("name"))

    sha = dict(VERIFY.CK_SHA.each(v))
    fmt = dict(VERIFY.CK_FORMAT.each(v))
    ncls = dict(VERIFY.CK_NCLASSES.each(v))
    loader = dict(VERIFY.SMOKE_LOADER.each(v))
    mean = dict(VERIFY.SMOKE_PER_FIELD_MEAN.each(v))
    lo = dict(VERIFY.SMOKE_PER_FIELD_MIN.each(v))
    hi = dict(VERIFY.SMOKE_PER_FIELD_MAX.each(v))
    med = dict(VERIFY.SMOKE_SCORE_MEDIAN.each(v))
    cap = dict(VERIFY.SMOKE_FIELDS_AT_CAP.each(v))
    nimg = dict(VERIFY.SMOKE_N_IMAGES.each(v))
    op_conf = dict(VERIFY.OP_CONF.each(v))
    op_iou = dict(VERIFY.OP_IOU.each(v))

    names = sorted(sha)
    out["arcCheckpoints"] = _joinlit(names)
    out["arcNCheckpoints"] = str(len(names))
    out["arcSmokeFields"] = _n(next(iter(nimg.values()), None))
    conf = next(iter(op_conf.values()), None)
    iou = next(iter(op_iou.values()), None)
    out["arcOperatingPoint"] = (
        f"conf {conf}, IoU {iou}, max\\_det {out.get('arcMagMaxDet', '--')}"
        if conf is not None else "not recorded")

    rows = []
    for n in names:
        key = "".join(ch for ch in n.title() if ch.isalnum())
        out[f"arcSha{key}"] = str(sha[n])[:12]
        out[f"arcFormat{key}"] = str(fmt.get(n, "--"))
        out[f"arcLoader{key}"] = str(loader.get(n, "--"))
        out[f"arcPerField{key}"] = _f(mean.get(n), 1)
        out[f"arcScoreMedian{key}"] = _f(med.get(n), 4)
        out[f"arcAtCap{key}"] = str(int(cap.get(n) or 0))
        rows.append(
            f"{n} & \\texttt{{{str(sha[n])[:12]}}} & {fmt.get(n, '--')} & "
            f"{int(ncls.get(n, 0))} & {_f(mean.get(n), 1)} & "
            f"{int(lo.get(n) or 0):,}--{int(hi.get(n) or 0):,} & "
            f"{_f(med.get(n), 4)} & {int(cap.get(n) or 0)}/"
            f"{int(nimg.get(n) or 0)} \\\\")
    out["arcCheckpointTable"] = " ".join(rows)

    # --- the fixed query budget --------------------------------------------- #
    fixed = _v(VERIFY.AGREE_FIXED.get(v), []) or []
    out["arcFixedBudgetArms"] = _joinlit(fixed) if fixed else "none"
    out["arcFixedBudgetSentence"] = (
        "No checkpoint returned a constant number of boxes per field, so no "
        "arm's per-field count is a decoder query budget rather than a "
        "detection count.")
    out["arcRatioMaxMin"] = _f(VERIFY.AGREE_RATIO.get(v), 2)
    if fixed:
        one = fixed[0]
        key = "".join(ch for ch in str(one).title() if ch.isalnum())
        out["arcFixedBudgetSentence"] = (
            f"{_join(fixed)} returned exactly the same number of boxes on "
            f"every field examined ({out.get('arcPerField' + key, '--')} per "
            f"field, with no variation between a parasite-dense smear and a "
            f"clean one). That is a decoder query budget, not a detection "
            f"count: any statistic that counts, averages or thresholds boxes "
            f"measures the output convention before it measures the "
            f"detector. It is the evidence for treating matched "
            f"post-processing as the only condition in which a checkpoint "
            f"comparison is interpretable.")

    # --- the detection ceiling ---------------------------------------------- #
    sat = sorted(n for n, c in cap.items() if int(c or 0) > 0)
    out["arcSaturatingArms"] = _joinlit(sat) if sat else "none"
    out["arcCapSentence"] = (
        "No checkpoint reached the package's default detection limit on any "
        "field examined, so no cache is censored at the top and the limit "
        "was left at its default.")
    out["arcNSaturating"] = str(len(sat))
    sugg = dict(VERIFY.PROBE_SUGGESTED.each(v))
    disc = dict(VERIFY.PROBE_DISCARDED.each(v))
    dshare = dict(VERIFY.PROBE_DISCARDED_SHARE.each(v))
    dbias = dict(VERIFY.PROBE_DENSITY_BIAS.each(v))
    dcap = dict(VERIFY.PROBE_DENSITY_CAPPED.each(v))
    dtrue = dict(VERIFY.PROBE_DENSITY_TRUE.each(v))
    dby = dict(VERIFY.PROBE_BY_CLASS.each(v))
    dmax = dict(VERIFY.PROBE_TRUE_MAX.each(v))
    dmat = dict(VERIFY.PROBE_MATERIAL.each(v))
    thr = dict(VERIFY.SMOKE_CENSORING.each(v))

    if sugg:
        out["arcCapProbed"] = _n(max(int(x) for x in sugg.values()))
    for n in sat:
        key = "".join(ch for ch in n.title() if ch.isalnum())
        out[f"arcCapDiscarded{key}"] = _n(disc.get(n))
        out[f"arcCapDiscardedShare{key}"] = _pct(dshare.get(n))
        out[f"arcCapDensityBias{key}"] = _pct(dbias.get(n))
        out[f"arcCapTrueMax{key}"] = _n(dmax.get(n))
        out[f"arcCapSuggested{key}"] = _n(sugg.get(n))
        out[f"arcCensoringThreshold{key}"] = _f(thr.get(n), 4)
    if sat:
        n = sat[0]
        key = "".join(ch for ch in str(n).title() if ch.isalnum())
        by = dby.get(n) or {}
        bits = ", ".join(f"{_species(k)} {int(x):,}" for k, x in
                         sorted(by.items(), key=lambda kv: -kv[1]))
        out["arcCapSentence"] = (
            f"{_join(sat)} returned exactly the package default detection "
            f"limit ({out.get('arcMagMaxDet', '--')}) on "
            f"{int(cap.get(n) or 0)} of {int(nimg.get(n) or 0)} fields at the "
            f"cached operating point. A field at the limit is truncated: the "
            f"highest-scoring boxes are kept and the remainder discarded, and "
            f"nothing in the pipeline's yield gate --- which checks a floor "
            f"--- detects a ceiling. Re-running those fields with the limit "
            f"lifted to {_n(next(iter(dict(VERIFY.PROBE_MAX_DET.each(v)).values()), None))} "
            f"showed the truncation discarding {_n(disc.get(n))} boxes "
            f"({_pct(dshare.get(n))} of the true output: {bits}), which moved "
            f"the parasites-per-erythrocyte statistic from "
            f"{_f(dtrue.get(n), 5)} to {_f(dcap.get(n), 5)}, a bias of "
            f"{_pct(dbias.get(n))}"
            + (". That is material for the density rule, which applies no "
               "score filter and therefore reads the censored tail directly. "
               if dmat.get(n) else ". ")
            + f"Above a score of {_f(thr.get(n), 4)} the truncation provably "
              f"changed nothing on these fields, since everything discarded "
              f"lay below it. The confirmatory pass was therefore run at a "
              f"limit of {_n(sugg.get(n))}, chosen from the probe's own "
              f"measured maximum, and the original limit is retained as a "
              f"pre-specified sensitivity arm.")
    return out


# --------------------------------------------------------------------------- #

def from_geometry(g: dict) -> dict:
    """What the geometry step removed from the detector cache, if anything.

    `run_experiment.sanitise_geometry` deletes every box of non-positive width
    or height before any condition reads the cache. It has to: a zero-area box
    has an empty IoU with everything, so it neither suppresses nor is
    suppressed, and it would pass the matched condition untouched --- the one
    thing that condition exists to prevent. But a step that mutates the cache
    after detection is a step the manuscript must be able to state, in both
    directions and with its own counts, including the boxes it removed that
    scored above the source study's published operating point. Writing the
    count to a JSON file the paper never reads is the reporting failure this
    study is about.
    """
    out: dict = {}
    if not g:
        out["arcGeomSentence"] = (
            "Whether the geometry step removed any detection was not assessed "
            "for this run: no geometry accounting was found beside the cache.")
        return out

    per = {str(k): (v or {}) for k, v in g.items() if isinstance(v, dict)}
    if not per:
        return out
    total = sum(int(r.get("n_total") or 0) for r in per.values())
    excl = sum(int(r.get("n_excluded") or 0) for r in per.values())
    above = sum(int(r.get("above_0_25") or 0) for r in per.values())
    fields = sum(int(r.get("n_fields_affected") or 0) for r in per.values())

    out["arcGeomNCheckpoints"] = _n(len(per))
    out["arcGeomBoxesChecked"] = _n(total)
    out["arcGeomExcluded"] = _n(excl)
    out["arcGeomAboveOperating"] = _n(above)
    out["arcGeomFieldsAffected"] = _n(fields)
    out["arcGeomShare"] = _pct(excl / total, 3) if total else None

    checked = ", ".join(sorted(per))
    if excl == 0:
        out["arcGeomSentence"] = (
            f"Before any condition read the cache, every detection was checked "
            f"for non-positive width or height, because a zero-area box has an "
            f"empty intersection with every other box and would therefore pass "
            f"the matched condition without being suppressed or suppressing "
            f"anything. Across the {_n(len(per))} checkpoints ({checked}), "
            f"{_n(total)} boxes were checked and {_n(excl)} were removed, so "
            f"on this run the step changed no cache and the three conditions "
            f"read the detections exactly as the checkpoints emitted them.")
    else:
        out["arcGeomSentence"] = (
            f"Before any condition read the cache, every detection was checked "
            f"for non-positive width or height, because a zero-area box has an "
            f"empty intersection with every other box and would therefore pass "
            f"the matched condition without being suppressed or suppressing "
            f"anything. Across the {_n(len(per))} checkpoints ({checked}), "
            f"{_n(excl)} of {_n(total)} boxes ({_pct(excl / total, 3)}) were "
            f"removed over {_n(fields)} fields, {_n(above)} of them scoring at "
            f"or above the source study's 0.25 operating point. The excluded "
            f"rows are written beside the cache with their class and score "
            f"accounting, so the exclusion is auditable rather than silent; "
            f"the analysis was run on the cache after this step.")
    return out


# --------------------------------------------------------------------------- #

def write(macros: dict, path: Path, sources: dict) -> None:
    lines = [
        "% archive.tex --- generated by make_archive_macros.py. Do not edit.",
        "% Every value below is read from a completed run, through the same",
        "% exp_schema keys the producers write. Nothing here is typed.",
        f"% generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
    ]
    for k, p in sources.items():
        lines.append(f"% {k:<8} {p}")
    lines.append("")
    bad = []
    for k in sorted(macros):
        v = macros[k]
        if v is None:
            continue
        # An unescaped % comments out the rest of the line and swallows the
        # closing brace; an unbalanced brace runs away the same way. Both kill
        # the build with an error that names neither the macro nor this file.
        if re.search(r"(?<!\\)%", str(v)) or \
                str(v).count("{") != str(v).count("}"):
            bad.append(k)
        lines.append("\\defresult{%s}{%s}" % (k, v))
    if bad:
        print("  ! %d macro(s) hold an unescaped %% or unbalanced braces and "
              "would break the LaTeX build: %s" % (len(bad), ", ".join(bad)),
              file=sys.stderr)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_archive_macros.py",
                                description=__doc__.split("\n\n")[0])
    p.add_argument("--eda", default="eda", help="folder holding eda.json")
    p.add_argument("--verify", default="verify",
                   help="folder holding verify.json")
    p.add_argument("--geometry", default="data/test",
                   help="folder holding geometry_exclusions.json, written "
                        "beside the detector cache by the detect phase")
    p.add_argument("--out", default="archive.tex")
    a = p.parse_args(argv)

    def read(folder, name):
        for cand in (Path(folder) / name, Path(folder)):
            if cand.is_file():
                return json.loads(cand.read_text(encoding="utf-8")), str(cand)
        return None, f"{Path(folder) / name} (absent)"

    e, e_src = read(a.eda, "eda.json")
    v, v_src = read(a.verify, "verify.json")
    g, g_src = read(a.geometry, "geometry_exclusions.json")
    if e is None and v is None:
        print("neither eda.json nor verify.json was found; nothing to write.",
              file=sys.stderr)
        return EXIT_PRECONDITION

    for doc, schema, label in ((e, EDA, "eda.json"), (v, VERIFY, "verify.json")):
        if doc is None:
            continue
        probs = schema.validate(doc)
        if probs:
            print(f"  ! {label} does not conform: {'; '.join(probs[:3])}")

    macros = {}
    macros.update(from_eda(e or {}))
    macros.update(from_verify(v or {}))
    macros.update(from_geometry(g or {}))
    write(macros, Path(a.out), {"eda": e_src, "verify": v_src,
                                "geometry": g_src})

    print(f"  wrote {len(macros)} archive macros -> {a.out}")
    print(f"    eda      {e_src}")
    print(f"    verify   {v_src}")
    print(f"    geometry {g_src}")
    missing = [k for k, val in macros.items() if val is None]
    if missing:
        print(f"    {len(missing)} macro(s) had no source value and were "
              f"left undefined (they render red): {', '.join(missing[:6])}")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
