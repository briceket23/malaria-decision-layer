#!/usr/bin/env python3
r"""eda_dataset.py --- what the archive says, before any model runs.

    python eda_dataset.py --root D:\malaria_france_data_set

Reads the released archive and answers the questions the experiment depends
on: can this cohort carry the estimands, what is in the label trees, and does
the transform behave the way the manuscript says it does. Touches no GPU,
writes nothing into the archive, and takes a few seconds.

WHAT IT ESTABLISHES

  Proposition 5    whether specificity is separable from centre. If every
                   uninfected smear comes from one site the two are perfectly
                   collinear, which is RANK DEFICIENCY, not low power.
  enrichment       the share of an infected patient's fields carrying a
                   parasite box. Near 1 means every accuracy in the paper is
                   an optimistic upper bound, unevenly across the rule grid.
  geometry         whether parasites really do sit nearer the field centre,
                   and what retention a pure 4:3 -> 16:9 crop would leave ---
                   a PREDICTION, testable against the measured retention.
  transform        which classes gained boxes. A crop removes boxes; it cannot
                   add one, so a class that grows was re-annotated.
  disjointness     every partition pair, not just the one the source states.
                   A smear in both train and test makes the whole test
                   estimate in-sample and nothing downstream could detect it.

OUTPUT

`eda.json`, whose every key is declared in `exp_schema.EDA` and validated
before the file is written; `EDA.md`; and figures. Nothing here spells a key
name inline --- a consumer reading a path that no producer writes is a failing
test in `test_experiment.py`, not a puzzle on someone's archive.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_schema as S                                        # noqa: E402
from exp_schema import EDA                                    # noqa: E402

EXIT_OK, EXIT_PRECONDITION, EXIT_UNEXPECTED = 0, 2, 4


def hr(t=""):
    print("\n" + ("--- " + t + " " if t else "").ljust(78, "-"))


def pct(x, dp=1):
    return "--" if x is None or (isinstance(x, float) and math.isnan(x)) \
        else f"{100 * x:.{dp}f}%"


def _mag():
    try:
        import mag
        from mag import discover, ingest
    except ImportError as exc:
        print(f"\nthis script builds on mag ({exc}).\n", file=sys.stderr)
        print(f'    "{sys.executable}" -m pip install -e .', file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)
    return mag, discover, ingest


# --------------------------------------------------------------------------- #
# 1. Cohort and partitions
# --------------------------------------------------------------------------- #

def survey_partitions(ingest, il, sets, report, log) -> None:
    """Per-partition size, and the smear-level disjointness that matters most.

    The source publication states disjointness only between training and
    validation. Every pair is checked here, because a smear in both train and
    test would make the whole test estimate in-sample and **nothing downstream
    could detect it** --- not a gate, not an interval, not a simulation.
    """
    problems, smears, detail = [], {}, {}
    for s in sets:
        try:
            p = ingest.build_patients(il, s)
        except Exception as exc:                              # noqa: BLE001
            msg = str(exc).splitlines()[0]
            detail[s] = {"error": msg}
            log(f"  {s:<12} -- {msg[:60]}")
            continue
        fields = p["n_images"].astype(int)
        smears[s] = set(p["patient_id"].astype(str))
        EDA.PART_SMEARS.put(report, int(len(p)), name=s)
        EDA.PART_IMAGES.put(report, int(fields.sum()), name=s)
        detail[s] = {
            "fields_min": int(fields.min()), "fields_max": int(fields.max()),
            "fields_median": float(fields.median()),
            "centres": sorted(p["centre"].dropna().astype(str).unique())}
        log(f"  {s:<12} {len(p):>4} smears  {int(fields.sum()):>7,} images   "
            f"fields {int(fields.min())}-{int(fields.max())}, "
            f"median {int(fields.median())}")

    names = sorted(smears)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            shared = smears[a] & smears[b]
            if not shared:
                continue
            # test vs test-zoom SHARING every smear is correct and expected:
            # they are one set of smears annotated twice, before and after the
            # transform. Any OTHER overlap is contamination.
            if {a, b} == {"test", "test-zoom"}:
                continue
            problems.append(f"{a} and {b} share {len(shared)} smear(s): the "
                            f"test estimate would be in-sample")
    EDA.PART_PROBLEMS.put(report, problems)
    for p_ in problems:
        log(f"  ! {p_}")
    return detail


def identifiability(ingest, il, set_name, report, log) -> None:
    """Proposition 5. The single most consequential fact about this cohort."""
    p = ingest.build_patients(il, set_name)
    unin = p[p["truth"] == "uninfected"]
    centres = sorted(unin["centre"].dropna().astype(str).unique())
    collinear = len(centres) <= 1

    EDA.IDENT_N_PATIENTS.put(report, int(len(p)))
    EDA.IDENT_N_UNINFECTED.put(report, int(len(unin)))
    EDA.IDENT_N_INFECTED.put(report, int((p["truth"] != "uninfected").sum()))
    EDA.IDENT_PREVALENCE.put(
        report, round(float((p["truth"] != "uninfected").mean()), 4))
    EDA.IDENT_CENTRES.put(report, centres)
    EDA.IDENT_SEPARABLE.put(report, not collinear)

    log(f"  {len(p)} patients: {int((p['truth'] != 'uninfected').sum())} "
        f"infected, {len(unin)} uninfected  "
        f"(prevalence {pct(EDA.IDENT_PREVALENCE.get(report))})")
    log(f"  uninfected come from: {', '.join(centres) or '--'}")
    if collinear:
        log("")
        log("  ! SPECIFICITY IS NOT SEPARABLE FROM CENTRE.")
        log("    Every uninfected smear comes from one site, so specificity and")
        log("    site are perfectly collinear. This is RANK DEFICIENCY, not low")
        log("    power: no sample size fixes it and no estimator recovers the")
        log("    separated effects. No domain-stratified specificity can be")
        log("    reported from this archive, in any rotation.")
    return p


# --------------------------------------------------------------------------- #
# 2. Annotations
# --------------------------------------------------------------------------- #

def survey_labels(ingest, archive, set_name, report, log, zoom=False):
    """Every box in one label tree."""
    lab = ingest.read_yolo_labels(archive, set_name)
    if not len(lab):
        log(f"  no labels under {set_name}")
        return lab

    n_fields = lab.groupby(["patient_id", "image_id"]).ngroups
    per_class = lab["cls"].value_counts()
    counts = {str(k): int(v) for k, v in per_class.items()}

    if zoom:
        EDA.LABZ_BOXES.put(report, int(len(lab)))
        EDA.LABZ_PER_CLASS.put(report, counts)
    else:
        EDA.LAB_SET.put(report, set_name)
        EDA.LAB_BOXES.put(report, int(len(lab)))
        EDA.LAB_FIELDS.put(report, int(n_fields))
        EDA.LAB_PER_FIELD.put(report,
                              round(len(lab) / max(n_fields, 1), 2))
        EDA.LAB_PER_CLASS.put(report, counts)

    log(f"  {len(lab):>9,} boxes over {n_fields:,} fields "
        f"({len(lab) / max(n_fields, 1):.2f}/field)")
    for k, v in per_class.items():
        log(f"      {str(k):<20} {v:>9,}  {pct(v / len(lab)):>7}")
    if len(per_class) > 1:
        log(f"  imbalance   {per_class.max() / per_class.min():,.1f}:1 "
            f"({per_class.idxmax()} to {per_class.idxmin()})")
    return lab


def centre_distance(lab, parasites, report, log) -> None:
    """Do parasite boxes sit nearer the field centre than erythrocytes?

    The manuscript INFERS this from two retention numbers and offers it as the
    explanation. It is directly measurable, so it is measured.

    YOLO coordinates are normalised, so the field centre is (0.5, 0.5) and the
    distance is comparable across images of any size. A 4:3 -> 16:9 crop keeps
    the full width and the central 75% of the height, so it removes boxes with
    |y - 0.5| > 0.375 --- which gives a PREDICTED retention per kind, testable
    against the retention the label trees actually show.
    """
    if not len(lab) or "x" not in lab:
        return
    d = lab.copy()
    d["r"] = ((d["x"] - 0.5) ** 2 + (d["y"] - 0.5) ** 2) ** 0.5
    d["cropped_by_16_9"] = (d["y"] - 0.5).abs() > 0.375

    EDA.GEO_PER_CLASS.put(report, {
        str(cls): {"n": int(len(g)),
                   "median_radius": round(float(g["r"].median()), 4),
                   "mean_radius": round(float(g["r"].mean()), 4),
                   "share_in_crop_margin":
                       round(float(g["cropped_by_16_9"].mean()), 4)}
        for cls, g in d.groupby("cls")})

    par = d[d["cls"].isin(parasites)]
    oth = d[~d["cls"].isin(parasites)]
    if not (len(par) and len(oth)):
        return

    p_r = round(float(par["r"].median()), 4)
    o_r = round(float(oth["r"].median()), 4)
    p_m = round(float(par["cropped_by_16_9"].mean()), 4)
    o_m = round(float(oth["cropped_by_16_9"].mean()), 4)
    EDA.GEO_PAR_RADIUS.put(report, p_r)
    EDA.GEO_OTH_RADIUS.put(report, o_r)
    EDA.GEO_PAR_MARGIN.put(report, p_m)
    EDA.GEO_OTH_MARGIN.put(report, o_m)
    EDA.GEO_NEARER.put(report, bool(p_r < o_r))
    EDA.GEO_PRED_PAR.put(report, round(1 - p_m, 4))
    EDA.GEO_PRED_OTH.put(report, round(1 - o_m, 4))

    log("  median radial distance from field centre (normalised):")
    log(f"      parasite classes   {p_r:.4f}")
    log(f"      blood classes      {o_r:.4f}")
    log("  share in the margin a 4:3 -> 16:9 crop removes:")
    log(f"      parasite classes   {pct(p_m)}")
    log(f"      blood classes      {pct(o_m)}")
    log("")
    if p_r < o_r:
        log("  -> parasites DO sit nearer the field centre. The manuscript's")
        log("     explanation for differential retention is supported by the")
        log("     geometry, not only inferred from the two totals.")
    else:
        log("  ! parasites do NOT sit nearer the field centre in this archive.")
        log("    The manuscript explains differential retention by saying they")
        log("    do. That explanation does not survive this measurement and")
        log("    should be removed or replaced, however plausible it reads.")
    log(f"  under a pure crop, retention would be "
        f"{pct(1 - p_m)} parasite / {pct(1 - o_m)} blood.")
    log("  Those are PREDICTIONS. transform_retention below measures what")
    log("  actually happened, and the two can be compared.")


def parasite_load(lab, parasites, patients, report, log) -> None:
    """Parasite-enrichment, as a count rather than a caveat."""
    if not len(lab):
        return
    par = lab[lab["cls"].isin(parasites)]

    truth = dict(zip(patients["patient_id"].astype(str),
                     patients["truth"].astype(str)))
    inf_ids = {k for k, v in truth.items() if v != "uninfected"}
    inf = lab[lab["patient_id"].astype(str).isin(inf_ids)]
    inf_par = par[par["patient_id"].astype(str).isin(inf_ids)]
    inf_fields = inf.groupby(["patient_id", "image_id"]).ngroups
    inf_hit = inf_par.groupby(["patient_id", "image_id"]).ngroups

    # Only over smears that HAVE a label file. Reindexing across every infected
    # patient and filling zero conflates "not annotated" with "no parasites
    # found", which on a partially-walked label tree reports a median of 0 for
    # an archive whose real median is in the dozens.
    labelled = set(lab["patient_id"].astype(str))
    counted = sorted(inf_ids & labelled)
    per_smear = (par.groupby("patient_id").size()
                 .reindex(counted, fill_value=0))

    enrich = round(inf_hit / max(inf_fields, 1), 4)
    EDA.LOAD_ENRICHMENT.put(report, enrich)
    EDA.LOAD_COVERAGE.put(report, int(len(counted)))
    EDA.LOAD_TOTAL.put(report, int(len(inf_ids)))
    if len(per_smear):
        EDA.LOAD_MEDIAN.put(report, float(per_smear.median()))
        EDA.LOAD_MIN.put(report, float(per_smear.min()))
        EDA.LOAD_MAX.put(report, float(per_smear.max()))

    log(f"  {pct(enrich)} of an infected patient's fields carry a parasite box")
    if len(per_smear):
        log(f"  parasite boxes per infected smear: median "
            f"{per_smear.median():.0f}, range {int(per_smear.min())}-"
            f"{int(per_smear.max())}  (over {len(counted)} annotated smears)")
    if len(counted) < len(inf_ids):
        log(f"  ! only {len(counted)} of {len(inf_ids)} infected smears carry "
            f"a label file.")
        log("    The counts above are over those; they are not a statement "
            "about the rest.")
    if enrich > 0.9:
        log("")
        log("  ! FIELDS ARE PARASITE-ENRICHED.")
        log("    Real screening is dominated by parasite-free fields. Every")
        log("    accuracy computed here is an optimistic upper bound, and the")
        log("    bias is UNEVEN: counting and density rules see an inflated,")
        log("    low-variance signal that would sit near zero under random")
        log("    field sampling. The RANKING of rules is at risk, not only the")
        log("    levels.")


# --------------------------------------------------------------------------- #
# 3. Transform
# --------------------------------------------------------------------------- #

def transform(ingest, archive, base, zoomed, report, log) -> None:
    """mag's own retention measurement, plus the classes that GREW."""
    try:
        tr = ingest.transform_retention(Path(archive), base, zoomed)
    except Exception as exc:                                  # noqa: BLE001
        log(f"  transform_retention failed: {type(exc).__name__}: {exc}")
        return
    if not tr:
        return
    for key, src in ((EDA.TF_PARASITE_KEPT, "parasite_kept"),
                     (EDA.TF_OTHER_KEPT, "other_kept"),
                     (EDA.TF_ALL_KEPT, "all_kept"),
                     (EDA.TF_BY_CLASS, "by_class"),
                     (EDA.TF_VERDICT, "verdict")):
        if tr.get(src) is not None:
            key.put(report, tr[src])

    log(f"  parasite boxes kept {pct(tr.get('parasite_kept'))}, "
        f"other {pct(tr.get('other_kept'))}, "
        f"all {pct(tr.get('all_kept'))}")
    if tr.get("verdict"):
        log(f"  mag's verdict: {tr['verdict']}")

    rising = {k: v for k, v in (tr.get("by_class") or {}).items()
              if (v.get("kept") or 0) > 1.0}
    EDA.TF_RISING.put(report, rising)
    if rising:
        log("")
        log("  ! CLASSES THAT GAIN BOXES AFTER THE TRANSFORM:")
        for k, v in sorted(rising.items()):
            log(f"      {k:<22} {v['base']:>6} -> {v['zoom']:>6}  "
                f"({v['kept']:.3f}x)")
        log("    A crop removes boxes; it cannot add one. These classes are")
        log("    not related to their base by any subtractive transform, so")
        log("    those smears were RE-ANNOTATED rather than merely re-cropped.")
        log("    This is a property of the released label trees, not a")
        log("    discrepancy between a paper and its data, and should be")
        log("    stated that way.")

    # The crop model, tested rather than assumed.
    pred_p, pred_o = EDA.GEO_PRED_PAR.get(report), EDA.GEO_PRED_OTH.get(report)
    if pred_p is not S.MISSING and tr.get("parasite_kept") is not None:
        gp = tr["parasite_kept"] - pred_p
        go = (tr["other_kept"] - pred_o
              if pred_o is not S.MISSING and tr.get("other_kept") is not None
              else None)
        log("")
        log("  crop model, predicted vs measured retention:")
        log(f"      parasite   {pct(pred_p)} predicted, "
            f"{pct(tr['parasite_kept'])} measured  ({gp:+.1%})")
        if go is not None:
            log(f"      blood      {pct(pred_o)} predicted, "
                f"{pct(tr['other_kept'])} measured  ({go:+.1%})")
            if (gp > 0) != (go > 0):
                log("")
                log("  ! THE GAPS HAVE OPPOSITE SIGNS.")
                log("    The margin test is on box CENTRES, so a box "
                    "straddling the new")
                log("    boundary may be clipped and kept rather than "
                    "dropped. That")
                log("    effect can only push measured retention ABOVE the "
                    "prediction,")
                log("    never below --- so one positive gap proves nothing. "
                    "Gaps of")
                log("    opposite sign cannot both be clipping: something "
                    "removed boxes")
                log("    the crop would keep AND something kept boxes it "
                    "would remove.")


# --------------------------------------------------------------------------- #
# 4. Images
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


def survey_images(archive, set_name, sample, report, log) -> None:
    """Dimensions and aspect ratios, on a sample.

    The source study resized test images to 16:9 with an unspecified zoom
    correction, and the six centres used cameras with differing native aspect
    ratios. Whether the images on disk are 16:9 or 4:3 is a fact worth having
    before reasoning about what the transform did.
    """
    try:
        from PIL import Image
    except ImportError:
        log("  (Pillow not installed; skipping. pip install pillow)")
        return

    root = Path(archive) / "images" / set_name
    paths = sorted(root.rglob("*.jpg"))[:sample] if root.is_dir() else []
    if not paths:
        log(f"  no images under {root}")
        return

    sizes, ratios = Counter(), Counter()
    for p in paths:
        try:
            with Image.open(p) as im:
                w, h = im.size
        except Exception:                                     # noqa: BLE001
            continue
        sizes[(w, h)] += 1
        ratios[round(w / h, 3)] += 1

    EDA.IMG_SAMPLED.put(report, len(paths))
    EDA.IMG_SIZES.put(report, {f"{w}x{h}": n for (w, h), n in sizes.most_common(8)})
    EDA.IMG_RATIOS.put(report, {str(r): n for r, n in ratios.most_common(8)})

    for (w, h), n in sizes.most_common(5):
        log(f"      {w}x{h:<6} {n:>5} images   ratio {name_ratio(w / h)}")
    if len(ratios) > 1:
        log(f"  ! {len(ratios)} distinct aspect ratios in the sample "
            f"({', '.join(name_ratio(r) for r, _ in ratios.most_common(4))}).")
        log("    A single geometric transform is unlikely to describe all "
            "contributing centres.")


# --------------------------------------------------------------------------- #
# 5. Figures
# --------------------------------------------------------------------------- #

def draw(report, lab, patients, parasites, out, log):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log("  (matplotlib not installed; skipping figures)")
        return []

    out.mkdir(parents=True, exist_ok=True)
    written = []
    PAR, OTH = "#c0504d", "#1f4e79"

    def _boxplot(ax, groups, names):
        """`labels=` became `tick_labels=` in matplotlib 3.9."""
        try:
            return ax.boxplot(groups, tick_labels=names, showfliers=False,
                              patch_artist=True)
        except TypeError:
            return ax.boxplot(groups, labels=names, showfliers=False,
                              patch_artist=True)

    def save(fig, name):
        p = out / f"{name}.png"
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close(fig)
        written.append(p.name)

    if patients is not None and len(patients):
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(patients["n_images"].astype(int), bins=30, color=OTH,
                edgecolor="white")
        ax.set_xlabel("fields per smear")
        ax.set_ylabel("smears")
        ax.set_title("Reading effort available per patient")
        save(fig, "fields_per_smear")

    if lab is not None and len(lab):
        vc = lab["cls"].value_counts()
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh([str(i) for i in vc.index][::-1], vc.values[::-1],
                color=[PAR if str(i) in parasites else OTH
                       for i in vc.index][::-1])
        ax.set_xscale("log")
        ax.set_xlabel("boxes (log scale)")
        ax.set_title("Annotations by class --- red = parasite")
        save(fig, "class_balance")

        d = lab.copy()
        d["r"] = ((d["x"] - 0.5) ** 2 + (d["y"] - 0.5) ** 2) ** 0.5
        groups = [g["r"].values for _, g in d.groupby("cls")]
        names = [str(k) for k, _ in d.groupby("cls")]
        fig, ax = plt.subplots(figsize=(8, 4))
        bp = _boxplot(ax, groups, names)
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(PAR if n in parasites else OTH)
        # the crop boundary the retention prediction is built on
        ax.axhline(0.375, color="#7f7f7f", ls="--", lw=1)
        ax.text(0.995, 0.375, " 4:3 -> 16:9 crop boundary", fontsize=7,
                color="#555", va="bottom", ha="right",
                transform=ax.get_yaxis_transform())
        ax.set_ylabel("distance from field centre (normalised)")
        ax.set_title("Are parasites nearer the field centre? red = parasite")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "centre_distance_by_class")

        pf = lab.groupby(["patient_id", "image_id"]).size()
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.hist(pf.values, bins=50, color=OTH, edgecolor="white")
        ax.set_xlabel("annotated boxes per field")
        ax.set_ylabel("fields")
        ax.set_title(f"Annotation density (median {pf.median():.0f}/field)")
        save(fig, "boxes_per_field")

        d["area"] = d["w"] * d["h"]
        groups = [g["area"].values for _, g in d.groupby("cls")]
        fig, ax = plt.subplots(figsize=(8, 4))
        bp = _boxplot(ax, groups, names)
        for patch, n in zip(bp["boxes"], names):
            patch.set_facecolor(PAR if n in parasites else OTH)
        ax.set_yscale("log")
        ax.set_ylabel("box area (fraction of field)")
        ax.set_title("Object size by class")
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "box_area_by_class")

    # predicted vs measured retention: the crop model, drawn
    pp, po = EDA.GEO_PRED_PAR.get(report), EDA.GEO_PRED_OTH.get(report)
    mp, mo = EDA.TF_PARASITE_KEPT.get(report), EDA.TF_OTHER_KEPT.get(report)
    if S.MISSING not in (pp, po, mp, mo):
        fig, ax = plt.subplots(figsize=(6, 3.6))
        x = [0, 1]
        ax.bar([i - 0.19 for i in x], [pp, po], width=0.38,
               color="#9aa7b1", label="predicted by a pure crop")
        ax.bar([i + 0.19 for i in x], [mp, mo], width=0.38,
               color=[PAR, OTH], label="measured in the label trees")
        for i, (a, b) in enumerate(((pp, mp), (po, mo))):
            ax.annotate(f"{b - a:+.1%}", (i + 0.19, b), ha="center",
                        va="bottom", fontsize=9,
                        color="#b00" if (b - a) > 0 else "#036")
        ax.set_xticks(x)
        ax.set_xticklabels(["parasite classes", "blood classes"])
        ax.set_ylabel("box retention")
        ax.set_ylim(0, 1.05)
        ax.set_title("Does a pure 4:3 -> 16:9 crop explain the transform?")
        ax.legend(fontsize=7)
        save(fig, "crop_model_fit")

    coh = EDA.COHORT_TABLE.get(report)
    if coh and coh is not S.MISSING:
        import pandas as pd
        t = pd.DataFrame(coh).drop(columns=["Total"], errors="ignore")
        t = t.drop(index=["Total"], errors="ignore")
        fig, ax = plt.subplots(figsize=(8, 4))
        t.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
        ax.set_ylabel("smears")
        ax.set_title("Cohort composition by centre")
        ax.legend(fontsize=7, ncol=2)
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        save(fig, "cohort_by_centre")

    log(f"  {len(written)} figures -> {out}")
    EDA.FIGURES.put(report, written)
    return written


# --------------------------------------------------------------------------- #
# 6. Report
# --------------------------------------------------------------------------- #

def write_md(report, path):
    g = lambda k, d="--": (k.get(report) if k.get(report) is not S.MISSING  # noqa: E731
                           else d)
    L = ["# EDA --- what the archive says", "",
         f"Archive: `{g(EDA.ARCHIVE)}`  ",
         f"`mag` {g(EDA.MAG_VERSION)}  ",
         f"Generated {g(EDA.GENERATED)}", ""]

    L += ["## Identifiability (Proposition 5)", "",
          f"- {g(EDA.IDENT_N_PATIENTS)} patients: "
          f"{g(EDA.IDENT_N_INFECTED)} infected, "
          f"{g(EDA.IDENT_N_UNINFECTED)} uninfected",
          f"- uninfected centres: "
          f"{', '.join(g(EDA.IDENT_CENTRES, []) or ['--'])}",
          f"- specificity separable from centre: "
          f"**{g(EDA.IDENT_SEPARABLE)}**", ""]
    if g(EDA.IDENT_SEPARABLE) is False:
        L += ["> Specificity and site are perfectly collinear. This is rank "
              "deficiency, not low power: no sample size fixes it and no "
              "estimator recovers the separated effects.", ""]

    L += ["## Annotations", "",
          f"- {g(EDA.LAB_BOXES):,} boxes, {g(EDA.LAB_PER_FIELD)}/field", ""]
    per = g(EDA.LAB_PER_CLASS, {})
    if per:
        L += ["| class | boxes |", "|---|---|"]
        L += [f"| {k} | {v:,} |" for k, v in
              sorted(per.items(), key=lambda kv: -kv[1])]
        L.append("")

    L += ["## Geometry and the transform", "",
          f"- median radius: parasite {g(EDA.GEO_PAR_RADIUS)}, "
          f"blood {g(EDA.GEO_OTH_RADIUS)}",
          f"- crop predicts retention: parasite {g(EDA.GEO_PRED_PAR)}, "
          f"blood {g(EDA.GEO_PRED_OTH)}",
          f"- measured retention: parasite {g(EDA.TF_PARASITE_KEPT)}, "
          f"blood {g(EDA.TF_OTHER_KEPT)}", ""]
    rising = g(EDA.TF_RISING, {})
    if rising:
        L += ["### Classes that gained boxes", "",
              "A crop removes boxes; it cannot add one.", "",
              "| class | base | after | ratio |", "|---|---|---|---|"]
        L += [f"| {k} | {v['base']} | {v['zoom']} | {v['kept']:.3f} |"
              for k, v in sorted(rising.items())]
        L.append("")

    L += ["## Enrichment", "",
          f"- {pct(g(EDA.LOAD_ENRICHMENT, None))} of an infected patient's "
          f"fields carry a parasite box",
          f"- over {g(EDA.LOAD_COVERAGE)} of {g(EDA.LOAD_TOTAL)} infected "
          f"smears that carry a label file", ""]

    figs = g(EDA.FIGURES, [])
    if figs:
        L += ["## Figures", ""] + [f"- `{f}`" for f in figs] + [""]
    Path(path).write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eda_dataset.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", "--archive", dest="root", default=".")
    p.add_argument("--image-list", dest="image_list", default=None)
    p.add_argument("--set", default="test", help="the analysis partition")
    p.add_argument("--zoom-set", dest="zoom_set", default="test-zoom")
    p.add_argument("--sets", nargs="+",
                   default=["train", "validation", "test"],
                   help="partitions to survey and check for overlap")
    p.add_argument("--sample-images", dest="sample_images", type=int,
                   default=400)
    p.add_argument("--out", default="eda")
    p.add_argument("--no-figures", dest="no_figures", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    mag, discover, ingest = _mag()

    print("=" * 78)
    print(" eda_dataset.py --- what the archive says, before any model runs")
    print("=" * 78)

    d = discover.discover(Path(a.root), max_depth=6)
    image_list = Path(a.image_list) if a.image_list else d.image_list
    if not image_list or not Path(image_list).is_file():
        print(f"\nno image_list found under {a.root}.", file=sys.stderr)
        print("  Pass it:  --image-list PATH", file=sys.stderr)
        return EXIT_PRECONDITION

    report: dict = {}
    EDA.ARCHIVE.put(report, str(d.archive or a.root))
    EDA.MAG_VERSION.put(report, f"{mag.__version__} build {mag.build_id()}")
    EDA.GENERATED.put(report, dt.datetime.now(dt.timezone.utc)
                      .isoformat(timespec="seconds"))
    EDA.SETS.put(report, list(a.sets))

    print(f"  mag       {mag.__version__}  build {mag.build_id()}")
    print(f"  archive   {d.archive}")
    print(f"  manifest  {image_list}")

    il = ingest.read_image_list(Path(image_list))
    EDA.MANIFEST_ROWS.put(report, int(len(il)))
    EDA.MANIFEST_COLUMNS.put(report, list(il.columns))
    print(f"  rows      {len(il):,}   columns: {', '.join(il.columns)}")

    hr("class map")
    if d.classes:
        got = ingest.read_classes(Path(d.classes))
        # classes.txt speaks the ARCHIVE's vocabulary ("Platelets",
        # "P. falciparum"); the protocol speaks the analysis vocabulary
        # ("platelet", "P_falciparum"). Comparing the two literally reports a
        # mismatch on a file that is correct -- which an earlier version of
        # this script did, on this archive. `ingest.CLASS_TO_SCHEMA` is the
        # declared translation between them, so the comparison runs through it.
        mapped = [ingest.CLASS_TO_SCHEMA.get(c, c) for c in got]
        EDA.CLASSES_DECLARED.put(report, list(got))
        EDA.CLASSES_SCHEMA.put(report, list(S.ALL_CLASSES))
        match = mapped == list(S.ALL_CLASSES)
        EDA.CLASSES_MATCH.put(report, bool(match))
        for i, (a_, b_) in enumerate(zip(got, mapped)):
            want = (S.ALL_CLASSES[i] if i < len(S.ALL_CLASSES) else "--")
            flag = "" if b_ == want else f"   != protocol {want}"
            print(f"  {i}  {a_:<24} -> {b_}{flag}")
        if not match:
            print("  ! classes.txt does NOT match the protocol index for "
                  "index.")
            print("    A permutation here relabels every box in the archive "
                  "and every")
            print("    downstream artefact would still look correct: the "
                  "table")
            print("    validates, the counts are plausible, no test recovers "
                  "it.")
        else:
            print(f"  {len(mapped)} classes, matching the protocol index for "
                  f"index (after the archive-to-protocol name mapping)")

    hr("partitions")
    survey_partitions(ingest, il, a.sets, report, print)

    hr(f"cohort --- {a.set}")
    coh = ingest.cohort_table(ingest.build_patients(il, a.set))
    EDA.COHORT_TABLE.put(report, json.loads(coh.to_json(orient="index")))
    EDA.COHORT_MARKDOWN.put(report, coh.to_string())
    print(coh.to_string())

    hr("identifiability (Proposition 5)")
    patients = identifiability(ingest, il, a.set, report, print)

    lab = None
    parasites = set(S.PARASITE_CLASSES)
    if d.archive:
        hr(f"annotations --- {a.set}")
        lab = survey_labels(ingest, d.archive, a.set, report, print)

        if lab is not None and len(lab):
            hr("geometry: are parasites nearer the field centre?")
            centre_distance(lab, parasites, report, print)

            hr("parasite enrichment")
            parasite_load(lab, parasites, patients, report, print)

        hr(f"annotations --- {a.zoom_set} (after the transform)")
        survey_labels(ingest, d.archive, a.zoom_set, report, print, zoom=True)

        hr("transform retention")
        transform(ingest, d.archive, a.set, a.zoom_set, report, print)

        hr("image dimensions")
        survey_images(d.archive, a.set, a.sample_images, report, print)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if not a.no_figures:
        hr("figures")
        draw(report, lab, patients, parasites, out / "figures", print)

    # The contract, enforced before anything reads this file.
    hr("schema")
    problems = EDA.validate(report)
    if problems:
        print(f"  ! {len(problems)} key(s) do not conform to exp_schema.EDA:")
        for p in problems:
            print(f"      {p}")
        print("\n  The file is still written --- a consumer reading through")
        print("  exp_schema will report the absent keys as UNKNOWN rather "
              "than")
        print("  as cleared. But this list is the drift, and it is the thing")
        print("  to fix.")
    else:
        print(f"  {len(EDA.keys)} declared keys, all conforming")

    hr("written")
    (out / "eda.json").write_text(json.dumps(report, indent=1, default=str),
                                  encoding="utf-8")
    write_md(report, out / "EDA.md")
    print(f"  {out / 'EDA.md'}")
    print(f"  {out / 'eda.json'}")
    if not a.no_figures:
        print(f"  {out / 'figures'}/")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
