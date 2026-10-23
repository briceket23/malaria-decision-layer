#!/usr/bin/env python3
r"""verify_models.py --- can these checkpoints detect? Find out in a minute.

    python verify_models.py --root D:\malaria_france_data_set

Opens each released `.pt`, checks its class map index for index, then **runs it
over a handful of real fields at mag's own operating point** and reports what
came back. Writes `verify.json`, whose keys are declared in
`exp_schema.VERIFY`, plus `VERIFY.md`. Touches no cache and writes nothing into
the archive.

WHY THIS EXISTS

`mag detect` is the expensive stage. Four failures make that pass worthless,
and all four are visible in seconds on twenty images.

  **A checkpoint that does not detect.** On one recorded run, yolov5x returned
  two detections across 4,508 fields. That is not a weak detector; something
  between the weights and the cache dropped everything. The cause was a
  repository-format pickle falling through to ultralytics' `YOLO`, which loads
  it WITHOUT ERROR and returns garbage --- so `fmt` is passed to `load_model`
  here, exactly as mag's own detect stage does.

  **A permuted class map.** If index-to-name is a permutation of the right one,
  every downstream artefact still looks correct: the table validates, the
  counts are plausible, the accuracy is merely somewhat lower than reported.
  No statistical test recovers it and no gate catches it.

  **The wrong operating point.** mag drives every checkpoint at `conf=0.001`,
  `iou=1.0` (suppression off) and `max_det=MAX_DET`. Ultralytics' defaults are
  `iou=0.7` and `max_det=300`. Omitting them smoke-tests a different detector
  from the one the pass will cache --- and the yolov5 adapter hides it, because
  its own signature defaults already match mag. This script passes them
  explicitly and records what it used.

  **A cache censored at the top.** `mag.detect.implausible_yield` guards the
  floor: too FEW boxes. Nothing guards the ceiling. A field returning exactly
  `max_det` boxes was truncated, and the cache that results is complete,
  well-formed, correctly hashed, and censored. Every gate accepts it. This
  script measures the saturation, then re-runs just the capped fields with the
  cap lifted to find out how much was lost and in which classes --- which is
  what decides whether it is a footnote or a re-run.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_schema as S                                        # noqa: E402
from exp_schema import VERIFY                                 # noqa: E402


def hush_amp_deprecations() -> None:
    """Silence the yolov5 fork's per-batch torch.cuda.amp FutureWarnings.

    The fork calls `torch.cuda.amp.autocast` inside AutoShape once per batch
    and torch 2.7 deprecated it. It would normally print once, but libraries
    around it open `warnings.catch_warnings()` blocks, and each exit
    invalidates the "already shown here" registry, so it reprints per batch.
    Only this exact deprecation is filtered; every other FutureWarning still
    prints. MALARIA_SHOW_WARNINGS=1 restores them.

    Called at import, BEFORE torch is imported anywhere in this process, so
    the filter is in place for the first batch.
    """
    import os
    import warnings
    if os.environ.get("MALARIA_SHOW_WARNINGS") == "1":
        return
    warnings.filterwarnings(
        "ignore",
        message=r"`torch\.cuda\.amp\.\w+\(args\.\.\.\)` is deprecated",
        category=FutureWarning)


hush_amp_deprecations()

EXIT_OK, EXIT_PRECONDITION, EXIT_FAILED = 0, 2, 3

#: How thin a yield has to be before it is a fault rather than a quiet field.
#: `mag.detect.implausible_yield` is the authority; this is the fallback.
FALLBACK_MIN_PER_FIELD = 0.5

#: How much the detection cap must move R6/R6b density before this script calls
#: it a failure rather than a note. A DECLARED CONVENTION, not a discovered
#: constant: it stands in for a question only the analysis stage can answer ---
#: whether any patient's density actually sits that close to rho.
MATERIAL_DENSITY_BIAS = 0.02

PARASITE = set(S.PARASITE_CLASSES)
RBC = S.RBC


def hr(t=""):
    print("\n" + ("--- " + t + " " if t else "").ljust(78, "-"))


def _mag():
    try:
        import mag
        from mag import detect, discover, ingest
    except ImportError as exc:
        print(f"\nthis script builds on mag ({exc}).\n", file=sys.stderr)
        print(f'    "{sys.executable}" -m pip install -e .', file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)
    return mag, detect, discover, ingest


# --------------------------------------------------------------------------- #
# 1. The file itself
# --------------------------------------------------------------------------- #

def true_format(detect, path, declared, log):
    """Which family the pickled model class actually belongs to.

    `read_checkpoint_metadata` decides by EXCEPTION: it tries `torch.load` and
    calls the file "yolov5-repo" only when the unpickler raises
    `ModuleNotFoundError` for `models`/`utils`. In a clean process that is
    right. In a CONTAMINATED one it is wrong, and silently so: anything that
    has already put top-level `models`/`utils` on `sys.path` --- importing the
    `yolov5` fork does it, and `_require_yolov5_modules` leaves them there
    after any successful load --- makes the unpickle succeed, the file is
    labelled "ultralytics", and `load_model` hands a YOLOv5 pickle to
    ultralytics' `YOLO`, which accepts it WITHOUT ERROR and returns raw tensor
    columns as confidences.

    This is not hypothetical. A dependency check three phases earlier in the
    same process, doing nothing worse than `import yolov5` to report its
    version, flipped yolov5x from "yolov5-repo" to "ultralytics" and turned
    2,724 boxes per field into a constant 14 with scores up to 638.8. The
    diagnostic changed the thing it was diagnosing.

    So the family is read from the model class itself. `models.yolo.*` is the
    repository format; `ultralytics.nn.*` is the package format. No import
    order can move that, and the answer does not depend on being right about
    which import did the contaminating.
    """
    try:
        import torch
        with detect._full_load():
            ck = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:                                         # noqa: BLE001
        return declared, None                # cannot probe; trust the declared
    model = ck.get("model") if isinstance(ck, dict) else None
    mod = type(model).__module__ if model is not None else ""
    if mod.split(".")[0] == "models":
        actual = "yolov5-repo"
    elif mod.split(".")[0] == "ultralytics":
        actual = "ultralytics"
    else:
        return declared, mod or None
    if actual != declared:
        log(f"  ! declared format {declared!r}, but the pickled model class "
            f"is {mod}")
        log(f"    Using {actual!r}. The declared value is decided by whether "
            f"an")
        log("    unpickle RAISES, which anything that has already imported "
            "the")
        log("    yolov5 modules in this process silently prevents. Loading a")
        log("    YOLOv5 pickle through ultralytics' YOLO succeeds and returns")
        log("    garbage, so this disagreement is caught here rather than in "
            "the")
        log("    cache.")
    return actual, mod


def inspect_weights(detect, name, path, expected, report, log,
                    repo=None) -> dict:
    """Hash it, open it, and check its class map index for index."""
    import hashlib

    problems: list = []
    VERIFY.CK_PATH.put(report, str(path), name=name)
    VERIFY.CK_PROBLEMS.put(report, problems, name=name)

    p = Path(path)
    if not p.is_file():
        problems.append("no such file")
        log(f"  ! no file at {p}")
        return {"problems": problems}

    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    VERIFY.CK_SHA.put(report, h.hexdigest(), name=name)
    VERIFY.CK_BYTES.put(report, int(p.stat().st_size), name=name)
    log(f"  file      {p.name}  {p.stat().st_size / 1e6:.1f} MB")
    log(f"  sha256    {h.hexdigest()}")

    try:
        meta = detect.read_checkpoint_metadata(p, repo)
    except Exception as exc:                                  # noqa: BLE001
        problems.append(f"could not open: {type(exc).__name__}: {exc}")
        log(f"  ! could not open: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:70]}")
        log("    A repository-format YOLOv5 pickle names `models.yolo` and")
        log("    `utils.*`, which exist only inside a yolov5 checkout. Point "
            "at one")
        log("    with --yolov5-repo, or install the packaged fork.")
        return {"problems": problems}

    # The declared format is only as good as the process it was read in;
    # reconcile it against the pickled model class before anything uses it.
    declared = meta.get("format")
    actual, mod = true_format(detect, p, declared, log)
    if actual != declared:
        meta["format"] = actual
        meta["declared_format"] = declared
        meta["model_class_module"] = mod
    if meta.get("format"):
        VERIFY.CK_FORMAT.put(report, str(meta["format"]), name=name)
    for k in ("format", "declared_format", "model_class_module",
              "task", "epoch", "date", "version"):
        if meta.get(k) is not None:
            log(f"  {k:<10}{meta[k]}")

    try:
        cmap = detect.class_map_from(meta, strict=True)
    except Exception as exc:                                  # noqa: BLE001
        problems.append(f"class map refused: {exc}")
        log(f"  ! class map refused: {str(exc).splitlines()[0][:70]}")
        return {"problems": problems, "metadata": meta}

    ordered = [cmap[i] for i in sorted(cmap)]
    VERIFY.CK_NCLASSES.put(report, len(ordered), name=name)

    mismatch = [f"index {i}: checkpoint {cmap.get(i)!r}, protocol {want!r}"
                for i, want in enumerate(expected) if cmap.get(i) != want]
    if len(ordered) != len(expected):
        mismatch.append(f"{len(ordered)} classes against {len(expected)}")

    if mismatch:
        problems.extend(mismatch)
        log(f"  ! CLASS MAP MISMATCH ({len(mismatch)})")
        for m in mismatch[:6]:
            log(f"      {m}")
        if set(ordered) == set(expected) and len(ordered) == len(expected):
            log("")
            log("    Same classes, DIFFERENT ORDER --- a permutation. Every")
            log("    downstream artefact would still look correct: the table")
            log("    validates, the counts are plausible, the confusion "
                "matrix is")
            log("    dense in sensible places. No statistical test recovers "
                "it.")
    else:
        log(f"  classes   {len(ordered)}, matching the protocol index for "
            f"index")
    return {"problems": problems, "metadata": meta, "class_map": cmap}


# --------------------------------------------------------------------------- #
# 2. The cap probe
# --------------------------------------------------------------------------- #

def probe_cap(model, paths, kw, max_det, probe_det, class_names, report, name,
              log) -> dict:
    """Re-run the capped fields with the cap lifted, and count what was lost.

    The censoring threshold says WHERE the cut fell. It does not say how deep
    it went, and only the depth decides what to do: a cap discarding forty
    boxes per field is a footnote, one discarding forty thousand is a re-run.
    The two look identical in the summary.

    The per-class split is the part that decides the direction.
    `rules._ev_density` divides parasite boxes by RBC boxes and applies no
    score filter to either, so a score-ranked truncation moves a RATIO. If the
    discarded tail is mostly RBC the denominator was cut and density is biased
    UP --- toward calling patients infected.
    """
    VERIFY.PROBE_MAX_DET.put(report, int(probe_det), name=name)
    VERIFY.PROBE_N_FIELDS.put(report, len(paths), name=name)
    kw2 = dict(kw)
    kw2["max_det"] = int(probe_det)
    log("")
    log(f"    probing {len(paths)} capped field(s) at max_det={probe_det:,} ...")

    per, cls_below, n_below, still = [], Counter(), 0, 0
    kept_par = kept_rbc = lost_par = lost_rbc = 0
    try:
        for batch in (paths[i:i + 2] for i in range(0, len(paths), 2)):
            for res in model.predict([str(p) for p in batch], **kw2):
                b = getattr(res, "boxes", None)
                n = 0 if b is None else len(b)
                per.append(int(n))
                if n >= probe_det:
                    still += 1
                if not n:
                    continue
                sc = [float(x) for x in b.conf.cpu().numpy()]
                cl = [int(x) for x in b.cls.cpu().numpy()]
                order = sorted(range(n), key=lambda i: sc[i], reverse=True)

                def nm(i):
                    return (class_names[cl[i]] if class_names
                            and 0 <= cl[i] < len(class_names) else str(cl[i]))

                for i in order[:max_det]:
                    k = nm(i)
                    kept_par += k in PARASITE
                    kept_rbc += k == RBC
                for i in order[max_det:]:
                    n_below += 1
                    k = nm(i)
                    cls_below[k] += 1
                    lost_par += k in PARASITE
                    lost_rbc += k == RBC
    except Exception as exc:                                  # noqa: BLE001
        log(f"    ! probe failed: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:60]}")
        return {}

    out = {}
    VERIFY.PROBE_TRUE_MEAN.put(
        report, round(sum(per) / max(len(per), 1), 1), name=name)
    VERIFY.PROBE_TRUE_MAX.put(report, max(per) if per else 0, name=name)
    VERIFY.PROBE_DISCARDED.put(report, int(n_below), name=name)
    VERIFY.PROBE_DISCARDED_SHARE.put(
        report, round(n_below / max(sum(per), 1), 3), name=name)
    VERIFY.PROBE_BY_CLASS.put(report, dict(cls_below.most_common()), name=name)
    VERIFY.PROBE_STILL_CAPPED.put(report, int(still), name=name)
    VERIFY.PROBE_KEPT_PAR.put(report, int(kept_par), name=name)
    VERIFY.PROBE_KEPT_RBC.put(report, int(kept_rbc), name=name)
    VERIFY.PROBE_LOST_PAR.put(report, int(lost_par), name=name)
    VERIFY.PROBE_LOST_RBC.put(report, int(lost_rbc), name=name)

    log(f"    uncapped: {sum(per):,} boxes over {len(per)} field(s), "
        f"{sum(per) / max(len(per), 1):.1f}/field (max {max(per):,})")
    log(f"    the cap keeps {max_det * len(paths):,} and DISCARDS "
        f"{n_below:,} ({n_below / max(sum(per), 1):.1%} of the true output)")
    if cls_below:
        log("    discarded tail by class: "
            + ", ".join(f"{k} {v:,}" for k, v in cls_below.most_common(5)))

    material = None
    if kept_rbc and (lost_rbc or lost_par):
        d_cap = kept_par / kept_rbc
        d_true = (kept_par + lost_par) / (kept_rbc + lost_rbc)
        bias = (d_cap / d_true - 1.0) if d_true else 0.0
        material = abs(bias) >= MATERIAL_DENSITY_BIAS
        VERIFY.PROBE_DENSITY_CAPPED.put(report, round(d_cap, 6), name=name)
        VERIFY.PROBE_DENSITY_TRUE.put(report, round(d_true, 6), name=name)
        VERIFY.PROBE_DENSITY_BIAS.put(report, round(bias, 4), name=name)
        VERIFY.PROBE_MATERIAL.put(report, bool(material), name=name)
        out["density_bias"] = bias
        log("")
        log(f"    density on these fields: {d_cap:.5f} capped vs "
            f"{d_true:.5f} true")
        log(f"    -> the cap moves R6/R6b density by {bias:+.1%} "
            f"({lost_rbc:,} RBC lost from the denominator, {lost_par:,} "
            f"parasite from the numerator)")
        if material:
            log("    That is material. R6 is one of the six MAIN_RULES.")
        else:
            log(f"    Under the {MATERIAL_DENSITY_BIAS:.0%} bar this script "
                f"treats as material.")
            log("    Declare it; it does not by itself justify re-running the "
                "pass.")
    out["material"] = material

    if still:
        log(f"    ! {still} field(s) STILL at the probe cap --- the true count "
            f"is higher again")

    # A cap that binds on 20 sampled fields binds harder on 4,508: the observed
    # maximum is an order statistic over the sample, not over the run.
    suggested = int(max(per) * 2) if per else None
    if suggested:
        VERIFY.PROBE_SUGGESTED.put(report, suggested, name=name)
        log("")
        log(f"    true maximum over these {len(per)} field(s): {max(per):,}. "
            f"Over a full")
        log(f"    pass it will be higher --- this is a SAMPLE maximum. "
            f"MAX_DET = {suggested:,}")
        log("    would clear it with margin.")
    out["suggested"] = suggested
    return out


# --------------------------------------------------------------------------- #
# 3. The smoke test
# --------------------------------------------------------------------------- #

def smoke_test(detect, name, path, images, conf, device, imgsz, report, log,
               fmt=None, repo=None, probe=30000, class_names=None) -> dict:
    """Run the model over real fields at mag's operating point."""
    problems: list = []
    notes: list = []
    VERIFY.SMOKE_N_IMAGES.put(report, len(images), name=name)
    VERIFY.SMOKE_PROBLEMS.put(report, problems, name=name)
    VERIFY.SMOKE_NOTES.put(report, notes, name=name)

    t0 = time.time()
    try:
        # `fmt` MUST be passed. mag's own detect stage does
        #     meta = read_checkpoint_metadata(weights, repo)
        #     model, loader = load_model(weights, meta.get("format"), repo)
        # and dropping the second argument is not harmless: the YOLOv5
        # repository checkpoint then falls through to ultralytics' `YOLO`,
        # which LOADS IT WITHOUT ERROR and returns garbage --- a fixed count of
        # boxes per field carrying "confidences" in the hundreds, because raw
        # tensor columns are read as scores.
        model, kind = detect.load_model(Path(path), fmt, repo)
    except Exception as exc:                                  # noqa: BLE001
        problems.append(f"load_model failed: {type(exc).__name__}: {exc}")
        log(f"  ! load_model failed: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:70]}")
        return {"problems": problems}
    VERIFY.SMOKE_LOADER.put(report, str(kind), name=name)
    log(f"  loader    {kind}  ({time.time() - t0:.1f}s)")

    # mag's detect stage drives every checkpoint at ONE operating point
    # (`_infer_group`). Ultralytics' own defaults are iou=0.7 (suppression ON)
    # and max_det=300; at those, an ultralytics-loaded checkpoint comes back
    # NMS-suppressed and clipped, which looks like a plausible per-field count
    # and is not one.
    max_det = int(getattr(detect, "MAX_DET", 3000))
    kw = {"conf": conf, "device": device, "verbose": False,
          "iou": 1.0, "agnostic_nms": False, "max_det": max_det}
    if imgsz:
        kw["imgsz"] = imgsz
    VERIFY.OP_CONF.put(report, float(conf), name=name)
    VERIFY.OP_IOU.put(report, 1.0, name=name)
    VERIFY.OP_AGNOSTIC.put(report, False, name=name)
    VERIFY.OP_MAX_DET.put(report, max_det, name=name)

    t0 = time.time()
    per_field, scores, classes, field_min = [], [], [], []
    bad_geom = Counter()
    min_w = min_h = None
    try:
        for batch in (images[i:i + 4] for i in range(0, len(images), 4)):
            for res in model.predict([str(p) for p in batch], **kw):
                b = getattr(res, "boxes", None)
                n = 0 if b is None else len(b)
                per_field.append(int(n))
                if n:
                    f = [float(x) for x in b.conf.cpu().numpy()]
                    scores += f
                    field_min.append(min(f))
                    cl = [int(x) for x in b.cls.cpu().numpy()]
                    classes += cl
                    # The same tensor the detect stage caches: xywhn, centre
                    # form, normalised. A degenerate box here becomes a
                    # BadInput when the cache is read, an hour later.
                    try:
                        wh = b.xywhn.cpu().numpy()[:, 2:4]
                        w_min, h_min = float(wh[:, 0].min()), float(wh[:, 1].min())
                        min_w = w_min if min_w is None else min(min_w, w_min)
                        min_h = h_min if min_h is None else min(min_h, h_min)
                        for i_, (w_, h_) in enumerate(wh):
                            if w_ <= 0 or h_ <= 0:
                                k = (class_names[cl[i_]]
                                     if class_names and 0 <= cl[i_] < len(class_names)
                                     else str(cl[i_]))
                                bad_geom[k] += 1
                    except Exception:                         # noqa: BLE001
                        pass       # a loader without xywhn: not this check's job
                else:
                    field_min.append(None)
    except Exception as exc:                                  # noqa: BLE001
        problems.append(f"predict failed: {type(exc).__name__}: {exc}")
        log(f"  ! predict failed: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0][:70]}")
        return {"problems": problems}

    secs = time.time() - t0
    total, with_any = sum(per_field), sum(1 for n in per_field if n)
    VERIFY.SMOKE_DETECTIONS.put(report, int(total), name=name)
    VERIFY.SMOKE_PER_FIELD_MEAN.put(
        report, round(total / max(len(per_field), 1), 1), name=name)
    VERIFY.SMOKE_PER_FIELD_MIN.put(
        report, min(per_field) if per_field else 0, name=name)
    VERIFY.SMOKE_PER_FIELD_MAX.put(
        report, max(per_field) if per_field else 0, name=name)
    VERIFY.SMOKE_FIELDS_WITH_ANY.put(report, int(with_any), name=name)
    VERIFY.SMOKE_SECONDS_PER_FIELD.put(
        report, round(secs / max(len(images), 1), 3), name=name)
    if scores:
        scores.sort()
        VERIFY.SMOKE_SCORE_MEDIAN.put(
            report, round(scores[len(scores) // 2], 4), name=name)
        VERIFY.SMOKE_SCORE_MAX.put(report, round(scores[-1], 4), name=name)

    log(f"  detections{total:>8,} over {len(images)} fields   "
        f"{total / max(len(per_field), 1):.1f}/field  "
        f"(range {min(per_field) if per_field else 0}-"
        f"{max(per_field) if per_field else 0})")
    log(f"  fields with at least one box: {with_any}/{len(per_field)}")
    if scores:
        log(f"  scores    median {scores[len(scores) // 2]:.4f}, max "
            f"{scores[-1]:.4f}, "
            f"{sum(1 for s in scores if s >= 0.25):,} above 0.25")
    log(f"  speed     {secs / max(len(images), 1):.3f}s/field")

    out = {"problems": problems, "notes": notes, "loader": kind}

    # --- geometry that suppression cannot see -------------------------------- #
    n_bad = sum(bad_geom.values())
    VERIFY.SMOKE_BAD_GEOM.put(report, int(n_bad), name=name)
    VERIFY.SMOKE_BAD_GEOM_SHARE.put(
        report, round(n_bad / max(total, 1), 5), name=name)
    if min_w is not None:
        VERIFY.SMOKE_MIN_W.put(report, round(min_w, 6), name=name)
        VERIFY.SMOKE_MIN_H.put(report, round(min_h, 6), name=name)
    if n_bad:
        VERIFY.SMOKE_BAD_GEOM_BY_CLASS.put(
            report, dict(bad_geom.most_common()), name=name)
        problems.append(f"{n_bad} of {total} boxes have a non-positive width "
                        f"or height")
        log("")
        log(f"  ! {n_bad:,} OF {total:,} BOXES HAVE w <= 0 OR h <= 0 "
            f"({n_bad / max(total, 1):.2%}).")
        log("    mag.schema.validate_detections REFUSES a cache containing "
            "these:")
        log("    a box of zero extent has zero area, so the IoU matrix clips "
            "it and")
        log("    it neither suppresses nor is suppressed --- it passes "
            "through the")
        log("    matched condition unchallenged, which is the one thing that "
            "condition")
        log("    exists to prevent. The refusal happens when the cache is "
            "READ, so")
        log("    without this check it arrives an hour after the pass.")
        log("    by class: " + ", ".join(f"{k} {v:,}"
                                         for k, v in bad_geom.most_common(6)))
        log("")
        log("    Do NOT fix this by dropping the rows. The count is evidence "
            "about")
        log("    a geometry conversion, and mag's own remedy says to check "
            "that")
        log("    conversion --- silently discarding the boxes would hide the "
            "cause")
        log("    and change the measurement at the same time.")

    # --- a confidence that is not a probability ---------------------------- #
    # mag.schema.validate_detections refuses a table whose scores leave [0, 1]:
    # a confidence outside it means the loader read the wrong tensor column, so
    # both the scores AND the geometry are suspect. Catching it here costs 20
    # images; catching it there costs the whole pass, because the cache is
    # written first and validated when it is read.
    if scores and (scores[0] < 0.0 or scores[-1] > 1.0):
        problems.append(f"scores outside [0,1]: min {scores[0]:.4g}, "
                        f"max {scores[-1]:.4g}")
        log("")
        log(f"  ! SCORES ARE NOT PROBABILITIES --- {scores[0]:.4g} to "
            f"{scores[-1]:.4g}")
        log("    The loader read the wrong tensor column. Both the scores and")
        log("    the geometry are suspect, and validate_detections will refuse")
        log("    the whole table --- after the pass has been paid for.")
        if fmt and kind and fmt.replace("-", "") not in str(kind).lower() \
                .replace("-", ""):
            log(f"    The checkpoint declares format {fmt!r} and it loaded as")
            log(f"    {kind!r}. Those disagree.")

    # --- the detection cap, reached ---------------------------------------- #
    n_sat = sum(1 for n in per_field if n >= max_det)
    VERIFY.SMOKE_FIELDS_AT_CAP.put(report, int(n_sat), name=name)
    VERIFY.SMOKE_CAP_SHARE.put(
        report, round(n_sat / max(len(per_field), 1), 3), name=name)
    if n_sat:
        cut = [s for n, s in zip(per_field, field_min)
               if n >= max_det and s is not None]
        thr = round(max(cut), 4) if cut else None
        if thr is not None:
            VERIFY.SMOKE_CENSORING.put(report, thr, name=name)
        log("")
        log(f"  ! {n_sat} OF {len(per_field)} FIELDS HIT THE DETECTION CAP "
            f"({max_det:,}).")
        log("    Those fields are truncated. Ultralytics keeps the "
            "highest-scoring")
        log("    max_det boxes and discards the rest, so the cache holds a "
            "censored")
        log("    sample of the field, not the field. mag's yield gate checks "
            "the")
        log("    floor, never the ceiling --- a cache like this passes every "
            "gate.")
        if thr is not None:
            log("")
            log(f"    Lowest score still standing on a capped field: {thr:.4g}.")
            log(f"    Above theta = {thr:.4g} the truncation provably changed "
                f"nothing:")
            log("    everything cut was below it. At or below it, counts are "
                "censored")
            log("    --- and the density rules apply no score filter at all.")

        res = probe_cap(model, [q for n, q in zip(per_field, images)
                                if n >= max_det],
                        kw, max_det, int(probe), class_names, report, name,
                        log) if probe else {}
        out.update(res)
        # The cap fired and the probe measured what it cost. Under the declared
        # bar this is a caveat to write down, not a reason to refuse the
        # checkpoint --- refusing would drop it out of the comparison and out
        # of the run over a sub-2% shift in one rule family.
        if res.get("material") is False:
            notes.append(
                f"{n_sat}/{len(per_field)} fields hit max_det={max_det}; "
                f"probe measured {res.get('density_bias', 0):+.1%} R6/R6b "
                f"density shift, under the {MATERIAL_DENSITY_BIAS:.0%} bar")
        else:
            problems.append(f"{n_sat} of {len(per_field)} fields returned "
                            f"exactly {max_det} boxes (max_det)")

    # --- a constant box count on a YOLO ------------------------------------ #
    # Expected for a detection transformer: RT-DETR emits a fixed query budget,
    # the same on a parasite-dense field and a clean one. NOT expected for a
    # YOLO, whose dense anchor output varies with what is in the field. A
    # constant that EQUALS the cap is saturation, already reported above.
    lo, hi = (min(per_field), max(per_field)) if per_field else (0, 0)
    if (lo == hi and lo > 0 and lo != max_det
            and "detr" not in str(kind).lower() and "detr" not in name.lower()):
        problems.append(f"a YOLO returned exactly {lo} boxes on every field")
        log("")
        log(f"  ! A YOLO RETURNED EXACTLY {lo} BOXES ON EVERY FIELD.")
        log("    Dense anchor output varies with field content. A constant "
            "count")
        log("    means the output is being truncated or misread, not "
            "detected.")

    # --- the yield verdict, from mag's own rule ---------------------------- #
    verdict = None
    try:
        verdict = detect.implausible_yield(total, with_any, len(per_field),
                                           conf)
    except Exception:                                         # noqa: BLE001
        if total / max(len(per_field), 1) < FALLBACK_MIN_PER_FIELD:
            verdict = (f"{total} detections over {len(per_field)} fields at a "
                       f"floor of {conf}")
    if verdict:
        problems.append(f"implausible yield: {verdict}")
        log("")
        log(f"  ! IMPLAUSIBLE YIELD --- {verdict}")
        log("    At a 0.001 floor with suppression effectively off, a stained")
        log("    field should return hundreds of boxes. This is a loading or")
        log("    driving fault, not a weak detector.")
    return out


# --------------------------------------------------------------------------- #
# 4. Cross-checkpoint
# --------------------------------------------------------------------------- #

def agreement(report, log) -> None:
    """Do the models return comparable amounts on the same pixels?

    Not a quality measure --- they saw identical fields, so a large spread is
    about output conventions rather than detection. A detection transformer
    emits a fixed query budget per field; a YOLO emits dense anchor output.
    That difference is why checkpoint comparisons are restricted to the
    matched condition.
    """
    means = dict(VERIFY.SMOKE_PER_FIELD_MEAN.each(report))
    probs = dict(VERIFY.SMOKE_PROBLEMS.each(report))
    ok = {k: v for k, v in means.items() if not probs.get(k)}

    dropped = sorted(k for k in means if probs.get(k))
    if dropped:
        VERIFY.AGREE_EXCLUDED.put(report, dropped)
        # `.each()` on the whole document, NOT `.get()` on a sub-dict: the key
        # carries a wildcard and the schema refuses to resolve one positionally.
        # That refusal is the guard working -- it turned a would-be silent
        # mis-read into an exception -- but it belongs in a test, not a run.
        caps = dict(VERIFY.SMOKE_FIELDS_AT_CAP.each(report))
        for k in dropped:
            cap = caps.get(k) or 0
            why = (f"{cap} field(s) at the detection cap" if cap
                   else (probs[k][0] if probs[k] else ""))
            log(f"  - {k} left out of the comparison: {why}")

    if len(ok) >= 2:
        VERIFY.AGREE_PER_FIELD.put(report, ok)
        lo, hi = min(ok.values()), max(ok.values())
        if lo:
            VERIFY.AGREE_RATIO.put(report, round(hi / lo, 2))
            log("  boxes per field: "
                + ", ".join(f"{k} {v}" for k, v in sorted(ok.items())))
            log(f"  max:min ratio  {round(hi / lo, 2)}")

    fixed = []
    for k in VERIFY.SMOKE_PER_FIELD_MEAN.names(report):
        sub = report["smoke"][k]
        lo_, hi_ = sub.get("per_field_min"), sub.get("per_field_max")
        if lo_ and lo_ == hi_:
            fixed.append(k)
            log(f"  ! {k} returned exactly {lo_} boxes on every field")
            log("    That is a query budget, not a detection count. Any "
                "statistic")
            log("    that counts, averages or thresholds boxes measures the")
            log("    decoder's output convention before it measures the "
                "detector.")
    VERIFY.AGREE_FIXED.put(report, sorted(fixed))


# --------------------------------------------------------------------------- #
# 5. Report
# --------------------------------------------------------------------------- #

def write_md(report, path):
    g = lambda k, d="--": (k.get(report) if k.get(report) is not S.MISSING  # noqa: E731
                           else d)
    L = ["# Model verification", "",
         f"Archive: `{g(VERIFY.ARCHIVE)}`  ",
         f"`mag` {g(VERIFY.MAG_VERSION)}, `MAX_DET` {g(VERIFY.MAG_MAX_DET)}  ",
         f"Device: `{g(VERIFY.DEVICE)}`", "",
         f"**{g(VERIFY.VERDICT)}**", "",
         "| name | classes | loader | boxes/field | at cap | verdict |",
         "|---|---|---|---|---|---|"]
    for nm in sorted(report.get("smoke", {})):
        sm = report["smoke"][nm]
        ck = report.get("checkpoints", {}).get(nm, {})
        bad = (ck.get("problems") or []) + (sm.get("problems") or [])
        L.append(f"| {nm} | {ck.get('n_classes', '--')} | "
                 f"{sm.get('loader', '--')} | "
                 f"{sm.get('per_field_mean', '--')} | "
                 f"{sm.get('fields_at_cap', 0)}/{sm.get('n_images', '--')} | "
                 f"{'**FAIL**' if bad else 'ok'} |")
    L.append("")

    for nm in sorted(report.get("smoke", {})):
        sm = report["smoke"][nm]
        ck = report.get("checkpoints", {}).get(nm, {})
        L += [f"### {nm}", "", f"- `{ck.get('path')}`",
              f"- sha256 `{ck.get('sha256', '--')}`"]
        op = sm.get("operating_point") or {}
        if op:
            L.append(f"- driven at conf={op.get('conf')}, iou={op.get('iou')}, "
                     f"max_det={op.get('max_det')} (mag's own operating point)")
        if sm.get("score_median") is not None:
            L.append(f"- scores: median {sm['score_median']}, "
                     f"max {sm.get('score_max')}")
        cp = sm.get("cap_probe") or {}
        if cp:
            L.append(f"- cap probe at max_det={cp.get('probe_max_det'):,}: "
                     f"**{cp.get('discarded', 0):,} boxes discarded** "
                     f"({cp.get('discarded_share', 0):.1%}); "
                     f"R6/R6b density {cp.get('density_bias', 0):+.1%}")
            if cp.get("discarded_by_class"):
                L.append("- discarded tail: " + ", ".join(
                    f"{k} {v:,}" for k, v in
                    list(cp["discarded_by_class"].items())[:6]))
            if cp.get("suggested_max_det"):
                L.append(f"- a non-binding cap for this checkpoint: "
                         f"`MAX_DET = {cp['suggested_max_det']:,}`")
        for kind, items in (("Notes", sm.get("notes")),
                            ("Problems", (ck.get("problems") or [])
                             + (sm.get("problems") or []))):
            if items:
                L += ["", f"**{kind}:**", ""] + [f"- {x}" for x in items]
        L.append("")
    Path(path).write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="verify_models.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", "--archive", dest="root", default=".")
    p.add_argument("--weights", nargs="*", metavar="NAME=PATH", default=None)
    p.add_argument("--set", default="test")
    p.add_argument("--n", type=int, default=20, help="fields to drive")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--yolov5-repo", dest="yolov5_repo", default=None)
    p.add_argument("--probe-cap", dest="probe_cap", type=int, default=30000,
                   metavar="N",
                   help="when a checkpoint saturates mag.detect.MAX_DET, "
                        "re-run just the capped fields at this cap to measure "
                        "how many boxes it discards, and in which classes. "
                        "0 disables.")
    p.add_argument("--out", default="verify")
    p.add_argument("--no-inference", dest="no_inference", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    mag, detect, discover, ingest = _mag()

    print("=" * 78)
    print(" verify_models.py --- can these checkpoints detect?")
    print("=" * 78)

    d = discover.discover(Path(a.root), max_depth=6)
    report: dict = {}
    VERIFY.ARCHIVE.put(report, str(d.archive or a.root))
    VERIFY.MAG_VERSION.put(report, f"{mag.__version__} build {mag.build_id()}")
    VERIFY.GENERATED.put(report, dt.datetime.now(dt.timezone.utc)
                         .isoformat(timespec="seconds"))
    VERIFY.DEVICE.put(report, str(a.device))
    VERIFY.MAG_MAX_DET.put(report, int(getattr(detect, "MAX_DET", 3000)))

    weights = {}
    for spec in (a.weights or []):
        if "=" not in spec:
            print(f"\n--weights takes NAME=PATH; got {spec!r}", file=sys.stderr)
            return EXIT_PRECONDITION
        k, v = spec.split("=", 1)
        weights[k] = v
    weights = weights or dict(d.weights or {})
    if not weights:
        print(f"\nno checkpoints found under {a.root}.", file=sys.stderr)
        print("  Pass them:  --weights rtdetr=PATH yolov5x=PATH",
              file=sys.stderr)
        return EXIT_PRECONDITION

    print(f"  mag       {mag.__version__}  build {mag.build_id()}")
    print(f"  archive   {d.archive}")
    print(f"  weights   {', '.join(sorted(weights))}")
    print(f"  MAX_DET   {getattr(detect, 'MAX_DET', 3000):,}  "
          f"(mag's own; the pass will use it)")

    if not a.no_inference:
        hr("device")
        try:
            gpu = detect.gpu_budget(a.device)
            VERIFY.GPU.put(report, gpu)
            for k, v in gpu.items():
                print(f"  {k:<14}{v}")
        except Exception as exc:                              # noqa: BLE001
            print(f"  gpu_budget failed: {type(exc).__name__}: {exc}")

    expected = list(S.ALL_CLASSES)
    meta_by_name = {}
    for name in sorted(weights):
        hr(name)
        meta_by_name[name] = inspect_weights(detect, name, weights[name],
                                             expected, report, print,
                                             repo=a.yolov5_repo)

    if not a.no_inference:
        images = detect._image_paths(Path(d.archive) / "images" / a.set)[:a.n]
        if not images:
            print(f"\nno images under "
                  f"{Path(d.archive) / 'images' / a.set}", file=sys.stderr)
            return EXIT_PRECONDITION
        hr(f"inference on {len(images)} real fields")
        print(f"  from {images[0].parent.parent}")
        for name in sorted(weights):
            if meta_by_name[name].get("problems"):
                print(f"\n  {name}: skipped --- it failed a check above")
                continue
            print(f"\n  {name}")
            fmt = (meta_by_name[name].get("metadata") or {}).get("format")
            smoke_test(detect, name, weights[name], images, a.conf, a.device,
                       a.imgsz, report, print, fmt=fmt, repo=a.yolov5_repo,
                       probe=a.probe_cap, class_names=expected)

        hr("agreement on identical fields")
        agreement(report, print)

        try:
            n_fields = int(ingest.build_patients(
                ingest.read_image_list(d.image_list), a.set)["n_images"].sum())
        except Exception:                                     # noqa: BLE001
            n_fields = None
        if n_fields:
            hr("projected full pass")
            VERIFY.PROJ_FIELDS.put(report, n_fields)
            total = 0.0
            for nm, spf in VERIFY.SMOKE_SECONDS_PER_FIELD.each(report):
                mins = spf * n_fields / 60
                total += mins
                print(f"  {nm:<10} {mins:>7.1f} min over {n_fields:,} fields")
            VERIFY.PROJ_TOTAL_MIN.put(report, round(total, 1))
            print(f"  {'total':<10} {total:>7.1f} min ({total / 60:.1f} h)")

    # --- verdict ------------------------------------------------------------ #
    failed = sorted({n for n, p in VERIFY.CK_PROBLEMS.each(report) if p}
                    | {n for n, p in VERIFY.SMOKE_PROBLEMS.each(report) if p})
    noted = sorted({n for n, x in VERIFY.SMOKE_NOTES.each(report)
                    if x and n not in failed})
    hr("verdict")
    if failed:
        verdict = (f"{len(failed)} of {len(weights)} checkpoint(s) failed: "
                   f"{', '.join(failed)}")
        print(f"  {verdict}")
        capped = [n for n in failed
                  if (report["smoke"].get(n, {}).get("fields_at_cap") or 0)]
        silent = [n for n in failed if n not in capped]
        # Two failures, two consequences. A silent checkpoint produces a cache
        # the yield gate REFUSES --- expensive, but loud. A saturated one
        # produces a cache every gate ACCEPTS: complete, well-formed, correctly
        # hashed, and censored.
        if silent:
            print(f"\n  {', '.join(silent)}: do not run `mag detect` against "
                  f"{'them' if len(silent) > 1 else 'it'}. The pass would "
                  f"produce a")
            print("  cache the yield gate then refuses, and a refused cache is "
                  "absent")
            print("  from every table in the paper.")
        if capped:
            print(f"\n  {', '.join(capped)}: the opposite problem. A capped "
                  f"pass produces a")
            print("  cache every gate ACCEPTS --- complete, well-formed, "
                  "correctly")
            print("  hashed, and censored at the top. Settle the cap before "
                  "spending")
            print("  the pass.")
    else:
        verdict = f"all {len(weights)} checkpoint(s) are safe to run"
        print(f"  {verdict}")
        print("\n  Class maps match index for index, and every model returned "
              "a")
        print("  plausible number of boxes at mag's own operating point.")
    for n in noted:
        for x in report["smoke"][n]["notes"]:
            print(f"  note: {n}: {x}")
    VERIFY.VERDICT.put(report, verdict)

    hr("schema")
    problems = VERIFY.validate(report)
    if problems:
        print(f"  ! {len(problems)} key(s) do not conform to "
              f"exp_schema.VERIFY:")
        for p in problems[:12]:
            print(f"      {p}")
    else:
        print(f"  {len(VERIFY.keys)} declared keys, all conforming")

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify.json").write_text(
        json.dumps(report, indent=1, default=str), encoding="utf-8")
    write_md(report, out / "VERIFY.md")
    print(f"\n  {out / 'VERIFY.md'}")
    print(f"  {out / 'verify.json'}")
    return EXIT_FAILED if failed else EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
