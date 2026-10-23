#!/usr/bin/env python3
r"""run_experiment.py --- the experiment, start to finish.

    python run_experiment.py --root ..\malaria_france_data_set

Runs, in order:

    preflight    read eda.json and verify.json; report what they establish
    ingest       mag ingest
    detect       mag detect, at a declared cap
    analyse      mag all, plus the macros mag does not emit
    sensitivity  the cap arm, when a checkpoint saturated the cap
    audit        mag audit, plus the v2 -> v3 wording guard
    manuscript   the findings ledger and every number the paper cites

IT REPORTS; IT DOES NOT REFUSE

Preflight prints everything the two offline checks establish, writes it to
`results/findings.csv`, and carries on. Nothing needs a flag to get past. A
check that halts a pipeline gets disabled within a week; a check that reports
gets read. The severities say how loudly, not whether:

    BLOCK    the manuscript is wrong without this sentence
    DECLARE  the manuscript must state it
    CAVEAT   the manuscript should state it

The one exception is the detection cap, and only because it is not a statement
about the archive --- it is a setting for the stage about to run. Detecting at
a cap the probe says will bind spends an hour producing a CENSORED cache that
every gate downstream accepts: complete, well-formed, correctly hashed, and
wrong. So `--max-det` defaults to a value that clears the probe's measurement,
and a binding cap is a loud warning rather than a silent default.

NOTHING HERE IS REIMPLEMENTED

Every stage calls `mag.cli.main()` or the functions underneath it, exactly as
`run_local.py` does. The two places this does more are documented hooks:
`report.write(..., extra=...)` for the derived macros, and a recorded
module-level override of `detect.MAX_DET` for the cap. Neither edits mag.

Every key it reads from `eda.json` or `verify.json` goes through
`exp_schema`, so a consumer looking for a key no producer writes is a failing
test rather than a puzzle on someone's archive.

EXIT CODES

    0   every stage completed
    2   a precondition is not met; an input or an earlier stage is missing
    3   a refusal the pipeline anticipated, WITH A REMEDY. Often a FINDING:
        the reconstruction gate refusing is a result about the released
        artefacts, not a fault in this code.
    5   the audit failed a check (mag's own code)
    4   anything else
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import exp_schema as S                                        # noqa: E402
import experiment_findings as EF                              # noqa: E402
from exp_schema import EDA, VERIFY, MISSING                   # noqa: E402


def hush_amp_deprecations() -> None:
    """Silence the yolov5 fork's per-batch torch.cuda.amp FutureWarnings.

    The fork calls `torch.cuda.amp.autocast` inside AutoShape once per batch
    and torch 2.7 deprecated it. It would normally print once, but libraries
    around it open `warnings.catch_warnings()` blocks, and each exit
    invalidates the "already shown here" registry, so it reprints per batch:
    over 4,508 fields that is thousands of identical two-line blocks between
    mag's progress lines, and the first complete run had to be interrupted
    because of it. Only this exact deprecation is filtered; every other
    FutureWarning still prints. MALARIA_SHOW_WARNINGS=1 restores them.

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

EXIT_OK, EXIT_PRECONDITION, EXIT_REFUSAL, EXIT_UNEXPECTED = 0, 2, 3, 4
EXIT_AUDIT = 5

STAGES = ("preflight", "ingest", "detect", "analyse", "sensitivity",
          "audit", "manuscript")

#: What `--max-det` falls back to when no probe measurement is available.
#: mag's own MAX_DET is 3000 and the real archive needs more; this is not a
#: number anyone should rely on, which is why preflight prefers the probe's.
DEFAULT_MAX_DET = 6580


def hr(t=""):
    print("\n" + ("--- " + t + " " if t else "").ljust(78, "-"))


def say(*a):
    print(*a, flush=True)


def _read_json(path):
    p = Path(path)
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


# --------------------------------------------------------------------------- #
# 1. mag, and the version it must be
# --------------------------------------------------------------------------- #

def require_mag():
    try:
        import mag
    except ImportError as exc:
        say(f"\nmag is not installed in {sys.executable} ({exc}).\n")
        say(f'    "{sys.executable}" -m pip install -e .')
        raise SystemExit(EXIT_PRECONDITION)
    major = str(getattr(mag, "__version__", "0")).split(".")[0]
    if major != "3":
        say(f"\nthis script targets mag 3.x and found {mag.__version__}.\n")
        say("  1.x and 3.x differ in ways that move NUMBERS, not just names:\n"
            "    BASELINE/CONFIRMATORY_RULE -> COMPARATOR/CANDIDATE\n"
            "    reproduction -> reconstruction (v3 SS3)\n"
            "    the apportionment -> a patient-clustered GEE (v3 SS9)")
        raise SystemExit(EXIT_PRECONDITION)
    return mag


# --------------------------------------------------------------------------- #
# 2. The macros mag 3.0.0 does not emit.
#
# Each is a composition of numbers mag already computes, or a fact about this
# run that the Study cannot know. They return {} rather than a placeholder when
# an input is absent, because mag renders an undefined \R{} RED in the PDF --
# and a placeholder renders as a value, which is the failure the \R mechanism
# exists to prevent.
# --------------------------------------------------------------------------- #

DERIVED_KEYS = ("jackknifeRange", "errOvaleVivaxShare", "riskConfCov90",
                "aurcConf", "nArmsAnalysed", "reconstructionShortfall",
                "capMaxDet", "capBinding", "capDensityBias", "capDiscarded",
                "enrichmentShare", "thinSpeciesList", "uninfectedCentreList",
                "cropModelGap")


def _pct(x, dp=1):
    if x is None or x is MISSING or (isinstance(x, float) and math.isnan(x)):
        return None
    return f"{100 * float(x):.{dp}f}\\%"


def _num(x, dp=4):
    if x is None or x is MISSING or (isinstance(x, float) and math.isnan(x)):
        return None
    return f"{float(x):.{dp}f}"


def derived_macros(s, ctx: dict | None = None) -> dict:
    """The derived macros, from a built Study plus this run's own record."""
    out: dict = {}
    ctx = ctx or {}

    # --- jackknifeRange ----------------------------------------------------- #
    # v3 SS9.3 keeps the descriptive ranges alongside the GEE attribution and
    # leaves each checkpoint out in turn. Needs >= 2 refits: a "range" over one
    # value is a point, and printing it as a range overstates what was assessed.
    ap = getattr(s, "apportionment", None) or {}
    diffs = [v["spread_rule"] - v["spread_checkpoint"]
             for v in (ap.get("leave_one_checkpoint_out") or {}).values()
             if v.get("spread_rule") is not None
             and v.get("spread_checkpoint") is not None]
    if len(diffs) >= 2:
        out["jackknifeRange"] = (f"{100 * min(diffs):.1f} to "
                                 f"{100 * max(diffs):.1f} percentage points")

    # --- errOvaleVivaxShare: the count, as a share -------------------------- #
    es = getattr(s, "error_summary", None) or {}
    if es.get("n_errors"):
        share = _pct((es.get("categories") or {}).get("ovale_vivax", 0)
                     / es["n_errors"])
        if share:
            out["errOvaleVivaxShare"] = share

    # --- riskConfCov90 / aurcConf ------------------------------------------- #
    # mag keys these by rule id because the candidate is chosen by the design
    # gate and its id is not known until that has run. Resolved from the
    # study's own confirmatory record, NEVER hard-coded: a hard-coded R4 would
    # print the wrong rule's risk the moment the selection changed.
    rid = (getattr(s, "confirmatory", None) or {}).get("rule")
    rc = (getattr(s, "selective", None) or {}).get(rid)
    if rid and isinstance(rc, dict):
        if rc.get("aurc") is not None:
            out["aurcConf"] = _num(rc["aurc"])
        op = (rc.get("operating") or {}).get("0.9")
        if op and op.get("risk") is not None:
            out["riskConfCov90"] = (f"{_pct(op['risk'])} with "
                                    f"{op['n_referred']} patients referred")

    fac = getattr(s, "factorial", None)
    if fac is not None and len(fac):
        out["nArmsAnalysed"] = str(fac["checkpoint"].nunique())

    # --- reconstructionShortfall -------------------------------------------- #
    # v3 SS3: a RECONSTRUCTION, not a reproduction. The manuscript states the
    # gap as a number of patients; subtracting two macros in LaTeX is a place
    # for arithmetic to be wrong.
    detail = getattr(getattr(s, "reconstruction", None), "detail", None) or {}
    if detail.get("n_correct") is not None and detail.get("published") is not None:
        gap = int(detail["n_correct"]) - int(detail["published"])
        out["reconstructionShortfall"] = (
            f"{abs(gap)} patient{'s' if abs(gap) != 1 else ''}"
            + (" above" if gap > 0 else " below" if gap < 0 else ""))

    # --- facts about THIS run, which the Study cannot know ------------------ #
    if ctx.get("max_det") is not None:
        out["capMaxDet"] = f"{int(ctx['max_det']):,}"
    if ctx.get("cap_binding") is not None:
        out["capBinding"] = ("yes --- the cache is censored"
                             if ctx["cap_binding"] else
                             "no field reached the cap")
    if ctx.get("density_bias") is not None:
        out["capDensityBias"] = f"{100 * float(ctx['density_bias']):+.1f}\\%"
    if ctx.get("discarded_share") is not None:
        out["capDiscarded"] = _pct(ctx["discarded_share"])
    if ctx.get("enrichment_share") is not None:
        out["enrichmentShare"] = _pct(ctx["enrichment_share"])
    if ctx.get("thin_species"):
        out["thinSpeciesList"] = ", ".join(
            f"{k} ({v})" for k, v in sorted(ctx["thin_species"].items()))
    if ctx.get("uninfected_centres"):
        out["uninfectedCentreList"] = ", ".join(ctx["uninfected_centres"])
    if ctx.get("crop_gaps"):
        # `:+.1%` writes a bare per-cent sign, which starts a comment in TeX
        # and would swallow the rest of the line if the manuscript ever called
        # this macro. The hazard scan only reads CALLED macros, so this was a
        # defect waiting for its first citation rather than one it would catch.
        out["cropModelGap"] = "; ".join(
            f"{k} {100 * v:+.1f}\\%" for k, v in
            sorted(ctx["crop_gaps"].items()))
    return out


# --------------------------------------------------------------------------- #
# 3. The v2 -> v3 manuscript guard.
# --------------------------------------------------------------------------- #

FORBIDDEN = (
    (r"rule\s+matters\s+more\s+than\s+the\s+detector",
     "SS9.4 forbids this phrasing outright"),
    (r"\b(twelve|ten|eleven|\d+)\s*times\s+(more|larger|greater)",
     "SS9.4 forbids any multiplier computed from a ratio of ranges"),
    (r"not\s+separately\s+estimable",
     "SS9.1: this v2 claim is false. Every patient contributes an outcome "
     "under every cell -- 170 repeated observations per cell, not one"),
    (r"\bapportion(ment|ed|ing)\b",
     "SS9 replaced the apportionment with a patient-clustered GEE; the word "
     "should not survive into v3 prose"),
    (r"\breproduc(tion|ed)\s+gate\b",
     "SS3: it is the RECONSTRUCTION gate. The published result is external"),
    (r"accumulation[^.]{0,40}across\s+fields\s+adds\s+(almost\s+)?nothing",
     "v4.1: the comparator is a CONSTANT function on this cache -- it calls "
     "every patient infected -- so its flat field-count curve is saturation "
     "and carries no evidence about what accumulating over fields is worth"),
    (r"five\s+fields\s+buy\s+nearly\s+what\s+twenty-five\s+buy",
     "v4.1: same retracted reading of a constant comparator's flat curve"),
    (r"helped\s+the\s+comparator\s+imperceptibly",
     "v4.1: same retracted reading; a constant rule is not helped or hindered"),
)

#: Wording that asserts something the identifiability finding forbids ---
#: applied only when THIS cohort has that problem, so the rule is a
#: consequence of the data rather than a universal asserted from nowhere.
FORBIDDEN_IF_UNIDENTIFIABLE = (
    (r"specificit(y|ies)\s+(by|per|across|stratified\s+by)\s+"
     r"(centre|center|site|domain|hospital)",
     "specificity is perfectly collinear with centre in this cohort"),
    (r"(centre|center|site|domain)-(stratified|specific|wise)\s+specificit",
     "specificity is perfectly collinear with centre in this cohort"),
)

#: v3 still defines these but retracted their v2 READING. Not an error to cite
#: them -- SS9.3 keeps them as descriptive ranges -- but they must not carry
#: the causal claim they carried in v2.
DEMOTED = ("spreadRule", "spreadCkpt", "ruleOverCkpt", "permNullRule",
           "apportionmentSentence")


#: Constructs whose ARGUMENT is a name rather than prose: a file path, a
#: cross-reference, a label. The forbidden-wording check must not read them.
#: `results/figures/apportionment` is the filename mag writes and cannot be
#: renamed from here; matching the word inside it failed a build over a path.
NOT_PROSE = (
    r"\\resultfigure\{[^}]*\}\{[^}]*\}\{[^}]*\}",
    r"\\includegraphics(\[[^]]*\])?\{[^}]*\}",
    r"\\(input|include|label|ref|eqref|cite|graphicspath|texttt|url)\{[^}]*\}",
    r"\\R\{[A-Za-z0-9]+\}",
    r"(?<!\\)%.*",
)


def check_manuscript(path, unidentifiable: bool = False) -> list:
    p = Path(path)
    if not p.is_file():
        return [("missing", f"no manuscript at {p}", "")]
    raw = p.read_text(encoding="utf-8", errors="replace")
    # Blank the non-prose regions rather than delete them, so a reported line
    # number still points at the line the author has to edit.
    text = raw
    for pattern in NOT_PROSE:
        text = re.sub(pattern,
                      lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)
    found = []
    patterns = list(FORBIDDEN)
    if unidentifiable:
        patterns += list(FORBIDDEN_IF_UNIDENTIFIABLE)
    for pattern, why in patterns:
        for m in re.finditer(pattern, text, re.I):
            line = text[:m.start()].count("\n") + 1
            found.append(("forbidden", f"line {line}: {m.group(0)!r}", why))
    cited = set(re.findall(r"\\R\{([A-Za-z0-9]+)\}", raw))
    for k in sorted(cited & set(DEMOTED)):
        found.append(("demoted", f"\\R{{{k}}}",
                      "v3 SS9.3 keeps this as a DESCRIPTIVE range; it must "
                      "not carry the v2 causal reading"))
    return found


# --------------------------------------------------------------------------- #
# 4. Context --- what this run knows that the Study does not
# --------------------------------------------------------------------------- #

def load_context(a) -> None:
    """Read the two artefacts once, evaluate the ledger, cache on `a`.

    Lazy and idempotent, so `--stages analyse` produces the same macros as
    `--stages preflight analyse`. Without this, a macro renders red because of
    stage selection rather than because the analysis could not compute it, and
    a red macro that is a puzzle gets resolved by deleting the \\R{}.
    """
    if getattr(a, "_ledger", None) is not None:
        return
    eda = _read_json(Path(a.eda) / "eda.json") or {}
    verify = _read_json(Path(a.verify) / "verify.json") or {}
    a._eda, a._verify = eda, verify
    a._ledger = EF.ledger(eda, verify)
    a._have_inputs = bool(eda) and bool(verify)

    need = 0
    for _, v in VERIFY.PROBE_SUGGESTED.each(verify):
        need = max(need, int(v))
    a._cap_need = need

    bias = None
    for _, v in VERIFY.PROBE_DENSITY_BIAS.each(verify):
        if bias is None or abs(v) > abs(bias):
            bias = v
    share = None
    for _, v in VERIFY.PROBE_DISCARDED_SHARE.each(verify):
        if share is None or v > share:
            share = v

    thin, gaps = {}, {}
    for r in a._ledger:
        if r["key"] == "rare_classes" and r["fired"]:
            thin = (r["data"] or {}).get("thin") or {}
        if r["key"] == "crop_model" and r["fired"]:
            gaps = (r["data"] or {}).get("gaps") or {}

    centres = EDA.IDENT_CENTRES.get(eda)
    a._context = {
        "max_det": a.max_det,
        "cap_binding": bool(need and a.max_det < need),
        "density_bias": bias,
        "discarded_share": share,
        "enrichment_share": (None if EDA.LOAD_ENRICHMENT.get(eda) is MISSING
                             else EDA.LOAD_ENRICHMENT.get(eda)),
        "thin_species": thin,
        "crop_gaps": gaps,
        "uninfected_centres": ([] if centres is MISSING else list(centres)),
    }


# --------------------------------------------------------------------------- #
# 5. Preflight --- report, then continue
# --------------------------------------------------------------------------- #

MARK = {True: "FIRED", False: "clear"}


def stage_preflight(a, d) -> int:
    load_context(a)
    eda, verify, rows = a._eda, a._verify, a._ledger

    if not a._have_inputs:
        missing = []
        if not eda:
            missing.append(f"  python eda_dataset.py --root {a.root} "
                           f"--out {a.eda}")
        if not verify:
            missing.append(f"  python verify_models.py --root {a.root} "
                           f"--out {a.verify}")
        say("  ! the offline checks have not run, or their JSON is "
            "unreadable:")
        for m in missing:
            say(m)
        say("")
        say("  The run continues. Every finding below will report UNKNOWN,")
        say("  which is not the same as clear --- and several macros the")
        say("  manuscript cites will render RED rather than wrong.")

    # --- the schema those files were supposed to conform to ----------------- #
    for name, doc, sch in (("eda.json", eda, EDA),
                           ("verify.json", verify, VERIFY)):
        if not doc:
            continue
        probs = sch.validate(doc)
        if probs:
            say(f"  ! {name} does not conform to exp_schema in "
                f"{len(probs)} place(s):")
            for p in probs[:6]:
                say(f"      {p}")
            if len(probs) > 6:
                say(f"      ... and {len(probs) - 6} more")

    hr("findings ledger")
    for r in rows:
        mark = "  ?  " if r["unknown"] else MARK[r["fired"]]
        say(f"  [{mark}] {r['severity']:<7} {r['key']:<18} {r['title']}")
        if r["detail"]:
            say(f"            {r['detail']}")

    n_fired = len(EF.fired(rows))
    n_block = len(EF.blocking(rows))
    unk = EF.unknown(rows)
    say("")
    say(f"  {n_fired} of {len(rows)} findings fired, {n_block} at BLOCK "
        f"severity.")
    if unk:
        say(f"  {len(unk)} could not be checked. A finding that was not "
            f"checked is not")
        say("  a finding that cleared:")
        for r in unk:
            say(f"      {r['key']}: {r['detail']}")
    say("")
    say("  None of these stops the run. They are limits on what the cohort "
        "and")
    say("  the released artefacts can support, and the manuscript must state")
    say("  them --- results/findings.csv carries each one with the sentence "
        "it")
    say("  owes and the section it belongs in.")

    # --- the cap, which IS a setting for the next stage --------------------- #
    if a._cap_need:
        hr("detection cap")
        for name, v in VERIFY.PROBE_SUGGESTED.each(verify):
            mx = dict(VERIFY.PROBE_TRUE_MAX.each(verify)).get(name, "?")
            nf = dict(VERIFY.PROBE_N_FIELDS.each(verify)).get(name, "?")
            say(f"  {name} needs >= {int(v):,}  (true max {mx:,} over {nf} "
                f"sampled fields)")
        say(f"  configured --max-det {a.max_det:,}")
        if a.max_det < a._cap_need:
            say("")
            say("  ! THE CONFIGURED CAP WILL BIND.")
            say(f"    At max_det={a.max_det:,} some fields will be truncated. "
                f"Unlike the")
            say("    findings above, this is not a fact about the archive --- "
                "it is a")
            say("    setting for the stage about to run, and a truncated "
                "cache passes")
            say("    every gate mag has: the manifest, the shard count and the "
                "schema")
            say("    are all satisfied by a censored table.")
            say("")
            say(f"      python run_experiment.py --root {a.root} "
                f"--max-det {a._cap_need}")
            say("")
            say("    Continuing at this cap; the censoring is recorded in "
                "capBinding.")
        else:
            say(f"  {a.max_det:,} clears it. No field should reach the cap.")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# 6. Stages that are mag's, unmodified
# --------------------------------------------------------------------------- #

def _weights_args(a, d) -> list:
    if a.weights:
        return list(a.weights)
    return [f"{k}={v}" for k, v in sorted((d.weights or {}).items())]


def stage_ingest(a, d) -> int:
    from mag import cli as mag_cli

    if not d.image_list:
        say(f"\nno image_list found under {a.root}. Point at it:")
        say(f"  python run_experiment.py --root {a.root} --image-list PATH")
        return EXIT_PRECONDITION
    argv = ["ingest", "--image-list", str(d.image_list),
            "--set", a.set, "--zoom-set", a.zoom_set,
            "--out", str(a.out), "--min-fields", str(a.min_fields)]
    if d.archive:
        argv += ["--archive", str(d.archive)]
    if d.classes:
        argv += ["--classes", str(d.classes)]
    if a.skip_layout:
        argv.append("--skip-layout")
    return _mag_call(mag_cli, argv)


def _class_family(detect, weights):
    """Which family the pickled model class belongs to, or None.

    `models.yolo.*` is the repository format; `ultralytics.nn.*` is the
    package format. Read from the object itself, so no import order can move
    the answer.
    """
    try:
        import torch
        with detect._full_load():
            ck = torch.load(weights, map_location="cpu", weights_only=False)
    except Exception:                                         # noqa: BLE001
        return None
    model = ck.get("model") if isinstance(ck, dict) else None
    root = (type(model).__module__ or "").split(".")[0] if model is not None \
        else ""
    return {"models": "yolov5-repo", "ultralytics": "ultralytics"}.get(root)


def install_format_guard() -> None:
    """Make mag's checkpoint-format detection independent of process state.

    `detect.run` reads the format itself and decides BY EXCEPTION: a file is
    "yolov5-repo" only when unpickling raises ModuleNotFoundError for
    `models`/`utils`. `_require_yolov5_modules` puts those on sys.path and,
    on success, leaves them there. So the moment ANYTHING in the process has
    loaded a YOLOv5 checkpoint --- the verify stage doing its job, twenty
    fields earlier --- every later read returns "ultralytics", `load_model`
    hands the v5 pickle to ultralytics' YOLO, which accepts it without error,
    and a 26-minute pass caches 2 detections across 4,508 fields.

    That is not hypothetical. It is what the first complete run did.

    This wraps `read_checkpoint_metadata` so the format is reconciled against
    the pickled model class before mag uses it. It is a runtime override of a
    library this suite otherwise only calls: the alternative is to run every
    checkpoint in its own process, which the driver now also does, and this
    is the belt to that pair of braces. It is recorded in run_state.json.
    """
    from mag import detect
    orig = detect.read_checkpoint_metadata
    if getattr(orig, "_format_guard", False):
        return

    def guarded(weights, repo=None):
        meta = orig(weights, repo)
        actual = _class_family(detect, weights)
        declared = meta.get("format")
        if actual and declared and actual != declared:
            say(f"  ! format guard: {Path(weights).parent.name} declares "
                f"{declared!r} but its pickled model class is {actual!r}.")
            say("    Using the class. mag decides the format by whether an "
                "unpickle")
            say("    raises, and something earlier in this process already "
                "imported")
            say("    the yolov5 modules, which prevents the raise. Loading a "
                "v5")
            say("    pickle through ultralytics' YOLO returns garbage.")
            meta["declared_format"] = declared
            meta["format"] = actual
        return meta

    guarded._format_guard = True
    detect.read_checkpoint_metadata = guarded


def set_cap(value: int) -> int:
    """Raise mag's detection cap for this process, and return the old one.

    `_infer_group` reads the module global at call time --- `max_det=MAX_DET`
    inside the function body, not a default bound at definition --- so setting
    the attribute is enough and no source is edited. It is recorded in the run
    state because a cache built at a different cap is a different cache, and
    nothing in the manifest's content hash would show it.
    """
    from mag import detect
    was = int(getattr(detect, "MAX_DET", 3000))
    detect.MAX_DET = int(value)
    return was


def _detect_argv(a, d, images, out, weights) -> list:
    argv = ["detect", "--images", str(images), "--weights", *weights,
            "--out", str(out), "--conf", str(a.conf),
            "--device", a.device, "--seed", str(a.seed)]
    patients = Path(a.out) / "patients.csv"
    if patients.is_file():
        argv += ["--patients", str(patients)]
    if a.imgsz:
        argv += ["--imgsz", str(a.imgsz)]
    if a.batch:
        argv += ["--batch", str(a.batch)]
    if a.yolov5_repo:
        argv += ["--yolov5-repo", str(a.yolov5_repo)]
    if a.assume_classes_txt:
        argv.append("--assume-classes-txt")
    if a.no_resume:
        argv.append("--no-resume")
    if a.half is True:
        argv.append("--half")
    elif a.half is False:
        argv.append("--no-half")
    return argv


def purge_suspect(cache: Path, log=say) -> list:
    """Delete the shards of any checkpoint mag marked suspect.

    `detect` RESUMES from `_shards/<ck>/`, and `_usable_shards` decides
    reusability from a fingerprint over the checkpoint sha, confidence floor,
    image size and class map. None of those changes when the LOADER was
    wrong. So a pass that cached 2 detections across 4,508 fields through the
    wrong model class leaves shards that a re-run happily reuses --- the fix
    lands, the run completes in seconds, and the cache is still garbage.

    A `.suspect.json` flag means mag's own yield gate refused that cache.
    Its shards are evidence of a bug, not of the detector, and they must not
    be resumed from.
    """
    purged = []
    for flag in sorted(Path(cache).glob("detections_*.suspect.json")):
        ck = flag.name[len("detections_"):-len(".suspect.json")]
        try:
            why = json.loads(flag.read_text(encoding="utf-8")).get("reason", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            why = ""
        sdir = Path(cache) / "_shards" / ck
        n = len(list(sdir.glob("*.parquet"))) if sdir.is_dir() else 0
        if sdir.is_dir():
            shutil.rmtree(sdir)
        for p in (Path(cache) / f"detections_{ck}.parquet",
                  Path(cache) / f"manifest_{ck}.json", flag):
            if p.exists():
                p.unlink()
        purged.append(ck)
        log(f"  purged the suspect cache for {ck} ({n} shards)")
        if why:
            log(f"    mag refused it: {why}")
        log("    Resume matches on the checkpoint hash, the floor, the image "
            "size and")
        log("    the class map --- none of which records WHICH LOADER ran. "
            "Without")
        log("    this the re-run would reuse the bad shards and finish in "
            "seconds.")
    return purged


def record_failure(results, stage: str, exc: BaseException) -> None:
    """Persist WHY a stage refused, the moment it refuses.

    The handback bundle is assembled from files, and the only record of the
    first real run's refusal was a line in the console log --- which was
    zipped one buffer short of complete. The bundle came back saying detect
    had returned 3 after eighteen minutes and nothing at all about why.

    A refusal is now a file, written before anything else can go wrong, and
    the bundle carries it whatever happens to the log.
    """
    import traceback
    p = Path(results)
    try:
        p.mkdir(parents=True, exist_ok=True)
        f = p / "stage_failures.json"
        try:
            prior = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError, UnicodeDecodeError):
            prior = []
        if not isinstance(prior, list):
            prior = []
        prior.append({
            "stage": stage,
            "utc": dt.datetime.now(dt.timezone.utc)
                     .isoformat(timespec="seconds"),
            "type": type(exc).__name__,
            "message": str(exc),
            "remedy": str(getattr(exc, "remedy", "") or ""),
            "traceback": traceback.format_exc(),
        })
        f.write_text(json.dumps(prior, indent=1), encoding="utf-8")
        say(f"  (recorded in {f})")
    except OSError as err:                                    # noqa: BLE001
        say(f"  ! could not record the failure: {err}")


def sanitise_geometry(cache: Path, log=say, conf_floor=None,
                      seed=None) -> dict:
    """Remove boxes of non-positive extent, counted and kept, not dropped.

    `mag.schema.validate_detections` REFUSES a cache containing any box with
    w <= 0 or h <= 0, and it is right to: such a box has zero area, so the IoU
    matrix clips it and it neither suppresses nor is suppressed. It therefore
    passes through the matched condition unchallenged --- the one thing that
    condition exists to prevent --- and inflates the suppressed arm relative
    to the raw one.

    So they cannot stay. But they must not merely vanish either: the count is
    evidence about the detector at a 0.001 floor with suppression off, and the
    manuscript has to be able to state it. Every excluded row is written to
    `excluded_geometry_<ck>.parquet` beside the cache, with a JSON summary by
    class and score band.

    Shards are filtered with the table. `provenance.inspect` compares the
    shard sum against the assembled table, so filtering one and not the other
    trades a schema refusal for a provenance refusal.
    """
    import pandas as pd
    from mag import cache as mcache

    out = {}
    shard_root = Path(cache) / "_shards"
    if not shard_root.is_dir():
        return out
    # Iterate the SHARDS, not the manifests. `cache.save` validates on write,
    # so a checkpoint whose pass produced degenerate boxes has NO manifest and
    # NO assembled table -- it raised at the end of inference, with every
    # shard already on disk. Working from the shards therefore also recovers
    # that pass instead of re-running it: the expensive part is done, and only
    # the assembly failed.
    for sdir in sorted(p for p in shard_root.iterdir() if p.is_dir()):
        ck = sdir.name
        shards = sorted(sdir.glob("*.parquet"))
        if not shards:
            continue
        df = pd.concat([pd.read_parquet(q) for q in shards],
                       ignore_index=True)
        bad = (df["w"] <= 0) | (df["h"] <= 0)
        n_bad = int(bad.sum())
        assembled = (Path(cache) / f"manifest_{ck}.json").is_file()
        rec = {"checkpoint": ck, "n_total": int(len(df)),
               "n_excluded": n_bad, "was_assembled": assembled,
               "share": round(n_bad / max(len(df), 1), 6)}
        if not n_bad:
            if not assembled:
                say(f"  ! {ck}: shards are clean but no manifest exists; "
                    f"re-assembling")
                _reassemble(cache, ck, df, mcache, log, conf_floor, seed)
                rec["reassembled"] = True
            out[ck] = rec
            continue

        excl = df[bad]
        rec["by_class"] = {str(k): int(v) for k, v in
                           excl["cls"].value_counts().items()}
        rec["score_min"] = float(excl["score"].min())
        rec["score_max"] = float(excl["score"].max())
        rec["score_median"] = float(excl["score"].median())
        rec["above_0_25"] = int((excl["score"] >= 0.25).sum())
        rec["zero_w"] = int((df["w"] <= 0).sum())
        rec["zero_h"] = int((df["h"] <= 0).sum())
        rec["n_fields_affected"] = int(excl["image_id"].nunique())
        excl.to_parquet(Path(cache) / f"excluded_geometry_{ck}.parquet",
                        index=False)

        log("")
        log(f"  ! {ck}: {n_bad:,} of {len(df):,} boxes have w <= 0 or h <= 0 "
            f"({n_bad / max(len(df), 1):.3%}),")
        log(f"    over {rec['n_fields_affected']:,} fields. By class: "
            + ", ".join(f"{k} {v:,}" for k, v in
                        sorted(rec["by_class"].items(),
                               key=lambda kv: -kv[1])[:6]))
        log(f"    their scores run {rec['score_min']:.4g} to "
            f"{rec['score_max']:.4g} (median {rec['score_median']:.4g}); "
            f"{rec['above_0_25']:,} above 0.25.")
        log("    A zero-area box is invisible to suppression and would "
            "survive the")
        log("    matched condition unchallenged, so it cannot enter the "
            "factorial.")
        log(f"    Excluded, and written to excluded_geometry_{ck}.parquet "
            f"with this")
        log("    accounting, so the exclusion is reported rather than "
            "silent.")

        # shards first, so their sum still equals the assembled table:
        # `provenance.inspect` compares the two, and filtering one without the
        # other trades a schema refusal for a provenance refusal.
        n_shard_rows = 0
        for q in shards:
            s = pd.read_parquet(q)
            keep = s[(s["w"] > 0) & (s["h"] > 0)].reset_index(drop=True)
            n_shard_rows += len(keep)
            keep.to_parquet(q, index=False)
        clean = df[~bad].reset_index(drop=True)
        _reassemble(cache, ck, clean, mcache, log, conf_floor, seed)
        rec["n_kept"] = int(len(clean))
        rec["shard_rows_after"] = n_shard_rows
        if n_shard_rows != len(clean):
            log(f"    ! shards now hold {n_shard_rows:,} rows against "
                f"{len(clean):,} in the table; provenance will refuse this.")
        out[ck] = rec

    if out:
        (Path(cache) / "geometry_exclusions.json").write_text(
            json.dumps(out, indent=1), encoding="utf-8")
    return out


def _reassemble(cache, ck, df, mcache, log=say,
                conf_floor=None, seed=None) -> None:
    """Write the table and manifest from the shards, via mag's own save.

    Most detection parameters come from `checkpoint_meta_<ck>.json`, which
    `detect.run` writes BEFORE inference and which therefore survives a pass
    that failed at assembly. It does NOT record the confidence floor or the
    seed, so those are passed in from this run's own arguments; falling back
    to mag's defaults would write a manifest describing a pass that never
    happened.
    """
    meta_path = Path(cache) / f"checkpoint_meta_{ck}.json"
    if not meta_path.is_file():
        log(f"    ! no checkpoint_meta_{ck}.json; cannot re-assemble. "
            f"Re-run detect for {ck}.")
        return
    m = json.loads(meta_path.read_text(encoding="utf-8"))
    try:
        floor = conf_floor if conf_floor is not None \
            else m.get("conf_floor", 0.001)
        man = mcache.save(df, Path(cache), ck,
                          checkpoint_sha=m["sha256"],
                          conf_floor=float(floor),
                          imgsz=int(m.get("imgsz", 640)),
                          nms="none", iou=None,
                          seed=int(seed if seed is not None
                                   else m.get("seed", 0)))
        log(f"    re-assembled {ck}: {man.n_detections:,} detections over "
            f"{man.n_images:,} fields")
    except Exception as exc:                                  # noqa: BLE001
        log(f"    ! re-assembly refused for {ck}: {type(exc).__name__}: "
            f"{str(exc).splitlines()[0]}")


def stage_detect(a, d) -> int:
    from mag import cli as mag_cli

    w = _weights_args(a, d)
    if not w:
        say("\nno checkpoints found or given. Pass them:")
        say("  --weights rtdetr=PATH yolov5x=PATH yolov8x=PATH")
        return EXIT_PRECONDITION

    install_format_guard()
    for st in a.sets:
        purge_suspect(Path(a.out) / st)
    was = set_cap(a.max_det)
    say(f"  detection cap: mag.detect.MAX_DET {was:,} -> {a.max_det:,}")
    if a.max_det != was:
        say("  A declared property of the detection stage. A cache built at "
            "one cap")
        say("  is not interchangeable with a cache built at another, and the")
        say("  manifest's content hash does not record it --- so it is "
            "recorded in")
        say("  run_state.json and in the capMaxDet macro.")

    rc = EXIT_OK
    for st in a.sets:
        images = Path(d.archive) / "images" / st if d.archive else None
        if images is None or not images.is_dir():
            say(f"\nno images at {images}")
            return EXIT_PRECONDITION
        rc = max(rc, _detect_set(a, d, mag_cli, images, Path(a.out) / st, w))
        # Geometry is checked AFTER the pass rather than during it: mag owns
        # the inference loop and this suite does not patch it. The exclusion
        # is applied to the cache, with the count preserved.
        hr("geometry")
        g = sanitise_geometry(Path(a.out) / st, conf_floor=a.conf,
                              seed=a.seed)
        if not any(v.get("n_excluded") for v in g.values()):
            say("  no box has a non-positive width or height")
    return rc


#: How many times detect may refuse and be repaired before the refusal
#: stands. One per checkpoint is the most that can be useful, since mag stops
#: its whole checkpoint loop at the first bad save; the extra one is slack.
MAX_DETECT_RECOVERIES = 4


def _mag_call(mag_cli, argv) -> int:
    """mag's CLI, called so that a refusal ARRIVES AS AN EXCEPTION.

    `mag.cli.main` catches MagError, prints it to stderr and returns 3. For a
    terminal that is exactly right. Here it is not: a return code carries
    neither the exception nor its remedy, and the refusal is the thing this
    stage has to act on.

    That cost the first real run. detect came back 3; the `except MagError`
    that was supposed to repair the cache never saw an exception to catch;
    the run stopped; and the only statement of the reason was a line on
    stderr, which is also the one thing a truncated log loses. Eighteen
    minutes of GPU time, and the handback could not say why it ended.

    So the parser is used and `fn` is called directly --- which is already
    how `build_and_write` drives `mag all`. The banner is printed first, as
    `main` would, so the log looks the same.
    """
    ns = mag_cli.build_parser().parse_args([str(x) for x in argv])
    banner = getattr(mag_cli, "_banner", None)
    if callable(banner):
        say(banner())
    validate = getattr(mag_cli, "_validate", None)
    if callable(validate):
        validate(ns)
    return int(ns.fn(ns) or 0)


def _detect_set(a, d, mag_cli, images: Path, cache: Path, w) -> int:
    """mag's detect over one set, recovering from a refusal at assembly.

    `mag.cache.save` validates as it writes, so a pass containing a box of
    non-positive extent raises AFTER inference: every shard is on disk and
    nothing is assembled. v2.4 ran the sanitiser only on the success path, so
    the refusal propagated and stopped the run --- and re-running reproduced
    it exactly, because resume reuses the shards, retries the assembly, and
    is refused again. Eighteen minutes of GPU time bought a cache that was
    one filter away from usable, twice.

    So: catch the refusal, filter the shards, re-assemble from them WITHOUT
    re-inference, and call mag again. The second call resumes past every
    checkpoint already assembled and goes on to the ones it never reached ---
    mag stops at the first bad save, so a later checkpoint may not have been
    attempted at all.

    A refusal the sanitiser cannot repair is re-raised untouched. This
    recovers ONE known failure; it is not a blanket retry.
    """
    from mag.errors import MagError

    for attempt in range(1, MAX_DETECT_RECOVERIES + 2):
        try:
            return _mag_call(mag_cli, _detect_argv(a, d, images, cache, w))
        except MagError as exc:
            head = (str(exc).splitlines() or [type(exc).__name__])[0]
            record_failure(a.results, f"detect/{cache.name}", exc)
            say(f"\n  {type(exc).__name__}: {head}")
            if attempt > MAX_DETECT_RECOVERIES:
                say(f"  refused {attempt} times; the refusal stands.")
                raise
            hr("recovery")
            say("  mag refused the cache AS IT WAS WRITTEN, which means the")
            say("  inference already happened: every shard is on disk under")
            say(f"  {cache / '_shards'}.")
            say("  Filtering the shards and re-assembling from them recovers")
            say("  the pass without spending the GPU again.")
            g = sanitise_geometry(cache, conf_floor=a.conf, seed=a.seed)
            if not any(v.get("n_excluded") or v.get("reassembled")
                       for v in g.values()):
                say("\n  Nothing in the cache could be repaired, so this is "
                    "not the")
                say("  geometry refusal. It stands, untouched.")
                raise
            kept = sum(int(v.get("n_kept") or v.get("n_total") or 0)
                       for v in g.values())
            say("\n  repaired. Calling mag again: assembled checkpoints are")
            say("  skipped by resume, and any checkpoint it never reached "
                "runs now.")
            say("")
            say("  EXPECT A LONG SILENCE HERE. mag re-assembles EVERY "
                "checkpoint on")
            say(f"  this call --- concatenating {kept:,} rows, validating "
                f"them, writing")
            say("  the table and hashing what it wrote --- and prints nothing "
                "while it")
            say("  does. On a cache this size that is minutes of pure disk "
                "and memory")
            say("  work with no progress line. It is not hung. run_all.py "
                "prints a")
            say("  heartbeat every minute so the difference is visible.")
    return EXIT_UNEXPECTED                                # pragma: no cover


def build_and_write(a, cache: Path, results: Path, ctx: dict,
                    checkpoints=None, quiet: bool = False):
    """mag's `cmd_all`, call for call, with `extra=` on the last step.

    Building ONCE is the point. `run`, `figures` and `report` each rebuild the
    study, so calling them in sequence does the work three times and can
    produce figures and text from two different builds. That is not
    hypothetical: an abstract once said the co-primary endpoint was met while
    the results section said it was not.
    """
    from mag import cli as mag_cli
    from mag import figures, report

    patients = Path(a.out) / "patients.csv"
    if not patients.is_file():
        say(f"\nno patients.csv at {patients}. Run the ingest stage first.")
        return None, EXIT_PRECONDITION

    argv = ["all", "--cache", str(cache), "--patients", str(patients),
            "--out", str(results), "--condition", a.condition,
            "--depth", a.depth, "--seed", str(a.seed),
            "--folds", str(a.folds), "--min-fields", str(a.min_fields),
            "--reps", str(a.reps), "--stress-reps", str(a.stress_reps)]
    if checkpoints or a.checkpoints:
        argv += ["--checkpoints", *(checkpoints or a.checkpoints)]
    if a.force_reconstruction:
        argv.append("--force-reconstruction")
    ns = mag_cli.build_parser().parse_args(argv)

    s = mag_cli._build(ns)
    written = mag_cli._write_outputs(s, results)
    stamp = figures.write_all(s, results / "figures")
    figures.check(results / "figures", s)      # refuses a mismatched pair
    extra = derived_macros(s, ctx)
    keys = report.write(s, results / "results.tex", extra=extra)
    if not quiet:
        mag_cli._report_to_console(s, a.condition)
        say(f"\nwrote {len(written)} result files, "
            f"{len(stamp['figures'])} figures and {len(keys)} macros")
        for why in stamp["skipped"]:
            say(f"  ! figure skipped: {why}")
    return (s, extra), EXIT_OK


def stage_analyse(a, d) -> int:
    load_context(a)
    built, rc = build_and_write(a, Path(a.out) / a.set, Path(a.results),
                                a._context)
    if built is None:
        return rc
    s, extra = built

    hr("macros this script adds")
    for k in DERIVED_KEYS:
        say(f"  {k:<24} {extra.get(k, 'NOT PRODUCED -- stays RED in the PDF')}")
    say("\n  A macro the analysis never computed renders red rather than "
        "blank,")
    say("  so an unfilled slot cannot pass silently.")
    say(f"\nprovenance: {s.provenance.line()}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# 7. Sensitivity --- the cap arm
# --------------------------------------------------------------------------- #

def melt_predictions(df):
    """mag writes predictions.csv WIDE: patient_id, then one column per arm
    named `<checkpoint>:<rule>`. Long form is what a diff can be keyed on."""
    idc = next((c for c in ("patient_id", "patient") if c in df.columns), None)
    if idc is None:
        return None
    arms = [c for c in df.columns if c != idc and ":" in c]
    if not arms:
        return None
    long = df.melt(id_vars=[idc], value_vars=arms,
                   var_name="arm", value_name="label")
    long[["checkpoint", "rule"]] = long["arm"].str.split(":", n=1, expand=True)
    return long.rename(columns={idc: "patient_id"})[
        ["patient_id", "checkpoint", "rule", "label"]]


def fork_cache(src: Path, dst: Path, redo: list) -> None:
    """Copy a cache directory, leaving out the checkpoints to re-detect.

    Cache file layout, from mag.detect.run: `detections_<ck>.parquet`,
    `manifest_<ck>.json`, `checkpoint_meta_<ck>.json`, an optional
    `detections_<ck>.suspect.json`, and resume shards under `_shards/<ck>/`.
    Everything else in the directory is checkpoint-independent and copies
    verbatim.
    """
    if dst.exists():
        shutil.rmtree(dst)
    skip = set()
    for ck in redo:
        skip |= {f"detections_{ck}.parquet", f"manifest_{ck}.json",
                 f"checkpoint_meta_{ck}.json", f"detections_{ck}.suspect.json"}

    def ignore(path, names):
        out = [n for n in names if n in skip]
        if Path(path).name == "_shards":
            out += [n for n in names if n in redo]
        return set(out)

    shutil.copytree(src, dst, ignore=ignore)


def stage_sensitivity(a, d) -> int:
    """Re-detect the saturating checkpoint at the ORIGINAL cap and diff.

    The main pass runs at a cap no field reaches, which is the defensible
    setting. A reader's question is not "was the cap high enough" but "would
    the answer have differed", and that is only answerable by running the
    other arm. The comparison is on `predictions.csv` --- mag's own
    per-patient labels, the quantity the manuscript reports, not a proxy.

    Arm B is a FULL factorial, not a one-checkpoint study. An earlier design
    analysed only the saturating checkpoint, and that breaks selection:
    `gates.select_checkpoint` is pre-registered on `declared="rtdetr"`, and a
    study whose cache holds only yolov5x has nothing for the declared primary
    to select. So arm B forks the arm A cache instead --- checkpoints whose
    observed per-field maximum never reached the LOW cap produce
    byte-identical output under either cap, and are copied rather than
    re-detected; only the saturating one is run again.
    """
    from mag import cli as mag_cli

    load_context(a)
    sat = sorted(k for k, v in VERIFY.SMOKE_FIELDS_AT_CAP.each(a._verify) if v)
    if not sat:
        say("  no checkpoint saturated the cap in the probe, so there is no "
            "arm to")
        say("  run. That is the result: the cap is not a free parameter here.")
        return EXIT_OK

    low = int(a.sensitivity_cap)
    say(f"  saturating checkpoint(s): {', '.join(sat)}")
    say(f"  arm A  max_det = {a.max_det:,}   (the main pass, already cached)")
    say(f"  arm B  max_det = {low:,}   (the cap that bound)")

    src_cache = Path(a.out) / a.set
    alt_cache = Path(a.out) / f"{a.set}_cap{low}"
    alt_results = Path(str(a.results).rstrip("/\\") + f"_cap{low}")

    # A copied checkpoint is only valid if the low cap could not have bound on
    # it either. per_field_max is over the sampled fields, so this is a
    # necessary check, not a proof --- which is why it is reported.
    hi_max = dict(VERIFY.SMOKE_PER_FIELD_MAX.each(a._verify))
    copied = [k for k in hi_max if k not in sat]
    risky = [k for k in copied if int(hi_max.get(k) or 0) >= low]
    if copied:
        say(f"  arm B reuses the cap-independent caches for: "
            f"{', '.join(sorted(copied))}")
        say("  (their observed per-field maxima never reached the low cap, so")
        say("  their output is identical under either arm)")
    for k in risky:
        say(f"  ! {k}: sampled per-field max {hi_max[k]} >= {low}; its copied "
            f"cache may not be cap-independent after all")

    have = all((alt_cache / f"manifest_{ck}.json").is_file() for ck in sat)
    if a.reuse_sensitivity and have:
        say(f"\n  reusing the arm B cache at {alt_cache}")
    else:
        if not src_cache.is_dir():
            say(f"  ! no arm A cache at {src_cache}. Run the detect stage "
                f"first.")
            return EXIT_PRECONDITION
        w = [x for x in _weights_args(a, d) if x.split("=", 1)[0] in sat]
        if not w:
            say("  ! the saturating checkpoint is not among the weights "
                "given or discovered.")
            return EXIT_PRECONDITION
        if not d.archive:
            say("  ! no archive, so arm B cannot be detected.")
            return EXIT_PRECONDITION
        say(f"\n  forking the arm A cache -> {alt_cache}")
        fork_cache(src_cache, alt_cache, sat)
        was = set_cap(low)
        say(f"  detection cap for arm B: {was:,} -> {low:,}")
        rc = _detect_set(a, d, mag_cli,
                         Path(d.archive) / "images" / a.set, alt_cache, w)
        sanitise_geometry(alt_cache, conf_floor=a.conf, seed=a.seed)
        set_cap(a.max_det)                     # leave the process as found
        if rc:
            return rc

    say("\n  analysing arm B (full factorial, same code path as arm A) ...")
    built, rc = build_and_write(a, alt_cache, alt_results,
                                {**a._context, "max_det": low},
                                checkpoints=a.checkpoints, quiet=True)
    if built is None:
        return rc
    return diff_arms(a, Path(a.results), alt_results, sat, low)


def diff_arms(a, res_a: Path, res_b: Path, checkpoints, low) -> int:
    import pandas as pd

    pa, pb = res_a / "predictions.csv", res_b / "predictions.csv"
    if not (pa.is_file() and pb.is_file()):
        say(f"  ! no predictions.csv to compare ({pa}, {pb})")
        return EXIT_PRECONDITION

    A, B = melt_predictions(pd.read_csv(pa)), melt_predictions(pd.read_csv(pb))
    if A is None or B is None:
        say("  ! predictions.csv is not in the expected `patient_id` + "
            "`checkpoint:rule` shape; cannot key the comparison")
        return EXIT_PRECONDITION

    A = A[A["checkpoint"].isin(checkpoints)]
    B = B[B["checkpoint"].isin(checkpoints)]
    m = A.merge(B, on=["patient_id", "checkpoint", "rule"],
                suffixes=("_hi", "_lo"))
    if m.empty:
        say("  ! the two arms share no rows; nothing to compare")
        return EXIT_PRECONDITION
    m["changed"] = m["label_hi"] != m["label_lo"]

    hr("cap sensitivity: patients whose label moved")
    total = int(m["changed"].sum())
    say(f"  {total} of {len(m):,} patient-rule cells changed between "
        f"max_det {a.max_det:,} and {low:,}")
    by = (m.groupby("rule")["changed"].agg(["sum", "count"])
          .sort_values("sum", ascending=False))
    rows = []
    for r, v in by.iterrows():
        if v["sum"]:
            flag = ("  <-- density rule, no score filter"
                    if str(r) in ("R6", "R6b") else "")
            say(f"    {str(r):<6} {int(v['sum']):>4} of {int(v['count']):>4}"
                f"{flag}")
        rows.append({"rule": r, "changed": int(v["sum"]),
                     "n": int(v["count"])})
    if total == 0:
        say("\n  No patient's call moved. The cap is immaterial to the "
            "reported")
        say("  results, and the manuscript can say so with a number behind "
            "it.")
    else:
        say("\n  The cap changed reported calls. It is a property of the "
            "detector")
        say("  under test and must be declared with its value, not left to a")
        say("  default.")

    out = res_a / "cap_sensitivity.json"
    out.write_text(json.dumps(
        {"cap_high": a.max_det, "cap_low": low,
         "checkpoints": sorted(checkpoints),
         "cells_compared": int(len(m)), "cells_changed": total,
         "by_rule": rows,
         "note": "per-patient labels from mag's own predictions.csv; the "
                 "quantity the manuscript reports, not a proxy"},
        indent=1), encoding="utf-8")
    say(f"\n  {out}")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# 8. Audit
# --------------------------------------------------------------------------- #

#: The macro files the manuscript inputs besides mag's own results.tex.
#: `mag.audit.check_macros` reads out_dir/results.tex and nothing else, which
#: is right for mag and wrong for this manuscript: it also inputs archive.tex
#: (the eda and verify layers) and tables.tex (the per-cell tabular bodies).
#: Audited against results.tex alone, every `arc*` and every table macro is
#: reported as undefined --- 56 false failures on the first complete run.
SIDE_MACRO_FILES = ("archive.tex", "tables.tex")


def _rescope_macro_check(findings: list, results: Path, ms, a) -> list:
    """Re-run mag's own macro check against ALL the macro files, not one.

    mag's check is the authority on what a macro failure IS; it is simply
    pointed at one file. So the check is re-run --- mag's function, mag's
    wording --- over the concatenation of every macro file the manuscript
    actually inputs, and its verdict replaces the narrower one.
    """
    from mag import audit as mag_audit

    if not ms:
        return findings
    here = Path(ms).resolve().parent
    extra = [here / n for n in SIDE_MACRO_FILES if (here / n).is_file()]
    extra += [Path(a.manuscript_dir) / n for n in SIDE_MACRO_FILES
              if getattr(a, "manuscript_dir", None)
              and (Path(a.manuscript_dir) / n).is_file()
              and (Path(a.manuscript_dir) / n) not in extra]
    tex = results / "results.tex"
    if not extra or not tex.is_file():
        return findings

    import tempfile
    parts = [tex.read_text(encoding="utf-8")]
    parts += [f.read_text(encoding="utf-8") for f in extra]
    with tempfile.TemporaryDirectory() as td:
        combined = Path(td) / "all_macros.tex"
        combined.write_text("\n".join(parts), encoding="utf-8")
        fresh = mag_audit.check_macros(combined, ms)
    say(f"  macros re-checked against results.tex plus "
        f"{', '.join(f.name for f in extra)}")
    out = [f for f in findings if f.get("check") != "macros"]
    out += [{"check": "macros", "severity": f.severity, "message": f.message,
             "detail": f.detail} for f in fresh]
    return out


def stage_audit(a, d) -> int:
    """mag's build audit, through its API rather than its exit code.

    Two changes from calling `mag audit` on the command line, both learned
    the hard way. First, the findings land in `results/audit_report.json` ---
    machine-readable, and part of the handback bundle --- instead of only on a
    console that scrolls away. Second, a failed audit no longer kills the run:
    the main loop is told to continue past this stage, because the manuscript
    stage must run REGARDLESS to record what the audit found. An audit that
    stops the pipeline before the pipeline can write down the audit's own
    verdict is a check defeating its own purpose.

    The audit still fails loudly when it fails --- an unfilled declarations
    file is supposed to fail; those ten fields are the authors' to write and
    no pipeline may invent them.
    """
    from mag import audit as mag_audit

    results = Path(a.results)
    findings = []
    n_checks = 0

    # Declarations are pre-loaded the way mag's own CLI does it: audit.run
    # would otherwise raise FileNotFoundError INSIDE a check when the path
    # names a file that is not there, taking every other check down with it.
    decl = None
    if a.declarations:
        p = Path(a.declarations)
        if p.is_file():
            try:
                decl = json.loads(p.read_text(encoding="utf-8"))
                decl.pop("_README", None)   # the template's own note would
                                            # trip the placeholder check
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                say(f"  ! {p} is not valid JSON ({exc}); auditing as if no "
                    f"declarations were given")
        else:
            say(f"  ! no declarations file at {p}; auditing without one")

    ms = a.manuscript if (a.manuscript and Path(a.manuscript).is_file()) \
        else None
    if a.manuscript and not ms:
        say(f"  ! no manuscript at {a.manuscript}; the manuscript-dependent "
            f"checks are skipped")

    fac = None
    fp = results / "factorial.csv"
    if fp.is_file():
        try:
            import pandas as pd
            fac = pd.read_csv(fp)
        except Exception:                                     # noqa: BLE001
            fac = None

    try:
        rep = mag_audit.run(results, manuscript=ms, declarations=decl,
                            factorial=fac)
        n_checks = rep.n_checks
        findings = [{"check": f.check, "severity": f.severity,
                     "message": f.message, "detail": f.detail}
                    for f in rep.findings]
        findings = _rescope_macro_check(findings, results, ms, a)
        say(f"  {rep.summary()}")
        for f in rep.findings:
            say(f"  [{f.severity:<4}] {f.check}: {f.message}")
    except Exception as exc:                                  # noqa: BLE001
        findings = [{"check": "audit-crashed", "severity": "fail",
                     "message": f"{type(exc).__name__}: {exc}", "detail": {}}]
        say(f"  ! the audit itself crashed: {type(exc).__name__}: {exc}")
        say("    That is recorded as a failure; the run continues so the")
        say("    manuscript stage can write it down.")

    if not a.declarations and any(f["check"] == "declarations"
                                  for f in findings):
        say("")
        say("  The audit cannot pass without a declarations file. It is "
            "yours to")
        say("  write --- ethics, funding, contributions, AI use, data "
            "availability.")
        say("  Fill declarations.json (a template is created by run_all.py) "
            "and re-run:")
        say("    python run_experiment.py --stages audit manuscript "
            "--declarations declarations.json")

    # The v2 -> v3 wording guard, folded into the same report.
    if ms:
        load_context(a)
        unident = any(r["fired"] for r in a._ledger
                      if r["key"] == "identifiability")
        found = check_manuscript(ms, unidentifiable=unident)
        hr("v2 -> v3 wording guard")
        if not found:
            say("  no retracted wording found")
        for kind, what, why in found:
            say(f"  ! {kind}: {what}")
            if why:
                say(f"      {why}")
            findings.append({"check": "wording_guard",
                             "severity": ("fail" if kind == "forbidden"
                                          else "warn"),
                             "message": f"{kind}: {what}",
                             "detail": {"why": why}})

    failed = [f for f in findings if f["severity"] == "fail"]
    (results / "audit_report.json").write_text(json.dumps(
        {"generated_utc": dt.datetime.now(dt.timezone.utc)
                            .isoformat(timespec="seconds"),
         "n_checks": n_checks,
         "passed": not failed,
         "n_failed": len(failed),
         "declarations_supplied": decl is not None,
         "manuscript_checked": bool(ms),
         "findings": findings},
        indent=1, default=str), encoding="utf-8")
    say(f"\n  {results / 'audit_report.json'}")
    return EXIT_AUDIT if failed else EXIT_OK


# --------------------------------------------------------------------------- #
# 9. Manuscript --- the ledger, and every number the paper cites
# --------------------------------------------------------------------------- #

def read_macros(tex: Path) -> dict:
    if not Path(tex).is_file():
        return {}
    return dict(re.findall(
        r"\\defresult\{([A-Za-z0-9]+)\}\{(.*)\}",
        Path(tex).read_text(encoding="utf-8", errors="replace")))


def stage_manuscript(a, d) -> int:
    import pandas as pd

    load_context(a)
    results = Path(a.results)
    results.mkdir(parents=True, exist_ok=True)
    rows = a._ledger

    pd.DataFrame([{k: r[k] for k in
                   ("key", "severity", "title", "fired", "unknown",
                    "detail", "where", "manuscript")} for r in rows]
                 ).to_csv(results / "findings.csv", index=False)

    macros = read_macros(results / "results.tex")
    cap = _read_json(results / "cap_sensitivity.json")
    summary = _read_json(results / "results.json") or {}
    audit = _read_json(results / "audit_report.json")

    (results / "manuscript_values.json").write_text(json.dumps({
        "generated_utc": dt.datetime.now(dt.timezone.utc)
                           .isoformat(timespec="seconds"),
        "provenance": macros.get("provenance"),
        "run": {"max_det": a.max_det, "condition": a.condition,
                "depth": a.depth, "seed": a.seed, "set": a.set},
        "context": a._context,
        "macros": macros,
        "findings": rows,
        "cap_sensitivity": cap,
        "audit": audit,
        "results_summary_keys": sorted(summary),
    }, indent=1, default=str), encoding="utf-8")

    write_results_md(a, results, rows, macros, cap, audit)

    hr("what the manuscript owes the reader")
    for r in EF.fired(rows):
        say(f"  [{r['severity']}] {r['where'] or 'unfiled'}")
        say(f"      {r['detail']}")
        say(f"      -> {r['manuscript']}")
        say("")
    say(f"  {results / 'findings.csv'}")
    say(f"  {results / 'manuscript_values.json'}")
    say(f"  {results / 'RESULTS.md'}")
    return EXIT_OK


def write_results_md(a, results: Path, rows, macros, cap, audit=None):
    L = ["# Results", "",
         f"Run {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}  ",
         f"Condition `{a.condition}`, depth `{a.depth}`, seed {a.seed}, "
         f"`max_det` {a.max_det:,}  ",
         f"Provenance: `{macros.get('provenance', '--')}`", ""]

    fired = EF.fired(rows)
    L += ["## Findings that constrain the claims", "",
          f"{len(fired)} of {len(rows)} fired.", "",
          "| severity | finding | measured | belongs in |",
          "|---|---|---|---|"]
    for r in fired:
        L.append(f"| {r['severity']} | {r['title']} | "
                 f"{r['detail'].replace('|', '/')} | {r['where']} |")
    L.append("")

    unk = EF.unknown(rows)
    if unk:
        L += ["### Not checked", "",
              "A finding that was not checked is not a finding that cleared.",
              ""]
        L += [f"- `{r['key']}`: {r['detail']}" for r in unk] + [""]

    if cap:
        L += ["## Cap sensitivity", "",
              f"- arms: `max_det` {cap['cap_high']:,} vs {cap['cap_low']:,} "
              f"on {', '.join(cap['checkpoints'])}",
              f"- {cap['cells_changed']} of {cap['cells_compared']:,} "
              f"patient-rule cells changed", ""]
        moved = [r for r in cap.get("by_rule", []) if r["changed"]]
        if moved:
            L += ["| rule | changed | of |", "|---|---|---|"]
            L += [f"| {r['rule']} | {r['changed']} | {r['n']} |" for r in moved]
            L.append("")

    if audit:
        L += ["## Build audit", "",
              f"- {audit.get('n_checks', 0)} checks, "
              f"{'passed' if audit.get('passed') else str(audit.get('n_failed', '?')) + ' FAILED'}",
              f"- declarations supplied: {audit.get('declarations_supplied')}",
              ""]
        bad = [f for f in (audit.get("findings") or [])
               if f.get("severity") == "fail"]
        if bad:
            L += [f"- [{f['check']}] {f['message']}" for f in bad] + [""]

    if macros:
        L += ["## Macros", "",
              "Every `\\R{}` the manuscript may cite. A key absent here "
              "renders RED in the PDF.", "", "```"]
        L += [f"{k:<32} {v}" for k, v in sorted(macros.items())]
        L += ["```", ""]

    (results / "RESULTS.md").write_text("\n".join(L), encoding="utf-8")


# --------------------------------------------------------------------------- #
# 10. Dispatch
# --------------------------------------------------------------------------- #

RUNNERS = {"preflight": stage_preflight, "ingest": stage_ingest,
           "detect": stage_detect, "analyse": stage_analyse,
           "sensitivity": stage_sensitivity, "audit": stage_audit,
           "manuscript": stage_manuscript}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_experiment.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", "--archive", dest="root", default=".")
    p.add_argument("--image-list", dest="image_list", default=None)
    p.add_argument("--weights", nargs="*", metavar="NAME=PATH", default=None)
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES,
                   help="run only these, in the order given here")
    p.add_argument("--sets", nargs="+", default=["test"])
    p.add_argument("--set", default="test")
    p.add_argument("--zoom-set", dest="zoom_set", default="test-zoom")
    p.add_argument("--out", default="data")
    p.add_argument("--results", default="results")
    p.add_argument("--eda", default="eda", help="where eda_dataset.py wrote")
    p.add_argument("--verify", default="verify",
                   help="where verify_models.py wrote")

    p.add_argument("--max-det", dest="max_det", type=int,
                   default=DEFAULT_MAX_DET, metavar="N",
                   help="mag.detect.MAX_DET for this run. Preflight warns "
                        "loudly if the cap probe says this will bind.")
    p.add_argument("--sensitivity-cap", dest="sensitivity_cap", type=int,
                   default=3000, help="the second arm's cap (mag's own)")
    p.add_argument("--reuse-sensitivity", dest="reuse_sensitivity",
                   action="store_true",
                   help="reuse an existing arm B cache instead of re-detecting")

    p.add_argument("--condition", choices=("raw", "matched", "source"),
                   default="matched")
    p.add_argument("--depth", choices=("core", "full"), default="full")
    p.add_argument("--checkpoints", nargs="*", default=None)
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--device", default="0")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--half", action="store_true", default=None)
    p.add_argument("--no-half", dest="half", action="store_false")
    p.add_argument("--yolov5-repo", dest="yolov5_repo", default=None)
    p.add_argument("--assume-classes-txt", dest="assume_classes_txt",
                   action="store_true")
    p.add_argument("--no-resume", dest="no_resume", action="store_true")
    p.add_argument("--skip-layout", dest="skip_layout", action="store_true")
    p.add_argument("--min-fields", dest="min_fields", type=int, default=25)
    p.add_argument("--folds", type=int, default=10)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--stress-reps", dest="stress_reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force-reconstruction", dest="force_reconstruction",
                   action="store_true")
    p.add_argument("--manuscript", default=None)
    p.add_argument("--declarations", default=None)
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    mag = require_mag()
    from mag import discover
    from mag.errors import MagError

    say("=" * 78)
    say(" run_experiment.py --- the experiment, start to finish")
    say("=" * 78)
    say(f"  mag       {mag.__version__}  build {mag.build_id()}")
    say(f"  stages    {' '.join(a.stages)}")
    say(f"  max_det   {a.max_det:,}")

    d = discover.discover(Path(a.root), max_depth=6)
    if a.image_list:
        d.image_list = Path(a.image_list)
    say(f"  archive   {d.archive}")

    state = {"started_utc": dt.datetime.now(dt.timezone.utc)
                              .isoformat(timespec="seconds"),
             "max_det": a.max_det, "condition": a.condition,
             "depth": a.depth, "seed": a.seed, "stages": {}}
    worst = EXIT_OK
    for name in a.stages:
        hr(name)
        if a.dry_run:
            say(f"  would run {name}")
            continue
        t0 = time.time()
        try:
            rc = RUNNERS[name](a, d)
        except MagError as exc:
            say(f"\n  {type(exc).__name__}: {exc}")
            if getattr(exc, "remedy", None):
                say(f"\n  {exc.remedy}")
            record_failure(a.results, name, exc)
            say("\n  A refusal with a remedy is often a FINDING, not a fault:")
            say("  the reconstruction gate refusing says something about the")
            say("  released artefacts, not about this code.")
            rc = EXIT_REFUSAL
        except KeyboardInterrupt:
            say("\n  interrupted")
            return 130
        state["stages"][name] = {"rc": rc, "seconds": round(time.time() - t0, 1)}
        worst = max(worst, rc)
        if rc and name == "audit":
            # A failed audit is a verdict to record, not a wall. The
            # manuscript stage runs next PRECISELY so that verdict lands in
            # findings.csv and manuscript_values.json instead of dying with
            # the process. Everything else that fails still stops the run:
            # a stage downstream of a missing input would only manufacture
            # confusion.
            say(f"\n  audit failed (rc={rc}); continuing so the manuscript "
                f"stage can record it.")
        elif rc:
            say(f"\n  {name} returned {rc}; stopping here.")
            break

    out = Path(a.results)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_state.json").write_text(json.dumps(state, indent=1),
                                        encoding="utf-8")
    hr("done")
    for k, v in state["stages"].items():
        say(f"  {k:<12} rc={v['rc']}  {v['seconds']}s")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
