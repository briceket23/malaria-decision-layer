#!/usr/bin/env python3
r"""run_all.py --- the whole experiment, one command, offline.

    python run_all.py --root ..\malaria_france_data_set --yolov5-repo ..\malaria_france_data_set\models\models\yolov5

That is the complete interface for the common case. It runs, in order:

    env         is this machine able to run the pass? (python, mag, torch,
                CUDA, ultralytics) --- checked BEFORE any time is spent
    eda         eda_dataset.py       what the archive says, no GPU
    verify      verify_models.py     can the checkpoints detect, at mag's own
                operating point, including the cap probe
    experiment  run_experiment.py    ingest -> detect -> analyse ->
                sensitivity -> audit -> manuscript
    manuscript  archive.tex from the eda/verify layers, and the register of
                which of the manuscript's numbers exist yet
    bundle      one zip with every result, log and report, ready to send back

Everything printed by every stage is also written to results/run_all.log, and
the final act is a PASTE-BACK SUMMARY: a short block of the headline numbers
you can paste into the chat even without uploading the bundle.

DECISIONS THIS SCRIPT MAKES, AND WHY

  * The detection cap is resolved from the cap probe's own measurement: the
    largest `suggested_max_det` in verify.json. If nothing saturates, mag's
    default stands. `--max-det` overrides both. Detecting at a cap the probe
    says will bind produces a CENSORED cache that every downstream gate
    accepts, which is the most expensive silent failure this project has.

  * `--force-reconstruction` is passed by default. Protocol v3 SS0 documents
    that on the real archive the comparator reconstructs 105/170 against a
    published 135, so the reconstruction gate WILL refuse --- and mag's own
    remedy text says to force past it and have the refusal recorded in the
    results. The refusal is a finding about the released artefacts; forcing
    records it instead of ending the run at minute forty.

  * If no declarations.json exists, a template is written with every value
    marked [FILL: ...]. The audit will fail on those placeholders, correctly:
    the ten declarations are the authors' to write and no pipeline may invent
    them. The run continues past the failed audit and records it, so one
    missing file never costs the detection pass.

  * A failed VERIFY does not stop the run when the failure is the known cap
    saturation --- that is precisely the condition the experiment corrects by
    raising the cap. Any other verify failure (a silent checkpoint, a
    permuted class map) stops everything, because an hour of GPU time against
    a broken checkpoint measures nothing.

WHAT TO SEND BACK

    results/handback_<timestamp>.zip

or, at minimum: results/run_all.log, results/RESULTS.md,
results/manuscript_values.json, verify/verify.json, eda/eda.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib
import importlib.util
import json
import os
import platform
import re
import sys
import time
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

PHASES = ("env", "eda", "verify", "experiment", "manuscript",
          "bundle")

EXIT_OK, EXIT_PRECONDITION = 0, 2

#: Bumped whenever a fix lands. Printed at startup, beside a short hash of
#: every file in the suite, because "which version am I running?" was
#: otherwise answerable only by reading tracebacks --- and a stale copy in a
#: sibling folder reproduced a bug that had already been fixed, costing a
#: whole round trip.
SUITE_VERSION = "3.0"

SUITE_FILES = ("exp_schema.py", "eda_dataset.py", "verify_models.py",
               "experiment_findings.py", "run_experiment.py",
               "make_archive_macros.py", "make_result_tables.py",
               "missing_results.py", "verify_offline.py", "run_all.py")


def suite_fingerprint() -> dict:
    """Short content hash per file, so a stale copy is visible immediately."""
    out = {}
    for n in SUITE_FILES:
        p = HERE / n
        try:
            out[n] = hashlib.sha256(p.read_bytes()).hexdigest()[:8]
        except OSError:
            out[n] = "MISSING"
    return out

#: The ten author-supplied declarations mag's audit requires. Written as a
#: template when none exists; the [FILL: ...] values fail the audit's
#: placeholder check BY DESIGN until a person replaces them.
DECLARATIONS_TEMPLATE = {
    "ethics_source_study": "[FILL: ethics approval of the SOURCE study, "
                           "e.g. committee name and reference]",
    "ethics_present_study": "[FILL: ethics statement for THIS re-analysis "
                            "of released, de-identified data]",
    "consent_for_publication": "[FILL: consent statement, or 'not "
                               "applicable' with the reason]",
    "funding": "[FILL: funding sources, or 'none']",
    "author_contributions": "[FILL: CRediT roles per author]",
    "ai_use": "[FILL: how AI assistance was used in analysis code and "
              "manuscript preparation]",
    "repository_url": "[FILL: URL of the code repository]",
    "osf_registration": "[FILL: OSF registration DOI or ID]",
    "code_archive_doi": "[FILL: Zenodo or equivalent archive DOI]",
    "data_availability": "[FILL: where the Plasmodium-v1 archive and the "
                         "released checkpoints can be obtained]",
}


def hr(t=""):
    print("\n" + ("=== " + t + " " if t else "").ljust(78, "="))


class Tee:
    """Everything printed goes to the console AND to run_all.log.

    The log is the artefact: a pasted console scrolls and truncates, and this
    session has already spent hours reconstructing state from partial
    transcripts. The bundle carries the log, so the whole run is reviewable.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for t in self.streams:
            try:
                t.write(s)
            except Exception:                                 # noqa: BLE001
                pass
        return len(s)

    def flush(self):
        for t in self.streams:
            try:
                t.flush()
            except Exception:                                 # noqa: BLE001
                pass

    def isatty(self):
        return False


# --------------------------------------------------------------------------- #
# env --- fail in seconds, not at minute forty
# --------------------------------------------------------------------------- #

def phase_env(a) -> int:
    info = {"python": sys.version.split()[0],
            "executable": sys.executable,
            "platform": platform.platform(),
            "started_utc": dt.datetime.now(dt.timezone.utc)
                             .isoformat(timespec="seconds"),
            "argv": sys.argv[1:],
            "packages": {}}
    problems = []

    try:
        import mag
        info["mag"] = f"{mag.__version__} build {mag.build_id()}"
        if str(mag.__version__).split(".")[0] != "3":
            problems.append(f"mag {mag.__version__} found; this suite "
                            f"targets 3.x")
    except ImportError:
        problems.append('mag is not installed:  pip install -e .')

    for m in ("numpy", "pandas", "pyarrow", "matplotlib", "statsmodels",
              "openpyxl", "PIL"):
        try:
            mod = importlib.import_module(m)
            info["packages"][m] = getattr(mod, "__version__", "?")
        except ImportError:
            problems.append(f"{m} is not importable")

    needs_gpu = "experiment" in a.phases or "verify" in a.phases
    if needs_gpu and not a.no_gpu_check:
        try:
            import torch
            info["packages"]["torch"] = torch.__version__
            info["cuda"] = bool(torch.cuda.is_available())
            if info["cuda"]:
                info["gpu"] = torch.cuda.get_device_name(0)
            else:
                problems.append(
                    "torch has no CUDA. Detection on CPU takes many hours. "
                    "Install a CUDA build, or pass --device cpu deliberately "
                    "with --no-gpu-check.")
        except ImportError:
            problems.append("torch is not importable; the detect stage "
                            "cannot run")
        try:
            import ultralytics
            info["packages"]["ultralytics"] = ultralytics.__version__
        except ImportError:
            problems.append("ultralytics is not importable; the detect "
                            "stage cannot run")

        # NEVER `import yolov5` here. Importing the fork puts top-level
        # `models`/`utils` into this process, and mag decides a checkpoint's
        # format by whether unpickling RAISES ModuleNotFoundError for exactly
        # those names. One import in a version check therefore relabels the
        # YOLOv5 checkpoint as "ultralytics", ultralytics' YOLO then loads it
        # without error, and the pass caches raw tensor columns as
        # confidences. That happened: a dependency check silently broke the
        # detector it was checking for. Presence and version are read from
        # metadata instead, which imports nothing.
        # `importlib` is imported at module level. A local `import
        # importlib.util` here would make the NAME local to this whole
        # function, and the `importlib.import_module` call above it would then
        # raise UnboundLocalError before reaching this line.
        try:
            import importlib.metadata as im
        except ImportError:                                   # pragma: no cover
            im = None
        have_v5 = importlib.util.find_spec("yolov5") is not None
        if have_v5:
            ver = "?"
            if im is not None:
                try:
                    ver = im.version("yolov5")
                except Exception:                             # noqa: BLE001
                    pass
            info["packages"]["yolov5"] = f"{ver} (present; not imported)"
        elif not a.yolov5_repo:
            problems.append(
                "neither the yolov5 package nor --yolov5-repo is available; "
                "the yolov5x checkpoint cannot load")

    if a.yolov5_repo:
        r = Path(a.yolov5_repo)
        if not r.is_dir():
            problems.append(f"--yolov5-repo names no directory: {r}")
        elif not (r / "models").is_dir():
            # The archive's `models\models\yolov5\` holds the CHECKPOINT, not
            # a source checkout; passing it here is a common and harmless-
            # looking mistake that leaves the real loader unfound.
            problems.append(
                f"--yolov5-repo has no models/ directory: {r}\n"
                f"    That is a checkpoint folder, not a yolov5 source "
                f"checkout. If the\n"
                f"    yolov5 package is installed, drop the flag entirely; "
                f"otherwise point\n"
                f"    it at a clone of github.com/ultralytics/yolov5.")

    for k, v in info.items():
        if k not in ("packages", "argv"):
            print(f"  {k:<12}{v}")
    for k, v in sorted(info["packages"].items()):
        print(f"    {k:<14}{v}")

    Path(a.results).mkdir(parents=True, exist_ok=True)
    (Path(a.results) / "environment.json").write_text(
        json.dumps(info, indent=1, default=str), encoding="utf-8")

    if problems:
        print()
        for p in problems:
            print(f"  ! {p}")
        if a.no_gpu_check:
            print("\n  --no-gpu-check given: continuing anyway.")
            return EXIT_OK
        print("\n  Nothing has been spent yet. Fix these and re-run --- or "
              "pass")
        print("  --no-gpu-check to proceed against this advice.")
        return EXIT_PRECONDITION
    return EXIT_OK


# --------------------------------------------------------------------------- #
# eda / verify --- the two producers, in-process so the tee captures them
# --------------------------------------------------------------------------- #

#: Known-harmless noise, matched line by line and counted rather than shown.
#:
#: The yolov5 fork calls `torch.cuda.amp.autocast` inside AutoShape, once per
#: BATCH, and torch 2.7 deprecated it. Ordinarily a warning at a fixed source
#: location prints once --- but torch and ultralytics open
#: `warnings.catch_warnings()` blocks internally, and every exit from one
#: invalidates the "already shown here" registry, so the same two lines are
#: re-emitted for every batch. Over 4,508 fields that is thousands of them,
#: and the first complete run had to be interrupted because mag's progress
#: lines were unreadable in between.
#:
#: There is nothing to fix on this side: the deprecation is about the fork's
#: own code. Two mechanisms, deliberately overlapping:
#:
#:   1. CHILD_WARNING_FILTERS is exported as PYTHONWARNINGS so the child never
#:      formats the warning at all --- cheapest, but a library that calls
#:      warnings.resetwarnings() would defeat it.
#:   2. is_noise() drops the lines out of the stream we relay. This cannot be
#:      defeated, and it COUNTS what it drops; the count is printed at the end
#:      of the phase, so the noise is suppressed rather than hidden.
#:
#: --show-warnings turns both off, for the case where a warning matters.
NOISE = (
    re.compile(r"FutureWarning: `torch\.cuda\.amp\."),
    re.compile(r"^\s*with amp\.autocast\(autocast\):\s*$"),
    re.compile(r"^\s*@torch\.cuda\.amp\.custom_"),
)

#: A warning prints as a header line plus the echoed source line under it.
#: Rather than enumerate every possible source line, the indented line
#: immediately after a suppressed header is dropped too --- but ONLY if it
#: also names the deprecated call. Dropping any indented follow-on line would
#: eat a progress line whenever Python cannot read the source file and prints
#: no echo, and silently losing real output is worse than showing a warning.
#: `\bamp\b` rather than `amp`, so a path containing "sample" is not a match.
NOISE_HEADER = re.compile(r"Warning: `torch\.cuda\.amp\.")
NOISE_CONT = re.compile(r"^\s+.*(\bamp\b|autocast|custom_fwd|custom_bwd)")

#: PYTHONWARNINGS entries: action:message:category:module:lineno, comma
#: separated. The message is a LITERAL prefix (warnings._setoption re.escapes
#: it), so it must contain no comma and no colon --- these prefixes stop
#: before both. Categories stay FutureWarning, so nothing else is hidden.
CHILD_WARNING_FILTERS = ",".join(
    "ignore:`torch.cuda.amp.%s(args...)` is deprecated:FutureWarning" % n
    for n in ("autocast", "custom_fwd", "custom_bwd"))

#: Set from --show-warnings in main(), read by run_script and child_env.
SHOW_WARNINGS = False


def is_noise(line: str) -> bool:
    return any(p.search(line) for p in NOISE)


def child_env() -> dict:
    """Environment for a phase subprocess.

    PYTHONWARNINGS is APPENDED to whatever the user already set, never
    replaced: someone debugging with PYTHONWARNINGS=error must keep it.
    """
    env = dict(os.environ)
    # Line-buffer the child. A Python process writing to a PIPE is BLOCK
    # buffered, so mag's progress lines sit in an 8 KB buffer until it fills.
    # During a long assembly that means minutes of apparent silence, and the
    # run has now been killed twice by hand for looking hung. It also means
    # an interrupt loses whatever was still in the buffer, so the log cannot
    # say where the run had got to.
    env["PYTHONUNBUFFERED"] = "1"
    if SHOW_WARNINGS:
        env["MALARIA_SHOW_WARNINGS"] = "1"
        return env
    env.pop("MALARIA_SHOW_WARNINGS", None)
    prior = env.get("PYTHONWARNINGS", "").strip()
    env["PYTHONWARNINGS"] = ((prior + "," if prior else "")
                             + CHILD_WARNING_FILTERS)
    return env


#: Sentinel yielded by the reader when nothing has arrived for a while.
_TICK = object()

#: How long a phase may say nothing before this script says it is still
#: working. Long enough not to interrupt normal progress lines, short enough
#: that a person does not conclude the run has died.
HEARTBEAT_SECONDS = 60


def _lines_with_heartbeat(p, every=None):
    """Yield the child's lines, and a heartbeat while it is quiet.

    Reading the pipe on a background thread, because iterating it directly
    blocks with no way to notice time passing. mag's expensive steps ---
    assembling nine million detections, hashing the table it just wrote ---
    print nothing at all, and a run that prints nothing for four minutes gets
    killed by the person watching it. That has now happened twice.
    """
    import queue
    import threading

    # Read the module attribute at CALL time. As a default argument it is
    # bound when this function is defined, so a caller --- the test suite ---
    # that lowers HEARTBEAT_SECONDS to make the wait bearable would still
    # wait the full minute, and the check would silently pass on a timeout
    # that never happened.
    every = HEARTBEAT_SECONDS if every is None else every
    q: "queue.Queue" = queue.Queue()

    def reader():
        try:
            for raw in p.stdout:
                q.put(raw)
        except Exception:                                     # noqa: BLE001
            pass
        finally:
            q.put(None)

    t = threading.Thread(target=reader, daemon=True)
    t.start()
    # Poll well inside the heartbeat, or the first tick lands a whole poll
    # interval late and a short quiet spell is missed entirely.
    poll = max(0.25, min(5.0, every / 4.0))
    t0 = last = time.time()
    while True:
        try:
            item = q.get(timeout=poll)
        except Exception:                                     # noqa: BLE001
            if time.time() - last >= every:
                mins = (time.time() - t0) / 60
                print(f"  [still working, {mins:.0f} min elapsed, no output "
                      f"from this stage yet --- this is normal during cache "
                      f"assembly]")
                last = time.time()
            yield _TICK
            continue
        if item is None:
            return
        last = time.time()
        yield item


def run_script(script: str, argv: list) -> int:
    """Run one suite script in its OWN process, streaming its output here.

    These used to be plain in-process calls, and that caused the worst
    failure this project has had. The verify stage loads the YOLOv5
    checkpoint through its repository loader, which puts top-level
    `models`/`utils` on sys.path and leaves them there. mag decides a
    checkpoint's format by whether unpickling RAISES on exactly those names,
    so the detect stage --- running minutes later in the same interpreter ---
    read the v5 checkpoint as "ultralytics", loaded it through the wrong
    class, and cached 2 detections across 4,508 fields after 26 minutes of
    GPU time.

    A phase is a program. Running it as one costs a process spawn and buys
    the guarantee that nothing it imports can reach the next phase.
    """
    import subprocess

    cmd = [sys.executable, str(HERE / script)] + [str(x) for x in argv]
    print(f"  $ {script} {' '.join(str(x) for x in argv)}\n")
    p = subprocess.Popen(cmd, cwd=str(Path.cwd()), stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", bufsize=1,
                         env=child_env())

    dropped = 0
    after_header = False
    try:
        for raw in _lines_with_heartbeat(p):
            if raw is _TICK:
                continue
            line = raw.rstrip("\n")
            if not SHOW_WARNINGS:
                if is_noise(line):
                    dropped += 1
                    after_header = bool(NOISE_HEADER.search(line))
                    continue
                # the echoed source line printed under a suppressed header
                if after_header and NOISE_CONT.match(line):
                    dropped += 1
                    after_header = False
                    continue
                after_header = False
            print(line)
    finally:
        p.stdout.close()

    rc = int(p.wait())
    if dropped:
        print(f"\n  [suppressed {dropped} lines of torch.cuda.amp "
              f"deprecation noise from the yolov5 fork; --show-warnings "
              f"keeps them]")
    return rc


def phase_eda(a) -> int:
    argv = ["--root", a.root, "--out", a.eda]
    if a.image_list:
        argv += ["--image-list", a.image_list]
    return run_script("eda_dataset.py", argv)


def phase_verify(a) -> int:
    argv = ["--root", a.root, "--out", a.verify, "--device", a.device]
    if a.yolov5_repo:
        argv += ["--yolov5-repo", a.yolov5_repo]
    rc = run_script("verify_models.py", argv)
    if rc == 0:
        return rc

    # A verify failure whose ONLY cause is cap saturation is the condition
    # the experiment exists to correct: the cap is raised before the pass.
    # Any other failure --- a silent checkpoint, a permuted class map, scores
    # that are not probabilities --- means an hour of GPU time would measure
    # the harness rather than the detectors, and the run stops here.
    doc = _read_json(Path(a.verify) / "verify.json") or {}
    only_cap = _failures_are_only_saturation(doc)
    if only_cap:
        print("\n  verify reported failures, and every one is the known cap")
        print("  saturation. The experiment corrects exactly this by raising")
        print("  the cap, so the run continues.")
        return EXIT_OK
    print("\n  verify reported failures beyond cap saturation. Running the")
    print("  full pass against a checkpoint that cannot detect measures")
    print("  nothing --- stopping here. Read verify/VERIFY.md.")
    return rc


def _failures_are_only_saturation(doc: dict) -> bool:
    import exp_schema as S
    any_fail = False
    for name in set(list(S.VERIFY.SMOKE_PROBLEMS.names(doc))
                    + list(S.VERIFY.CK_PROBLEMS.names(doc))):
        probs = (dict(S.VERIFY.SMOKE_PROBLEMS.each(doc)).get(name) or []) \
            + (dict(S.VERIFY.CK_PROBLEMS.each(doc)).get(name) or [])
        for p in probs:
            any_fail = True
            if "max_det" not in str(p):
                return False
    return any_fail


def _read_json(p):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None


# --------------------------------------------------------------------------- #
# experiment
# --------------------------------------------------------------------------- #

def resolve_max_det(a) -> int:
    """The cap the pass will run at, and where that number came from."""
    if a.max_det:
        print(f"  max_det   {a.max_det:,}  (given on the command line)")
        return a.max_det
    import exp_schema as S
    doc = _read_json(Path(a.verify) / "verify.json") or {}
    need = 0
    for name, v in S.VERIFY.PROBE_SUGGESTED.each(doc):
        if int(v) > need:
            need = int(v)
            src = name
    if need:
        print(f"  max_det   {need:,}  (the cap probe's measurement: 2x the "
              f"true per-field")
        print(f"            maximum {src} reached with the cap lifted)")
        return need
    mag_default = int(doc.get("mag_max_det") or 3000)
    print(f"  max_det   {mag_default:,}  (mag's default; no checkpoint "
          f"saturated it)")
    return mag_default


def ensure_declarations(a) -> str:
    if a.declarations:
        return a.declarations
    p = Path("declarations.json")
    if not p.is_file():
        p.write_text(json.dumps(DECLARATIONS_TEMPLATE, indent=1),
                     encoding="utf-8")
        print(f"  wrote {p} as a TEMPLATE. Its [FILL: ...] values will fail")
        print("  the audit until you replace them --- they are yours to "
              "write,")
        print("  and the run continues past that failure either way.")
    else:
        print(f"  using the existing {p}")
    return str(p)


def phase_experiment(a) -> int:
    mx = resolve_max_det(a)
    a._resolved_max_det = mx
    decl = ensure_declarations(a)

    argv = ["--root", a.root, "--out", a.out, "--results", a.results,
            "--eda", a.eda, "--verify", a.verify,
            "--max-det", str(mx), "--device", a.device,
            "--condition", a.condition,
            "--depth", ("core" if a.quick else "full"),
            "--seed", str(a.seed),
            "--declarations", decl]
    if a.condition != "matched":
        print(f"\n  --condition {a.condition}: this is NOT the primary "
              f"condition.")
        print("  The registered endpoint is evaluated under matched "
              "post-processing.")
        print("  A run under another condition is a secondary analysis and "
              "the manuscript")
        print("  must label it post hoc. Write it to its own --results "
              "directory so it")
        print("  cannot overwrite the primary run's object.")
    if a.image_list:
        argv += ["--image-list", a.image_list]
    if a.yolov5_repo:
        argv += ["--yolov5-repo", a.yolov5_repo]
    if a.manuscript:
        argv += ["--manuscript", a.manuscript]
    if a.batch:
        argv += ["--batch", str(a.batch)]
    if a.imgsz:
        argv += ["--imgsz", str(a.imgsz)]
    if a.reuse_sensitivity:
        argv += ["--reuse-sensitivity"]
    # Protocol v3 SS0: on the real archive the comparator reconstructs
    # 105/170 against a published 135, so the reconstruction gate refuses.
    # mag's remedy is to force past it and have the refusal RECORDED
    # (reconVerdict, reconOK, meta.forced_past_reconstruction). Without the
    # flag the run ends at the gate with nothing written down.
    #
    # This USED TO BE UNCONDITIONAL, which made the gate a label rather than a
    # gate: no run could ever stop at it, while the manuscript described a
    # refusal the analysis had proceeded past. A peer reviewer found the
    # discrepancy by reading this file. The force is now an explicit,
    # defaulted-on choice that says so on the console and can be switched off,
    # so a reader of the log can see which it was.
    if a.force_reconstruction:
        argv += ["--force-reconstruction"]
        print("\n  --force-reconstruction is ON (the driver's default).")
        print("  The reconstruction gate cannot stop this run; its verdict is")
        print("  still recorded, and the manuscript reports the forcing.")
        print("  Pass --no-force-reconstruction to stop at a refusal instead.")
    else:
        print("\n  --force-reconstruction is OFF: a refusal at the")
        print("  reconstruction gate will end the experiment stage.")
    return run_script("run_experiment.py", argv)


# --------------------------------------------------------------------------- #
# bundle --- everything I need, in one file
# --------------------------------------------------------------------------- #

def phase_manuscript(a) -> int:
    """Refresh the manuscript's macro files from whatever has been produced.

    Two generators, both cheap and both idempotent:

      make_archive_macros.py  archive.tex, from eda.json, verify.json and the
                              geometry accounting beside the cache --- the
                              cohort, the annotations, the checkpoints, and
                              what the geometry step removed. Complete as soon
                              as those stages have run, which is long before
                              the analysis.
      missing_results.py      MISSING_RESULTS.md and outstanding.tex --- every
                              `\R{}` the manuscript calls, whether it resolves,
                              and which stage produces the ones that do not.

    This runs whether or not the analysis finished. A manuscript that says
    which of its numbers exist is more useful than one that waits for all of
    them, and the register shrinks by itself as stages complete.
    """
    rc = run_script("make_archive_macros.py",
                    ["--eda", a.eda, "--verify", a.verify,
                     "--geometry", str(Path(a.out) / "test"),
                     "--out", str(Path(a.manuscript_dir) / "archive.tex")])
    if rc:
        return rc

    run_script("make_result_tables.py",
               ["--results", a.results,
                "--out", str(Path(a.manuscript_dir) / "tables.tex")])

    src = Path(a.manuscript) if a.manuscript else \
        Path(a.manuscript_dir) / "main.tex"
    if not src.is_file():
        print(f"\n  no manuscript at {src}; archive.tex was still written.")
        print("  Pass --manuscript to also refresh the outstanding-results "
              "register.")
        return EXIT_OK
    # rc 1 from missing_results means "results are outstanding", which is a
    # statement about the run and not a failure of this phase.
    run_script("missing_results.py",
               ["--manuscript", str(src),
                "--macros", str(Path(a.manuscript_dir) / "archive.tex"),
                str(Path(a.manuscript_dir) / "tables.tex"),
                str(Path(a.results) / "results.tex"),
                "--known", str(HERE / "results_keys.txt"),
                "--md", str(Path(a.manuscript_dir) / "MISSING_RESULTS.md"),
                "--tex", str(Path(a.manuscript_dir) / "outstanding.tex")])

    # Files every document needs beside it: the shared macro machinery and the
    # bibliography. BibTeX and \input both resolve relative to the directory
    # the engine runs in, which is the manuscript's. Copying rather than
    # symlinking, because Windows without developer mode cannot make a symlink
    # and this suite targets Windows.
    import shutil
    for name, why in (("resultmacros.tex", "no document can resolve a macro"),
                      ("refs.bib", "citations will render as [?]")):
        s_src, s_dst = HERE / name, src.parent / name
        if not s_src.is_file() or s_src.resolve() == s_dst.resolve():
            continue
        try:
            shutil.copyfile(s_src, s_dst)
        except OSError as exc:                       # read-only tree, say
            print(f"\n  could not place {name} beside the manuscript "
                  f"({exc}); {why}.")

    # The talk and the poster read the same generated macro layer as the
    # paper, so they get the same register run over them: a slide is exactly
    # where an unresolved macro goes unnoticed, and a poster is what people
    # photograph and quote later.
    companions = [p for p in (src.parent / "slides.tex",
                              src.parent / "poster.tex") if p.is_file()]
    for doc in companions:
        print(f"\n  --- {doc.name} ---")
        run_script("missing_results.py",
                   ["--manuscript", str(doc),
                    "--macros", str(Path(a.manuscript_dir) / "archive.tex"),
                    str(Path(a.manuscript_dir) / "tables.tex"),
                    str(Path(a.results) / "results.tex"),
                    "--known", str(HERE / "results_keys.txt"),
                    "--md", str(doc.with_suffix(".missing.md")),
                    "--tex", str(doc.with_suffix(".outstanding.tex"))])

    if a.pdf:
        compile_pdf(src)
        for doc in companions:
            compile_pdf(doc)
    return EXIT_OK


#: TeX engines to try, in order. pdflatex is what the manuscript targets;
#: the other two read the same source.
TEX_ENGINES = ("pdflatex", "xelatex", "lualatex")


def compile_pdf(src: Path) -> int:
    """LaTeX, BibTeX, LaTeX, LaTeX, run where the manuscript lives.

    Four passes, not two. The cross-references (\ref to a table or figure)
    resolve on a second LaTeX pass, and the numbered citations need a BibTeX
    run between the first and the second --- the first pass writes the keys
    into main.aux, BibTeX turns them into main.bbl, and the two passes after
    it place the numbers and then settle the reference list's own page
    breaks. Run from the manuscript's own directory, because that is what
    `archive.tex`, `refs.bib`, `results/figures/` and `eda/figures/` are
    relative to.

    A missing BibTeX is not fatal: the pass is skipped, the citations render
    as [?], and the run says so. A missing TeX installation is reported and is
    NOT an error either: every artefact the PDF is built from has already been
    written, and the PDF can be produced anywhere, including by someone else.
    """
    import shutil
    import subprocess

    engine = next((e for e in TEX_ENGINES if shutil.which(e)), None)
    if engine is None:
        print("\n  no LaTeX engine on PATH (looked for "
              + ", ".join(TEX_ENGINES) + ").")
        print("  Every input the PDF needs has still been written. To build "
              "it here,")
        print("  install MiKTeX (https://miktex.org) or TeX Live, reopen the "
              "shell, and")
        print(f"  run:  {TEX_ENGINES[0]} {src.name}   (twice), with  bibtex "
              f"{src.stem}  between the two, from {src.parent}")
        return EXIT_OK

    # BibTeX only for a document that actually cites something. The talk and
    # the poster sit in the same directory as refs.bib and have no
    # \bibliography, and running BibTeX over their .aux files would print a
    # warning per build for no gain.
    cites = "\\bibliography{" in src.read_text(encoding="utf-8",
                                               errors="replace")
    bibtex = shutil.which("bibtex") \
        if cites and (src.parent / "refs.bib").is_file() else None
    plan = f"{engine}, bibtex, {engine} x2" if bibtex else f"{engine} x2"
    print(f"\n  {plan}  ({src.name}, from {src.parent})")
    if cites and (src.parent / "refs.bib").is_file() and not bibtex:
        print("    no bibtex on PATH: citations will render as [?]. "
              "Everything else builds.")

    log = ""
    passes = (1, 2, 3) if bibtex else (1, 2)
    for k in passes:
        r = subprocess.run(
            [engine, "-interaction=nonstopmode", "-halt-on-error", src.name],
            cwd=str(src.parent), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
            errors="replace")
        log = r.stdout or ""
        if r.returncode:
            print(f"    pass {k} failed. The last lines of its output:")
            for line in log.strip().splitlines()[-15:]:
                print("      " + line)
            print(f"    The full log is {src.with_suffix('.log')}")
            return EXIT_OK          # a failed build is not a failed RUN
        if bibtex and k == 1:
            b = subprocess.run(
                [bibtex, src.stem], cwd=str(src.parent),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                encoding="utf-8", errors="replace")
            # BibTeX returns non-zero for a warning as well as for an error,
            # so its output is reported and the build continues: an
            # unformatted reference must not cost the paper its PDF.
            for line in (b.stdout or "").splitlines():
                if re.search(r"(?i)\b(error|warn|couldn't|didn't)\b", line):
                    print("      bibtex: " + line.strip())

    undefined = sorted(set(re.findall(
        r"Citation `([^']+)' on page \d+ undefined", log)))
    if undefined:
        print(f"    {len(undefined)} citation key(s) with no entry in "
              f"refs.bib: " + ", ".join(undefined[:6]))

    pdf = src.with_suffix(".pdf")
    unresolved = len(re.findall(r"macro (\w+) was not produced", log))
    names = sorted(set(re.findall(r"macro (\w+) was not produced", log)))
    if pdf.is_file():
        # "1 page", not "1 pages": the poster is a single page and reported
        # its size as nothing at all until this regex stopped requiring the s.
        pages = re.search(r"Output written on .*?\((\d+) pages?", log)
        n = int(pages.group(1)) if pages else 0
        print(f"    {pdf}"
              + (f"  ({n} page{'' if n == 1 else 's'})" if pages else ""))
    if names:
        print(f"    {len(names)} macro(s) still render red, {unresolved} "
              f"occurrence(s). Appendix A lists them.")
    else:
        print("    every macro resolved.")
    return EXIT_OK


def phase_bundle(a) -> int:
    results = Path(a.results)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    zpath = results / f"handback_{stamp}.zip"

    # FLUSH THE LOG BEFORE READING IT. run_all.log is written through a
    # buffered file object, and this function zips that file from disk. The
    # first handback came back with the log cut off at 16,317 bytes --- one
    # buffer short --- in the middle of the findings ledger, so the bundle
    # carried no record of why the run had stopped eighteen minutes later.
    # The complete file was on the machine all along; only the copy in the
    # zip was short.
    for s in (sys.stdout, sys.stderr):
        try:
            s.flush()
        except Exception:                                     # noqa: BLE001
            pass

    manuscript_files = []
    roots = [(Path(a.eda), "eda"), (Path(a.verify), "verify"),
             (results, "results")]
    for extra in sorted(Path(".").glob(f"{results.name}_cap*")):
        if extra.is_dir():
            roots.append((extra, extra.name))
    # The cache's own small JSON --- manifests, checkpoint_meta, the
    # geometry-exclusion accounting --- says what the detection stage
    # actually produced. Tables and shards stay out: they are gigabytes.
    for st in ("test", "validation", "train"):
        cs = Path(a.out) / st
        if cs.is_dir():
            roots.append((cs, f"cache_{st}", "*.json"))
    md = Path(getattr(a, "manuscript_dir", ".") or ".")
    for name in ("archive.tex", "tables.tex", "outstanding.tex",
                 "MISSING_RESULTS.md", "main.bbl", "main.pdf",
                 "slides.pdf", "poster.pdf"):
        if (md / name).is_file():
            manuscript_files.append((md / name, f"manuscript/{name}"))

    manifest = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p, arc in manuscript_files:
            z.write(p, arc)
            manifest.append({"file": arc, "bytes": p.stat().st_size,
                             "sha256": hashlib.sha256(
                                 p.read_bytes()).hexdigest()})
        for entry in roots:
            root, label = entry[0], entry[1]
            pattern = entry[2] if len(entry) > 2 else "*"
            if not root.is_dir():
                continue
            for p in sorted(root.rglob(pattern)):
                if not p.is_file():
                    continue
                if p.name.startswith("handback_") and p.suffix == ".zip":
                    continue                      # never bundle a bundle
                arc = f"{label}/{p.relative_to(root)}"
                z.write(p, arc)
                h = hashlib.sha256(p.read_bytes()).hexdigest()
                manifest.append({"file": arc, "bytes": p.stat().st_size,
                                 "sha256": h})
        summary = paste_back(a, to_console=False)
        z.writestr("HANDBACK.md",
                   "# Handback\n\n"
                   "Send this zip back into the conversation. It carries the "
                   "complete\nresults layer, both analysis arms, the findings "
                   "ledger, the audit\nreport, and the full run log.\n\n"
                   "## Headline summary\n\n```\n" + summary + "\n```\n")
        z.writestr("manifest.json", json.dumps(manifest, indent=1))

    mb = zpath.stat().st_size / 1e6
    print(f"  {zpath}  ({mb:.1f} MB, {len(manifest)} files)")
    if mb > 40:
        print("  ! larger than expected. If it is too big to upload, send at")
        print("    least: results/run_all.log, results/RESULTS.md,")
        print("    results/manuscript_values.json, verify/verify.json, "
              "eda/eda.json")
    return EXIT_OK


# --------------------------------------------------------------------------- #
# the paste-back summary
# --------------------------------------------------------------------------- #

def _g(d, *keys, default="--"):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return default if cur is None else cur


def paste_back(a, to_console=True) -> str:
    import run_experiment as RX

    results = Path(a.results)
    res = _read_json(results / "results.json") or {}
    cap = _read_json(results / "cap_sensitivity.json") or {}
    audit = _read_json(results / "audit_report.json") or {}
    state = _read_json(results / "run_state.json") or {}
    verify = _read_json(Path(a.verify) / "verify.json") or {}
    macros = RX.read_macros(results / "results.tex")

    conf = res.get("confirmatory") or {}
    # reconstruction is the GateResult's detail, flattened by _summary: the
    # counts live at the top level beside `passed`.
    rd = res.get("reconstruction") or {}
    mc = conf.get("mcnemar") or {}
    ni = conf.get("noninferiority") or {}
    iu = conf.get("iu") or {}
    mx_shown = (getattr(a, "_resolved_max_det", None) or a.max_det
                or macros.get("capMaxDet", "--"))

    L = ["================= PASTE-BACK SUMMARY (share this block) "
         "=================",
         f"generated   {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}",
         f"provenance  {macros.get('provenance', '--')}",
         f"max_det     {mx_shown}   capBinding: "
         f"{macros.get('capBinding', '--')}",
         "",
         f"reconstruction  {_g(rd, 'n_correct')}/{_g(rd, 'n_total')} vs "
         f"published {_g(rd, 'published')} (delta {_g(rd, 'delta')}, "
         f"tolerance {_g(rd, 'tolerance')})",
         f"  gate passed: {_g(rd, 'passed')}   forced past: "
         f"{_g(res, 'meta', 'forced_past_reconstruction')}   "
         f"checkpoint: {_g(rd, 'checkpoint')}",
         "",
         f"confirmatory [{conf.get('checkpoint', '--')}]  "
         f"{conf.get('rule', '--')} vs {conf.get('comparator', '--')}",
         f"  specificity {_g(conf, 'spec_base')} -> {_g(conf, 'spec_new')}"
         f"  delta {_g(conf, 'spec_delta')}  CI {_g(conf, 'spec_delta_ci')}",
         f"  sensitivity {_g(conf, 'sens_base')} -> {_g(conf, 'sens_new')}"
         f"  (n_pos {_g(conf, 'n_pos')}, n_neg {_g(conf, 'n_neg')})",
         f"  McNemar b={_g(mc, 'b')} c={_g(mc, 'c')} p={_g(mc, 'p')} "
         f"({_g(mc, 'alternative')})  valid={_g(mc, 'valid')}",
         f"  non-inferiority: non_inferior={_g(ni, 'non_inferior')}  "
         f"diff={_g(ni, 'diff')}  tango_lower={_g(ni, 'tango_lower')}  "
         f"margin={_g(ni, 'margin')}",
         f"  intersection-union reject: {_g(iu, 'reject')}  "
         f"components: {_g(iu, 'components')}",
         f"  primaryVerdict macro: {macros.get('primaryVerdict', '--')}",
         ""]

    if cap:
        moved = [r for r in (cap.get("by_rule") or []) if r.get("changed")]
        L.append(f"cap sensitivity  {cap.get('cells_changed', '--')} of "
                 f"{cap.get('cells_compared', '--')} patient-rule cells "
                 f"changed ({cap.get('cap_high')} vs {cap.get('cap_low')})")
        for r in moved[:5]:
            L.append(f"  {r['rule']:<8} {r['changed']} of {r['n']}")
        L.append("")

    fired = []
    try:
        import csv
        with open(results / "findings.csv", newline="",
                  encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if str(row.get("fired")).lower() == "true":
                    fired.append(f"{row['key']}:{row['severity']}")
    except OSError:
        pass
    L.append(f"findings fired  {len(fired)}: {', '.join(fired) or '--'}")
    L.append(f"audit           {audit.get('n_checks', '--')} checks, "
             f"{'PASSED' if audit.get('passed') else str(audit.get('n_failed', '--')) + ' failed'}"
             f" (declarations supplied: "
             f"{audit.get('declarations_supplied', '--')})")
    L.append(f"verify verdict  {verify.get('verdict', '--')}")
    if state.get("stages"):
        L.append("stage timings   " + "  ".join(
            f"{k}:{v.get('seconds', '?')}s"
            for k, v in state["stages"].items()))
    L.append("=" * 74)
    text = "\n".join(str(x) for x in L)
    if to_console:
        print()
        print(text)
    return text


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_all.py",
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", "--archive", dest="root", default=".",
                   help="the archive, or a folder above it")
    p.add_argument("--yolov5-repo", dest="yolov5_repo", default=None,
                   help="path to a yolov5 checkout, e.g. "
                        r"..\malaria_france_data_set\models\models\yolov5")
    p.add_argument("--image-list", dest="image_list", default=None)
    p.add_argument("--manuscript", default=None,
                   help="main.tex, for the audit's macro-coverage, the "
                        "v2->v3 wording guard and the outstanding-results "
                        "register")
    # A gate that is bypassed unconditionally is a label. Default on, because
    # the registered endpoint cannot be evaluated otherwise and the protocol
    # permits the forcing; switchable, because a reader must be able to see
    # that it was a choice.
    p.add_argument("--force-reconstruction", dest="force_reconstruction",
                   action="store_true", default=True,
                   help="proceed past a refusal at the reconstruction gate "
                        "(default; the refusal is recorded either way)")
    p.add_argument("--no-force-reconstruction", dest="force_reconstruction",
                   action="store_false",
                   help="stop the experiment stage at a reconstruction refusal")
    p.add_argument("--manuscript-dir", dest="manuscript_dir", default=".",
                   help="where archive.tex, outstanding.tex and "
                        "MISSING_RESULTS.md are written (default: beside "
                        "main.tex)")
    p.add_argument("--pdf", action="store_true",
                   help="also compile the manuscript (two LaTeX passes). "
                        "Needs pdflatex on PATH; without it the inputs are "
                        "still written and the run continues")
    p.add_argument("--declarations", default=None,
                   help="declarations.json; a template is created if absent")
    p.add_argument("--phases", nargs="+", default=list(PHASES),
                   choices=PHASES, help="run only these, in this order")
    p.add_argument("--device", default="0")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    # The confirmatory comparator is a constant function under `matched`,
    # because that condition applies no confidence floor. The same rule on the
    # same cache separates most of the negatives under `source`. Running the
    # paired comparison there is a secondary, post hoc analysis and a reviewer
    # will ask for it, so the switch is here rather than only in
    # run_experiment.py -- with its own --results directory, so it cannot
    # overwrite the registered run.
    p.add_argument("--condition", choices=("raw", "matched", "source"),
                   default="matched",
                   help="post-processing condition for the experiment stage. "
                        "matched is primary and registered; any other value "
                        "produces a secondary analysis that the manuscript "
                        "must label post hoc")
    p.add_argument("--max-det", dest="max_det", type=int, default=0,
                   help="override the cap; 0 = resolve from the cap probe")
    p.add_argument("--quick", action="store_true",
                   help="depth core instead of full: a fast first pass to "
                        "prove the plumbing before the real one")
    p.add_argument("--reuse-sensitivity", dest="reuse_sensitivity",
                   action="store_true",
                   help="reuse an existing arm B cache")
    p.add_argument("--no-gpu-check", dest="no_gpu_check", action="store_true")
    p.add_argument("--show-warnings", dest="show_warnings",
                   action="store_true",
                   help="keep the yolov5 fork's torch.cuda.amp deprecation "
                        "warnings; by default they are counted and dropped, "
                        "because they print once per batch and make the "
                        "detect stage's progress unreadable")
    p.add_argument("--eda", default="eda")
    p.add_argument("--verify", default="verify")
    p.add_argument("--out", default="data")
    p.add_argument("--results", default="results")
    return p


RUNNERS = {"env": phase_env, "eda": phase_eda, "verify": phase_verify,
           "experiment": phase_experiment, "manuscript": phase_manuscript,
           "bundle": phase_bundle}


def main(argv=None) -> int:
    global SHOW_WARNINGS
    a = build_parser().parse_args(argv)
    a._resolved_max_det = a.max_det or None
    SHOW_WARNINGS = bool(a.show_warnings)

    # A PowerShell console can be cp1252; never let an encoding kill a run
    # that has an hour of GPU time in it.
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except Exception:                                     # noqa: BLE001
            pass

    Path(a.results).mkdir(parents=True, exist_ok=True)
    log = open(Path(a.results) / "run_all.log", "a",
               encoding="utf-8", errors="replace")
    real_out, real_err = sys.stdout, sys.stderr
    sys.stdout = Tee(real_out, log)
    sys.stderr = Tee(real_err, log)

    try:
        fp = suite_fingerprint()
        print("#" * 78)
        print(f"# malaria experiment suite v{SUITE_VERSION}")
        print(f"# run_all.py  {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
        print(f"# from     {HERE}")
        print(f"# phases   {' '.join(a.phases)}")
        print("# files    " + "  ".join(f"{n.split('.')[0]}:{h}"
                                        for n, h in fp.items()))
        print("#" * 78)
        if any(v == "MISSING" for v in fp.values()):
            print("  ! a suite file is missing from this folder:",
                  ", ".join(n for n, v in fp.items() if v == "MISSING"))
            print("    Unpack the whole package into one folder and re-run.")
            return EXIT_PRECONDITION

        worst = EXIT_OK
        for name in a.phases:
            hr(name)
            t0 = time.time()
            try:
                rc = RUNNERS[name](a)
            except SystemExit as exc:          # producers raise on missing deps
                rc = int(exc.code or 0)
            except KeyboardInterrupt:
                print("\n  interrupted during '%s'." % name)
                print("  Nothing is lost: re-run the SAME command. Detection "
                      "resumes --- every")
                print("  smear already cached is skipped, and a checkpoint "
                      "that finished is not")
                print("  re-inferred. eda and verify are cheap and simply "
                      "run again.")
                return 130
            print(f"\n  [{name} finished in {time.time() - t0:.0f}s, rc={rc}]")
            worst = max(worst, rc)
            # audit failures surface as experiment rc=5; the bundle and the
            # summary must still be produced --- they are how the failure
            # travels back for interpretation.
            if rc and name in ("env", "eda"):
                print(f"\n  {name} failed; nothing later can run without it.")
                break
            if rc and name == "verify" and rc != EXIT_OK:
                print("\n  verify failed beyond the known saturation; "
                      "stopping.")
                break

        if "bundle" in a.phases or "experiment" in a.phases:
            paste_back(a)
        return worst
    finally:
        sys.stdout, sys.stderr = real_out, real_err
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
