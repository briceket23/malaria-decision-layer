#!/usr/bin/env python3
"""reproduce.py --- the paper's numbers, from the archive, on Windows or Linux.

    python reproduce.py --root D:\\malaria_france_data_set

That is the whole interface for the common case.

WHAT THIS IS, AND WHAT IT IS NOT

`mag` 3.0.0 already does the work. It ships five stages --- doctor, ingest,
detect, analyse, audit --- and a `run_local.py` that runs them in order. **This
script does not reimplement any of them.** It calls `mag.cli.main()` for every
stage, exactly as `run_local.py` does.

It exists for three things `mag` 3.0.0 does not do, and nothing else:

  1. **Six macros the manuscript cites and `mag` does not emit.**
     `jackknifeRange`, `errOvaleVivaxShare`, `riskConfCov90`, `aurcConf`,
     `nArmsAnalysed`, `reconstructionShortfall`. Each is a composition of
     numbers `mag` already computes, and each goes in through the documented
     `report.write(..., extra=...)` hook. Patching `mag` to add them would mean
     editing a package whose defects are pinned by its own test suite, to add
     arithmetic it already has the inputs for.

  2. **The v2 -> v3 manuscript guard.** Protocol v3 SS9 retracts the v2
     apportionment claim and SS9.4 forbids specific wording. A manuscript that
     still types "the rule matters more than the detector", or a multiplier
     computed from a ratio of ranges, is asserting something the analysis no
     longer supports. This script greps for those before you typeset.

  3. **A version guard.** `mag` 1.0.0 and 3.0.0 differ in ways that move
     numbers, not just names: `BASELINE`/`CONFIRMATORY_RULE` became
     `COMPARATOR`/`CANDIDATE`, `reproduction` became `reconstruction`
     (v3 SS3: the published result is external, so it is reconstructed, not
     reproduced), and the apportionment was replaced by a patient-clustered
     GEE. This script requires 3.x and says so rather than failing obscurely.

If you do not need those three, **use `run_local.py` instead** --- it is the
maintained path and this script has no advantage over it.

WHY NOT A .sh

PowerShell cannot execute a `.sh` file. Typing `.\\something.sh` hands it to
whatever application is registered for the extension --- normally an editor,
which opens it and prints its own startup log --- and nothing runs, with no
error. This is Python for that reason.

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
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

EXIT_OK, EXIT_PRECONDITION, EXIT_REFUSAL, EXIT_UNEXPECTED = 0, 2, 3, 4

STAGES = ("doctor", "ingest", "detect", "analyse", "audit")

#: Only `detect` needs a GPU, and only `doctor`/`ingest`/`detect` need the
#: archive. Starting from a cache means `--stages analyse audit`.
NEEDS_ARCHIVE = ("doctor", "ingest", "detect")
NEEDS_GPU = ("detect",)


# --------------------------------------------------------------------------- #
# 1. The six macros mag 3.0.0 does not emit.
#
# Every one is a composition of numbers mag already computes. They return {}
# rather than a placeholder when their input is absent, because mag renders an
# undefined \R{} RED in the PDF and a placeholder would render as a value --
# which is the failure the whole \R mechanism exists to prevent.
# --------------------------------------------------------------------------- #

DERIVED_KEYS = ("jackknifeRange", "errOvaleVivaxShare", "riskConfCov90",
                "aurcConf", "nArmsAnalysed", "reconstructionShortfall")


def _pct(x, dp=1):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return f"{100 * float(x):.{dp}f}\\%"


def _num(x, dp=4):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    return f"{float(x):.{dp}f}"


def derived_macros(s) -> dict:
    """The six, from a built Study. Absent inputs produce absent keys."""
    out = {}

    # --- jackknifeRange: the spread of (rule - checkpoint) across refits ---- #
    # v3 keeps the descriptive ranges (SS9.3) alongside the GEE attribution, and
    # leaves each checkpoint out in turn. The RANGE of the difference is what
    # says how fragile the ordering is. Needs >= 2 refits: a "range" over one
    # value is a point, and printing it as a range overstates what was assessed.
    ap = getattr(s, "apportionment", None) or {}
    diffs = [v["spread_rule"] - v["spread_checkpoint"]
             for v in (ap.get("leave_one_checkpoint_out") or {}).values()
             if v.get("spread_rule") is not None
             and v.get("spread_checkpoint") is not None]
    if len(diffs) >= 2:
        out["jackknifeRange"] = (f"{100 * min(diffs):.1f} to "
                                 f"{100 * max(diffs):.1f} percentage points")

    # --- errOvaleVivaxShare: the count, as a share ------------------------- #
    # mag emits the count. A count is uninterpretable without its denominator,
    # and the denominator moves between arms.
    es = getattr(s, "error_summary", None) or {}
    if es.get("n_errors"):
        share = _pct((es.get("categories") or {}).get("ovale_vivax", 0)
                     / es["n_errors"])
        if share:
            out["errOvaleVivaxShare"] = share

    # --- riskConfCov90 / aurcConf: the CANDIDATE rule's abstention numbers -- #
    # mag keys these by rule id (risk9R4, aurcR4) because the candidate is
    # chosen by the design gate and its id is not known until that has run. The
    # manuscript needs a stable name. The id is resolved from the study's own
    # confirmatory record, NEVER hard-coded: a hard-coded R4 would print the
    # wrong rule's risk the moment the selection changed, and print it
    # confidently.
    rid = (getattr(s, "confirmatory", None) or {}).get("rule")
    rc = (getattr(s, "selective", None) or {}).get(rid)
    if rid and isinstance(rc, dict):
        if rc.get("aurc") is not None:
            out["aurcConf"] = _num(rc["aurc"])
        op = (rc.get("operating") or {}).get("0.9")
        if op and op.get("risk") is not None:
            out["riskConfCov90"] = (f"{_pct(op['risk'])} with "
                                    f"{op['n_referred']} patients referred")

    # --- nArmsAnalysed ----------------------------------------------------- #
    # mag emits `armsAnalysed` (a joined list of display names) and `nArms`.
    # The limitation about the checkpoint factor is stated against the count.
    fac = getattr(s, "factorial", None)
    if fac is not None and len(fac):
        out["nArmsAnalysed"] = str(fac["checkpoint"].nunique())

    # --- reconstructionShortfall ------------------------------------------- #
    # v3 SS3: this is a RECONSTRUCTION, not a reproduction -- the published
    # result is an external claim about an external run. The manuscript states
    # the gap as a number of patients; subtracting two macros in LaTeX is a
    # place for arithmetic to be wrong.
    detail = getattr(getattr(s, "reconstruction", None), "detail", None) or {}
    if detail.get("n_correct") is not None and detail.get("published") is not None:
        gap = int(detail["n_correct"]) - int(detail["published"])
        out["reconstructionShortfall"] = (
            f"{abs(gap)} patient{'s' if abs(gap) != 1 else ''}"
            + (" above" if gap > 0 else " below" if gap < 0 else ""))
    return out


# --------------------------------------------------------------------------- #
# 2. The v2 -> v3 manuscript guard.
# --------------------------------------------------------------------------- #

#: Protocol v3 SS9.4 forbids these outright. A manuscript still carrying them is
#: asserting something the v3 analysis does not support.
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

#: Macros v3 still defines but whose v2 READING was retracted. Not an error to
#: cite them -- SS9.3 keeps them as descriptive ranges -- but they must not
#: carry the causal claim they carried in v2.
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


def check_manuscript(path) -> list:
    """Grep a manuscript for wording protocol v3 retracted. Returns findings.

    This is the same check `run_experiment.check_manuscript` runs; the two
    copies are kept identical in behaviour by
    `test_the_two_copies_of_the_wording_guard_agree`.
    """
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
    for pattern, why in FORBIDDEN:
        for m in re.finditer(pattern, text, re.I):
            line = text[:m.start()].count("\n") + 1
            found.append(("forbidden", f"line {line}: {m.group(0)!r}", why))
    cited = set(re.findall(r"\\R\{([A-Za-z0-9]+)\}", raw))
    for k in sorted(cited & set(DEMOTED)):
        found.append(("demoted", f"\\R{{{k}}}",
                      "v3 SS9.3 keeps this as a DESCRIPTIVE range; it must not "
                      "carry the v2 causal reading"))
    return found


# --------------------------------------------------------------------------- #
# 3. Stage dispatch. Every stage is mag's own.
# --------------------------------------------------------------------------- #

def hr(title=""):
    print("\n" + ("--- " + title + " " if title else "").ljust(78, "-"))


def require_mag():
    """Import mag, require 3.x, and say why rather than failing obscurely."""
    try:
        import mag
    except ImportError as exc:
        print(f"\nmag is not installed in {sys.executable} ({exc}).\n",
              file=sys.stderr)
        print(f'    "{sys.executable}" -m pip install -e .', file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)

    major = str(getattr(mag, "__version__", "0")).split(".")[0]
    if major != "3":
        print(f"\nthis script targets mag 3.x and found "
              f"{mag.__version__}.\n", file=sys.stderr)
        print("  mag 1.x and 3.x differ in ways that move NUMBERS, not just "
              "names:\n"
              "    - BASELINE/CONFIRMATORY_RULE  ->  COMPARATOR/CANDIDATE\n"
              "    - reproduction  ->  reconstruction (v3 SS3)\n"
              "    - the apportionment was replaced by a patient-clustered "
              "GEE (v3 SS9)\n"
              "  Running v3 prose against a v1 analysis, or the reverse, "
              "produces a\n  manuscript whose methods describe code that did "
              "not run.", file=sys.stderr)
        raise SystemExit(EXIT_PRECONDITION)
    return mag


def stage_analyse(a, d) -> int:
    """The one stage this script takes over, and only to add six macros.

    Mirrors `mag.cli.cmd_all` call for call --- build once, write outputs,
    draw figures, CHECK the figures against the study that is about to write
    the text, then write the macros --- with `extra=` on the last step.

    Building once is the whole point. `run`, `figures` and `report` each
    rebuild the study, so calling them in sequence does the work three times
    and can produce figures and text from two different builds. That is not
    hypothetical: an abstract once said the co-primary endpoint was met while
    the results section said it was not.
    """
    from mag import cli as mag_cli
    from mag import figures, report

    cache = d.cache if hasattr(d, "cache") else Path(a.out) / a.set
    patients = Path(a.out) / "patients.csv"
    if not patients.is_file():
        print(f"\nno patients.csv at {patients}.", file=sys.stderr)
        print("  Run the ingest stage first:  "
              "python reproduce.py --root ARCHIVE --stages ingest",
              file=sys.stderr)
        return EXIT_PRECONDITION

    # Reuse mag's own argument parser so this cannot drift from `mag all`.
    argv = ["all", "--cache", str(cache), "--patients", str(patients),
            "--out", str(a.results), "--condition", a.condition,
            "--depth", a.depth, "--seed", str(a.seed)]
    if a.force_reconstruction:
        argv.append("--force-reconstruction")
    ns = mag_cli.build_parser().parse_args(argv)

    s = mag_cli._build(ns)
    out = Path(a.results)
    written = mag_cli._write_outputs(s, out)
    stamp = figures.write_all(s, out / "figures")
    figures.check(out / "figures", s)          # refuses a mismatched pair

    extra = derived_macros(s)
    keys = report.write(s, out / "results.tex", extra=extra)
    mag_cli._report_to_console(s, a.condition)

    print(f"\nwrote {len(written)} result files, {len(stamp['figures'])} "
          f"figures and {len(keys)} macros")
    for why in stamp["skipped"]:
        print(f"  ! figure skipped: {why}")

    hr("macros this script added")
    for k in DERIVED_KEYS:
        if k in extra:
            print(f"  {k:<26} {extra[k]}")
        else:
            print(f"  {k:<26} NOT PRODUCED -- stays RED in the PDF")
    print("\n  A macro the analysis never computed renders red rather than")
    print("  blank, so an unfilled slot cannot pass silently. jackknifeRange")
    print("  needs three surviving arms; the selective pair needs the")
    print("  abstention analysis to have run.")

    print(f"\nprovenance: {s.provenance.line()}")
    return 0


def run_stage(a, name, d) -> int:
    """Dispatch one stage. Everything but `analyse` is mag's, unmodified."""
    from mag import cli as mag_cli
    from mag import discover

    hr(name)

    if name == "doctor":
        print(discover.format_report(d, discover.environment()))
        problems = discover.preflight(d, need_gpu="detect" in a.stages)
        for p in problems:
            print(f"  ! {p}")
        return EXIT_PRECONDITION if problems else EXIT_OK

    if name == "ingest":
        argv = ["ingest", "--image-list", str(d.image_list),
                "--set", a.set, "--out", str(a.out)]
        if getattr(d, "archive", None):
            argv += ["--archive", str(d.archive)]
        if getattr(d, "classes", None):
            argv += ["--classes", str(d.classes)]
        if a.zoom_set:
            argv += ["--zoom-set", a.zoom_set]
        return mag_cli.main(argv)

    if name == "detect":
        rc = 0
        for images in (d.image_sets or []):
            argv = ["detect", "--images", str(images),
                    "--out", str(Path(a.out) / a.set),
                    "--patients", str(Path(a.out) / "patients.csv"),
                    "--conf", str(a.conf), "--device", a.device]
            for spec in (a.weights or []):
                argv += ["--weights", spec]
            if a.no_resume:
                argv.append("--no-resume")
            rc = max(rc, mag_cli.main(argv))
        return rc

    if name == "analyse":
        return stage_analyse(a, d)

    if name == "audit":
        argv = ["audit", "--out", str(a.results)]
        if a.manuscript:
            argv += ["--manuscript", a.manuscript]
        if a.declarations:
            argv += ["--declarations", a.declarations]
        rc = mag_cli.main(argv)

        # The v2 -> v3 guard, after mag's own audit. Never fatal: it is about
        # PROSE, and the author is the only one who can fix prose.
        if a.manuscript:
            found = check_manuscript(a.manuscript)
            hr("protocol v3 wording guard")
            if not found:
                print("  no retracted v2 wording found.")
            for kind, what, why in found:
                print(f"  [{kind}] {what}")
                if why:
                    print(f"           {why}")
            if found:
                print("\n  These are PROSE findings, not analysis failures. "
                      "Protocol v3 SS9\n  retracts the v2 apportionment claim; "
                      "SS9.4 gives the required wording.")
        return rc

    print(f"unknown stage {name!r}", file=sys.stderr)
    return EXIT_UNEXPECTED


# --------------------------------------------------------------------------- #

def build_parser():
    ap = argparse.ArgumentParser(
        prog="reproduce.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""EXAMPLES

  # Everything, from the unzipped archive. Windows:
  python reproduce.py --root D:\malaria_france_data_set

  # The analysis only, from a cache you already have:
  python reproduce.py --root D:\malaria_france_data_set --stages analyse audit

  # Re-emit results and check the manuscript's wording:
  python reproduce.py --root . --stages analyse audit --manuscript ..\paper\main.tex

  # The reconstruction gate refused, and you want to continue with it recorded:
  python reproduce.py --root . --stages analyse --force-reconstruction
""")
    # `--root` is mag 3.0.0's own name for this (run_local.py --root), and is
    # the one to use. `--archive` is accepted because an earlier version of
    # this script called it that, and a renamed flag that fails with
    # "unrecognized arguments" costs a run for no reason.
    ap.add_argument("--root", "--archive", dest="root", default=".",
                    metavar="DIR",
                    help="the unzipped archive: the directory holding images/ "
                         "and image_list.xlsx. --archive is an accepted alias.")
    ap.add_argument("--stages", nargs="+", default=list(STAGES),
                    choices=list(STAGES),
                    help=f"default: {' '.join(STAGES)}")
    ap.add_argument("--out", default="data", help="cache and patients.csv")
    ap.add_argument("--results", default="results", help="results.tex and figures")
    ap.add_argument("--set", default="test")
    ap.add_argument("--zoom-set", dest="zoom_set", default="test-zoom")
    ap.add_argument("--condition", default="matched",
                    help="checkpoint comparisons are meaningful only in matched")
    ap.add_argument("--depth", default="full", choices=["core", "full"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="the cache is stored at this floor, PRE-suppression")
    ap.add_argument("--device", default="0", help="'0' for the first GPU, or 'cpu'")
    ap.add_argument("--weights", nargs="*", metavar="NAME=PATH", default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--force-reconstruction", action="store_true",
                    help="continue past a failed reconstruction gate, recorded")
    ap.add_argument("--manuscript", default=None,
                    help="main.tex; enables the audit's slot check and the "
                         "protocol v3 wording guard")
    ap.add_argument("--declarations", default=None,
                    help="declarations.json; author-supplied, checked for "
                         "presence and never filled in")
    return ap


#: Flags an earlier version of this script had, and where they went. argparse
#: says "unrecognized arguments: --x" and stops there, which for a renamed flag
#: is a dead end.
RETIRED = {
    "--cache": "the cache lives under --out (default: data/<set>); there is no "
               "separate flag. To analyse an existing one: "
               "--stages analyse audit --out DIR",
    "--no-gpu": "name the stages instead: --stages analyse audit",
    "--from": "name the stages instead: --stages analyse audit",
    "--to": "name the stages instead: --stages doctor",
    "--force": "mag 3.0.0 stages are idempotent by input, not by receipt; "
               "just re-run",
    "--force-reproduction": "renamed --force-reconstruction (v3 SS3: the "
                            "published result is external, so it is "
                            "reconstructed, not reproduced)",
    "--url": "remote indexing is gone; mag 3.0.0 reads the archive directly",
    "--configs": "mag 3.0.0 carries its protocol in code, not in configs/",
    "--dry-run": "not implemented here; use --stages doctor",
}


def main(argv=None) -> int:
    argv_in = list(sys.argv[1:] if argv is None else argv)
    retired = [x for x in argv_in if x.split("=")[0] in RETIRED]
    if retired:
        print("\nthese flags are from an earlier version of this script:\n",
              file=sys.stderr)
        for flag in retired:
            key = flag.split("=")[0]
            print(f"    {key}\n        {RETIRED[key]}\n", file=sys.stderr)
        print("  `python reproduce.py --help` lists what this version takes.",
              file=sys.stderr)
        return EXIT_PRECONDITION

    a = build_parser().parse_args(argv)
    mag = require_mag()
    from mag import discover
    from mag.errors import MagError

    print("=" * 78)
    print(" reproduce.py --- a thin layer over mag, plus six macros")
    print("=" * 78)
    print(f"  python   {sys.version.split()[0]}")
    print(f"  mag      {mag.__version__}  build {mag.build_id()}")
    print(f"  root     {Path(a.root).resolve()}")
    print(f"  stages   {' '.join(a.stages)}")
    print(f"  depth    {a.depth}   seed {a.seed}   condition {a.condition}")

    started = dt.datetime.now(dt.timezone.utc)
    try:
        d = discover.discover(a.root)
    except Exception as exc:                                 # noqa: BLE001
        needs = [s for s in a.stages if s in NEEDS_ARCHIVE]
        if needs:
            print(f"\ncould not read the archive at {a.root} ({exc})\n",
                  file=sys.stderr)
            print("  Point --root at the directory holding images/ and\n"
                  "  image_list.xlsx. If your extraction added a wrapping\n"
                  "  directory, go one level deeper.\n", file=sys.stderr)
            print(f"  Stages needing the archive: {', '.join(needs)}\n"
                  f"  To run without it:  --stages analyse audit",
                  file=sys.stderr)
            return EXIT_PRECONDITION
        d = None

    for name in STAGES:
        if name not in a.stages:
            continue
        try:
            rc = run_stage(a, name, d)
        except MagError as exc:
            print(f"\n{type(exc).__name__}: {exc}", file=sys.stderr)
            print("\n  This is a refusal mag anticipated and it carries a "
                  "remedy above.\n  If it is a GATE, it may be a FINDING "
                  "rather than a fault.", file=sys.stderr)
            return EXIT_REFUSAL
        except Exception as exc:                             # noqa: BLE001
            import traceback
            print(f"\n{type(exc).__name__}: {exc}\n", file=sys.stderr)
            traceback.print_exc()
            return EXIT_UNEXPECTED
        if rc != 0:
            print(f"\nstage {name} exited {rc}; stopping.", file=sys.stderr)
            return rc

    print("\n" + "=" * 78)
    print(" complete.")
    print("=" * 78)
    print(f"  started   {started.isoformat(timespec='seconds')}")
    print(f"  finished  "
          f"{dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")
    tex = Path(a.results) / "results.tex"
    if tex.is_file():
        print(f"\n  \\input{{{tex.as_posix()}}}")
        print("\n  Every \\R{} that renders RED is a macro this run did not")
        print("  produce. They are not typos.")
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
