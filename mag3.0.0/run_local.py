#!/usr/bin/env python3
"""One command, from the downloaded archive to the numbers the paper reports.

    python run_local.py --root /path/to/unzipped/archive

That is the whole interface for the common case. Everything below is for when
the common case does not apply.

WHAT IT DOES, IN ORDER

    doctor    what is on this machine, what is in the archive, what would block
    ingest    the archive's manifest -> patients.csv, plus the cohort census
    detect    the detectors over every field -> the detection cache
    analyse   the experiment -> results, figures, macros
    audit     the finished directory, checked before anything is typeset

Each stage is a `mag` subcommand and can be run on its own; this script runs
them in order, remembers what finished, and refuses to start a stage whose
inputs are not there. `--stages` picks a subset.

WHY IT IS A SCRIPT AND NOT A MAKEFILE

Because `detect` is the expensive stage and the only one that can be
interrupted in the middle of something worth keeping. It is resumable at smear
granularity, so a pass killed after 120 of 170 smears resumes at 121 -- and
that resume is only safe because each shard carries a fingerprint of the
parameters it was computed under. A Makefile's timestamp rule cannot express
that.

WHAT IT WILL NOT DO

It will not generate a cohort, invent a detection, or fall back to anything
when an input is missing. Every stage that cannot find what it needs stops and
says what is absent. The previous version of this project could produce a
results file with no archive present, and did; two external reviews were then
carried out on the resulting PDF. That is the failure this design exists to
make impossible rather than unlikely.

TIME AND SPACE

`ingest` takes seconds. `detect` is hours: on the released test partition, 170
smears and about 4,500 fields, three checkpoints. On a 6 GB card the batch size
is chosen for you and the pass fits. The cache is a few hundred megabytes; the
archive itself is about 6 GB and is only read.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from mag import cli as mag_cli                                  # noqa: E402
from mag import discover                                        # noqa: E402
from mag.errors import MagError                                 # noqa: E402

STAGES = ("doctor", "ingest", "detect", "analyse", "audit")


def hr(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}", flush=True)


def _state_path(out: Path) -> Path:
    return Path(out) / "run_state.json"


def _load_state(out: Path) -> dict:
    p = _state_path(out)
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:                                       # noqa: BLE001
        # a torn journal is not a reason to refuse to work; it is a reason to
        # forget what we thought we knew and check the artefacts instead
        return {}


def _save_state(out: Path, state: dict) -> None:
    p = _state_path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(p)


# --------------------------------------------------------------------------- #

def stage_doctor(a, d) -> int:
    hr("doctor -- what is here")
    print(discover.format_report(d, discover.environment()))
    problems = discover.preflight(d, need_gpu="detect" in a.stages)
    if problems:
        print("\nblocking:")
        for p in problems:
            print(f"  ! {p}")
        return 2
    print("\nno blocking problems.")
    return 0


def stage_ingest(a, d) -> int:
    hr("ingest -- the archive's manifest becomes the cohort")
    if not d.image_list:
        print("  ! image_list.xlsx was not found. It carries the patient "
              "grouping and the reference standard; nothing works without it.")
        return 2
    argv = ["ingest", "--image-list", str(d.image_list),
            "--set", a.set, "--out", str(a.out),
            "--min-fields", str(a.min_fields)]
    if d.archive:
        argv += ["--archive", str(d.archive)]
    if d.classes:
        argv += ["--classes", str(d.classes)]
    argv += ["--zoom-set", a.zoom_set or ""]
    return mag_cli.main(argv)


def stage_detect(a, d) -> int:
    hr("detect -- the detectors over every field")
    if not d.archive:
        print("  ! no archive directory; pass --archive with the directory "
              "that holds images/ and labels/")
        return 2
    weights = dict(d.weights)
    for spec in a.weights or []:
        name, _, path = spec.partition("=")
        weights[name] = Path(path).expanduser()
    if not weights:
        print("  ! no checkpoints found in the archive and none given. "
              "Pass --weights NAME=PATH.")
        return 2

    patients = Path(a.out) / "patients.csv"
    rc = 0
    for image_set in a.sets:
        images = Path(d.archive) / "images" / image_set
        if not images.is_dir():
            print(f"  . skipping '{image_set}': no such image directory")
            continue
        argv = ["detect", "--images", str(images),
                "--out", str(Path(a.out) / image_set),
                "--conf", str(a.conf), "--device", a.device,
                "--seed", str(a.seed),
                "--weights"] + [f"{k}={v}" for k, v in sorted(weights.items())]
        if patients.is_file():
            argv += ["--patients", str(patients)]
        if a.imgsz:
            argv += ["--imgsz", str(a.imgsz)]
        if a.batch:
            argv += ["--batch", str(a.batch)]
        if a.half is True:
            argv += ["--half"]
        elif a.half is False:
            argv += ["--no-half"]
        if a.yolov5_repo:
            argv += ["--yolov5-repo", a.yolov5_repo]
        if a.assume_classes_txt:
            argv += ["--assume-classes-txt"]
        if a.no_resume:
            argv += ["--no-resume"]
        rc = max(rc, mag_cli.main(argv))
    return rc


def stage_analyse(a, d) -> int:
    hr("analyse -- the experiment")
    cache = Path(a.out) / a.set
    patients = cache / "patients.csv"
    if not patients.is_file():
        patients = Path(a.out) / "patients.csv"
    if not patients.is_file():
        print(f"  ! no patients.csv under {a.out}. Run the ingest stage first.")
        return 2
    argv = ["all", "--cache", str(cache), "--patients", str(patients),
            "--out", str(a.results), "--tex", str(Path(a.results) / "results.tex"),
            "--condition", a.condition, "--depth", a.depth,
            "--seed", str(a.seed), "--min-fields", str(a.min_fields),
            "--reps", str(a.reps), "--stress-reps", str(a.stress_reps)]
    if a.force_reconstruction:
        argv.append("--force-reconstruction")
    return mag_cli.main(argv)


def stage_audit(a, d) -> int:
    hr("audit -- before anything is typeset")
    argv = ["audit", "--out", str(a.results), "--json"]
    if a.manuscript:
        argv += ["--manuscript", a.manuscript]
    if a.declarations:
        argv += ["--declarations", a.declarations]
    rc = mag_cli.main(argv)
    if rc and not a.declarations:
        print("\nThe declarations file is yours to write. It is a JSON object "
              "whose keys are listed above; every value is a fact about your "
              "study -- ethics, funding, CRediT, AI use, the repository URL, "
              "the registration and the code archive DOI -- and this pipeline "
              "has no way to know them and will not guess.")
    return rc


RUNNERS = {"doctor": stage_doctor, "ingest": stage_ingest,
           "detect": stage_detect, "analyse": stage_analyse,
           "audit": stage_audit}


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        "run_local.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=".",
                   help="the directory you unzipped the download into")
    p.add_argument("--archive", default=None,
                   help="override: the directory holding images/ and labels/")
    p.add_argument("--image-list", dest="image_list", default=None,
                   help="override: image_list.xlsx")
    p.add_argument("--weights", nargs="*", metavar="NAME=PATH", default=None,
                   help="checkpoints, in addition to any found in the archive")
    p.add_argument("--stages", nargs="+", default=list(STAGES),
                   choices=list(STAGES),
                   help="run only these, in the order given here")
    p.add_argument("--sets", nargs="+", default=["test"],
                   help="the partitions to run detection over")
    p.add_argument("--set", default="test",
                   help="the partition the cohort and the analysis come from")
    p.add_argument("--zoom-set", dest="zoom_set", default="test-zoom",
                   help="the second label tree, used to measure the source "
                        "study's transformation")
    p.add_argument("--out", default="data",
                   help="where patients.csv and the detection caches go")
    p.add_argument("--results", default="results",
                   help="where results, figures and macros go")
    p.add_argument("--condition", default="matched",
                   choices=["raw", "matched", "source"])
    p.add_argument("--depth", default="full", choices=["core", "full"])
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--device", default="0", help="'0' for the first GPU, or 'cpu'")
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--half", dest="half", action="store_true", default=None)
    p.add_argument("--no-half", dest="half", action="store_false")
    p.add_argument("--yolov5-repo", dest="yolov5_repo", default=None)
    p.add_argument("--assume-classes-txt", dest="assume_classes_txt",
                   action="store_true")
    p.add_argument("--no-resume", dest="no_resume", action="store_true")
    p.add_argument("--min-fields", dest="min_fields", type=int, default=25)
    p.add_argument("--reps", type=int, default=50)
    p.add_argument("--stress-reps", dest="stress_reps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--manuscript", default=None,
                   help="the .tex source, for the audit stage")
    p.add_argument("--declarations", default=None,
                   help="your declarations JSON, for the audit stage")
    p.add_argument("--force-reconstruction", dest="force_reconstruction",
                   action="store_true",
                   help="continue past a failed reconstruction gate. The gate "
                        "refusing is a FINDING; read it before using this")
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    root = Path(a.root).expanduser().resolve()
    d = discover.discover(root)
    if a.archive:
        d.archive = Path(a.archive).expanduser().resolve()
    if a.image_list:
        d.image_list = Path(a.image_list).expanduser().resolve()

    out = Path(a.out)
    state = _load_state(out)
    print(f"root      {root}")
    print(f"archive   {d.archive or '-- not found --'}")
    print(f"stages    {', '.join(a.stages)}")

    worst = 0
    for name in a.stages:
        t0 = time.time()
        try:
            rc = RUNNERS[name](a, d)
        except MagError as e:
            # the typed errors already carry a remedy; a traceback would bury it
            print(f"\n{type(e).__name__}: {e.problem}", file=sys.stderr)
            if e.remedy:
                print(f"\nWhat to do: {e.remedy}", file=sys.stderr)
            rc = 3
        except KeyboardInterrupt:
            print(f"\ninterrupted during '{name}'. Detection is resumable: "
                  "re-run the same command and it will pick up where it "
                  "stopped.", file=sys.stderr)
            return 130
        took = time.time() - t0
        state[name] = {"rc": rc, "seconds": round(took, 1)}
        _save_state(out, state)
        print(f"\n[{name}] {'ok' if rc == 0 else f'rc={rc}'} in {took:.1f}s")
        worst = max(worst, rc)
        if rc:
            print(f"\nStopping: '{name}' did not finish cleanly. Nothing after "
                  "it would be reading what it expects.", file=sys.stderr)
            break

    if worst == 0:
        print(f"\nDone. Results in {Path(a.results).resolve()}")
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
