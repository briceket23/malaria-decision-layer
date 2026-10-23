"""One entry point. Every command names its inputs; none of them has a default.

    mag ingest  --image-list XLSX --archive DIR   archive -> patients.csv
    mag detect  --images DIR --weights NAME=PATH  run the detectors
    mag doctor  --cache DIR --patients CSV        what is here, what would block
    mag rules                                     the registry, as a table
    mag run     --cache DIR --patients CSV        the experiment
    mag figures --cache DIR --patients CSV --out DIR
    mag report  --cache DIR --patients CSV --tex PATH
    mag all     --cache DIR --patients CSV --out DIR --tex PATH
    mag audit   --out DIR                         check before typesetting

The first two stages are the ones that touch the archive. `ingest` needs no GPU
and takes seconds; `detect` needs one and takes hours, and is resumable at smear
granularity so an interrupted pass costs only the smear it was inside.

`all` is the one to use. `run`, `figures` and `report` each build the study from
scratch, so running the three in sequence does the same work three times and --
worse -- can produce figures and text from two different builds if the code
changes in between. `all` computes the study once and writes everything from it,
which is why the figures it writes carry a provenance stamp that `report`
checks.

There is no `synth`, no `--demo`, and no default cache path. A command that
cannot find its input fails with a remedy rather than falling back to something
it can generate, because a pipeline that can generate its own input will
eventually be asked to explain a number that came from nowhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, build_id
from .errors import BadInput, MagError, MissingInput


def _banner() -> str:
    return (f"mag {__version__}  build {build_id()}  "
            f"from {Path(__file__).resolve().parent}")


# --------------------------------------------------------------------------- #

def cmd_doctor(a) -> int:
    from . import cache as cache_mod
    from . import gates, rules
    from .study import load_patients

    cache = Path(a.cache)
    print(f"cache      {cache}")
    names = cache_mod.available(cache) if cache.is_dir() else []
    print(f"checkpoints{'':1s}{', '.join(names) if names else '-- none found --'}")
    patients = load_patients(a.patients)
    print(f"patients   {len(patients)} rows, "
          f"{int(patients['n_images'].sum()):,} fields")
    print(f"rules      {len(rules.RULES)} fixed, {len(rules.FITTED)} fitted "
          f"({', '.join(rules.fitted_ids())})")
    ident = gates.identifiability(patients)
    print(f"identifiability  {'ok' if ident.passed else 'FAILS'}"
          f"{'' if ident.passed else ': ' + ident.message}")
    n_fields = int(patients["n_images"].sum())
    # against the same threshold `run` will use, not a hard-coded 25 -- doctor
    # said "will be skipped" while `run --min-fields 5` computed the curve
    eligible = int((patients["n_images"] >= a.min_fields).sum())
    print(f"field curve      {eligible} patients carry {a.min_fields}+ fields "
          f"({'computable' if eligible >= 20 else 'NOT computable, will be skipped'})")
    blocking = []
    for name in names:
        try:
            _, man = cache_mod.load(cache, name, verify=True)
        except MagError as e:
            blocking.append(f"{name}: {e.problem}")
            continue
        g = gates.yield_gate(man, n_fields)
        flag = "ok" if g.passed else "REFUSED"
        print(f"  {name:<10} {man.n_detections:>10,} detections  "
              f"{man.n_images:>6,} fields  floor {man.conf_floor}  [{flag}]")
        if not g.passed:
            blocking.append(g.message)
    if blocking:
        print("\nblocking:")
        for b in blocking:
            print(f"  ! {b}")
        return 2
    print("\nno blocking problems.")
    return 0


def cmd_ingest(a) -> int:
    """Archive -> patients.csv. No GPU, no detector, no network.

    This is the stage that decides who is in the cohort and what their
    reference diagnosis is, so it is deliberately the loudest one: every
    vocabulary it does not recognise is a refusal, and the census it prints is
    the manuscript's Table 1.
    """
    from . import ingest

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    il = ingest.read_image_list(a.image_list)
    print(f"image list  {len(il):,} rows, "
          f"{il['smear_name'].nunique():,} smears across "
          f"{il['set_name'].nunique()} partitions")
    if a.list_sets:
        for name, n in sorted(il.groupby("set_name")["smear_name"]
                              .nunique().items()):
            print(f"  {name:<12} {n:>5} smears")
        return 0

    if a.classes:
        # A mismatch here is reported, not fatal: the class order matters for
        # reading DETECTIONS, and this stage reads none.
        try:
            ingest.read_classes(a.classes)
            print("classes.txt matches the expected order")
        except MagError as e:
            print(f"  ! classes.txt: {e.problem}")

    patients = ingest.build_patients(il, a.set)
    dest = out / "patients.csv"
    patients.to_csv(dest, index=False, lineterminator="\n")
    print(f"patients    {len(patients)} rows, "
          f"{int(patients['n_images'].sum()):,} fields -> {dest}")
    print()
    print(ingest.cohort_table(patients).to_string())

    eligible = int((patients["n_images"] >= a.min_fields).sum())
    print(f"\n{eligible} patients carry {a.min_fields}+ fields")

    ok = True
    if not a.archive:
        print("\nlayout      not checked and the transform not measured: "
              "pass --archive to reconcile the manifest against the files on "
              "disk. Without it the field counts are the manifest's claim, "
              "not a fact about the archive")
    elif not a.skip_layout:
        try:
            chk = ingest.check_layout(a.archive, il, a.set)
            print(f"layout      {chk['images_on_disk']:,} images on disk against "
                  f"{chk['listed_images']:,} listed; "
                  f"{chk['label_files']:,} label files")
            if chk["n_listed_but_missing"]:
                ok = False
                print(f"  ! {chk['n_listed_but_missing']} listed images are "
                      "NOT on disk. The archive is incomplete; every "
                      "per-field denominator would be wrong. First few: "
                      + ", ".join(chk["listed_but_missing"][:5]))
            if chk["n_on_disk_but_unlisted"]:
                print(f"  . {chk['n_on_disk_but_unlisted']} images on disk are "
                      "not in the manifest; they are ignored, because the "
                      "manifest defines the cohort")
        except MagError as e:
            ok = False
            print(f"  ! layout could not be checked: {e.problem}")

    if a.zoom_set and a.archive:
        try:
            tr = ingest.transform_retention(a.archive, a.set, a.zoom_set)
            print(f"\ntransform   {a.set} -> {a.zoom_set}: "
                  f"{tr['parasite_kept'] * 100:.1f}% of parasite boxes kept, "
                  f"{tr['all_kept'] * 100:.1f}% of all boxes")
            print(f"            {tr['verdict']}")
            # written beside the patient table so the ANALYSIS can quote it
            # without needing the 6 GB archive on disk. A reviewer re-running
            # from a published cache does not have the archive, and a number
            # the manuscript quotes must come from somewhere they can check.
            tf = out / "transform.json"
            tf.write_text(json.dumps(tr, indent=1, sort_keys=True,
                                     default=str) + "\n", encoding="utf-8")
            print(f"            written to {tf}")
        except MagError as e:
            print(f"\ntransform   not measured: {e.problem}")

    if not ok:
        print("\ningest is NOT complete. Nothing above is safe to analyse.")
        return 2
    return 0


def cmd_detect(a) -> int:
    """Run the detectors and write the cache the analysis reads.

    This is the only stage that needs a GPU, and the only one that touches the
    images. It is resumable at smear granularity: a run interrupted after 100
    of 170 smears picks up at 101, and a shard is reused only when its
    fingerprint -- image list, confidence floor, image size, class map and the
    checkpoint's own SHA-256 -- matches exactly.
    """
    from . import detect

    weights = {}
    for spec in a.weights:
        if "=" not in spec:
            raise BadInput(
                f"--weights takes NAME=PATH, got {spec!r}",
                remedy=("The name becomes the checkpoint's name in every "
                        "table, so choose the one the manuscript uses: "
                        "`--weights rtdetr=/path/rtdetr.pt yolov8x=/path/y8.pt`"))
        name, _, path = spec.partition("=")
        p = Path(path).expanduser()
        if not p.is_file():
            raise MissingInput(f"no checkpoint file at {p}",
                               remedy="Check the path.", name=name)
        weights[name] = p
    if not weights:
        raise BadInput(
            "no checkpoints were given",
            remedy=("Pass at least one as --weights NAME=PATH. `mag doctor "
                    "--archive <dir>` lists the ones it can find in an "
                    "unpacked archive."))

    images = Path(a.images)
    out = Path(a.out)
    for name, w in sorted(weights.items()):
        print(f"\n=== {name} ===")
        meta = detect.read_checkpoint_metadata(w, a.yolov5_repo)
        cmap = detect.class_map_from(meta, strict=not a.assume_classes_txt)
        print(f"  {len(cmap)} classes, sha {meta['sha256'][:12]}, "
              f"format {meta['format']}")
        detect.run(w, images, out, name, cmap, conf_floor=a.conf,
                   imgsz=a.imgsz, batch=a.batch, device=a.device,
                   resume=not a.no_resume, progress=print, half=a.half,
                   yolov5_repo=a.yolov5_repo, seed=a.seed)

    if a.patients and Path(a.patients).is_file():
        # copied explicitly as UTF-8: Windows' cp1252 text default corrupted a
        # centre name on the round trip and the patient table then failed its
        # own vocabulary check
        (out / "patients.csv").write_text(
            Path(a.patients).read_text(encoding="utf-8"), encoding="utf-8")
        print("\ncopied the patient table beside the cache")

    bad = detect.suspect_checkpoints(out)
    if bad:
        print("\nmarked suspect and refused by the analysis stages:")
        for name, why in sorted(bad.items()):
            print(f"  ! {name}: {why}")
        return 2
    return 0


def cmd_rules(a) -> int:
    """The registry as a table, so the paper and the code can be read together."""
    from . import rules

    rows = [rules.describe(r) for r in rules.all_rule_ids()]
    if a.json:
        print(json.dumps(rows, indent=1, default=str))
        return 0
    import textwrap

    print(f"\n{'rule':<11}{'family':<13}{'defaults'}")
    print("-" * 78)
    for r in rows:
        defaults = ", ".join(f"{k}={v}" for k, v in r["defaults"].items()
                             if v is not None)
        print(f"{r['rule']:<11}{r['family']:<13}{defaults}")
        for line in textwrap.wrap(r["note"], 74):
            print(f"{'':<11}{line}")
        if r["fitted"]:
            print(f"{'':<11}FITTED ON {r['fitted_on']}. No exact test is valid "
                  "against this arm.")
        print()
    print("\nB0 trivial (the prior alone), B1 oracle (the ceiling on this "
          "cache), B2 the published rule under the published settings.")
    print("An exact test is valid only against a rule whose parameters were "
          "fixed before these patients were seen.")
    return 0


def _build(a):
    from . import study

    return study.run(a.cache, a.patients, checkpoints=a.checkpoints,
                     condition=a.condition, seed=a.seed,
                     force_past_reconstruction=a.force_reconstruction,
                     depth=a.depth, folds=a.folds, min_fields=a.min_fields,
                     reps=a.reps, stress_reps=getattr(a, "stress_reps", 30))


def _summary(s) -> dict:
    """Everything a run produced, as JSON. Frames go to their own CSVs."""
    return {
        "provenance": s.provenance.line(),
        "meta": s.meta,
        "identifiability": {"passed": s.identifiability.passed,
                            "message": s.identifiability.message,
                            **s.identifiability.detail},
        "excluded": s.excluded,
        "estimands": s.estimands,
        "cohort_check": s.cohort_check,
        "reconstruction": s.reconstruction.detail | {"passed": s.reconstruction.passed},
        "transform": s.transform,
        "design": {k: v for k, v in s.design.items() if k != "gate"},
        "confirmatory": s.confirmatory,
        "attribution": _attribution_json(s.attribution),
        "apportionment": s.apportionment,
        "bounds": s.bounds,
        "fitted": {f"{ck}:{rid}": v for (ck, rid), v in s.fitted.items()},
        "calibration": s.calibration,
        "selective": {k: (v if not isinstance(v, dict) else
                          {kk: vv for kk, vv in v.items()
                           if kk not in ("coverage", "risk")})
                      for k, v in s.selective.items()},
        "error_summary": s.error_summary,
        "domain": s.domain,
        "field_curve": s.field_curve,
        "power": s.power,
        "nulls": s.nulls,
        "sweep": s.sweep_summary,
        "stress": _stress_json(s.stress),
        "predictive": (_dc(s.predictive) if s.predictive else {}),
        "speciation_calibration": (_dc(s.speciation_calibration)
                                   if s.speciation_calibration else {}),
        "loco": s.loco,
    }


def _dc(obj):
    from dataclasses import asdict, is_dataclass

    return asdict(obj) if is_dataclass(obj) else dict(obj)


def _attribution_json(a):
    """The model, flattened. `None` when it could not be fitted."""
    if a is None:
        return {}
    return {"n_patients": a.n_patients, "n_rules": a.n_rules,
            "n_checkpoints": a.n_checkpoints,
            "rule_effect": a.rule_effect,
            "checkpoint_effect": a.checkpoint_effect,
            # `contrast` is the primary, probability-scale estimand;
            # `contrast_logodds` carries either the log-odds version or the
            # reason it is withheld. Both travel, so a reader of the JSON is
            # told why a scale is missing rather than finding it absent.
            "contrast": a.contrast,
            "contrast_logodds": a.contrast_logodds,
            "separation": a.separation,
            "numerics": a.numerics,
            "interaction": a.interaction,
            "rule_means": a.rule_means,
            "checkpoint_means": a.checkpoint_means,
            "agreement": a.agreement, "converged": a.converged,
            "note": a.note, "sentence": a.sentence}


def _stress_json(st):
    if st is None:
        return {}
    return {"rates": list(st.rates), "by_rule": st.by_rule,
            "ordering": st.ordering,
            "donor_mismatch": st.donor_mismatch,
            "n_donor_fields": st.n_donor_fields, "reps": st.reps,
            "note": st.note, "sentence": st.sentence}


def _write_outputs(s, out: Path) -> list:
    """The frames, each to its own file. Returns what was written."""
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    written = []
    s.factorial.to_csv(out / "factorial.csv", index=False)
    written.append("factorial.csv")
    s.provenance.write(out / "provenance.json")
    written.append("provenance.json")

    preds = pd.DataFrame({f"{ck}:{rid}": p for (ck, rid), p in
                          sorted(s.predictions.items(), key=str)})
    preds.index.name = "patient_id"
    preds.to_csv(out / "predictions.csv")
    written.append("predictions.csv")

    if s.errors is not None and len(s.errors):
        s.errors.to_csv(out / "errors.csv", index=False)
        written.append("errors.csv")
    if s.box_census is not None and len(s.box_census):
        s.box_census.to_csv(out / "box_census.csv", index=False)
        written.append("box_census.csv")
    if s.calibration:
        rows = [{"map": k, **{kk: vv for kk, vv in v.items()
                              if not isinstance(vv, (tuple, list, dict))}}
                for k, v in s.calibration.items() if isinstance(v, dict)]
        pd.DataFrame(rows).to_csv(out / "calibration.csv", index=False)
        written.append("calibration.csv")
    if s.selective:
        rows = []
        for rid, rc in s.selective.items():
            if not isinstance(rc, dict) or "operating" not in rc:
                continue
            for target, op in rc["operating"].items():
                rows.append({"rule": rid, "coverage_target": target,
                             "aurc": rc["aurc"], **op})
        if rows:
            pd.DataFrame(rows).to_csv(out / "selective.csv", index=False)
            written.append("selective.csv")
    if s.sweep is not None and len(getattr(s.sweep, "table", [])):
        s.sweep.table.to_csv(out / "sweep.csv", index=False)
        written.append("sweep.csv")

    if s.claims is not None and len(s.claims):
        s.claims.to_csv(out / "claims.csv", index=False)
        written.append("claims.csv")
    if s.stress is not None and s.stress.by_rule:
        rows = [{"rule": rid, "positive_field_rate": float(pi), **v}
                for rid, d in s.stress.by_rule.items() for pi, v in d.items()]
        pd.DataFrame(rows).to_csv(out / "stress.csv", index=False)
        written.append("stress.csv")
    if s.predictive is not None and s.predictive.at:
        rows = [{"setting": k, **{kk: vv for kk, vv in v.items()
                                  if not isinstance(vv, tuple)}}
                for k, v in s.predictive.at.items()]
        pd.DataFrame(rows).to_csv(out / "predictive.csv", index=False)
        written.append("predictive.csv")
    from . import rules as _rules

    _rules.comparability_table().to_csv(out / "rule_comparability.csv", index=False)
    _rules.parameter_table().to_csv(out / "rule_parameters.csv", index=False)
    written += ["rule_comparability.csv", "rule_parameters.csv"]

    (out / "results.json").write_text(
        json.dumps(_summary(s), indent=1, default=str), encoding="utf-8")
    written.append("results.json")
    return written


def _report_to_console(s, condition: str) -> None:
    c = s.confirmatory
    print(f"\nanalysed   {', '.join(sorted(set(s.factorial['checkpoint'])))}"
          f"   condition {condition}   depth {s.depth}")
    if s.excluded:
        for k, v in s.excluded.items():
            print(f"  ! excluded {k}: {v}")
    print(f"reconstruct {s.reconstruction.detail['n_correct']}/"
          f"{s.reconstruction.detail['n_total']} against a published "
          f"{s.reconstruction.detail['published']}  "
          f"({'reconstructible' if s.reconstruction.passed else 'NOT reconstructible'})")
    sel = s.design.get("selection_detail") or {}
    print(f"selection  {s.design['selected']} by {s.design['selection_rule']}"
          + (f"; max-discordance would choose {sel.get('by_max_discordance')}"
             if sel.get("by_max_discordance") else ""))
    print(f"confirmatory  {c['checkpoint']}:{c['rule']} vs {c['comparator']}")
    print(f"  specificity {c['spec_base']:.3f} -> {c['spec_new']:.3f}"
          f"   exact McNemar b={c['mcnemar']['b']} c={c['mcnemar']['c']} "
          f"p={c['mcnemar']['p']:.3g} ({c['mcnemar']['alternative']})")
    ni = c["noninferiority"]
    print(f"  sensitivity {ni['diff'] * 100:+.1f} pp, one-sided lower "
          f"{ni['one_sided_lower'] * 100:.1f}, margin {abs(ni['margin']) * 100:.1f}"
          f"  -> {'non-inferior' if ni['non_inferior'] else 'NOT shown non-inferior'}")
    print(f"  intersection-union: {'REJECT' if c['iu']['reject'] else 'no'}")
    spec = (s.power or {}).get("specificity") or {}
    if spec.get("sentence"):
        print(f"  detectable: {spec['sentence']}")
    if s.calibration.get("raw"):
        r = s.calibration["raw"]
        best = min((k for k in ("isotonic", "platt", "temperature")
                    if isinstance(s.calibration.get(k), dict)),
                   key=lambda k: s.calibration[k]["ece"], default=None)
        print(f"calibration  raw ECE {r['ece']:.4f} (null {r['null_reference']:.4f})"
              f", slope {r['slope']:.3f}"
              + (f"; best cross-fitted map {best} at ECE "
                 f"{s.calibration[best]['ece']:.4f}" if best else ""))
    if s.error_summary.get("n_errors"):
        e = s.error_summary
        print(f"errors      {e['n_errors']}, {100 * e['top_category_share']:.0f}% "
              f"{e['top_category']}, median fragility "
              f"{e['median_fragility']:.1f}")
    if s.attribution is not None and s.attribution.sentence:
        print(f"attribution {s.attribution.sentence[:150]}...")
    if s.stress is not None and s.stress.sentence:
        print(f"stress      {s.stress.sentence[:150]}...")
    if s.sweep_summary.get("sentence"):
        print(f"sweep       {s.sweep_summary['sentence']}")
    if s.meta.get("skipped_at_core_depth"):
        print(f"\nnot computed at --depth core: "
              f"{', '.join(s.meta['skipped_at_core_depth'])}")


# --------------------------------------------------------------------------- #

def cmd_run(a) -> int:
    s = _build(a)
    written = _write_outputs(s, Path(a.out))
    _report_to_console(s, a.condition)
    print(f"\nwrote {len(written)} files to {a.out}: {', '.join(written)}")
    return 0


def cmd_figures(a) -> int:
    from . import figures

    s = _build(a)
    stamp = figures.write_all(s, a.out)
    print(f"\nwrote {len(stamp['figures'])} figures to {a.out}")
    for name, files in sorted(stamp["figures"].items()):
        print(f"  {name:<20} {', '.join(files)}")
    for why in stamp["skipped"]:
        print(f"  ! skipped {why}")
    return 0


def cmd_report(a) -> int:
    from . import figures, report

    s = _build(a)
    if a.figures:
        figures.check(a.figures, s)
    keys = report.write(s, a.tex)
    print(f"wrote {a.tex} with {len(keys)} macros")
    print(f"provenance: {s.provenance.line()}")
    return 0


def cmd_all(a) -> int:
    """Build the study once; write results, figures and macros from it."""
    from . import figures, report

    s = _build(a)
    out = Path(a.out)
    written = _write_outputs(s, out)
    stamp = figures.write_all(s, out / "figures")
    # the stamp must match the study it was just written from. It cannot fail
    # here today; it is checked so that a future refactor which separates the
    # two fails loudly rather than shipping a mismatched pair.
    figures.check(out / "figures", s)
    keys = report.write(s, a.tex or (out / "results.tex"))
    _report_to_console(s, a.condition)
    print(f"\nwrote {len(written)} result files, {len(stamp['figures'])} "
          f"figures and {len(keys)} macros")
    for why in stamp["skipped"]:
        print(f"  ! figure skipped: {why}")
    print(f"provenance: {s.provenance.line()}")
    return 0


def cmd_audit(a) -> int:
    """Check a finished results directory before anything is typeset.

    Runs over artefacts rather than over a study object, deliberately: the
    question it answers is whether the files a manuscript will read are
    mutually consistent, and that is a property of the files. It can therefore
    be run on a directory produced weeks earlier, or on a colleague's, without
    re-running the analysis.

    Returns 0 when every check passes, 5 when any fails. The declarations file
    is author-supplied. This command checks that its fields are present and
    non-empty; it does not and will not supply them.
    """
    import json as _json

    from . import audit

    decl = None
    if a.declarations:
        p = Path(a.declarations)
        if not p.is_file():
            from .errors import BadInput
            raise BadInput(
                f"no declarations file at {p}",
                remedy=("Write one: a JSON object with the keys "
                        f"{', '.join(audit.REQUIRED_DECLARATIONS)}. Every "
                        "value is yours to supply -- ethics, funding, CRediT, "
                        "AI use, repository and registration identifiers are "
                        "facts about your study, and this tool has no way to "
                        "know them and will not guess."))
        decl = _json.loads(p.read_text(encoding="utf-8"))

    factorial = None
    fpath = Path(a.out) / "factorial.csv"
    if fpath.is_file():
        import pandas as pd
        factorial = pd.read_csv(fpath)

    rep = audit.run(a.out, manuscript=a.manuscript, declarations=decl,
                    factorial=factorial)
    print(rep.summary())
    for f in rep.findings:
        mark = "FAIL" if f.severity == "fail" else "warn"
        print(f"  [{mark}] {f.check}: {f.message}")
    if a.json:
        payload = {"passed": rep.passed, "n_checks": rep.n_checks,
                   "findings": [{"check": f.check, "severity": f.severity,
                                 "message": f.message, "detail": f.detail}
                                for f in rep.findings]}
        target = Path(a.out) / "audit.json"
        target.write_text(_json.dumps(payload, indent=2, sort_keys=True,
                                      default=str) + "\n", encoding="utf-8")
        print(f"wrote {target}")
    if not rep.passed:
        print("\nNothing here is fixed by re-running. Each line above names a "
              "file and a value; change the file.")
    return 0 if rep.passed else 5


# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser("mag", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=_banner())
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--cache", required=True,
                       help="a detection cache written by the detect stage")
        p.add_argument("--patients", required=True,
                       help="the patient table built from the released manifest")
        p.add_argument("--checkpoints", nargs="*", default=None)
        p.add_argument("--condition", default="matched",
                       choices=["raw", "matched", "source"],
                       help="post-processing; checkpoint comparisons are only "
                            "meaningful in 'matched'")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--depth", default="core", choices=["core", "full"],
                       help="core is every number a result depends on; full "
                            "adds the field curve, the null simulations and "
                            "the post hoc sweep, and cannot change a core "
                            "number")
        p.add_argument("--folds", type=int, default=10,
                       help="cross-fitting folds for the fitted rules; a value "
                            "at or above the patient count is leave-one-out, "
                            "which is documented as inadequate here")
        p.add_argument("--min-fields", dest="min_fields", type=int, default=25,
                       help="patients with fewer fields are excluded from the "
                            "field curve so its denominator does not move")
        p.add_argument("--reps", type=int, default=50,
                       help="resampling replicates for the field curve")
        p.add_argument("--force-reconstruction", "--force-reproduction",
                       dest="force_reconstruction", action="store_true",
                       help="continue past a failed reconstruction gate "
                            "and record that it was forced. The gate "
                            "refusing is a FINDING; read it first")
        p.add_argument("--stress-reps", dest="stress_reps", type=int,
                       default=30,
                       help="replicates for the field-sampling stress test")
        return p

    d = sub.add_parser("doctor", help="what is here and what would block")
    d.add_argument("--cache", required=True)
    d.add_argument("--patients", required=True)
    d.add_argument("--min-fields", dest="min_fields", type=int, default=25,
                   help="report the field curve against this threshold, the "
                        "same one `mag run` takes")
    d.set_defaults(fn=cmd_doctor)

    g = sub.add_parser(
        "ingest", help="archive -> patients.csv (no GPU, no detector)")
    g.add_argument("--image-list", dest="image_list", required=True,
                   help="the archive's image_list.xlsx, or a CSV export of it")
    g.add_argument("--archive", default=None,
                   help="the directory holding images/ and labels/; needed "
                        "only for the layout check and the transform "
                        "measurement")
    g.add_argument("--classes", default=None,
                   help="classes.txt, checked against the expected order")
    g.add_argument("--set", default="test",
                   help="the partition the cohort is drawn from")
    g.add_argument("--zoom-set", dest="zoom_set", default="test-zoom",
                   help="the second label tree, used to MEASURE the source "
                        "study's zoom transformation rather than infer it. "
                        "Pass an empty string to skip")
    g.add_argument("--out", default="data",
                   help="where patients.csv is written")
    g.add_argument("--min-fields", dest="min_fields", type=int, default=25,
                   help="report how many patients clear this field count")
    g.add_argument("--skip-layout", dest="skip_layout", action="store_true",
                   help="do not reconcile the manifest against the files on "
                        "disk. Only for a manifest-only check")
    g.add_argument("--list-sets", dest="list_sets", action="store_true",
                   help="print the partitions the manifest declares, and stop")
    g.set_defaults(fn=cmd_ingest)

    n = sub.add_parser("detect", help="run the detectors, write the cache")
    n.add_argument("--images", required=True,
                   help="the image directory for one partition, e.g. "
                        "<archive>/images/test")
    n.add_argument("--weights", nargs="+", required=True, metavar="NAME=PATH",
                   help="one or more checkpoints. NAME becomes the "
                        "checkpoint's name in every table")
    n.add_argument("--out", required=True,
                   help="the cache directory the analysis will read")
    n.add_argument("--patients", default=None,
                   help="patients.csv from `mag ingest`; copied beside the "
                        "cache so the two travel together")
    n.add_argument("--conf", type=float, default=0.001,
                   help="the confidence floor written into the cache. Low on "
                        "purpose: every rule's threshold is applied later, so "
                        "a cache cut at 0.25 could not evaluate a rule below "
                        "it")
    n.add_argument("--imgsz", type=int, default=None,
                   help="inference size; default is the checkpoint's own "
                        "training size")
    n.add_argument("--batch", type=int, default=None,
                   help="default is chosen from the card's memory")
    n.add_argument("--device", default="0", help="'0' for the first GPU, or 'cpu'")
    n.add_argument("--half", dest="half", action="store_true", default=None)
    n.add_argument("--no-half", dest="half", action="store_false")
    n.add_argument("--yolov5-repo", dest="yolov5_repo", default=None,
                   help="a clone of ultralytics/yolov5, for a checkpoint in "
                        "repository format")
    n.add_argument("--assume-classes-txt", dest="assume_classes_txt",
                   action="store_true",
                   help="when a checkpoint carries no class names, fall back "
                        "on classes.txt order. This is an ASSUMPTION about "
                        "which index means which species; declare it")
    n.add_argument("--no-resume", dest="no_resume", action="store_true",
                   help="delete finished shards and start the pass again")
    n.add_argument("--seed", type=int, default=0)
    n.set_defaults(fn=cmd_detect)

    q = sub.add_parser("rules", help="the rule registry, as a table")
    q.add_argument("--json", action="store_true")
    q.set_defaults(fn=cmd_rules)

    r = common(sub.add_parser("run", help="run the experiment"))
    r.add_argument("--out", default="results")
    r.set_defaults(fn=cmd_run)

    f = common(sub.add_parser("figures", help="draw the figures"))
    f.add_argument("--out", default="figures")
    f.set_defaults(fn=cmd_figures)

    t = common(sub.add_parser("report", help="write results.tex"))
    t.add_argument("--tex", required=True)
    t.add_argument("--figures", default=None,
                   help="refuse unless the figures in this directory came "
                        "from this same run")
    t.set_defaults(fn=cmd_report)

    z = common(sub.add_parser("all", help="results, figures and macros in one"))
    z.add_argument("--out", default="results")
    z.add_argument("--tex", default=None)
    z.set_defaults(fn=cmd_all)

    au = sub.add_parser(
        "audit", help="check a finished results directory before typesetting")
    au.add_argument("--out", required=True,
                    help="a directory written by `mag run` or `mag all`")
    au.add_argument("--manuscript", default=None,
                    help="the .tex source, checked for typed verdicts and for "
                         "macros the analysis never defined")
    au.add_argument("--declarations", default=None,
                    help="a JSON file of author-supplied declarations. "
                         "Checked for presence only; never filled in")
    au.add_argument("--json", action="store_true",
                    help="also write audit.json beside the results")
    au.set_defaults(fn=cmd_audit)
    return ap


def _validate(a) -> None:
    """Reject arguments that would quietly produce a different analysis."""
    from .errors import BadInput

    if getattr(a, "seed", 0) is not None and getattr(a, "seed", 0) < 0:
        raise BadInput(f"a seed of {a.seed}",
                       remedy="Seeds are non-negative integers.")
    folds = getattr(a, "folds", None)
    if folds is not None and folds < 2:
        # k = 0 divided by zero; k = 1 trained on every patient it then scored,
        # and both wrote results while `calibration.note` still claimed each
        # patient's probability came from a map fitted without that patient
        raise BadInput(f"{folds} cross-fitting folds",
                       remedy=("Use at least 2. Below that nothing is "
                               "cross-fitted and every fitted arm is the "
                               "baseline under another name."))
    reps = getattr(a, "reps", None)
    if reps is not None and reps < 1:
        raise BadInput(f"{reps} resampling replicates",
                       remedy="Use at least 1.")


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    print(_banner(), flush=True)
    try:
        _validate(a)
        return a.fn(a)
    except MagError as e:
        print(f"\n{type(e).__name__}: {e}", file=sys.stderr)
        return 3
    except Exception as e:                              # noqa: BLE001
        # a third-party exception -- a truncated parquet, a torn manifest --
        # reached the operator as a traceback with no remedy. It is still a
        # failure, but it is reported as one.
        print(f"\n{type(e).__name__}: {e}\n\n"
              "This is not a failure this package anticipated. If it names a "
              "file, that file is the place to look; otherwise re-run with the "
              "same arguments and report the trace.", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
