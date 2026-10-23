#!/usr/bin/env python3
"""test_experiment.py --- the checks that keep the suite honest.

    python test_experiment.py                 # fast: no archive, no GPU
    python test_experiment.py --with-archive   # also builds fixtures and runs
                                               # the real producers end to end

Three kinds of test, and the first two are why this file exists.

**Drift.** Both bugs this suite has had were a consumer looking for a key a
producer never wrote --- `predictions.csv` read as long when mag writes it
wide, and `centre_distance` read as nested when it is flat. Neither was caught
by a test, because the fixtures were written from the same memory as the
consumer. So: every key a consumer reads is declared in `exp_schema`, every
fixture here is BUILT through `exp_schema` rather than typed, and
`--with-archive` runs the real producers and validates their real output. A
misspelling is now an AttributeError; a drift is a failing test.

**Both directions.** A check that only ever fires is a constant with a function
around it, and it keeps firing after the problem is fixed, which teaches
everyone to ignore it. Every finding is fed an archive with its pathology (must
fire) and one without (must clear).

**Mechanics.** The cap override, the arm diff, the wording guard, the macros.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import exp_schema as S                                        # noqa: E402
import experiment_findings as EF                              # noqa: E402
import run_experiment as RX                                   # noqa: E402
from exp_schema import EDA, VERIFY, MISSING                   # noqa: E402

_results = []


def check(name, cond, detail=""):
    _results.append((name, bool(cond), detail))
    print(f"  [{'ok' if cond else 'FAIL'}] {name}"
          + (f"  --- {detail}" if detail and not cond else ""))


# --------------------------------------------------------------------------- #
# Fixtures, BUILT THROUGH THE SCHEMA so they cannot drift from it.
#
# Values are the ones measured on the real archive on 2026-08-27. Where a value
# is a judgement rather than a measurement it is marked.
# --------------------------------------------------------------------------- #

def build_eda(healthy: bool = False) -> dict:
    d: dict = {}
    EDA.ARCHIVE.put(d, "/fixture/Plasmodium-v1")
    EDA.MAG_VERSION.put(d, "3.0.0 build test")
    EDA.GENERATED.put(d, "2026-08-27T00:00:00+00:00")
    EDA.SETS.put(d, ["train", "validation", "test"])
    EDA.MANIFEST_ROWS.put(d, 29228)
    EDA.MANIFEST_COLUMNS.put(d, ["set_name", "origin", "species",
                                 "smear_name", "image_name", "image_path"])

    for st, (sm, im) in (("train", (403, 20830)), ("validation", (72, 3890)),
                         ("test", (170, 4508))):
        EDA.PART_SMEARS.put(d, sm, name=st)
        EDA.PART_IMAGES.put(d, im, name=st)
    EDA.PART_PROBLEMS.put(d, [])

    EDA.COHORT_TABLE.put(d, {
        "Montpellier": {"uninfected": 54, "P_falciparum": 11, "Total": 65},
        "Toulouse": {"P_falciparum": 5, "P_ovale": 5, "P_malariae": 5,
                     "P_vivax": 5, "Babesia_divergens": 1, "T_brucei": 1},
        "Lille": {"P_falciparum": 9, "P_ovale": 1, "P_malariae": 1,
                  "P_vivax": 2, "Babesia_divergens": 1, "T_brucei": 1},
        "Nantes": {"P_falciparum": 1},
        "Rouen": {"P_falciparum": 12, "P_ovale": 12, "P_malariae": 11,
                  "P_vivax": 10},
        "Saint-Louis": {"P_falciparum": 5, "P_ovale": 5, "P_malariae": 5,
                        "P_vivax": 5, "Babesia_divergens": 1, "T_brucei": 1},
        "Total": {"uninfected": 54, "P_falciparum": 43, "P_ovale": 23,
                  "P_malariae": 22, "P_vivax": 22, "Babesia_divergens": 3,
                  "T_brucei": 3, "Total": 170}}
        if not healthy else
        {"Montpellier": {"uninfected": 20, "P_falciparum": 11},
         "Rouen": {"uninfected": 20, "P_ovale": 12, "Babesia_divergens": 8},
         "Lille": {"uninfected": 14, "T_brucei": 9, "P_vivax": 10},
         "Total": {"uninfected": 54, "P_falciparum": 11, "P_ovale": 12,
                   "P_vivax": 10, "Babesia_divergens": 8, "T_brucei": 9,
                   "Total": 104}})

    EDA.IDENT_N_PATIENTS.put(d, 170)
    EDA.IDENT_N_INFECTED.put(d, 116)
    EDA.IDENT_N_UNINFECTED.put(d, 54)
    EDA.IDENT_PREVALENCE.put(d, 0.682)
    EDA.IDENT_CENTRES.put(d, ["Montpellier"] if not healthy
                          else ["Montpellier", "Rouen", "Lille"])
    EDA.IDENT_SEPARABLE.put(d, bool(healthy))

    EDA.LAB_SET.put(d, "test")
    EDA.LAB_BOXES.put(d, 359259)
    EDA.LAB_PER_FIELD.put(d, 79.8)
    EDA.LAB_FIELDS.put(d, 4502)
    EDA.LAB_PER_CLASS.put(d, {"RBC": 343287, "platelet": 10384,
                              "P_falciparum": 2965, "P_vivax": 706,
                              "P_ovale": 696, "P_malariae": 514, "WBC": 418,
                              "Babesia_divergens": 221, "T_brucei": 68}
                          if not healthy else
                          {"RBC": 5000, "P_falciparum": 900, "T_brucei": 400})

    EDA.GEO_PAR_RADIUS.put(d, 0.2481 if not healthy else 0.44)
    EDA.GEO_OTH_RADIUS.put(d, 0.4273 if not healthy else 0.42)
    EDA.GEO_PAR_MARGIN.put(d, 0.106)
    EDA.GEO_OTH_MARGIN.put(d, 0.303)
    EDA.GEO_NEARER.put(d, not healthy)
    EDA.GEO_PRED_PAR.put(d, 0.894)
    EDA.GEO_PRED_OTH.put(d, 0.697)

    EDA.LOAD_ENRICHMENT.put(d, 0.967 if not healthy else 0.11)
    EDA.LOAD_MEDIAN.put(d, 26.0)
    EDA.LOAD_MIN.put(d, 2.0)
    EDA.LOAD_MAX.put(d, 495.0)
    EDA.LOAD_COVERAGE.put(d, 116)
    EDA.LOAD_TOTAL.put(d, 116)

    # measured retention: pathological gaps have OPPOSITE signs; the healthy
    # pair sits within the clipping slack and on the same side
    EDA.TF_PARASITE_KEPT.put(d, 0.8718 if not healthy else 0.904)
    EDA.TF_OTHER_KEPT.put(d, 0.7646 if not healthy else 0.707)
    EDA.TF_ALL_KEPT.put(d, 0.7661)
    EDA.TF_BY_CLASS.put(d, {"Babesia_divergens": {"base": 221, "zoom": 250,
                                                  "kept": 1.1312}})
    EDA.TF_RISING.put(d, {"Babesia_divergens": {"base": 221, "zoom": 250,
                                                "kept": 1.1312}}
                      if not healthy else {})

    EDA.IMG_SAMPLED.put(d, 400)
    EDA.IMG_SIZES.put(d, {"3840x2160": 231, "2560x1920": 152,
                          "1920x1440": 17})
    EDA.IMG_RATIOS.put(d, {"1.778": 231, "1.333": 169} if not healthy
                       else {"1.778": 400})
    EDA.FIGURES.put(d, ["class_balance.png"])
    return d


def build_verify(healthy: bool = False) -> dict:
    d: dict = {}
    VERIFY.ARCHIVE.put(d, "/fixture/Plasmodium-v1")
    VERIFY.MAG_VERSION.put(d, "3.0.0 build test")
    VERIFY.GENERATED.put(d, "2026-08-27T00:00:00+00:00")
    VERIFY.DEVICE.put(d, "0")
    VERIFY.VERDICT.put(d, "fixture")
    VERIFY.MAG_MAX_DET.put(d, 3000)

    spec = {
        "rtdetr": dict(loader="RTDETR", det=6000, mean=300.0,
                       lo=300 if not healthy else 244,
                       hi=300 if not healthy else 289,
                       med=0.0939 if not healthy else 0.91, cap=0),
        "yolov5x": dict(loader="yolov5-repo", det=54487, mean=2724.3,
                        lo=1978, hi=3000 if not healthy else 2400,
                        med=0.9528, cap=5 if not healthy else 0),
        "yolov8x": dict(loader="YOLO", det=22834, mean=1141.7, lo=819,
                        hi=1348, med=0.9559, cap=0),
    }
    for nm, v in spec.items():
        VERIFY.CK_PATH.put(d, f"/fixture/{nm}.pt", name=nm)
        VERIFY.CK_SHA.put(d, "0" * 64, name=nm)
        VERIFY.CK_BYTES.put(d, 135_400_000, name=nm)
        VERIFY.CK_NCLASSES.put(d, 9, name=nm)
        VERIFY.CK_PROBLEMS.put(d, [], name=nm)
        VERIFY.SMOKE_N_IMAGES.put(d, 20, name=nm)
        VERIFY.SMOKE_LOADER.put(d, v["loader"], name=nm)
        VERIFY.SMOKE_DETECTIONS.put(d, v["det"], name=nm)
        VERIFY.SMOKE_PER_FIELD_MEAN.put(d, v["mean"], name=nm)
        VERIFY.SMOKE_PER_FIELD_MIN.put(d, v["lo"], name=nm)
        VERIFY.SMOKE_PER_FIELD_MAX.put(d, v["hi"], name=nm)
        VERIFY.SMOKE_FIELDS_WITH_ANY.put(d, 20, name=nm)
        VERIFY.SMOKE_SCORE_MEDIAN.put(d, v["med"], name=nm)
        VERIFY.SMOKE_SCORE_MAX.put(d, 0.98, name=nm)
        VERIFY.SMOKE_SECONDS_PER_FIELD.put(d, 0.1, name=nm)
        VERIFY.SMOKE_PROBLEMS.put(d, [], name=nm)
        VERIFY.SMOKE_NOTES.put(d, [], name=nm)
        VERIFY.OP_CONF.put(d, 0.001, name=nm)
        VERIFY.OP_IOU.put(d, 1.0, name=nm)
        VERIFY.OP_AGNOSTIC.put(d, False, name=nm)
        VERIFY.OP_MAX_DET.put(d, 3000, name=nm)
        VERIFY.SMOKE_FIELDS_AT_CAP.put(d, v["cap"], name=nm)
        VERIFY.SMOKE_CAP_SHARE.put(d, round(v["cap"] / 20, 3), name=nm)

    if not healthy:
        nm = "yolov5x"
        # material=True below, so verify_models records the saturation as a
        # PROBLEM (an immaterial one becomes a note). The fixture has to
        # match, or `agreement` has nothing to exclude and the exclusion path
        # goes untested -- which is exactly how a crash in it reached a real
        # archive.
        VERIFY.SMOKE_PROBLEMS.put(
            d, [f"5 of 20 fields returned exactly 3000 boxes (max_det)"],
            name=nm)
        VERIFY.SMOKE_CENSORING.put(d, 0.002, name=nm)
        VERIFY.PROBE_MAX_DET.put(d, 30000, name=nm)
        VERIFY.PROBE_N_FIELDS.put(d, 5, name=nm)
        VERIFY.PROBE_TRUE_MEAN.put(d, 3113.0, name=nm)
        VERIFY.PROBE_TRUE_MAX.put(d, 3290, name=nm)
        VERIFY.PROBE_DISCARDED.put(d, 565, name=nm)
        VERIFY.PROBE_DISCARDED_SHARE.put(d, 0.036, name=nm)
        VERIFY.PROBE_BY_CLASS.put(d, {"RBC": 555, "platelet": 7,
                                      "P_vivax": 2, "WBC": 1}, name=nm)
        VERIFY.PROBE_KEPT_PAR.put(d, 116, name=nm)
        VERIFY.PROBE_KEPT_RBC.put(d, 14970, name=nm)
        VERIFY.PROBE_LOST_PAR.put(d, 2, name=nm)
        VERIFY.PROBE_LOST_RBC.put(d, 555, name=nm)
        VERIFY.PROBE_DENSITY_CAPPED.put(d, 0.00775, name=nm)
        VERIFY.PROBE_DENSITY_TRUE.put(d, 0.0076, name=nm)
        VERIFY.PROBE_DENSITY_BIAS.put(d, 0.0204, name=nm)
        VERIFY.PROBE_MATERIAL.put(d, True, name=nm)
        VERIFY.PROBE_SUGGESTED.put(d, 6580, name=nm)
        VERIFY.PROBE_STILL_CAPPED.put(d, 0, name=nm)
    return d


EDA_BAD, VERIFY_BAD = build_eda(False), build_verify(False)
EDA_OK, VERIFY_OK = build_eda(True), build_verify(True)


def repaired(key):
    """The pathological archive with ONLY `key`'s cause put right."""
    e, v = copy.deepcopy(EDA_BAD), copy.deepcopy(VERIFY_BAD)
    good_e, good_v = EDA_OK, VERIFY_OK
    swap = {
        "cap_saturation": lambda: (v["smoke"].__setitem__(
            "yolov5x", copy.deepcopy(good_v["smoke"]["yolov5x"]))),
        "density_bias": lambda: v["smoke"]["yolov5x"].pop("cap_probe", None),
        "identifiability": lambda: e["identifiability"].update(
            good_e["identifiability"]),
        "enrichment": lambda: e["parasite_load"].update(
            good_e["parasite_load"]),
        "rising_class": lambda: e.__setitem__(
            "classes_rising_after_transform", {}),
        "crop_model": lambda: e["transform_retention"].update(
            {k: good_e["transform_retention"][k]
             for k in ("parasite_kept", "other_kept")}),
        "mixed_aspect": lambda: e["images"].__setitem__(
            "aspect_ratios", good_e["images"]["aspect_ratios"]),
        "rare_classes": lambda: e.__setitem__(
            "cohort_table", good_e["cohort_table"]),
        "fixed_budget": lambda: v["smoke"]["rtdetr"].update(
            {k: good_v["smoke"]["rtdetr"][k]
             for k in ("per_field_min", "per_field_max")}),
        "score_convention": lambda: v["smoke"]["rtdetr"].__setitem__(
            "score_median", good_v["smoke"]["rtdetr"]["score_median"]),
        "imbalance": lambda: e["labels_base"].__setitem__(
            "per_class", good_e["labels_base"]["per_class"]),
        "geometry": lambda: e["centre_distance"].update(
            {k: good_e["centre_distance"][k]
             for k in ("parasite_median_radius", "other_median_radius")}),
        # these two are already clear in the pathological fixture; the
        # "repaired" archive is the same one, and the test below asserts they
        # do NOT fire there either
        "partition_overlap": lambda: None,
        "operating_point": lambda: None,
    }
    swap[key]()
    return e, v


#: The findings that are ALREADY clear in the pathological fixture, because
#: the real archive does not have those problems. They are still checks worth
#: having --- both would invalidate the whole evaluation --- and the test below
#: asserts they fire when the fault is injected.
CLEAR_ON_REAL_ARCHIVE = {"partition_overlap", "operating_point"}


# --------------------------------------------------------------------------- #
# 1. Drift --- the tests that would have caught both bugs
# --------------------------------------------------------------------------- #

def test_no_orphan_keys():
    print("\nevery key a consumer reads is a key a producer declares")
    read = EF.keys_read()
    declared = EDA.paths() | VERIFY.paths()
    orphan = read - declared
    check("no consumer reads an undeclared key", not orphan, str(sorted(orphan)))
    check("consumers read something at all", len(read) > 20, str(len(read)))


def test_consumers_never_subscript_the_documents():
    print("\nconsumers reach the JSON only through exp_schema")
    src = (HERE / "experiment_findings.py").read_text(encoding="utf-8")
    bad = []
    for pat in ('eda[', 'verify[', 'eda.get(', 'verify.get('):
        if pat in src:
            bad.append(pat)
    check("experiment_findings.py has no direct subscripting", not bad,
          str(bad))
    check("every Finding declares what it reads",
          all(f.reads for f in EF.FINDINGS),
          str([f.key for f in EF.FINDINGS if not f.reads]))
    # a check that reads a key it did not declare would slip past the orphan
    # test above, so the declaration is compared to what the checker touches
    for f in EF.FINDINGS:
        names = {k.path.split(".")[0] for k in f.reads}
        check(f"{f.key} declares a coherent block set", bool(names), "")


def test_fixtures_conform_to_the_schema():
    print("\nthe fixtures are built THROUGH the schema, so they conform")
    for nm, doc, sch in (("eda (pathological)", EDA_BAD, EDA),
                         ("eda (healthy)", EDA_OK, EDA),
                         ("verify (pathological)", VERIFY_BAD, VERIFY),
                         ("verify (healthy)", VERIFY_OK, VERIFY)):
        probs = sch.validate(doc)
        check(f"{nm} conforms", not probs, "; ".join(probs[:3]))


# --------------------------------------------------------------------------- #
# 2. Both directions
# --------------------------------------------------------------------------- #

def test_findings_fire_on_the_real_archive():
    print("\nevery finding fires on the archive as measured")
    rows = {r["key"]: r for r in EF.ledger(EDA_BAD, VERIFY_BAD)}
    for f in EF.FINDINGS:
        r = rows[f.key]
        if f.key in CLEAR_ON_REAL_ARCHIVE:
            check(f"{f.key} is clear (the archive does not have it)",
                  not r["fired"] and not r["unknown"], r["detail"])
        else:
            check(f"{f.key} fires", r["fired"] and not r["unknown"],
                  f"fired={r['fired']} unknown={r['unknown']} {r['detail']}")


def test_findings_clear_when_repaired():
    print("\nevery finding CLEARS when only its own cause is repaired")
    for f in EF.FINDINGS:
        if f.key in CLEAR_ON_REAL_ARCHIVE:
            continue
        e, v = repaired(f.key)
        r = {x["key"]: x for x in EF.ledger(e, v)}[f.key]
        check(f"{f.key} clears", not r["fired"], f"still firing: {r['detail']}")


def test_the_two_latent_checks_fire_when_injected():
    print("\nthe checks that are clear on this archive still work")
    e = copy.deepcopy(EDA_BAD)
    EDA.PART_PROBLEMS.put(e, ["train and test share 3 smear(s)"])
    r = {x["key"]: x for x in EF.ledger(e, VERIFY_BAD)}["partition_overlap"]
    check("partition_overlap fires on an overlap", r["fired"], r["detail"])

    v = copy.deepcopy(VERIFY_BAD)
    VERIFY.OP_IOU.put(v, 0.7, name="yolov8x")
    VERIFY.OP_MAX_DET.put(v, 300, name="yolov8x")
    r = {x["key"]: x for x in EF.ledger(EDA_BAD, v)}["operating_point"]
    check("operating_point fires at ultralytics' defaults", r["fired"],
          r["detail"])
    check("it names the mismatch", "iou=0.7" in r["detail"], r["detail"])


def test_missing_input_is_unknown_not_clear():
    print("\na finding whose input is absent reports UNKNOWN, never 'clear'")
    rows = EF.ledger({}, {})
    unk = {r["key"] for r in EF.unknown(rows)}
    fired = {r["key"] for r in rows if r["fired"]}
    check("nothing fires on empty input", not fired, str(fired))
    check("every finding reports unknown", len(unk) == len(EF.FINDINGS),
          f"{len(unk)}/{len(EF.FINDINGS)}: missing "
          f"{[f.key for f in EF.FINDINGS if f.key not in unk]}")


# --------------------------------------------------------------------------- #
# 3. Schema mechanics
# --------------------------------------------------------------------------- #

def test_agreement_excludes_without_crashing():
    """The exclusion path in verify_models.agreement.

    It called `.get()` on a wildcard key with a sub-dict, which the schema
    refuses by design --- so a real run crashed after the inference was
    already paid for. The guard did its job; the test that should have caught
    it did not exist, because the fixture carried no failing checkpoint for
    the exclusion branch to run on.
    """
    import io
    import contextlib
    import verify_models as VM

    doc = copy.deepcopy(VERIFY_BAD)
    lines = []
    with contextlib.redirect_stdout(io.StringIO()):
        VM.agreement(doc, lines.append)
    text = "\n".join(lines)
    check("names the excluded checkpoint",
          "yolov5x left out of the comparison" in text, text[:200])
    check("says why, from the cap count", "at the detection cap" in text,
          text[:200])
    check("still compares the rest",
          "boxes per field" in text and "yolov8x" in text, text[:200])
    check("records the exclusion in the document",
          VERIFY.AGREE_EXCLUDED.get(doc) == ["yolov5x"],
          str(VERIFY.AGREE_EXCLUDED.get(doc)))

    clean = copy.deepcopy(VERIFY_OK)
    lines2 = []
    with contextlib.redirect_stdout(io.StringIO()):
        VM.agreement(clean, lines2.append)
    check("nothing excluded when nothing failed",
          not any("left out" in x for x in lines2), str(lines2[:2]))


def test_wildcard_keys_refuse_positional_get():
    print("\na wildcard key cannot be read positionally")
    try:
        VERIFY.SMOKE_FIELDS_AT_CAP.get({"fields_at_cap": 5})
        check("raises on .get() of a wildcard", False, "no ValueError")
    except ValueError as exc:
        check("raises on .get() of a wildcard", "wildcard" in str(exc),
              str(exc))
    try:
        EDA.IDENT_N_UNINFECTED.each({})
        check("raises on .each() of a plain key", False, "no ValueError")
    except ValueError as exc:
        check("raises on .each() of a plain key", "wildcard" in str(exc),
              str(exc))


def test_schema_mechanics():
    print("\nthe schema itself")
    d = {}
    EDA.IDENT_N_UNINFECTED.put(d, 54)
    check("put then get", EDA.IDENT_N_UNINFECTED.get(d) == 54, str(d))
    check("absent is MISSING, not None",
          EDA.LOAD_ENRICHMENT.get(d) is MISSING)
    check("MISSING is falsey", not EDA.LOAD_ENRICHMENT.get(d))

    v = {}
    VERIFY.SMOKE_FIELDS_AT_CAP.put(v, 5, name="yolov5x")
    VERIFY.SMOKE_FIELDS_AT_CAP.put(v, 0, name="rtdetr")
    check("wildcard each", dict(VERIFY.SMOKE_FIELDS_AT_CAP.each(v))
          == {"yolov5x": 5, "rtdetr": 0})
    check("wildcard names",
          sorted(VERIFY.SMOKE_FIELDS_AT_CAP.names(v)) == ["rtdetr", "yolov5x"])

    try:
        EDA.NO_SUCH_KEY_AT_ALL
        check("a typo raises", False, "no AttributeError")
    except AttributeError:
        check("a typo raises AttributeError at the call site", True)

    check("validate catches a type error",
          any("is str" in p for p in
              EDA.validate({**EDA_BAD, "manifest_rows": "many"})),
          str(EDA.validate({**EDA_BAD, "manifest_rows": "many"})[:2]))
    check("validate passes a good document", not EDA.validate(EDA_BAD))


# --------------------------------------------------------------------------- #
# 4. Driver mechanics
# --------------------------------------------------------------------------- #

def test_cap_override_reaches_inference():
    print("\nsetting mag.detect.MAX_DET reaches the real inference call")
    try:
        from mag import detect
    except ImportError:
        check("mag importable", False, "mag not installed")
        return
    seen = []

    class Stub:
        def predict(self, **kw):
            seen.append(kw["max_det"])
            return []

    args = (Stub(), [Path("a.jpg")], "S1", "ck", {0: "RBC"},
            0.001, 640, 1, "cpu", False, lambda *a: None)
    before = detect.MAX_DET
    try:
        detect._infer_group(*args)
        was = RX.set_cap(6580)
        detect._infer_group(*args)
        check("returns the previous value", was == before, f"{was}!={before}")
        check("the model is driven at both caps", seen == [before, 6580],
              str(seen))
    finally:
        detect.MAX_DET = before


def test_predictions_melt():
    print("\nthe arm diff keys on mag's ACTUAL predictions.csv shape")
    import pandas as pd
    wide = pd.DataFrame({"patient_id": ["S0", "S1"],
                         "rtdetr:R0": ["uninfected", "P_vivax"],
                         "rtdetr:R6": ["P_ovale", "uninfected"],
                         "yolov8x:R6": ["uninfected", "uninfected"]})
    long = RX.melt_predictions(wide)
    check("melts to patient/checkpoint/rule/label",
          list(long.columns) == ["patient_id", "checkpoint", "rule", "label"],
          str(list(long.columns)))
    check("one row per arm per patient", len(long) == 6, str(len(long)))
    check("splits on the FIRST colon only",
          set(long["checkpoint"]) == {"rtdetr", "yolov8x"},
          str(set(long["checkpoint"])))
    check("a long frame is rejected rather than mis-keyed",
          RX.melt_predictions(pd.DataFrame({"a": [1]})) is None)


def test_arm_diff():
    print("\nthe arm diff counts moved labels, by rule")
    import pandas as pd
    with tempfile.TemporaryDirectory() as td:
        A, B = Path(td) / "a", Path(td) / "b"
        A.mkdir(); B.mkdir()
        hi = pd.DataFrame({"patient_id": [f"S{i}" for i in range(10)],
                           "yolov8x:R4": ["uninfected"] * 10,
                           "yolov8x:R6": ["uninfected"] * 10})
        lo = hi.copy()
        lo.loc[0:2, "yolov8x:R6"] = "P_vivax"
        hi.to_csv(A / "predictions.csv", index=False)
        lo.to_csv(B / "predictions.csv", index=False)

        class Ns:
            max_det = 6580
        rc = RX.diff_arms(Ns(), A, B, ["yolov8x"], 3000)
        check("returns 0", rc == 0, str(rc))
        rec = json.loads((A / "cap_sensitivity.json").read_text())
        check("counts the moved cells", rec["cells_changed"] == 3, str(rec))
        by = {r["rule"]: r["changed"] for r in rec["by_rule"]}
        check("attributes them to R6", by == {"R6": 3, "R4": 0}, str(by))


def test_wording_guard():
    print("\nthe v2 -> v3 wording guard")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "ms.tex"
        p.write_text("The apportionment shows the rule matters more than the "
                     "detector.\nWe report specificity by centre.\n"
                     "See \\R{spreadRule}.\n", encoding="utf-8")
        base = RX.check_manuscript(p, unidentifiable=False)
        cond = RX.check_manuscript(p, unidentifiable=True)
        kinds = [k for k, _, _ in base]
        check("catches forbidden v3 wording", kinds.count("forbidden") >= 2,
              str(base))
        check("flags a demoted macro", "demoted" in kinds, str(base))
        check("specificity-by-centre only when unidentifiable",
              len(cond) > len(base), f"{len(base)} -> {len(cond)}")
        clean = Path(td) / "clean.tex"
        clean.write_text("The reconstruction gate refused; we report it.\n",
                         encoding="utf-8")
        check("a clean manuscript passes",
              RX.check_manuscript(clean, unidentifiable=True) == [])
        check("a missing manuscript is reported, not ignored",
              RX.check_manuscript(Path(td) / "nope.tex")[0][0] == "missing")


def test_derived_macros():
    import re as _re
    print("\na macro with no input is ABSENT, never a placeholder")

    class Empty:
        apportionment = {}
        error_summary = {}
        confirmatory = {}
        selective = {}
        factorial = None
        reconstruction = None

    check("empty study, empty context -> no macros",
          RX.derived_macros(Empty(), {}) == {})
    ctx = {"max_det": 6580, "cap_binding": False, "density_bias": 0.0204,
           "discarded_share": 0.036, "enrichment_share": 0.967,
           "thin_species": {"T_brucei": 3},
           "uninfected_centres": ["Montpellier"],
           "crop_gaps": {"parasite": -0.0222, "other": 0.0676}}
    out = RX.derived_macros(Empty(), ctx)
    check("capMaxDet formats with a separator",
          out.get("capMaxDet") == "6,580", str(out.get("capMaxDet")))
    check("bias renders signed", out.get("capDensityBias") == "+2.0\\%",
          str(out.get("capDensityBias")))
    check("crop gaps render both kinds, with the per-cent sign escaped",
          out.get("cropModelGap") == "other +6.8\\%; parasite -2.2\\%",
          str(out.get("cropModelGap")))
    bare = [k for k, v in out.items() if _re.search(r"(?<!\\)%", str(v))]
    check("no generated macro body carries a bare per-cent sign", not bare,
          str(bare))
    check("no macro holds a placeholder",
          all(str(v) not in ("", "None", "TODO", "nan", "MISSING")
              for v in out.values()), str(out))


def test_context_is_lazy_and_idempotent():
    print("\nthe run context loads regardless of which stages were selected")
    with tempfile.TemporaryDirectory() as td:
        e, v = Path(td) / "eda", Path(td) / "verify"
        e.mkdir(); v.mkdir()
        (e / "eda.json").write_text(json.dumps(EDA_BAD), encoding="utf-8")
        (v / "verify.json").write_text(json.dumps(VERIFY_BAD), encoding="utf-8")

        a = RX.build_parser().parse_args(
            ["--eda", str(e), "--verify", str(v), "--max-det", "3000"])
        RX.load_context(a)
        c = a._context
        check("cap requirement comes from the probe", a._cap_need == 6580,
              str(a._cap_need))
        check("a binding cap is recorded", c["cap_binding"] is True, str(c))
        check("density bias picked up", c["density_bias"] == 0.0204, str(c))
        check("thin species picked up", c["thin_species"] == {
            "Babesia_divergens": 3, "T_brucei": 3}, str(c["thin_species"]))
        check("enrichment picked up", c["enrichment_share"] == 0.967, str(c))
        check("uninfected centres picked up",
              c["uninfected_centres"] == ["Montpellier"], str(c))
        before = a._ledger
        RX.load_context(a)
        check("idempotent", a._ledger is before)


def test_preflight_never_refuses():
    print("\npreflight reports and continues, whatever it finds")
    with tempfile.TemporaryDirectory() as td:
        e, v = Path(td) / "eda", Path(td) / "verify"
        e.mkdir(); v.mkdir()
        (e / "eda.json").write_text(json.dumps(EDA_BAD), encoding="utf-8")
        (v / "verify.json").write_text(json.dumps(VERIFY_BAD), encoding="utf-8")
        for cap in (3000, 6580):
            a = RX.build_parser().parse_args(
                ["--eda", str(e), "--verify", str(v), "--max-det", str(cap)])
            import io
            import contextlib
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = RX.stage_preflight(a, None)
            check(f"rc is 0 at max_det={cap}", rc == 0, str(rc))
            if cap == 3000:
                check("a binding cap is still shouted about",
                      "WILL BIND" in buf.getvalue())
        # and with no inputs at all
        a = RX.build_parser().parse_args(
            ["--eda", str(Path(td) / "nope"), "--verify", str(Path(td) / "no")])
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = RX.stage_preflight(a, None)
        check("rc is 0 with no inputs", rc == 0, str(rc))
        check("but it says so", "have not run" in buf.getvalue())


def _fake_mag(fn):
    """A stand-in for `mag.cli` shaped like the parts `_mag_call` uses:
    build_parser().parse_args(argv) -> a namespace whose .fn does the work."""
    class P:
        @staticmethod
        def parse_args(argv):
            ns = argparse.Namespace()
            ns.fn = fn
            return ns

    class FakeCli:
        build_parser = staticmethod(lambda: P)
    return FakeCli


def test_a_mag_refusal_arrives_as_an_exception():
    """`mag.cli.main` returns 3 on a MagError; `_mag_call` must raise it.

    This is the root cause of the first real run's dead end. detect came
    back with rc=3, so the `except MagError` written to repair the cache
    never saw an exception, the run stopped with the cache one filter away
    from usable, and the reason existed only as a line on stderr. Checked
    against the REAL mag, so a change in its error handling fails here
    rather than eighteen minutes into a pass.
    """
    print("\na mag refusal reaches this code as an exception, not a code")
    import contextlib
    import io
    from mag import cli as mag_cli
    from mag.errors import MagError

    argv = ["ingest", "--archive", "/nonexistent-xyz",
            "--image-list", "/nonexistent-xyz/image_list.xlsx",
            "--out", "/nonexistent-xyz/out"]
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        rc = mag_cli.main(list(argv))
    check("mag.cli.main swallows the refusal and returns 3", rc == 3, str(rc))

    raised = None
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            RX._mag_call(mag_cli, list(argv))
    except MagError as exc:
        raised = exc
    check("_mag_call raises it instead", raised is not None)
    check("with the message intact",
          raised is not None and "image list" in str(raised),
          str(raised)[:80])
    check("and the remedy the handler prints",
          raised is not None and bool(getattr(raised, "remedy", None)))


def _write_shards(cache: Path, ck: str, n_bad: int, n_good: int = 40):
    """A cache in the exact state mag leaves after a refused assembly:
    every shard on disk, no table, no manifest, checkpoint_meta present."""
    import pandas as pd
    shard_dir = cache / "_shards" / ck
    shard_dir.mkdir(parents=True, exist_ok=True)
    rows, k = [], 0
    for smear in ("P001", "P002"):
        for i in range(n_good // 2):
            k += 1
            rows.append(dict(patient_id=smear, image_id=f"{smear}_f{i}",
                             checkpoint=ck, cls="RBC", score=0.9,
                             x=100.0 + k, y=100.0, w=20.0, h=20.0))
    for j in range(n_bad):
        rows.append(dict(patient_id="P001", image_id=f"P001_f{j}",
                         checkpoint=ck, cls="RBC", score=0.004,
                         x=5.0, y=5.0, w=0.0, h=3.0))
    df = pd.DataFrame(rows)
    for smear in ("P001", "P002"):
        part = df[df["patient_id"] == smear].reset_index(drop=True)
        part.to_parquet(shard_dir / f"{smear}.parquet", index=False)
        (shard_dir / f"{smear}.json").write_text('{"fingerprint": "x"}',
                                                 encoding="utf-8")
    (cache / f"checkpoint_meta_{ck}.json").write_text(
        json.dumps({"sha256": "a" * 64, "imgsz": 640, "format": "ultralytics"}),
        encoding="utf-8")
    return df


def test_detect_recovers_from_a_refused_assembly():
    """The failure that cost the first real run: 18 minutes, then rc=3.

    `mag.cache.save` validates as it writes, so a pass holding one box of
    non-positive extent raises AFTER inference -- every shard on disk,
    nothing assembled. v2.4 ran the sanitiser only on the success path, so
    the refusal propagated and stopped the run; re-running resumed the
    shards, retried the same assembly and was refused again.

    This drives `_detect_set` with a stand-in for mag that behaves the way
    mag does -- refusing while the shards hold a bad box, succeeding once
    they do not -- and asserts the pass is recovered WITHOUT re-inference.
    """
    print("\ndetect recovers a refused assembly instead of dying on it")
    import pandas as pd
    from mag.errors import BadInput, MissingInput

    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "test"
        cache.mkdir(parents=True)
        _write_shards(cache, "yolov5x", n_bad=7)

        calls = {"n": 0}

        def fake_detect(ns, _ck="yolov5x"):
            """mag's own behaviour: assemble from the shards and refuse
            while any of them holds a box of non-positive extent."""
            calls["n"] += 1
            shards = sorted((cache / "_shards" / _ck).glob("*.parquet"))
            df = pd.concat([pd.read_parquet(q) for q in shards],
                           ignore_index=True)
            bad = int(((df["w"] <= 0) | (df["h"] <= 0)).sum())
            if bad:
                raise BadInput(
                    f"{bad:,} detections have a non-positive width or height",
                    remedy="Check the geometry conversion.")
            return 0

        FakeMag = _fake_mag(fake_detect)

        class A:
            out, results = str(Path(td)), str(Path(td) / "results")
            conf, seed, device, imgsz, batch = 0.001, 0, "cpu", None, None
            yolov5_repo = assume_classes_txt = no_resume = None
            half = None
        A.results = str(Path(td) / "results")

        rc = RX._detect_set(A, None, FakeMag, Path(td) / "images", cache, [])
        check("the refused pass is recovered, not fatal", rc == 0, str(rc))
        check("mag was called again after the repair", calls["n"] == 2,
              str(calls["n"]))

        gj = cache / "geometry_exclusions.json"
        check("the exclusion accounting is written", gj.is_file())
        rec = json.loads(gj.read_text())["yolov5x"]
        check("it counts every excluded box", rec["n_excluded"] == 7,
              str(rec.get("n_excluded")))
        check("and says how many were kept", rec["n_kept"] == 40,
              str(rec.get("n_kept")))
        check("the excluded rows are kept, not dropped",
              (cache / "excluded_geometry_yolov5x.parquet").is_file())

        left = pd.concat([pd.read_parquet(q) for q in
                          sorted((cache / "_shards" / "yolov5x")
                                 .glob("*.parquet"))], ignore_index=True)
        check("the shards no longer hold a degenerate box",
              int(((left["w"] <= 0) | (left["h"] <= 0)).sum()) == 0)
        check("shard sum equals the table provenance compares it to",
              len(left) == rec["n_kept"], f"{len(left)} vs {rec['n_kept']}")
        check("the failure is recorded as a FILE, not only in the log",
              (Path(A.results) / "stage_failures.json").is_file())
        fail = json.loads(
            (Path(A.results) / "stage_failures.json").read_text())
        check("and it names the refusal", fail[0]["type"] == "BadInput"
              and "non-positive" in fail[0]["message"], str(fail[:1])[:160])

    # A refusal the sanitiser cannot repair must NOT be swallowed.
    with tempfile.TemporaryDirectory() as td:
        cache = Path(td) / "test"
        cache.mkdir(parents=True)
        _write_shards(cache, "rtdetr", n_bad=0)
        (cache / "manifest_rtdetr.json").write_text("{}", encoding="utf-8")

        def always_refuse(ns):
            raise MissingInput("no images under x", remedy="Point --archive")

        StubbornMag = _fake_mag(always_refuse)

        class A2:
            out, results = str(Path(td)), str(Path(td) / "results")
            conf, seed, device, imgsz, batch = 0.001, 0, "cpu", None, None
            yolov5_repo = assume_classes_txt = no_resume = None
            half = None

        raised = False
        try:
            RX._detect_set(A2, None, StubbornMag, Path(td) / "im", cache, [])
        except MissingInput:
            raised = True
        check("a refusal it cannot repair is re-raised untouched", raised)


def test_bundle_carries_a_complete_log():
    """The handback zip must hold the WHOLE log, not one buffer short.

    run_all.log is written through a buffered file object and the bundler
    zips that file from disk. The first handback arrived cut off at 16,317
    bytes, mid-sentence, so the bundle said the run had stopped after
    eighteen minutes and nothing about why. The complete file was on the
    machine the whole time.
    """
    print("\nthe handback bundle carries the log complete")
    import run_all as RA

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        results = root / "results"
        results.mkdir()
        log = open(results / "run_all.log", "a", encoding="utf-8")
        real_out = sys.stdout
        sys.stdout = RA.Tee(open(os.devnull, "w"), log)
        try:
            # comfortably more than one 8 KiB buffer, and an unterminated
            # tail is exactly what went missing
            for i in range(400):
                print(f"line {i:04d} " + "x" * 60)
            print("THE LAST LINE BEFORE THE BUNDLE", end="")

            # the REAL parser, so this cannot pass against a stub that has
            # drifted from the arguments phase_bundle actually reads
            A = RA.build_parser().parse_args(
                ["--eda", str(root / "eda"), "--verify", str(root / "verify"),
                 "--out", str(root / "data"), "--results", str(results)])
            rc = RA.phase_bundle(A)
        finally:
            sys.stdout = real_out

        check("the bundle is written", rc == 0, str(rc))
        zips = sorted(results.glob("handback_*.zip"))
        check("exactly one bundle", len(zips) == 1, str(zips))
        # close FIRST, so `on_disk` is the complete file the user keeps ---
        # comparing against the still-buffered file would let a bundler that
        # flushes nothing pass, since both copies would be equally short
        log.close()
        on_disk = (results / "run_all.log").read_bytes()
        with zipfile.ZipFile(zips[0]) as z:
            inside = z.read("results/run_all.log")
        # The bundler prints a line or two of its own AFTER writing the zip,
        # so the file on disk ends up slightly longer. What must hold is that
        # the zipped copy is a PREFIX of it and reaches everything written
        # before the bundle began --- including an unterminated last line,
        # which is the case that went missing.
        check("the zipped log is a prefix of the one on disk",
              on_disk.startswith(inside),
              f"{len(inside)} vs {len(on_disk)} bytes")
        check("and it reaches the last line written before bundling",
              b"THE LAST LINE BEFORE THE BUNDLE" in inside)


def test_a_quiet_phase_reports_that_it_is_working():
    """Silence must look like work, not like a hang.

    The real run has now been killed by hand twice, both times while a stage
    was busy and quiet. Two causes, both fixed here: a Python child writing
    to a pipe is BLOCK buffered, so its progress lines sit unseen until 8 KB
    accumulate; and mag's expensive steps --- concatenating nine million
    detections, hashing the table it just wrote --- print nothing at all.
    """
    print("\na quiet phase says it is still working")
    import contextlib
    import io
    import run_all as RA

    check("the child is told to run unbuffered",
          RA.child_env().get("PYTHONUNBUFFERED") == "1")

    with tempfile.TemporaryDirectory() as td:
        child = Path(td) / "slow.py"
        # quiet for longer than the heartbeat, then one line
        child.write_text("import time\ntime.sleep(2)\nprint('done')\n",
                         encoding="utf-8")
        real_here, real_beat = RA.HERE, RA.HEARTBEAT_SECONDS
        try:
            RA.HERE = Path(td)
            RA.HEARTBEAT_SECONDS = 1
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = RA.run_script("slow.py", [])
            out = buf.getvalue()
        finally:
            RA.HERE, RA.HEARTBEAT_SECONDS = real_here, real_beat

        check("the quiet child still finishes", rc == 0, str(rc))
        check("its output is relayed", "done" in out, out[-200:])
        check("and the silence was reported as work",
              "still working" in out, out[:300])
        check("the note says it is normal",
              "this is normal" in out, out[:300])


def test_console_noise_filter():
    """The yolov5 deprecation spam is dropped; nothing else is.

    A run was interrupted mid-detect because thousands of identical
    `torch.cuda.amp.autocast` FutureWarnings buried mag's progress lines. The
    filter that fixes it is one regex away from eating real output, so it is
    tested against the exact lines from that transcript --- and against the
    lines it must NOT touch.
    """
    print("\nthe console noise filter drops the spam and nothing else")
    import contextlib
    import io
    import os
    import warnings
    import run_all as RA

    hdr = (r"D:\malariaaggregation\malaria_venv\lib\site-packages\yolov5"
           r"\models\common.py:709: FutureWarning: "
           r"`torch.cuda.amp.autocast(args...)` is deprecated. Please use "
           r"`torch.amp.autocast('cuda', args...)` instead.")
    echo = r"  with amp.autocast(autocast):"
    lines = [
        "  smear 0012 ... 300 boxes",
        hdr, echo,
        "  smear 0013 ... 300 boxes",
        # a header Python could not echo a source line for: the NEXT line is
        # real output and must survive, even though it is indented and its
        # path contains "samples"
        r"D:\v\common.py:709: FutureWarning: `torch.cuda.amp.autocast"
        r"(args...)` is deprecated.",
        r"  smear 0014 from ..\samples\P_falciparum ... 300 boxes",
        # an echo form not in the literal list
        r"D:\v\x.py:5: FutureWarning: `torch.cuda.amp.custom_fwd(args...)`"
        r" is deprecated.",
        r"      with torch.cuda.amp.autocast(enabled=True):",
        "  ! format guard: yolov5 declares 'ultralytics'",
        r"D:\v\y.py:9: FutureWarning: a DIFFERENT warning you must see",
        "  done",
    ]

    with tempfile.TemporaryDirectory() as td:
        child = Path(td) / "noisy_child.py"
        child.write_text(
            "import sys\nfor s in %r:\n    print(s)\n" % (lines,),
            encoding="utf-8")
        real_here, real_show = RA.HERE, RA.SHOW_WARNINGS
        try:
            RA.HERE = Path(td)
            for show in (False, True):
                RA.SHOW_WARNINGS = show
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    rc = RA.run_script("noisy_child.py", [])
                out = buf.getvalue()
                tag = "--show-warnings" if show else "default"
                check(f"child ran ({tag})", rc == 0, str(rc))
                if show:
                    check("--show-warnings keeps the spam",
                          out.count("torch.cuda.amp.autocast(args...)") >= 2)
                    check("--show-warnings prints no suppression note",
                          "suppressed" not in out)
                    continue
                check("the deprecation header is dropped",
                      "is deprecated" not in out)
                check("its echoed source line is dropped",
                      "with amp.autocast" not in out)
                for keep in ("smear 0012", "smear 0013", "smear 0014",
                             "format guard", "a DIFFERENT warning", "done"):
                    check(f"kept: {keep}", keep in out)
                check("it reports how many lines it dropped",
                      "suppressed 5 lines" in out, out[-200:])
        finally:
            RA.HERE, RA.SHOW_WARNINGS = real_here, real_show

    # The child filter must be appended to a user's PYTHONWARNINGS, never
    # replace it: someone debugging with PYTHONWARNINGS=error keeps it.
    prior = os.environ.get("PYTHONWARNINGS")
    try:
        os.environ["PYTHONWARNINGS"] = "error::UserWarning"
        RA.SHOW_WARNINGS = False
        env = RA.child_env()
        check("a user's PYTHONWARNINGS survives",
              env["PYTHONWARNINGS"].startswith("error::UserWarning,"))
        check("ours is appended to it",
              "torch.cuda.amp.autocast" in env["PYTHONWARNINGS"])
        RA.SHOW_WARNINGS = True
        check("--show-warnings adds no filter and flags the child",
              RA.child_env().get("MALARIA_SHOW_WARNINGS") == "1"
              and "torch.cuda.amp" not in RA.child_env()["PYTHONWARNINGS"])
    finally:
        RA.SHOW_WARNINGS = False
        if prior is None:
            os.environ.pop("PYTHONWARNINGS", None)
        else:
            os.environ["PYTHONWARNINGS"] = prior

    # Every entry must be one Python itself accepts. A malformed entry makes
    # the interpreter complain on EVERY child process start.
    ok = True
    with warnings.catch_warnings():
        for entry in RA.CHILD_WARNING_FILTERS.split(","):
            try:
                warnings._setoption(entry)
            except Exception:                                 # noqa: BLE001
                ok = False
    check("every PYTHONWARNINGS entry parses", ok, RA.CHILD_WARNING_FILTERS)

    # and the in-process filter the two producers install matches the real
    # message, so a direct `python run_experiment.py` is quiet too
    import re as _re
    msg = _re.compile(r"`torch\.cuda\.amp\.\w+\(args\.\.\.\)` is deprecated")
    check("the in-process filter matches the real message",
          all(msg.match(m) for m in (
              "`torch.cuda.amp.autocast(args...)` is deprecated. Please use "
              "`torch.amp.autocast('cuda', args...)` instead.",
              "`torch.cuda.amp.custom_fwd(args...)` is deprecated.",
              "`torch.cuda.amp.custom_bwd(args...)` is deprecated.")))
    check("both producers install it before torch can load",
          all("hush_amp_deprecations()" in
              (HERE / f).read_text(encoding="utf-8")
              for f in ("run_experiment.py", "verify_models.py")))


# --------------------------------------------------------------------------- #
# 5. The integration test --- real producers, real fixture archives
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# 4b. The manuscript layer --- macros, LaTeX safety, and the register
# --------------------------------------------------------------------------- #

def test_archive_macros_are_latex_safe():
    r"""Every generated macro body must survive pdflatex.

    The first build of this manuscript died with `File ended while scanning
    use of \defresult`, and the error named neither the macro nor the file
    that wrote it. The cause was mag's own transform verdict, which contains
    a literal "75%": an unescaped percent comments out the rest of the line
    and swallows the closing brace. A generated macro file is written by
    code, so the check belongs in code.
    """
    print("\nevery generated archive macro is safe to put in a LaTeX file")
    import make_archive_macros as MAM

    eda = build_eda()
    ver = build_verify()
    # a verdict with the characters that actually broke the build
    S.EDA.TF_VERDICT.put(eda, "consistent with a 4:3 crop: 75% retained, "
                              "cost ~$3 & 100_000 boxes #1 {a} ^2 "
                              + chr(92) + "path")
    macros = {}
    macros.update(MAM.from_eda(eda))
    macros.update(MAM.from_verify(ver))
    check("it produced macros at all", len(macros) > 40, str(len(macros)))

    import re as _re
    unescaped = sorted(k for k, v in macros.items()
                       if v is not None and _re.search(r"(?<!\\)%", str(v)))
    check("no macro holds an unescaped percent", not unescaped,
          str(unescaped[:6]))
    unbalanced = sorted(k for k, v in macros.items()
                        if v is not None
                        and str(v).count("{") != str(v).count("}"))
    check("no macro holds unbalanced braces", not unbalanced,
          str(unbalanced[:6]))

    v = macros["arcTransformVerdict"]
    check("the verdict's percent is escaped", r"75\%" in v, v[:80])
    check("its underscore is escaped", r"100\_000" in v, v[:120])
    # a backslash and a brace, the pair that cannot be escaped by an ordered
    # sequence of str.replace calls: the first rule's output feeds the next
    check("a backslash becomes a control sequence, not a literal one",
          r"\textbackslash{}path" in v and "extbackslash" not in
          v.replace(r"\textbackslash", ""), v[-60:])
    check("braces are escaped exactly once",
          r"\{a\}" in v, v[-90:])

    # ...and the escaping must not reach values that are ALREADY LaTeX. The
    # first fix escaped those too, and \emph{B.\,divergens} came out of the
    # PDF as the literal text `extbackslash{}emph{B. extbackslash{},...}`.
    check("a formatted species name is left as LaTeX",
          r"\emph{" in macros["arcThinSpecies"]
          and "extbackslash" not in macros["arcThinSpecies"],
          macros["arcThinSpecies"])
    check("and so is a rising class",
          "extbackslash" not in macros["arcRisingClasses"],
          macros["arcRisingClasses"])


def test_manuscript_calls_only_macros_some_stage_produces():
    r"""Every \R{} in the manuscript resolves, or is named as outstanding.

    mag's audit fails a build whose manuscript calls a macro the analysis
    never defined --- in the v2 build those rendered as `[armsAnalysed]` and
    three reviewers read them as data. This is the same check, run without a
    build, so a typo in the manuscript is caught while it is being written
    rather than after an hour of GPU time.
    """
    print("\nthe manuscript calls only macros a stage can produce")
    man = HERE / "main.tex"
    if not man.is_file():
        check("main.tex is present", False, "no manuscript in this package")
        return
    import missing_results as MR
    import re as _re

    body = _re.sub(r"(?<!\\)%.*", "", man.read_text(encoding="utf-8"))
    called = set(MR.R_CALL.findall(body))
    check("the manuscript calls result macros", len(called) > 50,
          str(len(called)))

    known = {ln.strip() for ln in
             (HERE / "results_keys.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")}
    # the archive layer, generated from a schema-built fixture so this test
    # does not depend on anyone having run the real producers
    import make_archive_macros as MAM
    import make_result_tables as MRT
    arc = set(MAM.from_eda(build_eda())) | set(MAM.from_verify(build_verify()))
    # from_geometry names its macros in both directions -- an empty accounting
    # still produces the "not assessed" sentence -- so both branches are asked
    # for their names here rather than only the one a fixture happens to hit
    arc |= set(MAM.from_geometry({}))
    arc |= set(MAM.from_geometry({
        "rtdetr": {"checkpoint": "rtdetr", "n_total": 10, "n_excluded": 0,
                   "share": 0.0}}))
    arc |= set(MRT.MACRO_NAMES)
    # mag builds some names per checkpoint, so the fixed list carries only the
    # arms whichever run produced it had
    arc |= {k for k in called if any(p.match(k) for p in MR.TEMPLATED)}

    orphan = sorted(called - known - arc)
    check("no macro is called that no stage defines", not orphan,
          ", ".join(orphan[:8]))

    # and the reverse direction for the archive layer: a macro the manuscript
    # asks of make_archive_macros must be one it can actually produce
    asked = sorted(k for k in called if k.startswith("arc"))
    check("the manuscript asks for at least the archive layer",
          len(asked) > 20, str(len(asked)))
    unproduced = sorted(k for k in asked if k not in arc)
    check("every arc macro the manuscript calls is produced",
          not unproduced, ", ".join(unproduced[:8]))


def test_manuscript_passes_the_wording_guard():
    """No directional verdict is typed where a generated sentence exists.

    Protocol v3 SS17 retires "failed reproduction", the ranges-ratio
    attribution claim, and a specificity claim outside its estimand; SS19
    forbids typing a co-primary verdict a macro exists for. The v2 abstract
    said the conjunction held while its conclusion said the opposite, and
    both were typed.
    """
    print("\nthe manuscript types no verdict a macro exists for")
    man = HERE / "main.tex"
    if not man.is_file():
        check("main.tex is present", False, "no manuscript in this package")
        return
    try:
        from mag import audit
    except ImportError:                                       # pragma: no cover
        check("mag is importable for the wording guard", False, "no mag")
        return
    found = audit.check_no_typed_verdict(man)
    check("no banned wording in the manuscript", not found,
          "; ".join(str(f.message)[:90] for f in found[:3]))


def test_the_two_copies_of_the_wording_guard_agree():
    r"""`reproduce.py` ships without `run_experiment.py` and duplicates it.

    Two copies of one guard that disagree are worse than one guard: the
    offline runbook's copy read `results/figures/apportionment` --- the
    filename mag writes and this package cannot rename --- as forbidden v2
    prose, so the same manuscript passed in one place and failed in the
    other. The copies are kept behaviourally identical here.
    """
    print("\nthe offline copy of the wording guard reads the same as the driver's")
    import importlib.util

    src = HERE / "reproduce.py"
    if not src.is_file():
        check("reproduce.py is present", False, "no offline driver")
        return
    spec = importlib.util.spec_from_file_location("_reproduce_guard", src)
    off = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(off)
    except Exception as exc:                                  # pragma: no cover
        check("reproduce.py imports", False, str(exc)[:90])
        return

    check("the offline copy blanks non-prose regions too",
          set(getattr(off, "NOT_PROSE", ())) == set(RX.NOT_PROSE),
          "NOT_PROSE differs between the two copies")

    with tempfile.TemporaryDirectory() as td:
        cases = {
            "path.tex":
                "\\resultfigure{0.8\\linewidth}{results/figures/apportionment}"
                "{the analyse stage}\n",
            "prose.tex":
                "The apportionment shows the rule matters more.\n",
            "clean.tex":
                "The reconstruction gate refused; we report it.\n",
        }
        for name, body in cases.items():
            q = Path(td) / name
            q.write_text(body, encoding="utf-8")
            a = [(k, w) for k, w, _ in off.check_manuscript(q)]
            b = [(k, w) for k, w, _ in RX.check_manuscript(q)]
            check(f"the two copies agree on {name}", a == b, f"{a} vs {b}")

        q = Path(td) / "path.tex"
        check("a generated figure path is not read as prose",
              off.check_manuscript(q) == [], str(off.check_manuscript(q)))


def test_generated_clauses_are_not_used_as_sentences():
    r"""A lowercase clause after a full stop is a defect nothing else catches.

    `mag report` writes most prose macros as lowercase fragments with no
    terminal stop, to be placed INSIDE a sentence the manuscript supplies.
    Put one where a sentence starts and the paper prints a sentence beginning
    in lower case; put one before a capital and two sentences run together.
    Every other check passes: the macro resolved, the arithmetic agrees, the
    build is clean. Only the page is wrong, and only a reader sees it.
    """
    print("\ngenerated clauses are placed inside sentences, not as sentences")
    import missing_results as MR

    # The (body, source-file) shape `defined_macros` actually returns. The
    # first version of this test passed plain strings, so it passed against a
    # checker that read len(("body", "file")) == 2, called every macro too
    # short to be prose, and reported nothing at all on the real manuscript.
    defined = {
        "frag": ("across 276 configurations, 0 landed within 5 of the total",
                 "results.tex"),
        "whole": ("Diluting each patient's fields, the ordering was preserved.",
                  "results.tex"),
        "fragTable": ("R0 & 1 & 2 \\\\ R4 & 3 & 4 \\\\ R5 & 5 & 6 \\\\",
                      "tables.tex"),
    }

    bad = MR.sentence_shape(defined, "The sweep ran. \\R{frag} The next thing.")
    kinds = {why for _n, why, _e in bad}
    check("a lowercase body opening a sentence is reported",
          any("lower case" in k for k in kinds), str(sorted(kinds)))
    check("a body with no terminal stop before a capital is reported",
          any("terminal stop" in k for k in kinds), str(sorted(kinds)))

    ok = MR.sentence_shape(defined, "The sweep ran: \\R{frag}. The next thing.")
    check("the same body punctuated as a clause is not reported", not ok,
          str(ok[:2]))

    ok2 = MR.sentence_shape(defined, "It fell. \\R{whole} And then it rose.")
    check("a body that is already a sentence is not reported", not ok2,
          str(ok2[:2]))

    cell = MR.sentence_shape(defined, "claim & evidence & \\R{fragTable} \\\\")
    check("a tabular body in a cell is not reported", not cell, str(cell[:2]))

    # A closing brace is not by itself a sentence boundary. The punctuation
    # INSIDE the brace decides: a lead-in ending in a full stop opens a
    # sentence, one ending in a colon does not.
    stop = MR.sentence_shape(defined, "\\paragraph{Field subsampling.} \\R{frag}.")
    check("a braced lead-in ending in a stop is reported",
          any("lower case" in w for _n, w, _e in stop), str(stop[:2]))
    colon = MR.sentence_shape(defined, "\\textbf{No log-odds contrast:} \\R{frag}.")
    check("a braced lead-in ending in a colon is not reported", not colon,
          str(colon[:2]))
    item = MR.sentence_shape(defined, "\\begin{itemize}\\item \\R{frag}. x")
    check("the first thing in a list item is reported",
          any("lower case" in w for _n, w, _e in item), str(item[:2]))

    live = MR.sentence_shape(defined, "value \\R{frag} follows")
    check("a body inside a sentence is not reported", not live, str(live[:2]))

    # The shape the producer really returns, and the shape a caller might
    # hand it. Both must behave identically, because the failure mode of
    # getting it wrong is an empty report rather than an error.
    plain = {k: v[0] for k, v in defined.items()}
    check("a plain {name: body} mapping behaves the same",
          [r[:2] for r in MR.sentence_shape(plain, "It ran. \\R{frag} Next.")]
          == [r[:2] for r in MR.sentence_shape(defined, "It ran. \\R{frag} Next.")],
          "tuple and string mappings disagree")
    check("the real (body, source) mapping still reports",
          bool(MR.sentence_shape(defined, "It ran. \\R{frag} Next.")),
          "reported nothing for a tuple-valued mapping")


def test_the_leakage_check_is_reported_in_both_directions():
    r"""A clean split and a contaminated one both produce a sentence.

    A smear in both training and test makes the whole test estimate
    in-sample, and nothing downstream of it --- not a gate, not an interval,
    not a simulation --- can detect that. The EDA checks every pair of
    released partitions for it. The manuscript must therefore say what the
    check found in EVERY case, including the case where the check did not
    run: a macro defined only on the bad branch renders a red box on a clean
    archive and reads as a missing analysis rather than as a clean result.

    This also pins the label. `partitions.problems` is the DISJOINTNESS list,
    not a count of refused smears; an earlier version of the generator
    printed it under the heading "smears refused by a manifest check", which
    is exactly the class of mistake this paper is about.
    """
    print("\nthe partition-overlap check is reported whatever it found")
    import make_archive_macros as MAM

    clean = MAM.from_eda(build_eda())
    for k in ("arcNSets", "arcSetsChecked", "arcNSetPairs",
              "arcNPartitionOverlaps", "arcLeakageSentence", "arcFlowTable"):
        check(f"{k} is defined on a clean archive", k in clean, "missing")
    check("a clean archive says no smear is shared",
          "no smear appears in more than one" in clean["arcLeakageSentence"],
          clean["arcLeakageSentence"][:70])
    check("the flow row names the overlap check, not a refusal count",
          "partition pairs sharing a smear" in clean["arcFlowTable"]
          and "refused" not in clean["arcFlowTable"],
          clean["arcFlowTable"][:90])

    dirty = build_eda()
    EDA.PART_PROBLEMS.put(dirty, ["train and test share 3 smear(s): the test "
                                  "estimate would be in-sample"])
    d = MAM.from_eda(dirty)
    check("a contaminated archive says the estimate is in-sample",
          "in-sample" in d["arcLeakageSentence"], d["arcLeakageSentence"][:70])
    check("and does not claim the partitions are clean",
          "no smear appears in more than one" not in d["arcLeakageSentence"],
          d["arcLeakageSentence"][:70])

    unchecked = build_eda()
    EDA.SETS.put(unchecked, [])
    u = MAM.from_eda(unchecked)
    check("an unchecked archive says so rather than implying a clean split",
          "not assessed" in u["arcLeakageSentence"],
          u["arcLeakageSentence"][:70])


def test_the_design_arms_table_uses_a_closed_vocabulary():
    r"""Every arm is registered, pre-specified or post hoc --- and says which.

    The arms table exists so a post hoc arm and a registered one cannot look
    identical on the page. That only works if the last column is drawn from a
    closed set: a fourth word, or a blank, and the distinction quietly stops
    meaning anything.
    """
    print("\nthe design-arms table declares every arm from a closed set")
    import re as _re

    man = HERE / "main.tex"
    if not man.is_file():
        check("main.tex is present", False, "not in this package")
        return
    src = man.read_text(encoding="utf-8")
    m = _re.search(r"\\label\{tab:arms\}(.*?)\\bottomrule", src, _re.S)
    check("the arms table is in the manuscript", m is not None)
    if m is None:
        return
    body = m.group(1)
    body = body[body.index("\\midrule") + len("\\midrule"):]
    rows = [r for r in body.split("\\\\") if r.strip()]
    check("it lists a dozen arms or more", len(rows) >= 12, f"{len(rows)} rows")

    allowed = ("registered", "pre-specified", "post hoc")
    undeclared = []
    for r in rows:
        cells = r.split("&")
        if len(cells) < 5:
            undeclared.append(r.strip()[:40])
            continue
        last = _re.sub(r"\\[a-zA-Z]+|\{|\}", "", cells[-1]).strip().lower()
        if not any(last.startswith(a) for a in allowed):
            undeclared.append(cells[0].strip()[:40] + " -> " + last[:30])
    check("every arm is declared registered, pre-specified or post hoc",
          not undeclared, "; ".join(undeclared[:4]))

    check("the post hoc arm is marked in the table",
          "post hoc" in body, "no arm is labelled post hoc")


def test_talk_and_poster_quote_the_same_macro_layer():
    r"""The slides and the poster type no number the paper generates.

    A talk and a poster are where a result gets retyped: someone copies a
    figure off a page at midnight, the analysis is re-run the next week, and
    the paper and the slide disagree by two patients for the rest of their
    lives. Both documents therefore read the same generated macro layer as
    the manuscript, and this test pins three things about them: they call
    macros, they call only macros some stage produces, and every number they
    show arrives through \R{} rather than as a digit in the source.
    """
    print("\nthe talk and the poster quote the macro layer, not each other")
    import missing_results as MR
    import make_archive_macros as MAM
    import make_result_tables as MRT
    import re as _re

    known = {ln.strip() for ln in
             (HERE / "results_keys.txt").read_text(encoding="utf-8").splitlines()
             if ln.strip() and not ln.startswith("#")}
    arc = set(MAM.from_eda(build_eda())) | set(MAM.from_verify(build_verify()))
    # from_geometry names its macros in both directions -- an empty accounting
    # still produces the "not assessed" sentence -- so both branches are asked
    # for their names here rather than only the one a fixture happens to hit
    arc |= set(MAM.from_geometry({}))
    arc |= set(MAM.from_geometry({
        "rtdetr": {"checkpoint": "rtdetr", "n_total": 10, "n_excluded": 0,
                   "share": 0.0}}))
    arc |= set(MRT.MACRO_NAMES)

    for name in ("slides.tex", "poster.tex"):
        doc = HERE / name
        if not doc.is_file():
            check(f"{name} is present", False, "not in this package")
            continue
        body = _re.sub(r"(?<!\\)%.*", "", doc.read_text(encoding="utf-8"))
        called = set(MR.R_CALL.findall(body))
        check(f"{name} quotes the analysis", len(called) > 30,
              f"{len(called)} macros called")
        orphan = sorted(called - known - arc
                        - {k for k in called
                           if any(p.match(k) for p in MR.TEMPLATED)})
        check(f"{name} calls no macro that no stage defines", not orphan,
              ", ".join(orphan[:8]))

        # A typed result would be a decimal or a percentage in the prose. The
        # source legitimately carries integers --- font sizes, column counts,
        # rule identifiers, protocol constants that the paper also states as
        # parameters --- so this looks for the shape a RESULT has: a decimal
        # with a percent sign, which nothing in a layout ever needs.
        prose = body
        for pattern in (r"\\R\{[A-Za-z0-9]+\}",
                        r"\\(?:fontsize|setlength|includegraphics|vskip|"
                        r"vspace|hspace|resultfigure|usepackage|"
                        r"documentclass|definecolor|node|draw|scale)"
                        r"(\[[^]]*\])?(\{[^}]*\})*",
                        r"\\begin\{tikzpicture\}.*?\\end\{tikzpicture\}"):
            prose = _re.sub(pattern, " ", prose, flags=_re.S)
        typed = _re.findall(r"\d+\.\d+\\%", prose)
        check(f"{name} types no percentage of its own", not typed,
              ", ".join(typed[:6]))


def test_bibliography_has_no_orphans_either_way():
    r"""Every \cite resolves, every entry is cited, every entry is locatable.

    A numbered reference list fails silently in both directions: a key with
    no entry renders as `[?]` and a reader skims past it, and an entry no
    sentence cites never appears in the PDF at all, so a bibliography can
    carry a reference the paper does not make and nobody notices. Both are
    checked here, before a build, against the same two files the build reads.

    The third check is the one that matters for a paper making a
    reproducibility argument: an entry with neither a DOI nor a URL cannot be
    followed, and this paper is in no position to ship an unfollowable
    citation.
    """
    print("\nthe bibliography has no orphans and every entry is locatable")
    import re as _re

    man, bib = HERE / "main.tex", HERE / "refs.bib"
    if not (man.is_file() and bib.is_file()):
        check("main.tex and refs.bib are both present",
              False, f"main.tex={man.is_file()} refs.bib={bib.is_file()}")
        return

    body = _re.sub(r"(?<!\\)%.*", "", man.read_text(encoding="utf-8"))
    cited = set()
    for m in _re.finditer(r"\\cite\{([^}]*)\}", body):
        cited |= {k.strip() for k in m.group(1).split(",") if k.strip()}

    raw = bib.read_text(encoding="utf-8")
    entries = _re.findall(r"(?m)^@(\w+)\{([^,]+),((?:[^@])*)", raw)
    keys = {k.strip() for _, k, _ in entries}

    check("the manuscript cites the literature", len(cited) > 10,
          f"{len(cited)} keys cited")
    dangling = sorted(cited - keys)
    check("every cited key has an entry", not dangling,
          ", ".join(dangling[:8]))
    unused = sorted(keys - cited)
    check("every entry is cited somewhere", not unused,
          ", ".join(unused[:8]))

    unlocatable = sorted(k.strip() for _, k, fields in entries
                         if "doi" not in fields.lower()
                         and "url" not in fields.lower())
    check("every entry carries a DOI or a URL", not unlocatable,
          ", ".join(unlocatable[:8]))

    # Braces must balance inside every entry or BibTeX reads the rest of the
    # file as one record and the reference list silently loses entries. This
    # walks from each entry's opening brace to its match rather than counting
    # over a regex slice, because a slice that stops at the next `@' also
    # swallows the comment block above it.
    unbalanced = []
    for m in _re.finditer(r"(?m)^@\w+\{", raw):
        depth, i = 0, m.end() - 1
        while i < len(raw):
            if raw[i] == "{":
                depth += 1
            elif raw[i] == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth:
            unbalanced.append(raw[m.end():m.end() + 40].split(",")[0])
    check("every entry's braces balance", not unbalanced,
          ", ".join(unbalanced[:8]))


def test_latex_hazards_are_found_and_neutralised():
    r"""A character `mag report` does not escape must not reach a build.

    mag escapes % & # and _ in a macro body and nothing else. Two of its
    speciation macros carry the multiclass Brier formula as prose ---
    `(p_ik - y_ik)^2` --- and the bare caret is `! Missing $ inserted.` in
    text mode, which took the rest of the sentence into maths with it. The
    first build against a real results.tex produced two errors and a 1451pt
    overfull line from that one character.
    """
    print("\nLaTeX hazards in generated macros are found, and the caret is "
          "neutralised")
    import missing_results as MR

    defined = {
        "specBrierFormula": (r"BS = mean\_i sum\_k (p\_ik - y\_ik)^2", "r.tex"),
        "niStatement": (r"$\hat\Delta_{\mathrm{sens}} = 0.0$ points", "r.tex"),
        "arcCohortTable": ("centre & a & b \\\\ x & 1 & 2 \\\\", "archive.tex"),
        "someProse": ("a bare 50% and an & tab", "r.tex"),
    }
    found = {h[0]: h for h in MR.latex_hazards(defined)}

    check("the bare caret is found", "specBrierFormula" in found,
          str(sorted(found)))
    check("and reported as handled by the manuscript",
          found.get("specBrierFormula", (None,) * 6)[5] is not None)
    check("a subscript inside real maths is not flagged",
          "niStatement" not in found,
          "escaping it would print the estimator's name with a literal "
          "underscore")
    check("a tabular body's & is not flagged", "arcCohortTable" not in found,
          "the & is the point of a tabular row set")
    check("a bare percent in prose IS flagged", "someProse" in found,
          str(sorted(found)))
    unhandled = [h for h in MR.latex_hazards(defined) if h[5] is None]
    check("and is reported as unhandled", any(h[0] == "someProse"
                                              for h in unhandled))

    # The documents' side of the contract. The guard lives in
    # resultmacros.tex, which the paper, the slides and the poster all input:
    # one copy, because three copies of a catcode trick are three chances for
    # one of them to drift and take a build down.
    rm = HERE / "resultmacros.tex"
    if rm.is_file():
        src = rm.read_text(encoding="utf-8")
        check("resultmacros.tex reads results.tex with ^ made ordinary",
              "\\catcode`\\^=12" in src and "\\begingroup" in src,
              "without this the caret is a hard build error")
        for name in ("main.tex", "slides.tex", "poster.tex"):
            doc = HERE / name
            if not doc.is_file():
                continue
            body = doc.read_text(encoding="utf-8")
            check(f"{name} gets its macros from resultmacros.tex",
                  "resultmacros.tex" in body
                  and "\\catcode`\\^=12" not in body,
                  "either it does not input the shared file, or it carries "
                  "its own copy of the guard")


def test_pdf_build_degrades_without_latex():
    """No TeX on PATH must not fail the run, and must say what to install.

    The inputs to the PDF are the deliverable; the PDF is a convenience. A
    machine without MiKTeX still produces every macro file, every figure and
    every table, and the build can happen anywhere --- so a missing engine is
    reported, not raised.
    """
    print("\nthe PDF step is optional and says so when LaTeX is absent")
    import contextlib
    import io
    import shutil
    import run_all as RA

    real_which = shutil.which
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "main.tex"
        src.write_text("\\documentclass{article}\\begin{document}x"
                       "\\end{document}\n", encoding="utf-8")
        shutil.which = lambda name, *a, **k: None
        try:
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = RA.compile_pdf(src)
            out = buf.getvalue()
        finally:
            shutil.which = real_which
        check("a missing engine is not a failure", rc == RA.EXIT_OK, str(rc))
        check("it names the engines it looked for",
              all(e in out for e in RA.TEX_ENGINES), out[:120])
        check("it says the inputs were still written",
              "has still been written" in out, out[:200])
        check("and how to build it later", "miktex.org" in out.lower(),
              out[:200])


def test_outstanding_register_is_honest():
    """The register counts what is resolved and what is not, and says why."""
    print("\nthe outstanding-results register reports the real state")
    import missing_results as MR

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "doc.tex").write_text(
            "A \\R{arcNPatients} B \\R{primaryVerdict} C \\R{notAThing}\n"
            "% a comment mentioning \\R{phantom} must not be counted\n"
            "and an author slot [FILL: OSF registration DOI or ID]\n",
            encoding="utf-8")
        (d / "macros.tex").write_text(
            "\\defresult{arcNPatients}{170}\n", encoding="utf-8")
        (d / "known.txt").write_text("primaryVerdict\narcNPatients\n",
                                     encoding="utf-8")
        rc = MR.main(["--manuscript", str(d / "doc.tex"),
                      "--macros", str(d / "macros.tex"),
                      "--known", str(d / "known.txt"),
                      "--md", str(d / "out.md"), "--tex", str(d / "out.tex")])
        md = (d / "out.md").read_text(encoding="utf-8")
        tex = (d / "out.tex").read_text(encoding="utf-8")

        check("it exits non-zero while anything is outstanding", rc == 1,
              str(rc))
        check("a resolved macro is listed with its value",
              "`arcNPatients` = 170" in md, md[:200])
        check("an outstanding macro is listed with what it means",
              "primaryVerdict" in md and "generated co-primary verdict" in md)
        check("a macro no stage defines is called out separately",
              "Called but never produced by any stage" in md
              and "notAThing" in md)
        check("a macro named only in a comment is not counted",
              "phantom" not in md, "comments were scanned as calls")
        check("author-supplied slots are listed",
              "[FILL: OSF registration DOI or ID]" in md)
        check("the appendix carries the counts",
              "resolved from a completed run" in tex
              and "awaiting an analysis stage" in tex)


# --------------------------------------------------------------------------- #
# 5. The integration test --- real producers, real fixture archives
# --------------------------------------------------------------------------- #

def test_producers_against_a_real_archive(tmp: Path):
    print("\nthe real producers, on a built archive, conform to the schema")
    import make_fixture

    for flavour, healthy in (("pathological", False), ("healthy", True)):
        arch = tmp / f"arch_{flavour}"
        make_fixture.build(arch, healthy=healthy, seed=0)
        out = tmp / f"eda_{flavour}"
        rc = subprocess.run(
            [sys.executable, str(HERE / "eda_dataset.py"),
             "--root", str(arch), "--out", str(out), "--no-figures"],
            capture_output=True, text=True)
        check(f"eda_dataset runs on the {flavour} archive", rc.returncode == 0,
              rc.stderr[-300:])
        doc = json.loads((out / "eda.json").read_text())
        probs = EDA.validate(doc)
        check(f"its eda.json conforms ({flavour})", not probs,
              "; ".join(probs[:3]))

        rows = {r["key"]: r for r in EF.ledger(doc, VERIFY_BAD)}
        for key in ("identifiability", "rising_class", "mixed_aspect",
                    "enrichment"):
            r = rows[key]
            check(f"{key} {'fires' if not healthy else 'clears'} on the real "
                  f"{flavour} archive",
                  r["fired"] == (not healthy) and not r["unknown"],
                  f"fired={r['fired']} unknown={r['unknown']} {r['detail']}")


# --------------------------------------------------------------------------- #

def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="test_experiment.py")
    p.add_argument("--with-archive", dest="with_archive", action="store_true",
                   help="also build fixture archives and run the real "
                        "producers against them (~30s)")
    a = p.parse_args(argv)

    print("=" * 74)
    print(" test_experiment.py")
    print("=" * 74)
    for fn in (test_no_orphan_keys,
               test_consumers_never_subscript_the_documents,
               test_fixtures_conform_to_the_schema,
               test_findings_fire_on_the_real_archive,
               test_findings_clear_when_repaired,
               test_the_two_latent_checks_fire_when_injected,
               test_missing_input_is_unknown_not_clear,
               test_agreement_excludes_without_crashing,
               test_wildcard_keys_refuse_positional_get,
               test_schema_mechanics,
               test_cap_override_reaches_inference,
               test_predictions_melt,
               test_arm_diff,
               test_wording_guard,
               test_derived_macros,
               test_context_is_lazy_and_idempotent,
               test_preflight_never_refuses,
               test_a_mag_refusal_arrives_as_an_exception,
               test_detect_recovers_from_a_refused_assembly,
               test_bundle_carries_a_complete_log,
               test_console_noise_filter,
               test_a_quiet_phase_reports_that_it_is_working,
               test_archive_macros_are_latex_safe,
               test_manuscript_calls_only_macros_some_stage_produces,
               test_manuscript_passes_the_wording_guard,
               test_generated_clauses_are_not_used_as_sentences,
               test_the_two_copies_of_the_wording_guard_agree,
               test_the_leakage_check_is_reported_in_both_directions,
               test_the_design_arms_table_uses_a_closed_vocabulary,
               test_talk_and_poster_quote_the_same_macro_layer,
               test_bibliography_has_no_orphans_either_way,
               test_latex_hazards_are_found_and_neutralised,
               test_pdf_build_degrades_without_latex,
               test_outstanding_register_is_honest):
        fn()

    if a.with_archive:
        with tempfile.TemporaryDirectory() as td:
            test_producers_against_a_real_archive(Path(td))
    else:
        print("\n  (--with-archive skipped: the real producers are not run)")

    bad = [n for n, ok, _ in _results if not ok]
    print("\n" + "-" * 74)
    print(f"  {len(_results) - len(bad)} passed, {len(bad)} failed")
    for n in bad:
        print(f"    FAIL {n}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
