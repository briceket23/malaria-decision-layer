"""The two stages that touch the archive, and the cache they hand over.

`ingest` and `detect` sit upstream of every number in the paper: `ingest`
decides who is in the cohort and what their reference diagnosis is, `detect`
decides what evidence exists about them. A defect here is invisible downstream
because everything downstream is consistent with it.

Neither stage can be tested against the real archive in CI -- it is 6 GB of
images behind a registration -- so these tests build a small archive with the
same SHAPE and check the rules that turn it into tables. Nothing here fabricates
a detection: the detector itself is never run, and the one test that exercises
`detect.run` end to end drives it with a stub whose output is fixed and
declared in the test.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from mag import cache as cache_mod
from mag import detect, ingest
from mag.errors import BadInput, MissingInput
from mag.schema import UNINFECTED

COHORT = [("Montpellier", "Uninfected", 4),
          ("Rouen", "P. falciparum", 3),
          ("Saint Louis", "Babesia", 2)]


def build_archive(root, *, n_fields=3, zoom_keeps=None, set_name="test"):
    """An archive with the released one's layout, at 1/2000 the size.

    Returns (archive_dir, image_list_csv). `zoom_keeps`, when given, is the
    number of boxes the test-zoom tree keeps per field, so the transform
    measurement has something to measure.
    """
    rows, pid = [], 0
    for centre, species, k in COHORT:
        for _ in range(k):
            pid += 1
            smear = f"S{pid:03d}"
            for f in range(n_fields):
                image = f"{smear}-{f:02d}.jpg"
                rows.append({"set_name": set_name, "origin": centre,
                             "species": species, "smear_name": smear,
                             "image_name": image,
                             "image_path": f"images/{set_name}/{smear}/{image}"})
    il = pd.DataFrame(rows)
    csv = root / "image_list.csv"
    il.to_csv(csv, index=False)

    archive = root / "archive"
    for r in rows:
        p = archive / r["image_path"]
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"\xff\xd8\xff")            # not a real JPEG; never decoded
        stem = r["image_name"].rsplit(".", 1)[0]
        body = "3 0.5 0.5 0.05 0.05\n1 0.2 0.2 0.10 0.10\n"
        lab = archive / "labels" / set_name / r["smear_name"] / f"{stem}.txt"
        lab.parent.mkdir(parents=True, exist_ok=True)
        lab.write_text(body)
        if zoom_keeps is not None:
            z = archive / "labels" / "test-zoom" / r["smear_name"] / f"{stem}.txt"
            z.parent.mkdir(parents=True, exist_ok=True)
            z.write_text("".join(body.splitlines(keepends=True)[:zoom_keeps]))
    (archive / "classes.txt").write_text("\n".join(ingest.CLASSES_TXT))
    return archive, csv


@pytest.fixture
def archive(tmp_path):
    return build_archive(tmp_path)


# --------------------------------------------------------------------------- #
# ingest
# --------------------------------------------------------------------------- #

def test_the_cohort_comes_from_the_manifest_not_the_filesystem(archive):
    """`n_images` is the manifest's count. The filesystem is reconciled against
    it separately, so that a missing file is a reported discrepancy rather than
    a silently smaller denominator."""
    _, csv = archive
    il = ingest.read_image_list(csv)
    pat = ingest.build_patients(il, "test")
    assert len(pat) == sum(k for _, _, k in COHORT)
    assert set(pat["n_images"]) == {3}
    assert pat["patient_id"].is_unique


def test_the_centre_and_species_vocabularies_are_mapped_not_passed_through(archive):
    _, csv = archive
    pat = ingest.build_patients(ingest.read_image_list(csv), "test")
    # the manifest says "Saint Louis"; the analysis vocabulary is "Saint-Louis"
    assert "Saint-Louis" in set(pat["centre"])
    assert "Saint Louis" not in set(pat["centre"])
    assert set(pat["truth"]) == {UNINFECTED, "P_falciparum", "Babesia_divergens"}


def test_an_unknown_species_is_refused_rather_than_becoming_a_category(tmp_path):
    """A spelling the archive has never used would otherwise appear quietly as
    a seventh label and change every denominator that groups on it."""
    _, csv = build_archive(tmp_path)
    il = pd.read_csv(csv)
    il.loc[0, "species"] = "P. knowlesi"
    il.to_csv(csv, index=False)
    with pytest.raises(BadInput) as e:
        ingest.read_image_list(csv)
    assert "P. knowlesi" in str(e.value)
    assert "deliberately" in str(e.value.remedy)


def test_a_blank_diagnosis_names_the_spreadsheet_row(tmp_path):
    _, csv = build_archive(tmp_path)
    il = pd.read_csv(csv)
    il.loc[2, "species"] = None
    il.to_csv(csv, index=False)
    with pytest.raises(BadInput) as e:
        ingest.read_image_list(csv)
    assert "spreadsheet row 4" in str(e.value), "header row + 1-based counting"


def test_a_smear_carrying_two_diagnoses_is_refused(tmp_path):
    _, csv = build_archive(tmp_path)
    il = pd.read_csv(csv)
    first = il.loc[0, "smear_name"]
    il.loc[il["smear_name"] == first, "species"] = ["Uninfected", "P. ovale",
                                                    "P. ovale"]
    il.to_csv(csv, index=False)
    with pytest.raises(BadInput) as e:
        ingest.build_patients(ingest.read_image_list(csv), "test")
    assert "more than one centre or species" in str(e.value)
    assert first in str(e.value.context.get("smears", []))


def test_an_absent_partition_lists_the_ones_that_exist(archive):
    _, csv = archive
    with pytest.raises(BadInput) as e:
        ingest.build_patients(ingest.read_image_list(csv), "validation")
    assert "test" in str(e.value.context.get("available", []))


def test_a_reordered_classes_file_is_refused(tmp_path):
    p = tmp_path / "classes.txt"
    p.write_text("\n".join(reversed(ingest.CLASSES_TXT)))
    with pytest.raises(BadInput) as e:
        ingest.read_classes(p)
    assert "relabel every box" in str(e.value.remedy)


def test_a_class_file_with_a_byte_order_mark_still_reads(tmp_path):
    """Excel writes one. The first class name silently became "\\ufeffWBC"."""
    p = tmp_path / "classes.txt"
    p.write_bytes(("﻿" + "\n".join(ingest.CLASSES_TXT)).encode("utf-8"))
    assert ingest.read_classes(p) == ingest.CLASSES_TXT


def test_a_missing_image_is_reported_not_absorbed(archive):
    arch, csv = archive
    il = ingest.read_image_list(csv)
    victim = arch / il.loc[0, "image_path"]
    victim.unlink()
    chk = ingest.check_layout(arch, il, "test")
    assert chk["n_listed_but_missing"] == 1
    assert chk["listed_but_missing"][0].endswith(victim.name.lower())


def test_the_layout_check_is_indifferent_to_windows_path_separators(archive):
    """The manifest was authored on Windows and carries backslashes. Compared
    literally against POSIX paths, every one of 4,508 images read as missing."""
    arch, csv = archive
    il = ingest.read_image_list(csv)
    il["image_path"] = il["image_path"].str.replace("/", "\\", regex=False)
    chk = ingest.check_layout(arch, il, "test")
    assert chk["n_listed_but_missing"] == 0


def test_labels_are_read_per_field_and_empty_files_are_counted(archive):
    arch, _ = archive
    lab = ingest.read_yolo_labels(arch, "test")
    n_fields = sum(k for _, _, k in COHORT) * 3
    assert lab.attrs["n_label_files"] == n_fields
    assert lab.attrs["n_empty_files"] == 0
    assert len(lab) == 2 * n_fields
    assert set(lab["cls"]) == {"P_falciparum", "RBC"}  # class ids 3 and 1


def test_an_empty_label_file_is_a_field_with_no_annotation_not_a_missing_field(
        archive):
    arch, _ = archive
    victim = sorted((arch / "labels" / "test").rglob("*.txt"))[0]
    victim.write_text("")
    lab = ingest.read_yolo_labels(arch, "test")
    assert lab.attrs["n_empty_files"] == 1
    assert lab.attrs["n_label_files"] == sum(k for _, _, k in COHORT) * 3


def test_a_class_id_outside_the_vocabulary_is_refused(archive):
    """A negative id indexes from the end of the tuple and would silently
    relabel the box as T. brucei."""
    arch, _ = archive
    victim = sorted((arch / "labels" / "test").rglob("*.txt"))[0]
    victim.write_text("-1 0.5 0.5 0.1 0.1\n")
    with pytest.raises(BadInput) as e:
        ingest.read_yolo_labels(arch, "test")
    assert "outside 0..8" in str(e.value)


def test_a_missing_label_tree_says_the_analysis_can_proceed_without_it(archive):
    arch, _ = archive
    with pytest.raises(MissingInput) as e:
        ingest.read_yolo_labels(arch, "nosuchset")
    assert "you can proceed without them" in str(e.value.remedy)


def test_the_transform_is_measured_from_the_two_label_trees_not_inferred(
        tmp_path):
    """The source study's "zoom correction" is not an ambiguity to reason
    about: the archive ships both label trees, so it is a count."""
    arch, _ = build_archive(tmp_path, zoom_keeps=1)
    tr = ingest.transform_retention(arch, "test", "test-zoom")
    assert tr["parasite_kept"] == pytest.approx(1.0)
    assert tr["all_kept"] == pytest.approx(0.5)
    assert tr["boxes_base"] == 2 * tr["boxes_zoom"]
    assert tr["verdict"], "a measurement without a verdict is half an answer"


def test_the_cohort_table_is_the_manuscripts_table_one(archive):
    _, csv = archive
    pat = ingest.build_patients(ingest.read_image_list(csv), "test")
    ct = ingest.cohort_table(pat)
    assert ct.loc["Total", "Total"] == len(pat)
    assert ct.loc["Montpellier", UNINFECTED] == 4
    assert ct.loc["Rouen", "P_falciparum"] == 3
    # every centre row sums to its total, so the table cannot lose a patient
    for centre in [c for c in ct.index if c != "Total"]:
        assert ct.loc[centre].drop("Total").sum() == ct.loc[centre, "Total"]


# --------------------------------------------------------------------------- #
# detect: the parts that need no detector
# --------------------------------------------------------------------------- #

def test_a_checkpoint_with_no_class_names_refuses_before_it_guesses():
    with pytest.raises(BadInput) as e:
        detect.class_map_from({"file": "w.pt"}, strict=True)
    assert "--assume-classes-txt" in str(e.value.remedy)
    assert "say so in the manuscript" in str(e.value.remedy)


def test_the_fallback_is_the_archives_class_order_and_nothing_else():
    cmap = detect.class_map_from({"file": "w.pt"}, strict=False)
    assert len(cmap) == len(ingest.CLASSES_TXT)
    assert cmap[0] == "WBC" and cmap[3] == "P_falciparum"


def test_an_unmatched_class_name_is_refused_not_dropped():
    """It used to be dropped silently, and every detection of that class
    vanished with no count and no message."""
    with pytest.raises(BadInput) as e:
        detect.class_map_from({"file": "w.pt",
                               "names": {0: "WBC", 1: "cytoplasm"}})
    assert "cytoplasm" in str(e.value)
    assert "dropped silently" in str(e.value.remedy)


def test_class_names_are_matched_through_the_usual_abbreviations():
    cmap = detect.class_map_from({"names": ["wbc", "rbc", "plt", "falciparum",
                                            "ovale", "malariae", "vivax",
                                            "babesia", "t.brucei"]})
    assert cmap[3] == "P_falciparum" and cmap[8] == "T_brucei"
    assert cmap[2] == "platelet"


def test_a_silent_checkpoint_is_named_as_a_bug_not_reported_as_a_result():
    """4,508 fields returning 2 boxes is a loader that was never driven
    correctly. The pipeline says so and refuses the cache."""
    why = detect.implausible_yield(2, 2, 4508, 0.001)
    assert why and "0.04%" in why
    assert detect.implausible_yield(50_000, 4400, 4508, 0.001) == ""


def test_a_high_confidence_floor_makes_a_sparse_cache_expected_not_suspect():
    """At a 0.25 floor a sparse cache is the floor doing its job. The check
    only fires where sparsity cannot be explained that way."""
    assert detect.implausible_yield(2, 2, 4508, 0.30) == ""


def test_a_suspect_cache_is_discovered_by_name(tmp_path):
    (tmp_path / "detections_bad.suspect.json").write_text(
        json.dumps({"checkpoint": "bad", "reason": "silent"}))
    assert detect.suspect_checkpoints(tmp_path) == {"bad": "silent"}
    assert detect.suspect_checkpoints(tmp_path / "nope") == {}


def test_the_batch_size_is_chosen_from_the_card_not_discovered_by_oom():
    b = detect.gpu_budget("cpu")
    assert b["cuda"] is False and b["half"] is False
    assert b["batch"] >= 1


# --------------------------------------------------------------------------- #
# detect.run, driven by a stub
# --------------------------------------------------------------------------- #

class _Boxes:
    def __init__(self, xywhn, conf, cls):
        self.xywhn, self.conf, self.cls = xywhn, conf, cls

    def __len__(self):
        return len(self.conf)


class _Result:
    @classmethod
    def of(cls, path, idx, conf, xywhn):
        r = cls.__new__(cls)
        r.path = str(path)
        r.boxes = _Boxes(xywhn, conf, idx)
        return r

    def __init__(self, path, n):
        self.path = str(path)
        rng = np.random.default_rng(abs(hash(self.path)) % (2 ** 32))
        self.boxes = _Boxes(rng.uniform(0.1, 0.9, size=(n, 4)),
                            rng.uniform(0.3, 0.99, size=n),
                            rng.integers(0, 9, size=n))


class _StubModel:
    """A detector that returns a fixed number of boxes per image.

    It exists so that `detect.run`'s sharding, resume and cache-writing can be
    tested without a GPU. It is a TEST DOUBLE for the plumbing, not a source of
    results: it lives in the test file, is never importable from `mag`, and
    `test_no_fabrication` asserts that no such thing exists in the package.
    """

    def __init__(self, n_boxes=4):
        self.n_boxes = n_boxes
        self.calls = 0

    def predict(self, source, **_):
        self.calls += 1
        for s in source:
            yield _Result(s, self.n_boxes)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    arch, csv = build_archive(tmp_path)
    model = _StubModel()
    monkeypatch.setattr(detect, "load_model", lambda *a, **k: (model, "stub"))
    monkeypatch.setattr(detect, "read_checkpoint_metadata",
                        lambda w, repo=None: {"file": str(w), "sha256": "a" * 64,
                                              "format": "stub"})
    return arch, csv, model


def _run(arch, out, name="stub", **kw):
    cmap = detect.class_map_from({"file": "w.pt"}, strict=False)
    return detect.run(arch / "weights.pt", arch / "images" / "test", out, name,
                      cmap, device="cpu", progress=lambda *a: None, **kw)


def test_the_cache_a_run_writes_is_the_cache_the_analysis_reads(stubbed,
                                                                tmp_path):
    """The join between the two halves of the package. If `detect` wrote a
    column `cache.load` did not expect, every test above and below this one
    would still pass."""
    arch, _, _ = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    det, man = cache_mod.load(out, "stub", verify=True)
    assert man.checkpoint == "stub"
    assert man.nms == "none", "the cache is pre-suppression by design"
    assert man.conf_floor == pytest.approx(0.001)
    assert man.n_detections == len(det)
    n_fields = sum(k for _, _, k in COHORT) * 3
    assert det["image_id"].nunique() == n_fields
    assert det["patient_id"].nunique() == sum(k for _, _, k in COHORT)


def test_the_manifest_hash_describes_the_file_on_disk(stubbed, tmp_path):
    """`cache.save` hashes what it just wrote, not what it held in memory: a
    round trip through parquet can move a float's last place, and a manifest
    that disagrees with the file beside it makes the next load refuse the cache
    and demand an hour of GPU time to reproduce the same mismatch."""
    arch, _, _ = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    cache_mod.load(out, "stub", verify=True)          # would raise on mismatch


def test_a_second_run_reuses_every_shard_and_re_infers_nothing(stubbed,
                                                               tmp_path):
    arch, _, model = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    first = model.calls
    assert first > 0
    _run(arch, out)
    assert model.calls == first, "resume re-ran a smear whose shard was valid"


def test_a_changed_confidence_floor_invalidates_the_shards(stubbed, tmp_path):
    """The fingerprint covers the floor, the image size, the class map, the
    image list and the checkpoint's own SHA. A shard reused across a changed
    floor would be a cache whose manifest describes a run that never happened."""
    arch, _, model = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    before = model.calls
    _run(arch, out, conf_floor=0.10)
    assert model.calls > before


def test_no_resume_starts_the_pass_again(stubbed, tmp_path):
    arch, _, model = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    before = model.calls
    _run(arch, out, resume=False)
    assert model.calls > before


def test_a_cache_that_does_not_cover_the_image_set_is_refused(stubbed,
                                                              tmp_path):
    """Every per-patient denominator would mean something different from the
    one the manifest declares."""
    arch, _, _ = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    # delete one smear's shard and re-run with a fingerprint that still matches,
    # after removing that smear's images so the assembly cannot recreate it
    shard = sorted((out / "_shards" / "stub").glob("*.parquet"))[0]
    smear = shard.stem
    for p in (arch / "images" / "test" / smear).glob("*"):
        p.unlink()
    (arch / "images" / "test" / smear).rmdir()
    shard.unlink()
    # the smear is gone from the image set too, so the run succeeds on what is
    # left -- the stale shard sidecar must not resurrect it
    _run(arch, out)
    det, _ = cache_mod.load(out, "stub", verify=True)
    assert smear not in set(det["patient_id"])


def test_an_empty_image_directory_refuses_with_somewhere_to_look(stubbed,
                                                                 tmp_path):
    arch, _, _ = stubbed
    with pytest.raises(MissingInput) as e:
        detect.run(arch / "w.pt", tmp_path / "empty", tmp_path / "c", "stub",
                   {0: "WBC"}, device="cpu", progress=lambda *a: None)
    assert "mag doctor" in str(e.value.remedy)


def test_a_silent_run_marks_the_cache_suspect_and_the_analysis_refuses_it(
        monkeypatch, tmp_path):
    arch, _ = build_archive(tmp_path, n_fields=40)
    model = _StubModel(n_boxes=0)
    monkeypatch.setattr(detect, "load_model", lambda *a, **k: (model, "stub"))
    monkeypatch.setattr(detect, "read_checkpoint_metadata",
                        lambda w, repo=None: {"file": str(w), "sha256": "a" * 64,
                                              "format": "stub"})
    out = tmp_path / "cache"
    _run(arch, out)
    flag = out / "detections_stub.suspect.json"
    assert flag.is_file()
    payload = json.loads(flag.read_text())
    assert payload["n_detections"] == 0
    assert detect.suspect_checkpoints(out) == {"stub": payload["reason"]}


def test_the_suspect_flag_is_cleared_when_a_later_run_is_healthy(monkeypatch,
                                                                 tmp_path):
    """A stale flag would refuse a cache that has since been fixed, and the
    operator would be told to re-run something they had just re-run."""
    arch, _ = build_archive(tmp_path, n_fields=40)
    silent = _StubModel(n_boxes=0)
    monkeypatch.setattr(detect, "read_checkpoint_metadata",
                        lambda w, repo=None: {"file": str(w), "sha256": "a" * 64,
                                              "format": "stub"})
    monkeypatch.setattr(detect, "load_model", lambda *a, **k: (silent, "stub"))
    out = tmp_path / "cache"
    _run(arch, out)
    assert (out / "detections_stub.suspect.json").is_file()

    healthy = _StubModel(n_boxes=6)
    monkeypatch.setattr(detect, "load_model", lambda *a, **k: (healthy, "stub"))
    _run(arch, out, resume=False)
    assert not (out / "detections_stub.suspect.json").exists()
    assert detect.suspect_checkpoints(out) == {}


def test_the_boxes_reach_the_cache_in_the_schemas_coordinate_convention(
        stubbed, tmp_path):
    """Normalised centre-x, centre-y, width, height. Suppression, the density
    rules and the box census all assume it, and none of them would fail
    loudly on xyxy."""
    arch, _, _ = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    det, _ = cache_mod.load(out, "stub", verify=True)
    for col in ("x", "y", "w", "h"):
        assert det[col].between(0.0, 1.0).all(), col
        assert det[col].dtype == np.float64
    assert det["score"].between(0.0, 1.0).all()


def test_a_class_index_the_map_does_not_cover_cannot_reach_the_cache(
        monkeypatch, tmp_path):
    """The stub emits indices 0..8; the map covers 0..2. The rest are dropped
    at the box, which is the defensive half of the strict check in
    `class_map_from` -- and what is dropped must never be relabelled."""
    arch, _ = build_archive(tmp_path)
    model = _StubModel(n_boxes=9)
    monkeypatch.setattr(detect, "load_model", lambda *a, **k: (model, "stub"))
    monkeypatch.setattr(detect, "read_checkpoint_metadata",
                        lambda w, repo=None: {"file": str(w), "sha256": "a" * 64,
                                              "format": "stub"})
    out = tmp_path / "cache"
    detect.run(arch / "w.pt", arch / "images" / "test", out, "stub",
               {0: "WBC", 1: "RBC", 2: "platelet"}, device="cpu",
               progress=lambda *a: None)
    det, _ = cache_mod.load(out, "stub", verify=True)
    assert set(det["cls"]) <= {"WBC", "RBC", "platelet"}


def test_the_checkpoint_metadata_travels_beside_the_cache(stubbed, tmp_path):
    """`provenance` matches this file's SHA against the manifest's. Without it
    a cache cannot be traced to the weights that produced it."""
    arch, _, _ = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    meta = json.loads((out / "checkpoint_meta_stub.json").read_text())
    assert meta["sha256"] == "a" * 64
    _, man = cache_mod.load(out, "stub", verify=True)
    assert man.checkpoint_sha256 == meta["sha256"]


def test_a_torn_shard_is_redone_rather_than_read(stubbed, tmp_path):
    arch, _, model = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    victim = sorted((out / "_shards" / "stub").glob("*.parquet"))[0]
    victim.write_bytes(b"not a parquet file")
    before = model.calls
    _run(arch, out)
    assert model.calls > before
    cache_mod.load(out, "stub", verify=True)


def test_a_shard_without_its_fingerprint_is_redone(stubbed, tmp_path):
    """Written by an older version, so there is nothing to compare against.
    Reusing it would mean trusting a run whose parameters are unknown."""
    arch, _, model = stubbed
    out = tmp_path / "cache"
    _run(arch, out)
    sidecar = sorted((out / "_shards" / "stub").glob("*.json"))[0]
    sidecar.unlink()
    before = model.calls
    _run(arch, out)
    assert model.calls > before


# --------------------------------------------------------------------------- #
# The joint: a cache this package WROTE, analysed by this package
# --------------------------------------------------------------------------- #

class _StructuredModel:
    """A stub whose boxes carry a signal, so the analysis has something to do.

    Infected smears get high-confidence boxes of their own species; every smear
    gets blood cells and a trickle of low-confidence parasite noise. This is the
    same structure `tests/fixtures.py` builds directly into a cache -- but here
    it goes through `detect.run`, so the test exercises the real writer and the
    real reader against each other rather than a hand-written cache in the
    middle.

    It is a TEST DOUBLE and lives in the test file. `test_no_fabrication`
    asserts that nothing like it exists inside the package.
    """

    def __init__(self, truth_of, classes, spec_bias=0.0, seed=0):
        self.truth_of = truth_of
        self.classes = list(classes)
        self.spec_bias = spec_bias
        self.rng = np.random.default_rng(seed)

    def predict(self, source, **_):
        for s in source:
            smear = pathlib.Path(s).parent.name
            truth = self.truth_of.get(smear, UNINFECTED)
            idx, conf = [], []
            for c, n in ((0, 2), (1, 30), (2, 3)):          # WBC, RBC, platelet
                idx += [c] * n
                conf += list(self.rng.uniform(0.3, 0.99, n))
            for _ in range(self.rng.poisson(1.2 + self.spec_bias)):
                idx.append(int(self.rng.integers(3, 9)))
                conf.append(float(self.rng.uniform(0.001, 0.20)))
            if truth != UNINFECTED:
                k = self.classes.index(truth)
                for _ in range(int(self.rng.integers(2, 7))):
                    idx.append(k)
                    conf.append(float(self.rng.uniform(0.55, 0.99)))
            n = len(idx)
            yield _Result.of(s, np.asarray(idx), np.asarray(conf),
                             self.rng.uniform(0.1, 0.9, size=(n, 4)))


def test_a_cache_this_package_wrote_is_a_cache_it_can_analyse(monkeypatch,
                                                              tmp_path):
    """The seam between the two halves of the repository.

    Every other test on either side uses a cache built by its own fixtures. If
    `detect` wrote a column, a dtype or a coordinate convention that `study`
    did not expect, all of them would still pass and only this one would fail.
    """
    from mag import cli, ingest as ing

    big = [("Montpellier", "Uninfected", 20), ("Rouen", "P. falciparum", 16),
           ("Toulouse", "P. ovale", 12), ("Lille", "P. vivax", 10),
           ("Nantes", "P. malariae", 8), ("Saint Louis", "Babesia", 5)]
    monkeypatch.setattr("tests.test_acquire.COHORT", big, raising=False)
    globals()["COHORT"] = big
    try:
        arch, csv = build_archive(tmp_path, n_fields=6)
        patients = ing.build_patients(ing.read_image_list(csv), "test")
        truth_of = dict(zip(patients["patient_id"].astype(str),
                            patients["truth"].astype(str)))
        classes = list(ing.CLASS_TO_SCHEMA[c] for c in ing.CLASSES_TXT)

        monkeypatch.setattr(detect, "read_checkpoint_metadata",
                            lambda w, repo=None: {"file": str(w),
                                                  "sha256": "b" * 64,
                                                  "format": "stub"})
        cache = tmp_path / "cache"
        for name, bias in (("rtdetr", 0.0), ("yolov8x", 0.8)):
            model = _StructuredModel(truth_of, classes, spec_bias=bias,
                                     seed=hash(name) % 1000)
            monkeypatch.setattr(detect, "load_model",
                                lambda *a, _m=model, **k: (_m, "stub"))
            detect.run(tmp_path / f"{name}.pt", arch / "images" / "test",
                       cache, name,
                       detect.class_map_from({"file": "w.pt"}, strict=False),
                       device="cpu", progress=lambda *a: None)

        pcsv = cache / "patients.csv"
        patients.to_csv(pcsv, index=False, lineterminator="\n")

        out = tmp_path / "out"
        rc = cli.main(["all", "--cache", str(cache), "--patients", str(pcsv),
                       "--out", str(out), "--depth", "core",
                       "--min-fields", "5", "--reps", "3",
                       "--force-reconstruction"])
        assert rc == 0
        res = json.loads((out / "results.json").read_text())
        assert res["provenance"], "the cache carried no provenance line"
        assert res["confirmatory"]["mcnemar"]["alternative"] == "greater"
        fac = pd.read_csv(out / "factorial.csv")
        assert set(fac["checkpoint"]) >= {"rtdetr", "yolov8x"}
        # every arm scored the same cohort: tp+fn+tn+fp is the patient count,
        # and a checkpoint that lost a patient somewhere in the cache would
        # show up here and nowhere else
        totals = (fac["tp"] + fac["fn"] + fac["tn"] + fac["fp"]).unique()
        assert len(totals) == 1, f"arms scored different cohorts: {totals}"
        assert int(totals[0]) == len(patients)
        from mag import figures as figmod
        assert (out / "figures" / figmod.STAMP).is_file()
    finally:
        globals()["COHORT"] = [("Montpellier", "Uninfected", 4),
                               ("Rouen", "P. falciparum", 3),
                               ("Saint Louis", "Babesia", 2)]


# --------------------------------------------------------------------------- #
# The transform measurement, from ingest to the manuscript
# --------------------------------------------------------------------------- #

def test_the_transform_measurement_travels_from_ingest_to_the_macros(tmp_path):
    """The archive ships both label trees, so the source study's zoom
    correction is a count. `ingest` counts it beside the patient table and the
    analysis quotes it -- without needing the 6 GB archive, which a reviewer
    re-running from a published cache does not have."""
    from mag import cli, report, study

    arch, csv = build_archive(tmp_path, zoom_keeps=1)
    data = tmp_path / "data"
    rc = cli.main(["ingest", "--image-list", str(csv), "--archive", str(arch),
                   "--out", str(data), "--min-fields", "1"])
    assert rc == 0
    tf = json.loads((data / "transform.json").read_text())
    assert tf["parasite_kept"] == pytest.approx(1.0)
    assert tf["all_kept"] == pytest.approx(0.5)

    loaded = study.load_transform(data / "patients.csv", data)
    assert loaded["measured"] is True
    s = study.Study()
    s.transform = loaded
    macros = report._transform(s)
    assert macros["tfMeasured"] == "1"
    # the percent sign is LaTeX-escaped; unescaped it comments out the rest
    # of the line and the sentence after it vanishes from the PDF
    assert macros["tfKept"] == r"50.0\%"
    assert macros["tfParasiteKept"] == r"100.0\%"
    assert macros["annBase"] != macros["annAfterTf"]
    assert "direct count, not an inference" in macros["tfSentence"]


def test_an_unmeasured_transform_says_so_rather_than_quoting_a_fraction(
        tmp_path):
    """A retained fraction nobody counted is exactly the kind of number the
    previous build was reviewed on."""
    from mag import report, study

    missing = study.load_transform(tmp_path / "patients.csv", tmp_path)
    assert missing["measured"] is False
    assert missing["why"]

    s = study.Study()
    s.transform = missing
    m = report._transform(s)
    assert m["tfMeasured"] == "0"
    assert m["tfKept"] == "NOT MEASURED"
    assert m["annRecovered"] == "NOT MEASURED"
    assert "No retained fraction is reported" in m["tfSentence"]


def test_a_torn_transform_file_is_reported_not_partially_believed(tmp_path):
    from mag import study

    (tmp_path / "transform.json").write_text("{not json")
    got = study.load_transform(tmp_path / "patients.csv", tmp_path)
    assert got["measured"] is False
    assert "could not be read" in got["why"]
