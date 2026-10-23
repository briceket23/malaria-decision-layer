"""The guarantee the rewrite exists for: no number without a measurement.

The predecessor could generate a cohort and write a results file from it. It
did, and two external reviews were then carried out on the resulting PDF; of
roughly thirty reported numerical contradictions, three were real and the rest
were the placeholder file. These tests pin the arrangement that makes that
impossible rather than unlikely.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import fixtures
from mag import cli, report, study
from mag.errors import MissingInput, ProvenanceError

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "mag"


def test_the_package_contains_no_generator():
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for forbidden in ("import fixtures", "from fixtures",
                          "def patient_table", "def write_cache"):
            assert forbidden not in text, f"{py.name} can fabricate input"


def test_the_package_contains_no_stub_detector():
    """`detect` can be tested without a GPU because the test supplies a stub.

    The stub lives in `tests/test_acquire.py` and is reachable only from there.
    A stub inside the package would be one monkeypatch -- or one `--dry-run`
    flag added in a hurry -- away from writing a cache of invented boxes that
    every downstream stage would treat as measurements.
    """
    for py in SRC.rglob("*.py"):
        text = py.read_text(encoding="utf-8")
        for forbidden in ("class _StubModel", "def fake_predict",
                          "def _synth", "random_boxes", "--dry-run"):
            assert forbidden not in text, f"{py.name} can invent detections"


def test_the_only_writer_of_a_detection_cache_is_the_detector():
    """`cache.save` is called from exactly one place in the package.

    Nothing else may write `detections_<ck>.parquet`, because the manifest
    beside it asserts a SHA-256 for the weights that produced it. A second
    writer is a second way for that assertion to become untrue.
    """
    callers = sorted(py.name for py in SRC.rglob("*.py")
                     if "save_cache(" in py.read_text(encoding="utf-8")
                     or "cache.save(" in py.read_text(encoding="utf-8"))
    assert callers == ["detect.py"], callers


def test_ingest_reads_the_manifest_and_never_completes_it():
    """A patient the manifest does not carry is not a patient. The refusals in
    `ingest` all raise; none fills a value in."""
    import re

    text = (SRC / "ingest.py").read_text(encoding="utf-8")
    # A crosstab cell with no patients really is zero, so `fillna(0)` on the
    # census is fine. What must never appear is a default for a value that
    # identifies a patient: a filled centre or diagnosis is an invented one.
    for forbidden in (r"truth[^\n]*fillna", r"centre[^\n]*fillna",
                      r"species[^\n]*fillna", r"origin[^\n]*fillna",
                      r"or UNINFECTED", r"dropna\(subset"):
        assert not re.search(forbidden, text), \
            f"ingest.py substitutes a patient-identifying value: {forbidden}"


def test_the_fixture_module_is_not_a_tool():
    text = (ROOT / "tests" / "fixtures.py").read_text(encoding="utf-8")
    assert "if __name__" not in text, "the fixture builder became executable"


#: Commands that ANALYSE. Each must be told which cache and which cohort; a
#: default for either is how a pipeline quietly analyses the wrong thing.
#: `detect` is not among them: it WRITES a cache, and its optional --patients
#: only copies the table beside the output as a convenience.
ANALYSING = ("doctor", "run", "figures", "report", "all")


def test_no_analysing_command_has_a_default_cache():
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    for name in ANALYSING:
        assert name in choices, f"{name} is no longer a command"
        for arg in choices[name]._actions:
            if arg.dest in ("cache", "patients"):
                assert arg.required, f"{name} --{arg.dest} has become optional"
                assert arg.default in (None, []), \
                    f"{name} --{arg.dest} has a default"


def test_detect_writes_where_it_is_told_and_nowhere_by_default():
    """It reads images and writes a cache. Both are required, because a
    default output directory is how one checkpoint's cache overwrites
    another's."""
    choices = cli.build_parser()._subparsers._group_actions[0].choices
    for dest in ("images", "out", "weights"):
        arg = next(x for x in choices["detect"]._actions if x.dest == dest)
        assert arg.required, f"detect --{dest} has become optional"


def test_a_missing_cache_is_an_error_with_a_remedy(tmp_path):
    with pytest.raises(MissingInput) as ei:
        study.run(tmp_path / "nothing", tmp_path / "nobody.csv")
    assert ei.value.remedy, "the error tells the operator nothing to do"


def test_a_cache_without_inference_artefacts_cannot_write_results(tmp_path):
    cache, pcsv = fixtures.write_cache(tmp_path, provenanced=False)
    s = study.run(cache, pcsv, condition="matched", force_past_reconstruction=True)
    assert s.provenance.usable is False
    assert any("shards" in r or "checkpoint_meta" in r for r in s.provenance.reasons)

    tex = tmp_path / "out" / "results.tex"
    with pytest.raises(ProvenanceError):
        report.write(s, tex)
    assert not tex.exists(), "a refused write still left a file behind"


def test_a_cache_with_them_can(tmp_path):
    cache, pcsv = fixtures.write_cache(tmp_path, provenanced=True)
    s = study.run(cache, pcsv, condition="matched", force_past_reconstruction=True)
    assert s.provenance.usable, s.provenance.reasons
    tex = tmp_path / "out" / "results.tex"
    keys = report.write(s, tex)
    assert tex.exists() and "provenance" in keys
    assert s.provenance.line() in tex.read_text()


def test_a_hash_mismatch_between_manifest_and_metadata_is_refused(tmp_path):
    import json
    cache, pcsv = fixtures.write_cache(tmp_path, provenanced=True)
    meta = cache / "checkpoint_meta_rtdetr.json"
    d = json.loads(meta.read_text())
    d["sha256"] = "f" * 64
    meta.write_text(json.dumps(d))
    s = study.run(cache, pcsv, condition="matched", force_past_reconstruction=True)
    assert s.provenance.usable is False
    assert any("not the same run" in r for r in s.provenance.reasons)
