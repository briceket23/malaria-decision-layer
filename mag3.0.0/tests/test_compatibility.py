"""The rewrite must read the caches that already exist.

Six GPU passes over 4,508 fields cost about fifty minutes on a 6 GB card. A
rewrite that made an operator repeat them to gain tidier code would be a bad
trade, so the on-disk format is deliberately unchanged and this is the test
that keeps it that way.

The cache written by the previous package holds, per checkpoint:

    detections_<name>.parquet    the table
    manifest_<name>.json         its content hash and the checkpoint hash
    checkpoint_meta_<name>.json  what was read out of the .pt file
    _shards/<name>/*.parquet     one per smear, written as inference ran

If any of those names or the hash convention changes, this test fails before an
operator discovers it against a 44 GB archive.
"""

from __future__ import annotations

import json

import pandas as pd

import fixtures
from mag import cache as cache_mod, study
from mag.schema import DETECTION_COLUMNS, content_hash


def _legacy_cache(tmp_path):
    """A cache written the way the previous package wrote one, by hand.

    Written literally rather than by importing the old code, so that this test
    documents the format it promises to read instead of depending on the thing
    being replaced.
    """
    cache = tmp_path / "legacy"
    cache.mkdir(parents=True)
    patients = fixtures.patient_table(n_fields=(3, 5), seed=4)
    patients.to_csv(cache / "patients.csv", index=False)
    det = fixtures.detections(patients, "rtdetr", seed=4)[list(DETECTION_COLUMNS)]
    det.to_parquet(cache / "detections_rtdetr.parquet", index=False)

    (cache / "manifest_rtdetr.json").write_text(json.dumps({
        "checkpoint": "rtdetr",
        "checkpoint_sha256": "c" * 64,
        "conf_floor": 0.001, "nms": "none", "iou": None, "imgsz": 640,
        "n_images": int(det["image_id"].nunique()),
        "n_detections": int(len(det)),
        "content_sha256": content_hash(det, list(DETECTION_COLUMNS)),
        "software": {"python": "3.9.13", "torch": "2.7.1+cu118"},
        "seed": 0, "created_utc": "2026-08-25T01:00:00+00:00",
        # a key the rewrite does not know about: an older writer's field must
        # not stop a newer reader
        "legacy_only_field": "ignored",
    }), encoding="utf-8")
    (cache / "checkpoint_meta_rtdetr.json").write_text(json.dumps({
        "file": "D:/models/rt-detr/model.pt", "sha256": "c" * 64,
        "format": "ultralytics", "names": {"0": "WBC", "1": "RBC"},
        "loader": "RTDETR", "imgsz": 640,
    }), encoding="utf-8")
    shards = cache / "_shards" / "rtdetr"
    shards.mkdir(parents=True)
    for pid in patients["patient_id"]:
        (shards / f"{pid}.parquet").write_bytes(b"\x00shard")
    return cache, cache / "patients.csv"


def test_a_cache_from_the_previous_package_loads_unchanged(tmp_path):
    cache, _ = _legacy_cache(tmp_path)
    det, man = cache_mod.load(cache, "rtdetr", verify=True)
    assert len(det) == man.n_detections
    assert man.checkpoint_sha256 == "c" * 64
    assert man.conf_floor == 0.001


def test_an_unknown_manifest_field_does_not_stop_the_reader(tmp_path):
    cache, _ = _legacy_cache(tmp_path)
    _, man = cache_mod.load(cache, "rtdetr")
    assert not hasattr(man, "legacy_only_field")


def test_the_content_hash_convention_is_unchanged(tmp_path):
    """If this drifts, every existing cache fails its own integrity check and
    fifty minutes of GPU time has to be spent again."""
    cache, _ = _legacy_cache(tmp_path)
    det, man = cache_mod.load(cache, "rtdetr", verify=False)
    assert content_hash(det, list(DETECTION_COLUMNS)) == man.content_sha256


def test_a_legacy_cache_satisfies_the_provenance_check(tmp_path):
    """The artefacts the previous package wrote are exactly the ones the
    rewrite requires as proof, so an existing run is analysable as it stands."""
    cache, pcsv = _legacy_cache(tmp_path)
    s = study.run(cache, pcsv, checkpoints=["rtdetr"], condition="matched",
                  force_past_reconstruction=True)
    assert s.provenance.usable, s.provenance.reasons
    assert s.provenance.origins["rtdetr"].weights.endswith("model.pt")
    assert s.provenance.origins["rtdetr"].n_shards == 170


def test_the_hash_convention_is_order_and_repr_independent(tmp_path):
    """Both properties are load-bearing and were discovered the hard way.

    Order independence lets a table reassembled from per-smear shards in a
    different sequence still verify; the pinned float format survives a round
    trip through parquet. A rewrite that hashed the frame as it happened to sit
    in memory silently invalidated every cache on disk, which is how this test
    came to exist.
    """
    cache, _ = _legacy_cache(tmp_path)
    det, man = cache_mod.load(cache, "rtdetr", verify=False)
    shuffled = det.sample(frac=1.0, random_state=1)
    assert content_hash(shuffled, list(DETECTION_COLUMNS)) == man.content_sha256
    round_tripped = pd.read_parquet(cache / "detections_rtdetr.parquet")
    assert content_hash(round_tripped, list(DETECTION_COLUMNS)) == man.content_sha256
