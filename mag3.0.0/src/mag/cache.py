"""Reading and writing the detection cache.

The on-disk format is deliberately unchanged from the predecessor's. Six GPU
passes over 4,508 fields cost about fifty minutes on a 6 GB card, and a rewrite
that made an operator repeat them to gain nothing but tidier code would be a
bad trade. What has changed is that nothing here will invent a cache, supply a
default path, or return a partial table as though it were whole.

Two design points carry over because they were paid for:

  * caching happens BEFORE non-maximum suppression, at a 0.001 floor, keeping
    the full per-class score vector. Class-agnostic suppression deletes the
    losing species box when two species compete for one parasite, which
    destroys the distribution the aggregation rules are about;
  * the manifest records the content hash of the table it describes, and the
    hash is checked on load. A stale cache read as fresh is the failure mode
    that produces results nobody can reproduce.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from .errors import IntegrityError, MagError, MissingInput
from .schema import DETECTION_COLUMNS, content_hash, validate_detections


@dataclass
class Manifest:
    checkpoint: str
    checkpoint_sha256: str
    conf_floor: float
    nms: str                      # "none" | "per_class" | "class_agnostic"
    iou: float | None
    imgsz: int
    n_images: int
    n_detections: int
    content_sha256: str
    software: dict = field(default_factory=dict)
    seed: int = 0
    created_utc: str = ""

    @classmethod
    def load(cls, path) -> "Manifest":
        path = Path(path)
        try:
            d = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:                        # noqa: BLE001
            # a torn JSON reached the operator as a JSONDecodeError traceback,
            # and `mag doctor` -- whose whole job is to report what is wrong --
            # crashed on it rather than listing it as a blocking problem
            raise IntegrityError(
                f"the manifest {path.name} could not be read "
                f"({type(exc).__name__}: {exc})",
                remedy=("The manifest is truncated or corrupt, usually a run "
                        "interrupted mid-write. Re-run detection for this "
                        "checkpoint."),
                path=str(path)) from exc
        known = {f for f in cls.__dataclass_fields__}
        try:
            return cls(**{k: v for k, v in d.items() if k in known})
        except TypeError as exc:
            # a manifest missing required fields raised a bare TypeError from
            # the dataclass constructor, which named the fields but not the file
            raise IntegrityError(
                f"the manifest {path.name} is missing required fields ({exc})",
                remedy=("This manifest was not written by this pipeline, or "
                        "was written by a version that predates a field it "
                        "now requires. Re-run detection for this checkpoint."),
                path=str(path), found=sorted(d)) from exc

    def write(self, path) -> Path:
        """Atomic. An interrupted write left a torn JSON that no reader could
        parse, while the table beside it was written atomically."""
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        os.replace(tmp, path)
        return path


def software_versions() -> dict:
    import sys

    out = {"python": sys.version.split()[0]}
    for mod in ("numpy", "pandas", "pyarrow", "scipy", "sklearn", "torch",
                "ultralytics"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "present")
        except Exception:
            pass
    return out


def _write_frame(df: pd.DataFrame, path: Path) -> Path:
    """Atomic, and honest about the fallback.

    A failed parquet write used to leave the previous parquet in place, and the
    manifest was then built from a table nobody had just written. Here the
    temporary file is renamed over the target only on success, and a fallback
    to CSV removes the stale parquet rather than leaving two candidates.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return path
    except Exception:
        tmp.unlink(missing_ok=True)
        csv = path.with_suffix(".csv")
        tmp_csv = csv.with_suffix(".csv.tmp")
        df.to_csv(tmp_csv, index=False, lineterminator="\n")
        os.replace(tmp_csv, csv)
        if path.exists():
            path.unlink()
        return csv


def _read_frame(path: Path) -> pd.DataFrame:
    """Read a cached table, or refuse it by name.

    `float_precision="round_trip"` is load-bearing on the CSV path. Pandas'
    default parser is off by one unit in the last place on many values, and for
    some of them that changes the tenth significant digit -- exactly the
    precision `schema.content_hash` pins. Every CSV-fallback cache therefore
    failed its own integrity check on the very next load, and the refusal told
    the operator the table had been tampered with and to re-run detection: an
    hour of GPU time, reproducing the same failure.

    A truncated file raises a third-party exception -- `ArrowInvalid`, a
    `ParserError` -- which reached the operator as a traceback. It is still a
    failure, but it is one this package names.
    """
    path = Path(path)
    try:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path, float_precision="round_trip",
                           dtype={"patient_id": "string", "image_id": "string",
                                  "checkpoint": "string", "cls": "string"})
    except MagError:
        raise
    except Exception as exc:                            # noqa: BLE001
        raise IntegrityError(
            f"the detection table at {path.name} could not be read "
            f"({type(exc).__name__}: {exc})",
            remedy=("The file is truncated or corrupt -- usually an inference "
                    "run that was interrupted mid-write. Delete it and re-run "
                    "detection for this checkpoint; the other checkpoints in "
                    "the cache are unaffected."),
            path=str(path),
            size_bytes=path.stat().st_size if path.exists() else 0) from exc


def detections_path(cache_dir, checkpoint: str) -> Path:
    """Where a checkpoint's table is, whichever format it landed in."""
    root = Path(cache_dir)
    for suffix in (".parquet", ".csv"):
        p = root / f"detections_{checkpoint}{suffix}"
        if p.exists():
            return p
    raise MissingInput(
        f"no detection table for '{checkpoint}' in {root}",
        remedy=(f"Run `mag detect --checkpoint {checkpoint}` first, or name a "
                "cache directory that already holds one. This package will "
                "not proceed without measured detections."),
        looked_for=[f"detections_{checkpoint}.parquet",
                    f"detections_{checkpoint}.csv"])


def save(detections: pd.DataFrame, cache_dir, checkpoint: str, *,
         checkpoint_sha: str, conf_floor: float, imgsz: int,
         nms: str = "none", iou: float | None = None, seed: int = 0) -> Manifest:
    root = Path(cache_dir)
    root.mkdir(parents=True, exist_ok=True)
    det = validate_detections(detections)
    written = _write_frame(det, root / f"detections_{checkpoint}.parquet")
    # Hash what is ON DISK, not what is in memory. A round trip through parquet
    # -- and especially through the CSV fallback -- can change a float's last
    # place, and a manifest that disagrees with the file beside it is worse
    # than no manifest: the next `load` refuses the cache as tampered with and
    # tells the operator to re-run detection, which is an hour of GPU time
    # spent reproducing the same mismatch.
    on_disk = validate_detections(_read_frame(written))
    man = Manifest(
        checkpoint=checkpoint, checkpoint_sha256=checkpoint_sha,
        conf_floor=float(conf_floor), nms=nms, iou=iou, imgsz=int(imgsz),
        n_images=int(on_disk["image_id"].nunique()),
        n_detections=int(len(on_disk)),
        content_sha256=content_hash(on_disk, list(DETECTION_COLUMNS)),
        software=software_versions(), seed=int(seed),
        created_utc=_dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"))
    man.write(root / f"manifest_{checkpoint}.json")
    return man


def load(cache_dir, checkpoint: str, *, verify: bool = True):
    """Return (detections, manifest), refusing a table that has drifted."""
    root = Path(cache_dir)
    path = detections_path(root, checkpoint)
    man_path = root / f"manifest_{checkpoint}.json"
    if not man_path.exists():
        raise MissingInput(
            f"'{checkpoint}' has a detection table but no manifest in {root}",
            remedy=("A table without a manifest cannot be checked against the "
                    "checkpoint it came from. Re-run detection for it."),
            table=str(path))
    man = Manifest.load(man_path)
    det = validate_detections(_read_frame(path))
    if verify:
        got = content_hash(det, list(DETECTION_COLUMNS))
        if got != man.content_sha256:
            raise IntegrityError(
                f"cache integrity failure for '{checkpoint}': the manifest "
                f"records {man.content_sha256[:12]} and the table hashes to "
                f"{got[:12]}",
                remedy=("The table and its manifest describe different data. "
                        "The usual causes, in order of likelihood: the file "
                        "was written by a different run; it was round-tripped "
                        "through a tool that narrowed a float column to 32 "
                        "bits (check the dtypes below -- score, x, y, w and h "
                        "must be float64); or it was edited by hand. Re-run "
                        "detection rather than analysing it."),
                table=str(path), rows=len(det),
                dtypes={c: str(det[c].dtype) for c in det.columns})
    return det, man


def available(cache_dir) -> list[str]:
    """Checkpoint names this cache can actually analyse.

    Enumerated from the MANIFESTS, requiring a matching table -- not the other
    way round. Globbing the tables meant that a stray copy left beside them,
    `detections_rtdetr.backup-2024.parquet`, was announced as a checkpoint
    called `rtdetr.backup-2024`, whose missing manifest then aborted the whole
    analysis. Refusing an incomplete pair is right; refusing because of a file
    nobody asked to analyse is not.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return []
    names = []
    for man in sorted(root.glob("manifest_*.json")):
        name = man.stem[len("manifest_"):]
        if any((root / f"detections_{name}{sfx}").exists()
               for sfx in (".parquet", ".csv")):
            names.append(name)
    return sorted(names)


def orphans(cache_dir) -> dict:
    """Files in the cache that look like a checkpoint but cannot be analysed.

    Reported by `mag doctor` rather than acted on. A half-written pair is worth
    seeing; it is not worth stopping for.
    """
    root = Path(cache_dir)
    if not root.is_dir():
        return {}
    tables, manifests = set(), set()
    for pattern in ("detections_*.parquet", "detections_*.csv"):
        for p in root.glob(pattern):
            tables.add(p.stem[len("detections_"):])
    for p in root.glob("manifest_*.json"):
        manifests.add(p.stem[len("manifest_"):])
    return {"table_without_manifest": sorted(tables - manifests),
            "manifest_without_table": sorted(manifests - tables)}
