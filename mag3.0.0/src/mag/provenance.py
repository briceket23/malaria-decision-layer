"""Chain of custody. No result leaves this package without one.

The predecessor could write a `results.tex` from any table of numbers. It did,
from a rehearsal cohort, and two external reviewers then read the resulting PDF
as a scientific report and spent most of their effort reconciling
contradictions that existed only in that file. The artefact was
indistinguishable from a finished paper, and no amount of care downstream can
fix an artefact that cannot say what it is.

So a `Provenance` is required to write results, and it is `usable` only when
the cache carries the two things a pass over the archive leaves behind and
nothing else does:

  * `checkpoint_meta_<name>.json` --- the metadata read out of the `.pt` file
    itself, whose recorded SHA-256 must equal the manifest's; and
  * `_shards/<name>/*.parquet` --- the per-smear tables written as inference
    proceeded.

A manifest can carry any string in its checkpoint field. Those two artefacts
are produced only by `detect.run`, which is the point.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class CheckpointOrigin:
    name: str
    sha256: str
    weights: str
    n_detections: int
    n_images: int
    n_shards: int
    #: the hash of the detection table itself, so that two runs on different
    #: caches cannot produce the same provenance line
    content_sha256: str = ""


#: The share of the cohort that must be covered by non-empty inference shards.
#: A shard per smear is what a real pass leaves; a handful of stubs is what a
#: forgery leaves, and the two were indistinguishable while shards were counted
#: by filename.
SHARD_COVERAGE = 0.5


@dataclass
class Provenance:
    cache: str = ""
    image_set: str = ""
    post: str = ""
    origins: dict = field(default_factory=dict)      # name -> CheckpointOrigin
    patients_hash: str = ""
    n_patients: int = 0
    version: str = ""
    build: str = ""
    excluded: dict = field(default_factory=dict)     # name -> why it was refused
    reasons: list = field(default_factory=list)      # why this is not usable

    @property
    def usable(self) -> bool:
        return not self.reasons

    def line(self) -> str:
        """The stamp the manuscript prints on its title page.

        It carries the patient-table hash and the per-checkpoint content
        hashes, because it is also what `figures.check` compares. Without them
        two runs on DIFFERENT patient tables produced identical lines -- 23
        patients relabelled, a different oracle bound -- and the check that
        exists to catch exactly that passed.
        """
        names = ", ".join(sorted(self.origins))
        digests = "+".join(
            (self.origins[k].content_sha256 or "")[:8]
            for k in sorted(self.origins)) or "none"
        return (f"cache {self.cache}/{self.image_set}, mag {self.version} "
                f"build {self.build}, post-processing {self.post}, "
                f"{len(self.origins)} checkpoints ({names}) [{digests}], "
                f"{self.n_patients} patients [{self.patients_hash[:8]}]")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["origins"] = {k: asdict(v) if hasattr(v, "__dataclass_fields__") else v
                        for k, v in self.origins.items()}
        d["usable"] = self.usable
        return d

    def write(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.as_dict(), indent=1, default=str),
                        encoding="utf-8")
        return path

    def require_usable(self) -> None:
        from .errors import ProvenanceError

        if self.usable:
            return
        raise ProvenanceError(
            "these numbers cannot be traced to a run on the released archive:\n"
            + "\n".join(f"  - {r}" for r in self.reasons),
            remedy=("This is the guarantee working, not a malfunction. Produce "
                    "a cache with `mag detect`, then re-run the analysis. "
                    "There is no flag that writes results without it."))


def inspect(cache_dir, image_set: str, manifests: dict, patients_hash: str,
            n_patients: int, post: str, excluded=None) -> Provenance:
    """Build the record from what is actually on disk beside the cache."""
    from . import __version__, build_id

    root = Path(cache_dir)
    p = Provenance(cache=root.name, image_set=str(image_set), post=str(post),
                   patients_hash=str(patients_hash or ""),
                   n_patients=int(n_patients or 0),
                   version=__version__, build=build_id(),
                   excluded=dict(excluded or {}))
    reasons = []
    if not manifests:
        reasons.append("no checkpoint was analysed")

    for name, man in (manifests or {}).items():
        n_det = int(getattr(man, "n_detections", 0) or 0)
        n_img = int(getattr(man, "n_images", 0) or 0)
        sha = str(getattr(man, "checkpoint_sha256", "") or "")
        if n_det <= 0:
            reasons.append(f"{name}: the cache holds no detections")
        if n_img <= 0:
            reasons.append(f"{name}: the cache covers no image")

        meta_path = root / f"checkpoint_meta_{name}.json"
        weights = ""
        if not meta_path.is_file():
            reasons.append(f"{name}: no {meta_path.name} beside the cache, so "
                           "no weights file was ever opened for it")
        else:
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                reasons.append(f"{name}: {meta_path.name} is unreadable ({e})")
                meta = {}
            weights = str(meta.get("file", ""))
            if not weights or not meta.get("sha256"):
                reasons.append(f"{name}: the checkpoint metadata names no "
                               "weights file and no hash")
            elif not sha:
                # `elif sha and ...` skipped the cross-check silently when the
                # manifest's hash was empty, and recorded NO reason -- so a
                # cache whose manifest simply omitted the checkpoint hash was
                # declared usable and wrote results.
                reasons.append(
                    f"{name}: the manifest records no checkpoint hash, so the "
                    "metadata beside it cannot be cross-checked against the "
                    "detections it claims to describe")
            elif meta["sha256"] != sha:
                reasons.append(
                    f"{name}: the manifest records checkpoint {sha[:12]} and "
                    f"the metadata {str(meta['sha256'])[:12]}; these "
                    "detections and that checkpoint are not the same run")

        # Shards were counted by filename alone, so `touch anything.parquet`
        # satisfied the one artefact only a real inference pass leaves. They
        # are now required to hold bytes, and the count is checked against the
        # cohort: a single stub beside a 170-patient table is not a pass.
        shard_dir = root / "_shards" / name
        shards = ([q for q in shard_dir.glob("*.parquet") if q.stat().st_size > 0]
                  if shard_dir.is_dir() else [])
        n_shards = len(shards)
        if n_shards <= 0:
            reasons.append(f"{name}: no non-empty inference shards under "
                           f"_shards/{name}, so this table was assembled "
                           "rather than inferred")
        elif n_patients and n_shards < max(int(n_patients) * SHARD_COVERAGE, 1):
            reasons.append(
                f"{name}: {n_shards} inference shards for {n_patients} "
                f"patients is under the {SHARD_COVERAGE:.0%} this package "
                "requires as evidence that a pass actually ran")

        p.origins[name] = CheckpointOrigin(
            name=name, sha256=sha, weights=weights, n_detections=n_det,
            n_images=n_img, n_shards=n_shards,
            content_sha256=str(getattr(man, "content_sha256", "") or ""))

    if not p.patients_hash:
        reasons.append("the patient table carries no content hash")
    if p.n_patients <= 0:
        reasons.append("no patients were analysed")
    p.reasons = reasons
    return p
