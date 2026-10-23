"""TEST FIXTURES ONLY. Not part of the package, not reachable from the CLI.

These builders make small controlled inputs so the analysis can be exercised
without a GPU and without the 44 GB archive. Three properties are load-bearing
and are themselves tested in `test_no_fabrication.py`:

  * nothing under `src/mag` imports this module, and no command can invoke it,
    so the shipped package cannot produce an observation;
  * every builder writes into a pytest `tmp_path`;
  * `real_looking_cache` deliberately does NOT satisfy the provenance check
    unless a test asks it to, because the whole point is that a fixture is not
    a measurement.

The predecessor shipped a generator as `make dry`. It wrote a results file
shaped exactly like a real one, and two external reviews were then carried out
on the resulting PDF. That is the failure this arrangement prevents.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from mag import cache as cache_mod
from mag.schema import BLOOD_CLASSES, PARASITE_CLASSES, UNINFECTED

#: The released cohort's structure, from the source publication's Table 1.
COHORT = [
    ("Montpellier", UNINFECTED, 54), ("Montpellier", "P_falciparum", 11),
    ("Toulouse", "P_falciparum", 5), ("Toulouse", "P_ovale", 5),
    ("Toulouse", "P_malariae", 5), ("Toulouse", "P_vivax", 5),
    ("Toulouse", "Babesia_divergens", 1), ("Toulouse", "T_brucei", 1),
    ("Lille", "P_falciparum", 9), ("Lille", "P_ovale", 1),
    ("Lille", "P_malariae", 1), ("Lille", "P_vivax", 2),
    ("Lille", "Babesia_divergens", 1), ("Lille", "T_brucei", 1),
    ("Nantes", "P_falciparum", 1),
    ("Rouen", "P_falciparum", 12), ("Rouen", "P_ovale", 12),
    ("Rouen", "P_malariae", 11), ("Rouen", "P_vivax", 10),
    ("Saint-Louis", "P_falciparum", 5), ("Saint-Louis", "P_ovale", 5),
    ("Saint-Louis", "P_malariae", 5), ("Saint-Louis", "P_vivax", 5),
    ("Saint-Louis", "Babesia_divergens", 1), ("Saint-Louis", "T_brucei", 1),
]


def patient_table(n_fields=(20, 31), seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows, pid = [], 0
    for centre, truth, k in COHORT:
        for _ in range(k):
            pid += 1
            rows.append({"patient_id": f"P{pid:04d}", "centre": centre,
                         "truth": truth,
                         "n_images": int(rng.integers(*n_fields))})
    return pd.DataFrame(rows)


def detections(patients: pd.DataFrame, checkpoint: str, *, seed: int = 0,
               spec_bias: float = 0.0) -> pd.DataFrame:
    """Detections with a deliberate, documented structure.

    Infected patients receive true-class boxes; every patient receives blood
    cells and a trickle of low-confidence parasite noise, which is what makes
    a sum-with-no-threshold rule call everyone infected. `spec_bias` raises the
    noise so that two checkpoints differ, which the checkpoint-factor tests
    need.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in patients.itertuples(index=False):
        for f in range(int(r.n_images)):
            image_id = f"{r.patient_id}-{f:03d}"
            for c, n in zip(BLOOD_CLASSES, (2, 40, 3)):
                for _ in range(n):
                    rows.append((r.patient_id, image_id, checkpoint, c,
                                 float(rng.uniform(0.3, 0.99))))
            n_noise = rng.poisson(1.5 + spec_bias)
            for _ in range(n_noise):
                rows.append((r.patient_id, image_id, checkpoint,
                             str(rng.choice(PARASITE_CLASSES)),
                             float(rng.uniform(0.001, 0.20))))
            if r.truth != UNINFECTED:
                for _ in range(rng.integers(2, 7)):
                    rows.append((r.patient_id, image_id, checkpoint, r.truth,
                                 float(rng.uniform(0.55, 0.99))))
    det = pd.DataFrame(rows, columns=["patient_id", "image_id", "checkpoint",
                                      "cls", "score"])
    n = len(det)
    det["x"] = rng.uniform(0.1, 0.9, n)
    det["y"] = rng.uniform(0.1, 0.9, n)
    det["w"] = rng.uniform(0.04, 0.09, n)
    det["h"] = rng.uniform(0.04, 0.09, n)

    # Duplicates. A real detector proposes the same object several times, which
    # is the entire reason suppression exists and the reason the raw and
    # matched conditions differ. A fixture of well-separated boxes would leave
    # the whole matched path untested.
    dup = det.sample(frac=0.35, random_state=int(seed) + 11).copy()
    jitter = rng.normal(0.0, 0.004, size=(len(dup), 2))
    dup["x"] = (dup["x"].to_numpy() + jitter[:, 0]).clip(0.05, 0.95)
    dup["y"] = (dup["y"].to_numpy() + jitter[:, 1]).clip(0.05, 0.95)
    dup["score"] = (dup["score"].to_numpy() * 0.9).clip(0.001, 0.999)
    return pd.concat([det, dup], ignore_index=True)


def write_cache(tmp_path: Path, checkpoints=("rtdetr", "yolov8x"), *,
                seed: int = 0, provenanced: bool = False,
                n_fields=(4, 7)) -> tuple[Path, Path]:
    """A cache directory plus a patients.csv. Returns (cache_dir, patients_csv).

    With `provenanced=False` (the default) the cache lacks the checkpoint
    metadata and shards that only a real pass leaves behind, so `report.write`
    refuses it. Tests that need the writing path ask for `provenanced=True`
    and are, by construction, testing the writer rather than a result.
    """
    cache = Path(tmp_path) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    patients = patient_table(n_fields=n_fields, seed=seed)
    pcsv = cache / "patients.csv"
    patients.to_csv(pcsv, index=False)

    for i, ck in enumerate(checkpoints):
        det = detections(patients, ck, seed=seed + i, spec_bias=0.6 * i)
        man = cache_mod.save(det, cache, ck, checkpoint_sha=f"{i:064d}",
                             conf_floor=0.001, imgsz=640, seed=seed)
        if provenanced:
            (cache / f"checkpoint_meta_{ck}.json").write_text(json.dumps(
                {"file": f"/weights/{ck}/model.pt", "sha256": man.checkpoint_sha256,
                 "names": {}, "format": "fixture"}), encoding="utf-8")
            sd = cache / "_shards" / ck
            sd.mkdir(parents=True, exist_ok=True)
            # one non-empty shard per smear, which is what a real inference
            # pass leaves. Three stubs used to satisfy the provenance check;
            # they no longer do, and they should not have.
            for pid in patients["patient_id"]:
                (sd / f"{pid}.parquet").write_bytes(b"\x00shard")
    return cache, pcsv
