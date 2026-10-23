#!/usr/bin/env python3
"""make_fixture.py --- a small synthetic archive with the real archive's shape.

    python make_fixture.py --out /tmp/fixture

Not a substitute for the released data and not meant to look like it. It exists
so the suite can be run end to end without a GPU or a 30 GB download, and so
the tests exercise the same code paths a real archive does: the same manifest
columns, the same label-tree layout, the same partition names, and --- on
purpose --- the same three pathologies the real archive has, so a check that
fires there fires here too.

Baked in deliberately:

  * every uninfected smear from ONE centre  -> Proposition 5 fails
  * infected fields nearly all carry a parasite box -> enrichment
  * one class GAINS boxes after the transform -> re-annotation
  * two species carried by 3 smears -> no per-class claim
  * two aspect ratios among the images
  * parasites drawn nearer the field centre than erythrocytes

`--healthy` builds the same archive without them, which is what the tests use
to prove each check clears rather than merely firing forever.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

CENTRES = ["Montpellier", "Toulouse", "Lille", "Nantes", "Rouen", "Saint Louis"]
SPECIES = ["Uninfected", "P. falciparum", "P. ovale", "P. malariae",
           "P. vivax", "Babesia", "Trypanosoma brucei"]
CLASSES_TXT = ("WBC", "RBC", "Platelets", "P. falciparum", "P. ovale",
               "P. malariae", "P. vivax", "Babesia", "Trypanosoma brucei")
PARASITE_IDX = {3, 4, 5, 6, 7, 8}
RBC_IDX, WBC_IDX, PLT_IDX = 1, 0, 2

PNG_1x1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8ffff3f0005fe02fea6b8a30d0000000049454e"
    "44ae426082")


def _jpg(path: Path, w: int, h: int):
    """A real JPEG of the requested size, if Pillow is here; else a stub."""
    try:
        from PIL import Image
        Image.new("RGB", (w, h), (200, 180, 190)).save(path, "JPEG", quality=20)
    except ImportError:
        path.write_bytes(PNG_1x1)


def build(out: Path, healthy: bool = False, seed: int = 0):
    rng = random.Random(seed)
    root = out / "Plasmodium-v1.zip" / "Plasmodium-v1"
    for sub in ("images", "labels"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(CLASSES_TXT) + "\n",
                                      encoding="utf-8")

    rows = []
    plan = {"train": 24, "validation": 8, "test": 34}

    def species_for(i, part):
        if part != "test":
            return SPECIES[i % len(SPECIES)]
        # 10 uninfected, then the infected classes, with two species thin
        if i < 10:
            return "Uninfected"
        if i >= 31:                                   # 3 smears
            return "Trypanosoma brucei" if i == 33 else "Babesia"
        return SPECIES[1 + (i % 4)]

    def centre_for(i, part, sp):
        if part != "test":
            return CENTRES[i % len(CENTRES)]
        if sp == "Uninfected":
            # THE pathology: every uninfected smear from one site, unless the
            # fixture is asked for a cohort that can carry a specificity.
            return CENTRES[0] if not healthy else CENTRES[i % 3]
        return CENTRES[1 + (i % 5)]

    for part, n in plan.items():
        for i in range(n):
            sp = species_for(i, part)
            ct = centre_for(i, part, sp)
            # distinct per partition: "train" and "test" both begin with T,
            # and a shared prefix makes smear ids collide across partitions
            # -- which is exactly what the partition_overlap check catches.
            tag = {"train": "TR", "validation": "VA", "test": "TE"}[part]
            smear = f"{tag}{i:04d}-uuid"
            n_fields = rng.randint(4, 9)
            for f in range(n_fields):
                rows.append({"set_name": part, "origin": ct, "species": sp,
                             "smear_name": smear, "image_name": f"f{f:03d}.jpg",
                             "image_path": f"images/{part}/{smear}/f{f:03d}.jpg"})

    import pandas as pd
    df = pd.DataFrame(rows)

    # test-zoom is the SAME smears re-annotated after the transform
    zoom = df[df["set_name"] == "test"].copy()
    zoom["set_name"] = "test-zoom"
    zoom["image_path"] = zoom["image_path"].str.replace("/test/", "/test-zoom/",
                                                        regex=False)
    df = pd.concat([df, zoom], ignore_index=True)
    df.to_excel(root / "image_list.xlsx", index=False)

    def draw_boxes(infected: bool, zoomed: bool, sp_idx: int):
        """One field's worth of YOLO lines."""
        lines = []
        for _ in range(rng.randint(30, 60)):          # erythrocytes, spread out
            y = rng.uniform(0.02, 0.98)
            lines.append((RBC_IDX, rng.uniform(.05, .95), y, 0.09, 0.09))
        for _ in range(rng.randint(0, 3)):
            lines.append((PLT_IDX, rng.uniform(.05, .95),
                          rng.uniform(.02, .98), 0.03, 0.03))
        if rng.random() < 0.3:
            lines.append((WBC_IDX, rng.uniform(.1, .9), rng.uniform(.1, .9),
                          0.16, 0.16))
        if infected:
            # enrichment: an infected field nearly always carries a parasite,
            # and parasites sit NEARER the centre than the erythrocytes
            # enrichment is the pathology: in a randomly-sampled field set
            # most fields of an infected patient carry no parasite at all
            hit = 0.97 if not healthy else 0.10
            n_par = rng.randint(1, 6) if rng.random() < hit else 0
            for _ in range(n_par):
                spread = 0.5 if not healthy else 0.95
                lines.append((sp_idx,
                              0.5 + rng.uniform(-.45, .45),
                              0.5 + rng.uniform(-.45, .45) * spread,
                              0.05, 0.05))
        if zoomed:
            # a pure 4:3 -> 16:9 crop: drop |y - 0.5| > 0.375, rescale the rest
            kept = []
            for c, x, y, w, h in lines:
                if abs(y - 0.5) > 0.375:
                    continue
                kept.append((c, x, 0.5 + (y - 0.5) / 0.75, w, h / 0.75))
            lines = kept
            # ... except Babesia, which GAINS boxes: a crop cannot add one, so
            # these smears were re-annotated rather than merely re-cropped
            if not healthy and sp_idx == 7:
                for _ in range(rng.randint(2, 5)):
                    lines.append((7, rng.uniform(.1, .9), rng.uniform(.1, .9),
                                  0.05, 0.05))
        return lines

    sp_to_idx = {"P. falciparum": 3, "P. ovale": 4, "P. malariae": 5,
                 "P. vivax": 6, "Babesia": 7, "Trypanosoma brucei": 8}

    for part in ("test", "test-zoom"):
        base = "test"
        sub = df[df["set_name"] == part]
        for smear, g in sub.groupby("smear_name"):
            sp = g["species"].iloc[0]
            infected = sp != "Uninfected"
            idx = sp_to_idx.get(sp, 3)
            ldir = root / "labels" / part / smear
            idir = root / "images" / part / smear
            ldir.mkdir(parents=True, exist_ok=True)
            idir.mkdir(parents=True, exist_ok=True)
            for _, r in g.iterrows():
                stem = Path(r["image_name"]).stem
                rng.seed(hash((smear, stem)) & 0xFFFF)   # same field both sets
                lines = draw_boxes(infected, part.endswith("zoom"), idx)
                (ldir / f"{stem}.txt").write_text(
                    "\n".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                              for c, x, y, w, h in lines) + "\n",
                    encoding="utf-8")
                if part == base:
                    # two aspect ratios, as in the real archive
                    w, h = ((3840, 2160) if (hash(smear) % 3) else (2560, 1920))
                    if healthy:
                        w, h = 3840, 2160
                    _jpg(idir / r["image_name"], w // 12, h // 12)
                else:
                    _jpg(idir / r["image_name"], 320, 180)

    for part in ("train", "validation"):
        for smear, g in df[df["set_name"] == part].groupby("smear_name"):
            idir = root / "images" / part / smear
            idir.mkdir(parents=True, exist_ok=True)
            for _, r in g.iterrows():
                _jpg(idir / r["image_name"], 320, 180)

    return root


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="make_fixture.py")
    p.add_argument("--out", default="fixture")
    p.add_argument("--healthy", action="store_true",
                   help="build WITHOUT the pathologies, so checks can clear")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args(argv)
    root = build(Path(a.out), healthy=a.healthy, seed=a.seed)
    print(f"archive at {root}")
    print(f"  {'healthy' if a.healthy else 'with the real archive pathologies'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
