"""Turn the released archive into the two tables the analysis contracts define.

This module is what the reconnaissance gate was for. The four questions it was
supposed to answer are answered by the archive's own manifest, `image_list.xlsx`:

  * filenames        -- UUID.jpg, one directory per smear
  * patient encoding -- `smear_name` is the patient; the image path is
                        images/<set>/<smear_name>/<image_name>
  * test mapping     -- `set_name` partitions the archive, and the test rows
                        carry `origin` (centre) and `species` (the reference
                        standard) for all 170 smears
  * weights          -- models/models/{rt-detr,yolov5,yolov8}

It also answers a question the manuscript could not close by argument. The
source study's "zoom correction" is not an in-place ambiguity to be inferred:
the archive ships `test` and `test-zoom` as two label trees over the same
smears. Counting annotations in both measures the transformation directly, which
is what `transform_retention()` does.

Nothing here needs a GPU. Run it before inference, not after.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .errors import BadInput, MissingInput
from .schema import (BLOOD_CLASSES, CENTRES, PARASITE_CLASSES, UNINFECTED,
                     validate_patients)

#: Image extensions, matched case-insensitively. `rglob("*.jpg")` is
#: case-SENSITIVE on POSIX and case-INSENSITIVE on Windows, so a single
#: `.JPG` used to make the archive audit differently on the two platforms and
#: moved the measured transform retention from 0.75 to 0.94.
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"})


def iter_images(root: Path):
    return (q for q in Path(root).rglob("*")
            if q.is_file() and q.suffix.lower() in IMAGE_SUFFIXES)


def iter_labels(root: Path):
    return (q for q in Path(root).rglob("*")
            if q.is_file() and q.suffix.lower() == ".txt")


#: The archive's species vocabulary mapped onto the analysis label space.
#: Written out rather than derived, because a silent mismatch here would
#: mislabel patients and every downstream number with them.
SPECIES_TO_LABEL = {
    "Uninfected": UNINFECTED,
    "P. falciparum": "P_falciparum",
    "P. ovale": "P_ovale",
    "P. malariae": "P_malariae",
    "P. vivax": "P_vivax",
    "Babesia": "Babesia_divergens",
    "Trypanosoma brucei": "T_brucei",
}

#: `origin` spells one centre differently from CENTRES.
ORIGIN_TO_CENTRE = {c: c for c in CENTRES} | {"Saint Louis": "Saint-Louis"}

#: classes.txt, in file order. The index is the YOLO class id.
CLASSES_TXT = ("WBC", "RBC", "Platelets", "P. falciparum", "P. ovale",
               "P. malariae", "P. vivax", "Babesia", "Trypanosoma brucei")

CLASS_TO_SCHEMA = {
    "WBC": "WBC", "RBC": "RBC", "Platelets": "platelet",
    "P. falciparum": "P_falciparum", "P. ovale": "P_ovale",
    "P. malariae": "P_malariae", "P. vivax": "P_vivax",
    "Babesia": "Babesia_divergens", "Trypanosoma brucei": "T_brucei",
}


# --------------------------------------------------------------------------- #
# The archive manifest
# --------------------------------------------------------------------------- #

def read_image_list(path: Path) -> pd.DataFrame:
    """Read image_list.xlsx (or a CSV export of it) and check its shape."""
    path = Path(path)
    if not path.is_file():
        raise MissingInput(
            f"no image list at {path}",
            remedy=("This is the archive's own `image_list.xlsx`. It carries "
                    "the patient grouping and the reference standard, and "
                    "nothing in this pipeline works without it."))
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(path, sep="\t")
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        try:
            df = pd.read_excel(path)
        except ImportError as exc:
            # pandas raises a bare ImportError naming the engine, which reads
            # as a broken install rather than a missing optional dependency
            raise MissingInput(
                f"reading {path.name} needs a spreadsheet engine ({exc})",
                remedy=("Install one: `pip install openpyxl`. Or export the "
                        "sheet to CSV and pass that instead -- this reads "
                        "either, and the CSV path needs nothing extra."),
                path=str(path)) from exc
    need = {"set_name", "origin", "species", "smear_name", "image_name", "image_path"}
    missing = need - set(df.columns)
    if missing:
        raise BadInput(
            f"the image list is missing columns: {sorted(missing)}",
            remedy=("This should be the archive's own `image_list.xlsx`, "
                    "unmodified. If you exported it to CSV, export the sheet "
                    "that carries the per-image rows and keep its header."),
            found=sorted(df.columns), needed=sorted(need), path=str(path))
    for col, table in (("species", SPECIES_TO_LABEL), ("origin", ORIGIN_TO_CENTRE)):
        blank = df[col].isna()
        if blank.any():
            raise BadInput(
                f"{int(blank.sum())} rows have an empty '{col}' cell "
                f"(first at spreadsheet row {int(blank.idxmax()) + 2})",
                remedy=("Every row needs a centre and a species: they become "
                        "the patient's centre and reference diagnosis. Fix the "
                        "manifest rather than letting the rows through -- a "
                        "patient with no diagnosis silently changes the "
                        "denominator of every endpoint."),
                column=col, path=str(path))
        unknown = set(df[col]) - set(table)
        if unknown:
            raise BadInput(
                f"unknown {col} in the image list: "
                f"{sorted(map(repr, unknown))}",
                remedy=(f"The {col} vocabulary is written out in ingest.py so "
                        "that a spelling the archive has never used is a "
                        "refusal rather than a new category appearing quietly "
                        "in a table. If the archive really does use this "
                        "value, add it there deliberately."),
                known=sorted(table), path=str(path))
    return df


def read_classes(path: Path) -> tuple[str, ...]:
    """classes.txt, verified against the vocabulary this package expects.

    A silently reordered class file would relabel every detection, so the order
    is checked rather than trusted.
    """
    lines = tuple(l.strip() for l in
              Path(path).read_text(encoding="utf-8-sig").splitlines() if l.strip())
    if lines != CLASSES_TXT:
        raise BadInput(
            "classes.txt does not match the expected order",
            remedy=("The class order fixes the index map used to read every "
                    "detection. A reordered file would relabel every box "
                    "without any other symptom, so re-map CLASS_TO_SCHEMA "
                    "deliberately before continuing."),
            expected=list(CLASSES_TXT), found=list(lines), path=str(path))
    return lines


def build_patients(image_list: pd.DataFrame, set_name: str = "test") -> pd.DataFrame:
    """One row per smear: the patient table the analysis contract requires."""
    sub = image_list[image_list["set_name"] == set_name]
    if sub.empty:
        raise BadInput(
            f"the image list has no rows for set_name={set_name!r}",
            remedy=("Pass the partition the released manifest actually uses. "
                    "`mag ingest --list-sets` prints them."),
            available=sorted(map(str, image_list["set_name"].unique())))
    g = (sub.groupby("smear_name")
            .agg(centre=("origin", "first"), species=("species", "first"),
                 n_images=("image_name", "size"),
                 n_centres=("origin", "nunique"), n_species=("species", "nunique"))
            .reset_index())
    bad = g[(g.n_centres > 1) | (g.n_species > 1)]
    if len(bad):
        raise BadInput(
            f"{len(bad)} smears carry more than one centre or species",
            remedy=("One smear is one patient, and a patient has one centre "
                    "and one reference diagnosis. Resolve the manifest rows "
                    "for these smears before building a cohort from them."),
            smears=[str(x) for x in bad["smear_name"].tolist()[:10]])
    out = pd.DataFrame({
        "patient_id": g["smear_name"].astype(str),
        "centre": g["centre"].map(ORIGIN_TO_CENTRE),
        "truth": g["species"].map(SPECIES_TO_LABEL),
        "n_images": g["n_images"].astype("int64"),
    }).sort_values("patient_id").reset_index(drop=True)
    return validate_patients(out)


def cohort_table(patients: pd.DataFrame) -> pd.DataFrame:
    """Centre x species, in the manuscript's Table 1 layout."""
    ct = pd.crosstab(patients["centre"], patients["truth"],
                     margins=True, margins_name="Total")
    cols = [c for c in (UNINFECTED,) + PARASITE_CLASSES if c in ct.columns] + ["Total"]
    rows = [c for c in CENTRES if c in ct.index] + ["Total"]
    return ct.reindex(index=rows, columns=cols).fillna(0).astype(int)


# --------------------------------------------------------------------------- #
# The label trees
# --------------------------------------------------------------------------- #

def read_yolo_labels(archive: Path, set_name: str = "test") -> pd.DataFrame:
    """Every ground-truth box in labels/<set_name>, as a long table.

    YOLO format: one box per line, `cls cx cy w h`, all normalised. An empty
    file is a field with no annotation and is kept as zero rows, not skipped:
    the count of annotated fields is itself a reconciliation quantity.
    """
    root = Path(archive) / "labels" / set_name
    if not root.is_dir():
        raise MissingInput(
            f"no label tree at {root}",
            remedy=("The ground-truth boxes live at "
                    "<archive>/labels/<set_name>/<smear>/<field>.txt. This "
                    "reads them for the annotation census and the transform "
                    "measurement; neither is needed for the main analysis, so "
                    "you can proceed without them."),
            looked_in=str(root))
    rows, n_files, n_empty = [], 0, 0
    for txt in sorted(iter_labels(root)):
        n_files += 1
        pid = txt.parent.name if txt.parent != root else ""
        body = txt.read_text(encoding="utf-8-sig", errors="replace").strip()
        if not body:
            n_empty += 1
            continue
        for line in body.splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            k = int(float(parts[0]))
            if not 0 <= k < len(CLASSES_TXT):
                # A negative id would index from the end of the tuple and
                # silently relabel the box as T. brucei.
                raise BadInput(
                    f"{txt.name}: class id {k} is outside "
                    f"0..{len(CLASSES_TXT) - 1}",
                    remedy=("This label file does not belong to this class "
                            "vocabulary. Check classes.txt against the label "
                            "tree before reading either."),
                    path=str(txt))
            rows.append((pid, txt.stem, CLASS_TO_SCHEMA[CLASSES_TXT[k]],
                         float(parts[1]), float(parts[2]),
                         float(parts[3]), float(parts[4])))
    det = pd.DataFrame(rows, columns=["patient_id", "image_id", "cls",
                                      "x", "y", "w", "h"])
    det.attrs["n_label_files"] = n_files
    det.attrs["n_empty_files"] = n_empty
    det.attrs["set_name"] = set_name
    return det


def annotation_census(labels: pd.DataFrame) -> pd.DataFrame:
    """Annotations by class, with the parasite subtotal separated.

    The parasite subtotal is the quantity the manuscript reconciles: erythrocyte
    and leucocyte boxes dominate the raw count and would swamp any change in the
    classes the study is about.
    """
    c = labels["cls"].value_counts()
    rows = [{"cls": k, "n": int(c.get(k, 0)),
             "kind": "parasite" if k in PARASITE_CLASSES else "nuisance"}
            for k in list(PARASITE_CLASSES) + list(BLOOD_CLASSES)]
    df = pd.DataFrame(rows)
    return df.sort_values(["kind", "cls"]).reset_index(drop=True)


def transform_retention(archive: Path, base: str = "test",
                        zoomed: str = "test-zoom") -> dict:
    """Measure the source study's geometric transform instead of inferring it.

    The manuscript could not decide between an anisotropic resize, which
    discards nothing, and a crop, which for a 4:3 field would retain about 75%
    of a uniformly distributed annotation set. The archive settles it: both
    label trees are shipped, so the retained fraction is a count, not an
    argument.

    Returned quantities map onto the manuscript's reconciliation table:
        annRecovered -> parasite annotations in `base`
        annAfterTf   -> parasite annotations in `zoomed`
        tfKept       -> their ratio
    """
    a = read_yolo_labels(archive, base)
    b = read_yolo_labels(archive, zoomed)
    pa = a[a["cls"].isin(PARASITE_CLASSES)]
    pb = b[b["cls"].isin(PARASITE_CLASSES)]
    kept = len(pb) / len(pa) if len(pa) else float("nan")
    all_kept = len(b) / len(a) if len(a) else float("nan")
    n_oa, n_ob = len(a) - len(pa), len(b) - len(pb)
    other_kept = n_ob / n_oa if n_oa else float("nan")

    # Which reading of the transform is consistent with the counts? Note the
    # plural: the class-differential retention carries information the total
    # alone does not. On the real archive, all boxes survive at 76.6% -- the
    # crop figure -- while parasite boxes survive at 87.2%. A uniform crop
    # cannot do that; a crop of the field MARGINS can, when parasites sit
    # nearer the centre than erythrocytes. And they do, mechanically: the
    # operator centres the object of interest before capturing the field.
    if len(pa) == 0:
        verdict = "no annotations in the base set; nothing to compare"
    elif kept > 0.995 and all_kept > 0.995:
        verdict = ("consistent with an anisotropic resize: essentially every "
                   "annotation survives, so no region was discarded")
    elif 0.70 <= all_kept <= 0.80 and kept >= other_kept + 0.05:
        verdict = ("consistent with a 4:3 to 16:9 crop of the field margins: "
                   "all-box retention sits at the 75% a uniform crop preserves, "
                   "while parasite boxes survive preferentially, as expected "
                   "when parasites are centred by the operator")
    elif 0.70 <= kept <= 0.80:
        verdict = ("consistent with a 4:3 to 16:9 crop: the retained fraction "
                   "is close to the 75% of area such a crop preserves")
    else:
        verdict = ("consistent with neither a pure resize nor a uniform 4:3 to "
                   "16:9 crop; the retained fraction is stated and the "
                   "mechanism left open")

    return {
        "base_set": base, "zoom_set": zoomed,
        "fields_base": int(a.attrs["n_label_files"]),
        "fields_zoom": int(b.attrs["n_label_files"]),
        "empty_fields_base": int(a.attrs["n_empty_files"]),
        "empty_fields_zoom": int(b.attrs["n_empty_files"]),
        "boxes_base": int(len(a)), "boxes_zoom": int(len(b)),
        "parasite_base": int(len(pa)), "parasite_zoom": int(len(pb)),
        "parasite_kept": float(kept),
        "all_kept": float(all_kept),
        "other_base": int(n_oa), "other_zoom": int(n_ob),
        "other_kept": float(other_kept),
        "smears_base": int(a["patient_id"].nunique()),
        "smears_zoom": int(b["patient_id"].nunique()),
        "verdict": verdict,
        "by_class": (pd.concat(
            [pa["cls"].value_counts().rename("base"),
             pb["cls"].value_counts().rename("zoom")], axis=1)
            .fillna(0).astype(int).assign(
                kept=lambda d: (d["zoom"] / d["base"].replace(0, pd.NA)).astype(float))
            .to_dict("index")),
    }


def check_layout(archive: Path, image_list: pd.DataFrame,
                 set_name: str = "test") -> dict:
    """Does the archive on disk match the manifest? Report, do not assume."""
    archive = Path(archive)
    sub = image_list[image_list["set_name"] == set_name]
    def norm(s: str) -> str:
        # An image_list written on Windows carries backslashes; the disk side
        # was normalised and the manifest side was not, so every row counted as
        # both missing and unlisted.
        return str(s).replace("\\", "/").lstrip("./").casefold()

    listed = {norm(v) for v in sub["image_path"].astype(str)}
    on_disk = {norm(p.relative_to(archive).as_posix())
               for p in iter_images(archive / "images" / set_name)}
    labels_dir = archive / "labels" / set_name
    n_labels = sum(1 for _ in iter_labels(labels_dir)) if labels_dir.is_dir() else 0
    return {
        "listed_images": len(listed),
        "images_on_disk": len(on_disk),
        "listed_but_missing": sorted(listed - on_disk)[:20],
        "on_disk_but_unlisted": sorted(on_disk - listed)[:20],
        "n_listed_but_missing": len(listed - on_disk),
        "n_on_disk_but_unlisted": len(on_disk - listed),
        "label_files": n_labels,
        "labels_match_images": n_labels == len(on_disk),
    }
