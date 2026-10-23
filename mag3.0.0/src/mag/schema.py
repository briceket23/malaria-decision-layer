"""The vocabulary, and the checks that keep it honest.

Everything that crosses a module boundary in this package is one of the frames
defined here, validated on the way in. The validation is not defensive
programming for its own sake: two of the predecessor's confirmed bugs were a
patient with no prediction being scored as a false positive, and a zero-padded
identifier read back as an integer so that a join silently matched nothing.
Both were type errors that a schema check catches at the boundary.
"""

from __future__ import annotations

import hashlib

import pandas as pd

UNINFECTED = "uninfected"

#: The six parasite classes the detectors emit, in the order the released
#: `classes.txt` declares them. Order matters twice: it fixes the class index
#: map when reading a checkpoint, and it is the declared tie order for argmax.
PARASITE_CLASSES = ("P_falciparum", "P_ovale", "P_malariae", "P_vivax",
                    "Babesia_divergens", "T_brucei")

#: Non-parasite classes. RBC is load-bearing: the density rules divide by it.
BLOOD_CLASSES = ("WBC", "RBC", "platelet")

ALL_CLASSES = BLOOD_CLASSES + PARASITE_CLASSES
LABELS = (UNINFECTED,) + PARASITE_CLASSES

#: The source study's five-way grouping, kept so that the baseline reproduces
#: the published number. Babesia and Trypanosoma are not Plasmodium and are not
#: pooled with it; they are their own labels in the six-way analysis.
GROUPING_5 = {"Babesia_divergens": "other", "T_brucei": "other"}

DETECTION_COLUMNS = ("patient_id", "image_id", "checkpoint", "cls", "score",
                     "x", "y", "w", "h")
PATIENT_COLUMNS = ("patient_id", "centre", "truth", "n_images")

#: Every centre the released archive draws from. Written out rather than
#: inferred from whatever the manifest happens to contain, so that a spelling
#: the archive has never used is a refusal at ingest rather than a seventh
#: centre appearing quietly in a table. `ingest` maps the manifest's `origin`
#: column onto this vocabulary.
CENTRES = ("Montpellier", "Toulouse", "Lille", "Nantes", "Rouen", "Saint-Louis")

#: Patients from the two centres that also supplied training material. Used
#: only to report the internal/external split; it cannot rescue specificity,
#: which is confounded with centre by construction (see gates.identifiability).
INTERNAL_CENTRES = ("Montpellier", "Rouen")


def regroup(label: str, grouping: dict | None = None) -> str:
    return (grouping or {}).get(label, label)


def content_hash(df: pd.DataFrame, columns=None) -> str:
    """A hash of a frame's content, independent of machine, order and repr.

    Three things are pinned, and each of them for a reason that was paid for:

      * the line terminator, because `to_csv` resolves an unset one to
        `os.linesep` even when returning a string, so the same table hashed
        differently on Windows and Linux and every cache failed its own
        integrity check when moved between them;
      * the row order, by sorting, so that a table reassembled from per-smear
        shards in a different sequence is still the same table;
      * the float format, because a round trip through parquet or CSV can
        change the last digit of a repr and a manifest that disagrees with its
        own file is worse than no manifest.

    This is byte-for-byte the convention the previous package used. It is not
    a detail: change it and every existing cache fails verification, which on
    this study is fifty minutes of GPU time per image set.

    One known limitation, kept deliberately. `%.10g` truncates, so two tables
    differing only beyond the tenth significant digit hash the same. A
    round-trippable `%.17g` would close that band and would also invalidate
    every cache on disk, so the band stays and is written down here instead:
    this hash detects corruption and mismatched writes, not adversarial
    tampering at the eleventh digit.
    """
    cols = list(columns or df.columns)
    sub = df[cols].sort_values(cols, kind="mergesort").reset_index(drop=True)
    # Canonicalise the dtypes before rendering, so that an integer column read
    # back as a float, or an object column read back as pandas "string", hashes
    # the same. It does NOT make the hash survive a NARROWING: 0.1 in float32
    # is genuinely a different number from 0.1 in float64, and no rendering
    # convention can pretend otherwise. `cache.load` names that as the likely
    # cause when the check fails, rather than telling the operator their table
    # was tampered with.
    for c in cols:
        if pd.api.types.is_float_dtype(sub[c]) or pd.api.types.is_integer_dtype(sub[c]):
            sub[c] = sub[c].astype("float64")
        else:
            sub[c] = sub[c].astype("string")
    payload = sub.to_csv(index=False, float_format="%.10g",
                         lineterminator="\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_columns(df: pd.DataFrame, columns, what: str) -> None:
    from .errors import BadInput

    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise BadInput(
            f"{what} is missing {len(missing)} required column(s): "
            f"{', '.join(missing)}",
            remedy=("This frame did not come from this pipeline, or came from "
                    "an older version of it. Regenerate it rather than adding "
                    "the columns by hand."),
            found=list(df.columns))


def validate_detections(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce a detection table into the schema, or refuse it."""
    from .errors import BadInput

    _require_columns(df, DETECTION_COLUMNS, "the detection table")
    out = df.copy()
    for c in ("patient_id", "image_id", "checkpoint", "cls"):
        out[c] = out[c].astype("string")
    # A missing identifier is not a missing value to be tolerated: `patient_id`
    # decides whose smear this is and `image_id` decides which field, and the
    # aggregation codes an absent one as -1, which is a valid index into the
    # row above. A detection with no image_id was therefore filed under the
    # previous patient's last field, confidence and all.
    for c in ("patient_id", "image_id", "checkpoint", "cls"):
        n_null = int(out[c].isna().sum())
        if n_null:
            raise BadInput(
                f"the detection table has {n_null:,} rows with no {c}",
                remedy=("Every detection must name its patient, its field, the "
                        "checkpoint that produced it and the class it claims. "
                        "Re-run inference rather than filling these in."))
    for c in ("score", "x", "y", "w", "h"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    if out[["score", "x", "y", "w", "h"]].isna().any().any():
        raise BadInput("the detection table holds non-numeric geometry or scores",
                       remedy="Re-run inference for the affected checkpoint.")
    # A negative width or height gives a box of negative extent. `_iou_matrix`
    # clips its area to zero, so such a box neither suppresses nor is
    # suppressed -- it passes straight through the standardising condition
    # whose entire purpose is to make two checkpoints comparable.
    bad_geom = int(((out["w"] <= 0) | (out["h"] <= 0)).sum())
    if bad_geom:
        raise BadInput(
            f"{bad_geom:,} detections have a non-positive width or height",
            remedy=("A box of zero or negative extent is invisible to "
                    "suppression and would survive the matched condition "
                    "unchallenged. Check the geometry conversion in the "
                    "detect stage; xywh must be centre-form and positive."))
    bad = sorted(set(out["cls"].dropna()) - set(ALL_CLASSES))
    if bad:
        raise BadInput(
            f"the detection table names {len(bad)} class(es) this study does "
            f"not define: {', '.join(bad)}",
            remedy=("The class map derived from the checkpoint disagrees with "
                    "classes.txt. Check the map printed by the detect stage "
                    "before spending GPU time on it."),
            known=list(ALL_CLASSES))
    if not out["score"].between(0.0, 1.0).all():
        raise BadInput("detection scores outside [0, 1]",
                       remedy="A confidence that is not a probability means "
                              "the loader read the wrong tensor column.")
    return out[list(DETECTION_COLUMNS)]


def validate_patients(df: pd.DataFrame) -> pd.DataFrame:
    from .errors import BadInput

    _require_columns(df, PATIENT_COLUMNS, "the patient table")
    out = df.copy()
    for c in ("patient_id", "centre", "truth"):
        out[c] = out[c].astype("string")
    out["n_images"] = pd.to_numeric(out["n_images"], errors="coerce").astype("Int64")
    # A patient with no reference diagnosis is not a patient with an unknown
    # one: depending only on the dtype pandas gave the column, `truth != UNINFECTED`
    # either drops them from the specificity denominator or scores them as a
    # false negative. Two different wrong answers for the same input.
    n_null = int(out["truth"].isna().sum())
    if n_null:
        raise BadInput(
            f"{n_null} patients have no reference diagnosis",
            remedy=("Every patient must carry a label. Drop the unlabelled "
                    "patients from the cohort explicitly, so the denominator "
                    "of every endpoint is the set of patients you chose."))
    if out["patient_id"].isna().any():
        raise BadInput("the patient table has rows with no patient_id",
                       remedy="Rebuild it from the released image_list.xlsx.")
    bad_n = out["n_images"].isna() | (out["n_images"] < 0)
    if bool(bad_n.any()):
        raise BadInput(
            f"{int(bad_n.sum())} patients have a missing or negative field count",
            remedy=("n_images is the number of fields examined for that "
                    "patient and is the denominator of every per-field "
                    "statistic. Rebuild it from the released manifest."))
    bad = sorted(set(out["truth"].dropna()) - set(LABELS))
    if bad:
        raise BadInput(f"the patient table holds unknown labels: {', '.join(bad)}",
                       remedy="Rebuild it from the released image_list.xlsx.",
                       known=list(LABELS))
    if out["patient_id"].duplicated().any():
        dup = out.loc[out["patient_id"].duplicated(), "patient_id"].tolist()[:3]
        raise BadInput(f"the patient table repeats {len(dup)}+ identifiers",
                       remedy="One row per patient. Rebuild from the manifest.",
                       first=dup)
    return out[list(PATIENT_COLUMNS)]
