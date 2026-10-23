"""From a detection table to what a rule actually sees.

A rule never touches the detection table. It sees a `PatientView`: for one
patient, the confidences of every detection of every class, grouped by field.
That indirection is what makes the rule registry declarative and what makes the
post-processing condition a property of the view rather than a branch inside
fourteen rules.

Post-processing is a standardising condition, not a third experimental factor.
Detectors of different families do not emit comparable box sets until identical
suppression has been applied: on the released weights RT-DETR returns a fixed
300 boxes per field --- its query budget, the same on a parasite-dense smear as
on a clean one --- while a YOLO head returns several hundred from a dense
anchor grid, most of which duplicate each other. Any statistic that counts,
averages or thresholds boxes measures that difference before it measures the
detector, which is why checkpoint comparisons are reported in the `matched`
condition and why the condition is recorded in the provenance line.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .schema import BLOOD_CLASSES, PARASITE_CLASSES

#: The source study's inference settings, kept as a comparator rather than as
#: the analysis condition: class-agnostic suppression at IoU 0.45 and a 0.25
#: confidence floor.
SOURCE_CONF = 0.25
SOURCE_IOU = 0.45
MATCHED_IOU = 0.45

CONDITIONS = ("raw", "matched", "source")


@dataclass
class PatientView:
    """Everything about one patient that a rule may use."""

    patient_id: str
    n_fields: int
    #: class -> array of confidences, one entry per surviving detection
    scores: dict = field(default_factory=dict)
    #: class -> array of per-field maxima (0.0 where the class is absent)
    per_field_max: dict = field(default_factory=dict)
    #: class -> array of per-field counts
    per_field_count: dict = field(default_factory=dict)

    def total(self, cls: str) -> float:
        s = self.scores.get(cls)
        return float(s.sum()) if s is not None and s.size else 0.0

    def n_boxes(self, cls: str) -> int:
        s = self.scores.get(cls)
        return int(s.size) if s is not None else 0

    def parasite_total(self) -> float:
        return sum(self.total(c) for c in PARASITE_CLASSES)


def _iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """Pairwise IoU for xywh (centre-form, normalised) boxes."""
    if len(boxes) == 0:
        return np.zeros((0, 0))
    x, y, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1, y1, x2, y2 = x - w / 2, y - h / 2, x + w / 2, y + h / 2
    area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ix1 = np.maximum(x1[:, None], x1[None, :])
    iy1 = np.maximum(y1[:, None], y1[None, :])
    ix2 = np.minimum(x2[:, None], x2[None, :])
    iy2 = np.minimum(y2[:, None], y2[None, :])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    union = area[:, None] + area[None, :] - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(union > 0, inter / union, 0.0)


def _suppress(order: np.ndarray, boxes: np.ndarray, iou_thr: float) -> np.ndarray:
    """Greedy suppression over one group, highest score first.

    `order` is positional indices into `boxes`, already sorted by descending
    score. The IoU matrix is built once per group rather than per comparison,
    and groups of one skip it entirely: with 4,508 fields and up to nine
    classes there are tens of thousands of groups, and a per-group DataFrame
    round trip dominated the runtime of the whole analysis.
    """
    b = boxes[order]
    keep, dead = [], np.zeros(len(order), dtype=bool)
    iou = _iou_matrix(b)
    for i in range(len(order)):
        if dead[i]:
            continue
        keep.append(order[i])
        dead |= iou[i] > iou_thr
        dead[i] = True
    return np.asarray(keep, dtype=int)


def apply_condition(det: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Return the detections a given post-processing condition leaves standing.

    `raw` is the cache as stored. `matched` applies per-class suppression at
    IoU 0.45 identically to every checkpoint, which is the only condition in
    which two detector families are comparable. `source` reproduces the
    published operating point: a 0.25 floor and class-agnostic suppression,
    kept as a comparator because it is what the source study ran, not because
    it is a good idea --- class-agnostic suppression deletes the losing species
    when two compete for one parasite, so no distribution over species is left
    to recalibrate.
    """
    from .errors import BadInput

    if condition not in CONDITIONS:
        raise BadInput(f"unknown post-processing condition {condition!r}",
                       remedy=f"Use one of: {', '.join(CONDITIONS)}.")
    if condition == "raw" or det.empty:
        return det

    work = det
    if condition == "source":
        work = work[work["score"] >= SOURCE_CONF]
        if work.empty:
            return work
    iou_thr = SOURCE_IOU if condition == "source" else MATCHED_IOU

    # One pass in numpy: group by field (and class, unless class-agnostic),
    # sort by descending score inside each group, suppress, keep the survivors.
    #
    # A FIELD IS A (PATIENT, IMAGE) PAIR. Keying on image_id alone -- which
    # this did -- makes suppression run across patients the moment two of them
    # use the same image name, which per-patient frame numbering (000.jpg,
    # 001.jpg) guarantees. One patient's boxes then delete another's: on a
    # fixture re-keyed that way, 28% of detections under `matched` and 38%
    # under `source` were wrongly removed, and a patient whose only detection
    # lost a tie became uninfected. It also turns "one field" into "every
    # patient's copy of frame 003", and the O(n^2) IoU matrix for that group
    # exhausted memory.
    img = pd.factorize(
        pd.MultiIndex.from_arrays([work["patient_id"].astype(str),
                                   work["image_id"].astype(str)]),
        sort=False)[0].astype(np.int64)
    if condition == "source":
        gid = img
    else:
        cls = pd.factorize(work["cls"], sort=False)[0].astype(np.int64)
        gid = img * (cls.max() + 1) + cls
    score = work["score"].to_numpy(dtype=float)
    boxes = work[["x", "y", "w", "h"]].to_numpy(dtype=float)
    positions = np.arange(len(work))

    # sort by group, then by descending score, so each group is a contiguous
    # slice already in the order greedy suppression wants
    lex = np.lexsort((-score, gid))
    gid_s, pos_s = gid[lex], positions[lex]
    bounds = np.flatnonzero(np.diff(gid_s)) + 1
    keep_positions = []
    for lo, hi in zip(np.concatenate(([0], bounds)),
                      np.concatenate((bounds, [len(gid_s)]))):
        block = pos_s[lo:hi]
        if len(block) == 1:
            keep_positions.append(block)
            continue
        kept = _suppress(np.arange(len(block)), boxes[block], iou_thr)
        keep_positions.append(block[kept])
    if not keep_positions:
        return work.iloc[0:0]
    keep = np.sort(np.concatenate(keep_positions))
    return work.iloc[keep]


def build_views(det: pd.DataFrame, patients: pd.DataFrame,
                condition: str = "raw", *, report: dict | None = None) -> dict:
    """One `PatientView` per patient in `patients`, in that order.

    Every patient gets a view, including one for whom the detector emitted
    nothing. A patient silently absent from this mapping is how a missing
    prediction became a scored false positive in the predecessor.

    The aggregation is done in two grouped passes over the whole table rather
    than per patient per class. The obvious nested loop is 170 x 9 small
    groupbys and dominated the runtime of the entire analysis.

    Pass a dict as `report` to receive the count of detections that could not
    be matched to a patient, a class and a field. Rows are dropped either way,
    because filing them somewhere is worse -- but a silent drop is how a join
    that matched nothing produced a complete set of results, so the number
    travels into `Study.meta` and out to the reader.
    """
    work = apply_condition(det, condition)
    classes = tuple(BLOOD_CLASSES) + tuple(PARASITE_CLASSES)
    ids = [str(p) for p in patients["patient_id"]]
    n_by_id = dict(zip(ids, pd.to_numeric(patients["n_images"],
                                          errors="coerce").fillna(0).astype(int)))

    views = {pid: PatientView(patient_id=pid, n_fields=max(n_by_id.get(pid, 0), 0),
                              scores={c: np.zeros(0) for c in classes},
                              per_field_max={c: np.zeros(0) for c in classes},
                              per_field_count={c: np.zeros(0, dtype=int)
                                               for c in classes})
             for pid in ids}
    if report is not None:
        report.update(n_detections=int(len(work)), n_dropped=0,
                      n_patients=len(ids), condition=condition)
    if work.empty:
        return views

    known = {c: j for j, c in enumerate(classes)}
    pat_of = {p: i for i, p in enumerate(ids)}
    pat = work["patient_id"].astype(str).map(pat_of).to_numpy()
    cls = work["cls"].astype(str).map(known).to_numpy()
    img_raw = work["image_id"].astype(str)
    n_before = len(work)
    # A row whose patient, class or field cannot be resolved is dropped here
    # rather than allowed to land in some other patient's arrays by an index
    # that happened to be in range. `pd.factorize` codes a missing value as -1,
    # which is a perfectly valid index into the row above -- so a detection
    # with no image_id was silently filed under the previous patient's last
    # field, complete with its confidence.
    keep = ~(pd.isna(pat) | pd.isna(cls) | img_raw.isna().to_numpy()
             | (img_raw == "nan").to_numpy() | (img_raw == "<NA>").to_numpy())
    dropped = int(n_before - int(keep.sum()))
    if report is not None:
        report.update(n_dropped=dropped,
                      dropped_share=dropped / max(n_before, 1))
    if dropped:
        work = work.loc[keep]
        pat, cls = pat[keep], cls[keep]
        # A join that matches nothing produces a completely plausible study --
        # every patient with no detections, the trivial baseline as the answer.
        # That is the zero-padded-identifier failure this package was written
        # after, so a total miss is refused rather than reported.
        if not len(work):
            from .errors import BadInput

            raise BadInput(
                f"none of the {n_before:,} detections matched a patient, a "
                "class and a field in this cohort",
                remedy=("The detection table and the patient table do not "
                        "join. Check the identifier format on both sides -- a "
                        "zero-padded id read back as an integer is the usual "
                        "cause -- rather than analysing an empty result."),
                example_detection_ids=sorted(
                    set(det["patient_id"].astype(str)))[:3],
                example_patient_ids=ids[:3])
    pat = pat.astype(np.int64)
    cls = cls.astype(np.int64)
    score = work["score"].to_numpy(dtype=float)

    # --- every score, by patient and class, in one sort ----------------------
    # The obvious version groups the frame by (patient, class) and pulls each
    # group's column, which is 170 x 9 small pandas objects and was most of the
    # runtime of a core run. A single lexsort puts every group in a contiguous
    # slice and `np.split` hands out views into one array.
    key = pat * len(classes) + cls
    order = np.argsort(key, kind="stable")
    key_s, score_s = key[order], score[order]
    cuts = np.flatnonzero(np.diff(key_s)) + 1
    for block_key, block in zip(key_s[np.concatenate(([0], cuts))],
                                np.split(score_s, cuts)):
        views[ids[block_key // len(classes)]].scores[
            classes[block_key % len(classes)]] = block

    # --- per-field maxima and counts, by scattering into one dense array -----
    # Fields are numbered once, globally: patient p's fields occupy the slice
    # [start[p], start[p] + n_fields[p]). One scatter then fills every
    # (field, class) cell, and each view takes a slice of it.
    img_codes = pd.factorize(work["image_id"].astype(str).to_numpy(),
                             sort=True)[0].astype(np.int64)
    n_img = int(img_codes.max()) + 1
    det_key = pat * n_img + img_codes
    # the distinct (patient, field) pairs, in patient-then-image order, which is
    # the order the per-field arrays have always been in
    field_key = np.unique(det_key)
    n_fields_total = int(field_key.size)
    starts = np.zeros(len(ids) + 1, dtype=np.int64)
    starts[1:] = np.cumsum(np.bincount(field_key // n_img, minlength=len(ids)))
    # every detection's key is present in field_key by construction, so a
    # searchsorted is an exact lookup rather than an approximation
    slots = np.searchsorted(field_key, det_key)

    fmax = np.zeros((n_fields_total, len(classes)), dtype=float)
    fcount = np.zeros((n_fields_total, len(classes)), dtype=np.int64)
    # reduceat over contiguous (field, class) blocks rather than `np.maximum.at`
    # -- the unbuffered ufunc.at form is an order of magnitude slower and this
    # runs over every detection in the cache
    cell = slots * len(classes) + cls
    o = np.argsort(cell, kind="stable")
    cell_s, score_s2 = cell[o], score[o]
    block_starts = np.concatenate(([0], np.flatnonzero(np.diff(cell_s)) + 1))
    block_max = np.maximum.reduceat(score_s2, block_starts)
    block_n = np.diff(np.concatenate((block_starts, [len(cell_s)])))
    rows, cols = np.divmod(cell_s[block_starts], len(classes))
    fmax[rows, cols] = block_max
    fcount[rows, cols] = block_n

    for i, pid in enumerate(ids):
        v = views[pid]
        lo, hi = int(starts[i]), int(starts[i + 1])
        v.n_fields = max(v.n_fields, hi - lo)
        if hi == lo:
            continue
        block_max = fmax[lo:hi]
        block_count = fcount[lo:hi]
        for j, c in enumerate(classes):
            v.per_field_max[c] = block_max[:, j]
            v.per_field_count[c] = block_count[:, j]
    return views
