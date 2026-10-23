#!/usr/bin/env python3
"""exp_schema.py --- every key in eda.json and verify.json, defined once.

WHY THIS EXISTS

Twice now a check reported nothing wrong because it was looking in the wrong
place. `centre_distance` writes flat keys --- `parasite_median_radius` --- and
a consumer went looking for a nested `median_radius.parasite` that never
existed. The data was in the file the whole time. The consumer said UNKNOWN,
which at least is not a lie, but only because someone had thought to build an
UNKNOWN state; a shade less care and it would have said "clear".

The cause is not carelessness. It is that a producer and a consumer in two
different files each spell a key from memory, and nothing compares the two
spellings until an archive is in front of them. Tests do not catch it, because
the test fixtures are written from the same memory as the consumer.

So keys are not spelled in those files any more. Every key is a `Key` object
here, with its path, its type and what it means. Producers write through it,
consumers read through it, and `test_experiment.py` asserts that the set of
keys read is a subset of the set of keys written. A typo is now an
AttributeError at import, and a drift is a failing test --- neither can reach
an archive.

USAGE

    from exp_schema import EDA, VERIFY

    report = {}
    EDA.IDENT_N_UNINFECTED.put(report, 54)          # producer
    ...
    problems = EDA.validate(report)                 # before writing

    n = EDA.IDENT_N_UNINFECTED.get(doc)             # consumer
    if n is MISSING: ...                            # absent, not None

Wildcards use `*` and iterate:

    for name, n in VERIFY.SMOKE_FIELDS_AT_CAP.each(doc):
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator


class _Missing:
    """Absent, as distinct from present-and-null.

    `d.get(k)` cannot tell "the producer wrote null" from "the producer never
    ran that block", and those mean different things: the first is a measured
    absence, the second is a check that did not happen. Every finding in this
    suite turns on that distinction, so it gets a sentinel rather than None.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __bool__(self):
        return False

    def __repr__(self):
        return "MISSING"


MISSING = _Missing()


@dataclass(frozen=True)
class Key:
    """One path into a JSON document, with its meaning attached."""
    path: str
    kind: type | tuple = object
    doc: str = ""
    required: bool = False

    @property
    def parts(self) -> list:
        return self.path.split(".")

    @property
    def wild(self) -> bool:
        return "*" in self.parts

    @property
    def block(self) -> str:
        return self.parts[0]

    # -- read ------------------------------------------------------------- #

    def get(self, d, default=MISSING):
        """The value at this path, or `default` (MISSING) if any hop is absent."""
        if self.wild:
            raise ValueError(f"{self.path} is a wildcard; use .each()")
        cur = d
        for p in self.parts:
            if not isinstance(cur, dict) or p not in cur:
                return default
            cur = cur[p]
        return cur

    def each(self, d) -> Iterator:
        """(name, value) for every match of the single `*` hop.

        The misuse check runs HERE, at the call, not inside the generator
        body: a generator does not execute until it is iterated, so a
        deferred `raise` surfaces somewhere else entirely --- or, if the
        caller never iterates, not at all. That is the same shape of bug as
        reading a wildcard key positionally, and it should not be possible in
        the thing built to prevent it.
        """
        if not self.wild:
            raise ValueError(f"{self.path} has no wildcard; use .get()")
        return self._each(d)

    def _each(self, d) -> Iterator:
        i = self.parts.index("*")
        before, after = self.parts[:i], self.parts[i + 1:]
        cur = d
        for p in before:
            if not isinstance(cur, dict) or p not in cur:
                return
            cur = cur[p]
        if not isinstance(cur, dict):
            return
        for name, sub in cur.items():
            v = sub
            ok = True
            for p in after:
                if not isinstance(v, dict) or p not in v:
                    ok = False
                    break
                v = v[p]
            if ok:
                yield name, v

    def names(self, d) -> list:
        """The wildcard hop's keys, whether or not the tail is present."""
        if not self.wild:
            raise ValueError(f"{self.path} has no wildcard")
        i = self.parts.index("*")
        cur = d
        for p in self.parts[:i]:
            if not isinstance(cur, dict) or p not in cur:
                return []
            cur = cur[p]
        return list(cur) if isinstance(cur, dict) else []

    # -- write ------------------------------------------------------------ #

    def put(self, d, value, name: str | None = None) -> None:
        """Set this path, creating intermediate dicts. `name` fills the `*`."""
        parts = list(self.parts)
        if self.wild:
            if name is None:
                raise ValueError(f"{self.path} needs a name for the wildcard")
            parts[parts.index("*")] = name
        cur = d
        for p in parts[:-1]:
            nxt = cur.get(p)
            if not isinstance(nxt, dict):
                nxt = {}
                cur[p] = nxt
            cur = nxt
        cur[parts[-1]] = value

    def ok(self, value) -> bool:
        if value is MISSING or value is None:
            return not self.required
        if self.kind is object:
            return True
        if self.kind is float:
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        return isinstance(value, self.kind)


@dataclass
class Schema:
    """A named set of keys, and the checks that hold a document to them."""
    name: str
    keys: list = field(default_factory=list)

    def __post_init__(self):
        # bind each key as an attribute so a typo is an AttributeError at the
        # call site rather than a silent MISSING at runtime
        for k, key in self._named.items():
            setattr(self, k, key)

    _named: dict = field(default_factory=dict, repr=False)

    @classmethod
    def build(cls, name: str, named: dict) -> "Schema":
        s = cls(name=name, keys=list(named.values()))
        s._named = dict(named)
        for attr, key in named.items():
            setattr(s, attr, key)
        return s

    def __iter__(self):
        return iter(self.keys)

    def by_path(self, path: str):
        for k in self.keys:
            if k.path == path:
                return k
        return None

    def paths(self) -> set:
        return {k.path for k in self.keys}

    def validate(self, d: dict) -> list:
        """Problems with a document, as strings. Empty means it conforms."""
        out = []
        for k in self.keys:
            if k.wild:
                if k.required and not list(k.each(d)):
                    out.append(f"{self.name}: no match for required "
                               f"{k.path!r}")
                for nm, v in k.each(d):
                    if not k.ok(v):
                        out.append(f"{self.name}: {k.path} [{nm}] is "
                                   f"{type(v).__name__}, want "
                                   f"{getattr(k.kind, '__name__', k.kind)}")
                continue
            v = k.get(d)
            if v is MISSING:
                if k.required:
                    out.append(f"{self.name}: required key {k.path!r} absent")
                continue
            if not k.ok(v):
                out.append(f"{self.name}: {k.path} is {type(v).__name__}, "
                           f"want {getattr(k.kind, '__name__', k.kind)}")
        return out

    def describe(self) -> str:
        w = max((len(k.path) for k in self.keys), default=0)
        lines = [f"# {self.name}", ""]
        for k in sorted(self.keys, key=lambda x: x.path):
            flag = "*" if k.required else " "
            lines.append(f"{flag} {k.path:<{w}}  {k.doc}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# eda.json --- what the archive says, before any model runs
# --------------------------------------------------------------------------- #

EDA = Schema.build("eda.json", {
    # -- provenance -------------------------------------------------------- #
    "ARCHIVE": Key("archive", str, "the archive root the run read", True),
    "MAG_VERSION": Key("mag", str, "mag version and build", True),
    "GENERATED": Key("generated_utc", str, "when this file was written", True),
    "SETS": Key("sets", list, "partitions surveyed"),

    # -- manifest ---------------------------------------------------------- #
    "MANIFEST_ROWS": Key("manifest_rows", int, "rows in image_list.xlsx", True),
    "MANIFEST_COLUMNS": Key("manifest_columns", list, "its columns"),

    # -- class map --------------------------------------------------------- #
    "CLASSES_DECLARED": Key("classes.declared", list,
                            "classes.txt, in file order"),
    "CLASSES_SCHEMA": Key("classes.schema", list,
                          "the protocol's names, in index order"),
    "CLASSES_MATCH": Key("classes.match", bool,
                         "the two agree index for index"),

    # -- partitions -------------------------------------------------------- #
    "PART_SMEARS": Key("partitions.by_set.*.smears", int,
                       "smears in this partition"),
    "PART_IMAGES": Key("partitions.by_set.*.images", int,
                       "fields in this partition"),
    "PART_PROBLEMS": Key("partitions.problems", list,
                         "partition checks that failed"),

    # -- cohort ------------------------------------------------------------ #
    "COHORT_TABLE": Key("cohort_table", dict,
                        "centre -> species -> smears, plus Total", True),
    "COHORT_MARKDOWN": Key("cohort_markdown", str, "the same, rendered"),

    # -- identifiability (Proposition 5) ----------------------------------- #
    "IDENT_N_PATIENTS": Key("identifiability.n_patients", int,
                            "patients in the analysis set", True),
    "IDENT_N_INFECTED": Key("identifiability.n_infected", int, "", True),
    "IDENT_N_UNINFECTED": Key("identifiability.n_uninfected", int, "", True),
    "IDENT_PREVALENCE": Key("identifiability.prevalence", float, "", True),
    "IDENT_CENTRES": Key("identifiability.uninfected_centres", list,
                         "centres contributing an uninfected smear", True),
    "IDENT_SEPARABLE": Key("identifiability.specificity_separable_from_centre",
                           bool,
                           "False means specificity and site are collinear",
                           True),

    # -- annotations ------------------------------------------------------- #
    "LAB_SET": Key("labels_base.set", str, "the partition these came from"),
    "LAB_BOXES": Key("labels_base.boxes", int, "boxes in the base set", True),
    "LAB_PER_FIELD": Key("labels_base.boxes_per_field", float, ""),
    "LAB_FIELDS": Key("labels_base.fields_with_boxes", int, ""),
    "LAB_PER_CLASS": Key("labels_base.per_class", dict,
                         "class -> box count, base set", True),
    "LABZ_BOXES": Key("labels_zoom.boxes", int, "boxes after the transform"),
    "LABZ_PER_CLASS": Key("labels_zoom.per_class", dict,
                          "class -> box count, transformed set"),

    # -- geometry ---------------------------------------------------------- #
    # FLAT, because that is what the computation produces. A consumer that
    # wants them nested does not get to invent a nesting.
    "GEO_PAR_RADIUS": Key("centre_distance.parasite_median_radius", float,
                          "median normalised radius, parasite classes"),
    "GEO_OTH_RADIUS": Key("centre_distance.other_median_radius", float,
                          "median normalised radius, blood classes"),
    "GEO_PAR_MARGIN": Key("centre_distance.parasite_margin_share", float,
                          "share of parasite boxes a 4:3->16:9 crop removes"),
    "GEO_OTH_MARGIN": Key("centre_distance.other_margin_share", float,
                          "the same for blood classes"),
    "GEO_NEARER": Key("centre_distance.parasites_nearer_centre", bool, ""),
    "GEO_PRED_PAR": Key("centre_distance.predicted_retention_parasite", float,
                        "1 - margin share: retention a pure crop would leave"),
    "GEO_PRED_OTH": Key("centre_distance.predicted_retention_other", float, ""),
    "GEO_PER_CLASS": Key("centre_distance.per_class", dict,
                         "class -> {n, median_radius, mean_radius, "
                         "share_in_crop_margin}"),

    # -- enrichment -------------------------------------------------------- #
    "LOAD_ENRICHMENT": Key("parasite_load.enrichment_among_infected", float,
                           "share of an infected patient's fields carrying a "
                           "parasite box"),
    "LOAD_MEDIAN": Key("parasite_load.parasite_boxes_per_infected_smear.median",
                       float, ""),
    "LOAD_MIN": Key("parasite_load.parasite_boxes_per_infected_smear.min",
                    float, ""),
    "LOAD_MAX": Key("parasite_load.parasite_boxes_per_infected_smear.max",
                    float, ""),
    "LOAD_COVERAGE": Key("parasite_load.infected_smears_with_labels", int,
                         "infected smears that carry any annotation"),
    "LOAD_TOTAL": Key("parasite_load.infected_smears_total", int, ""),

    # -- transform --------------------------------------------------------- #
    "TF_PARASITE_KEPT": Key("transform_retention.parasite_kept", float,
                            "measured parasite-box retention"),
    "TF_OTHER_KEPT": Key("transform_retention.other_kept", float, ""),
    "TF_ALL_KEPT": Key("transform_retention.all_kept", float, ""),
    "TF_BY_CLASS": Key("transform_retention.by_class", dict,
                       "class -> {base, zoom, kept}"),
    "TF_VERDICT": Key("transform_retention.verdict", str, ""),
    "TF_RISING": Key("classes_rising_after_transform", dict,
                     "classes whose count GREW; a crop cannot add a box"),

    # -- images ------------------------------------------------------------ #
    "IMG_SAMPLED": Key("images.sampled", int, "images opened"),
    "IMG_SIZES": Key("images.sizes", dict, "'WxH' -> count"),
    "IMG_RATIOS": Key("images.aspect_ratios", dict,
                      "aspect ratio -> count; keys are numbers as strings"),

    "FIGURES": Key("figures", list, "figure filenames written"),
})


# --------------------------------------------------------------------------- #
# verify.json --- can these checkpoints detect?
# --------------------------------------------------------------------------- #

VERIFY = Schema.build("verify.json", {
    "ARCHIVE": Key("archive", str, "", True),
    "MAG_VERSION": Key("mag", str, "", True),
    "GENERATED": Key("generated_utc", str, "", True),
    "DEVICE": Key("device", str, ""),
    "VERDICT": Key("verdict", str, "the run's one-line conclusion", True),
    "GPU": Key("gpu", dict, "mag.detect.gpu_budget()"),
    "MAG_MAX_DET": Key("mag_max_det", int,
                       "mag.detect.MAX_DET as this run found it", True),

    # -- the file itself --------------------------------------------------- #
    "CK_PATH": Key("checkpoints.*.path", str, ""),
    "CK_SHA": Key("checkpoints.*.sha256", str, ""),
    "CK_BYTES": Key("checkpoints.*.bytes", int, ""),
    "CK_NCLASSES": Key("checkpoints.*.n_classes", int, ""),
    "CK_FORMAT": Key("checkpoints.*.metadata.format", str, ""),
    "CK_PROBLEMS": Key("checkpoints.*.problems", list, ""),

    # -- the smoke test ---------------------------------------------------- #
    "SMOKE_N_IMAGES": Key("smoke.*.n_images", int, "fields driven"),
    "SMOKE_LOADER": Key("smoke.*.loader", str,
                        "RTDETR | YOLO | yolov5-repo"),
    "SMOKE_DETECTIONS": Key("smoke.*.detections", int, ""),
    "SMOKE_PER_FIELD_MEAN": Key("smoke.*.per_field_mean", float, ""),
    "SMOKE_PER_FIELD_MIN": Key("smoke.*.per_field_min", int, ""),
    "SMOKE_PER_FIELD_MAX": Key("smoke.*.per_field_max", int, ""),
    "SMOKE_FIELDS_WITH_ANY": Key("smoke.*.fields_with_any", int, ""),
    "SMOKE_SCORE_MEDIAN": Key("smoke.*.score_median", float, ""),
    "SMOKE_SCORE_MAX": Key("smoke.*.score_max", float, ""),
    "SMOKE_SECONDS_PER_FIELD": Key("smoke.*.seconds_per_field", float, ""),
    # Geometry. `mag.schema.validate_detections` REFUSES a cache holding any
    # box with w <= 0 or h <= 0 -- such a box has zero area, so `_iou_matrix`
    # clips it and it neither suppresses nor is suppressed, passing through
    # the matched condition unchallenged. That refusal happens when the cache
    # is READ, which is after the pass is paid for. Measuring it on twenty
    # fields costs nothing and moves the discovery to before the hour.
    "SMOKE_BAD_GEOM": Key("smoke.*.non_positive_geometry", int,
                          "boxes with w <= 0 or h <= 0"),
    "SMOKE_BAD_GEOM_SHARE": Key("smoke.*.non_positive_geometry_share", float, ""),
    "SMOKE_BAD_GEOM_BY_CLASS": Key("smoke.*.non_positive_geometry_by_class",
                                   dict, ""),
    "SMOKE_MIN_W": Key("smoke.*.min_width", float, "smallest normalised w"),
    "SMOKE_MIN_H": Key("smoke.*.min_height", float, "smallest normalised h"),
    "SMOKE_PROBLEMS": Key("smoke.*.problems", list, ""),
    "SMOKE_NOTES": Key("smoke.*.notes", list,
                       "measured, reported, not disqualifying"),

    # -- the operating point the smoke test used --------------------------- #
    # Recorded because it MUST equal mag's own, and a run at ultralytics'
    # defaults measures a different detector from the one the pass will cache.
    "OP_CONF": Key("smoke.*.operating_point.conf", float, ""),
    "OP_IOU": Key("smoke.*.operating_point.iou", float, ""),
    "OP_AGNOSTIC": Key("smoke.*.operating_point.agnostic_nms", bool, ""),
    "OP_MAX_DET": Key("smoke.*.operating_point.max_det", int, ""),

    # -- the detection cap ------------------------------------------------- #
    "SMOKE_FIELDS_AT_CAP": Key("smoke.*.fields_at_cap", int,
                               "fields returning exactly max_det boxes"),
    "SMOKE_CAP_SHARE": Key("smoke.*.fields_at_cap_share", float, ""),
    "SMOKE_CENSORING": Key("smoke.*.censoring_threshold", float,
                           "lowest score still standing on a capped field"),

    # -- the cap probe ----------------------------------------------------- #
    "PROBE_MAX_DET": Key("smoke.*.cap_probe.probe_max_det", int, ""),
    "PROBE_N_FIELDS": Key("smoke.*.cap_probe.n_fields", int, ""),
    "PROBE_TRUE_MEAN": Key("smoke.*.cap_probe.true_per_field_mean", float, ""),
    "PROBE_TRUE_MAX": Key("smoke.*.cap_probe.true_per_field_max", int, ""),
    "PROBE_DISCARDED": Key("smoke.*.cap_probe.discarded", int, ""),
    "PROBE_DISCARDED_SHARE": Key("smoke.*.cap_probe.discarded_share", float, ""),
    "PROBE_BY_CLASS": Key("smoke.*.cap_probe.discarded_by_class", dict, ""),
    "PROBE_KEPT_PAR": Key("smoke.*.cap_probe.kept_parasite", int, ""),
    "PROBE_KEPT_RBC": Key("smoke.*.cap_probe.kept_rbc", int, ""),
    "PROBE_LOST_PAR": Key("smoke.*.cap_probe.lost_parasite", int, ""),
    "PROBE_LOST_RBC": Key("smoke.*.cap_probe.lost_rbc", int, ""),
    "PROBE_DENSITY_CAPPED": Key("smoke.*.cap_probe.density_capped", float, ""),
    "PROBE_DENSITY_TRUE": Key("smoke.*.cap_probe.density_true", float, ""),
    "PROBE_DENSITY_BIAS": Key("smoke.*.cap_probe.density_bias", float,
                              "signed relative shift in R6/R6b density"),
    "PROBE_MATERIAL": Key("smoke.*.cap_probe.material", bool, ""),
    "PROBE_SUGGESTED": Key("smoke.*.cap_probe.suggested_max_det", int,
                           "a cap the sampled fields do not reach"),
    "PROBE_STILL_CAPPED": Key("smoke.*.cap_probe.still_at_probe_cap", int, ""),

    # -- cross-checkpoint --------------------------------------------------- #
    "AGREE_PER_FIELD": Key("agreement.per_field_mean", dict, ""),
    "AGREE_RATIO": Key("agreement.ratio_max_to_min", float, ""),
    "AGREE_EXCLUDED": Key("agreement.excluded", list, ""),
    "AGREE_FIXED": Key("agreement.fixed_budget", list,
                       "checkpoints emitting a constant per-field count"),
    "PROJ_FIELDS": Key("projection.fields", int, ""),
    "PROJ_TOTAL_MIN": Key("projection.total_minutes", float, ""),
})


#: The names the protocol uses. `mag.schema` is the authority; these are the
#: fallback for a checkout without mag importable, and they are compared
#: against mag's in the test suite so the two cannot drift either.
try:                                                          # pragma: no cover
    from mag.schema import PARASITE_CLASSES, BLOOD_CLASSES, ALL_CLASSES
except Exception:                                             # noqa: BLE001
    PARASITE_CLASSES = ("P_falciparum", "P_ovale", "P_malariae", "P_vivax",
                        "Babesia_divergens", "T_brucei")
    BLOOD_CLASSES = ("WBC", "RBC", "platelet")
    ALL_CLASSES = BLOOD_CLASSES + PARASITE_CLASSES

RBC = "RBC"

SCHEMAS = {"eda.json": EDA, "verify.json": VERIFY}


def main() -> int:
    """`python exp_schema.py` prints the contract."""
    for s in SCHEMAS.values():
        print(s.describe())
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
