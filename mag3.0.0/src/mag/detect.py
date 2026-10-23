"""Run the released checkpoints over the images and write the detection cache.

This is the only module that needs torch and a GPU. Everything else in the
project reads the cache it produces, which is what makes the analysis testable
without one.

Two decisions here are deliberate and are argued in the manuscript's Methods:

  * Cache BEFORE non-maximum suppression, at a confidence floor of 0.001,
    keeping the full per-class score vector. The source study's inference used
    class-agnostic NMS at IoU 0.45 and a 0.25 floor. Class-agnostic suppression
    deletes the losing species box when two species compete for one parasite,
    so there is no distribution over species left to recalibrate; and the 0.25
    floor truncates exactly the low-confidence mass the study is about. Caching
    below both keeps the source settings available as a comparator (see
    `io.apply_source_settings`) without imposing them on everything.

  * Record the checkpoint's embedded training arguments. Ultralytics stores
    imgsz, optimizer, lr0, augmentation and the class-name mapping inside the
    .pt file; none of it appears in the source publication, and the class-index
    to species mapping is not optional information.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .cache import save as save_cache
from .errors import BadInput, IntegrityError, MissingInput
from .schema import file_sha256


@contextmanager
def _full_load():
    """torch.load with weights_only forced off, restored afterwards.

    torch >= 2.6 defaults weights_only=True, and third-party loaders (the
    yolov5 package among them) call torch.load without the argument, so a
    checkpoint that needs full unpickling fails inside code we do not control.
    """
    import torch
    orig = torch.load

    def patched(*a, **k):
        k.setdefault("weights_only", False)
        return orig(*a, **k)

    torch.load = patched
    try:
        yield
    finally:
        torch.load = orig


def _v5_module_roots(repo=None):
    """Directories that can supply top-level `models` and `utils`, best first.

    A repository-format checkpoint pickles its classes under those two
    top-level names. Two things provide them: a clone of
    github.com/ultralytics/yolov5, where they *are* the top level and nothing
    else is needed, and the packaged fork on PyPI, where they sit inside the
    installed package directory.

    The fork's location is resolved with importlib rather than by importing it.
    `import yolov5` executes the fork's own __init__, which pulls in sahi,
    roboflow, boto3 and huggingface-hub; a version clash anywhere in that stack
    raises, and the checkpoint we came here to load is then unreachable for a
    reason that has nothing to do with the checkpoint. find_spec reads the
    package's location without running any of it.
    """
    roots = []
    if repo:
        roots.append(("--yolov5-repo", Path(repo).expanduser().resolve()))
    try:
        spec = importlib.util.find_spec("yolov5")
    except Exception:                      # a half-installed or shadowed name
        spec = None
    locs = list(getattr(spec, "submodule_search_locations", None) or [])
    if locs:
        roots.append(("the installed yolov5 package", Path(locs[0])))
    return roots


def _v5_help(weights, attempts) -> str:
    from .discover import yolov5_error_hint

    lines = [
        f"{weights} is a repository-format YOLOv5 checkpoint: its pickle refers "
        "to the original ultralytics/yolov5 top-level modules ('models', "
        "'utils'), which neither torch nor the ultralytics package provides.",
        "",
        "Tried, in order:",
    ]
    lines += ([f"  - {a}" for a in attempts]
              or ["  - nothing: no source of those modules is installed or given"])
    hints = []
    for a in attempts:
        h = yolov5_error_hint(a)
        if h and h not in hints:
            hints.append(h)
    if hints:
        lines += [""] + [f"  -> {h}" for h in hints]
    lines += [
        "",
        "Either of these fixes it, and every finished smear is kept:",
        "",
        "  1. the packaged fork, if it imports cleanly here:",
        "         pip install yolov5",
        "     If it is already installed and an attempt above shows it failing "
        "to import, that error is the thing to fix -- usually a clashing "
        "opencv, numpy or huggingface-hub in the same environment. "
        "`python -c \"import yolov5\"` reproduces it on its own.",
        "",
        "  2. or a clone of the upstream repository, which carries these "
        "modules at its top level, needs nothing else installed, and whose "
        "current default branch no longer imports pkg_resources:",
        "         git clone --depth 1 https://github.com/ultralytics/yolov5",
        "         python run_local.py ... --yolov5-repo ./yolov5",
    ]
    return "\n".join(lines)


def _require_yolov5_modules(weights, cause: Exception, repo=None) -> str:
    """Put top-level `models`/`utils` on the path, or say exactly what failed.

    Returns the source that worked. Raises with every attempt and its real
    error rather than a single guess, because "install the fork" is wrong
    advice when the fork is installed and broken.
    """
    attempts = []
    for label, root in _v5_module_roots(repo):
        if not (root / "models").is_dir():
            attempts.append(f"{label} ({root}): no models/ directory there")
            continue
        before = set(sys.modules)
        added = str(root) not in sys.path
        if added:
            # front-inserted, as the fork itself does: the unpickler needs
            # top-level 'models'/'utils' to resolve to this copy
            sys.path.insert(0, str(root))
        try:
            importlib.import_module("models.yolo")
            return f"{label} ({root})"
        except Exception as e:
            attempts.append(f"{label} ({root}): {type(e).__name__}: {e}")
            if added:
                try:
                    sys.path.remove(str(root))
                except ValueError:
                    pass
            # a half-imported package would poison the next attempt
            for name in sorted(set(sys.modules) - before):
                if name.split(".")[0] in ("models", "utils", "yolov5"):
                    sys.modules.pop(name, None)
    raise ModuleNotFoundError(_v5_help(weights, attempts)) from cause


class _SilenceDeprecatedHalf(logging.Filter):
    """ultralytics 8.4 logs a deprecation for the half kwarg on EVERY predict
    call -- thousands of lines that bury the progress reporting. The kwarg
    still works; the noise does not need to."""

    def filter(self, record) -> bool:
        return "'half' is deprecated" not in str(record.getMessage())


def read_checkpoint_metadata(weights: Path, repo=None) -> dict:
    """Dump everything the checkpoint knows about how it was trained.

    NOTE: a .pt file is a pickle. Loading one executes code from the archive.
    Do this in a throwaway environment, and only because these weights come
    from a named Zenodo record with a DOI.
    """
    import torch                                   # imported lazily on purpose
    fmt = "ultralytics"
    with _full_load():
        try:
            ck = torch.load(weights, map_location="cpu", weights_only=False)
        except ModuleNotFoundError as e:
            if getattr(e, "name", "") not in ("models", "utils"):
                raise
            _require_yolov5_modules(weights, e, repo)
            ck = torch.load(weights, map_location="cpu", weights_only=False)
            fmt = "yolov5-repo"
    meta = {"file": str(weights), "sha256": file_sha256(Path(weights)),
            "format": fmt}
    for key in ("train_args", "date", "version", "epoch", "best_fitness"):
        if isinstance(ck, dict) and key in ck:
            meta[key] = _jsonable(ck[key])
    model = ck.get("model") if isinstance(ck, dict) else None
    if model is not None:
        meta["names"] = _jsonable(getattr(model, "names", None))
        meta["nc"] = _jsonable(getattr(model, "nc", None))
        if "train_args" not in meta:
            meta["args"] = _jsonable(getattr(model, "args", None))
    return meta


def _jsonable(o):
    try:
        json.dumps(o)
        return o
    except TypeError:
        return str(o)


def class_map_from(meta: dict, strict: bool = True) -> dict[int, str]:
    """Map the checkpoint's own class indices onto this project's vocabulary.

    The checkpoint carries `names`; classes.txt carries the archive's order.
    They should agree, and when they do not, guessing is the wrong move: a
    silently wrong map relabels every detection and every number downstream.
    So the checkpoint's names win, they are matched by string, and anything
    unmatched raises rather than being dropped.
    """
    from .ingest import CLASSES_TXT, CLASS_TO_SCHEMA

    names = meta.get("names")
    if not names:
        if strict:
            raise BadInput(
                f"{meta.get('file')} carries no class names",
                remedy=("Pass --assume-classes-txt to fall back on the "
                        "archive's classes.txt order. That is an assumption "
                        "about which index means which species, so if you use "
                        "it, say so in the manuscript: a wrong order relabels "
                        "every detection with no other symptom."),
                checkpoint=str(meta.get("file")))
        return {i: CLASS_TO_SCHEMA[c] for i, c in enumerate(CLASSES_TXT)}
    # Past this point the checkpoint DOES carry names. `strict=False` means
    # "fall back when names are absent", not "stop checking the vocabulary":
    # an unmatched name used to be dropped silently, and every detection of
    # that class vanished with no count and no message.

    if isinstance(names, dict):
        pairs = [(int(k), str(v)) for k, v in names.items()]
    else:
        pairs = list(enumerate(str(v) for v in names))

    out, unmatched = {}, []
    for idx, raw in sorted(pairs):
        key = _match_class(raw)
        if key is None:
            unmatched.append((idx, raw))
        else:
            out[idx] = key
    if unmatched:
        raise BadInput(
            f"{len(unmatched)} checkpoint class name(s) are not in the "
            f"archive vocabulary: {unmatched}",
            remedy=("An unmatched name used to be dropped silently, and every "
                    "detection of that class vanished with no count and no "
                    "message. Either this is not a checkpoint trained on this "
                    "archive, or _match_class needs the synonym added "
                    "deliberately."),
            known=list(CLASS_TO_SCHEMA))
    return out


def _match_class(raw: str) -> str | None:
    """Tolerate spacing, case and the usual abbreviations; nothing more."""
    from .ingest import CLASS_TO_SCHEMA

    norm = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    table = {norm(k): v for k, v in CLASS_TO_SCHEMA.items()}
    table.update({
        "platelet": "platelet", "plt": "platelet",
        "pfalciparum": "P_falciparum", "falciparum": "P_falciparum",
        "povale": "P_ovale", "ovale": "P_ovale",
        "pmalariae": "P_malariae", "malariae": "P_malariae",
        "pvivax": "P_vivax", "vivax": "P_vivax",
        "babesia": "Babesia_divergens", "babesiadivergens": "Babesia_divergens",
        "trypanosomabrucei": "T_brucei", "tbrucei": "T_brucei",
        "trypanosoma": "T_brucei",
    })
    return table.get(norm(raw))


class _V5Boxes:
    """The slice of the ultralytics Boxes surface run() reads, over a repo-v5
    Detections row tensor [n, 6] = (x, y, w, h, conf, cls), all normalised."""

    def __init__(self, t):
        self.xywhn = t[:, 0:4]
        self.conf = t[:, 4]
        self.cls = t[:, 5]
        self._n = int(t.shape[0])

    def __len__(self):
        return self._n


class _V5Result:
    def __init__(self, t):
        self.boxes = _V5Boxes(t) if t is not None and len(t) else None


#: Below this floor every detector in this study fires on essentially every
#: field: a blood smear image is dense with erythrocytes, and the released
#: checkpoints all carry an RBC class. A pass that comes back nearly empty is
#: therefore evidence about the loader, not about the detector.
SPARSE_FLOOR = 0.05


def implausible_yield(n_detections: int, fields_with_boxes: int,
                      n_fields: int, conf_floor: float) -> str:
    """Say why a finished pass cannot be a real measurement, or "" if it can.

    This exists because of a real incident: a repository-format YOLOv5
    checkpoint loaded without error, ran to completion over 4,508 fields, and
    produced two boxes. Two boxes is not a weak detector -- it is output that
    never reached the cache. Left unflagged it would have entered the factorial
    as "YOLOv5 fails at every rule", which is a claim about my adapter wearing
    the clothes of a finding.

    The thresholds are deliberately far from any defensible behaviour: at a
    0.001 floor the working checkpoints in this study cover 100% of fields.
    """
    if float(conf_floor) > SPARSE_FLOOR or n_fields <= 0:
        return ""                     # at a high floor, sparse output is normal
    if fields_with_boxes < 0.5 * n_fields:
        return (f"only {fields_with_boxes:,} of {n_fields:,} fields "
                f"({fields_with_boxes / n_fields:.2%}) produced any box at a "
                f"confidence floor of {conf_floor}")
    if n_detections < n_fields:
        return (f"{n_detections:,} boxes over {n_fields:,} fields is under one "
                f"per field at a confidence floor of {conf_floor}")
    return ""


def suspect_checkpoints(cache_dir) -> dict:
    """Checkpoint -> reason, for caches a run refused to trust."""
    out = {}
    for q in sorted(Path(cache_dir).glob("detections_*.suspect.json")):
        try:
            d = json.loads(q.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            d = {}
        name = d.get("checkpoint") or q.name[len("detections_"):-len(".suspect.json")]
        out[name] = d.get("reason", "marked suspect")
    return out


def _build_v5_model(weights):
    """AutoShape around the checkpoint, whichever source provided the modules.

    The fork exposes a one-call loader; a bare clone does not, and there the
    same object is built from the repository's own `models.common` -- which is
    what the fork's loader does internally anyway. Both end at AutoShape, whose
    knobs are the ones this study fixes.
    """
    try:
        import yolov5                     # the fork's convenience API
    except Exception:
        yolov5 = None                     # a clone, or a fork that will not import
    if yolov5 is not None and hasattr(yolov5, "load"):
        with _full_load():
            return yolov5.load(str(weights))
    import torch
    from models.common import AutoShape, DetectMultiBackend   # noqa: E402
    with _full_load():
        return AutoShape(DetectMultiBackend(str(weights), device=torch.device("cpu")))


class YOLOv5RepoAdapter:
    """A repository-format YOLOv5 checkpoint behind the predict() surface.

    The released yolov5 weights were trained with github.com/ultralytics/yolov5,
    whose pickled model classes the ultralytics package cannot load. The
    packaged fork (pip install yolov5) can, through its AutoShape pipeline, and
    AutoShape happens to expose exactly the knobs this study fixes: a confidence
    floor, an IoU threshold (1.0 leaves overlapping boxes standing, our
    pre-suppression condition), class-agnostic off, and a detection cap. Its
    `xywhn` output is normalised (x, y, w, h, conf, cls) -- the cache schema.

    Half precision is not forced here: AutoShape runs mixed precision on CUDA by
    itself, and the `half` argument is accepted only to match the call shape.
    """

    def __init__(self, weights, repo=None):
        self._m = _build_v5_model(weights)
        self._device = None

    def predict(self, source, imgsz=640, conf=0.001, iou=1.0,
                agnostic_nms=False, max_det=3000, verbose=False,
                device="0", stream=True, half=None, **_):
        m = self._m
        if self._device != str(device):
            want = ("cpu" if str(device).lower() in ("cpu", "-1")
                    else f"cuda:{str(device).split(',')[0]}")
            m.to(want)
            self._device = str(device)
        m.conf = float(conf)
        m.iou = float(iou)
        m.agnostic = bool(agnostic_nms)
        m.max_det = int(max_det)
        m.classes = None
        res = m([str(q) for q in source], size=int(imgsz))
        for t in res.xywhn:
            yield _V5Result(t)


def load_model(weights: Path, fmt: str | None = None, repo=None):
    """Pick the right ultralytics class for the architecture in the file.

    RT-DETR and YOLO are different model classes in ultralytics, and loading a
    detection transformer through `YOLO` is not reliable across versions.
    """
    from ultralytics import YOLO
    if fmt == "yolov5-repo":
        _require_yolov5_modules(weights, ModuleNotFoundError("models"), repo)
        with _full_load():
            return YOLOv5RepoAdapter(weights, repo), "yolov5-repo"
    name = Path(weights).as_posix().lower()
    if "rtdetr" in name or "rt-detr" in name or "detr" in name:
        try:
            from ultralytics import RTDETR
        except ImportError as e:
            raise ImportError(
                "this looks like an RT-DETR checkpoint and ultralytics does not "
                "expose RTDETR in this version. Upgrade ultralytics, or pass the "
                "file under a name without 'detr' if you have verified that YOLO "
                f"loads it correctly. ({e})") from e
        # Any other failure is re-raised: the docstring above says loading a
        # detection transformer through YOLO is unreliable, and swallowing the
        # error did exactly that whenever the constructor threw.
        return RTDETR(str(weights)), "RTDETR"
    try:
        return YOLO(str(weights)), "YOLO"
    except ModuleNotFoundError as e:
        if getattr(e, "name", "") not in ("models", "utils"):
            raise
        # reached when load_model is called without the metadata pass
        _require_yolov5_modules(weights, e, repo)
        with _full_load():
            return YOLOv5RepoAdapter(weights, repo), "yolov5-repo"


#: VRAM budget -> batch size. The released checkpoints are run at a 0.001
#: confidence floor with duplicate suppression effectively disabled, so a
#: single erythrocyte-dense field can carry thousands of boxes and the
#: post-processing buffers, not the backbone, are what exhausts a small card.
#: These are deliberately conservative: a run that finishes slowly beats one
#: that dies at 80%.
BATCH_FOR_VRAM = ((6.5, 2), (10.0, 4), (14.0, 8), (24.0, 12), (float("inf"), 16))

#: Above this a field is assumed to be erythrocyte-dense and is processed alone.
MAX_DET = 3000


def _as_array(x):
    """A detector's boxes as a numpy array, whatever it handed back.

    Ultralytics returns CUDA tensors on a GPU and CPU tensors otherwise, and
    some versions return plain arrays on the CPU path. Calling `.cpu().numpy()`
    unconditionally works on two of those three and raises an AttributeError on
    the third, inside the batch loop, after the pass has already spent its time
    on the smears before it.
    """
    for step in ("detach", "cpu", "numpy"):
        fn = getattr(x, step, None)
        if callable(fn):
            x = fn()
    import numpy as _np

    return _np.asarray(x)


def _image_paths(images_root: Path) -> list[Path]:
    """Every field under a partition, found case-insensitively.

    The suffix set is `ingest`'s, imported rather than repeated: two copies of
    it drifted once, and a single `.JPG` then made the archive audit
    differently on Linux and on Windows.
    """
    from .ingest import IMAGE_SUFFIXES

    return sorted(p for p in Path(images_root).rglob("*")
                  if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def gpu_budget(device: str = "0") -> dict:
    """What the card can take. Reported before inference, not discovered by OOM."""
    out = {"device": device, "cuda": False, "total_gb": None,
           "batch": 4, "half": False, "name": None}
    if str(device).lower() in ("cpu", "-1"):
        out["batch"] = 4
        return out
    try:
        import torch
        if not torch.cuda.is_available():
            out["note"] = "CUDA not available; falling back to CPU"
            return out
        idx = 0 if device in ("", None) else int(str(device).split(",")[0])
        props = torch.cuda.get_device_properties(idx)
        total = props.total_memory / 1e9
        out.update(cuda=True, total_gb=round(total, 1), name=props.name, half=True)
        out["batch"] = next(b for cap, b in BATCH_FOR_VRAM if total < cap)
    except Exception as e:                      # noqa: BLE001 - reported, not hidden
        out["note"] = f"could not query the GPU ({type(e).__name__}: {e})"
    return out


def _fingerprint(images, conf_floor, imgsz, class_map, sha) -> dict:
    """What a shard was computed from. Resume is only safe if this matches.

    Keying resume on the smear name alone was wrong twice over. An archive that
    was still extracting when the first run started left smears whose shard
    covered 7 of 10 fields, and the resumed run reported them complete -- the
    manifest then recorded 7 fields and every per-field denominator downstream
    was wrong. And a re-run at a different --conf reused shards computed at the
    old floor while stamping the manifest with the new one, so a cache truncated
    at 0.25 could claim the 0.001 floor the whole study rests on.
    """
    return {"images": sorted(q.stem for q in images),
            "conf_floor": float(conf_floor), "imgsz": int(imgsz),
            "class_map": {str(k): v for k, v in sorted(class_map.items())},
            "checkpoint_sha256": sha}


def _write_atomic(df: pd.DataFrame, path: Path) -> None:
    """A crash mid-write must not leave a shard that resume then trusts."""
    tmp = path.with_name(path.name + ".tmp")
    try:
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _usable_shards(shard_dir: Path, want: dict, by_smear: dict) -> set:
    """Smears whose shard is present, readable, and computed from these inputs."""
    ok = set()
    for q in shard_dir.glob("*.parquet"):
        smear = q.stem
        side = q.with_suffix(".json")
        if smear not in by_smear:
            continue
        if not side.exists():
            continue                       # written by an older version: redo it
        try:
            got = json.loads(side.read_text(encoding="utf-8"))
            pq.read_metadata(q)            # cheap: catches a truncated file
        except Exception:
            continue
        if got == _fingerprint(by_smear[smear], want["conf_floor"], want["imgsz"],
                               {int(k): v for k, v in want["class_map"].items()},
                               want["checkpoint_sha256"]):
            ok.add(smear)
    return ok


def run(weights: Path, images_root: Path, out_dir: Path, checkpoint: str,
        class_map: dict[int, str], conf_floor: float = 0.001,
        imgsz: int | None = None, batch: int | None = None, device: str = "0",
        patient_from_name=None, resume: bool = True,
        progress=print, half: bool | None = None, yolov5_repo=None,
        seed: int = 0) -> Path:
    """Run inference and write `detections_<checkpoint>.parquet` + manifest.

    `class_map` maps the checkpoint's class indices to our vocabulary; get it
    from `class_map_from(read_checkpoint_metadata(weights))` rather than assuming
    an order. `patient_from_name` maps an image filename to its smear; the
    released layout puts the smear in the parent directory, which is the default.

    Sizing. `batch` and `imgsz` default to None and are then chosen for you:
    the batch from the card's VRAM (2 on a 6 GB card, 16 on a 24 GB one) and
    imgsz from the checkpoint's own training arguments, because reproducing a
    published number at a different input size is not reproducing it. Half
    precision is used on CUDA. An out-of-memory error halves the batch and
    retries down to 1 before giving up, so a small card finishes slowly instead
    of failing.

    Resumability. 4,508 fields times three checkpoints times two image sets is
    27,048 forward passes. Each smear is written as soon as it finishes, next to
    a JSON fingerprint of the fields, floor, image size, class map and
    checkpoint hash it came from; a resumed run reuses a shard only when that
    fingerprint still matches, and re-infers otherwise. Shards are written
    atomically, so an interruption cannot leave one that resume would trust.
    """
    weights, images_root, out_dir = Path(weights), Path(images_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = out_dir / "_shards" / checkpoint
    shard_dir.mkdir(parents=True, exist_ok=True)

    meta = read_checkpoint_metadata(weights, yolov5_repo)

    if imgsz is None:
        args = meta.get("train_args") or meta.get("args") or {}
        imgsz = int(args.get("imgsz", 640)) if isinstance(args, dict) else 640
        progress(f"  image size {imgsz} (from the checkpoint's training arguments)")
    budget = gpu_budget(device)
    if str(device).lower() not in ("cpu", "-1") and not budget.get("cuda"):
        # gpu_budget used to SAY "falling back to CPU" while run() kept passing
        # the CUDA device to ultralytics, which raises
        # "Invalid CUDA 'device=0' requested" on a CPU-only torch build.
        # The fallback is now real -- but it is the slow path, and it says so.
        progress("  ! this torch build has no CUDA"
                 + (f" ({budget['note']})" if budget.get("note") else "")
                 + "; running on the CPU instead.")
        progress("  ! expect hours, not minutes. For the GPU, install the CUDA "
                 "build of torch (see requirements-gpu.txt), then re-run; "
                 "finished smears are kept.")
        device = "cpu"
        half = False
    if batch is None:
        batch = budget["batch"]
        progress(f"  batch {batch} for {budget.get('name') or device}"
                 + (f", {budget['total_gb']} GB" if budget["total_gb"] else "")
                 + (f" -- {budget['note']}" if budget.get("note") else ""))
    if half is None:
        half = bool(budget["half"])

    from .ingest import IMAGE_SUFFIXES

    paths = _image_paths(images_root)
    if not paths:
        raise MissingInput(
            f"no images under {images_root}",
            remedy=("Point --archive at the directory that holds images/ and "
                    "labels/, and check that the partition you named exists "
                    "under images/. `mag doctor` lists what it found."),
            looked_in=str(images_root),
            suffixes=sorted(IMAGE_SUFFIXES))

    by_smear: dict[str, list[Path]] = {}
    for pth in paths:
        pid = patient_from_name(pth.name) if patient_from_name else pth.parent.name
        by_smear.setdefault(pid, []).append(pth)

    want = {"conf_floor": float(conf_floor), "imgsz": int(imgsz),
            "class_map": {str(k): v for k, v in sorted(class_map.items())},
            "checkpoint_sha256": meta["sha256"]}

    if resume:
        done = _usable_shards(shard_dir, want, by_smear)
        stale = {q.stem for q in shard_dir.glob("*.parquet")} - done
        if stale:
            progress(f"  {len(stale)} shards do not match the current inputs "
                     f"(different fields, floor, image size or checkpoint) and "
                     f"will be recomputed")
    else:
        done = set()
        for q in list(shard_dir.glob("*.parquet")) + list(shard_dir.glob("*.json")):
            q.unlink()

    todo = [sm for sm in sorted(by_smear) if sm not in done]
    progress(f"  {len(by_smear)} smears, {len(paths):,} fields; "
             f"{len(done)} reusable, {len(todo)} to infer")

    model = loader = None
    if todo:
        model, loader = load_model(weights, meta.get("format"), yolov5_repo)
        meta["loader"] = loader
    meta.update(n_images=len(paths), n_smears=len(by_smear), batch=int(batch),
                half=bool(half), imgsz=int(imgsz), device=str(device),
                gpu=budget.get("name"))
    (out_dir / f"checkpoint_meta_{checkpoint}.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    # one filter for the whole loop: ultralytics logs the half deprecation on
    # every predict call, which is thousands of lines burying the progress
    lg = logging.getLogger("ultralytics")
    quiet = _SilenceDeprecatedHalf()
    lg.addFilter(quiet)
    t0 = time.time()
    for k, smear in enumerate(todo, 1):
        group = by_smear[smear]
        rows = _infer_group(model, group, smear, checkpoint, class_map,
                            conf_floor, imgsz, batch, device, half, progress)
        _write_atomic(_shard_frame(rows), shard_dir / f"{smear}.parquet")
        (shard_dir / f"{smear}.json").write_text(
            json.dumps(_fingerprint(group, conf_floor, imgsz, class_map,
                                    meta["sha256"]), indent=1), encoding="utf-8")
        if k % 10 == 0 or k == len(todo):
            rate = k / max(time.time() - t0, 1e-9)
            progress(f"  {k}/{len(todo)} smears  "
                     f"({rate * 60:.1f}/min, ~{(len(todo) - k) / rate / 60:.1f} min left)")

    lg.removeFilter(quiet)
    _release(model)

    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise IntegrityError(
            f"no shards were written under {shard_dir}",
            remedy=("Every smear failed, or the image directory held nothing "
                    "this loader recognised. Re-run with a single smear and "
                    "read the loader's own output before repeating the pass."),
            images_root=str(images_root))
    if len(shards) != len(by_smear):
        raise IntegrityError(
            f"{len(shards)} shards for {len(by_smear)} smears in {images_root}",
            remedy=("A cache that does not cover the image set would give "
                    "every per-patient denominator a different meaning from "
                    "the one the manifest declares. Re-run detection for this "
                    "checkpoint; shards that are already complete are reused."),
            shard_dir=str(shard_dir))
    df = pd.concat([pd.read_parquet(q) for q in shards], ignore_index=True)
    if df.empty:
        df = _shard_frame([])
    man = save_cache(df, out_dir, checkpoint, checkpoint_sha=meta["sha256"],
                     conf_floor=conf_floor, imgsz=imgsz, nms="none", iou=None,
                     seed=seed)
    covered = int(df["image_id"].nunique())
    progress(f"  cached {man.n_detections:,} detections; {covered:,} of "
             f"{len(paths):,} fields produced at least one box, across "
             f"{len(shards)} smears")

    why = implausible_yield(man.n_detections, covered, len(paths), conf_floor)
    flag = out_dir / f"detections_{checkpoint}.suspect.json"
    if why:
        flag.write_text(json.dumps(
            {"checkpoint": checkpoint, "reason": why, "loader": meta.get("loader"),
             "format": meta.get("format"), "weights": str(weights),
             "checkpoint_sha256": meta["sha256"], "conf_floor": float(conf_floor),
             "imgsz": int(imgsz), "n_detections": int(man.n_detections),
             "fields_with_boxes": covered, "n_fields": len(paths)},
            indent=1), encoding="utf-8")
        progress(f"  ! {why}\n"
                 f"  ! this cache is marked suspect and the analysis stages "
                 f"will refuse it. A checkpoint this silent has not been "
                 f"loaded or driven correctly; that is a bug to find, not a "
                 f"result to report. Delete {flag.name} only if you have "
                 f"established otherwise.")
    elif flag.exists():
        flag.unlink()
    return out_dir


def _release(model) -> None:
    """Give the card back between checkpoints. On 6 GB this is not optional."""
    try:
        del model
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _infer_group(model, group, smear, checkpoint, class_map, conf_floor,
                 imgsz, batch, device, half, progress) -> list[dict]:
    """One smear, with an out-of-memory retry that halves the batch.

    `stream=True` makes ultralytics yield one Results at a time instead of
    materialising the whole batch, which is the difference between fitting and
    not fitting on a small card at this detection ceiling.
    """
    rows: list[dict] = []
    b = int(batch)
    while True:
        try:
            rows.clear()
            for i in range(0, len(group), b):
                chunk = group[i:i + b]
                kw = dict(source=[str(q) for q in chunk], imgsz=imgsz,
                          conf=conf_floor, iou=1.0, agnostic_nms=False,
                          max_det=MAX_DET, verbose=False, device=device,
                          stream=True)
                if half:
                    # passed only when wanted: ultralytics 8.4 deprecates the
                    # kwarg, and passing half=False still tripped the warning
                    # on every batch of a CPU run.
                    kw["half"] = True
                for pth, res in zip(chunk, model.predict(**kw)):
                    bx = res.boxes
                    if bx is None or len(bx) == 0:
                        continue
                    xywhn = _as_array(bx.xywhn)
                    cls = _as_array(bx.cls).astype(int)
                    conf = _as_array(bx.conf)
                    for (x, y, w, h), c, sc in zip(xywhn, cls, conf):
                        name = class_map.get(int(c))
                        if name is None:
                            continue
                        rows.append({"patient_id": smear, "image_id": pth.stem,
                                     "checkpoint": checkpoint, "cls": name,
                                     "score": float(sc), "x": float(x),
                                     "y": float(y), "w": float(w), "h": float(h)})
            return rows
        except TypeError as e:
            if half and "half" in str(e):
                # a future ultralytics that removes the kwarg outright
                half = False
                continue
            raise
        except Exception as e:                  # noqa: BLE001
            if not _is_oom(e) or b <= 1:
                raise
            b = max(1, b // 2)
            progress(f"    out of memory on {smear}; retrying at batch {b}")
            _empty_cache()


def _is_oom(e: Exception) -> bool:
    text = f"{type(e).__name__}: {e}".lower()
    return "out of memory" in text or "cuda_error_out_of_memory" in text \
        or type(e).__name__ == "OutOfMemoryError"


def _empty_cache() -> None:
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


def _shard_frame(rows: list[dict]) -> pd.DataFrame:
    """A typed frame even when a smear yielded no detection above the floor."""
    cols = ["patient_id", "image_id", "checkpoint", "cls",
            "score", "x", "y", "w", "h"]
    if not rows:
        return pd.DataFrame({c: pd.Series(dtype="float64" if c in
                                          ("score", "x", "y", "w", "h") else "object")
                             for c in cols})
    return pd.DataFrame(rows, columns=cols)
