"""Find the downloaded archive, check the machine, and drive the pipeline.

Everything else in this package takes explicit paths, which is right for a
library and tedious for a person who has just unzipped 44.8 GB. This module
takes one root directory and works out the rest, then runs the stages in the
order the protocol requires, with the gates enforced rather than advisory.

Design rules, all of them learned the hard way:

  * Discover, then report, then act. Every path this module infers is printed
    before anything expensive happens, because a wrong guess about which .pt
    file is "the" YOLOv8 checkpoint is not recoverable after the fact.
  * Never skip a gate silently. The reproduction gate and the design gate can
    fail, and a failure is a finding, not an error to be worked around.
  * Assume Windows. The released archive is most often unzipped on a laptop;
    nothing here shells out, and every path is a `pathlib.Path`.
"""

from __future__ import annotations

import json
import platform
import os
import shutil
import sys
import warnings
from dataclasses import dataclass, field, asdict
from pathlib import Path

#: Directory-name fragments that identify each released checkpoint. Order
#: matters: "rt-detr" must be tested before a bare "detr" would match anything.
CHECKPOINT_HINTS = (
    ("rtdetr", ("rt-detr", "rtdetr", "rt_detr")),
    ("yolov5x", ("yolov5", "yolo-v5", "yolo_v5", "v5x")),
    ("yolov8x", ("yolov8", "yolo-v8", "yolo_v8", "v8x")),
)

WEIGHT_SUFFIXES = (".pt", ".pth")

#: Roughly what the caches and shards need. The images are already on disk;
#: this is headroom for parquet output, not for the archive itself.
DISK_HEADROOM_GB = 4.0


@dataclass
class Discovery:
    root: Path
    archive: Path | None = None
    image_list: Path | None = None
    classes: Path | None = None
    weights: dict = field(default_factory=dict)
    image_sets: list = field(default_factory=list)
    label_sets: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.archive and self.image_list and self.weights)

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("root", "archive", "image_list", "classes"):
            d[k] = str(d[k]) if d[k] else None
        d["weights"] = {k: str(v) for k, v in self.weights.items()}
        return d


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #

def walk(root: Path, max_depth: int = 6):
    """One depth-pruned, symlink-following, loop-safe pass over the tree.

    Replaces four independent `rglob` traversals. On a 74k-entry fixture those
    cost 890 ms against 52 ms for a single walk, and the released tree is about
    twice that -- on a cold external drive with a virus scanner in the path the
    difference is minutes, and `discover()` runs before any stage, including
    `--stages paper`.

    `rglob` also refuses to descend into symlinked directories, so an archive
    kept on another volume and linked in -- the ordinary Linux layout -- was
    invisible: the manifest was found (it is a file symlink) and the archive was
    not, which reads as an arbitrary failure. `os.walk(followlinks=True)` fixes
    that, and a visited-inode set keeps it terminating.
    """
    root = Path(root)
    seen: set = set()
    base = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        here = Path(dirpath)
        try:
            st = here.stat()
            key = (st.st_dev, st.st_ino)
        except OSError:
            dirnames[:] = []
            continue
        if st.st_ino and key in seen:
            # st_ino == 0 means this filesystem exposes no stable ids (FAT and
            # some network shares); deduping on (dev, 0) would prune the whole
            # tree after the first directory, so loop protection is skipped
            # there -- os.walk itself does not revisit without symlink cycles.
            dirnames[:] = []
            continue
        seen.add(key)
        if len(here.parts) - base >= max_depth:
            dirnames[:] = []
        yield here, dirnames, filenames


def _shallowest(paths, root: Path):
    """Prefer the copy nearest the root: unzipping twice is common."""
    return min(paths, key=lambda p: (len(p.relative_to(root).parts), str(p)))


def discover(root: Path, max_depth: int = 6) -> Discovery:
    """Work out where everything is, from one directory."""
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"not a directory: {root}")
    d = Discovery(root=root)

    archives, lists, classes, weights = [], [], [], []
    for here, dirnames, filenames in walk(root, max_depth):
        names = set(dirnames)
        if "images" in names and "labels" in names:
            archives.append(here)
        for f in filenames:
            low = f.lower()
            if low.startswith("image_list.") and low.rsplit(".", 1)[-1] in \
                    ("xlsx", "xls", "csv"):
                lists.append(here / f)
            elif low == "classes.txt":
                classes.append(here / f)
            elif Path(low).suffix in WEIGHT_SUFFIXES:
                weights.append(here / f)

    if archives:
        d.archive = _shallowest(archives, root)
        if len(archives) > 1:
            d.notes.append(
                f"{len(archives)} directories look like the archive; using the "
                f"shallowest. Others: {[str(c) for c in archives if c != d.archive][:3]}")
        d.image_sets = sorted(p.name for p in (d.archive / "images").iterdir() if p.is_dir())
        d.label_sets = sorted(p.name for p in (d.archive / "labels").iterdir() if p.is_dir())
    else:
        d.notes.append("no directory containing both images/ and labels/ was "
                       "found (symlinked directories are followed, so this "
                       "means the layout differs from the released one)")

    if lists:
        d.image_list = _shallowest(lists, root)
    else:
        d.notes.append("image_list.xlsx not found; pass --image-list explicitly")

    if classes:
        d.classes = _shallowest(classes, root)
    else:
        d.notes.append("classes.txt not found; class ids will be taken from the "
                       "checkpoints themselves")

    d.weights, notes = _match_weights(weights, root)
    d.notes.extend(notes)
    missing = [n for n, _ in CHECKPOINT_HINTS if n not in d.weights]
    if missing:
        d.notes.append(f"no weights found for: {missing}. Pass them with "
                       "--weights name=path.")
    return d


def _match_weights(paths, root: Path) -> tuple[dict, list]:
    """Assign weight files to checkpoint names, and say so when it is a guess.

    The hint is matched against the path RELATIVE TO ROOT. Matching the absolute
    path let the unzip directory's own name decide: with the download extracted
    into `yolov5-malaria-france/`, every checkpoint matched "yolov5" first, the
    YOLOv8 weights were cached as `detections_yolov5x.parquet`, and the only
    warning said YOLOv8 weights were missing -- which reads as "the archive
    ships two", inviting the user to patch it and leave the swap in place.
    """
    buckets: dict[str, list] = {}
    for p in paths:
        try:
            hay = p.relative_to(root).as_posix().lower()
        except ValueError:
            hay = p.name.lower()
        for name, hints in CHECKPOINT_HINTS:
            if any(h in hay for h in hints):
                buckets.setdefault(name, []).append(p)
                break

    out, notes = {}, []
    for name, cand in buckets.items():
        best = [p for p in cand if p.stem.lower() == "best"]
        pool = best or cand
        pool = sorted(pool, key=lambda p: (len(p.relative_to(root).parts),
                                           -p.stat().st_size, str(p)))
        out[name] = pool[0]
        if len(pool) > 1:
            notes.append(
                f"{len(pool)} candidates for {name}; using {out[name]}. "
                f"Others: {[f'{p} ({p.stat().st_size / 1e6:.0f} MB)' for p in pool[1:4]]}. "
                "Override with --weights if this is wrong.")
    return out, notes


def discover_weights(root: Path) -> dict[str, Path]:
    """Match .pt files to checkpoint names by the directory they sit in.

    `best.pt` wins over `last.pt`; then the copy nearest the root, which is what
    `_shallowest` does for the archive and the manifest and what this used to
    skip -- with the download unzipped twice, the weights came from the copy
    that was NOT chosen as the archive.
    """
    root = Path(root)
    paths = [Path(dp) / f for dp, _, fs in walk(root) for f in fs
             if Path(f).suffix.lower() in WEIGHT_SUFFIXES]
    return _match_weights(paths, root)[0]


# --------------------------------------------------------------------------- #
# The machine
# --------------------------------------------------------------------------- #

def yolov5_error_hint(text) -> str:
    """Name the known cause behind an import failure, when it is one of these.

    setuptools 82.0 removed pkg_resources, which every yolov5 release before
    the packaging switch imports at module scope -- and `pip install yolov5`
    upgrades setuptools on the way in, so installing the fork is what breaks
    the fork. That is worth saying outright rather than leaving to a search.
    """
    t = str(text or "")
    if "pkg_resources" in t:
        return ("pkg_resources was removed in setuptools 82.0 and yolov5 "
                "imports it. Put it back with:  pip install \"setuptools<82\"")
    if "numpy" in t and ("dtype" in t or "_ARRAY_API" in t or "multiarray" in t):
        return ("this is the numpy 1.x/2.x ABI split: a package in the stack "
                "was built against the other major version. Rebuild or pin "
                "whichever the traceback names.")
    if "cv2" in t or "opencv" in t.lower():
        return ("two OpenCV builds in one environment shadow each other. Keep "
                "one:  pip uninstall -y opencv-python opencv-python-headless  "
                "then install just the headless build.")
    return ""


def _probe_yolov5():
    """Whether the packaged fork imports here, and the real reason if not.

    The released yolov5 checkpoint is repository-format and can only be loaded
    through the fork's modules or a clone of the upstream repository. Finding
    that out an hour into the GPU stage, after the other two checkpoints have
    run, is the expensive way; the doctor says it in seconds. `not installed`
    and `installed but its own imports fail` need different fixes, so they are
    reported differently.
    """
    import importlib.util
    try:
        if importlib.util.find_spec("yolov5") is None:
            return None, "not installed (pip install yolov5)"
    except Exception as e:
        return None, f"{type(e).__name__}: {e}"
    try:
        import yolov5
        return getattr(yolov5, "__version__", "present"), None
    except Exception as e:
        return None, (f"installed, but importing it raises "
                      f"{type(e).__name__}: {e}")


def environment() -> dict:
    """What is installed, and is there a GPU. Reported before anything costly."""
    env = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()} {platform.release()}",
        "executable": sys.executable,
    }
    for mod in ("numpy", "pandas", "pyarrow", "matplotlib", "openpyxl",
                "scipy", "sklearn", "torch", "ultralytics"):
        try:
            m = __import__(mod)
            env[mod] = getattr(m, "__version__", "present")
        except Exception:
            env[mod] = None
    env["yolov5"], env["yolov5_error"] = _probe_yolov5()
    try:
        import torch
        env["cuda_available"] = bool(torch.cuda.is_available())
        env["cuda_devices"] = [torch.cuda.get_device_name(i)
                               for i in range(torch.cuda.device_count())]
        env["torch_cuda"] = getattr(torch.version, "cuda", None)
    except Exception:
        env["cuda_available"] = False
        env["cuda_devices"] = []
    env["pdflatex"] = shutil.which("pdflatex")
    return env


def disk_free_gb(path: Path) -> float:
    try:
        return shutil.disk_usage(Path(path)).free / 1e9
    except Exception:
        return float("nan")


def preflight(d: Discovery, need_gpu: bool) -> list[str]:
    """Blocking problems, returned as sentences. Empty means go."""
    env = environment()
    problems = []
    if not d.archive:
        problems.append("The archive was not found. Pass --archive with the "
                        "directory that holds images/ and labels/.")
    if not d.image_list:
        problems.append("image_list.xlsx was not found. It carries the patient "
                        "grouping and the reference standard; nothing works "
                        "without it.")
    if d.image_list and d.image_list.suffix.lower() in (".xlsx", ".xls") \
            and not env.get("openpyxl"):
        problems.append("openpyxl is not installed and the manifest is a "
                        "spreadsheet. Run: pip install openpyxl")
    for mod in ("numpy", "pandas", "matplotlib", "scipy", "sklearn"):
        if not env.get(mod):
            name = "scikit-learn" if mod == "sklearn" else mod
            problems.append(f"{name} is not installed. Run: pip install -r requirements.txt")
    if need_gpu:
        if not env.get("torch"):
            problems.append("torch is not installed, and inference needs it. "
                            "Install the build that matches your CUDA version "
                            "from pytorch.org, then: pip install ultralytics")
        if not env.get("ultralytics"):
            problems.append("ultralytics is not installed. Run: pip install ultralytics")
        if not d.weights:
            problems.append("No checkpoint weights were found under the root.")
    free = disk_free_gb(d.root)
    if free == free and free < DISK_HEADROOM_GB:
        problems.append(f"Only {free:.1f} GB free on the volume holding the "
                        f"archive; the caches and shards want about "
                        f"{DISK_HEADROOM_GB:.0f} GB.")
    return problems


def format_report(d: Discovery, env: dict | None = None) -> str:
    env = env or environment()
    L = [f"root            {d.root}",
         f"archive         {d.archive or '-- NOT FOUND --'}",
         f"image sets      {', '.join(d.image_sets) or '--'}",
         f"label sets      {', '.join(d.label_sets) or '--'}",
         f"image list      {d.image_list or '-- NOT FOUND --'}",
         f"classes.txt     {d.classes or '-- not found, will use checkpoint names --'}",
         "weights"]
    if d.weights:
        for name, p in sorted(d.weights.items()):
            L.append(f"  {name:<10}  {p}  ({p.stat().st_size / 1e6:.0f} MB)")
    else:
        L.append("  -- NONE FOUND --")
    L += ["",
          f"python          {env['python']}  ({env['platform']})",
          f"torch           {env.get('torch') or 'not installed'}"
          f"   cuda={env.get('cuda_available')}"
          f"   {', '.join(env.get('cuda_devices') or []) or ''}",
          f"ultralytics     {env.get('ultralytics') or 'not installed'}",
          f"yolov5 (fork)   {env.get('yolov5') or 'not importable'}"
          f"{'' if env.get('yolov5') else '  <- ' + str(env.get('yolov5_error'))}",
          f"pandas/pyarrow  {env.get('pandas')} / {env.get('pyarrow')}",
          f"scipy/sklearn   {env.get('scipy')} / {env.get('sklearn')}",
          f"openpyxl        {env.get('openpyxl') or 'not installed'}",
          f"pdflatex        {env.get('pdflatex') or 'not on PATH (the PDF step will be skipped)'}",
          f"free disk       {disk_free_gb(d.root):.1f} GB"]
    if d.notes:
        L += ["", "notes"] + [f"  - {n}" for n in d.notes]
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# State, so a long run can be resumed and audited
# --------------------------------------------------------------------------- #

class State:
    """A small JSON journal beside the outputs. Not a database."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.data = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError) as e:
                backup = self.path.with_suffix(".corrupt.json")
                self.path.replace(backup)
                warnings.warn(
                    f"the run journal at {self.path} was unreadable ({e}); it "
                    f"has been moved to {backup} and a fresh one started. "
                    "Finished stages will be re-run.", RuntimeWarning)

    def get(self, key, default=None):
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=1, default=str), encoding="utf-8")
        os.replace(tmp, self.path)     # atomic: a crash cannot truncate it
        return value

    def done(self, key) -> bool:
        return bool(self.data.get(key, {}).get("ok"))
