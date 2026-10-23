#!/usr/bin/env python3
"""Where do the YOLOv5 boxes disappear? One image, one layer at a time.

    python scripts/probe_v5.py <model.pt> <one-field.jpg>

The released yolov5 checkpoint ran to completion over 4,508 fields and returned
two boxes. That is not a weak detector, so something between the weights and
the cache is dropping everything, and this prints each stage so the stage that
loses them is visible rather than guessed at:

  1. the raw forward pass, before any suppression -- how many anchors carry
     objectness above the floor, and how high the best score gets. If this is
     already empty, the fault is the weights or the preprocessing; if it is
     healthy, the fault is downstream.
  2. the fork's own non-maximum suppression at the study's operating point
     (floor 0.001, IoU 1.0) and at the stock one (0.25 / 0.45). An IoU of 1.0
     is meant to suppress nothing; if the stock setting returns boxes and the
     study's returns none, the suppression is the culprit.
  3. the same call through the pipeline's adapter, which is what actually
     wrote the cache.

Nothing here is written to disk and no cache is touched.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))


def main(argv) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2
    weights, image = Path(argv[1]), Path(argv[2])
    for p in (weights, image):
        if not p.is_file():
            print(f"not a file: {p}")
            return 2

    import numpy as np
    import torch
    from PIL import Image

    print(f"weights  {weights}")
    print(f"image    {image}")

    import yolov5
    print(f"yolov5   {getattr(yolov5, '__version__', '?')}")
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"device   {dev}   torch {torch.__version__}")

    from mag.detect import _full_load
    with _full_load():
        m = yolov5.load(str(weights))
    m.to(dev)
    names = getattr(m, "names", None)
    print(f"classes  {names}")

    # ---------------------------------------------------------------- 1. raw
    im = Image.open(image).convert("RGB")
    print(f"\nimage size {im.size[0]}x{im.size[1]}")
    x = np.asarray(im.resize((640, 640)), dtype=np.float32) / 255.0
    t = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
    with torch.no_grad():
        y = m.model(t)
    raw = y[0] if isinstance(y, (list, tuple)) else y
    print(f"\n1. raw forward   output {tuple(raw.shape)}")
    if raw.ndim == 3 and raw.shape[-1] > 5:
        obj = raw[..., 4]
        cls = raw[..., 5:]
        print(f"   objectness   max {float(obj.max()):.6f}   "
              f"mean {float(obj.mean()):.6f}   "
              f"above 0.001: {int((obj > 0.001).sum()):,} of {obj.numel():,}")
        print(f"   class score  max {float(cls.max()):.6f}")
        print(f"   obj x cls    max {float((obj[..., None] * cls).max()):.6f}")
        if float(obj.max()) < 0.01:
            print("   -> the model itself is silent on this image: the fault is "
                  "upstream of suppression (weights, input size or scaling).")
    else:
        print("   unexpected layout -- this is not the [batch, anchors, 5+nc] "
              "tensor the yolov5 suppression expects, which would explain an "
              "empty result on its own.")

    # ------------------------------------------------------ 2. the fork's NMS
    print("\n2. the fork's own suppression, same image")
    for conf, iou, note in ((0.001, 1.0, "the study's operating point"),
                            (0.001, 0.45, "study floor, stock IoU"),
                            (0.25, 0.45, "stock yolov5 defaults")):
        m.conf, m.iou, m.max_det = float(conf), float(iou), 3000
        m.agnostic, m.classes = False, None
        try:
            r = m([str(image)], size=640)
            n = int(len(r.xywhn[0]))
            top = float(r.xywhn[0][:, 4].max()) if n else float("nan")
            print(f"   conf={conf:<6} iou={iou:<5} -> {n:>5} boxes   "
                  f"best {top:.4f}   ({note})")
        except Exception as e:                                   # noqa: BLE001
            print(f"   conf={conf:<6} iou={iou:<5} -> {type(e).__name__}: {e}")

    # ------------------------------------------------------- 3. the adapter
    print("\n3. through the pipeline's adapter")
    try:
        from mag import detect as infer
        ad = infer.YOLOv5RepoAdapter(weights)
        out = list(ad.predict([str(image)], imgsz=640, conf=0.001, iou=1.0,
                              agnostic_nms=False, max_det=3000,
                              device="0" if dev != "cpu" else "cpu"))
        b = out[0].boxes if out else None
        print(f"   {0 if b is None else len(b)} boxes reached the cache schema")
    except Exception as e:                                       # noqa: BLE001
        print(f"   {type(e).__name__}: {e}")

    print("\nSend this whole output back. Section 1 says whether the weights "
          "are producing anything at all; 2 says whether suppression is eating "
          "it; 3 says whether the adapter is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
