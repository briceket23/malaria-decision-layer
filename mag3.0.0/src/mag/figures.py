"""Figures, generated from the same `Study` object as the text.

The rule this module exists to enforce: a figure and the sentence beside it must
descend from one run. In the predecessor they did not. `analyse` wrote the
numbers, a separate script drew the figures, and the two were run hours apart
against different caches; a reviewer found a figure whose bars disagreed with
the table above them by four percentage points, and neither artefact carried
anything that would have revealed which was stale.

So every function here takes the `Study` and nothing else, `write_all` stamps
`PROVENANCE.txt` with the provenance line and the build id, and `check` refuses
to let a manuscript build against figures whose stamp does not match the results
it is about to typeset.

Matplotlib only, no seaborn, no style sheets: a figure that renders differently
depending on a package's default palette is a figure that cannot be regenerated
in five years.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

STAMP = "PROVENANCE.txt"
FORMATS = ("pdf", "png")
DPI = 200

#: One colour per role, declared once. Chosen to survive greyscale printing and
#: the two most common forms of colour blindness; the pairs are also separated
#: by lightness so that no comparison in any figure depends on hue alone.
INK = {"baseline": "#1f4e79", "candidate": "#c0504d", "oracle": "#404040",
       "trivial": "#909090", "null": "#b0b0b0", "accent": "#e08214",
       "good": "#2b7a3d", "grid": "#dddddd"}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": DPI, "savefig.dpi": DPI, "font.size": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": INK["grid"], "grid.linewidth": 0.5,
        "axes.axisbelow": True, "figure.autolayout": True,
    })
    return plt


def _save(fig, out: Path, name: str) -> list:
    out.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in FORMATS:
        p = out / f"{name}.{ext}"
        fig.savefig(p, bbox_inches="tight")
        written.append(p)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return written


# --------------------------------------------------------------------------- #

def factorial_heatmap(s, out: Path) -> list:
    """Accuracy for every (rule, checkpoint) cell, with the bounds beside it."""
    plt = _mpl()
    fac = s.factorial
    real = fac[~fac.rule.str.startswith("B")]
    cell = real.pivot_table(index="rule", columns="checkpoint", values="accuracy")
    fig, ax = plt.subplots(figsize=(1.6 + 1.1 * cell.shape[1], 0.32 * cell.shape[0] + 1.4))
    im = ax.imshow(cell.to_numpy(), aspect="auto", cmap="viridis",
                   vmin=float(np.nanmin(cell.to_numpy())),
                   vmax=float(np.nanmax(cell.to_numpy())))
    ax.set_xticks(range(cell.shape[1]), list(cell.columns), rotation=0)
    ax.set_yticks(range(cell.shape[0]), list(cell.index))
    ax.grid(False)
    for i in range(cell.shape[0]):
        for j in range(cell.shape[1]):
            v = cell.iloc[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center",
                        fontsize=7,
                        color="white" if v < np.nanmean(cell.to_numpy()) else "black")
    fig.colorbar(im, ax=ax, label="patient-level accuracy (%)", fraction=0.05)
    fitted = sorted(set(real.loc[real.get("fitted", False) == True, "rule"])) \
        if "fitted" in real.columns else []
    ax.set_title("Every cell of the factorial"
                 + (f"\n({', '.join(fitted)} are cross-fitted; no exact test "
                    "is valid against them)" if fitted else ""),
                 fontsize=8)
    return _save(fig, out, "factorial")


def apportionment(s, out: Path) -> list:
    """Spread across rules against spread across checkpoints, with the null."""
    plt = _mpl()
    ap = s.apportionment or {}
    if not ap:
        return []
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    labels = [f"rules\n(n={ap['n_rules']})", f"checkpoints\n(n={ap['n_checkpoints']})"]
    obs = [100 * ap["spread_rule"], 100 * ap["spread_checkpoint"]]
    # None where no null is defined -- the rule factor has none in a design with
    # one observation per cell, and an absent reference must look absent
    null = [ap.get("perm_null_rule_p95"), ap.get("perm_null_checkpoint_p95")]
    pair = [100 * ap["pairwise_rule_median"], 100 * ap["pairwise_checkpoint_median"]]
    boot = (ap.get("bootstrap") or {})
    x = np.arange(2)
    ax.bar(x - 0.18, obs, width=0.34, color=INK["baseline"], label="observed range")
    ax.bar(x + 0.18, pair, width=0.34, color=INK["accent"],
           label="median pairwise difference")
    for i, key in enumerate(("spread_rule_ci", "spread_checkpoint_ci")):
        ci = boot.get(key)
        if ci:
            ax.plot([i - 0.18, i - 0.18], [100 * ci[0], 100 * ci[1]],
                    color=INK["oracle"], lw=1.2, zorder=4,
                    label="bootstrap 95% interval" if i == 0 else None)
    drawn_null = False
    for i, n in enumerate(null):
        if n is None or not np.isfinite(n):
            continue
        ax.plot([i - 0.36, i + 0.36], [100 * n, 100 * n], color=INK["null"],
                lw=1.4, ls="--", zorder=3,
                label="permutation null, 95th centile" if not drawn_null else None)
        drawn_null = True
    ax.set_xticks(x, labels)
    ax.set_ylabel("accuracy spread (percentage points)")
    ax.legend(fontsize=6.5, frameon=False)
    diff = boot.get("difference_ci")
    ax.set_title("A range over {} rules and a range over {} checkpoints\n"
                 "are not the same statistic".format(ap["n_rules"],
                                                     ap["n_checkpoints"])
                 + (f"\ndifference {100 * boot['difference']:.1f} pp "
                    f"(95% CI {100 * diff[0]:.1f} to {100 * diff[1]:.1f})"
                    if diff else ""), fontsize=8)
    return _save(fig, out, "apportionment")


def calibration(s, out: Path) -> list:
    """Reliability diagrams: raw against each cross-fitted map."""
    plt = _mpl()
    cal = getattr(s, "calibration", None) or {}
    if not cal:
        return []
    keys = [k for k in ("raw", "isotonic", "platt", "temperature")
            if isinstance(cal.get(k), dict)]
    if not keys:
        return []
    fig, axes = plt.subplots(1, len(keys), figsize=(2.0 * len(keys), 2.3),
                             sharey=True)
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, keys):
        rep = cal[k]
        detail = rep.get("detail") or ()
        conf = [d["conf"] for d in detail]
        acc = [d["acc"] for d in detail]
        ax.plot([0, 1], [0, 1], color=INK["null"], lw=0.8, ls="--")
        ax.plot(conf, acc, "o-", color=INK["baseline"], ms=3, lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("predicted")
        ax.set_title(f"{k}\nECE {rep['ece']:.3f} (null {rep['null_reference']:.3f})"
                     f"\nslope {rep['slope']:.2f}", fontsize=7)
    axes[0].set_ylabel("observed frequency")
    fig.suptitle("Reliability, equal-mass bins, cross-fitted", fontsize=8, y=1.06)
    return _save(fig, out, "calibration")


def risk_coverage(s, out: Path) -> list:
    """Risk against coverage for the baseline and the confirmatory rule."""
    plt = _mpl()
    sel = {k: v for k, v in (getattr(s, "selective", None) or {}).items()
           if isinstance(v, dict) and "coverage" in v}
    if not sel:
        return []
    fig, ax = plt.subplots(figsize=(3.2, 2.4))
    colours = [INK["baseline"], INK["candidate"], INK["accent"], INK["good"]]
    for (name, rc), colour in zip(sorted(sel.items()), colours):
        cov = np.asarray(rc["coverage"], dtype=float)
        risk = np.asarray(rc["risk"], dtype=float)
        ax.plot(100 * cov, 100 * risk, lw=1.3, color=colour,
                label=f"{name} (AURC {rc['aurc']:.3f})")
    ax.set_xlabel("coverage (% of patients answered)")
    ax.set_ylabel("risk among answered (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    ax.set_title("Does the confidence know when it is wrong?", fontsize=8)
    return _save(fig, out, "risk_coverage")


def box_census(s, out: Path) -> list:
    """Boxes per field, raw against matched, per checkpoint."""
    plt = _mpl()
    census = getattr(s, "box_census", None)
    if census is None or len(census) == 0:
        return []
    fig, ax = plt.subplots(figsize=(0.9 * len(census) + 1.8, 2.4))
    x = np.arange(len(census))
    ax.bar(x - 0.19, census["raw_per_field"], width=0.36,
           color=INK["null"], label="raw")
    ax.bar(x + 0.19, census["matched_per_field"], width=0.36,
           color=INK["baseline"], label="matched (identical suppression)")
    if "fixed_budget" in census.columns:
        for i, fixed in enumerate(census["fixed_budget"]):
            if bool(fixed):
                ax.text(i - 0.19, census["raw_per_field"].iloc[i], "fixed\nbudget",
                        ha="center", va="bottom", fontsize=6, color=INK["candidate"])
    ax.set_xticks(x, list(census["checkpoint"]))
    ax.set_yscale("log")
    ax.set_ylabel("detections per field (log scale)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title("Two detector families are not comparable\nuntil the same "
                 "suppression has been applied", fontsize=8)
    return _save(fig, out, "box_census")


def field_curve(s, out: Path) -> list:
    """Accuracy against the number of fields examined, per rule."""
    plt = _mpl()
    fc = getattr(s, "field_curve", None) or {}
    by_rule = fc.get("by_rule") or {}
    if not by_rule:
        return []
    plt_fig, ax = plt.subplots(figsize=(3.2, 2.4))
    colours = [INK["baseline"], INK["candidate"], INK["accent"], INK["good"],
               INK["oracle"]]
    for (rid, curve), colour in zip(sorted(by_rule.items()), colours):
        ks = sorted(int(k) for k in curve)
        mean = [100 * curve[k]["mean"] if k in curve else 100 * curve[str(k)]["mean"]
                for k in ks]
        lo = [100 * (curve.get(k) or curve[str(k)])["lo"] for k in ks]
        hi = [100 * (curve.get(k) or curve[str(k)])["hi"] for k in ks]
        ax.plot(ks, mean, "o-", ms=3, lw=1.2, color=colour, label=rid)
        ax.fill_between(ks, lo, hi, color=colour, alpha=0.13, lw=0)
    ax.set_xlabel("fields examined per patient")
    ax.set_ylabel("accuracy (%)")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"Fields needed, over {fc.get('n_eligible', 0)} patients with "
                 f"at least {fc.get('min_fields_required', 0)}", fontsize=8)
    return _save(plt_fig, out, "field_curve")


def errors(s, out: Path) -> list:
    """What kind of mistake, and how few boxes would have changed it."""
    plt = _mpl()
    tbl = getattr(s, "errors", None)
    if tbl is None or len(tbl) == 0:
        return []
    wrong = tbl[~tbl["correct"]]
    if wrong.empty:
        return []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.0, 2.4))
    cat = wrong["category"].value_counts()
    ax1.barh(list(cat.index)[::-1], list(cat.values)[::-1], color=INK["baseline"])
    ax1.set_xlabel("patients")
    ax1.set_title("Error taxonomy", fontsize=8)
    frag = wrong.loc[np.isfinite(wrong["fragility"]), "fragility"]
    if len(frag):
        ax2.hist(frag, bins=min(30, max(5, int(frag.max()))),
                 color=INK["candidate"])
        ax2.axvline(2, color=INK["oracle"], ls="--", lw=1)
        ax2.set_xlabel("detections that would have to be removed to change "
                       "the call")
        ax2.set_ylabel("patients")
        ax2.set_title("Fragility", fontsize=8)
    return _save(fig, out, "errors")


def reconstruction_sweep(s, out: Path) -> list:
    """Where the published number sits relative to everything we could get."""
    plt = _mpl()
    sw = getattr(s, "sweep", None)
    if sw is None or len(getattr(sw, "table", [])) == 0:
        return []
    t = sw.table
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    ax.hist(t["n_correct"], bins=range(int(t["n_correct"].min()),
                                       int(t["n_correct"].max()) + 2),
            color=INK["null"])
    ax.axvline(sw.target, color=INK["candidate"], lw=1.4,
               label=f"published {sw.target}/{sw.n_total}")
    best = t.iloc[0]
    ax.axvline(best["n_correct"], color=INK["baseline"], lw=1.4, ls="--",
               label=f"closest reachable {int(best['n_correct'])}")
    ax.set_xlabel(f"patients correct out of {sw.n_total}")
    ax.set_ylabel("configurations")
    ax.legend(fontsize=6.5, frameon=False)
    ax.set_title(f"{len(t)} configurations swept (post hoc)", fontsize=8)
    return _save(fig, out, "reconstruction_sweep")


# --------------------------------------------------------------------------- #

ALL = (factorial_heatmap, apportionment, calibration, risk_coverage,
       box_census, field_curve, errors, reconstruction_sweep)


def write_all(s, out) -> dict:
    """Draw everything the study supports, and stamp what it came from.

    A figure whose inputs are absent is skipped and named in `skipped`, never
    drawn from a default. An empty panel in a manuscript is a missing analysis;
    a fabricated one is worse.
    """
    from . import build_id

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    # `skipped` and `failed` are two different things and were one list. A
    # figure the study has no data for is a legitimate absence; a figure that
    # raised is a coding bug, and `mag all` went on to write the manuscript
    # macros and return 0 with the bug printed among the absences.
    written, skipped, failed = {}, [], []
    for fn in ALL:
        try:
            paths = fn(s, out)
        except Exception as exc:                       # noqa: BLE001
            failed.append(f"{fn.__name__}: {type(exc).__name__}: {exc}")
            continue
        if paths:
            written[fn.__name__] = [str(p.name) for p in paths]
        else:
            skipped.append(f"{fn.__name__}: the study carries no data for it")

    stamp = {
        "build": build_id(),
        "provenance": s.provenance.line() if s.provenance else None,
        "condition": s.meta.get("condition"),
        "cache": s.meta.get("cache"),
        "seed": s.meta.get("seed"),
        "figures": written, "skipped": skipped, "failed": failed,
    }
    (out / STAMP).write_text(json.dumps(stamp, indent=1, default=str) + "\n",
                             encoding="utf-8")
    return stamp


def check(figures_dir, s) -> None:
    """Refuse a manuscript build whose figures came from a different run."""
    from .errors import ProvenanceError

    p = Path(figures_dir) / STAMP
    if not p.exists():
        raise ProvenanceError(
            f"no {STAMP} beside the figures in {figures_dir}",
            remedy="Run `mag figures` so the figures carry the run they "
                   "descend from. A manuscript will not be built against "
                   "figures of unknown origin.")
    stamp = json.loads(p.read_text(encoding="utf-8"))
    live = s.provenance.line() if s.provenance else None
    if stamp.get("provenance") != live:
        raise ProvenanceError(
            "the figures and the results come from different runs",
            remedy="Re-run `mag figures` against this cache before building "
                   "the manuscript.",
            figures=stamp.get("provenance"), results=live)
    # The provenance line does not carry the seed, and two runs of the same
    # cache at different seeds produce different bootstrap intervals and
    # different subsamples. Checked separately rather than folded into the
    # line, which is also printed on the title page.
    if str(stamp.get("seed")) != str(s.meta.get("seed")):
        raise ProvenanceError(
            "the figures and the results were run at different seeds",
            remedy="Re-run `mag figures --seed` to match the results.",
            figures_seed=stamp.get("seed"), results_seed=s.meta.get("seed"))
    if stamp.get("failed"):
        raise ProvenanceError(
            f"{len(stamp['failed'])} figures failed to draw on the run that "
            "produced this stamp",
            remedy=("Fix them and re-run `mag figures`. A manuscript built "
                    "against a figure set with holes in it will typeset "
                    "missing panels without saying so."),
            failed=stamp["failed"])
