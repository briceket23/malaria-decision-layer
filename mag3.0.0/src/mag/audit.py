"""The check that stands between a results object and a PDF.

Protocol v3 §19. It runs after the analysis and before LaTeX, and a failure is
an error rather than a warning. Every check here exists because the defect it
catches reached three external reviewers in the previous build:

  1. unresolved macros rendered as `[armsAnalysed]` and were read as data;
  2. the abstract and the conclusion stated opposite endpoint verdicts;
  3. figures and numbers came from different runs;
  4. an area-under-risk-coverage appeared as 0.027 in the text and 0.471 in a
     figure;
  5. a reported sensitivity did not follow from its own 2x2;
  6. a Brier score printed as `0.000`, where 0.004 and 0.040 look identical;
  7. results carried no hash that tied them to a run;
  8. the declarations were placeholders.

Check 8 is the one to be clear about: **it never invents a declaration.** It
verifies that ethics, funding, CRediT, AI-use, repository, OSF and Zenodo
identifiers are non-empty and fails if they are not. Those are author-supplied
and always will be. A pipeline that filled them in would be manufacturing the
one part of a paper that must come from a person.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Metrics that must never be printed at a precision where two materially
#: different values collapse to the same string.
PRECISION_FLOOR = {"brier": 4, "ece": 4, "ici": 4, "aurc": 4,
                   "null_reference": 4}

#: Author-supplied declarations. Present and non-empty, or no PDF.
REQUIRED_DECLARATIONS = (
    "ethics_source_study", "ethics_present_study", "consent_for_publication",
    "funding", "author_contributions", "ai_use", "repository_url",
    "osf_registration", "code_archive_doi", "data_availability",
)


@dataclass
class Finding:
    check: str
    severity: str          # "fail" | "warn"
    message: str
    detail: dict = field(default_factory=dict)


@dataclass
class AuditReport:
    findings: list = field(default_factory=list)
    n_checks: int = 0

    @property
    def failed(self) -> list:
        return [f for f in self.findings if f.severity == "fail"]

    @property
    def passed(self) -> bool:
        return not self.failed

    def raise_if_failed(self) -> None:
        from .errors import IntegrityError

        if self.passed:
            return
        lines = "\n".join(f"  [{f.check}] {f.message}" for f in self.failed)
        raise IntegrityError(
            f"the build audit found {len(self.failed)} blocking problem(s):\n{lines}",
            remedy=("Fix them and re-run. This audit is the reason the "
                    "previous build's defects reached three reviewers: it did "
                    "not exist. Do not bypass it -- there is no flag."),
            n_failed=len(self.failed), n_checks=self.n_checks)

    def summary(self) -> str:
        if self.passed:
            return f"build audit: {self.n_checks} checks, all passed"
        return (f"build audit: {len(self.failed)} of {self.n_checks} checks "
                "FAILED")


# --------------------------------------------------------------------------- #

def _macro_bodies(tex: str) -> dict:
    return {m.group(1): m.group(2)
            for m in re.finditer(r"\\defresult\{([^}]*)\}\{(.*)\}", tex)}


def _used_macros(source: str) -> set:
    """Macro names a manuscript actually calls, as `\\R{name}`."""
    return set(re.findall(r"\\R\{([A-Za-z0-9]+)\}", source))


def check_macros(tex_path, manuscript_path=None) -> list:
    """Every macro the manuscript calls is defined, and none is empty."""
    out = []
    tex = Path(tex_path).read_text(encoding="utf-8")
    defined = _macro_bodies(tex)

    empty = sorted(k for k, v in defined.items() if not v.strip())
    if empty:
        out.append(Finding(
            "macros", "fail",
            f"{len(empty)} macro(s) are defined but empty: "
            f"{', '.join(empty[:6])}",
            {"empty": empty}))

    placeholder = sorted(
        k for k, v in defined.items()
        if re.fullmatch(r"\[.*\]|TODO|TBD|xxx+|\?+", v.strip(), re.IGNORECASE))
    if placeholder:
        out.append(Finding(
            "macros", "fail",
            f"{len(placeholder)} macro(s) still hold a placeholder: "
            f"{', '.join(placeholder[:6])}",
            {"placeholder": placeholder}))

    if manuscript_path and Path(manuscript_path).is_file():
        used = _used_macros(Path(manuscript_path).read_text(encoding="utf-8"))
        missing = sorted(used - set(defined))
        if missing:
            out.append(Finding(
                "macros", "fail",
                f"the manuscript calls {len(missing)} macro(s) the analysis "
                f"never defined: {', '.join(missing[:6])}. In the previous "
                "build these rendered as [name] and were read as data",
                {"missing": missing}))
        unused = sorted(set(defined) - used)
        if unused:
            out.append(Finding(
                "macros", "warn",
                f"{len(unused)} macro(s) are computed but never used",
                {"unused": unused[:20]}))
    return out


def check_no_typed_verdict(manuscript_path) -> list:
    """No directional claim is typed where a generated sentence exists.

    The previous build's abstract said the co-primary conjunction was met while
    its conclusion said sensitivity fell outside the margin. Both were typed.
    The generated sentences exist so that cannot happen; this check is what
    stops someone typing one anyway.
    """
    if not manuscript_path or not Path(manuscript_path).is_file():
        return []
    text = Path(manuscript_path).read_text(encoding="utf-8")
    # strip comments and macro calls before looking for prose
    body = re.sub(r"(?<!\\)%.*", "", text)
    body = re.sub(r"\\R\{[A-Za-z0-9]+\}", " ", body)

    banned = [
        (r"co-primary\s+(endpoint\s+)?(was|is)\s+met", "a typed co-primary verdict"),
        (r"conjunction\s+(was|is)\s+met", "a typed conjunction verdict"),
        (r"sensitivity\s+was\s+non-inferior", "a typed non-inferiority verdict"),
        (r"non-inferiority\s+was\s+(not\s+)?(demonstrated|established)",
         "a typed non-inferiority verdict"),
        (r"failed\s+reproduction", "retired vocabulary (protocol v3 §3)"),
        (r"the\s+rule\s+matters\s+more\s+than", "a forbidden attribution claim"),
        (r"the\s+rule\s+controls\s+false\s+positives",
         "a specificity claim outside its estimand"),
    ]
    out = []
    for pattern, why in banned:
        for m in re.finditer(pattern, body, re.IGNORECASE):
            line = body[:m.start()].count("\n") + 1
            out.append(Finding(
                "typed-verdict", "fail",
                f"line {line}: {why} — \"{m.group(0)}\". Use the generated "
                "macro instead",
                {"line": line, "match": m.group(0)}))
    return out


def check_arithmetic(factorial) -> list:
    """Every reported rate follows from its own confusion counts."""
    out = []
    if factorial is None or not len(factorial):
        return out
    for r in factorial.itertuples(index=False):
        tp, fn = int(r.tp), int(r.fn)
        tn, fp = int(r.tn), int(r.fp)
        for name, k, n, got in (("sensitivity", tp, tp + fn, r.sensitivity),
                                ("specificity", tn, tn + fp, r.specificity)):
            if n == 0:
                continue
            want = k / n
            if not np.isfinite(got) or abs(float(got) - want) > 1e-9:
                out.append(Finding(
                    "arithmetic", "fail",
                    f"{r.checkpoint}:{r.rule} reports {name} {float(got):.6f} "
                    f"but its counts give {want:.6f} ({k}/{n})",
                    {"checkpoint": str(r.checkpoint), "rule": str(r.rule)}))
    return out


def check_precision(tex_path) -> list:
    """No metric printed where two materially different values look alike."""
    out = []
    tex = Path(tex_path).read_text(encoding="utf-8")
    for name, body in _macro_bodies(tex).items():
        low = name.lower()
        for metric, places in PRECISION_FLOOR.items():
            if metric not in low:
                continue
            m = re.fullmatch(r"-?\d+\.(\d+)", body.strip())
            if m and len(m.group(1)) < places:
                out.append(Finding(
                    "precision", "fail",
                    f"{name} = {body} is printed to {len(m.group(1))} decimals; "
                    f"{metric} needs {places}. A Brier of 0.000 is "
                    "indistinguishable from 0.0004",
                    {"macro": name, "value": body}))
    return out


def check_cross_artefact(tex_path, results_json) -> list:
    """A number appearing in two artefacts has one value.

    The previous build reported an AURC of 0.027 in the text and 0.471 in a
    figure. Both descended from the same analysis; neither descended from the
    same object.
    """
    out = []
    tex = Path(tex_path).read_text(encoding="utf-8")
    macros = _macro_bodies(tex)
    try:
        res = json.loads(Path(results_json).read_text(encoding="utf-8"))
    except Exception as exc:                                # noqa: BLE001
        return [Finding("cross-artefact", "fail",
                        f"results.json could not be read ({exc})", {})]

    def dig(path, obj):
        for key in path:
            if not isinstance(obj, dict) or key not in obj:
                return None
            obj = obj[key]
        return obj

    # (macro name, path into results.json, scale applied when rendering)
    #
    # Every pair here is REQUIRED to resolve. The first version of this list
    # still read ("reproduction", "n_correct") after the results key had been
    # renamed to "reconstruction", and because an unresolvable path was skipped
    # in silence the check went on reporting success while comparing nothing.
    # That is the same class of defect the audit exists to catch, so a pair
    # that does not resolve on both sides is now a finding in its own right.
    pairs = [
        ("primaryB", ("confirmatory", "mcnemar", "b"), 1),
        ("primaryC", ("confirmatory", "mcnemar", "c"), 1),
        ("reconN", ("reconstruction", "n_correct"), 1),
        ("reconTotal", ("reconstruction", "n_total"), 1),
        ("reconPublished", ("reconstruction", "published"), 1),
        ("attrNPatients", ("attribution", "n_patients"), 1),
    ]
    compared = 0
    for macro, path, scale in pairs:
        body = macros.get(macro)
        want = dig(path, res)
        if body is None or want is None:
            out.append(Finding(
                "cross-artefact", "fail",
                f"{macro} could not be compared: "
                + ("it is not defined in results.tex"
                   if body is None else "")
                + (" and " if body is None and want is None else "")
                + ("results.json has nothing at "
                   f"{'.'.join(path)}" if want is None else "")
                + ". A pair that cannot be compared is a check that is not "
                  "running, not a check that passed",
                {"macro": macro, "path": list(path)}))
            continue
        body = body.strip()
        if not re.fullmatch(r"-?[\d.]+", body):
            out.append(Finding(
                "cross-artefact", "fail",
                f"{macro} is \"{body}\" in results.tex, which is not a number "
                "and cannot be checked against results.json",
                {"macro": macro, "tex": body}))
            continue
        compared += 1
        if abs(float(body) - float(want) * scale) > 1e-6:
            out.append(Finding(
                "cross-artefact", "fail",
                f"{macro} is {body} in results.tex but "
                f"{want} in results.json",
                {"macro": macro, "tex": body, "json": want}))
    if compared == 0:
        out.append(Finding(
            "cross-artefact", "fail",
            "no macro could be compared against results.json at all; the "
            "cross-artefact check is not running", {}))
    return out


def check_provenance(results_json) -> list:
    out = []
    try:
        res = json.loads(Path(results_json).read_text(encoding="utf-8"))
    except Exception as exc:                                # noqa: BLE001
        return [Finding("provenance", "fail",
                        f"results.json could not be read ({exc})", {})]
    meta = res.get("meta") or {}
    for key in ("build", "version", "seed", "condition", "cache"):
        if meta.get(key) in (None, ""):
            out.append(Finding("provenance", "fail",
                               f"the results carry no {key}", {}))
    if not res.get("provenance"):
        out.append(Finding("provenance", "fail",
                           "the results carry no provenance line", {}))
    return out


def check_declarations(declarations) -> list:
    """Author-supplied, checked for presence, never filled in."""
    out = []
    if declarations is None:
        return [Finding(
            "declarations", "fail",
            "no declarations file was supplied. Ethics, funding, CRediT, "
            "AI use, repository and registration identifiers are "
            "author-supplied and this pipeline will not invent them",
            {"required": list(REQUIRED_DECLARATIONS)})]
    if not isinstance(declarations, dict):
        declarations = json.loads(Path(declarations).read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_DECLARATIONS
               if not str(declarations.get(k, "")).strip()]
    if missing:
        out.append(Finding(
            "declarations", "fail",
            f"{len(missing)} declaration(s) are empty: {', '.join(missing)}",
            {"missing": missing}))
    # a declaration that still reads like a slot is not a declaration
    slots = [k for k, v in declarations.items()
             if re.fullmatch(r"\[.*\]|TODO|TBD|N/?A", str(v).strip(),
                             re.IGNORECASE)]
    if slots:
        out.append(Finding(
            "declarations", "fail",
            f"{len(slots)} declaration(s) still hold a placeholder: "
            f"{', '.join(slots)}",
            {"placeholder": slots}))
    return out


def check_figures(figures_dir, provenance_line, seed) -> list:
    """Figures descend from this run, and none failed to draw."""
    from .figures import STAMP

    out = []
    p = Path(figures_dir) / STAMP if figures_dir else None
    if p is None or not p.is_file():
        return [Finding("figures", "fail",
                        f"no {STAMP} beside the figures", {})]
    stamp = json.loads(p.read_text(encoding="utf-8"))
    if stamp.get("provenance") != provenance_line:
        out.append(Finding(
            "figures", "fail",
            "the figures and the results come from different runs",
            {"figures": stamp.get("provenance"), "results": provenance_line}))
    if str(stamp.get("seed")) != str(seed):
        out.append(Finding("figures", "fail",
                           "the figures were drawn at a different seed",
                           {"figures": stamp.get("seed"), "results": seed}))
    if stamp.get("failed"):
        out.append(Finding(
            "figures", "fail",
            f"{len(stamp['failed'])} figure(s) failed to draw",
            {"failed": stamp["failed"]}))
    return out


# --------------------------------------------------------------------------- #

def run(out_dir, *, manuscript=None, declarations=None,
        factorial=None) -> AuditReport:
    """Every check, over a finished results directory."""
    out_dir = Path(out_dir)
    tex = out_dir / "results.tex"
    results = out_dir / "results.json"
    figures = out_dir / "figures"

    report = AuditReport()
    checks = []

    if not tex.is_file():
        report.findings.append(Finding(
            "macros", "fail", f"no results.tex at {tex}", {}))
    else:
        checks += [("macros", lambda: check_macros(tex, manuscript)),
                   ("precision", lambda: check_precision(tex))]
    if results.is_file():
        checks.append(("provenance", lambda: check_provenance(results)))
        if tex.is_file():
            checks.append(("cross-artefact",
                           lambda: check_cross_artefact(tex, results)))
    else:
        report.findings.append(Finding(
            "provenance", "fail", f"no results.json at {results}", {}))

    if factorial is not None:
        checks.append(("arithmetic", lambda: check_arithmetic(factorial)))
    if manuscript:
        checks.append(("typed-verdict",
                       lambda: check_no_typed_verdict(manuscript)))
    checks.append(("declarations", lambda: check_declarations(declarations)))

    if results.is_file() and figures.is_dir():
        try:
            res = json.loads(results.read_text(encoding="utf-8"))
            checks.append(("figures", lambda: check_figures(
                figures, res.get("provenance"), (res.get("meta") or {}).get("seed"))))
        except Exception:                                   # noqa: BLE001
            pass

    for _, fn in checks:
        report.findings.extend(fn())
    report.n_checks = len(checks)
    return report
