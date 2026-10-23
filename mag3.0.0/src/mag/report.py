"""Results out, as macros a manuscript can read and cannot contradict.

Protocol v3. Three rules govern this module, and the third is new.


Two rules govern this module.

First, nothing is written without a usable provenance. A results file that
cannot name the checkpoint hashes and inference shards it descends from is
indistinguishable, once it sits beside a manuscript, from one that can.

Second, every directional claim leaves here as a finished sentence. A paper
that types "sensitivity fell outside the margin" beside a macro that renders
whatever the run produced will contradict itself the moment the run changes,
and one did: an abstract said the co-primary endpoint was met while the results
section said it was not, and two reviewers spent pages on the discrepancy. The
outcome sentences below are generated from the same objects as the tables, so
the two cannot disagree.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .provenance import Provenance
from .estimand import SENSITIVITY, SPECIFICITY
from .study import CANDIDATE, COMPARATOR, Study

ARM_NAMES = {"rtdetr": "RT-DETR", "yolov5x": "YOLOv5x", "yolov8x": "YOLOv8x"}


def _pct(x, dp=1):
    return "---" if x is None or not np.isfinite(x) else f"{100 * float(x):.{dp}f}\\%"


def _num(x, dp=3):
    return "---" if x is None or not np.isfinite(x) else f"{float(x):.{dp}f}"


def _p(x):
    if x is None or not np.isfinite(x):
        return "---"
    return "$<$0.0001" if x < 1e-4 else f"{float(x):.4f}"


def _count(x):
    """A count that may be a median, so may be a half. `1` beats `1.0`."""
    if x is None or not np.isfinite(x):
        return "---"
    v = float(x)
    return f"{v:.0f}" if abs(v - round(v)) < 1e-9 else f"{v:.1f}"


def _ci(lo, hi, dp=1, scale=100):
    if lo is None or not np.isfinite(lo):
        return "---"
    return f"{scale * lo:.{dp}f} to {scale * hi:.{dp}f}"


#: Characters that change the meaning of a LaTeX line if they arrive raw.
#: `%` is the dangerous one: it opens a comment, so an unescaped per-cent sign
#: inside a generated sentence deletes the rest of that line from the PDF
#: silently. Sentences composed in this module already write `\%`; sentences
#: that arrive from the analysis modules -- which are plain prose and should
#: not have to know they will end up in LaTeX -- do not.
_TEX_ESCAPES = ((r"(?<!\\)%", r"\\%"), (r"(?<!\\)&", r"\\&"),
                (r"(?<!\\)#", r"\\#"), (r"(?<!\\)_", r"\\_"))


def _tex(value: str) -> str:
    r"""Make one macro body safe to typeset, without double-escaping.

    Regions between dollar signs are left alone. Sentences composed here carry
    real mathematics -- `$\hat\Delta_{\mathrm{sens}}$` -- and escaping the
    subscript underscore inside it turns the operator into a literal glyph, so
    the manuscript printed the estimator's name with a visible underscore and
    `sens` sitting at full size on the baseline. No LaTeX error, no warning,
    and only that one macro affected.
    """
    import re

    parts = re.split(r"(\$[^$]*\$)", str(value))
    for i, part in enumerate(parts):
        if part.startswith("$") and part.endswith("$") and len(part) > 1:
            continue                      # math mode: the author meant it
        for pattern, replacement in _TEX_ESCAPES:
            part = re.sub(pattern, replacement, part)
        parts[i] = part
    return "".join(parts)


def _macro_name(value: str) -> str:
    r"""A checkpoint name, reduced to something a `\defresult` key may hold.

    Macro NAMES are interpolated into the key itself, where `_tex` cannot
    help: `\defresult{accR0rt_detr}` breaks the parse, rather than merely
    setting a glyph wrongly. Everything outside the ASCII alphanumerics is
    dropped.

    Digits are KEPT. Stripping them collides `yolov5x` with `yolov8x`, which
    are two of the three checkpoints this study runs, and the caller's
    injectivity check would then refuse a perfectly ordinary cache.
    """
    import re

    out = re.sub(r"[^A-Za-z0-9]", "", str(value))
    return out or "arm"


def _join(items):
    items = [str(i) for i in items]
    if len(items) <= 1:
        return items[0] if items else "none"
    return ", ".join(items[:-1]) + " and " + items[-1]


def _arm(name):
    return ARM_NAMES.get(str(name), str(name))


def _coprimary_verdict(c: dict) -> str:
    """The co-primary result, with each clause reading its OWN component.

    The version this replaces read only the conjunction and then hard-coded
    which half had failed, so whenever the SPECIFICITY component was the one
    that failed it printed "the specificity component was met and
    non-inferiority of sensitivity was not demonstrated" -- both halves false,
    machine-generated, and therefore trusted more than the typed prose it was
    written to replace. Kept as its own function so the four cases can be
    tested without building a study.
    """
    comp = c["iu"]["components"]
    spec_ok = bool(comp.get("specificity_superior"))
    sens_ok = bool(comp.get("sensitivity_non_inferior"))
    p = _p(c["mcnemar"]["p"])
    spec = ("superiority on specificity was "
            + ("demonstrated" if spec_ok else "not demonstrated")
            + f" (exact $p$ = {p})")
    sens = ("non-inferiority of sensitivity was "
            + ("demonstrated" if sens_ok else "not demonstrated"))
    return (f"{spec} and {sens}, so the intersection--union "
            + ("primary endpoint was met" if c["iu"]["reject"]
               else "primary endpoint was not established"))


def macros(s: Study) -> dict:
    """Every quantity the manuscript names, plus the sentences it must not type."""
    out: dict[str, str] = {}
    fac = s.factorial

    # --- what the analysis rests on ----------------------------------------
    arms = sorted(set(fac["checkpoint"]))
    out["armsAnalysed"] = _join([_arm(a) for a in arms])
    out["nArms"] = str(len(arms))
    out["armsExcluded"] = (_join([_arm(a) for a in sorted(s.excluded)])
                           if s.excluded else "none")
    out["nArmsExcluded"] = str(len(s.excluded))
    out["armsExcludedWhy"] = ("; ".join(f"{_arm(k)}: {v}"
                                        for k, v in sorted(s.excluded.items()))
                              if s.excluded else "no checkpoint was excluded")
    out["condition"] = str(s.meta.get("condition", ""))

    # --- identifiability ----------------------------------------------------
    ident = s.identifiability
    out["identifiable"] = "1" if ident.passed else "0"
    out["identifiabilityNote"] = (
        ident.message or "uninfected patients span more than one centre, so "
        "specificity is not collinear with domain")
    out["nUninfected"] = str(ident.detail.get("n_uninfected", 0))
    out["uninfectedCentres"] = _join(ident.detail.get("centres", []))

    # --- reconstruction -------------------------------------------------------
    rep = s.reconstruction
    out["reconN"] = str(rep.detail["n_correct"])
    out["reconTotal"] = str(rep.detail["n_total"])
    out["reconPublished"] = str(rep.detail["published"])
    out["reconDirection"] = rep.detail["direction"]
    out["reconVerdict"] = ("reconstructible from the released artefacts"
                       if rep.passed else
                       "not reconstructible from the released artefacts")
    out["reconOK"] = "1" if rep.passed else "0"
    sens, spec = rep.detail.get("sensitivity"), rep.detail.get("specificity")
    where = []
    if sens is not None and np.isfinite(sens):
        where.append("sensitivity " + _pct(sens) +
                     (" --- no infected patient was missed" if sens >= 0.999 else ""))
    if spec is not None and np.isfinite(spec):
        where.append("specificity " + _pct(spec) +
                     (" --- every uninfected patient was called infected"
                      if spec <= 0.001 else ""))
    out["reconWhere"] = "; ".join(where) if where else "---"

    # --- the factorial ------------------------------------------------------
    seen_keys = {}
    for ck in arms:
        key = _macro_name(ck)
        if key in seen_keys:
            raise ValueError(
                f"checkpoints {seen_keys[key]!r} and {ck!r} reduce to the same "
                f"macro name {key!r}; rename one before building the manuscript")
        seen_keys[key] = ck
        row = fac[(fac.checkpoint == ck) & (fac.rule == COMPARATOR)]
        if row.empty:
            continue
        r = row.iloc[0]
        out[f"specR0{key}"] = f"{int(r.tn)}/{int(r.tn + r.fp)} ({_pct(r.specificity)})"
        out[f"sensR0{key}"] = _pct(r.sensitivity)
        out[f"accR0{key}"] = _pct(r.accuracy)
    real = fac[~fac.rule.str.startswith("B")]
    # the best cell of the factorial is the best cell among arms whose
    # parameters were fixed in advance. A cross-fitted arm can top this table
    # and be quoted with no caveat, which is the comparison `_apportion` goes
    # out of its way to exclude and `fittedSentence` goes out of its way to
    # qualify -- it must not reappear here unlabelled.
    fixed_only = (real[~real["fitted"].astype(bool)]
                  if "fitted" in real.columns else real)
    best = fixed_only.loc[fixed_only["accuracy"].idxmax()]
    out["bestCell"] = f"{_arm(best.checkpoint)} with {best.rule}"
    out["bestCellAcc"] = _pct(best.accuracy)
    out["bestCellIsFitted"] = "0"
    out["oracle"] = _pct(max(s.bounds["oracle"].values()))
    out["trivial"] = _pct(max(s.bounds["trivial"].values()))

    # --- design -------------------------------------------------------------
    out["confCheckpoint"] = _arm(s.design["selected"])
    out["selectionRule"] = s.design["selection_rule"].replace("_", " ")
    out["headroomList"] = _join([f"{_arm(k)} {v}"
                                 for k, v in sorted(s.design["headroom"].items())])
    out["confFamilySize"] = str(len(s.design["family"]))

    # --- the co-primary endpoint, as arithmetic and as a sentence -----------
    c = s.confirmatory
    mc, ni = c["mcnemar"], c["noninferiority"]
    out["primaryB"], out["primaryC"] = str(mc["b"]), str(mc["c"])
    out["primaryP"] = _p(mc["p"])
    out["primaryAlternative"] = mc["alternative"]
    out["specConfBase"] = _pct(c["spec_base"])
    out["specConfNew"] = _pct(c["spec_new"])
    out["specDelta"] = _pct(c["spec_delta"])
    out["specDeltaCi"] = _ci(*c["spec_delta_ci"])
    out["sensConfBase"] = _pct(c["sens_base"])
    out["sensConfNew"] = _pct(c["sens_new"])
    out["niDiff"] = f"{100 * ni['diff']:.1f}"
    out["niLower"] = f"{100 * ni['one_sided_lower']:.1f}"
    # the margin is stored signed, as the bound the lower limit must clear.
    # It is printed unsigned with the direction in words, because "margin -5.0"
    # beside "the bound lies inside the margin" reads as a contradiction.
    out["niMargin"] = f"{abs(100 * ni['margin']):.1f}"
    out["niMet"] = "1" if ni["non_inferior"] else "0"
    out["iuReject"] = "1" if c["iu"]["reject"] else "0"
    out["niStatement"] = (
        f"$\\hat\\Delta_{{\\mathrm{{sens}}}} = {100 * ni['diff']:.1f}$ "
        f"percentage points, one-sided 95\\% lower bound "
        f"{100 * ni['one_sided_lower']:.1f} against a non-inferiority margin of "
        f"{abs(100 * ni['margin']):.1f} percentage points; "
        + ("the bound lies inside the margin, so non-inferiority was "
           "demonstrated" if ni["non_inferior"] else
           "the bound lies outside the margin, so non-inferiority was not "
           "demonstrated"))
    out["primaryVerdict"] = _coprimary_verdict(c)
    out["primaryFailed"] = (
        _join([f.replace("_", " ") for f in c["iu"]["failed"]])
        if c["iu"]["failed"] else "no component failed")

    # --- the sensitivity analysis on the selection criterion ----------------
    sel = s.design.get("selection_detail") or {}
    out["selAltCheckpoint"] = _arm(sel.get("by_max_discordance") or "---")
    out["selAgree"] = ("1" if sel.get("agree") else
                       "0" if sel.get("agree") is False else "---")
    sa = c.get("sensitivity_analysis") or {}
    out["selSentence"] = (
        "the pre-registered criterion chose the checkpoint on which the "
        "published rule leaves the most room to improve. That criterion is not "
        "level-preserving --- headroom contains the count the test statistic is "
        "built from --- and "
        + ("a criterion that is, maximum discordance, selects the same "
           "checkpoint, so the inflation cannot have affected this result"
           if sa.get("same_as_primary") else
           f"a criterion that is, maximum discordance, selects "
           f"{_arm(sa.get('checkpoint'))} instead; that comparison is reported "
           "beside the primary one"))
    if not sa.get("same_as_primary") and sa.get("mcnemar"):
        out["selAltB"] = str(sa["mcnemar"]["b"])
        out["selAltC"] = str(sa["mcnemar"]["c"])
        out["selAltP"] = _p(sa["mcnemar"]["p"])
        out["selAltSpecDelta"] = _pct(sa["spec_delta"])

    out.update(_estimands(s))
    out.update(_transform(s))
    out.update(_attribution(s))
    out.update(_stress(s))
    out.update(_predictive(s))
    out.update(_loco(s))
    out.update(_calibration(s))
    out.update(_speciation_calibration(s))
    out.update(_selective(s))
    out.update(_diagnostics(s))
    out.update(_power(s))
    out.update(_nulls(s))
    out.update(_sweep(s))
    out.update(_fitted(s))

    # --- apportionment ------------------------------------------------------
    ap = s.apportionment or {}
    if ap:
        out["spreadRule"] = _pct(ap["spread_rule"])
        out["spreadCkpt"] = _pct(ap["spread_checkpoint"])
        out["pairwiseRule"] = _pct(ap["pairwise_rule_median"])
        out["pairwiseCkpt"] = _pct(ap["pairwise_checkpoint_median"])
        # `perm_null_rule_p95` is deliberately None: there is no permutation
        # that removes a rule effect without removing the values it is made of.
        # A macro that quietly rendered "0.0%" there would be read as a null of
        # zero rather than as the absence of one.
        out["permNullRule"] = _pct(ap.get("perm_null_rule_p95"))
        out["permNullCkpt"] = _pct(ap.get("perm_null_checkpoint_p95"))
        out["ruleOverCkpt"] = "1" if ap["rule_exceeds_checkpoint"] else "0"
        out["ruleExceedsNull"] = (
            "was not compared against a permutation null, which this design "
            "does not admit for the rule factor"
            if ap.get("perm_null_rule_p95") is None else
            "exceeded" if ap["spread_rule"] > ap["perm_null_rule_p95"]
            else "did not exceed")
        jack = ap.get("leave_one_checkpoint_out") or {}
        out["jackknifeArms"] = str(len(jack))
        out["jackknifeStable"] = (
            "did not change" if ap.get("ordering_stable") else
            "changed" if jack else "could not be assessed with "
            f"{ap['n_checkpoints']} checkpoints")
        out["apportionmentSentence"] = (
            f"the spread across rules ({_pct(ap['spread_rule'])}) "
            + ("exceeded" if ap["rule_exceeds_checkpoint"] else "did not exceed")
            + f" the spread across checkpoints ({_pct(ap['spread_checkpoint'])}); "
            "both are descriptive ranges over factors with different numbers of "
            f"levels ({ap['n_rules']} against {ap['n_checkpoints']}), and the "
            "median absolute pairwise difference, which does not grow with the "
            f"number of levels, was {_pct(ap['pairwise_rule_median'])} between "
            f"two rules against {_pct(ap['pairwise_checkpoint_median'])} between "
            "two checkpoints")
        boot = ap.get("bootstrap") or {}
        if boot.get("difference_ci"):
            lo, hi = boot["difference_ci"]
            out["spreadDiff"] = _pct(boot["difference"])
            out["spreadDiffCi"] = _ci(lo, hi)
            out["spreadDiffExcludesZero"] = "1" if boot["excludes_zero"] else "0"
            out["apportionmentInterval"] = (
                f"the difference between the two spreads was "
                f"{_pct(boot['difference'])} (95\\% bootstrap interval over "
                f"patients {_ci(lo, hi)} percentage points), which "
                + ("excludes zero" if boot["excludes_zero"] else
                   "includes zero, so the ordering of the two spreads is not "
                   "resolved by this cohort"))
        if ap.get("perm_null_checkpoint_p") is not None:
            out["permCkptP"] = _p(ap["perm_null_checkpoint_p"])
        out["permNullNote"] = ap.get("perm_null_note", "")
        out["nFittedExcluded"] = str(ap.get("n_fitted_rules_excluded", 0))
    return out


# --------------------------------------------------------------------------- #
# The reviewer-requested sections. One function each, so that a macro can be
# traced to the analysis that produced it without reading the whole module.
# --------------------------------------------------------------------------- #

def _estimands(s: Study) -> dict:
    """The scopes, as macros, so the manuscript cannot quote a number without one."""
    out = {
        "scopeSpecificity": SPECIFICITY.scope,
        "scopeSensitivity": SENSITIVITY.scope,
        "specificityIdentifiable": "1" if SPECIFICITY.identifiable else "0",
        "specificityWhyNot": SPECIFICITY.why_not,
        "specificityRequiredForm": SPECIFICITY.required_form,
    }
    check = getattr(s, "cohort_check", None) or {}
    out["scopeStillHolds"] = "1" if check.get("specificity_scope_still_holds") else "0"
    out["negativeCentres"] = _join(check.get("negative_centres", [])) or "none"
    out["estimandSentence"] = (
        "Specificity in this archive is not a site-invariant quantity: "
        + SPECIFICITY.why_not
        + ". Sensitivity is conditioned on fields selected to contain a "
          "parasite, so it is an estimate for the archive's own sampling "
          "regime and not for screening.")
    return out


def _transform(s: Study) -> dict:
    """The source study's zoom correction, as a count rather than an argument.

    Closes the reconciliation the previous manuscript could not: the archive
    ships `test` and `test-zoom` as two label trees over the same smears, so the
    retained fraction of annotations is measurable. `mag ingest` counts it and
    writes it beside the patient table; this reads it.

    When it was never counted, every macro here says so in words the manuscript
    can print. It does not fall back to a plausible fraction, because a retained
    fraction nobody counted is exactly the kind of number the previous build
    was reviewed on.
    """
    tf = getattr(s, "transform", None) or {}
    if not tf.get("measured"):
        why = tf.get("why", "the two label trees were never counted")
        return {
            "tfMeasured": "0",
            "tfKept": "NOT MEASURED",
            "tfParasiteKept": "NOT MEASURED",
            "annBase": "NOT MEASURED",
            "annAfterTf": "NOT MEASURED",
            "annRecovered": "NOT MEASURED",
            "tfVerdict": f"not measured: {why}",
            "tfSentence": (
                "The transformation between the two released label trees was "
                f"not measured in this run ({why}). No retained fraction is "
                "reported."),
        }
    base, zoom = int(tf.get("boxes_base", 0)), int(tf.get("boxes_zoom", 0))
    pbase, pzoom = int(tf.get("parasite_base", 0)), int(tf.get("parasite_zoom", 0))
    out = {
        "tfMeasured": "1",
        "tfBaseSet": str(tf.get("base_set", "")),
        "tfZoomSet": str(tf.get("zoom_set", "")),
        "tfKept": _pct(tf.get("all_kept")),
        "tfParasiteKept": _pct(tf.get("parasite_kept")),
        "tfOtherKept": _pct(tf.get("other_kept")),
        "annBase": f"{base:,}",
        "annAfterTf": f"{zoom:,}",
        "annRecovered": f"{zoom - base:,}" if zoom >= base else f"-{base - zoom:,}",
        "annParasiteBase": f"{pbase:,}",
        "annParasiteAfterTf": f"{pzoom:,}",
        "tfFieldsBase": f"{int(tf.get('fields_base', 0)):,}",
        "tfFieldsZoom": f"{int(tf.get('fields_zoom', 0)):,}",
        "tfVerdict": str(tf.get("verdict", "")),
    }
    out["tfSentence"] = (
        f"Counting annotations in both released label trees, the "
        f"{tf.get('zoom_set', 'zoomed')} tree retained "
        f"{out['tfParasiteKept']} of the {pbase:,} parasite boxes and "
        f"{out['tfKept']} of the {base:,} boxes overall across "
        f"{out['tfFieldsBase']} fields. This is a direct count, not an "
        f"inference from the images: {tf.get('verdict', '')}.")
    return out


def _attribution(s: Study) -> dict:
    """§9: the model, not the ranges. The ranges live in the apportionment."""
    a = getattr(s, "attribution", None)
    if a is None:
        return {}
    out = {"attrNRules": str(a.n_rules), "attrNCheckpoints": str(a.n_checkpoints),
           "attrNPatients": str(a.n_patients)}
    if not a.contrast:
        out["attributionSentence"] = a.note or (
            "the factorial model was not fitted; no attribution is reported")
        out["attrFitted"] = "0"
        return out
    out["attrFitted"] = "1"
    c = a.contrast
    # The primary contrast is on the PROBABILITY scale and is always defined;
    # the log-odds form is reported only when no level is separated. The
    # reconstructed published rule calls every uninfected patient infected, so
    # separation is the expected state rather than an edge case.
    out["attrContrast"] = _pct(c["estimate"])
    out["attrContrastCi"] = _ci(c["lo"], c["hi"])
    out["attrExcludesZero"] = "1" if c["excludes_zero"] else "0"
    out["attrRuleSpread"] = _pct(c["rule_spread"])
    out["attrCkptSpread"] = _pct(c["checkpoint_spread"])
    out["attrNBoot"] = str(c["n_boot"])
    lo = getattr(a, "contrast_logodds", {}) or {}
    out["attrLogOddsReported"] = "1" if lo.get("reported") else "0"
    if lo.get("reported"):
        out["attrContrastLogOdds"] = _num(lo["estimate_logodds"], 3)
        out["attrContrastLogOddsCi"] = f"{lo['lo']:.3f} to {lo['hi']:.3f}"
    else:
        out["attrLogOddsWhy"] = lo.get("why", "not reported")
    sep = a.separation or {}
    out["attrSeparated"] = "1" if sep.get("any") else "0"
    out["attrNSeparatedLevels"] = str(sep.get("n_separated_levels", 0))
    out["attrSeparatedLevels"] = _join(
        [d["level"] for d in sep.get("rules", []) + sep.get("checkpoints", [])]
    ) or "none"
    if a.rule_effect:
        out["attrRuleSpreadLogOdds"] = _num(a.rule_effect.get("spread_logodds"), 3)
    if a.checkpoint_effect:
        out["attrCkptSpreadLogOdds"] = _num(a.checkpoint_effect.get("spread_logodds"), 3)
    inter = a.interaction or {}
    if inter:
        out["attrInteractionChi2"] = _num(inter.get("wald_chi2"), 2)
        out["attrInteractionDf"] = str(inter.get("n_terms", ""))
        out["attrInteractionP"] = _p(inter.get("p"))
        out["attrInteractionDetected"] = "1" if inter.get("detected") else "0"
    ag = a.agreement or {}
    out["attrAgreeChecked"] = "1" if ag.get("checked") else "0"
    out["attrAgree"] = "1" if ag.get("agree") else "0"
    if ag.get("checked"):
        out["attrAgreeGap"] = _num(ag.get("max_abs_coefficient_difference"), 6)
    else:
        out["attrAgreeWhy"] = ag.get("why", "not checked")
    out["attributionSentence"] = a.sentence
    out["attributionCaveat"] = (
        "the marginal effects come from a patient-clustered generalised "
        "estimating equation with an exchangeable working correlation and "
        "robust standard errors; the descriptive ranges that accompany them "
        "are ranges over factors with different numbers of levels and are not "
        "variance components")
    return out


def _stress(s: Study) -> dict:
    """§13: does the ordering of the rules survive field dilution?"""
    st = getattr(s, "stress", None)
    if st is None or not st.ordering:
        return {"stressSentence": (getattr(st, "note", "") or
                                   "the field-sampling stress test was not run")}
    o = st.ordering
    out = {
        "stressRates": _join([f"{r:g}" for r in st.rates]),
        "stressMinTau": _num(o.get("min_tau"), 2),
        "stressStable": "1" if o.get("stable") else "0",
        "stressInverted": "1" if o.get("inverted") else "0",
        "stressTopChanges": "1" if o.get("top_rule_changes") else "0",
        "stressTopRules": _join(o.get("top_rules_seen", [])),
        "stressDonorMismatch": str(
            (st.donor_mismatch or {}).get("n_with_no_same_centre_donor", 0)),
        "stressNDonorFields": f"{st.n_donor_fields:,}",
        "stressSentence": st.sentence,
    }
    lowest = min(float(k) for k in o["by_rate"])
    for rid, rates in st.by_rule.items():
        low = rates.get(f"{lowest:g}") or {}
        top = rates.get(f"{max(st.rates):g}") or {}
        if low and top:
            out[f"stressSens{_macro_name(rid)}"] = (
                f"{_pct(top['sensitivity'])} at pi=1 to "
                f"{_pct(low['sensitivity'])} at pi={lowest:g}")
    return out


def _predictive(s: Study) -> dict:
    """§14: predictive values at prevalences a reader chooses."""
    pv = getattr(s, "predictive", None)
    if pv is None or not pv.at:
        return {}
    out = {"pvSentence": pv.sentence, "pvScope": pv.scope,
           "pvCohortPrevalence": _pct(pv.cohort_prevalence)}
    for key, v in pv.at.items():
        tag = "Cohort" if key == "cohort" else _macro_name(f"p{key}")
        out[f"ppv{tag}"] = _pct(v["ppv"])
        out[f"npv{tag}"] = _pct(v["npv"])
        if np.isfinite(v["ppv_ci"][0]):
            out[f"ppv{tag}Ci"] = _ci(*v["ppv_ci"])
    return out


def _loco(s: Study) -> dict:
    """§15: leave-one-centre-out, on speciation only."""
    l = getattr(s, "loco", None) or {}
    if not l.get("by_centre"):
        return {}
    out = {
        "locoOverall": _pct(l.get("overall_speciation_accuracy")),
        "locoMaxSwing": _pct(l.get("max_swing")),
        "locoNCentres": str(l.get("n_interpretable_centres", 0)),
        "locoSentence": (
            "Holding each centre out in turn, speciation accuracy moved by at "
            f"most {_pct(l.get('max_swing'))} from its all-centre value of "
            f"{_pct(l.get('overall_speciation_accuracy'))} across the "
            f"{l.get('n_interpretable_centres', 0)} centres with enough "
            "infected patients to estimate it. " + l.get("note", "")),
    }
    return out


def _speciation_calibration(s: Study) -> dict:
    """§11.B: the species posterior, never pooled with the infection decision."""
    sc = getattr(s, "speciation_calibration", None)
    if sc is None or not sc.n:
        return {}
    return {
        "specBrier": _num(sc.brier, 4),
        "specBrierFormula": sc.brier_formula,
        "specCalN": str(sc.n),
        "specTopAccuracy": _pct(sc.top_label_accuracy),
        "specMeanTopProb": _num(sc.mean_top_probability, 4),
        "specUninfectedMass": sc.uninfected_mass,
        "specCalNote": sc.note,
        "specCalSentence": (
            f"Among the {sc.n} infected patients carrying any parasite "
            f"evidence, the species posterior had a multiclass Brier score of "
            f"{_num(sc.brier, 4)} ({sc.brier_formula}) and its top label was "
            f"correct for {_pct(sc.top_label_accuracy)}. This is reported "
            "separately from the infection calibration above and is never "
            "pooled with it: the two are different questions on different "
            "denominators."),
    }


def _calibration(s: Study) -> dict:
    cal = getattr(s, "calibration", None) or {}
    out = {}
    names = {"raw": "Raw", "isotonic": "Iso", "platt": "Platt",
             "temperature": "Temp"}
    best, best_ece = None, float("inf")
    for key, suffix in names.items():
        rep = cal.get(key)
        if not isinstance(rep, dict):
            continue
        out[f"brier{suffix}"] = _num(rep["brier"], 4)
        out[f"ece{suffix}"] = _num(rep["ece"], 4)
        out[f"slope{suffix}"] = _num(rep["slope"], 3)
        out[f"intercept{suffix}"] = _num(rep["intercept"], 3)
        out[f"ici{suffix}"] = _num(rep["ici"], 4)
        if key != "raw" and np.isfinite(rep["ece"]) and rep["ece"] < best_ece:
            best, best_ece = key, rep["ece"]
        if key != "raw":
            out[f"foldsDegenerate{suffix}"] = str(rep.get("folds_degenerate", 0))
    raw = cal.get("raw")
    if isinstance(raw, dict):
        out["eceNull"] = _num(raw["null_reference"], 4)
        out["calFolds"] = str((cal.get("platt") or {}).get("folds", "---"))
        direction = ("too extreme" if raw["slope"] < 1 else "too timid")
        # "reduced" is a claim about a comparison, so it is made by comparing.
        # The version this replaces said "reduced" unconditionally, including
        # on a run where every fold degenerated and all four rows carried the
        # identical error.
        improved = best is not None and best_ece < raw["ece"] - 1e-12
        out["calibrationSentence"] = (
            f"the uncalibrated screening statistic had a Brier score of "
            f"{_num(raw['brier'], 4)} and an expected calibration error of "
            f"{_num(raw['ece'], 4)} against a reference of "
            f"{_num(raw['null_reference'], 4)} for a perfectly calibrated "
            f"predictor at this sample size; its calibration slope of "
            f"{_num(raw['slope'], 3)} indicates scores that are {direction}"
            + ((f", and the best of the three cross-fitted maps ({best}) "
                + ("reduced" if improved else "did not reduce")
                + f" the calibration error, to {_num(best_ece, 4)}")
               if best is not None else ""))
        out["calibrationImproved"] = "1" if improved else "0"
    out["calibrationCaveat"] = (
        "calibration acts on the screening statistic alone. It is a monotone "
        "map on one number, so it can move the infected/uninfected boundary and "
        "cannot reorder the species; no speciation result here is attributable "
        "to it.")
    return out


def _selective(s: Study) -> dict:
    sel = {k: v for k, v in (getattr(s, "selective", None) or {}).items()
           if isinstance(v, dict) and "aurc" in v}
    if not sel:
        return {}
    out = {}
    for rid, rc in sel.items():
        out[f"aurc{rid}"] = _num(rc["aurc"], 4)
        for target in ("0.95", "0.9", "0.8"):
            op = (rc.get("operating") or {}).get(target)
            if op:
                tag = target.replace("0.", "")
                out[f"risk{tag}{rid}"] = _pct(op["risk"])
                out[f"referred{tag}{rid}"] = str(op["n_referred"])
    base = sel.get(COMPARATOR) or {}
    cand = sel.get(CANDIDATE) or {}
    op90 = (base.get("operating") or {}).get("0.9") or {}
    if op90:
        out["selectiveSentence"] = (
            f"referring the {op90['n_referred']} least confident patients "
            f"(10\\% of the cohort) left a risk of {_pct(op90['risk'])} among "
            "those still answered automatically"
            + (f", against an area under the risk--coverage curve of "
               f"{_num(base.get('aurc'), 4)} for the published rule and "
               f"{_num(cand.get('aurc'), 4)} for the counting rule"
               if cand else "")
            + (". A lower area is better, and the comparison is meaningful "
               "because both rules are ranked by the same confidence"))
    out["selectiveCaveat"] = (
        "the coverage threshold is taken by rank rather than by quantile value: "
        "dozens of patients sit at exactly zero confidence and no quantile "
        "separates them, which in the predecessor produced a reported risk at "
        "90\\% coverage that was the full-cohort risk with zero patients "
        "referred.")
    return out


def _diagnostics(s: Study) -> dict:
    out = {}
    es = getattr(s, "error_summary", None) or {}
    if es.get("n_errors"):
        out["errN"] = str(es["n_errors"])
        out["errTopCategory"] = str(es["top_category"]).replace("_", " ")
        out["errTopCategoryShare"] = _pct(es["top_category_share"])
        out["errTopPred"] = str(es["top_predicted"]).replace("_", " ")
        out["errTopPredShare"] = _pct(es["top_predicted_share"])
        out["errOvaleVivax"] = str((es.get("categories") or {}).get("ovale_vivax", 0))
        out["errMedianFragility"] = _num(es.get("median_fragility"), 1)
        out["errMedianFragilityFP"] = _num(
            es.get("median_fragility_false_positive"), 1)
        out["errFlippedByTwo"] = _pct(es.get("share_flipped_by_two_boxes"))
        out["errSentence"] = (
            f"of {es['n_errors']} misclassified patients, "
            f"{_pct(es['top_category_share'])} fell into a single category "
            f"({str(es['top_category']).replace('_', ' ')}) and "
            f"{_pct(es['top_predicted_share'])} were called "
            f"{str(es['top_predicted']).replace('_', ' ')}; the median false "
            "positive would have been called correctly after removing "
            f"{_count(es.get('median_fragility_false_positive'))} of its "
            "detections, and "
            f"{_pct(es.get('share_flipped_by_two_boxes'))} of them would have "
            "turned on two boxes or fewer")
    census = getattr(s, "box_census", None)
    if census is not None and len(census):
        out["boxRawRatio"] = _num(census.attrs.get("ratio_raw"), 1)
        out["boxMatchedRatio"] = _num(census.attrs.get("ratio_matched"), 1)
        for r in census.itertuples(index=False):
            key = _macro_name(r.checkpoint)
            out[f"boxRaw{key}"] = _num(r.raw_per_field, 1)
            out[f"boxMatched{key}"] = _num(r.matched_per_field, 1)
        fixed = [str(r.checkpoint) for r in census.itertuples(index=False)
                 if getattr(r, "fixed_budget", False)]
        out["fixedBudgetArms"] = _join([_arm(f) for f in fixed]) if fixed else "none"
        out["boxSentence"] = (
            f"before suppression the busiest checkpoint emitted "
            f"{_num(census.attrs.get('ratio_raw'), 1)} times as many boxes per "
            "field as the quietest; after identical suppression the ratio was "
            f"{_num(census.attrs.get('ratio_matched'), 1)}"
            + (f". {_join([_arm(f) for f in fixed])} emitted exactly the same "
               "number of boxes on every field, which is a fixed query budget "
               "and not a detection count" if fixed else ""))
    dom = getattr(s, "domain", None) or {}
    if dom and np.isfinite(dom.get("difference_in_differences", float("nan"))):
        out["ddExtInt"] = _pct(dom["difference_in_differences"])
        out["ddCi"] = _ci(*dom["ci"])
        out["ddNInternal"] = str(dom["n_internal"])
        out["ddNExternal"] = str(dom["n_external"])
        out["ddSentence"] = (
            "on speciation, the candidate's advantage over the baseline "
            f"differed between internal and external centres by "
            f"{_pct(dom['difference_in_differences'])} (95\\% interval "
            f"{_ci(*dom['ci'])} percentage points, "
            f"{dom['n_internal']} internal against {dom['n_external']} external "
            "patients). Specificity cannot appear in this contrast: every "
            "uninfected smear in the cohort comes from one centre")
    fc = getattr(s, "field_curve", None) or {}
    if fc.get("by_rule"):
        out["fieldCurveN"] = str(fc.get("n_eligible", 0))
        out["fieldCurveMin"] = str(fc.get("min_fields_required", 0))
        by = fc["by_rule"].get(COMPARATOR) or {}
        for k in (5, 10, 25):
            v = by.get(k) or by.get(str(k))
            if v:
                out[f"fields{k}"] = _pct(v["mean"])
        five = (by.get(5) or by.get("5") or {}).get("mean")
        top = max((v["mean"] for v in by.values()), default=None)
        if five is not None and top is not None:
            out["fieldCurveSentence"] = (
                f"over the {fc['n_eligible']} patients contributing at least "
                f"{fc['min_fields_required']} fields, the published rule "
                f"reached {_pct(five)} on five fields against {_pct(top)} on "
                "the full subsample examined")
    elif fc.get("note"):
        out["fieldCurveSentence"] = fc["note"]
    return out


def _power(s: Study) -> dict:
    p = getattr(s, "power", None) or {}
    spec, sens = p.get("specificity") or {}, p.get("sensitivity") or {}
    out = {}
    if spec:
        out["powerNDiscordant"] = str(spec.get("n_discordant", 0))
        out["powerCritB"] = (str(spec["critical_b_at_realised_discordance"])
                             if spec.get("critical_b_at_realised_discordance")
                             is not None else "no attainable value")
        out["powerMinDiscordant"] = str(
            spec.get("min_discordant_pairs_that_can_reject", "---"))
        det = spec.get("detectable") or {}
        out["powerDetectableOdds"] = _num(det.get("odds"), 1)
        out["powerSpecSentence"] = spec.get("sentence", "")
        rn = (spec.get("required_n_for_3_to_1") or {}).get("n")
        out["powerRequiredN"] = str(rn) if rn else "more than 5000"
    if sens:
        out["powerNiAtZero"] = _num(sens.get("power_at_zero_true_difference"), 2)
        det = sens.get("detectable") or {}
        out["powerNiDetectable"] = _pct(abs(det["delta"])) if det.get(
            "delta") is not None and np.isfinite(det.get("delta", float("nan"))) \
            else "not attainable at this sample size"
        out["powerSensSentence"] = sens.get("sentence", "")
    agree = p.get("method_agreement") or {}
    if agree:
        out["powerApproxAgrees"] = "1" if agree.get("agrees") else "0"
        out["powerApproxGap"] = _num(agree.get("abs_difference"), 4)
    out["powerCaveat"] = (
        "these are detectable effects, not observed power. Observed power is a "
        "monotone function of the p-value and would restate the test result as "
        "though it explained it.")
    return out


def _nulls(s: Study) -> dict:
    n = getattr(s, "nulls", None) or {}
    out = {}
    wc = n.get("winners_curse") or {}
    detail = wc.get("detail") or {}
    if detail:
        out["wcBestCell"] = str(detail.get("best_cell", "---")).replace("_", " ")
        out["wcObserved"] = _pct(wc.get("observed"))
        out["wcNull"] = _pct(wc.get("null_mean"))
        out["wcP"] = _p(wc.get("p_value"))
        out["wcShrunkBest"] = _pct(detail.get("shrunk_best"))
        out["wcSentence"] = detail.get("sentence", "")
    cf = n.get("crossfit") or {}
    if cf.get("k10"):
        out["cfOptimism"] = _pct(cf["k10"]["mean_apparent_gain"])
        out["cfOptimismP95"] = _pct(cf["k10"]["p95_apparent_gain"])
        if cf.get("loo"):
            out["cfLooOptimism"] = _pct(cf["loo"]["mean_apparent_gain"])
            out["cfLooPenalty"] = _pct(cf.get("loo_penalty"))
        out["cfSentence"] = (
            "on a score constructed to carry no information, the best of the "
            "three cross-fitted calibrators still appeared to gain "
            f"{_pct(cf['k10']['mean_apparent_gain'])} over the best constant "
            f"predictor at 10 folds (95th centile "
            f"{_pct(cf['k10']['p95_apparent_gain'])}); an observed fitted-rule "
            "gain must clear that before it is called an improvement")
    sn = n.get("selection") or {}
    if sn.get("headroom"):
        out["selSizeHeadroom"] = _pct(sn["headroom"]["size_after_selection"], 2)
        out["selSizeDiscordance"] = _pct(sn["discordance"]["size_after_selection"], 2)
        out["selSizeFixed"] = _pct(sn["fixed"]["size_after_selection"], 2)
        out["selSizeSentence"] = sn.get("verdict", "")
        out["selSizeDegenerate"] = "1" if sn.get("degenerate") else "0"
    elif sn.get("note"):
        out["selSizeSentence"] = sn["note"]
    return out


def _sweep(s: Study) -> dict:
    sw = getattr(s, "sweep_summary", None) or {}
    if not sw:
        return {}
    out = {"sweepN": str(sw.get("n_configurations", 0)),
           "sweepHits": str(sw.get("n_reproducing", 0)),
           "sweepAnyReproduces": "1" if sw.get("any_reproduces") else "0",
           # named, because the sweep and the reproduction gate can run on
           # different checkpoints and a manuscript that prints both totals
           # without saying which detector produced each is comparing nothing
           "sweepCheckpoint": _arm(sw.get("checkpoint", "---")),
           "sweepSentence": sw.get("sentence", "")}
    closest = sw.get("closest") or {}
    if closest:
        out["sweepClosest"] = (
            f"{closest['rule']} at {closest['theta_key']}"
            f"={closest['theta']:g} under {closest['condition']}")
        out["sweepClosestN"] = f"{closest['n_correct']}/{sw.get('n_total', '')}"
        out["sweepClosestGap"] = str(closest["gap"])
    agreement = sw.get("agreement_at_target") or {}
    if agreement:
        lo, hi = agreement["specificity_range"]
        out["sweepSpecRange"] = _ci(lo, hi)
        lo, hi = agreement["sensitivity_range"]
        out["sweepSensRange"] = _ci(lo, hi)
        out["sweepAgreementSentence"] = (
            "of the configurations that reach the published total, the "
            f"specificity ranged over {out['sweepSpecRange']} and the "
            f"sensitivity over {out['sweepSensRange']} percentage points, so "
            "reaching the headline number is not the same as reproducing the "
            "result behind it")
    return out


def _fitted(s: Study) -> dict:
    fitted = getattr(s, "fitted", None) or {}
    if not fitted:
        return {}
    ids = sorted({rid for _, rid in fitted})
    out = {"fittedArms": _join(ids), "nFittedArms": str(len(ids))}
    degenerate = sorted({rid for (_, rid), d in fitted.items()
                         if d.get("folds_fitted", 0) == 0})
    out["fittedDegenerate"] = _join(degenerate) if degenerate else "none"
    out["fittedAllDegenerate"] = "1" if len(degenerate) == len(ids) else "0"
    # a fitted arm every one of whose folds fell back to the identity map is
    # the baseline under another name, and the sentence below must not call it
    # an improvement
    out["fittedWarning"] = (
        f"every fold of {_join(degenerate)} degenerated to the identity map, so "
        "the arm is the published rule under another name and its accuracy is "
        "not evidence about calibration" if degenerate else
        "no fitted arm degenerated: every fold carried both classes and was "
        "calibrated")
    fac = s.factorial
    if "fitted" in fac.columns and bool(fac["fitted"].any()):
        sub = fac[fac["fitted"].astype(bool)]
        best = sub.loc[sub["accuracy"].idxmax()]
        fixed = fac[(~fac["fitted"].astype(bool)) & (~fac.rule.str.startswith("B"))]
        best_fixed = fixed.loc[fixed["accuracy"].idxmax()]
        out["fittedBest"] = f"{_arm(best.checkpoint)} with {best.rule}"
        out["fittedBestAcc"] = _pct(best.accuracy)
        out["fittedGain"] = _pct(best.accuracy - best_fixed.accuracy)
        cf = ((getattr(s, "nulls", None) or {}).get("crossfit") or {}).get("k10")
        if degenerate and best.rule in degenerate:
            out["fittedSentence"] = (
                f"the nominally best fitted arm ({best.rule} on "
                f"{_arm(best.checkpoint)}) reached {_pct(best.accuracy)}, but "
                + out["fittedWarning"])
            return out
        out["fittedSentence"] = (
            f"the best cross-fitted rule ({best.rule} on "
            f"{_arm(best.checkpoint)}) reached {_pct(best.accuracy)} against "
            f"{_pct(best_fixed.accuracy)} for the best rule with parameters "
            f"fixed in advance, a difference of "
            f"{_pct(best.accuracy - best_fixed.accuracy)}"
            + (f", to be read against the {_pct(cf['mean_apparent_gain'])} such "
               "a procedure gains on a score carrying no information at all"
               if cf else "")
            + ". No exact test is reported against any fitted arm: its "
              "parameters come from the patients it is scored on, so its "
              "discordant pairs are not exchangeable with the baseline's")
    return out


def write(s: Study, path, *, extra: dict | None = None) -> dict:
    """Write results.tex, or refuse and say why."""
    prov: Provenance = s.provenance
    if prov is None:
        from .errors import ProvenanceError
        raise ProvenanceError(
            "this study carries no provenance record",
            remedy="Build it with study.run, which attaches one.")
    prov.require_usable()

    keys = macros(s)
    keys.update(extra or {})
    keys["provenance"] = prov.line()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["% Generated by `mag report`. Do not edit.",
             f"% {prov.line()}",
             "% A key the analysis never computed stays red in the PDF rather",
             "% than vanishing, so an unfilled slot cannot pass silently.",
             ""]
    lines += [f"\\defresult{{{k}}}{{{_tex(keys[k])}}}" for k in sorted(keys)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return keys
