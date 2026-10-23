"""The experiment: one pure function from a cache to a set of results.

Protocol v3. `run` takes a cache directory and a patient table and returns a
`Study`. It touches no GPU, downloads nothing, generates nothing, and reads
nothing outside the directory it was given.

The gate order is fixed by the protocol and each gate is evaluated before
anything it could contaminate:

    identifiability -> yield -> declared checkpoint -> factorial
        -> reconstruction -> design -> confirm

Identifiability first because it changes what any later number may claim. Yield
next because a checkpoint that did not detect must leave the factorial before it
can contribute a cell. The confirmatory checkpoint is DECLARED rather than
selected -- v2 chose it by maximum headroom and that inflates the size of the
test that follows (§8.3). Reconstruction before design because a design gate
evaluated on a comparator that does not reconstruct is a gate on the wrong
comparator. Design before the confirmatory test, which is the only place a
p-value is computed at all.

Everything after `_describe` is downstream of the confirmatory test and cannot
change a number it produced. That separation is deliberate and is why it is a
different function: a reader wondering whether the stress test could have
influenced the headline need only observe that it runs afterwards and writes to
different fields.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import cache as cache_mod
from . import estimand, evidence, gates, metrics, rules, stats
from .errors import BadInput, GateRefusal, MissingInput
from .provenance import Provenance, inspect
from .schema import GROUPING_5, UNINFECTED, content_hash, validate_patients

#: The confirmatory comparison, named in the registration. One rule against the
#: reconstructed published rule, on the declared checkpoint.
COMPARATOR = "R0"
CANDIDATE = "R4"
NI_MARGIN = 0.05
ALPHA = 0.05

#: How much of the analysis to run. `core` is every number a result depends on;
#: `full` adds the descriptions, the nulls, the stress test and the post hoc
#: sweep. Nothing in `full` can change a number in `core`.
DEPTHS = ("core", "full")


@dataclass
class Study:
    provenance: Provenance = None
    estimands: dict = field(default_factory=dict)
    cohort_check: dict = field(default_factory=dict)
    identifiability: gates.GateResult = None
    yields: dict = field(default_factory=dict)
    excluded: dict = field(default_factory=dict)
    factorial: pd.DataFrame = None
    predictions: dict = field(default_factory=dict)
    ties: dict = field(default_factory=dict)
    fitted: dict = field(default_factory=dict)
    reconstruction: gates.GateResult = None
    design: dict = field(default_factory=dict)
    confirmatory: dict = field(default_factory=dict)
    #: §9 primary: the patient-clustered model. `apportionment` is descriptive.
    attribution: object = None
    apportionment: dict = field(default_factory=dict)
    bounds: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)

    # --- descriptions, all downstream of the confirmatory test -------------
    calibration: dict = field(default_factory=dict)
    speciation_calibration: object = None
    selective: dict = field(default_factory=dict)
    predictive: object = None
    box_census: pd.DataFrame = None
    errors: pd.DataFrame = None
    error_summary: dict = field(default_factory=dict)
    domain: dict = field(default_factory=dict)
    loco: dict = field(default_factory=dict)
    field_curve: dict = field(default_factory=dict)
    stress: object = None
    power: dict = field(default_factory=dict)
    nulls: dict = field(default_factory=dict)
    sweep: object = None
    sweep_summary: dict = field(default_factory=dict)
    claims: pd.DataFrame = None
    #: The source study's zoom transformation, MEASURED by `mag ingest` from
    #: the two label trees and read from disk here. Empty when the archive was
    #: not present at ingest time, which is a fact to report and never one to
    #: fill in: the alternative is a manuscript quoting a retained fraction
    #: nobody counted.
    transform: dict = field(default_factory=dict)

    def frame(self) -> pd.DataFrame:
        return self.factorial

    @property
    def depth(self) -> str:
        return str(self.meta.get("depth", "core"))


def load_patients(path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise MissingInput(
            f"no patient table at {path}",
            remedy=("Build one from the released image_list.xlsx with the "
                    "detect stage, or point --patients at an existing one. "
                    "This package will not infer a cohort."))
    df = pd.read_csv(path, dtype={"patient_id": "string", "centre": "string",
                                  "truth": "string"})
    return validate_patients(df)


def _row(checkpoint: str, rid: str, family: str, truth: pd.Series,
         pred: pd.Series, ties: int = 0, *, fitted: bool = False) -> dict:
    scr = metrics.screening(truth, pred)
    ov = metrics.overall(truth, pred, GROUPING_5)
    sp = metrics.speciation(truth, pred, GROUPING_5)
    band = metrics.band_referenced_accuracy(truth, pred)
    return {"checkpoint": checkpoint, "rule": rid, "family": family,
            "fitted": bool(fitted),
            "in_main_paper": rid in rules.MAIN_RULES,
            "specificity": scr["specificity"],
            "spec_lo": scr["specificity_ci"][0], "spec_hi": scr["specificity_ci"][1],
            "tn": scr["tn"], "fp": scr["fp"],
            "sensitivity": scr["sensitivity"], "tp": scr["tp"], "fn": scr["fn"],
            "accuracy": ov["accuracy"], "acc_lo": ov["ci"][0], "acc_hi": ov["ci"][1],
            "n_correct": ov["n_correct"],
            "balanced_accuracy": ov["balanced_accuracy"],
            "speciation_accuracy": sp["accuracy"],
            "band": band["band"], "band_unambiguous": band["unambiguous"],
            "argmax_ties": int(ties),
            "exact_test_valid": bool(not fitted and rid in rules.FIXED_RULES)}


def _screening_discordance(truth: pd.Series, a: pd.Series, b: pd.Series) -> int:
    """Discordant pairs among the uninfected: what the exact test conditions on."""
    neg = truth == UNINFECTED
    ca = (a.reindex(truth.index)[neg] == UNINFECTED).to_numpy()
    cb = (b.reindex(truth.index)[neg] == UNINFECTED).to_numpy()
    return int((ca != cb).sum())


# --------------------------------------------------------------------------- #

def load_transform(patients_csv, cache_dir) -> dict:
    """The zoom-transformation measurement, if `mag ingest` wrote one.

    `ingest` counts the annotations in the `test` and `test-zoom` label trees
    and writes `transform.json` beside the patient table. The analysis reads it
    rather than recomputing it, because recomputing would mean this stage needs
    the 6 GB archive on disk -- and a reviewer re-running the analysis from a
    published cache does not have it.

    Absent is a normal state and is reported as such. The alternative is a
    manuscript quoting a retained fraction that nobody counted.
    """
    for base in (Path(patients_csv).parent, Path(cache_dir),
                 Path(cache_dir).parent):
        p = base / "transform.json"
        if p.is_file():
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
            except Exception as exc:                        # noqa: BLE001
                return {"measured": False,
                        "why": f"{p} could not be read ({exc})"}
            d["measured"] = True
            d["source"] = str(p)
            return d
    return {"measured": False,
            "why": ("`mag ingest` was run without --archive, or its output "
                    "was not kept beside the patient table, so the two label "
                    "trees were never counted")}


def run(cache_dir, patients_csv, *, checkpoints=None, condition: str = "matched",
        seed: int = 0, force_past_reconstruction: bool = False,
        depth: str = "core", folds: int = 10, min_fields: int = 25,
        reps: int = 50, stress_reps: int = 30) -> Study:
    from . import __version__, build_id

    if depth not in DEPTHS:
        raise BadInput(f"unknown depth {depth!r}",
                       remedy=f"Use one of: {', '.join(DEPTHS)}.")
    cache_dir = Path(cache_dir)
    patients = load_patients(patients_csv)
    measured_transform = load_transform(patients_csv, cache_dir)
    order = patients["patient_id"].tolist()
    truth = patients.set_index("patient_id")["truth"]
    n_fields_expected = int(pd.to_numeric(patients["n_images"]).sum())

    names = list(checkpoints) if checkpoints else cache_mod.available(cache_dir)
    if not names:
        raise MissingInput(
            f"no detection cache in {cache_dir}",
            remedy="Run the detect stage first. Nothing here fabricates one.")

    s = Study()
    s.transform = measured_transform
    s.meta = {"cache": str(cache_dir), "condition": condition, "seed": seed,
              "checkpoints_requested": names, "depth": depth,
              "folds": int(folds), "version": __version__, "build": build_id(),
              "protocol": "v3"}

    # --- the estimands, before anything they qualify ------------------------
    s.estimands = estimand.declaration()
    s.cohort_check = estimand.check_cohort(patients)

    # --- identifiability ----------------------------------------------------
    s.identifiability = gates.identifiability(patients)

    # --- load, and refuse arms that did not detect --------------------------
    loaded, manifests = {}, {}
    for name in names:
        det, man = cache_mod.load(cache_dir, name, verify=True)
        g = gates.yield_gate(man, n_fields_expected)
        s.yields[name] = g
        if not g.passed:
            s.excluded[name] = g.message
            continue
        loaded[name] = det
        manifests[name] = man
    if not loaded:
        raise GateRefusal(
            "every checkpoint was refused by the yield gate:\n"
            + "\n".join(f"  - {m}" for m in s.excluded.values()),
            remedy=("These caches record almost no detections. That is a "
                    "loading or driving fault, not a property of the "
                    "detectors. Fix detection and re-run; do not analyse them "
                    "as weak results."))

    s.provenance = inspect(cache_dir, cache_dir.name, manifests,
                           content_hash(patients), len(patients), condition,
                           s.excluded)

    # --- the factorial ------------------------------------------------------
    join = {}
    views = {}
    for name, det in loaded.items():
        rep = {}
        views[name] = evidence.build_views(det, patients, condition, report=rep)
        join[name] = rep
    s.meta["join"] = join
    dropped = {k: v for k, v in join.items() if v.get("n_dropped")}
    if dropped:
        s.meta["join_warning"] = (
            "detections were dropped for want of a matching patient, class or "
            "field: " + "; ".join(
                f"{k} {v['n_dropped']:,} of {v['n_detections']:,} "
                f"({v.get('dropped_share', 0):.1%})" for k, v in dropped.items()))

    rows = []
    for name, v in views.items():
        for rid in rules.rule_ids():
            pred, _, _, ties = rules.predict(v, order, rid)
            s.predictions[(name, rid)] = pred
            s.ties[(name, rid)] = int((ties > 1).sum())
            rows.append(_row(name, rid, rules.RULES[rid].family, truth, pred,
                             s.ties[(name, rid)]))
        for rid in rules.fitted_ids():
            pred, _, ties, detail = rules.fit_predict(v, order, truth, rid,
                                                      k=folds, seed=seed)
            s.predictions[(name, rid)] = pred
            s.ties[(name, rid)] = int((ties > 1).sum())
            s.fitted[(name, rid)] = detail
            rows.append(_row(name, rid, rules.FITTED[rid].family, truth, pred,
                             s.ties[(name, rid)], fitted=True))
        b0 = rules.trivial(order, truth)
        b1 = rules.oracle(v, order, truth)
        s.predictions[(name, "B0")] = b0
        s.predictions[(name, "B1")] = b1
        rows.append(_row(name, "B0", "bound", truth, b0))
        rows.append(_row(name, "B1", "bound", truth, b1))
    s.factorial = pd.DataFrame(rows)
    s.bounds = {
        "oracle": {k: float(v) for k, v in
                   s.factorial[s.factorial.rule == "B1"]
                   .set_index("checkpoint")["accuracy"].items()},
        "trivial": {k: float(v) for k, v in
                    s.factorial[s.factorial.rule == "B0"]
                    .set_index("checkpoint")["accuracy"].items()}}

    # --- reconstruction of the published rule -------------------------------
    # NOT "reproduction". Protocol v3 §3: the published result is an external
    # fact we did not compute; what we can say is whether it is reconstructible
    # from the released artefacts under the documented procedure.
    primary_ck = (gates.DECLARED_CHECKPOINT if gates.DECLARED_CHECKPOINT in loaded
                  else sorted(loaded)[0])
    src_views = evidence.build_views(loaded[primary_ck], patients, "source")
    src_pred, _, _, _ = rules.predict(src_views, order, COMPARATOR)
    src_ov = metrics.overall(truth, src_pred, GROUPING_5)
    src_scr = metrics.screening(truth, src_pred)
    s.reconstruction = gates.reconstruction_gate(src_ov["n_correct"], len(order))
    s.reconstruction.detail.update(
        checkpoint=primary_ck, sensitivity=src_scr["sensitivity"],
        specificity=src_scr["specificity"], condition="source",
        vocabulary=("not reconstructible from the released artefacts; this is "
                    "a finding about the release, not evidence about the "
                    "original implementation"))

    s.predictions[(primary_ck, "B2")] = src_pred
    s.factorial = pd.concat(
        [s.factorial, pd.DataFrame([_row(primary_ck, "B2", "bound", truth,
                                         src_pred)])], ignore_index=True)
    s.bounds["source_settings"] = {
        primary_ck: float(src_ov["accuracy"]),
        "n_correct": int(src_ov["n_correct"]), "condition": "source"}

    if not s.reconstruction.passed and not force_past_reconstruction:
        raise GateRefusal(
            f"[reconstruction gate] {s.reconstruction.message}",
            remedy=("This is a finding, not a bug. Report it, and treat every "
                    "later comparison as being against our reconstruction of "
                    "the published rule rather than against the published "
                    "system. Pass --force-reconstruction to continue and have "
                    "that recorded in the results."),
            **s.reconstruction.detail)
    s.meta["forced_past_reconstruction"] = bool(
        not s.reconstruction.passed and force_past_reconstruction)

    # --- the declared checkpoint, then one confirmatory comparison ----------
    heads = {name: gates.headroom(truth, s.predictions[(name, COMPARATOR)])
             for name in loaded}
    discs = {name: _screening_discordance(truth, s.predictions[(name, COMPARATOR)],
                                          s.predictions[(name, CANDIDATE)])
             for name in loaded}
    chosen, sel_rule, sel_detail = gates.select_checkpoint(
        heads, discs, criterion=gates.PRIMARY_SELECTION)
    dg = gates.design_gate(truth, s.predictions[(chosen, COMPARATOR)],
                           s.predictions[(chosen, CANDIDATE)],
                           checkpoint=chosen, rule=CANDIDATE)
    s.design = {"headroom": heads, "discordance": discs, "selected": chosen,
                "selection_rule": sel_rule, "selection_detail": sel_detail,
                "gate": dg, "family": [f"{chosen}:{CANDIDATE}"]}
    gates.refuse(dg, remedy=(
        "The confirmatory family cannot produce a non-null result as "
        "specified. Rebuild the family before unblinding; do not work around "
        "this by testing a different rule after seeing the data."))

    s.confirmatory = _confirm(truth, s.predictions[(chosen, COMPARATOR)],
                              s.predictions[(chosen, CANDIDATE)],
                              checkpoint=chosen, seed=seed)

    # the level-preserving criterion, as a pre-specified sensitivity analysis
    alt = sel_detail.get("by_max_discordance")
    if alt and alt != chosen:
        alt_gate = gates.design_gate(truth, s.predictions[(alt, COMPARATOR)],
                                     s.predictions[(alt, CANDIDATE)],
                                     checkpoint=alt, rule=CANDIDATE)
        if alt_gate.passed:
            s.confirmatory["sensitivity_analysis"] = _confirm(
                truth, s.predictions[(alt, COMPARATOR)],
                s.predictions[(alt, CANDIDATE)], checkpoint=alt, seed=seed)
            s.confirmatory["sensitivity_analysis"]["design_gate"] = "passed"
        else:
            s.confirmatory["sensitivity_analysis"] = {
                "checkpoint": alt, "same_as_primary": False,
                "design_gate": "refused", "why": alt_gate.message,
                "note": ("the level-preserving criterion selects a checkpoint "
                         "whose comparison the design gate refuses, so no test "
                         "is reported for it: a p-value from a comparison that "
                         "cannot produce a non-null result is not evidence")}
    else:
        s.confirmatory["sensitivity_analysis"] = {
            "checkpoint": alt or chosen, "same_as_primary": True,
            "note": ("the level-preserving criterion selects the same "
                     "checkpoint the protocol declares, so the size inflation "
                     "of a headroom-selected design cannot have affected this "
                     "result")}

    # --- §9 attribution: the model first, the ranges as description ---------
    s.attribution = _attribute(s, truth, order, loaded, seed=seed)
    s.apportionment = _apportion(s.factorial, s.predictions, truth, seed=seed)

    # --- everything downstream ----------------------------------------------
    _describe(s, loaded, views, patients, truth, order, chosen,
              condition=condition, seed=seed, depth=depth, folds=folds,
              min_fields=min_fields, reps=reps, stress_reps=stress_reps,
              reconstruction_ck=primary_ck)
    s.claims = claim_table(s)
    return s


# --------------------------------------------------------------------------- #

def _confirm(truth, comparator, candidate, *, checkpoint: str, seed: int,
             comparator_rule: str = COMPARATOR,
             candidate_rule: str = CANDIDATE) -> dict:
    """The co-primary pair, both components one-sided in the same direction.

    The rule ids are parameters and the exchangeability flag is derived from
    them: reading two module constants instead made the flag `True` for every
    possible call, which is a literal with extra steps on the one guard that
    keeps an invalid p-value off the page.
    """
    arms_fixed = (comparator_rule in rules.FIXED_RULES
                  and candidate_rule in rules.FIXED_RULES)
    base_s = metrics.screening(truth, comparator)
    cand_s = metrics.screening(truth, candidate)

    # Protocol v3 §10.2: the full 2x2 first. It is the inferential object, and
    # without a and d a reader cannot reconstruct the result.
    spec_table = stats.paired_table(base_s["correct_neg"], cand_s["correct_neg"])
    sens_table = stats.paired_table(base_s["correct_pos"], cand_s["correct_pos"])

    sup = stats.mcnemar_exact(base_s["correct_neg"], cand_s["correct_neg"],
                              alternative="greater", arms_are_fixed=arms_fixed)
    ni = stats.noninferiority(base_s["correct_pos"], cand_s["correct_pos"],
                              NI_MARGIN, alpha=ALPHA, seed=seed)
    iu = stats.intersection_union({
        "specificity_superior": bool(sup.p <= ALPHA and sup.valid),
        "sensitivity_non_inferior": bool(ni["non_inferior"])})

    delta = stats.bootstrap(
        list(zip(base_s["correct_neg"], cand_s["correct_neg"])),
        lambda u: float(np.mean([c for _, c in u]) - np.mean([b for b, _ in u])),
        seed=seed)
    return {
        "checkpoint": checkpoint, "rule": candidate_rule,
        "comparator": comparator_rule, "alpha": ALPHA,
        "arms_are_fixed": arms_fixed,
        "specificity_table": spec_table,
        "sensitivity_table": sens_table,
        "mcnemar": {"b": sup.b, "c": sup.c, "p": sup.p,
                    "n_discordant": sup.n_discordant,
                    "alternative": sup.alternative, "valid": sup.valid,
                    "reason": sup.reason},
        "noninferiority": ni,
        "spec_base": base_s["specificity"], "spec_new": cand_s["specificity"],
        "spec_delta": delta.point, "spec_delta_ci": (delta.lo, delta.hi),
        "sens_base": base_s["sensitivity"], "sens_new": cand_s["sensitivity"],
        "n_neg": base_s["n_neg"], "n_pos": base_s["n_pos"],
        "iu": iu,
        "scope_specificity": estimand.SPECIFICITY.scope,
        "scope_sensitivity": estimand.SENSITIVITY.scope,
        "governs": "the exact test decides; the interval describes",
    }


def _attribute(s: Study, truth, order, loaded, *, seed: int):
    """§9: the patient-clustered model over the fixed-rule factorial.

    Fixed rules only. A fitted rule's family was chosen on these patients, so
    including it in a marginal rule effect measures partly how much a fitted
    rule can fit -- the very comparison the attribution is about.
    """
    from . import attribute

    cks = sorted(loaded)
    rule_ids = [r for r in rules.rule_ids()]
    if len(cks) < 2 or len(rule_ids) < 2:
        out = attribute.Attribution(n_rules=len(rule_ids), n_checkpoints=len(cks))
        out.note = (f"{len(rule_ids)} rules and {len(cks)} checkpoints survived; "
                    "a two-factor model needs at least two levels of each")
        return out
    df = attribute.long_frame(s.predictions, truth, order, rules=rule_ids,
                              checkpoints=cks, grouping=GROUPING_5)
    return attribute.fit(df)


def _describe(s: Study, loaded: dict, views: dict, patients: pd.DataFrame,
              truth: pd.Series, order, chosen: str, *, condition: str,
              seed: int, depth: str, folds: int, min_fields: int = 25,
              reps: int = 50, stress_reps: int = 30,
              reconstruction_ck: str | None = None) -> None:
    """The descriptive analyses, in place on `s`.

    Everything here is downstream of the confirmatory test and none of it can
    change a number the test produced.
    """
    from . import (calibrate, diagnostics, power as power_mod, predictive,
                   selective, simulate)

    v = views[chosen]
    base_pred = s.predictions[(chosen, COMPARATOR)]
    cand_pred = s.predictions[(chosen, CANDIDATE)]

    # --- 11.A calibration of the infection decision -------------------------
    raw = np.array([rules.RULES[COMPARATOR].apply(v[str(p)])["confidence"]
                    for p in order], dtype=float)
    y = (truth.reindex(order).to_numpy() != UNINFECTED).astype(float)
    s.calibration = {"raw": _as_dict(calibrate.report(raw, y, seed=seed))}
    for kind in calibrate.CALIBRATORS:
        cf = calibrate.crossfit_detail(raw, y, kind, k=folds, seed=seed)
        rep = _as_dict(calibrate.report(cf.calibrated, y, seed=seed))
        rep.update(folds=cf.k, folds_fitted=cf.n_folds_fitted,
                   folds_degenerate=cf.n_folds_degenerate, cross_fitted=True)
        s.calibration[kind] = rep
    s.calibration["endpoint"] = "binary infection screening"
    s.calibration["note"] = (
        "the raw row is the screening statistic as the rule computes it; the "
        "three others are cross-fitted, so each patient's probability comes "
        "from a map fitted without that patient. Reported to four decimals: a "
        "Brier of 0.000 is indistinguishable from 0.0004. The expected "
        "calibration error of a perfectly calibrated predictor is not zero at "
        "this sample size, which is what null_reference records.")

    # --- 11.B calibration of the species posterior, kept separate -----------
    ev = pd.DataFrame(
        {p: rules.RULES[COMPARATOR].apply(v[str(p)])["evidence"] for p in order}).T
    from .schema import PARASITE_CLASSES

    s.speciation_calibration = calibrate.speciation_report(
        ev, truth.reindex(order), PARASITE_CLASSES, seed=seed)

    # --- selective prediction ------------------------------------------------
    for rid in (COMPARATOR, CANDIDATE):
        ev_r = pd.DataFrame(
            {p: rules.RULES[rid].apply(v[str(p)])["evidence"] for p in order}).T
        conf = selective.confidence(ev_r)
        rc = selective.risk_coverage(s.predictions[(chosen, rid)], truth, conf,
                                     GROUPING_5)
        s.selective[rid] = {"coverage": rc.coverage, "risk": rc.risk,
                            "aurc": rc.aurc, "operating": rc.operating}
    s.selective["note"] = (
        "kappa = (largest minus second largest class posterior) times the "
        "screening score, identically for every rule. The coverage threshold "
        "is taken by rank because dozens of patients sit at exactly zero "
        "confidence and no quantile separates them. The headline is workload, "
        "not the area under the curve.")

    # --- prevalence-standardised predictive values --------------------------
    base_s = metrics.screening(truth, base_pred)
    cand_s = metrics.screening(truth, cand_pred)
    s.predictive = predictive.at_prevalences(
        cand_s["correct_pos"], cand_s["correct_neg"], seed=seed)

    # --- what the detectors emitted -----------------------------------------
    s.box_census = diagnostics.box_census(
        loaded, int(pd.to_numeric(patients["n_images"]).sum()))

    # --- what kind of mistakes, and how fragile ------------------------------
    s.errors = diagnostics.error_table(truth.reindex(order), base_pred, patients,
                                       v, COMPARATOR)
    s.error_summary = diagnostics.error_summary(s.errors)
    s.error_summary["checkpoint"] = chosen
    s.error_summary["rule"] = COMPARATOR

    # --- domain -------------------------------------------------------------
    s.domain = diagnostics.domain_contrast(truth.reindex(order), base_pred,
                                           cand_pred, patients, seed=seed)
    s.loco = diagnostics.leave_one_centre_out(truth.reindex(order), cand_pred,
                                              patients, grouping=GROUPING_5)

    # --- detectable effects --------------------------------------------------
    rep = power_mod.analyse(s.confirmatory, base_s["correct_pos"],
                            cand_s["correct_pos"], base_s["n_neg"], seed=seed)
    s.power = {"specificity": rep.specificity, "sensitivity": rep.sensitivity,
               "method_agreement": rep.agreement, "note": rep.note}

    if depth != "full":
        s.meta["skipped_at_core_depth"] = [
            "field_curve", "field-sampling stress test",
            "nulls (winner's curse, cross-fit, selection)",
            "reconstruction sweep"]
        return

    # --- the expensive descriptions ------------------------------------------
    s.field_curve = diagnostics.field_curve(
        loaded[chosen], patients, truth, [COMPARATOR, CANDIDATE],
        condition=condition, seed=seed, min_fields=min_fields, reps=reps)

    # §13 the stress test: does the ordering survive dilution?
    from . import stress as stress_mod

    s.stress = stress_mod.run(
        loaded[chosen], patients, truth,
        [r for r in rules.MAIN_RULES if r in rules.RULES],
        condition=condition, reps=stress_reps, seed=seed, grouping=GROUPING_5)

    from .schema import regroup

    t_grouped = truth.reindex(order).map(lambda x: regroup(x, GROUPING_5))
    correct = pd.DataFrame({
        f"{ck}:{rid}": (s.predictions[(ck, rid)].reindex(order)
                        .map(lambda x: regroup(x, GROUPING_5))
                        .to_numpy() == t_grouped.to_numpy()).astype(float)
        for (ck, rid) in sorted(s.predictions, key=str)
        if not str(rid).startswith("B")}, index=order)
    s.nulls["winners_curse"] = _as_dict(simulate.winners_curse(correct, seed=seed))
    s.nulls["crossfit"] = simulate.crossfit_null(
        n=len(order), prevalence=float(y.mean()), reps=120, seed=seed)
    n_neg = int(base_s["n_neg"])
    if n_neg > 0 and len(s.design["headroom"]) >= 2:
        rates = [h / n_neg for h in s.design["headroom"].values()]
        disc = float(np.mean([d / n_neg for d in s.design["discordance"].values()]))
        s.nulls["selection"] = simulate.selection_comparison(
            n_neg, rates, n_checkpoints=len(rates), discordance=disc,
            reps=3000, seed=seed)
    else:
        s.nulls["selection"] = {
            "note": (f"{len(s.design['headroom'])} checkpoint(s) survived the "
                     "yield gate, so there was no selection to simulate")}

    from . import search

    sweep_ck = (reconstruction_ck if reconstruction_ck in loaded else chosen)
    s.sweep = search.sweep(loaded[sweep_ck], patients, truth)
    s.sweep_summary = search.summarise(s.sweep)
    s.sweep_summary["checkpoint"] = sweep_ck
    s.sweep_summary["is_reconstruction_checkpoint"] = bool(
        sweep_ck == reconstruction_ck)


def _as_dict(obj) -> dict:
    from dataclasses import asdict, is_dataclass

    return asdict(obj) if is_dataclass(obj) else dict(obj)


# --------------------------------------------------------------------------- #
# §18 the claim-evidence-scope table, as a build artefact
# --------------------------------------------------------------------------- #

def claim_table(s: Study) -> pd.DataFrame:
    """Every claim the paper makes, its evidence, its scope, and its verdict.

    Protocol v3 §18. Generated from the results object rather than written by
    an author, so that a claim cannot survive a run that fails to support it.
    The last two rows are permanently "not assessed" and are here because their
    absence is the thing a reader most needs to see.
    """
    c = s.confirmatory or {}
    iu = c.get("iu") or {}
    comp = iu.get("components") or {}
    attr = s.attribution
    stress = s.stress

    def verdict(ok, unknown=False):
        return "not assessed" if unknown else ("supported" if ok else "not established")

    rows = [
        {"claim": "the aggregation rule is consequential",
         "evidence": ("marginal rule effect, patient-clustered"
                      if attr and attr.contrast else "not computed"),
         "scope": "this cohort, the levels surviving the yield gate",
         "verdict": verdict(bool(attr and attr.contrast.get("excludes_zero")))},
        {"claim": f"{CANDIDATE} improves specificity over the reconstructed rule",
         "evidence": (f"paired 2x2, exact p = {c['mcnemar']['p']:.4g}"
                      if c else "not computed"),
         "scope": estimand.SPECIFICITY.scope,
         "verdict": verdict(bool(comp.get("specificity_superior")))},
        {"claim": f"{CANDIDATE} is superior overall",
         "evidence": "intersection-union co-primary endpoint",
         "scope": "both components must pass",
         "verdict": verdict(bool(iu.get("reject")))},
        {"claim": "detector checkpoints differ",
         "evidence": ("marginal checkpoint effect, patient-clustered"
                      if attr and attr.checkpoint_effect else "not computed"),
         "scope": "matched post-processing only",
         "verdict": verdict(bool(
             attr and attr.checkpoint_effect
             and any(not lv.get("reference") and (lv["lo"] > 0 or lv["hi"] < 0)
                     for lv in attr.checkpoint_effect["levels"].values())))},
        {"claim": "the rule effect exceeds the checkpoint effect",
         "evidence": "contrast of marginal effects; NOT variance components",
         "scope": "this cohort, these levels",
         "verdict": verdict(bool(attr and attr.contrast.get("excludes_zero")))},
        {"claim": "the published result is reconstructible from the release",
         "evidence": (f"{s.reconstruction.detail['n_correct']}/"
                      f"{s.reconstruction.detail['n_total']} against a published "
                      f"{s.reconstruction.detail['published']}"
                      if s.reconstruction else "not computed"),
         "scope": "the released artefacts under the documented procedure",
         "verdict": verdict(bool(s.reconstruction and s.reconstruction.passed))},
        {"claim": "the ordering of the rules survives field dilution",
         "evidence": (f"Kendall tau {stress.ordering['min_tau']:.2f} at the "
                      "lowest positive-field rate"
                      if stress and stress.ordering else "not computed"),
         "scope": "a stated resampling model, not screening performance",
         "verdict": ("supported" if stress and stress.ordering.get("stable")
                     else "not established" if stress and stress.ordering
                     else "not assessed")},
        {"claim": "the screening statistic is calibrated",
         "evidence": "expected calibration error against its finite-sample null",
         "scope": "within finite-sample variability; not an inferential test",
         "verdict": "not assessed"},
        {"claim": "clinical utility",
         "evidence": "—",
         "scope": "outside the estimand",
         "verdict": "not assessed"},
        {"claim": "deployment readiness",
         "evidence": "—",
         "scope": "outside the dataset",
         "verdict": "not assessed"},
    ]
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Descriptive apportionment. Secondary to §9 and labelled everywhere.
# --------------------------------------------------------------------------- #

def _bootstrap_spreads(cell: pd.DataFrame, predictions: dict, truth: pd.Series,
                       *, seed: int, n_boot: int = 1000) -> dict:
    """Resample patients; recompute both spreads and their difference."""
    from .schema import regroup

    rules_idx = list(cell.index)
    ck_idx = list(cell.columns)
    order = list(truth.index)
    t = truth.map(lambda x: regroup(x, GROUPING_5)).to_numpy()

    grid = np.empty((len(order), len(rules_idx), len(ck_idx)), dtype=float)
    for i, rid in enumerate(rules_idx):
        for j, ck in enumerate(ck_idx):
            pred = predictions.get((ck, rid))
            if pred is None:
                return {"bootstrap": {"note": "a cell had no predictions"}}
            p = pred.reindex(order).map(lambda x: regroup(x, GROUPING_5)).to_numpy()
            grid[:, i, j] = (p == t).astype(float)

    def spread_pair(acc):
        r, c = acc.mean(axis=1), acc.mean(axis=0)
        return float(r.max() - r.min()), float(c.max() - c.min())

    obs_r, obs_c = spread_pair(grid.mean(axis=0))
    rng = np.random.default_rng(seed)
    diffs, rs, cs = (np.empty(n_boot) for _ in range(3))
    n = len(order)
    for b in range(n_boot):
        sr, sc = spread_pair(grid[rng.integers(0, n, size=n)].mean(axis=0))
        rs[b], cs[b], diffs[b] = sr, sc, sr - sc
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"bootstrap": {
        "n_boot": int(n_boot), "n_patients": n,
        "spread_rule_ci": (float(np.quantile(rs, 0.025)), float(np.quantile(rs, 0.975))),
        "spread_checkpoint_ci": (float(np.quantile(cs, 0.025)), float(np.quantile(cs, 0.975))),
        "difference": float(obs_r - obs_c), "difference_ci": (float(lo), float(hi)),
        "excludes_zero": bool(lo > 0 or hi < 0),
        "note": "patients are the resampling unit; both spreads are recomputed "
                "on the same resample so their difference retains the "
                "correlation between them"}}


def _apportion(factorial: pd.DataFrame, predictions: dict, truth: pd.Series,
               *, seed: int, n_perm: int = 500) -> dict:
    """Descriptive ranges. SECONDARY to the model in §9, and labelled so.

    A range grows with the number of levels, and the two factors here have very
    different level counts. The previous version made this the primary
    attribution analysis; protocol v3 §9.3 demotes it to a description that
    accompanies the figure.
    """
    is_bound = factorial.rule.str.startswith("B")
    is_fitted = (factorial["fitted"] if "fitted" in factorial.columns
                 else pd.Series(False, index=factorial.index)).fillna(False)
    real = factorial[~is_bound & ~is_fitted.astype(bool)]
    cell = real.pivot_table(index="rule", columns="checkpoint", values="accuracy")
    if cell.empty:
        return {}

    def spreads(m: pd.DataFrame) -> dict:
        r, c = m.mean(axis=1), m.mean(axis=0)

        def pw(v):
            a = np.asarray(v, dtype=float)
            return np.abs(a[:, None] - a[None, :])[np.triu_indices(len(a), 1)]

        rd, cd = pw(r.to_numpy()), pw(c.to_numpy())
        return {"spread_rule": float(r.max() - r.min()),
                "spread_checkpoint": float(c.max() - c.min()),
                "pairwise_rule_median": float(np.median(rd)) if rd.size else float("nan"),
                "pairwise_checkpoint_median": float(np.median(cd)) if cd.size else float("nan"),
                "n_rules": int(m.shape[0]), "n_checkpoints": int(m.shape[1])}

    out = spreads(cell)
    out["rule_exceeds_checkpoint"] = out["spread_rule"] > out["spread_checkpoint"]
    out["status"] = ("descriptive; the attribution claim is made by the "
                     "patient-clustered model in `attribution`, not here")

    rng = np.random.default_rng(seed)
    vals = cell.to_numpy()
    null_c = []
    for _ in range(n_perm):
        m = pd.DataFrame(np.apply_along_axis(rng.permutation, 1, vals),
                         index=cell.index, columns=cell.columns)
        null_c.append(spreads(m)["spread_checkpoint"])
    out["perm_null_checkpoint_p95"] = float(np.quantile(null_c, 0.95))
    out["perm_null_checkpoint_p"] = float(
        (np.asarray(null_c) >= out["spread_checkpoint"] - 1e-12).mean())
    out["perm_null_rule_p95"] = None
    out["perm_null_note"] = (
        "the checkpoint null permutes each rule's values across checkpoints. No "
        "permutation null is reported for the rule factor: with one observation "
        "per cell-mean there is no permutation that removes a rule effect "
        "without also removing the values it is made of.")

    out.update(_bootstrap_spreads(cell, predictions, truth, seed=seed))

    jack = {}
    if cell.shape[1] > 2:
        for drop in cell.columns:
            jack[str(drop)] = spreads(cell.drop(columns=[drop]))
    out["leave_one_checkpoint_out"] = jack
    out["ordering_stable"] = (
        all((v["spread_rule"] > v["spread_checkpoint"]) ==
            out["rule_exceeds_checkpoint"] for v in jack.values())
        if jack else None)
    out["cell_means"] = {str(c): {str(r): float(cell.loc[r, c])
                                  for r in cell.index} for c in cell.columns}
    with_fitted = factorial[~is_bound]
    cell_f = with_fitted.pivot_table(index="rule", columns="checkpoint",
                                     values="accuracy")
    if cell_f.shape[0] > cell.shape[0]:
        sf = spreads(cell_f)
        out["including_fitted_rules"] = {
            **sf,
            "rule_exceeds_checkpoint": sf["spread_rule"] > sf["spread_checkpoint"],
            "note": ("a sensitivity analysis, not the primary apportionment: "
                     "the fitted rules' parameters come from these patients")}
    out["n_fitted_rules_excluded"] = int(cell_f.shape[0] - cell.shape[0])
    return out
