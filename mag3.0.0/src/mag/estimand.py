"""What each endpoint actually estimates, and the scope that travels with it.

Protocol v3 §4. This module exists because of a single observation from an
external review that changes the paper rather than qualifying it:

    the dataset does not merely limit precision, it changes what is being
    estimated

Two facts about this archive do that. Every uninfected patient comes from one
centre, so specificity is not a property of a rule but of a cohort. And every
field from an infected patient was selected to contain a parasite, so
sensitivity is conditioned on sampling that screening does not provide.

Neither is a limitation to be confessed in a final paragraph. Both are part of
the estimand, and an estimand that appears only in the Discussion has already
been read as something else by then. So each endpoint carries its scope as a
string, the string travels into every generated sentence that quotes the
number, and `report` cannot print the number without it.

The scopes are written HERE, before any analysis, and are properties of the
dataset rather than of any result. That is what makes them safe to write before
the run -- and what makes it dishonest to soften them after it.
"""

from __future__ import annotations

from dataclasses import dataclass

#: The centre that supplied every uninfected patient in the released test set.
#: Named rather than inferred: if a future cohort differs, the identifiability
#: gate says so and this scope is rebuilt from the patient table.
NEGATIVE_CENTRE = "Montpellier"


@dataclass(frozen=True)
class Estimand:
    """One endpoint's target, its scope, and the wording it forces."""

    name: str
    #: what the quantity is, in one sentence, with its conditioning explicit
    target: str
    #: the population and conditions the estimate is FOR
    scope: str
    #: wording that must never be used about this endpoint
    forbidden: tuple = ()
    #: the form a sentence quoting this number must take
    required_form: str = ""
    #: True when the design cannot separate this quantity from a nuisance one
    identifiable: bool = True
    why_not: str = ""

    def qualify(self, sentence: str) -> str:
        """Attach the scope to a sentence that quotes this endpoint."""
        return f"{sentence.rstrip('.')} ({self.scope})."


SPECIFICITY = Estimand(
    name="screening specificity",
    target=("the probability that the rule calls a patient uninfected, among "
            "uninfected patients"),
    scope=(f"among uninfected patients from the {NEGATIVE_CENTRE} cohort of "
           "this archive, under the reconstructed pipeline"),
    forbidden=("the rule controls false positives",
               "specificity of the rule",
               "site-invariant specificity"),
    required_form=("reduced false-positive calls in the "
                   f"{NEGATIVE_CENTRE}-negative subset of this archive"),
    identifiable=False,
    why_not=("every uninfected patient in the released test set comes from one "
             "centre, which also supplied training material, so specificity is "
             "perfectly collinear with centre and with internal/external "
             "domain. This is rank deficiency, not low power: no estimator "
             "separates them and no sample size fixes it"),
)

SENSITIVITY = Estimand(
    name="screening sensitivity",
    target=("the probability that the rule calls a patient infected, among "
            "infected patients"),
    scope=("among infected patients, when every examined field is known to "
           "contain a parasite"),
    forbidden=("sensitivity in screening",
               "the rule detects infection in N% of cases"),
    required_form=("called N% of infected patients infected, on smears whose "
                   "every field was selected to contain a parasite"),
    identifiable=True,
)

ACCURACY = Estimand(
    name="patient-level accuracy",
    target="the proportion of patients whose label the rule reproduces",
    scope=("in a cohort of 68% prevalence with a species mix chosen by a "
           "referral centre; a design-dependent summary, not a headline"),
    forbidden=("accuracy of the system", "overall performance"),
    required_form=("reproduced the reference label for N% of patients in this "
                   "cohort"),
    identifiable=True,
)

SPECIATION = Estimand(
    name="speciation accuracy",
    target=("the proportion of truly infected patients whose species the rule "
            "reproduces"),
    scope=("among the truly infected patients, regardless of the screening "
           "call, so the denominator is identical across rules"),
    forbidden=(),
    required_form="",
    identifiable=True,
)

#: Keyed for lookup from a metric name, so that a result object can attach its
#: own scope without the caller choosing one.
ESTIMANDS = {e.name: e for e in (SPECIFICITY, SENSITIVITY, ACCURACY, SPECIATION)}


def scope_of(metric: str) -> str:
    """The scope string for a metric name, or a refusal to invent one."""
    key = {"specificity": "screening specificity",
           "sensitivity": "screening sensitivity",
           "accuracy": "patient-level accuracy",
           "speciation_accuracy": "speciation accuracy"}.get(metric, metric)
    e = ESTIMANDS.get(key)
    return e.scope if e else ""


def declaration() -> dict:
    """Every estimand, for the results object and the Methods section.

    Written into `results.json` so that a reader of the machine-readable output
    sees the same scopes as a reader of the paper, and so that a future run on a
    different cohort produces different scopes rather than the same prose.
    """
    return {
        e.name: {
            "target": e.target,
            "scope": e.scope,
            "identifiable": e.identifiable,
            "why_not": e.why_not,
            "forbidden": list(e.forbidden),
            "required_form": e.required_form,
        }
        for e in ESTIMANDS.values()
    }


def check_cohort(patients) -> dict:
    """Confirm the scopes still describe the cohort in hand.

    The scopes above are written for the released test set. If a future cohort
    draws its uninfected patients from more than one centre, SPECIFICITY.scope
    is wrong and this says so rather than letting stale prose describe new data.
    """
    from .schema import UNINFECTED

    neg = patients[patients["truth"] == UNINFECTED]
    centres = sorted(set(neg["centre"].dropna().astype(str)))
    single = len(centres) == 1
    matches = single and centres[0] == NEGATIVE_CENTRE
    return {
        "negative_centres": centres,
        "n_uninfected": int(len(neg)),
        "specificity_scope_still_holds": bool(matches),
        "note": ("the specificity scope names "
                 f"{NEGATIVE_CENTRE}; this cohort's uninfected patients come "
                 f"from {', '.join(centres) if centres else 'no centre'}"
                 + ("" if matches else
                    " -- the scope in estimand.SPECIFICITY must be rebuilt "
                    "before any specificity figure is reported")),
    }
