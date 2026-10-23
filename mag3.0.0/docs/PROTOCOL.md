# Analysis protocol v3

**One specification. The code implements it; the manuscript reports against it;
neither is allowed to say anything the other does not.**

Supersedes v2. Written after three independent reviews of the v2 manuscript
build and after an adversarial audit of the v2 analysis package. Every change
from v2 is traceable to a specific finding, and the traceability matrix in §1
says which.

---

## 0. Read this first: the reviewed build was not a results build

All three reviews describe numbers that do not exist. The PDF they read carries
its own watermark — *"DRAFT — the numbers in this build are NOT from a run on
the archive"* — and the figures they quote are demo-generator output. This is
not a quibble about provenance; it changes what the paper is about:

| the reviews were shown | the last real run produced | consequence |
|---|---|---|
| reproduction 156/170, **above** the published 135 | **96/170**, *below* it | the direction of the reproducibility finding reverses |
| specificity under R0 = 74.1% | **0.000** — every uninfected patient called infected | the failure is total, not partial |
| R4−R0 specificity +25.9 pp, co-primary **met** | **+87.0 pp**; NI **inconclusive** (bound −6.0 vs −5.0 margin) | the endpoint fails on a different component than reviewed |
| Brier 0.000; AURC 0.027 in text vs 0.471 in figure | generator artefacts | not defects to repair in prose |

Reviewer 1 builds a recommendation on "the reimplementation scored higher — 21
patients." The real finding is the opposite and is stronger: **the released
artefacts under the documented rule call every uninfected patient infected.**

Two consequences bind everything below.

**First, no number enters prose until the re-run produces it.** The last real
cache no longer exists; the archive must be re-processed. Until then every
quantity in this protocol is a slot, not a value.

**Second, the reviewers' methodological points stand regardless.** They were
reasoning about the design, and on the design they are largely right — including
on one defect this package had already found by simulation and one it had got
conceptually wrong. Those are §4 and §9.

---

## 1. What changed from v2, and why

Status column: **✔** implemented in `mag` v1 · **~** partial · **✘** absent, new
work. Reviewer column: R1, R2, R3 are the three reviews in the order supplied.

| § | change | driver | status |
|---|---|---|---|
| 2 | Contribution hierarchy inverted: the measurement problem leads, R4 demoted to a demonstration | R1 §17–18, R3 | ✔ (prose) |
| 3 | Three quantities separated by name everywhere: published / reconstructed / comparator | R2 §1.3 | ✘ |
| 3 | "Failed reproduction" → "not reconstructible from the released artefacts" | R1 §4, R2 §1.3 | ✘ (vocabulary) |
| 5 | Rule set reduced to six in the main paper; full catalogue to supplement | R2 §4.1 | ✘ |
| 5 | Rule comparability table: input quantity, scale, threshold meaning, dependencies | R2 §4.2 | ✘ |
| 6 | Parameter rationale table with pre-registration status | R1 §31 | ✘ |
| 7 | Proposition statements in main text, proofs to supplement | R1 §7 | ✘ |
| 8 | **Checkpoint pre-declared**, headroom demoted to descriptive feasibility | R2 §2.1 + own simulation | ~ |
| 9 | **Factorial attribution by patient-clustered model with interaction**; ranges demoted | R2 §2.4–2.5 | ✘ |
| 9 | Retract the claim that interaction "is not estimable" — it is | R2 §2.5 | ✘ |
| 10 | Primary endpoint presented as paired discordances first | R2 §2.2, R1 §3 | ~ |
| 11 | **Calibration split**: binary infection vs conditional speciation | R2 §5.1, R1 §12 | ✘ |
| 12 | Abstention reported as workload–performance, not AURC alone | R2 §5.3 | ~ |
| 13 | **Field-sampling stress test** — the highest-value new experiment | R2 §4.3 | ✘ |
| 14 | Prevalence-standardised predictive values at declared prevalences | R2 §3.3 | ✘ |
| 15 | Leave-one-centre-out where estimable | R2 §9 | ✘ |
| 16 | Accuracy demoted from headline; screening pair promoted | R2 §3.3 | ✘ (prose) |
| 17 | WHO-band analysis moved wholly to supplement | R1 §14, R2 | ✘ |
| 18 | Specificity restated as a Montpellier-cohort estimate, not site-invariant | R2 §3.1, R1 §16 | ~ |
| 19 | Claim–evidence–scope table as a build artefact | R1 §34 | ✘ |
| 20 | Build audit that fails the PDF on any unresolved macro or cross-artefact mismatch | R1 §2, R2 §1.2, R3 | ~ |
| — | *(already fixed before these reviews)* one generated sentence per directional claim | R1 §3, R3 §1 | ✔ |
| — | *(already fixed)* Tango score limit replaces the percentile bootstrap on NI | own audit | ✔ |
| — | *(already fixed)* exact-test validity flag derived per arm | own audit | ✔ |

### Points I am not adopting, and why

**R1 §24 / R3 §7 — "the AURC figure disagrees with the text", "Table 5 disagrees
with Figure 3."** These are not editing tasks. In v2 the figures and the numbers
were written by different code paths; in v3 both descend from one `Study` object
and a stamp check refuses a manuscript build across a mismatch. The class of
defect is removed rather than the instances repaired.

**R1 §15 / R3 §11 — a model card giving each checkpoint's architecture and
training set.** Adopt the table, but it must be sourced from the released
artefacts and the source publication, not reconstructed. Where the release does
not state something — early-stopping set membership, augmentation — the cell
reads NOT REPORTED. A model card filled in by inference is worse than none.

**R3 §14 — "the study has approximately 80% power to detect 6% at α=0.05."**
Do not write this. It is a *pre-hoc* power statement computed after the sample
size was fixed by someone else's test split, and it assumes a base sensitivity.
§10.4 replaces it with the detectable-effect calculation, which answers the same
reader question without assuming the answer.

**R1 §26 — a boxed "minimum reporting standard" takeaway.** Adopt, with one
change: it must be derived from what this study found to be undocumented, not
asserted as a general standard. A recommendation is evidence, not a slogan.

---

## 2. The question, and the hierarchy of claims

The paper is a **methodological diagnostic-AI paper demonstrated on malaria
microscopy**, not a malaria detector paper. The transferable statement is:

> A detector and a patient-level diagnostic system are not the same object. The
> patient decision is a function of the checkpoint, the post-processing, the
> aggregation rule, the threshold and the calibration — and patient-level
> results are routinely reported naming only the first.

**Contributions, in the order the paper makes them:**

1. **A measurement problem.** Patient-level performance is not determined by
   detector output alone; a decision layer sits between them and is rarely
   reported.
2. **An experimental demonstration.** Under matched post-processing and fixed
   weights, varying the aggregation rule moved patient-level performance
   further than varying the checkpoint. *(Scope: this cohort, these levels.)*
3. **A reproducibility finding.** The published patient-level result was not
   reconstructible from the released artefacts under the documented procedure.
4. **A design framework.** Identifiability and design gates that refuse
   comparisons incapable of producing a non-null result, before unblinding.
5. **A clinical caution.** This dataset cannot establish deployment-level
   performance, and the protocol says so before the results rather than after.

**R4 is evidence for contribution 1. It is not the paper's endpoint.** No
sentence may present it as a recommended rule.

---

## 3. Three quantities that must never share a name

The single most common way this manuscript misleads is by letting one word do
three jobs. Fixed vocabulary, used in the title, abstract, methods, results,
discussion, figures and code:

| name | symbol | what it is |
|---|---|---|
| **the published result** | — | the accuracy the source publication reports. An external fact. We did not compute it and cannot verify it |
| **the reconstructed rule** | **R0** | our implementation of the published rule as documented, run on the released weights and images |
| **the comparator** | **R0** | the same object, in its role as the baseline of every paired test |

The confirmatory comparison is therefore **R4 against our reconstruction of the
published rule**, never "against the published system." Required sentence,
generated:

> The released artefacts did not reconstruct the source study's reported
> patient-level accuracy. We treated the reconstructed rule as the prespecified
> comparator for all paired analyses, and interpret the discrepancy as a
> reproducibility finding about the release rather than as evidence about the
> original implementation.

**Vocabulary ban.** "Failed reproduction" is retired. It licenses the reading
*the original paper was wrong*, which this study cannot support. Use **"not
reconstructible from the released artefacts"**.

---

## 4. The estimands, stated before any analysis

Reviewer 2's strongest point: the dataset does not merely limit precision, it
**changes what is being estimated**. Each endpoint below carries its estimand,
and the estimand carries its scope into every sentence that quotes it.

### 4.1 Screening specificity — a cohort quantity, not a property of a rule

All 54 uninfected patients come from Montpellier, which also supplied training
material. Specificity is therefore perfectly collinear with centre and with
internal/external domain. This is **rank deficiency, not low power**: no
estimator separates them and no sample size fixes it.

> **Estimand.** The probability that the rule calls a patient uninfected, among
> uninfected patients from the Montpellier cohort of this archive, under the
> reconstructed RT-DETR pipeline.

Every specificity sentence carries that scope. Forbidden: "the rule controls
false positives." Required form: "the rule reduced false-positive calls in the
Montpellier-negative subset of this archive."

### 4.2 Screening sensitivity — conditioned on enriched fields

Every field from an infected patient contains at least one parasite by the
archive's selection protocol. Accumulation and counting rules are being
evaluated under sampling far more favourable than screening.

> **Estimand.** The probability that the rule calls a patient infected, among
> infected patients, **when every examined field is known to contain a
> parasite.**

That last clause is not a limitation to be confessed later. It is part of the
estimand, and §13 measures how far the answer moves when it is relaxed.

### 4.3 Accuracy — demoted

At 68% prevalence with a non-representative species mix, accuracy is a
design-dependent summary. It remains in the factorial because it is the one
quantity defined for every cell, and it is **never a headline**. The screening
pair leads; balanced accuracy and per-class recall follow; accuracy is reported
as design-dependent and labelled so in every table.

### 4.4 Speciation — a fixed denominator

Evaluated on the truly infected patients regardless of the screening call, so
the denominator is identical across rules and the pairing survives.

---

## 5. The decision layer

### 5.1 Rules: six in the main paper

v2 carried fourteen. On 170 patients that is a framework competing with its own
finding, and several rules were near-duplicates. The **main paper reports six**,
chosen to span the decision principles rather than to fill a table:

| id | principle | why it is in the main set |
|---|---|---|
| R0 | summed confidence, threshold at 0⁺ | the reconstructed published rule; the comparator |
| R1 | extremal — the single strongest detection | the opposite pole from accumulation |
| R4 | counting — at least *m* detections above τ | the confirmatory candidate |
| R5 | field voting — share of fields carrying evidence | the only rule normalised by fields examined |
| R6 | density — parasites per erythrocyte | the only rule with a clinical unit |
| B1 | oracle | the ceiling on what any function of this cache could return |

The remaining rules (R2, R3, R6b, R7, R8, and the fitted family R9–R13) move to
**Supplementary Table S4** with their full definitions. They are still computed
— the factorial needs them — but they do not compete for the reader's attention.

### 5.2 The comparability table (new, required)

Reviewer 2 §4.2 is correct that fixed parameters make rules *pre-registered*
without making them *comparable*. Their thresholds live on different scales and
depend on different things. One table, in the main paper:

| rule | input quantity | threshold means | depends on field count | depends on duplicate boxes | depends on RBC detection | can abstain | yields a probability |
|---|---|---|---|---|---|---|---|
| R0 | Σ confidence | total evidence | no | **yes** | no | via κ | no |
| R1 | max confidence | one detection's score | no | no | no | via κ | no |
| R4 | count ≥ τ | number of detections | no | **yes** | no | via κ | no |
| R5 | share of fields voting | fraction of fields | **yes** | no | no | via κ | no |
| R6 | parasites per RBC | parasitaemia | no | **yes** | **yes** | via κ | no |
| R9–R11 | calibrated screening probability | probability | no | **yes** | no | via κ | **yes** |

The table states what the comparison *is*: **a comparison of fixed operational
rules at pre-registered parameters**, not of rule families at their best. That
declaration goes in the Methods, because the alternative reading — that each
rule was given its best shot — is the one a reader will otherwise assume.

### 5.3 Parameter rationale (new, required)

| parameter | value | rule | source | pre-registered |
|---|---|---|---|---|
| τ | 0.50 | R4 | the source study's own reported operating point | yes |
| m | 2 | R4 | minimum evidence: one box is a single-detection artefact | yes |
| k | 5 | R3 | top-*k* robustness, an order of magnitude below the median box count | yes |
| ρ | 0.05 | R6 | parasitaemia threshold in routine reporting | yes |
| ρ | 0.001 | R6b | a screening-sensitive threshold, declared as a sensitivity analysis | yes |
| β | 10 | R8 | log-sum-exp sharpness placing it between max and sum | yes |
| θ_vote | 0.50 | R5 | matches τ so the two counting-family rules share a floor | yes |

**None was fitted to this dataset.** That is what makes the confirmatory test
valid, and it is why the table exists.

---

## 6. Post-processing harmonisation

Retitled, per R1 §10, to say what it demonstrates:

> **Why raw detector outputs are not comparable across architectures**

Three conditions: `raw` (as cached), `matched` (per-class NMS at IoU 0.45,
applied identically to every checkpoint), `source` (the published operating
point: 0.25 floor, class-agnostic NMS at 0.45).

**`matched` is the primary condition and the only one in which a checkpoint
comparison is interpretable.** On the released weights RT-DETR emits a fixed
query budget per field — the same count on a parasite-dense smear as on a clean
one — so any statistic that counts, averages or thresholds boxes measures the
decoder before it measures the detector. The box census (§16) is the evidence.

Post-processing is a **standardising condition, not a third experimental
factor.** It is not crossed into the factorial and no p-value is computed across
it.

---

## 7. Propositions

Statements and their practical consequence in the main text; **proofs in
Supplementary Methods**.

| | statement | what it licenses or forbids |
|---|---|---|
| P1 | scale invariance | a rule's threshold may be reported in its own units without loss |
| P2 | screening equivalence | a per-class recalibration cannot change a screening call, so enrolling a calibrated rule as a confirmatory comparison on D₁ guarantees *b*=*c*=0 |
| P3 | headroom | a baseline that is already perfect on the negatives offers no win, whatever the candidate does |
| P4 | oracle ceiling | the gap between the oracle and the baseline is the only quantity here that does not depend on which rules were tried |
| P5 | design identifiability | when every uninfected patient shares one centre, specificity and domain are one column |

P2 and P3 are the operative ones: they are why comparisons are removed **before**
unblinding rather than reported and discounted afterwards.

---

## 8. Gates

Order is fixed and each gate is evaluated before anything it could contaminate:

```
identifiability → yield → pre-declared checkpoint → factorial
    → reconstruction → design → confirm
```

### 8.1 Identifiability (P5) — qualifies, does not stop

Fails on this archive. Its verdict travels into every specificity sentence.

### 8.2 Yield — post hoc, and declared so

Written after a released checkpoint returned two boxes across 4,508 fields.
Thresholds stated rather than tuned; far from any defensible detector behaviour,
because it catches a harness fault and does not grade detectors. A refused arm is
**named with its cause** — which requires the YOLOv5x probe, not silence.

### 8.3 Checkpoint selection — CHANGED

**v2 selected the confirmatory checkpoint by maximum baseline headroom.** Both an
independent reviewer (R2 §2.1) and this package's own null simulation found the
same defect, from opposite directions. The simulation settles it:

```
headroom  = b + (patients both rules get wrong)
```

where *b* is the numerator of the test statistic. Selecting the largest headroom
selects, in part, the largest *b*. Simulated at three comparable arms the size
rises to **≈6.5% against a nominal 5%** — an inflation, not the promised
conservatism, and the argument v2 gave for it ("headroom reads the baseline
only") is simply wrong.

**v3 pre-declares the checkpoint.** RT-DETR, named in the registration, chosen
because it is the architecture the source study reports — a fact independent of
any outcome in this dataset. Headroom is retained as a **descriptive feasibility
statement** reported alongside, never as a selector.

Two analyses remain, both pre-specified:

- **max-discordance selection**, reported as a sensitivity analysis. It is
  level-preserving by construction: the exact McNemar conditions on *n* = *b*+*c*,
  and conditional on *n* the statistic is Binomial(*n*, ½) whatever else is true,
  so a rule that is a function of the counts alone leaves that distribution
  untouched. Simulation agrees.
- **the size simulation itself**, reported in the supplement, because a design
  paper that found a size inflation in its own procedure should show it.

### 8.4 Reconstruction gate

Refuses when the reconstructed rule does not land within tolerance of the
published figure. **Refusal is the expected outcome and is a finding.** Forcing
past it is permitted, is recorded in the results object, and changes the
comparator's description in every generated sentence.

### 8.5 Design gate (P2, P3)

Refuses a confirmatory comparison whose candidate is screening-equivalent to the
comparator, or whose comparator has no headroom. Its discordance count is taken
**among the uninfected**, the subset the specificity test conditions on.

Applied to the sensitivity-analysis arm as well as the primary — v2 tested the
alternative arm without gating it and printed *p* = 1.0000 for a comparison that
could not have produced a non-null result.

---

## 9. Factorial attribution — REBUILT

This is the paper's second contribution and v2's weakest analysis.

### 9.1 The v2 claim, and why it does not survive

v2 compared a range across fourteen rules with a range across two or three
checkpoints. A range grows with the number of levels; the two factors have
wildly different level counts; the comparison is outlier-sensitive. v2 also
claimed the interaction "is arithmetically the residual and is not separately
estimable."

**That claim is wrong, and R2 §2.5 is right to reject it.** Every patient
contributes an outcome under *every* rule–checkpoint cell. The design is a
complete factorial with **170 repeated patient-level observations per cell**, not
one observation per cell. What v2 actually lacked was independent residual error
*at the cell-mean level* — a different and much narrower statement.

### 9.2 The v3 primary analysis

A patient-clustered model on the binary patient-level outcome:

```
logit P(Y_irc = 1) = μ + α_r + β_c + (αβ)_rc + u_i
```

*r* rule, *c* checkpoint, *u_i* a patient effect. Fitted as a GEE with an
exchangeable working correlation and robust (sandwich) standard errors clustered
on patient. Reported:

- marginal rule effect and its clustered interval,
- marginal checkpoint effect and its clustered interval,
- **the rule × checkpoint interaction**, with a test,
- the contrast between the two marginal effects, which is the paper's claim.

**Pre-declared estimand:** the difference between the marginal rule effect and
the marginal checkpoint effect, on the log-odds scale, over the levels that
survive the yield gate.

### 9.3 What the ranges become

Descriptive, secondary, and labelled. The permutation null for the checkpoint
factor is retained (it is valid: permuting each rule's values across checkpoints
leaves that rule's mean untouched). **No permutation null is reported for the
rule factor** — with one observation per cell-mean there is no permutation that
removes a rule effect without removing the values it is made of, and v2's
attempt returned a null equal to the observed spread to the last decimal.

The patient-level bootstrap of the *difference between the two spreads* stays,
because it answers the reader's real question with the patients as the
resampling unit.

### 9.4 Required wording

Forbidden: *"the rule matters more than the detector"*, and any multiplier
("twelve times more") computed from a ratio of ranges.

Required form:

> Within the levels surviving the yield gate and under matched post-processing,
> the marginal effect of the aggregation rule on patient-level accuracy exceeded
> the marginal effect of the detector checkpoint by [Δ] on the log-odds scale
> (95% CI [·], patient-clustered). The descriptive accuracy range across rules
> was [·] percentage points against [·] across checkpoints; these are ranges over
> factors with different numbers of levels and are not variance components.

---

## 10. The confirmatory comparison

### 10.1 Structure

Intersection–union co-primary on the pre-declared checkpoint, one candidate
(R4) against the comparator (R0):

- **specificity**, superiority, one-sided exact McNemar;
- **sensitivity**, non-inferiority at a 5 percentage-point margin.

Both components one-sided **in the same direction** at α = 0.05. An IU test needs
no multiplicity correction and none is applied; it also inherits the *worst*
component's level, which is why §10.3 matters.

### 10.2 Presentation: discordances first

The inferential object is the paired 2×2. Report it as such, before any
percentage-point difference — a reader currently cannot reconstruct the result
from the manuscript.

| | comparator correct | comparator wrong |
|---|---|---|
| **candidate correct** | *a* | ***b*** |
| **candidate wrong** | ***c*** | *d* |

Required for each endpoint: *a*, *b*, *c*, *d*; the discordant total; the exact
*p*; and the direction. Generated sentence, e.g.:

> On the specificity endpoint the candidate corrected *b* comparator false
> positives and introduced *c*, giving *n* discordant pairs and an exact
> one-sided *p* of [·].

### 10.3 Non-inferiority — method fixed

Δ = sensitivity(R4) − sensitivity(R0). Non-inferiority requires the one-sided
95% **lower** bound on Δ to exceed −0.05.

**The bound is Tango's score interval.** The percentile bootstrap v2 used is
only first-order accurate, and at this endpoint's parameters — a comparator
sensitivity near 0.97, so per-patient differences almost always 0 or −1 — it does
not hold its level: simulated type I error at the NI null was **0.063 to 0.076
against a nominal 0.05**, and because this is half of a co-primary the whole
endpoint inherited it. Tango holds (0.017–0.044 simulated) and retains 0.93
power. Both limits are computed and reported; Tango governs.

### 10.4 Detectable effect, not observed power

Observed power is a monotone function of the *p*-value and is not reported.
Reported instead: given the realised sample size and discordance, the smallest
true effect the procedure would find with 80% probability — a question whose
answer does not change after the data.

Including the fact that below five discordant pairs **no split whatsoever**
reaches a one-sided 0.05. That is a property of the design, not of the detector.

### 10.5 The verdict

One generated sentence, each clause reading its own component, reused verbatim
in abstract, results, discussion and conclusion. It is never typed by hand.

If specificity passes and sensitivity does not: **the co-primary endpoint was not
met, and no superiority claim is made.** The design working as intended is a
result, not a disappointment.

---

## 11. Calibration — SPLIT

v2 mixed binary screening calibration with six-class calibration in one set of
numbers. Two analyses, reported separately, never pooled.

### 11.A Infection (binary)

P(infected) per patient. Brier, ECE against its finite-sample null, calibration
slope and intercept, ICI, reliability diagram.

### 11.B Speciation (multiclass, conditional)

The class posterior among truly infected patients only. Multiclass Brier stated
by formula, per-class reliability, and an explicit statement of whether the
probabilities sum to one and how the uninfected mass is handled.

### 11.C Required reporting

- **Four decimal places.** A Brier reported as `0.000` is a formatting failure
  that all three reviewers flagged, and it is indistinguishable from a genuine
  zero, which would be impossible here.
- Every metric **cross-fitted out of fold**; the fold count and the number of
  degenerate folds reported beside it.
- Reliability diagrams carry **bin counts, intervals, and a score histogram**. A
  connected line over sparse bins is not a reliability diagram.
- Calibration acts on the screening statistic only — a monotone map on one
  number — so it can move the infection boundary and **cannot reorder species.**
  No speciation result may be attributed to it.
- ECE is not an inferential test. Permitted wording: *"the observed ECE lies
  within the finite-sample variability expected under the stated null."*
  Forbidden: any wording implying calibration was established.

---

## 12. Selective prediction

One confidence, identical across rules, so a difference between curves is a
difference between rules: κ = (π₍₁₎ − π₍₂₎) · *u*, the posterior margin times the
screening score. Coverage taken **by rank**, ties broken on patient identifier.

**The headline is workload, not AURC.** Required form:

> At 90% automated coverage the system referred [n] of [N] patients and achieved
> [·] sensitivity and [·] specificity among those still answered automatically.

Reported per endpoint and separately for infected and uninfected patients, with
the direction of the confidence stated and whether abstention is applied before
or after species assignment.

---

## 13. Field-sampling stress test — NEW, and the highest-value addition

Reviewer 2 §4.3, adopted in full. This is worth more than any additional rule.

Every field from an infected patient contains a parasite. Accumulation rules are
therefore evaluated under an assumption that screening violates. The concern is
not that all performance is inflated — it is that **the ranking of the rules may
be an artefact of the enrichment**, since R0 accumulates evidence without bound
and R4 requires a count.

**Design.** Hold the detector cache fixed. For each infected patient, construct
smears at a declared positive-field rate π ∈ {1.0 (as archived), 0.5, 0.25, 0.1}
by combining that patient's enriched fields with parasite-free fields drawn from
the uninfected patients of the *same* centre. Re-run every rule at every π.

**Reported:** each rule's screening sensitivity and specificity against π, and —
the actual question — **whether the ordering of the rules is stable across π.**

**Declared in advance:** this is a *sensitivity analysis under a stated
resampling model*, not an estimate of screening performance. The negative fields
come from a different patient, so it tests the rules' response to evidence
dilution and not the full screening problem.

If the ordering inverts at low π, that is a more important result than the
confirmatory comparison, and the protocol says so before the answer is known.

---

## 14. Prevalence-standardised predictive values — NEW

The cohort is 68% infected. Sensitivity and specificity transport; predictive
values do not. Report PPV and NPV at **declared target prevalences** (e.g. 1%,
5%, 20%) computed from the cohort's sensitivity and specificity, each labelled
with the prevalence assumed and with the estimand's scope from §4.1 attached.

Never report a predictive value at the cohort's own prevalence without saying it
is a referral-centre prevalence.

---

## 15. Domain analyses

- **Internal vs external**, difference in differences, **on speciation only.**
  Specificity cannot appear: every uninfected patient is internal, so the
  contrast would compare one centre with the empty set.
- **Leave-one-centre-out** on speciation, for the centres with enough infected
  patients to estimate anything, with the excluded centres named.
- **A sensitivity analysis excluding the two three-patient classes**, because a
  per-class recall over three patients moves in steps of 33 points.

---

## 16. Descriptive analyses

**Box census** — boxes per field, raw against matched, per checkpoint, with the
fixed-query-budget flag. This is the evidence for §6 and should be a main figure.

**Error taxonomy with fragility** — one row per patient: what was called, what it
should have been, the category, and how many detections would have to be removed
to change the call. Fragility distinguishes a result from a coincidence.

Required caveat, per R1 §32: a category share of 100% does **not** mean the
detector is perfect. It means that under *this decision rule*, every observed
patient-level error arrived through that pathway. The generated sentence says so.

**Field-count curve** — restricted to patients with ≥25 fields so the denominator
does not move along the x-axis, sampling universe padded to the *declared* field
count. Retitled to carry its scope:

> Sensitivity to field subsampling in the parasite-enriched test cohort

and the enrichment caveat appears **in the Results**, not only the Discussion.

**Reconstruction sweep** — declared post hoc everywhere. Reports not only whether
any configuration reaches the published total, but the **sensitivity/specificity
split of those that do**: reaching 135/170 with a different split is not
reproducing the result.

---

## 17. Forbidden and required language

| forbidden | required |
|---|---|
| "failed reproduction" | "not reconstructible from the released artefacts" |
| "the rule matters more than the detector" | the §9.4 marginal-effects form |
| "R4 improves malaria diagnosis" | "the aggregation rule changed patient-level performance in this cohort" |
| "the rule controls false positives" | "reduced false-positive calls in the Montpellier-negative subset of this archive" |
| "condemns a smear" | "a single surviving parasite detection produced a positive patient-level call" |
| "the detector separates species better than the aggregate does" | "the residual speciation errors are consistent with aggregation-induced loss of class-specific evidence" — with the confusion analysis attached |
| "pre-registered" unqualified | "registered before archive download and before analysis, after the source study's headline results were public" |
| "checkpoint factor" as an architecture comparison | "released-checkpoint variation within one detector system" |
| any Brier at three decimals | four decimals |
| "trial registration" | "registered analysis plan" — this is a secondary analysis, not a trial |

---

## 18. The claim–evidence–scope table

A **build artefact**, generated from the results object, printed in the paper.
Every row's verdict comes from the analysis, not from an author.

| claim | evidence | scope | status |
|---|---|---|---|
| aggregation is consequential | marginal rule effect, clustered CI | this cohort, surviving levels | generated |
| R4 improves specificity | discordance table, exact *p* | Montpellier negatives, RT-DETR | generated |
| R4 is superior overall | co-primary verdict | — | generated |
| checkpoints differ | marginal checkpoint effect | matched condition only | generated |
| rule effect > checkpoint effect | contrast of marginals | not variance components | generated |
| the published result is reconstructible | reconstruction gate | released artefacts | generated |
| rule ordering survives dilution | §13 stress test | stated resampling model | generated |
| clinical utility | — | outside the estimand | **not assessed** |
| deployment readiness | — | outside the dataset | **not assessed** |

---

## 19. Build integrity

**No PDF exists until this passes.** The audit runs after the results object and
before LaTeX; a failure is an error, not a warning.

1. **No unresolved macro.** Any `\R{}` without a definition fails the build. In
   v2 they rendered as `[armsAnalysed]` and three reviewers read them as data.
2. **No hand-typed directional claim.** Every verdict is a generated sentence,
   each clause reading its own component.
3. **One results object.** Figures and tables descend from the same `Study`; the
   figure stamp carries the provenance line and the seed, and the manuscript
   build refuses on a mismatch.
4. **Cross-artefact agreement.** Every number appearing in both a table and a
   figure is checked equal. v2 had AURC at 0.027 in text and 0.471 in a figure.
5. **Arithmetic.** Sensitivity, specificity and accuracy recomputed from the
   confusion counts and checked against the reported values.
6. **Precision floors.** Brier and ECE at four decimals; no metric printed at a
   precision that cannot distinguish 0.004 from 0.040.
7. **Provenance.** Checkpoint hashes, patient-table hash, seeds, versions, build
   id — present, or no results are written.
8. **Declarations complete.** Ethics, funding, CRediT, AI-use, repository, OSF
   and Zenodo identifiers present. **These are author-supplied; the pipeline
   checks that they are non-empty and never invents one.**

---

## 20. Manuscript architecture

```
Title            "Detector versus decision rule: a registered secondary
                  analysis of patient-level diagnostic aggregation in
                  automated malaria microscopy"

Abstract         structured: Background / Methods / Results / Conclusions
                 — generated verdicts only

1 Introduction   four paragraphs, ~25-30% shorter than v2
                 1.1 clinical problem
                 1.2 the detector → patient decision gap
                 1.3 the evidence gap
                 1.4 three explicit objectives

2 Methods        design & registration · dataset & estimands (§4 first) ·
                 checkpoints · post-processing · decision framework ·
                 the six rules + comparability · gates · endpoints ·
                 statistical analysis · provenance

3 Results        3.1 cohort and archive verification
                 3.2 reconstruction of the published pipeline
                 3.3 detector-output harmonisation
                 3.4 primary confirmatory comparison (discordances first)
                 3.5 rule vs checkpoint attribution (model, then ranges)
                 3.6 calibration — infection, then speciation
                 3.7 selective prediction as workload
                 3.8 error mechanisms
                 3.9 sensitivity analyses incl. the field-sampling stress test

4 Discussion     principal findings · why aggregation matters · why checkpoint
                 comparisons are hard · reproducibility implications ·
                 scope of inference · limitations in three tiers · generalisation

5 Conclusion     the minimum reporting standard, derived from what this study
                 found undocumented

Declarations · Data and code availability · References · Supplement
```

**Objectives → analysis → result matrix**, printed in the Methods, so every
analysis in the paper maps to an objective and nothing orphaned survives.

**Five main figures.** (1) study architecture; (2) reconstruction audit — where
the discrepancy sits; (3) the factorial heatmap; (4) the primary paired endpoint
as a discordance plot; (5) calibration and risk–coverage. Everything else to the
supplement.

**Limitations in three tiers**, per R3 §8: structural and unfixable (specificity
confounded with centre; enriched fields; imperfect reference standard);
study-specific and addressed by design (excluded arm, three-patient classes,
no multi-label); generalisation (one dataset, one detector system, referral
population).

**Scope of inference** appears **early**, not buried in the Discussion:
technical validity ≠ diagnostic validity ≠ clinical utility ≠ deployment
readiness.

---

## 21. What this protocol refuses to claim

Written here, before the analysis, so it cannot be softened afterwards:

- that the source publication is wrong;
- that R4 is a recommended rule;
- that any specificity estimate is site-invariant;
- that any performance figure transports to screening;
- that calibration was established;
- that the rule factor dominates the detector as a general fact;
- that anything here bears on clinical utility or deployment readiness.

The paper's defensible central message, which the design will support whichever
way the endpoint falls:

> Patient-level decision rules are consequential, this dataset is insufficient to
> support a deployment recommendation, and a registered design can expose that
> limitation rather than conceal it.

---

## 22. Acquisition — how the archive becomes evidence

Sections 1–21 describe what is done with the detections. This section describes
where the detections come from, and it is part of the protocol rather than an
appendix because two of its choices change what every downstream number means.

### 22.1 The cohort is the manifest's, not the filesystem's

`patient_id` is the archive's `smear_name`; one smear is one patient. `centre`
and `truth` come from the manifest's `origin` and `species` columns, mapped onto
this analysis's vocabulary through tables written out in `mag/ingest.py` rather
than derived. `n_images` is the manifest's count of rows for that smear.

Three refusals, all of them because the alternative is a silent change of
denominator:

* A species or centre value the mapping does not contain is refused. Passing it
  through would create a category the analysis has never seen and change the
  denominator of every endpoint that groups on it.
* A smear carrying more than one centre or species is refused. A patient has one
  of each; anything else means the patient table is not well defined.
* An image the manifest lists but the archive does not hold makes ingest exit
  non-zero. The count is what per-field statistics divide by, and a quietly
  smaller cohort is undetectable downstream.

Images present on disk but absent from the manifest are ignored and counted. The
manifest defines the cohort.

### 22.2 The transformation is measured, not inferred

The source study applied a "zoom correction" whose mechanism the publication
does not fully specify. The archive ships `test` and `test-zoom` as two label
trees over the same smears, so the retained fraction of annotations is a count.
`ingest.transform_retention` reports it per class with a verdict, and where the
retention is consistent with more than one mechanism the verdict says so and
leaves the mechanism open. **This protocol does not infer the transformation
from the images.**

### 22.3 The cache is pre-suppression, at a declared floor

Detection runs at a confidence floor of 0.001 with suppression effectively
disabled (`iou=1.0`, `max_det=3000`), and the manifest records `nms="none"`.

This is a protocol commitment, not a default. Every rule's threshold is applied
downstream in `mag/evidence.py`, so a cache written at the published 0.25
operating point could not evaluate any rule below it, and the three
post-processing conditions of §6 would collapse into one. The cost is a cache
roughly an order of magnitude larger; the benefit is that raw, matched and
source are all computable from the same measured pass.

### 22.4 The class map comes from the checkpoint, or is declared

Class indices are mapped to this analysis's vocabulary using the names embedded
in the checkpoint itself, matched through a synonym table that tolerates case,
spacing and the usual abbreviations and nothing else. Any name that does not
match is a refusal naming every unmatched index.

Where a checkpoint carries no names at all, the fallback to `classes.txt` order
is available but must be requested explicitly (`--assume-classes-txt`) and
**declared in the manuscript**. It is an assumption about which index means
which species; a wrong order relabels every detection with no other symptom.

### 22.5 A silent detector is a bug, not a result

A run producing implausibly few detections — fewer boxes than fields, or boxes
in under half the fields, at a floor of 0.05 or below — writes
`detections_<name>.suspect.json` beside its own cache, and every analysis stage
then refuses that arm.

This is separate from, and upstream of, the yield gate of §8. The gate protects
the analysis; this protects the interpretation. A checkpoint that returns two
boxes from four and a half thousand fields has not been loaded or driven
correctly, and reporting it as a weak detector would be a claim about the
checkpoint that the evidence does not support. The manuscript must state which
arms were excluded and why.

### 22.6 Reproducibility of the acquisition

Detection is resumable at smear granularity. A shard is reused only when its
fingerprint matches exactly: the smear's image list, the confidence floor, the
image size, the class map, and the checkpoint's SHA-256. Any change to any of
those invalidates the shard rather than mixing two runs in one cache.

The cache carries, and every downstream stage verifies: the checkpoint's
SHA-256 in both the manifest and `checkpoint_meta_<name>.json`; a content hash
of the detection table pinned to an exact serialisation; and the per-smear
shards themselves, which `mag/provenance.py` requires to cover at least half the
cohort before any result may be written. A table assembled by any other means
has none of those and cannot acquire them.
