# Malaria Decision Layer

[![reproduce](https://github.com/briceket23/malaria-decision-layer/actions/workflows/reproduce.yml/badge.svg)](https://github.com/briceket23/malaria-decision-layer/actions/workflows/reproduce.yml)

Analysis engine, driver scripts and cached results for the study
**Patient-level screening is not determined by detector weights alone: a
secondary reanalysis of the downstream inference configuration in automated
malaria microscopy.**

The study asks how the decisions taken *after* an object detector ---
confidence floor, suppression, aggregation across fields, patient-level
threshold --- determine the screening call a patient receives, and whether the
public Plasmodium-v1 release carries enough of that configuration for a third
party to rebuild its published patient-level result.

> **Scope.** This is a secondary computational analysis of one public archive.
> Its results are not evidence of prospective clinical validity, deployment
> readiness, or performance in settings the archive does not represent.

---

## Contents

- [For reviewers: check the numbers in a few minutes](#for-reviewers-check-the-numbers-in-a-few-minutes)
- [The question](#the-question)
- [Findings](#findings)
- [Dataset and cohort](#dataset-and-cohort)
- [Methods in brief](#methods-in-brief)
- [Repository layout](#repository-layout)
- [Requirements and installation](#requirements-and-installation)
- [Re-running the experiment from the raw archive](#re-running-the-experiment-from-the-raw-archive)
- [Reproducibility record](#reproducibility-record)
- [Limitations](#limitations)
- [Responsible interpretation](#responsible-interpretation)
- [Citation](#citation)
- [Contributing](#contributing)
- [Licence](#licence)

---

## For reviewers: check the numbers in a few minutes

Cloning this repository is enough to verify every reported number it holds.
No GPU, no network, none of the raw archive; about one minute on a recent
laptop:

```bash
git clone https://github.com/briceket23/malaria-decision-layer.git
cd malaria-decision-layer
python -m pip install -r requirements.txt      # Python 3.9 to 3.12
python orchestrate.py reproduce
```

or, with Docker:

```bash
docker build -t malaria-decision-layer .
docker run --rm malaria-decision-layer
```

`reproduce` runs four steps and ends with `REPRODUCED` or `NOT REPRODUCED`:

| step | what it checks | result on the completed run |
|---|---|---|
| **compare** | every stored quantity --- per-patient calls, factorial cells, result macros, `results.json` blocks --- against `baseline/frozen_results.json`, with the 49 confusion tables recomputed from the per-patient calls | **10,212 identical**, 0 consequential |
| **revision** | re-derives the revision layer from the cached artefacts: marginal spans, symmetric permutation nulls (10,000 replicates, seed 0), exact McNemar, Tango score bounds, Wilson intervals | written to `rederived/`, never over the frozen file |
| **diff** | the re-derived values against the frozen `manuscript/revision.tex` | **47 identical**, 0 different, 3 not re-derivable here |
| **test** | the offline suites, then the 163-assertion experiment suite against the engine | all pass |

A change of a single digit in any reported quantity makes `compare` report it
as CONSEQUENTIAL and exit non-zero; this was tested by altering one macro and,
separately, one factorial cell. The same command runs on every push in GitHub
Actions on Python 3.9 (the run's interpreter) and 3.12, together with the
engine's own 292 tests; the badge above is its result.

**What it cannot check.** `cohortDupCheck` needs the archive manifest, which is
part of the Zenodo deposit; `r0Threshold` and `r0ThresholdSource` need the
analysis plan's threshold record. `reproduce` names all three and does not
count them as passes. Rebuilding the detection cache itself needs the raw
archive and a CUDA GPU ([below](#re-running-the-experiment-from-the-raw-archive)).
The reference labels used to recompute the confusion tables are the engine's
per-patient copy in `results/errors.csv`; their primary source is the
archive's manifest.

---

## The question

A detector trained for automated malaria microscopy reports parasites in
single microscope fields. A clinic needs one call per patient. Between the two
sits a chain of choices:

1. **Detection** --- the released checkpoint scores candidate boxes on each field.
2. **Confidence floor** --- boxes below a score are discarded.
3. **Suppression** --- overlapping boxes are merged, per class or across classes.
4. **Aggregation** --- a rule turns the surviving boxes of all a patient's fields
   into one score.
5. **Threshold** --- the score becomes infected or uninfected, and a species.

We call steps 2--5 the *downstream inference configuration*. This repository
holds each checkpoint's detections fixed --- one cached pass per checkpoint ---
and varies only that configuration, so any change in a patient's call is
attributable to it and not to the detector.

**The principle the study tests:** a patient-level result is a property of
the whole inference pipeline, not of the detector weights alone. A release
that publishes the weights but not the configuration does not let anyone else
check its patient-level claim.

### Questions

- **Q1 --- How far do downstream settings move patient-level screening** when
  the cached detections and the aggregation rule are held fixed?
- **Q2 --- Do the released artefacts rebuild the published patient-level
  result?** If not, where do the rebuilt and published results disagree?
- **Q3 --- Are the three released checkpoints comparable** once their output
  passes through the same downstream processing?

These are questions about inference behaviour and computational
reproducibility. They do not establish clinical utility.

---

## Findings

Every value below is generated by the engine and stored in this repository;
`orchestrate.py reproduce` checks it.

### 1. The downstream configuration alone can change the diagnosis categorically

On the RT-DETR checkpoint, the same cached detections were passed through two
post-processing conditions, with the same aggregation rule (summed confidence)
and an unchanged threshold:

| | published operating point | matched post-processing |
|---|---:|---:|
| confidence floor | 0.25 | none |
| suppression | class-agnostic, IoU 0.45 | per class, IoU 0.45 |
| **sensitivity** | 99.1% | **100.0%** |
| **specificity** | 74.1% | **0.0%** |

Without a confidence floor the rule sums the entire low-confidence tail, and
every patient clears the threshold: all 54 uninfected patients are called
infected. Nothing about the detector changed between the two columns.

A high sensitivity therefore says little on its own. It has to be read with
the specificity and with the configuration that produced both.

### 2. The published patient-level result was not rebuilt from the released artefacts

Scored in the release's six-class scheme (infected status **and** species must
both be right), the documented rule classified **105 of 170** patients
correctly, against **116 of 170** in the source publication. In the five-class
scheme, which merges *P. ovale* and *P. vivax*, it was 122 against 135.

The divergence is located, not just counted:

- **Uninfected patients** matched the publication exactly: 40 of 54 correctly
  called uninfected, as published.
- **The operating point** matched: sensitivity 99.1%, specificity 74.1%.
- **Infected patients** were detected as positive in 115 of 116 cases.

Screening behaviour was therefore reproduced; the shortfall lies in **species
assignment** among infected patients.

A post hoc sweep of 276 configurations found 3 that reach the published 116
exactly --- but at specificities of 92.6% to 96.3%, against the published
74.1%. A total matched at a different operating point is a different result,
not a reconstruction. We read this as a finding about what the release
carries, not as evidence about the original implementation.

### 3. Raw detection counts are not comparable across the released checkpoints

The archive releases three checkpoints: RT-DETR, YOLOv5x and YOLOv8x. RT-DETR
returned exactly **300 boxes on every field** before suppression --- on a
parasite-dense smear and on a clean one alike. That is a decoder query budget,
not a detection count; after matched suppression it averaged 170.1 boxes per
field.

Any statistic computed over unfiltered output therefore counts queries, not
parasites. A comparison of checkpoints is meaningless until their output
conventions are documented and harmonised, which is why matched
post-processing is the primary condition here.

### Predictive values do not transport

The same screening pair gives a positive predictive value of **94.1%** at this
cohort's prevalence (68.2% infected) and **7.0%** at a prevalence of 1%. No
accuracy figure in this repository is a deployment estimate.

---

## Dataset and cohort

The analysis uses the public **Plasmodium-v1** archive and its three released
detector checkpoints, from the Zenodo deposit
[doi:10.5281/zenodo.8358829](https://doi.org/10.5281/zenodo.8358829)
(version 1.0, CC BY 4.0). The archive is not redistributed here.

| test partition | value |
|---|---:|
| patients (one smear each) | 170 |
| infected | 116 (68.2%) |
| uninfected | 54 |
| microscope fields | 4,508 |
| centres | 6 |

The reference standard is the species recorded against each smear in the
archive's manifest. How it was produced (number of readers, adjudication,
molecular confirmation) is described in the source publication and is not
recoverable from the release; so far as the release shows, it is single-read
light microscopy. Every accuracy here is measured against those labels, not
against independently verified infection status.

### Two properties of the cohort that shape every estimate

**All 54 uninfected patients come from one centre, Montpellier.** Specificity
and centre are therefore perfectly collinear. Specificity can be computed for
these patients, but the cohort cannot separate a centre effect from an
infection-status effect, and no specificity here is an estimate for uninfected
patients from other centres.

**Infected patients' fields are parasite-enriched: 96.7% of them carry an
annotated parasite.** Sensitivity is therefore measured under sampling far
more favourable than routine screening, where many fields of an infected smear
are empty. This is a field-level property of the sample, not a prevalence and
not a correction factor.

---

## Methods in brief

### Detector output held fixed

Each checkpoint was run once over all 4,508 test fields at a 0.001 confidence
floor with suppression disabled, and the detections were cached. Every
condition and every threshold is computed from that one pass, so the
experiments measure the downstream configuration, not detector variability.
They do not measure robustness to image acquisition, retraining or domain
shift.

### Rules and conditions

The engine implements 14 aggregation rules plus three reference baselines,
including the oracle bound; six
are reported in the main analysis. Each rule is stated as a formula over the
cached detection set in `ALGORITHMS.md`. Three post-processing conditions are
defined: `raw` (as cached), `matched` (per-class suppression, IoU 0.45, applied
identically to every checkpoint --- the primary condition) and `source` (the
published operating point).

### Three tasks, kept apart

- **Screening** --- infected or uninfected. Sensitivity and specificity belong here.
- **Speciation** --- the species call among patients who are infected.
- **Joint** --- a patient is correct only if both are right. The factorial's
  accuracy is a joint-task figure.

Finding 2 is a speciation divergence, not a screening one; keeping the tasks
apart is what makes that visible.

### Units of analysis

The unit of analysis is the **patient**. Fields from one patient are not
independent observations, and patients are clustered by centre. No field-level
quantity is reported as patient-level inference.

### Statistics

| procedure | used for |
|---|---|
| Wilson intervals | every factorial cell |
| exact one-sided McNemar test | paired specificity comparison of two rules |
| Tango's score bound | paired non-inferiority of sensitivity |
| intersection--union test | the co-primary endpoint |
| permutation nulls, 10,000 replicates | marginal spans of rule and checkpoint |
| cross-fitted calibration, Brier score, ECE | the screening statistic |

Monte Carlo *p*-values are (1+r)/(1+B). Every seeded procedure uses seed 0.

A patient-clustered GEE model of the rule × checkpoint interaction is
implemented in the engine, but **no coefficient or Wald statistic from it is
reported**: two independent implementations disagreed by 0.076 on the log-odds
scale against a pre-set tolerance of 0.001, and an interaction statistic that
cannot be reproduced across implementations is not treated as evidence.

The pre-specified comparison of the counting rule R4 against the summed-
confidence comparator R0 is reported exactly as planned, but under matched
post-processing the comparator called every patient infected, so its
specificity component is a comparison against a constant. That is stated
wherever the endpoint appears; it is not read as evidence of an advantage over
a functioning comparator.

---

## Repository layout

```
orchestrate.py         every stage, one entry point; `reproduce` is the reviewer's
mag/                   the analysis engine, mag 3.0.0 (build 8d3140ff57):
  src/mag/               every decision rule, gate and statistical procedure
  tests/                 its own 292 tests (`orchestrate.py engine-test`)
  docs/                  PROTOCOL.md, IMPLEMENTATION.md, RUNBOOK.md
analysis/              the driver scripts
  run_all.py             the whole experiment from the raw archive
  run_experiment.py      drives the engine's stages
  revision_analyses.py   the auditable revision layer
  baseline_tools.py      freeze the completed run; compare every later build
  ontology_correction.py six-class vs five-class scoring
  verify_models.py       checkpoint identity and the detection-cap probe
  eda_dataset.py         archive verification and identifiability
  make_*.py              write the generated result layers
  test_*.py              the driver's test suites
  OFFLINE_RUNBOOK.md     step-by-step guide to a full re-run
manuscript/            GENERATED result layers only --- every reported quantity
                       as a \defresult{name}{value} line --- and the result
                       figures. See manuscript/README.md. The paper's sources
                       are not in this repository.
results/               the completed run: per-patient calls, the factorial,
                       results.json, environment.json, provenance.json
results_source/        the source-condition companion run
data/                  per-checkpoint manifests, geometry-exclusion record
baseline/              frozen_results.json, the reference for every compare
ALGORITHMS.md          every stage as pseudocode, every rule as a formula
Dockerfile             Python 3.9 with the exact pins; runs `reproduce`
.github/workflows/     `reproduce` and the engine tests on every push
requirements.txt       pinned to the versions in results/environment.json
```

---

## Requirements and installation

**To verify the results:** Python 3.9 to 3.12 and the packages in
`requirements.txt`, which pins the exact versions of the completed run
(numpy 2.0.2, pandas 2.3.3, pyarrow 21.0.0, matplotlib 3.9.4, statsmodels
0.14.6). numpy 2.0.2 has no build for Python 3.13 or later. Nothing needs
installing beyond that: `orchestrate.py` puts the engine in `mag/src` on the
path itself.

**To use the engine on its own:**

```bash
pip install ./mag            # analysis only
pip install "./mag[gpu]"     # adds torch, ultralytics, openpyxl, pillow for detection
mag --help
```

`python orchestrate.py check` reports the interpreter, the libraries, and the
engine's build id, and confirms it matches the build the completed run recorded.

---

## Re-running the experiment from the raw archive

Only needed to rebuild the detection cache itself. Nothing in the reviewer
section depends on it. It requires the Plasmodium-v1 images and checkpoints
from the Zenodo deposit, the GPU extras above, and a CUDA GPU. The reported
run used Python 3.9.13, PyTorch 2.7.1+cu118, Ultralytics 8.4.127 and an
NVIDIA GeForce RTX 4050 Laptop GPU on Windows 10; GPU memory needs depend on
model, image size and batch size.

```bash
python orchestrate.py check                      # what can run here
python orchestrate.py run  --root <archive>      # detection and analysis
python orchestrate.py full --root <archive>      # run -> revision -> compare
```

On the real archive the reconstruction gate refuses, by design; proceeding
past it needs `--accept-reconstruction-refusal "<reason>"`, and the reason is
written to `results/force_reason.json`. `analysis/OFFLINE_RUNBOOK.md` walks
through the run, its expected duration, and what its console messages mean.

---

## Reproducibility record

| item | where it is recorded |
|---|---|
| dataset | Zenodo doi:10.5281/zenodo.8358829, version 1.0; test partition as released |
| checkpoints | file hashes in `results/provenance.json` |
| detection cache | per-checkpoint manifests in `data/`, with hash, floor, image size and seed |
| inference configuration | every rule as a formula in `ALGORITHMS.md`; parameters in `results/` |
| engine | `mag/`, build `8d3140ff57`, matching `results/provenance.json` |
| software and hardware | `results/environment.json`, written by the run itself |
| randomness | seed 0 for every seeded procedure; detection is deterministic given the cache |
| outputs | per-patient calls, the factorial, all result macros, figures, logs |
| proof of no drift | `baseline/frozen_results.json`, checked by `orchestrate.py compare` |

`results/environment.json` and `results/provenance.json` are kept exactly as the
run wrote them, including the local paths they contain; editing them would
break the frozen baseline.

The analysis was not registered in a public registry. Analyses are labelled
pre-specified or post hoc throughout, against the authors' dated analysis plan.

---

## Limitations

1. **One public archive, secondary analysis.** No prospective performance,
   operational feasibility or patient outcome is assessed.
2. **Uninfected patients come from a single centre.** Specificity and centre
   are perfectly collinear; this is rank deficiency, not low power.
3. **Parasite-enriched fields.** Sensitivity is measured under sampling far
   more favourable than routine screening, and the bias is uneven across
   rules, so rankings as well as levels are at risk.
4. **Reference standard.** Single-read light microscopy, as far as the
   release shows, with no molecular confirmation and no inter-observer
   agreement to bound label noise. Two species are carried by too few patients
   for a per-class rate to be stable, and one label per patient places
   co-infection outside the analysis.
5. **Investigator choices.** The inference settings, including the unfloored
   primary condition, are choices whose consequences this study measures but
   cannot neutralise.
6. **Reconstruction referent.** Because the published result was not rebuilt,
   the comparator is this study's implementation of the documented rule.
7. **No external validation.** Results from this archive do not show that any
   decision rule suits other laboratories, populations, staining or imaging.

---

## Responsible interpretation

This is a research resource for studying how detector output and downstream
decisions interact in malaria microscopy. It is **not** a validated clinical
diagnostic system and must not replace laboratory assessment. Nothing in it
recommends any rule for clinical use.

The transferable content is the decomposition and the design --- what a release
must state for a patient-level claim to be checkable --- not any performance
figure.

---

## Citation

If you use this software or its outputs, please cite the repository using
`CITATION.cff` (GitHub's *Cite this repository* button reads it). The
associated article is under review; its reference and DOI will be added here
on publication, together with an archived release DOI.

Source data: Plasmodium-v1, doi:10.5281/zenodo.8358829 (CC BY 4.0).

---

## Contributing

Issues and pull requests that improve transparency and reproducibility are
welcome --- for example clean-environment reproduction reports, further tests of
aggregation or suppression rules, or clearer documentation of a detector's
output convention.

Any change must leave `python orchestrate.py reproduce` reporting
`REPRODUCED`. A change that moves a reported number is CONSEQUENTIAL: it must
be declared in `baseline/approved_changes.json` with a reason and described in
`CHANGELOG.md`.

---

## Licence

Code: MIT; see `LICENSE` and `mag/LICENSE`. The Plasmodium-v1 archive and its
checkpoints are distributed by their authors under CC BY 4.0 and are not
included here.
