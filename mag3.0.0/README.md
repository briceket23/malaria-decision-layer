# mag — patient-level aggregation in automated malaria microscopy

Reference implementation for the re-analysis of the public Guemas et al. malaria
detection archive. It takes the released images and the released weights, runs
the detectors, applies fourteen patient-level aggregation rules to the same
detections, and reports what separates them — under a protocol written before
the results were seen and enforced by the code.

Every number the manuscript reports comes out of a run of this package on the
released archive. There is no synthetic generation anywhere in `src/mag`, no
`--demo`, no default cache path, and no fallback that manufactures an input a
stage cannot find. A stage that has nothing to read stops and says so.

---

## What this answers

The source study reports a patient-level result. Reaching a patient-level
result from a field-level detector requires an **aggregation rule**, and the
aggregation rule is a modelling choice that the literature reports as if it
were an implementation detail. This package makes that choice the object of
study.

An aggregation rule is the triple `(E, g, θ)`: an evidence functional over a
patient's fields, a screening statistic, and a threshold. Fourteen of them are
implemented — from the trivial baseline through counting, accumulating,
field-normalised and density-referenced rules, to four rules whose parameters
are fitted by cross-fitting — together with three bounds that say what any rule
could achieve on this cohort.

Two questions follow, and the package answers both:

**Does the rule matter more than the detector?** A patient-clustered model on
the full rule × checkpoint factorial, fitted as a GEE with robust standard
errors and cross-checked against an independently written estimator. Reported
on the probability scale, because complete separation is the expected state on
this data and a log-odds contrast is undefined where a rule's outcome never
varies.

**Does the published result reconstruct?** Not "does it reproduce" — the
published number is an external fact this package did not compute. What can be
established is whether it is *reconstructible* from the released artefacts
under the documented procedure. The gate that asks reports its answer as a
finding about the release, not as evidence about the original implementation.

---

## Install

```bash
git clone <this repository>
cd mag
python -m pip install -e .
```

Python 3.9 or newer. The analysis needs numpy, pandas, pyarrow, matplotlib and
statsmodels, all pulled in by that command.

For the detection stage — the only stage that needs a GPU — also:

```bash
python -m pip install -e ".[gpu]"
```

which adds torch, ultralytics, openpyxl and Pillow. A reviewer re-running the
analysis from a published detection cache never needs these.

```bash
python -m pip install -e ".[dev]" && python -m pytest -q
```

runs the test suite: 292 tests, about three minutes.

---

## Run it

The whole pipeline, from the unzipped archive:

```bash
python run_local.py --root /path/to/unzipped/archive
```

That runs five stages in order and stops at the first one that does not finish
cleanly. Each is also a `mag` subcommand you can run on its own.

### 1. `doctor` — what is here

```bash
python run_local.py --root ARCHIVE --stages doctor
```

Reports what it found in the archive, what is installed, how much disk is free,
and what would block the stages you asked for. Run this first; it takes
seconds and it is where a CPU-only torch build or a missing spreadsheet engine
becomes visible, rather than three hours into a detection pass.

### 2. `ingest` — the archive becomes a cohort

```bash
mag ingest --image-list ARCHIVE/image_list.xlsx --archive ARCHIVE --out data
```

Reads the archive's own manifest and writes `data/patients.csv`: one row per
smear, with its centre, its reference diagnosis and its field count. Prints the
cohort census that becomes Table 1, reconciles the manifest against the files
actually on disk, and — where the archive ships both label trees — *measures*
the source study's zoom transformation rather than reasoning about it.

Every vocabulary it does not recognise is a refusal. A species spelling the
archive has never used would otherwise appear quietly as a seventh label and
change the denominator of every endpoint that groups on it.

`n_images` is the manifest's count, not the filesystem's. The two are
reconciled separately so that a missing file is a reported discrepancy rather
than a silently smaller denominator.

### 3. `detect` — the detectors over every field

```bash
mag detect --images ARCHIVE/images/test --out data/test \
           --weights rtdetr=W/rtdetr.pt yolov8x=W/yolov8x.pt yolov5x=W/yolov5x.pt \
           --patients data/patients.csv --device 0
```

The expensive stage: on the released test partition that is 170 smears and
about 4,500 fields per checkpoint. It writes a **pre-suppression cache** at a
0.001 confidence floor — deliberately low, because every rule's threshold is
applied downstream and a cache cut at 0.25 could not evaluate a rule below it.

It is resumable at smear granularity. A pass killed after 120 of 170 smears
resumes at 121, and a shard is reused only when its fingerprint — the image
list, the confidence floor, the image size, the class map and the checkpoint's
own SHA-256 — matches exactly. Batch size is chosen from the card's memory and
halves on an out-of-memory error rather than aborting the pass.

Three things it refuses to do quietly. A checkpoint whose class names are not
in the archive's vocabulary is rejected rather than having the unmatched
classes dropped. A checkpoint carrying no class names at all is rejected unless
you pass `--assume-classes-txt`, which is an assumption about which index means
which species and must be declared. And a run that produces implausibly few
detections — the failure mode that once returned 2 boxes from 4,508 fields —
marks its own cache suspect, and every analysis stage then refuses it.

For a YOLOv5 checkpoint in repository format, pass `--yolov5-repo` with a clone
of `ultralytics/yolov5`, or install the `yolov5` PyPI fork. When such a
checkpoint returns nothing, `python scripts/probe_v5.py WEIGHTS.pt FIELD.jpg`
isolates *where* the boxes vanish — in the weights, in the preprocessing, in
the fork's suppression, or in this package's adapter — by running one image
through each layer separately and printing what survives.

### 4. `analyse` — the experiment

```bash
mag all --cache data/test --patients data/test/patients.csv \
        --out results --tex results/results.tex --depth full
```

Builds the study once and writes results, figures and LaTeX macros from that
one build. Running `run`, `figures` and `report` separately does the same work
three times and can produce figures and text from two different builds; `all`
cannot, which is why the figures it writes carry a provenance stamp that
`report` checks.

The gates run in a fixed order and each can stop the study:

| gate | asks |
|---|---|
| identifiability | is the endpoint separable from centre at all? |
| yield | did this checkpoint produce a plausible number of detections? |
| declared checkpoint | did the pre-declared arm survive? |
| factorial | is the full rule × checkpoint grid complete? |
| reconstruction | does the published result reconstruct from the released artefacts? |
| design | can the confirmatory comparison produce a non-null result? |
| confirm | the intersection–union co-primary test |

The reconstruction gate stops the run by default. `--force-reconstruction`
continues past it and records that it was forced — but the gate refusing is a
**finding**, and it belongs in the paper before it belongs in a flag.

### 5. `audit` — before anything is typeset

```bash
mag audit --out results --manuscript paper/main.tex \
          --declarations declarations.json --json
```

Checks the finished directory rather than the study object, so it can be run on
results produced weeks earlier. It catches: a macro the manuscript calls that
the analysis never defined; a macro defined but empty; a directional verdict
typed into the prose where a generated sentence exists; retired vocabulary; a
rate that does not follow from its own confusion counts; a Brier score printed
to three decimals, where 0.000 and 0.0004 are different findings and the same
string; a number that disagrees between `results.tex` and `results.json`;
figures drawn from a different run or a different seed; and a declaration still
holding a placeholder.

It exits non-zero if any check fails, and there is no flag to bypass it.

`declarations.json` is **yours to write**. Copy `declarations.example.json`,
which carries all ten keys with a note on what each one wants, and replace every
value. The keys are:

```
ethics_source_study      ethics_present_study     consent_for_publication
funding                  author_contributions     ai_use
repository_url           osf_registration         code_archive_doi
data_availability
```

The audit checks that each is present and non-empty. It does not and will not
supply them: they are facts about your study, this pipeline has no way to know
them, and a pipeline that filled them in would be inventing an ethics approval.

---

## What comes out

In `results/`:

| file | what it is |
|---|---|
| `results.json` | every number, machine-readable, with the provenance line |
| `results.tex` | LaTeX macros — the manuscript reads these, never a typed number |
| `factorial.csv` | every rule × checkpoint cell, with confusion counts |
| `predictions.csv` | the per-patient call of every arm |
| `errors.csv` | every misclassified patient, with a fragility count |
| `calibration.csv` | Brier, ECE, ICI and the null reference, per condition |
| `claims.csv` | each claim, its estimand, its evidence and its scope |
| `stress.csv` | rule ordering under field-sampling dilution |
| `predictive.csv` | PPV and NPV at declared prevalences, never bare |
| `selective.csv` | risk–coverage and AURC, per arm |
| `sweep.csv` | the reconstruction sweep |
| `box_census.csv` | boxes per field, raw against suppressed |
| `rule_comparability.csv`, `rule_parameters.csv` | the rule registry, as tables |
| `provenance.json` | the chain of custody |
| `figures/` | every figure, plus a stamp tying it to this run |
| `audit.json` | the audit, when `--json` was passed |

---

## How it refuses to fabricate

This matters more than any feature, because the predecessor could generate a
cohort and write a results file from it — and did, and two external reviews
were then carried out on the resulting PDF. Of roughly thirty reported
numerical contradictions, three were real and the rest were the placeholder
file.

Five arrangements make that impossible rather than unlikely, and each is pinned
by a test in `tests/test_no_fabrication.py`:

The package contains no generator. No module under `src/mag` may import the
test fixtures, define a cohort builder, or contain a stub detector. The stub
that lets `detect` be tested without a GPU lives in `tests/test_acquire.py` and
is reachable only from there.

No analysing command has a default cache or a default cohort. A default path is
how a pipeline quietly analyses the wrong thing.

`cache.save` is called from exactly one place in the package. Nothing but the
detector writes a detection cache, because the manifest beside it asserts a
SHA-256 for the weights that produced it, and a second writer is a second way
for that assertion to become untrue.

`ingest` reads the manifest and never completes it. There is no default centre,
no default diagnosis, no `fillna` on a value that identifies a patient.

A cache cannot be reported without its chain of custody. `report.write` refuses
unless `checkpoint_meta_<name>.json` is present, its SHA matches the manifest's,
and the per-smear shards cover at least half the cohort. A hand-assembled table
has none of those and cannot acquire them.

The cache also carries a content hash pinned to an exact serialisation, so a
table edited after the fact — or round-tripped through a tool that narrows a
float column to 32 bits — is refused by name on the next load rather than
analysed.

---

## What the numbers mean, and what they do not

Two properties of this archive change what is being estimated, not merely how
precisely it is estimated. Both are declared in `mag/estimand.py` before any
analysis runs, and the scope travels with every generated sentence that quotes
the number.

**Specificity is not identifiable here.** Every uninfected patient in the
released test set comes from one centre, which also supplied training material.
Specificity is therefore perfectly collinear with centre and with
internal/external domain. This is rank deficiency, not low power: no estimator
separates them and no sample size fixes it. The package reports specificity as
a property of *that cohort*, and refuses the wording that would make it a
property of the rule.

**Sensitivity is conditioned on enriched sampling.** Every field from an
infected patient was selected to contain a parasite. That is not a sampling
regime screening provides, and the rules respond to it differently by
construction. The concern is not that the figures are inflated — it is that the
*ordering* of the rules might be an artefact of the enrichment, and the
ordering is what the second contribution rests on. So the package measures it:
`mag.stress` dilutes each infected patient's fields with parasite-free donor
fields down to declared positive-field rates, holding the field count and the
detector cache fixed, and reports Kendall's τ of the rule ordering against the
undiluted one. Nothing is re-inferred and no detection is invented.

**Predictive values are never reported bare.** This cohort is 68% infected
because a referral centre assembled it. A PPV computed at that prevalence and
quoted without it will be read as a screening PPV and is roughly an order of
magnitude too high for one. `mag.predictive` reports a small table across
declared prevalences with patient-level bootstrap intervals, and attaches the
assumption to every cell.

---

## Repository layout

```
src/mag/
  ingest.py       archive -> patients.csv; vocabularies, layout, transform
  detect.py       the detectors -> the cache; sharding, resume, refusals
  discover.py     find the archive and the weights; report the environment
  cache.py        read and write the cache, with integrity checks
  schema.py       the vocabularies and the two table contracts
  evidence.py     suppression conditions and per-patient views
  rules.py        the fourteen rules and three bounds, as a registry
  estimand.py     what each endpoint estimates, and its scope
  metrics.py      screening, overall, speciation, band-referenced
  stats.py        exact McNemar, Tango non-inferiority, intersection-union
  gates.py        the seven gates, in protocol order
  calibrate.py    Brier, ECE, ICI, isotonic, Platt, temperature
  selective.py    risk-coverage and AURC
  attribute.py    the rule x checkpoint model, and its cross-check
  stress.py       field-sampling dilution and ordering stability
  predictive.py   PPV and NPV at declared prevalences
  power.py        detectable effects, exactly, not observed power
  simulate.py     the null simulations
  search.py       the reconstruction sweep
  diagnostics.py  errors, field curve, leave-one-centre-out
  study.py        the experiment: one build, everything from it
  report.py       the LaTeX macros
  figures.py      the figures, with a provenance stamp
  audit.py        the build audit
  provenance.py   the chain of custody
  cli.py          one entry point
docs/
  PROTOCOL.md         the specification the code and the manuscript both obey
  IMPLEMENTATION.md   each protocol section mapped to its implementation
  RUNBOOK.md          the operational sequence, with timings
scripts/
  probe_v5.py         isolate where a YOLOv5 checkpoint's detections vanish
tests/              292 tests
run_local.py        the staged driver
declarations.example.json   the ten author-supplied declarations, as a template
```

---

## Reproducing a published run

Every run stamps a provenance line naming the cache, the package version, the
build id, the post-processing condition, the checkpoints with their content
hashes, and the cohort hash. Two runs agreeing on that line and disagreeing on
a number is a bug in this package, and the test suite has a test that says so.

To check a result you have been given: run `mag all` with the same
`--condition`, `--depth` and `--seed` against the same cache, and compare
`results.json`. The content hashes in the provenance line are what establish
that the two runs saw the same detections.

---

## Citation and licence

MIT, see `LICENSE`. `CITATION.cff` carries the citation metadata; its author
list is marked `NOT REPORTED — supply before release` and is yours to complete,
as are the repository URL, the registration and the code archive DOI in
`declarations.json`.

This package analyses a public archive it did not create. Cite the source
dataset and the source study alongside it.
