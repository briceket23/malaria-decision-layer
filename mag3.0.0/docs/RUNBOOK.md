# Runbook — producing the paper's results on your machine

From an unzipped archive to a `results.tex` the manuscript builds against.
Written for Windows, since that is where the GPU is; every command works
unchanged on Linux with `/` for `\`.

One repository now, five stages. Only one of them needs a GPU:

| stage | GPU | time | what it produces |
|---|---|---|---|
| `doctor` | no | seconds | what is here, what would block |
| `ingest` | no | seconds | `patients.csv`, the cohort census, the transform measurement |
| `detect` | **yes** | ~50 min per image set | the detection cache |
| `analyse` | no | ~4 min | results, figures, macros |
| `audit` | no | seconds | every check that must pass before typesetting |

> **Set this once and the rest is copy-paste.** `$ARCHIVE` is the directory you
> unzipped the download into — the one containing `images/` and
> `image_list.xlsx`.

```powershell
cd D:\malariaaggregation
.\malaria_venv\Scripts\Activate.ps1
$ARCHIVE = "D:\malaria_france_data_set"     # <-- edit this to your path
```

---

## Step 1 — environment (once)

```powershell
pip install -e .
pip install -e ".[gpu]"
```

The second line adds torch, ultralytics, openpyxl and Pillow. Install a CUDA
build of torch, not the default wheel:

```powershell
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

**The `setuptools` trap, which cost an hour last time.** `pip install yolov5`
upgrades `setuptools`, and setuptools 82 removed `pkg_resources`, which the
yolov5 fork imports at module load. The install succeeds and the import then
fails with a bare `ModuleNotFoundError`. If you hit it:

```powershell
pip install "setuptools<70.0.0"
```

Verify before spending GPU time:

```powershell
python run_local.py --root $ARCHIVE --stages doctor
mag --version
```

`mag --version` prints a **build id** — a hash of every `.py` in the package. It
appears on the first line of every run and in every `results.json`. If a number
ever looks stale, check that id before checking anything else.

---

## Step 2 — ingest (no GPU, seconds)

```powershell
mag ingest --image-list $ARCHIVE\image_list.xlsx --archive $ARCHIVE --out data
```

Writes `data\patients.csv` and prints three things worth reading before you
spend an hour on detection:

**The cohort census.** Centre by species, with margins. This is Table 1. Check
the totals against what you expect before anything else runs on them.

**The layout reconciliation.** How many images the manifest lists against how
many are on disk. A listed image that is missing makes `ingest` exit non-zero,
because every per-field denominator would be wrong and no later stage could
detect it.

**The transform measurement.** Where the archive ships both `test` and
`test-zoom` label trees, this counts the annotations in each and reports the
retained fraction with a verdict. The source study's "zoom correction" is
therefore measured, not argued about.

---

## Step 3 — detection (GPU, resumable)

```powershell
mag detect --images $ARCHIVE\images\test --out data\test `
           --patients data\patients.csv `
           --weights rtdetr=$ARCHIVE\models\models\rt-detr\weights\best.pt `
                     yolov8x=$ARCHIVE\models\models\yolov8\weights\best.pt `
                     yolov5x=$ARCHIVE\models\models\yolov5\weights\best.pt `
           --device 0
```

`run_local.py --stages detect` does the same thing for every partition in
`--sets` and finds the weights in the archive itself, if they are where the
archive normally puts them.

It writes:

```
data\test\
  patients.csv
  detections_rtdetr.parquet      manifest_rtdetr.json      checkpoint_meta_rtdetr.json
  detections_yolov8x.parquet     manifest_yolov8x.json     checkpoint_meta_yolov8x.json
  detections_yolov5x.parquet     manifest_yolov5x.json     checkpoint_meta_yolov5x.json
  _shards\rtdetr\*.parquet       ...one per smear...
```

**Resumable — one shard per smear.** An interruption costs at most one smear;
run the same command again. A shard is reused only when its fingerprint — the
image list, the confidence floor, the image size, the class map and the
checkpoint's own SHA-256 — matches exactly, so changing any of those correctly
invalidates the work rather than silently mixing two runs.

Two things to avoid:

- **Do not pass `--no-resume`** after a successful run. It deletes the shards.
- **Do not delete `_shards\`.** `mag` requires non-empty shards covering at
  least half the cohort as its evidence that a real inference pass happened.
  Without them it refuses to write results, by design and with no bypass flag.

On a 6 GB card the batch size is chosen for you and halves on an
out-of-memory error rather than aborting. If it is still unhappy, `--batch 1`
and `--no-half` are the two dials worth trying, in that order.

---

## Step 4 — the YOLOv5x yield problem

On the last run, **yolov5x returned 2 detections across 4,508 fields**. That is
not a weak detector; something between the weights and the cache is dropping
everything.

Two mechanisms now catch it. `detect` marks its own cache suspect and writes
`detections_yolov5x.suspect.json`, and every analysis stage then refuses that
arm. Independently, the yield gate would refuse it too, naming it in
`\armsExcludedWhy`. So the analysis is safe either way — but the paper has to
say *why* the arm was excluded, and for that you need the probe:

```powershell
python .\scripts\probe_v5.py <yolov5x-weights.pt> <one-field.jpg>
```

It prints three stages — the raw forward pass before any suppression, the fork's
own NMS at both the study's operating point and the stock one, and the same call
through the adapter that actually wrote the cache — so the stage that loses the
boxes is visible rather than guessed at. Nothing is written to disk.

If it turns out to be the repository format, re-run detection with
`--yolov5-repo` pointing at a clone of `ultralytics/yolov5`:

```powershell
git clone --depth 1 https://github.com/ultralytics/yolov5
mag detect ... --yolov5-repo .\yolov5
```

Whatever the probe shows, **report it**. A checkpoint excluded by a stated,
pre-declared rule with a named cause is a methods paragraph. A checkpoint
quietly missing from a table is a reviewer's first question.

---

## Step 5 — analysis

```powershell
mag all --cache data\test --patients data\test\patients.csv `
        --condition matched --depth full `
        --out results\test-matched --tex results\test-matched\results.tex
```

`--depth full` adds the field curve, the null simulations, the field-sampling
stress test and the reconstruction sweep. It cannot change a `core` number; the
test suite has a test that says so.

Run the same command with `--condition raw` and `--condition source` into their
own directories. Post-processing is a **standardising condition**, not a third
factor: `matched` is where checkpoints are comparable, `raw` says whether the
checkpoint effect is really a post-processing effect, and `source` is the
published operating point.

| condition | depth | what it is for |
|---|---|---|
| **matched** | full | **the primary analysis** |
| raw | core | is the checkpoint effect post-processing? |
| source | core | the published operating point |

Each lands in its own directory so no two runs can overwrite each other's
`results.tex`.

---

## Step 6 — what the gates will say

**Expect refusals. They are the point.** A gate that stops the run has found
something the paper must report; it has not malfunctioned. Exit code 3 means a
gate spoke.

### The reconstruction gate will almost certainly refuse

The comparator scored **96/170** against a published 135:

```
GateRefusal: [reconstruction gate] the comparator scores 96/170,
39 below the published 135
```

**This is the paper's central finding**, and note the vocabulary: the published
number is an external fact this package did not compute, so the claim is that
it is *not reconstructible from the released artefacts under the documented
procedure* — a finding about the release, not evidence about the original
implementation. Record the exact numbers, then re-run with:

```powershell
mag all ... --force-reconstruction
```

Every affected `results.json` then carries `meta.forced_past_reconstruction:
true`, and `\reconVerdict` renders "not reconstructible from the released
artefacts" throughout the manuscript. From that point every downstream
comparison is against *our own operationalisation* of the published rule rather
than against the published system, and the generated sentences say so. That is
not a workaround; it is the honest framing, and it is why the flag exists
rather than the gate being a warning.

### The `source` condition may refuse at the *design* gate

`--force-reconstruction` does **not** wave this one through, and should not:

```
GateRefusal: [design gate] rtdetr:R4 is screening-equivalent to the comparator:
the comparison cannot produce a non-null result on this endpoint
```

Under the published post-processing the comparator has no headroom — it already
calls every uninfected patient the way the candidate does — so no candidate can
win, whatever it does. **That refusal is itself a result**: the published
operating point leaves nothing for an aggregation rule to improve on the
screening endpoint. Report it; do not go looking for a different rule to test,
which is exactly the practice this paper criticises.

### The identifiability gate will fail

All 54 uninfected patients come from Montpellier, so specificity is perfectly
collinear with centre. This does not stop anything — it **qualifies what may be
claimed**, and `\identifiabilityNote` carries the wording into the manuscript.
No sample size fixes it; it is rank deficiency, not low power.

### The attribution model will report complete separation

Expected, not a failure. R0's specificity is 0.000, so its level's outcome never
varies and its log-odds effect is unbounded. The package detects this, withholds
the log-odds contrast with a stated reason, and reports the probability-scale
contrast — which remains defined because it is not fitted. `\attrSeparated` and
`\attrLogOddsWhy` carry that into the manuscript.

---

## Step 7 — the audit, before you typeset anything

Write `declarations.json` first — `cp declarations.example.json
declarations.json` and replace every value. Every one of them is yours:

```json
{
  "ethics_source_study":     "...",
  "ethics_present_study":    "...",
  "consent_for_publication": "...",
  "funding":                 "...",
  "author_contributions":    "...",
  "ai_use":                  "...",
  "repository_url":          "...",
  "osf_registration":        "...",
  "code_archive_doi":        "...",
  "data_availability":       "..."
}
```

Then:

```powershell
mag audit --out results\test-matched --manuscript paper\main.tex `
          --declarations declarations.json --json
```

It exits non-zero if any check fails and there is no flag to bypass it. What it
catches, in the order it usually catches them: a macro the manuscript calls that
the analysis never defined (these rendered as `[name]` in the previous build and
were read as data); a directional verdict typed into the prose where a generated
sentence exists (the previous build's abstract said the co-primary conjunction
was met while its conclusion said the opposite — both were typed); a Brier score
printed to three decimals; a number that disagrees between `results.tex` and
`results.json`; and figures drawn from a different run or seed.

---

## Step 8 — where each part of the paper comes from

Everything below is in `results\test-matched\` unless noted.

| manuscript section | file | notes |
|---|---|---|
| Table 1 — the cohort | printed by `mag ingest` | centre × species, with margins |
| Table 2 — the factorial | `factorial.csv` | one row per (checkpoint, rule); `exact_test_valid` marks the arms an exact test is valid against |
| Table 3 — rule comparability | `rule_comparability.csv`, `rule_parameters.csv` | what each rule's statistic is on, and where every threshold came from |
| §Estimands | `results.json` → `estimands` | the target and scope of each endpoint, declared before the run |
| §Reconstruction | `\reconN`, `\reconTotal`, `\reconWhere`, `\reconVerdict` | plus `figures\reconstruction_sweep.pdf` and `\sweepAgreementSentence` |
| §Primary endpoint | `\primaryVerdict`, `\niStatement`, `\primaryB`, `\primaryC`, `\primaryP` | each clause reads its own component |
| §Attribution | `results.json` → `attribution`, `\attributionSentence` | the patient-clustered model, its interaction test, and the agreement between two implementations |
| §Design and selection | `\selSentence`, `\selSizeSentence` | including the size inflation of the retired criterion |
| §Power | `\powerSpecSentence`, `\powerSensSentence`, `\powerCaveat` | detectable effects, never observed power |
| §Calibration | `calibration.csv`, `figures\calibration.pdf`, `\calibrationSentence` | ECE against its finite-sample null |
| §Speciation calibration | `results.json` → `speciation_calibration` | kept separate from the binary infection calibration |
| §Triage / abstention | `selective.csv`, `figures\risk_coverage.pdf`, `\selectiveSentence` | |
| §Error analysis | `errors.csv`, `figures\errors.pdf`, `\errSentence` | fragility is the column reviewers will want |
| §Post-processing | `box_census.csv`, `figures\box_census.pdf`, `\boxSentence` | compare against `results\test-raw\` |
| §Field-sampling stress | `stress.csv`, `\stressSentence` | Kendall's τ of the rule ordering under dilution |
| §Predictive values | `predictive.csv`, `\pvSentence` | at declared prevalences, never bare |
| §Fields needed | `figures\field_curve.pdf`, `\fieldCurveSentence` | |
| §Claims | `claims.csv` | each claim, its estimand, its evidence, its scope |
| Additional file — nulls | `results.json` → `nulls` | winner's curse, cross-fit optimism, selection level |
| Additional file — sweep | `sweep.csv` | declared post hoc everywhere |
| Title page provenance | `\provenance` | cache, build id, checkpoint hashes, patient-table hash |

### Building the manuscript

Point the LaTeX build at the primary run's macros and figures. Never type a
number the macros already carry: the audit will catch a typed verdict, but it
cannot catch a typed number that happens to be plausible.

---

## If something goes wrong

**A typed error with a remedy.** Every refusal this package raises names what is
missing and what to do about it. There is no traceback because there is no bug —
re-reading the remedy is faster than re-running.

**`cache integrity failure`.** The table and its manifest describe different
data. Usual causes, in order: the file was written by a different run; it was
round-tripped through a tool that narrowed a float column to 32 bits; it was
edited by hand. Re-run detection rather than analysing it.

**A number that looks stale.** Check `mag --version`'s build id against the one
in `results.json`. Everything else is downstream of that.

**A figure and the text disagreeing.** That cannot survive `mag audit`, and if
you are seeing it, the figures and the results came from different runs. Re-run
`mag all`, which builds the study once and writes both from it.
