# Offline runbook

The complete experiment for *Separating the detector from the decision rule*,
packaged to run on your machine against the real Plasmodium-v1 archive.
Nothing in this package contains results. Every number comes out of your run.

## 0. What you need

Already true on your machine, from your own transcripts:

- Windows, PowerShell, the `malaria_venv` virtual environment (Python 3.9.13)
- `mag 3.0.0 build 8d3140ff57` installed (`pip install -e .` in the mag folder)
- torch 2.7.1+cu118 with CUDA available (RTX 4050 Laptop, 6.4 GB)
- the archive at `..\malaria_france_data_set` (relative to your working folder)
- the `yolov5` package installed (7.0.14 in your venv) -- no --yolov5-repo needed

Disk: keep roughly 3 GB free. The uncapped yolov5x cache is the largest
artefact, and the sensitivity arm adds a second copy of it.

## 1. Unpack

Put all files from this package into the SAME folder you ran v2.4 from
(`D:\malariaaggregation\malaria_experiment_suite`, per your last banner),
replacing the older versions of the same names. Use that folder and not
another one: its `data\test\` holds the shards from the eighteen-minute
pass, and this version repairs and reuses them instead of re-running the GPU.
The files:

    exp_schema.py            the JSON contract every script reads and writes
    eda_dataset.py           archive EDA (no GPU)
    verify_models.py         checkpoint verification + cap probe
    experiment_findings.py   the 14-check findings ledger
    run_experiment.py        ingest -> detect -> analyse -> sensitivity ->
                             audit -> manuscript
    run_all.py               ONE COMMAND that runs everything above
    test_experiment.py       the test suite (optional, no GPU, ~5 s)
    make_fixture.py          builds a tiny synthetic archive (only used by
                             test_experiment.py --with-archive)
    declarations.template.json   the ten author declarations, as a template

  the manuscript layer:

    main.tex                 the v3 manuscript. Every number in it is a
                             macro; nothing is typed
    make_archive_macros.py   archive.tex, from eda.json and verify.json
    make_result_tables.py    tables.tex: the factorial, field-curve, stress
                             and cap-arm tables, from results.json and
                             factorial.csv
    missing_results.py       MISSING_RESULTS.md and outstanding.tex: which of
                             the manuscript's numbers exist yet
    results_keys.txt         every macro name mag can define (NAMES ONLY)
    verify_offline.py        proves, before the run, that a complete run
                             fills every macro and writes every figure
    archive.tex              generated; overwritten by every run
    tables.tex               generated; the per-cell tabular bodies
    outstanding.tex          generated; the PDF's appendix A

## 2. Optional: prove the plumbing first (no GPU, ~1 minute)

    python test_experiment.py --with-archive

Expect: `186 passed, 0 failed`. This runs against a generated fixture and
touches neither your archive nor your GPU.

## 2c. Optional: prove the manuscript will fill (no GPU, ~3 minutes)

    python verify_offline.py --manuscript main.tex

This runs the real analysis pipeline --- mag's own `all`, the same
run_experiment stages, the same macro generators --- against a SYNTHETIC
cohort and a SYNTHETIC detection cache, then reports how many of the
manuscript's macros resolved and whether every figure was written.

Expect:

    macros the manuscript calls        156
    resolved by a complete run         156
    STILL RED after a complete run     0
    figures ... written  box_census / factorial / apportionment / calibration

It proves the plumbing and nothing else. Every number it produces is
invented, is labelled SYNTHETIC on every artefact, and none may be quoted.
What it buys is that a macro with no producer, or a figure written under a
different name, is found in three minutes rather than in a PDF after an hour
of GPU time.

## 2b. Confirm you are running the right copy

Every run now prints a banner. It must say:

    # malaria experiment suite v3.0

If it does not, you are running an older copy from a different folder. Note
the `# from  <path>` line in the banner -- that is the folder Python actually
loaded, and it is the one to replace. Extract the package there, or delete the
old folder entirely and extract fresh.

## 3. The run

ONE command produces every value, every figure and the compiled PDF:

    python run_all.py --root ..\malaria_france_data_set --manuscript main.tex --pdf

Run it from inside the suite folder, with the venv active. `--manuscript
main.tex` is what makes the manuscript layer part of the run; `--pdf` adds the
LaTeX build at the end and is skipped with a message if no TeX engine is on
PATH. If pdflatex is present but bibtex is not, the run says so, builds
anyway, and the citations render as `[?]`.

`--pdf` builds THREE documents, not one: `main.tex` (the paper, with a BibTeX
pass between the LaTeX passes for the numbered reference list), `slides.tex`
(the conference talk) and `poster.tex` (the A0 poster). The talk and the
poster read the same generated macro layer as the paper through
`resultmacros.tex`, so no number can differ between them, and the
outstanding-results register runs over all three.

DO NOT pass --yolov5-repo. The `yolov5` package (7.0.14) is installed in your
venv and mag uses it automatically. `models\models\yolov5\` holds the yolov5
CHECKPOINT, not a source checkout -- an earlier version of this runbook told
you to pass it as --yolov5-repo, which was wrong. Only pass the flag if the
yolov5 package is NOT installed, and then point it at a clone of
github.com/ultralytics/yolov5 (a folder with a models/ subdirectory). The env
phase now rejects a folder that has no models/ in it.

Optional additions:

    --manuscript main.tex        adds the audit's macro-coverage check, the
                                 v2->v3 wording guard, and the register of
                                 which of the manuscript's numbers exist yet
    --manuscript-dir .           where archive.tex, outstanding.tex and
                                 MISSING_RESULTS.md are written
    --pdf                        build main.tex at the end: pdflatex, bibtex,
                                 then pdflatex twice more
    --quick                      depth core instead of full: a fast first pass
                                 to prove everything connects, before the real
                                 one (you can re-run without --quick after;
                                 the detection cache is reused automatically)
    --batch 1                    only if you hit CUDA out-of-memory
    --show-warnings              put the yolov5 fork's torch.cuda.amp
                                 deprecation warnings back in the console
                                 (see section 4)

Everything printed is also written to `results\run_all.log`.

**Re-running is always safe.** Detection resumes: every smear already cached
is skipped and a checkpoint that finished is not re-inferred. Your
`data\test\` holds the shards from the run that stopped at assembly, and
this version repairs them from disk rather than re-inferring them --- expect
detect to reach the `--- recovery ---` block within a couple of minutes
rather than after another eighteen.

### What it does, in order

1. **env** -- checks python, mag, torch, CUDA, ultralytics BEFORE spending
   any time. A missing piece stops the run in seconds with the fix printed.
2. **eda** -- reads the archive: cohort, partitions and their disjointness,
   annotations, geometry, the transform. Writes `eda\eda.json`, `EDA.md`,
   figures. Minutes.
3. **verify** -- opens each checkpoint, checks the class map index for index,
   drives 20 real fields at mag's own operating point (`conf=0.001, iou=1.0,
   max_det=MAX_DET`), and if any field returns exactly the cap, re-runs just
   those fields with the cap lifted to measure what the truncation discards
   and in which classes. Writes `verify\verify.json`, `VERIFY.md`. A few
   minutes. **A failure here that is ONLY cap saturation does not stop the
   run** -- the experiment corrects it by raising the cap. Any other failure
   (a silent checkpoint, a permuted class map) stops everything, because an
   hour of GPU time against a broken checkpoint measures nothing.
4. **experiment** -- the pipeline itself:
   - *preflight*: prints the findings ledger from eda.json and verify.json;
     reports and CONTINUES. Nothing here needs a flag.
   - *ingest*: `mag ingest` (patients.csv, layout checks).
   - *detect*: `mag detect` over all 4,508 test fields, three checkpoints, at
     a cap resolved from the probe's own measurement (expected ~6,580; the
     console states the number and where it came from). This is the long
     stage: roughly 20-40 minutes on your card. It RESUMES -- if interrupted,
     re-running skips completed smears.
   - *analyse*: `mag all` at depth full -- the factorial, gates, confirmatory
     comparison, GEE attribution, calibration, stress, power, sweep -- plus
     the derived macros mag does not emit. First full-depth run on real data;
     expect several minutes to half an hour.
   - *sensitivity*: forks the cache, re-detects ONLY the saturating
     checkpoint at mag's original cap 3000 (the other checkpoints' caches are
     cap-independent and are reused verbatim), analyses the full factorial
     again, and diffs per-patient labels between the two caps. Writes
     `results\cap_sensitivity.json`.
   - *audit*: mag's build audit through its API, plus the wording guard if
     --manuscript was given. Writes `results\audit_report.json`. **A failed
     audit does not stop the run** -- the failure is recorded and the
     manuscript stage still runs.
   - *manuscript*: `findings.csv`, `manuscript_values.json`, `RESULTS.md`.
5. **manuscript** -- regenerates `archive.tex` from the eda and verify layers
   (this works even when the analysis has not run) and, with `--manuscript`,
   the outstanding-results register. Seconds.
6. **bundle** -- flushes the log, then zips eda\, verify\, results\,
   results_cap3000\, the cache's own small JSON (manifests, checkpoint
   metadata, the geometry-exclusion accounting --- never the tables or
   shards) and the log into `results\handback_<timestamp>.zip` (expected a
   few MB), and prints the PASTE-BACK SUMMARY.

### What lands where

Every path is relative to the folder you run the command from. Nothing is
written outside it.

| path | written by | what it is |
|---|---|---|
| `eda\eda.json` | eda | every archive measurement, as JSON |
| `eda\EDA.md` | eda | the same, readable |
| `eda\figures\*.png` | eda | 7 figures: cohort, class balance, fields per smear, boxes per field, box area, centre distance, crop-model fit |
| `verify\verify.json` | verify | checkpoint hashes, per-field behaviour, the cap probe |
| `verify\VERIFY.md` | verify | the same, readable |
| `data\patients.csv` | ingest | the patient table: one row per smear |
| `data\test\detections_<ck>.parquet` | detect | the detection cache, one table per checkpoint |
| `data\test\manifest_<ck>.json` | detect | its provenance: hash, floor, image size, seed |
| `data\test\_shards\<ck>\*.parquet` | detect | per-smear shards, which is what resume reads |
| `data\test\geometry_exclusions.json` | detect | boxes excluded for non-positive extent, by class |
| `results\results.json` | analyse | the built Study, as JSON |
| `results\results.tex` | analyse | **every result macro** the manuscript reads |
| `results\factorial.csv` | analyse | one row per rule x checkpoint cell |
| `results\predictions.csv` | analyse | per-patient calls, one column per arm |
| `results\errors.csv`, `box_census.csv`, `calibration.csv`, `selective.csv`, `claims.csv`, `rule_comparability.csv`, `rule_parameters.csv`, `predictive.csv` | analyse | the frames behind the tables |
| `results\figures\*.png` and `*.pdf` | analyse | 6 figures: factorial, apportionment, box census, calibration, errors, risk-coverage |
| `results\figures\PROVENANCE.txt` | analyse | the stamp the build checks figures against |
| `results\provenance.json` | analyse | checkpoint hashes, patient-table hash, seed, versions |
| `results_cap3000\` | sensitivity | the whole results layer again at the original cap |
| `results\cap_sensitivity.json` | sensitivity | which patient-level calls differ between the two caps |
| `results\audit_report.json` | audit | the build audit's verdict, check by check |
| `results\findings.csv` | manuscript | the findings ledger |
| `results\manuscript_values.json` | manuscript | every macro as a flat key/value map |
| `results\RESULTS.md` | manuscript | the readable summary |
| `archive.tex` | manuscript phase | the archive and checkpoint macros |
| `outstanding.tex` | manuscript phase | appendix A of the PDF |
| `MISSING_RESULTS.md` | manuscript phase | the register, for the repository |
| `main.pdf` | `--pdf` | the compiled manuscript |
| `results\run_all.log` | all | the complete console record |
| `results\handback_<stamp>.zip` | bundle | everything above except the caches and figures' PDFs |

The manuscript reads `archive.tex`, `results\results.tex`,
`results\figures\` and `eda\figures\` **by relative path**, so
`main.tex` must sit in the folder those are under. That is the default.

### Putting the outputs in a git repository instead

Point each layer at the repository and keep the caches out of it:

    python run_all.py --root ..\malaria_france_data_set ^
        --eda    ..\myrepo\eda ^
        --verify ..\myrepo\verify ^
        --results ..\myrepo\results ^
        --out    data ^
        --manuscript ..\myrepo\main.tex --manuscript-dir ..\myrepo --pdf

`--out` is the detection cache and is the one thing that should NOT go in a
repository: it is gigabytes and it is reproducible from the checkpoints.
Add `data/` to `.gitignore` and commit the rest.

### Expected total time

Roughly 1 to 2 hours, dominated by detect (~20-40 min), analyse at depth
full (unknown on real data -- this is its first real run; if it exceeds an
hour, interrupt and re-run with `--quick` to get a core-depth result while we
look at it), and the sensitivity arm (~10-15 min).

## 4. Things you will see that are NOT problems

- `[still working, N min elapsed, no output from this stage yet]`. mag prints
  nothing while it concatenates the shards, validates them, writes the table
  and hashes what it wrote. On the yolov5x cache that is around nine and a
  half million rows and takes minutes. It is not hung; the heartbeat exists
  so the difference is visible, because the run has twice been killed by hand
  during exactly this step.
- `BadInput: N detections have a non-positive width or height` immediately
  followed by `--- recovery ---`. The pass is already on disk; the shards are
  filtered, the table re-assembled from them, and mag called again. No field
  is inferred twice. Expect a long silence straight afterwards, for the
  reason above.
- `[suppressed N lines of torch.cuda.amp deprecation noise from the yolov5
  fork]` at the end of a phase. The fork calls `torch.cuda.amp.autocast`
  inside AutoShape once per BATCH and torch 2.7 deprecated it; over 4,508
  fields that is thousands of identical two-line warnings, and it is why the
  first complete run had to be interrupted. v2.4 filters exactly that
  deprecation, in two places (`PYTHONWARNINGS` in the child, and the line
  stream this script relays), counts what it drops, and prints the count.
  **Every other warning still prints.** `--show-warnings` puts them back.
- `reconstruction gate: the comparator scores N/170, below the published 135`
  -- Protocol v3 SS0 predicts this. The run forces past it BY DESIGN and
  records the refusal; it is a finding about the released artefacts.
- `audit: declarations still hold a placeholder` -- correct until you fill
  `declarations.json`. The ten fields (ethics, funding, CRediT, AI use,
  repository, OSF, Zenodo, data availability) are yours to write; no pipeline
  may invent them. The run completes regardless.
- The findings ledger firing 12 or more findings -- those are the measured
  properties of the archive (identifiability, enrichment, the re-annotated
  class, ...), reported so the manuscript states them.

## 5. If something fails

| Symptom | Meaning | Do |
|---|---|---|
| env phase lists problems | a dependency is missing | follow the printed pip lines; nothing was spent |
| `no image_list found` | discovery missed the xlsx | add `--image-list ..\malaria_france_data_set\Plasmodium-v1.zip\Plasmodium-v1\image_list.xlsx` |
| CUDA out of memory in detect | 6 GB card at an unlucky batch | mag halves the batch automatically; if it still dies, add `--batch 1` |
| verify fails, NOT about max_det | a checkpoint cannot detect | stop; send `verify\VERIFY.md` and `results\run_all.log` back before anything else |
| detect interrupted | -- | re-run the same command; completed smears are skipped |
| `ProvenanceError ... no non-empty inference shards` | a cache was assembled rather than inferred | delete the named cache folder and re-run; this guard is working, not broken |
| `format guard: ... declares 'ultralytics' but its pickled model class is 'yolov5-repo'` | the format was corrected before loading | nothing -- this is the fix working |
| `purged the suspect cache for <ck>` | a previous pass cached garbage; its shards would otherwise be resumed | nothing -- that checkpoint re-infers from scratch |
| `N boxes have w <= 0 or h <= 0` after detect | degenerate geometry, excluded with accounting | nothing -- the count and the rows are kept in `geometry_exclusions.json` and `excluded_geometry_<ck>.parquet` |
| `BadInput: N detections have a non-positive width or height` followed by `--- recovery ---` | mag refused the cache AT ASSEMBLY; the inference is already on disk | nothing -- the shards are filtered, the pass is re-assembled without re-running the GPU, and mag is called again |
| `results\stage_failures.json` exists | a stage refused at some point | read it: it holds the exception, its remedy and the traceback, whatever happened to the log |
| `declared format 'ultralytics', but the pickled model class is models.yolo` | the format was reconciled against the model class | nothing -- this is the fix working; the correct loader is used |
| yolov5x shows `14.0/field` or `scores max 638` | the wrong loader was used | stop; this must not happen any more, send the log |
| analyse takes over an hour | first full-depth run on real data | interrupt, re-run with `--quick`, send that bundle; we diagnose from the log |

## 6. After filling declarations.json (optional, any time later)

    python run_experiment.py --stages audit manuscript --results results --eda eda --verify verify --declarations declarations.json --max-det 6580
    python run_all.py --root ..\malaria_france_data_set --phases bundle

(Use the max_det value your run printed, if different.)

## 6b. The manuscript

`main.tex` compiles on its own, before any analysis:

    python make_archive_macros.py --eda eda --verify verify --out archive.tex
    python missing_results.py --manuscript main.tex
    pdflatex main && bibtex main && pdflatex main && pdflatex main

The talk and the poster need no BibTeX --- they cite nothing --- so two
passes each:

    pdflatex slides && pdflatex slides
    pdflatex poster && pdflatex poster

`resultmacros.tex` must sit beside all three; the manuscript phase copies it
there for you. The poster is A0 portrait; change `size=a0` to `size=a1` in its
`beamerposter` options for A1, or to
`size=custom,width=91.4,height=121.9` for a 36x48 inch US board. Nothing else
changes --- beamerposter rescales the type.

`refs.bib` ships with the suite and must sit beside `main.tex`; the manuscript
phase copies it there for you. Its 32 entries were each checked against
PubMed, a publisher page, arXiv/CVF, Zenodo or the WHO catalogue, and every
one carries a DOI or a URL.

Every number in it is a macro. The ones your eda and verify layers already
produced resolve; the ones the analysis has not produced render in red as
`[macroName]`, and Appendix A lists them with the stage that produces each.
Figures the analysis has not written render as a framed statement of which
stage writes them, so a missing figure never stops the build.

`run_all.py --manuscript main.tex` refreshes both generated files as part of
the run, so the register shrinks by itself as stages complete.

## 7. What to send back

    results\handback_<timestamp>.zip

and/or paste the PASTE-BACK SUMMARY block from the end of the console.
If the zip is too large to upload, send at minimum:
`results\run_all.log`, `results\RESULTS.md`, `results\manuscript_values.json`,
`verify\verify.json`, `eda\eda.json`.

With that in hand, the next steps on my side are interpretation of every
number against the protocol, and the section-by-section revision of the
manuscript -- including SS9.4's required wording, the estimand statements the
identifiability finding forces, and the detection-stage paragraph the cap
findings require.

## 8. Design decisions carried over from the session

Each of these exists because something specific went wrong earlier:

- **Every JSON key is defined once, in exp_schema.py.** Twice, a consumer
  read a key a producer never wrote (predictions.csv read as long when mag
  writes it wide; centre_distance read as nested when it is flat). Key drift
  is now a failing test, not a silent UNKNOWN.
- **verify drives mag's exact operating point.** An earlier version ran at
  ultralytics' defaults (iou=0.7, max_det=300) and measured a different
  detector from the one the pass caches -- yolov8x looked like 132 boxes per
  field when it is ~1,140.
- **`fmt` is always passed to the model loader.** Without it, the yolov5
  repository pickle loads through the wrong class WITHOUT ERROR and returns
  garbage that an earlier version certified as "safe to run".
- **The cap is measured, then cleared.** Nothing in mag guards the ceiling;
  a saturated cache passes every gate while being censored at the top. The
  probe measures the loss; the pass runs at a cap the probe says no field
  reaches; the sensitivity arm quantifies what the old cap would have changed.
- **Preflight reports and continues.** The earlier --acknowledge gate made
  results unreachable until a flag was typed; a check that halts gets
  disabled, a check that reports gets read.
- **The audit cannot kill the run.** It used to exit before the manuscript
  stage could record its verdict. Now the verdict is a file in the bundle.
- **Windows first.** No .sh (PowerShell hands it to an editor and nothing
  runs); ASCII-only console output; UTF-8 file IO; verified to parse under
  Python 3.9.
- **One log, one bundle.** Pasted consoles truncate; the zip carries the
  complete record.
- **Known noise is counted, not hidden.** A per-batch deprecation warning
  made the detect stage's progress unreadable and cost an interrupted run.
  It is filtered by message --- not by category, not by silencing warnings
  wholesale --- and the number dropped is printed, so a suppression can never
  be mistaken for silence.
- **A refusal from mag arrives as an exception.** `mag.cli.main` catches its
  own errors, prints them to stderr and returns 3. That is right for a
  terminal and wrong for a caller that has to ACT on the refusal: the first
  real run's detect came back 3, the recovery never saw an exception to
  catch, and the reason existed only as a console line. The CLI parser is
  now used and its `fn` called directly, as `mag all` already was.
- **The GPU is spent once.** `cache.save` validates as it writes, so a pass
  holding a box of zero extent refuses AFTER inference, with every shard on
  disk. The shards are filtered, the table re-assembled from them, and mag
  called again --- no field is inferred twice.
- **A failure is a file.** `results\stage_failures.json` records the
  exception, its remedy and the traceback the moment a stage refuses. The
  first handback said only that detect had returned 3 after eighteen
  minutes.
- **The log is flushed before it is zipped.** That first handback's log was
  cut off one buffer short, mid-sentence, and the complete file sat on the
  machine unread.
