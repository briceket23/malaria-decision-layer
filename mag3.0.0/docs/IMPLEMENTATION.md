# Implementation status — protocol v3 against the code

Every section of `PROTOCOL.md` mapped to the module that implements it and the
tests that hold it in place. This is the document to read when checking whether
a protocol commitment is real or aspirational.

**Status at this build: every section implemented, 292 tests passing on
Python 3.9, 3.11 and 3.12, Linux and Windows.**

```
28 modules   11,056 lines of package code
11 test files  292 tests
```

---

## Traceability

| § | requirement | module | tests |
|---|---|---|---|
| 3 | three quantities separated by name | `report`, `study`, `gates` | `test_audit`, `test_protocol_v3` |
| 4 | estimands attached to every endpoint | `estimand` | `test_protocol_v3` (5) |
| 5 | rule registry as data; fixed and fitted disjoint | `rules` | `test_analyses`, `test_invariants` |
| 5.1 | six rules in the main paper, rest to supplement | `rules.MAIN_RULES` | `test_wiring` |
| 5.2 | rule comparability table | `rules.comparability_table` | `test_wiring` |
| 5.3 | parameter rationale table | `rules.PARAMETER_RATIONALE` | `test_analyses` |
| 6 | three post-processing conditions | `evidence.apply_condition` | `test_invariants`, `test_audit` |
| 7 | propositions P1–P5 | `gates`, `rules` | `test_invariants` |
| 8 | seven gates in protocol order | `gates`, `study` | `test_experiment` |
| 8.3 | checkpoint pre-declared; headroom descriptive | `gates.select_checkpoint` | `test_analyses`, `test_audit` |
| 8.5 | design gate on both arms | `gates.design_gate` | `test_audit` |
| 9 | patient-clustered factorial model with interaction | `attribute` | `test_protocol_v3` (10) |
| 9.1 | the interaction IS estimable | `attribute.fit` | `test_protocol_v3` |
| 9.3 | ranges demoted to descriptive | `study._apportion` | `test_wiring` |
| 10 | intersection–union co-primary | `stats`, `study._confirm` | `test_analyses`, `test_wiring` |
| 10.2 | exact McNemar, one-sided, full 2×2 | `stats.mcnemar_exact`, `stats.paired_table` | `test_analyses` |
| 10.3 | Tango score interval for non-inferiority | `stats.noninferiority` | `test_analyses`, `test_audit` |
| 11 | calibration split: infection vs speciation | `calibrate` | `test_calibration` (26) |
| 11.C | four decimals, bin counts | `report`, `audit.PRECISION_FLOOR` | `test_protocol_v3` |
| 12 | abstention as workload, not AURC alone | `selective` | `test_analyses` |
| 13 | field-sampling stress test | `stress` | `test_protocol_v3` (7) |
| 14 | prevalence-standardised PPV/NPV | `predictive` | `test_protocol_v3` (5) |
| 15 | leave-one-centre-out, speciation only | `diagnostics.leave_one_centre_out` | `test_analyses` |
| 16 | error taxonomy with its caveat | `diagnostics.error_summary` | `test_analyses` |
| 17 | forbidden and required language | `estimand`, `audit.check_no_typed_verdict` | `test_protocol_v3` |
| 18 | claim–evidence–scope table as a build artefact | `study.claim_table` | `test_wiring` |
| 19 | build audit gating the PDF | `audit`, `cli.cmd_audit` | `test_protocol_v3` (16) |
| 20 | manuscript architecture | `report` | `test_wiring` |
| 22.1 | cohort from the manifest, with refusals | `ingest` | `test_acquire` (12) |
| 22.2 | transformation measured, not inferred | `ingest.transform_retention`, `study.load_transform`, `report._transform` | `test_acquire` (3) |
| 22.3 | pre-suppression cache at a declared floor | `detect.run`, `cache.save` | `test_acquire` |
| 22.4 | class map from the checkpoint, or declared | `detect.class_map_from` | `test_acquire` (4) |
| 22.5 | a silent detector is a bug, not a result | `detect.implausible_yield` | `test_acquire` (4) |
| 22.6 | resumable, fingerprinted, provenanced | `detect`, `provenance` | `test_acquire` (6), `test_no_fabrication` |
| — | schema validation at every boundary | `schema` | `test_audit` |
| — | content hash, cross-platform | `schema.content_hash` | `test_compatibility` |
| — | no fabrication, structurally | five arrangements | `test_no_fabrication` (11) |

---

## The analyses protocol v3 added

### `estimand.py` — §4

Four estimands, each carrying its target, its scope, the wording it forbids and
the wording it requires. `SPECIFICITY.identifiable` is `False` with a stated
reason: every uninfected patient comes from one centre, so specificity is
collinear with centre and with internal/external domain. That is rank
deficiency, not low power, and the distinction is the point — a reader told the
estimate is imprecise will ask for a larger sample; a reader told it is
collinear knows no sample fixes it.

`check_cohort` re-derives the scope from the patient table in hand, so a future
cohort that no longer matches the written scope is detected rather than
described by stale prose.

### `attribute.py` — §9

This replaces v2's weakest analysis and retracts one of its claims.

v2 compared a range across fourteen rules with a range across two checkpoints
and called the difference an apportionment. A range grows with the number of
levels; the two factors have wildly different level counts; the comparison is
outlier-sensitive. It then asserted the interaction "is arithmetically the
residual and is not separately estimable."

**That assertion is false.** Every patient contributes an outcome under every
rule × checkpoint cell — 170 repeated observations per cell, not one. v3 fits

```
logit P(Y_irc = 1) = mu + alpha_r + beta_c + (alpha beta)_rc + u_i
```

as a GEE with an exchangeable working correlation and robust standard errors
clustered on patient, and tests the interaction.

Two implementation decisions matter:

**statsmodels, deliberately.** The first dependency outside numpy and pandas.
A reviewer challenging the paper's second contribution should be able to check
it against an implementation they recognise; "we wrote our own GEE" invites a
question the paper does not need. `sandwich_check` fits the same marginal model
by Fisher scoring with a cluster-robust covariance written out in forty lines,
and the run reports the agreement between the two. If they disagree, the
analysis says so and neither is reported.

**The probability scale is primary.** Complete separation is the expected state
on this data — R0's specificity is 0.000, so its level's outcome never varies
and its log-odds effect is unbounded. An earlier version fitted anyway and the
two implementations diverged by 4×10⁸ on a fixture: a number that would have
been printed. `detect_separation` finds separated levels, the log-odds contrast
is withheld with a stated reason where any exists, and the probability-scale
contrast — which is not fitted and remains defined — governs.

### `stress.py` — §13

Three independent reviews asked for one additional experiment between them, and
this is it.

Every field from an infected patient in this archive was selected to contain a
parasite. The rules respond to that differently by construction: an accumulating
rule sums evidence without bound over fields that all carry some, a counting
rule needs a threshold number of confident detections, a field-normalised rule
divides by the fields examined. The concern is not that the figures are
inflated — it is that the **ordering** of the rules may be an artefact of the
enrichment, and the ordering is what the second contribution rests on.

The design holds the detector cache fixed. Nothing is re-inferred and no
detection is invented: each infected patient's smear is rebuilt at a declared
positive-field rate by keeping a share of that patient's own fields and
replacing the rest with parasite-free fields from uninfected patients. The field
count is held constant so a field-normalised rule stays comparable across rates.
Uninfected patients are untouched.

Reported: each rule's sensitivity and specificity against the rate, and
Kendall's τ-b of the ordering against the undiluted one.

One consequence of this archive's structure is stated in every output: every
uninfected patient comes from one centre, so for an infected patient from
anywhere else the donor fields carry a domain shift. `donor_mismatch` counts
exactly how many patients that affects.

### `predictive.py` — §14

This cohort is 68% infected because a referral centre assembled it. A PPV
computed at that prevalence and quoted without it reads as a screening PPV and
is roughly an order of magnitude too high for one.

The module refuses to report a predictive value without the prevalence it
assumed, and reports several — 1%, 5% and 20%, declared in the protocol rather
than fitted — with patient-level bootstrap intervals. The cohort's own
prevalence is reported too, and labelled as a referral-centre quantity, because
a reader will look for it and its absence invites exactly the reconstruction the
module exists to prevent.

Intervals are bootstrapped rather than delta-method: at 54 uninfected patients
the specificity interval is wide and asymmetric, and a symmetric approximation
puts the PPV bound in the wrong place.

### `audit.py` — §19

The check that did not exist, which is why the previous build's defects reached
three reviewers. It runs over a finished results directory rather than a study
object, so it can be run on results produced weeks earlier, or on a colleague's.

It catches: a macro the manuscript calls that the analysis never defined (these
rendered as `[name]` and were read as data); a macro defined but empty; a
directional verdict typed where a generated sentence exists (the previous
build's abstract said the co-primary conjunction was met while its conclusion
said the opposite — both typed); retired vocabulary; a rate that does not follow
from its own confusion counts; a metric printed to too few decimals; a number
that disagrees between `results.tex` and `results.json`; figures from a
different run or seed; and a declaration still holding a placeholder.

Exit code 5 if anything fails. There is no bypass flag.

One check deserves its own note. `check_cross_artefact` compares named macros
against paths into `results.json`, and its first version still read
`("reproduction", "n_correct")` after the key had been renamed to
`reconstruction`. Because an unresolvable path was skipped in silence, the check
reported success while comparing nothing — the same class of defect the audit
exists to catch. A pair that does not resolve on both sides is now a failure in
its own right, and `test_a_pair_that_cannot_be_compared_is_a_failure_not_a_silent_pass`
holds it there.

---

## The acquisition layer — §22

`ingest` and `detect` were a separate package in the previous arrangement. They
are here now, because a repository that cannot get from the archive to the
results is not a reproducible pipeline.

`ingest` needs no GPU and takes seconds. `detect` needs one, takes hours, and is
resumable at smear granularity with a fingerprint covering the image list, the
confidence floor, the image size, the class map and the checkpoint's SHA-256.

The seam between the two halves is tested directly:
`test_a_cache_this_package_wrote_is_a_cache_it_can_analyse` drives `detect.run`
with a structured stub, then runs `mag all` over the cache it produced. Every
other test on either side uses a cache built by its own fixtures — if `detect`
wrote a column, a dtype or a coordinate convention that `study` did not expect,
all of them would still pass and only that one would fail.

---

## Defects this rebuild found in itself

Recorded because a defect found once and not written down is a defect that comes
back. Each has a test named after it.

**`_kendall` was not tie-corrected.** The denominator added the tied pairs where
τ-b subtracts them. On two identical rankings containing one tie it returned
0.67 instead of 1.0 — and since ordering stability is decided at τ ≥ 0.8, that
would have reported a stable ordering as unstable. Ties are the normal case here
(several rules reach identical sensitivity), so the error was not rare. Now
verified against scipy across 200 random tie-heavy cases, agreeing to 2×10⁻¹⁶.

**The cross-artefact check was silently disconnected.** See §19 above.

**`cache.save` hashed memory, not disk.** A round trip through parquet, and
especially through the CSV fallback, can move a float's last place. The manifest
then disagrees with the file beside it, the next load refuses the cache as
tampered with, and the operator is told to re-run detection — an hour of GPU
time spent reproducing the same mismatch. It now hashes what it just wrote.

**`__version__` had drifted to 1.0.0** while `pyproject.toml` said 3.0.0. The
version is stamped into every provenance record, so a drifted version silently
mislabels which code produced a result. A test now compares the two.

**Complete separation broke the attribution fit.** Both implementations diverged
by 4×10⁸ on the fixture. Restructured so the probability contrast is primary and
always defined.

**The GEE's link overflow was promoted to an error.** Separation drives the
linear predictor to infinity and numpy warns; this package's own filter turns
warnings from `mag.*` into errors. Silenced at the two calls that can raise it —
and *recorded*, because an overflow with no separated level would mean something
else entirely.

---

## What still requires a human

Nothing in this repository can supply these, and nothing in it will guess:

- the author list, affiliations and CRediT statement;
- the ethics approval for the present study, and the source study's;
- consent for publication;
- funding;
- the AI-use declaration;
- the repository URL, the registration identifier and the code archive DOI;
- the data availability statement.

`mag audit` checks that each is present and non-empty in `declarations.json` and
refuses the build otherwise. `CITATION.cff` carries the same gap, marked.

---

## On revisiting the manuscript

The Results, the Abstract's results clause, the Discussion's first paragraph and
the Conclusion wait for a run against the real archive. Everything else can be
written now, because none of it depends on a value: the section architecture
(§20), the contribution hierarchy (§2), the estimands and their scopes (§4) —
which are properties of the *dataset*, not of any result — the forbidden and
required language (§17), the claim–evidence–scope skeleton (§18) with verdict
cells empty, and the Methods in full.

That is a substantial share of the paper, and it is the share that is safe to
write. Three reviewers read a build whose numbers were placeholders and produced
pages of analysis about results that do not exist — including a recommendation
built on the reconstruction discrepancy running in the *opposite* direction to
the real one. Writing a Results section now, against the same absent numbers,
would reproduce that situation with better wording.
