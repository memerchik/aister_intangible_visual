# Phase 5 v2 findings — pairwise ceramic correction

> This report preserves the v2 result. V3 has since completed and was also
> rejected; see [`PHASE_5_EVOLUTION.md`](PHASE_5_EVOLUTION.md) for the complete
> record and [`PHASE_5_V3_FINDINGS.md`](PHASE_5_V3_FINDINGS.md) for the latest
> decision.

## Decision

Phase 5 v2 is complete and **rejected for promotion**. The frozen pairwise
Opishnyan/Bubnivka correction did not improve the selection-aware nested result
over Phase 5 v1 and was not stable across source-atomic outer searches. It
failed 6 of 18 required gates, so it must not advance to calibration,
packaging, or sealed evaluation.

Phase 5 v1 remains the stronger development result. Neither v1 nor v2 is an
accepted deployable candidate.

The 330-image test set remains sealed and unevaluated. V2 loaded only the 1,693
development embeddings named in its allowlist; it produced no test embedding,
prediction, metric, threshold, or error analysis.

## Frozen experiment

The v2 experiment was declared before its first model fit in
[`phases/phase_05_source_robustness/experiment_contract_v2.json`](phases/phase_05_source_robustness/experiment_contract_v2.json).
Its SHA-256 is
`481fcab8c19b412a59354f53735a3c6299da5c16ec06ed1c6e4ffbe49c1a5a68`.

The experiment replayed the v1 base procedure with only its three frozen
finalists. It selected a base first, then compared 65 correction choices:

- one uncorrected reference;
- four fixed pairwise feature families;
- logistic-probe `C` in `{10, 100}`;
- source-group exponent in `{0.0, 0.5}`;
- correction blend in `{0.25, 0.5, 0.75, 1.0}`.

Each binary head trained only on Opishnyan and Bubnivka rows in the current
training folds. At application time it preserved the base probability mass of
the pair, changed only their conditional log-odds, and left Ornek,
Petrykivka, and Kosiv probability columns unchanged. Corrections also had to
remain within frozen accuracy, macro-F1, source-robustness, Opishnyan
source-group-recall, and other-class-recall limits.

The nested lifecycle held out each canonical source-atomic fold, selected the
base on inner OOF predictions, selected the correction on a second inner OOF
comparison, and predicted the untouched outer fold once. The uncorrected v2
path reproduced the pinned v1 per-image decision fingerprint before v2 was
accepted as valid experimental evidence.

## Results

| Evaluation | Accuracy | Macro F1 | Balanced accuracy | Boundary score | Robustness score |
|---|---:|---:|---:|---:|---:|
| Phase 5 v1 adaptive nested OOF | **93.9752%** | **93.6135%** | **94.3388%** | 86.7567% | **71.0883%** |
| Phase 5 v2 adaptive nested OOF | 93.7980% | 93.4211% | 94.0571% | 86.3769% | 70.7670% |
| V2 full-development selected recipe OOF | 93.8571% | 93.5464% | 94.2274% | **86.7883%** | 70.9745% |

The v2 adaptive procedure made 1,588 correct and 105 incorrect predictions,
three fewer correct than v1. It changed 18 v1 decisions: 7 former errors were
fixed, 10 correct predictions regressed, and one error changed to another
wrong class.

| Ceramic diagnostic | V1 nested | V2 nested | Change |
|---|---:|---:|---:|
| Opishnyan recall | 87.4251% | 87.4251% | 0.0000 pp |
| Bubnivka recall | 90.1408% | 88.7324% | -1.4085 pp |
| Bubnivka precision | 79.6680% | 79.7468% | +0.0788 pp |
| Ceramic macro F1 | 90.7297% | 90.4092% | -0.3206 pp |
| Opishnyan -> Bubnivka errors | 45 | 45 | 0 |
| Bubnivka -> Opishnyan errors | 17 | 20 | +3 |

The correction moved three Opishnyan errors back from Bubnivka but also moved
three correct Opishnyan predictions toward Bubnivka, leaving the leading error
count unchanged. For Bubnivka it restored four prior errors but changed seven
correct predictions to Opishnyan. This is the main reason aggregate accuracy,
Bubnivka recall, and ceramic macro F1 fell.

## Selection instability

V2 found no repeatable correction recipe. The five outer searches selected:

| Outer fold | Selected base | Selected correction |
|---:|---|---|
| 0 | three-view CLS ensemble | letterbox CLS+patch, `C=10`, source `0.5`, blend `1.0` |
| 1 | global CLS | none |
| 2 | three-view CLS ensemble | global CLS, `C=10`, source `0.5`, blend `0.25` |
| 3 | three-view CLS ensemble | global CLS, `C=100`, source `0.0`, blend `0.75` |
| 4 | global CLS | global CLS, `C=10`, source `0.5`, blend `0.5` |

The full-development search chose global CLS as the base and letterbox
CLS+patch with `C=100`, source exponent `0.5`, and blend `0.25` as the
correction. That recipe was ineligible in four of the five outer inner
searches and was never within the frozen boundary-score tie window of the best
eligible correction. Correction stability therefore scored 0 of the required
4 outer searches. The base recipe remained stable at exactly 4 of 5.

The full-development correction itself looked promising: compared with the
uncorrected full-development base, it changed 11 decisions, fixed 7, regressed
2, and raised accuracy from 93.5617% to 93.8571%. The nested result shows why
that gain cannot be trusted as a promotion result: selecting the correction on
all development folds favors a recipe that does not transfer consistently
between the large, heterogeneous source cohorts.

## Promotion gates

Twelve of 18 frozen gates passed. Six failed:

| Failed gate | Required | Observed | Margin |
|---|---:|---:|---:|
| Opishnyan recall | >= 87.5000% | 87.4251% | -0.0749 pp |
| Bubnivka precision | >= 82.0000% | 79.7468% | -2.2532 pp |
| Ceramic macro F1 | >= 91.0000% | 90.4092% | -0.5908 pp |
| Opishnyan -> Bubnivka errors | <= 40 | 45 | +5 errors |
| Bubnivka recall | >= 90.9600% | 88.7324% | -2.2276 pp |
| Correction stability | 4 of 5 outer searches | 0 of 5 | -4 searches |

The aggregate accuracy, macro-F1, difficult Opishnyan slices, large named
sources, worst fold, fold variation, three non-target recall floors, and base
stability gates passed. Frozen thresholds were not changed after observing the
result.

## Reproducibility and integrity

The v2 runner is
[`scripts/run_phase5_v2_pairwise.py`](scripts/run_phase5_v2_pairwise.py), and
its SHA-256 is
`5e5afb6a952b9b8fc00cb7e7fa259e288661124394ba608e8415a44ffb98b479`.
The 13 canonical artifacts are under
[`outputs/phase_5_source_robustness_v2/`](outputs/phase_5_source_robustness_v2/),
and `metrics.json` has SHA-256
`6b01234a4b04985d26185bdded6fb5382f5186e04895d1571687ad1011b174d3`.

A separate full run reproduced all 13 artifacts byte for byte, including the
confusion-matrix PNG. Independent output tests reconstruct development
coverage, probability semantics, ordinary and source-group metrics, pairwise
boundary metrics, confusion outputs, correction eligibility and ranking,
both stability rules, all 18 promotion gates, input/code/cache provenance, and
the comparison with v1.

The recorded environment is CPython 3.9.6, NumPy 2.0.2, scikit-learn 1.6.1,
Matplotlib 3.9.4, and threadpoolctl 3.6.0. Fits used one BLAS thread and treated
convergence warnings as errors.

## What this teaches us

The remaining problem is not solved by a generic binary probe over the current
global, letterbox, center, CLS, and patch-summary embeddings. A single
full-development score can make such a correction look useful, but the effect
changes direction across held-out source cohorts. The v1 source-balanced base
is more dependable than the v2 adaptive correction and remains the best
development result, although it still failed its own promotion gates.

Do not move to Phase 6 or the sealed test. Any further modelling must be a new,
separately frozen development experiment. It should address the source-dependent
ceramic boundary rather than expanding this same pairwise grid, retain v1 as
the reference, and preserve every existing gate and source-atomic isolation
rule.
