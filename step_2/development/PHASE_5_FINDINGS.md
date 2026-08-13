# Phase 5 findings — source robustness

> This report preserves the v1 result. V2 and v3 have since completed and were
> also rejected; see [`PHASE_5_EVOLUTION.md`](PHASE_5_EVOLUTION.md) for the
> complete record and [`PHASE_5_V3_FINDINGS.md`](PHASE_5_V3_FINDINGS.md) for
> the latest decision.

## Decision

Phase 5 is complete, but the candidate is **rejected for promotion** under the
frozen experiment contract. The adaptive nested procedure reached 93.9752%
development accuracy and materially improved the difficult Opishnyan source
slices, but it failed 5 of 17 predeclared promotion gates. Phase 6 calibration
and packaging must not begin from this result.

The 330-image test set remains sealed and unevaluated. No test image was
embedded, predicted, calibrated, or inspected during this phase.

## Frozen evaluation

The experiment was declared before fitting in
[`phases/phase_05_source_robustness/experiment_contract.json`](phases/phase_05_source_robustness/experiment_contract.json),
whose SHA-256 is
`55d0dd0d3665b4b84b33a9c9461763c6858452bd0b10b679fd9b77fbe630a87f`.

It used all 1,693 development images and the five canonical source-atomic
folds. For each outer fold, the other four folds formed a complete inner OOF
search over 12 base configurations:

- three frozen DINOv3 feature families;
- logistic-probe `C` in `{10, 100}`;
- source-group exponent in `{0.0, 0.5}`.

The base recipe was selected before comparing no specialist with one fixed
three-way ceramic conditional specialist. Every source-group size and class
normalizer was recomputed from the current training subset. Each fit actively
verified that its training and validation source groups were disjoint.

The candidate family remains development-informed because Phase 4 errors and
a bounded view pre-screen shaped the search. Nested OOF estimates the adaptive
Phase 5 procedure more honestly than reusing its selected OOF score, but it is
not an unbiased production estimate. Only the later one-time sealed evaluation
can provide that estimate.

## Results

| Evaluation | Accuracy | Macro F1 | Balanced accuracy | Robustness score |
|---|---:|---:|---:|---:|
| Phase 4 fixed reference | 93.0892% | 92.8447% | 94.0378% | 64.4664% |
| Phase 5 adaptive nested OOF | **93.9752%** | **93.6135%** | **94.3388%** | **71.0883%** |
| Phase 5 full-development selected recipe OOF | 93.5617% | 93.2398% | 94.0818% | 70.9101% |

The adaptive outer procedure made 1,591 correct predictions, 15 more than the
Phase 4 reference. Thirty-eight decisions changed: 26 Phase 4 errors were
fixed and 11 formerly correct predictions regressed.

The full-development recipe selected for possible later work is:

```text
feature family:         global_cls
embedding:              DINOv3 global-fivecrop CLS
C:                      10
source-group exponent:  0.5
ceramic specialist:     none
```

It was within the robustness tie window in 4 of 5 outer searches, satisfying
the stability rule. The three-view ensemble was selected in outer folds 0, 2,
and 3, while the global recipe was selected in folds 1 and 4. Every outer
search rejected the specialist. In the full-development comparison the
specialist slightly raised ordinary accuracy, but reduced the robustness score
from 70.9101% to 68.3707%, so it was not selected.

## What improved

| Diagnostic | Phase 4 | Phase 5 nested | Change |
|---|---:|---:|---:|
| Opishnyan recall | 83.4331% | **87.4251%** | +3.9920 pp |
| Opishnyan source-group recall | 51.0767% | **59.0742%** | +7.9976 pp |
| Opishnyan → Bubnivka errors | 58 | **45** | -13 |
| Hard-three Opishnyan pooled accuracy | 64.6739% | **73.9130%** | +9.2391 pp |
| Challenging Opishnyan accuracy | 62.6794% | **71.7703%** | +9.0909 pp |
| Hard-three cohort-macro accuracy | 61.9939% | **71.5619%** | +9.5680 pp |

All five adaptive outer recipes used source exponent `0.5`. This is strong
evidence that training-time source balancing is useful for this dataset. The
largest gains occurred in the two source-heavy folds that had limited Phase 4:
fold 3 rose from 87.4269% to 88.8889%, and fold 4 from 85.7567% to 88.4273%.

## Why promotion failed

The frozen contract required all 17 gates to pass. Twelve passed, including
aggregate accuracy and macro F1, all three hard-source slice gates, the large
named-source gate, worst-fold accuracy, fold variation, three of four
non-target recall floors, and recipe stability.

| Failed gate | Required | Observed | Margin |
|---|---:|---:|---:|
| Opishnyan recall | >= 87.5000% | 87.4251% | -0.0749 pp |
| Bubnivka precision | >= 82.0000% | 79.6680% | -2.3320 pp |
| Ceramic macro F1 | >= 91.0000% | 90.7297% | -0.2703 pp |
| Opishnyan → Bubnivka errors | <= 40 | 45 | +5 errors |
| Bubnivka recall | >= 90.9600% | 90.1408% | -0.8192 pp |

Opishnyan gained 20 correct classifications relative to Phase 4, while
Bubnivka lost 6. The leading remaining error is still Opishnyan → Bubnivka
(45 cases), followed by Bubnivka → Opishnyan (17 cases). The result therefore
improves the intended weak class and source slices, but does not yet provide a
clean enough ceramic boundary to freeze for deployment.

The Opishnyan recall gate is only one additional correct image away, and the
Bubnivka recall floor is two correct images away. Nevertheless, changing the
frozen thresholds after observing the results would invalidate the experiment.
The rejection is retained.

## Reproducibility and integrity

The canonical runner is
[`scripts/run_phase5_source_robustness.py`](scripts/run_phase5_source_robustness.py),
and the 12 required artifacts are under
[`outputs/phase_5_source_robustness/`](outputs/phase_5_source_robustness/).

An independent run with CPython 3.9.6, NumPy 2.0.2, scikit-learn 1.6.1,
Matplotlib 3.9.4, and one BLAS thread reproduced all 12 artifacts byte for byte,
including the PNG. The Phase 5 exact-reference path also reproduced all 1,693
Phase 4 argmax decisions; its stable prediction fingerprint is
`d6f5b90c2c6b8af39fb8e6bffbe3e44ba3e6d99b3f8f069b8f1febfe95bab307`.

The output tests independently recompute the metrics, confusion matrix,
source-group score, diagnostic slices, promotion gates, selection semantics,
and provenance from the prediction files and frozen inputs.

## Recommended next work

Do not move to calibration or the sealed test. If work continues, it should be
a separately frozen Phase 5 v2 development experiment focused narrowly on the
Opishnyan/Bubnivka decision boundary while preserving the source gains. A
bounded candidate set should consider development-only per-crop or patch-level
DINOv3 evidence, a predeclared hierarchical ceramic objective, and a small
source-weighting grid. It must use the same nested source-atomic protocol and
must not target individual error IDs or relax the v1 gates after the fact.
