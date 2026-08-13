# Phase 5 v5 findings — final fixed motif consensus

## Decision

Phase 5 v5 is complete and **rejected for promotion**. It passed 18 of 21
required gates, the most of any iteration, but the contract requires all 21.
Phase 6 remains blocked, and the 330-image test set remains sealed and
unevaluated.

V5 reached 94.5068% nested accuracy, 94.1957% macro F1, 94.9226% balanced
accuracy, and a project-best 76.8884% source-robustness score over all 1,693
development images. It made 1,600 correct and 93 incorrect predictions.

The three failed gates were:

| Gate | Required | V5 | Shortfall |
|---|---:|---:|---:|
| Bubnivka precision | >= 82.0000% | 79.8354% | 2.1646 pp |
| Opishnyan -> Bubnivka errors | <= 40 | 46 | 6 errors |
| Worst outer-fold accuracy | >= 87.5000% | 86.8421% | 0.6579 pp / 3 images |

## What v5 tested

V5 was declared as the final Phase 5 attempt and frozen before its first fit at
[`phases/phase_05_source_robustness/experiment_contract_v5.json`](phases/phase_05_source_robustness/experiment_contract_v5.json).
Its contract SHA-256 is
`c9473a7e20c57049fc4bc648f1133c5269434a955f955f390f10594adb80a70d`.

The experiment removed v4's post-base descriptor search. After replaying the
exact v1 base selection inside each outer training side, it always fitted the
same four heads:

1. the selected unmodified base representation;
2. base plus the uniform-mean motif descriptor;
3. base plus the texture-weighted-mean motif descriptor;
4. base plus the top-two-texture motif descriptor.

The four probability rows were converted to centered log probabilities,
averaged with fixed weights of `0.25` each, and passed through softmax. There
was one recipe, no blend search, no class-specific threshold, no pair
correction, and no per-image rule.

The stability gate remained substantive: on each outer training pool, the
fixed consensus had to satisfy the frozen non-regression limits relative to
that pool's selected base. It was eligible in all 5 of 5 outer searches,
exceeding the required 4 of 5.

## Result compared with v4

| Metric | V4 | V5 | Better |
|---|---:|---:|---|
| Nested accuracy | **94.6840%** | 94.5068% | V4 |
| Macro F1 | **94.3622%** | 94.1957% | V4 |
| Balanced accuracy | **95.0256%** | 94.9226% | V4 |
| Source robustness | 74.2219% | **76.8884%** | V5 |
| Opishnyan recall | **88.4232%** | 88.0240% | V4 |
| Bubnivka recall | 90.6103% | **91.0798%** | V5 |
| Bubnivka precision | **81.0924%** | 79.8354% | V4 |
| Ceramic macro F1 | **91.6200%** | 91.3437% | V4 |
| Opishnyan -> Bubnivka | **43** | 46 | V4 |
| Post-base stability | 2/5 | **5/5** | V5 |
| Gates passed | 17/21 | **18/21** | V5 |

V5 changed 22 v4 decisions: it fixed 9 v4 errors, regressed 12 correct v4
predictions, and changed 1 error to another wrong class. The net result was
three fewer correct images than v4.

The fixed consensus accomplished its intended stability goal and recovered
Bubnivka recall above its floor. It also produced the best source-robustness
score of all five experiments. The equal consensus nevertheless admitted more
Opishnyan examples as Bubnivka, lowering Bubnivka precision and increasing the
asymmetric error count. Outer fold 3 reached only 297/342 correct; 300/342
would have passed the worst-fold gate.

## What the sample size does and does not prove

The result does **not** prove that passing all gates is mathematically
impossible with the current 1,693 images. V4 was only four favorable boundary
changes away from simultaneously satisfying its three class-boundary gates,
and v5 demonstrates that both Bubnivka recall and method stability can pass.

It does show that the current dataset cannot support a defensible claim that
the complete gate set has been met after five development-informed attempts.
Two gates failed every version:

- Bubnivka precision >= 82%;
- Opishnyan-to-Bubnivka errors <= 40.

The Bubnivka class has 213 development examples, compared with 501 Opishnyan
examples. In v5, 194/213 Bubnivka images were recovered, while 49 non-Bubnivka
images were predicted as Bubnivka. With 194 true positives, precision would
require no more than 42 false positives—seven fewer than observed. The related
Opishnyan error gate requires six of the 46 Opishnyan-to-Bubnivka errors to be
resolved.

These estimates are sensitive to a small number of images. Approximate 95%
Wilson intervals are 86.49–94.22% for Bubnivka recall, 74.34–84.40% for
Bubnivka precision, and 82.85–90.02% for the worst fold. The gate thresholds
lie inside these uncertainty ranges. That is evidence of limited precision,
not permission to relax a gate after observing it.

## Can a larger dataset improve these gates?

Yes, provided the added data contributes **new independent variation**. Merely
adding adjacent frames, reposts, or more images from the existing dominant
sources would make the count larger without addressing the observed failure.
The most useful future additions are:

- new physical objects and acquisition sources for both Opishnyan and
  Bubnivka, especially examples whose ornament language is visually similar;
- balanced coverage of plates, cups, figurines, tiles, textiles, museum
  displays, shop photographs, and clean catalog images;
- difficult but unambiguous boundary examples, reviewed by a domain expert;
- source/object group metadata captured at ingestion so related images remain
  atomic during splitting;
- enough new independent sources to reduce the dominance of the current large
  Opishnyan cohorts in individual folds.

When data grows, it must become a new dataset and split version. Promotion
thresholds should be declared before training, source-atomic folds rebuilt,
and a new sealed evaluation partition reserved. The current development rows
and repeated v1-v5 findings cannot be treated as a fresh unbiased test.

## Reproducibility and isolation

- The canonical result is under
  [`outputs/phase_5_source_robustness_v5/`](outputs/phase_5_source_robustness_v5/).
- The runner is
  [`scripts/run_phase5_v5_fixed_consensus.py`](scripts/run_phase5_v5_fixed_consensus.py).
- The array-only consensus implementation is
  [`src/ornament_classifier/fixed_consensus.py`](src/ornament_classifier/fixed_consensus.py).
- Two complete v5 runs reproduced all 14 artifacts byte for byte.
- The previously reproduced motif cache remained unchanged.
- The exact v1 base decision fingerprint reproduced before v5 was accepted.
- No sealed-test image was embedded, predicted, calibrated, or evaluated.

## Final Phase 5 conclusion

V4 remains the strongest aggregate and class-boundary candidate; v5 remains
the strongest source-robust and most stable motif procedure. Neither is
approved. A sixth experiment on the same repeatedly inspected development set
would add selection pressure without adding independent evidence, so Phase 5
modelling stops here.

The next legitimate path is to wait for a materially larger, source-diverse
dataset and establish a new evaluation contract. Until then, any useful
prototype should be described as development-stage and should route uncertain
ceramic predictions to human review rather than claim production readiness.
