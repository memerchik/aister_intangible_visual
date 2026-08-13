# Phase 5 experiment record — v1 through v5

This document is the chronological record of all five completed Phase 5
experiments. Each iteration was frozen before fitting, used only the 1,693
development images under the five source-atomic folds, and left the 330-image
test set sealed.

See [`PHASE_5_GATES.md`](PHASE_5_GATES.md) for the complete gate definitions,
thresholds, observed values, and pass/fail matrix.

## Shared evaluation lifecycle

Every iteration followed the same non-negotiable sequence:

1. validate the source-atomic v3 split and content fingerprints;
2. load only content-addressed development embedding arrays allowed by the
   current frozen contract;
3. hold out one canonical outer fold;
4. select candidates with pooled OOF predictions from the other four folds;
5. refit the selected procedure on the four outer-training folds and predict
   the untouched outer fold once;
6. aggregate all five outer predictions as the selection-aware nested result;
7. repeat the selection procedure on all development folds only to describe a
   possible later recipe;
8. evaluate every frozen promotion and stability gate;
9. reproduce the complete artifact set independently;
10. stop unless every required gate passes.

Probabilities in all iterations are uncalibrated ranking scores. None of the
experiments trained a final serialized model, selected a confidence threshold,
or measured production accuracy.

## V1 — source-balanced DINOv3 selection

**Question.** Can source-balanced training and bounded DINOv3 view selection
repair the weak source cohorts without sacrificing aggregate quality?

**Frozen work.** V1 compared 12 base configurations across three DINOv3
feature families, two logistic-probe `C` values, and two source-group
exponents. It selected the base first, then compared no specialist with one
fixed three-way ceramic specialist.

**Selected full-development recipe.** Global-fivecrop DINOv3 CLS, `C=10`,
source exponent `0.5`, no specialist.

**What worked.** V1 fixed 26 Phase 4 errors, introduced 11 regressions, and
raised accuracy to 93.9752%. Opishnyan source-group recall rose by 7.9976
percentage points, hard-three accuracy rose by 9.2391 points, and
Opishnyan-to-Bubnivka errors fell from 58 to 45. Source weighting `0.5` was
selected in every outer fold.

**Why it stopped.** V1 failed 5 of 17 gates: Opishnyan recall, Bubnivka
precision, ceramic macro F1, Opishnyan-to-Bubnivka errors, and Bubnivka recall.
The result was rejected despite being the strongest aggregate result.

Evidence: [v1 contract](phases/phase_05_source_robustness/experiment_contract.json),
[v1 findings](PHASE_5_FINDINGS.md),
[v1 outputs](outputs/phase_5_source_robustness/).

## V2 — DINOv3 pairwise ceramic correction

**Question.** Can a binary Opishnyan/Bubnivka head correct the remaining
asymmetric boundary while preserving v1's source gains?

**Frozen work.** V2 replayed the three v1 base finalists, then compared 65
pairwise choices across four DINOv3 CLS/patch families, two `C` values, two
source exponents, and four blend weights. Pair correction preserved total pair
mass and every non-pair probability column.

**Selected full-development recipe.** V1 global CLS base plus letterbox
CLS+patch binary correction, `C=100`, source exponent `0.5`, blend `0.25`.

**What worked.** The full-development correction fixed 7 decisions and
regressed 2, improving its selection-conditional accuracy to 93.8571%.

**Why it stopped.** Nested accuracy fell to 93.7980%. V2 changed 18 v1
decisions, fixed 7, and regressed 10. Bubnivka recall fell to 88.7324%, while
Opishnyan-to-Bubnivka errors remained at 45. Five outer searches selected five
different correction recipes, and the full recipe was stable in 0 of 5. V2
failed 6 of 18 gates and was rejected.

Evidence: [v2 contract](phases/phase_05_source_robustness/experiment_contract_v2.json),
[v2 findings](PHASE_5_V2_FINDINGS.md),
[v2 outputs](outputs/phase_5_source_robustness_v2/).

## V3 — cross-encoder consensus

**Question.** Is the missing boundary evidence present in independent frozen
encoders rather than additional DINOv3-only pair heads?

**Frozen work.** V3 replayed the v1 base procedure and compared 28 consensus
choices built from DINOv2, ConvNeXt, CLIP, MobileNetV3, and DINOv3 patch
summaries. It tested fixed multiclass fusion, pair consensus, and majority-gated
pair consensus at three blends. It added explicit v1 non-regression checks and
allowed Phase 6 only on an all-pass result.

**Selected full-development recipe.** V1 global CLS base plus structural-four
multiclass consensus at blend `0.15`.

**What worked.** V3 recovered Bubnivka recall to 92.0188%, reduced
Bubnivka-to-Opishnyan errors from 17 in v1 to 13, and passed all four
class-recall floors. It performed better than v2 on accuracy and macro F1.

**Why it stopped.** The gain reversed the other side of the boundary:
Opishnyan recall fell to 86.4271%, Opishnyan-to-Bubnivka errors rose to 50,
and source robustness fell to 68.2122%. Five outer searches again selected five
different fusions; the full recipe was stable in 0 of 5. V3 failed 9 of 21
gates and was rejected.

Evidence: [v3 contract](phases/phase_05_source_robustness/experiment_contract_v3.json),
[v3 findings](PHASE_5_V3_FINDINGS.md),
[v3 outputs](outputs/phase_5_source_robustness_v3/).

## V4 — label-free motif localization

**Question.** Can localized high-texture ornament regions reduce dependence on
the carrier object and improve the ceramic boundary across sources?

**Frozen work.** V4 used a two-stage freeze. A label-, source-, and error-free
extraction contract first produced six DINOv3 tile embeddings for every
development image and four fixed motif descriptors. A separate classifier
contract was then frozen before any fit. It replayed the three exact v1 base
finalists and compared no motif with uniform, texture-weighted, texture-top-two,
and texture-dispersion descriptors at two `C` values.

**Selected full-development recipe.** Global DINOv3 CLS base plus the
texture-weighted motif descriptor, both with `C=10` and source exponent `0.5`.

**What worked.** V4 produced the best result on every aggregate metric:
94.6840% accuracy, 94.3622% macro F1, 95.0256% balanced accuracy, and 74.2219%
source robustness. Relative to v1, it fixed 22 errors and regressed 10, a net
gain of 12 correct images. It was the first iteration to pass the Opishnyan
recall and ceramic-macro-F1 gates and reduced Opishnyan-to-Bubnivka errors to
43.

**Why it stopped.** Bubnivka precision reached 81.0924% but remained below
82%; Opishnyan-to-Bubnivka errors remained three above the limit; Bubnivka
recall was 90.6103% against a 90.9600% floor; and the full motif recipe was
stable in only 2 of 5 outer searches. V4 passed 17 of 21 gates but was rejected
under the all-required rule.

Evidence:
[motif extraction contract](phases/phase_05_source_robustness/motif_extraction_contract_v4.json),
[v4 experiment contract](phases/phase_05_source_robustness/experiment_contract_v4.json),
[v4 findings](PHASE_5_V4_FINDINGS.md),
[v4 outputs](outputs/phase_5_source_robustness_v4/).

## V5 — fixed motif consensus

**Question.** Can one fixed consensus over global and complementary motif
evidence remove v4's recipe instability while passing the remaining boundary
gates?

**Frozen work.** V5 was declared the final attempt. It retained the exact v1
base selection, then always combined four heads: the selected base and the
uniform, texture-weighted, and top-two motif augmentations. Their centered log
probabilities were averaged with fixed `0.25` weights. There was one post-base
recipe and no descriptor, blend, threshold, or pair-correction search.

**What worked.** V5 passed 18 of 21 gates, the highest gate count. The fixed
consensus was eligible in all 5 outer searches, Bubnivka recall passed at
91.0798%, and source robustness reached a project-best 76.8884%.

**Why it stopped.** Bubnivka precision fell to 79.8354%,
Opishnyan-to-Bubnivka errors rose to 46, and outer fold 3 reached 86.8421%
against the 87.5% floor. V5 changed 22 v4 decisions, fixing 9 and regressing
12, for three fewer correct images. It was rejected under the unchanged
all-required rule.

Evidence: [v5 contract](phases/phase_05_source_robustness/experiment_contract_v5.json),
[v5 findings](PHASE_5_V5_FINDINGS.md),
[v5 outputs](outputs/phase_5_source_robustness_v5/).

## Final comparison

| Metric | V1 | V2 | V3 | V4 | V5 | Best |
|---|---:|---:|---:|---:|---:|---|
| Nested accuracy | 93.9752% | 93.7980% | 93.9161% | **94.6840%** | 94.5068% | V4 |
| Macro F1 | 93.6135% | 93.4211% | 93.5815% | **94.3622%** | 94.1957% | V4 |
| Balanced accuracy | 94.3388% | 94.0571% | 94.5147% | **95.0256%** | 94.9226% | V4 |
| Source robustness | 71.0883% | 70.7670% | 68.2122% | 74.2219% | **76.8884%** | V5 |
| Opishnyan recall | 87.4251% | 87.4251% | 86.4271% | **88.4232%** | 88.0240% | V4 |
| Bubnivka recall | 90.1408% | 88.7324% | **92.0188%** | 90.6103% | 91.0798% | V3 |
| Bubnivka precision | 79.6680% | 79.7468% | 78.4000% | **81.0924%** | 79.8354% | V4 |
| Ceramic macro F1 | 90.7297% | 90.4092% | 90.6397% | **91.6200%** | 91.3437% | V4 |
| Opishnyan -> Bubnivka | 45 | 45 | 50 | **43** | 46 | V4 |
| Post-base stability | — | 0/5 correction | 0/5 fusion | 2/5 motif | **5/5 fixed** | V5 |
| Gates passed | 12/17 | 12/18 | 12/21 | 17/21 | **18/21** | V5 |
| Promotion result | Reject | Reject | Reject | Reject | Reject | None |

## Combined conclusions

1. Source-balanced training is the most reliable improvement found. V1's
   exponent `0.5` remains supported across folds.
2. The bottleneck is a source-dependent ceramic boundary, not aggregate
   representation quality. All versions exceed 93.7% nested accuracy.
3. DINOv3-only pair correction is unstable and should not be expanded.
4. Independent encoders contain Bubnivka evidence, but their fusion transfers
   error to Opishnyan and weakens the hard source cohorts.
5. Motif localization is the first materially different method to deliver a
   clear nested gain. It improves both aggregate and source-robust metrics, but
   its aggregation recipe remains source-sensitive.
6. V5 proves that a stable motif procedure and the Bubnivka recall floor are
   attainable, but stability alone does not resolve the class boundary.
7. Full-development results in v2 through v5 do not replace nested adaptive
   evidence. Candidate selection must remain inside each outer training side.
8. Bubnivka precision and Opishnyan-to-Bubnivka errors failed every iteration.
   They should be treated as data/representation limitations, not tuned away
   with further inspection of the same development rows.
9. Phase 6 remains blocked. The sealed test cannot be used to choose among v1
   through v5.

## Current boundary

No Phase 5 candidate is approved. V4 is the strongest aggregate/boundary
candidate; v5 is the strongest source-robust and most stable motif candidate.
The five experiments have repeatedly informed development, so another method
search on the same rows would add selection pressure without independent
evidence. Modelling stops until materially more source-diverse data is
available. Phase 6 can begin only after a future version receives an
all-required `promote` decision; observed gates cannot be relaxed
retroactively.
