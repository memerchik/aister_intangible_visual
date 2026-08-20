# Phase 5 promotion-gate matrix

This document records every promotion gate evaluated across Phase 5 v1 through
v5. A candidate can move to Phase 6 only when every gate required by its
frozen contract passes. `—` means that the gate did not exist in that
iteration and therefore was not evaluated.

V1 evaluated 17 gates, v2 evaluated 18, and v3, v4, and v5 each evaluated 21.
There are 24 distinct gate names across the five versions because the
pair-correction, consensus-fusion, motif-recipe, and fixed-consensus stability
gates are method-specific.

## Shared gates

These 17 gates were evaluated in every version.

| Gate | Required | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|---:|
| Overall accuracy | >= 92.5000% | 93.9752% ✓ | 93.7980% ✓ | 93.9161% ✓ | 94.6840% ✓ | 94.5068% ✓ |
| Overall macro F1 | >= 92.3400% | 93.6135% ✓ | 93.4211% ✓ | 93.5815% ✓ | 94.3622% ✓ | 94.1957% ✓ |
| Opishnyan recall | >= 87.5000% | 87.4251% ✗ | 87.4251% ✗ | 86.4271% ✗ | 88.4232% ✓ | 88.0240% ✓ |
| Bubnivka precision | >= 82.0000% | 79.6680% ✗ | 79.7468% ✗ | 78.4000% ✗ | 81.0924% ✗ | 79.8354% ✗ |
| Ceramic macro F1 | >= 91.0000% | 90.7297% ✗ | 90.4092% ✗ | 90.6397% ✗ | 91.6200% ✓ | 91.3437% ✓ |
| Opishnyan → Bubnivka errors | <= 40 | 45 ✗ | 45 ✗ | 50 ✗ | 43 ✗ | 46 ✗ |
| Challenging-Opishnyan accuracy | >= 70.0000% | 71.7703% ✓ | 71.7703% ✓ | 68.8995% ✗ | 74.1627% ✓ | 72.7273% ✓ |
| Hard-three Opishnyan pooled accuracy | >= 72.0000% | 73.9130% ✓ | 73.9130% ✓ | 71.1957% ✗ | 76.0870% ✓ | 73.9130% ✓ |
| Hard-three Opishnyan cohort-macro accuracy | >= 70.0000% | 71.5619% ✓ | 72.5917% ✓ | 69.9594% ✗ | 74.4859% ✓ | 74.8739% ✓ |
| Large named-source macro accuracy | >= 87.5000% | 87.5472% ✓ | 88.8561% ✓ | 88.4381% ✓ | 88.4244% ✓ | 88.5791% ✓ |
| Worst outer-fold accuracy | >= 87.5000% | 88.4273% ✓ | 89.0208% ✓ | 88.4273% ✓ | 88.8889% ✓ | 86.8421% ✗ |
| Outer-fold accuracy standard deviation | <= 5.5000 pp | 4.8979 pp ✓ | 4.4320 pp ✓ | 4.9327 pp ✓ | 4.5921 pp ✓ | 5.0521 pp ✓ |
| Ornek recall floor | >= 96.6300% | 98.6333% ✓ | 98.6333% ✓ | 98.6333% ✓ | 99.0888% ✓ | 98.8610% ✓ |
| Bubnivka recall floor | >= 90.9600% | 90.1408% ✗ | 88.7324% ✗ | 92.0188% ✓ | 90.6103% ✗ | 91.0798% ✓ |
| Petrykivka recall floor | >= 95.3100% | 96.9231% ✓ | 96.9231% ✓ | 96.9231% ✓ | 98.0769% ✓ | 98.0769% ✓ |
| Kosiv recall floor | >= 95.8600% | 98.5714% ✓ | 98.5714% ✓ | 98.5714% ✓ | 98.9286% ✓ | 98.5714% ✓ |
| Base-recipe stability | >= 4 of 5 outer searches | 4/5 ✓ | 4/5 ✓ | 4/5 ✓ | 4/5 ✓ | 4/5 ✓ |

## Iteration-specific gates

| Gate | Required | V1 | V2 | V3 | V4 | V5 |
|---|---:|---:|---:|---:|---:|---:|
| Pair-correction recipe stability | >= 4 of 5 outer searches | — | 0/5 ✗ | — | — | — |
| Consensus-fusion recipe stability | >= 4 of 5 outer searches | — | — | 0/5 ✗ | — | — |
| Motif-recipe stability | >= 4 of 5 outer searches | — | — | — | 2/5 ✗ | — |
| Fixed-consensus stability | >= 4 of 5 outer searches | — | — | — | — | 5/5 ✓ |
| V1 accuracy non-regression | >= 93.7752% | — | — | 93.9161% ✓ | 94.6840% ✓ | 94.5068% ✓ |
| V1 macro-F1 non-regression | >= 93.4135% | — | — | 93.5815% ✓ | 94.3622% ✓ | 94.1957% ✓ |
| V1 source-robustness non-regression | >= 70.5883% | — | — | 68.2122% ✗ | 74.2219% ✓ | 76.8884% ✓ |

The v3–v5 non-regression floors allow at most a 0.20 percentage-point accuracy
drop, a 0.20 point macro-F1 drop, and a 0.50 point source-robustness drop from
the recorded v1 result.

## Gate rationale and failure interpretation

These gates are project-specific decision rules defined for the frozen Phase 5
experiments. They combine aggregate quality, class-boundary behavior,
source-cohort robustness, fold stability, and selection stability. A failed
gate identifies the condition that prevented promotion; it does not by itself
prove a single cause.

### Shared gates

| Gate | Why it is tracked | What a failure indicates |
|---|---|---|
| Overall accuracy | Provides a direct aggregate measure of correct five-way decisions across all development images. | The procedure does not meet the minimum overall correctness required for promotion, even if some classes perform well. |
| Overall macro F1 | Gives every class equal weight while combining precision and recall. | Aggregate accuracy may be masking weak or imbalanced class-level behavior. |
| Opishnyan recall | Protects the largest class from being lost across the difficult ceramic boundary. | Too many true Opishnyan images are assigned to other classes. |
| Bubnivka precision | Limits false Bubnivka assignments, especially Opishnyan images admitted into the smaller Bubnivka class. | A Bubnivka prediction is not sufficiently specific under the required operating point. |
| Ceramic macro F1 | Summarizes balanced performance over Opishnyan, Bubnivka, and Kosiv ceramics. | The method has not achieved the required joint precision/recall quality across the three ceramic classes. |
| Opishnyan → Bubnivka errors | Directly bounds the dominant asymmetric confusion in absolute image counts. | The model still crosses the Opishnyan/Bubnivka boundary too often, regardless of its overall score. |
| Challenging-Opishnyan accuracy | Tests Opishnyan images from the three named difficult cohorts together with images lacking a named source cohort. | Performance is too dependent on easier or better represented Opishnyan sources. |
| Hard-three Opishnyan pooled accuracy | Measures image-weighted accuracy across `gray_catalog_b`, `shelf_figurines`, and `shop_cubbies`. | The combined difficult-source image pool remains below the required accuracy. |
| Hard-three Opishnyan cohort-macro accuracy | Gives each of the three difficult cohorts equal influence, independent of cohort size. | At least one difficult source can remain weak even when a larger cohort raises the pooled result. |
| Large named-source macro accuracy | Gives equal weight to every identified source cohort with at least 20 development images. | Performance does not transfer consistently across the larger known acquisition sources. |
| Worst outer-fold accuracy | Sets a floor on the weakest source-held-out fold. | At least one held-out source grouping is substantially less reliable than the aggregate result. |
| Outer-fold accuracy standard deviation | Limits variability across the five source-atomic outer folds. | Performance is too sensitive to which source groups are held out. |
| Ornek recall floor | Prevents a new procedure from sacrificing established Ornek recognition. | Too many Ornek examples are missed despite possible gains elsewhere. |
| Bubnivka recall floor | Prevents gains in Bubnivka precision from being obtained simply by avoiding Bubnivka predictions. | Too many true Bubnivka images are missed. |
| Petrykivka recall floor | Prevents the ceramic-focused work from regressing Petrykivka recognition. | Too many Petrykivka examples are missed despite possible gains elsewhere. |
| Kosiv recall floor | Protects the third ceramic class while the Opishnyan/Bubnivka boundary is adjusted. | Too many true Kosiv images are missed. |
| Base-recipe stability | Checks whether the full-development base winner remains competitive in at least four source-held-out selections. | The selected base recipe is sensitive to the particular sources available during selection. |

### Iteration-specific gates

| Gate | Why it is tracked | What a failure indicates |
|---|---|---|
| Pair-correction recipe stability | Checks whether the v2 binary correction transfers across outer source splits. | The chosen correction settings are source-dependent and do not repeat reliably. |
| Consensus-fusion recipe stability | Checks whether the v3 encoder and blend choice remains competitive across outer source splits. | The apparent fusion benefit depends on the selection subset or held-out sources. |
| Motif-recipe stability | Checks whether the v4 motif descriptor and classifier settings repeat across outer source splits. | The selected motif aggregation recipe is not stable enough across source compositions. |
| Fixed-consensus stability | Verifies that the v5 fixed four-head recipe remains eligible under the frozen limits in at least four outer searches. | Even without post-base recipe selection, the consensus does not transfer consistently across source-held-out evaluations. |
| V1 accuracy non-regression | Keeps later methods within the allowed aggregate-accuracy loss from the v1 reference. | A later method trades away more overall correct decisions than the contract permits. |
| V1 macro-F1 non-regression | Keeps later methods within the allowed class-balanced F1 loss from v1. | A later method sacrifices too much equal-weight class performance. |
| V1 source-robustness non-regression | Protects the source-cohort improvement established by v1. | A later method's aggregate or boundary gain is accompanied by an unacceptable loss on source-robust evaluation. |

## Gates failed by all five versions

Exactly two shared gates were not satisfied by any Phase 5 candidate:

1. **Bubnivka precision >= 82.0000%.** V4 is best at 81.0924%, still 0.9076
   percentage points short.
2. **Opishnyan → Bubnivka errors <= 40.** V4 is best at 43, three errors above
   the limit; v1/v2 produced 45, v3 produced 50, and v5 produced 46.

These two gates describe the persistent failure: the classifier has
not resolved the Opishnyan/Bubnivka ceramic boundary without admitting too many
Opishnyan examples as Bubnivka. V4 fixes the previously universal Opishnyan
recall and ceramic-macro-F1 failures, but its Bubnivka recall is 0.3497 points
below the class floor.

The method-specific stability failures are also important, but they are not
shared all-version failures: the base recipe passes at 4/5, v2 pair correction
and v3 fusion each scored 0/5, v4 motif aggregation scored 2/5, and the fixed
v5 consensus passed at 5/5. V5 proves that a stable motif procedure is possible
on the current data; it does not resolve the persistent boundary.

## Gate definitions

- Ceramic macro F1 is the unweighted mean of the Opishnyan, Bubnivka, and
  Kosiv per-class F1 scores.
- The hard-three slice contains the Opishnyan `gray_catalog_b`,
  `shelf_figurines`, and `shop_cubbies` source cohorts.
- Challenging Opishnyan contains true Opishnyan rows in those three cohorts or
  rows with no named source-atomic cohort.
- Large named sources are nonblank source-atomic cohorts with at least 20
  development images; their gate is the unweighted macro accuracy across
  those cohorts.
- Worst-fold accuracy and fold-accuracy standard deviation are calculated
  across the five source-atomic outer folds.
- A recipe is stable when the full-development winner remains eligible and
  within the frozen score tie window in at least four of five outer searches.
  V5 had no post-base winner selection; its fixed consensus instead had to
  satisfy the frozen eligibility limits in at least four of five searches.

The canonical values above come from the five immutable `metrics.json`
artifacts under `outputs/phase_5_source_robustness*/`.
