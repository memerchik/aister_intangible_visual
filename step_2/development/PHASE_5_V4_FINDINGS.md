# Phase 5 v4 findings — label-free motif localization

> Historical iteration: v4 remains the strongest aggregate Phase 5 result but
> is rejected. The final, also rejected v5 result is documented in
> [`PHASE_5_V5_FINDINGS.md`](PHASE_5_V5_FINDINGS.md).

## Decision

Phase 5 v4 is complete and **rejected for promotion**. It is the strongest
development result in the project so far: the selection-aware nested procedure
reached 94.6840% accuracy, 94.3622% macro F1, and a 74.2219%
source-robustness score over all 1,693 development images. It passed 17 of 21
required gates, but an all-pass result is mandatory. Phase 6 remains blocked,
and the 330-image test set remains sealed and unevaluated.

The four failed gates were:

| Gate | Required | V4 | Shortfall |
|---|---:|---:|---:|
| Bubnivka precision | >= 82.0000% | 81.0924% | 0.9076 pp |
| Opishnyan -> Bubnivka errors | <= 40 | 43 | 3 errors |
| Bubnivka recall | >= 90.9600% | 90.6103% | 0.3497 pp |
| Motif-recipe stability | >= 4/5 outer searches | 2/5 | 2 searches |

## What v4 tested

V4 was deliberately different from the v2 and v3 correction/fusion
experiments. It asked whether the object background was obscuring the ornament
and whether a classifier built from localized high-texture regions would
generalize better across cups, plates, rugs, paintings, and other carriers.

The work was frozen in two stages:

1. The
   [motif-extraction contract](phases/phase_05_source_robustness/motif_extraction_contract_v4.json)
   was frozen before any tile was encoded. It generated proposals without
   labels, source IDs, prior error membership, or test access.
2. After the cache existed, the
   [v4 experiment contract](phases/phase_05_source_robustness/experiment_contract_v4.json)
   was frozen before the first v4 classifier fit. This prevented the observed
   descriptor values from changing the candidate set.

For each development image, the extractor scored 18 square windows from two
scales and a 3x3 location grid, retained six diverse high-texture proposals,
and encoded each proposal with the same frozen DINOv3 ViT-S/16 representation.
Four fixed 384-dimensional motif descriptors were derived from the six tile
embeddings: uniform mean, texture-weighted mean, top-two texture mean, and
texture dispersion. No encoder fine-tuning or learned proposal mechanism was
used.

The classifier first replayed the exact three v1 base finalists. It then
compared the no-motif path with the four descriptors at `C=1` and `C=10`, all
with source exponent `0.5`. Selection was repeated inside every outer fold;
the outer fold remained untouched until one procedure had been chosen from its
four-fold training side.

## Main result

| Metric | V1 | V2 | V3 | V4 |
|---|---:|---:|---:|---:|
| Nested accuracy | 93.9752% | 93.7980% | 93.9161% | **94.6840%** |
| Macro F1 | 93.6135% | 93.4211% | 93.5815% | **94.3622%** |
| Balanced accuracy | 94.3388% | 94.0571% | 94.5147% | **95.0256%** |
| Source robustness | 71.0883% | 70.7670% | 68.2122% | **74.2219%** |
| Opishnyan recall | 87.4251% | 87.4251% | 86.4271% | **88.4232%** |
| Bubnivka recall | 90.1408% | 88.7324% | **92.0188%** | 90.6103% |
| Bubnivka precision | 79.6680% | 79.7468% | 78.4000% | **81.0924%** |
| Ceramic macro F1 | 90.7297% | 90.4092% | 90.6397% | **91.6200%** |
| Opishnyan -> Bubnivka | 45 | 45 | 50 | **43** |

V4 made 1,603 correct and 90 incorrect nested predictions. Relative to v1,
35 decisions changed: 22 v1 errors were fixed, 10 correct decisions regressed,
and 3 errors changed to another wrong class, for a net gain of 12 correct
images. V4 is the first Phase 5 iteration to pass both the Opishnyan-recall and
ceramic-macro-F1 gates. Consequently, only two shared gates remained failed by
all four versions at that point: Bubnivka precision and
Opishnyan-to-Bubnivka errors. Both also failed the later v5 experiment.

The full-development selection chose global DINOv3 CLS (`C=10`, source
exponent `0.5`) plus the texture-weighted motif mean (`C=10`, source exponent
`0.5`). Its selection-conditional OOF result was weaker than the adaptive
nested result: 94.1524% accuracy, 93.8604% macro F1, 78.8618% Bubnivka
precision, and 49 Opishnyan-to-Bubnivka errors. It is reported for recipe
description only and is not promotion evidence.

## Stability finding

The base recipe remained stable in 4 of 5 outer searches. The full-development
motif recipe was within the frozen eligibility/tie window in only outer folds
1 and 2, so motif stability was 2 of 5 rather than the required 4 of 5.

The selected outer procedures were:

| Outer fold | Base | Motif path |
|---|---|---|
| 0 | Three-view CLS | Texture top-two mean, `C=10` |
| 1 | Global CLS | Uniform mean, `C=10` |
| 2 | Three-view CLS | Texture-weighted mean, `C=10` |
| 3 | Three-view CLS | No motif descriptor |
| 4 | Global CLS | Uniform mean, `C=10` |

This variation shows that motif localization contains useful signal, but the
best way to aggregate it remains source-sensitive. It should not be hidden by
choosing the single full-development winner after inspecting all folds.

## Reproducibility and isolation audit

- Extraction contract SHA-256:
  `19df4814d2d915c461f267eed89282e5474abb1ac0d37fc50d9dbf25be7c33ff`.
- Experiment contract SHA-256:
  `7df72082a99d835a1bbbc91c9eb69ca399d1242cd97e9ca64a0ea38135ec42da`.
- Motif cache SHA-256:
  `97ee78769b6b4349b58e9b2cb27c9f4b58748ecf7cd9fbcfcc554c2fa122db40`.
- The cache contains 1,693 image rows, six proposals per image, 10,158 tile
  embeddings, and four aggregate descriptor arrays.
- A clean second extraction reproduced the cache byte for byte, including its
  fingerprint
  `7c0e76849ec8bc31ab433aea1d2264f638db81284c150935b04c10b73aa3cf64`.
- Loading the local Transformers-format DINOv3 weights into the equivalent
  extraction graph was checked against the frozen Phase 4 center embeddings;
  maximum absolute error was `3.0174851417541504e-07`.
- Two complete classifier runs independently reproduced all 14 output
  artifacts byte for byte.
- The v1 base decision fingerprint reproduced before motif results were
  accepted.
- No sealed-test embeddings, predictions, metrics, or labels were used.

Canonical evidence is in
[`outputs/phase_5_source_robustness_v4/`](outputs/phase_5_source_robustness_v4/).
The runner is
[`scripts/run_phase5_v4_motif.py`](scripts/run_phase5_v4_motif.py), the strict
cache boundary is
[`src/ornament_classifier/embeddings_v4.py`](src/ornament_classifier/embeddings_v4.py),
and the proposal implementation is
[`src/ornament_classifier/motif.py`](src/ornament_classifier/motif.py).

## Conclusion and next boundary

Motif-localized DINOv3 features are a real improvement, not a failed idea:
they produce the best aggregate, balanced, source-robust, and ceramic results
seen so far. They do not yet satisfy the product contract. The remaining
failure is narrow but consequential: Bubnivka predictions still absorb too
many Opishnyan examples, Bubnivka recall is slightly below its floor, and the
chosen motif aggregation is not stable enough across sources.

V4 must therefore remain a rejected development candidate. Phase 6 cannot
begin from it, the gates must not be relaxed after seeing these results, and
the sealed test cannot be used to decide whether the remaining shortfall is
acceptable.
