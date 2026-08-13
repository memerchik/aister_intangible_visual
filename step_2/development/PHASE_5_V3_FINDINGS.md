# Phase 5 v3 findings — cross-encoder consensus

> Historical iteration: v3 remains rejected evidence. The later, also
> rejected v4 result is documented in
> [`PHASE_5_V4_FINDINGS.md`](PHASE_5_V4_FINDINGS.md).

## Decision

Phase 5 v3 is complete and **rejected for promotion**. It reached 93.9161%
nested development accuracy and recovered Bubnivka recall to 92.0188%, but it
shifted the ceramic boundary away from Opishnyan, reduced source robustness,
and selected no stable fusion recipe. It failed 9 of 21 required gates.

`phase_6_transition_allowed` is therefore `false`. Phase 6 calibration,
packaging, and sealed evaluation must not begin from v1, v2, or v3.

The 330-image test set remains sealed and unevaluated. No test image was
embedded, predicted, calibrated, or inspected.

## Frozen experiment

The v3 contract was frozen before the first v3 fit at
[`phases/phase_05_source_robustness/experiment_contract_v3.json`](phases/phase_05_source_robustness/experiment_contract_v3.json).
Its SHA-256 is
`828cc5aa45724518e10808929589722133ce336ee272cb8bda20b5089c0dc980`.

V3 retained the exact three-candidate v1 base procedure and compared 28
predeclared fusion choices after base selection:

- one unmodified v1 reference;
- three auxiliary families: four independent encoders, four
  structure-oriented heads, and a broad six-head family;
- multiclass centered-log fusion, pair-only consensus, and two-thirds-majority
  gated pair consensus;
- fixed blend weights of `0.15`, `0.30`, and `0.50`.

The auxiliary evidence came from frozen development embeddings produced by
DINOv2, ConvNeXt, CLIP, MobileNetV3, and DINOv3 letterbox/global patch
summaries. Every head was a five-class logistic probe with `C=10` and
fold-local source exponent `0.5`. Shared heads were fitted once per current
train/validation boundary and reused across candidates.

V3 never used source IDs, image IDs, prior error membership, object type, or
motif visibility as inference inputs. Pair fusion could operate only on rows
whose base argmax was Opishnyan or Bubnivka. It preserved their combined
probability mass and every non-pair probability column.

Selection remained nested and sequential: choose the base on inner OOF
predictions, choose the fusion on the same inner pool without the outer fold,
refit the selected procedure on the four outer-training folds, and predict the
outer fold once. The uncorrected v3 path reproduced the v1 nested decision
fingerprint before v3 outputs were accepted.

## Results

| Evaluation | Accuracy | Macro F1 | Balanced accuracy | Source robustness |
|---|---:|---:|---:|---:|
| Phase 5 v1 nested | **93.9752%** | **93.6135%** | 94.3388% | **71.0883%** |
| Phase 5 v2 nested | 93.7980% | 93.4211% | 94.0571% | 70.7670% |
| Phase 5 v3 nested | 93.9161% | 93.5815% | **94.5147%** | 68.2122% |
| V3 full-development selected recipe OOF | 93.7389% | 93.3974% | 94.2385% | 70.9935% |

V3 made 1,590 correct and 103 incorrect nested predictions: one fewer correct
than v1 and two more than v2. Relative to v1 it changed 14 decisions, fixed 6,
regressed 7, and changed one error to another wrong class.

| Ceramic diagnostic | V1 | V2 | V3 |
|---|---:|---:|---:|
| Opishnyan recall | **87.4251%** | **87.4251%** | 86.4271% |
| Bubnivka recall | 90.1408% | 88.7324% | **92.0188%** |
| Bubnivka precision | **79.6680%** | 79.7468% | 78.4000% |
| Ceramic macro F1 | **90.7297%** | 90.4092% | 90.6397% |
| Opishnyan -> Bubnivka | **45** | **45** | 50 |
| Bubnivka -> Opishnyan | 17 | 20 | **13** |

Cross-encoder evidence did supply useful information: it recovered four
Bubnivka-to-Opishnyan mistakes relative to v1. The tradeoff was unacceptable,
however. Five additional Opishnyan images moved toward Bubnivka, Bubnivka
precision fell, and the source-heavy Opishnyan cohorts regressed.

## Selection and stability

The five outer searches again selected five different fusions:

| Outer fold | Base | Fusion |
|---:|---|---|
| 0 | three-view DINOv3 CLS | structural-four multiclass, blend `0.30` |
| 1 | global DINOv3 CLS | independent-four multiclass, blend `0.30` |
| 2 | three-view DINOv3 CLS | independent-four multiclass, blend `0.15` |
| 3 | three-view DINOv3 CLS | independent-four majority pair, blend `0.50` |
| 4 | global DINOv3 CLS | none |

The full-development winner was global DINOv3 CLS plus structural-four
multiclass fusion at blend `0.15`. It was ineligible in three outer searches
and outside the readiness tie window in all five. Fusion stability was 0 of the
required 4 searches. Base stability remained 4 of 5.

The full-development fusion appeared mildly useful in isolation: it changed 8
decisions relative to the unmodified full-development base, fixed 5, regressed
2, and raised accuracy from 93.5617% to 93.7389%. As in v2, nested evaluation
shows that this full-development gain is not a stable cross-source result.

## Promotion gates

V3 retained all v1 gates, added fusion stability, and added three explicit v1
non-regression checks. Twelve of 21 passed; nine failed:

| Failed gate | Required | Observed |
|---|---:|---:|
| Opishnyan recall | >= 87.5000% | 86.4271% |
| Bubnivka precision | >= 82.0000% | 78.4000% |
| Ceramic macro F1 | >= 91.0000% | 90.6397% |
| Opishnyan -> Bubnivka errors | <= 40 | 50 |
| Challenging Opishnyan accuracy | >= 70.0000% | 68.8995% |
| Hard-three Opishnyan pooled accuracy | >= 72.0000% | 71.1957% |
| Hard-three cohort-macro accuracy | >= 70.0000% | 69.9594% |
| Fusion stability | >= 4 of 5 | 0 of 5 |
| V1 source-robustness non-regression | >= 70.5883% | 68.2122% |

Aggregate accuracy, macro F1, all four class-recall floors, the large named
source gate, worst-fold accuracy, fold variation, base stability, and the v1
accuracy/macro-F1 non-regression checks passed. The all-required policy keeps
the decision at `reject`.

## Reproducibility and integrity

The canonical runner is
[`scripts/run_phase5_v3_consensus.py`](scripts/run_phase5_v3_consensus.py),
whose SHA-256 is
`a62266e0e3062103df24b43fb5ceb8ceafebfce2b3ef5a42ae3c728791a76260`.
The 13 canonical artifacts are under
[`outputs/phase_5_source_robustness_v3/`](outputs/phase_5_source_robustness_v3/).
`metrics.json` has SHA-256
`c624e2c6a8254716f0fc62abe2d493e3d40ad67be447787ed86528aee14b89e0`.

A second complete run reproduced all 13 files byte for byte, including the
PNG. Independent tests reconstruct the metrics, source scores, readiness
score, grids, eligibility, ranking, both stability rules, all 21 gates,
confusion outputs, provenance, and v1/v2/v3 comparison.

## Conclusion

Cross-encoder consensus is not the missing final correction. It improves the
Bubnivka side of the boundary but transfers the error burden to Opishnyan and
remains source-dependent. The strongest development procedure remains v1,
which itself failed five gates.

Do not begin Phase 6. Further work needs a newly authorized and separately
frozen experiment; it cannot be presented as a revision of v3 or validated on
the sealed test.
