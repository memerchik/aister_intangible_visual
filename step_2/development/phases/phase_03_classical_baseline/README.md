# Phase 03 — source-blocked classical baseline

**Status: complete.** Phase 3 is the frozen handcrafted-feature reference on
the source-atomic v3 development folds. It contains no sealed-test predictions.

## Outcome

The selected `knn_cosine_k5` candidate over HSV, HOG, and LBP features reaches:

- 63.3196% aggregate out-of-fold accuracy;
- 59.2031% macro F1;
- 59.1724% balanced accuracy;
- 87.9504% top-3 accuracy;
- 61.4701% ± 10.8879 percentage points unweighted fold accuracy.

The large fold variation is an expected consequence of honest source-held-out
evaluation. This result replaces the invalidated earlier v1 score and is the
classical reference for subsequent development comparisons.

## Canonical evidence

- [Phase 3 findings](../../PHASE_3_FINDINGS.md)
- [Output inventory and reproduction notes](../../outputs/phase_3_classical_baseline/README.md)
- [Canonical metrics](../../outputs/phase_3_classical_baseline/metrics.json)
- [Candidate comparison](../../outputs/phase_3_classical_baseline/model_selection_summary.csv)
- [OOF predictions](../../outputs/phase_3_classical_baseline/oof_predictions.csv)
- [Confusion matrix](../../outputs/phase_3_classical_baseline/confusion_matrix.png)

## Rebuild surface

- Runner: [`run_classical_baseline.py`](../../scripts/run_classical_baseline.py)
- Environment: [`requirements.txt`](../../requirements.txt)
- Output tests: [`test_phase_3_baseline_outputs.py`](../../tests/test_phase_3_baseline_outputs.py)

## Boundary

This phase establishes a reference, not a deployment candidate. It must remain
reproducible against the v3 development contract and must not be retuned using
the sealed test set.
