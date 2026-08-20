# Phase 3: classical baseline

Phase 3 provides the handcrafted-feature reference on the source-atomic v3
development folds. It contains no sealed-test predictions.

## Results

The selected `knn_cosine_k5` model uses HSV, HOG, and LBP features and reaches:

- 63.3196% aggregate out-of-fold accuracy;
- 59.2031% macro F1;
- 59.1724% balanced accuracy;
- 87.9504% top-3 accuracy;
- 61.4701% ± 10.8879 percentage points unweighted fold accuracy.

The large difference between folds reflects the honest source-held-out
evaluation. This corrected result replaces the earlier score produced with the
superseded v1 split.

## Files and findings

- [Phase 3 findings](../../PHASE_3_FINDINGS.md)
- [Output inventory](../../outputs/phase_3_classical_baseline/README.md)
- [Metrics](../../outputs/phase_3_classical_baseline/metrics.json)
- [Candidate comparison](../../outputs/phase_3_classical_baseline/model_selection_summary.csv)
- [Out-of-fold predictions](../../outputs/phase_3_classical_baseline/oof_predictions.csv)
- [Confusion matrix](../../outputs/phase_3_classical_baseline/confusion_matrix.png)

## Code and tests

- Experiment runner: [`run_classical_baseline.py`](../../scripts/run_classical_baseline.py)
- Environment: [`requirements.txt`](../../requirements.txt)
- Output tests: [`test_phase_3_baseline_outputs.py`](../../tests/test_phase_3_baseline_outputs.py)

This result is a comparison baseline rather than a deployment candidate. It is
defined against the v3 development folds; the sealed test set was not used.
