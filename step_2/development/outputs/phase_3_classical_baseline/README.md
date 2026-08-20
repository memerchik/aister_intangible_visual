# Phase 3 classical baseline results

This folder contains the saved results from running the Step 1 handcrafted
features and model candidates on the source-atomic v3 development folds. It
contains no predictions for the 330-image sealed test set.

## Files

- `metrics.json`: experiment scope, selected model, aggregate metrics,
  fingerprints, validation diagnostics, and software versions.
- `model_selection_summary.csv`: five-fold mean and variation for every model.
- `cv_fold_metrics.csv`: one result for each model and held-out fold.
- `fixed_validation_metrics.csv`: secondary train-to-validation diagnostic.
- `oof_predictions.csv`: one held-out prediction for each of the 1,693
  development images.
- `per_class_metrics.csv`: precision, recall, F1, and support by class.
- `confusion_matrix.csv` and `confusion_matrix.png`: aggregate confusion matrix.
- `confusion_pairs.csv`: directed errors ranked by frequency.
- `diagnostic_slices.csv`: accuracy by provenance, review cohort, motif
  visibility, object type, source cohort, and fold.

## Result

The selected baseline is `knn_cosine_k5` with HSV, HOG, and LBP features.
Models were ranked by unweighted mean five-fold accuracy, then macro F1 and
model name for ties.

It reached 63.3196% aggregate out-of-fold accuracy, 59.2031% macro F1, 59.1724%
balanced accuracy, and 87.9504% top-3 accuracy. Unweighted fold accuracy was
61.4701% ± 10.8879 percentage points. The difference comes from the deliberately
uneven source-blocked fold sizes.

## Reproduce

Install [`../../requirements.txt`](../../requirements.txt), then run from the
repository root:

```bash
python3 step_2/development/scripts/run_classical_baseline.py
```

See [`../../PHASE_3_FINDINGS.md`](../../PHASE_3_FINDINGS.md) for interpretation.
The earlier result based on split v1 has been superseded.
