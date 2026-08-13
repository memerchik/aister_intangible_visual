# Phase 3 classical baseline outputs

These artifacts reproduce the frozen Step 1 classical feature and model candidates on the source-atomic v3 development protocol. They contain no predictions for the 330-image sealed test set.

- `metrics.json`: evaluation scope, selected candidate, aggregate metrics, split and feature fingerprints, fixed-validation diagnostic, and software versions.
- `model_selection_summary.csv`: unweighted five-fold mean and variation for each candidate.
- `cv_fold_metrics.csv`: one result per candidate and held-out fold.
- `fixed_validation_metrics.csv`: the secondary fixed train/validation diagnostic.
- `oof_predictions.csv`: exactly one held-out prediction for each of the 1,693 development images.
- `per_class_metrics.csv`: precision, recall, F1, and support for the selected model.
- `confusion_matrix.csv` and `.png`: aggregate OOF confusion matrix.
- `confusion_pairs.csv`: ranked directed errors.
- `diagnostic_slices.csv`: accuracy by prior folder, review cohort, motif visibility, object type, source cohort, and CV fold.

The selected baseline is `knn_cosine_k5` with HSV, HOG, and LBP features. Selection uses unweighted mean five-fold accuracy; ties use mean macro F1 and then model name.

Its aggregate OOF results are 63.3196% accuracy, 59.2031% macro F1, 59.1724% balanced accuracy, and 87.9504% top-3 accuracy. Unweighted fold accuracy is 61.4701% ± 10.8879 percentage points. The difference reflects the intentionally unequal source-blocked fold sizes.

Rebuild from the repository root after installing `step_02/requirements.txt`:

```bash
python3 step_02/scripts/run_classical_baseline.py
```

See [`../../PHASE_3_FINDINGS.md`](../../PHASE_3_FINDINGS.md) for interpretation. The earlier v1 output was superseded and is not a valid benchmark.
