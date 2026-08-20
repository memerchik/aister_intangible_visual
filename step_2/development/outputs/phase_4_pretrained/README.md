# Phase 4 pretrained encoder results

This folder contains the saved comparison of frozen pretrained encoders and
view policies on the source-atomic v3 development folds. It contains no
embeddings or predictions for the 330-image sealed test set.

## Files

- `metrics.json`: experiment identity, integrity checks, selection policy,
  aggregate metrics, model provenance, and software versions.
- `representation_summary.csv`: metrics for all 13 representations and the
  selected candidate.
- `cv_fold_metrics.csv`: one held-out result for each representation and fold.
- `probe_screen.csv`: secondary train-to-validation comparison of `C` values.
- `representation_metadata.csv`: model revisions, hashes, input policies,
  embedding fingerprints, runtimes, and cache identities.
- `oof_predictions.csv`: one held-out prediction and class-score vector for
  each of the 1,693 development images.
- `per_class_metrics.csv`: precision, recall, F1, and support by class.
- `confusion_matrix.csv` and `confusion_matrix.png`: aggregate confusion matrix.
- `confusion_pairs.csv`: directed errors ranked by frequency.
- `diagnostic_slices.csv`: accuracy by provenance, review status, motif
  visibility, object type, source cohort, and fold.

## Result

The selected representation is `dinov3_vits16__cls__global_fivecrop`: DINOv3
ViT-S/16 CLS features averaged over a letterboxed global view and five crops,
followed by a class-balanced logistic probe with fixed `C=10`.

It reached 93.0892% aggregate out-of-fold accuracy, 92.8447% macro F1, 94.0378%
balanced accuracy, and 99.5275% top-3 accuracy. Unweighted fold accuracy was
93.0985% ± 5.9735 percentage points.

The same out-of-fold results selected the representation, and earlier prototype
work influenced the candidates. These are exploratory development metrics, not
an independent performance estimate. The class-score columns are uncalibrated.

## Reproduce

Use the pinned [`../../requirements-phase4.txt`](../../requirements-phase4.txt)
environment with the required pretrained weights already available, then run:

```bash
python3 step_2/development/scripts/run_pretrained_embeddings.py
```

DINOv3 access is gated and its custom licence needs review before deployment.
See [`../../PHASE_4_FINDINGS.md`](../../PHASE_4_FINDINGS.md) for the complete
interpretation.
