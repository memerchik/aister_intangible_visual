# Phase 4 pretrained-representation outputs

These artifacts screen frozen pretrained encoders and view policies on the source-atomic v3 development folds. They contain no embeddings or predictions for the 330-image sealed test set.

- `metrics.json`: experiment and split identity, integrity checks, selection policy, aggregate metrics, probability policy, selected model provenance, and software versions.
- `representation_summary.csv`: aggregate and unweighted fold metrics for all 13 representations, including the selection flag.
- `cv_fold_metrics.csv`: one held-out result per representation and source-blocked fold.
- `probe_screen.csv`: diagnostic fixed train-to-validation `C` screen; it does not control OOF fitting.
- `representation_metadata.csv`: model revisions, weight/configuration hashes, input policies, embedding fingerprints, runtimes, and cache identities.
- `oof_predictions.csv`: one held-out prediction and uncalibrated class score vector for each of the 1,693 development images.
- `per_class_metrics.csv`: aggregate OOF precision, recall, F1, and support for the selected representation.
- `confusion_matrix.csv` and `.png`: selected aggregate OOF confusion matrix.
- `confusion_pairs.csv`: ranked selected-representation errors.
- `diagnostic_slices.csv`: accuracy by prior folder, review status, motif visibility, object type, global source cohort, and CV fold.

The selected representation is `dinov3_vits16__cls__global_fivecrop`: DINOv3 ViT-S/16 CLS features averaged over a letterboxed global view and five crops, followed by a class-balanced logistic probe with fixed `C=10`.

Its exploratory aggregate OOF result is 93.0892% accuracy, 92.8447% macro F1, 94.0378% balanced accuracy, and 99.5275% top-3 accuracy. Unweighted fold accuracy is 93.0985% ± 5.9735 percentage points.

These numbers are selection-conditional, because the same OOF results selected the representation and earlier prototype work informed the candidate set and fixed probe setting. They are not an unbiased performance estimate. Probability columns are uncalibrated ranking scores, not deployable confidence values.

Rebuild from the repository root in the separately pinned `step_02/requirements-phase4.txt` environment, with the required pretrained weights already available:

```bash
python3 step_02/scripts/run_pretrained_embeddings.py
```

DINOv3 access is gated and its custom licence requires legal review before deployment. The run does not create a final serialized model or any deployment service. See [`../../PHASE_4_FINDINGS.md`](../../PHASE_4_FINDINGS.md) for the full interpretation.
