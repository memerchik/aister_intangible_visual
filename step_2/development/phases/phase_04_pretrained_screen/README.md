# Phase 4: pretrained encoder comparison

Phase 4 compares frozen image encoders and view policies with a fixed,
class-balanced logistic probe on the source-atomic v3 folds. The results are an
exploratory model comparison rather than a deployable model.

## Results

The selected `dinov3_vits16__cls__global_fivecrop` representation reaches:

- 93.0892% aggregate out-of-fold accuracy;
- 92.8447% macro F1;
- 94.0378% balanced accuracy;
- 99.5275% top-3 accuracy;
- 93.0985% ± 5.9735 percentage points unweighted fold accuracy.

The same out-of-fold results were used to select the representation, and the
earlier prototype influenced the candidates tested. These numbers therefore
describe development performance rather than an unbiased 93% production
estimate. The probability values are also uncalibrated.

## Files and findings

- [Phase 4 findings](../../PHASE_4_FINDINGS.md)
- [Output inventory](../../outputs/phase_4_pretrained/README.md)
- [Metrics](../../outputs/phase_4_pretrained/metrics.json)
- [Representation comparison](../../outputs/phase_4_pretrained/representation_summary.csv)
- [Out-of-fold predictions](../../outputs/phase_4_pretrained/oof_predictions.csv)
- [Confusion matrix](../../outputs/phase_4_pretrained/confusion_matrix.png)

## Code and tests

- Experiment runner: [`run_pretrained_embeddings.py`](../../scripts/run_pretrained_embeddings.py)
- Pinned environment: [`requirements-phase4.txt`](../../requirements-phase4.txt)
- Output tests: [`test_phase_4_pretrained_outputs.py`](../../tests/test_phase_4_pretrained_outputs.py)

The 330-image test set remains untouched. The selected DINOv3 checkpoint is
gated and uses a custom licence; both the model licence and dataset-image rights
need review before deployment.
