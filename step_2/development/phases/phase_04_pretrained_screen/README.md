# Phase 04 — frozen pretrained-representation screen

**Status: complete at the exploratory development level.** Phase 4 compares
frozen image encoders and view policies with a fixed class-balanced logistic
probe on the source-atomic v3 folds. It does not create a deployable model.

## Outcome

The selected `dinov3_vits16__cls__global_fivecrop` representation reaches:

- 93.0892% aggregate out-of-fold accuracy;
- 92.8447% macro F1;
- 94.0378% balanced accuracy;
- 99.5275% top-3 accuracy;
- 93.0985% ± 5.9735 percentage points unweighted fold accuracy.

These metrics are **selection-conditional and exploratory**. The same OOF
results selected the representation, and earlier prototype work informed the
candidate family and probe setting. They show a strong development candidate,
not an unbiased 93% production guarantee. Scores are also uncalibrated.

## Canonical evidence

- [Phase 4 findings](../../PHASE_4_FINDINGS.md)
- [Output inventory and reproduction notes](../../outputs/phase_4_pretrained/README.md)
- [Canonical metrics](../../outputs/phase_4_pretrained/metrics.json)
- [Representation comparison](../../outputs/phase_4_pretrained/representation_summary.csv)
- [OOF predictions](../../outputs/phase_4_pretrained/oof_predictions.csv)
- [Confusion matrix](../../outputs/phase_4_pretrained/confusion_matrix.png)

## Rebuild surface

- Runner: [`run_pretrained_embeddings.py`](../../scripts/run_pretrained_embeddings.py)
- Pinned environment: [`requirements-phase4.txt`](../../requirements-phase4.txt)
- Output tests: [`test_phase_4_pretrained_outputs.py`](../../tests/test_phase_4_pretrained_outputs.py)

## Boundary

The runner, metadata, and outputs are frozen evidence for this completed screen.
The 330-image test set remains untouched. The selected DINOv3 checkpoint is
gated and uses a custom licence; legal review and resolution of dataset-image
rights are required before deployment.
