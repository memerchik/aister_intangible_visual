# Development phases

This directory provides a short guide to each part of the Step 2 research. The
datasets, scripts, tests, and result files remain in their main directories;
the phase pages link to them and summarize what was learned.

## Status

| Phase | Status | Work | Main report |
|---|---|---|---|
| [1 — data audit](phase_01_data_truth/README.md) | Complete | Inventory, duplicate detection, and visual review | [`PHASE_1_FINDINGS.md`](../PHASE_1_FINDINGS.md) |
| [2 — evaluation protocol](phase_02_evaluation_protocol/README.md) | Complete | Source-atomic v3 splits and leakage controls | [`PHASE_2_FINDINGS.md`](../PHASE_2_FINDINGS.md) |
| [3 — classical baseline](phase_03_classical_baseline/README.md) | Complete | Handcrafted-feature reference | [`PHASE_3_FINDINGS.md`](../PHASE_3_FINDINGS.md) |
| [4 — pretrained encoders](phase_04_pretrained_screen/README.md) | Complete | Frozen-encoder comparison | [`PHASE_4_FINDINGS.md`](../PHASE_4_FINDINGS.md) |
| [5 — source robustness](phase_05_source_robustness/README.md) | Complete; no candidate met every release gate | Source-aware selection, correction, fusion, motif localization, and stability testing | [`PHASE_5_EVOLUTION.md`](../PHASE_5_EVOLUTION.md) |

All five research phases have finished, but the project does not yet have an
independently validated production model. The 330-image test set remains
sealed and unevaluated.

## Application preview

[`../../v0_5/`](../../v0_5/README.md) is a separate, runnable application built
from the fixed v4 full-development recipe. It provides immediate rankings,
motif visualization, human confirmation, and optional data contribution. Its
scores are uncalibrated, and it does not access the sealed test set or claim
production accuracy.

## File locations

- [`data/`](../data/README.md) contains the verified Step 2 image copy.
- [`metadata/`](../metadata/SCHEMA.md) contains the inventory and audit data.
- [`review/`](../review/IMAGE_REVIEW_INDEX.md) contains manual review and
  source-cohort evidence.
- [`splits/`](../splits/README.md) defines the current evaluation split.
- [`scripts/`](../scripts) contains dataset builders and experiment runners.
- [`tests/`](../tests) contains integrity and reproducibility checks.
- [`outputs/`](../outputs) contains saved experiment results.
- [`src/`](../src/README.md) contains reusable modelling code.

The phase folders themselves contain documentation and links only.

## Evaluation protocol

Model development uses [`train.csv`](../splits/train.csv),
[`validation.csv`](../splits/validation.csv), and the source-blocked folds in
[`development.csv`](../splits/development.csv). [`test.csv`](../splits/test.csv)
is reserved for one evaluation after the complete model and decision policy
have been frozen. It is excluded from embeddings, predictions, error analysis,
model selection, calibration, and threshold design during development.

[`registry.json`](registry.json) provides the phase map in a small,
machine-readable format.
