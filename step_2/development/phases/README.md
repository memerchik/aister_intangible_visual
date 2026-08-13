# Step 2 development phase map

This directory is the navigation layer for Step 02. It organizes the work by
phase without relocating, copying, or renaming any canonical artifact.

## Status

| Phase | Status | Role | Primary evidence |
|---|---|---|---|
| [01 — data truth](phase_01_data_truth/README.md) | Complete | Inventory, duplicate detection, and visual adjudication | [`PHASE_1_FINDINGS.md`](../PHASE_1_FINDINGS.md) |
| [02 — evaluation protocol](phase_02_evaluation_protocol/README.md) | Complete | Source-atomic v3 splits and leakage controls | [`PHASE_2_FINDINGS.md`](../PHASE_2_FINDINGS.md) |
| [03 — classical baseline](phase_03_classical_baseline/README.md) | Complete | Frozen handcrafted-feature reference | [`PHASE_3_FINDINGS.md`](../PHASE_3_FINDINGS.md) |
| [04 — pretrained screen](phase_04_pretrained_screen/README.md) | Complete | Frozen-encoder development screen | [`PHASE_4_FINDINGS.md`](../PHASE_4_FINDINGS.md) |
| [05 — source robustness](phase_05_source_robustness/README.md) | V1–v5 complete; all rejected; modelling closed pending new data | Nested source-robust selection, correction, fusion, motif localization/consensus, and promotion decisions | [`PHASE_5_EVOLUTION.md`](../PHASE_5_EVOLUTION.md) ([v5](../PHASE_5_V5_FINDINGS.md), [v4](../PHASE_5_V4_FINDINGS.md), [v3](../PHASE_5_V3_FINDINGS.md), [v2](../PHASE_5_V2_FINDINGS.md), [v1](../PHASE_5_FINDINGS.md)) |

“Complete” means that the phase's declared development work and evidence are
complete. It does not mean that a deployable classifier or an unbiased final
performance estimate exists. In particular, the 330-image test set remains
sealed and unevaluated.

## Provisional application track

[`../../v0_5/`](../../v0_5/README.md) is a separate, runnable
human-assisted product preview built from the fixed v4 full-development recipe.
It provides immediate rankings, motif visualization, human confirmation, and
opt-in data contribution without claiming that Phase 5 passed. It is not a
sixth Phase 5 experiment, does not start Phase 6, and does not access the sealed
test. The formal phase map below therefore remains unchanged.

## Where canonical files remain

- [`data/`](../data/README.md) contains the unchanged Step 02 image copy.
- [`metadata/`](../metadata/SCHEMA.md) is the base inventory and audit layer.
- [`review/`](../review/IMAGE_REVIEW_INDEX.md) contains human adjudication and
  source-cohort evidence.
- [`splits/`](../splits/README.md) is the current evaluation contract.
- [`scripts/`](../scripts) contains the reproducible phase runners.
- [`tests/`](../tests) contains integrity and reproducibility checks.
- [`outputs/`](../outputs) contains immutable experiment evidence.

The phase folders contain documentation and links only. A link from a phase
does not make the linked artifact phase-local, and no dataset or model output
should be duplicated here. Development implementation remains under `../src/`,
with phase-specific outputs under `../outputs/`. The deployable runtime is
owned separately by `../../v0_5/`.

## Evaluation rule

Model development may use [`train.csv`](../splits/train.csv),
[`validation.csv`](../splits/validation.csv), and the source-blocked folds in
[`development.csv`](../splits/development.csv). The sealed
[`test.csv`](../splits/test.csv) manifest must not be used for embeddings,
predictions, error analysis, selection, calibration, or threshold design. It
may be evaluated exactly once only after the entire model and decision policy
have been frozen in a later phase.

[`registry.json`](registry.json) provides the same phase map in a small
machine-readable form.
