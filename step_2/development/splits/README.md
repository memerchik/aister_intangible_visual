# Phase 2 source-atomic evaluation contract

This directory is the single source of truth for Step 02 dataset membership. Paths in the CSV files are relative to `step_02/data/`.

## Current production split

- `train.csv`: 1,369 images for fitting model parameters.
- `validation.csv`: 324 images for development-only diagnostics.
- `test.csv`: 330 images sealed until the complete model and decision policy are frozen.
- `development.csv`: the 1,693 non-test images, with source-blocked `cv_fold` values 0–4.

The five development folds contain 500, 244, 270, 342, and 337 images. All five classes occur in every production partition and every fold.

This is an approximate, not exact, stratification. The 70/15/15 production targets are about 1,416/304/303, but large acquisition cohorts cannot be split. In particular, one 261-image Opishnyan source cohort occupies all of fold 0 for that cohort and creates most of the CV imbalance. Report aggregate OOF metrics and unweighted fold mean/variation together.

## Leakage controls

The assignment unit is the union of, in precedence order:

1. the 28 visually reviewed global same-acquisition-source cohorts;
2. the 46 visually adjudicated pretrained-similarity components;
3. manually confirmed source groups;
4. manually confirmed physical-object groups;
5. identical included image bytes.

The global audit directly assigns 982 included images to conservative source cohorts; propagation through the prior semantic groups gives 993 source-atomic members. The final split has 1,051 indivisible groups, with a maximum size of 261.

No source-atomic group, global source cohort, semantic group, confirmed source group, confirmed object group, or included content hash crosses a production split or a development CV fold. The 32 excluded images remain in `split_manifest.csv` without an assignment:

- 28 exact or visually identical re-encoded copies;
- 4 images without a usable visible motif.

Source cohorts are visual inferences from repeated watermarks, backdrops, camera/export signatures, or coherent sessions. Unassigned images are not proven independent, and shared provenance is not a licence to redistribute an image.

## Superseded versions

`step02_grouped_v1` was replaced after a pretrained-embedding audit revealed missed same-object/source relationships. `step02_semantic_grouped_v2` integrated those relationships but was replaced after all 28 global acquisition cohorts were found to cross its evaluation boundaries. Neither earlier test partition remains a benchmark.

The current version is `step02_source_atomic_v3`, seed `20260719`. Its assignment fingerprint is:

`59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`

## Usage rules

1. Treat `old_split` and all superseded Step 02 assignments as provenance only.
2. Use only train/validation and the five development folds for model selection.
3. Keep all source/group fields intact in every downstream experiment.
4. Report aggregate OOF metrics plus unweighted fold mean and standard deviation because fold sizes differ materially.
5. Freeze preprocessing, representation, model, hyperparameters, calibration, and rejection policy before any test evaluation.
6. Evaluate `test.csv` exactly once for the final frozen candidate; never tune from its predictions or errors.
7. Report macro F1, balanced accuracy, per-class recall, confusion matrix, and top-k accuracy alongside top-1 accuracy.

The v3 sealed test has not been evaluated.

## Reproduction

Run from the repository root:

```bash
python3 step_02/scripts/build_global_source_cohorts.py
python3 step_02/scripts/build_splits.py
python3 -m unittest discover -s step_02/tests -v
```

`split_audit.json` records exact input/review fingerprints, target and observed allocations, CSV hashes, group counts, and all leakage checks.
