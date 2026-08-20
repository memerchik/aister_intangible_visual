# Step 2 dataset copy

This is the verified Step 2 copy of the labeled images from `step_1/data/`.
Keeping a separate copy preserves the original prototype while the audited
workflow uses this folder.

- `dataset_dev/` and `dataset_test/` retain the original folder layout for
  provenance.
- The folder names are not the current evaluation split. Current train,
  validation, development-fold, and sealed-test assignments are defined in
  [`../splits/`](../splits/README.md).
- The audit retained 2,023 of the 2,055 images. Exclusions and review decisions
  are recorded in [`../metadata/`](../metadata/SCHEMA.md) and
  [`../review/`](../review/IMAGE_REVIEW_INDEX.md).
