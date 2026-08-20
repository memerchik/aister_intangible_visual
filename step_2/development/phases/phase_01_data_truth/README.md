# Phase 1: data audit

Phase 1 established what images exist, which records are usable, and which
images are duplicates or belong together. This phase contains data review only;
model training and evaluation splits are handled in later phases.

## Results

- 2,055 supported, decodable image files were inventoried across five labels.
- All 143 multi-image candidate groups, covering 429 images, were reviewed.
- 28 exact or visually identical copies and 4 images without a usable motif
  were excluded, leaving 2,023 modelling records.
- `step_1/` remained unchanged as the prototype and recovery copy.
- Image provenance and redistribution licences remain unknown.

## Files and findings

- [Phase 1 findings](../../PHASE_1_FINDINGS.md)
- [Data-copy notes](../../data/README.md)
- [Metadata schema](../../metadata/SCHEMA.md)
- [Inventory manifest](../../metadata/manifest.csv)
- [Audit summary](../../metadata/data_audit.json)
- [Review index](../../review/IMAGE_REVIEW_INDEX.md)
- [Completed review workbook](../../review/outputs/019f6af8-e0cf-7513-85b2-f6eb3920df46/phase_1_image_review.xlsx)
- [Machine-readable review decisions](../../review/outputs/019f6af8-e0cf-7513-85b2-f6eb3920df46/phase_1_manual_review.json)

## Code and tests

- Dataset inventory: [`build_manifest.py`](../../scripts/build_manifest.py)
- Integrity tests: [`test_phase_1_outputs.py`](../../tests/test_phase_1_outputs.py)
- Review tests: [`test_manual_review_outputs.py`](../../tests/test_manual_review_outputs.py)

The generated manifest is the base image inventory. Manual decisions are stored
in the review files and joined by later scripts rather than written back into
the original manifest.

The original `dataset_dev` and `dataset_test` directories are retained as
provenance. Current evaluation assignments come from the Phase 2 split files,
not from those directory names.
