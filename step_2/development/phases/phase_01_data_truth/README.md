# Phase 01 — data truth

**Status: complete.** Phase 1 established what images exist, which records are
usable, and which images must remain linked or be excluded. It did not train a
model or create the current evaluation assignment.

## Outcome

- 2,055 supported, decodable image files were inventoried across five labels.
- Visual review completed all 143 multi-image candidate groups covering 429
  images.
- 28 exact or visually identical copies and 4 images without a usable motif
  were marked for exclusion, leaving 2,023 modelling records.
- `step_1/` remained unchanged as the prototype and recovery copy.
- Image provenance and redistribution licences remain unknown.

## Canonical evidence

- [Phase 1 findings](../../PHASE_1_FINDINGS.md)
- [Data-copy notes](../../data/README.md)
- [Metadata schema](../../metadata/SCHEMA.md)
- [Inventory manifest](../../metadata/manifest.csv)
- [Audit summary](../../metadata/data_audit.json)
- [Review index](../../review/IMAGE_REVIEW_INDEX.md)
- [Completed review workbook](../../review/outputs/019f6af8-e0cf-7513-85b2-f6eb3920df46/phase_1_image_review.xlsx)
- [Machine-readable adjudication](../../review/outputs/019f6af8-e0cf-7513-85b2-f6eb3920df46/phase_1_manual_review.json)

## Rebuild surface

- Runner: [`build_manifest.py`](../../scripts/build_manifest.py)
- Integrity tests: [`test_phase_1_outputs.py`](../../tests/test_phase_1_outputs.py)
- Review tests: [`test_manual_review_outputs.py`](../../tests/test_manual_review_outputs.py)

The manifest is the reproducible base inventory. Human decisions remain in the
review artifacts and are joined downstream; they are not written back into the
base manifest.

## Boundary

The original `dataset_dev` and `dataset_test` locations record prior provenance
only. They are not valid evaluation partitions. Phase 2 owns all current split
assignments.
