# Phase 1 findings - data truth

Phase 1 has created a reproducible inventory, a conservative grouping review layer for every image, and a completed visual review of all multi-image candidates. It has not changed labels, moved images, created a new evaluation split, or trained a model.

## Verified dataset

- The Step 2 data directory was copied from `step_1/data/` and verified with a recursive byte comparison.
- There are 2,055 supported image files across five ornament labels.
- All 2,055 images can be decoded.
- The dataset fingerprint is recorded in `metadata/data_audit.json`.
- The original Step 1 development/test assignment is retained only as metadata.

## Automated findings

- 7 exact duplicate groups contain 7 extra copies.
- 2 exact duplicate pairs cross the old development/test boundary.
- Conservative perceptual matching identifies 77 image-pair candidates in 47 groups; 11 of those pairs cross the old boundary.
- Filename-family matching adds related multi-view candidates such as numbered suffixes.
- In total, 143 multi-image candidate object groups contain 429 images.
- 13 candidate object groups cross the old development/test boundary.
- No candidate group crosses ornament labels under the current conservative rules.
- 44 images have an aspect ratio more extreme than 2:1 and 3 files exceed 10 MB. No image has a side shorter than 96 pixels.

Candidate groups intentionally favor leakage prevention. Perceptually similar pottery photographed against the same background is not necessarily the same physical object, so every multi-image candidate was reviewed visually.

## Completed manual review

The completed review is stored in `review/outputs/019f6af8-e0cf-7513-85b2-f6eb3920df46/phase_1_image_review.xlsx` and the machine-readable companion `phase_1_manual_review.json`. The Markdown index remains a clickable local fallback.

- All 143 multi-image candidate groups and all 429 images in those groups have a `complete` review status.
- The review resolves the 429 images into 210 confirmed object groups and 137 assigned source groups.
- Group decisions: 111 same physical object, 21 partitioned into multiple objects, 7 same source but different objects, and 4 unrelated false matches.
- All 13 groups that cross the old development/test boundary are resolved: 6 same object, 3 partitioned, 2 same source/different objects, and 2 false matches.
- 28 byte-identical or visually identical re-encoded photographs are marked `exclude_exact_duplicate`.
- 4 plain candlestick images have no usable visible motif and are marked `exclude_low_quality` for motif training.
- The remaining reviewed images comprise 206 primary views and 191 useful related views that must stay with their object/source group in any future split.
- Motif visibility among reviewed images: 406 clear, 12 partial, 7 small, and 4 not visible.
- No reviewed image showed clear visual evidence that its provided ornament label was wrong.

The review is intentionally visual. Provenance and licence remain unknown because neither can be established from pixels or filenames alone.

## Information that pixels and filenames cannot establish reliably

All 2,055 records remain explicitly inventoried for:

- physical object grouping;
- object type;
- motif visibility;
- original source/provenance;
- image licence.

The group-level and per-image fields for the 429 multi-image candidates are complete in the review outputs. The remaining 1,626 singleton images do not need duplicate/object-relationship adjudication, but their optional object-type, motif-visibility, provenance, and licence metadata have not been manually filled.

## Phase boundary

Phase 2 must not treat the old `dataset_test` directory as a sealed test set. Split generation can now consume the completed object/source decisions, while retaining conservative grouping for the 1,626 singleton images. No additional image collection is required for this work.
