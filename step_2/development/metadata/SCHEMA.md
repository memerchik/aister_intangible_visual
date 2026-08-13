# Phase 1 metadata schema

## Machine-derived fields

- `image_id`: stable identifier derived from ornament label and filename, independent of the current Step 1 split.
- `relative_path`: path relative to `step_02/data/`.
- `current_split`: original `dataset_dev` or `dataset_test` assignment.
- `ornament_label`: folder-provided class label.
- `content_sha256`: byte-level content digest.
- `dataset_fingerprint`: stored in `data_audit.json`; digest of all relative paths and file digests.
- `exact_duplicate_cluster_id`: populated only for byte-identical files.
- `perceptual_review_cluster_id`: populated for images joined by conservative hash-similarity rules.
- `filename_family`: class-scoped filename stem after removing a final numeric view suffix.
- `candidate_object_group_id`: conservative union of exact, perceptual, and filename-family hints. Every image receives a value; images without a detected relationship receive a unique singleton group.
- `quality_flags`: semicolon-separated automated warnings, or `ok`.

## Human-review fields

The audit does not fabricate these values from weak evidence:

- `source_group_id`: common website, collection, photographer, or acquisition batch.
- `object_group_id`: images of the same physical object or artwork.
- `object_type`: cup, plate, rug, painting, textile, figurine, and so on.
- `motif_visibility`: suggested controlled values are `clear`, `partial`, `small`, `obscured`, and `not_visible`.
- `provenance`: original source or collection reference.
- `license`: reuse terms for the image.
- `metadata_review_status`: starts as `needs_manual_review`.

`review_queue.csv` prioritizes unreadable files and cross-split duplicate candidates before ordinary metadata review.
