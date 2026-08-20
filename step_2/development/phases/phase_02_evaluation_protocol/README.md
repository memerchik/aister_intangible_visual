# Phase 2: evaluation protocol

Phase 2 converts the Phase 1 inventory and review decisions into the current
`step02_source_atomic_v3` development and sealed-test split. Classifier training
starts in Phase 3.

## Results

- The 2,023 usable images are assigned to 1,369 train, 324 validation, and 330
  sealed-test records.
- The 1,693 development images use five source-blocked folds containing
  500, 244, 270, 342, and 337 images.
- 28 visually reviewed acquisition cohorts are incorporated into 1,051
  indivisible source-atomic groups.
- No known source, semantic, object, or identical-byte group crosses a
  production split or development fold.
- The deterministic assignment seed is `20260719`.

The split is intentionally uneven because the largest acquisition cohort has
261 images and cannot be divided without leaking its source signature.

## Files and findings

- [Phase 2 findings](../../PHASE_2_FINDINGS.md)
- [Split files and usage notes](../../splits/README.md)
- [Split audit](../../splits/split_audit.json)
- [Complete split manifest](../../splits/split_manifest.csv)
- [Development folds](../../splits/development.csv)
- [Pretrained-similarity review](../../review/pretrained_similarity/audit.json)
- [Global source-cohort review](../../review/global_source_cohorts/README.md)
- [Global source decisions](../../review/global_source_cohorts/adjudication.json)

## Code and tests

- Similarity audit: [`audit_pretrained_similarity.py`](../../scripts/audit_pretrained_similarity.py)
- Source-cohort projection: [`build_global_source_cohorts.py`](../../scripts/build_global_source_cohorts.py)
- Split builder: [`build_splits.py`](../../scripts/build_splits.py)
- Split tests: [`test_phase_2_splits.py`](../../tests/test_phase_2_splits.py)
- Similarity-review tests: [`test_pretrained_similarity_audit_outputs.py`](../../tests/test_pretrained_similarity_audit_outputs.py)

Split versions v1 and v2 are retained as leakage-correction history and are not
valid benchmarks. V3 is the current evaluation split. Its 330-image test
partition has not been evaluated.
