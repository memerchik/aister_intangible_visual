# Phase 02 — source-atomic evaluation protocol

**Status: complete.** Phase 2 converts the Phase 1 inventory and reviews into
the current `step02_source_atomic_v3` development and sealed-test contract. It
does not train or select a classifier.

## Outcome

- 2,023 usable images are allocated as 1,369 train, 324 validation, and 330
  sealed test records.
- The 1,693 development images use five source-blocked folds of
  500/244/270/342/337 images.
- 28 visually reviewed acquisition cohorts are incorporated into 1,051
  indivisible source-atomic groups.
- Known source, semantic, object, and byte-identity leakage is zero across both
  production and development-fold boundaries.
- The deterministic assignment seed is `20260719`.

The unequal partitions are intentional: the largest acquisition cohort has 261
images and cannot be divided without knowingly leaking its source signature.

## Canonical evidence

- [Phase 2 findings](../../PHASE_2_FINDINGS.md)
- [Split contract and usage rules](../../splits/README.md)
- [Split audit](../../splits/split_audit.json)
- [Complete split manifest](../../splits/split_manifest.csv)
- [Development folds](../../splits/development.csv)
- [Pretrained-similarity adjudication](../../review/pretrained_similarity/audit.json)
- [Global source-cohort review](../../review/global_source_cohorts/README.md)
- [Global source adjudication](../../review/global_source_cohorts/adjudication.json)

## Rebuild surface

- Similarity audit: [`audit_pretrained_similarity.py`](../../scripts/audit_pretrained_similarity.py)
- Source-cohort projection: [`build_global_source_cohorts.py`](../../scripts/build_global_source_cohorts.py)
- Split builder: [`build_splits.py`](../../scripts/build_splits.py)
- Split tests: [`test_phase_2_splits.py`](../../tests/test_phase_2_splits.py)
- Similarity-audit tests: [`test_pretrained_similarity_audit_outputs.py`](../../tests/test_pretrained_similarity_audit_outputs.py)

## Boundary

Split versions v1 and v2 are superseded leakage-correction history and are not
benchmarks. V3 is the only current contract. The 330-image test partition is
sealed and has not been evaluated.
