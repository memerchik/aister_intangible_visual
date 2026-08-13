# Phase 2 findings - source-atomic evaluation

Phase 2 converted the completed visual review into a reproducible evaluation substrate. It did not train or select a model. The current `step02_source_atomic_v3` test set remains sealed.

## Why v1 and v2 were replaced

The original `step02_grouped_v1` split isolated the relationships known from Phase 1, but a later pretrained-similarity audit exposed missed same-object and same-source links across production and CV boundaries. All 2,023 included images were screened; 46 candidate components involving 118 images were visually adjudicated. The resulting semantic unions produced `step02_semantic_grouped_v2`.

V2 still treated many visually obvious acquisition batches as independent. A full-inventory contact-sheet review found 28 conservative same-acquisition-source cohorts directly containing 982 images, or 48.54% of the usable inventory. Every cohort crossed a v2 production boundary and a CV/evaluation boundary; 25 touched the then-test partition. V2 was therefore also invalidated and resealed rather than used for performance reporting.

The v3 splitter unions the global cohorts with the semantic, manually confirmed source, and physical-object relationships. Propagation through those prior groups places 993 images in the source-atomic cohort membership. Directly unassigned images are not claimed to have independent provenance.

## V3 result

- 2,023 of 2,055 images are usable for modelling.
- 32 records are excluded: 28 duplicate or re-encoded photographs and 4 images without a usable visible motif.
- The production split contains 1,369 train, 324 validation, and 330 sealed-test images.
- The 1,693 development images have source-blocked CV folds of 500, 244, 270, 342, and 337 images.
- All five ornament classes occur in every production split and every CV fold.
- There are 1,051 source-atomic allocation groups; the largest contains 261 images.

## Balance trade-off

The 70/15/15 targets for 2,023 included images are approximately 1,416/304/303. V3 differs by -47/+20/+27 images because acquisition cohorts are indivisible.

Most class allocations remain close to target. The main production imbalance is Bubnivka: 167/46/67 instead of approximately 196/42/42. Opishnyan is 397/104/88 instead of approximately 412/89/88. The other three classes differ by at most two images from their per-class targets.

CV imbalance is larger because the 261-image Opishnyan roundmark catalogue must occupy one fold. Fold 0 contains 500 images, including 261 Opishnyan examples; folds 1 and 2 contain only 5 and 31 Opishnyan examples. Every other class is allocated almost equally across folds. Keeping the watermark/source batch intact is more important than presenting artificially even folds.

Both aggregate OOF metrics and unweighted mean-plus-standard-deviation fold metrics must therefore be reported. Neither alone describes this grouped evaluation adequately.

## Leakage audit

The v3 assignment keeps the following relationships on exactly one side of every production and development-CV boundary:

- source-atomic split groups and global source cohorts;
- visually adjudicated semantic source/object groups;
- manually confirmed source groups and physical-object groups;
- included content hashes.

The canonical audit reports zero violations for all of those controls in both production and CV. The old `dataset_dev`/`dataset_test` folders were ignored during assignment and remain provenance only.

## Reproducibility

The allocator is deterministic at seed `20260719`. Its version and assignment fingerprint are:

- version: `step02_source_atomic_v3`
- fingerprint: `59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`

`split_audit.json` records exact CSV hashes, target and observed allocations, review fingerprints, and every leakage check. The automated suite verifies inventory integrity, exclusions, review inputs, exact partition membership, class coverage, group/hash isolation, and deterministic reconstruction.

## Interpretation

Scores from the original folder split, Step 02 v1, and Step 02 v2 are not comparable final benchmarks: each predates a leakage correction. V3 is the first protocol that blocks the repeated acquisition signatures identified by the full-inventory audit.

The source labels remain visually inferred rather than externally proven. Large within-class source cohorts also mean this dataset cannot simultaneously provide strict source isolation, exact 70/15/15 balance, and equal five-fold class counts. Phase 3 and Phase 4 correctly preserve the source boundary and disclose the resulting imbalance. The v3 sealed test has not been opened for model evaluation.
