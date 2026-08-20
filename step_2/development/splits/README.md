# Dataset splits and evaluation protocol

This directory defines the current Step 2 dataset assignments. Image paths in
the CSV files are relative to [`../data/`](../data/README.md).

## Current split

The current version is `step02_source_atomic_v3`, generated with seed
`20260719`.

- `train.csv`: 1,369 images used to fit model parameters.
- `validation.csv`: 324 images used for secondary development diagnostics.
- `development.csv`: all 1,693 non-test images, with source-blocked `cv_fold`
  values from 0 to 4.
- `test.csv`: 330 images reserved for one final evaluation after the model and
  decision policy are frozen.
- `split_manifest.csv`: all 2,055 inventory records, including the 32 excluded
  images and their reasons.

The five development folds contain 500, 244, 270, 342, and 337 images. Each
class appears in every partition and every fold. The sizes are intentionally
uneven because related images remain together; one Opishnyan acquisition cohort
alone contains 261 images. Reports therefore include both aggregate out-of-fold
metrics and the unweighted mean and variation across folds.

## Leakage controls

Related records are joined into one assignment group using, in order:

1. 28 visually reviewed acquisition-source cohorts;
2. 46 visually reviewed pretrained-similarity components;
3. manually confirmed source groups;
4. manually confirmed physical-object groups;
5. identical included image bytes.

The final split contains 1,051 indivisible groups. No reviewed acquisition
cohort, semantic group, source group, physical object, or included content hash
crosses a production split or a development fold.

The 32 excluded images are kept in `split_manifest.csv` without an assignment:

- 28 exact or visually identical copies;
- 4 images without a usable visible motif.

Source cohorts are inferred from repeated watermarks, backdrops, camera/export
signatures, and coherent photography sessions. Images without a cohort are not
proven to have independent provenance. A shared source also says nothing about
whether an image may legally be redistributed.

## Earlier split versions

`step02_grouped_v1` was replaced after an embedding review found missed
same-object and same-source relationships. `step02_semantic_grouped_v2` added
those relationships but was replaced after 28 broader acquisition cohorts were
found to cross its partitions. Results based on those earlier test partitions
are not current benchmarks.

The v3 assignment fingerprint is:

`59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`

## Using the split

- Treat `old_split` and superseded Step 2 assignments as provenance only.
- Use the train, validation, and five development folds for model development.
- Keep the source and group columns when creating experiment data.
- Report aggregate out-of-fold metrics and the unweighted fold mean and
  standard deviation.
- Report macro F1, balanced accuracy, per-class recall, confusion matrix, and
  top-k accuracy alongside top-1 accuracy.
- Keep the test set out of embeddings, predictions, error analysis, selection,
  calibration, and threshold design until the final system is frozen.

The v3 test set has not been evaluated.

## Rebuild and validate

Run from the repository root:

```bash
python3 step_2/development/scripts/build_global_source_cohorts.py
python3 step_2/development/scripts/build_splits.py
python3 -m unittest discover -s step_2/development/tests -v
```

`split_audit.json` records input fingerprints, observed allocations, CSV hashes,
group counts, and the leakage checks performed by the builder.
