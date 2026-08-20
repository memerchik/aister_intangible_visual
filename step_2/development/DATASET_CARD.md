# Dataset card

## Overview

The Step 2 dataset is a research collection for five-way visual classification
of Ukrainian ornament traditions. Images show ornaments on varied carriers,
including ceramics, textiles, paintings, and photographed objects. The labels
are operational folder-level classes used by this project:

1. `01_opishnyan_ceramics`
2. `02_ornek`
3. `03_bubnivka_ceramics`
4. `04_petrykivka_painting`
5. `05_kosiv_ceramics`

The verified copy used by Step 2 is under [`data/`](data/). The original Step 1
copy remains under [`../../step_1/`](../../step_1/).

## Composition

The image audit found 2,055 supported, readable files. Review retained 2,023
images and excluded 32: 28 exact or re-encoded duplicate copies and four
images without a usable visible motif.

| Label | Audited files | Retained images | Train | Validation | Sealed test |
|---|---:|---:|---:|---:|---:|
| Opishnyan ceramics | 604 | 589 | 397 | 104 | 88 |
| Ornek | 516 | 516 | 361 | 78 | 77 |
| Bubnivka ceramics | 281 | 280 | 167 | 46 | 67 |
| Petrykivka painting | 307 | 306 | 214 | 46 | 46 |
| Kosiv ceramics | 347 | 332 | 230 | 50 | 52 |
| **Total** | **2,055** | **2,023** | **1,369** | **324** | **330** |

The canonical counts are recorded in
[`metadata/data_audit.json`](metadata/data_audit.json) and
[`splits/split_audit.json`](splits/split_audit.json). Individual records,
image properties, hashes, review fields, and operational labels are in
[`metadata/manifest.csv`](metadata/manifest.csv).

## Review and grouping

Phase 1 reviewed 143 candidate multi-image groups covering 429 images. The
review produced 210 confirmed object groups and 137 confirmed source groups.
A later global source review identified 28 same-acquisition cohorts directly
covering 982 images; propagation through reviewed relationships assigned 993
images to those cohorts.

The current split policy treats identical bytes, confirmed physical objects,
reviewed semantic sources, and global acquisition-source cohorts as atomic.
This produces 1,051 source-atomic groups. The largest contains 261 images, so
source integrity takes priority over evenly sized folds.

Review evidence is stored under [`review/`](review/), and the full split audit
is in [`splits/split_audit.json`](splits/split_audit.json).

## Evaluation partitions

The current assignment is `step02_source_atomic_v3`, generated with seed
`20260719`.

| Evaluation set | Images | Role |
|---|---:|---|
| Development | 1,693 | Five-fold, source-blocked model selection and analysis |
| Fold 0 | 500 | One source-atomic development fold |
| Fold 1 | 244 | One source-atomic development fold |
| Fold 2 | 270 | One source-atomic development fold |
| Fold 3 | 342 | One source-atomic development fold |
| Fold 4 | 337 | One source-atomic development fold |
| Sealed test | 330 | Reserved for one evaluation of a fully frozen future system |

The development set is the union of the 1,369-image train partition and the
324-image validation partition. The original `dataset_dev` and `dataset_test`
folder names are provenance fields only; they do not define the current
evaluation boundary.

The assignment fingerprint is
`59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`.
The split audit reports no known crossings of source-atomic groups, semantic
groups, confirmed objects, or identical content across the production or
development-fold boundaries.

## Labels and annotations

Each retained image has exactly one of the five operational class labels. The
dataset does not contain a negative or unknown class, multilabel tradition
annotations, motif-part annotations, historical-period labels, or complete
object metadata. Object type, motif visibility, provenance, and licence fields
were added during review where evidence was available; missing values remain
explicit rather than inferred.

The labels support the current flat classification task. They do not by
themselves establish that every image has only one possible cultural,
historical, geographic, or stylistic interpretation.

## Provenance and rights

The original image provenance and redistribution licence are unresolved for
all 2,055 audited files. This repository therefore does not present the image
collection as a generally redistributable public dataset. Anyone sharing,
publishing, or deploying the images must complete a separate provenance,
consent, and rights review.

The model weights used by the later experiments are separate from the image
dataset. The selected DINOv3 checkpoint is gated and distributed under its own
custom licence.

## Intended use

Appropriate uses include:

- reproducing the recorded research experiments in a controlled environment;
- studying source-aware evaluation and error patterns within these five
  operational labels;
- comparing methods on the frozen development folds without accessing the
  sealed test set;
- supporting human review and collecting structured feedback through the v0.5
  demonstration application.

The dataset and current models are not validated for autonomous cultural
attribution, open-world recognition, population-level performance claims,
rights decisions, or other high-stakes uses. Results on the development set do
not establish performance on new sources or objects.

## Maintenance

New data should receive a new dataset and split version rather than being
inserted into the existing frozen evaluation records. Additions require the
same readability checks, duplicate detection, semantic and source grouping,
rights review, and source-atomic partitioning used here. The 330-image test set
must remain untouched until the model, confidence policy, and evaluation plan
are fully frozen.

