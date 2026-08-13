# AISTER Ukrainian ornament classifier

This repository identifies five Ukrainian ornament traditions from images. It
now has three clear boundaries:

| Folder | Purpose |
|---|---|
| [`step_1/`](step_1/) | Preserved prototype and workshop materials |
| [`step_2/development/`](step_2/development/README.md) | Complete leakage-audited research history, datasets, phase outputs, scripts, and tests |
| [`step_2/v0_5/`](step_2/v0_5/README.md) | Self-contained provisional web application and Render deployment root |

Start with [`step_2/README.md`](step_2/README.md) for the short project map.
Before a public commit, run the
[`public-release safety check`](PUBLIC_RELEASE_CHECKLIST.md).

## Supported labels

1. `01_opishnyan_ceramics`
2. `02_ornek`
3. `03_bubnivka_ceramics`
4. `04_petrykivka_painting`
5. `05_kosiv_ceramics`

## Current result

The audited Step 2 dataset contains 2,055 images, with 2,023 retained after
review. Phase 5 ran five frozen development experiments. V4 produced the best
aggregate source-blocked development result at 94.6840% accuracy; v5 was the
most stable and passed 18 of 21 promotion gates. None passed every gate, so no
candidate advanced to calibration or the one-shot sealed evaluation. The
330-image test partition remains sealed.

The separate v0.5 application packages the fixed v4 full-development recipe as
a transparent, human-assisted showcase. It returns an immediate best visual
match, all five uncalibrated ranking scores, and six motif regions. It does not
claim production accuracy, calibrated confidence, unknown-class detection, or
formal Phase 6 status.

## Run the showcase locally

Install the local dependencies and start the app from the repository root:

```bash
python3 -m pip install -r step_2/v0_5/requirements-local.txt
python3 step_2/v0_5/server.py --open
```

The server verifies the exact gated DINOv3 weight before loading. See the
[`v0.5 setup guide`](step_2/v0_5/README.md) for local weight options and the
prepared Render free-tier deployment.

## Research navigation

- [`development/README.md`](step_2/development/README.md) — results and reproduction commands
- [`development/phases/README.md`](step_2/development/phases/README.md) — phase map
- [`development/ARCHITECTURE.md`](step_2/development/ARCHITECTURE.md) — ownership and relocation rules
- [`development/PHASE_5_EVOLUTION.md`](step_2/development/PHASE_5_EVOLUTION.md) — v1–v5 record
- [`development/PHASE_5_GATES.md`](step_2/development/PHASE_5_GATES.md) — all promotion gates

## Evaluation rules

- Treat the original `dataset_dev` and `dataset_test` names as provenance only.
- Develop only from `step_2/development/splits/development.csv` and its
  source-blocked folds.
- Do not embed, predict, calibrate, select, or inspect errors using
  `step_2/development/splits/test.csv` before the full future pipeline is
  frozen.
- Keep acquisition-source, semantic-source, physical-object, and identical-byte
  groups on one side of every evaluation boundary.
- Preserve frozen Phase 1–5 evidence. Historical `step_02/...` strings inside
  those artifacts are intentional provenance and resolve through
  [`relocation_manifest.json`](step_2/development/relocation_manifest.json).

Dataset-image provenance and licensing remain unresolved. The chosen DINOv3
checkpoint is gated and uses a custom licence, so model and data rights require
review before any public release beyond a controlled research showcase.
