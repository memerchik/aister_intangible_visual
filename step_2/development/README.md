# Step 2 research and model development

This folder contains the audited development work that followed the Step 1
prototype. It includes the dataset copy, review records, leakage-controlled
splits, experiment code, tests, and saved results. The original prototype and
workshop remain unchanged in `step_1/`.

## Start here

- [`phases/README.md`](phases/README.md) summarizes the five development phases.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) explains the directory layout and the
  relocation from the old `step_02/` path.
- [`PHASE_1_FINDINGS.md`](PHASE_1_FINDINGS.md) covers the dataset audit.
- [`PHASE_2_FINDINGS.md`](PHASE_2_FINDINGS.md) describes the evaluation split.
- [`PHASE_3_FINDINGS.md`](PHASE_3_FINDINGS.md) reports the classical baseline.
- [`PHASE_4_FINDINGS.md`](PHASE_4_FINDINGS.md) reports the pretrained encoder
  comparison.
- [`PHASE_5_EVOLUTION.md`](PHASE_5_EVOLUTION.md) follows experiments v1–v5.
- [`PHASE_5_GATES.md`](PHASE_5_GATES.md) lists every promotion gate and result.
- [`../v0_5/README.md`](../v0_5/README.md) explains the standalone application.

Historical result files may still contain `step_02/...` paths. These strings
are preserved as experiment provenance and are mapped to the current files by
[`relocation_manifest.json`](relocation_manifest.json).

## Development summary

| Phase | Work completed | Main files |
|---|---|---|
| 1 — data audit | Inventoried 2,055 images, reviewed duplicate and related-image groups, and retained 2,023 usable records. | `metadata/`, `review/`, `PHASE_1_FINDINGS.md` |
| 2 — evaluation protocol | Created source-atomic splits that keep related acquisitions, objects, and identical images together. | `splits/`, `PHASE_2_FINDINGS.md` |
| 3 — classical baseline | Evaluated HSV, HOG, and LBP features on the corrected folds. | `outputs/phase_3_classical_baseline/` |
| 4 — pretrained encoders | Compared 13 frozen representation and view combinations. | `outputs/phase_4_pretrained/` |
| 5 — source robustness | Evaluated five nested procedures. V4 had the best aggregate accuracy; v5 was the most stable, but neither met every release gate. | `outputs/phase_5_source_robustness*/` |

## Dataset and evaluation

The current split is `step02_source_atomic_v3`, generated with seed `20260719`.

| Partition | Images | Use |
|---|---:|---|
| Train | 1,369 | Model fitting and a secondary validation diagnostic |
| Validation | 324 | Secondary diagnostic only |
| Development | 1,693 | Five-fold, source-blocked model selection and analysis |
| Sealed test | 330 | Reserved for one final evaluation of a fully frozen future system |

The five development folds contain 500, 244, 270, 342, and 337 images. Their
sizes differ because source groups are kept intact; the largest group alone
contains 261 Opishnyan images. [`splits/split_audit.json`](splits/split_audit.json)
records zero known crossings of source groups, reviewed semantic groups,
physical objects, or identical image bytes.

The split assignment fingerprint is
`59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`.
Earlier v1 and v2 splits are retained only as history because later visual
reviews found related images on opposite sides of their evaluation boundaries.
The original `dataset_dev` and `dataset_test` folder names also describe
provenance, not the current evaluation assignment.

## Results

| Experiment | Accuracy | Macro F1 | Balanced accuracy | Top-3 accuracy |
|---|---:|---:|---:|---:|
| Phase 3 `knn_cosine_k5` | 63.3196% | 59.2031% | 59.1724% | 87.9504% |
| Phase 4 `dinov3_vits16__cls__global_fivecrop` | 93.0892% | 92.8447% | 94.0378% | 99.5275% |
| Phase 5 v1 adaptive nested procedure | 93.9752% | 93.6135% | 94.3388% | **99.8819%** |
| Phase 5 v2 pairwise correction | 93.7980% | 93.4211% | 94.0571% | 99.8819% |
| Phase 5 v3 encoder consensus | 93.9161% | 93.5815% | 94.5147% | 99.8819% |
| Phase 5 v4 motif-localized procedure | **94.6840%** | **94.3622%** | **95.0256%** | — |
| Phase 5 v5 fixed motif consensus | 94.5068% | 94.1957% | 94.9226% | — |

V4 produced the strongest overall development result. V5 passed 18 of 21
promotion gates and was the most stable across source-held-out searches, but it
still missed the required Bubnivka precision, Opishnyan-to-Bubnivka error, and
worst-fold accuracy thresholds. No experiment was promoted.

These are development results, not an independent estimate of performance on
new sources. The same development data influenced candidate design and model
selection, the class scores are uncalibrated, and the test set has not been
evaluated. Most remaining errors are concentrated at the Opishnyan/Bubnivka
boundary. Detailed comparisons are available in the Phase 5 findings files.

The [`../v0_5/`](../v0_5/README.md) application uses the fixed v4
full-development recipe to provide a human-assisted preview. It is useful for
demonstration and expert feedback, but it is not a validated production model.

## Directory guide

| Path | Contents |
|---|---|
| `data/` | Verified copy of the available image dataset |
| `metadata/` | Image inventory, audit results, and review queue |
| `review/` | Manual adjudication and source-cohort evidence |
| `splits/` | Current partitions, folds, split policy, hashes, and audit |
| `scripts/` | Dataset builders and experiment runners |
| `tests/` | Integrity, isolation, determinism, and output checks |
| `outputs/` | Saved metrics, predictions, diagnostics, and experiment records |
| `src/` | Reusable modelling code used by the later experiments |
| `phases/` | Short phase summaries and links to the files above |

## Reproduce the work

Run commands from the repository root. Dataset and split generation use the
main development environment:

```bash
python3 step_2/development/scripts/build_manifest.py
python3 step_2/development/scripts/build_global_source_cohorts.py
python3 step_2/development/scripts/build_splits.py
python3 -m unittest discover -s step_2/development/tests -v
```

The classical baseline uses [`requirements.txt`](requirements.txt):

```bash
python3 step_2/development/scripts/run_classical_baseline.py
```

The pretrained encoder comparison uses
[`requirements-phase4.txt`](requirements-phase4.txt). The gated DINOv3 weights
must already be available locally.

```bash
python3 step_2/development/scripts/run_pretrained_embeddings.py
```

Phase 5 uses [`requirements-phase5.txt`](requirements-phase5.txt) and existing
development embedding caches:

```bash
MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_source_robustness.py

MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v2_pairwise.py

MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v3_consensus.py
```

V4 first builds a motif cache with
[`requirements-phase5-v4-extraction.txt`](requirements-phase5-v4-extraction.txt),
then runs the classifier:

```bash
PYTHONPATH=.cache/step_02/vision_runtime:.venv \
python3 step_2/development/scripts/run_phase5_v4_motif_extraction.py

MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v4_motif.py
```

The v5 fixed-consensus experiment uses the same Phase 5 environment:

```bash
MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v5_fixed_consensus.py
```

The sealed test set is intentionally absent from these workflows. Test IDs are
used only by integrity and disjointness checks; no test images are embedded or
scored during development.

## Current limitations

Additional tuning on the repeatedly reviewed development records would make
the reported results less independent. A new modelling cycle will therefore
need more source-diverse images, especially around the Opishnyan/Bubnivka
distinction. That cycle should receive a new dataset and split version, gates
defined before fitting, and a new untouched final evaluation partition.

Image provenance and redistribution rights are not yet resolved. The selected
`facebook/dinov3-vits16-pretrain-lvd1689m` checkpoint is gated and distributed
under the custom DINOv3 licence. Both the dataset and model licence require
review before a general public release.
