# Step 2 development — production-development track

This workspace turns the Step 1 prototype into a leakage-audited ornament classifier using only the images already available in the repository. `step_1/` remains intact as the historical baseline and workshop; `step_2/development/` is the source of truth for all current development claims.

## Navigation

- [`phases/README.md`](phases/README.md) — phase-by-phase index and artifact manifests;
- [`ARCHITECTURE.md`](ARCHITECTURE.md) — canonical paths, ownership, and compatibility rules;
- [`PHASE_1_FINDINGS.md`](PHASE_1_FINDINGS.md) — data inventory and visual review;
- [`PHASE_2_FINDINGS.md`](PHASE_2_FINDINGS.md) — source-atomic evaluation protocol;
- [`PHASE_3_FINDINGS.md`](PHASE_3_FINDINGS.md) — corrected classical baseline;
- [`PHASE_4_FINDINGS.md`](PHASE_4_FINDINGS.md) — pretrained representation screen;
- [`PHASE_5_FINDINGS.md`](PHASE_5_FINDINGS.md) — Phase 5 v1 nested source-robustness result;
- [`PHASE_5_V2_FINDINGS.md`](PHASE_5_V2_FINDINGS.md) — Phase 5 v2 pairwise correction result;
- [`PHASE_5_V3_FINDINGS.md`](PHASE_5_V3_FINDINGS.md) — Phase 5 v3 cross-encoder consensus result;
- [`PHASE_5_V4_FINDINGS.md`](PHASE_5_V4_FINDINGS.md) — Phase 5 v4 motif-localization result;
- [`PHASE_5_V5_FINDINGS.md`](PHASE_5_V5_FINDINGS.md) — final Phase 5 result and data-growth guidance;
- [`PHASE_5_EVOLUTION.md`](PHASE_5_EVOLUTION.md) — chronological v1–v5 steps and conclusions;
- [`PHASE_5_GATES.md`](PHASE_5_GATES.md) — complete promotion-gate matrix for v1–v5;
- [`../v0_5/README.md`](../v0_5/README.md) — self-contained provisional human-assisted application ([findings](../v0_5/FINDINGS.md));
- [`splits/README.md`](splits/README.md) — current evaluation contract.

The `phases/` hierarchy is a navigation layer. It does not duplicate datasets,
review evidence, split manifests, scripts, or outputs. The complete development
workspace moved from historical `step_02/` to `step_2/development/`; frozen
records retain their original path strings, while `relocation_manifest.json`
binds those hashes to the current physical files.

## Current status

Phases 1–5 are complete at the development-experiment level:

| Phase | Outcome | Canonical evidence |
|---|---|---|
| 01 — data truth | Copied and verified the dataset, inventoried 2,055 images, reviewed duplicate and multi-image relationships, and retained 2,023 usable records. | `metadata/`, `review/`, `PHASE_1_FINDINGS.md` |
| 02 — evaluation protocol | Built source-atomic v3 partitions that keep acquisition source, semantic source, physical object, and identical image bytes together. | `splits/`, `PHASE_2_FINDINGS.md` |
| 03 — classical baseline | Reran the frozen HSV/HOG/LBP reference on the corrected development folds. | `outputs/phase_3_classical_baseline/`, `PHASE_3_FINDINGS.md` |
| 04 — pretrained screen | Compared 13 frozen representation/view candidates on the same source-blocked folds. | `outputs/phase_4_pretrained/`, `PHASE_4_FINDINGS.md` |
| 05 — source robustness | Ran frozen v1–v5 nested experiments. V4 is strongest overall; v5 is most stable/source-robust but failed 3 of 21 gates. All were rejected and modelling is closed pending new data. | `outputs/phase_5_source_robustness*/`, `PHASE_5_EVOLUTION.md` |

The 330-image source-atomic test set remains sealed. There is still no promoted
or calibrated production model, confidence policy, unknown-class rejection
threshold, or unbiased final evaluation. A separate
[`../v0_5/`](../v0_5/README.md) productization track now
packages the fixed v4 full-development recipe as an explicitly provisional
human-assisted preview. It does not open Phase 6 or change any Phase 5 result.

## Evaluation source of truth

The canonical split version is `step02_source_atomic_v3`, built deterministically with seed `20260719`.

| Partition | Images | Permitted use |
|---|---:|---|
| Train | 1,369 | Model development and the secondary fixed-validation diagnostic |
| Validation | 324 | Secondary diagnostic only; it is source-heavy and must not determine the final claim alone |
| Development | 1,693 | Source-blocked five-fold selection and analysis |
| Sealed test | 330 | One future evaluation after the complete pipeline is frozen |

Development folds contain 500, 244, 270, 342, and 337 images. Their unequal sizes are intentional: the largest indivisible source cohort contains 261 Opishnyan images. `splits/split_audit.json` records zero known production or CV violations for source-atomic groups, adjudicated semantic source/object groups, confirmed physical objects, and included content hashes.

The assignment fingerprint is `59718f69dc0b3f6c2eeee2b372d3e5381b48aa957e15c0128abbdda3719f9f4b`.

Earlier v1 and v2 splits were superseded when progressively broader visual audits identified same-object, same-source, and same-acquisition relationships crossing their boundaries. Scores produced under those protocols are historical diagnostics, not valid current estimates. The original `dataset_dev`/`dataset_test` folder assignment is provenance only.

## Development results

| Candidate | Accuracy | Macro F1 | Balanced accuracy | Top-3 accuracy |
|---|---:|---:|---:|---:|
| Phase 3 `knn_cosine_k5` | 63.3196% | 59.2031% | 59.1724% | 87.9504% |
| Phase 4 `dinov3_vits16__cls__global_fivecrop` | 93.0892% | 92.8447% | 94.0378% | 99.5275% |
| Phase 5 v1 adaptive nested procedure | 93.9752% | 93.6135% | 94.3388% | **99.8819%** |
| Phase 5 v2 pairwise nested procedure | 93.7980% | 93.4211% | 94.0571% | 99.8819% |
| Phase 5 v3 cross-encoder consensus | 93.9161% | 93.5815% | 94.5147% | 99.8819% |
| Phase 5 v4 motif-localized procedure | **94.6840%** | **94.3622%** | **95.0256%** | — |
| Phase 5 v5 fixed motif consensus | 94.5068% | 94.1957% | 94.9226% | — |

The Phase 5 adaptive outer procedure makes 1,591 correct and 102 incorrect
predictions. Relative to Phase 4, it fixes 26 errors and introduces 11, for a
net gain of 15 correct images. It raises Opishnyan recall from 83.4331% to
87.4251%, hard-three Opishnyan accuracy from 64.6739% to 73.9130%, and the
source-robustness score from 64.4664% to 71.0883%.

The v1 full-development selection chose DINOv3 global-fivecrop CLS features
with `C=10`, source exponent `0.5`, and no specialist. Its
selection-conditional OOF accuracy is 93.5617%. V2 added a frozen pairwise
correction search. Its adaptive nested accuracy fell to 93.7980%, Bubnivka
recall fell to 88.7324%, the 45 Opishnyan-to-Bubnivka errors were unchanged,
and the full-development correction recipe was stable in 0 of 5 outer
searches. Neither candidate is promoted.

V3 tested independent DINOv2, ConvNeXt, CLIP, MobileNetV3, and DINOv3
evidence. It recovered Bubnivka recall to 92.0188%, but Opishnyan recall fell
to 86.4271%, Opishnyan-to-Bubnivka errors rose to 50, source robustness fell
to 68.2122%, and the full fusion recipe was stable in 0 of 5 outer searches.
V3 failed 9 of 21 gates and is also rejected. Calibration and packaging remain
blocked.

V4 tested label-free motif-localized DINOv3 descriptors. Its adaptive nested
procedure improved to 94.6840% accuracy, 94.3622% macro F1, and 74.2219%
source robustness, with 90 total errors. It is the first iteration to pass the
Opishnyan-recall and ceramic-macro-F1 gates. It still failed Bubnivka precision
(81.0924%), Opishnyan-to-Bubnivka errors (43), Bubnivka recall (90.6103%), and
motif stability (2/5). V4 is therefore also rejected and Phase 6 remains
blocked.

V5 removed descriptor selection and always averaged the base plus three motif
heads. It passed 18 of 21 gates, recovered Bubnivka recall to 91.0798%, was
stable in all 5 outer searches, and raised source robustness to 76.8884%. It
still failed Bubnivka precision (79.8354%), Opishnyan-to-Bubnivka errors (46),
and worst-fold accuracy (86.8421%). It is rejected, and no sixth experiment is
planned on the repeatedly inspected development set.

These remain **development-informed metrics, not an unbiased production
estimate**. Phase 4 diagnostics shaped the candidate family, and full-development
OOF results selected the final recipe. Current probability values are
uncalibrated ranking scores and cannot support an acceptance or rejection
threshold.

The strongest v4 error is concentrated rather than evenly distributed:

- Opishnyan recall is 88.4232% and now passes its gate;
- 43 of 90 errors are Opishnyan → Bubnivka, three above the gate;
- Bubnivka recall is 90.6103%, 0.3497 points below its floor;
- Bubnivka precision is 81.0924%, 0.9076 points below its gate;
- mixed displays/collections remain difficult in the limited reviewed object-type slice.

V2 did not resolve this concentration. It changed 18 v1 nested decisions,
fixing 7 and regressing 10. Bubnivka-to-Opishnyan errors rose from 17 to 20,
and five different correction recipes were selected across the five outer
searches.

V4 and v5 clear the 90% goal only as development results. They do not establish
90% on unseen sources or in production.

## Canonical artifacts

Shared artifacts remain outside individual phase folders because later phases consume them:

| Path | Role |
|---|---|
| `data/` | Verified Step 02 copy of the available image dataset |
| `metadata/` | Base manifest, class summary, duplicate candidates, audit, and review queue |
| `review/` | Human visual adjudication, similarity review, and 28 global source cohorts |
| `splits/` | Canonical source-atomic partitions, folds, policy, hashes, and audit |
| `scripts/` | Reproducible phase runners retained at stable paths |
| `tests/` | Integrity, isolation, determinism, and output-contract checks |
| `outputs/phase_3_classical_baseline/` | Canonical Phase 3 evidence |
| `outputs/phase_4_pretrained/` | Canonical full Phase 4 evidence |
| `outputs/phase_4_pretrained_quick/` | Smoke-run diagnostic; not the canonical Phase 4 report |
| `outputs/phase_5_source_robustness/` | Preserved canonical Phase 5 v1 search, predictions, diagnostics, and rejection evidence |
| `outputs/phase_5_source_robustness_v2/` | Canonical Phase 5 v2 pairwise search, predictions, diagnostics, and rejection evidence |
| `outputs/phase_5_source_robustness_v3/` | Canonical Phase 5 v3 consensus search, predictions, diagnostics, and rejection evidence |
| `outputs/phase_5_source_robustness_v4/` | Canonical Phase 5 v4 motif-localized search, predictions, diagnostics, and rejection evidence |
| `outputs/phase_5_source_robustness_v5/` | Canonical final Phase 5 fixed-consensus predictions, diagnostics, and rejection evidence |
| `src/` | Reusable production-oriented code from Phase 5 onward |
| `../v0_5/` | Self-contained provisional UI, service, packaged fixed-v4 model, product contract, and opt-in contribution quarantine; not formal Phase 6 evidence |

See [`ARCHITECTURE.md`](ARCHITECTURE.md) before adding or relocating files.

## Rebuild and validate

Run Phase 1–2 integrity generation from the repository root:

```bash
python3 step_2/development/scripts/build_manifest.py
python3 step_2/development/scripts/build_global_source_cohorts.py
python3 step_2/development/scripts/build_splits.py
python3 -m unittest discover -s step_2/development/tests -v
```

Rebuild the classical reference after installing `step_2/development/requirements.txt`:

```bash
python3 step_2/development/scripts/run_classical_baseline.py
```

Phase 4 has a separately pinned environment in `requirements-phase4.txt`. The required pretrained weights must already be locally available; DINOv3 access is gated.

```bash
python3 step_2/development/scripts/run_pretrained_embeddings.py
```

Phase 5 reuses only the allowlisted cached development embeddings and has a
smaller pinned environment in `requirements-phase5.txt`:

```bash
MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_source_robustness.py
```

Run the separately frozen v2 correction experiment with the same environment:

```bash
MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v2_pairwise.py
```

Run the frozen v3 cross-encoder consensus experiment with:

```bash
MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v3_consensus.py
```

V4 has two separately frozen stages. First build the development-only motif
cache with `requirements-phase5-v4-extraction.txt`, then run the pinned Phase 5
classifier environment:

```bash
PYTHONPATH=.cache/step_02/vision_runtime:.venv \
python3 step_2/development/scripts/run_phase5_v4_motif_extraction.py

MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v4_motif.py
```

Run the final frozen v5 fixed consensus with the Phase 5 environment:

```bash
MPLCONFIGDIR=.cache/matplotlib \
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/development/scripts/run_phase5_v5_fixed_consensus.py
```

Run the provisional assisted v0.5 application locally with the already pinned
vision environment:

```bash
PYTHONPYCACHEPREFIX=.cache/pycache \
/usr/bin/python3 step_2/v0_5/server.py --open
```

The packaged model can be rebuilt deterministically from development-only
caches with `step_2/development/scripts/build_assisted_v0_5_model.py`. Ordinary
predictions are memory-only. Images are written to a quarantined review queue
only after explicit contribution consent, and no contribution is automatically
eligible for training.

Do not run a test-evaluation command during development. Reading test identifiers for integrity and disjointness checks is permitted; embedding test images, fitting against them, or producing test predictions is not.

## Next phase boundary

Phase 5 v1–v5 are complete and rejected. Do not proceed to Phase 6 calibration
or the sealed evaluation. Bubnivka precision and Opishnyan-to-Bubnivka errors
failed every iteration. A sixth search on the same repeatedly inspected rows
would not provide independent evidence, so modelling is closed until
materially more source-diverse data is available.

When the dataset grows, create a new version, rebuild source-atomic folds,
refreeze gates before fitting, and reserve a new sealed evaluation partition.
Do not target individual current error IDs or relax an observed gate.

Only an accepted, fully frozen future candidate can move through the formal
calibration, sealed-test, and production-release path. The v0.5 assisted
interface is intentionally usable before then: it returns an immediate visual
suggestion, all five uncalibrated ranking scores, motif-region visualization,
and a non-blocking Opishnyan/Bubnivka comparison warning. Its engineering can
be reused later, but none of its product behavior promotes v4 or validates a
production accuracy claim.

Dataset provenance and image licences remain unresolved. The selected `facebook/dinov3-vits16-pretrain-lvd1689m` checkpoint is gated and governed by the custom DINOv3 licence. Legal review of both the dataset and any chosen model dependency is required before deployment.
