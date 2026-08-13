# Step 2 development architecture and path contract

This document defines the research workspace after it moved from `step_02/` to
`step_2/development/`. The move separates research from the deployable
`step_2/v0_5/` app. Completed evidence keeps its original recorded path and
hash; a relocation adapter resolves those historical references.

## Design principles

1. **One canonical copy per artifact.** A phase README links to shared data, review evidence, splits, scripts, and outputs; it does not copy them.
2. **Stable completed evidence.** Phase 1–5 artifact contents, including all five Phase 5 iterations, remain unchanged. Historical paths resolve through `relocation_manifest.json` rather than being rewritten inside frozen evidence.
3. **Shared truth stays shared.** `metadata/`, `review/`, and `splits/` are inputs to several phases and therefore do not belong inside a single phase directory.
4. **The sealed test is a contract.** Development code may verify test membership and isolation, but must not embed test images or produce test predictions before the pipeline is frozen.
5. **Experiment runners are separated from reusable code.** `scripts/` preserves reproducible phase entry points. Production-oriented components belong in `src/` from Phase 5 onward.
6. **Structure communicates status.** A phase directory can exist as a plan without implying that its experiments, artifacts, or claims are complete.

## Directory roles

```text
step_2/
├── development/
│   ├── README.md                       # research status and operating rules
│   ├── ARCHITECTURE.md                 # this ownership contract
│   ├── relocation_manifest.json        # old-path/current-file hash bridge
│   ├── PHASE_*_FINDINGS.md             # completed narrative reports
│   ├── phases/                          # phase navigation and contracts
│   ├── data/                            # protected Step 2 dataset copy
│   ├── metadata/                        # reproducible base inventory
│   ├── review/                          # visual adjudication evidence
│   ├── splits/                          # canonical evaluation contract
│   ├── scripts/                         # reproducible phase runners
│   ├── tests/                           # integrity and output contracts
│   ├── outputs/                         # frozen experiment evidence
│   └── src/ornament_classifier/         # reusable development code
└── v0_5/                                # standalone assisted application
```

The `phases/` folders are not execution roots. Commands continue to run from the repository root and use the canonical paths documented below.

## Phase-to-artifact map

| Phase | Reads | Canonical writes/evidence | Runner or implementation |
|---|---|---|---|
| 01 — data truth | `data/` | `metadata/`, manual-review material under `review/`, `PHASE_1_FINDINGS.md` | `scripts/build_manifest.py` plus completed visual adjudication |
| 02 — evaluation protocol | `metadata/`, `review/` | `splits/`, `review/pretrained_similarity/`, `review/global_source_cohorts/`, `PHASE_2_FINDINGS.md` | `scripts/audit_pretrained_similarity.py`, `scripts/build_global_source_cohorts.py`, `scripts/build_splits.py` |
| 03 — classical baseline | `splits/development.csv`, `data/` | `outputs/phase_3_classical_baseline/`, `PHASE_3_FINDINGS.md` | `scripts/run_classical_baseline.py` |
| 04 — pretrained screen | `splits/development.csv`, `data/` | `outputs/phase_4_pretrained/`, `PHASE_4_FINDINGS.md` | `scripts/run_pretrained_embeddings.py` |
| 05 — source robustness | Frozen v5 contract, v4 extraction contract/cache, allowlisted Phase 4 caches, preserved v1–v4 evidence | Versioned `outputs/phase_5_source_robustness*/` and `PHASE_5*_FINDINGS.md` for v1–v5; all rejected | Five versioned runners under `scripts/`, composed from reusable modules under `src/ornament_classifier/` |

Phase ownership describes provenance, not exclusive access. For example, Phase 4 consumes the Phase 2 split contract, and future phases must continue to respect it.

## Artifact status vocabulary

Use these labels consistently in documentation and generated metadata:

| Status | Meaning | Current examples |
|---|---|---|
| Canonical | Source of truth for the current protocol or a frozen reported result | `splits/`, `outputs/phase_3_classical_baseline/`, `outputs/phase_4_pretrained/`, all five `outputs/phase_5_source_robustness*/` iterations |
| Historical | Preserved for provenance but superseded for current claims | Step 1 metrics and prior Step 02 Phase 5 results described in findings |
| Diagnostic | Useful for engineering checks but not the reported experiment | `outputs/phase_4_pretrained_quick/`, fixed train-to-validation screens |
| Cache | Rebuildable performance aid, never the only result record | Pretrained embedding caches |
| Planned | Scope or scaffold with no completed evidence | A future phase directory before its contract is executed |

A directory name alone must never promote a diagnostic or planned artifact to canonical status.

## Stability rules

### Paths that remain stable after the relocation

Do not move or duplicate these established locations as part of phase organization:

- `data/`
- `metadata/`
- `review/`
- `splits/`
- `scripts/`
- `tests/`
- `outputs/phase_3_classical_baseline/`
- `outputs/phase_4_pretrained/`
- `outputs/phase_5_source_robustness/`
- `outputs/phase_5_source_robustness_v2/`
- `outputs/phase_5_source_robustness_v3/`
- `outputs/phase_5_source_robustness_v4/`
- `outputs/phase_5_source_robustness_v5/`
- top-level `PHASE_*_FINDINGS.md` reports

This is especially important for the review workbook and its image links, the split and output fingerprints, and the Phase 4 cache identity.

If a completed runner must later change for a substantive reason, record the new behavior as a new version and regenerate every artifact whose provenance pins that runner. Do not silently overwrite evidence while continuing to quote the old hash or metric.

### New Phase 5+ code

Reusable components should live under `src/ornament_classifier/` and have narrow responsibilities such as data loading, view generation, feature extraction, probing, calibration, serialization, and inference. Experiment entry points should compose those modules rather than copy code from a notebook or an earlier runner.

New outputs should receive a phase-specific directory under `outputs/` only after the output contract is defined. Temporary experiments and caches must be distinguishable from canonical evidence. A phase README should link to the resulting path rather than contain a second copy.

The assisted application is deliberately outside both `phases/` and the
development runtime because it packages
a rejected development candidate for transparent human-assisted use; it is not
promotion evidence. Its product contract, static UI, service, runtime inference,
small model artifact, and contribution quarantine live under sibling `v0_5/`.
Only deterministic model packaging remains a development script. This track
must keep the sealed-test boundary and must never rewrite canonical experiment
outputs.

### Dataset copies

`step_1/data/` and `step_2/development/data/` intentionally remain separate.
Step 1 is the protected prototype snapshot; Step 2 development is the verified
copy used by the active audit and split manifests. Never copy the dataset into
`phases/`, `outputs/`, `src/`, or `v0_5/`.

### Frozen pre-relocation paths

Canonical JSON, CSV, and generated metrics may contain `step_02/...`. Do not
edit those records. `ProjectPaths.resolve_recorded_path()` maps that prefix to
`step_2/development/`, while repository-relative `.cache/step_02/...` remains
at the repository root. `relocation_manifest.json` records every code file
whose hash changed solely to support the move; tests verify both the historical
and current hashes.

## Evaluation lifecycle

All modelling after this restructure must follow this order:

1. Load included records and source-blocked folds from `splits/development.csv`.
2. Develop, select, and compare candidates without test-image embeddings or predictions.
3. Quantify source sensitivity and report aggregate OOF metrics together with unweighted fold mean and variation.
4. Freeze preprocessing, representation, classifier, serialization, calibration, and human-review rules.
5. Record all artifact and environment fingerprints.
6. Evaluate the frozen pipeline once on the 330 records in `splits/test.csv`.
7. Preserve that result without further test-driven tuning; any later model is a new version requiring a new evaluation contract.

The current project has completed an initial pass through steps 2 and 3 and
five selection-aware Phase 5 robustness iterations. V1 improved the target
source slices but failed its frozen promotion contract. V2 tested a bounded
pairwise correction, performed slightly worse in nested OOF, and failed
correction stability. V3 tested bounded multi-representation consensus,
recovered part of v2's aggregate loss, but failed 9 of 21 gates and selected a
different fusion recipe in every outer search. V4 tested label-free motif
localization and produced the strongest aggregate result, but failed 4 of 21
gates. V5 fixed the motif stability problem and passed 18 of 21 gates, but the
persistent Opishnyan/Bubnivka boundary remained and one fold fell below its
floor. Lifecycle step 4 remains blocked; modelling is closed pending new data,
and an accepted future candidate is required before calibration, packaging, or
sealed evaluation.

## Compatibility checklist

Any structural or Phase 5+ change should leave the following true:

- all integrity and output-contract tests pass;
- the dataset fingerprint is unchanged unless a deliberate new dataset version is declared;
- split v3 rebuilds deterministically and its assignment fingerprint remains unchanged;
- production and CV leakage checks remain zero;
- canonical Phase 3, Phase 4, and Phase 5 v1–v5 files retain their recorded contents and provenance;
- Phase 4 can reproduce from its existing validated cache and environment;
- Markdown, review-workbook, and artifact-manifest links resolve;
- no sealed-test embeddings, predictions, metrics, or tuned thresholds exist before the frozen evaluation phase.

For project status, return to [`README.md`](README.md). For per-phase navigation, use [`phases/README.md`](phases/README.md).
