# Phase 05 — source robustness

**Status: v1 through v5 complete; all candidates rejected. Phase 5 modelling
is closed pending materially more source-diverse data.** Phase 6 calibration,
packaging, and sealed evaluation remain blocked.

## Final outcome: v5

The final frozen v5 experiment replaced v4's descriptor search with one fixed
four-head consensus. Its selection-aware nested procedure reached 94.5068%
accuracy, 94.1957% macro F1, 94.9226% balanced accuracy, and a project-best
76.8884% source-robustness score over all 1,693 development records.

V5 passed 18 of 21 gates. It fixed v4's Bubnivka-recall and motif-stability
failures—the consensus was eligible in all 5 outer searches—but failed
Bubnivka precision (79.8354%), Opishnyan-to-Bubnivka errors (46), and worst
outer-fold accuracy (86.8421%). The all-required contract therefore rejected
it. The sealed test was not touched.

Read [`../../PHASE_5_V5_FINDINGS.md`](../../PHASE_5_V5_FINDINGS.md) for the
complete result and data-growth implications,
[`../../PHASE_5_EVOLUTION.md`](../../PHASE_5_EVOLUTION.md) for the chronological
record, and [`../../PHASE_5_GATES.md`](../../PHASE_5_GATES.md) for every gate.

## V5 contract and implementation

- [Frozen v5 contract](experiment_contract_v5.json)
- [V5 fixed-consensus runner](../../scripts/run_phase5_v5_fixed_consensus.py)
- [V5 canonical outputs](../../outputs/phase_5_source_robustness_v5/)
- [Fixed consensus primitive](../../src/ornament_classifier/fixed_consensus.py)
- [V5 output-integrity tests](../../tests/test_phase_5_v5_outputs.py)
- [Pinned Phase 5 classifier environment](../../requirements-phase5.txt)

There was one post-base recipe: the selected base plus uniform,
texture-weighted, and top-two motif heads, averaged as equal centered log
probabilities. There was no descriptor/blend search, threshold tuning, pair
correction, or per-image rule. Two complete runs reproduced all 14 artifacts
byte for byte, and the v1 base fingerprint reproduced before the result was
accepted.

## Preserved v1–v4 evidence

| Version | Main method | Accuracy | Gates | Decision |
|---|---|---:|---:|---|
| [V1](../../PHASE_5_FINDINGS.md) | Source-balanced DINOv3 selection | 93.9752% | 12/17 | Reject |
| [V2](../../PHASE_5_V2_FINDINGS.md) | DINOv3 pair correction | 93.7980% | 12/18 | Reject |
| [V3](../../PHASE_5_V3_FINDINGS.md) | Cross-encoder consensus | 93.9161% | 12/21 | Reject |
| [V4](../../PHASE_5_V4_FINDINGS.md) | Label-free motif localization | **94.6840%** | 17/21 | Reject |
| [V5](../../PHASE_5_V5_FINDINGS.md) | Fixed motif consensus | 94.5068% | **18/21** | Reject |

The corresponding contracts, runners, and immutable outputs remain at their
versioned paths. V4 remains the strongest aggregate/boundary result; v5 is the
most source-robust and stable motif procedure.

## Final evaluation boundary

Two gates failed all five versions: Bubnivka precision and
Opishnyan-to-Bubnivka errors. A sixth search on the repeatedly inspected
development rows would add selection pressure without new independent
evidence. Modelling therefore stops until the dataset gains new independent
objects and acquisition sources, especially for the Opishnyan/Bubnivka
boundary.

When data grows, create a new dataset/split version, refreeze gates, rebuild
source-atomic folds, and reserve a new sealed evaluation partition. Until a
future all-pass promotion result exists, Phase 6 and the current sealed test
remain closed.
