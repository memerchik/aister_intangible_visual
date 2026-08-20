# Phase 5: source robustness

Phase 5 contains five experiments designed to improve performance across held-
out acquisition sources. V4 achieved the best overall accuracy, while v5 was
the most stable. No version met every release gate, so the project has not yet
moved to calibration or final test evaluation.

## V5 result

The final v5 experiment replaced v4's descriptor search with one fixed
four-head consensus. Its selection-aware nested evaluation reached 94.5068%
accuracy, 94.1957% macro F1, 94.9226% balanced accuracy, and a project-best
76.8884% source-robustness score over all 1,693 development records.

V5 passed 18 of 21 gates. It fixed v4's Bubnivka-recall and motif-stability
failures, and the consensus was selected in all five outer searches. It still
missed the required Bubnivka precision (79.8354%), Opishnyan-to-Bubnivka error
count (46), and worst outer-fold accuracy (86.8421%). The sealed test was not
used.

The detailed records are available in:

- [`PHASE_5_V5_FINDINGS.md`](../../PHASE_5_V5_FINDINGS.md) for the v5 result;
- [`PHASE_5_EVOLUTION.md`](../../PHASE_5_EVOLUTION.md) for the v1–v5 history;
- [`PHASE_5_GATES.md`](../../PHASE_5_GATES.md) for all promotion gates.

## V5 implementation

- [Experiment definition](experiment_contract_v5.json)
- [Fixed-consensus runner](../../scripts/run_phase5_v5_fixed_consensus.py)
- [Saved v5 outputs](../../outputs/phase_5_source_robustness_v5/)
- [Consensus implementation](../../src/ornament_classifier/fixed_consensus.py)
- [Output-integrity tests](../../tests/test_phase_5_v5_outputs.py)
- [Pinned environment](../../requirements-phase5.txt)

V5 averages the selected base model with uniform, texture-weighted, and top-two
motif heads using equal centered log probabilities. It has no descriptor or
blend search, threshold tuning, pair correction, or per-image rule. Two full
runs reproduced all 14 output files byte for byte.

## Earlier experiments

| Version | Method | Accuracy | Gates passed | Result |
|---|---|---:|---:|---|
| [V1](../../PHASE_5_FINDINGS.md) | Source-balanced DINOv3 selection | 93.9752% | 12/17 | Not promoted |
| [V2](../../PHASE_5_V2_FINDINGS.md) | DINOv3 pair correction | 93.7980% | 12/18 | Not promoted |
| [V3](../../PHASE_5_V3_FINDINGS.md) | Cross-encoder consensus | 93.9161% | 12/21 | Not promoted |
| [V4](../../PHASE_5_V4_FINDINGS.md) | Label-free motif localization | **94.6840%** | 17/21 | Not promoted |
| [V5](../../PHASE_5_V5_FINDINGS.md) | Fixed motif consensus | 94.5068% | **18/21** | Not promoted |

The corresponding experiment definitions, runners, and saved outputs remain at
their versioned paths. V4 has the strongest aggregate result; v5 has the most
stable motif procedure and the best source-robustness score.

## Why development stops here

Bubnivka precision and Opishnyan-to-Bubnivka errors missed their gates in every
version. More tuning on the same repeatedly reviewed records would increase
selection bias without adding independent evidence. A new modelling cycle will
need additional objects and acquisition sources, especially examples that
clarify the Opishnyan/Bubnivka distinction. It will also need a new split
version, gates defined before fitting, and a new untouched final evaluation
partition.
