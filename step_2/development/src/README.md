# Development Python package

`ornament_classifier` contains reusable modelling code for the Phase 5
experiments. It loads only the development split and checks the source-atomic
evaluation contract before returning records or cached features.

## Modules

| Module | Purpose |
|---|---|
| `paths.py` | Resolves current and historically recorded repository paths |
| `contracts.py` | Loads and validates development records and split metadata |
| `embeddings.py` | Loads the feature blocks used by the original Phase 5 experiment |
| `robustness.py` | Source weighting, candidate evaluation, and diagnostic helpers |
| `pairwise.py`, `embeddings_v2.py` | Pairwise correction and the v2 feature-cache contract |
| `consensus.py`, `embeddings_v3.py` | Cross-encoder fusion and the v3 cache contract |
| `motif.py`, `embeddings_v4.py` | Label-free motif regions and the two-stage v4 cache contract |
| `fixed_consensus.py` | Fixed centered-log-probability consensus used by v5 |

The experiment runners live in [`../scripts/`](../scripts), and their saved
results live in [`../outputs/`](../outputs). None of the loaders in this package
provides access to the sealed test set.

## Use from Python

Add `step_2/development/src` to `PYTHONPATH`, then load the development contract:

```python
from ornament_classifier import load_development_contract

development = load_development_contract()
```

The standalone application has its own runtime in
[`../../v0_5/src/aister_runtime/`](../../v0_5/src/aister_runtime/inference.py).
This keeps upload-time inference independent of the research dataset, split
files, and experiment code.
