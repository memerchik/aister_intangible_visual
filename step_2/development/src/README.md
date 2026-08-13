# Reusable implementation

`ornament_classifier` is the reusable package boundary for Phase 5 and later.
It centralizes development-only paths and validates the source-atomic
development contract before modelling code can use it.

Phase 5 v2 adds array-only pairwise correction primitives in `pairwise.py` and
a contract-pinned six-block cache loader in `embeddings_v2.py`. Neither module
exposes a sealed-test path.

Phase 5 v3 adds array-only consensus-fusion and deterministic candidate-ranking
primitives in `consensus.py`, plus a frozen-contract allowlist loader in
`embeddings_v3.py`. It also exposes no sealed-test path.

Phase 5 v4 adds deterministic, label-free motif proposal and aggregation
primitives in `motif.py`, plus the two-stage-contract cache boundary in
`embeddings_v4.py`. The extraction and experiment runners remain under
`scripts/`; neither reusable module exposes a sealed-test path.

Phase 5 v5 adds the array-only, fixed centered-log-probability consensus in
`fixed_consensus.py`. It has no file access, learned blend, class-specific
threshold, or sealed-test path.

The provisional application runtime has moved to sibling
[`../../v0_5/src/aister_runtime/`](../../v0_5/src/aister_runtime/inference.py). It
owns upload-time inference and has no dependency on this development package.
The development builder still reproduces and packages the fixed v4 recipe into
that standalone app; this does not constitute Phase 6 promotion.

Add `step_2/development/src` to `PYTHONPATH`, then load the contract with:

```python
from ornament_classifier import load_development_contract

development = load_development_contract()
```

The public API intentionally has no sealed-test path or loader. The sealed test
is reserved for the separate, one-shot evaluation procedure after the model,
calibration, and review policy have been frozen.
