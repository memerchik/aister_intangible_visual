"""Strict access to the frozen Phase 5 v3 embedding allowlist.

This loader reuses the hardened Phase 5 cache primitives but pins the v3
contract, its seven cache files, all v1/v2 evidence inputs, and the current
development split. It exposes no arbitrary path, cache scan, extraction
fallback, or sealed-evaluation API.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .contracts import ContractError, load_development_contract
from .embeddings import (
    EmbeddingBlock,
    EmbeddingCacheError,
    EmbeddingRequests,
    _LoadContext,
    _development_data_fingerprint,
    _load_embedding_block,
    _normalize_requests,
    _required_mapping,
    _required_sha256,
    _required_string,
    _resolve_contract_path,
    _sha256_file,
    _validate_current_development,
    _verify_file_hash,
)
from .paths import ProjectPaths


PHASE_5_V3_CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract_v3.json"
)
EXPECTED_PHASE_5_V3_CONTRACT_SHA256 = (
    "828cc5aa45724518e10808929589722133ce336ee272cb8bda20b5089c0dc980"
)


def load_embedding_block_v3(
    cache_name: str,
    array_name: str,
    *,
    paths: Optional[ProjectPaths] = None,
) -> EmbeddingBlock:
    """Load one cache array named by the exact frozen v3 allowlist."""

    context = _load_v3_context(paths or ProjectPaths.discover())
    return _load_embedding_block(context, cache_name, array_name)


def load_allowlisted_embeddings_v3(
    requests: EmbeddingRequests,
    *,
    paths: Optional[ProjectPaths] = None,
) -> Dict[str, EmbeddingBlock]:
    """Load several v3 blocks after validating shared inputs once."""

    normalized = _normalize_requests(requests)
    context = _load_v3_context(paths or ProjectPaths.discover())
    return {
        output_name: _load_embedding_block(context, cache_name, array_name)
        for output_name, cache_name, array_name in normalized
    }


def _load_v3_context(paths: ProjectPaths) -> _LoadContext:
    expected_path = paths.step_root / PHASE_5_V3_CONTRACT_RELATIVE_PATH
    contract_path = expected_path.resolve()
    if contract_path != expected_path.resolve() or not contract_path.is_file():
        raise EmbeddingCacheError(
            f"Phase 5 v3 experiment contract is missing: {expected_path}"
        )
    observed_contract_sha256 = _sha256_file(contract_path)
    if observed_contract_sha256 != EXPECTED_PHASE_5_V3_CONTRACT_SHA256:
        raise EmbeddingCacheError(
            "Phase 5 v3 experiment contract changed after it was frozen"
        )
    try:
        experiment = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingCacheError(
            f"Cannot load Phase 5 v3 experiment contract: {error}"
        ) from error
    if not isinstance(experiment, dict):
        raise EmbeddingCacheError("Phase 5 v3 contract must be a JSON object")
    _validate_v3_header(experiment)
    input_contract = _required_mapping(experiment, "input_contract")
    _validate_v3_input_files(paths, input_contract)
    try:
        development = load_development_contract(paths)
    except ContractError as error:
        raise EmbeddingCacheError(
            f"Current development split contract is invalid: {error}"
        ) from error
    data_fingerprint_sha256 = _development_data_fingerprint(development)
    _validate_current_development(
        experiment,
        input_contract,
        development,
        data_fingerprint_sha256,
    )
    return _LoadContext(
        paths=paths,
        development=development,
        contract_path=contract_path,
        contract_sha256=observed_contract_sha256,
        experiment=experiment,
        data_fingerprint_sha256=data_fingerprint_sha256,
    )


def _validate_v3_header(experiment: object) -> None:
    if not isinstance(experiment, dict):
        raise EmbeddingCacheError("Phase 5 v3 contract must be an object")
    if experiment.get("schema_version") != 1:
        raise EmbeddingCacheError("Unsupported Phase 5 v3 contract schema")
    if experiment.get("phase") != "phase_05_source_robustness":
        raise EmbeddingCacheError("Experiment contract is not for Phase 5")
    if experiment.get("iteration") != "v3":
        raise EmbeddingCacheError("Experiment contract is not Phase 5 v3")
    if experiment.get("status") != "frozen_before_fit":
        raise EmbeddingCacheError("Phase 5 v3 contract is not frozen before fit")
    scope = _required_mapping(experiment, "scope")
    if scope.get("sealed_test_access") != "forbidden":
        raise EmbeddingCacheError("Phase 5 v3 requires sealed access to be forbidden")


def _validate_v3_input_files(paths: ProjectPaths, input_contract: object) -> None:
    if not isinstance(input_contract, dict):
        raise EmbeddingCacheError("Phase 5 v3 input contract must be an object")
    pairs = (
        ("development_csv", "development_csv_sha256"),
        ("split_audit", "split_audit_sha256"),
        ("phase_5_v1_contract", "phase_5_v1_contract_sha256"),
        ("phase_5_v1_metrics", "phase_5_v1_metrics_sha256"),
        ("phase_5_v1_full_selection", "phase_5_v1_full_selection_sha256"),
        ("phase_5_v1_nested_predictions", "phase_5_v1_nested_predictions_sha256"),
        ("phase_5_v2_contract", "phase_5_v2_contract_sha256"),
        ("phase_5_v2_metrics", "phase_5_v2_metrics_sha256"),
        ("phase_5_v2_full_selection", "phase_5_v2_full_selection_sha256"),
        ("phase_5_v2_nested_predictions", "phase_5_v2_nested_predictions_sha256"),
    )
    phase_root = paths.step_root / "phases" / "phase_05_source_robustness"
    output_root = paths.step_root / "outputs"
    canonical = {
        "development_csv": paths.development_manifest_path.resolve(),
        "split_audit": paths.split_audit_path.resolve(),
        "phase_5_v1_contract": (phase_root / "experiment_contract.json").resolve(),
        "phase_5_v1_metrics": (
            output_root / "phase_5_source_robustness" / "metrics.json"
        ).resolve(),
        "phase_5_v1_full_selection": (
            output_root
            / "phase_5_source_robustness"
            / "full_development_selection.csv"
        ).resolve(),
        "phase_5_v1_nested_predictions": (
            output_root / "phase_5_source_robustness" / "nested_oof_predictions.csv"
        ).resolve(),
        "phase_5_v2_contract": (phase_root / "experiment_contract_v2.json").resolve(),
        "phase_5_v2_metrics": (
            output_root / "phase_5_source_robustness_v2" / "metrics.json"
        ).resolve(),
        "phase_5_v2_full_selection": (
            output_root
            / "phase_5_source_robustness_v2"
            / "full_development_selection.csv"
        ).resolve(),
        "phase_5_v2_nested_predictions": (
            output_root
            / "phase_5_source_robustness_v2"
            / "nested_oof_predictions.csv"
        ).resolve(),
    }
    for path_key, digest_key in pairs:
        resolved = _resolve_contract_path(
            paths, _required_string(input_contract, path_key)
        )
        if resolved != canonical[path_key]:
            raise EmbeddingCacheError(
                f"{path_key} does not resolve to the canonical v3 input"
            )
        _verify_file_hash(
            resolved,
            _required_sha256(input_contract, digest_key),
            path_key,
        )
