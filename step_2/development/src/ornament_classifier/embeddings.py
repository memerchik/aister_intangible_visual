"""Strict, development-only access to the frozen Phase 5 embedding caches.

Only cache files and arrays named in the frozen Phase 5 experiment contract can
be loaded.  The loader intentionally has no directory scan, arbitrary path
argument, cache-generation fallback, or dependency on the Phase 4 experiment
runner.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import numpy as np

from .contracts import ContractError, DevelopmentContract, load_development_contract
from .paths import ProjectPaths


PHASE_5_CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract.json"
)
UNIT_NORM_RTOL = 5e-4
UNIT_NORM_ATOL = 5e-4

EmbeddingRequest = Tuple[str, str]
EmbeddingRequests = Union[
    Mapping[str, EmbeddingRequest],
    Iterable[EmbeddingRequest],
]


class EmbeddingCacheError(ValueError):
    """Raised when an allowlisted embedding cache violates its frozen contract."""


@dataclass(frozen=True)
class EmbeddingBlock:
    """One verified embedding matrix in canonical development-row order."""

    cache_name: str
    array_name: str
    values: np.ndarray
    image_ids: Tuple[str, ...]
    metadata: Mapping[str, object]
    provenance: Mapping[str, object]
    cache_path: Path
    cache_sha256: str
    array_sha256: str
    fingerprint: str
    data_fingerprint_sha256: str


@dataclass(frozen=True)
class _LoadContext:
    paths: ProjectPaths
    development: DevelopmentContract
    contract_path: Path
    contract_sha256: str
    experiment: Mapping[str, object]
    data_fingerprint_sha256: str


def load_embedding_block(
    cache_name: str,
    array_name: str,
    *,
    paths: Optional[ProjectPaths] = None,
) -> EmbeddingBlock:
    """Load one explicitly allowlisted Phase 5 development embedding matrix.

    ``cache_name`` and ``array_name`` must exactly match entries in
    ``experiment_contract.json``.  Callers cannot supply a cache path.
    """

    context = _load_context(paths or ProjectPaths.discover())
    return _load_embedding_block(context, cache_name, array_name)


def load_allowlisted_embeddings(
    requests: EmbeddingRequests,
    *,
    paths: Optional[ProjectPaths] = None,
) -> Dict[str, EmbeddingBlock]:
    """Load several verified blocks without repeatedly validating shared inputs.

    An iterable contains ``(cache_name, array_name)`` pairs and is returned with
    ``"cache_name.array_name"`` keys.  A mapping provides caller-selected output
    names whose values are the same two-item pairs.
    """

    normalized = _normalize_requests(requests)
    context = _load_context(paths or ProjectPaths.discover())
    return {
        output_name: _load_embedding_block(context, cache_name, array_name)
        for output_name, cache_name, array_name in normalized
    }


def _normalize_requests(
    requests: EmbeddingRequests,
) -> Tuple[Tuple[str, str, str], ...]:
    if isinstance(requests, Mapping):
        raw_requests = tuple(
            (output_name, request) for output_name, request in requests.items()
        )
    else:
        try:
            raw_requests = tuple(
                (f"{request[0]}.{request[1]}", request) for request in requests
            )
        except (IndexError, TypeError) as error:
            raise EmbeddingCacheError(
                "Embedding requests must be two-item (cache_name, array_name) pairs"
            ) from error

    normalized = []
    seen_output_names = set()
    for output_name, request in raw_requests:
        if not isinstance(output_name, str) or not output_name:
            raise EmbeddingCacheError("Embedding request output names must be non-empty")
        if (
            not isinstance(request, (tuple, list))
            or len(request) != 2
            or not all(isinstance(value, str) and value for value in request)
        ):
            raise EmbeddingCacheError(
                "Embedding requests must be two-item (cache_name, array_name) pairs"
            )
        if output_name in seen_output_names:
            raise EmbeddingCacheError(
                f"Duplicate embedding request output name: {output_name!r}"
            )
        seen_output_names.add(output_name)
        normalized.append((output_name, request[0], request[1]))
    if not normalized:
        raise EmbeddingCacheError("At least one embedding block must be requested")
    return tuple(normalized)


def _load_context(paths: ProjectPaths) -> _LoadContext:
    contract_path = (paths.step_root / PHASE_5_CONTRACT_RELATIVE_PATH).resolve()
    expected_contract_path = paths.step_root / PHASE_5_CONTRACT_RELATIVE_PATH
    if contract_path != expected_contract_path.resolve() or not contract_path.is_file():
        raise EmbeddingCacheError(
            f"Phase 5 experiment contract is missing: {expected_contract_path}"
        )
    contract_sha256 = _sha256_file(contract_path)
    try:
        experiment = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EmbeddingCacheError(
            f"Cannot load Phase 5 experiment contract: {error}"
        ) from error
    if not isinstance(experiment, dict):
        raise EmbeddingCacheError("Phase 5 experiment contract must be a JSON object")

    _validate_experiment_header(experiment)
    input_contract = _required_mapping(experiment, "input_contract")
    _validate_input_files(paths, input_contract)
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
        contract_sha256=contract_sha256,
        experiment=experiment,
        data_fingerprint_sha256=data_fingerprint_sha256,
    )


def _validate_experiment_header(experiment: Mapping[str, object]) -> None:
    if experiment.get("schema_version") != 1:
        raise EmbeddingCacheError("Unsupported Phase 5 experiment-contract schema")
    if experiment.get("phase") != "phase_05_source_robustness":
        raise EmbeddingCacheError("Experiment contract is not for Phase 5")
    if experiment.get("status") != "frozen_before_fit":
        raise EmbeddingCacheError("Phase 5 experiment contract is not frozen before fit")
    scope = _required_mapping(experiment, "scope")
    if scope.get("sealed_test_access") != "forbidden":
        raise EmbeddingCacheError(
            "Phase 5 embedding access requires sealed-test access to be forbidden"
        )


def _validate_input_files(
    paths: ProjectPaths,
    input_contract: Mapping[str, object],
) -> None:
    pairs = (
        ("development_csv", "development_csv_sha256"),
        ("split_audit", "split_audit_sha256"),
        ("phase_4_metrics", "phase_4_metrics_sha256"),
    )
    expected_canonical_paths = {
        "development_csv": paths.development_manifest_path.resolve(),
        "split_audit": paths.split_audit_path.resolve(),
    }
    for path_key, digest_key in pairs:
        relative_path = _required_string(input_contract, path_key)
        expected_digest = _required_sha256(input_contract, digest_key)
        resolved = _resolve_contract_path(paths, relative_path)
        if path_key in expected_canonical_paths and resolved != expected_canonical_paths[path_key]:
            raise EmbeddingCacheError(
                f"{path_key} does not resolve to the canonical development input"
            )
        _verify_file_hash(resolved, expected_digest, path_key)


def _validate_current_development(
    experiment: Mapping[str, object],
    input_contract: Mapping[str, object],
    development: DevelopmentContract,
    data_fingerprint_sha256: str,
) -> None:
    scope = _required_mapping(experiment, "scope")
    count = scope.get("development_image_count")
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        raise EmbeddingCacheError("Invalid development_image_count in experiment contract")
    if len(development.records) != count:
        raise EmbeddingCacheError(
            "Current development image count disagrees with the Phase 5 contract"
        )

    outer_folds = scope.get("outer_folds")
    if not isinstance(outer_folds, list) or tuple(outer_folds) != development.fold_ids:
        raise EmbeddingCacheError(
            "Current development fold IDs disagree with the Phase 5 contract"
        )
    if input_contract.get("split_version") != development.split_version:
        raise EmbeddingCacheError("Current split version disagrees with Phase 5")
    if input_contract.get("split_seed") != development.split_seed:
        raise EmbeddingCacheError("Current split seed disagrees with Phase 5")

    expected_assignment = _required_sha256(
        input_contract, "split_assignment_fingerprint_sha256"
    )
    if development.audit.get("assignment_fingerprint_sha256") != expected_assignment:
        raise EmbeddingCacheError(
            "Current split assignment fingerprint disagrees with Phase 5"
        )
    expected_data = _required_sha256(input_contract, "data_fingerprint_sha256")
    if data_fingerprint_sha256 != expected_data:
        raise EmbeddingCacheError(
            "Current development ID/content order fingerprint disagrees with Phase 5"
        )


def _load_embedding_block(
    context: _LoadContext,
    cache_name: str,
    array_name: str,
) -> EmbeddingBlock:
    allowlist = _required_mapping(context.experiment, "cache_allowlist")
    cache_spec = allowlist.get(cache_name)
    if not isinstance(cache_spec, dict):
        raise EmbeddingCacheError(f"Embedding cache is not allowlisted: {cache_name!r}")
    arrays = _required_mapping(cache_spec, "arrays")
    array_spec = arrays.get(array_name)
    if not isinstance(array_spec, dict):
        raise EmbeddingCacheError(
            f"Embedding array is not allowlisted: {cache_name}.{array_name}"
        )

    cache_path = _resolve_contract_path(
        context.paths, _required_string(cache_spec, "path")
    )
    if cache_path.suffix != ".npz":
        raise EmbeddingCacheError("Allowlisted embedding caches must be NPZ files")
    expected_cache_sha256 = _required_sha256(cache_spec, "file_sha256")
    _verify_file_hash(cache_path, expected_cache_sha256, cache_name)

    expected_fingerprint = _required_sha256(cache_spec, "fingerprint")
    expected_array_sha256 = _required_sha256(array_spec, "sha256")
    expected_shape = _required_shape(array_spec)
    expected_dtype = _required_string(array_spec, "dtype")
    if expected_shape[0] != len(context.development.records):
        raise EmbeddingCacheError(
            f"Allowlisted row count disagrees with development: {cache_name}.{array_name}"
        )
    if expected_dtype != "float32":
        raise EmbeddingCacheError(
            f"Allowlisted embeddings must be float32: {cache_name}.{array_name}"
        )

    try:
        with np.load(cache_path, allow_pickle=False) as cached:
            required_arrays = {"fingerprint", "image_ids", "metadata_json", array_name}
            missing = sorted(required_arrays - set(cached.files))
            if missing:
                raise EmbeddingCacheError(
                    f"Embedding cache is missing arrays: {', '.join(missing)}"
                )
            fingerprint = _read_scalar_string(cached["fingerprint"], "fingerprint")
            image_ids_array = np.asarray(cached["image_ids"])
            metadata_text = _read_scalar_string(
                cached["metadata_json"], "metadata_json"
            )
            values = np.array(cached[array_name], copy=True, order="C")
    except EmbeddingCacheError:
        raise
    except (OSError, ValueError, TypeError) as error:
        raise EmbeddingCacheError(
            f"Cannot safely load allowlisted embedding cache {cache_path}: {error}"
        ) from error

    if fingerprint != expected_fingerprint:
        raise EmbeddingCacheError(f"Embedding fingerprint mismatch: {cache_name}")
    image_ids = _validate_image_ids(image_ids_array, context.development)
    metadata = _parse_metadata(metadata_text, cache_name)
    if metadata.get("embedding_fingerprint_sha256") != expected_fingerprint:
        raise EmbeddingCacheError(
            f"Embedding metadata fingerprint mismatch: {cache_name}"
        )
    identity = metadata.get("cache_identity")
    if not isinstance(identity, dict):
        raise EmbeddingCacheError(f"Embedding cache identity is missing: {cache_name}")
    if identity.get("data_fingerprint") != context.data_fingerprint_sha256:
        raise EmbeddingCacheError(
            f"Embedding cache targets a different development dataset: {cache_name}"
        )

    _validate_values(
        values,
        expected_shape,
        expected_dtype,
        expected_array_sha256,
        cache_name,
        array_name,
    )
    values.flags.writeable = False
    provenance = {
        "phase": context.experiment["phase"],
        "experiment_version": context.experiment.get("experiment_version"),
        "experiment_contract_path": str(context.contract_path),
        "experiment_contract_sha256": context.contract_sha256,
        "cache_name": cache_name,
        "array_name": array_name,
        "cache_sha256": expected_cache_sha256,
        "array_sha256": expected_array_sha256,
        "embedding_fingerprint_sha256": expected_fingerprint,
        "data_fingerprint_sha256": context.data_fingerprint_sha256,
        "split_assignment_fingerprint_sha256": _required_sha256(
            _required_mapping(context.experiment, "input_contract"),
            "split_assignment_fingerprint_sha256",
        ),
    }
    return EmbeddingBlock(
        cache_name=cache_name,
        array_name=array_name,
        values=values,
        image_ids=image_ids,
        metadata=metadata,
        provenance=provenance,
        cache_path=cache_path,
        cache_sha256=expected_cache_sha256,
        array_sha256=expected_array_sha256,
        fingerprint=expected_fingerprint,
        data_fingerprint_sha256=context.data_fingerprint_sha256,
    )


def _validate_image_ids(
    values: np.ndarray,
    development: DevelopmentContract,
) -> Tuple[str, ...]:
    if values.ndim != 1 or values.dtype.kind != "U":
        raise EmbeddingCacheError(
            "Embedding image_ids must be a one-dimensional Unicode array"
        )
    image_ids = tuple(str(value) for value in values.tolist())
    expected = tuple(record.image_id for record in development.records)
    if len(image_ids) != len(expected):
        raise EmbeddingCacheError("Embedding image ID count disagrees with development")
    if image_ids != expected:
        raise EmbeddingCacheError(
            "Embedding image IDs do not match current development-row order"
        )
    return image_ids


def _validate_values(
    values: np.ndarray,
    expected_shape: Tuple[int, int],
    expected_dtype: str,
    expected_sha256: str,
    cache_name: str,
    array_name: str,
) -> None:
    reference = f"{cache_name}.{array_name}"
    if values.shape != expected_shape or values.ndim != 2:
        raise EmbeddingCacheError(
            f"Embedding shape mismatch for {reference}: {values.shape}"
        )
    if values.dtype.name != expected_dtype:
        raise EmbeddingCacheError(
            f"Embedding dtype mismatch for {reference}: {values.dtype.name}"
        )
    actual_sha256 = _sha256_array(values)
    if actual_sha256 != expected_sha256:
        raise EmbeddingCacheError(f"Embedding array hash mismatch for {reference}")
    if not np.isfinite(values).all():
        raise EmbeddingCacheError(f"Embedding contains non-finite values: {reference}")
    norms = np.linalg.norm(values, axis=1)
    if not np.allclose(
        norms,
        1.0,
        rtol=UNIT_NORM_RTOL,
        atol=UNIT_NORM_ATOL,
    ):
        raise EmbeddingCacheError(f"Embedding rows are not unit normalized: {reference}")


def _parse_metadata(value: str, cache_name: str) -> Dict[str, object]:
    try:
        metadata = json.loads(value)
    except json.JSONDecodeError as error:
        raise EmbeddingCacheError(
            f"Embedding metadata is invalid JSON: {cache_name}"
        ) from error
    if not isinstance(metadata, dict):
        raise EmbeddingCacheError(
            f"Embedding metadata must be an object: {cache_name}"
        )
    return metadata


def _read_scalar_string(values: np.ndarray, name: str) -> str:
    if values.shape != () or values.dtype.kind != "U":
        raise EmbeddingCacheError(f"Embedding cache {name} must be a Unicode scalar")
    return str(values.item())


def _required_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise EmbeddingCacheError(f"Experiment contract field {key!r} must be an object")
    return value


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise EmbeddingCacheError(
            f"Experiment contract field {key!r} must be a non-empty string"
        )
    return value


def _required_sha256(mapping: Mapping[str, object], key: str) -> str:
    value = _required_string(mapping, key).lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise EmbeddingCacheError(
            f"Experiment contract field {key!r} must be a SHA-256 digest"
        )
    return value


def _required_shape(array_spec: Mapping[str, object]) -> Tuple[int, int]:
    shape = array_spec.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in shape
        )
    ):
        raise EmbeddingCacheError("Allowlisted embedding shape must contain two sizes")
    return shape[0], shape[1]


def _resolve_contract_path(paths: ProjectPaths, relative_path: str) -> Path:
    try:
        return paths.resolve_recorded_path(relative_path)
    except ValueError as error:
        raise EmbeddingCacheError(
            f"Experiment-contract path escapes the repository: {relative_path!r}"
        ) from error


def _development_data_fingerprint(development: DevelopmentContract) -> str:
    digest = hashlib.sha256()
    for record in development.records:
        digest.update(
            f"{record.image_id}\x1f{record.content_sha256}\n".encode("utf-8")
        )
    return digest.hexdigest()


def _verify_file_hash(path: Path, expected_sha256: str, label: str) -> None:
    if not path.is_file():
        raise EmbeddingCacheError(f"Required allowlisted file is missing: {path}")
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise EmbeddingCacheError(f"File hash mismatch for {label}: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    except OSError as error:
        raise EmbeddingCacheError(f"Cannot hash required file {path}: {error}") from error
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(values)
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()
