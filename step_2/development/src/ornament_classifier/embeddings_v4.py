"""Strict loader for the frozen Phase 5 v4 motif-tile cache."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import numpy as np

from .contracts import load_development_contract
from .paths import ProjectPaths


CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/experiment_contract_v4.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "7df72082a99d835a1bbbc91c9eb69ca399d1242cd97e9ca64a0ea38135ec42da"
)


class MotifCacheError(ValueError):
    """Raised when the v4 cache or its provenance violates the frozen contract."""


@dataclass(frozen=True)
class MotifEmbeddingCache:
    embeddings: Mapping[str, np.ndarray]
    tile_embeddings: np.ndarray
    proposal_ids: np.ndarray
    proposal_boxes: np.ndarray
    proposal_scores: np.ndarray
    proposal_scales: np.ndarray
    proposal_positions: np.ndarray
    provenance: Mapping[str, object]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x1f")
    digest.update("x".join(str(value) for value in array.shape).encode("ascii"))
    digest.update(b"\n")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _load_contract(paths: ProjectPaths) -> tuple[Mapping[str, object], Path]:
    path = (paths.step_root / CONTRACT_RELATIVE_PATH).resolve()
    expected_path = paths.step_root / CONTRACT_RELATIVE_PATH
    if path != expected_path.resolve() or not path.is_file():
        raise MotifCacheError("Frozen Phase 5 v4 experiment contract is missing")
    if _sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        raise MotifCacheError("Phase 5 v4 experiment contract changed after freezing")
    try:
        contract = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise MotifCacheError(f"Cannot load Phase 5 v4 contract: {error}") from error
    if not isinstance(contract, dict):
        raise MotifCacheError("Phase 5 v4 contract must be an object")
    if contract.get("iteration") != "v4" or contract.get("status") != "frozen_before_fit":
        raise MotifCacheError("Phase 5 v4 contract has an invalid status")
    scope = contract.get("scope")
    if not isinstance(scope, dict) or scope.get("sealed_test_access") != "forbidden":
        raise MotifCacheError("Phase 5 v4 sealed access must be forbidden")
    return contract, path


def load_motif_embeddings_v4(
    *, paths: Optional[ProjectPaths] = None
) -> MotifEmbeddingCache:
    resolved = paths or ProjectPaths.discover()
    contract, contract_path = _load_contract(resolved)
    spec = contract.get("motif_cache")
    if not isinstance(spec, dict):
        raise MotifCacheError("Phase 5 v4 motif_cache contract is invalid")
    relative = spec.get("path")
    expected_file_hash = spec.get("file_sha256")
    if not isinstance(relative, str) or not isinstance(expected_file_hash, str):
        raise MotifCacheError("Phase 5 v4 motif cache path or hash is invalid")
    repository_root = resolved.workspace_root.resolve()
    try:
        path = resolved.resolve_recorded_path(relative)
    except ValueError as error:
        raise MotifCacheError("Phase 5 v4 motif cache escapes the repository") from error
    if not path.is_file() or _sha256_file(path) != expected_file_hash:
        raise MotifCacheError("Phase 5 v4 motif cache is missing or changed")

    required_files = tuple(spec.get("required_arrays", ()))
    array_specs = spec.get("arrays")
    if not required_files or len(required_files) != len(set(required_files)):
        raise MotifCacheError("Phase 5 v4 required cache arrays are invalid")
    if not isinstance(array_specs, dict):
        raise MotifCacheError("Phase 5 v4 array specifications are invalid")
    try:
        with np.load(path, allow_pickle=False) as cached:
            if set(cached.files) != set(required_files):
                raise MotifCacheError("Phase 5 v4 cache members disagree with the contract")
            arrays = {name: np.array(cached[name], copy=True) for name in cached.files}
    except (OSError, ValueError) as error:
        if isinstance(error, MotifCacheError):
            raise
        raise MotifCacheError(f"Cannot read Phase 5 v4 motif cache: {error}") from error

    development = load_development_contract(resolved)
    expected_ids = [record.image_id for record in development.records]
    expected_content = [record.content_sha256 for record in development.records]
    if arrays["image_ids"].tolist() != expected_ids:
        raise MotifCacheError("Phase 5 v4 motif cache image order changed")
    if arrays["content_sha256s"].tolist() != expected_content:
        raise MotifCacheError("Phase 5 v4 motif cache content hashes changed")
    if str(arrays["fingerprint"].item()) != spec.get("fingerprint"):
        raise MotifCacheError("Phase 5 v4 motif cache fingerprint changed")
    try:
        metadata = json.loads(str(arrays["metadata_json"].item()))
    except (ValueError, json.JSONDecodeError) as error:
        raise MotifCacheError("Phase 5 v4 motif cache metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise MotifCacheError("Phase 5 v4 motif cache metadata must be an object")
    if metadata.get("cache_fingerprint_sha256") != spec.get("fingerprint"):
        raise MotifCacheError("Phase 5 v4 metadata fingerprint changed")
    if metadata.get("sealed_test_evaluated") is not False:
        raise MotifCacheError("Phase 5 v4 motif cache sealed status is invalid")
    identity = metadata.get("cache_identity")
    if not isinstance(identity, dict):
        raise MotifCacheError("Phase 5 v4 motif cache identity is invalid")
    if identity.get("extraction_contract_sha256") != spec.get("extraction_contract_sha256"):
        raise MotifCacheError("Phase 5 v4 extraction contract provenance changed")

    for name, declared in array_specs.items():
        if name not in arrays or not isinstance(declared, dict):
            raise MotifCacheError(f"Phase 5 v4 array is missing: {name}")
        value = arrays[name]
        if list(value.shape) != declared.get("shape"):
            raise MotifCacheError(f"Phase 5 v4 array shape changed: {name}")
        if str(value.dtype) != declared.get("dtype"):
            raise MotifCacheError(f"Phase 5 v4 array dtype changed: {name}")
        if _sha256_array(value) != declared.get("sha256"):
            raise MotifCacheError(f"Phase 5 v4 array hash changed: {name}")
        if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
            raise MotifCacheError(f"Phase 5 v4 array is non-finite: {name}")

    descriptor_names = tuple(str(value) for value in spec.get("descriptor_names", ()))
    embeddings = {}
    for name in descriptor_names:
        key = f"embedding__{name}"
        value = arrays[key]
        if value.ndim != 2 or value.shape != (len(expected_ids), 384):
            raise MotifCacheError(f"Phase 5 v4 descriptor shape is invalid: {name}")
        norms = np.linalg.norm(value, axis=1)
        if not np.allclose(norms, 1.0, rtol=5e-4, atol=5e-4):
            raise MotifCacheError(f"Phase 5 v4 descriptor is not unit-normalized: {name}")
        embeddings[name] = value
    tiles = arrays["tile_embeddings"]
    if tiles.shape != (len(expected_ids), 6, 384):
        raise MotifCacheError("Phase 5 v4 tile embedding shape is invalid")
    if not np.allclose(np.linalg.norm(tiles, axis=2), 1.0, rtol=5e-4, atol=5e-4):
        raise MotifCacheError("Phase 5 v4 tile embeddings are not unit-normalized")
    provenance = {
        "experiment_contract_path": str(contract_path.relative_to(repository_root)),
        "experiment_contract_sha256": EXPECTED_CONTRACT_SHA256,
        "cache_path": str(path.relative_to(repository_root)),
        "cache_sha256": expected_file_hash,
        "cache_fingerprint_sha256": spec.get("fingerprint"),
        "extraction_contract_sha256": spec.get("extraction_contract_sha256"),
        "metadata": metadata,
    }
    return MotifEmbeddingCache(
        embeddings=embeddings,
        tile_embeddings=tiles,
        proposal_ids=arrays["proposal_ids"],
        proposal_boxes=arrays["proposal_boxes"],
        proposal_scores=arrays["proposal_scores"],
        proposal_scales=arrays["proposal_scales"],
        proposal_positions=arrays["proposal_positions"],
        provenance=provenance,
    )
