#!/usr/bin/env python3
"""Package the fixed v4 full-development recipe for the assisted v0.5 app.

This is packaging, not a new model-selection experiment. It reads development
embeddings only, never opens the sealed test manifest or test predictions, and
records the rejected/provisional status in the generated model manifest.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Mapping


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SOURCE_ROOT = STEP_ROOT / "src"
APPLICATION_ROOT = STEP_ROOT.parent / "v0_5"
APPLICATION_SOURCE_ROOT = APPLICATION_ROOT / "src"
for source_root in (SOURCE_ROOT, APPLICATION_SOURCE_ROOT):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import numpy as np  # noqa: E402

from aister_runtime.inference import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    CLASS_ORDER,
    MODEL_ID,
    MODEL_WEIGHT_SHA256,
    sha256_array,
    sha256_file,
)
from ornament_classifier.contracts import load_development_contract  # noqa: E402
from ornament_classifier.paths import ProjectPaths  # noqa: E402
from ornament_classifier.robustness import (  # noqa: E402
    compute_source_group_weights,
    fit_logistic_probe,
)


GLOBAL_CACHE = REPO_ROOT / ".cache/step_02/pretrained_embeddings/dinov3_vits16__global_fivecrop__2783a020edd6f819.npz"
MOTIF_CACHE = REPO_ROOT / ".cache/step_02/pretrained_embeddings/dinov3_vits16__motif_tiles_v4.npz"
GLOBAL_CACHE_SHA256 = "cf04c78f40db527306f7775e1c9277f99ec2ab97c8aaf5180e09f87e6af35bfa"
MOTIF_CACHE_SHA256 = "97ee78769b6b4349b58e9b2cb27c9f4b58748ecf7cd9fbcfcc554c2fa122db40"
V4_CONTRACT = STEP_ROOT / "phases/phase_05_source_robustness/experiment_contract_v4.json"
V4_CONTRACT_SHA256 = "7df72082a99d835a1bbbc91c9eb69ca399d1242cd97e9ca64a0ea38135ec42da"
V4_METRICS = STEP_ROOT / "outputs/phase_5_source_robustness_v4/metrics.json"
V4_METRICS_SHA256 = "c53116230d3e558ab31538dee25c1c35131fb0e6176002bcc9214f70fbc727f3"
DEVELOPMENT_OOF_ACCURACY = 0.941523922031896
DEVELOPMENT_OOF_MACRO_F1 = 0.9386036923727085


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=APPLICATION_ROOT / "artifacts",
    )
    return parser.parse_args()


def stable_json(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", dir=str(path.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def deterministic_npz(arrays: Mapping[str, np.ndarray]) -> bytes:
    """Return a compressed NumPy archive with stable ordering and timestamps."""

    archive = io.BytesIO()
    with zipfile.ZipFile(
        archive,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as zipped:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]), allow_pickle=False)
            information = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            information.compress_type = zipfile.ZIP_DEFLATED
            information.external_attr = 0o600 << 16
            zipped.writestr(information, member.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
    return archive.getvalue()


def validate_inputs(paths: ProjectPaths) -> None:
    expected = {
        GLOBAL_CACHE: GLOBAL_CACHE_SHA256,
        MOTIF_CACHE: MOTIF_CACHE_SHA256,
        V4_CONTRACT: V4_CONTRACT_SHA256,
        V4_METRICS: V4_METRICS_SHA256,
    }
    for path, digest in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise RuntimeError(f"Pinned v0.5 packaging input changed: {path}")
    product_contract = APPLICATION_ROOT / "product_contract.json"
    payload = json.loads(product_contract.read_text(encoding="utf-8"))
    if payload.get("evaluation_boundary", {}).get("sealed_test_access") != "forbidden":
        raise RuntimeError("The v0.5 product contract violates the sealed-test boundary")


def load_training_arrays(development) -> tuple[np.ndarray, np.ndarray]:
    with np.load(GLOBAL_CACHE, allow_pickle=False) as stored:
        base_ids = np.array(stored["image_ids"], copy=True)
        base = np.array(stored["embedding__cls"], dtype=np.float32, copy=True)
    with np.load(MOTIF_CACHE, allow_pickle=False) as stored:
        motif_ids = np.array(stored["image_ids"], copy=True)
        motif = np.array(
            stored["embedding__texture_weighted_mean"], dtype=np.float32, copy=True
        )
    expected_ids = np.asarray([record.image_id for record in development.records])
    if not np.array_equal(base_ids, expected_ids) or not np.array_equal(motif_ids, expected_ids):
        raise RuntimeError("The v0.5 packaging cache image order changed")
    return np.concatenate((base, motif), axis=1), expected_ids


def main() -> None:
    args = parse_args()
    paths = ProjectPaths.discover()
    validate_inputs(paths)
    development = load_development_contract(paths)
    if len(development.records) != 1693:
        raise RuntimeError("The v0.5 model requires all 1,693 development records")
    features, image_ids = load_training_arrays(development)
    labels = np.asarray([record.ornament_label for record in development.records], dtype=object)
    groups = np.asarray([record.split_group_id for record in development.records], dtype=object)
    if tuple(sorted(set(labels.tolist()))) != CLASS_ORDER:
        raise RuntimeError("The v0.5 class contract changed")
    weights = compute_source_group_weights(labels, groups, 0.5)
    fitted = fit_logistic_probe(
        features,
        labels,
        c_value=10.0,
        sample_weight=weights,
        classes=CLASS_ORDER,
        seed=20260719,
    )
    classes = np.asarray(fitted.classes)
    coefficients = np.asarray(fitted.estimator.coef_, dtype=np.float64)
    intercept = np.asarray(fitted.estimator.intercept_, dtype=np.float64)
    output_directory = args.output_directory.resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact_path = output_directory / "v4_assisted_model.npz"
    write_atomic(
        artifact_path,
        deterministic_npz(
            {
                "classes": classes,
                "coefficients": coefficients,
                "intercept": intercept,
            }
        ),
    )
    product_contract = APPLICATION_ROOT / "product_contract.json"
    manifest = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "application_version": "v0.5",
        "model_id": "v4_full_development_global_plus_texture_weighted_motif",
        "status": "provisional_human_assisted_not_promoted",
        "recipe": {
            "encoder": MODEL_ID,
            "encoder_weight_sha256": MODEL_WEIGHT_SHA256,
            "base": "global_fivecrop_cls",
            "motif": "texture_weighted_mean",
            "feature_dimensions": 768,
            "classifier": "sklearn_logistic_regression_l2_lbfgs",
            "c_value": 10.0,
            "source_group_exponent": 0.5,
            "fit_seed": 20260719,
            "iterations": list(fitted.iterations),
        },
        "training": {
            "scope": "development_only",
            "image_count": len(development.records),
            "image_ids_sha256": sha256_array(image_ids),
            "split_version": development.split_version,
            "split_assignment_fingerprint_sha256": development.audit.get(
                "assignment_fingerprint_sha256"
            ),
            "global_embedding_cache_sha256": GLOBAL_CACHE_SHA256,
            "motif_embedding_cache_sha256": MOTIF_CACHE_SHA256,
        },
        "development_evidence": {
            "scope": "selection_conditional_oof_not_sealed_test",
            "accuracy": DEVELOPMENT_OOF_ACCURACY,
            "macro_f1": DEVELOPMENT_OOF_MACRO_F1,
            "phase_5_v4_contract_sha256": V4_CONTRACT_SHA256,
            "phase_5_v4_metrics_sha256": V4_METRICS_SHA256,
            "promotion_decision": "reject",
        },
        "product_contract_path": str(product_contract.relative_to(REPO_ROOT)),
        "product_contract_sha256": sha256_file(product_contract),
        "array_sha256": {
            "classes": sha256_array(classes),
            "coefficients": sha256_array(coefficients),
            "intercept": sha256_array(intercept),
        },
        "artifact_sha256": sha256_file(artifact_path),
        "score_type": "uncalibrated_ranking_score",
        "calibrated": False,
        "unknown_class_rejection": False,
        "sealed_test_evaluated": False,
        "public_deployment_ready": False,
    }
    manifest_path = output_directory / "model_manifest.json"
    write_atomic(manifest_path, stable_json(manifest).encode("utf-8"))
    print(
        json.dumps(
            {
                "artifact": str(artifact_path),
                "artifact_sha256": manifest["artifact_sha256"],
                "manifest": str(manifest_path),
                "development_image_count": len(development.records),
                "sealed_test_evaluated": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
