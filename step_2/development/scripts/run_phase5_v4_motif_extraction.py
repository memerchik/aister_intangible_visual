#!/usr/bin/env python3
"""Extract the frozen, development-only Phase 5 v4 motif-tile cache.

This is a label-free extraction stage. It validates the source-atomic
development manifest and every image byte, selects deterministic motif
proposals from pixels only, and applies the already-local DINOv3 ViT-S/16
checkpoint. It has no sealed-test loader or prediction path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import platform
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SOURCE_ROOT = STEP_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np  # noqa: E402
import PIL  # noqa: E402
from PIL import Image  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

from ornament_classifier.contracts import load_development_contract  # noqa: E402
from ornament_classifier.motif import (  # noqa: E402
    aggregate_tile_embeddings,
    select_motif_proposals,
)
from ornament_classifier.paths import ProjectPaths  # noqa: E402


EXTRACTION_CONTRACT_RELATIVE_PATH = Path(
    "phases/phase_05_source_robustness/motif_extraction_contract_v4.json"
)
EXPECTED_EXTRACTION_CONTRACT_SHA256 = (
    "19df4814d2d915c461f267eed89282e5474abb1ac0d37fc50d9dbf25be7c33ff"
)
SCRIPT_PATH = Path(__file__).resolve()
MODEL_ARCHITECTURE = "vit_small_patch16_dinov3_qkvb"
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--weight-path", type=Path, default=None)
    parser.add_argument("--batch-images", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(b"\x1f")
    digest.update("x".join(str(value) for value in array.shape).encode("ascii"))
    digest.update(b"\n")
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_extraction_contract(paths: ProjectPaths) -> Tuple[Mapping[str, object], Path]:
    path = (paths.step_root / EXTRACTION_CONTRACT_RELATIVE_PATH).resolve()
    if not path.is_file():
        raise ValueError(f"Frozen v4 extraction contract is missing: {path}")
    observed = sha256_file(path)
    if observed != EXPECTED_EXTRACTION_CONTRACT_SHA256:
        raise ValueError(
            "V4 extraction contract changed after freezing: "
            f"expected {EXPECTED_EXTRACTION_CONTRACT_SHA256}, observed {observed}"
        )
    contract = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(contract, dict):
        raise ValueError("V4 extraction contract must be an object")
    if contract.get("iteration") != "v4" or contract.get("status") != "frozen_before_extraction":
        raise ValueError("V4 extraction contract has an invalid status")
    scope = contract.get("scope")
    if not isinstance(scope, dict) or scope.get("sealed_test_access") != "forbidden":
        raise ValueError("V4 extraction must forbid sealed-test access")
    return contract, path


def _contract_mapping(contract: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = contract.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"V4 extraction contract field {key!r} must be an object")
    return value


def validate_inputs(
    contract: Mapping[str, object], paths: ProjectPaths, records: Sequence[object]
) -> None:
    inputs = _contract_mapping(contract, "input_contract")
    for key, hash_key in (
        ("development_csv", "development_csv_sha256"),
        ("split_audit", "split_audit_sha256"),
        ("reference_center_cache", "reference_center_cache_sha256"),
    ):
        relative = inputs.get(key)
        expected = inputs.get(hash_key)
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise ValueError(f"V4 extraction input {key!r} is invalid")
        path = (paths.step_root.parent / relative).resolve()
        try:
            path.relative_to(paths.step_root.parent.resolve())
        except ValueError as error:
            raise ValueError(f"V4 extraction input escapes the repository: {key}") from error
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"V4 extraction input hash disagrees: {key}")
    if len(records) != int(inputs.get("development_image_count", -1)):
        raise ValueError("V4 extraction development count disagrees")
    motif_path = STEP_ROOT / "src" / "ornament_classifier" / "motif.py"
    if sha256_file(motif_path) != inputs.get("motif_module_sha256"):
        raise ValueError("V4 motif proposal implementation changed after freezing")


def resolve_weight(contract: Mapping[str, object], override: Path | None) -> Path:
    model = _contract_mapping(contract, "encoder")
    expected_sha = model.get("weight_sha256")
    if model.get("model_id") != MODEL_ID or not isinstance(expected_sha, str):
        raise ValueError("V4 encoder contract is invalid")
    if override is not None:
        path = override.expanduser().resolve()
    else:
        path = (
            Path.home()
            / ".cache"
            / "huggingface"
            / "hub"
            / "models--facebook--dinov3-vits16-pretrain-lvd1689m"
            / "blobs"
            / expected_sha
        )
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise ValueError("The frozen local DINOv3 weight is missing or changed")
    return path


def convert_dinov3_model(weight_path: Path) -> torch.nn.Module:
    """Convert the local Transformers state dict to the equivalent timm graph."""

    with safe_open(str(weight_path), framework="pt", device="cpu") as handle:
        source = {key: handle.get_tensor(key) for key in handle.keys()}
    model = timm.create_model(
        MODEL_ARCHITECTURE,
        pretrained=False,
        num_classes=0,
        img_size=224,
    )
    state: Dict[str, torch.Tensor] = {
        "cls_token": source["embeddings.cls_token"],
        "reg_token": source["embeddings.register_tokens"],
        "patch_embed.proj.weight": source["embeddings.patch_embeddings.weight"],
        "patch_embed.proj.bias": source["embeddings.patch_embeddings.bias"],
        "norm.weight": source["norm.weight"],
        "norm.bias": source["norm.bias"],
    }
    for index in range(12):
        old = f"layer.{index}."
        new = f"blocks.{index}."
        state[new + "gamma_1"] = source[old + "layer_scale1.lambda1"]
        state[new + "gamma_2"] = source[old + "layer_scale2.lambda1"]
        for name in ("norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias"):
            state[new + name] = source[old + name]
        state[new + "attn.q_bias"] = source[old + "attention.q_proj.bias"]
        state[new + "attn.v_bias"] = source[old + "attention.v_proj.bias"]
        state[new + "attn.qkv.weight"] = torch.cat(
            (
                source[old + "attention.q_proj.weight"],
                source[old + "attention.k_proj.weight"],
                source[old + "attention.v_proj.weight"],
            ),
            dim=0,
        )
        state[new + "attn.proj.weight"] = source[old + "attention.o_proj.weight"]
        state[new + "attn.proj.bias"] = source[old + "attention.o_proj.bias"]
        state[new + "mlp.fc1.weight"] = source[old + "mlp.up_proj.weight"]
        state[new + "mlp.fc1.bias"] = source[old + "mlp.up_proj.bias"]
        state[new + "mlp.fc2.weight"] = source[old + "mlp.down_proj.weight"]
        state[new + "mlp.fc2.bias"] = source[old + "mlp.down_proj.bias"]
    model.load_state_dict(state, strict=True)
    model.eval()
    if {parameter.device.type for parameter in model.parameters()} != {"cpu"}:
        raise RuntimeError("V4 motif extraction requires a CPU-only model")
    return model


def images_to_tensor(images: Sequence[Image.Image]) -> torch.Tensor:
    mean = np.asarray(MEAN, dtype=np.float32).reshape(1, 1, 3)
    std = np.asarray(STD, dtype=np.float32).reshape(1, 1, 3)
    arrays = []
    for image in images:
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - mean) / std
        arrays.append(np.transpose(array, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays)).float()


def center_view(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    scale = 256 / min(image.width, image.height)
    resized = image.resize(
        (max(256, round(image.width * scale)), max(256, round(image.height * scale))),
        Image.Resampling.BICUBIC,
    )
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    return resized.crop((left, top, left + 224, top + 224))


def embed_views(model: torch.nn.Module, images: Sequence[Image.Image]) -> np.ndarray:
    with torch.inference_mode():
        features = model.forward_features(images_to_tensor(images))[:, 0]
    values = features.detach().cpu().numpy()
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values = values / np.maximum(norms, 1e-12)
    return values.astype(np.float32)


def validate_conversion(
    model: torch.nn.Module,
    contract: Mapping[str, object],
    paths: ProjectPaths,
    records: Sequence[object],
) -> Mapping[str, object]:
    validation = _contract_mapping(contract, "conversion_validation")
    indices = tuple(int(value) for value in validation.get("development_indices", ()))
    if not indices or any(index < 0 or index >= len(records) for index in indices):
        raise ValueError("V4 conversion validation indices are invalid")
    inputs = _contract_mapping(contract, "input_contract")
    reference_path = paths.step_root.parent / str(inputs["reference_center_cache"])
    with np.load(reference_path, allow_pickle=False) as cached:
        reference = np.asarray(cached["embedding__cls"])[list(indices)]
    images = []
    for index in indices:
        record = records[index]
        with Image.open(paths.resolve_development_image(record.relative_path)) as image:
            images.append(center_view(image))
    observed = embed_views(model, images)
    maximum = float(np.max(np.abs(observed - reference)))
    cosine = np.sum(observed * reference, axis=1)
    tolerance = float(validation.get("maximum_absolute_error", -1))
    if maximum > tolerance or np.any(cosine < float(validation.get("minimum_cosine", 2))):
        raise RuntimeError("Converted DINOv3 graph does not reproduce frozen embeddings")
    return {
        "development_indices": indices,
        "maximum_absolute_error": maximum,
        "cosine_similarity": cosine.tolist(),
        "passed": True,
    }


def verify_image(record: object, paths: ProjectPaths) -> Path:
    path = paths.resolve_development_image(record.relative_path)
    if sha256_file(path) != record.content_sha256:
        raise ValueError(f"Development image bytes changed: {record.image_id}")
    return path


def extract(
    model: torch.nn.Module,
    records: Sequence[object],
    paths: ProjectPaths,
    contract: Mapping[str, object],
    batch_images: int,
) -> Mapping[str, np.ndarray]:
    if batch_images < 1:
        raise ValueError("--batch-images must be positive")
    policy = _contract_mapping(contract, "proposal_policy")
    scales = tuple(float(value) for value in policy.get("scales", ()))
    positions = tuple(float(value) for value in policy.get("positions", ()))
    proposal_count = int(policy.get("proposal_count", 0))
    maximum_iou = float(policy.get("maximum_iou", -1))
    score_size = int(policy.get("score_size", 0))
    input_size = int(_contract_mapping(contract, "encoder").get("input_size", 0))
    if input_size != 224:
        raise ValueError("V4 extractor currently requires the frozen 224-pixel input")

    count = len(records)
    dimension = 384
    tile_embeddings = np.empty((count, proposal_count, dimension), dtype=np.float32)
    boxes = np.empty((count, proposal_count, 4), dtype=np.int32)
    scores = np.empty((count, proposal_count), dtype=np.float64)
    proposal_ids = np.empty((count, proposal_count), dtype="<U24")
    scales_array = np.empty((count, proposal_count), dtype=np.float32)
    positions_array = np.empty((count, proposal_count, 2), dtype=np.float32)
    descriptors = {
        name: np.empty((count, dimension), dtype=np.float32)
        for name in (
            "uniform_mean",
            "texture_weighted_mean",
            "texture_top2_mean",
            "texture_dispersion",
        )
    }

    for start in range(0, count, batch_images):
        batch_records = records[start : start + batch_images]
        views: List[Image.Image] = []
        batch_proposals = []
        for local_index, record in enumerate(batch_records):
            path = verify_image(record, paths)
            with Image.open(path) as opened:
                image = opened.convert("RGB")
                proposals = select_motif_proposals(
                    image,
                    scales=scales,
                    positions=positions,
                    proposal_count=proposal_count,
                    maximum_iou=maximum_iou,
                    score_size=score_size,
                )
                batch_proposals.append(proposals)
                for proposal in proposals:
                    views.append(
                        image.crop(proposal.box).resize(
                            (input_size, input_size), Image.Resampling.BICUBIC
                        )
                    )
        embedded = embed_views(model, views).reshape(
            len(batch_records), proposal_count, dimension
        )
        for local_index, proposals in enumerate(batch_proposals):
            global_index = start + local_index
            tile_embeddings[global_index] = embedded[local_index]
            for proposal_index, proposal in enumerate(proposals):
                boxes[global_index, proposal_index] = proposal.box
                scores[global_index, proposal_index] = proposal.texture_score
                proposal_ids[global_index, proposal_index] = proposal.proposal_id
                scales_array[global_index, proposal_index] = proposal.scale
                positions_array[global_index, proposal_index] = (
                    proposal.x_position,
                    proposal.y_position,
                )
            aggregated = aggregate_tile_embeddings(
                embedded[local_index], scores[global_index]
            )
            for name in descriptors:
                descriptors[name][global_index] = aggregated[name]
        completed = min(start + batch_images, count)
        if completed == count or completed % 100 == 0:
            print(f"V4 motif extraction: {completed}/{count}", flush=True)

    result: Dict[str, np.ndarray] = {
        "image_ids": np.asarray([record.image_id for record in records]),
        "content_sha256s": np.asarray([record.content_sha256 for record in records]),
        "proposal_ids": proposal_ids,
        "proposal_boxes": boxes,
        "proposal_scores": scores,
        "proposal_scales": scales_array,
        "proposal_positions": positions_array,
        "tile_embeddings": tile_embeddings,
    }
    result.update({f"embedding__{name}": values for name, values in descriptors.items()})
    return result


def write_deterministic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        with zipfile.ZipFile(
            temporary, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
        ) as archive:
            for name in sorted(arrays):
                buffer = io.BytesIO()
                np.lib.format.write_array(
                    buffer, np.ascontiguousarray(arrays[name]), allow_pickle=False
                )
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o600 << 16
                archive.writestr(info, buffer.getvalue(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def main() -> None:
    args = parse_args()
    if args.threads < 1:
        raise ValueError("--threads must be positive")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(20260719)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    paths = ProjectPaths.discover()
    contract, contract_path = load_extraction_contract(paths)
    development = load_development_contract(paths)
    records = development.records
    validate_inputs(contract, paths, records)
    weight_path = resolve_weight(contract, args.weight_path)
    model = convert_dinov3_model(weight_path)
    conversion = validate_conversion(model, contract, paths, records)
    arrays = dict(extract(model, records, paths, contract, args.batch_images))
    array_metadata = {
        name: {
            "shape": list(values.shape),
            "dtype": str(values.dtype),
            "sha256": sha256_array(values),
        }
        for name, values in sorted(arrays.items())
    }
    identity = {
        "experiment_version": contract.get("experiment_version"),
        "extraction_contract_sha256": EXPECTED_EXTRACTION_CONTRACT_SHA256,
        "split_assignment_fingerprint_sha256": development.audit.get(
            "assignment_fingerprint_sha256"
        ),
        "image_ids_sha256": sha256_array(arrays["image_ids"]),
        "content_sha256s_sha256": sha256_array(arrays["content_sha256s"]),
        "weight_sha256": sha256_file(weight_path),
        "proposal_policy": contract.get("proposal_policy"),
        "aggregation_policy": contract.get("aggregation_policy"),
        "motif_module_sha256": sha256_file(
            STEP_ROOT / "src" / "ornament_classifier" / "motif.py"
        ),
        "extractor_script_sha256": sha256_file(SCRIPT_PATH),
    }
    fingerprint = hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()
    metadata = {
        "cache_fingerprint_sha256": fingerprint,
        "cache_identity": identity,
        "arrays": array_metadata,
        "conversion_validation": conversion,
        "encoder": contract.get("encoder"),
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
        },
        "device": "cpu",
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "sealed_test_evaluated": False,
    }
    arrays["fingerprint"] = np.asarray(fingerprint)
    arrays["metadata_json"] = np.asarray(stable_json(metadata))
    output_spec = _contract_mapping(contract, "output_contract")
    default_output = paths.step_root.parent / str(output_spec.get("cache_path"))
    output = (args.output or default_output).resolve()
    if args.output is None and output != default_output.resolve():
        raise ValueError("Frozen v4 extraction cache path resolution failed")
    write_deterministic_npz(output, arrays)
    print(
        json.dumps(
            {
                "cache": str(output),
                "cache_sha256": sha256_file(output),
                "cache_fingerprint_sha256": fingerprint,
                "development_image_count": len(records),
                "tile_count": int(len(records) * arrays["tile_embeddings"].shape[1]),
                "conversion_maximum_absolute_error": conversion[
                    "maximum_absolute_error"
                ],
                "sealed_test_evaluated": False,
                "extraction_contract": str(contract_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
