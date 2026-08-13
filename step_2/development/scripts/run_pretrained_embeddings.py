#!/usr/bin/env python3
"""Evaluate frozen pretrained vision embeddings on Phase 2 development data.

This Phase 4 runner accepts only the development manifest. It never reads or
embeds the sealed test manifest. Its selected-representation OOF metrics are
exploratory model-selection diagnostics, not an unbiased performance estimate:
the representation is selected on those same OOF results and the candidate set
and probe C were informed by earlier prototype exploration. Encoder outputs are
cached only after image-byte validation and are keyed by data, model artifacts,
preprocessing, runtime, platform, and deterministic execution policy.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import tempfile
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
DATA_ROOT = STEP_ROOT / "data"
DEFAULT_DEVELOPMENT = STEP_ROOT / "splits" / "development.csv"
DEFAULT_SPLIT_AUDIT = STEP_ROOT / "splits" / "split_audit.json"
DEFAULT_OUTPUT_DIR = STEP_ROOT / "outputs" / "phase_4_pretrained"
DEFAULT_QUICK_OUTPUT_DIR = STEP_ROOT / "outputs" / "phase_4_pretrained_quick"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "step_02" / "pretrained_embeddings"
DEFAULT_TIMM_HUB_CACHE = REPO_ROOT / ".cache" / "step_02" / "huggingface" / "hub"
DEFAULT_DINOV3_SOURCE = "facebook/dinov3-vits16-pretrain-lvd1689m"
SCRIPT_PATH = Path(__file__).resolve()

os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import PIL  # noqa: E402
import sklearn  # noqa: E402
import timm  # noqa: E402
import torch  # noqa: E402
import transformers  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from transformers import AutoModel  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402


PHASE = "step02_phase_4_pretrained_embeddings"
EXPERIMENT_VERSION = "frozen_pretrained_views_v2"
SEED = 20260719
EXPECTED_FOLDS = ("0", "1", "2", "3", "4")
DINO_INPUT_SIZE = 224
DINO_RESIZE_SHORT_SIDE = 256
PROBE_MAX_ITERATIONS = 3000
C_GRID = (0.01, 0.1, 1.0, 10.0)
OUTER_C = 10.0
METRIC_NAMES = (
    "accuracy",
    "balanced_accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "top_3_accuracy",
)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class TimmEncoderSpec:
    alias: str
    model_name: str
    hub_id: str
    weight_filename: str
    pooling_name: str
    license_review: str = "Upstream model card/license review required before deployment"


@dataclass(frozen=True)
class InputPolicy:
    input_size: int
    resize_short_side: int
    crop_pct: float
    crop_mode: str
    interpolation: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    source: str

    def as_dict(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "resize_short_side": self.resize_short_side,
            "crop_pct": self.crop_pct,
            "crop_mode": self.crop_mode,
            "interpolation": self.interpolation,
            "mean": self.mean,
            "std": self.std,
            "source": self.source,
        }


TIMM_ENCODERS = (
    TimmEncoderSpec(
        alias="dinov2_small_reg4",
        model_name="vit_small_patch14_reg4_dinov2.lvd142m",
        hub_id="timm/vit_small_patch14_reg4_dinov2.lvd142m",
        weight_filename="model.safetensors",
        pooling_name="model_pool",
    ),
    TimmEncoderSpec(
        alias="convnext_tiny_in22k",
        model_name="convnext_tiny.fb_in22k_ft_in1k",
        hub_id="timm/convnext_tiny.fb_in22k_ft_in1k",
        weight_filename="model.safetensors",
        pooling_name="model_pool",
    ),
    TimmEncoderSpec(
        alias="mobilenetv3_large",
        model_name="mobilenetv3_large_100.ra_in1k",
        hub_id="timm/mobilenetv3_large_100.ra_in1k",
        weight_filename="model.safetensors",
        pooling_name="model_pool",
    ),
    TimmEncoderSpec(
        alias="clip_vit_b32",
        model_name="vit_base_patch32_clip_quickgelu_224.openai",
        hub_id="timm/vit_base_patch32_clip_224.openai",
        weight_filename="pytorch_model.bin",
        pooling_name="model_pool",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", type=Path, default=DEFAULT_DEVELOPMENT)
    parser.add_argument("--split-audit", type=Path, default=DEFAULT_SPLIT_AUDIT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--timm-hub-cache", type=Path, default=DEFAULT_TIMM_HUB_CACHE)
    parser.add_argument("--dinov3-source", default=DEFAULT_DINOV3_SOURCE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run only DINOv3 center-crop representations (for development checks).",
    )
    args = parser.parse_args()
    if args.output_dir is None:
        args.output_dir = DEFAULT_QUICK_OUTPUT_DIR if args.quick else DEFAULT_OUTPUT_DIR
    if args.quick and args.output_dir.resolve() == DEFAULT_OUTPUT_DIR.resolve():
        parser.error("--quick cannot write to the canonical Phase 4 output directory")
    return args


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def display_path(path: Path) -> str:
    """Return a stable repo-relative path when possible, otherwise an absolute path."""

    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def extraction_runtime_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "Pillow": PIL.__version__,
        "torch": torch.__version__,
        "timm": timm.__version__,
        "transformers": transformers.__version__,
    }


def platform_identity() -> dict[str, str]:
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "architecture": platform.architecture()[0],
        "processor": platform.processor(),
        "python_implementation": platform.python_implementation(),
    }


def configure_cpu_determinism(threads: int) -> dict[str, object]:
    if threads < 1:
        raise ValueError("--threads must be at least 1")
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "mkldnn"):
        torch.backends.mkldnn.deterministic = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if not torch.are_deterministic_algorithms_enabled():
        raise RuntimeError("PyTorch deterministic algorithms could not be enabled")
    return {
        "requested": True,
        "device": "cpu",
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "torch_float32_matmul_precision": torch.get_float32_matmul_precision(),
        "torch_threads": threads,
        "torch_interop_threads": torch.get_num_interop_threads(),
        "sklearn_blas_threads_during_fit": 1,
        "numpy_seed": SEED,
        "torch_seed": SEED,
        "scope": (
            "Deterministic execution is asserted for this CPU/runtime; bitwise identity "
            "is not claimed across different architectures or library builds."
        ),
    }


def execution_identity(
    batch_size: int, threads: int, determinism: dict[str, object]
) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    return {
        "requested_batch_size": batch_size,
        "threads": threads,
        "device": "cpu",
        "platform": platform_identity(),
        "determinism": determinism,
    }


def effective_batch_size(requested_batch_size: int, input_size: int) -> int:
    """Conservatively cap large-ViT batches as attention grows quadratically in tokens."""

    scale = (DINO_INPUT_SIZE / input_size) ** 4
    return max(1, min(requested_batch_size, int(requested_batch_size * scale)))


def expected_development_sha256(split_audit: dict[str, object], path: Path) -> str:
    for audit_key in ("output_file_sha256", "generated_csv_sha256"):
        recorded = split_audit.get(audit_key)
        if isinstance(recorded, str):
            return recorded
        if isinstance(recorded, dict):
            candidates = (
                path.name,
                "development.csv",
                "development",
                display_path(path),
                str(path),
            )
            for candidate in candidates:
                value = recorded.get(candidate)
                if isinstance(value, str):
                    return value
    raise ValueError(
        "split_audit.json must provide output_file_sha256 or generated_csv_sha256 "
        "for development.csv"
    )


def crossing_identifiers(
    rows: list[dict[str, str]], identifier: str, boundary: str, allow_blank: bool = False
) -> list[str]:
    assignments: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        value = row[identifier]
        if not value:
            if allow_blank:
                continue
            raise ValueError(f"Development row {row['image_id']} has blank {identifier}")
        assignments[value].add(row[boundary])
    return sorted(value for value, observed in assignments.items() if len(observed) > 1)


def validate_development(
    rows: list[dict[str, str]], path: Path, split_audit: dict[str, object]
) -> dict[str, object]:
    expected_sha = expected_development_sha256(split_audit, path)
    observed_sha = sha256_file(path)
    if observed_sha != expected_sha:
        raise ValueError(
            "Development CSV SHA-256 does not match split_audit.json: "
            f"expected {expected_sha}, observed {observed_sha}"
        )

    required_fields = {
        "image_id",
        "relative_path",
        "ornament_label",
        "content_sha256",
        "confirmed_object_group_id",
        "confirmed_source_group_id",
        "split_group_id",
        "inclusion_status",
        "production_split",
        "cv_fold",
        "split_version",
        "split_seed",
    }
    if not rows:
        raise ValueError("Development input is empty")
    missing_fields = required_fields - set(rows[0])
    if missing_fields:
        raise ValueError(f"Development input is missing fields: {sorted(missing_fields)}")

    production_counts = split_audit.get("production_split_counts")
    if not isinstance(production_counts, dict):
        raise ValueError("split_audit.json is missing production_split_counts")
    expected_count = sum(int(production_counts[name]) for name in ("train", "validation"))
    if len(rows) != expected_count:
        raise ValueError(f"Expected {expected_count:,} development rows, found {len(rows):,}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Development image IDs are not unique")
    if {row["production_split"] for row in rows} != {"train", "validation"}:
        raise ValueError("Input must contain only Phase 2 train and validation rows")
    if {row["cv_fold"] for row in rows} != set(EXPECTED_FOLDS):
        raise ValueError("Input does not contain all five expected CV folds")
    if any(row["inclusion_status"] != "included" for row in rows):
        raise ValueError("Input contains an excluded image")

    expected_version = str(split_audit["split_version"])
    expected_seed = str(split_audit["split_seed"])
    if {row["split_version"] for row in rows} != {expected_version}:
        raise ValueError("Development split_version does not match split_audit.json")
    if {row["split_seed"] for row in rows} != {expected_seed}:
        raise ValueError("Development split_seed does not match split_audit.json")

    class_counts = split_audit.get("class_counts")
    if not isinstance(class_counts, dict) or not class_counts:
        raise ValueError("split_audit.json is missing class_counts")
    expected_classes = set(class_counts)
    if {row["ornament_label"] for row in rows} != expected_classes:
        raise ValueError("Development class set does not match split_audit.json")
    for boundary, values in (
        ("production_split", ("train", "validation")),
        ("cv_fold", EXPECTED_FOLDS),
    ):
        for value in values:
            observed_classes = {
                row["ornament_label"] for row in rows if row[boundary] == value
            }
            if observed_classes != expected_classes:
                raise ValueError(
                    f"{boundary}={value} does not contain every expected class"
                )

    identifier_policies = [
        ("split_group_id", False),
        ("confirmed_object_group_id", False),
        ("confirmed_source_group_id", True),
        ("content_sha256", False),
    ]
    optional_identifier_policies = (
        ("global_source_cohort_id", True),
        ("source_atomic_cohort_ids", True),
        ("pre_source_cohort_split_group_id", False),
        ("source_atomic_split_group_id", False),
    )
    for optional_identifier, allow_blank in optional_identifier_policies:
        if optional_identifier in rows[0]:
            identifier_policies.append((optional_identifier, allow_blank))
    for identifier, allow_blank in identifier_policies:
        for boundary in ("production_split", "cv_fold"):
            violations = crossing_identifiers(rows, identifier, boundary, allow_blank)
            if violations:
                raise ValueError(
                    f"{identifier} crosses {boundary}: {violations[:5]}"
                )

    data_root = DATA_ROOT.resolve()
    observed_hashes: list[tuple[str, str]] = []
    for row in rows:
        image_path = (DATA_ROOT / row["relative_path"]).resolve()
        try:
            image_path.relative_to(data_root)
        except ValueError as error:
            raise ValueError(
                f"Development image escapes the data root: {row['relative_path']}"
            ) from error
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing development image: {row['relative_path']}")
        observed_hash = sha256_file(image_path)
        if observed_hash != row["content_sha256"]:
            raise ValueError(
                "Development image byte hash does not match the manifest for "
                f"{row['image_id']}: expected {row['content_sha256']}, "
                f"observed {observed_hash}"
            )
        observed_hashes.append((row["image_id"], observed_hash))

    byte_digest = hashlib.sha256()
    for image_id, observed_hash in observed_hashes:
        byte_digest.update(f"{image_id}\x1f{observed_hash}\n".encode("utf-8"))
    return {
        "all_image_bytes_sha256_verified": True,
        "verified_image_count": len(observed_hashes),
        "verified_image_fingerprint_sha256": byte_digest.hexdigest(),
    }


def resize_short_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    scale = size / min(width, height)
    resized = (max(size, round(width * scale)), max(size, round(height * scale)))
    return image.resize(resized, Image.Resampling.BICUBIC)


def center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = (width - size) // 2
    top = (height - size) // 2
    return image.crop((left, top, left + size, top + size))


def five_crops(image: Image.Image, size: int) -> list[Image.Image]:
    width, height = image.size
    positions = (
        (0, 0),
        (width - size, 0),
        (0, height - size),
        (width - size, height - size),
        ((width - size) // 2, (height - size) // 2),
    )
    return [image.crop((left, top, left + size, top + size)) for left, top in positions]


def letterbox(image: Image.Image, size: int, mean: tuple[float, float, float]) -> Image.Image:
    width, height = image.size
    scale = size / max(width, height)
    resized_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    resized = image.resize(resized_size, Image.Resampling.BICUBIC)
    fill = tuple(round(channel * 255) for channel in mean)
    canvas = Image.new("RGB", (size, size), fill)
    offset = ((size - resized.width) // 2, (size - resized.height) // 2)
    canvas.paste(resized, offset)
    return canvas


def image_views(
    image: Image.Image,
    view_mode: str,
    mean: tuple[float, float, float],
    input_size: int,
    resized_short_side: int,
) -> list[Image.Image]:
    image = image.convert("RGB")
    if view_mode == "center_crop":
        return [center_crop(resize_short_side(image, resized_short_side), input_size)]
    if view_mode == "letterbox":
        return [letterbox(image, input_size, mean)]
    if view_mode == "global_fivecrop":
        resized = resize_short_side(image, resized_short_side)
        return [letterbox(image, input_size, mean), *five_crops(resized, input_size)]
    raise ValueError(f"Unsupported view mode: {view_mode}")


def images_to_tensor(
    images: list[Image.Image],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
) -> torch.Tensor:
    arrays = []
    mean_array = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
    std_array = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)
    for image in images:
        array = np.asarray(image, dtype=np.float32) / 255.0
        array = (array - mean_array) / std_array
        arrays.append(np.transpose(array, (2, 0, 1)))
    return torch.from_numpy(np.stack(arrays)).float()


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def aggregate_views(values: np.ndarray, image_count: int, views_per_image: int) -> np.ndarray:
    values = normalize_rows(values)
    values = values.reshape(image_count, views_per_image, values.shape[1]).mean(axis=1)
    return normalize_rows(values).astype(np.float32)


def assert_cpu_model(model: torch.nn.Module) -> None:
    devices = {parameter.device.type for parameter in model.parameters()}
    if devices != {"cpu"}:
        raise RuntimeError(f"Phase 4 deterministic extraction requires CPU-only parameters: {devices}")


def data_fingerprint(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(f'{row["image_id"]}\x1f{row["content_sha256"]}\n'.encode("utf-8"))
    return digest.hexdigest()


def dinov3_model_provenance(model_source: str) -> dict[str, object]:
    weight = resolve_dinov3_weight(model_source)
    snapshot = weight.parent
    config = snapshot / "config.json"
    preprocessor = snapshot / "preprocessor_config.json"
    if not config.is_file() or not preprocessor.is_file():
        raise FileNotFoundError(
            "DINOv3 cache must include config.json and preprocessor_config.json "
            f"beside the weights: {snapshot}"
        )
    return {
        "model_source": model_source,
        "revision": snapshot.name,
        "weight_sha256": sha256_file(weight),
        "model_config_sha256": sha256_file(config),
        "preprocessor_config_sha256": sha256_file(preprocessor),
    }


def timm_pretrained_config(spec: TimmEncoderSpec) -> dict[str, object]:
    config = timm.get_pretrained_cfg(spec.model_name)
    if config is None:
        raise ValueError(f"timm has no pretrained configuration for {spec.model_name}")
    return config.to_dict()


def timm_input_policy(spec: TimmEncoderSpec) -> InputPolicy:
    config = timm_pretrained_config(spec)
    input_shape = tuple(int(value) for value in config["input_size"])
    if len(input_shape) != 3 or input_shape[0] != 3 or input_shape[1] != input_shape[2]:
        raise ValueError(
            f"Only square three-channel timm inputs are supported, found {input_shape}"
        )
    input_size = input_shape[1]
    crop_pct = float(config.get("crop_pct", 1.0))
    if not 0 < crop_pct <= 1:
        raise ValueError(f"Invalid timm crop_pct for {spec.model_name}: {crop_pct}")
    interpolation = str(config.get("interpolation", "bicubic"))
    if interpolation != "bicubic":
        raise ValueError(
            f"Only bicubic native interpolation is implemented, found {interpolation}"
        )
    crop_mode = str(config.get("crop_mode", "center"))
    if crop_mode != "center":
        raise ValueError(
            f"Only native center-crop preprocessing is implemented, found {crop_mode}"
        )
    return InputPolicy(
        input_size=input_size,
        resize_short_side=max(input_size, int(input_size / crop_pct)),
        crop_pct=crop_pct,
        crop_mode=crop_mode,
        interpolation=interpolation,
        mean=tuple(float(value) for value in config["mean"]),
        std=tuple(float(value) for value in config["std"]),
        source="timm.get_pretrained_cfg(model_name)",
    )


def timm_model_provenance(spec: TimmEncoderSpec, hub_cache: Path) -> dict[str, object]:
    weight = resolve_timm_weight(spec, hub_cache)
    config = timm_pretrained_config(spec)
    policy = timm_input_policy(spec)
    model_configuration = {
        "model_name": spec.model_name,
        "architecture": config.get("architecture"),
        "num_classes_removed": True,
    }
    return {
        "model_source": spec.hub_id,
        "revision": weight.parent.name,
        "weight_sha256": sha256_file(weight),
        "model_config_sha256": hashlib.sha256(
            stable_json(model_configuration).encode("utf-8")
        ).hexdigest(),
        "pretrained_config_sha256": hashlib.sha256(
            stable_json(config).encode("utf-8")
        ).hexdigest(),
        "preprocessor_config_sha256": hashlib.sha256(
            stable_json(policy.as_dict()).encode("utf-8")
        ).hexdigest(),
    }


def embedding_identity(
    rows: list[dict[str, str]],
    encoder_alias: str,
    model_provenance: dict[str, object],
    view_mode: str,
    input_policy: InputPolicy,
    execution: dict[str, object],
) -> dict[str, object]:
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "extraction_runtime": extraction_runtime_versions(),
        "encoder_alias": encoder_alias,
        "model_provenance": model_provenance,
        "view_mode": view_mode,
        "input_policy": input_policy.as_dict(),
        "execution": execution,
        "data_fingerprint": data_fingerprint(rows),
    }


def embedding_fingerprint(identity: dict[str, object]) -> str:
    return hashlib.sha256(stable_json(identity).encode("utf-8")).hexdigest()


def load_embedding_cache(
    path: Path,
    rows: list[dict[str, str]],
    expected_fingerprint: str,
    expected_keys: set[str],
    expected_identity: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as cached:
        required = {"fingerprint", "image_ids", "metadata_json"}
        if not required.issubset(cached.files):
            raise ValueError(f"Incomplete embedding cache: {path}")
        if str(cached["fingerprint"].item()) != expected_fingerprint:
            return None
        image_ids = [str(value) for value in cached["image_ids"]]
        if image_ids != [row["image_id"] for row in rows]:
            raise ValueError(f"Embedding cache image IDs do not match development input: {path}")
        metadata = json.loads(str(cached["metadata_json"].item()))
        embeddings = {
            key.removeprefix("embedding__"): cached[key]
            for key in cached.files
            if key.startswith("embedding__")
        }
    if set(embeddings) != expected_keys:
        raise ValueError(
            f"Embedding cache keys do not match {sorted(expected_keys)}: {path}"
        )
    if not isinstance(metadata, dict):
        raise ValueError(f"Embedding cache metadata is not an object: {path}")
    if metadata.get("embedding_fingerprint_sha256") != expected_fingerprint:
        raise ValueError(f"Embedding cache metadata fingerprint is invalid: {path}")
    if metadata.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError(f"Embedding cache experiment version is invalid: {path}")
    if metadata.get("script_sha256") != sha256_file(SCRIPT_PATH):
        raise ValueError(f"Embedding cache script hash is invalid: {path}")
    if metadata.get("extraction_runtime") != extraction_runtime_versions():
        raise ValueError(f"Embedding cache runtime metadata is invalid: {path}")
    if stable_json(metadata.get("cache_identity")) != stable_json(expected_identity):
        raise ValueError(f"Embedding cache identity metadata is invalid: {path}")
    for name, values in embeddings.items():
        if values.ndim != 2 or values.shape[0] != len(rows) or values.shape[1] <= 0:
            raise ValueError(f"Invalid cached {name} shape {values.shape}: {path}")
        if values.dtype != np.float32:
            raise ValueError(f"Cached {name} must be float32, found {values.dtype}: {path}")
        if not np.isfinite(values).all():
            raise ValueError(f"Cached {name} contains non-finite values: {path}")
        norms = np.linalg.norm(values, axis=1)
        if not np.allclose(norms, 1.0, rtol=5e-4, atol=5e-4):
            raise ValueError(f"Cached {name} rows are not unit normalized: {path}")
    if expected_keys == {"cls", "patch_mean", "cls_patch_concat"}:
        cls_dimension = embeddings["cls"].shape[1]
        if embeddings["patch_mean"].shape[1] != cls_dimension:
            raise ValueError(f"DINO cached pooling dimensions disagree: {path}")
        if embeddings["cls_patch_concat"].shape[1] != 2 * cls_dimension:
            raise ValueError(f"DINO cached concatenation dimension is invalid: {path}")
    return embeddings, metadata


def save_embedding_cache(
    path: Path,
    rows: list[dict[str, str]],
    fingerprint: str,
    embeddings: dict[str, np.ndarray],
    metadata: dict[str, object],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {f"embedding__{name}": values for name, values in embeddings.items()}
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary_path = Path(handle.name)
            np.savez_compressed(
                handle,
                fingerprint=np.asarray(fingerprint),
                image_ids=np.asarray([row["image_id"] for row in rows]),
                metadata_json=np.asarray(stable_json(metadata)),
                **arrays,
            )
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def resolve_timm_weight(spec: TimmEncoderSpec, hub_cache: Path) -> Path:
    repository = hub_cache / f'models--{spec.hub_id.replace("/", "--")}'
    ref = repository / "refs" / "main"
    if not ref.is_file():
        raise FileNotFoundError(f"Missing cached revision for {spec.hub_id}: {ref}")
    revision = ref.read_text(encoding="utf-8").strip()
    weight = repository / "snapshots" / revision / spec.weight_filename
    if not weight.is_file():
        raise FileNotFoundError(f"Missing cached weights for {spec.hub_id}: {weight}")
    return weight


def resolve_dinov3_weight(model_source: str) -> Path:
    source = Path(model_source)
    if source.exists():
        weight = source / "model.safetensors" if source.is_dir() else source
        if weight.is_file():
            return weight.resolve()
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(model_source, "model.safetensors")
    if not isinstance(cached, str):
        raise FileNotFoundError(
            f"DINOv3 weights are not cached for {model_source}; access to the gated model is required"
        )
    # Keep the snapshot path (rather than resolving its blob symlink) so the
    # immutable revision and adjacent config artifacts remain discoverable.
    return Path(cached)


def extract_dinov3(
    rows: list[dict[str, str]],
    model_source: str,
    view_mode: str,
    batch_size: int,
    execution: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    weight_path = resolve_dinov3_weight(model_source)
    weight_sha = sha256_file(weight_path)
    provenance = dinov3_model_provenance(model_source)
    input_policy = InputPolicy(
        input_size=DINO_INPUT_SIZE,
        resize_short_side=DINO_RESIZE_SHORT_SIDE,
        crop_pct=DINO_INPUT_SIZE / DINO_RESIZE_SHORT_SIDE,
        crop_mode="project_multiview",
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        source="project DINOv3 multi-view policy; native normalization and input size",
    )
    actual_batch_size = effective_batch_size(batch_size, input_policy.input_size)
    model = AutoModel.from_pretrained(
        str(weight_path.parent), local_files_only=True
    ).eval()
    assert_cpu_model(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    register_count = int(model.config.num_register_tokens)
    collected: dict[str, list[np.ndarray]] = {
        "cls": [],
        "patch_mean": [],
        "cls_patch_concat": [],
    }
    start = time.perf_counter()
    view_count = 1 if view_mode in {"center_crop", "letterbox"} else 6
    with torch.inference_mode():
        for start_index in range(0, len(rows), actual_batch_size):
            batch_rows = rows[start_index : start_index + actual_batch_size]
            views: list[Image.Image] = []
            for row in batch_rows:
                with Image.open(DATA_ROOT / row["relative_path"]) as image:
                    views.extend(
                        image_views(
                            image,
                            view_mode,
                            input_policy.mean,
                            input_policy.input_size,
                            input_policy.resize_short_side,
                        )
                    )
            inputs = images_to_tensor(views, input_policy.mean, input_policy.std)
            outputs = model(pixel_values=inputs)
            cls = outputs.pooler_output.detach().cpu().numpy()
            patch_start = 1 + register_count
            patch_mean = outputs.last_hidden_state[:, patch_start:].mean(dim=1).detach().cpu().numpy()
            concat = np.concatenate([cls, patch_mean], axis=1)
            for name, values in (
                ("cls", cls),
                ("patch_mean", patch_mean),
                ("cls_patch_concat", concat),
            ):
                collected[name].append(aggregate_views(values, len(batch_rows), view_count))
            completed = min(start_index + batch_size, len(rows))
            if completed == len(rows) or completed % 320 == 0:
                print(f"  DINOv3 {view_mode}: {completed}/{len(rows)}", flush=True)
    seconds = time.perf_counter() - start
    embeddings = {name: np.vstack(parts) for name, parts in collected.items()}
    metadata = {
        "encoder_alias": "dinov3_vits16",
        "model_id": model_source,
        "model_source": model_source,
        "license": "DINOv3 License (custom; deployment review required)",
        "weight_sha256": weight_sha,
        "model_revision": provenance["revision"],
        "model_config_sha256": provenance["model_config_sha256"],
        "preprocessor_config_sha256": provenance["preprocessor_config_sha256"],
        "parameter_count": parameter_count,
        "view_mode": view_mode,
        "views_per_image": view_count,
        "effective_batch_size": actual_batch_size,
        "input_policy": input_policy.as_dict(),
        "execution_identity": execution,
        "embedding_dimensions": {name: int(values.shape[1]) for name, values in embeddings.items()},
        "nondeterministic_observations": {
            "extraction_seconds": seconds,
            "images_per_second": len(rows) / seconds,
        },
        "experiment_version": EXPERIMENT_VERSION,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "extraction_runtime": extraction_runtime_versions(),
    }
    return embeddings, metadata, weight_sha


def extract_timm(
    rows: list[dict[str, str]],
    spec: TimmEncoderSpec,
    hub_cache: Path,
    batch_size: int,
    execution: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], str]:
    weight_path = resolve_timm_weight(spec, hub_cache)
    weight_sha = sha256_file(weight_path)
    provenance = timm_model_provenance(spec, hub_cache)
    input_policy = timm_input_policy(spec)
    actual_batch_size = effective_batch_size(batch_size, input_policy.input_size)
    arguments: dict[str, object] = {
        "pretrained": True,
        "num_classes": 0,
        "pretrained_cfg_overlay": {"file": str(weight_path)},
    }
    model = timm.create_model(spec.model_name, **arguments).eval()
    assert_cpu_model(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    collected: list[np.ndarray] = []
    start = time.perf_counter()
    with torch.inference_mode():
        for start_index in range(0, len(rows), actual_batch_size):
            batch_rows = rows[start_index : start_index + actual_batch_size]
            views: list[Image.Image] = []
            for row in batch_rows:
                with Image.open(DATA_ROOT / row["relative_path"]) as image:
                    views.extend(
                        image_views(
                            image,
                            "center_crop",
                            input_policy.mean,
                            input_policy.input_size,
                            input_policy.resize_short_side,
                        )
                    )
            inputs = images_to_tensor(views, input_policy.mean, input_policy.std)
            outputs = model(inputs)
            if isinstance(outputs, (tuple, list)):
                outputs = outputs[0]
            values = outputs.detach().cpu().numpy()
            if values.ndim > 2:
                values = values.reshape(values.shape[0], -1)
            collected.append(normalize_rows(values).astype(np.float32))
            completed = min(start_index + batch_size, len(rows))
            if completed == len(rows) or completed % 320 == 0:
                print(f"  {spec.alias}: {completed}/{len(rows)}", flush=True)
    seconds = time.perf_counter() - start
    embeddings = {spec.pooling_name: np.vstack(collected)}
    metadata = {
        "encoder_alias": spec.alias,
        "model_id": spec.model_name,
        "model_source": spec.hub_id,
        "hub_id": spec.hub_id,
        "license": spec.license_review,
        "weight_sha256": weight_sha,
        "model_revision": provenance["revision"],
        "model_config_sha256": provenance["model_config_sha256"],
        "pretrained_config_sha256": provenance["pretrained_config_sha256"],
        "preprocessor_config_sha256": provenance["preprocessor_config_sha256"],
        "parameter_count": parameter_count,
        "view_mode": "center_crop",
        "views_per_image": 1,
        "effective_batch_size": actual_batch_size,
        "input_policy": input_policy.as_dict(),
        "execution_identity": execution,
        "embedding_dimensions": {
            spec.pooling_name: int(embeddings[spec.pooling_name].shape[1])
        },
        "nondeterministic_observations": {
            "extraction_seconds": seconds,
            "images_per_second": len(rows) / seconds,
        },
        "experiment_version": EXPERIMENT_VERSION,
        "script_sha256": sha256_file(SCRIPT_PATH),
        "extraction_runtime": extraction_runtime_versions(),
    }
    return embeddings, metadata, weight_sha


def get_or_extract_dinov3(
    rows: list[dict[str, str]],
    model_source: str,
    view_mode: str,
    cache_dir: Path,
    batch_size: int,
    use_cache: bool,
    execution: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], Path]:
    provenance = dinov3_model_provenance(model_source)
    input_policy = InputPolicy(
        input_size=DINO_INPUT_SIZE,
        resize_short_side=DINO_RESIZE_SHORT_SIDE,
        crop_pct=DINO_INPUT_SIZE / DINO_RESIZE_SHORT_SIDE,
        crop_mode="project_multiview",
        interpolation="bicubic",
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        source="project DINOv3 multi-view policy; native normalization and input size",
    )
    identity = embedding_identity(
        rows,
        "dinov3_vits16",
        provenance,
        view_mode,
        input_policy,
        execution,
    )
    fingerprint = embedding_fingerprint(identity)
    cache_path = cache_dir / f"dinov3_vits16__{view_mode}__{fingerprint[:16]}.npz"
    if use_cache:
        cached = load_embedding_cache(
            cache_path,
            rows,
            fingerprint,
            {"cls", "patch_mean", "cls_patch_concat"},
            identity,
        )
        if cached is not None:
            print(f"Loaded cache: {cache_path.name}", flush=True)
            return cached[0], cached[1], cache_path
    print(f"Extracting DINOv3 {view_mode}", flush=True)
    embeddings, metadata, _ = extract_dinov3(
        rows, model_source, view_mode, batch_size, execution
    )
    metadata["embedding_fingerprint_sha256"] = fingerprint
    metadata["cache_identity"] = identity
    if use_cache:
        save_embedding_cache(cache_path, rows, fingerprint, embeddings, metadata)
    return embeddings, metadata, cache_path


def get_or_extract_timm(
    rows: list[dict[str, str]],
    spec: TimmEncoderSpec,
    hub_cache: Path,
    cache_dir: Path,
    batch_size: int,
    use_cache: bool,
    execution: dict[str, object],
) -> tuple[dict[str, np.ndarray], dict[str, object], Path]:
    provenance = timm_model_provenance(spec, hub_cache)
    input_policy = timm_input_policy(spec)
    identity = embedding_identity(
        rows,
        spec.alias,
        provenance,
        "center_crop",
        input_policy,
        execution,
    )
    fingerprint = embedding_fingerprint(identity)
    cache_path = cache_dir / f"{spec.alias}__center_crop__{fingerprint[:16]}.npz"
    if use_cache:
        cached = load_embedding_cache(
            cache_path, rows, fingerprint, {spec.pooling_name}, identity
        )
        if cached is not None:
            print(f"Loaded cache: {cache_path.name}", flush=True)
            return cached[0], cached[1], cache_path
    print(f"Extracting {spec.alias} center_crop", flush=True)
    embeddings, metadata, _ = extract_timm(
        rows, spec, hub_cache, batch_size, execution
    )
    metadata["embedding_fingerprint_sha256"] = fingerprint
    metadata["cache_identity"] = identity
    if use_cache:
        save_embedding_cache(cache_path, rows, fingerprint, embeddings, metadata)
    return embeddings, metadata, cache_path


def representation_name(encoder: str, pooling: str, view_mode: str) -> str:
    return f"{encoder}__{pooling}__{view_mode}"


def calculate_metrics(
    truth: np.ndarray,
    predicted: np.ndarray,
    probabilities: np.ndarray,
    class_count: int,
) -> dict[str, float]:
    labels = np.arange(class_count)
    precision, recall, f1, _ = precision_recall_fscore_support(
        truth,
        predicted,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    top_three = np.argsort(-probabilities, axis=1, kind="stable")[:, :3]
    return {
        "accuracy": float(accuracy_score(truth, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predicted)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top_3_accuracy": float(np.mean(np.any(top_three == truth[:, None], axis=1))),
    }


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    c_value: float,
) -> LogisticRegression:
    probe = LogisticRegression(
        C=c_value,
        class_weight="balanced",
        max_iter=PROBE_MAX_ITERATIONS,
        random_state=SEED,
        solver="lbfgs",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        try:
            with threadpool_limits(limits=1):
                probe.fit(features[train_indices], labels[train_indices])
        except ConvergenceWarning as error:
            raise RuntimeError(
                f"Logistic probe failed to converge within {PROBE_MAX_ITERATIONS} iterations"
            ) from error
    if not hasattr(probe, "n_iter_") or np.any(probe.n_iter_ >= PROBE_MAX_ITERATIONS):
        raise RuntimeError(
            "Logistic probe convergence could not be established from n_iter_"
        )
    return probe


def probe_iterations(probe: LogisticRegression) -> int:
    return int(np.max(probe.n_iter_))


def tune_probes(
    representations: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    labels: np.ndarray,
    class_count: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    production = np.asarray([row["production_split"] for row in rows])
    train_indices = np.flatnonzero(production == "train")
    validation_indices = np.flatnonzero(production == "validation")
    screen_rows: list[dict[str, object]] = []
    diagnostic_best_c: dict[str, float] = {}
    for name in sorted(representations):
        features = representations[name]
        candidates: list[dict[str, object]] = []
        for c_value in C_GRID:
            probe = fit_probe(features, labels, train_indices, c_value)
            predicted = probe.predict(features[validation_indices]).astype(np.int64)
            probabilities = probe.predict_proba(features[validation_indices])
            metrics = calculate_metrics(
                labels[validation_indices], predicted, probabilities, class_count
            )
            row = {
                "representation": name,
                "metric_status": "diagnostic_production_holdout",
                "c_value": c_value,
                "diagnostic_only": "True",
                "train_examples": len(train_indices),
                "validation_examples": len(validation_indices),
                "iterations": probe_iterations(probe),
                "max_iterations": PROBE_MAX_ITERATIONS,
                "converged": "True",
                **metrics,
            }
            candidates.append(row)
            screen_rows.append(row)
        best = sorted(
            candidates,
            key=lambda row: (
                -float(row["accuracy"]),
                -float(row["macro_f1"]),
                float(row["c_value"]),
            ),
        )[0]
        diagnostic_best_c[name] = float(best["c_value"])
        print(
            f"Screen {name}: C={best['c_value']:g}, "
            f"accuracy={best['accuracy']:.4f}, macro_f1={best['macro_f1']:.4f}",
            flush=True,
        )
    return screen_rows, diagnostic_best_c


def cross_validate(
    representations: dict[str, np.ndarray],
    rows: list[dict[str, str]],
    labels: np.ndarray,
    class_count: int,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, tuple[np.ndarray, np.ndarray]],
]:
    folds = np.asarray([row["cv_fold"] for row in rows])
    fold_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    predictions_by_representation: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for name in sorted(representations):
        features = representations[name]
        oof_predicted = np.full(len(rows), -1, dtype=np.int64)
        oof_probabilities = np.zeros((len(rows), class_count), dtype=np.float64)
        for fold in EXPECTED_FOLDS:
            validation_indices = np.flatnonzero(folds == fold)
            train_indices = np.flatnonzero(folds != fold)
            probe = fit_probe(features, labels, train_indices, OUTER_C)
            predicted = probe.predict(features[validation_indices]).astype(np.int64)
            probabilities = probe.predict_proba(features[validation_indices])
            oof_predicted[validation_indices] = predicted
            oof_probabilities[validation_indices] = probabilities
            metrics = calculate_metrics(
                labels[validation_indices], predicted, probabilities, class_count
            )
            fold_rows.append(
                {
                    "representation": name,
                    "metric_status": "exploratory_candidate_cv",
                    "c_value": OUTER_C,
                    "cv_fold": fold,
                    "train_examples": len(train_indices),
                    "validation_examples": len(validation_indices),
                    "iterations": probe_iterations(probe),
                    "max_iterations": PROBE_MAX_ITERATIONS,
                    "converged": "True",
                    **metrics,
                }
            )
        if np.any(oof_predicted < 0):
            raise RuntimeError(f"Missing OOF predictions for {name}")
        predictions_by_representation[name] = (oof_predicted, oof_probabilities)
        aggregate = calculate_metrics(labels, oof_predicted, oof_probabilities, class_count)
        per_class = precision_recall_fscore_support(
            labels,
            oof_predicted,
            labels=np.arange(class_count),
            zero_division=0,
        )
        recalls = per_class[1]
        candidate_folds = [row for row in fold_rows if row["representation"] == name]
        summary: dict[str, object] = {
            "representation": name,
            "metric_status": "exploratory_selection_conditional",
            "c_value": OUTER_C,
            "fold_count": len(candidate_folds),
            "aggregate_accuracy": aggregate["accuracy"],
            "aggregate_macro_f1": aggregate["macro_f1"],
            "aggregate_balanced_accuracy": aggregate["balanced_accuracy"],
            "aggregate_top_3_accuracy": aggregate["top_3_accuracy"],
            "worst_class_recall": float(recalls.min()),
            "all_folds_converged": "True",
            "max_fold_iterations": max(int(row["iterations"]) for row in candidate_folds),
        }
        for metric in METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in candidate_folds])
            summary[f"mean_{metric}"] = float(values.mean())
            summary[f"std_{metric}"] = float(values.std(ddof=1))
        summaries.append(summary)
        print(
            f"CV {name}: accuracy={summary['mean_accuracy']:.4f}±{summary['std_accuracy']:.4f}, "
            f"macro_f1={summary['mean_macro_f1']:.4f}",
            flush=True,
        )
    return fold_rows, summaries, predictions_by_representation


def select_representation(summaries: list[dict[str, object]]) -> str:
    best_accuracy = max(float(row["mean_accuracy"]) for row in summaries)
    near_ties = [row for row in summaries if float(row["mean_accuracy"]) >= best_accuracy - 0.005]
    selected = sorted(
        near_ties,
        key=lambda row: (
            -float(row["mean_macro_f1"]),
            -float(row["worst_class_recall"]),
            str(row["representation"]),
        ),
    )[0]
    ordered = sorted(
        summaries,
        key=lambda row: (
            -float(row["mean_accuracy"]),
            -float(row["mean_macro_f1"]),
            str(row["representation"]),
        ),
    )
    for rank, row in enumerate(ordered, start=1):
        row["accuracy_rank"] = rank
        row["within_half_point_of_best"] = str(
            float(row["mean_accuracy"]) >= best_accuracy - 0.005
        )
        row["selected"] = str(row is selected)
    summaries[:] = ordered
    return str(selected["representation"])


def per_class_rows(
    truth: np.ndarray, predicted: np.ndarray, class_names: list[str]
) -> list[dict[str, object]]:
    precision, recall, f1, support = precision_recall_fscore_support(
        truth,
        predicted,
        labels=np.arange(len(class_names)),
        zero_division=0,
    )
    return [
        {
            "class_index": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(class_names)
    ]


def diagnostic_slice_rows(
    rows: list[dict[str, str]], predicted: np.ndarray, probabilities: np.ndarray, labels: np.ndarray
) -> list[dict[str, object]]:
    dimensions: dict[str, list[str]] = {
        "old_split": [row["old_split"] for row in rows],
        "review_cohort": [
            "manually_reviewed" if row["manual_review_status"] == "complete" else "singleton"
            for row in rows
        ],
        "motif_visibility": [row["motif_visibility"] for row in rows],
        "object_type": [row["object_type"] for row in rows],
    }
    cohort_field = (
        "source_atomic_cohort_ids"
        if "source_atomic_cohort_ids" in rows[0]
        else "global_source_cohort_id"
    )
    if cohort_field in rows[0]:
        dimensions["global_source_cohort"] = [
            row[cohort_field] or "unassigned" for row in rows
        ]
    dimensions["cv_fold"] = [row["cv_fold"] for row in rows]
    uncalibrated_max_probability = probabilities.max(axis=1)
    result: list[dict[str, object]] = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            indices = np.asarray([index for index, observed in enumerate(values) if observed == value])
            if len(indices) < 5:
                continue
            correct = predicted[indices] == labels[indices]
            result.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "image_count": len(indices),
                    "correct_count": int(correct.sum()),
                    "error_count": int((~correct).sum()),
                    "accuracy": float(correct.mean()),
                    "mean_uncalibrated_max_probability": float(
                        uncalibrated_max_probability[indices].mean()
                    ),
                }
            )
    return result


def confusion_pair_rows(confusion: np.ndarray, class_names: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for true_index, true_class in enumerate(class_names):
        support = int(confusion[true_index].sum())
        for predicted_index, predicted_class in enumerate(class_names):
            count = int(confusion[true_index, predicted_index])
            if true_index == predicted_index or count == 0:
                continue
            rows.append(
                {
                    "true_class": true_class,
                    "predicted_class": predicted_class,
                    "error_count": count,
                    "fraction_of_true_class": count / support,
                }
            )
    rows.sort(key=lambda row: (-int(row["error_count"]), str(row["true_class"]), str(row["predicted_class"])))
    return rows


def save_confusion_plot(confusion: np.ndarray, class_names: list[str], path: Path) -> None:
    short_names = [name.split("_", 1)[1].replace("_", " ") for name in class_names]
    figure, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(confusion, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set_xticks(range(len(class_names)), labels=short_names, rotation=35, ha="right")
    axis.set_yticks(range(len(class_names)), labels=short_names)
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Pretrained embedding model: development OOF confusion matrix")
    threshold = confusion.max() / 2 if confusion.size else 0
    for row_index in range(confusion.shape[0]):
        for column_index in range(confusion.shape[1]):
            value = int(confusion[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    np.random.seed(SEED)
    determinism = configure_cpu_determinism(args.threads)
    execution = execution_identity(args.batch_size, args.threads, determinism)
    native_input_sizes = {DINO_INPUT_SIZE}
    if not args.quick:
        native_input_sizes.update(
            timm_input_policy(spec).input_size for spec in TIMM_ENCODERS
        )
    execution["effective_batch_size_by_input"] = {
        str(input_size): effective_batch_size(args.batch_size, input_size)
        for input_size in sorted(native_input_sizes)
    }

    with args.split_audit.open(encoding="utf-8") as handle:
        split_audit = json.load(handle)
    rows = read_csv(args.development)
    image_integrity = validate_development(rows, args.development, split_audit)
    if image_integrity["verified_image_fingerprint_sha256"] != data_fingerprint(rows):
        raise RuntimeError("Verified image-byte fingerprint disagrees with the manifest")
    class_names = sorted({row["ornament_label"] for row in rows})
    class_to_index = {name: index for index, name in enumerate(class_names)}
    labels = np.asarray([class_to_index[row["ornament_label"]] for row in rows], dtype=np.int64)

    representations: dict[str, np.ndarray] = {}
    representation_metadata: dict[str, dict[str, object]] = {}
    cache_files: list[str] = []
    view_modes = ("center_crop",) if args.quick else ("center_crop", "letterbox", "global_fivecrop")
    for view_mode in view_modes:
        embeddings, metadata, cache_path = get_or_extract_dinov3(
            rows,
            args.dinov3_source,
            view_mode,
            args.cache_dir,
            args.batch_size,
            use_cache=not args.no_cache,
            execution=execution,
        )
        if not args.no_cache:
            if not cache_path.is_file():
                raise RuntimeError(f"Expected embedding cache was not written: {cache_path}")
            cache_files.append(display_path(cache_path))
        for pooling, values in embeddings.items():
            name = representation_name("dinov3_vits16", pooling, view_mode)
            representations[name] = values
            representation_metadata[name] = {**metadata, "pooling": pooling}

    if not args.quick:
        for spec in TIMM_ENCODERS:
            embeddings, metadata, cache_path = get_or_extract_timm(
                rows,
                spec,
                args.timm_hub_cache,
                args.cache_dir,
                args.batch_size,
                use_cache=not args.no_cache,
                execution=execution,
            )
            if not args.no_cache:
                if not cache_path.is_file():
                    raise RuntimeError(f"Expected embedding cache was not written: {cache_path}")
                cache_files.append(display_path(cache_path))
            for pooling, values in embeddings.items():
                name = representation_name(spec.alias, pooling, "center_crop")
                representations[name] = values
                representation_metadata[name] = {**metadata, "pooling": pooling}

    screen_rows, diagnostic_best_c = tune_probes(
        representations, rows, labels, len(class_names)
    )
    fold_rows, summary_rows, predictions_by_representation = cross_validate(
        representations,
        rows,
        labels,
        len(class_names),
    )
    selected_name = select_representation(summary_rows)
    predicted, probabilities = predictions_by_representation[selected_name]
    aggregate = calculate_metrics(labels, predicted, probabilities, len(class_names))
    per_class = per_class_rows(labels, predicted, class_names)
    confusion = confusion_matrix(labels, predicted, labels=np.arange(len(class_names)))
    top_order = np.argsort(-probabilities, axis=1, kind="stable")

    prediction_rows: list[dict[str, object]] = []
    for index, row in enumerate(rows):
        output: dict[str, object] = {
            "image_id": row["image_id"],
            "evaluation_status": "exploratory_selected_representation_oof",
            "relative_path": row["relative_path"],
            "true_class": row["ornament_label"],
            "predicted_class": class_names[int(predicted[index])],
            "is_correct": str(int(predicted[index]) == int(labels[index])),
            "uncalibrated_max_probability": float(probabilities[index].max()),
            "top_1_class": class_names[int(top_order[index, 0])],
            "top_2_class": class_names[int(top_order[index, 1])],
            "top_3_class": class_names[int(top_order[index, 2])],
            "cv_fold": row["cv_fold"],
            "production_split": row["production_split"],
            "old_split": row["old_split"],
            "manual_review_status": row["manual_review_status"],
            "confirmed_object_group_id": row["confirmed_object_group_id"],
            "confirmed_source_group_id": row["confirmed_source_group_id"],
            "object_type": row["object_type"],
            "motif_visibility": row["motif_visibility"],
        }
        for class_index, class_name in enumerate(class_names):
            output[f"probability_{class_name}"] = float(probabilities[index, class_index])
        for optional_group_field in (
            "split_group_id",
            "global_source_cohort_id",
            "source_atomic_cohort_ids",
            "pre_source_cohort_split_group_id",
            "source_atomic_split_group_id",
        ):
            if optional_group_field in row:
                output[optional_group_field] = row[optional_group_field]
        prediction_rows.append(output)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_fields = (
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "top_3_accuracy",
    )
    write_csv(
        output_dir / "probe_screen.csv",
        screen_rows,
        (
            "representation",
            "metric_status",
            "c_value",
            "diagnostic_only",
            "train_examples",
            "validation_examples",
            "iterations",
            "max_iterations",
            "converged",
            *metric_fields,
        ),
    )
    write_csv(
        output_dir / "cv_fold_metrics.csv",
        fold_rows,
        (
            "representation",
            "metric_status",
            "c_value",
            "cv_fold",
            "train_examples",
            "validation_examples",
            "iterations",
            "max_iterations",
            "converged",
            *metric_fields,
        ),
    )
    summary_fields = (
        "accuracy_rank",
        "selected",
        "within_half_point_of_best",
        "representation",
        "metric_status",
        "c_value",
        "fold_count",
        "aggregate_accuracy",
        "aggregate_macro_f1",
        "aggregate_balanced_accuracy",
        "aggregate_top_3_accuracy",
        "worst_class_recall",
        "all_folds_converged",
        "max_fold_iterations",
        *(field for metric in METRIC_NAMES for field in (f"mean_{metric}", f"std_{metric}")),
    )
    write_csv(output_dir / "representation_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "oof_predictions.csv", prediction_rows, tuple(prediction_rows[0].keys()))
    write_csv(
        output_dir / "per_class_metrics.csv",
        per_class,
        ("class_index", "class_name", "precision", "recall", "f1", "support"),
    )
    write_csv(
        output_dir / "diagnostic_slices.csv",
        diagnostic_slice_rows(rows, predicted, probabilities, labels),
        (
            "dimension",
            "value",
            "image_count",
            "correct_count",
            "error_count",
            "accuracy",
            "mean_uncalibrated_max_probability",
        ),
    )
    write_csv(
        output_dir / "confusion_pairs.csv",
        confusion_pair_rows(confusion, class_names),
        ("true_class", "predicted_class", "error_count", "fraction_of_true_class"),
    )
    confusion_rows = [
        {
            "true_class": class_name,
            **{
                name: int(confusion[index, column])
                for column, name in enumerate(class_names)
            },
        }
        for index, class_name in enumerate(class_names)
    ]
    write_csv(output_dir / "confusion_matrix.csv", confusion_rows, ("true_class", *class_names))
    save_confusion_plot(confusion, class_names, output_dir / "confusion_matrix.png")

    selected_fold_rows = [row for row in fold_rows if row["representation"] == selected_name]
    selected_summary = next(row for row in summary_rows if row["representation"] == selected_name)
    metadata_rows = []
    for name in sorted(representation_metadata):
        metadata = representation_metadata[name]
        metadata_rows.append(
            {
                "representation": name,
                "encoder_alias": metadata["encoder_alias"],
                "model_id": metadata["model_id"],
                "model_source": metadata["model_source"],
                "license": metadata["license"],
                "weight_sha256": metadata["weight_sha256"],
                "model_revision": metadata["model_revision"],
                "model_config_sha256": metadata["model_config_sha256"],
                "pretrained_config_sha256": metadata.get(
                    "pretrained_config_sha256", ""
                ),
                "preprocessor_config_sha256": metadata[
                    "preprocessor_config_sha256"
                ],
                "parameter_count": metadata["parameter_count"],
                "view_mode": metadata["view_mode"],
                "views_per_image": metadata["views_per_image"],
                "effective_batch_size": metadata["effective_batch_size"],
                "pooling": metadata["pooling"],
                "embedding_dimension": metadata["embedding_dimensions"][metadata["pooling"]],
                "input_policy_json": stable_json(metadata["input_policy"]),
                "execution_identity_json": stable_json(
                    metadata["execution_identity"]
                ),
                "cache_identity_json": stable_json(metadata["cache_identity"]),
                "nondeterministic_extraction_seconds": metadata[
                    "nondeterministic_observations"
                ]["extraction_seconds"],
                "nondeterministic_images_per_second": metadata[
                    "nondeterministic_observations"
                ]["images_per_second"],
                "embedding_fingerprint_sha256": metadata["embedding_fingerprint_sha256"],
                "experiment_version": metadata["experiment_version"],
                "script_sha256": metadata["script_sha256"],
                "extraction_runtime_json": stable_json(metadata["extraction_runtime"]),
            }
        )
    write_csv(
        output_dir / "representation_metadata.csv",
        metadata_rows,
        (
            "representation",
            "encoder_alias",
            "model_id",
            "model_source",
            "license",
            "weight_sha256",
            "model_revision",
            "model_config_sha256",
            "pretrained_config_sha256",
            "preprocessor_config_sha256",
            "parameter_count",
            "view_mode",
            "views_per_image",
            "effective_batch_size",
            "pooling",
            "embedding_dimension",
            "input_policy_json",
            "execution_identity_json",
            "cache_identity_json",
            "nondeterministic_extraction_seconds",
            "nondeterministic_images_per_second",
            "embedding_fingerprint_sha256",
            "experiment_version",
            "script_sha256",
            "extraction_runtime_json",
        ),
    )

    metrics_payload = {
        "phase": PHASE,
        "experiment_version": EXPERIMENT_VERSION,
        "run_mode": "quick" if args.quick else "full",
        "script_sha256": sha256_file(SCRIPT_PATH),
        "evaluation_scope": "development_only_exploratory_model_selection",
        "sealed_test_evaluated": False,
        "performance_estimate": {
            "available": False,
            "reason": (
                "The reported selected-representation OOF metrics are selection-conditional: "
                "the same OOF results select the representation, and the candidate set and "
                "C=10 were informed by earlier prototype exploration."
            ),
            "required_next_step": (
                "Freeze the complete pipeline, then evaluate exactly once on the untouched "
                "sealed test set."
            ),
        },
        "aggregate_oof_metric_status": (
            "exploratory_selection_conditional_not_an_unbiased_performance_estimate"
        ),
        "probability_policy": {
            "calibrated": False,
            "threshold_selection_used": False,
            "interpretation": (
                "predict_proba values are uncalibrated ranking scores; no deployment or "
                "rejection threshold may be selected from this run"
            ),
        },
        "selection_rule": (
            "highest mean five-fold accuracy; representations within 0.5 percentage points "
            "use mean macro F1, then worst-class recall, then name; because selection uses "
            "these OOF results, the selected representation's OOF metrics are optimistic"
        ),
        "probe_tuning_rule": (
            "fixed production train-to-validation C grid is diagnostic only and does not "
            "influence outer-fold fitting or OOF predictions"
        ),
        "outer_cv_probe_rule": (
            "class-balanced logistic regression with C=10, frozen after earlier prototype "
            "exploration and before this split run; it is not an independently unbiased "
            "hyperparameter choice"
        ),
        "outer_c_provenance": "frozen_after_prior_prototype_exploration",
        "split_version": split_audit["split_version"],
        "split_seed": split_audit["split_seed"],
        "split_assignment_fingerprint_sha256": split_audit["assignment_fingerprint_sha256"],
        "development_integrity": {
            "csv_sha256": sha256_file(args.development),
            "audit_csv_sha256": expected_development_sha256(split_audit, args.development),
            "class_coverage_verified": True,
            "production_group_isolation_verified": True,
            "cv_group_isolation_verified": True,
            "group_isolation_fields_verified": [
                field
                for field in (
                    "split_group_id",
                    "confirmed_object_group_id",
                    "confirmed_source_group_id",
                    "content_sha256",
                    "global_source_cohort_id",
                    "source_atomic_cohort_ids",
                    "pre_source_cohort_split_group_id",
                    "source_atomic_split_group_id",
                )
                if field in rows[0]
            ],
            **image_integrity,
        },
        "development_image_count": len(rows),
        "sealed_test_image_count": split_audit["production_split_counts"]["test"],
        "class_names": class_names,
        "selected_representation": selected_name,
        "selected_c": OUTER_C,
        "diagnostic_best_c_by_representation": diagnostic_best_c,
        "selected_representation_metadata": representation_metadata[selected_name],
        "selected_summary": selected_summary,
        "aggregate_oof_metrics": aggregate,
        "selected_fold_metrics": selected_fold_rows,
        "per_class_metrics": per_class,
        "representation_count": len(representations),
        "probe_convergence": {
            "warnings_treated_as_errors": True,
            "max_iterations": PROBE_MAX_ITERATIONS,
            "all_fits_converged": all(
                row["converged"] == "True" for row in (*screen_rows, *fold_rows)
            ),
            "maximum_observed_iterations": max(
                int(row["iterations"]) for row in (*screen_rows, *fold_rows)
            ),
            "sklearn_blas_threads_per_fit": 1,
        },
        "cache_enabled": not args.no_cache,
        "cache_files": sorted(set(cache_files)),
        "data_fingerprint_sha256": data_fingerprint(rows),
        "input_policy": {
            "timm_models": (
                "model-native timm input size, crop percentage, bicubic interpolation, "
                "normalization mean, and normalization standard deviation"
            ),
            "dinov3": (
                "native 224-pixel size and ImageNet normalization with project-defined "
                "center-crop, letterbox, and global-five-crop view policies"
            ),
            "representation_policies": {
                name: metadata["input_policy"]
                for name, metadata in sorted(representation_metadata.items())
            },
            "comparison_limitation": (
                "Encoder rankings combine backbone and model-native preprocessing; differing "
                "native resolutions mean this is not a controlled backbone-only comparison."
            ),
        },
        "seed": SEED,
        "execution_identity": execution,
        "determinism": determinism,
        "timing_policy": (
            "Fields prefixed nondeterministic_ are observational timings and are excluded "
            "from experiment and cache identity."
        ),
        "source_group_fields": {
            field: len({row[field] for row in rows})
            for field in (
                "global_source_cohort_id",
                "source_atomic_cohort_ids",
                "pre_source_cohort_split_group_id",
                "source_atomic_split_group_id",
            )
            if field in rows[0]
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "Pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "timm": timm.__version__,
            "transformers": transformers.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(
        json.dumps(
            {
                "selected_representation": selected_name,
                "aggregate_oof_metrics": aggregate,
                "sealed_test_evaluated": False,
                "output_dir": str(output_dir),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
