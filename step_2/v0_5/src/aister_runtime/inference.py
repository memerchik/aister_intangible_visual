"""Self-contained provisional inference for the v0.5 application.

This module packages the fixed full-development Phase 5 v4 recipe without
opening the formal Phase 6 or sealed-test workflow. The returned values are
uncalibrated ranking scores, and the application must preserve that wording.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .motif import aggregate_tile_embeddings, select_motif_proposals


APP_VERSION = "v0.5"
ARTIFACT_SCHEMA_VERSION = 1
MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"
MODEL_ARCHITECTURE = "vit_small_patch16_dinov3_qkvb"
MODEL_WEIGHT_SHA256 = (
    "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d"
)
CLASS_ORDER = (
    "01_opishnyan_ceramics",
    "02_ornek",
    "03_bubnivka_ceramics",
    "04_petrykivka_painting",
    "05_kosiv_ceramics",
)
CERAMIC_BOUNDARY_CLASSES = frozenset(
    ("01_opishnyan_ceramics", "03_bubnivka_ceramics")
)
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_DECODED_PIXELS = 30_000_000
MIN_IMAGE_DIMENSION = 32
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224
RESIZE_SHORT_SIDE = 256

CLASS_CATALOG: Mapping[str, Mapping[str, str]] = {
    "01_opishnyan_ceramics": {
        "name": "Opishnyan ceramics",
        "short_name": "Opishnyan",
        "kind": "Painted ceramics",
        "look_for": "Compare the painted composition, line rhythm, palette, and how the ornament follows the ceramic form.",
        "source_url": "",
    },
    "02_ornek": {
        "name": "Örnek",
        "short_name": "Örnek",
        "kind": "Crimean Tatar ornament",
        "look_for": "A symbolic composition built from floral or geometric signs across embroidery, weaving, pottery, engraving, and other crafts.",
        "source_url": "https://ich.unesco.org/en/RL/ornek-a-crimean-tatar-ornament-and-knowledge-about-it-01601?RL=01601",
    },
    "03_bubnivka_ceramics": {
        "name": "Bubnivka ceramics",
        "short_name": "Bubnivka",
        "kind": "Painted ceramics",
        "look_for": "Compare the painted composition, repeated border elements, palette, and the relationship between motif and vessel shape.",
        "source_url": "",
    },
    "04_petrykivka_painting": {
        "name": "Petrykivka painting",
        "short_name": "Petrykivka",
        "kind": "Decorative painting",
        "look_for": "Fantastic flowers and other natural elements arranged with flowing, highly decorative brushwork.",
        "source_url": "https://ich.unesco.org/en/RL/petrykivka-decorative-painting-as-a-phenomenon-of-the-ukrainian-ornamental-folk-art-00893",
    },
    "05_kosiv_ceramics": {
        "name": "Kosiv ceramics",
        "short_name": "Kosiv",
        "kind": "Painted ceramics",
        "look_for": "Graphic contour drawing, figurative scenes, and the characteristic green and yellow ceramic palette.",
        "source_url": "https://ich.unesco.org/en/RL/tradition-of-kosiv-painted-ceramics-01456?RL=01456",
    },
}


class AssistedInferenceError(RuntimeError):
    """Raised when the provisional inference contract cannot be satisfied."""


class InvalidImageError(ValueError):
    """Raised when an uploaded image violates the application input policy."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise AssistedInferenceError("The encoder returned an empty feature vector")
    return matrix / norms


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / np.sum(exponentials, axis=1, keepdims=True)


@dataclass(frozen=True)
class LinearModelArtifact:
    classes: Tuple[str, ...]
    coefficients: np.ndarray
    intercept: np.ndarray
    manifest: Mapping[str, object]

    @classmethod
    def load(cls, artifact_directory: Path) -> "LinearModelArtifact":
        manifest_path = artifact_directory / "model_manifest.json"
        weights_path = artifact_directory / "v4_assisted_model.npz"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise AssistedInferenceError(f"Cannot load the v0.5 model manifest: {error}") from error
        if not isinstance(manifest, dict):
            raise AssistedInferenceError("The v0.5 model manifest must be an object")
        if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
            raise AssistedInferenceError("Unsupported v0.5 model artifact schema")
        if manifest.get("sealed_test_evaluated") is not False:
            raise AssistedInferenceError("The v0.5 artifact violates the sealed-test boundary")
        expected_file_hash = manifest.get("artifact_sha256")
        if not isinstance(expected_file_hash, str) or sha256_file(weights_path) != expected_file_hash:
            raise AssistedInferenceError("The v0.5 model artifact is missing or changed")
        try:
            with np.load(weights_path, allow_pickle=False) as stored:
                if set(stored.files) != {"classes", "coefficients", "intercept"}:
                    raise AssistedInferenceError("The v0.5 model artifact members changed")
                classes_array = np.array(stored["classes"], copy=True)
                coefficients = np.array(stored["coefficients"], dtype=np.float64, copy=True)
                intercept = np.array(stored["intercept"], dtype=np.float64, copy=True)
        except (OSError, ValueError) as error:
            raise AssistedInferenceError(f"Cannot read the v0.5 model artifact: {error}") from error
        classes = tuple(str(value) for value in classes_array.tolist())
        if classes != CLASS_ORDER:
            raise AssistedInferenceError("The v0.5 class order changed")
        if coefficients.shape != (len(CLASS_ORDER), 768) or intercept.shape != (len(CLASS_ORDER),):
            raise AssistedInferenceError("The v0.5 model parameter shape changed")
        expected_arrays = manifest.get("array_sha256")
        if not isinstance(expected_arrays, dict):
            raise AssistedInferenceError("The v0.5 array manifest is invalid")
        observed_arrays = {
            "classes": sha256_array(classes_array),
            "coefficients": sha256_array(coefficients),
            "intercept": sha256_array(intercept),
        }
        if observed_arrays != expected_arrays:
            raise AssistedInferenceError("The v0.5 model parameter fingerprint changed")
        return cls(classes, coefficients, intercept, manifest)

    def predict_scores(self, features: np.ndarray) -> np.ndarray:
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != self.coefficients.shape[1]:
            raise ValueError("Inference features do not match the packaged v0.5 model")
        if not np.isfinite(matrix).all():
            raise ValueError("Inference features must be finite")
        return _softmax(matrix @ self.coefficients.T + self.intercept[None, :])


def decode_data_url(data_url: str) -> Tuple[bytes, str]:
    if not isinstance(data_url, str) or not data_url.startswith("data:image/"):
        raise InvalidImageError("Upload a JPEG, PNG, or WebP image")
    try:
        header, encoded = data_url.split(",", 1)
    except ValueError as error:
        raise InvalidImageError("The uploaded image payload is incomplete") from error
    if ";base64" not in header:
        raise InvalidImageError("The uploaded image must use base64 encoding")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as error:
        raise InvalidImageError("The uploaded image encoding is invalid") from error
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("The image is larger than the 12 MB limit")
    return image_bytes, header[5:].split(";", 1)[0].lower()


def decode_image(image_bytes: bytes) -> Tuple[Image.Image, str]:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise InvalidImageError("The uploaded image is empty")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise InvalidImageError("The image is larger than the 12 MB limit")
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            detected_format = str(opened.format or "").upper()
            if detected_format not in {"JPEG", "PNG", "WEBP"}:
                raise InvalidImageError("Only JPEG, PNG, and WebP images are supported")
            width, height = opened.size
            if width * height > MAX_DECODED_PIXELS:
                raise InvalidImageError("The decoded image is larger than the 30 megapixel limit")
            if min(width, height) < MIN_IMAGE_DIMENSION:
                raise InvalidImageError("Both image dimensions must be at least 32 pixels")
            oriented = ImageOps.exif_transpose(opened)
            oriented.load()
            if "A" in oriented.getbands():
                rgba = oriented.convert("RGBA")
                background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                image = Image.alpha_composite(background, rgba).convert("RGB")
            else:
                image = oriented.convert("RGB")
    except InvalidImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise InvalidImageError("The file could not be decoded as an image") from error
    return image, detected_format.lower()


class Dinov3FeatureExtractor:
    """Load the pinned local encoder and reproduce the fixed v4 representations."""

    def __init__(self, weight_path: Optional[Path] = None, threads: int = 4):
        if threads < 1:
            raise ValueError("threads must be positive")
        self.weight_path = self._resolve_weight(weight_path)
        self.threads = threads
        self._model = None

    @staticmethod
    def _resolve_weight(override: Optional[Path]) -> Path:
        environment_path = os.environ.get("AISTER_DINOV3_WEIGHT", "").strip()
        if override is not None:
            path = override.expanduser().resolve()
        elif environment_path:
            path = Path(environment_path).expanduser().resolve()
        else:
            path = (
                Path.home()
                / ".cache"
                / "huggingface"
                / "hub"
                / "models--facebook--dinov3-vits16-pretrain-lvd1689m"
                / "blobs"
                / MODEL_WEIGHT_SHA256
            )
        if not path.is_file() or sha256_file(path) != MODEL_WEIGHT_SHA256:
            raise AssistedInferenceError(
                "The pinned local DINOv3 weight is unavailable or changed; no download is attempted"
            )
        return path

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        import timm
        from safetensors import safe_open

        torch.set_num_threads(self.threads)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # A host process may have configured this global setting already.
            pass
        torch.manual_seed(20260719)
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.set_float32_matmul_precision("highest")
        with safe_open(str(self.weight_path), framework="pt", device="cpu") as handle:
            source = {key: handle.get_tensor(key) for key in handle.keys()}
        model = timm.create_model(
            MODEL_ARCHITECTURE,
            pretrained=False,
            num_classes=0,
            img_size=INPUT_SIZE,
        )
        state: Dict[str, object] = {
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
            raise AssistedInferenceError("The v0.5 encoder must run on CPU")
        self._model = model

    @staticmethod
    def _letterbox(image: Image.Image) -> Image.Image:
        scale = INPUT_SIZE / max(image.width, image.height)
        resized_size = (
            max(1, round(image.width * scale)),
            max(1, round(image.height * scale)),
        )
        resized = image.resize(resized_size, Image.Resampling.BICUBIC)
        fill = tuple(round(channel * 255) for channel in IMAGENET_MEAN)
        canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), fill)
        canvas.paste(
            resized,
            ((INPUT_SIZE - resized.width) // 2, (INPUT_SIZE - resized.height) // 2),
        )
        return canvas

    @staticmethod
    def _global_views(image: Image.Image) -> Tuple[Image.Image, ...]:
        scale = RESIZE_SHORT_SIDE / min(image.width, image.height)
        resized = image.resize(
            (
                max(RESIZE_SHORT_SIDE, round(image.width * scale)),
                max(RESIZE_SHORT_SIDE, round(image.height * scale)),
            ),
            Image.Resampling.BICUBIC,
        )
        positions = (
            (0, 0),
            (resized.width - INPUT_SIZE, 0),
            (0, resized.height - INPUT_SIZE),
            (resized.width - INPUT_SIZE, resized.height - INPUT_SIZE),
            (
                (resized.width - INPUT_SIZE) // 2,
                (resized.height - INPUT_SIZE) // 2,
            ),
        )
        crops = tuple(
            resized.crop((left, top, left + INPUT_SIZE, top + INPUT_SIZE))
            for left, top in positions
        )
        return (Dinov3FeatureExtractor._letterbox(image), *crops)

    @staticmethod
    def _images_to_tensor(images: Sequence[Image.Image]):
        import torch

        mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(1, 1, 3)
        std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(1, 1, 3)
        arrays = []
        for image in images:
            array = np.asarray(image, dtype=np.float32) / 255.0
            arrays.append(np.transpose((array - mean) / std, (2, 0, 1)))
        return torch.from_numpy(np.stack(arrays)).float()

    def _embed_views(self, images: Sequence[Image.Image]) -> np.ndarray:
        import torch

        self.load()
        if self._model is None:
            raise AssistedInferenceError("The encoder did not initialize")
        with torch.inference_mode():
            features = self._model.forward_features(self._images_to_tensor(images))[:, 0]
        return _normalize_rows(features.detach().cpu().numpy()).astype(np.float32)

    def extract(self, image: Image.Image) -> Tuple[np.ndarray, Tuple[Mapping[str, object], ...]]:
        proposals = select_motif_proposals(image)
        global_views = self._global_views(image)
        motif_views = tuple(
            image.crop(proposal.box).resize(
                (INPUT_SIZE, INPUT_SIZE), Image.Resampling.BICUBIC
            )
            for proposal in proposals
        )
        embeddings = self._embed_views((*global_views, *motif_views))
        global_embedding = _normalize_rows(embeddings[:6].mean(axis=0)[None, :])[0]
        motif_embedding = aggregate_tile_embeddings(
            embeddings[6:], [proposal.texture_score for proposal in proposals]
        )["texture_weighted_mean"]
        features = np.concatenate((global_embedding, motif_embedding))[None, :]
        regions = tuple(
            {
                "rank": rank,
                "x": proposal.left / image.width,
                "y": proposal.top / image.height,
                "width": (proposal.right - proposal.left) / image.width,
                "height": (proposal.bottom - proposal.top) / image.height,
                "texture_score": proposal.texture_score,
            }
            for rank, proposal in enumerate(proposals, start=1)
        )
        return features.astype(np.float32), regions


class AssistedOrnamentPredictor:
    """Thread-safe immediate predictor used by the v0.5 local application."""

    def __init__(
        self,
        application_root: Optional[Path] = None,
        weight_path: Optional[Path] = None,
        threads: int = 4,
    ):
        root = (
            Path(application_root).expanduser().resolve()
            if application_root is not None
            else Path(__file__).resolve().parents[2]
        )
        artifact_directory = root / "artifacts"
        self.artifact = LinearModelArtifact.load(artifact_directory)
        self.extractor = Dinov3FeatureExtractor(weight_path=weight_path, threads=threads)
        self._lock = threading.Lock()

    def warmup(self) -> None:
        self.extractor.load()

    def predict_image(self, image: Image.Image) -> Mapping[str, object]:
        with self._lock:
            features, regions = self.extractor.extract(image)
            scores = self.artifact.predict_scores(features)[0]
        ranked_indices = np.argsort(-scores, kind="stable")
        ranking = []
        for rank, index in enumerate(ranked_indices, start=1):
            label = self.artifact.classes[int(index)]
            catalog = CLASS_CATALOG[label]
            ranking.append(
                {
                    "rank": rank,
                    "label": label,
                    "name": catalog["name"],
                    "short_name": catalog["short_name"],
                    "kind": catalog["kind"],
                    "look_for": catalog["look_for"],
                    "source_url": catalog["source_url"],
                    "score": float(scores[index]),
                }
            )
        top_label = str(ranking[0]["label"])
        return {
            "application_version": APP_VERSION,
            "model_id": self.artifact.manifest.get("model_id"),
            "score_type": "uncalibrated_ranking_score",
            "top_match": ranking[0],
            "ranking": ranking,
            "ranking_gap": float(ranking[0]["score"] - ranking[1]["score"]),
            "review_recommended": top_label in CERAMIC_BOUNDARY_CLASSES,
            "review_reason": (
                "Opishnyan and Bubnivka remain the model's most difficult visual boundary. Compare the top alternatives before confirming."
                if top_label in CERAMIC_BOUNDARY_CLASSES
                else "The result is still a visual suggestion; qualified review is required for official identification."
            ),
            "motif_regions": regions,
            "image": {"width": image.width, "height": image.height},
            "calibrated": False,
            "sealed_test_evaluated": False,
        }

    def predict_bytes(self, image_bytes: bytes) -> Tuple[Mapping[str, object], str]:
        image, detected_format = decode_image(image_bytes)
        return self.predict_image(image), detected_format
