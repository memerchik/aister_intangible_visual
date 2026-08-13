"""Locate or securely download the exact gated DINOv3 deployment weight."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .inference import MODEL_WEIGHT_SHA256


MODEL_WEIGHT_URL = (
    "https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m/"
    "resolve/main/model.safetensors"
)
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


def _valid_weight(path: Path) -> bool:
    if not path.is_file():
        return False
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(DOWNLOAD_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest() == MODEL_WEIGHT_SHA256


def _local_cache_weight() -> Path:
    return (
        Path.home()
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--facebook--dinov3-vits16-pretrain-lvd1689m"
        / "blobs"
        / MODEL_WEIGHT_SHA256
    )


def ensure_model_weight() -> Path:
    """Return the pinned weight, downloading it only during deployment startup."""

    configured = os.environ.get("AISTER_DINOV3_WEIGHT", "").strip()
    configured_path = Path(configured).expanduser().resolve() if configured else None
    candidates = tuple(path for path in (configured_path, _local_cache_weight()) if path)
    for candidate in candidates:
        if _valid_weight(candidate):
            return candidate
    if configured_path is not None and configured_path.exists():
        raise RuntimeError("AISTER_DINOV3_WEIGHT exists but does not match the pinned hash")

    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
        "HF_HUB_TOKEN", ""
    ).strip()
    if not token:
        raise RuntimeError(
            "The gated DINOv3 weight is unavailable. Provide HF_TOKEN after accepting "
            "the model terms, or set AISTER_DINOV3_WEIGHT to the verified local file."
        )

    target = configured_path or Path("/tmp/aister-model/model.safetensors")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".download")
    request = Request(
        MODEL_WEIGHT_URL,
        headers={"Authorization": f"Bearer {token}", "User-Agent": "AISTER-v0.5"},
    )
    digest = hashlib.sha256()
    try:
        print("Downloading the gated DINOv3 weight for this runtime...", flush=True)
        with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != MODEL_WEIGHT_SHA256:
            raise RuntimeError("The downloaded DINOv3 weight failed hash verification")
        os.replace(temporary, target)
        return target
    except (HTTPError, URLError) as error:
        raise RuntimeError(
            "The gated DINOv3 weight could not be downloaded; verify HF_TOKEN access"
        ) from error
    finally:
        if temporary.exists():
            temporary.unlink()
