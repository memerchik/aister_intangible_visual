#!/usr/bin/env python3
"""Fail when a prospective public Git commit contains private or unsafe files.

The check scans the current tracked checkout plus unignored new files. It never
prints matched secret values. Run it from anywhere inside the repository before
staging or committing a public release.
"""

from __future__ import annotations

import re
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path


MAX_FILE_BYTES = 50 * 1024 * 1024
CHECKPOINT_SUFFIXES = {
    ".ckpt",
    ".onnx",
    ".p12",
    ".pem",
    ".pfx",
    ".pt",
    ".pth",
    ".safetensors",
}
SECRET_PATTERNS = {
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----"),
    "GitHub token": re.compile(rb"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})"),
    "Hugging Face token": re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "Google API key": re.compile(rb"\bAIza[A-Za-z0-9_-]{30,}\b"),
    "Slack token": re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    "credential-bearing URL": re.compile(rb"(?i)https?://[^\s/:]+:[^\s/@]+@"),
}
HOME_PATH_PATTERNS = {
    "macOS home path": re.compile(rb"/Users/[^/\s\"']+"),
    "Linux home path": re.compile(rb"/home/[^/\s\"']+"),
    "Windows home path": re.compile(rb"[A-Za-z]:\\Users\\[^\\\s\"']+"),
}
PNG_PRIVACY_METADATA = re.compile(
    rb"(?i)(?:"
    rb"(?:author|artist|creator|owner|copyright|contact)\s*="
    rb"|\b(?:author|artist|ownername|copyright|creatorcontactinfo)\b"
    rb"|\bdc:(?:creator|rights)\b"
    rb"|\bphotoshop:AuthorsPosition\b"
    rb"|\bIptc4xmpCore:CreatorContactInfo\b"
    rb"|\bexif:CameraOwnerName\b"
    rb")"
)
KNOWN_FROZEN_HOME_PATH_FILES = {
    Path("step_1/outputs/test_predictions.csv"),
    Path("step_1/workshop_materials/notebooks/classical_vision_hands_on.ipynb"),
    Path("step_1/workshop_materials/outputs/workshop_predictions.csv"),
    Path("step_2/development/outputs/phase_5_source_robustness/metrics.json"),
    Path("step_2/development/outputs/phase_5_source_robustness_v2/metrics.json"),
    Path("step_2/development/outputs/phase_5_source_robustness_v3/metrics.json"),
    Path("step_2/development/outputs/phase_5_source_robustness_v4/metrics.json"),
    Path("step_2/development/outputs/phase_5_source_robustness_v5/metrics.json"),
}


def git_paths(*arguments: str) -> list[Path]:
    output = subprocess.check_output(("git", *arguments, "-z"))
    return [Path(value.decode()) for value in output.split(b"\0") if value]


def prospective_files() -> list[Path]:
    ordered = git_paths("ls-files") + git_paths(
        "ls-files", "--others", "--exclude-standard"
    )
    unique: list[Path] = []
    for path in ordered:
        if path.is_file() and path not in unique:
            unique.append(path)
    return unique


def forbidden_path_reason(path: Path) -> str | None:
    normalized = path.as_posix()
    if path.name.startswith(".env"):
        return "environment file"
    if path.name in {"id_rsa", "id_ed25519"}:
        return "private-key filename"
    if path.suffix.lower() in CHECKPOINT_SUFFIXES:
        return "private key or model checkpoint extension"
    if path.name.endswith(".inspect.ndjson"):
        return "generated inspection sidecar"
    forbidden_prefixes = (
        ".cache/",
        "step_1/workshop_materials/data/dataset_dev/",
        "step_1/workshop_materials/data/dataset_test/",
        "step_2/development/data/dataset_dev/",
        "step_2/development/data/dataset_test/",
        "step_2/v0_5/contributions/pending/",
        "step_2/v0_5/model/",
    )
    if normalized.startswith(forbidden_prefixes):
        return "raw data, local cache, contribution, or private model path"
    return None


def png_metadata_chunks(data: bytes) -> tuple[list[bytes], list[str]]:
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return [], []

    chunks: list[bytes] = []
    issues: list[str] = []
    offset = 8
    while offset + 12 <= len(data):
        length = int.from_bytes(data[offset : offset + 4], "big")
        kind = data[offset + 4 : offset + 8]
        end = offset + 12 + length
        if end > len(data):
            issues.append("malformed PNG metadata")
            break
        payload = data[offset + 8 : offset + 8 + length]
        try:
            if kind == b"tEXt":
                keyword, value = payload.split(b"\0", 1)
                chunks.append(keyword + b"=" + value)
            elif kind == b"zTXt":
                keyword, compressed = payload.split(b"\0", 1)
                chunks.append(keyword + b"=" + zlib.decompress(compressed[1:]))
            elif kind == b"iTXt":
                keyword, remainder = payload.split(b"\0", 1)
                compression_flag = remainder[0]
                remainder = remainder[2:]
                _, remainder = remainder.split(b"\0", 1)
                _, value = remainder.split(b"\0", 1)
                if compression_flag == 1:
                    value = zlib.decompress(value)
                chunks.append(keyword + b"=" + value)
            elif kind == b"eXIf":
                issues.append("embedded PNG EXIF metadata")
        except (IndexError, ValueError, zlib.error):
            issues.append("malformed PNG text metadata")
        offset = end
        if kind == b"IEND":
            break
    return chunks, issues


def content_chunks(path: Path) -> tuple[list[bytes], list[bytes], list[str]]:
    data = path.read_bytes()
    chunks = [data]
    png_chunks, issues = png_metadata_chunks(data)
    chunks.extend(png_chunks)
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            for member in archive.namelist():
                if member.endswith((".xml", ".rels", ".json", ".txt", ".csv")):
                    chunks.append(archive.read(member))
    return chunks, png_chunks, issues


def main() -> int:
    repository = subprocess.check_output(
        ("git", "rev-parse", "--show-toplevel"), text=True
    ).strip()
    root = Path(repository)
    failures: list[tuple[Path, str]] = []
    warnings: list[tuple[Path, str]] = []

    for relative in prospective_files():
        path = root / relative
        reason = forbidden_path_reason(relative)
        if reason:
            failures.append((relative, reason))
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            failures.append((relative, "file is larger than 50 MiB"))
            continue
        try:
            chunks, png_chunks, content_issues = content_chunks(path)
        except (OSError, zipfile.BadZipFile) as error:
            failures.append((relative, f"could not scan file: {type(error).__name__}"))
            continue
        for reason in content_issues:
            failures.append((relative, reason))
        if any(PNG_PRIVACY_METADATA.search(chunk) for chunk in png_chunks):
            failures.append((relative, "identity-bearing PNG metadata"))
        for label, pattern in SECRET_PATTERNS.items():
            if any(pattern.search(chunk) for chunk in chunks):
                failures.append((relative, label))
        for label, pattern in HOME_PATH_PATTERNS.items():
            if relative == Path("scripts/check_public_push.py"):
                continue
            if not any(pattern.search(chunk) for chunk in chunks):
                continue
            if relative in KNOWN_FROZEN_HOME_PATH_FILES:
                warnings.append((relative, f"known historical {label}"))
            else:
                failures.append((relative, label))

    print(f"Scanned {len(prospective_files())} tracked or commit-eligible files.")
    for path, reason in sorted(set(warnings)):
        print(f"WARNING {path}: {reason}")
    if failures:
        for path, reason in sorted(set(failures)):
            print(f"BLOCKED {path}: {reason}")
        print("Public-push safety check failed. No secret values were printed.")
        return 1
    print("Public-push safety check passed. No raw data or credential pattern is commit-eligible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
