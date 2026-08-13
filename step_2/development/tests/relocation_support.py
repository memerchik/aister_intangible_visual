"""Test helpers for frozen evidence recorded before the Step 02 relocation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


DEVELOPMENT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = DEVELOPMENT_ROOT.parents[1]
MANIFEST_PATH = DEVELOPMENT_ROOT / "relocation_manifest.json"
MANIFEST = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_recorded_path(recorded: str) -> Path:
    """Map a frozen path to its current physical location."""

    path = Path(recorded)
    parts = path.parts
    if not path.is_absolute() and parts and parts[0] == "step_02":
        return DEVELOPMENT_ROOT.joinpath(*parts[1:]).resolve()
    if path.is_absolute() and "step_02" in parts:
        marker = parts.index("step_02")
        return DEVELOPMENT_ROOT.joinpath(*parts[marker + 1 :]).resolve()
    if path.is_absolute():
        return path.resolve()
    return (WORKSPACE_ROOT / path).resolve()


def assert_recorded_file(
    testcase: Any,
    recorded: str,
    expected_sha256: str | None = None,
) -> Path:
    """Assert existence and either exact or explicitly attested hash identity."""

    path = resolve_recorded_path(recorded)
    testcase.assertTrue(path.is_file(), recorded)
    if expected_sha256 is None:
        return path

    actual = sha256_file(path)
    if actual == expected_sha256:
        return path

    entry = MANIFEST["changed_files"].get(recorded)
    testcase.assertIsNotNone(entry, f"unrecorded relocation hash change: {recorded}")
    testcase.assertEqual(entry["historical_sha256"], expected_sha256, recorded)
    testcase.assertEqual(entry["current_path"], path.relative_to(WORKSPACE_ROOT).as_posix())
    testcase.assertEqual(entry["current_sha256"], actual, recorded)
    return path
