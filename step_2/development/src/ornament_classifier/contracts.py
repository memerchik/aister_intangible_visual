"""Validated, development-only inputs for reusable modelling code."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

from .paths import ProjectPaths


DEVELOPMENT_DESTINATIONS = frozenset(("train", "validation"))
REQUIRED_COLUMNS = frozenset(
    (
        "image_id",
        "relative_path",
        "ornament_label",
        "content_sha256",
        "split_group_id",
        "inclusion_status",
        "production_split",
        "cv_fold",
        "split_version",
        "split_seed",
    )
)


class ContractError(ValueError):
    """Raised when canonical development inputs violate their contract."""


@dataclass(frozen=True)
class DevelopmentRecord:
    image_id: str
    relative_path: str
    ornament_label: str
    content_sha256: str
    split_group_id: str
    production_split: str
    cv_fold: str
    split_version: str
    split_seed: int
    source_atomic_split_group_id: str = ""
    source_atomic_cohort_ids: str = ""
    global_source_cohort_id: str = ""
    object_type: str = ""
    motif_visibility: str = ""


@dataclass(frozen=True)
class DevelopmentContract:
    paths: ProjectPaths
    records: Tuple[DevelopmentRecord, ...]
    audit: Mapping[str, object]
    split_version: str
    split_seed: int
    fold_ids: Tuple[str, ...]

    def image_path(self, record: DevelopmentRecord) -> Path:
        return self.paths.resolve_development_image(record.relative_path)

    def records_for_fold(self, fold_id: str) -> Tuple[DevelopmentRecord, ...]:
        if fold_id not in self.fold_ids:
            raise ContractError(
                f"Unknown development fold {fold_id!r}; expected one of {self.fold_ids}"
            )
        return tuple(record for record in self.records if record.cv_fold == fold_id)


def load_split_audit(paths: Optional[ProjectPaths] = None) -> Dict[str, object]:
    """Load the canonical split provenance used by development code."""

    resolved = paths or ProjectPaths.discover()
    try:
        with resolved.split_audit_path.open(encoding="utf-8") as handle:
            audit = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ContractError(f"Cannot load {resolved.split_audit_path}: {error}") from error
    if not isinstance(audit, dict):
        raise ContractError("split_audit.json must contain a JSON object")
    return audit


def load_development_contract(
    paths: Optional[ProjectPaths] = None,
) -> DevelopmentContract:
    """Load only development.csv and validate it against split_audit.json."""

    resolved = paths or ProjectPaths.discover()
    audit = load_split_audit(resolved)
    try:
        split_version = audit["split_version"]
        split_seed = audit["split_seed"]
        fold_count = audit["policy"]["cv_folds"]
        destination_counts = audit["production_split_counts"]
        fold_counts = audit["development_cv_fold_counts"]
        if not isinstance(split_version, str) or not split_version:
            raise TypeError("split_version")
        if not isinstance(split_seed, int) or isinstance(split_seed, bool):
            raise TypeError("split_seed")
        if not isinstance(fold_count, int) or fold_count < 2:
            raise TypeError("cv_folds")
        fold_ids = tuple(str(index) for index in range(fold_count))
        expected_destinations = {
            name: int(destination_counts[name])
            for name in sorted(DEVELOPMENT_DESTINATIONS)
        }
        expected_folds = {fold_id: int(fold_counts[fold_id]) for fold_id in fold_ids}
    except (KeyError, TypeError, ValueError) as error:
        raise ContractError("split_audit.json has an invalid development contract") from error

    try:
        with resolved.development_manifest_path.open(
            encoding="utf-8", newline=""
        ) as handle:
            reader = csv.DictReader(handle)
            missing = sorted(REQUIRED_COLUMNS - set(reader.fieldnames or ()))
            if missing:
                raise ContractError(
                    "Development manifest is missing columns: " + ", ".join(missing)
                )
            rows = list(reader)
    except OSError as error:
        raise ContractError(
            f"Cannot load {resolved.development_manifest_path}: {error}"
        ) from error

    records = tuple(
        _record_from_row(
            row,
            number,
            resolved,
            split_version,
            split_seed,
            fold_ids,
        )
        for number, row in enumerate(rows, start=2)
    )
    if not records:
        raise ContractError("Development manifest is empty")
    if len({record.image_id for record in records}) != len(records):
        raise ContractError("Development manifest contains duplicate image IDs")

    observed_destinations = Counter(record.production_split for record in records)
    observed_folds = Counter(record.cv_fold for record in records)
    if dict(observed_destinations) != expected_destinations:
        raise ContractError("Development destination counts disagree with the split audit")
    if dict(observed_folds) != expected_folds:
        raise ContractError("Development fold counts disagree with the split audit")

    return DevelopmentContract(
        paths=resolved,
        records=records,
        audit=audit,
        split_version=split_version,
        split_seed=split_seed,
        fold_ids=fold_ids,
    )


def _record_from_row(
    row: Mapping[str, str],
    number: int,
    paths: ProjectPaths,
    split_version: str,
    split_seed: int,
    fold_ids: Tuple[str, ...],
) -> DevelopmentRecord:
    empty = [name for name in REQUIRED_COLUMNS if not row.get(name, "").strip()]
    if empty:
        raise ContractError(f"Development row {number} has empty fields: {sorted(empty)}")
    if row["inclusion_status"] != "included":
        raise ContractError(f"Development row {number} is not included")
    if row["production_split"] not in DEVELOPMENT_DESTINATIONS:
        raise ContractError(f"Development row {number} has a forbidden destination")
    if row["cv_fold"] not in fold_ids:
        raise ContractError(f"Development row {number} has an invalid fold")
    if row["split_version"] != split_version or row["split_seed"] != str(split_seed):
        raise ContractError(f"Development row {number} has stale split provenance")
    source_atomic_group = row.get("source_atomic_split_group_id", "").strip()
    if source_atomic_group and source_atomic_group != row["split_group_id"]:
        raise ContractError(
            f"Development row {number} disagrees on its source-atomic group"
        )

    digest = row["content_sha256"].lower()
    if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise ContractError(f"Development row {number} has an invalid SHA-256")
    try:
        image_path = paths.resolve_development_image(row["relative_path"])
    except ValueError as error:
        raise ContractError(f"Development row {number} has an unsafe path") from error
    if not image_path.is_file():
        raise ContractError(f"Development image is missing: {image_path}")

    return DevelopmentRecord(
        image_id=row["image_id"],
        relative_path=row["relative_path"],
        ornament_label=row["ornament_label"],
        content_sha256=digest,
        split_group_id=row["split_group_id"],
        production_split=row["production_split"],
        cv_fold=row["cv_fold"],
        split_version=row["split_version"],
        split_seed=split_seed,
        source_atomic_split_group_id=source_atomic_group,
        source_atomic_cohort_ids=row.get("source_atomic_cohort_ids", "").strip(),
        global_source_cohort_id=row.get("global_source_cohort_id", "").strip(),
        object_type=row.get("object_type", "").strip(),
        motif_visibility=row.get("motif_visibility", "").strip(),
    )
