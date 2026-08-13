"""Canonical development paths for Phase 5 and later.

This module is deliberately development-only. A sealed-test path is not part of
the reusable modelling API; final evaluation must use a separate entry point
after the full decision pipeline is frozen.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union


PathLike = Union[str, Path]


@dataclass(frozen=True)
class ProjectPaths:
    """Resolve stable Step 02 inputs without exposing the sealed test."""

    step_root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "step_root", self.step_root.expanduser().resolve())

    @classmethod
    def discover(cls) -> "ProjectPaths":
        """Build paths from the installed source-tree location."""

        return cls(Path(__file__).resolve().parents[2])

    @classmethod
    def from_step_root(cls, step_root: PathLike) -> "ProjectPaths":
        """Build paths for an explicitly supplied Step 02 root."""

        return cls(Path(step_root))

    @property
    def data_root(self) -> Path:
        return self.step_root / "data"

    @property
    def workspace_root(self) -> Path:
        """Repository root containing ``step_2/`` and the shared cache."""

        if self.step_root.name == "development" and self.step_root.parent.name == "step_2":
            return self.step_root.parents[1]
        return self.step_root.parent

    def resolve_recorded_path(self, relative_path: str) -> Path:
        """Resolve an immutable pre-restructure path without rewriting evidence.

        Frozen Phase 5 contracts record research files under the historical
        ``step_02/`` prefix. After the navigation-only move, that prefix maps
        to ``step_2/development/``. Other paths, notably ``.cache/step_02``,
        remain relative to the repository root.
        """

        supplied = Path(relative_path)
        if supplied.is_absolute() or not supplied.parts or ".." in supplied.parts:
            raise ValueError("Recorded paths must be non-empty repository-relative paths")
        if supplied.parts[0] == "step_02":
            resolved = (self.step_root / Path(*supplied.parts[1:])).resolve()
            boundary = self.step_root
        else:
            resolved = (self.workspace_root / supplied).resolve()
            boundary = self.workspace_root
        try:
            resolved.relative_to(boundary.resolve())
        except ValueError as error:
            raise ValueError(f"Recorded path escapes its boundary: {relative_path!r}") from error
        return resolved

    @property
    def development_source_roots(self) -> tuple[Path, Path]:
        """Physical source folders referenced by development-manifest rows.

        The historical folder names are provenance only. The canonical
        ``production_split`` field, not the old folder name, defines whether a
        row belongs to development.
        """

        return (
            self.data_root / "dataset_dev",
            self.data_root / "dataset_test",
        )

    @property
    def development_manifest_path(self) -> Path:
        return self.step_root / "splits" / "development.csv"

    @property
    def split_audit_path(self) -> Path:
        return self.step_root / "splits" / "split_audit.json"

    def resolve_development_image(self, relative_path: str) -> Path:
        """Resolve a manifest image while preventing absolute/path escapes."""

        supplied = Path(relative_path)
        if supplied.is_absolute() or not supplied.parts:
            raise ValueError("Development image paths must be non-empty and relative")
        roots_by_name = {root.name: root for root in self.development_source_roots}
        if supplied.parts[0] not in roots_by_name:
            raise ValueError(
                "Development image paths must remain inside a canonical source folder"
            )

        resolved_root = roots_by_name[supplied.parts[0]].resolve()
        candidate = (self.data_root / supplied).resolve()
        try:
            candidate.relative_to(resolved_root)
        except ValueError as error:
            raise ValueError(
                f"Development image path escapes its data root: {relative_path!r}"
            ) from error
        return candidate
