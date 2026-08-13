#!/usr/bin/env python3
"""Build the deterministic, leakage-safe Step 02 evaluation split.

The original dataset_dev/dataset_test folders are treated only as provenance.
Split assignment is driven by the completed Phase 1 source/object review, the
pretrained-similarity adjudication, and the complete global acquisition-source
cohort review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Iterable, Sequence


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
DEFAULT_MANIFEST = STEP_ROOT / "metadata" / "manifest.csv"
DEFAULT_REVIEW = (
    STEP_ROOT
    / "review"
    / "outputs"
    / "019f6af8-e0cf-7513-85b2-f6eb3920df46"
    / "phase_1_manual_review.json"
)
DEFAULT_OUTPUT_DIR = STEP_ROOT / "splits"
DEFAULT_SIMILARITY_REVIEW_DIR = STEP_ROOT / "review" / "pretrained_similarity"
DEFAULT_SIMILARITY_AUDIT = DEFAULT_SIMILARITY_REVIEW_DIR / "audit.json"
DEFAULT_SIMILARITY_COMPONENTS = DEFAULT_SIMILARITY_REVIEW_DIR / "component_candidates.csv"
DEFAULT_SIMILARITY_MEMBERS = DEFAULT_SIMILARITY_REVIEW_DIR / "component_members.csv"
DEFAULT_SIMILARITY_PAIRS = DEFAULT_SIMILARITY_REVIEW_DIR / "pair_candidates.csv"
DEFAULT_ADJUDICATION = DEFAULT_SIMILARITY_REVIEW_DIR / "adjudication.json"
DEFAULT_GLOBAL_SOURCE_ADJUDICATION = (
    STEP_ROOT / "review" / "global_source_cohorts" / "adjudication.json"
)

SPLIT_VERSION = "step02_source_atomic_v3"
GLOBAL_SOURCE_REVIEW_VERSION = "global_source_cohorts_v1"
DEFAULT_SEED = 20260719
PRODUCTION_SPLITS = ("train", "validation", "test")
PRODUCTION_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
CV_FOLDS = 5

OUTPUT_FIELDS = (
    "image_id",
    "relative_path",
    "ornament_label",
    "old_split",
    "file_name",
    "content_sha256",
    "width",
    "height",
    "quality_flags",
    "candidate_object_group_id",
    "manual_review_status",
    "confirmed_object_group_id",
    "confirmed_source_group_id",
    "presemantic_split_group_id",
    "semantic_split_group_id",
    "global_source_cohort_id",
    "source_atomic_cohort_ids",
    "pre_source_cohort_split_group_id",
    "source_atomic_split_group_id",
    "split_group_id",
    "semantic_review_component_ids",
    "semantic_review_decision",
    "semantic_merge_component_ids",
    "semantic_merge_group_ids",
    "semantic_merge_relationships",
    "semantic_similarity_audit_fingerprint_sha256",
    "semantic_adjudication_file_sha256",
    "global_source_inventory_projection_fingerprint_sha256",
    "global_source_cohort_file_sha256",
    "object_type",
    "motif_visibility",
    "label_status",
    "training_use",
    "inclusion_status",
    "exclusion_reason",
    "production_split",
    "cv_fold",
    "split_version",
    "split_seed",
)


@dataclass(frozen=True)
class AllocationGroup:
    group_id: str
    label: str
    image_ids: tuple[str, ...]

    @property
    def size(self) -> int:
        return len(self.image_ids)


@dataclass(frozen=True)
class SemanticMergeGroup:
    component_id: str
    group_id: str
    relationship: str
    image_ids: tuple[str, ...]


@dataclass(frozen=True)
class SemanticAdjudication:
    audit_fingerprint_sha256: str
    adjudication_sha256: str
    component_members: dict[str, tuple[str, ...]]
    component_decisions: dict[str, str]
    merge_groups: tuple[SemanticMergeGroup, ...]


@dataclass(frozen=True)
class GlobalSourceCohort:
    """One explicitly reviewed same-acquisition-source cohort.

    ``image_ids`` contains direct human-reviewed members only. Applying the
    cohort later propagates the relationship through every member's existing
    semantic-v2 group.
    """

    cohort_id: str
    ornament_label: str
    relationship: str
    image_ids: tuple[str, ...]
    evidence: str
    notes: str


@dataclass(frozen=True)
class GlobalSourceAdjudication:
    """Strictly bound global-source review used to make source-atomic groups.

    File schema (unknown extra keys are tolerated for reviewer provenance)::

        {
          "review_version": "global_source_cohorts_v1",
          "review_status": "complete",
          "inventory_projection_fingerprint_sha256": "<sha256>",
          "cohorts": [{
            "cohort_id": "<unique>",
            "ornament_label": "<one inventory class>",
            "relationship": "same_acquisition_source",
            "image_ids": ["<at least two explicit, non-overlapping IDs>"],
            "evidence": "<non-empty human evidence>",
            "notes": "<string; may be empty>"
          }]
        }

    The inventory/projection fingerprint binds the review to all post-semantic
    rows, including exclusions, so inventory or semantic-group changes require
    a fresh review.
    """

    review_version: str
    inventory_projection_fingerprint_sha256: str
    adjudication_sha256: str
    cohorts: tuple[GlobalSourceCohort, ...]


class UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        # Root choice is deterministic and independent of adjudication file ordering.
        first, second = sorted((left_root, right_root))
        self.parent[second] = first


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--review-json", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--similarity-audit", type=Path, default=DEFAULT_SIMILARITY_AUDIT)
    parser.add_argument(
        "--similarity-components", type=Path, default=DEFAULT_SIMILARITY_COMPONENTS
    )
    parser.add_argument("--similarity-members", type=Path, default=DEFAULT_SIMILARITY_MEMBERS)
    parser.add_argument("--similarity-pairs", type=Path, default=DEFAULT_SIMILARITY_PAIRS)
    parser.add_argument("--adjudication-json", type=Path, default=DEFAULT_ADJUDICATION)
    parser.add_argument(
        "--global-source-adjudication",
        type=Path,
        default=DEFAULT_GLOBAL_SOURCE_ADJUDICATION,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def stable_digest(*parts: object) -> str:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def stable_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def integer_targets(total: int, ratios: dict[str, float], order: Sequence[str]) -> dict[str, int]:
    raw = {name: total * ratios[name] for name in order}
    targets = {name: math.floor(raw[name]) for name in order}
    remainder = total - sum(targets.values())
    ranked = sorted(order, key=lambda name: (-(raw[name] - targets[name]), order.index(name)))
    for name in ranked[:remainder]:
        targets[name] += 1
    return targets


def allocate_groups(
    groups: Sequence[AllocationGroup],
    targets: dict[str, int],
    destinations: Sequence[str],
    *,
    seed: int,
    namespace: str,
) -> dict[str, str]:
    """Allocate groups without splitting them, using singleton groups to hit exact targets."""

    if sum(group.size for group in groups) != sum(targets.values()):
        raise ValueError("Allocation target does not match the number of images")

    assignments: dict[str, str] = {}
    remaining = dict(targets)
    multi = sorted(
        (group for group in groups if group.size > 1),
        key=lambda group: (-group.size, stable_digest(seed, namespace, group.group_id)),
    )
    singletons = sorted(
        (group for group in groups if group.size == 1),
        key=lambda group: stable_digest(seed, namespace, group.group_id),
    )

    for group in multi:
        eligible = [name for name in destinations if remaining[name] >= group.size]
        if not eligible:
            raise ValueError(
                f"Cannot place group {group.group_id} of size {group.size}; remaining={remaining}"
            )
        destination = max(
            eligible,
            key=lambda name: (
                remaining[name] / max(targets[name], 1),
                remaining[name],
                -destinations.index(name),
            ),
        )
        assignments[group.group_id] = destination
        remaining[destination] -= group.size

    for group in singletons:
        eligible = [name for name in destinations if remaining[name] > 0]
        if not eligible:
            raise ValueError(f"No remaining destination for singleton {group.group_id}")
        destination = max(
            eligible,
            key=lambda name: (
                remaining[name] / max(targets[name], 1),
                remaining[name],
                -destinations.index(name),
            ),
        )
        assignments[group.group_id] = destination
        remaining[destination] -= 1

    if any(remaining.values()):
        raise ValueError(f"Allocation did not consume targets: {remaining}")
    return assignments


def _normalized_allocation_objective(
    *,
    class_counts: dict[str, dict[str, int]],
    total_counts: dict[str, int],
    class_targets: dict[str, dict[str, int]],
    total_targets: dict[str, int],
    labels: Sequence[str],
    destinations: Sequence[str],
) -> Fraction:
    """Return normalized class plus total L1 target deviation exactly."""

    objective = Fraction(0, 1)
    for label in labels:
        for destination in destinations:
            target = class_targets[label][destination]
            objective += Fraction(
                abs(class_counts[label][destination] - target), max(target, 1)
            )
    for destination in destinations:
        target = total_targets[destination]
        objective += Fraction(
            abs(total_counts[destination] - target), max(target, 1)
        )
    return objective


def allocate_groups_approximately(
    groups: Sequence[AllocationGroup],
    ratios: dict[str, float],
    destinations: Sequence[str],
    *,
    seed: int,
    namespace: str,
) -> dict[str, str]:
    """Deterministically stratify indivisible groups near requested ratios.

    Large acquisition cohorts can exceed a class destination's target, so an
    exact quota allocator is invalid for source-atomic v3. This greedy allocator
    minimizes normalized class plus total L1 deviation and forces one group of
    every class into every destination whenever enough groups exist. Groups are
    ordered largest-relative-to-class first, spreading the most consequential
    cohorts before small groups fill remaining gaps. A deterministic improving
    move pass then reduces the same objective without sacrificing class cover.
    """

    if not groups:
        return {}
    if set(ratios) != set(destinations):
        raise ValueError("Allocation ratios and destinations differ")
    if any(ratios[destination] <= 0 for destination in destinations):
        raise ValueError("Every approximate-allocation ratio must be positive")
    if not math.isclose(sum(ratios.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Approximate-allocation ratios must sum to one")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("Allocation group IDs are not unique")

    groups_by_label: dict[str, list[AllocationGroup]] = defaultdict(list)
    for group in groups:
        if group.size < 1:
            raise ValueError(f"Allocation group is empty: {group.group_id}")
        groups_by_label[group.label].append(group)
    labels = tuple(sorted(groups_by_label))
    for label in labels:
        if len(groups_by_label[label]) < len(destinations):
            raise ValueError(
                f"Cannot retain class {label} in every destination: "
                f"{len(groups_by_label[label])} groups for {len(destinations)} destinations"
            )

    class_totals = {
        label: sum(group.size for group in groups_by_label[label]) for label in labels
    }
    total = sum(class_totals.values())
    class_targets = {
        label: integer_targets(class_totals[label], ratios, destinations)
        for label in labels
    }
    total_targets = integer_targets(total, ratios, destinations)
    class_counts = {
        label: {destination: 0 for destination in destinations} for label in labels
    }
    class_group_counts = {
        label: {destination: 0 for destination in destinations} for label in labels
    }
    total_counts = {destination: 0 for destination in destinations}
    remaining_groups = Counter(group.label for group in groups)

    ordered = sorted(
        groups,
        key=lambda group: (
            -Fraction(group.size, class_totals[group.label]),
            -group.size,
            stable_digest(seed, namespace, group.label, group.group_id),
        ),
    )
    assignments: dict[str, str] = {}
    group_by_id = {group.group_id: group for group in groups}

    for group in ordered:
        label = group.label
        empty_destinations = [
            destination
            for destination in destinations
            if class_group_counts[label][destination] == 0
        ]
        # If every remaining group is needed for missing destinations, placing
        # this group anywhere else would make full class coverage impossible.
        eligible = (
            empty_destinations
            if len(empty_destinations) == remaining_groups[label]
            else list(destinations)
        )
        scored: list[tuple[Fraction, str, str]] = []
        for destination in eligible:
            class_counts[label][destination] += group.size
            total_counts[destination] += group.size
            score = _normalized_allocation_objective(
                class_counts=class_counts,
                total_counts=total_counts,
                class_targets=class_targets,
                total_targets=total_targets,
                labels=labels,
                destinations=destinations,
            )
            class_counts[label][destination] -= group.size
            total_counts[destination] -= group.size
            scored.append(
                (
                    score,
                    stable_digest(
                        seed, namespace, "destination", group.group_id, destination
                    ),
                    destination,
                )
            )
        _, _, destination = min(scored)
        assignments[group.group_id] = destination
        class_counts[label][destination] += group.size
        class_group_counts[label][destination] += 1
        total_counts[destination] += group.size
        remaining_groups[label] -= 1

    current_objective = _normalized_allocation_objective(
        class_counts=class_counts,
        total_counts=total_counts,
        class_targets=class_targets,
        total_targets=total_targets,
        labels=labels,
        destinations=destinations,
    )
    while True:
        improving_moves: list[tuple[Fraction, str, str, str]] = []
        for group_id in sorted(assignments):
            group = group_by_id[group_id]
            source = assignments[group_id]
            if class_group_counts[group.label][source] <= 1:
                continue
            for destination in destinations:
                if destination == source:
                    continue
                class_counts[group.label][source] -= group.size
                class_counts[group.label][destination] += group.size
                total_counts[source] -= group.size
                total_counts[destination] += group.size
                candidate_objective = _normalized_allocation_objective(
                    class_counts=class_counts,
                    total_counts=total_counts,
                    class_targets=class_targets,
                    total_targets=total_targets,
                    labels=labels,
                    destinations=destinations,
                )
                class_counts[group.label][source] += group.size
                class_counts[group.label][destination] -= group.size
                total_counts[source] += group.size
                total_counts[destination] -= group.size
                if candidate_objective < current_objective:
                    improving_moves.append(
                        (
                            candidate_objective,
                            stable_digest(
                                seed,
                                namespace,
                                "improving_move",
                                group_id,
                                source,
                                destination,
                            ),
                            group_id,
                            destination,
                        )
                    )
        if not improving_moves:
            break
        candidate_objective, _, group_id, destination = min(improving_moves)
        group = group_by_id[group_id]
        source = assignments[group_id]
        assignments[group_id] = destination
        class_counts[group.label][source] -= group.size
        class_counts[group.label][destination] += group.size
        class_group_counts[group.label][source] -= 1
        class_group_counts[group.label][destination] += 1
        total_counts[source] -= group.size
        total_counts[destination] += group.size
        current_objective = candidate_objective

    return assignments


def exclusion_reason(review_row: dict[str, object]) -> str:
    training_use = str(review_row["training_use"])
    if training_use == "exclude_exact_duplicate":
        return "exact_or_reencoded_duplicate"
    if training_use == "exclude_low_quality":
        return "no_usable_visible_motif"
    if str(review_row["label_status"]) == "incorrect":
        return "incorrect_label_requires_relabeling"
    return ""


def enrich_manifest(
    manifest: list[dict[str, str]], review: dict[str, object], seed: int
) -> list[dict[str, object]]:
    manifest_by_id = {row["image_id"]: row for row in manifest}
    if len(manifest_by_id) != len(manifest):
        raise ValueError("Manifest image IDs are not unique")

    members_by_candidate: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        members_by_candidate[row["candidate_object_group_id"]].append(row["image_id"])
    expected_review_ids = {
        image_id
        for members in members_by_candidate.values()
        if len(members) > 1
        for image_id in members
    }

    reviewed_rows = {
        str(row["image_id"]): row for row in review["image_decisions"]  # type: ignore[index]
    }
    if len(reviewed_rows) != len(review["image_decisions"]):  # type: ignore[arg-type,index]
        raise ValueError("Manual review image IDs are not unique")
    if set(reviewed_rows) != expected_review_ids:
        missing = sorted(expected_review_ids - set(reviewed_rows))
        unexpected = sorted(set(reviewed_rows) - expected_review_ids)
        raise ValueError(
            f"Manual review does not exactly cover multi-image candidates; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    enriched: list[dict[str, object]] = []
    for source in manifest:
        image_id = source["image_id"]
        manual = reviewed_rows.get(image_id)
        if manual is None:
            object_group_id = f"obj_single_{image_id.removeprefix('img_')}"
            source_group_id = ""
            object_type = "unreviewed"
            motif_visibility = "unreviewed"
            label_status = "provided_folder_label"
            training_use = "keep_primary"
            manual_review_status = "not_required_singleton"
            reason = ""
        else:
            if manual["review_status"] != "complete":
                raise ValueError(f"Incomplete manual review for {image_id}")
            object_group_id = str(manual["confirmed_object_group_id"])
            source_group_id = str(manual["confirmed_source_group_id"])
            object_type = str(manual["object_type"])
            motif_visibility = str(manual["motif_visibility"])
            label_status = str(manual["label_status"])
            training_use = str(manual["training_use"])
            manual_review_status = "complete"
            reason = exclusion_reason(manual)

        if not object_group_id:
            raise ValueError(f"Missing object group for {image_id}")
        if training_use == "needs_decision":
            raise ValueError(f"Unresolved training decision for {image_id}")

        included = not reason
        split_group_id = source_group_id or object_group_id
        enriched.append(
            {
                "image_id": image_id,
                "relative_path": source["relative_path"],
                "ornament_label": source["ornament_label"],
                "old_split": source["current_split"],
                "file_name": source["file_name"],
                "content_sha256": source["content_sha256"],
                "width": source["width"],
                "height": source["height"],
                "quality_flags": source["quality_flags"],
                "candidate_object_group_id": source["candidate_object_group_id"],
                "manual_review_status": manual_review_status,
                "confirmed_object_group_id": object_group_id,
                "confirmed_source_group_id": source_group_id,
                "split_group_id": split_group_id,
                "object_type": object_type,
                "motif_visibility": motif_visibility,
                "label_status": label_status,
                "training_use": training_use,
                "inclusion_status": "included" if included else "excluded",
                "exclusion_reason": reason,
                "production_split": "",
                "cv_fold": "",
                "split_version": SPLIT_VERSION,
                "split_seed": seed,
            }
        )
    return enriched


def load_semantic_adjudication(
    rows: Sequence[dict[str, object]],
    *,
    audit_path: Path,
    component_path: Path,
    member_path: Path,
    pair_path: Path,
    adjudication_path: Path,
) -> SemanticAdjudication:
    """Load and strictly bind the human adjudication to its similarity audit."""

    if not adjudication_path.is_file():
        raise FileNotFoundError(
            "The semantic split cannot be built before similarity adjudication is complete: "
            f"missing {adjudication_path}"
        )
    required_paths = (audit_path, component_path, member_path, pair_path)
    missing_paths = [str(path) for path in required_paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError(f"Missing similarity-audit inputs: {missing_paths}")

    with audit_path.open(encoding="utf-8") as handle:
        similarity_audit = json.load(handle)
    with adjudication_path.open(encoding="utf-8") as handle:
        adjudication = json.load(handle)
    if not isinstance(similarity_audit, dict) or not isinstance(adjudication, dict):
        raise ValueError("Similarity audit and adjudication must be JSON objects")

    audit_output_paths = {
        "pair_candidates.csv": pair_path,
        "component_candidates.csv": component_path,
        "component_members.csv": member_path,
    }
    actual_output_hashes = {
        name: sha256_file(path) for name, path in audit_output_paths.items()
    }
    if actual_output_hashes != similarity_audit.get("output_sha256"):
        raise ValueError("Similarity-audit CSV hashes do not match audit.json")
    fingerprint_payload = {
        "audit_version": similarity_audit.get("audit_version"),
        "manifest_sha256": similarity_audit.get("input", {}).get("manifest_sha256"),
        "embedding_fingerprint_sha256": similarity_audit.get("embedding", {}).get(
            "embedding_fingerprint_sha256"
        ),
        "thresholds": similarity_audit.get("thresholds"),
        "output_sha256": actual_output_hashes,
    }
    actual_audit_fingerprint = hashlib.sha256(
        stable_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    if actual_audit_fingerprint != similarity_audit.get("audit_fingerprint_sha256"):
        raise ValueError("Similarity-audit fingerprint does not validate")
    if adjudication.get("audit_fingerprint_sha256") != actual_audit_fingerprint:
        raise ValueError("Adjudication targets a different similarity-audit fingerprint")
    if adjudication.get("review_status") != "complete":
        raise ValueError("Top-level semantic adjudication is not complete")

    component_rows = read_csv(component_path)
    member_rows = read_csv(member_path)
    pair_rows = read_csv(pair_path)
    component_by_id = {row["component_id"]: row for row in component_rows}
    if len(component_by_id) != len(component_rows):
        raise ValueError("Similarity component IDs are not unique")

    members_by_component: dict[str, list[str]] = defaultdict(list)
    member_metadata: dict[str, dict[str, str]] = {}
    for member in member_rows:
        component_id = member["component_id"]
        image_id = member["image_id"]
        if image_id in member_metadata:
            raise ValueError(f"Similarity-audit image appears in multiple components: {image_id}")
        members_by_component[component_id].append(image_id)
        member_metadata[image_id] = member
    if set(members_by_component) != set(component_by_id):
        raise ValueError("Similarity component/member files cover different component IDs")
    for component_id, component in component_by_id.items():
        expected_members = sorted(component["image_ids"].split("|"))
        observed_members = sorted(members_by_component[component_id])
        if observed_members != expected_members:
            raise ValueError(f"Similarity component membership mismatch for {component_id}")
        if len(observed_members) != int(component["member_count"]):
            raise ValueError(f"Similarity member count mismatch for {component_id}")
        observed_pair_count = sum(
            pair["component_id"] == component_id for pair in pair_rows
        )
        if observed_pair_count != int(component["pair_count"]):
            raise ValueError(f"Similarity pair count mismatch for {component_id}")

    summary = similarity_audit.get("summary", {})
    expected_component_count = int(summary.get("candidate_component_count", -1))
    if expected_component_count != 46 or len(component_by_id) != expected_component_count:
        raise ValueError("Semantic adjudication must cover the complete 46-component audit")
    if len(member_rows) != int(summary.get("candidate_image_count", -1)):
        raise ValueError("Similarity candidate image count does not match audit.json")
    if len(pair_rows) != int(summary.get("candidate_pair_count", -1)):
        raise ValueError("Similarity candidate pair count does not match audit.json")

    row_by_id = {str(row["image_id"]): row for row in rows}
    included_count = sum(row["inclusion_status"] == "included" for row in rows)
    excluded_count = len(rows) - included_count
    audit_input = similarity_audit.get("input", {})
    if included_count != int(audit_input.get("included_image_count", -1)):
        raise ValueError("Current included inventory differs from the similarity audit")
    if excluded_count != int(audit_input.get("excluded_image_count", -1)):
        raise ValueError("Current excluded inventory differs from the similarity audit")
    for image_id, member in member_metadata.items():
        row = row_by_id.get(image_id)
        if row is None or row["inclusion_status"] != "included":
            raise ValueError(f"Similarity candidate is absent or excluded: {image_id}")
        for audit_field, current_field in (
            ("ornament_label", "ornament_label"),
            ("confirmed_object_group_id", "confirmed_object_group_id"),
            ("confirmed_source_group_id", "confirmed_source_group_id"),
        ):
            if member[audit_field] != str(row[current_field]):
                raise ValueError(
                    f"Similarity member provenance changed for {image_id}: {audit_field}"
                )

    components = adjudication.get("components")
    if not isinstance(components, list) or len(components) != 46:
        raise ValueError("Adjudication components must be a list of exactly 46 decisions")
    decision_by_component: dict[str, dict[str, object]] = {}
    for decision in components:
        if not isinstance(decision, dict):
            raise ValueError("Every adjudication component must be an object")
        component_id = str(decision.get("component_id", ""))
        if not component_id or component_id in decision_by_component:
            raise ValueError(f"Missing or duplicate adjudication component ID: {component_id}")
        decision_by_component[component_id] = decision
    if set(decision_by_component) != set(component_by_id):
        missing = sorted(set(component_by_id) - set(decision_by_component))
        unexpected = sorted(set(decision_by_component) - set(component_by_id))
        raise ValueError(
            "Adjudication does not exactly cover audited components; "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )

    allowed_decisions = {
        "same_physical_object",
        "same_source_distinct_objects",
        "distinct_unrelated",
        "partitioned",
    }
    allowed_relationships = {"same_physical_object", "same_source"}
    component_decisions: dict[str, str] = {}
    merge_groups: list[SemanticMergeGroup] = []
    seen_group_ids: set[str] = set()
    for component_id in sorted(decision_by_component):
        component = decision_by_component[component_id]
        if component.get("review_status") != "complete":
            raise ValueError(f"Incomplete semantic component decision: {component_id}")
        if "notes" not in component or not isinstance(component["notes"], str):
            raise ValueError(f"Semantic component notes are missing: {component_id}")
        decision_name = str(component.get("decision", ""))
        if decision_name not in allowed_decisions:
            raise ValueError(f"Invalid decision for {component_id}: {decision_name}")
        component_decisions[component_id] = decision_name
        groups = component.get("groups")
        if not isinstance(groups, list):
            raise ValueError(f"Adjudication groups must be a list for {component_id}")
        expected_members = set(members_by_component[component_id])
        if decision_name == "distinct_unrelated" and groups:
            raise ValueError(f"Distinct/unrelated component must not merge groups: {component_id}")
        if decision_name in {"same_physical_object", "same_source_distinct_objects"}:
            if len(groups) != 1:
                raise ValueError(f"Simple semantic decision requires one group: {component_id}")
        if decision_name == "partitioned" and not groups:
            raise ValueError(f"Partitioned decision requires at least one merge group: {component_id}")

        listed_members: set[str] = set()
        for group in groups:
            if not isinstance(group, dict):
                raise ValueError(f"Adjudication group must be an object: {component_id}")
            group_id = str(group.get("group_id", ""))
            relationship = str(group.get("relationship", ""))
            image_ids = group.get("image_ids")
            if not group_id or group_id in seen_group_ids:
                raise ValueError(f"Missing or duplicate semantic group ID: {group_id}")
            if relationship not in allowed_relationships:
                raise ValueError(f"Invalid semantic relationship in {group_id}: {relationship}")
            if not isinstance(image_ids, list) or len(image_ids) < 2:
                raise ValueError(f"Semantic merge group needs at least two images: {group_id}")
            normalized_ids = tuple(sorted(str(image_id) for image_id in image_ids))
            if len(set(normalized_ids)) != len(normalized_ids):
                raise ValueError(f"Duplicate image in semantic merge group: {group_id}")
            normalized_set = set(normalized_ids)
            if not normalized_set <= expected_members:
                raise ValueError(f"Semantic merge group contains non-component member: {group_id}")
            if listed_members & normalized_set:
                raise ValueError(f"Partitioned semantic groups overlap in {component_id}")
            listed_members.update(normalized_set)
            seen_group_ids.add(group_id)
            merge_groups.append(
                SemanticMergeGroup(component_id, group_id, relationship, normalized_ids)
            )

        if decision_name in {"same_physical_object", "same_source_distinct_objects"}:
            if listed_members != expected_members:
                raise ValueError(f"Simple semantic group must cover all members: {component_id}")
            expected_relationship = (
                "same_physical_object"
                if decision_name == "same_physical_object"
                else "same_source"
            )
            if merge_groups[-1].relationship != expected_relationship:
                raise ValueError(f"Decision/relationship mismatch for {component_id}")

    return SemanticAdjudication(
        audit_fingerprint_sha256=actual_audit_fingerprint,
        adjudication_sha256=sha256_file(adjudication_path),
        component_members={
            component_id: tuple(sorted(image_ids))
            for component_id, image_ids in members_by_component.items()
        },
        component_decisions=component_decisions,
        merge_groups=tuple(sorted(merge_groups, key=lambda group: group.group_id)),
    )


def apply_semantic_adjudication(
    rows: list[dict[str, object]], adjudication: SemanticAdjudication
) -> None:
    """Union adjudicated relations through their existing source/object groups."""

    row_by_id = {str(row["image_id"]): row for row in rows}
    base_group_ids = {str(row["split_group_id"]) for row in rows}
    unions = UnionFind(base_group_ids)
    for group in adjudication.merge_groups:
        member_base_groups = sorted(
            {str(row_by_id[image_id]["split_group_id"]) for image_id in group.image_ids}
        )
        for base_group_id in member_base_groups[1:]:
            unions.union(member_base_groups[0], base_group_id)

    bases_by_root: dict[str, set[str]] = defaultdict(set)
    for base_group_id in sorted(base_group_ids):
        bases_by_root[unions.find(base_group_id)].add(base_group_id)
    semantic_id_by_root = {
        root: (
            next(iter(base_ids))
            if len(base_ids) == 1
            else f"semgrp_{stable_digest('semantic_v2', *sorted(base_ids))[:16]}"
        )
        for root, base_ids in bases_by_root.items()
    }

    merge_components_by_root: dict[str, set[str]] = defaultdict(set)
    merge_groups_by_root: dict[str, set[str]] = defaultdict(set)
    relationships_by_root: dict[str, set[str]] = defaultdict(set)
    for group in adjudication.merge_groups:
        root = unions.find(str(row_by_id[group.image_ids[0]]["split_group_id"]))
        merge_components_by_root[root].add(group.component_id)
        merge_groups_by_root[root].add(group.group_id)
        relationships_by_root[root].add(group.relationship)

    component_by_image = {
        image_id: component_id
        for component_id, image_ids in adjudication.component_members.items()
        for image_id in image_ids
    }
    for row in rows:
        image_id = str(row["image_id"])
        base_group_id = str(row["split_group_id"])
        root = unions.find(base_group_id)
        semantic_group_id = semantic_id_by_root[root]
        direct_component = component_by_image.get(image_id, "")
        row["presemantic_split_group_id"] = base_group_id
        row["semantic_split_group_id"] = semantic_group_id
        row["split_group_id"] = semantic_group_id
        row["semantic_review_component_ids"] = direct_component
        row["semantic_review_decision"] = (
            adjudication.component_decisions[direct_component] if direct_component else ""
        )
        row["semantic_merge_component_ids"] = "|".join(
            sorted(merge_components_by_root[root])
        )
        row["semantic_merge_group_ids"] = "|".join(sorted(merge_groups_by_root[root]))
        row["semantic_merge_relationships"] = "|".join(
            sorted(relationships_by_root[root])
        )
        row["semantic_similarity_audit_fingerprint_sha256"] = (
            adjudication.audit_fingerprint_sha256
        )
        row["semantic_adjudication_file_sha256"] = adjudication.adjudication_sha256


def source_inventory_projection_fingerprint(
    rows: Sequence[dict[str, object]],
) -> str:
    """Hash the exact post-semantic inventory projected for source review.

    The byte contract is deliberately simple so the independent review tool can
    reproduce it: rows sorted by ``image_id``; UTF-8 fields separated by ASCII
    unit separators; one LF-terminated record per row. All included and
    excluded rows participate.
    """

    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        digest.update(
            "\x1f".join(
                str(row[field])
                for field in (
                    "image_id",
                    "ornament_label",
                    "inclusion_status",
                    "semantic_split_group_id",
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def load_global_source_adjudication(
    rows: Sequence[dict[str, object]], adjudication_path: Path
) -> GlobalSourceAdjudication:
    """Load and strictly validate the complete global source-cohort review."""

    if not adjudication_path.is_file():
        raise FileNotFoundError(
            "The source-atomic split cannot be built before global source-cohort "
            f"adjudication is complete: missing {adjudication_path}"
        )
    with adjudication_path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Global source-cohort adjudication must be a JSON object")
    if payload.get("review_version") != GLOBAL_SOURCE_REVIEW_VERSION:
        raise ValueError(
            "Global source-cohort review_version must be "
            f"{GLOBAL_SOURCE_REVIEW_VERSION!r}"
        )
    if payload.get("review_status") != "complete":
        raise ValueError("Top-level global source-cohort adjudication is not complete")

    actual_projection_fingerprint = source_inventory_projection_fingerprint(rows)
    recorded_projection_fingerprint = payload.get(
        "inventory_projection_fingerprint_sha256"
    )
    if recorded_projection_fingerprint != actual_projection_fingerprint:
        raise ValueError(
            "Global source-cohort adjudication targets a different inventory/"
            "semantic projection fingerprint"
        )

    raw_cohorts = payload.get("cohorts")
    if not isinstance(raw_cohorts, list):
        raise ValueError("Global source-cohort cohorts must be a list")
    row_by_id = {str(row["image_id"]): row for row in rows}
    labels = {str(row["ornament_label"]) for row in rows}
    seen_cohort_ids: set[str] = set()
    seen_image_ids: set[str] = set()
    cohorts: list[GlobalSourceCohort] = []
    previous_cohort_id = ""
    for raw_cohort in raw_cohorts:
        if not isinstance(raw_cohort, dict):
            raise ValueError("Every global source cohort must be a JSON object")
        cohort_id = str(raw_cohort.get("cohort_id", ""))
        label = str(raw_cohort.get("ornament_label", ""))
        relationship = str(raw_cohort.get("relationship", ""))
        image_ids = raw_cohort.get("image_ids")
        evidence = raw_cohort.get("evidence")
        notes = raw_cohort.get("notes")
        if not cohort_id or cohort_id in seen_cohort_ids:
            raise ValueError(f"Missing or duplicate global source cohort ID: {cohort_id}")
        if cohort_id <= previous_cohort_id:
            raise ValueError("Global source cohorts must be sorted by cohort_id")
        previous_cohort_id = cohort_id
        if label not in labels:
            raise ValueError(f"Unknown ornament label in source cohort {cohort_id}: {label}")
        if relationship != "same_acquisition_source":
            raise ValueError(
                f"Invalid relationship in source cohort {cohort_id}: {relationship}"
            )
        if not isinstance(image_ids, list) or len(image_ids) < 2:
            raise ValueError(f"Source cohort {cohort_id} needs at least two image IDs")
        normalized_ids = tuple(str(image_id) for image_id in image_ids)
        if tuple(sorted(normalized_ids)) != normalized_ids:
            raise ValueError(f"Source cohort image IDs must be sorted: {cohort_id}")
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError(f"Duplicate image ID within source cohort: {cohort_id}")
        overlap = seen_image_ids & set(normalized_ids)
        if overlap:
            raise ValueError(
                "Global source cohorts overlap on explicit image IDs: "
                f"{sorted(overlap)[:5]}"
            )
        if not isinstance(evidence, str) or not evidence.strip():
            raise ValueError(f"Source cohort evidence is missing: {cohort_id}")
        if not isinstance(notes, str):
            raise ValueError(f"Source cohort notes must be a string: {cohort_id}")
        for image_id in normalized_ids:
            row = row_by_id.get(image_id)
            if row is None:
                raise ValueError(f"Unknown image in source cohort {cohort_id}: {image_id}")
            if row["inclusion_status"] != "included":
                raise ValueError(f"Excluded image in source cohort {cohort_id}: {image_id}")
            if str(row["ornament_label"]) != label:
                raise ValueError(
                    f"Source cohort {cohort_id} crosses labels at image {image_id}"
                )
        seen_cohort_ids.add(cohort_id)
        seen_image_ids.update(normalized_ids)
        cohorts.append(
            GlobalSourceCohort(
                cohort_id=cohort_id,
                ornament_label=label,
                relationship=relationship,
                image_ids=normalized_ids,
                evidence=evidence,
                notes=notes,
            )
        )

    return GlobalSourceAdjudication(
        review_version=GLOBAL_SOURCE_REVIEW_VERSION,
        inventory_projection_fingerprint_sha256=actual_projection_fingerprint,
        adjudication_sha256=sha256_file(adjudication_path),
        cohorts=tuple(cohorts),
    )


def apply_global_source_adjudication(
    rows: list[dict[str, object]], adjudication: GlobalSourceAdjudication
) -> None:
    """Union direct source cohorts through complete existing semantic groups."""

    row_by_id = {str(row["image_id"]): row for row in rows}
    semantic_group_ids = {str(row["semantic_split_group_id"]) for row in rows}
    unions = UnionFind(semantic_group_ids)
    direct_cohort_by_image: dict[str, str] = {}
    for cohort in adjudication.cohorts:
        member_semantic_groups = sorted(
            {
                str(row_by_id[image_id]["semantic_split_group_id"])
                for image_id in cohort.image_ids
            }
        )
        for semantic_group_id in member_semantic_groups[1:]:
            unions.union(member_semantic_groups[0], semantic_group_id)
        for image_id in cohort.image_ids:
            direct_cohort_by_image[image_id] = cohort.cohort_id

    semantics_by_root: dict[str, set[str]] = defaultdict(set)
    for semantic_group_id in sorted(semantic_group_ids):
        semantics_by_root[unions.find(semantic_group_id)].add(semantic_group_id)
    source_id_by_root = {
        root: (
            next(iter(semantic_ids))
            if len(semantic_ids) == 1
            else f"srcgrp_{stable_digest('source_atomic_v3', *sorted(semantic_ids))[:16]}"
        )
        for root, semantic_ids in semantics_by_root.items()
    }
    cohort_ids_by_root: dict[str, set[str]] = defaultdict(set)
    for cohort in adjudication.cohorts:
        root = unions.find(
            str(row_by_id[cohort.image_ids[0]]["semantic_split_group_id"])
        )
        cohort_ids_by_root[root].add(cohort.cohort_id)

    for row in rows:
        image_id = str(row["image_id"])
        semantic_group_id = str(row["semantic_split_group_id"])
        root = unions.find(semantic_group_id)
        source_group_id = source_id_by_root[root]
        row["global_source_cohort_id"] = direct_cohort_by_image.get(image_id, "")
        row["source_atomic_cohort_ids"] = "|".join(sorted(cohort_ids_by_root[root]))
        row["pre_source_cohort_split_group_id"] = semantic_group_id
        row["source_atomic_split_group_id"] = source_group_id
        row["split_group_id"] = source_group_id
        row["global_source_inventory_projection_fingerprint_sha256"] = (
            adjudication.inventory_projection_fingerprint_sha256
        )
        row["global_source_cohort_file_sha256"] = adjudication.adjudication_sha256


def source_atomic_membership_fingerprint(rows: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        digest.update(
            "\x1f".join(
                str(row[field])
                for field in (
                    "image_id",
                    "global_source_cohort_id",
                    "source_atomic_cohort_ids",
                    "pre_source_cohort_split_group_id",
                    "source_atomic_split_group_id",
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def semantic_membership_fingerprint(rows: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        digest.update(
            "\x1f".join(
                str(row[field])
                for field in (
                    "image_id",
                    "presemantic_split_group_id",
                    "semantic_split_group_id",
                    "semantic_review_component_ids",
                    "semantic_review_decision",
                    "semantic_merge_component_ids",
                    "semantic_merge_group_ids",
                    "semantic_merge_relationships",
                )
            ).encode("utf-8")
        )
        digest.update(b"\n")
    return digest.hexdigest()


def allocation_groups(rows: Sequence[dict[str, object]]) -> list[AllocationGroup]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["split_group_id"])].append(row)

    result: list[AllocationGroup] = []
    for group_id, members in grouped.items():
        labels = {str(row["ornament_label"]) for row in members}
        if len(labels) != 1:
            raise ValueError(f"Split group {group_id} crosses labels: {sorted(labels)}")
        result.append(
            AllocationGroup(
                group_id=group_id,
                label=next(iter(labels)),
                image_ids=tuple(sorted(str(row["image_id"]) for row in members)),
            )
        )
    return result


def assign_production_splits(rows: list[dict[str, object]], seed: int) -> None:
    included = [row for row in rows if row["inclusion_status"] == "included"]
    groups = allocation_groups(included)
    assignment = allocate_groups_approximately(
        groups,
        PRODUCTION_RATIOS,
        PRODUCTION_SPLITS,
        seed=seed,
        namespace="production:source_atomic_v3",
    )

    for row in included:
        row["production_split"] = assignment[str(row["split_group_id"])]


def assign_cv_folds(rows: list[dict[str, object]], seed: int) -> None:
    development = [
        row
        for row in rows
        if row["inclusion_status"] == "included" and row["production_split"] != "test"
    ]
    groups = allocation_groups(development)
    destinations = tuple(str(index) for index in range(CV_FOLDS))
    ratios = {name: 1 / CV_FOLDS for name in destinations}
    assignment = allocate_groups_approximately(
        groups,
        ratios,
        destinations,
        seed=seed,
        namespace="cv:source_atomic_v3",
    )

    for row in development:
        row["cv_fold"] = assignment[str(row["split_group_id"])]


def allocation_diagnostics(
    rows: Sequence[dict[str, object]],
    *,
    destination_field: str,
    destinations: Sequence[str],
    ratios: dict[str, float],
) -> dict[str, object]:
    """Record target/observed class and total imbalance for an allocation."""

    labels = tuple(sorted({str(row["ornament_label"]) for row in rows}))
    totals_by_label = Counter(str(row["ornament_label"]) for row in rows)
    target_by_class = {
        label: integer_targets(totals_by_label[label], ratios, destinations)
        for label in labels
    }
    target_total = integer_targets(len(rows), ratios, destinations)
    observed_by_class = {
        label: {
            destination: sum(
                str(row["ornament_label"]) == label
                and str(row[destination_field]) == destination
                for row in rows
            )
            for destination in destinations
        }
        for label in labels
    }
    observed_total = {
        destination: sum(
            str(row[destination_field]) == destination for row in rows
        )
        for destination in destinations
    }
    deviation_by_class = {
        label: {
            destination: (
                observed_by_class[label][destination]
                - target_by_class[label][destination]
            )
            for destination in destinations
        }
        for label in labels
    }
    deviation_total = {
        destination: observed_total[destination] - target_total[destination]
        for destination in destinations
    }
    objective = _normalized_allocation_objective(
        class_counts=observed_by_class,
        total_counts=observed_total,
        class_targets=target_by_class,
        total_targets=target_total,
        labels=labels,
        destinations=destinations,
    )
    return {
        "ratios": {destination: ratios[destination] for destination in destinations},
        "target_total": target_total,
        "observed_total": observed_total,
        "deviation_total": deviation_total,
        "target_by_class": target_by_class,
        "observed_by_class": observed_by_class,
        "deviation_by_class": deviation_by_class,
        "max_absolute_total_deviation": max(
            (abs(value) for value in deviation_total.values()), default=0
        ),
        "max_absolute_class_deviation": max(
            (
                abs(value)
                for destinations_by_class in deviation_by_class.values()
                for value in destinations_by_class.values()
            ),
            default=0,
        ),
        "normalized_class_plus_total_l1_objective": round(float(objective), 12),
        "every_class_in_every_destination": all(
            observed_by_class[label][destination] > 0
            for label in labels
            for destination in destinations
        ),
    }


def build_summary_rows(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    included = [row for row in rows if row["inclusion_status"] == "included"]
    counts = Counter((str(row["production_split"]), str(row["ornament_label"])) for row in included)
    totals = Counter(str(row["ornament_label"]) for row in included)
    summary: list[dict[str, object]] = []
    for split in PRODUCTION_SPLITS:
        for label in sorted(totals):
            count = counts[(split, label)]
            summary.append(
                {
                    "production_split": split,
                    "ornament_label": label,
                    "image_count": count,
                    "class_total": totals[label],
                    "class_fraction": f"{count / totals[label]:.6f}",
                }
            )
    return summary


def assignment_fingerprint(rows: Sequence[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["image_id"])):
        payload = "\x1f".join(
            str(row[field])
            for field in (
                "image_id",
                "inclusion_status",
                "exclusion_reason",
                "presemantic_split_group_id",
                "semantic_split_group_id",
                "global_source_cohort_id",
                "source_atomic_cohort_ids",
                "pre_source_cohort_split_group_id",
                "source_atomic_split_group_id",
                "split_group_id",
                "production_split",
                "cv_fold",
                "semantic_similarity_audit_fingerprint_sha256",
                "semantic_adjudication_file_sha256",
                "global_source_inventory_projection_fingerprint_sha256",
                "global_source_cohort_file_sha256",
            )
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def audit(
    rows: Sequence[dict[str, object]],
    review: dict[str, object],
    semantic: SemanticAdjudication,
    source: GlobalSourceAdjudication,
    seed: int,
) -> dict[str, object]:
    included = [row for row in rows if row["inclusion_status"] == "included"]
    excluded = [row for row in rows if row["inclusion_status"] == "excluded"]
    split_counts = Counter(str(row["production_split"]) for row in included)
    class_counts = Counter(str(row["ornament_label"]) for row in included)
    excluded_counts = Counter(str(row["exclusion_reason"]) for row in excluded)
    cv_counts = Counter(str(row["cv_fold"]) for row in included if row["production_split"] != "test")

    split_group_splits: dict[str, set[str]] = defaultdict(set)
    source_atomic_group_splits: dict[str, set[str]] = defaultdict(set)
    source_cohort_splits: dict[str, set[str]] = defaultdict(set)
    semantic_group_splits: dict[str, set[str]] = defaultdict(set)
    object_group_splits: dict[str, set[str]] = defaultdict(set)
    content_splits: dict[str, set[str]] = defaultdict(set)
    candidate_splits: dict[str, set[str]] = defaultdict(set)
    for row in included:
        split = str(row["production_split"])
        split_group_splits[str(row["split_group_id"])].add(split)
        source_atomic_group_splits[str(row["source_atomic_split_group_id"])].add(split)
        if row["global_source_cohort_id"]:
            source_cohort_splits[str(row["global_source_cohort_id"])].add(split)
        semantic_group_splits[str(row["semantic_split_group_id"])].add(split)
        object_group_splits[str(row["confirmed_object_group_id"])].add(split)
        content_splits[str(row["content_sha256"])].add(split)
        candidate_splits[str(row["candidate_object_group_id"])].add(split)

    split_group_violations = sorted(group for group, splits in split_group_splits.items() if len(splits) > 1)
    source_atomic_group_violations = sorted(
        group
        for group, splits in source_atomic_group_splits.items()
        if len(splits) > 1
    )
    source_cohort_violations = sorted(
        cohort for cohort, splits in source_cohort_splits.items() if len(splits) > 1
    )
    semantic_group_violations = sorted(
        group for group, splits in semantic_group_splits.items() if len(splits) > 1
    )
    object_group_violations = sorted(group for group, splits in object_group_splits.items() if len(splits) > 1)
    content_hash_violations = sorted(value for value, splits in content_splits.items() if len(splits) > 1)
    adjudicated_candidate_crossings = sorted(
        group for group, splits in candidate_splits.items() if len(splits) > 1
    )

    development = [row for row in included if row["production_split"] != "test"]
    cv_group_folds: dict[str, set[str]] = defaultdict(set)
    cv_source_atomic_folds: dict[str, set[str]] = defaultdict(set)
    cv_source_cohort_folds: dict[str, set[str]] = defaultdict(set)
    cv_semantic_folds: dict[str, set[str]] = defaultdict(set)
    cv_object_folds: dict[str, set[str]] = defaultdict(set)
    cv_content_folds: dict[str, set[str]] = defaultdict(set)
    for row in development:
        fold = str(row["cv_fold"])
        cv_group_folds[str(row["split_group_id"])].add(fold)
        cv_source_atomic_folds[str(row["source_atomic_split_group_id"])].add(fold)
        if row["global_source_cohort_id"]:
            cv_source_cohort_folds[str(row["global_source_cohort_id"])].add(fold)
        cv_semantic_folds[str(row["semantic_split_group_id"])].add(fold)
        cv_object_folds[str(row["confirmed_object_group_id"])].add(fold)
        cv_content_folds[str(row["content_sha256"])].add(fold)

    cv_group_violations = sorted(group for group, folds in cv_group_folds.items() if len(folds) > 1)
    cv_source_atomic_violations = sorted(
        group
        for group, folds in cv_source_atomic_folds.items()
        if len(folds) > 1
    )
    cv_source_cohort_violations = sorted(
        cohort
        for cohort, folds in cv_source_cohort_folds.items()
        if len(folds) > 1
    )
    cv_semantic_violations = sorted(
        group for group, folds in cv_semantic_folds.items() if len(folds) > 1
    )
    cv_object_violations = sorted(group for group, folds in cv_object_folds.items() if len(folds) > 1)
    cv_content_violations = sorted(value for value, folds in cv_content_folds.items() if len(folds) > 1)

    class_split_counts: dict[str, dict[str, int]] = {}
    for label in sorted(class_counts):
        class_split_counts[label] = {
            split: sum(
                row["ornament_label"] == label and row["production_split"] == split
                for row in included
            )
            for split in PRODUCTION_SPLITS
        }

    groups = allocation_groups(included)
    presemantic_groups = {
        str(row["presemantic_split_group_id"]) for row in included
    }
    semantic_group_bases: dict[str, set[str]] = defaultdict(set)
    for row in included:
        semantic_group_bases[str(row["semantic_split_group_id"])].add(
            str(row["presemantic_split_group_id"])
        )
    source_group_semantics: dict[str, set[str]] = defaultdict(set)
    for row in included:
        source_group_semantics[str(row["source_atomic_split_group_id"])].add(
            str(row["pre_source_cohort_split_group_id"])
        )
    merge_decision_counts = Counter(
        semantic.component_decisions.values()
    )
    production_diagnostics = allocation_diagnostics(
        included,
        destination_field="production_split",
        destinations=PRODUCTION_SPLITS,
        ratios=PRODUCTION_RATIOS,
    )
    cv_destinations = tuple(str(index) for index in range(CV_FOLDS))
    cv_ratios = {destination: 1 / CV_FOLDS for destination in cv_destinations}
    cv_diagnostics = allocation_diagnostics(
        development,
        destination_field="cv_fold",
        destinations=cv_destinations,
        ratios=cv_ratios,
    )
    return {
        "split_version": SPLIT_VERSION,
        "split_seed": seed,
        "policy": {
            "production_ratios": PRODUCTION_RATIOS,
            "cv_folds": CV_FOLDS,
            "group_precedence": (
                "global same-acquisition-source union over human-adjudicated semantic "
                "union, then confirmed_source_group_id, then confirmed_object_group_id"
            ),
            "allocator": (
                "deterministic indivisible-group approximate stratification minimizing "
                "normalized class plus total L1 target deviation"
            ),
            "test_policy": "sealed; never use for feature, model, threshold, or hyperparameter selection",
            "old_split_policy": "provenance only; ignored for assignment",
        },
        "source_image_count": len(rows),
        "included_image_count": len(included),
        "excluded_image_count": len(excluded),
        "excluded_reason_counts": dict(sorted(excluded_counts.items())),
        "production_split_counts": {split: split_counts[split] for split in PRODUCTION_SPLITS},
        "class_counts": dict(sorted(class_counts.items())),
        "class_split_counts": class_split_counts,
        "production_allocation": production_diagnostics,
        "development_cv_fold_counts": {str(index): cv_counts[str(index)] for index in range(CV_FOLDS)},
        "development_cv_class_counts": {
            str(index): {
                label: sum(
                    row["cv_fold"] == str(index) and row["ornament_label"] == label
                    for row in development
                )
                for label in sorted(class_counts)
            }
            for index in range(CV_FOLDS)
        },
        "development_cv_allocation": cv_diagnostics,
        "split_group_count": len(groups),
        "presemantic_split_group_count": len(presemantic_groups),
        "semantic_split_group_count": len(semantic_group_bases),
        "semantic_merged_group_count": sum(
            len(base_ids) > 1 for base_ids in semantic_group_bases.values()
        ),
        "source_atomic_split_group_count": len(source_group_semantics),
        "source_atomic_merged_group_count": sum(
            len(semantic_ids) > 1 for semantic_ids in source_group_semantics.values()
        ),
        "largest_split_group_size": max(group.size for group in groups),
        "manual_reviewed_image_count": sum(row["manual_review_status"] == "complete" for row in rows),
        "manual_reviewed_group_count": len(review["group_decisions"]),  # type: ignore[arg-type,index]
        "singleton_image_count": sum(row["manual_review_status"] == "not_required_singleton" for row in rows),
        "split_group_leakage_violations": split_group_violations,
        "source_atomic_group_leakage_violations": source_atomic_group_violations,
        "global_source_cohort_leakage_violations": source_cohort_violations,
        "semantic_group_leakage_violations": semantic_group_violations,
        "object_group_leakage_violations": object_group_violations,
        "content_hash_leakage_violations": content_hash_violations,
        "cv_split_group_leakage_violations": cv_group_violations,
        "cv_source_atomic_group_leakage_violations": cv_source_atomic_violations,
        "cv_global_source_cohort_leakage_violations": cv_source_cohort_violations,
        "cv_semantic_group_leakage_violations": cv_semantic_violations,
        "cv_object_group_leakage_violations": cv_object_violations,
        "cv_content_hash_leakage_violations": cv_content_violations,
        "adjudicated_candidate_groups_crossing_splits": adjudicated_candidate_crossings,
        "semantic_adjudication": {
            "review_status": "complete",
            "similarity_audit_fingerprint_sha256": semantic.audit_fingerprint_sha256,
            "adjudication_file_sha256": semantic.adjudication_sha256,
            "reviewed_component_count": len(semantic.component_members),
            "decision_counts": dict(sorted(merge_decision_counts.items())),
            "merge_group_count": len(semantic.merge_groups),
            "candidate_image_count": len(
                {
                    image_id
                    for image_ids in semantic.component_members.values()
                    for image_id in image_ids
                }
            ),
            "semantic_group_membership_fingerprint_sha256": (
                semantic_membership_fingerprint(rows)
            ),
        },
        "global_source_adjudication": {
            "review_version": source.review_version,
            "review_status": "complete",
            "inventory_projection_fingerprint_sha256": (
                source.inventory_projection_fingerprint_sha256
            ),
            "adjudication_file_sha256": source.adjudication_sha256,
            "cohort_count": len(source.cohorts),
            "direct_member_image_count": sum(
                len(cohort.image_ids) for cohort in source.cohorts
            ),
            "propagated_member_image_count": sum(
                bool(row["source_atomic_cohort_ids"]) for row in rows
            ),
            "relationship_counts": dict(
                sorted(Counter(cohort.relationship for cohort in source.cohorts).items())
            ),
            "source_atomic_group_membership_fingerprint_sha256": (
                source_atomic_membership_fingerprint(rows)
            ),
        },
        "assignment_fingerprint_sha256": assignment_fingerprint(rows),
    }


def validate(rows: Sequence[dict[str, object]], audit_payload: dict[str, object]) -> None:
    if len(rows) != 2055:
        raise ValueError(f"Expected 2,055 source rows, found {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("Output image IDs are not unique")
    if {row["split_version"] for row in rows} != {SPLIT_VERSION}:
        raise ValueError("Output rows do not all use the source-atomic v3 split version")
    if any(not row["presemantic_split_group_id"] for row in rows):
        raise ValueError("A row lacks its pre-semantic group provenance")
    if any(not row["semantic_split_group_id"] for row in rows):
        raise ValueError("A row lacks its semantic split group")
    if any(not row["pre_source_cohort_split_group_id"] for row in rows):
        raise ValueError("A row lacks its pre-source-cohort group provenance")
    if any(not row["source_atomic_split_group_id"] for row in rows):
        raise ValueError("A row lacks its source-atomic split group")
    if any(
        row["pre_source_cohort_split_group_id"] != row["semantic_split_group_id"]
        for row in rows
    ):
        raise ValueError("Pre-source-cohort group differs from the semantic split group")
    if any(
        row["split_group_id"] != row["source_atomic_split_group_id"] for row in rows
    ):
        raise ValueError("Canonical split group differs from the source-atomic split group")
    similarity_audit_fingerprints = {
        str(row["semantic_similarity_audit_fingerprint_sha256"]) for row in rows
    }
    adjudication_file_hashes = {
        str(row["semantic_adjudication_file_sha256"]) for row in rows
    }
    semantic_audit = audit_payload["semantic_adjudication"]
    if not isinstance(semantic_audit, dict):
        raise ValueError("Semantic adjudication audit is malformed")
    if similarity_audit_fingerprints != {
        str(semantic_audit["similarity_audit_fingerprint_sha256"])
    }:
        raise ValueError("Row-level semantic similarity-audit fingerprint mismatch")
    if adjudication_file_hashes != {str(semantic_audit["adjudication_file_sha256"])}:
        raise ValueError("Row-level semantic adjudication file hash mismatch")
    if semantic_membership_fingerprint(rows) != semantic_audit[
        "semantic_group_membership_fingerprint_sha256"
    ]:
        raise ValueError("Semantic group membership fingerprint mismatch")

    source_audit = audit_payload["global_source_adjudication"]
    if not isinstance(source_audit, dict):
        raise ValueError("Global source adjudication audit is malformed")
    source_projection_fingerprints = {
        str(row["global_source_inventory_projection_fingerprint_sha256"])
        for row in rows
    }
    source_file_hashes = {
        str(row["global_source_cohort_file_sha256"]) for row in rows
    }
    if source_projection_fingerprints != {
        str(source_audit["inventory_projection_fingerprint_sha256"])
    }:
        raise ValueError("Row-level global-source projection fingerprint mismatch")
    if source_file_hashes != {str(source_audit["adjudication_file_sha256"])}:
        raise ValueError("Row-level global-source adjudication file hash mismatch")
    if source_inventory_projection_fingerprint(rows) != source_audit[
        "inventory_projection_fingerprint_sha256"
    ]:
        raise ValueError("Global-source inventory/projection fingerprint mismatch")
    if source_atomic_membership_fingerprint(rows) != source_audit[
        "source_atomic_group_membership_fingerprint_sha256"
    ]:
        raise ValueError("Source-atomic group membership fingerprint mismatch")
    propagated_cohorts_by_group: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        propagated_cohorts_by_group[str(row["source_atomic_split_group_id"])].add(
            str(row["source_atomic_cohort_ids"])
        )
    if any(len(values) != 1 for values in propagated_cohorts_by_group.values()):
        raise ValueError("Source-atomic propagated cohort provenance is inconsistent")
    if any(
        row["global_source_cohort_id"]
        and str(row["global_source_cohort_id"])
        not in str(row["source_atomic_cohort_ids"]).split("|")
        for row in rows
    ):
        raise ValueError("A direct source cohort is absent from propagated provenance")

    included = [row for row in rows if row["inclusion_status"] == "included"]
    excluded = [row for row in rows if row["inclusion_status"] == "excluded"]
    if any(not row["production_split"] for row in included):
        raise ValueError("An included row lacks a production split")
    if any(row["production_split"] for row in excluded):
        raise ValueError("An excluded row has a production split")
    if any(row["production_split"] == "test" and row["cv_fold"] for row in included):
        raise ValueError("A sealed test row has a development CV fold")
    if any(row["production_split"] != "test" and not row["cv_fold"] for row in included):
        raise ValueError("A development row lacks a CV fold")
    if set(row["production_split"] for row in included) != set(PRODUCTION_SPLITS):
        raise ValueError("Not all production splits are populated")
    if any(audit_payload[name] for name in (
        "split_group_leakage_violations",
        "source_atomic_group_leakage_violations",
        "global_source_cohort_leakage_violations",
        "semantic_group_leakage_violations",
        "object_group_leakage_violations",
        "content_hash_leakage_violations",
        "cv_split_group_leakage_violations",
        "cv_source_atomic_group_leakage_violations",
        "cv_global_source_cohort_leakage_violations",
        "cv_semantic_group_leakage_violations",
        "cv_object_group_leakage_violations",
        "cv_content_hash_leakage_violations",
    )):
        raise ValueError("Leakage audit failed")

    labels = {str(row["ornament_label"]) for row in included}
    for split in PRODUCTION_SPLITS:
        if {str(row["ornament_label"]) for row in included if row["production_split"] == split} != labels:
            raise ValueError(f"Split {split} does not contain every class")
    for fold in range(CV_FOLDS):
        if {
            str(row["ornament_label"])
            for row in included
            if row["production_split"] != "test" and row["cv_fold"] == str(fold)
        } != labels:
            raise ValueError(f"CV fold {fold} does not contain every class")
    expected_production_diagnostics = allocation_diagnostics(
        included,
        destination_field="production_split",
        destinations=PRODUCTION_SPLITS,
        ratios=PRODUCTION_RATIOS,
    )
    if audit_payload["production_allocation"] != expected_production_diagnostics:
        raise ValueError("Production allocation diagnostics do not match rows")
    cv_destinations = tuple(str(index) for index in range(CV_FOLDS))
    cv_ratios = {destination: 1 / CV_FOLDS for destination in cv_destinations}
    expected_cv_diagnostics = allocation_diagnostics(
        [row for row in included if row["production_split"] != "test"],
        destination_field="cv_fold",
        destinations=cv_destinations,
        ratios=cv_ratios,
    )
    if audit_payload["development_cv_allocation"] != expected_cv_diagnostics:
        raise ValueError("Development CV allocation diagnostics do not match rows")
    if not expected_production_diagnostics["every_class_in_every_destination"]:
        raise ValueError("Production allocation does not retain every class")
    if not expected_cv_diagnostics["every_class_in_every_destination"]:
        raise ValueError("Development CV allocation does not retain every class")


def validate_generated_csv_hashes(
    output_dir: Path, audit_payload: dict[str, object]
) -> None:
    expected = audit_payload.get("generated_csv_sha256")
    if not isinstance(expected, dict) or not expected:
        raise ValueError("Audit lacks generated CSV SHA-256 hashes")
    actual = {name: sha256_file(output_dir / name) for name in sorted(expected)}
    if actual != expected:
        raise ValueError("Generated CSV SHA-256 validation failed")


def main() -> None:
    args = parse_args()
    manifest = read_csv(args.manifest)
    with args.review_json.open(encoding="utf-8") as handle:
        review = json.load(handle)

    rows = enrich_manifest(manifest, review, args.seed)
    semantic = load_semantic_adjudication(
        rows,
        audit_path=args.similarity_audit,
        component_path=args.similarity_components,
        member_path=args.similarity_members,
        pair_path=args.similarity_pairs,
        adjudication_path=args.adjudication_json,
    )
    apply_semantic_adjudication(rows, semantic)
    source = load_global_source_adjudication(
        rows, args.global_source_adjudication
    )
    apply_global_source_adjudication(rows, source)
    assign_production_splits(rows, args.seed)
    assign_cv_folds(rows, args.seed)
    rows.sort(key=lambda row: (str(row["ornament_label"]), str(row["relative_path"])))

    audit_payload = audit(rows, review, semantic, source, args.seed)
    validate(rows, audit_payload)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "split_manifest.csv", rows, OUTPUT_FIELDS)
    for split in PRODUCTION_SPLITS:
        split_rows = [
            row
            for row in rows
            if row["inclusion_status"] == "included" and row["production_split"] == split
        ]
        write_csv(args.output_dir / f"{split}.csv", split_rows, OUTPUT_FIELDS)
    development_rows = [
        row
        for row in rows
        if row["inclusion_status"] == "included" and row["production_split"] != "test"
    ]
    write_csv(args.output_dir / "development.csv", development_rows, OUTPUT_FIELDS)
    write_csv(
        args.output_dir / "split_summary.csv",
        build_summary_rows(rows),
        ("production_split", "ornament_label", "image_count", "class_total", "class_fraction"),
    )
    csv_names = (
        "split_manifest.csv",
        "train.csv",
        "validation.csv",
        "test.csv",
        "development.csv",
        "split_summary.csv",
    )
    audit_payload["generated_csv_sha256"] = {
        name: sha256_file(args.output_dir / name) for name in csv_names
    }
    validate_generated_csv_hashes(args.output_dir, audit_payload)
    with (args.output_dir / "split_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit_payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(audit_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
