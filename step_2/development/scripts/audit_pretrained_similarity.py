#!/usr/bin/env python3
"""Build a review-only DINOv3 similarity audit for every included image.

The audit deliberately does not merge objects, sources, or split groups. It
emits high-similarity pairs and their connected components so that those
relationships can be adjudicated before the evaluation split is resealed.

Run this script with the Phase 4 vision runtime (NumPy, Torch, Transformers,
and the locally cached DINOv3 weights are required).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_pretrained_embeddings as pretrained  # noqa: E402


np = pretrained.np
torch = pretrained.torch

STEP_ROOT = SCRIPT_DIR.parent
REPO_ROOT = STEP_ROOT.parents[1]
DEFAULT_MANIFEST = STEP_ROOT / "splits" / "split_manifest.csv"
DEFAULT_OUTPUT_DIR = STEP_ROOT / "review" / "pretrained_similarity"
DEFAULT_CACHE_DIR = REPO_ROOT / ".cache" / "step_02" / "pretrained_embeddings"
DEFAULT_MODEL_SOURCE = pretrained.DEFAULT_DINOV3_SOURCE

AUDIT_VERSION = "step02_dinov3_similarity_v1"
EXPECTED_INCLUDED_IMAGES = 2023
CONCAT_THRESHOLD = 0.93
PATCH_THRESHOLD = 0.98
CRITICAL_CONCAT_THRESHOLD = 0.95
CRITICAL_PATCH_THRESHOLD = 0.985

PAIR_FIELDS = (
    "rank",
    "component_id",
    "review_priority",
    "priority_score",
    "image_id_a",
    "relative_path_a",
    "ornament_label_a",
    "production_split_a",
    "cv_fold_a",
    "old_split_a",
    "candidate_object_group_id_a",
    "confirmed_object_group_id_a",
    "confirmed_source_group_id_a",
    "image_id_b",
    "relative_path_b",
    "ornament_label_b",
    "production_split_b",
    "cv_fold_b",
    "old_split_b",
    "candidate_object_group_id_b",
    "confirmed_object_group_id_b",
    "confirmed_source_group_id_b",
    "cosine_cls_patch_concat",
    "cosine_patch_mean",
    "trigger_concat",
    "trigger_patch_mean",
    "same_label",
    "same_candidate_group",
    "same_content_sha256",
    "cross_production_split",
    "cross_cv_fold",
    "touches_current_test",
    "current_evaluation_boundary",
)

COMPONENT_FIELDS = (
    "component_id",
    "review_priority",
    "member_count",
    "pair_count",
    "max_cosine_cls_patch_concat",
    "max_cosine_patch_mean",
    "ornament_labels",
    "production_splits",
    "cv_folds",
    "cross_production_split",
    "cross_cv_fold",
    "touches_current_test",
    "image_ids",
    "relative_paths",
)

MEMBER_FIELDS = (
    "component_id",
    "image_id",
    "relative_path",
    "ornament_label",
    "production_split",
    "cv_fold",
    "old_split",
    "candidate_object_group_id",
    "confirmed_object_group_id",
    "confirmed_source_group_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--embedding-cache", type=Path, action="append", default=[])
    parser.add_argument("--no-discover-cache", action="store_true")
    parser.add_argument("--dinov3-source", default=DEFAULT_MODEL_SOURCE)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--concat-threshold", type=float, default=CONCAT_THRESHOLD)
    parser.add_argument("--patch-threshold", type=float, default=PATCH_THRESHOLD)
    parser.add_argument(
        "--critical-concat-threshold", type=float, default=CRITICAL_CONCAT_THRESHOLD
    )
    parser.add_argument(
        "--critical-patch-threshold", type=float, default=CRITICAL_PATCH_THRESHOLD
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path, rows: Sequence[dict[str, object]], fields: Sequence[str]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
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
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def validate_manifest(all_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    required = {
        "image_id",
        "relative_path",
        "ornament_label",
        "content_sha256",
        "candidate_object_group_id",
        "confirmed_object_group_id",
        "confirmed_source_group_id",
        "inclusion_status",
        "production_split",
        "cv_fold",
    }
    if not all_rows:
        raise ValueError("Split manifest is empty")
    missing_fields = required - set(all_rows[0])
    if missing_fields:
        raise ValueError(f"Manifest is missing fields: {sorted(missing_fields)}")
    rows = [row for row in all_rows if row["inclusion_status"] == "included"]
    if len(rows) != EXPECTED_INCLUDED_IMAGES:
        raise ValueError(
            f"Expected {EXPECTED_INCLUDED_IMAGES} included images, found {len(rows)}"
        )
    image_ids = [row["image_id"] for row in rows]
    if len(set(image_ids)) != len(image_ids):
        raise ValueError("Included image IDs are not unique")
    if {row["production_split"] for row in rows} != {"train", "validation", "test"}:
        raise ValueError("Included images do not contain train, validation, and test")
    missing_files = [
        row["relative_path"]
        for row in rows
        if not (pretrained.DATA_ROOT / row["relative_path"]).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Missing included images: {missing_files[:5]}")
    return rows


def compatible_cache(
    path: Path,
    rows_by_id: dict[str, dict[str, str]],
    weight_sha256: str,
) -> tuple[list[str], dict[str, np.ndarray]] | None:
    """Load a cache only when its complete fingerprint matches current rows."""

    try:
        with np.load(path, allow_pickle=False) as cached:
            required = {"fingerprint", "image_ids", "embedding__patch_mean", "embedding__cls_patch_concat"}
            if not required.issubset(cached.files):
                return None
            image_ids = [str(value) for value in cached["image_ids"]]
            if not image_ids or len(set(image_ids)) != len(image_ids):
                return None
            if any(image_id not in rows_by_id for image_id in image_ids):
                return None
            cache_rows = [rows_by_id[image_id] for image_id in image_ids]
            expected = pretrained.embedding_fingerprint(
                cache_rows,
                "dinov3_vits16",
                pretrained.DEFAULT_DINOV3_SOURCE,
                weight_sha256,
                "center_crop",
                pretrained.IMAGENET_MEAN,
                pretrained.IMAGENET_STD,
            )
            if str(cached["fingerprint"].item()) != expected:
                return None
            arrays = {
                "patch_mean": cached["embedding__patch_mean"].astype(np.float32),
                "cls_patch_concat": cached["embedding__cls_patch_concat"].astype(np.float32),
            }
    except (OSError, ValueError, KeyError):
        return None
    if any(values.shape[0] != len(image_ids) for values in arrays.values()):
        return None
    return image_ids, arrays


def collect_embeddings(
    rows: list[dict[str, str]],
    model_source: str,
    cache_paths: Iterable[Path],
    batch_size: int,
) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    rows_by_id = {row["image_id"]: row for row in rows}
    weight_path = pretrained.resolve_dinov3_weight(model_source)
    weight_sha256 = pretrained.sha256_file(weight_path)
    collected: dict[str, dict[str, np.ndarray]] = {
        "patch_mean": {},
        "cls_patch_concat": {},
    }
    used_cache_files: list[str] = []

    for path in sorted({path.resolve() for path in cache_paths}):
        if not path.is_file():
            continue
        loaded = compatible_cache(path, rows_by_id, weight_sha256)
        if loaded is None:
            continue
        image_ids, arrays = loaded
        contributed = False
        for offset, image_id in enumerate(image_ids):
            for name in collected:
                value = arrays[name][offset]
                previous = collected[name].get(image_id)
                if previous is not None and not np.allclose(previous, value, rtol=0.0, atol=1e-6):
                    raise ValueError(f"Conflicting cached embeddings for {image_id}: {path}")
                if previous is None:
                    collected[name][image_id] = value
                    contributed = True
        if contributed:
            used_cache_files.append(display_path(path))

    missing_rows = [
        row for row in rows if row["image_id"] not in collected["cls_patch_concat"]
    ]
    cached_count = len(rows) - len(missing_rows)
    extraction_metadata: dict[str, object] = {}
    if missing_rows:
        print(f"Extracting {len(missing_rows)} uncached all-data audit embeddings", flush=True)
        extracted, extraction_metadata, extracted_weight_sha = pretrained.extract_dinov3(
            missing_rows, model_source, "center_crop", batch_size
        )
        if extracted_weight_sha != weight_sha256:
            raise ValueError("DINOv3 weight changed during embedding extraction")
        for name in collected:
            for row, value in zip(missing_rows, extracted[name]):
                collected[name][row["image_id"]] = value.astype(np.float32)

    embeddings = {
        name: np.stack([values[row["image_id"]] for row in rows]).astype(np.float32)
        for name, values in collected.items()
    }
    for name, values in embeddings.items():
        if not np.isfinite(values).all():
            raise ValueError(f"Non-finite {name} embeddings")
        embeddings[name] = pretrained.normalize_rows(values).astype(np.float32)

    full_fingerprint = pretrained.embedding_fingerprint(
        rows,
        "dinov3_vits16",
        pretrained.DEFAULT_DINOV3_SOURCE,
        weight_sha256,
        "center_crop",
        pretrained.IMAGENET_MEAN,
        pretrained.IMAGENET_STD,
    )
    metadata = {
        "encoder_alias": "dinov3_vits16",
        "model_id": pretrained.DEFAULT_DINOV3_SOURCE,
        "model_source": model_source,
        "view_mode": "center_crop",
        "weight_sha256": weight_sha256,
        "embedding_fingerprint_sha256": full_fingerprint,
        "embedding_dimensions": {
            name: int(values.shape[1]) for name, values in embeddings.items()
        },
        "cached_image_count": cached_count,
        "extracted_image_count": len(missing_rows),
        "source_cache_files": used_cache_files,
    }
    if extraction_metadata:
        metadata["parameter_count"] = int(extraction_metadata["parameter_count"])
    return embeddings, metadata


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1


def boolean(value: bool) -> str:
    return "true" if value else "false"


def component_identifier(image_ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(image_ids)).encode("utf-8")
    return f"simcomp_{hashlib.sha256(payload).hexdigest()[:16]}"


def boundary_name(left: str, right: str) -> str:
    if left == right:
        return ""
    values = {left, right}
    if "test" in values:
        return "test_vs_development"
    return "train_vs_validation"


def build_candidates(
    rows: list[dict[str, str]],
    embeddings: dict[str, np.ndarray],
    concat_threshold: float,
    patch_threshold: float,
    critical_concat_threshold: float,
    critical_patch_threshold: float,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
]:
    image_count = len(rows)
    upper_left, upper_right = np.triu_indices(image_count, 1)
    object_ids = np.asarray([row["confirmed_object_group_id"] for row in rows])
    source_ids = np.asarray([row["confirmed_source_group_id"] for row in rows])
    same_object = (object_ids[upper_left] != "") & (
        object_ids[upper_left] == object_ids[upper_right]
    )
    same_source = (source_ids[upper_left] != "") & (
        source_ids[upper_left] == source_ids[upper_right]
    )
    eligible = ~(same_object | same_source)

    concat_matrix = embeddings["cls_patch_concat"] @ embeddings["cls_patch_concat"].T
    patch_matrix = embeddings["patch_mean"] @ embeddings["patch_mean"].T
    concat_values = concat_matrix[upper_left, upper_right]
    patch_values = patch_matrix[upper_left, upper_right]
    selected = eligible & (
        (concat_values >= concat_threshold) | (patch_values >= patch_threshold)
    )
    selected_left = upper_left[selected]
    selected_right = upper_right[selected]
    selected_concat = concat_values[selected]
    selected_patch = patch_values[selected]

    union_find = UnionFind(image_count)
    for left, right in zip(selected_left, selected_right):
        union_find.union(int(left), int(right))

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for index in set(selected_left.tolist()) | set(selected_right.tolist()):
        members_by_root[union_find.find(int(index))].append(int(index))
    component_by_index: dict[int, str] = {}
    for members in members_by_root.values():
        component_id = component_identifier([rows[index]["image_id"] for index in members])
        for index in members:
            component_by_index[index] = component_id

    raw_pairs: list[dict[str, object]] = []
    numeric_by_component: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for left, right, concat_similarity, patch_similarity in zip(
        selected_left, selected_right, selected_concat, selected_patch
    ):
        left_index, right_index = int(left), int(right)
        row_a, row_b = rows[left_index], rows[right_index]
        if row_b["image_id"] < row_a["image_id"]:
            row_a, row_b = row_b, row_a
        critical = (
            float(concat_similarity) >= critical_concat_threshold
            or float(patch_similarity) >= critical_patch_threshold
        )
        component_id = component_by_index[left_index]
        priority_score = max(
            float(concat_similarity) / concat_threshold,
            float(patch_similarity) / patch_threshold,
        )
        numeric_by_component[component_id].append(
            (float(concat_similarity), float(patch_similarity))
        )
        raw_pairs.append(
            {
                "component_id": component_id,
                "review_priority": "critical" if critical else "review",
                "priority_score": f"{priority_score:.9f}",
                "image_id_a": row_a["image_id"],
                "relative_path_a": row_a["relative_path"],
                "ornament_label_a": row_a["ornament_label"],
                "production_split_a": row_a["production_split"],
                "cv_fold_a": row_a["cv_fold"],
                "old_split_a": row_a["old_split"],
                "candidate_object_group_id_a": row_a["candidate_object_group_id"],
                "confirmed_object_group_id_a": row_a["confirmed_object_group_id"],
                "confirmed_source_group_id_a": row_a["confirmed_source_group_id"],
                "image_id_b": row_b["image_id"],
                "relative_path_b": row_b["relative_path"],
                "ornament_label_b": row_b["ornament_label"],
                "production_split_b": row_b["production_split"],
                "cv_fold_b": row_b["cv_fold"],
                "old_split_b": row_b["old_split"],
                "candidate_object_group_id_b": row_b["candidate_object_group_id"],
                "confirmed_object_group_id_b": row_b["confirmed_object_group_id"],
                "confirmed_source_group_id_b": row_b["confirmed_source_group_id"],
                "cosine_cls_patch_concat": f"{float(concat_similarity):.9f}",
                "cosine_patch_mean": f"{float(patch_similarity):.9f}",
                "trigger_concat": boolean(float(concat_similarity) >= concat_threshold),
                "trigger_patch_mean": boolean(float(patch_similarity) >= patch_threshold),
                "same_label": boolean(row_a["ornament_label"] == row_b["ornament_label"]),
                "same_candidate_group": boolean(
                    bool(row_a["candidate_object_group_id"])
                    and row_a["candidate_object_group_id"]
                    == row_b["candidate_object_group_id"]
                ),
                "same_content_sha256": boolean(
                    row_a["content_sha256"] == row_b["content_sha256"]
                ),
                "cross_production_split": boolean(
                    row_a["production_split"] != row_b["production_split"]
                ),
                "cross_cv_fold": boolean(
                    bool(row_a["cv_fold"])
                    and bool(row_b["cv_fold"])
                    and row_a["cv_fold"] != row_b["cv_fold"]
                ),
                "touches_current_test": boolean(
                    "test" in {row_a["production_split"], row_b["production_split"]}
                ),
                "current_evaluation_boundary": boundary_name(
                    row_a["production_split"], row_b["production_split"]
                ),
            }
        )

    raw_pairs.sort(
        key=lambda row: (
            row["review_priority"] != "critical",
            -float(row["priority_score"]),
            -float(row["cosine_cls_patch_concat"]),
            -float(row["cosine_patch_mean"]),
            row["image_id_a"],
            row["image_id_b"],
        )
    )
    for rank, row in enumerate(raw_pairs, start=1):
        row["rank"] = rank

    pair_count_by_component = Counter(row["component_id"] for row in raw_pairs)
    priority_by_component: dict[str, str] = {}
    for row in raw_pairs:
        component_id = str(row["component_id"])
        if row["review_priority"] == "critical":
            priority_by_component[component_id] = "critical"
        else:
            priority_by_component.setdefault(component_id, "review")

    component_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    components = sorted(
        (
            (component_by_index[members[0]], sorted(members, key=lambda index: rows[index]["image_id"]))
            for members in members_by_root.values()
        ),
        key=lambda item: item[0],
    )
    for component_id, members in components:
        member_data = [rows[index] for index in members]
        similarities = numeric_by_component[component_id]
        production_splits = sorted({row["production_split"] for row in member_data})
        cv_folds = sorted({row["cv_fold"] for row in member_data if row["cv_fold"]})
        component_rows.append(
            {
                "component_id": component_id,
                "review_priority": priority_by_component[component_id],
                "member_count": len(members),
                "pair_count": pair_count_by_component[component_id],
                "max_cosine_cls_patch_concat": f"{max(value[0] for value in similarities):.9f}",
                "max_cosine_patch_mean": f"{max(value[1] for value in similarities):.9f}",
                "ornament_labels": "|".join(
                    sorted({row["ornament_label"] for row in member_data})
                ),
                "production_splits": "|".join(production_splits),
                "cv_folds": "|".join(cv_folds),
                "cross_production_split": boolean(len(production_splits) > 1),
                "cross_cv_fold": boolean(len(cv_folds) > 1),
                "touches_current_test": boolean("test" in production_splits),
                "image_ids": "|".join(row["image_id"] for row in member_data),
                "relative_paths": "|".join(row["relative_path"] for row in member_data),
            }
        )
        for row in member_data:
            member_rows.append(
                {
                    "component_id": component_id,
                    "image_id": row["image_id"],
                    "relative_path": row["relative_path"],
                    "ornament_label": row["ornament_label"],
                    "production_split": row["production_split"],
                    "cv_fold": row["cv_fold"],
                    "old_split": row["old_split"],
                    "candidate_object_group_id": row["candidate_object_group_id"],
                    "confirmed_object_group_id": row["confirmed_object_group_id"],
                    "confirmed_source_group_id": row["confirmed_source_group_id"],
                }
            )
    component_rows.sort(
        key=lambda row: (
            row["review_priority"] != "critical",
            -float(row["max_cosine_cls_patch_concat"]),
            -float(row["max_cosine_patch_mean"]),
            row["component_id"],
        )
    )
    component_order = {
        str(row["component_id"]): index for index, row in enumerate(component_rows)
    }
    member_rows.sort(
        key=lambda row: (component_order[str(row["component_id"])], row["image_id"])
    )

    candidate_indices = set(selected_left.tolist()) | set(selected_right.tolist())
    pair_labels = Counter(
        "|".join(sorted((str(row["ornament_label_a"]), str(row["ornament_label_b"]))))
        for row in raw_pairs
    )
    summary = {
        "total_unordered_pair_count": int(len(upper_left)),
        "excluded_confirmed_same_object_pair_count": int(same_object.sum()),
        "excluded_confirmed_same_source_pair_count": int(same_source.sum()),
        "excluded_confirmed_relation_union_pair_count": int((same_object | same_source).sum()),
        "eligible_pair_count": int(eligible.sum()),
        "candidate_pair_count": len(raw_pairs),
        "candidate_image_count": len(candidate_indices),
        "candidate_component_count": len(component_rows),
        "critical_pair_count": sum(
            row["review_priority"] == "critical" for row in raw_pairs
        ),
        "critical_component_count": sum(
            row["review_priority"] == "critical" for row in component_rows
        ),
        "same_label_pair_count": sum(row["same_label"] == "true" for row in raw_pairs),
        "cross_label_pair_count": sum(row["same_label"] == "false" for row in raw_pairs),
        "same_candidate_group_pair_count": sum(
            row["same_candidate_group"] == "true" for row in raw_pairs
        ),
        "cross_production_split_pair_count": sum(
            row["cross_production_split"] == "true" for row in raw_pairs
        ),
        "cross_cv_fold_pair_count": sum(
            row["cross_cv_fold"] == "true" for row in raw_pairs
        ),
        "touches_current_test_pair_count": sum(
            row["touches_current_test"] == "true" for row in raw_pairs
        ),
        "cross_production_split_component_count": sum(
            row["cross_production_split"] == "true" for row in component_rows
        ),
        "cross_cv_fold_component_count": sum(
            row["cross_cv_fold"] == "true" for row in component_rows
        ),
        "touches_current_test_component_count": sum(
            row["touches_current_test"] == "true" for row in component_rows
        ),
        "component_size_distribution": dict(
            sorted(Counter(str(row["member_count"]) for row in component_rows).items())
        ),
        "pair_label_distribution": dict(sorted(pair_labels.items())),
    }
    return raw_pairs, component_rows, member_rows, summary


def main() -> None:
    args = parse_args()
    np.random.seed(pretrained.SEED)
    torch.manual_seed(pretrained.SEED)
    torch.set_num_threads(args.threads)

    all_rows = read_csv(args.manifest)
    rows = validate_manifest(all_rows)
    cache_paths = list(args.embedding_cache)
    if not args.no_discover_cache and args.cache_dir.is_dir():
        cache_paths.extend(
            args.cache_dir.glob("dinov3_vits16__center_crop__*.npz")
        )
    embeddings, embedding_metadata = collect_embeddings(
        rows, args.dinov3_source, cache_paths, args.batch_size
    )
    pair_rows, component_rows, member_rows, summary = build_candidates(
        rows,
        embeddings,
        args.concat_threshold,
        args.patch_threshold,
        args.critical_concat_threshold,
        args.critical_patch_threshold,
    )

    output_dir = args.output_dir
    pair_path = output_dir / "pair_candidates.csv"
    component_path = output_dir / "component_candidates.csv"
    member_path = output_dir / "component_members.csv"
    audit_path = output_dir / "audit.json"
    write_csv(pair_path, pair_rows, PAIR_FIELDS)
    write_csv(component_path, component_rows, COMPONENT_FIELDS)
    write_csv(member_path, member_rows, MEMBER_FIELDS)

    output_hashes = {
        "pair_candidates.csv": sha256_file(pair_path),
        "component_candidates.csv": sha256_file(component_path),
        "component_members.csv": sha256_file(member_path),
    }
    thresholds = {
        "candidate_union_rule": (
            f"cls_patch_concat >= {args.concat_threshold:g} OR "
            f"patch_mean >= {args.patch_threshold:g}"
        ),
        "cls_patch_concat": args.concat_threshold,
        "patch_mean": args.patch_threshold,
        "critical_cls_patch_concat": args.critical_concat_threshold,
        "critical_patch_mean": args.critical_patch_threshold,
    }
    fingerprint_payload = {
        "audit_version": AUDIT_VERSION,
        "manifest_sha256": sha256_file(args.manifest),
        "embedding_fingerprint_sha256": embedding_metadata[
            "embedding_fingerprint_sha256"
        ],
        "thresholds": thresholds,
        "output_sha256": output_hashes,
    }
    audit_fingerprint = hashlib.sha256(
        stable_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    audit = {
        "audit_version": AUDIT_VERSION,
        "purpose": "review-only semantic duplicate and source candidate discovery",
        "auto_merge_performed": False,
        "current_split_status": "invalidated_pending_similarity_adjudication_and_reseal",
        "input": {
            "manifest": display_path(args.manifest),
            "manifest_sha256": fingerprint_payload["manifest_sha256"],
            "manifest_row_count": len(all_rows),
            "included_image_count": len(rows),
            "excluded_image_count": len(all_rows) - len(rows),
        },
        "embedding": embedding_metadata,
        "thresholds": thresholds,
        "exclusion_rule": "exclude pairs already sharing a confirmed object or non-empty confirmed source group",
        "summary": summary,
        "output_sha256": output_hashes,
        "audit_fingerprint_sha256": audit_fingerprint,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "included_images": len(rows),
                "candidate_pairs": summary["candidate_pair_count"],
                "candidate_components": summary["candidate_component_count"],
                "touches_current_test_pairs": summary[
                    "touches_current_test_pair_count"
                ],
                "audit_fingerprint_sha256": audit_fingerprint,
                "output_dir": display_path(output_dir),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
