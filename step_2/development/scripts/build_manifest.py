#!/usr/bin/env python3
"""Build the reproducible Step 02 Phase 1 image manifest and data audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
UNREVIEWED = "unreviewed"
UNKNOWN = "unknown"


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
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def parse_args() -> argparse.Namespace:
    development_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=development_root / "data")
    parser.add_argument("--output-root", type=Path, default=development_root / "metadata")
    return parser.parse_args()


def sha256_bytes(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_id(prefix: str, *parts: str) -> str:
    value = "\0".join(parts).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(value).hexdigest()[:16]}"


def bits_to_hex(bits: Iterable[bool]) -> str:
    value = 0
    for index, flag in enumerate(bits):
        value |= int(flag) << index
    return f"{value:016x}"


def perceptual_hashes(image: Image.Image) -> tuple[str, str]:
    grayscale = image.convert("L")
    difference_pixels = list(
        grayscale.resize((9, 8), Image.Resampling.LANCZOS).get_flattened_data()
    )
    difference_bits = (
        difference_pixels[row * 9 + column + 1] > difference_pixels[row * 9 + column]
        for row in range(8)
        for column in range(8)
    )

    average_pixels = list(
        grayscale.resize((8, 8), Image.Resampling.LANCZOS).get_flattened_data()
    )
    average = sum(average_pixels) / len(average_pixels)
    average_bits = (pixel > average for pixel in average_pixels)
    return bits_to_hex(difference_bits), bits_to_hex(average_bits)


def filename_family(class_name: str, filename: str) -> str:
    stem = Path(filename).stem.lower()
    normalized = re.sub(r"_(\d+)$", "", stem)
    return f"{class_name}/{normalized}"


def inspect_image(path: Path) -> dict[str, Any]:
    with Image.open(path) as opened:
        image_format = opened.format or "unknown"
        source_mode = opened.mode
        width, height = opened.size
        try:
            exif_orientation = opened.getexif().get(274)
        except Exception:
            exif_orientation = None
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        dhash, ahash = perceptual_hashes(image)

    aspect_ratio = width / height
    flags: list[str] = []
    if min(width, height) < 96:
        flags.append("low_resolution")
    if max(aspect_ratio, 1.0 / aspect_ratio) > 2.0:
        flags.append("extreme_aspect_ratio")
    if path.stat().st_size > 10 * 1024 * 1024:
        flags.append("large_file")

    return {
        "readable": True,
        "image_format": image_format,
        "source_mode": source_mode,
        "width": width,
        "height": height,
        "aspect_ratio": round(aspect_ratio, 6),
        "exif_orientation": exif_orientation if exif_orientation is not None else "",
        "dhash": dhash,
        "ahash": ahash,
        "quality_flags": ";".join(flags) if flags else "ok",
        "read_error": "",
    }


def hamming_distance(left: str, right: str) -> int:
    return (int(left, 16) ^ int(right, 16)).bit_count()


def cluster_ids(groups: Iterable[list[str]], prefix: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for members in groups:
        if len(members) < 2:
            continue
        cluster_id = stable_id(prefix, *sorted(members))
        for member in members:
            result[member] = cluster_id
    return result


def union_groups(union_find: UnionFind) -> list[list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for value in union_find.parent:
        grouped[union_find.find(value)].append(value)
    return [sorted(members) for members in grouped.values()]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[3]
    data_root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        path
        for path in data_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise FileNotFoundError(f"No supported images found under {data_root}")

    rows: list[dict[str, Any]] = []
    identity_keys: set[str] = set()
    for path in image_paths:
        relative_path = path.relative_to(data_root).as_posix()
        parts = Path(relative_path).parts
        if len(parts) < 3:
            raise ValueError(f"Expected <split>/<class>/<file>, got {relative_path}")
        current_split, class_name, filename = parts[0], parts[1], parts[-1]
        identity_key = f"{class_name}/{filename.lower()}"
        if identity_key in identity_keys:
            identity_key = relative_path.lower()
        identity_keys.add(identity_key)

        content_sha256 = sha256_bytes(path)
        base = {
            "image_id": stable_id("img", identity_key),
            "relative_path": relative_path,
            "current_split": current_split,
            "ornament_label": class_name,
            "file_name": filename,
            "extension": path.suffix.lower(),
            "file_size_bytes": path.stat().st_size,
            "content_sha256": content_sha256,
            "filename_family": filename_family(class_name, filename),
        }
        try:
            base.update(inspect_image(path))
        except Exception as error:
            base.update(
                {
                    "readable": False,
                    "image_format": "",
                    "source_mode": "",
                    "width": "",
                    "height": "",
                    "aspect_ratio": "",
                    "exif_orientation": "",
                    "dhash": "",
                    "ahash": "",
                    "quality_flags": "unreadable",
                    "read_error": f"{type(error).__name__}: {error}",
                }
            )
        rows.append(base)

    ids = [str(row["image_id"]) for row in rows]
    by_id = {str(row["image_id"]): row for row in rows}

    exact_by_hash: dict[str, list[str]] = defaultdict(list)
    family_groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        image_id = str(row["image_id"])
        exact_by_hash[str(row["content_sha256"])].append(image_id)
        family_groups[str(row["filename_family"])].append(image_id)

    exact_groups = [sorted(group) for group in exact_by_hash.values() if len(group) > 1]
    exact_cluster_by_id = cluster_ids(exact_groups, "exact")

    perceptual_union = UnionFind(ids)
    candidate_union = UnionFind(ids)
    pair_details: dict[tuple[str, str], dict[str, Any]] = {}

    for group in exact_groups:
        for left, right in itertools.combinations(group, 2):
            perceptual_union.union(left, right)
            candidate_union.union(left, right)
            pair_details[(left, right)] = {"reasons": {"exact_duplicate"}}

    readable_rows = [row for row in rows if row["readable"]]
    for left_row, right_row in itertools.combinations(readable_rows, 2):
        left = str(left_row["image_id"])
        right = str(right_row["image_id"])
        dhash_distance = hamming_distance(str(left_row["dhash"]), str(right_row["dhash"]))
        ahash_distance = hamming_distance(str(left_row["ahash"]), str(right_row["ahash"]))
        left_aspect = float(left_row["aspect_ratio"])
        right_aspect = float(right_row["aspect_ratio"])
        aspect_delta = abs(left_aspect - right_aspect) / max(left_aspect, right_aspect)
        if dhash_distance <= 4 and ahash_distance <= 4 and aspect_delta <= 0.10:
            perceptual_union.union(left, right)
            candidate_union.union(left, right)
            detail = pair_details.setdefault(tuple(sorted((left, right))), {"reasons": set()})
            detail["reasons"].add("perceptual_review")
            detail.update(
                {
                    "dhash_distance": dhash_distance,
                    "ahash_distance": ahash_distance,
                    "aspect_ratio_delta": round(aspect_delta, 6),
                }
            )

    for family, members in family_groups.items():
        if len(members) < 2:
            continue
        anchor = min(members)
        for member in members:
            candidate_union.union(anchor, member)
        for left, right in itertools.combinations(sorted(members), 2):
            detail = pair_details.setdefault((left, right), {"reasons": set()})
            detail["reasons"].add("filename_family")
            detail["filename_family"] = family

    perceptual_groups = [group for group in union_groups(perceptual_union) if len(group) > 1]
    all_candidate_groups = union_groups(candidate_union)
    candidate_groups = [group for group in all_candidate_groups if len(group) > 1]
    perceptual_cluster_by_id = cluster_ids(perceptual_groups, "visual")
    candidate_cluster_by_id = {
        member: stable_id("candidate", *group)
        for group in all_candidate_groups
        for member in group
    }

    for row in rows:
        image_id = str(row["image_id"])
        row.update(
            {
                "exact_duplicate_cluster_id": exact_cluster_by_id.get(image_id, ""),
                "perceptual_review_cluster_id": perceptual_cluster_by_id.get(image_id, ""),
                "candidate_object_group_id": candidate_cluster_by_id[image_id],
                "source_group_id": UNREVIEWED,
                "object_group_id": UNREVIEWED,
                "object_type": UNREVIEWED,
                "motif_visibility": UNREVIEWED,
                "provenance": UNKNOWN,
                "license": UNKNOWN,
                "label_status": "provided_folder_label",
                "metadata_review_status": "needs_manual_review",
                "notes": "",
            }
        )

    duplicate_rows: list[dict[str, Any]] = []
    for (left, right), detail in sorted(pair_details.items()):
        left_row = by_id[left]
        right_row = by_id[right]
        duplicate_rows.append(
            {
                "left_image_id": left,
                "right_image_id": right,
                "left_path": left_row["relative_path"],
                "right_path": right_row["relative_path"],
                "left_split": left_row["current_split"],
                "right_split": right_row["current_split"],
                "left_label": left_row["ornament_label"],
                "right_label": right_row["ornament_label"],
                "same_split": left_row["current_split"] == right_row["current_split"],
                "same_label": left_row["ornament_label"] == right_row["ornament_label"],
                "candidate_reasons": ";".join(sorted(detail["reasons"])),
                "dhash_distance": detail.get("dhash_distance", ""),
                "ahash_distance": detail.get("ahash_distance", ""),
                "aspect_ratio_delta": detail.get("aspect_ratio_delta", ""),
                "filename_family": detail.get("filename_family", ""),
            }
        )

    duplicate_ids = {row["left_image_id"] for row in duplicate_rows} | {
        row["right_image_id"] for row in duplicate_rows
    }
    cross_split_ids = {
        image_id
        for pair in duplicate_rows
        if not pair["same_split"]
        for image_id in (pair["left_image_id"], pair["right_image_id"])
    }
    review_rows: list[dict[str, Any]] = []
    for row in rows:
        image_id = str(row["image_id"])
        reasons: list[str] = []
        priority = 3
        if not row["readable"]:
            priority = 1
            reasons.append("unreadable_image")
        if image_id in cross_split_ids:
            priority = min(priority, 1)
            reasons.append("cross_split_duplicate_candidate")
        elif image_id in duplicate_ids:
            priority = min(priority, 2)
            reasons.append("duplicate_or_group_candidate")
        reasons.extend(
            [
                "confirm_object_group",
                "assign_object_type",
                "review_motif_visibility",
                "record_provenance",
                "record_license",
            ]
        )
        review_rows.append(
            {
                "priority": priority,
                "image_id": image_id,
                "relative_path": row["relative_path"],
                "ornament_label": row["ornament_label"],
                "candidate_object_group_id": row["candidate_object_group_id"],
                "review_reasons": ";".join(reasons),
                "review_status": "pending",
                "reviewer_notes": "",
            }
        )
    review_rows.sort(key=lambda row: (int(row["priority"]), str(row["relative_path"])))

    class_counter = Counter((str(row["current_split"]), str(row["ornament_label"])) for row in rows)
    class_summary = [
        {"current_split": split, "ornament_label": label, "image_count": count}
        for (split, label), count in sorted(class_counter.items())
    ]

    fingerprint = hashlib.sha256()
    for row in sorted(rows, key=lambda item: str(item["relative_path"])):
        fingerprint.update(str(row["relative_path"]).encode("utf-8"))
        fingerprint.update(b"\0")
        fingerprint.update(str(row["content_sha256"]).encode("ascii"))
        fingerprint.update(b"\n")

    exact_cross_split_pairs = sum(
        1
        for pair in duplicate_rows
        if "exact_duplicate" in pair["candidate_reasons"] and not pair["same_split"]
    )
    perceptual_cross_split_pairs = sum(
        1
        for pair in duplicate_rows
        if "perceptual_review" in pair["candidate_reasons"] and not pair["same_split"]
    )
    candidate_cross_split_group_count = sum(
        len({by_id[image_id]["current_split"] for image_id in group}) > 1
        for group in candidate_groups
    )
    candidate_cross_label_group_count = sum(
        len({by_id[image_id]["ornament_label"] for image_id in group}) > 1
        for group in candidate_groups
    )
    try:
        data_root_display = data_root.relative_to(repo_root).as_posix()
    except ValueError:
        data_root_display = data_root.as_posix()

    audit = {
        "phase": "step_02_phase_1_data_truth",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_root": data_root_display,
        "dataset_fingerprint_sha256": fingerprint.hexdigest(),
        "image_count": len(rows),
        "readable_image_count": sum(bool(row["readable"]) for row in rows),
        "unreadable_image_count": sum(not bool(row["readable"]) for row in rows),
        "class_count": len({row["ornament_label"] for row in rows}),
        "split_count": len({row["current_split"] for row in rows}),
        "format_counts": dict(sorted(Counter(str(row["image_format"]) for row in rows).items())),
        "quality_flag_counts": dict(sorted(Counter(str(row["quality_flags"]) for row in rows).items())),
        "exact_duplicate_group_count": len(exact_groups),
        "exact_duplicate_extra_copy_count": sum(len(group) - 1 for group in exact_groups),
        "exact_cross_split_pair_count": exact_cross_split_pairs,
        "perceptual_review_group_count": len(perceptual_groups),
        "perceptual_review_pair_count": sum(
            "perceptual_review" in pair["candidate_reasons"] for pair in duplicate_rows
        ),
        "perceptual_cross_split_pair_count": perceptual_cross_split_pairs,
        "candidate_object_group_count": len(all_candidate_groups),
        "candidate_multi_image_group_count": len(candidate_groups),
        "images_in_multi_image_candidate_groups": sum(len(group) for group in candidate_groups),
        "candidate_cross_split_group_count": candidate_cross_split_group_count,
        "candidate_cross_label_group_count": candidate_cross_label_group_count,
        "manual_review_image_count": len(review_rows),
        "unknown_provenance_count": sum(row["provenance"] == UNKNOWN for row in rows),
        "unknown_license_count": sum(row["license"] == UNKNOWN for row in rows),
        "class_summary": class_summary,
    }

    manifest_fields = [
        "image_id",
        "relative_path",
        "current_split",
        "ornament_label",
        "file_name",
        "extension",
        "file_size_bytes",
        "content_sha256",
        "readable",
        "image_format",
        "source_mode",
        "width",
        "height",
        "aspect_ratio",
        "exif_orientation",
        "dhash",
        "ahash",
        "filename_family",
        "exact_duplicate_cluster_id",
        "perceptual_review_cluster_id",
        "candidate_object_group_id",
        "quality_flags",
        "read_error",
        "source_group_id",
        "object_group_id",
        "object_type",
        "motif_visibility",
        "provenance",
        "license",
        "label_status",
        "metadata_review_status",
        "notes",
    ]
    duplicate_fields = [
        "left_image_id",
        "right_image_id",
        "left_path",
        "right_path",
        "left_split",
        "right_split",
        "left_label",
        "right_label",
        "same_split",
        "same_label",
        "candidate_reasons",
        "dhash_distance",
        "ahash_distance",
        "aspect_ratio_delta",
        "filename_family",
    ]
    review_fields = [
        "priority",
        "image_id",
        "relative_path",
        "ornament_label",
        "candidate_object_group_id",
        "review_reasons",
        "review_status",
        "reviewer_notes",
    ]

    write_csv(output_root / "manifest.csv", rows, manifest_fields)
    write_csv(output_root / "class_summary.csv", class_summary, list(class_summary[0]))
    write_csv(output_root / "duplicate_candidates.csv", duplicate_rows, duplicate_fields)
    write_csv(output_root / "review_queue.csv", review_rows, review_fields)
    with (output_root / "data_audit.json").open("w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(json.dumps(audit, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
