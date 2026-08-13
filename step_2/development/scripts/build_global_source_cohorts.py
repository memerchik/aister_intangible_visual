#!/usr/bin/env python3
"""Build the reviewed global acquisition/source cohort adjudication.

The rules in this file are provenance for the completed visual audit.  The
canonical JSON stores explicit image IDs so later split construction does not
depend on filenames or filename ordering.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SPLIT_MANIFEST = STEP_ROOT / "splits" / "split_manifest.csv"
OUTPUT_DIR = STEP_ROOT / "review" / "global_source_cohorts"
ADJUDICATION_PATH = OUTPUT_DIR / "adjudication.json"
MEMBERS_PATH = OUTPUT_DIR / "cohort_members.csv"
README_PATH = OUTPUT_DIR / "README.md"

REVIEW_VERSION = "global_source_cohorts_v1"
RELATIONSHIP = "same_acquisition_source"
EXPECTED_INVENTORY_ROWS = 2_055
EXPECTED_INCLUDED_ROWS = 2_023
EXPECTED_COHORTS = 28
EXPECTED_MEMBERS = 982


@dataclass(frozen=True)
class CohortSpec:
    cohort_id: str
    ornament_label: str
    expected_count: int
    source_rule: str
    selector: Callable[[int], bool]
    evidence: str
    notes: str


def in_ranges(*ranges: tuple[int, int]) -> Callable[[int], bool]:
    return lambda number: any(start <= number <= end for start, end in ranges)


SPECS = (
    CohortSpec(
        "src_global_bub_blue_fair_table",
        "03_bubnivka_ceramics",
        10,
        "first filename integer 308-317",
        in_ranges((308, 317)),
        "One outdoor vendor sequence uses the same vivid blue tablecloth, daylight, camera viewpoint, and display arrangement.",
        "Distinct wares are treated as one acquisition session; object identity is not asserted.",
    ),
    CohortSpec(
        "src_global_bub_dark_cloth_closeups",
        "03_bubnivka_ceramics",
        13,
        "first filename integer 294-307",
        in_ranges((294, 307)),
        "A contiguous close-up sequence shares the same dark patterned support, lighting, crop style, and camera rendering.",
        "Some records may also be related views; this cohort conservatively captures their common acquisition source.",
    ),
    CohortSpec(
        "src_global_bub_fair_vendor_a",
        "03_bubnivka_ceramics",
        23,
        "first filename integer 83-106",
        in_ranges((83, 106)),
        "A coherent fair/vendor sequence shares the same plates, display tables, camera rendering, and 2048-pixel image family.",
        "The first close-ups and later table views are inferred to be one session; no creator identity is asserted.",
    ),
    CohortSpec(
        "src_global_bub_gray_cloth_plates",
        "03_bubnivka_ceramics",
        67,
        "first filename integer 127-193",
        in_ranges((127, 193)),
        "Every record is a centered plate photographed on the same diagonal gray plaid cloth with identical square framing and lighting.",
        "This is the clearest large acquisition cohort in Bubnivka.",
    ),
    CohortSpec(
        "src_global_bub_leaf_stump",
        "03_bubnivka_ceramics",
        7,
        "first filename integer 110-116",
        in_ranges((110, 116)),
        "Distinct wares are staged on the same tree stump and autumn-leaf outdoor set with matching camera rendering.",
        "Same photo session is visually well supported.",
    ),
    CohortSpec(
        "src_global_bub_mesh_plate_pairs",
        "03_bubnivka_ceramics",
        10,
        "first filename integer 236-240",
        in_ranges((236, 240)),
        "Five distinct plates and reverse views use the same gray-black mesh backdrop, scale, lighting, and overhead framing.",
        "Existing object/view groups remain valid inside this broader source cohort.",
    ),
    CohortSpec(
        "src_global_bub_museum_interior",
        "03_bubnivka_ceramics",
        30,
        "first filename integer 205-235",
        in_ranges((205, 235)),
        "A contiguous museum or workshop collection sequence shares wooden shelving, woven display cloths, room context, and camera rendering.",
        "The common institution/session is inferred from repeated environment rather than an explicit watermark.",
    ),
    CohortSpec(
        "src_global_bub_vognik_watermark",
        "03_bubnivka_ceramics",
        33,
        "first filename integer 1-35 or 263-270",
        in_ranges((1, 35), (263, 270)),
        "All records carry the same lower-right tree-and-horses 'Vohnik' source watermark, despite varied objects and backgrounds.",
        "The separated filename ranges are joined only because the exact source mark recurs.",
    ),
    CohortSpec(
        "src_global_kos_black_museum_cards",
        "05_kosiv_ceramics",
        9,
        "first filename integer 15-23",
        in_ranges((15, 23)),
        "Objects share the same black museum-card layout, Ukrainian caption typography, and lower-right institutional emblem.",
        "The visible card template is a strong source cue.",
    ),
    CohortSpec(
        "src_global_kos_huclia_watermark",
        "05_kosiv_ceramics",
        6,
        "first filename integer 354-359",
        in_ranges((354, 359)),
        "The records share HUCLIA.INFO branding and one museum documentation source, including repeated display-case context.",
        "Same website/institution source is explicit.",
    ),
    CohortSpec(
        "src_global_kos_lowres_dark_archive",
        "05_kosiv_ceramics",
        11,
        "first filename integer 72-83",
        in_ranges((72, 83)),
        "A contiguous low-resolution archive batch uses near-black catalog backgrounds, matching tiny image dimensions, and consistent object framing.",
        "Source identity is unknown, but the acquisition/export signature is distinctive.",
    ),
    CohortSpec(
        "src_global_kos_museum_emblem",
        "05_kosiv_ceramics",
        5,
        "first filename integer 269-275",
        in_ranges((269, 275)),
        "All available records share the same maroon museum-card treatment, caption label, and white lower-right emblem.",
        "Missing filename integers are absent from the included inventory and are not implied members.",
    ),
    CohortSpec(
        "src_global_kos_seller_wood_watermark",
        "05_kosiv_ceramics",
        32,
        "first filename integer 112-145",
        in_ranges((112, 145)),
        "A seller/catalog batch repeatedly uses the same warm wooden surfaces, shop interior, product staging, and kosivceramics.com branding.",
        "Varied scenes remain linked by explicit site branding and recurring inventory context.",
    ),
    CohortSpec(
        "src_global_kos_shop_camera_series",
        "05_kosiv_ceramics",
        23,
        "first filename integer 148-170",
        in_ranges((148, 170)),
        "One shop or workshop visit shares the same interior, vendor, displays, color rendering, and a dominant 1920x1282 camera family.",
        "The relationship is same acquisition session, not same physical object.",
    ),
    CohortSpec(
        "src_global_kos_textile_archive",
        "05_kosiv_ceramics",
        26,
        "first filename integer 45-70",
        in_ranges((45, 70)),
        "Tiles and vessels share the same fine white textile support, square archive export family, lighting, and catalog crop style.",
        "Two visually contiguous sub-batches are joined by their acquisition surface and export signature.",
    ),
    CohortSpec(
        "src_global_opn_black_card",
        "01_opishnyan_ceramics",
        3,
        "first filename integer 60-62",
        in_ranges((60, 62)),
        "Three distinct figurines use the same black product-card background, object scale, and small upper-right source mark.",
        "Small but explicit cross-boundary source cohort.",
    ),
    CohortSpec(
        "src_global_opn_gray_catalog_b",
        "01_opishnyan_ceramics",
        104,
        "first filename integer 116-224",
        in_ranges((116, 224)),
        "A large museum catalog batch shares one gray sweep, soft floor shadow, centered object framing, and consistent studio lighting.",
        "Kept separate from the earlier gray catalog batch because the export dimensions and catalog sequence differ.",
    ),
    CohortSpec(
        "src_global_opn_museum_display_logo",
        "01_opishnyan_ceramics",
        31,
        "first filename integer 85-115",
        in_ranges((85, 115)),
        "Museum/display photographs share the same interior documentation style and recurring circular lower-right source logo.",
        "The background varies within one documented collection visit.",
    ),
    CohortSpec(
        "src_global_opn_roundmark_catalog",
        "01_opishnyan_ceramics",
        261,
        "first filename integer 245-384",
        in_ranges((245, 384)),
        "All records share the same gray studio sweep and identical circular lower-right maker/source watermark across many distinct objects.",
        "At 261 images this cohort cannot yield balanced five-fold source-blocked CV; splitting it would reintroduce the exact watermark across folds.",
    ),
    CohortSpec(
        "src_global_opn_shelf_figurines",
        "01_opishnyan_ceramics",
        38,
        "first filename integer 237-243",
        in_ranges((237, 243)),
        "Distinct figurines are photographed in the same beige shelf niche with the same dried-flower prop, patterned textile, and phone camera.",
        "Multiple views within a filename family remain nested inside this source session.",
    ),
    CohortSpec(
        "src_global_opn_shop_cubbies",
        "01_opishnyan_ceramics",
        42,
        "first filename integer 228-236",
        in_ranges((228, 236)),
        "Product photographs share a gray linen tabletop, wooden cubby shelving, vertical phone format, and repeated shop inventory background.",
        "Different object families are one shop acquisition cohort.",
    ),
    CohortSpec(
        "src_global_opn_studio_gray_a",
        "01_opishnyan_ceramics",
        77,
        "first filename integer 1-58",
        in_ranges((1, 58)),
        "A professional museum batch shares the same gray studio sweep, lighting geometry, centered framing, and 2048-pixel camera/export family.",
        "Kept separate from src_global_opn_gray_catalog_b because common institutional provenance cannot be proved visually.",
    ),
    CohortSpec(
        "src_global_orn_blue_velvet_jewelry",
        "02_ornek",
        7,
        "first filename integer 133-139",
        in_ranges((133, 139)),
        "Distinct metal ornaments use the same royal-blue velvet, lighting direction, catalog framing, and photographic treatment.",
        "One jewelry catalog session is visually explicit.",
    ),
    CohortSpec(
        "src_global_orn_countertop_pottery",
        "02_ornek",
        29,
        "first filename integer 2-28",
        in_ranges((2, 28)),
        "Cups, plates, and bird figurines share the same speckled gray counter, pale window/backsplash, phone-camera rendering, and scale.",
        "Existing multi-view object families remain nested inside the broader tabletop session.",
    ),
    CohortSpec(
        "src_global_orn_stone_plate_batch",
        "02_ornek",
        7,
        "first filename integer 29-35",
        in_ranges((29, 35)),
        "Distinct colorful plates are photographed overhead on the same dark mottled stone slab with matching vertical framing.",
        "One product-photo session is strongly supported.",
    ),
    CohortSpec(
        "src_global_orn_timestamp_unglazed",
        "02_ornek",
        17,
        "first filename integer 186-202",
        in_ranges((186, 202)),
        "Unglazed wares share the same low-resolution camera family, yellow embedded date/time stamp, white cloth, and recurring outdoor setup.",
        "The timestamp and camera signature link scene variations to one photographer/source.",
    ),
    CohortSpec(
        "src_global_pet_gold_x_watermark",
        "04_petrykivka_painting",
        18,
        "first filename integer 252-269",
        in_ranges((252, 269)),
        "Every artwork reproduction carries the same gold lower-right X-shaped source watermark and warm catalog background treatment.",
        "The source mark is class-correlated and crosses production and CV boundaries in semantic-v2.",
    ),
    CohortSpec(
        "src_global_pet_rukotvory_watermark",
        "04_petrykivka_painting",
        43,
        "first filename integer 270-312",
        in_ranges((270, 312)),
        "The records share RUKOTVORY/РУКОТВОРИ site branding, consistent web-export treatment, and recurring product-documentation style.",
        "One website/source cohort; artworks and physical objects remain distinct.",
    ),
)


def read_manifest() -> list[dict[str, str]]:
    with SPLIT_MANIFEST.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_INVENTORY_ROWS:
        raise ValueError(f"Expected {EXPECTED_INVENTORY_ROWS} inventory rows, got {len(rows)}")
    if len({row["image_id"] for row in rows}) != len(rows):
        raise ValueError("split_manifest.csv contains duplicate image IDs")
    included = sum(row["inclusion_status"] == "included" for row in rows)
    if included != EXPECTED_INCLUDED_ROWS:
        raise ValueError(f"Expected {EXPECTED_INCLUDED_ROWS} included rows, got {included}")
    return rows


def first_filename_integer(file_name: str) -> int:
    match = re.search(r"\d+", file_name)
    if match is None:
        raise ValueError(f"Filename has no integer: {file_name}")
    return int(match.group())


def inventory_projection_fingerprint(rows: list[dict[str, str]]) -> str:
    digest = hashlib.sha256()
    for row in sorted(rows, key=lambda item: item["image_id"]):
        payload = "\x1f".join(
            (
                row["image_id"],
                row["ornament_label"],
                row["inclusion_status"],
                row["semantic_split_group_id"],
            )
        )
        digest.update(payload.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def build_cohorts(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    cohorts: list[dict[str, object]] = []
    claimed: dict[str, str] = {}
    for spec in sorted(SPECS, key=lambda item: item.cohort_id):
        selected = sorted(
            (
                row
                for row in rows
                if row["inclusion_status"] == "included"
                and row["ornament_label"] == spec.ornament_label
                and spec.selector(first_filename_integer(row["file_name"]))
            ),
            key=lambda row: row["image_id"],
        )
        if len(selected) != spec.expected_count:
            raise ValueError(
                f"{spec.cohort_id}: expected {spec.expected_count} members, got {len(selected)}"
            )
        for row in selected:
            previous = claimed.get(row["image_id"])
            if previous is not None:
                raise ValueError(
                    f"Image {row['image_id']} overlaps {previous} and {spec.cohort_id}"
                )
            claimed[row["image_id"]] = spec.cohort_id
        cohorts.append(
            {
                "cohort_id": spec.cohort_id,
                "ornament_label": spec.ornament_label,
                "relationship": RELATIONSHIP,
                "image_ids": [row["image_id"] for row in selected],
                "evidence": spec.evidence,
                "notes": spec.notes,
            }
        )
    if len(cohorts) != EXPECTED_COHORTS:
        raise ValueError(f"Expected {EXPECTED_COHORTS} cohorts, got {len(cohorts)}")
    if len(claimed) != EXPECTED_MEMBERS:
        raise ValueError(f"Expected {EXPECTED_MEMBERS} unique members, got {len(claimed)}")
    return cohorts


def write_adjudication(
    rows: list[dict[str, str]], cohorts: list[dict[str, object]]
) -> str:
    fingerprint = inventory_projection_fingerprint(rows)
    payload = {
        "review_version": REVIEW_VERSION,
        "review_status": "complete",
        "inventory_projection_fingerprint_sha256": fingerprint,
        "cohorts": cohorts,
    }
    ADJUDICATION_PATH.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return fingerprint


def write_members(rows: list[dict[str, str]], cohorts: list[dict[str, object]]) -> None:
    by_id = {row["image_id"]: row for row in rows}
    spec_by_id = {spec.cohort_id: spec for spec in SPECS}
    fields = (
        "cohort_id",
        "cohort_size",
        "ornament_label",
        "relationship",
        "image_id",
        "relative_path",
        "file_name",
        "source_rule",
        "semantic_split_group_id",
        "current_production_split",
        "current_cv_fold",
        "evidence",
        "notes",
    )
    with MEMBERS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for cohort in cohorts:
            cohort_id = str(cohort["cohort_id"])
            spec = spec_by_id[cohort_id]
            image_ids = list(cohort["image_ids"])
            for image_id in image_ids:
                row = by_id[str(image_id)]
                writer.writerow(
                    {
                        "cohort_id": cohort_id,
                        "cohort_size": len(image_ids),
                        "ornament_label": cohort["ornament_label"],
                        "relationship": RELATIONSHIP,
                        "image_id": image_id,
                        "relative_path": row["relative_path"],
                        "file_name": row["file_name"],
                        "source_rule": spec.source_rule,
                        "semantic_split_group_id": row["semantic_split_group_id"],
                        "current_production_split": row["production_split"],
                        "current_cv_fold": row["cv_fold"],
                        "evidence": cohort["evidence"],
                        "notes": cohort["notes"],
                    }
                )


def distribution_text(values: Counter[str], order: tuple[str, ...]) -> str:
    return ", ".join(f"{key}:{values.get(key, 0)}" for key in order)


def write_readme(
    rows: list[dict[str, str]],
    cohorts: list[dict[str, object]],
    fingerprint: str,
) -> None:
    by_id = {row["image_id"]: row for row in rows}
    spec_by_id = {spec.cohort_id: spec for spec in SPECS}
    split_versions = {row["split_version"] for row in rows}
    if len(split_versions) != 1:
        raise ValueError(f"Expected one split version, got {sorted(split_versions)}")
    current_split_version = next(iter(split_versions))
    lines = [
        "# Global acquisition/source cohort review",
        "",
        "This directory records the completed full-inventory visual audit used to prevent class-correlated acquisition cues from crossing evaluation boundaries. All 2,023 included images were reviewed in naturally sorted contact sheets with the semantic-v2 production split and CV fold overlaid. The review identified 28 conservative, non-overlapping cohorts containing 982 images (48.54% of the included inventory). The remaining 1,041 images are unassigned because no repeated source relationship was visually strong enough; unassigned does not mean proven-independent provenance.",
        "",
        f"`adjudication.json` is canonical and contains explicit, sorted image IDs. `cohort_members.csv` is a readable projection with the current `{current_split_version}` boundary assignments. Filename rules below only document how the visually reviewed cohorts can be reproduced; downstream split code must consume the explicit IDs.",
        "",
        "## Method and decision rule",
        "",
        "A cohort was accepted only when distinct objects or artworks shared an exact watermark/logo, a repeated backdrop and lighting setup, an embedded camera timestamp/export signature, or a coherent photographed visit/session. Ornament style alone was not accepted as source evidence. Each accepted relationship is `same_acquisition_source`; it does not claim that the members show the same physical object.",
        "",
        f"Inventory projection fingerprint: `{fingerprint}`. It hashes all 2,055 semantic-v2 inventory rows in image-ID order over `image_id`, `ornament_label`, `inclusion_status`, and `semantic_split_group_id`, separated by unit separators and newline records.",
        "",
        f"## Reviewed cohorts and `{current_split_version}` boundary distribution",
        "",
        "| Cohort | Rule provenance | Images | Semantic groups | Production split | CV fold (`test` is `-`) | Evidence |",
        "|---|---|---:|---:|---|---|---|",
    ]
    for cohort in cohorts:
        cohort_id = str(cohort["cohort_id"])
        spec = spec_by_id[cohort_id]
        members = [by_id[str(image_id)] for image_id in cohort["image_ids"]]
        production = Counter(row["production_split"] for row in members)
        folds = Counter(row["cv_fold"] or "-" for row in members)
        semantic_groups = len({row["semantic_split_group_id"] for row in members})
        evidence = str(cohort["evidence"]).replace("|", "\\|")
        lines.append(
            "| "
            + " | ".join(
                (
                    f"`{cohort_id}`",
                    spec.source_rule,
                    str(len(members)),
                    str(semantic_groups),
                    distribution_text(production, ("train", "validation", "test")),
                    distribution_text(folds, ("0", "1", "2", "3", "4", "-")),
                    evidence,
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Before source-cohort integration, all 28 cohorts crossed more than one semantic-v2 production split and more than one CV/evaluation boundary. Twenty-five touched the semantic-v2 test split; the other three crossed train/validation and development folds. The current table is regenerated from the split version named in its heading. In source-atomic v3, every cohort occupies exactly one production split and every development cohort occupies exactly one CV fold.",
            "",
            "The largest cohort, `src_global_opn_roundmark_catalog`, has 261 images and 88 pre-cohort semantic groups. A strict source-blocked five-fold protocol therefore cannot also keep equal per-class fold sizes. Evaluation should preserve the source boundary and report the resulting fold imbalance or use explicit leave-source-cohort diagnostics; splitting the cohort would knowingly leak the identical watermark.",
            "",
            "## Deliberately unmerged and uncertain cases",
            "",
            "- The two large Opishnyan gray-studio cohorts may share an institution, but they remain separate because export dimensions and catalog sequences differ and common provenance cannot be proved visually.",
            "- Bubnivka fair, museum-interior, and dark-cloth cohorts are session inferences from repeated environments and camera signatures; unlike the Vohnik cohort, they do not have an explicit source mark.",
            "- Kosiv textile-archive and shop-camera cohorts are strongly coherent acquisition batches, but their institution or photographer names remain unknown.",
            "- Ornek ranges 344-365, 366-384, 449-456, and 457-480; Kosiv 174-201; and several Petrykivka illustration runs were not grouped. Their style is coherent, but the review could not distinguish one creator/source from motif-level similarity without external provenance.",
            "- Provenance and license remain unknown. A shared watermark supports evaluation grouping, not authorization to redistribute or deploy an image.",
            "",
            "## Rebuild and validate",
            "",
            "From the repository root:",
            "",
            "```bash",
            "python3 step_2/development/scripts/build_global_source_cohorts.py",
            "```",
            "",
            "The generator fails if inventory counts, expected cohort counts, inclusion status, class labels, image-ID uniqueness, or non-overlap change.",
            "",
        ]
    )
    README_PATH.write_text("\n".join(lines), encoding="utf-8")


def validate_outputs(rows: list[dict[str, str]]) -> None:
    payload = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    if list(payload) != [
        "review_version",
        "review_status",
        "inventory_projection_fingerprint_sha256",
        "cohorts",
    ]:
        raise ValueError("Unexpected adjudication top-level schema or ordering")
    cohorts = payload["cohorts"]
    cohort_ids = [cohort["cohort_id"] for cohort in cohorts]
    if cohort_ids != sorted(cohort_ids) or len(cohort_ids) != len(set(cohort_ids)):
        raise ValueError("Cohort IDs are not sorted and unique")
    by_id = {row["image_id"]: row for row in rows}
    seen: set[str] = set()
    for cohort in cohorts:
        if set(cohort) != {
            "cohort_id",
            "ornament_label",
            "relationship",
            "image_ids",
            "evidence",
            "notes",
        }:
            raise ValueError(f"Unexpected cohort schema for {cohort['cohort_id']}")
        image_ids = cohort["image_ids"]
        if image_ids != sorted(image_ids) or len(image_ids) < 2:
            raise ValueError(f"Invalid image ID ordering/count for {cohort['cohort_id']}")
        for image_id in image_ids:
            if image_id in seen:
                raise ValueError(f"Overlapping image ID: {image_id}")
            seen.add(image_id)
            row = by_id[image_id]
            if row["inclusion_status"] != "included":
                raise ValueError(f"Excluded image in cohort: {image_id}")
            if row["ornament_label"] != cohort["ornament_label"]:
                raise ValueError(f"Label mismatch for {image_id}")
        if cohort["relationship"] != RELATIONSHIP or not cohort["evidence"]:
            raise ValueError(f"Invalid relationship/evidence for {cohort['cohort_id']}")
    if len(seen) != EXPECTED_MEMBERS:
        raise ValueError(f"Expected {EXPECTED_MEMBERS} adjudicated members, got {len(seen)}")
    with MEMBERS_PATH.open(encoding="utf-8", newline="") as handle:
        member_rows = list(csv.DictReader(handle))
    if len(member_rows) != EXPECTED_MEMBERS:
        raise ValueError(f"Expected {EXPECTED_MEMBERS} CSV rows, got {len(member_rows)}")


def main() -> None:
    rows = read_manifest()
    cohorts = build_cohorts(rows)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = write_adjudication(rows, cohorts)
    write_members(rows, cohorts)
    write_readme(rows, cohorts, fingerprint)
    validate_outputs(rows)
    print(f"Wrote {len(cohorts)} cohorts with {EXPECTED_MEMBERS} unique included images")
    print(f"Inventory projection fingerprint: {fingerprint}")
    for path in (ADJUDICATION_PATH, MEMBERS_PATH, README_PATH):
        print(f"{path.relative_to(REPO_ROOT)} sha256={sha256_file(path)}")


if __name__ == "__main__":
    main()
