# Global acquisition/source cohort review

This directory records the full-inventory visual audit used to keep repeated
acquisition cues on one side of each evaluation split. All 2,023 included
images were reviewed in naturally sorted contact sheets with the semantic-v2
split and fold overlaid. The review identified 28 conservative, non-overlapping
cohorts containing 982 images (48.54% of the included inventory). The remaining
1,041 images are unassigned because no repeated source relationship was strong
enough to identify visually; this does not prove that their sources are
independent.

`adjudication.json` contains the reviewed, sorted image IDs used by the split
builder. `cohort_members.csv` presents the same cohorts with their current
`step02_source_atomic_v3` assignments. The filename rules below explain how the
reviewed cohorts were identified; the split code uses the explicit IDs rather
than inferring membership from filenames.

## Method and decision rule

A cohort was accepted only when distinct objects or artworks shared an exact watermark/logo, a repeated backdrop and lighting setup, an embedded camera timestamp/export signature, or a coherent photographed visit/session. Ornament style alone was not accepted as source evidence. Each accepted relationship is `same_acquisition_source`; it does not claim that the members show the same physical object.

Inventory projection fingerprint: `bc90b6022b2f8bb6183e7d6f9cb73ab1908900077c70ad5ca4ea9355b01c81aa`. It hashes all 2,055 semantic-v2 inventory rows in image-ID order over `image_id`, `ornament_label`, `inclusion_status`, and `semantic_split_group_id`, separated by unit separators and newline records.

## Reviewed cohorts and `step02_source_atomic_v3` boundary distribution

| Cohort | Rule provenance | Images | Semantic groups | Production split | CV fold (`test` is `-`) | Evidence |
|---|---|---:|---:|---|---|---|
| `src_global_bub_blue_fair_table` | first filename integer 308-317 | 10 | 10 | train:10, validation:0, test:0 | 0:0, 1:0, 2:0, 3:10, 4:0, -:0 | One outdoor vendor sequence uses the same vivid blue tablecloth, daylight, camera viewpoint, and display arrangement. |
| `src_global_bub_dark_cloth_closeups` | first filename integer 294-307 | 13 | 13 | train:0, validation:13, test:0 | 0:0, 1:0, 2:13, 3:0, 4:0, -:0 | A contiguous close-up sequence shares the same dark patterned support, lighting, crop style, and camera rendering. |
| `src_global_bub_fair_vendor_a` | first filename integer 83-106 | 23 | 22 | train:23, validation:0, test:0 | 0:0, 1:23, 2:0, 3:0, 4:0, -:0 | A coherent fair/vendor sequence shares the same plates, display tables, camera rendering, and 2048-pixel image family. |
| `src_global_bub_gray_cloth_plates` | first filename integer 127-193 | 67 | 62 | train:0, validation:0, test:67 | 0:0, 1:0, 2:0, 3:0, 4:0, -:67 | Every record is a centered plate photographed on the same diagonal gray plaid cloth with identical square framing and lighting. |
| `src_global_bub_leaf_stump` | first filename integer 110-116 | 7 | 7 | train:7, validation:0, test:0 | 0:0, 1:0, 2:0, 3:0, 4:7, -:0 | Distinct wares are staged on the same tree stump and autumn-leaf outdoor set with matching camera rendering. |
| `src_global_bub_mesh_plate_pairs` | first filename integer 236-240 | 10 | 5 | train:10, validation:0, test:0 | 0:0, 1:0, 2:10, 3:0, 4:0, -:0 | Five distinct plates and reverse views use the same gray-black mesh backdrop, scale, lighting, and overhead framing. |
| `src_global_bub_museum_interior` | first filename integer 205-235 | 30 | 29 | train:30, validation:0, test:0 | 0:0, 1:0, 2:0, 3:30, 4:0, -:0 | A contiguous museum or workshop collection sequence shares wooden shelving, woven display cloths, room context, and camera rendering. |
| `src_global_bub_vognik_watermark` | first filename integer 1-35 or 263-270 | 33 | 33 | train:0, validation:33, test:0 | 0:0, 1:0, 2:0, 3:0, 4:33, -:0 | All records carry the same lower-right tree-and-horses 'Vohnik' source watermark, despite varied objects and backgrounds. |
| `src_global_kos_black_museum_cards` | first filename integer 15-23 | 9 | 9 | train:0, validation:0, test:9 | 0:0, 1:0, 2:0, 3:0, 4:0, -:9 | Objects share the same black museum-card layout, Ukrainian caption typography, and lower-right institutional emblem. |
| `src_global_kos_huclia_watermark` | first filename integer 354-359 | 6 | 6 | train:6, validation:0, test:0 | 0:0, 1:0, 2:0, 3:6, 4:0, -:0 | The records share HUCLIA.INFO branding and one museum documentation source, including repeated display-case context. |
| `src_global_kos_lowres_dark_archive` | first filename integer 72-83 | 11 | 11 | train:0, validation:0, test:11 | 0:0, 1:0, 2:0, 3:0, 4:0, -:11 | A contiguous low-resolution archive batch uses near-black catalog backgrounds, matching tiny image dimensions, and consistent object framing. |
| `src_global_kos_museum_emblem` | first filename integer 269-275 | 5 | 5 | train:5, validation:0, test:0 | 0:0, 1:0, 2:0, 3:5, 4:0, -:0 | All available records share the same maroon museum-card treatment, caption label, and white lower-right emblem. |
| `src_global_kos_seller_wood_watermark` | first filename integer 112-145 | 32 | 31 | train:0, validation:0, test:32 | 0:0, 1:0, 2:0, 3:0, 4:0, -:32 | A seller/catalog batch repeatedly uses the same warm wooden surfaces, shop interior, product staging, and kosivceramics.com branding. |
| `src_global_kos_shop_camera_series` | first filename integer 148-170 | 23 | 22 | train:0, validation:23, test:0 | 0:0, 1:0, 2:0, 3:0, 4:23, -:0 | One shop or workshop visit shares the same interior, vendor, displays, color rendering, and a dominant 1920x1282 camera family. |
| `src_global_kos_textile_archive` | first filename integer 45-70 | 26 | 26 | train:0, validation:26, test:0 | 0:0, 1:0, 2:0, 3:0, 4:26, -:0 | Tiles and vessels share the same fine white textile support, square archive export family, lighting, and catalog crop style. |
| `src_global_opn_black_card` | first filename integer 60-62 | 3 | 3 | train:0, validation:0, test:3 | 0:0, 1:0, 2:0, 3:0, 4:0, -:3 | Three distinct figurines use the same black product-card background, object scale, and small upper-right source mark. |
| `src_global_opn_gray_catalog_b` | first filename integer 116-224 | 104 | 96 | train:0, validation:104, test:0 | 0:0, 1:0, 2:0, 3:104, 4:0, -:0 | A large museum catalog batch shares one gray sweep, soft floor shadow, centered object framing, and consistent studio lighting. |
| `src_global_opn_museum_display_logo` | first filename integer 85-115 | 31 | 31 | train:31, validation:0, test:0 | 0:0, 1:0, 2:31, 3:0, 4:0, -:0 | Museum/display photographs share the same interior documentation style and recurring circular lower-right source logo. |
| `src_global_opn_roundmark_catalog` | first filename integer 245-384 | 261 | 88 | train:261, validation:0, test:0 | 0:261, 1:0, 2:0, 3:0, 4:0, -:0 | All records share the same gray studio sweep and identical circular lower-right maker/source watermark across many distinct objects. |
| `src_global_opn_shelf_figurines` | first filename integer 237-243 | 38 | 7 | train:38, validation:0, test:0 | 0:0, 1:0, 2:0, 3:0, 4:38, -:0 | Distinct figurines are photographed in the same beige shelf niche with the same dried-flower prop, patterned textile, and phone camera. |
| `src_global_opn_shop_cubbies` | first filename integer 228-236 | 42 | 9 | train:42, validation:0, test:0 | 0:0, 1:0, 2:0, 3:0, 4:42, -:0 | Product photographs share a gray linen tabletop, wooden cubby shelving, vertical phone format, and repeated shop inventory background. |
| `src_global_opn_studio_gray_a` | first filename integer 1-58 | 77 | 58 | train:0, validation:0, test:77 | 0:0, 1:0, 2:0, 3:0, 4:0, -:77 | A professional museum batch shares the same gray studio sweep, lighting geometry, centered framing, and 2048-pixel camera/export family. |
| `src_global_orn_blue_velvet_jewelry` | first filename integer 133-139 | 7 | 7 | train:0, validation:0, test:7 | 0:0, 1:0, 2:0, 3:0, 4:0, -:7 | Distinct metal ornaments use the same royal-blue velvet, lighting direction, catalog framing, and photographic treatment. |
| `src_global_orn_countertop_pottery` | first filename integer 2-28 | 29 | 23 | train:0, validation:0, test:29 | 0:0, 1:0, 2:0, 3:0, 4:0, -:29 | Cups, plates, and bird figurines share the same speckled gray counter, pale window/backsplash, phone-camera rendering, and scale. |
| `src_global_orn_stone_plate_batch` | first filename integer 29-35 | 7 | 6 | train:0, validation:0, test:7 | 0:0, 1:0, 2:0, 3:0, 4:0, -:7 | Distinct colorful plates are photographed overhead on the same dark mottled stone slab with matching vertical framing. |
| `src_global_orn_timestamp_unglazed` | first filename integer 186-202 | 17 | 17 | train:0, validation:0, test:17 | 0:0, 1:0, 2:0, 3:0, 4:0, -:17 | Unglazed wares share the same low-resolution camera family, yellow embedded date/time stamp, white cloth, and recurring outdoor setup. |
| `src_global_pet_gold_x_watermark` | first filename integer 252-269 | 18 | 18 | train:0, validation:18, test:0 | 0:0, 1:0, 2:0, 3:18, 4:0, -:0 | Every artwork reproduction carries the same gold lower-right X-shaped source watermark and warm catalog background treatment. |
| `src_global_pet_rukotvory_watermark` | first filename integer 270-312 | 43 | 43 | train:0, validation:0, test:43 | 0:0, 1:0, 2:0, 3:0, 4:0, -:43 | The records share RUKOTVORY/РУКОТВОРИ site branding, consistent web-export treatment, and recurring product-documentation style. |

Before source-cohort integration, all 28 cohorts crossed more than one semantic-v2 production split and more than one CV/evaluation boundary. Twenty-five touched the semantic-v2 test split; the other three crossed train/validation and development folds. The current table is regenerated from the split version named in its heading. In source-atomic v3, every cohort occupies exactly one production split and every development cohort occupies exactly one CV fold.

The largest cohort, `src_global_opn_roundmark_catalog`, has 261 images and 88 pre-cohort semantic groups. A strict source-blocked five-fold protocol therefore cannot also keep equal per-class fold sizes. Evaluation should preserve the source boundary and report the resulting fold imbalance or use explicit leave-source-cohort diagnostics; splitting the cohort would knowingly leak the identical watermark.

## Uncertain cases kept separate

- The two large Opishnyan gray-studio cohorts may share an institution, but they remain separate because export dimensions and catalog sequences differ and common provenance cannot be proved visually.
- Bubnivka fair, museum-interior, and dark-cloth cohorts are session inferences from repeated environments and camera signatures; unlike the Vohnik cohort, they do not have an explicit source mark.
- Kosiv textile-archive and shop-camera cohorts are strongly coherent acquisition batches, but their institution or photographer names remain unknown.
- Ornek ranges 344-365, 366-384, 449-456, and 457-480; Kosiv 174-201; and several Petrykivka illustration runs were not grouped. Their style is coherent, but the review could not distinguish one creator/source from motif-level similarity without external provenance.
- Provenance and license remain unknown. A shared watermark supports evaluation grouping, not authorization to redistribute or deploy an image.

## Rebuild and validate

From the repository root:

```bash
python3 step_2/development/scripts/build_global_source_cohorts.py
```

The generator fails if inventory counts, expected cohort counts, inclusion status, class labels, image-ID uniqueness, or non-overlap change.
