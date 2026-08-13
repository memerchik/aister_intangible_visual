from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SRC_ROOT = STEP_ROOT / "src"

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.embeddings import (  # noqa: E402
    EmbeddingCacheError,
    load_allowlisted_embeddings,
    load_embedding_block,
)
from ornament_classifier.paths import ProjectPaths  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_array(values: np.ndarray) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(values).tobytes(order="C")
    ).hexdigest()


class TinyPhase5Fixture:
    def __init__(self, root: Path) -> None:
        self.repository_root = root / "repository"
        self.step_root = self.repository_root / "step_02"
        self.paths = ProjectPaths.from_step_root(self.step_root)
        self.contract_path = (
            self.step_root
            / "phases"
            / "phase_05_source_robustness"
            / "experiment_contract.json"
        )
        self.cache_path = (
            self.repository_root / ".cache" / "step_02" / "fixture_embeddings.npz"
        )
        self.development_path = self.step_root / "splits" / "development.csv"
        self.audit_path = self.step_root / "splits" / "split_audit.json"
        self.metrics_path = self.step_root / "outputs" / "phase_4" / "metrics.json"
        self.image_ids = ("img_a", "img_b")
        self.content_hashes = ("a" * 64, "b" * 64)
        self.fingerprint = "c" * 64
        self.values = np.eye(2, dtype=np.float32)
        self._build()

    def _build(self) -> None:
        for path in (
            self.contract_path.parent,
            self.cache_path.parent,
            self.development_path.parent,
            self.metrics_path.parent,
            self.step_root / "data" / "dataset_dev" / "class_a",
        ):
            path.mkdir(parents=True, exist_ok=True)
        for name in ("a.jpg", "b.jpg"):
            (self.step_root / "data" / "dataset_dev" / "class_a" / name).write_bytes(
                b"fixture"
            )

        fields = (
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
        rows = (
            {
                "image_id": "img_a",
                "relative_path": "dataset_dev/class_a/a.jpg",
                "ornament_label": "class_a",
                "content_sha256": self.content_hashes[0],
                "split_group_id": "group_a",
                "inclusion_status": "included",
                "production_split": "train",
                "cv_fold": "0",
                "split_version": "fixture_split",
                "split_seed": "7",
            },
            {
                "image_id": "img_b",
                "relative_path": "dataset_dev/class_a/b.jpg",
                "ornament_label": "class_a",
                "content_sha256": self.content_hashes[1],
                "split_group_id": "group_b",
                "inclusion_status": "included",
                "production_split": "validation",
                "cv_fold": "1",
                "split_version": "fixture_split",
                "split_seed": "7",
            },
        )
        with self.development_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

        audit = {
            "split_version": "fixture_split",
            "split_seed": 7,
            "policy": {"cv_folds": 2},
            "production_split_counts": {"train": 1, "validation": 1, "test": 0},
            "development_cv_fold_counts": {"0": 1, "1": 1},
            "assignment_fingerprint_sha256": "d" * 64,
        }
        self.audit_path.write_text(
            json.dumps(audit, sort_keys=True), encoding="utf-8"
        )
        self.metrics_path.write_text("{}\n", encoding="utf-8")
        self.write_cache()

        data_digest = hashlib.sha256()
        for image_id, content_hash in zip(self.image_ids, self.content_hashes):
            data_digest.update(f"{image_id}\x1f{content_hash}\n".encode("utf-8"))
        self.contract = {
            "schema_version": 1,
            "phase": "phase_05_source_robustness",
            "experiment_version": "fixture_v1",
            "status": "frozen_before_fit",
            "scope": {
                "development_image_count": 2,
                "outer_folds": ["0", "1"],
                "sealed_test_access": "forbidden",
            },
            "input_contract": {
                "development_csv": "step_02/splits/development.csv",
                "development_csv_sha256": sha256_file(self.development_path),
                "split_audit": "step_02/splits/split_audit.json",
                "split_audit_sha256": sha256_file(self.audit_path),
                "split_version": "fixture_split",
                "split_seed": 7,
                "split_assignment_fingerprint_sha256": "d" * 64,
                "data_fingerprint_sha256": data_digest.hexdigest(),
                "phase_4_metrics": "step_02/outputs/phase_4/metrics.json",
                "phase_4_metrics_sha256": sha256_file(self.metrics_path),
            },
            "cache_allowlist": {
                "fixture": {
                    "path": ".cache/step_02/fixture_embeddings.npz",
                    "file_sha256": sha256_file(self.cache_path),
                    "fingerprint": self.fingerprint,
                    "arrays": {
                        "embedding__cls": {
                            "shape": [2, 2],
                            "dtype": "float32",
                            "sha256": sha256_array(self.values),
                        }
                    },
                }
            },
        }
        self.write_contract()

    def write_cache(
        self,
        *,
        values: np.ndarray | None = None,
        image_ids: np.ndarray | None = None,
        fingerprint: str | None = None,
        metadata: object | None = None,
    ) -> None:
        selected_values = self.values if values is None else values
        selected_ids = (
            np.asarray(self.image_ids) if image_ids is None else image_ids
        )
        selected_fingerprint = self.fingerprint if fingerprint is None else fingerprint
        selected_metadata = (
            {
                "embedding_fingerprint_sha256": selected_fingerprint,
                "cache_identity": {
                    "data_fingerprint": getattr(self, "contract", {})
                    .get("input_contract", {})
                    .get("data_fingerprint_sha256", ""),
                },
            }
            if metadata is None
            else metadata
        )
        if not selected_metadata["cache_identity"]["data_fingerprint"]:
            digest = hashlib.sha256()
            for image_id, content_hash in zip(self.image_ids, self.content_hashes):
                digest.update(
                    f"{image_id}\x1f{content_hash}\n".encode("utf-8")
                )
            selected_metadata["cache_identity"]["data_fingerprint"] = digest.hexdigest()
        np.savez_compressed(
            self.cache_path,
            fingerprint=np.asarray(selected_fingerprint),
            image_ids=selected_ids,
            metadata_json=np.asarray(json.dumps(selected_metadata, sort_keys=True)),
            embedding__cls=selected_values,
        )

    def write_contract(self) -> None:
        self.contract_path.write_text(
            json.dumps(self.contract, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def accept_cache_file(self) -> None:
        self.contract["cache_allowlist"]["fixture"]["file_sha256"] = sha256_file(
            self.cache_path
        )

    def accept_array(self, values: np.ndarray) -> None:
        spec = self.contract["cache_allowlist"]["fixture"]["arrays"][
            "embedding__cls"
        ]
        spec["shape"] = list(values.shape)
        spec["dtype"] = values.dtype.name
        spec["sha256"] = sha256_array(values)


class Phase5CanonicalCacheLoaderTests(unittest.TestCase):
    def test_all_frozen_blocks_load_with_exact_provenance(self) -> None:
        blocks = load_allowlisted_embeddings(
            {
                "center": ("dinov3_center", "embedding__cls"),
                "letterbox": ("dinov3_letterbox", "embedding__cls"),
                "letterbox_concat": (
                    "dinov3_letterbox",
                    "embedding__cls_patch_concat",
                ),
                "global": ("dinov3_global", "embedding__cls"),
            }
        )

        self.assertEqual(set(blocks), {"center", "letterbox", "letterbox_concat", "global"})
        self.assertEqual(blocks["center"].values.shape, (1693, 384))
        self.assertEqual(blocks["letterbox_concat"].values.shape, (1693, 768))
        for block in blocks.values():
            self.assertEqual(block.values.dtype, np.dtype("float32"))
            self.assertFalse(block.values.flags.writeable)
            self.assertEqual(len(block.image_ids), 1693)
            self.assertEqual(sha256_file(block.cache_path), block.cache_sha256)
            self.assertEqual(sha256_array(block.values), block.array_sha256)
            self.assertEqual(
                block.provenance["data_fingerprint_sha256"],
                block.data_fingerprint_sha256,
            )
            self.assertEqual(
                block.metadata["embedding_fingerprint_sha256"], block.fingerprint
            )

    def test_iterable_requests_use_stable_reference_keys(self) -> None:
        blocks = load_allowlisted_embeddings(
            (("dinov3_global", "embedding__cls"),)
        )
        self.assertEqual(
            tuple(blocks), ("dinov3_global.embedding__cls",)
        )

    def test_unknown_cache_and_array_are_rejected(self) -> None:
        with self.assertRaisesRegex(EmbeddingCacheError, "not allowlisted"):
            load_embedding_block("unknown", "embedding__cls")
        with self.assertRaisesRegex(EmbeddingCacheError, "not allowlisted"):
            load_embedding_block("dinov3_global", "embedding__unknown")

    def test_loader_has_no_scan_or_phase_4_runner_dependency(self) -> None:
        source = (
            SRC_ROOT / "ornament_classifier" / "embeddings.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn(".glob(", source)
        self.assertNotIn(".rglob(", source)
        self.assertNotIn("run_pretrained_embeddings", source)
        self.assertIn("allow_pickle=False", source)


class Phase5CacheTamperTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.fixture = TinyPhase5Fixture(Path(self.temporary.name))

    def load(self):
        return load_embedding_block(
            "fixture", "embedding__cls", paths=self.fixture.paths
        )

    def test_tiny_valid_fixture_loads(self) -> None:
        block = self.load()
        np.testing.assert_array_equal(block.values, np.eye(2, dtype=np.float32))
        self.assertEqual(block.image_ids, ("img_a", "img_b"))

    def test_cache_file_hash_is_mandatory(self) -> None:
        with self.fixture.cache_path.open("ab") as handle:
            handle.write(b"tamper")
        with self.assertRaisesRegex(EmbeddingCacheError, "File hash mismatch"):
            self.load()

    def test_internal_fingerprint_is_mandatory(self) -> None:
        self.fixture.write_cache(fingerprint="e" * 64)
        self.fixture.accept_cache_file()
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "fingerprint mismatch"):
            self.load()

    def test_current_image_id_order_is_mandatory(self) -> None:
        self.fixture.write_cache(image_ids=np.asarray(("img_b", "img_a")))
        self.fixture.accept_cache_file()
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "development-row order"):
            self.load()

    def test_array_hash_is_mandatory(self) -> None:
        changed = np.asarray(((0.0, 1.0), (1.0, 0.0)), dtype=np.float32)
        self.fixture.write_cache(values=changed)
        self.fixture.accept_cache_file()
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "array hash mismatch"):
            self.load()

    def test_nonfinite_and_nonunit_arrays_are_rejected_after_hashing(self) -> None:
        cases = (
            (
                np.asarray(((np.nan, 0.0), (0.0, 1.0)), dtype=np.float32),
                "non-finite",
            ),
            (
                np.asarray(((2.0, 0.0), (0.0, 1.0)), dtype=np.float32),
                "unit normalized",
            ),
        )
        for values, message in cases:
            with self.subTest(message=message):
                self.fixture.write_cache(values=values)
                self.fixture.accept_cache_file()
                self.fixture.accept_array(values)
                self.fixture.write_contract()
                with self.assertRaisesRegex(EmbeddingCacheError, message):
                    self.load()

    def test_shape_dtype_and_object_payload_are_rejected(self) -> None:
        wrong_shape = np.ones((2, 3), dtype=np.float32)
        wrong_shape /= np.linalg.norm(wrong_shape, axis=1, keepdims=True)
        self.fixture.write_cache(values=wrong_shape)
        self.fixture.accept_cache_file()
        self.fixture.contract["cache_allowlist"]["fixture"]["arrays"][
            "embedding__cls"
        ]["sha256"] = sha256_array(wrong_shape)
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "shape"):
            self.load()

        self.fixture = TinyPhase5Fixture(Path(self.temporary.name) / "dtype")
        wrong_dtype = np.eye(2, dtype=np.float64)
        self.fixture.write_cache(values=wrong_dtype)
        self.fixture.accept_cache_file()
        self.fixture.accept_array(wrong_dtype)
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "float32"):
            self.load()

        self.fixture = TinyPhase5Fixture(Path(self.temporary.name) / "object")
        self.fixture.write_cache(image_ids=np.asarray(self.fixture.image_ids, dtype=object))
        self.fixture.accept_cache_file()
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "safely load"):
            self.load()

    def test_current_input_file_hash_and_data_fingerprint_are_mandatory(self) -> None:
        with self.fixture.development_path.open("a", encoding="utf-8") as handle:
            handle.write("\n")
        with self.assertRaisesRegex(EmbeddingCacheError, "File hash mismatch"):
            self.load()

        self.fixture = TinyPhase5Fixture(Path(self.temporary.name) / "fingerprint")
        text = self.fixture.development_path.read_text(encoding="utf-8")
        self.fixture.development_path.write_text(
            text.replace("a" * 64, "f" * 64), encoding="utf-8"
        )
        self.fixture.contract["input_contract"]["development_csv_sha256"] = (
            sha256_file(self.fixture.development_path)
        )
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "order fingerprint"):
            self.load()

    def test_contract_cannot_redirect_cache_outside_repository(self) -> None:
        self.fixture.contract["cache_allowlist"]["fixture"]["path"] = "../outside.npz"
        self.fixture.write_contract()
        with self.assertRaisesRegex(EmbeddingCacheError, "escapes the repository"):
            self.load()


if __name__ == "__main__":
    unittest.main()
