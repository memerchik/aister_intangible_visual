from __future__ import annotations

import ast
import hashlib
import sys
import unittest
from pathlib import Path


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SRC_ROOT = STEP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.embeddings_v2 import (  # noqa: E402
    EXPECTED_PHASE_5_V2_CONTRACT_SHA256,
    load_allowlisted_embeddings_v2,
)


class Phase5V2CacheLoaderTests(unittest.TestCase):
    def test_contract_is_exactly_pinned_before_cache_access(self) -> None:
        path = (
            STEP_ROOT
            / "phases"
            / "phase_05_source_robustness"
            / "experiment_contract_v2.json"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            EXPECTED_PHASE_5_V2_CONTRACT_SHA256,
        )

    def test_all_six_v2_blocks_load_with_frozen_provenance(self) -> None:
        requests = {
            "center_cls": ("dinov3_center", "embedding__cls"),
            "center_concat": (
                "dinov3_center",
                "embedding__cls_patch_concat",
            ),
            "letterbox_cls": ("dinov3_letterbox", "embedding__cls"),
            "letterbox_concat": (
                "dinov3_letterbox",
                "embedding__cls_patch_concat",
            ),
            "global_cls": ("dinov3_global", "embedding__cls"),
            "global_concat": (
                "dinov3_global",
                "embedding__cls_patch_concat",
            ),
        }
        blocks = load_allowlisted_embeddings_v2(requests)
        self.assertEqual(set(blocks), set(requests))
        for block in blocks.values():
            self.assertEqual(block.values.shape[0], 1693)
            self.assertFalse(block.values.flags.writeable)
            self.assertEqual(
                block.provenance["experiment_contract_sha256"],
                EXPECTED_PHASE_5_V2_CONTRACT_SHA256,
            )

    def test_v2_loader_has_no_scan_arbitrary_path_or_sealed_api(self) -> None:
        path = SRC_ROOT / "ornament_classifier" / "embeddings_v2.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(attributes & {"glob", "rglob", "iglob"})
        lowered = source.lower()
        self.assertNotIn('"test.csv"', lowered)
        self.assertNotIn("'test.csv'", lowered)
        self.assertNotIn("np.load", source)


if __name__ == "__main__":
    unittest.main()
