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

from ornament_classifier.embeddings_v3 import (  # noqa: E402
    EXPECTED_PHASE_5_V3_CONTRACT_SHA256,
    load_allowlisted_embeddings_v3,
)


class Phase5V3CacheLoaderTests(unittest.TestCase):
    def test_contract_is_exactly_pinned_before_cache_access(self) -> None:
        path = (
            STEP_ROOT
            / "phases"
            / "phase_05_source_robustness"
            / "experiment_contract_v3.json"
        )
        self.assertEqual(
            hashlib.sha256(path.read_bytes()).hexdigest(),
            EXPECTED_PHASE_5_V3_CONTRACT_SHA256,
        )

    def test_all_nine_v3_blocks_load_with_frozen_provenance(self) -> None:
        requests = {
            "center_cls": ("dinov3_center", "embedding__cls"),
            "letter_cls": ("dinov3_letterbox", "embedding__cls"),
            "letter_concat": ("dinov3_letterbox", "embedding__cls_patch_concat"),
            "global_cls": ("dinov3_global", "embedding__cls"),
            "global_concat": ("dinov3_global", "embedding__cls_patch_concat"),
            "dinov2": ("dinov2_center", "embedding__model_pool"),
            "convnext": ("convnext_center", "embedding__model_pool"),
            "clip": ("clip_center", "embedding__model_pool"),
            "mobilenet": ("mobilenet_center", "embedding__model_pool"),
        }
        blocks = load_allowlisted_embeddings_v3(requests)
        self.assertEqual(set(blocks), set(requests))
        for block in blocks.values():
            self.assertEqual(block.values.shape[0], 1693)
            self.assertFalse(block.values.flags.writeable)
            self.assertEqual(
                block.provenance["experiment_contract_sha256"],
                EXPECTED_PHASE_5_V3_CONTRACT_SHA256,
            )

    def test_loader_has_no_scan_arbitrary_path_or_sealed_api(self) -> None:
        path = SRC_ROOT / "ornament_classifier" / "embeddings_v3.py"
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        attributes = {
            node.func.attr
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        self.assertFalse(attributes & {"glob", "rglob", "iglob"})
        self.assertNotIn("test.csv", source.lower())
        self.assertNotIn("np.load", source)


if __name__ == "__main__":
    unittest.main()
