from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import numpy as np


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
SRC_ROOT = STEP_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ornament_classifier.embeddings_v4 import (  # noqa: E402
    EXPECTED_CONTRACT_SHA256,
    MotifCacheError,
    load_motif_embeddings_v4,
)


class Phase5V4CacheLoaderTests(unittest.TestCase):
    def test_all_descriptors_and_proposals_load_with_frozen_provenance(self) -> None:
        cache = load_motif_embeddings_v4()
        self.assertEqual(
            set(cache.embeddings),
            {
                "uniform_mean",
                "texture_weighted_mean",
                "texture_top2_mean",
                "texture_dispersion",
            },
        )
        self.assertTrue(
            all(value.shape == (1693, 384) for value in cache.embeddings.values())
        )
        self.assertEqual(cache.tile_embeddings.shape, (1693, 6, 384))
        self.assertEqual(cache.proposal_boxes.shape, (1693, 6, 4))
        self.assertEqual(cache.proposal_scores.shape, (1693, 6))
        self.assertEqual(
            cache.provenance["experiment_contract_sha256"],
            EXPECTED_CONTRACT_SHA256,
        )
        self.assertEqual(
            cache.provenance["cache_sha256"],
            "97ee78769b6b4349b58e9b2cb27c9f4b58748ecf7cd9fbcfcc554c2fa122db40",
        )
        self.assertFalse(
            cache.provenance["metadata"]["sealed_test_evaluated"]
        )

    def test_every_embedding_and_tile_is_unit_normalized(self) -> None:
        cache = load_motif_embeddings_v4()
        for values in cache.embeddings.values():
            np.testing.assert_allclose(
                np.linalg.norm(values, axis=1), 1.0, rtol=5e-4, atol=5e-4
            )
        np.testing.assert_allclose(
            np.linalg.norm(cache.tile_embeddings, axis=2),
            1.0,
            rtol=5e-4,
            atol=5e-4,
        )

    def test_loader_has_no_arbitrary_cache_or_sealed_api(self) -> None:
        source = (
            SRC_ROOT / "ornament_classifier" / "embeddings_v4.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("rglob(", source)
        self.assertNotIn("glob(", source)
        self.assertNotIn('"test.csv"', source)
        self.assertNotIn("'test.csv'", source)
        with self.assertRaises(TypeError):
            load_motif_embeddings_v4(path=Path("arbitrary.npz"))  # type: ignore[call-arg]


if __name__ == "__main__":
    unittest.main()
