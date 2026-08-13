from __future__ import annotations

import hashlib
import json
import re
import unittest
from pathlib import Path

from relocation_support import assert_recorded_file


STEP_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = STEP_ROOT.parents[1]
CONTRACT_PATH = (
    STEP_ROOT
    / "phases"
    / "phase_05_source_robustness"
    / "motif_extraction_contract_v4.json"
)
EXTRACTOR_PATH = STEP_ROOT / "scripts" / "run_phase5_v4_motif_extraction.py"
MOTIF_PATH = STEP_ROOT / "src" / "ornament_classifier" / "motif.py"
EXPECTED_CONTRACT_SHA256 = (
    "19df4814d2d915c461f267eed89282e5474abb1ac0d37fc50d9dbf25be7c33ff"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class Phase5V4ExtractionContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        cls.source = EXTRACTOR_PATH.read_text(encoding="utf-8")

    def test_contract_is_frozen_before_extraction_and_exact(self) -> None:
        self.assertEqual(sha256_file(CONTRACT_PATH), EXPECTED_CONTRACT_SHA256)
        self.assertEqual(self.contract["iteration"], "v4")
        self.assertEqual(self.contract["status"], "frozen_before_extraction")
        self.assertTrue(self.contract["scope"]["development_only"])
        self.assertFalse(self.contract["scope"]["labels_permitted_for_extraction"])
        self.assertFalse(self.contract["scope"]["source_ids_permitted_for_extraction"])
        self.assertEqual(self.contract["scope"]["sealed_test_access"], "forbidden")

    def test_inputs_policy_encoder_and_output_are_bounded(self) -> None:
        inputs = self.contract["input_contract"]
        self.assertEqual(inputs["development_image_count"], 1693)
        for key, hash_key in (
            ("development_csv", "development_csv_sha256"),
            ("split_audit", "split_audit_sha256"),
            ("reference_center_cache", "reference_center_cache_sha256"),
            ("motif_module", "motif_module_sha256"),
        ):
            assert_recorded_file(self, inputs[key], inputs[hash_key])
        policy = self.contract["proposal_policy"]
        self.assertEqual(policy["scales"], [0.45, 0.65])
        self.assertEqual(policy["positions"], [0.0, 0.5, 1.0])
        self.assertEqual(policy["proposal_count"], 6)
        self.assertEqual(policy["maximum_iou"], 0.6)
        self.assertTrue(policy["label_free"])
        self.assertEqual(self.contract["encoder"]["weight_sha256"], "4610ad75edef83e75afdebf162d148dc628045ea6cbb83d67d4708c709c4f91d")
        required = self.contract["output_contract"]["required_arrays"]
        self.assertEqual(len(required), 14)
        self.assertEqual(len(required), len(set(required)))

    def test_extractor_pins_contract_and_contains_no_sealed_path(self) -> None:
        match = re.search(
            r'EXPECTED_EXTRACTION_CONTRACT_SHA256\s*=\s*\(\s*"([0-9a-f]{64})"',
            self.source,
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.group(1), EXPECTED_CONTRACT_SHA256)
        self.assertNotIn('"test.csv"', self.source)
        self.assertNotIn("'test.csv'", self.source)
        self.assertIn("torch.use_deterministic_algorithms(True", self.source)
        self.assertIn("write_deterministic_npz", self.source)
        self.assertIn("convert_dinov3_model", self.source)
        self.assertIn("validate_conversion", self.source)

    def test_motif_module_is_the_frozen_label_free_implementation(self) -> None:
        self.assertEqual(sha256_file(MOTIF_PATH), self.contract["input_contract"]["motif_module_sha256"])
        source = MOTIF_PATH.read_text(encoding="utf-8")
        for forbidden in ("ornament_label", "source_atomic", "cv_fold", "test.csv"):
            self.assertNotIn(forbidden, source)


if __name__ == "__main__":
    unittest.main()
