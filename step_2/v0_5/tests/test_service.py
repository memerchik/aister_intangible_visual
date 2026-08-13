import base64
import importlib.util
import json
import sys
import tempfile
import threading
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
SERVER_PATH = APP_ROOT / "server.py"
SOURCE_ROOT = APP_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

spec = importlib.util.spec_from_file_location("aister_v0_5_server", SERVER_PATH)
server_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = server_module
spec.loader.exec_module(server_module)


class StubPredictor:
    def predict_bytes(self, image_bytes):
        result = {
            "application_version": "v0.5",
            "top_match": {
                "label": "02_ornek",
                "name": "Örnek",
                "score": 0.8,
            },
            "ranking": [
                {"rank": 1, "label": "02_ornek", "name": "Örnek", "score": 0.8},
                {"rank": 2, "label": "04_petrykivka_painting", "name": "Petrykivka painting", "score": 0.1},
            ],
            "motif_regions": [],
            "calibrated": False,
            "sealed_test_evaluated": False,
        }
        return result, "png"


class BlockingPredictor(StubPredictor):
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()

    def predict_bytes(self, image_bytes):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("blocking predictor timed out")
        return super().predict_bytes(image_bytes)


class AssistedApplicationServiceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.contribution_root = Path(self.temporary.name) / "pending"
        self.service = server_module.ApplicationService(
            StubPredictor(), self.contribution_root
        )
        self.image_bytes = b"small-test-image"
        encoded = base64.b64encode(self.image_bytes).decode("ascii")
        self.payload = {
            "image": "data:image/png;base64," + encoded,
            "file_name": "ornament.png",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def test_prediction_is_memory_only_and_writes_no_files(self):
        result = self.service.predict(self.payload)
        self.assertRegex(result["prediction_id"], r"^pred_[0-9a-f]{32}$")
        self.assertEqual(result["storage"], "memory_only_unless_explicitly_contributed")
        self.assertFalse(self.contribution_root.exists())

    def test_contribution_requires_explicit_consent(self):
        result = self.service.predict(self.payload)
        with self.assertRaisesRegex(ValueError, "consent"):
            self.service.contribute(
                {
                    "prediction_id": result["prediction_id"],
                    "final_label": "02_ornek",
                    "consent": False,
                }
            )
        self.assertFalse(self.contribution_root.exists())

    def test_consented_contribution_is_quarantined_and_not_training_eligible(self):
        result = self.service.predict(self.payload)
        saved = self.service.contribute(
            {
                "prediction_id": result["prediction_id"],
                "final_label": "04_petrykivka_painting",
                "note": "  Human   correction.  ",
                "consent": True,
            }
        )
        self.assertEqual(saved["status"], "pending_expert_review")
        files = sorted(self.contribution_root.iterdir())
        self.assertEqual(len(files), 2)
        metadata_path = next(path for path in files if path.suffix == ".json")
        metadata = json.loads(metadata_path.read_text())
        self.assertEqual(metadata["review_status"], "pending_expert_review")
        self.assertFalse(metadata["training_eligible"])
        self.assertFalse(metadata["automatic_training_permitted"])
        self.assertEqual(len(metadata["image_content_sha256"]), 64)
        self.assertEqual(metadata["rights_review_status"], "pending")
        self.assertEqual(metadata["deduplication_status"], "pending")
        self.assertEqual(metadata["source_object_grouping_status"], "pending")
        self.assertEqual(metadata["expert_label_status"], "pending")
        self.assertTrue(metadata["changed_from_suggestion"])
        self.assertEqual(metadata["note"], "Human correction.")

    def test_health_never_claims_calibration_or_sealed_evaluation(self):
        health = self.service.health()
        self.assertEqual(health["status"], "ready")
        self.assertFalse(health["calibrated"])
        self.assertFalse(health["sealed_test_evaluated"])

    def test_disabled_contributions_do_not_cache_or_write_images(self):
        service = server_module.ApplicationService(
            StubPredictor(), self.contribution_root, contributions_enabled=False
        )
        result = service.predict(self.payload)
        self.assertIsNone(result["prediction_id"])
        self.assertEqual(result["storage"], "none")
        self.assertFalse(service.public_config()["contributions_enabled"])
        with self.assertRaisesRegex(ValueError, "disabled"):
            service.contribute(
                {
                    "prediction_id": "pred_" + "0" * 32,
                    "final_label": "02_ornek",
                    "consent": True,
                }
            )
        self.assertFalse(self.contribution_root.exists())

    def test_only_one_prediction_is_admitted_without_queueing_uploads(self):
        predictor = BlockingPredictor()
        service = server_module.ApplicationService(
            predictor, self.contribution_root, contributions_enabled=False
        )
        completed = []
        first = threading.Thread(target=lambda: completed.append(service.predict(self.payload)))
        first.start()
        self.assertTrue(predictor.started.wait(timeout=2))
        self.assertTrue(service.health()["prediction_busy"])
        self.assertEqual(service.health()["maximum_concurrent_predictions"], 1)
        with self.assertRaisesRegex(server_module.PredictionBusyError, "already"):
            service.predict(self.payload)
        predictor.release.set()
        first.join(timeout=5)
        self.assertFalse(first.is_alive())
        self.assertEqual(len(completed), 1)
        self.assertFalse(service.health()["prediction_busy"])

    def test_prediction_slot_is_released_after_an_error(self):
        class FailingPredictor:
            def predict_bytes(self, image_bytes):
                raise RuntimeError("expected failure")

        service = server_module.ApplicationService(
            FailingPredictor(), self.contribution_root, contributions_enabled=False
        )
        with self.assertRaisesRegex(RuntimeError, "expected failure"):
            service.predict(self.payload)
        self.assertFalse(service.health()["prediction_busy"])


class AssistedInterfaceContractTests(unittest.TestCase):
    def test_interface_contains_required_transparency_and_accessibility_copy(self):
        html = (APP_ROOT / "static/index.html").read_text()
        self.assertIn("ranking score", html)
        self.assertIn("not calibrated probabilities", html)
        self.assertIn("Prediction images are not saved by default", html)
        self.assertIn("Opishnyan/Bubnivka", html)
        self.assertIn("Skip to main content", html)
        self.assertIn('aria-live="polite"', html)

    def test_interface_uses_local_assets_and_no_inline_script(self):
        html = (APP_ROOT / "static/index.html").read_text()
        self.assertIn('href="/app.css"', html)
        self.assertIn('src="/app.js"', html)
        self.assertIn('src="/aister-logo.png"', html)
        self.assertTrue((APP_ROOT / "static/aister-logo.png").is_file())
        self.assertIn("/aister-logo.png", server_module.STATIC_FILES)
        self.assertNotIn("<script>", html)
        self.assertNotIn("fonts.googleapis.com", html)
        javascript = (APP_ROOT / "static/app.js").read_text()
        self.assertNotIn("innerHTML", javascript)

    def test_render_boundary_uses_only_the_standalone_application(self):
        render_yaml = (APP_ROOT.parents[1] / "render.yaml").read_text()
        dockerfile = (APP_ROOT / "Dockerfile").read_text()
        server_source = (APP_ROOT / "server.py").read_text()
        self.assertIn("rootDir: step_2/v0_5", render_yaml)
        self.assertIn('value: "false"', render_yaml)
        self.assertIn("AISTER_ISOLATE_PREDICTIONS", render_yaml)
        self.assertIn('value: "true"', render_yaml)
        self.assertIn("prediction_worker.py", dockerfile)
        self.assertNotIn("development", dockerfile)
        self.assertNotIn("ornament_classifier", server_source)
        self.assertIn("aister_runtime.inference", server_source)


if __name__ == "__main__":
    unittest.main()
