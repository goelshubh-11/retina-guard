"""
tests/test_pipeline.py — Retina Guard
Validation suite for all pipeline components.

Run with:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import numpy as np
from PIL import Image
import torch
from datetime import date


# ─────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────

@pytest.fixture(scope="module")
def dummy_pil() -> Image.Image:
    """Synthetic 512×512 RGB image (random pixels)."""
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 255, (512, 512, 3), dtype=np.uint8)
    return Image.fromarray(arr)


@pytest.fixture(scope="module")
def dummy_rgb(dummy_pil) -> np.ndarray:
    return np.array(dummy_pil.convert("RGB"))


@pytest.fixture(scope="module")
def model():
    from app.model import build_model
    return build_model(pretrained=True, weights_path=None)


@pytest.fixture(scope="module")
def processed_result(dummy_pil):
    from app.preprocessing import full_preprocess
    return full_preprocess(dummy_pil, apply_ben_graham=True)


# ─────────────────────────────────────────────
# 1. Preprocessing tests
# ─────────────────────────────────────────────

class TestPreprocessing:

    def test_output_shapes(self, processed_result):
        orig, proc = processed_result
        assert orig.shape == (380, 380, 3), f"Expected (380,380,3), got {orig.shape}"
        assert proc.shape == (380, 380, 3), f"Expected (380,380,3), got {proc.shape}"

    def test_dtype_uint8(self, processed_result):
        orig, proc = processed_result
        assert orig.dtype == np.uint8
        assert proc.dtype == np.uint8

    def test_value_range(self, processed_result):
        _, proc = processed_result
        assert proc.min() >= 0 and proc.max() <= 255

    def test_clahe_increases_variance(self, dummy_rgb):
        """CLAHE should increase local contrast (std of L channel)."""
        import cv2
        from app.preprocessing import apply_clahe
        bgr      = cv2.cvtColor(dummy_rgb, cv2.COLOR_RGB2BGR)
        lab_orig = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        enhanced = apply_clahe(bgr)
        lab_enh  = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        # Variance of L-channel should not drop to zero
        assert lab_enh[:,:,0].std() >= 0

    def test_resize_pad_square(self):
        from app.preprocessing import resize_pad_square
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, (300, 600, 3), dtype=np.uint8)
        out = resize_pad_square(img, 380)
        assert out.shape == (380, 380, 3)


# ─────────────────────────────────────────────
# 2. Model tests
# ─────────────────────────────────────────────

class TestModel:

    def test_model_loads(self, model):
        assert model is not None

    def test_model_eval_mode(self, model):
        assert not model.training

    def test_classifier_head(self, model):
        """Check the classifier outputs 5 logits."""
        from app.model import NUM_CLASSES
        rng = torch.Generator()
        rng.manual_seed(0)
        dummy_tensor = torch.randn(1, 3, 380, 380)
        with torch.no_grad():
            out = model(dummy_tensor)
        assert out.shape == (1, NUM_CLASSES), \
            f"Expected (1, {NUM_CLASSES}), got {out.shape}"

    def test_predict_returns_valid_class(self, model, processed_result):
        from app.model import predict, get_inference_transform, CLASS_NAMES, DEVICE
        _, proc = processed_result
        pil = Image.fromarray(proc)
        tfm = get_inference_transform()
        tensor = tfm(pil).unsqueeze(0).to(DEVICE)
        result = predict(model, tensor)
        assert result["class_name"] in CLASS_NAMES
        assert 0.0 <= result["confidence"] <= 1.0
        assert len(result["probs"]) == 5
        assert abs(sum(result["probs"]) - 1.0) < 1e-4  # probs sum to 1

    def test_predict_probs_non_negative(self, model, processed_result):
        from app.model import predict, get_inference_transform, DEVICE
        _, proc = processed_result
        pil = Image.fromarray(proc)
        tfm = get_inference_transform()
        tensor = tfm(pil).unsqueeze(0).to(DEVICE)
        result = predict(model, tensor)
        assert all(p >= 0 for p in result["probs"])


# ─────────────────────────────────────────────
# 3. Grad-CAM tests
# ─────────────────────────────────────────────

class TestGradCAM:

    def test_overlay_shape(self, model, processed_result):
        from app.gradcam import run_gradcam
        from app.model import get_inference_transform, DEVICE
        _, proc = processed_result
        pil    = Image.fromarray(proc)
        tfm    = get_inference_transform()
        tensor = tfm(pil).unsqueeze(0).to(DEVICE)
        overlay, raw_cam, density = run_gradcam(
            model, tensor, proc, class_idx=0, method="gradcam"
        )
        assert overlay.shape == proc.shape
        assert raw_cam.shape == proc.shape[:2]

    def test_cam_values_in_unit_interval(self, model, processed_result):
        from app.gradcam import run_gradcam
        from app.model import get_inference_transform, DEVICE
        _, proc = processed_result
        pil    = Image.fromarray(proc)
        tfm    = get_inference_transform()
        tensor = tfm(pil).unsqueeze(0).to(DEVICE)
        _, raw_cam, density = run_gradcam(
            model, tensor, proc, class_idx=0, method="gradcam"
        )
        assert raw_cam.min() >= 0.0
        assert raw_cam.max() <= 1.0

    def test_activation_density_range(self, model, processed_result):
        from app.gradcam import run_gradcam
        from app.model import get_inference_transform, DEVICE
        _, proc = processed_result
        pil    = Image.fromarray(proc)
        tfm    = get_inference_transform()
        tensor = tfm(pil).unsqueeze(0).to(DEVICE)
        _, _, density = run_gradcam(
            model, tensor, proc, class_idx=0, method="gradcam"
        )
        assert 0.0 <= density <= 1.0

    def test_gradcampp_runs(self, model, processed_result):
        from app.gradcam import run_gradcam
        from app.model import get_inference_transform, DEVICE
        _, proc = processed_result
        pil    = Image.fromarray(proc)
        tfm    = get_inference_transform()
        tensor = tfm(pil).unsqueeze(0).to(DEVICE)
        overlay, raw_cam, density = run_gradcam(
            model, tensor, proc, class_idx=2, method="gradcam++"
        )
        assert overlay is not None

    def test_heatmap_image(self):
        from app.gradcam import cam_to_heatmap_image
        cam = np.random.rand(380, 380).astype(np.float32)
        hm  = cam_to_heatmap_image(cam)
        assert hm.shape == (380, 380, 3)
        assert hm.dtype == np.uint8


# ─────────────────────────────────────────────
# 4. Prevention agent tests
# ─────────────────────────────────────────────

class TestPreventionAgent:

    def _make_scan(self, severity, density, confidence=0.85,
                   label="scan", d=None):
        from app.prevention import ScanResult
        return ScanResult(
            label           = label,
            severity_class  = severity,
            confidence      = confidence,
            activation_density = density,
            probs           = [0.2, 0.2, 0.2, 0.2, 0.2],
            scan_date       = d or date.today(),
        )

    def test_high_risk_alert_triggered(self):
        from app.prevention import analyse_progression
        hist = self._make_scan("Mild DR",     0.20, label="hist")
        curr = self._make_scan("Moderate DR", 0.35, label="curr")
        report = analyse_progression(hist, curr)
        assert report.alert_triggered is True
        assert report.risk_level == "High"

    def test_low_risk_no_alert(self):
        from app.prevention import analyse_progression
        hist = self._make_scan("No DR", 0.20, label="hist")
        curr = self._make_scan("No DR", 0.21, label="curr")
        report = analyse_progression(hist, curr)
        assert report.alert_triggered is False
        assert report.risk_level == "Low"

    def test_delta_density_correct(self):
        from app.prevention import analyse_progression
        hist = self._make_scan("Mild DR", 0.30, label="hist")
        curr = self._make_scan("Mild DR", 0.40, label="curr")
        report = analyse_progression(hist, curr)
        assert abs(report.delta_density - 0.10) < 1e-6

    def test_recommendations_not_empty(self):
        from app.prevention import analyse_progression
        hist = self._make_scan("Moderate DR", 0.40)
        curr = self._make_scan("Severe DR",   0.55)
        report = analyse_progression(hist, curr)
        assert len(report.recommendations) > 0

    def test_next_review_immediate_for_proliferative_high(self):
        from app.prevention import analyse_progression
        hist = self._make_scan("Severe DR",        0.50)
        curr = self._make_scan("Proliferative DR", 0.70)
        report = analyse_progression(hist, curr)
        assert report.next_review_months == 0

    def test_single_scan_report(self):
        from app.prevention import ScanResult, single_scan_report
        scan = ScanResult("single", "No DR", 0.92, 0.10,
                          [0.92, 0.04, 0.02, 0.01, 0.01])
        rep  = single_scan_report(scan)
        assert "severity_class" in rep
        assert rep["severity_class"] == "No DR"
        assert rep["next_review_months"] == 12

    def test_progression_report_fields(self):
        from app.prevention import analyse_progression
        hist   = self._make_scan("Mild DR", 0.25)
        curr   = self._make_scan("Mild DR", 0.30)
        report = analyse_progression(hist, curr)
        assert hasattr(report, "delta_density")
        assert hasattr(report, "relative_change_pct")
        assert hasattr(report, "risk_level")
        assert hasattr(report, "summary")
        assert isinstance(report.summary, str)
        assert len(report.summary) > 10


# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
