"""
gradcam.py — Retina Guard
Grad-CAM visualisation using pytorch-grad-cam.

Highlights the retinal regions that most influenced the model's decision —
essential for clinical verification and trust.
"""

from __future__ import annotations
import numpy as np
import torch
import cv2
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# ─────────────────────────────────────────────
# Target layer resolver
# ─────────────────────────────────────────────

def get_target_layers(model: torch.nn.Module) -> list:
    """
    Return the last convolutional block of EfficientNet-B3 as the
    Grad-CAM target layer. This is the feature map before global pooling.
    """
    # EfficientNet-B3 feature extractor is model.features
    # Block 7 (index -1) is the final inverted-residual block
    return [model.features[-1]]


# ─────────────────────────────────────────────
# Activation density metric
# ─────────────────────────────────────────────

def compute_activation_density(cam_map: np.ndarray,
                                threshold: float = 0.5) -> float:
    """
    Compute the fraction of the image whose Grad-CAM activation exceeds
    `threshold`. Used by the prevention agent to measure lesion spread.

    Parameters
    ----------
    cam_map   : HxW float32, values in [0, 1]
    threshold : normalised activation floor

    Returns
    -------
    density : float in [0, 1]
    """
    return float((cam_map > threshold).mean())


# ─────────────────────────────────────────────
# Main Grad-CAM runner
# ─────────────────────────────────────────────

def run_gradcam(
    model:        torch.nn.Module,
    input_tensor: torch.Tensor,
    rgb_image:    np.ndarray,
    class_idx:    int | None = None,
    method:       str = "gradcam",
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Generate a Grad-CAM heatmap overlaid on the retinal image.

    Parameters
    ----------
    model        : EfficientNet-B3 (eval mode)
    input_tensor : (1, 3, H, W) normalised tensor on correct device
    rgb_image    : HxWx3 uint8, the preprocessed retinal image
    class_idx    : If None, use the predicted class (most useful clinically)
    method       : "gradcam" | "gradcam++" | "eigencam"

    Returns
    -------
    heatmap_overlay  : HxWx3 uint8 — colourised Grad-CAM on image
    raw_cam          : HxW float32 in [0,1] — raw activation map
    activation_density : float  — fraction of pixels above 0.5 activation
    """
    target_layers = get_target_layers(model)
    targets       = [ClassifierOutputTarget(class_idx)] if class_idx is not None else None

    # Choose CAM variant
    cam_cls = {
        "gradcam":   GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam":  EigenCAM,
    }.get(method, GradCAM)

    with cam_cls(model=model, target_layers=target_layers) as cam:
        raw_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # HxW float32

    # Resize cam to match display image
    h, w      = rgb_image.shape[:2]
    raw_cam_r = cv2.resize(raw_cam, (w, h), interpolation=cv2.INTER_LINEAR)
    raw_cam_r = np.clip(raw_cam_r, 0, 1)

    # Float RGB for overlay function
    rgb_float = rgb_image.astype(np.float32) / 255.0

    heatmap_overlay = show_cam_on_image(
        rgb_float, raw_cam_r, use_rgb=True, colormap=cv2.COLORMAP_INFERNO
    )

    density = compute_activation_density(raw_cam_r, threshold=0.5)

    return heatmap_overlay, raw_cam_r, density


# ─────────────────────────────────────────────
# Standalone heatmap (no overlay) — for side-panel display
# ─────────────────────────────────────────────

def cam_to_heatmap_image(cam: np.ndarray) -> np.ndarray:
    """Convert a raw HxW float cam map to a colourised HxWx3 uint8 heatmap."""
    cam_uint8 = (cam * 255).astype(np.uint8)
    heatmap   = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
