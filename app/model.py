"""
model.py — Retina Guard
EfficientNet-B3 classifier for 5-class Diabetic Retinopathy grading.

APTOS 2019 severity scale
  0 → No DR
  1 → Mild DR
  2 → Moderate DR
  3 → Severe DR
  4 → Proliferative DR
"""

from __future__ import annotations
import os
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
from pathlib import Path

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
NUM_CLASSES   = 5
IMG_SIZE      = 380           # EfficientNet-B3 native size
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR",
]

CLASS_COLOURS = {
    "No DR":              "#22c55e",   # green
    "Mild DR":            "#84cc16",   # lime
    "Moderate DR":        "#eab308",   # yellow
    "Severe DR":          "#f97316",   # orange
    "Proliferative DR":   "#ef4444",   # red
}

SEVERITY_ADVICE = {
    "No DR": (
        "No signs of diabetic retinopathy detected. "
        "Continue routine annual screening."
    ),
    "Mild DR": (
        "Early microaneurysms present. "
        "Optimise HbA1c, blood pressure, and lipid control. "
        "Repeat screening in 9–12 months."
    ),
    "Moderate DR": (
        "Moderate non-proliferative changes detected. "
        "Refer to ophthalmology within 3 months. "
        "Intensify systemic risk-factor management."
    ),
    "Severe DR": (
        "Severe non-proliferative diabetic retinopathy. "
        "Urgent ophthalmology referral within 1 month. "
        "High risk of progression to proliferative stage."
    ),
    "Proliferative DR": (
        "⚠️  Proliferative diabetic retinopathy — sight-threatening. "
        "Immediate ophthalmology referral. "
        "Laser photocoagulation or anti-VEGF treatment required."
    ),
}


# ─────────────────────────────────────────────
# Model definition
# ─────────────────────────────────────────────

def build_model(pretrained: bool = True,
                weights_path: str | None = None) -> nn.Module:
    """
    Build EfficientNet-B3 with a custom 5-class classifier head.

    If `weights_path` is provided and exists, load fine-tuned DR weights.
    Otherwise fall back to ImageNet pretrained weights (demo mode).
    """
    if pretrained and weights_path is None:
        base = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    else:
        base = efficientnet_b3(weights=None)

    # Replace classifier
    in_features = base.classifier[1].in_features
    base.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, NUM_CLASSES),
    )

    # Load fine-tuned weights if available
    if weights_path and Path(weights_path).exists():
        state = torch.load(weights_path, map_location=DEVICE)
        # Handle both plain state-dict and checkpoint dicts
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        base.load_state_dict(state, strict=False)
        print(f"[RetinaGuard] Loaded weights from {weights_path}")
    else:
        print("[RetinaGuard] Running in DEMO mode — ImageNet weights only.")

    base.to(DEVICE)
    base.eval()
    return base


# ─────────────────────────────────────────────
# Inference transform
# ─────────────────────────────────────────────

def get_inference_transform() -> T.Compose:
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────
# Prediction helper
# ─────────────────────────────────────────────

@torch.no_grad()
def predict(model: nn.Module,
            input_tensor: torch.Tensor) -> dict:
    """
    Run a single forward pass and return a prediction dict.

    Parameters
    ----------
    model        : the EfficientNet-B3 model
    input_tensor : (1, 3, H, W) float tensor on DEVICE

    Returns
    -------
    {
      "class_idx"  : int,
      "class_name" : str,
      "confidence" : float,
      "probs"      : list[float],   # softmax probabilities
    }
    """
    logits = model(input_tensor)
    probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    idx    = int(probs.argmax())
    return {
        "class_idx":  idx,
        "class_name": CLASS_NAMES[idx],
        "confidence": float(probs[idx]),
        "probs":      probs.tolist(),
    }


# ─────────────────────────────────────────────
# Singleton loader (cached across Streamlit reruns)
# ─────────────────────────────────────────────

_MODEL_CACHE: nn.Module | None = None

def get_model(weights_path: str | None = None) -> nn.Module:
    """Return the globally cached model, building it if necessary."""
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = build_model(pretrained=True, weights_path=weights_path)
    return _MODEL_CACHE
