"""
preprocessing.py — Retina Guard
Retinal image preprocessing pipeline using CLAHE + Ben Graham's method.
"""

import cv2
import numpy as np
from PIL import Image
import io


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
IMG_SIZE = 380          # EfficientNet-B3 native resolution
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID  = (8, 8)


# ─────────────────────────────────────────────
# Core helpers
# ─────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    """Ensure image is 3-channel BGR (OpenCV format)."""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    return img


def apply_clahe(img_bgr: np.ndarray,
                clip_limit: float = CLAHE_CLIP_LIMIT,
                tile_grid: tuple  = CLAHE_TILE_GRID) -> np.ndarray:
    """
    Apply CLAHE per channel in LAB space — preserves hue, lifts contrast.
    Returns BGR image.
    """
    lab   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l     = clahe.apply(l)
    lab   = cv2.merge([l, a, b])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def ben_graham_preprocess(img_bgr: np.ndarray,
                           sigmaX: int = 10) -> np.ndarray:
    """
    Ben Graham's preprocessing: subtract a Gaussian-blurred version of the
    image to remove low-frequency illumination artefacts and reveal fine
    vascular details (microaneurysms, haemorrhages).
    """
    blurred = cv2.GaussianBlur(img_bgr, (0, 0), sigmaX)
    result  = cv2.addWeighted(img_bgr, 4, blurred, -4, 128)
    return result


def crop_fundus_circle(img_bgr: np.ndarray,
                        tol: int = 7) -> np.ndarray:
    """
    Auto-crop the black border around the fundus image using a pixel
    brightness threshold so the model sees only the retinal disc.
    """
    gray   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    mask   = gray > tol
    rows   = np.any(mask, axis=1)
    cols   = np.any(mask, axis=0)
    r0, r1 = np.where(rows)[0][[0, -1]]
    c0, c1 = np.where(cols)[0][[0, -1]]
    return img_bgr[r0:r1+1, c0:c1+1]


def resize_pad_square(img_bgr: np.ndarray,
                       size: int = IMG_SIZE) -> np.ndarray:
    """Letterbox-resize to a square canvas preserving aspect ratio."""
    h, w   = img_bgr.shape[:2]
    scale  = size / max(h, w)
    nh, nw = int(h * scale), int(w * scale)
    resized = cv2.resize(img_bgr, (nw, nh), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y0     = (size - nh) // 2
    x0     = (size - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = resized
    return canvas


# ─────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────

def full_preprocess(pil_image: Image.Image,
                    apply_ben_graham: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline.

    Parameters
    ----------
    pil_image       : PIL Image (RGB)
    apply_ben_graham: Whether to apply Ben Graham's local normalisation

    Returns
    -------
    (original_rgb, processed_rgb)  — both HxWx3 uint8, size IMG_SIZE
    """
    img_rgb = np.array(pil_image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # 1 — Crop black fundus border
    try:
        img_bgr = crop_fundus_circle(img_bgr)
    except (ValueError, IndexError):
        pass   # image has no dark border — skip

    # 2 — CLAHE contrast enhancement
    img_bgr = apply_clahe(img_bgr)

    # 3 — Optional Ben Graham local normalisation
    if apply_ben_graham:
        img_bgr = ben_graham_preprocess(img_bgr)

    # 4 — Letterbox resize to model input size
    img_bgr = resize_pad_square(img_bgr, IMG_SIZE)

    # Convert back to RGB for display / torch
    processed_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Also return a "clean" original (only cropped + resized) for side-by-side display
    orig_bgr  = cv2.cvtColor(np.array(pil_image.convert("RGB")),
                              cv2.COLOR_RGB2BGR)
    try:
        orig_bgr = crop_fundus_circle(orig_bgr)
    except (ValueError, IndexError):
        pass
    orig_bgr     = resize_pad_square(orig_bgr, IMG_SIZE)
    original_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    return original_rgb, processed_rgb


def pil_from_bytes(file_bytes: bytes) -> Image.Image:
    """Convert raw file bytes (from st.file_uploader) to PIL Image."""
    return Image.open(io.BytesIO(file_bytes))
