"""
main.py — Retina Guard
Full Streamlit dashboard for Diabetic Retinopathy screening.

Run:
    streamlit run app/main.py
"""

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import io
import numpy as np
import torch
from PIL import Image
from datetime import date
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Internal modules ───────────────────────────────────────────────────────────
from app.preprocessing import full_preprocess, pil_from_bytes
from app.model          import (
    get_model, get_inference_transform, predict,
    CLASS_NAMES, CLASS_COLOURS, SEVERITY_ADVICE, DEVICE
)
from app.gradcam        import run_gradcam, cam_to_heatmap_image
from app.prevention     import (
    ScanResult, analyse_progression, single_scan_report
)


# ══════════════════════════════════════════════════════════════════════════════
# Page config & CSS
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title  = "Retina Guard",
    page_icon   = "🔬",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

st.markdown("""
<style>
/* ── Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1b2a 50%, #0a1628 100%);
    color: #e2e8f0;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(13, 27, 42, 0.95);
    border-right: 1px solid rgba(56, 189, 248, 0.15);
}

/* ── Header banner ── */
.rg-header {
    background: linear-gradient(90deg,
        rgba(56, 189, 248, 0.08) 0%,
        rgba(99, 102, 241, 0.08) 50%,
        rgba(236, 72, 153, 0.05) 100%);
    border: 1px solid rgba(56, 189, 248, 0.2);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
}
.rg-header::before {
    content: '';
    position: absolute;
    inset: 0;
    background: radial-gradient(circle at 80% 50%,
        rgba(56,189,248,0.06) 0%, transparent 60%);
    pointer-events: none;
}
.rg-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.rg-header p {
    color: #94a3b8;
    margin: 0.4rem 0 0;
    font-size: 1rem;
}

/* ── Cards ── */
.rg-card {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(56, 189, 248, 0.15);
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
}

/* ── Metric pill ── */
.rg-metric {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 999px;
    padding: 0.35rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
    color: #38bdf8;
}

/* ── Alert box ── */
.rg-alert-high {
    background: rgba(239, 68, 68, 0.1);
    border: 1px solid rgba(239, 68, 68, 0.5);
    border-left: 4px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.rg-alert-moderate {
    background: rgba(234, 179, 8, 0.08);
    border: 1px solid rgba(234, 179, 8, 0.4);
    border-left: 4px solid #eab308;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}
.rg-alert-low {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.35);
    border-left: 4px solid #22c55e;
    border-radius: 10px;
    padding: 1rem 1.25rem;
}

/* ── Probability bar ── */
.prob-bar-wrap { margin: 0.3rem 0; }
.prob-label    { font-size: 0.8rem; color: #94a3b8; margin-bottom: 2px; }

/* ── Section headings ── */
.rg-section-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: #38bdf8;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin: 1.5rem 0 0.75rem;
    border-bottom: 1px solid rgba(56,189,248,0.2);
    padding-bottom: 0.4rem;
}

/* ── Image captions ── */
.img-caption {
    text-align: center;
    font-size: 0.78rem;
    color: #64748b;
    margin-top: 0.25rem;
    font-style: italic;
}

/* Hide streamlit branding */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbar"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Loading EfficientNet-B3 model…")
def load_model():
    weights = st.session_state.get("weights_path")
    return get_model(weights_path=weights)


def tensor_from_rgb(rgb: np.ndarray) -> torch.Tensor:
    """Convert HxWx3 uint8 numpy array to (1,3,H,W) tensor on DEVICE."""
    pil  = Image.fromarray(rgb)
    tfm  = get_inference_transform()
    return tfm(pil).unsqueeze(0).to(DEVICE)


def run_full_pipeline(pil_image: Image.Image,
                       model,
                       cam_method: str,
                       apply_bg: bool) -> dict:
    """Run preprocess → predict → grad-cam. Return all artefacts."""
    original_rgb, processed_rgb = full_preprocess(pil_image, apply_ben_graham=apply_bg)
    tensor   = tensor_from_rgb(processed_rgb)
    pred     = predict(model, tensor)
    overlay, raw_cam, density = run_gradcam(
        model, tensor, processed_rgb,
        class_idx  = pred["class_idx"],
        method     = cam_method,
    )
    heatmap  = cam_to_heatmap_image(raw_cam)

    return {
        "original_rgb":  original_rgb,
        "processed_rgb": processed_rgb,
        "overlay":       overlay,
        "heatmap":       heatmap,
        "raw_cam":       raw_cam,
        "pred":          pred,
        "density":       density,
    }


def render_probability_bars(probs: list[float], class_names: list[str]):
    """Render styled horizontal probability bars."""
    fig, ax = plt.subplots(figsize=(5, 2.5))
    fig.patch.set_facecolor("#0a0f1e")
    ax.set_facecolor("#0a0f1e")

    colours = [
        "#22c55e", "#84cc16", "#eab308", "#f97316", "#ef4444"
    ]
    bars = ax.barh(class_names, probs, color=colours, height=0.55, zorder=3)
    ax.set_xlim(0, 1)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.spines[:].set_visible(False)
    ax.xaxis.set_visible(False)
    ax.set_facecolor("#0a0f1e")
    for spine in ax.spines.values():
        spine.set_color("none")
    ax.tick_params(axis="y", colors="#cbd5e1")

    # Value labels
    for bar, p in zip(bars, probs):
        ax.text(
            min(p + 0.02, 0.98), bar.get_y() + bar.get_height()/2,
            f"{p*100:.1f}%",
            va="center", ha="left", fontsize=7.5, color="#e2e8f0",
        )

    plt.tight_layout(pad=0.4)
    return fig


def severity_badge(class_name: str) -> str:
    colour = CLASS_COLOURS.get(class_name, "#94a3b8")
    return (
        f'<span style="background:{colour}22; color:{colour}; '
        f'border:1px solid {colour}55; border-radius:999px; '
        f'padding:0.2rem 0.85rem; font-size:0.9rem; font-weight:600;">'
        f'{class_name}</span>'
    )


def alert_box(risk: str, message: str):
    cls = {
        "High":     "rg-alert-high",
        "Moderate": "rg-alert-moderate",
        "Low":      "rg-alert-low",
    }.get(risk, "rg-alert-low")
    icons = {"High": "🔴", "Moderate": "🟡", "Low": "🟢"}
    st.markdown(
        f'<div class="{cls}">'
        f'<strong>{icons.get(risk,"")} {risk} Risk</strong><br>'
        f'{message}</div>',
        unsafe_allow_html=True
    )


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        "<h2 style='color:#38bdf8; margin-bottom:0'>⚙️ Settings</h2>",
        unsafe_allow_html=True
    )
    st.markdown("---")

    mode = st.radio(
        "Analysis mode",
        ["Single Scan", "Progression Comparison"],
        help="Single scan: analyse one retinal image.\n"
             "Progression: compare a historical vs. current scan."
    )

    st.markdown("**Grad-CAM method**")
    cam_method = st.selectbox(
        label=" ",
        options=["gradcam", "gradcam++", "eigencam"],
        format_func=lambda x: {
            "gradcam": "Grad-CAM (standard)",
            "gradcam++": "Grad-CAM++ (sharper)",
            "eigencam": "Eigen-CAM (no-backprop)",
        }[x],
        label_visibility="collapsed",
    )

    apply_bg = st.checkbox("Ben Graham preprocessing", value=True,
                            help="Subtract Gaussian blur to reveal fine vascular details.")

    st.markdown("---")
    st.markdown("**Custom model weights** (optional)")
    weights_file = st.file_uploader("Upload .pth / .pt weights", type=["pth","pt"])
    weights_path = None
    if weights_file:
        import tempfile, pathlib
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp.write(weights_file.read())
        tmp.close()
        weights_path = tmp.name
        st.session_state["weights_path"] = weights_path
        st.success("Weights loaded — reload model ↓")
        if st.button("🔄 Reload model"):
            st.cache_resource.clear()
            st.rerun()

    st.markdown("---")
    st.markdown(
        "<p style='font-size:0.75rem; color:#475569;'>"
        "Retina Guard v1.0 · EfficientNet-B3 · Grad-CAM<br>"
        "For research / educational use only.<br>"
        "Not a medical device.</p>",
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Header
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<div class="rg-header">
  <h1>🔬 Retina Guard</h1>
  <p>AI-Powered Diabetic Retinopathy Detection & Progression Monitoring</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Load model once
# ══════════════════════════════════════════════════════════════════════════════

model = load_model()


# ══════════════════════════════════════════════════════════════════════════════
# ── MODE 1: Single Scan ────────────────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

if mode == "Single Scan":
    col_up, col_info = st.columns([1, 2], gap="large")

    with col_up:
        st.markdown('<div class="rg-section-title">📤 Upload Retinal Image</div>',
                    unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Fundus photograph (JPEG / PNG / TIFF)",
            type=["jpg", "jpeg", "png", "tiff", "tif"],
        )
        patient_name = st.text_input("Patient name / ID", placeholder="e.g. PT-00142")
        scan_date    = st.date_input("Scan date", value=date.today())

    if not uploaded:
        with col_info:
            st.info("👈  Upload a fundus photograph to begin analysis.")
        st.stop()

    # ── Run pipeline ───────────────────────────────────────────────────────
    pil_img = pil_from_bytes(uploaded.read())

    with st.spinner("Running detection pipeline…"):
        result = run_full_pipeline(pil_img, model, cam_method, apply_bg)

    pred    = result["pred"]
    density = result["density"]

    # ── Header metrics ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Diagnosis", pred["class_name"])
    with m2:
        st.metric("Confidence", f"{pred['confidence']*100:.1f} %")
    with m3:
        st.metric("Lesion Density", f"{density*100:.2f} %")
    with m4:
        review = single_scan_report(
            ScanResult("current", pred["class_name"], pred["confidence"],
                       density, pred["probs"], scan_date)
        )["next_review_months"]
        st.metric("Next Review", f"{review} mo" if review > 0 else "Immediate")

    st.markdown("---")

    # ── Image panels ──────────────────────────────────────────────────────
    st.markdown('<div class="rg-section-title">🖼️ Visualisation</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.image(result["original_rgb"], use_container_width=True)
        st.markdown('<p class="img-caption">Original</p>', unsafe_allow_html=True)
    with c2:
        st.image(result["processed_rgb"], use_container_width=True)
        st.markdown('<p class="img-caption">CLAHE Preprocessed</p>', unsafe_allow_html=True)
    with c3:
        st.image(result["overlay"], use_container_width=True)
        st.markdown(f'<p class="img-caption">Grad-CAM Overlay ({cam_method})</p>',
                    unsafe_allow_html=True)
    with c4:
        st.image(result["heatmap"], use_container_width=True)
        st.markdown('<p class="img-caption">Activation Heatmap</p>', unsafe_allow_html=True)

    # ── Diagnosis & probs ─────────────────────────────────────────────────
    st.markdown('<div class="rg-section-title">📊 Classification Result</div>',
                unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1], gap="large")
    with r1:
        st.markdown(
            f"**Severity:** {severity_badge(pred['class_name'])}  "
            f"&nbsp;&nbsp; Confidence: **{pred['confidence']*100:.1f} %**",
            unsafe_allow_html=True,
        )
        st.markdown("")
        advice = SEVERITY_ADVICE[pred["class_name"]]
        st.info(advice)

    with r2:
        st.markdown("**Class probabilities**")
        fig = render_probability_bars(pred["probs"], CLASS_NAMES)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # ── Clinical recommendations ──────────────────────────────────────────
    st.markdown('<div class="rg-section-title">🩺 Clinical Recommendations</div>',
                unsafe_allow_html=True)

    recs = single_scan_report(
        ScanResult("current", pred["class_name"], pred["confidence"],
                   density, pred["probs"], scan_date)
    )["recommendations"]

    for rec in recs:
        st.markdown(f"• {rec}")

    # ── Colour-coded Grad-CAM bar ─────────────────────────────────────────
    st.markdown('<div class="rg-section-title">🌡️ Activation Density Legend</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<span class="rg-metric">Lesion coverage: {density*100:.2f} % of field</span>',
        unsafe_allow_html=True,
    )
    fig2, ax2 = plt.subplots(figsize=(6, 0.45))
    fig2.patch.set_facecolor("#0a0f1e")
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax2.imshow(gradient, aspect="auto", cmap="inferno")
    ax2.axvline(x=density*255, color="white", linewidth=2.5, linestyle="--")
    ax2.set_axis_off()
    ax2.set_title("Low ◄─── Activation ───► High",
                  fontsize=7.5, color="#94a3b8", pad=2)
    plt.tight_layout(pad=0)
    st.pyplot(fig2, use_container_width=True)
    plt.close(fig2)


# ══════════════════════════════════════════════════════════════════════════════
# ── MODE 2: Progression Comparison ────────────────────────────────────────
# ══════════════════════════════════════════════════════════════════════════════

else:
    st.markdown(
        '<div class="rg-section-title">📤 Upload Scans for Progression Analysis</div>',
        unsafe_allow_html=True,
    )
    up_col1, up_col2 = st.columns(2, gap="large")

    with up_col1:
        st.markdown("**Historical scan (baseline)**")
        hist_file = st.file_uploader("Historical fundus image",
                                      type=["jpg","jpeg","png","tiff"],
                                      key="hist")
        hist_date = st.date_input("Historical scan date",
                                   value=date(2024, 1, 1), key="hd")
        hist_label = st.text_input("Label", value="Baseline", key="hl")

    with up_col2:
        st.markdown("**Current scan**")
        curr_file = st.file_uploader("Current fundus image",
                                      type=["jpg","jpeg","png","tiff"],
                                      key="curr")
        curr_date = st.date_input("Current scan date",
                                   value=date.today(), key="cd")
        curr_label = st.text_input("Label", value="Current", key="cl")

    if not hist_file or not curr_file:
        st.info("👆  Upload both scans to enable progression analysis.")
        st.stop()

    # ── Run both pipelines ─────────────────────────────────────────────────
    hist_pil = pil_from_bytes(hist_file.read())
    curr_pil = pil_from_bytes(curr_file.read())

    with st.spinner("Analysing both scans…"):
        hist_res = run_full_pipeline(hist_pil, model, cam_method, apply_bg)
        curr_res = run_full_pipeline(curr_pil, model, cam_method, apply_bg)

    hist_pred = hist_res["pred"]
    curr_pred = curr_res["pred"]

    # Build ScanResult objects
    hist_scan = ScanResult(
        hist_label, hist_pred["class_name"], hist_pred["confidence"],
        hist_res["density"], hist_pred["probs"], hist_date,
    )
    curr_scan = ScanResult(
        curr_label, curr_pred["class_name"], curr_pred["confidence"],
        curr_res["density"], curr_pred["probs"], curr_date,
    )

    report = analyse_progression(hist_scan, curr_scan)

    # ── Top metrics ────────────────────────────────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric("Historical", hist_pred["class_name"])
    with m2:
        st.metric("Current",    curr_pred["class_name"])
    with m3:
        delta_str = f"{report.relative_change_pct:+.1f} %"
        st.metric("Δ Lesion density", delta_str,
                  delta=delta_str if report.delta_density != 0 else None)
    with m4:
        st.metric("Risk level", report.risk_level)
    with m5:
        rv = report.next_review_months
        st.metric("Next review", f"{rv} mo" if rv > 0 else "⚠️ Immediate")

    st.markdown("---")

    # ── Alert ─────────────────────────────────────────────────────────────
    alert_box(report.risk_level, report.summary)

    # ── Side-by-side Grad-CAM ──────────────────────────────────────────────
    st.markdown('<div class="rg-section-title">🖼️ Side-by-Side Grad-CAM Comparison</div>',
                unsafe_allow_html=True)

    lc, rc = st.columns(2, gap="large")
    with lc:
        st.markdown(f"#### {hist_label} · {hist_date}")
        hc1, hc2 = st.columns(2)
        with hc1:
            st.image(hist_res["original_rgb"], use_container_width=True)
            st.markdown('<p class="img-caption">Original</p>', unsafe_allow_html=True)
        with hc2:
            st.image(hist_res["overlay"], use_container_width=True)
            st.markdown('<p class="img-caption">Grad-CAM</p>', unsafe_allow_html=True)
        st.markdown(
            f"Severity: {severity_badge(hist_pred['class_name'])}  "
            f"  Confidence: **{hist_pred['confidence']*100:.1f} %**  "
            f"  Density: **{hist_res['density']*100:.2f} %**",
            unsafe_allow_html=True,
        )
        fig_h = render_probability_bars(hist_pred["probs"], CLASS_NAMES)
        st.pyplot(fig_h, use_container_width=True)
        plt.close(fig_h)

    with rc:
        st.markdown(f"#### {curr_label} · {curr_date}")
        cc1, cc2 = st.columns(2)
        with cc1:
            st.image(curr_res["original_rgb"], use_container_width=True)
            st.markdown('<p class="img-caption">Original</p>', unsafe_allow_html=True)
        with cc2:
            st.image(curr_res["overlay"], use_container_width=True)
            st.markdown('<p class="img-caption">Grad-CAM</p>', unsafe_allow_html=True)
        st.markdown(
            f"Severity: {severity_badge(curr_pred['class_name'])}  "
            f"  Confidence: **{curr_pred['confidence']*100:.1f} %**  "
            f"  Density: **{curr_res['density']*100:.2f} %**",
            unsafe_allow_html=True,
        )
        fig_c = render_probability_bars(curr_pred["probs"], CLASS_NAMES)
        st.pyplot(fig_c, use_container_width=True)
        plt.close(fig_c)

    # ── Activation density comparison bar ──────────────────────────────────
    st.markdown('<div class="rg-section-title">📈 Lesion Density Comparison</div>',
                unsafe_allow_html=True)

    fig3, ax3 = plt.subplots(figsize=(7, 1.8))
    fig3.patch.set_facecolor("#0a0f1e")
    ax3.set_facecolor("#0a0f1e")

    labels = [hist_label, curr_label]
    values = [hist_res["density"]*100, curr_res["density"]*100]
    colours = ["#38bdf8", "#f472b6"]
    bars3 = ax3.barh(labels, values, color=colours, height=0.5)
    ax3.set_xlim(0, max(max(values)*1.2, 5))
    ax3.spines[:].set_visible(False)
    ax3.tick_params(colors="#cbd5e1")
    ax3.xaxis.label.set_color("#94a3b8")
    ax3.set_xlabel("Activation density (%)", color="#94a3b8", fontsize=8)
    ax3.axvline(x=values[0], color="#38bdf8", linestyle="--", linewidth=1, alpha=0.5)

    for bar, v in zip(bars3, values):
        ax3.text(v + 0.3, bar.get_y() + bar.get_height()/2,
                 f"{v:.2f} %", va="center", color="#e2e8f0", fontsize=9)

    plt.tight_layout(pad=0.4)
    st.pyplot(fig3, use_container_width=True)
    plt.close(fig3)

    # ── Recommendations ────────────────────────────────────────────────────
    st.markdown('<div class="rg-section-title">🩺 Prevention Recommendations</div>',
                unsafe_allow_html=True)

    for rec in report.recommendations:
        st.markdown(f"• {rec}")
