"""
main.py - Retina Guard
Streamlit dashboard for diabetic retinopathy screening.

Run from the project root:
    python -m streamlit run app/main.py
"""

from __future__ import annotations
import hashlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date

import streamlit as st

from app import ui
from app.preprocessing import full_preprocess, pil_from_bytes
from app.model import build_model, rgb_to_tensor, predict, CLASS_NAMES, SEVERITY_ADVICE
from app.gradcam import run_gradcam, cam_to_heatmap_image
from app.prevention import ScanResult, analyse_progression, single_scan_report

st.set_page_config(
    page_title="Retina Guard",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(ui.CSS, unsafe_allow_html=True)

CAM_LABELS = {
    "gradcam": "Grad-CAM",
    "gradcam++": "Grad-CAM++",
    "eigencam": "Eigen-CAM",
}
IMAGE_TYPES = ["jpg", "jpeg", "png", "tif", "tiff"]


# ---------------------------------------------------------------------------
# Cached model + analysis
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading EfficientNet-B3")
def load_model(weights_key: str, _weights_bytes: bytes | None = None):
    path = None
    if _weights_bytes:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pth")
        tmp.write(_weights_bytes)
        tmp.close()
        path = tmp.name
    return build_model(pretrained=True, weights_path=path)


@st.cache_data(show_spinner=False, max_entries=16)
def analyse(file_bytes: bytes, cam_method: str, ben_graham: bool, weights_key: str, _model):
    """Preprocess -> classify -> Grad-CAM. Cached so widget changes do not re-run it."""
    pil = pil_from_bytes(file_bytes)
    original, processed = full_preprocess(pil, apply_ben_graham=ben_graham)
    tensor = rgb_to_tensor(processed)
    pred = predict(_model, tensor)
    overlay, raw_cam, density = run_gradcam(
        _model, tensor, processed, class_idx=pred["class_idx"], method=cam_method
    )
    return {
        "original": original,
        "processed": processed,
        "overlay": overlay,
        "heatmap": cam_to_heatmap_image(raw_cam),
        "pred": pred,
        "density": density,
    }


def show_image(img) -> None:
    """Fit an image to its column across Streamlit versions."""
    try:
        st.image(img, width="stretch")
    except Exception:
        st.image(img, use_container_width=True)


def md(html: str) -> None:
    st.markdown(html, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    md(ui.sidebar_brand())

    md(ui.label("Analysis mode"))
    mode = st.radio(
        "Analysis mode",
        ["Single scan", "Progression comparison"],
        label_visibility="collapsed",
    )

    md(ui.label("Explainability"))
    cam_method = st.selectbox(
        "Explainability method",
        list(CAM_LABELS),
        format_func=lambda k: CAM_LABELS[k],
        label_visibility="collapsed",
    )

    md(ui.label("Preprocessing"))
    ben_graham = st.checkbox(
        "Ben Graham local normalisation",
        value=True,
        help="Subtracts a Gaussian-blurred copy to suppress uneven illumination "
             "and bring out fine vessel detail.",
    )

    md(ui.label("Model weights"))
    weights_file = st.file_uploader(
        "Fine-tuned weights (.pth or .pt)", type=["pth", "pt"], label_visibility="collapsed"
    )

weights_bytes = weights_file.getvalue() if weights_file else None
weights_key = hashlib.sha1(weights_bytes).hexdigest() if weights_bytes else "imagenet"
model = load_model(weights_key, _weights_bytes=weights_bytes)
using_custom = weights_bytes is not None

with st.sidebar:
    md(ui.model_status(using_custom))

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
md(ui.topbar())

if not using_custom:
    md(ui.notice(
        "Baseline model",
        "No fine-tuned weights are loaded, so the classification head is untrained and "
        "predicted grades are illustrative only. Upload a checkpoint trained on APTOS "
        "in the sidebar for meaningful results.",
    ))


# ---------------------------------------------------------------------------
# Single scan
# ---------------------------------------------------------------------------
def render_single() -> None:
    md(ui.section("Study input"))
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 2, 2], gap="medium")
        uploaded = c1.file_uploader("Fundus photograph", type=IMAGE_TYPES)
        patient = c2.text_input("Patient ID", placeholder="e.g. PT-00142")
        scan_date = c3.date_input("Scan date", value=date.today())

    if not uploaded:
        md(ui.section("How it works"))
        md(ui.steps([
            ("01", "Upload", "Provide a fundus photograph in JPEG, PNG or TIFF format."),
            ("02", "Preprocess", "Border crop, CLAHE contrast enhancement and optional "
                                 "Ben Graham normalisation."),
            ("03", "Classify", "EfficientNet-B3 assigns one of five APTOS severity grades."),
            ("04", "Explain", "Grad-CAM highlights the regions that drove the prediction."),
        ]))
        md(ui.disclaimer())
        st.stop()

    with st.spinner("Running analysis"):
        res = analyse(uploaded.getvalue(), cam_method, ben_graham, weights_key, model)

    pred, density = res["pred"], res["density"]
    scan = ScanResult("Current", pred["class_name"], pred["confidence"],
                      density, pred["probs"], scan_date)
    rep = single_scan_report(scan)

    md(ui.record_strip([
        ("Patient", patient or "Not provided"),
        ("Scan date", str(scan_date)),
        ("Image", uploaded.name),
        ("Explainability", CAM_LABELS[cam_method]),
    ]))
    md(ui.cards([
        ui.card("Predicted grade", pred["class_name"], f"Class {pred['class_idx']} of 4",
                ui.SEVERITY_COLOURS.get(pred["class_name"])),
        ui.card("Confidence", f"{pred['confidence'] * 100:.1f}%", "Softmax probability"),
        ui.card("Activation density", f"{density * 100:.1f}%", "Image area above 0.5 activation"),
        ui.card("Suggested follow-up", ui.review_text(rep["next_review_months"]),
                "Screening interval"),
    ]))

    tab_exp, tab_cls, tab_rec, tab_rep = st.tabs(
        ["Explainability", "Classification", "Recommendations", "Report"]
    )

    with tab_exp:
        cols = st.columns(4, gap="medium")
        panels = [
            ("Original", res["original"], "Fig. 1"),
            ("Enhanced", res["processed"], "Fig. 2"),
            (f"{CAM_LABELS[cam_method]} overlay", res["overlay"], "Fig. 3"),
            ("Activation map", res["heatmap"], "Fig. 4"),
        ]
        for col, (name, img, fig) in zip(cols, panels):
            with col:
                show_image(img)
                md(ui.caption(name, fig))
        md(ui.section("Activation density"))
        md(ui.density_bar(density))
        md('<div class="rg-note">Bright areas in the overlay mark where the network '
           'concentrated when choosing the predicted grade. They support, but do not '
           'replace, clinical review of the image.</div>')

    with tab_cls:
        left, right = st.columns([1, 1], gap="large")
        with left:
            md(ui.label("Predicted grade"))
            md(ui.pill(pred["class_name"]))
            md(f'<div class="rg-note">{SEVERITY_ADVICE[pred["class_name"]]}</div>')
        with right:
            md(ui.label("Class probabilities"))
            md(ui.prob_bars(pred["probs"], CLASS_NAMES, pred["class_idx"]))

    with tab_rec:
        md(ui.rec_list(rep["recommendations"]))

    with tab_rep:
        report = ui.text_report(
            "Retina Guard - single scan report",
            [("Patient", patient or "Not provided"), ("Scan date", str(scan_date)),
             ("Image", uploaded.name), ("Explainability", CAM_LABELS[cam_method]),
             ("Weights", "Custom" if using_custom else "Baseline (untrained head)")],
            [("Predicted grade", pred["class_name"]),
             ("Confidence", f"{pred['confidence'] * 100:.1f}%"),
             ("Activation density", f"{density * 100:.1f}%"),
             ("Suggested follow-up", ui.review_text(rep["next_review_months"]))],
            rep["recommendations"],
        )
        st.download_button(
            "Download report (.txt)", report,
            file_name=f"retina_guard_{(patient or 'scan').replace(' ', '_')}_{scan_date}.txt",
            mime="text/plain",
        )
        st.code(report, language=None)

    md(ui.disclaimer())


# ---------------------------------------------------------------------------
# Progression comparison
# ---------------------------------------------------------------------------
def render_progression() -> None:
    md(ui.section("Study input", "Baseline and follow-up scans of the same eye"))
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            md(ui.label("Historical scan"))
            hist_file = st.file_uploader("Historical fundus image", type=IMAGE_TYPES, key="hist")
            h1, h2 = st.columns(2)
            hist_label = h1.text_input("Label", value="Baseline", key="hl")
            hist_date = h2.date_input("Scan date", value=date(2025, 11, 18), key="hd")
    with right:
        with st.container(border=True):
            md(ui.label("Current scan"))
            curr_file = st.file_uploader("Current fundus image", type=IMAGE_TYPES, key="curr")
            c1, c2 = st.columns(2)
            curr_label = c1.text_input("Label", value="Follow-up", key="cl")
            curr_date = c2.date_input("Scan date", value=date.today(), key="cd")

    if not (hist_file and curr_file):
        md('<div class="rg-note">Upload both scans to run the progression analysis.</div>')
        md(ui.section("How progression is measured"))
        md(ui.steps([
            ("01", "Analyse both scans", "Each image is classified and a Grad-CAM map is produced."),
            ("02", "Measure activation", "The share of the image above 0.5 activation is computed."),
            ("03", "Compare", "The relative change in activation density is calculated."),
            ("04", "Alert", "An increase above 10% raises a prevention alert for the next six months."),
        ]))
        md(ui.disclaimer())
        st.stop()

    with st.spinner("Analysing both scans"):
        hist = analyse(hist_file.getvalue(), cam_method, ben_graham, weights_key, model)
        curr = analyse(curr_file.getvalue(), cam_method, ben_graham, weights_key, model)

    hp, cp = hist["pred"], curr["pred"]
    hist_scan = ScanResult(hist_label, hp["class_name"], hp["confidence"],
                           hist["density"], hp["probs"], hist_date)
    curr_scan = ScanResult(curr_label, cp["class_name"], cp["confidence"],
                           curr["density"], cp["probs"], curr_date)
    report = analyse_progression(hist_scan, curr_scan)

    headline = {
        "High": "Prevention alert: high risk of progression",
        "Moderate": "Moderate change in activation density",
        "Low": "No significant progression detected",
    }[report.risk_level]
    md(ui.section("Result"))
    md(ui.banner(report.risk_level, headline, report.summary))
    md(ui.cards([
        ui.card("Historical grade", hp["class_name"], str(hist_date),
                ui.SEVERITY_COLOURS.get(hp["class_name"])),
        ui.card("Current grade", cp["class_name"], str(curr_date),
                ui.SEVERITY_COLOURS.get(cp["class_name"])),
        ui.card("Change in activation", ui.format_pct(report.relative_change_pct),
                "Relative to baseline"),
        ui.card("Risk level", report.risk_level, "Alert threshold: +10%",
                ui.RISK_COLOURS[report.risk_level]),
        ui.card("Suggested follow-up", ui.review_text(report.next_review_months),
                "Screening interval"),
    ]))

    tab_cmp, tab_rec, tab_rep = st.tabs(["Comparison", "Recommendations", "Report"])

    with tab_cmp:
        md(ui.compare_table(
            ["Measure", hist_label, curr_label],
            [
                ["Scan date", str(hist_date), str(curr_date)],
                ["Predicted grade", hp["class_name"], cp["class_name"]],
                ["Confidence", f"{hp['confidence'] * 100:.1f}%", f"{cp['confidence'] * 100:.1f}%"],
                ["Activation density", f"{hist['density'] * 100:.2f}%", f"{curr['density'] * 100:.2f}%"],
            ],
        ))
        md(ui.section("Side-by-side Grad-CAM"))
        lcol, rcol = st.columns(2, gap="large")
        for col, name, res_, p in ((lcol, hist_label, hist, hp), (rcol, curr_label, curr, cp)):
            with col:
                md(ui.label(name))
                a, b = st.columns(2, gap="small")
                with a:
                    show_image(res_["original"])
                    md(ui.caption("Original"))
                with b:
                    show_image(res_["overlay"])
                    md(ui.caption(f"{CAM_LABELS[cam_method]} overlay"))
                md(ui.pill(p["class_name"]))
                md(ui.prob_bars(p["probs"], CLASS_NAMES, p["class_idx"]))

    with tab_rec:
        md(ui.rec_list(report.recommendations))

    with tab_rep:
        text = ui.text_report(
            "Retina Guard - progression report",
            [("Historical scan", f"{hist_label} ({hist_date})"),
             ("Current scan", f"{curr_label} ({curr_date})"),
             ("Explainability", CAM_LABELS[cam_method]),
             ("Weights", "Custom" if using_custom else "Baseline (untrained head)")],
            [("Historical grade", hp["class_name"]), ("Current grade", cp["class_name"]),
             ("Historical activation", f"{hist['density'] * 100:.2f}%"),
             ("Current activation", f"{curr['density'] * 100:.2f}%"),
             ("Relative change", ui.format_pct(report.relative_change_pct)),
             ("Risk level", report.risk_level),
             ("Suggested follow-up", ui.review_text(report.next_review_months))],
            report.recommendations,
            summary=report.summary,
        )
        st.download_button("Download report (.txt)", text,
                           file_name=f"retina_guard_progression_{curr_date}.txt",
                           mime="text/plain")
        st.code(text, language=None)

    md(ui.disclaimer())


if mode == "Single scan":
    render_single()
else:
    render_progression()
