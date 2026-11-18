"""
ui.py - Retina Guard
Presentation layer: the stylesheet and small HTML component builders used by
the Streamlit dashboard. Everything here is pure Python (no Streamlit, no
PyTorch) so it can be unit-tested on its own.

Note: HTML snippets are deliberately built without indentation or blank
lines, because Markdown treats indented text as a code block.
"""

from __future__ import annotations
from html import escape

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
SEVERITY_COLOURS = {
    "No DR":            "#2fbf8a",
    "Mild DR":          "#9bc53d",
    "Moderate DR":      "#e2b93b",
    "Severe DR":        "#ee8a3a",
    "Proliferative DR": "#e5484d",
}

RISK_COLOURS = {
    "High":     "#e5484d",
    "Moderate": "#e2b93b",
    "Low":      "#2fbf8a",
}

ACCENT = "#5aa9ff"

# ---------------------------------------------------------------------------
# Stylesheet
# ---------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0b1220;
    --surface: #111a2b;
    --surface-2: #162236;
    --border: #22304a;
    --text: #e6edf7;
    --muted: #8b9bb4;
    --accent: #5aa9ff;
}

html, body, [class*="css"], .stApp {
    font-family: 'Inter', -apple-system, 'Segoe UI', Roboto, sans-serif;
}
.stApp { background: var(--bg); color: var(--text); }
.block-container { padding-top: 3.2rem; padding-bottom: 3rem; max-width: 1280px; }

/* Sidebar: always open, so there is no collapse control to lose */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebarCollapseButton"] { display: none; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Widgets */
[data-testid="stFileUploaderDropzone"] {
    background: var(--surface);
    border: 1px dashed #2f4266;
    border-radius: 8px;
}
.stButton > button, .stDownloadButton > button {
    background: var(--surface-2);
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 6px;
    font-weight: 500;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    border-color: var(--accent);
    color: var(--accent);
}
.stTabs [data-baseweb="tab-list"] { gap: 1.75rem; border-bottom: 1px solid var(--border); }
.stTabs [data-baseweb="tab"] { padding: 0.65rem 0; color: var(--muted); font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--text); }

/* Top bar */
.rg-topbar {
    display: flex; justify-content: space-between; align-items: flex-end;
    padding-bottom: 1rem; margin-bottom: 1.25rem;
    border-bottom: 1px solid var(--border);
}
.rg-title { font-size: 1.65rem; font-weight: 600; letter-spacing: -0.01em; color: var(--text); }
.rg-sub { font-size: 0.9rem; color: var(--muted); margin-top: 0.15rem; }
.rg-tags { display: flex; gap: 0.5rem; flex-wrap: wrap; }
.rg-tags span {
    font-size: 0.72rem; color: var(--muted);
    border: 1px solid var(--border); border-radius: 4px; padding: 0.2rem 0.55rem;
}

/* Sidebar brand and labels */
.rg-brand { display: flex; align-items: center; gap: 0.55rem; font-weight: 600; font-size: 1.05rem; margin-bottom: 0.25rem; }
.rg-mark { width: 10px; height: 10px; border-radius: 50%; background: var(--accent); display: inline-block; }
.rg-brand-sub { font-size: 0.78rem; color: var(--muted); margin-bottom: 1.1rem; }
.rg-label {
    font-size: 0.7rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; margin: 1.1rem 0 0.35rem;
}
.rg-model-status { font-size: 0.8rem; color: var(--muted); line-height: 1.5; margin-top: 0.5rem; }
.rg-model-status b { color: var(--text); font-weight: 600; }

/* Section headings */
.rg-section {
    font-size: 0.95rem; font-weight: 600; color: var(--text);
    margin: 1.9rem 0 0.8rem; padding-bottom: 0.45rem; border-bottom: 1px solid var(--border);
}
.rg-section small { color: var(--muted); font-weight: 400; margin-left: 0.6rem; font-size: 0.8rem; }

/* Record strip */
.rg-record {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 0; border: 1px solid var(--border); border-radius: 8px;
    background: var(--surface); margin: 1.25rem 0 1rem;
}
.rg-record > div { padding: 0.7rem 1rem; border-right: 1px solid var(--border); }
.rg-record > div:last-child { border-right: none; }
.rg-record .k { font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
.rg-record .v { font-size: 0.9rem; color: var(--text); margin-top: 0.15rem; word-break: break-word; }

/* Summary cards */
.rg-cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; margin-bottom: 0.5rem; }
.rg-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 0.9rem 1.05rem;
}
.rg-card .k { font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.rg-card .v { font-size: 1.45rem; font-weight: 600; margin-top: 0.3rem; font-variant-numeric: tabular-nums; color: var(--text); }
.rg-card .s { font-size: 0.78rem; color: var(--muted); margin-top: 0.2rem; }

/* Severity pill */
.rg-pill {
    display: inline-block; font-size: 0.82rem; font-weight: 600;
    padding: 0.15rem 0.7rem; border-radius: 4px;
}

/* Banners and notices */
.rg-banner {
    border: 1px solid var(--border); border-left-width: 4px;
    border-radius: 8px; padding: 0.95rem 1.2rem; margin: 0.25rem 0 1.1rem;
}
.rg-banner .t { font-size: 0.74rem; letter-spacing: 0.09em; text-transform: uppercase; font-weight: 700; }
.rg-banner .b { font-size: 0.92rem; color: var(--text); margin-top: 0.3rem; line-height: 1.55; }
.rg-notice {
    background: var(--surface-2); border: 1px solid var(--border);
    border-left: 4px solid #e2b93b; border-radius: 8px;
    padding: 0.8rem 1.1rem; margin-bottom: 1rem; font-size: 0.86rem; color: var(--muted); line-height: 1.55;
}
.rg-notice b { color: var(--text); font-weight: 600; }

/* Probability bars */
.rg-bars { margin-top: 0.25rem; }
.rg-bar-row { display: grid; grid-template-columns: 125px 1fr 56px; align-items: center; gap: 12px; padding: 5px 0; }
.rg-bar-row .n { font-size: 0.85rem; color: var(--muted); }
.rg-bar-row.top .n { color: var(--text); font-weight: 600; }
.rg-track { height: 8px; background: #1b2a44; border-radius: 4px; overflow: hidden; }
.rg-fill { height: 100%; border-radius: 4px; }
.rg-bar-row .p { font-size: 0.82rem; color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
.rg-bar-row.top .p { color: var(--text); font-weight: 600; }

/* Density scale */
.rg-density { margin-top: 0.4rem; }
.rg-density .track { position: relative; height: 10px; background: #1b2a44; border-radius: 5px; }
.rg-density .fill { height: 100%; border-radius: 5px; background: var(--accent); }
.rg-density .scale { display: flex; justify-content: space-between; font-size: 0.72rem; color: var(--muted); margin-top: 0.3rem; }

/* Figures */
.rg-cap { font-size: 0.78rem; color: var(--muted); margin: 0.35rem 0 0.8rem; }
.rg-cap b { color: var(--text); font-weight: 600; }
.rg-note { font-size: 0.85rem; color: var(--muted); line-height: 1.6; margin-top: 0.4rem; }

/* Table */
.rg-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.rg-table th {
    text-align: left; font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase;
    color: var(--muted); font-weight: 600; padding: 0.55rem 0.8rem; border-bottom: 1px solid var(--border);
}
.rg-table td { padding: 0.65rem 0.8rem; border-bottom: 1px solid var(--border); color: var(--text); font-variant-numeric: tabular-nums; }
.rg-table td:first-child { color: var(--muted); }

/* Recommendations */
ol.rg-recs { list-style: none; counter-reset: rec; padding: 0; margin: 0.4rem 0 0; }
ol.rg-recs li {
    counter-increment: rec; position: relative;
    padding: 0.7rem 0 0.7rem 2.4rem; border-bottom: 1px solid var(--border);
    font-size: 0.92rem; line-height: 1.55; color: var(--text);
}
ol.rg-recs li::before {
    content: counter(rec, decimal-leading-zero); position: absolute; left: 0; top: 0.72rem;
    font-size: 0.78rem; color: var(--accent); font-weight: 600; font-variant-numeric: tabular-nums;
}

/* Pipeline steps (empty state) */
.rg-steps { display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 12px; margin-top: 1.1rem; }
.rg-step { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 1rem 1.1rem; }
.rg-step .n { font-size: 0.74rem; color: var(--accent); font-weight: 700; letter-spacing: 0.08em; }
.rg-step .h { font-size: 0.98rem; font-weight: 600; margin: 0.25rem 0 0.3rem; color: var(--text); }
.rg-step .d { font-size: 0.84rem; color: var(--muted); line-height: 1.5; }

.rg-disclaimer {
    font-size: 0.76rem; color: var(--muted); border-top: 1px solid var(--border);
    margin-top: 2.5rem; padding-top: 0.9rem; line-height: 1.6;
}
</style>
"""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def review_text(months: int) -> str:
    """Human-readable follow-up interval."""
    if months <= 0:
        return "Immediate"
    return f"{months} month" + ("" if months == 1 else "s")


def format_pct(pct: float, cap: float = 999.0) -> str:
    """Signed percentage, capped so a near-zero baseline cannot print absurd values."""
    if pct > cap:
        return f">+{cap:.0f}%"
    if pct < -cap:
        return f"<-{cap:.0f}%"
    return f"{pct:+.1f}%"


def _hex_alpha(colour: str, alpha_hex: str) -> str:
    return f"{colour}{alpha_hex}"


# ---------------------------------------------------------------------------
# Components (each returns an HTML string)
# ---------------------------------------------------------------------------
def topbar() -> str:
    return (
        '<div class="rg-topbar"><div>'
        '<div class="rg-title">Retina Guard</div>'
        '<div class="rg-sub">Diabetic retinopathy screening and progression monitoring</div>'
        '</div><div class="rg-tags">'
        '<span>EfficientNet-B3</span><span>Grad-CAM</span><span>Research use only</span>'
        '</div></div>'
    )


def sidebar_brand() -> str:
    return (
        '<div class="rg-brand"><span class="rg-mark"></span>Retina Guard</div>'
        '<div class="rg-brand-sub">Analysis controls</div>'
    )


def label(text: str) -> str:
    return f'<div class="rg-label">{escape(text)}</div>'


def section(title: str, sub: str | None = None) -> str:
    extra = f"<small>{escape(sub)}</small>" if sub else ""
    return f'<div class="rg-section">{escape(title)}{extra}</div>'


def caption(text: str, figure: str | None = None) -> str:
    prefix = f"<b>{escape(figure)}</b> " if figure else ""
    return f'<div class="rg-cap">{prefix}{escape(text)}</div>'


def model_status(custom: bool) -> str:
    if custom:
        state, detail = "Custom weights", "Fine-tuned checkpoint loaded."
    else:
        state, detail = "Baseline", "ImageNet backbone, untrained classifier head."
    return f'<div class="rg-model-status"><b>{state}</b><br>{detail}</div>'


def pill(name: str) -> str:
    c = SEVERITY_COLOURS.get(name, "#8b9bb4")
    return (
        f'<span class="rg-pill" style="background:{_hex_alpha(c, "22")};'
        f'color:{c};border:1px solid {_hex_alpha(c, "66")};">{escape(name)}</span>'
    )


def card(title: str, value: str, sub: str = "", colour: str | None = None) -> str:
    border = f' style="border-left:3px solid {colour};"' if colour else ""
    sub_html = f'<div class="s">{escape(sub)}</div>' if sub else ""
    return (
        f'<div class="rg-card"{border}><div class="k">{escape(title)}</div>'
        f'<div class="v">{escape(value)}</div>{sub_html}</div>'
    )


def cards(items: list[str]) -> str:
    return '<div class="rg-cards">' + "".join(items) + "</div>"


def record_strip(pairs: list[tuple[str, str]]) -> str:
    cells = "".join(
        f'<div><div class="k">{escape(k)}</div><div class="v">{escape(v)}</div></div>'
        for k, v in pairs
    )
    return f'<div class="rg-record">{cells}</div>'


def banner(level: str, title: str, body: str) -> str:
    c = RISK_COLOURS.get(level, "#8b9bb4")
    return (
        f'<div class="rg-banner" style="border-left-color:{c};background:{_hex_alpha(c, "14")};">'
        f'<div class="t" style="color:{c};">{escape(title)}</div>'
        f'<div class="b">{escape(body)}</div></div>'
    )


def notice(title: str, body: str) -> str:
    return f'<div class="rg-notice"><b>{escape(title)}.</b> {escape(body)}</div>'


def prob_bars(probs: list[float], names: list[str], top_idx: int | None = None) -> str:
    if top_idx is None:
        top_idx = max(range(len(probs)), key=lambda i: probs[i])
    rows = []
    for i, (p, n) in enumerate(zip(probs, names)):
        colour = SEVERITY_COLOURS.get(n, ACCENT)
        width = max(0.0, min(1.0, p)) * 100
        cls = "rg-bar-row top" if i == top_idx else "rg-bar-row"
        rows.append(
            f'<div class="{cls}"><div class="n">{escape(n)}</div>'
            f'<div class="rg-track"><div class="rg-fill" style="width:{width:.1f}%;background:{colour};"></div></div>'
            f'<div class="p">{p * 100:.1f}%</div></div>'
        )
    return '<div class="rg-bars">' + "".join(rows) + "</div>"


def density_bar(density: float) -> str:
    width = max(0.0, min(1.0, density)) * 100
    return (
        '<div class="rg-density"><div class="track">'
        f'<div class="fill" style="width:{width:.1f}%;"></div></div>'
        '<div class="scale"><span>0%</span><span>50%</span><span>100%</span></div></div>'
    )


def steps(items: list[tuple[str, str, str]]) -> str:
    body = "".join(
        f'<div class="rg-step"><div class="n">{escape(n)}</div>'
        f'<div class="h">{escape(h)}</div><div class="d">{escape(d)}</div></div>'
        for n, h, d in items
    )
    return f'<div class="rg-steps">{body}</div>'


def compare_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{escape(str(c))}</td>" for c in row) + "</tr>" for row in rows
    )
    return f'<table class="rg-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def rec_list(recs: list[str]) -> str:
    return '<ol class="rg-recs">' + "".join(f"<li>{escape(r)}</li>" for r in recs) + "</ol>"


def disclaimer() -> str:
    return (
        '<div class="rg-disclaimer">Retina Guard is a research and teaching prototype. '
        'It is not a certified medical device, and its output must be reviewed by a '
        'qualified ophthalmologist before any clinical decision is made.</div>'
    )


# ---------------------------------------------------------------------------
# Plain-text report (for download)
# ---------------------------------------------------------------------------
def text_report(title: str, header: list[tuple[str, str]], findings: list[tuple[str, str]],
                recommendations: list[str], summary: str | None = None) -> str:
    line = "=" * 64
    out = [line, title.upper(), line]
    out += [f"{k:<22}{v}" for k, v in header]
    out += ["", "FINDINGS", "-" * 64]
    out += [f"{k:<22}{v}" for k, v in findings]
    if summary:
        out += ["", "SUMMARY", "-" * 64, summary]
    out += ["", "RECOMMENDATIONS", "-" * 64]
    out += [f"{i}. {r}" for i, r in enumerate(recommendations, 1)]
    out += ["", line,
            "Research prototype. Not a medical device. Review by a qualified",
            "ophthalmologist is required before any clinical decision.", line]
    return "\n".join(out)
