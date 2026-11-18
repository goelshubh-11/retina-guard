"""
prevention.py — Retina Guard
Prevention Agent: tracks lesion progression between historical and
current retinal scans and triggers clinical alerts.

Algorithm
---------
1. Receive Grad-CAM activation densities from both scans.
2. Compute Delta-Change  = current_density - historical_density
3. Compute Relative-Change = delta / max(historical_density, ε)
4. If relative change > PROGRESSION_THRESHOLD  → issue Prevention Alert
5. Return a structured report with risk level, recommendations, and timeline.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
import numpy as np


# ─────────────────────────────────────────────
# Thresholds & risk tiers
# ─────────────────────────────────────────────

PROGRESSION_THRESHOLD  = 0.10   # 10 % relative increase → high-risk
MODERATE_THRESHOLD     = 0.05   # 5 % relative increase  → moderate-risk
EPSILON                = 1e-6   # avoid div-by-zero


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

@dataclass
class ScanResult:
    """Summary of a single retinal scan."""
    label:              str
    severity_class:     str
    confidence:         float
    activation_density: float
    probs:              list[float]
    scan_date:          date = field(default_factory=date.today)


@dataclass
class ProgressionReport:
    """Full prevention analysis comparing two scan results."""
    historical:          ScanResult
    current:             ScanResult
    delta_density:       float
    relative_change_pct: float
    risk_level:          str          # "Low" | "Moderate" | "High"
    alert_triggered:     bool
    recommendations:     list[str]
    next_review_months:  int
    summary:             str


# ─────────────────────────────────────────────
# Core prevention agent
# ─────────────────────────────────────────────

def analyse_progression(
    historical: ScanResult,
    current:    ScanResult,
) -> ProgressionReport:
    """
    Compare two scans and produce a ProgressionReport.

    Parameters
    ----------
    historical : ScanResult from the older/baseline scan
    current    : ScanResult from the recent scan

    Returns
    -------
    ProgressionReport
    """
    hist_d = historical.activation_density
    curr_d = current.activation_density

    delta          = curr_d - hist_d
    relative_chg   = delta / max(hist_d, EPSILON)
    rel_pct        = relative_chg * 100.0

    # ── Risk tier ──────────────────────────────
    if relative_chg > PROGRESSION_THRESHOLD:
        risk_level       = "High"
        alert_triggered  = True
    elif relative_chg > MODERATE_THRESHOLD:
        risk_level       = "Moderate"
        alert_triggered  = False
    else:
        risk_level       = "Low"
        alert_triggered  = False

    # ── Severity-aware recommendations ─────────
    recommendations = _build_recommendations(
        current.severity_class, risk_level, delta, current.confidence
    )

    # ── Next review window ──────────────────────
    next_review = _next_review_months(current.severity_class, risk_level)

    # ── Natural-language summary ────────────────
    summary = _build_summary(historical, current, rel_pct, risk_level, next_review)

    return ProgressionReport(
        historical          = historical,
        current             = current,
        delta_density       = delta,
        relative_change_pct = rel_pct,
        risk_level          = risk_level,
        alert_triggered     = alert_triggered,
        recommendations     = recommendations,
        next_review_months  = next_review,
        summary             = summary,
    )


# ─────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────

def _build_recommendations(severity: str,
                             risk: str,
                             delta: float,
                             confidence: float) -> list[str]:
    base = []

    # Systemic control always first
    base.append("Optimise glycaemic control: target HbA1c < 7 %.")
    base.append("Monitor blood pressure (target < 130/80 mmHg).")
    base.append("Control dyslipidaemia with statin therapy if indicated.")

    if severity in ("No DR", "Mild DR"):
        if risk == "Low":
            base.append("Continue annual retinal photography screening.")
        else:
            base.append("Advance screening interval to every 6 months.")
    elif severity == "Moderate DR":
        base.append("Ophthalmology referral within 3 months.")
        base.append("Fluorescein angiography to map capillary dropout.")
    elif severity == "Severe DR":
        base.append("Urgent ophthalmology referral — within 4 weeks.")
        base.append("Consider intravitreal anti-VEGF (e.g., Ranibizumab).")
        base.append("Pan-retinal photocoagulation assessment required.")
    elif severity == "Proliferative DR":
        base.append("IMMEDIATE ophthalmology referral.")
        base.append("Vitreoretinal surgery evaluation if traction detachment.")
        base.append("Anti-VEGF injection and/or pan-retinal photocoagulation.")

    if risk == "High":
        base.append(
            "Prevention Alert: lesion spread increased by > 10 % since last scan. "
            "High risk of vision loss within 6 months without intervention."
        )
    elif risk == "Moderate":
        base.append(
            "Moderate progression detected. "
            "Strengthen systemic risk-factor management."
        )

    if confidence < 0.6:
        base.append(
            "Model confidence is below 60 % — consider re-imaging with better "
            "mydriasis or refer for slit-lamp biomicroscopy."
        )

    return base


def _next_review_months(severity: str, risk: str) -> int:
    table = {
        ("No DR",            "Low"):      12,
        ("No DR",            "Moderate"): 9,
        ("No DR",            "High"):     6,
        ("Mild DR",          "Low"):      9,
        ("Mild DR",          "Moderate"): 6,
        ("Mild DR",          "High"):     3,
        ("Moderate DR",      "Low"):      6,
        ("Moderate DR",      "Moderate"): 3,
        ("Moderate DR",      "High"):     1,
        ("Severe DR",        "Low"):      3,
        ("Severe DR",        "Moderate"): 2,
        ("Severe DR",        "High"):     1,
        ("Proliferative DR", "Low"):      1,
        ("Proliferative DR", "Moderate"): 1,
        ("Proliferative DR", "High"):     0,   # immediate
    }
    return table.get((severity, risk), 6)


def _build_summary(hist: ScanResult,
                    curr: ScanResult,
                    rel_pct: float,
                    risk: str,
                    next_review: int) -> str:
    direction = "increased" if rel_pct > 0 else "decreased"
    review    = f"in {next_review} month(s)" if next_review > 0 else "IMMEDIATELY"

    return (
        f"Comparing {hist.label} ({hist.severity_class}, "
        f"{hist.scan_date}) → {curr.label} ({curr.severity_class}, "
        f"{curr.scan_date}): Grad-CAM lesion density has {direction} by "
        f"{abs(rel_pct):.1f} %. Risk level: {risk}. "
        f"Recommend follow-up {review}."
    )


# ─────────────────────────────────────────────
# Quick single-scan summary (no baseline)
# ─────────────────────────────────────────────

def single_scan_report(result: ScanResult) -> dict:
    """Return clinical recommendations for a single scan (no baseline)."""
    recs  = _build_recommendations(
        result.severity_class, "Low", 0.0, result.confidence
    )
    review = _next_review_months(result.severity_class, "Low")
    return {
        "severity_class":   result.severity_class,
        "confidence":       result.confidence,
        "recommendations":  recs,
        "next_review_months": review,
    }
