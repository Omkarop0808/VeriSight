"""Error Level Analysis (ELA) — Detects image tampering and inconsistencies.

ELA works by resaving the image at a known quality level and comparing
the differences. Regions with different compression levels appear brighter,
which can indicate splicing, manipulation, or AI-generated composite areas.
"""
import numpy as np
from PIL import Image
import cv2
import base64
import io


def analyze_ela(image: Image.Image, quality: int = 90) -> dict:
    """
    Perform Error Level Analysis on an image.

    Args:
        image: PIL Image to analyze
        quality: JPEG quality for resave comparison (default 90)

    Returns:
        dict with ELA analysis results and visualization
    """
    # Ensure RGB
    img_rgb = image.convert("RGB").resize((512, 512))
    img_array = np.array(img_rgb)

    # ─── Resave at specified quality ─────────────────────
    buffer = io.BytesIO()
    img_rgb.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    resaved = Image.open(buffer).convert("RGB")
    resaved_array = np.array(resaved)

    # ─── Compute ELA ─────────────────────────────────────
    # Difference between original and resaved
    ela_array = np.abs(img_array.astype(np.float32) - resaved_array.astype(np.float32))

    # Scale for visibility (multiply by a factor)
    scale_factor = 20
    ela_scaled = np.clip(ela_array * scale_factor, 0, 255).astype(np.uint8)

    # ─── Compute grayscale ELA for analysis ──────────────
    ela_gray = np.mean(ela_array, axis=2)

    # ─── Statistical Analysis ────────────────────────────
    ela_mean = float(np.mean(ela_gray))
    ela_std = float(np.std(ela_gray))
    ela_max = float(np.max(ela_gray))
    ela_min = float(np.min(ela_gray))

    # ─── Region-based analysis ───────────────────────────
    # Split image into 4x4 grid and analyze each region
    h, w = ela_gray.shape
    grid_size = 4
    region_h = h // grid_size
    region_w = w // grid_size

    region_means = []
    for i in range(grid_size):
        for j in range(grid_size):
            region = ela_gray[i*region_h:(i+1)*region_h, j*region_w:(j+1)*region_w]
            region_means.append(float(np.mean(region)))

    # Variance between regions (high variance = suspicious)
    region_variance = float(np.var(region_means))
    region_max = max(region_means)
    region_min = min(region_means)
    uniformity_ratio = region_min / (region_max + 1e-8)

    # ─── Risk Assessment ─────────────────────────────────
    risk_score = 0
    findings = []

    # Uniform ELA (AI images tend to have very uniform error levels)
    if ela_std < 2.0:
        findings.append({
            "type": "warning",
            "message": "Very uniform error levels — consistent with AI-generated images",
            "severity": "medium"
        })
        risk_score += 25

    # High uniformity across regions
    if uniformity_ratio > 0.8:
        findings.append({
            "type": "warning",
            "message": "Error levels are highly uniform across all regions — unusual for real photos",
            "severity": "medium"
        })
        risk_score += 20

    # Very high variance (possible splicing)
    if region_variance > 50:
        findings.append({
            "type": "warning",
            "message": "High variance between image regions — possible image splicing or compositing",
            "severity": "high"
        })
        risk_score += 30

    # Low overall ELA (heavily compressed or generated)
    if ela_mean < 1.5:
        findings.append({
            "type": "info",
            "message": "Very low error levels — image is either heavily compressed or AI-generated",
            "severity": "low"
        })
        risk_score += 10

    # Natural error distribution
    if 3.0 < ela_mean < 15.0 and ela_std > 3.0 and uniformity_ratio < 0.7:
        findings.append({
            "type": "info",
            "message": "Error level distribution is consistent with natural photography",
            "severity": "low"
        })
        risk_score -= 15

    risk_score = max(0, min(100, risk_score))

    # ─── Generate heatmap visualization ──────────────────
    # Create colored ELA heatmap
    ela_norm = ((ela_gray - ela_gray.min()) / (ela_gray.max() - ela_gray.min() + 1e-8) * 255).astype(np.uint8)
    ela_heatmap = cv2.applyColorMap(ela_norm, cv2.COLORMAP_JET)
    ela_heatmap_rgb = cv2.cvtColor(ela_heatmap, cv2.COLOR_BGR2RGB)

    # ELA overlay on original
    ela_overlay = cv2.addWeighted(img_array, 0.5, ela_scaled, 0.5, 0)

    # Convert to base64
    ela_heatmap_b64 = _array_to_base64(ela_heatmap_rgb)
    ela_overlay_b64 = _array_to_base64(ela_overlay)
    ela_raw_b64 = _array_to_base64(ela_scaled)

    return {
        "ela_heatmap_base64": ela_heatmap_b64,
        "ela_overlay_base64": ela_overlay_b64,
        "ela_raw_base64": ela_raw_b64,
        "risk_score": risk_score,
        "findings": findings,
        "metrics": {
            "mean_error": round(ela_mean, 4),
            "std_error": round(ela_std, 4),
            "max_error": round(ela_max, 4),
            "min_error": round(ela_min, 4),
            "region_variance": round(region_variance, 4),
            "uniformity_ratio": round(uniformity_ratio, 4),
            "quality_used": quality,
        },
        "summary": _generate_summary(findings, risk_score)
    }


def _array_to_base64(img_array: np.ndarray) -> str:
    """Convert numpy array to base64 PNG string."""
    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _generate_summary(findings: list, risk_score: int) -> str:
    """Generate human-readable summary."""
    if risk_score >= 50:
        return f"🔴 ELA reveals significant compression inconsistencies (score: {risk_score}/100). Image may be manipulated or AI-generated."
    elif risk_score >= 25:
        return f"🟡 Some ELA anomalies detected (score: {risk_score}/100). Results warrant further investigation."
    else:
        return f"🟢 ELA appears consistent (score: {risk_score}/100). No significant manipulation detected."
