"""Frequency Domain Analyzer — FFT spectral analysis for AI image detection.

GAN-generated images often exhibit periodic patterns in the frequency domain
that are invisible to the human eye but clearly visible in FFT analysis.
These patterns arise from the upsampling operations in neural networks.
"""
import numpy as np
from PIL import Image
import cv2
import base64
import io


def analyze_frequency(image: Image.Image) -> dict:
    """
    Perform FFT-based frequency analysis on an image.

    GAN/diffusion model artifacts often appear as:
    - Regular grid patterns in the frequency domain
    - Unusual high-frequency energy distribution
    - Periodic peaks at specific frequencies

    Args:
        image: PIL Image to analyze

    Returns:
        dict with frequency analysis results and visualization
    """
    # Convert to grayscale numpy array
    img_gray = np.array(image.convert("L").resize((512, 512))).astype(np.float32)

    # ─── 2D FFT ──────────────────────────────────────────
    f_transform = np.fft.fft2(img_gray)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log1p(np.abs(f_shift))

    # Normalize for visualization
    mag_norm = (magnitude_spectrum - magnitude_spectrum.min()) / (magnitude_spectrum.max() - magnitude_spectrum.min() + 1e-8)
    mag_visual = (mag_norm * 255).astype(np.uint8)

    # Apply color map for better visualization
    mag_colored = cv2.applyColorMap(mag_visual, cv2.COLORMAP_INFERNO)
    mag_colored_rgb = cv2.cvtColor(mag_colored, cv2.COLOR_BGR2RGB)

    # ─── Power Spectrum Analysis ─────────────────────────
    power_spectrum = np.abs(f_shift) ** 2
    total_power = power_spectrum.sum()

    # Radial power distribution (azimuthally averaged)
    h, w = power_spectrum.shape
    center_y, center_x = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2).astype(int)
    max_r = min(center_x, center_y)

    radial_profile = np.zeros(max_r)
    for r in range(max_r):
        mask = R == r
        if mask.any():
            radial_profile[r] = power_spectrum[mask].mean()

    # ─── Anomaly Detection ───────────────────────────────
    # Check for unusual peaks in the radial profile
    if len(radial_profile) > 10:
        # Smooth the profile
        from numpy import convolve
        kernel = np.ones(5) / 5
        smoothed = convolve(radial_profile, kernel, mode='same')

        # Detect peaks (deviations from smooth trend)
        deviation = np.abs(radial_profile - smoothed)
        threshold = np.std(deviation) * 3
        peaks = np.where(deviation > threshold)[0]
        peak_count = len(peaks)
    else:
        peaks = []
        peak_count = 0

    # ─── High-frequency energy ratio ─────────────────────
    # AI images often have different high-freq energy distribution
    mid_freq_start = max_r // 4
    high_freq_start = max_r // 2

    low_freq_power = radial_profile[:mid_freq_start].sum() if mid_freq_start > 0 else 0
    mid_freq_power = radial_profile[mid_freq_start:high_freq_start].sum()
    high_freq_power = radial_profile[high_freq_start:].sum()

    total_radial = low_freq_power + mid_freq_power + high_freq_power + 1e-8
    hf_ratio = high_freq_power / total_radial
    mf_ratio = mid_freq_power / total_radial
    lf_ratio = low_freq_power / total_radial

    # ─── Grid Pattern Detection ──────────────────────────
    # Check for periodic patterns (common in GAN outputs)
    grid_score = 0
    if peak_count > 3:
        # Multiple peaks suggest periodic artifacts
        grid_score = min(100, peak_count * 15)

    # ─── Azimuthal Variance ──────────────────────────────
    # Real images have more varied frequency content; AI tends to be more uniform
    angles = np.arctan2(Y - center_y, X - center_x)
    angle_bins = np.linspace(-np.pi, np.pi, 36)
    azimuthal_power = []
    for i in range(len(angle_bins) - 1):
        mask = (angles >= angle_bins[i]) & (angles < angle_bins[i+1]) & (R > 10) & (R < max_r)
        if mask.any():
            azimuthal_power.append(power_spectrum[mask].mean())

    azimuthal_variance = float(np.var(azimuthal_power)) if azimuthal_power else 0
    # Normalize variance to 0-100 scale
    az_uniformity = 100 - min(100, azimuthal_variance / (np.mean(azimuthal_power) + 1e-8) * 100) if azimuthal_power else 50

    # ─── Risk Assessment ─────────────────────────────────
    risk_score = 0
    findings = []

    # High frequency energy analysis
    if hf_ratio < 0.05:
        findings.append({
            "type": "warning",
            "message": "Very low high-frequency energy — image may be overly smooth (common in AI)",
            "severity": "medium"
        })
        risk_score += 25
    elif hf_ratio > 0.4:
        findings.append({
            "type": "info",
            "message": "Strong high-frequency content — consistent with natural photography",
            "severity": "low"
        })
        risk_score -= 10

    # Grid pattern
    if grid_score > 30:
        findings.append({
            "type": "warning",
            "message": f"Periodic frequency peaks detected ({peak_count} anomalous peaks) — possible GAN artifacts",
            "severity": "high" if grid_score > 60 else "medium"
        })
        risk_score += grid_score // 2

    # Azimuthal uniformity
    if az_uniformity > 80:
        findings.append({
            "type": "warning",
            "message": "Highly uniform frequency distribution — AI images tend to be more isotropic",
            "severity": "low"
        })
        risk_score += 10

    risk_score = max(0, min(100, risk_score))

    # ─── Generate visualization ──────────────────────────
    spectrum_b64 = _array_to_base64(mag_colored_rgb)

    return {
        "spectrum_base64": spectrum_b64,
        "risk_score": risk_score,
        "findings": findings,
        "metrics": {
            "high_freq_ratio": round(float(hf_ratio), 4),
            "mid_freq_ratio": round(float(mf_ratio), 4),
            "low_freq_ratio": round(float(lf_ratio), 4),
            "peak_count": int(peak_count),
            "grid_score": int(grid_score),
            "azimuthal_uniformity": round(float(az_uniformity), 2),
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
    warnings = [f for f in findings if f["type"] == "warning"]
    if risk_score >= 50:
        return f"🔴 Frequency analysis shows significant anomalies (score: {risk_score}/100). Patterns suggest possible AI generation."
    elif risk_score >= 25:
        return f"🟡 Some frequency anomalies detected (score: {risk_score}/100). Results are inconclusive."
    else:
        return f"🟢 Frequency spectrum appears natural (score: {risk_score}/100). No significant AI-generation patterns found."
