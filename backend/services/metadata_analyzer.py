"""Metadata Analyzer — Extracts and analyzes EXIF/image metadata for forensic clues.

AI-generated images often have:
- Stripped/missing EXIF data
- No camera model information
- Missing GPS data
- Specific software signatures (e.g., ComfyUI, Automatic1111)
- C2PA/Content Credentials watermarks
"""
import io
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Optional


def analyze_metadata(image: Image.Image, raw_bytes: bytes = None) -> dict:
    """
    Extract and analyze image metadata for forensic indicators.

    Args:
        image: PIL Image
        raw_bytes: Raw image bytes (for deeper analysis)

    Returns:
        dict with metadata analysis results
    """
    metadata = {}
    findings = []
    risk_score = 0

    # ─── EXIF Data Extraction ────────────────────────────
    exif_data = {}
    try:
        raw_exif = image._getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                # Convert bytes to string for serialization
                if isinstance(value, bytes):
                    try:
                        value = value.decode("utf-8", errors="replace")
                    except Exception:
                        value = str(value)[:100]
                exif_data[str(tag_name)] = str(value)[:200]
        else:
            findings.append({
                "type": "warning",
                "message": "No EXIF data found — common in AI-generated images",
                "severity": "medium"
            })
            risk_score += 25
    except Exception:
        findings.append({
            "type": "warning",
            "message": "EXIF data could not be extracted",
            "severity": "low"
        })

    # ─── Camera Info Analysis ────────────────────────────
    camera_make = exif_data.get("Make", "")
    camera_model = exif_data.get("Model", "")
    software = exif_data.get("Software", "")

    if camera_make or camera_model:
        findings.append({
            "type": "info",
            "message": f"Camera: {camera_make} {camera_model}".strip(),
            "severity": "low"
        })
        risk_score -= 15  # Camera info reduces risk
    else:
        if exif_data:  # Has EXIF but no camera
            findings.append({
                "type": "warning",
                "message": "EXIF present but no camera model — possible metadata stripping",
                "severity": "medium"
            })
            risk_score += 15

    # ─── Software Detection ──────────────────────────────
    ai_software_markers = [
        "stable diffusion", "comfyui", "automatic1111", "midjourney",
        "dall-e", "dalle", "flux", "invoke", "diffusers", "novelai",
        "leonardo", "playground", "adobe firefly", "bing image creator",
        "a1111", "sdxl", "sd3"
    ]

    if software:
        software_lower = software.lower()
        is_ai_software = any(marker in software_lower for marker in ai_software_markers)
        if is_ai_software:
            findings.append({
                "type": "critical",
                "message": f"AI generation software detected: {software}",
                "severity": "critical"
            })
            risk_score += 50
        else:
            findings.append({
                "type": "info",
                "message": f"Software: {software}",
                "severity": "low"
            })

    # ─── Date/Time Analysis ──────────────────────────────
    datetime_original = exif_data.get("DateTimeOriginal", "")
    datetime_digitized = exif_data.get("DateTimeDigitized", "")

    if datetime_original:
        findings.append({
            "type": "info",
            "message": f"Original date: {datetime_original}",
            "severity": "low"
        })
        risk_score -= 10
    else:
        if exif_data and len(exif_data) > 3:
            findings.append({
                "type": "warning",
                "message": "No original capture date — unusual for camera photos",
                "severity": "low"
            })
            risk_score += 5

    # ─── GPS Data ────────────────────────────────────────
    gps_info = exif_data.get("GPSInfo", "")
    if gps_info:
        findings.append({
            "type": "info",
            "message": "GPS data present — indicates real camera capture",
            "severity": "low"
        })
        risk_score -= 20
    else:
        if camera_make:
            findings.append({
                "type": "info",
                "message": "No GPS data (may be stripped for privacy)",
                "severity": "low"
            })

    # ─── Image Properties ────────────────────────────────
    width, height = image.size
    mode = image.mode
    format_info = getattr(image, "format", "Unknown") or "Unknown"

    properties = {
        "width": width,
        "height": height,
        "mode": mode,
        "format": format_info,
        "aspect_ratio": f"{width/height:.3f}" if height > 0 else "N/A",
        "megapixels": f"{(width * height) / 1e6:.2f} MP"
    }

    # Common AI image sizes
    ai_common_sizes = [
        (512, 512), (768, 768), (1024, 1024), (1024, 768),
        (768, 1024), (1024, 1360), (1360, 1024), (1536, 1024),
        (1024, 1536), (2048, 2048), (1920, 1080), (1080, 1920)
    ]

    if (width, height) in ai_common_sizes:
        findings.append({
            "type": "warning",
            "message": f"Image dimensions ({width}×{height}) match common AI generation sizes",
            "severity": "low"
        })
        risk_score += 10

    # ─── C2PA / Content Credentials ──────────────────────
    c2pa_detected = False
    if raw_bytes:
        try:
            # Check for C2PA markers in raw bytes
            if b"c2pa" in raw_bytes.lower() if isinstance(raw_bytes, (bytes, bytearray)) else False:
                c2pa_detected = True
                findings.append({
                    "type": "critical",
                    "message": "C2PA Content Credentials detected — image has provenance data",
                    "severity": "high"
                })
        except Exception:
            pass

    # ─── Clamp risk score ────────────────────────────────
    risk_score = max(0, min(100, risk_score))

    return {
        "exif_data": exif_data,
        "findings": findings,
        "risk_score": risk_score,
        "image_properties": properties,
        "has_camera_info": bool(camera_make or camera_model),
        "has_gps": bool(gps_info),
        "has_datetime": bool(datetime_original),
        "c2pa_detected": c2pa_detected,
        "exif_count": len(exif_data),
        "summary": _generate_summary(findings, risk_score)
    }


def _generate_summary(findings: list, risk_score: int) -> str:
    """Generate a human-readable summary."""
    critical = [f for f in findings if f["type"] == "critical"]
    warnings = [f for f in findings if f["type"] == "warning"]

    if critical:
        return f"⚠️ {len(critical)} critical finding(s) detected. " + critical[0]["message"]
    elif len(warnings) >= 2:
        return f"🟡 Multiple metadata anomalies detected (risk score: {risk_score}/100). Metadata patterns suggest possible AI generation."
    elif warnings:
        return f"🟡 Minor metadata anomaly: {warnings[0]['message']}"
    else:
        return f"✅ Metadata appears consistent with authentic camera capture (risk score: {risk_score}/100)."
