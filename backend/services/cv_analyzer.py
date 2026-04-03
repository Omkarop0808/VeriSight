"""Computer Vision Optics Analyzer — Extracts physical lens heuristics.

Analyzes Laplacian variance, Canny edge density, and Shannon entropy
to mathematically differentiate physical camera focal planes from
synthetic image generators.
"""
import cv2
import numpy as np
from PIL import Image
from scipy.stats import entropy

def analyze_cv_optics(image: Image.Image) -> dict:
    img_rgb = image.convert("RGB")
    img_array = np.array(img_rgb)
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # ─── Laplacian Variance (Blur/Depth of Field) ────────
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # ─── Canny Edge Density ──────────────────────────────
    edges = cv2.Canny(gray, 100, 200)
    edge_density = (np.count_nonzero(edges) / edges.size) * 100
    
    # ─── Histogram Entropy ───────────────────────────────
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.ravel() / (hist.sum() + 1e-8)
    shannon_entropy = entropy(hist_norm, base=2)
    
    # ─── Risk Calculation ────────────────────────────────
    risk_score = 0
    findings = []
    
    # Image Generation models often lack natural focal blur (hyper-sharpness)
    if laplacian_var > 2500:
        risk_score += 25
        findings.append({"type": "warning", "message": "High focal sharpness across field; lacks natural camera depth-of-field blur."})
    elif 100 < laplacian_var < 1500:
        # Natural optics
        risk_score -= 20
        findings.append({"type": "info", "message": "Natural focal plane blurring detected (consistent with physical lenses)."})
        
    if edge_density > 18:
        risk_score += 15
        findings.append({"type": "warning", "message": "Unnaturally high edge structural density (synthetic textures)."})
        
    if shannon_entropy > 7.8:
        risk_score += 10
        findings.append({"type": "warning", "message": "Color entropy is atypically maximized or constrained."})
        
    risk_score = max(0, min(100, risk_score))
    
    return {
        "risk_score": risk_score,
        "metrics": {
            "laplacian_variance": round(float(laplacian_var), 2),
            "edge_density_pct": round(float(edge_density), 2),
            "shannon_entropy": round(float(shannon_entropy), 2)
        },
        "findings": findings
    }
