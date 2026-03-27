"""
DeepSight AI — FastAPI Backend Server
Dual-engine AI-generated image & deepfake detector
"""
import os
import sys
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import io

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.classifier import load_model, classify_image
from backend.services.gradcam_service import generate_heatmap
from backend.services.gemini_forensics import configure_gemini, analyze_image_forensically
from backend.services.score_combiner import combine_verdicts

# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="DeepSight AI",
    description="Dual-engine AI-generated image & deepfake detector",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Analytics Store ─────────────────────────────────────────
analytics = {
    "total_scans": 0,
    "real_count": 0,
    "fake_count": 0,
    "confidences": [],
    "recent_scans": [],
}

# ─── Startup ─────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    """Load model and configure Gemini on startup."""
    checkpoint = os.path.join("backend", "checkpoints", "checkpoint_phase2.pth")
    if not os.path.exists(checkpoint):
        # Also check parent directory structure
        alt_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "AI-Generated-Deepfake-Image-Detector",
            "AI Images Detector",
            "checkpoints",
            "checkpoint_phase2.pth"
        )
        if os.path.exists(alt_checkpoint):
            checkpoint = alt_checkpoint

    load_model(checkpoint)
    configure_gemini()


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "🛡️ DeepSight AI — Dual-Engine Deepfake Detector", "status": "running"}


@app.post("/api/analyze")
async def analyze_image(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.50),
    use_gemini: bool = Form(default=True),
):
    """
    Analyze a single image for AI-generation / deepfake artifacts.

    Returns ML classification, Grad-CAM heatmap, Gemini forensics, and combined verdict.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Validate size
        if image.width * image.height > 4096 * 4096:
            raise HTTPException(400, "Image too large. Max 4096x4096 pixels.")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {str(e)}")

    # ─── Engine 1: ML Classification ─────────────────────
    ml_result = classify_image(image, threshold)

    # ─── Grad-CAM Heatmap ────────────────────────────────
    try:
        target_class = 1  # Analyze for "Fake" class
        heatmap_b64, raw_cam_b64 = generate_heatmap(image, target_class)
    except Exception as e:
        print(f"⚠️ Grad-CAM error: {e}")
        heatmap_b64 = None
        raw_cam_b64 = None

    # ─── Engine 2: Gemini Forensics ──────────────────────
    gemini_result = None
    if use_gemini:
        try:
            gemini_result = await analyze_image_forensically(image)
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")

    # ─── Combine Verdicts ────────────────────────────────
    combined = combine_verdicts(ml_result, gemini_result)

    processing_time = round(time.time() - start_time, 2)

    # ─── Update Analytics ────────────────────────────────
    analytics["total_scans"] += 1
    if combined["final_label"] == "Fake":
        analytics["fake_count"] += 1
    else:
        analytics["real_count"] += 1
    analytics["confidences"].append(combined["final_confidence"])
    analytics["recent_scans"].append({
        "id": analysis_id,
        "filename": file.filename,
        "verdict": combined["final_label"],
        "confidence": combined["final_confidence"],
        "timestamp": datetime.now().isoformat(),
    })
    # Keep only last 50
    analytics["recent_scans"] = analytics["recent_scans"][-50:]

    # ─── Build Response ──────────────────────────────────
    # Remove non-serializable logits from ml_result
    ml_result_clean = {k: v for k, v in ml_result.items() if k != "logits"}

    response = {
        "id": analysis_id,
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(),
        "ml_result": ml_result_clean,
        "gemini_result": gemini_result,
        "combined_verdict": combined,
        "heatmap_base64": heatmap_b64,
        "raw_cam_base64": raw_cam_b64,
        "processing_time": processing_time,
    }

    return JSONResponse(content=response)


@app.post("/api/batch")
async def batch_analyze(
    files: list[UploadFile] = File(...),
    threshold: float = Form(default=0.50),
):
    """Analyze multiple images in batch (ML only for speed)."""
    results = []

    for file in files[:20]:  # Limit to 20 images
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert("RGB")
            ml_result = classify_image(image, threshold)
            ml_result_clean = {k: v for k, v in ml_result.items() if k != "logits"}
            combined = combine_verdicts(ml_result)

            results.append({
                "filename": file.filename,
                "ml_result": ml_result_clean,
                "combined_verdict": combined,
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "error": str(e),
            })

    total = len(results)
    real_count = sum(1 for r in results if r.get("combined_verdict", {}).get("final_label") == "Real")
    fake_count = sum(1 for r in results if r.get("combined_verdict", {}).get("final_label") == "Fake")

    return {
        "total_images": total,
        "results": results,
        "summary": {
            "real_count": real_count,
            "fake_count": fake_count,
            "fake_percentage": round(fake_count / total * 100, 1) if total > 0 else 0,
        }
    }


@app.get("/api/stats")
async def get_stats():
    """Get analysis statistics."""
    avg_conf = (
        sum(analytics["confidences"]) / len(analytics["confidences"])
        if analytics["confidences"]
        else 0.0
    )

    return {
        "total_scans": analytics["total_scans"],
        "real_count": analytics["real_count"],
        "fake_count": analytics["fake_count"],
        "avg_confidence": round(avg_conf, 4),
        "recent_scans": analytics["recent_scans"][-10:],
    }


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ─── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
