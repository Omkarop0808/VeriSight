"""
DeepSight AI — FastAPI Backend Server
Dual-engine AI-generated image & deepfake detector with full forensic analysis.
"""
import os
import sys
import uuid
import time
import asyncio
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from PIL import Image
import io

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.services.classifier import load_model, classify_image
from backend.services.gradcam_service import generate_heatmap
from backend.services.gemini_forensics import configure_gemini, analyze_image_forensically
from backend.services.score_combiner import combine_verdicts
from backend.services.metadata_analyzer import analyze_metadata
from backend.services.frequency_analyzer import analyze_frequency
from backend.services.ela_analyzer import analyze_ela
from backend.services.report_generator import generate_report
from backend.services.checkpoint_downloader import download_checkpoint, is_checkpoint_available

# ─── Analytics Store ─────────────────────────────────────────
analytics = {
    "total_scans": 0,
    "real_count": 0,
    "fake_count": 0,
    "confidences": [],
    "recent_scans": [],
}


# ─── Lifespan (replaces deprecated @app.on_event) ───────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    # ── Startup ──
    # Auto-download checkpoint if not present
    if not is_checkpoint_available():
        download_checkpoint()

    checkpoint = os.path.join("backend", "checkpoints", "checkpoint_phase2.pth")
    if not os.path.exists(checkpoint):
        alt_checkpoint = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "..", "AI-Generated-Deepfake-Image-Detector",
            "AI Images Detector", "checkpoints", "checkpoint_phase2.pth"
        )
        if os.path.exists(alt_checkpoint):
            checkpoint = alt_checkpoint

    load_model(checkpoint)
    configure_gemini()
    print("🛡️ DeepSight AI server ready!")

    yield  # App is running

    # ── Shutdown ──
    print("🛡️ DeepSight AI server shutting down...")


# ─── App Setup ───────────────────────────────────────────────
app = FastAPI(
    title="DeepSight AI",
    description="Dual-engine AI-generated image & deepfake detector with full forensic analysis",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────
@app.get("/")
async def root():
    return {
        "message": "🛡️ DeepSight AI — Dual-Engine Deepfake Detector",
        "version": "2.0.0",
        "status": "running",
        "engines": ["ConvNeXtV2 (ML)", "Gemini 3.1 Flash (LLM)", "Metadata", "FFT", "ELA"],
    }


@app.post("/api/analyze")
async def analyze_image_endpoint(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.50),
    use_gemini: bool = Form(default=True),
    include_metadata: bool = Form(default=True),
    include_frequency: bool = Form(default=True),
    include_ela: bool = Form(default=True),
):
    """
    Full forensic analysis of a single image.

    Returns ML classification, Grad-CAM, Gemini forensics, metadata,
    frequency analysis, ELA, and combined verdict.
    """
    start_time = time.time()
    analysis_id = str(uuid.uuid4())[:8]

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
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
        heatmap_b64, raw_cam_b64 = generate_heatmap(image, target_class=1)
    except Exception as e:
        print(f"⚠️ Grad-CAM error: {e}")
        heatmap_b64, raw_cam_b64 = None, None

    # ─── Engine 2: Gemini Forensics ──────────────────────
    gemini_result = None
    if use_gemini:
        try:
            gemini_result = await analyze_image_forensically(image)
        except Exception as e:
            print(f"⚠️ Gemini error: {e}")

    # ─── Metadata Analysis ───────────────────────────────
    metadata_result = None
    if include_metadata:
        try:
            metadata_result = analyze_metadata(image, contents)
        except Exception as e:
            print(f"⚠️ Metadata error: {e}")

    # ─── Frequency Analysis ──────────────────────────────
    frequency_result = None
    if include_frequency:
        try:
            frequency_result = analyze_frequency(image)
        except Exception as e:
            print(f"⚠️ Frequency error: {e}")

    # ─── ELA ─────────────────────────────────────────────
    ela_result = None
    if include_ela:
        try:
            ela_result = analyze_ela(image)
        except Exception as e:
            print(f"⚠️ ELA error: {e}")

    # ─── Combine Verdicts ────────────────────────────────
    combined = combine_verdicts(ml_result, gemini_result, metadata_result, frequency_result, ela_result)
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
    analytics["recent_scans"] = analytics["recent_scans"][-50:]

    # ─── Build Response ──────────────────────────────────
    ml_result_clean = {k: v for k, v in ml_result.items() if k != "logits"}

    response = {
        "id": analysis_id,
        "filename": file.filename,
        "timestamp": datetime.now().isoformat(),
        "ml_result": ml_result_clean,
        "gemini_result": gemini_result,
        "metadata_result": metadata_result,
        "frequency_result": {k: v for k, v in (frequency_result or {}).items() if k != "spectrum_base64"} if frequency_result else None,
        "ela_result": {k: v for k, v in (ela_result or {}).items() if not k.endswith("_base64")} if ela_result else None,
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

    for file in files[:20]:
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


@app.post("/api/report/pdf")
async def generate_pdf_report(
    file: UploadFile = File(...),
    threshold: float = Form(default=0.50),
):
    """Generate a full PDF forensic report."""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    start_time = time.time()

    ml_result = classify_image(image, threshold)

    try:
        heatmap_b64, _ = generate_heatmap(image, target_class=1)
    except Exception:
        heatmap_b64 = None

    try:
        gemini_result = await analyze_image_forensically(image)
    except Exception:
        gemini_result = None

    metadata_result = analyze_metadata(image, contents)
    frequency_result = analyze_frequency(image)
    ela_result = analyze_ela(image)

    combined = combine_verdicts(ml_result, gemini_result, metadata_result, frequency_result, ela_result)
    processing_time = round(time.time() - start_time, 2)

    pdf_bytes = generate_report(
        image=image,
        combined_result=combined,
        ml_result=ml_result,
        gemini_result=gemini_result,
        heatmap_b64=heatmap_b64,
        metadata_result=metadata_result,
        frequency_result=frequency_result,
        ela_result=ela_result,
        filename=file.filename,
        processing_time=processing_time,
    )

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename=DeepSight_Report_{file.filename}.pdf"}
    )


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
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "checkpoint_available": is_checkpoint_available(),
    }


# ─── Run ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
