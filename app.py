"""
DeepSight AI — Streamlit Frontend
Stunning glassmorphism dark-theme UI for AI-generated image detection
"""
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
import io
import os
import sys
import time
import base64
import json
import asyncio
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models.convnext import build_model
from backend.transforms import test_transforms
from backend.services.gradcam_service import generate_heatmap
from backend.services.gemini_forensics import configure_gemini, analyze_image_forensically
from backend.services.score_combiner import combine_verdicts
from backend.services.classifier import load_model, classify_image, get_model_for_gradcam

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="DeepSight AI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0a0a1a 0%, #0d1526 25%, #0a0f1e 50%, #0d0d25 75%, #0a0a1a 100%);
        font-family: 'Inter', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main Title */
    .hero-title {
        font-size: 3.2rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff, #7b2ff7, #ff6b9d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }

    .hero-subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.1rem;
        font-weight: 400;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Glass Card */
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 28px;
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        margin-bottom: 20px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }

    /* Verdict Cards */
    .verdict-real {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.08), rgba(0, 200, 100, 0.03));
        border: 2px solid rgba(0, 255, 136, 0.3);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(0, 255, 136, 0.1);
    }

    .verdict-fake {
        background: linear-gradient(135deg, rgba(255, 59, 48, 0.08), rgba(255, 100, 100, 0.03));
        border: 2px solid rgba(255, 59, 48, 0.3);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 0 40px rgba(255, 59, 48, 0.1);
    }

    .verdict-label {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
    }

    .verdict-real .verdict-label {
        color: #00ff88;
    }

    .verdict-fake .verdict-label {
        color: #ff3b30;
    }

    .confidence-text {
        font-size: 1.3rem;
        color: #a0aec0;
        margin-top: 8px;
    }

    /* Artifact Scores */
    .artifact-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 12px;
    }

    .artifact-title {
        font-weight: 700;
        color: #e2e8f0;
        font-size: 0.95rem;
        margin-bottom: 4px;
    }

    .artifact-desc {
        color: #718096;
        font-size: 0.82rem;
        line-height: 1.4;
    }

    /* Risk Badge */
    .risk-critical { color: #ff3b30; font-weight: 800; }
    .risk-high { color: #ff9500; font-weight: 700; }
    .risk-medium { color: #ffcc00; font-weight: 600; }
    .risk-low { color: #00ff88; font-weight: 600; }

    /* Engine Tags */
    .engine-tag {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
    }

    .engine-ml {
        background: rgba(0, 212, 255, 0.15);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.3);
    }

    .engine-gemini {
        background: rgba(123, 47, 247, 0.15);
        color: #a78bfa;
        border: 1px solid rgba(123, 47, 247, 0.3);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 35, 0.95);
        border-right: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Upload Area */
    .stFileUploader {
        border-radius: 16px;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 600;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0, 212, 255, 0.3);
    }

    /* Agreement badge */
    .agreement-yes {
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    .agreement-no {
        background: rgba(255, 204, 0, 0.1);
        color: #ffcc00;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
    }

    /* Powered by text */
    .powered-by {
        text-align: center;
        color: #4a5568;
        font-size: 0.8rem;
        margin-top: 40px;
        padding: 20px;
    }
</style>
""", unsafe_allow_html=True)


# ─── Initialize Model ────────────────────────────────────────
@st.cache_resource
def init_model():
    """Load model on first run."""
    # Check multiple possible checkpoint paths
    paths = [
        os.path.join("backend", "checkpoints", "checkpoint_phase2.pth"),
        os.path.join("checkpoints", "checkpoint_phase2.pth"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "..", "AI-Generated-Deepfake-Image-Detector",
                     "AI Images Detector", "checkpoints", "checkpoint_phase2.pth"),
    ]

    checkpoint = None
    for p in paths:
        if os.path.exists(p):
            checkpoint = p
            break

    model = load_model(checkpoint)
    configure_gemini()
    return model


init_model()

# ─── Session State ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_real" not in st.session_state:
    st.session_state.total_real = 0
if "total_fake" not in st.session_state:
    st.session_state.total_fake = 0


# ─── Hero Section ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🛡️ DeepSight AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Dual-Engine AI-Generated Image & Deepfake Detector — Powered by ConvNeXtV2 + Gemini Vision</p>', unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    threshold = st.slider(
        "Detection Threshold",
        0.0, 1.0, 0.50, 0.01,
        help="Higher = stricter (more confidence needed to label as Fake)"
    )

    use_gemini = st.toggle("🔍 Gemini Forensics", value=True,
                            help="Enable Gemini 2.0 Flash deep forensic analysis")

    gemini_key = st.text_input("Gemini API Key", type="password", 
                                placeholder="Enter key to enable forensics",
                                help="Get free key: aistudio.google.com/apikey")
    if gemini_key:
        configure_gemini(gemini_key)
        st.success("✅ Gemini configured!")

    st.divider()

    st.markdown("### 📊 Session Stats")
    col1, col2 = st.columns(2)
    col1.metric("✅ Real", st.session_state.total_real)
    col2.metric("❌ Fake", st.session_state.total_fake)

    total = st.session_state.total_real + st.session_state.total_fake
    if total > 0:
        fake_pct = st.session_state.total_fake / total * 100
        st.progress(fake_pct / 100, text=f"{fake_pct:.0f}% Fake detected")

    st.divider()
    st.markdown("### 🧠 About")
    st.markdown("""
    **Engine 1:** ConvNeXtV2-Base  
    Trained on 400K+ images  
    Detects: DALL-E3, FLUX, Midjourney, SDXL, StyleGAN2+
    
    **Engine 2:** Gemini 2.0 Flash  
    Multi-dimensional forensic analysis  
    6-category artifact detection
    """)


# ─── Main Content ─────────────────────────────────────────────
tab_analyze, tab_batch, tab_history = st.tabs(["🔍 Analyze", "📊 Batch Mode", "📋 History"])


# ─── Tab: Single Analysis ─────────────────────────────────────
with tab_analyze:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Drag and drop or click to upload",
            type=["jpg", "jpeg", "png", "webp"],
            help="Supports JPG, PNG, WebP. Max 50MB.",
            label_visibility="collapsed",
        )

        from streamlit_paste_button import paste_image_button as pbutton
        paste_result = pbutton("📋 Paste from Clipboard", text_color="#a78bfa", background_color="rgba(123,47,247,0.1)")

        # Handle image input
        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except Exception:
                st.error("❌ Invalid image file")

        elif paste_result and paste_result.image_data is not None:
            try:
                image = paste_result.image_data.convert("RGB")
            except Exception:
                st.error("❌ Invalid pasted image")

        if image:
            st.image(image, caption="Input Image", use_container_width=True)

            analyze_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">📋 Results</div>', unsafe_allow_html=True)

        if image and analyze_btn:
            start_time = time.time()

            # ─── Engine 1: ML Classification ──────────
            with st.spinner("🧠 Engine 1: ConvNeXtV2 classifying..."):
                ml_result = classify_image(image, threshold)

            # ─── Grad-CAM ─────────────────────────────
            with st.spinner("🎨 Generating Grad-CAM heatmap..."):
                try:
                    heatmap_b64, raw_cam_b64 = generate_heatmap(image, target_class=1)
                except Exception as e:
                    st.warning(f"Grad-CAM unavailable: {e}")
                    heatmap_b64, raw_cam_b64 = None, None

            # ─── Engine 2: Gemini Forensics ───────────
            gemini_result = None
            if use_gemini:
                with st.spinner("🔍 Engine 2: Gemini forensic analysis..."):
                    try:
                        gemini_result = asyncio.run(analyze_image_forensically(image))
                    except Exception as e:
                        st.warning(f"Gemini unavailable: {e}")

            # ─── Combine Verdicts ─────────────────────
            combined = combine_verdicts(ml_result, gemini_result)
            processing_time = round(time.time() - start_time, 2)

            # Update stats
            if combined["final_label"] == "Fake":
                st.session_state.total_fake += 1
            else:
                st.session_state.total_real += 1

            # ─── Display Verdict ──────────────────────
            verdict_class = "verdict-fake" if combined["final_label"] == "Fake" else "verdict-real"
            verdict_emoji = "❌" if combined["final_label"] == "Fake" else "✅"

            st.markdown(f"""
            <div class="{verdict_class}">
                <p class="verdict-label">{verdict_emoji} {combined["final_label"]}</p>
                <p class="confidence-text">Confidence: {combined["final_confidence"]*100:.1f}%</p>
                <p style="margin-top: 8px;">
                    <span class="engine-tag engine-ml">🧠 ConvNeXtV2</span>
                    {"<span class='engine-tag engine-gemini'>🔍 Gemini</span>" if gemini_result else ""}
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # Risk + Agreement
            risk = combined["risk_level"]
            risk_class = f"risk-{risk.lower()}"
            st.markdown(f"""
            **Risk Level:** <span class="{risk_class}">{risk}</span> &nbsp;|&nbsp;
            **Engines:** {"<span class='agreement-yes'>✓ Agree</span>" if combined["agreement"] else "<span class='agreement-no'>⚠ Disagree</span>"} &nbsp;|&nbsp;
            **Time:** {processing_time}s
            """, unsafe_allow_html=True)

            st.divider()

            # ─── Engine Details ───────────────────────
            col_e1, col_e2 = st.columns(2)

            with col_e1:
                st.markdown("##### 🧠 Engine 1: ML Classification")
                st.metric("Label", ml_result["label"])
                st.progress(ml_result["fake_probability"],
                           text=f"Fake: {ml_result['fake_probability']*100:.1f}%")
                st.progress(ml_result["real_probability"],
                           text=f"Real: {ml_result['real_probability']*100:.1f}%")

            with col_e2:
                if gemini_result and gemini_result.get("confidence", 0) > 0:
                    st.markdown("##### 🔍 Engine 2: Gemini Forensics")
                    st.metric("Verdict", gemini_result.get("overall_verdict", "N/A"))
                    st.caption(gemini_result.get("explanation", ""))
                else:
                    st.markdown("##### 🔍 Engine 2: Gemini Forensics")
                    st.info("Set Gemini API key in sidebar to enable")

            st.divider()

            # ─── Grad-CAM Heatmap ─────────────────────
            if heatmap_b64:
                st.markdown("##### 🎨 Grad-CAM Heatmap — What the Model Sees")
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(image.resize((256, 256)), caption="Original", use_container_width=True)
                with col_heat:
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    st.image(heatmap_img, caption="Grad-CAM Overlay", use_container_width=True)

            st.divider()

            # ─── Artifact Breakdown ───────────────────
            if gemini_result and gemini_result.get("artifacts"):
                st.markdown("##### 🔬 Artifact Breakdown (6-Category Forensic Analysis)")

                artifacts = gemini_result["artifacts"]
                cols = st.columns(3)
                for i, artifact in enumerate(artifacts):
                    with cols[i % 3]:
                        score = artifact.get("score", 0)
                        severity = artifact.get("severity", "low")
                        category = artifact.get("category", "Unknown")
                        desc = artifact.get("description", "N/A")

                        # Color based on score
                        if score >= 70:
                            bar_color = "🔴"
                        elif score >= 40:
                            bar_color = "🟡"
                        else:
                            bar_color = "🟢"

                        st.markdown(f"""
                        <div class="artifact-card">
                            <div class="artifact-title">{bar_color} {category}</div>
                            <div class="artifact-desc">{desc}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(score / 100, text=f"{score}/100")

                st.divider()

                # Detailed Analysis
                if gemini_result.get("detailed_analysis"):
                    st.markdown("##### 📝 Detailed Forensic Analysis")
                    st.markdown(f"> {gemini_result['detailed_analysis']}")

            # Save to history
            st.session_state.history.append({
                "filename": uploaded_file.name if uploaded_file else "Pasted Image",
                "verdict": combined["final_label"],
                "confidence": combined["final_confidence"],
                "risk": combined["risk_level"],
                "time": processing_time,
            })

        elif not image:
            st.markdown("""
            <div class="glass-card" style="text-align: center; padding: 60px 20px;">
                <p style="font-size: 4rem; margin: 0;">🖼️</p>
                <p style="color: #8892b0; font-size: 1.1rem; margin-top: 16px;">
                    Upload an image to analyze
                </p>
                <p style="color: #4a5568; font-size: 0.85rem;">
                    Supports: JPG, PNG, WebP • Max 50MB
                </p>
            </div>
            """, unsafe_allow_html=True)


# ─── Tab: Batch Mode ──────────────────────────────────────────
with tab_batch:
    st.markdown('<div class="section-header">📊 Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown("Upload multiple images to analyze them all at once. Uses ML engine only for speed.")

    batch_files = st.file_uploader(
        "Upload images",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
        key="batch_upload"
    )

    if batch_files:
        st.info(f"📁 {len(batch_files)} images ready for analysis")

        if st.button("🚀 Analyze All", type="primary", use_container_width=True):
            results = []
            progress = st.progress(0, text="Analyzing...")

            for i, f in enumerate(batch_files[:20]):
                try:
                    img = Image.open(f).convert("RGB")
                    result = classify_image(img, threshold)
                    combined = combine_verdicts(result)
                    results.append({
                        "File": f.name,
                        "Verdict": combined["final_label"],
                        "Confidence": f"{combined['final_confidence']*100:.1f}%",
                        "Risk": combined["risk_level"],
                    })
                except Exception as e:
                    results.append({
                        "File": f.name,
                        "Verdict": "Error",
                        "Confidence": "N/A",
                        "Risk": str(e)[:30],
                    })
                progress.progress((i + 1) / len(batch_files), text=f"Analyzing {i+1}/{len(batch_files)}...")

            progress.empty()

            # Summary
            real_c = sum(1 for r in results if r["Verdict"] == "Real")
            fake_c = sum(1 for r in results if r["Verdict"] == "Fake")

            col1, col2, col3 = st.columns(3)
            col1.metric("Total", len(results))
            col2.metric("✅ Real", real_c)
            col3.metric("❌ Fake", fake_c)

            st.dataframe(results, use_container_width=True, hide_index=True)


# ─── Tab: History ─────────────────────────────────────────────
with tab_history:
    st.markdown('<div class="section-header">📋 Analysis History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.dataframe(
            list(reversed(st.session_state.history)),
            use_container_width=True,
            hide_index=True,
        )

        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.session_state.total_real = 0
            st.session_state.total_fake = 0
            st.rerun()
    else:
        st.info("No analysis history yet. Upload an image to get started!")


# ─── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="powered-by">
    🛡️ DeepSight AI — Built for Smart India Hackathon 2025<br>
    Powered by ConvNeXtV2 (400K images) + Gemini 2.0 Flash + Grad-CAM<br>
    Problem Statement 7: Image Classification & Artifact Identification for AI-Generated Images
</div>
""", unsafe_allow_html=True)
