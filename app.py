"""
VeriSight AI — Streamlit Frontend v2.0
Premium glassmorphism dark-theme UI for AI-generated image detection.
Features: Dual-engine analysis, Grad-CAM, Metadata, FFT, ELA, PDF reports.
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
import gc
import numpy as np
import requests
from dotenv import load_dotenv

# ─── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="VeriSight AI — Deepfake Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ensure environment variables are loaded immediately for Windows users
load_dotenv(override=True)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backend.models.convnext import build_model
from backend.transforms import test_transforms
from backend.services.gradcam_service import generate_heatmap
from backend.services.gemini_forensics import configure_gemini, analyze_image_forensically, generate_expert_summary
# Gemini initialization is now done lazily within init_model() or as needed
from backend.services.score_combiner import combine_verdicts
from backend.services.classifier import load_model, classify_image, get_model_for_gradcam
from backend.services.metadata_analyzer import analyze_metadata
from backend.services.frequency_analyzer import analyze_frequency
from backend.services.ela_analyzer import analyze_ela
from backend.services.report_generator import generate_report
from backend.services.checkpoint_downloader import download_checkpoint, is_checkpoint_available
from backend.services.vit_service import analyze_vit_regional as analyze_vit, load_vit_model
from backend.services.anatomy_analyzer import analyze_anatomy
from backend.services.regional_analyzer import analyze_regional_inconsistency as analyze_regional

# ─── Custom CSS ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Styles - Shadcn/Magic UI Dark Mode (Neuroblastic) */
    .stApp {
        background: #09090b; /* Shadcn Dark */
        font-family: 'Inter', sans-serif;
        color: #e2e8f0;
    }

    /* Override Streamlit components to match Shadcn UI */
    .stButton > button {
        background-color: #fafafa !important;
        color: #18181b !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        border: none !important;
        box-shadow: 0 4px 14px 0 rgba(255, 255, 255, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background-color: #e4e4e7 !important;
        transform: translateY(-1px) !important;
    }
    
    /* Neobrutalist input fields */
    .stTextInput > div > div > input {
        background: transparent !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        border-radius: 6px !important;
        color: white !important;
    }

    /* Restoring Sidebar Toggle Visibility */
    header[data-testid="stHeader"] {
        visibility: visible !important;
        background: transparent !important;
        z-index: 999;
    }
    
    /* Style the sidebar toggle button (collapsedControl) */
    button[data-testid="base-control"] {
        color: #00d4ff !important;
        background: rgba(0, 212, 255, 0.1) !important;
        border-radius: 50% !important;
        padding: 5px !important;
    }

    #MainMenu, footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main Title */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 900;
        background: linear-gradient(135deg, #00d4ff 0%, #7b2ff7 40%, #ff6b9d 70%, #00d4ff 100%);
        background-size: 300% 300%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
        letter-spacing: -0.03em;
        animation: gradient-flow 6s ease infinite;
    }

    @keyframes gradient-flow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    .hero-subtitle {
        text-align: center;
        color: #8892b0;
        font-size: 1.05rem;
        font-weight: 400;
        margin-top: 0;
        margin-bottom: 2rem;
    }

    /* Engine badges in subtitle */
    .engine-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.72rem;
        font-weight: 600;
        margin: 0 3px;
        vertical-align: middle;
    }
    .badge-ml { background: rgba(0,212,255,0.12); color: #00d4ff; border: 1px solid rgba(0,212,255,0.25); }
    .badge-llm { background: rgba(123,47,247,0.12); color: #a78bfa; border: 1px solid rgba(123,47,247,0.25); }
    .badge-meta { background: rgba(255,107,157,0.12); color: #ff6b9d; border: 1px solid rgba(255,107,157,0.25); }
    .badge-fft { background: rgba(0,255,136,0.12); color: #00ff88; border: 1px solid rgba(0,255,136,0.25); }
    .badge-ela { background: rgba(255,204,0,0.12); color: #ffcc00; border: 1px solid rgba(255,204,0,0.25); }

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
        transition: all 0.3s ease;
    }
    .glass-card:hover {
        border-color: rgba(0, 212, 255, 0.15);
        box-shadow: 0 8px 40px rgba(0, 212, 255, 0.08);
    }

    /* Verdict Cards */
    .verdict-real {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.06), rgba(0, 200, 100, 0.02));
        border: 2px solid rgba(0, 255, 136, 0.25);
        border-radius: 24px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 0 60px rgba(0, 255, 136, 0.08);
        animation: verdict-glow-real 3s ease-in-out infinite;
    }
    @keyframes verdict-glow-real {
        0%, 100% { box-shadow: 0 0 40px rgba(0, 255, 136, 0.08); }
        50% { box-shadow: 0 0 80px rgba(0, 255, 136, 0.15); }
    }

    .verdict-fake {
        background: linear-gradient(135deg, rgba(255, 59, 48, 0.06), rgba(255, 100, 100, 0.02));
        border: 2px solid rgba(255, 59, 48, 0.25);
        border-radius: 24px;
        padding: 32px;
        text-align: center;
        box-shadow: 0 0 60px rgba(255, 59, 48, 0.08);
        animation: verdict-glow-fake 3s ease-in-out infinite;
    }
    @keyframes verdict-glow-fake {
        0%, 100% { box-shadow: 0 0 40px rgba(255, 59, 48, 0.08); }
        50% { box-shadow: 0 0 80px rgba(255, 59, 48, 0.15); }
    }

    .verdict-label {
        font-size: 2.8rem;
        font-weight: 900;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .verdict-real .verdict-label { color: #00ff88; }
    .verdict-fake .verdict-label { color: #ff3b30; }

    .confidence-text {
        font-size: 1.2rem;
        color: #a0aec0;
        margin-top: 8px;
    }

    /* Artifact Scores */
    .artifact-card {
        background: rgba(255, 255, 255, 0.025);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 16px;
        padding: 18px;
        margin-bottom: 12px;
        transition: all 0.2s ease;
    }
    .artifact-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(0, 212, 255, 0.15);
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
        line-height: 1.5;
    }

    .artifact-region {
        color: #4a5568;
        font-size: 0.75rem;
        font-style: italic;
        margin-top: 4px;
    }

    /* Risk Badge */
    .risk-critical { color: #ff3b30; font-weight: 800; text-transform: uppercase; }
    .risk-high { color: #ff9500; font-weight: 700; }
    .risk-medium { color: #ffcc00; font-weight: 600; }
    .risk-low { color: #00ff88; font-weight: 600; }

    /* Engine Tags */
    .engine-tag {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 4px;
    }
    .engine-ml {
        background: rgba(0, 212, 255, 0.12);
        color: #00d4ff;
        border: 1px solid rgba(0, 212, 255, 0.25);
    }
    .engine-gemini {
        background: rgba(123, 47, 247, 0.12);
        color: #a78bfa;
        border: 1px solid rgba(123, 47, 247, 0.25);
    }

    /* Sidebar Styles */
    section[data-testid="stSidebar"] {
        background: #09090b !important; /* Match main background */
        border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Ensure the sidebar toggle icon is white/visible */
    button[kind="header"] {
        color: white !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 12px;
        padding: 4px;
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
        border-radius: 14px;
        padding: 16px;
    }

    /* Section headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 2px solid rgba(0, 212, 255, 0.2);
    }

    /* Sub-section headers */
    .sub-header {
        font-size: 1rem;
        font-weight: 600;
        color: #a0aec0;
        margin-bottom: 10px;
        margin-top: 16px;
    }

    /* Finding cards */
    .finding-warning {
        background: rgba(255, 149, 0, 0.08);
        border-left: 3px solid #ff9500;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .finding-critical {
        background: rgba(255, 59, 48, 0.08);
        border-left: 3px solid #ff3b30;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #e2e8f0;
    }
    .finding-info {
        background: rgba(0, 212, 255, 0.06);
        border-left: 3px solid #00d4ff;
        padding: 10px 14px;
        border-radius: 0 8px 8px 0;
        margin: 6px 0;
        font-size: 0.85rem;
        color: #e2e8f0;
    }

    /* Agreement badge */
    .agreement-yes {
        background: rgba(0, 255, 136, 0.1);
        color: #00ff88;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }
    .agreement-no {
        background: rgba(255, 204, 0, 0.1);
        color: #ffcc00;
        padding: 5px 14px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.82rem;
        display: inline-block;
    }

    /* Score ring */
    .score-ring {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        font-weight: 800;
        font-size: 1.1rem;
        margin: 4px;
    }
    .score-high { background: rgba(255,59,48,0.15); color: #ff3b30; border: 2px solid rgba(255,59,48,0.3); }
    .score-medium { background: rgba(255,204,0,0.15); color: #ffcc00; border: 2px solid rgba(255,204,0,0.3); }
    .score-low { background: rgba(0,255,136,0.15); color: #00ff88; border: 2px solid rgba(0,255,136,0.3); }

    /* Stats bar */
    .stats-bar {
        display: flex;
        gap: 12px;
        margin: 8px 0;
    }
    .stat-chip {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 8px 14px;
        font-size: 0.82rem;
        color: #a0aec0;
        flex: 1;
        text-align: center;
    }
    .stat-chip strong {
        color: #e2e8f0;
        display: block;
        font-size: 1.1rem;
    }

    /* Powered by text */
    .powered-by {
        text-align: center;
        color: #4a5568;
        font-size: 0.78rem;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid rgba(255,255,255,0.04);
    }

    /* Pulsing dot */
    .pulse-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #00ff88;
        animation: pulse 2s ease-in-out infinite;
        margin-right: 6px;
        vertical-align: middle;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(0.8); }
    }
</style>
""", unsafe_allow_html=True)


# init_model is no longer called globally to save 700MB+ RAM during boot.
# It is now called lazily by the classification service.

# ─── Session State ────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []
if "total_real" not in st.session_state:
    st.session_state.total_real = 0
if "total_fake" not in st.session_state:
    st.session_state.total_fake = 0
if "current_page" not in st.session_state:
    st.session_state.current_page = "🏠 Dashboard"


# ─── Hero Section ─────────────────────────────────────────────
st.markdown('<h1 class="hero-title">🛡️ VeriSight AI</h1>', unsafe_allow_html=True)
st.markdown("""
<p class="hero-subtitle">
    Multi-Engine AI-Generated Image & Deepfake Forensic Platform<br>
    <span class="engine-badge badge-ml">🧠 ConvNeXtV2</span>
    <span class="engine-badge badge-llm">🔮 Gemini</span>
    <span class="engine-badge badge-meta">📋 Metadata</span>
    <span class="engine-badge badge-fft">📊 FFT</span>
    <span class="engine-badge badge-ela">🔬 ELA</span>
</p>
""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧭 Navigation")
    
    # Sync radio with session state for cross-page navigation
    page_options = ["🏠 Dashboard", "🔬 Deep Forensics Lab"]
    try:
        current_index = page_options.index(st.session_state.current_page)
    except ValueError:
        current_index = 0
        
    page_selection = st.radio(
        "Go to", 
        page_options, 
        index=current_index,
        label_visibility="collapsed",
        key="nav_radio"
    )
    st.session_state.current_page = page_selection
    st.divider()

    st.markdown("### ⚙️ Analysis Settings")

    threshold = st.slider(
        "Detection Threshold",
        0.0, 1.0, 0.50, 0.01,
        help="Higher = stricter (more confidence needed to label as Fake)"
    )

    use_gemini = st.toggle("🔮 Gemini Forensics", value=True,
                            help="Enable Gemini deep forensic analysis")

    run_metadata = st.toggle("📋 Metadata Analysis", value=True,
                              help="Analyze EXIF data and image properties")
    run_frequency = st.toggle("📊 Frequency (FFT)", value=True,
                               help="FFT spectral analysis for GAN patterns")
    run_ela = st.toggle("🔬 Error Level Analysis", value=True,
                         help="ELA for compression inconsistencies")

    st.divider()

    st.markdown("### 🔑 API Configuration")
    # 1. Fetch from Environment/Files (.env or Streamlit Secrets)
    env_key = os.getenv("GEMINI_API_KEY", "")
    
    # 2. UI Input (Session State used to persist manual overrides)
    if "manual_api_key" not in st.session_state:
        st.session_state.manual_api_key = env_key

    gemini_key = st.text_input(
        "Gemini API Key", 
        type="password", 
        value=st.session_state.manual_api_key,
        placeholder="Enter key for forensics",
        help="Get free: aistudio.google.com/apikey"
    )
    
    # 3. Dynamic Configuration & Status
    if gemini_key:
        st.session_state.manual_api_key = gemini_key
        configure_gemini(gemini_key)
        
        if gemini_key == env_key and env_key != "":
            st.success("✅ Gemini configured! (Using .env)")
        else:
            st.success("✅ Gemini configured! (Using Manual Entry)")
            if st.button("🔄 Reset to .env key", use_container_width=True):
                st.session_state.manual_api_key = env_key
                st.rerun()
    else:
        st.warning("⚠️ Enter Gemini API Key to enable Engine 2")

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
    <span class="pulse-dot"></span> **7-Signal Forensic Pipeline**

    **Forensic Engine 1:** ConvNeXtV2-Base
    Trained on 400K+ images — detects DALL-E3, FLUX, Midjourney, SDXL, StyleGAN2+

    **Forensic Engine 2:** Gemini 2.5 Pro
    Contextual reasoning, anatomical validation, and AI origin indexing

    **Forensic Engine 3:** Vision Transformer (ViT)
    Deep structural anomaly mapping and contextual coherence check

    **Forensic Signal 4:** FFT Spectral Analysis
    Detection of periodic upscaling artifacts in the frequency domain

    **Forensic Signal 5:** ELA (Error Level Analysis)
    JPEG compression layer consistency and local modification mapping

    **Forensic Signal 6:** Metadata / EXIF Scan
    Lens physics, GPS, and creation tool signature verification

    **Forensic Signal 7:** Texture & Noise Profiling
    Analysis of pixel-level noise distributions and surface smoothness
    """, unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────
def render_findings(findings: list):
    """Render analysis findings with styled cards."""
    for f in findings:
        css_class = f"finding-{f.get('type', 'info')}"
        icon = {"critical": "🔴", "warning": "🟡", "info": "🔵"}.get(f["type"], "ℹ️")
        st.markdown(f'<div class="{css_class}">{icon} {f["message"]}</div>', unsafe_allow_html=True)


def get_score_class(score: int) -> str:
    if score >= 60:
        return "score-high"
    elif score >= 30:
        return "score-medium"
    return "score-low"


def load_image_from_url(url: str) -> Image.Image:
    """Download and open image from URL."""
    response = requests.get(url, timeout=15, stream=True)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")


# ─── Main Content ─────────────────────────────────────────────
if st.session_state.current_page == "🔬 Deep Forensics Lab":
    from backend.services.deep_forensics_page import render_deep_forensics_page
    render_deep_forensics_page()
    st.stop()

tab_analyze, tab_forensics, tab_batch, tab_history = st.tabs([
    "🔍 Analyze", "🔬 Technical Details", "📊 Batch Mode", "📋 History"
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: Single Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_analyze:
    col_upload, col_result = st.columns([1, 1], gap="large")

    with col_upload:
        st.markdown('<div class="section-header">📤 Upload Image</div>', unsafe_allow_html=True)

        # Input methods
        input_method = st.radio(
            "Input method",
            ["📁 Upload File", "📸 Camera", "🌐 URL", "📋 Paste"],
            horizontal=True,
            label_visibility="collapsed",
        )

        image = None
        source_name = "Unknown"

        if input_method == "📁 Upload File":
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload",
                type=["jpg", "jpeg", "png", "webp"],
                help="Supports JPG, PNG, WebP. Max 50MB.",
                label_visibility="collapsed",
            )
            if uploaded_file is not None:
                try:
                    image = Image.open(uploaded_file).convert("RGB")
                    source_name = uploaded_file.name
                except Exception:
                    st.error("❌ Invalid image file")

        elif input_method == "📸 Camera":
            camera_file = st.camera_input("Take Live Photo", label_visibility="collapsed")
            if camera_file is not None:
                try:
                    image = Image.open(camera_file).convert("RGB")
                    source_name = "Live Camera Capture"
                except Exception:
                    st.error("❌ Invalid camera capture")

        elif input_method == "🌐 URL":
            url = st.text_input("Image URL", placeholder="https://example.com/image.jpg")
            if url:
                try:
                    with st.spinner("Downloading image..."):
                        image = load_image_from_url(url)
                        source_name = url.split("/")[-1][:50]
                except Exception as e:
                    st.error(f"❌ Failed to load image: {e}")

        elif input_method == "📋 Paste":
            try:
                from streamlit_paste_button import paste_image_button as pbutton
                paste_result = pbutton("📋 Paste from Clipboard", text_color="#a78bfa",
                                        background_color="rgba(123,47,247,0.1)")
                if paste_result and paste_result.image_data is not None:
                    image = paste_result.image_data.convert("RGB")
                    source_name = "Pasted Image"
            except ImportError:
                st.info("Paste functionality requires: `pip install streamlit-paste-button`")

        if image:
            st.image(image, caption=f"Input: {source_name}", use_container_width=True)

            # Image info
            w, h = image.size
            st.markdown(f"""
            <div class="stats-bar">
                <div class="stat-chip"><strong>{w}×{h}</strong>Resolution</div>
                <div class="stat-chip"><strong>{(w*h)/1e6:.1f}MP</strong>Megapixels</div>
                <div class="stat-chip"><strong>{image.mode}</strong>Mode</div>
            </div>
            """, unsafe_allow_html=True)

            analyze_btn = st.button("🔬 Analyze Image", type="primary", use_container_width=True)

    with col_result:
        st.markdown('<div class="section-header">📋 Analysis Results</div>', unsafe_allow_html=True)

        using_cached = False
        if image and not analyze_btn:
            if "last_analysis" in st.session_state and st.session_state.last_analysis.get("source_name") == source_name:
                using_cached = True

        if image and (analyze_btn or using_cached):
            if using_cached:
                la = st.session_state.last_analysis
                ml_result = la.get("ml_result")
                heatmap_b64 = la.get("heatmap_b64")
                raw_cam_b64 = la.get("raw_cam_b64")
                gemini_result = la.get("gemini_result")
                vit_result = la.get("vit_result")
                anatomy_result = la.get("anatomy_result")
                regional_result = la.get("regional_result")
                metadata_result = la.get("metadata_result")
                frequency_result = la.get("frequency_result")
                ela_result = la.get("ela_result")
                combined = la.get("combined")
                processing_time = la.get("processing_time", 0.0)
                engines_used = len(combined.get("analysis_engines", ["ConvNeXtV2"]))
            else:
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
                    with st.spinner("🔮 Engine 2: Gemini forensic analysis..."):
                        try:
                            # configure_gemini() is already called in the sidebar via gemini_key input
                            gemini_result = asyncio.run(analyze_image_forensically(image))
                        except Exception as e:
                            st.warning(f"Gemini unavailable: {e}")
                            
                # ─── Engine 3: Vision Transformer ─────────
                vit_result = None
                with st.spinner("👁️ Engine 3: ViT Regional analysis..."):
                    try:
                        vit_result = analyze_vit(image)
                    except Exception as e:
                        st.warning(f"ViT unavailable: {e}")
    
                # ─── Anatomy Check ────────────────────────
                anatomy_result = None
                with st.spinner("🦴 Analyzing anatomical consistency..."):
                    try:
                        anatomy_result = analyze_anatomy(image)
                    except Exception as e:
                        st.warning(f"Anatomy check failed: {e}")
    
                # ─── Regional Inconsistency ───────────────
                regional_result = None
                with st.spinner("🧩 Checking regional inconsistencies..."):
                    try:
                        regional_result = analyze_regional(image)
                    except Exception as e:
                        st.warning(f"Regional check failed: {e}")
    
                # ─── Signal 3: Metadata ───────────────────
                metadata_result = None
                is_live_camera = False
                
                with st.spinner("📋 Analyzing metadata..."):
                    try:
                        raw_bytes = None
                        if input_method == "📁 Upload File" and uploaded_file:
                            uploaded_file.seek(0)
                            raw_bytes = uploaded_file.read()
                        elif input_method == "📸 Camera" and camera_file:
                            camera_file.seek(0)
                            raw_bytes = camera_file.read()
                            is_live_camera = True
                            
                        if run_metadata:
                            metadata_result = analyze_metadata(image, raw_bytes)
                            
                        # Ensure camera flag is passed to the engine
                        if is_live_camera:
                            if metadata_result is None:
                                metadata_result = {}
                            metadata_result["is_live_camera"] = True
                    except Exception as e:
                        st.warning(f"Metadata analysis failed: {e}")
    
                # ─── Signal 4: Frequency ──────────────────
                frequency_result = None
                if run_frequency:
                    with st.spinner("📊 FFT spectral analysis..."):
                        try:
                            frequency_result = analyze_frequency(image)
                        except Exception as e:
                            st.warning(f"Frequency analysis failed: {e}")
    
                # ─── Signal 5: ELA ────────────────────────
                ela_result = None
                if run_ela:
                    with st.spinner("🔬 Error Level Analysis..."):
                        try:
                            ela_result = analyze_ela(image)
                        except Exception as e:
                            st.warning(f"ELA failed: {e}")
    
                # ─── Combine Verdicts ─────────────────────
                combined = combine_verdicts(
                    ml_result, 
                    gemini_result, 
                    vit_result, 
                    metadata_result, 
                    frequency_result, 
                    ela_result,
                    anatomy_result,
                    regional_result,
                    threshold
                )
                
                # --- Expert Investigative Summary ---
                # Check if Gemini already provided a summary in the first pass to save quota
                expert_statement = gemini_result.get("expert_investigative_summary") if gemini_result else None
                
                if expert_statement:
                    combined["expert_statement"] = expert_statement
                elif not gemini_result or gemini_result.get("error"):
                    # CRITICAL: If Gemini failed the first time, skip the second call to save quota
                    combined["expert_statement"] = "Investigative summary skipped: Primary forensic engine is offline."
                else:
                    with st.spinner("🕵️ Finalizing expert investigative summary..."):
                        try:
                            signals = {
                                "ml_result": ml_result,
                                "vit_result": vit_result,
                                "gemini_result": gemini_result,
                                "anatomy_result": anatomy_result,
                                "regional_result": regional_result,
                                "metadata_result": metadata_result,
                                "combined": combined
                            }
                            expert_statement = asyncio.run(generate_expert_summary(signals))
                            combined["expert_statement"] = expert_statement
                        except Exception as e:
                            combined["expert_statement"] = f"Summary engine error: {str(e)}"
    
                processing_time = round(time.time() - start_time, 2)
    
                # Update stats (only when analyzing fresh)
                verdict_lbl = combined["final_label"].lower()
                if "ai" in verdict_lbl or "fake" in verdict_lbl or "suspicious" in verdict_lbl:
                    st.session_state.total_fake += 1
                else:
                    st.session_state.total_real += 1
    
                # Store results in session for Deep Forensics tab
                st.session_state.last_analysis = {
                    "image": image,
                    "ml_result": ml_result,
                    "gemini_result": gemini_result,
                    "vit_result": vit_result,
                    "anatomy_result": anatomy_result,
                    "regional_result": regional_result,
                    "metadata_result": metadata_result,
                    "frequency_result": frequency_result,
                    "ela_result": ela_result,
                    "combined": combined,
                    "heatmap_b64": heatmap_b64,
                    "raw_cam_b64": raw_cam_b64,
                    "vit_heatmap_b64": vit_result.get("heatmap_b64") if vit_result else None,
                    "processing_time": processing_time,
                    "source_name": source_name,
                }
                
                # Force garbage collection after heavy ML inference
                gc.collect()

            # ─── Display Turnitin-Style Verdict ──────────────────────
            # Calculate total AI probability percentage
            fake_prob_pct = int(combined.get("combined_fake_probability", 0) * 100)
            
            if "Suspicious" in combined["final_label"] or combined["final_label"] == "Partially AI Generated":
                color = "#ff9500" # Orange
                idx_label = combined["final_label"]
                idx_desc = "Structural anomalies or regional AI generation detected."
            elif fake_prob_pct >= 55:
                color = "#ff3b30" # Red
                idx_label = "AI-Generated"
                idx_desc = "Highly likely generated by an AI model or deeply manipulated."
            elif fake_prob_pct >= 35:
                color = "#ffcc00" # Yellow
                idx_label = "Inconclusive"
                idx_desc = "Manual review recommended. Contains subtle algorithmic artifacts."
            elif "Possible False Positive" in combined["final_label"]:
                color = "#ff9500" # Orange
                idx_label = "Possible False Positive"
                idx_desc = "Image scored high for AI generation but contains strong physical hardware markers (EXIF, natural noise). Manual review recommended."
            else:
                color = "#00ff88" # Green
                idx_label = "Authentic (Real)"
                idx_desc = "No significant AI generation signatures detected."

            prob_gen = combined.get("probable_generator", "Unknown")
            source_text = f"<br><span style='color: #00d4ff; font-weight: 600;'>🎯 Detected Source: {prob_gen}</span>" if prob_gen and prob_gen != "Unknown" and color == "#ff3b30" else ""
            gemini_tag = "<span style='background: rgba(123,47,247,0.2); color: #c4b5fd; border: 1px solid rgba(123,47,247,0.4); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;'>🔮 Gemini</span>" if gemini_result and gemini_result.get("confidence", 0) > 0 else ""

            html_content = f"""
            <div style="display: flex; align-items: center; background: rgba(255,255,255,0.03); padding: 24px; border-radius: 16px; border-left: 8px solid {color}; margin-bottom: 24px; box-shadow: 0 4px 24px rgba(0,0,0,0.2);">
                <div style="min-width: 140px; text-align: center; border-right: 1px solid rgba(255,255,255,0.1); margin-right: 24px;">
                    <h1 style="font-size: 3.8rem; font-weight: 900; margin: 0; color: {color}; line-height: 1;">{fake_prob_pct}%</h1>
                </div>
                <div>
                    <h2 style="margin: 0 0 4px 0; color: #e2e8f0; font-size: 1.5rem; font-weight: 700;">AI Probability Index: {idx_label}</h2>
                    <p style="margin: 0; color: #a0aec0; font-size: 0.95rem; line-height: 1.4;">
                        {idx_desc}{source_text}
                    </p>
                    <div style="margin-top: 12px; display: flex; flex-wrap: wrap; gap: 8px;">
                        <span style="background: rgba(123,47,247,0.2); color: #c4b5fd; border: 1px solid rgba(123,47,247,0.4); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;">🧠 ConvNeXtV2</span>
                        <span style="background: rgba(123,47,247,0.2); color: #c4b5fd; border: 1px solid rgba(123,47,247,0.4); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;">👁️ Regional ViT</span>
                        <span style="background: rgba(0,212,255,0.1); color: #00d4ff; border: 1px solid rgba(0,212,255,0.3); padding: 4px 10px; border-radius: 12px; font-size: 0.8rem;">🦴 Anatomy Check</span>
                        {gemini_tag}
                    </div>
                </div>
            </div>
            """
            st.markdown(html_content, unsafe_allow_html=True)

            # Warning if physical photo metrics are heavily penalized
            if combined.get("false_positive_warning"):
                st.warning("⚠️ **System Warning:** The ConvNeXt/ViT models scored this image very high for synthesis, but DeepSight's forensic physics engine verified explicit hardware sensor data (EXIF / Natural Grain). Treat the AI prediction cautiously.")
                
            if gemini_result and gemini_result.get("overall_verdict", "").lower() == "fake" and combined.get("is_strong_real_photo"):
                st.warning("⚠️ **Note:** AI semantic analysis (Gemini) conflicts with metadata and noise analysis — treat with caution.")

            with st.expander("ℹ️ How is this verdict calculated?"):
                st.markdown("The **AI Probability Index** is a weighted ensemble score combining structural anomalies, texture analysis, regional patch-checking, and metadata risk. If an explicit anatomical anomaly (like an extra appendage) is found, it acts as a high-priority override, pushing the index directly to **Suspicious / AI-Generated**.")

            if combined.get("is_strong_real_photo") and color == "#00ff88":
                with st.expander("✅ Real Photo Indicators", expanded=True):
                    st.markdown("""
                    **The system definitively classified this as a Real Photograph because of these physical markers:**
                    - ✅ **Hardware EXIF Present:** The image contains embedded camera/smart-phone lens data.
                    - ✅ **Natural Sensor Noise:** Fast-Fourier Transform (FFT) detected organic shot noise, not synthetic checkerboard grids.
                    - ✅ **Consistent Compression:** Error Level Analysis shows uniform physical compression across regions.
                    - ✅ **Anatomical Coherence:** Passed rigid human-joint checks with no impossibilities detected.
                    """)


            # Expert Closing Statement Panel
            if combined.get("expert_statement"):
                st.markdown(f"""
                <div style="background: rgba(123, 47, 247, 0.05); border: 1px dashed rgba(123, 47, 247, 0.3); border-radius: 12px; padding: 20px; margin-bottom: 24px;">
                    <div style="display: flex; align-items: center; margin-bottom: 8px;">
                        <span style="font-size: 1.2rem; margin-right: 8px;">⚖️</span>
                        <strong style="color: #a78bfa; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.05em;">Final Investigative Closing Statement</strong>
                    </div>
                    <p style="margin: 0; color: #e2e8f0; font-size: 1rem; line-height: 1.6; font-style: italic;">
                        "{combined['expert_statement']}"
                    </p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("")

            # Risk + Agreement + Time
            risk = combined["risk_level"]
            risk_class = f"risk-{risk.lower()}"
            engines_used = len(combined.get("analysis_engines", ["ConvNeXtV2"]))

            st.markdown(f"""
            **Risk Level:** <span class="{risk_class}">{risk}</span> &nbsp;|&nbsp;
            **Engines:** {"<span class='agreement-yes'>✓ Agree</span>" if combined["agreement"] else "<span class='agreement-no'>⚠ Disagree</span>"} &nbsp;|&nbsp;
            **Time:** {processing_time}s &nbsp;|&nbsp;
            **Signals:** {engines_used}
            """, unsafe_allow_html=True)

            # Quick Navigation to Deep Forensics
            if st.button("🔬 Open Deep Forensics Lab", use_container_width=True, type="primary"):
                st.session_state.current_page = "🔬 Deep Forensics Lab"
                st.rerun()

            st.divider()

            # ─── Multi-Signal Diagnostic Chart ─────────────────
            st.markdown("##### 📊 Multi-Signal Diagnostic Breakdown")
            st.caption("Visual representation of risk levels evaluated across independent analytical layers.")
            
            # Construct dictionary for chart
            chart_data = {
                "Engine / Signal": ["ConvNeXtV2 (Texture)", "Metadata Risk", "FFT Risk", "ELA Risk"],
                "AI Risk Score": [
                    int(ml_result["fake_probability"] * 100),
                    metadata_result.get("risk_score", 0) if metadata_result else 0,
                    frequency_result.get("risk_score", 0) if frequency_result else 0,
                    ela_result.get("risk_score", 0) if ela_result else 0
                ]
            }
            if gemini_result and gemini_result.get("confidence", 0) > 0:
                chart_data["Engine / Signal"].insert(1, "Gemini (Anatomy)")
                gemini_risk = int(gemini_result["confidence"] * 100) if gemini_result.get("overall_verdict", "").lower() == "fake" else 100 - int(gemini_result["confidence"] * 100)
                chart_data["AI Risk Score"].insert(1, gemini_risk)
            
            # Map robust hex colors directly to prevent Altair condition parsing errors
            def get_color(score):
                if score >= 70: return "#ff3b30"
                if score >= 40: return "#ff9500"
                return "#00ff88"
                
            chart_data["BarColor"] = [get_color(s) for s in chart_data["AI Risk Score"]]

            import pandas as pd
            import altair as alt
            
            df_chart = pd.DataFrame(chart_data)
            
            # Use Altair for a beautiful horizontal bar chart
            chart = alt.Chart(df_chart).mark_bar(cornerRadiusEnd=4).encode(
                x=alt.X('AI Risk Score:Q', scale=alt.Scale(domain=[0, 100]), title='Fake Probability / Risk (%)'),
                y=alt.Y('Engine / Signal:N', sort='-x', title=''),
                color=alt.Color('BarColor:N', scale=None),
                tooltip=['Engine / Signal', 'AI Risk Score']
            ).properties(height=250)
            
            st.altair_chart(chart, use_container_width=True)

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
                    st.markdown("##### 🔮 Engine 2: Gemini Forensics")
                    st.metric("Verdict", gemini_result.get("overall_verdict", "N/A"))
                    st.caption(gemini_result.get("explanation", ""))
                else:
                    st.markdown("##### 🔮 Engine 2: Gemini Forensics")
                    err_msg = gemini_result.get("explanation", "Set Gemini API key in sidebar or .env to enable") if gemini_result else "Set Gemini API key in sidebar to enable"
                    st.info(err_msg)

            st.divider()

            # ─── Grad-CAM Heatmap ─────────────────────
            if heatmap_b64:
                st.markdown("##### 🎨 Grad-CAM — What the Model Sees")
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(image.resize((256, 256)), caption="Original", use_container_width=True)
                with col_heat:
                    heatmap_bytes = base64.b64decode(heatmap_b64)
                    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
                    st.image(heatmap_img, caption="Grad-CAM Overlay", use_container_width=True)
                
                with st.expander("ℹ️ How to interpret this heatmap"):
                    st.markdown("The **Grad-CAM (Gradient-weighted Class Activation Mapping)** visualization highlights the specific pixels the ConvNeXtV2 model focused on. **Warmer colors (red/orange)** indicate regions exhibiting high synthetic texture anomalies or blending errors.")

            st.divider()

            # ─── Artifact Breakdown ───────────────────
            if gemini_result and gemini_result.get("artifacts"):
                st.markdown("##### 🔬 6-Category Artifact Breakdown")
                st.caption("Deep inspection of lighting, physics, anatomy, and structural consistency.")

                artifacts = gemini_result["artifacts"]
                cols = st.columns(3)
                for i, artifact in enumerate(artifacts):
                    with cols[i % 3]:
                        score = artifact.get("score", 0)
                        severity = artifact.get("severity", "low")
                        category = artifact.get("category", "Unknown")
                        desc = artifact.get("description", "N/A")
                        regions = artifact.get("regions", "")

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
                            {"<div class='artifact-region'>📍 " + regions + "</div>" if regions and regions != "N/A" else ""}
                        </div>
                        """, unsafe_allow_html=True)
                        st.progress(score / 100, text=f"{score}/100")

                st.divider()

                # Detailed Analysis
                if gemini_result.get("detailed_analysis"):
                    st.markdown("##### 📝 Detailed Forensic Analysis")
                    st.markdown(f"> {gemini_result['detailed_analysis']}")

            st.divider()

            # ─── Auxiliary Signal Summaries ────────────
            if metadata_result or frequency_result or ela_result:
                st.markdown("##### 📡 Auxiliary Signal Summary")
                aux_cols = st.columns(3)

                if metadata_result:
                    with aux_cols[0]:
                        score = metadata_result.get("risk_score", 0)
                        sc = get_score_class(score)
                        st.markdown(f'<div class="score-ring {sc}">{score}</div>', unsafe_allow_html=True)
                        st.caption("📋 Metadata Risk")
                        st.caption(metadata_result.get("summary", "")[:100])

                if frequency_result:
                    with aux_cols[1]:
                        score = frequency_result.get("risk_score", 0)
                        sc = get_score_class(score)
                        st.markdown(f'<div class="score-ring {sc}">{score}</div>', unsafe_allow_html=True)
                        st.caption("📊 Frequency Risk")
                        st.caption(frequency_result.get("summary", "")[:100])

                if ela_result:
                    with aux_cols[2]:
                        score = ela_result.get("risk_score", 0)
                        sc = get_score_class(score)
                        st.markdown(f'<div class="score-ring {sc}">{score}</div>', unsafe_allow_html=True)
                        st.caption("🔬 ELA Risk")
                        st.caption(ela_result.get("summary", "")[:100])

            st.divider()

            # ─── PDF Download ─────────────────────────
            st.markdown("##### 📄 Download Report")
            try:
                pdf_bytes = generate_report(
                    image=image,
                    combined_result=combined,
                    ml_result=ml_result,
                    gemini_result=gemini_result,
                    heatmap_b64=heatmap_b64,
                    metadata_result=metadata_result,
                    frequency_result=frequency_result,
                    ela_result=ela_result,
                    filename=source_name,
                    processing_time=processing_time,
                )
                st.download_button(
                    "📥 Download PDF Forensic Report",
                    data=pdf_bytes,
                    file_name=f"VeriSight_Report_{source_name}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"PDF generation failed: {e}")

            # Save to history
            st.session_state.history.append({
                "filename": source_name,
                "verdict": combined["final_label"],
                "confidence": combined["final_confidence"],
                "risk": combined["risk_level"],
                "engines": engines_used,
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
                    Supports: JPG, PNG, WebP • Upload, paste, or enter URL
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.info("💡 **Tip:** Use the sidebar on the left to access the **Deep Forensics Lab**, adjust detection thresholds, or configure your Gemini API Key. Click the **>** icon in the top left if the sidebar is hidden.")
            
            if st.button("🔬 Go to Deep Forensics Lab Directly", use_container_width=True):
                st.session_state.current_page = "🔬 Deep Forensics Lab"
                st.rerun()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: Deep Forensics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_forensics:
    st.markdown('<div class="section-header">🔬 Deep Forensic Analysis</div>', unsafe_allow_html=True)

    if "last_analysis" in st.session_state:
        la = st.session_state.last_analysis

        # ─── Metadata Deep Dive ──────────────────────
        st.markdown('<div class="sub-header">📋 Metadata / EXIF Analysis</div>', unsafe_allow_html=True)

        if la.get("metadata_result"):
            meta = la["metadata_result"]

            col_meta1, col_meta2 = st.columns([1, 1])

            with col_meta1:
                st.metric("Metadata Risk Score", f"{meta['risk_score']}/100", help="Normal Range: 0-30. Scores > 70 strongly indicate the image was generated by an AI service that strips metadata.")
                st.metric("EXIF Fields Found", meta["exif_count"], help="Normal Range: 15+. Photographed images typically contain rich EXIF data (GPS, Lens, F-Stop). AI generators output 0.")
                st.metric("Camera Info", "✅ Yes" if meta["has_camera_info"] else "❌ No")
                st.metric("GPS Data", "✅ Yes" if meta["has_gps"] else "❌ No")
                st.metric("Capture Date", "✅ Yes" if meta["has_datetime"] else "❌ No")

            with col_meta2:
                st.markdown("**Findings:**")
                render_findings(meta.get("findings", []))

            # EXIF table
            if meta.get("exif_data"):
                with st.expander("📂 Raw EXIF Data", expanded=False):
                    exif_items = list(meta["exif_data"].items())[:30]
                    st.table({"Field": [k for k, v in exif_items],
                              "Value": [v[:80] for k, v in exif_items]})

            # Image properties
            props = meta.get("image_properties", {})
            if props:
                st.markdown(f"""
                <div class="stats-bar">
                    <div class="stat-chip"><strong>{props.get('width', '?')}×{props.get('height', '?')}</strong>Dimensions</div>
                    <div class="stat-chip"><strong>{props.get('megapixels', '?')}</strong>Megapixels</div>
                    <div class="stat-chip"><strong>{props.get('format', '?')}</strong>Format</div>
                    <div class="stat-chip"><strong>{props.get('aspect_ratio', '?')}</strong>Aspect</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Enable Metadata Analysis in sidebar and analyze an image first")

        st.divider()

        # ─── Frequency Analysis ──────────────────────
        st.markdown('<div class="sub-header">📊 Frequency Domain (FFT) Analysis</div>', unsafe_allow_html=True)

        if la.get("frequency_result"):
            freq = la["frequency_result"]

            col_f1, col_f2 = st.columns([1, 1])

            with col_f1:
                st.metric("Frequency Risk Score", f"{freq['risk_score']}/100", help="Normal Range: 0-30. Scores > 70 suggest synthetic high-frequency noise patterns.")
                metrics = freq.get("metrics", {})
                st.metric("High-Freq Ratio", f"{metrics.get('high_freq_ratio', 0):.4f}", help="Normal Range: ~0.01-0.15. AI generators often smear high frequencies, creating unnaturally smooth ratios.")
                st.metric("Grid Score", f"{metrics.get('grid_score', 0)}/100", help="Normal Range: 0-20. High grid scores indicate upscaling artifacts (checkerboard patterns) typical of GANs/Diffusion.")
                st.metric("Peak Anomalies", metrics.get("peak_count", 0), help="Normal Range: 0. Any isolated frequency peak usually indicates repeating manipulative patterns (like deepfake blending).")
                st.metric("Azimuthal Uniformity", f"{metrics.get('azimuthal_uniformity', 0):.1f}%", help="Normal Range: < 60%. Highly uniform frequencies (>80%) suggest artificial, perfectly generated noise.")

            with col_f2:
                spectrum_b64 = freq.get("spectrum_base64")
                if spectrum_b64:
                    spec_bytes = base64.b64decode(spectrum_b64)
                    spec_img = Image.open(io.BytesIO(spec_bytes))
                    st.image(spec_img, caption="FFT Power Spectrum (center=low freq, edges=high freq)",
                             use_container_width=True)

            st.markdown("**Findings:**")
            render_findings(freq.get("findings", []))
        else:
            st.info("Enable Frequency Analysis in sidebar and analyze an image first")

        st.divider()

        # ─── ELA Analysis ────────────────────────────
        st.markdown('<div class="sub-header">🔬 Error Level Analysis (ELA)</div>', unsafe_allow_html=True)

        if la.get("ela_result"):
            ela = la["ela_result"]

            col_e1, col_e2, col_e3 = st.columns(3)

            with col_e1:
                st.markdown("**Original**")
                st.image(la["image"].resize((256, 256)), use_container_width=True)

            with col_e2:
                ela_heatmap_b64 = ela.get("ela_heatmap_base64")
                if ela_heatmap_b64:
                    ela_bytes = base64.b64decode(ela_heatmap_b64)
                    ela_img = Image.open(io.BytesIO(ela_bytes))
                    st.markdown("**ELA Heatmap**")
                    st.image(ela_img, use_container_width=True)

            with col_e3:
                ela_overlay_b64 = ela.get("ela_overlay_base64")
                if ela_overlay_b64:
                    overlay_bytes = base64.b64decode(ela_overlay_b64)
                    overlay_img = Image.open(io.BytesIO(overlay_bytes))
                    st.markdown("**ELA Overlay**")
                    st.image(overlay_img, use_container_width=True)

            # ELA metrics
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                st.metric("ELA Risk Score", f"{ela['risk_score']}/100", help="Normal Range: 0-30. Risk > 70 indicates substantial saving/compression inconsistencies often found in spliced or AI-edited regions.")
                metrics = ela.get("metrics", {})
                st.metric("Mean Error", f"{metrics.get('mean_error', 0):.4f}", help="Normal Range: 2.0-6.0. Extraneous mean error points to heavily modified pixel compression blocks.")
                st.metric("Std Error", f"{metrics.get('std_error', 0):.4f}")

            with col_m2:
                st.metric("Region Uniformity", f"{metrics.get('uniformity_ratio', 0):.4f}")
                st.metric("Max Error", f"{metrics.get('max_error', 0):.4f}")
                st.metric("JPEG Quality Used", metrics.get("quality_used", 90))

            st.markdown("**Findings:**")
            render_findings(ela.get("findings", []))
        else:
            st.info("Enable ELA in sidebar and analyze an image first")

    else:
        st.markdown("""
        <div class="glass-card" style="text-align: center; padding: 60px 20px;">
            <p style="font-size: 4rem; margin: 0;">🔬</p>
            <p style="color: #8892b0; font-size: 1.1rem; margin-top: 16px;">
                Analyze an image in the "Analyze" tab first
            </p>
            <p style="color: #4a5568; font-size: 0.85rem;">
                Deep forensic results (Metadata, FFT, ELA) will appear here
            </p>
        </div>
        """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: Batch Mode
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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
        st.info(f"📁 {len(batch_files)} images ready for analysis (max 20)")

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
                        "Fake %": f"{result['fake_probability']*100:.1f}%",
                    })
                except Exception as e:
                    results.append({
                        "File": f.name,
                        "Verdict": "Error",
                        "Confidence": "N/A",
                        "Risk": str(e)[:30],
                        "Fake %": "N/A",
                    })
                progress.progress((i + 1) / min(len(batch_files), 20),
                                  text=f"Analyzing {i+1}/{min(len(batch_files), 20)}...")

            progress.empty()

            # Summary
            real_c = sum(1 for r in results if r["Verdict"] == "Real")
            fake_c = sum(1 for r in results if r["Verdict"] == "Fake")
            err_c = sum(1 for r in results if r["Verdict"] == "Error")

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", len(results))
            col2.metric("✅ Real", real_c)
            col3.metric("❌ Fake", fake_c)
            col4.metric("⚠️ Errors", err_c)

            st.dataframe(results, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB: History
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_history:
    st.markdown('<div class="section-header">📋 Analysis History</div>', unsafe_allow_html=True)

    if st.session_state.history:
        st.dataframe(
            list(reversed(st.session_state.history)),
            use_container_width=True,
            hide_index=True,
        )

        col_clear, col_export = st.columns(2)
        with col_clear:
            if st.button("🗑️ Clear History", use_container_width=True):
                st.session_state.history = []
                st.session_state.total_real = 0
                st.session_state.total_fake = 0
                st.rerun()
        with col_export:
            history_json = json.dumps(st.session_state.history, indent=2)
            st.download_button(
                "📥 Export History (JSON)",
                data=history_json,
                file_name="verisight_history.json",
                mime="application/json",
                use_container_width=True,
            )
    else:
        st.info("No analysis history yet. Upload an image to get started!")


# ─── Footer ──────────────────────────────────────────────────
st.markdown("""
<div class="powered-by">
    🛡️ <strong>VeriSight AI v2.0</strong> — Enterprise-grade Forensic Platform<br>
    5-Engine Pipeline: ConvNeXtV2 (400K images) + Gemini 3.1 Flash + Metadata + FFT + ELA + Grad-CAM<br>
    Detecting AI-Generated Images & Identifying Deepfake Artifacts.
</div>
""", unsafe_allow_html=True)
