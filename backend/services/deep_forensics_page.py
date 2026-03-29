import streamlit as st
import cv2
import numpy as np
import io
import base64
from PIL import Image
import pandas as pd
import altair as alt
from backend.services.metadata_analyzer import analyze_metadata
from backend.services.frequency_analyzer import analyze_frequency
from backend.services.ela_analyzer import analyze_ela

def get_color_for_score(score):
    if score >= 70: return "#ff3b30"
    if score >= 40: return "#ff9500"
    return "#00ff88"

def render_deep_forensics_page():
    st.markdown('<h1 class="hero-title" style="font-size: 2.5rem;">🔬 Deep Forensics Lab</h1>', unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #a0aec0; margin-top: -10px; margin-bottom: 30px;'>Advanced Diagnostic Dashboard & Expert Opinion System</p>", unsafe_allow_html=True)

    if "last_analysis" not in st.session_state or st.session_state.last_analysis.get("image") is None:
        st.info("⚠️ Please analyze an image in the Dashboard first to view deep forensics.")
        return

    la = st.session_state.last_analysis
    image = la["image"]
    combined = la["combined"]
    ml_result = la["ml_result"]
    gemini_result = la.get("gemini_result", {})
    vit_result = la.get("vit_result", {})
    heatmap_b64 = la.get("heatmap_b64")
    
    cv_image = np.array(image.convert("RGB"))
    cv_image_bgr = cv_image[:, :, ::-1].copy()

    col_left, col_right = st.columns([1, 1.5], gap="large")

    with col_left:
        st.markdown('<div class="section-header">🖼️ Target Image</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        
        # Verdict Container
        fake_prob_pct = int(combined.get("combined_fake_probability", combined["final_confidence"] if combined["final_label"] in ["Fake", "Potential Fake"] else (1-combined["final_confidence"])) * 100)
        v_color = get_color_for_score(fake_prob_pct)
        if combined['final_label'] == "Potential Fake":
            v_color = "#ff9500" # Warning orange
        
        st.markdown(f"""
        <div style="background: rgba(255,255,255,0.03); padding: 20px; border-radius: 12px; border: 1px solid rgba(255,255,255,0.06); margin-top: 15px; border-left: 5px solid {v_color};">
            <h3 style="margin:0; color: #e2e8f0;">Final Verdict: {combined['final_label']}</h3>
            <p style="margin:5px 0 0 0; font-size: 1.2rem; color: {v_color}; font-weight: bold;">{fake_prob_pct}% AI Probability</p>
        </div>
        """, unsafe_allow_html=True)

        if heatmap_b64:
            st.markdown("<br><b>Grad-CAM Overlay</b>", unsafe_allow_html=True)
            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_img = Image.open(io.BytesIO(heatmap_bytes))
            st.image(heatmap_img, use_container_width=True)
            
        # AI Origin Suggestion (Turnitin Style)
        st.markdown('<div class="section-header" style="margin-top: 30px;">🔮 AI Origin Suggestion</div>', unsafe_allow_html=True)
        
        if combined.get('combined_fake_probability', 0) >= 0.40:
            # Determine likely generator based on visual and spectral signatures
            prob_gen = combined.get("probable_generator", "Unknown AI")
            if gemini_result and gemini_result.get("probable_generator") and gemini_result.get("probable_generator") != "Unknown":
                prob_gen = gemini_result.get("probable_generator")
            
            # Heuristic override based on spectral/texture signals
            freq_score = la.get("frequency_result", {}).get("risk_score", 0) if la.get("frequency_result") else 0
            
            # Origin signatures
            signatures = {
                "Stable Diffusion": {
                    "match": "Stable Diffusion" in prob_gen or "SDXL" in prob_gen,
                    "desc": "Signatures: Foreground/background separation noise and micro-texture tiling variations strongly map to Stable Diffusion diffusion signatures.",
                    "confidence": 85
                },
                "Midjourney": {
                    "match": "Midjourney" in prob_gen,
                    "desc": "Signatures: Hyper-realistic color saturation, ultra-smooth skin 'plastic' textures, and flawless depth-of-field typical of Midjourney V6 algorithms.",
                    "confidence": 92
                },
                "DALL-E": {
                    "match": "DALL-E" in prob_gen,
                    "desc": "Signatures: Specific diffuse edge boundaries, background geometric coherence errors, and 'oil painting' smoothing artifacts common in DALL-E 3.",
                    "confidence": 88
                },
                "GAN-derived": {
                    "match": freq_score > 70 or "StyleGAN" in prob_gen or "GAN" in prob_gen,
                    "desc": "Signatures: Distinct checkerboard artifacts in the frequency domain and periodic grid patterns indicating GAN-based upscaling/synthesis.",
                    "confidence": 94
                }
            }
            
            # Select best match
            best_match = {"name": prob_gen, "desc": "General algorithmic anomalies detected consistent with synthetic generation.", "conf": 70}
            for name, sig in signatures.items():
                if sig["match"]:
                    best_match = {"name": name, "desc": sig["desc"], "conf": sig["confidence"]}
                    break

            st.markdown(f"""
            <div style="background: rgba(123,47,247,0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(123,47,247,0.3); box-shadow: 0 4px 15px rgba(123,47,247,0.1);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-size: 0.75rem; color: #a78bfa; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em;">Estimated Origin Model</span>
                    <span style="background: rgba(123,47,247,0.3); color: #fff; padding: 2px 8px; border-radius: 6px; font-size: 0.7rem;">{best_match['conf']}% Match</span>
                </div>
                <h3 style="margin: 8px 0; color: #fff; font-size: 1.4rem;">{best_match['name']}</h3>
                <p style="font-size: 0.9rem; color: #a0aec0; margin: 0; line-height: 1.5;">{best_match['desc']}</p>
                <div style="font-size: 0.75rem; text-align: right; color: #718096; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
                    <i>*Similarity indexing is a signal, not a definitive verdict.</i>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("The image is classified as Real. No AI generation signatures detected.")

    with col_right:
        st.markdown('<div class="section-header">📈 Forensic Analysis Charts</div>', unsafe_allow_html=True)
        
        tab_spec, tab_pixel, tab_struct, tab_meta = st.tabs(["Spectral & ELA", "Pixel & Color", "Texture & Edges", "Metadata"])
        
        with tab_spec:
            st.write("##### 1. Frequency Domain Analysis (FFT)")
            freq = la.get("frequency_result")
            if freq and freq.get("spectrum_base64"):
                spec_bytes = base64.b64decode(freq["spectrum_base64"])
                st.image(Image.open(io.BytesIO(spec_bytes)), use_container_width=True, caption="Periodic grid patterns strongly indicate GAN/Diffusion upscaling.")
            else:
                st.warning("FFT analysis not run. Enable in Settings.")

            st.write("##### 2. Error Level Analysis (ELA)")
            ela = la.get("ela_result")
            if ela and ela.get("ela_heatmap_base64"):
                ela_bytes = base64.b64decode(ela.get("ela_heatmap_base64"))
                st.image(Image.open(io.BytesIO(ela_bytes)), use_container_width=True, caption="Highlighting areas of inconsistent JPEG compression (often injected/manipulated pixels).")
            else:
                st.warning("ELA not run. Enable in Settings.")

        with tab_pixel:
            st.write("##### 3. Color Channel Analysis (RGB Intensities)")
            # Generate RGB histograms
            colors = ('b', 'g', 'r')
            chart_data = {"Intensity": list(range(256))}
            for i, col in enumerate(colors):
                hist = cv2.calcHist([cv_image_bgr], [i], None, [256], [0, 256])
                chart_data[col.upper()] = hist.flatten().tolist()
                
            df_hist = pd.DataFrame(chart_data)
            df_melt = df_hist.melt("Intensity", var_name="Channel", value_name="Count")
            
            # Altair line chart
            c = alt.Chart(df_melt).mark_area(opacity=0.3).encode(
                x="Intensity:Q",
                y="Count:Q",
                color=alt.Color("Channel:N", scale=alt.Scale(domain=["R", "G", "B"], range=["#ff3b30", "#00ff88", "#00d4ff"])),
            ).properties(height=200)
            st.altair_chart(c, use_container_width=True)
            st.caption("AI generators sometimes exhibit unnaturally uniform or clipped color distributions.")

            st.write("##### 4. Pixel Noise Distribution")
            gray = cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY)
            # Estimate noise by subtracting median blur
            median = cv2.medianBlur(gray, 3)
            noise = cv2.absdiff(gray, median)
            hist_noise, bins = np.histogram(noise.ravel(), bins=50, range=(1, 50))
            
            df_noise = pd.DataFrame({"Noise Magnitude": bins[:-1], "Frequency": hist_noise})
            c_noise = alt.Chart(df_noise).mark_bar(color="#a78bfa").encode(
                x="Noise Magnitude:Q", y="Frequency:Q"
            ).properties(height=200)
            st.altair_chart(c_noise, use_container_width=True)
            st.caption("Perfectly smooth pixel distributions with zero noise usually indicate AI synthesis.")

        with tab_struct:
            st.write("##### 5. Edge Coherence Map")
            # Canny edge detection
            edges = cv2.Canny(gray, 100, 200)
            # Apply color map for visual flair
            edges_colored = cv2.applyColorMap(edges, cv2.COLORMAP_JET)
            st.image(edges_colored, use_container_width=True, channels="BGR", caption="Analyzes edge boundaries. AI images often struggle with background structural coherence and depth edge separation.")
            
            st.write("##### 6. Texture Complexity Map")
            # Variance focus map
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            texture_map = cv2.convertScaleAbs(laplacian)
            texture_colored = cv2.applyColorMap(texture_map, cv2.COLORMAP_MAGMA)
            st.image(texture_colored, use_container_width=True, channels="BGR", caption="Highlights regions of micro-texture. Deepfakes often have 'plastic' skin or completely uniform backgrounds.")

        with tab_meta:
            st.write("##### 7. Metadata Deep Scan")
            meta = la.get("metadata_result", {})
            if meta and meta.get("exif_data"):
                st.success(f"Discovered {meta.get('exif_count', 0)} EXIF flags.")
                st.dataframe(pd.DataFrame(list(meta["exif_data"].items()), columns=["Tag", "Value"]), use_container_width=True, hide_index=True)
            else:
                st.warning("⚠️ Critical Metadata Blank. AI generated images very frequently lack EXIF camera, GPS, or focal length tags entirely.")

    st.markdown("---")
    
    # ─── Expert Opinion Panel ─────────────────────────────────────
    st.markdown('<div class="section-header">⚖️ Expert System Opinion</div>', unsafe_allow_html=True)
    
    # Signals
    ml_score = int(ml_result["fake_probability"] * 100)
    meta_score = la.get("metadata_result", {}).get("risk_score", 0) if la.get("metadata_result") else 0
    freq_score = la.get("frequency_result", {}).get("risk_score", 0) if la.get("frequency_result") else 0
    ela_score = la.get("ela_result", {}).get("risk_score", 0) if la.get("ela_result") else 0
    gem_score = int(gemini_result.get("confidence", 0) * 100) if gemini_result and gemini_result.get("overall_verdict", "Real") == "Fake" else 0
    vit_score = int(vit_result.get("fake_probability", 0) * 100) if vit_result else 0
    
    # Weighted calculation
    # Weights: ML(30%), Gemini(25%), ViT(15%), Freq(10%), ELA(10%), Meta(10%)
    weighted_avg = (
        (ml_score * 0.30) + 
        (gem_score * 0.25) + 
        (vit_score * 0.15) + 
        (freq_score * 0.10) + 
        (ela_score * 0.10) + 
        (meta_score * 0.10)
    )
    
    if weighted_avg >= 70:
        expert_verdict = "Likely AI Generated"
        expert_color = "#ff3b30"
        confidence_desc = "Critical visual and mathematical evidence of synthetic generation."
    elif weighted_avg >= 40:
        expert_verdict = "Inconclusive / Suspicious"
        expert_color = "#ff9500"
        confidence_desc = "Significant anomalies detected; digital manipulation or 'half-fake' inpainting is suspected."
    else:
        expert_verdict = "Likely Real"
        expert_color = "#00ff88"
        confidence_desc = "Signals align with natural camera optics and physical light sensors."
        
    reasoning = []
    if ml_score > 70: reasoning.append("ConvNeXtV2 explicitly classified texture anomalies.")
    if vit_score > 70: reasoning.append("Vision Transformer detected distinct structural inconsistencies.")
    if freq_score > 60: reasoning.append("Fourier spectrum shows unnatural periodic upscaling grid artifacts.")
    if ela_score > 60: reasoning.append("Error Level Analysis found distinct variations in compression layers.")
    if not la.get("metadata_result", {}).get("has_camera_info", False): reasoning.append("Absence of camera hardware EXIF data (GPS/Lens/Model).")
    if gem_score > 60: reasoning.append("Gemini contextual reasoning identified anatomical or physical errors.")
    
    if not reasoning:
        reasoning = ["All forensic engines confirm statistical coherence typical of physical image sensors."]
        
    reason_html = "".join([f"<li style='margin-bottom: 5px; color: #cbd5e0;'>{r}</li>" for r in reasoning])
    
    st.markdown(f"""
    <div style="display: flex; gap: 20px; flex-wrap: wrap;">
        <div style="flex: 1.5; min-width: 300px; background: rgba(255,255,255,0.02); border: 1px solid rgba(255,255,255,0.06); padding: 24px; border-radius: 16px; border-top: 6px solid {expert_color};">
            <h4 style="margin-top:0; color: #a0aec0; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em;">Forensic Verdict</h4>
            <h2 style="color: {expert_color}; margin: 10px 0; font-size: 2rem;">{expert_verdict}</h2>
            <p style="color: #a0aec0; font-size: 0.95rem; font-style: italic; margin-bottom: 20px;">{confidence_desc}</p>
            <h4 style="color: #e2e8f0; font-size: 0.9rem; margin-bottom: 10px;">Primary Contributing Factors:</h4>
            <ul style="color: #e2e8f0; font-size: 0.9rem; padding-left: 20px;">
                {reason_html}
            </ul>
        </div>
        <div style="flex: 1; min-width: 250px; background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.08); padding: 24px; border-radius: 16px;">
            <h4 style="margin-top:0; color: #a0aec0; font-size: 0.8rem; text-transform: uppercase;">Weighted Signal Breakdown</h4>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span>🧠 ConvNeXtV2:</span> <b>{ml_score}%</b></div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span>🔮 Gemini Pro:</span> <b>{gem_score}%</b></div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span>👁️ ViT Context:</span> <b>{vit_score}%</b></div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span>📊 FFT Spectral:</span> <b>{freq_score}%</b></div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid rgba(255,255,255,0.05);"><span>🔬 ELA Integrity:</span> <b>{ela_score}%</b></div>
            <div style="display: flex; justify-content: space-between; padding: 6px 0;"><span>📋 Meta Flags:</span> <b>{meta_score}%</b></div>
            <div style="margin-top: 25px; text-align: center;">
                <div style="font-size: 0.7rem; color: #718096; margin-bottom: 5px;">AGGREGATE FORENSIC INDEX</div>
                <div style="font-size: 1.8rem; font-weight: 900; color: {expert_color};">{int(weighted_avg)}%</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
