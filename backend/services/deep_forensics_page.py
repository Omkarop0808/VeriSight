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
    anatomy_result = la.get("anatomy_result", {})
    regional_result = la.get("regional_result", {})
    heatmap_b64 = la.get("heatmap_b64")
    vit_heatmap_b64 = la.get("vit_heatmap_b64")
    
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
            st.markdown("<br><b>ConvNeXtV2 Grad-CAM Overlay</b>", unsafe_allow_html=True)
            heatmap_bytes = base64.b64decode(heatmap_b64)
            st.image(Image.open(io.BytesIO(heatmap_bytes)), use_container_width=True)
            
        if vit_heatmap_b64:
            st.markdown("<br><b>Regional AI Detection Map (ViT)</b>", unsafe_allow_html=True)
            vit_heatmap_bytes = base64.b64decode(vit_heatmap_b64)
            st.image(Image.open(io.BytesIO(vit_heatmap_bytes)), use_container_width=True, caption="Highlights suspicious patches with higher AI probability.")
            
        # AI Origin Suggestion (Turnitin Style)
        st.markdown('<div class="section-header" style="margin-top: 30px;">🔮 AI Origin Suggestion</div>', unsafe_allow_html=True)
        
        if combined.get('combined_fake_probability', 0) >= 0.40 and combined.get("final_label") != "Likely Real":
            # Determine likely generator based on visual and spectral signatures
            prob_gen = combined.get("probable_generator", "")
            if gemini_result and gemini_result.get("probable_generator") and gemini_result.get("probable_generator") != "Unknown":
                prob_gen += " " + gemini_result.get("probable_generator")
            
            gem_desc = gemini_result.get("verbal_analysis", "").lower() if gemini_result else ""
            
            # Heuristic override based on spectral/texture signals
            freq_score = la.get("frequency_result", {}).get("risk_score", 0) if la.get("frequency_result") else 0
            has_checkerboard = la.get("frequency_result", {}).get("metrics", {}).get("grid_score", 0) > 40 if la.get("frequency_result") else False
            has_anatomy_errors = la.get("anatomy_result", {}).get("is_suspicious", False) if la.get("anatomy_result") else False
            
            # User-defined Forensic Fingerprint Mapping
            signatures = {
                "Stable Diffusion": {
                    "match": has_checkerboard or has_anatomy_errors or "tiling" in gem_desc or "repetition" in gem_desc or "stable diffusion" in prob_gen.lower(),
                    "desc": "Signatures: Checkerboard artifacts in FFT, tiling repetition, or severe anatomical boundary failures (e.g., extra fingers).",
                    "confidence": 85 if has_anatomy_errors else (92 if has_checkerboard else 75)
                },
                "Midjourney": {
                    "match": "smooth" in gem_desc or "dreamlike" in gem_desc or "painterly" in gem_desc or "midjourney" in prob_gen.lower(),
                    "desc": "Signatures: Hyper-realistic color saturation, overly smooth skin with dreamy soft-focus backgrounds, and painterly textures.",
                    "confidence": 88
                },
                "DALL-E 3": {
                    "match": "asymmetry" in gem_desc or "soft edge" in gem_desc or "photorealistic" in gem_desc or "dall-e" in prob_gen.lower(),
                    "desc": "Signatures: Slight facial asymmetry paired with perfect skin, subtle edge softening/halos, and highly coherent but hyper-clean geometric generations.",
                    "confidence": 82
                },
                "Modern Diffusion (Generic)": {
                    "match": freq_score > 60 and not has_checkerboard,
                    "desc": "Signatures: Complete absence of natural camera sensor noise. The noise pattern is unnaturally clean and uniform.",
                    "confidence": 70
                }
            }
            
            scored_matches = []
            for name, sig in signatures.items():
                if sig["match"]:
                    scored_matches.append({"name": name, "desc": sig["desc"], "conf": sig["confidence"]})
            
            # Sort by confidence
            scored_matches = sorted(scored_matches, key=lambda x: x["conf"], reverse=True)
            
            if len(scored_matches) > 0:
                best_match = scored_matches[0]
                second_match = scored_matches[1] if len(scored_matches) > 1 else None
            else:
                best_match = {"name": "Insufficient signals for origin identification", "desc": "The model detects synthesis but lacks specific architectural fingerprints (e.g. strong checkerboarding or specific aesthetic textures) to determine the exact generator.", "conf": 0}
                second_match = None

            alt_html = ""
            if second_match:
                alt_html = f"<div style='margin-top: 10px; padding-top: 10px; border-top: 1px dashed rgba(255,255,255,0.1);'><span style='font-size: 0.7rem; color: #a78bfa; text-transform: uppercase;'>Alternative Possibility</span><h4 style='margin: 4px 0 2px 0; font-size: 1rem; color: #e2e8f0;'>{second_match['name']} <span style='font-size: 0.8rem; color: #718096;'>({second_match['conf']}% Match)</span></h4><p style='font-size: 0.8rem; color: #718096; margin: 0;'>{second_match['desc']}</p></div>"

            # Removed indentation to prevent Streamlit 4-space markdown bug
            clean_html = f"""<div style="background: rgba(123,47,247,0.1); padding: 20px; border-radius: 12px; border: 1px solid rgba(123,47,247,0.3); box-shadow: 0 4px 15px rgba(123,47,247,0.1);">
<div style="display: flex; justify-content: space-between; align-items: center;">
<span style="font-size: 0.75rem; color: #a78bfa; text-transform: uppercase; font-weight: 800; letter-spacing: 0.05em;">Most Likely Origin</span>
<span style="background: rgba(123,47,247,0.3); color: #fff; padding: 2px 8px; border-radius: 6px; font-size: 0.7rem;">{best_match['conf']}% Match</span>
</div>
<h3 style="margin: 8px 0; color: #fff; font-size: 1.4rem;">{best_match['name']}</h3>
<p style="font-size: 0.9rem; color: #a0aec0; margin: 0; line-height: 1.5;">{best_match['desc']}</p>
{alt_html}
<div style="font-size: 0.75rem; text-align: right; color: #718096; margin-top: 12px; border-top: 1px solid rgba(255,255,255,0.05); padding-top: 8px;">
<i>*Similarity indexing is probabilistically derived from structural signatures.</i>
</div>
</div>"""
            st.markdown(clean_html, unsafe_allow_html=True)
            
            # Explicit Anatomy Override Note
            if la.get("anatomy_result", {}).get("is_suspicious", False):
                st.markdown("<div style='margin-top: 10px; padding: 12px; background: rgba(255,59,48,0.1); border-left: 3px solid #ff3b30; border-radius: 0 6px 6px 0;'><p style='margin: 0; font-size: 0.85rem; color: #ff3b30;'><b>🦴 Anatomical Override Triggered:</b> Generative models (e.g., Stable Diffusion, FLUX) are notorious for miscalculating human skeletal physics. The detected structural anomaly heavily biases the verdict towards AI synthesis.</p></div>", unsafe_allow_html=True)
        else:
            st.info("The image is classified as Real. No AI generation signatures detected.")

    with col_right:
        st.markdown('<div class="section-header">📈 Forensic Analysis Charts</div>', unsafe_allow_html=True)
        
        tab_spec, tab_pixel, tab_struct, tab_meta = st.tabs(["Spectral & ELA", "Pixel & Color", "Texture & Edges", "Metadata"])
        
        with tab_spec:
            col_fft, col_ela_cell = st.columns(2)
            with col_fft:
                st.write("##### 1. Frequency Domain (FFT)")
                freq = la.get("frequency_result")
                if freq and freq.get("spectrum_base64"):
                    spec_bytes = base64.b64decode(freq["spectrum_base64"])
                    st.image(Image.open(io.BytesIO(spec_bytes)), use_container_width=True, caption="Periodic grid artifacts.")
                else:
                    st.warning("FFT unavailable.")

            with col_ela_cell:
                st.write("##### 2. Error Level Analysis (ELA)")
                ela = la.get("ela_result")
                if ela and ela.get("ela_heatmap_base64"):
                    ela_bytes = base64.b64decode(ela.get("ela_heatmap_base64"))
                    st.image(Image.open(io.BytesIO(ela_bytes)), use_container_width=True, caption="Compression layer inconsistencies.")
                else:
                    st.warning("ELA unavailable.")
            
            st.divider()
            st.write("##### 3. Composite Image Interpreter (Structural & Regional)")
            if regional_result and regional_result.get("composite_b64"):
                comp_bytes = base64.b64decode(regional_result["composite_b64"])
                st.image(Image.open(io.BytesIO(comp_bytes)), use_container_width=True, caption="Variance Grid Division")
                
                # Dynamic interpreter
                partially_fake = vit_result.get("is_partially_fake", False) if vit_result else False
                if partially_fake or regional_result.get("is_inconsistent", False):
                    st.error("🚨 **Composite Synthesis Detected:** The platform identified distinct structural or noise-layer differences across sectors of this image. It is highly likely that one half (or specific subjects) are authentic photographs, while the other half was synthetically generated or deeply manipulated (e.g., Face-Swaps, Generative Fill).")
                else:
                    st.success("✅ **Spatially Coherent:** Noise distributions and lighting algorithms are consistent across all major quadrants. No 'half-fake/half-real' compositing detected.")
                
                st.info(f"V-Inconsistency Index: {regional_result['v_inconsistency']} | E-Inconsistency Index: {regional_result['e_inconsistency']}")
            else:
                st.warning("Regional analysis not run.")

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
            st.write("##### 5. Anatomical Consistency Check")
            if anatomy_result and anatomy_result.get("overlay_b64"):
                ana_bytes = base64.b64decode(anatomy_result["overlay_b64"])
                st.image(Image.open(io.BytesIO(ana_bytes)), use_container_width=True, caption=anatomy_result["explanation"])
                
                # Show flags
                f = anatomy_result.get("flags", {})
                cols = st.columns(3)
                cols[0].metric("Hands Detected", f.get("hand_count", 0))
                cols[1].metric("Fingers Count", f.get("total_fingers", 0))
                cols[2].metric("Limbs Detected", f.get("arms_count", 0) + f.get("legs_count", 0))
            else:
                st.warning("Anatomy check unavailable.")

            st.divider()
            st.write("##### 6. Edge & Texture Analysis")
            col_edge, col_tex = st.columns(2)
            with col_edge:
                edges = cv2.Canny(cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY), 100, 200)
                st.image(cv2.applyColorMap(edges, cv2.COLORMAP_JET), use_container_width=True, channels="BGR", caption="Edge Coherence Map")
            with col_tex:
                laplacian = cv2.Laplacian(cv2.cvtColor(cv_image_bgr, cv2.COLOR_BGR2GRAY), cv2.CV_64F)
                st.image(cv2.applyColorMap(cv2.convertScaleAbs(laplacian), cv2.COLORMAP_MAGMA), use_container_width=True, channels="BGR", caption="Texture Complexity Map")

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
    
    # Synchronize with backend score_combiner.py (Weighted + Max-Pooling)
    expert_score = int(la.get("combined", {}).get("combined_fake_probability", 0) * 100)
    weighted_avg = expert_score

    
    expert_verdict = la.get("combined", {}).get("final_label", "Unknown")
    
    if "AI Generated" in expert_verdict:
        expert_color = "#ff3b30"
        confidence_desc = "Critical visual and mathematical evidence of synthetic generation."
    elif "Inconclusive" in expert_verdict or "Suspicious" in expert_verdict or "Partially" in expert_verdict:
        expert_color = "#ff9500"
        confidence_desc = "Significant anomalies detected; digital manipulation or 'half-fake' inpainting is suspected."
    else:
        expert_color = "#00ff88"
        confidence_desc = "Signals align with natural camera optics and physical light sensors."
    
    reasoning = []
    if ml_score > 70: reasoning.append("ConvNeXtV2 explicitly classified texture anomalies.")
    if vit_score > 70: reasoning.append("Vision Transformer detected distinct regional AI artifacts.")
    if anatomy_result and anatomy_result.get("is_suspicious"): 
        for a in anatomy_result.get("anomalies", []): reasoning.append(f"🦴 {a}")
    if regional_result and regional_result.get("is_inconsistent"): reasoning.append("🧩 Composite detection flagged significant regional noise/compression variance.")
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
    
    # ─── Regional Anomaly Interpreter ─────────────────────────────
    st.markdown('<div class="section-header">🔍 Regional Local Inspector</div>', unsafe_allow_html=True)
    st.write("This table breaks down the image by specific subject regions to identify 'Local Gen' vs 'Global Auth' discrepancy.")
    
    analysis_rows = []
    
    # Check if Gemini output specific regional / face tracking details
    gem_inspect = gemini_result.get("regional_inspection", []) if isinstance(gemini_result, dict) else []
    
    if len(gem_inspect) > 0:
        for reg in gem_inspect:
            status = "🚨 FAKE" if reg.get("is_fake", False) else "✅ AUTHENTIC"
            ai_sig = "HIGH" if reg.get("confidence", 0) > 60 and reg.get("is_fake") else "LOW"
            if not reg.get("is_fake"): ai_sig = "LOW"
            
            analysis_rows.append({
                "Region": reg.get("subject_id", "Unknown Subject"),
                "AI Signal": ai_sig,
                "Artifacts": reg.get("explanation", "N/A"),
                "Status": status
            })
    else:
        # Fallback to ViT or Regional Result if Gemini hasn't generated regional data
        if regional_result and regional_result.get("anomalies"):
            for a in regional_result["anomalies"]:
                analysis_rows.append({"Region": a.get("region", "Unknown"), "AI Signal": "HIGH", "Artifacts": a.get("desc", "N/A"), "Status": "🚨 FAKE"})
        
        if vit_result and vit_result.get("is_partially_fake"):
            analysis_rows.append({"Region": "Global (Spectral)", "AI Signal": "MED", "Artifacts": "Grid-based upscaling artifacts", "Status": "🔬 SUSPICIOUS"})
            
        if not analysis_rows:
            analysis_rows.append({"Region": "General", "AI Signal": "LOW", "Artifacts": "No significant regional anomalies", "Status": "✅ AUTHENTIC"})
        
    st.table(pd.DataFrame(analysis_rows))

