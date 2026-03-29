"""PDF Report Generator — Creates professional forensic analysis reports.

Generates downloadable PDF reports with:
- Analysis verdict and confidence
- Grad-CAM heatmap visualization
- Artifact breakdown scores
- Metadata analysis
- Frequency analysis
- ELA results
- Recommendations
"""
import io
import base64
import json
import datetime
import importlib
from PIL import Image
from fpdf import FPDF


import re

def clean_text(text):
    """Ensure text is a string and handle basic cleaning by stripping/replacing Unicode/Emoji."""
    if not isinstance(text, str):
        return str(text)
    
    # Replace common meaningful emojis with text equivalents
    replacements = {
        "🔴": "[CRITICAL]", "🟡": "[WARNING]", "🔵": "[INFO]", "🟢": "[OK]",
        "✓": "[YES]", "✗": "[NO]", "🎯": "[TARGET]", "🧠": "[ML]",
        "🔬": "[SCIENCE]", "🔮": "[AI]", "📊": "[STATS]", "📋": "[META]",
        "🛡️": "[SEC]", "⚖️": "[VERDICT]"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    # Strip remaining emojis and unsupported unicode (allowing Latin-1, Basic Latin, punctuation)
    text = re.sub(r'[^\x00-\x7F\xA0-\xFF\u0100-\u017F\u2018-\u201D\u2013\u2014]', '', text)
    
    words = re.split(r'(\s+)', text)
    safe_words = []
    for w in words:
        if not re.match(r'\s', w) and len(w) > 45:
            chunked = " ".join([w[i:i+45] for i in range(0, len(w), 45)])
            safe_words.append(chunked)
        else:
            safe_words.append(w)
            
    return "".join(safe_words)


class ForensicReport(FPDF):
    """Custom PDF class with header/footer."""
    
    def cell(self, *args, **kwargs):
        """Override cell to ensure all text is cleaned and handle space errors."""
        if 'txt' in kwargs:
            kwargs['txt'] = clean_text(kwargs['txt'])
        elif 'text' in kwargs:
            kwargs['text'] = clean_text(kwargs['text'])
        elif len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = clean_text(args[2])
            args = tuple(args)
        
        try:
            super().cell(*args, **kwargs)
        except Exception as e:
            if "Not enough horizontal space" in str(e):
                self.set_x(10)
                super().cell(*args, **kwargs)
            else:
                raise e

    def multi_cell(self, *args, **kwargs):
        """Override multi_cell to ensure all text is cleaned and handle space errors."""
        if 'txt' in kwargs:
            kwargs['txt'] = clean_text(kwargs['txt'])
        elif 'text' in kwargs:
            kwargs['text'] = clean_text(kwargs['text'])
        elif len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = clean_text(args[2])
            args = tuple(args)
            
        try:
            super().multi_cell(*args, **kwargs)
        except Exception as e:
            if "Not enough horizontal space" in str(e):
                self.set_x(10)
                # If width was 0 or negative, force epw
                if len(args) > 0 and args[0] <= 0:
                    args = list(args)
                    args[0] = self.epw
                    args = tuple(args)
                super().multi_cell(*args, **kwargs)
            else:
                raise e

    def header(self):
        self.set_fill_color(10, 10, 26)
        self.rect(0, 0, 210, 297, 'F')
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(0, 212, 255)
        self.set_x(10)
        self.multi_cell(0, 15, "VeriSight AI", align="C")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(136, 146, 176)
        self.set_x(10)
        self.multi_cell(0, 6, "AI-Generated Image Forensic Analysis Report", align="C")
        self.ln(5)
        # Divider line
        self.set_draw_color(0, 212, 255)
        self.set_line_width(0.5)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(8)

    def footer(self):
        self.set_y(-20)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(74, 85, 104)
        self.set_x(10)
        self.multi_cell(0, 10, f"VeriSight AI Report | Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Page {self.page_no()}/{{nb}}", align="C")


def generate_report(
    image: Image.Image,
    combined_result: dict,
    ml_result: dict,
    gemini_result: dict = None,
    vit_result: dict = None,
    anatomy_result: dict = None,
    regional_result: dict = None,
    heatmap_b64: str = None,
    vit_heatmap_b64: str = None,
    metadata_result: dict = None,
    frequency_result: dict = None,
    ela_result: dict = None,
    filename: str = "Unknown",
    processing_time: float = 0.0,
) -> bytes:
    """
    Generate a comprehensive PDF forensic report.

    Returns:
        bytes of the PDF file
    """
    pdf = ForensicReport()
    # ─── Register Unicode Fonts ──────────────────────────
    import os
    font_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets", "fonts")
    
    # Use DejaVuSans for Unicode support
    try:
        pdf.add_font("DejaVu", "", os.path.join(font_path, "DejaVuSans.ttf"))
        pdf.add_font("DejaVu", "B", os.path.join(font_path, "DejaVuSans-Bold.ttf"))
        default_font = "DejaVu"
    except Exception as e:
        print(f"⚠️ Font loading failed: {e}. Falling back to Helvetica.")
        default_font = "Helvetica"

    pdf.alias_nb_pages()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=25)

    # ─── Report Info ─────────────────────────────────────
    pdf.set_font(default_font, "", 9)
    pdf.set_text_color(136, 146, 176)
    
    # Restrict filename length to physically avoid cell horizontal overflow
    safe_fn = filename if len(filename) < 40 else filename[:37] + "..."
    pdf.multi_cell(0, 5, f"File: {safe_fn}  |  Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}  |  Runtime: {processing_time}s")
    pdf.ln(2)

    # ─── Verdict Section ─────────────────────────────────
    verdict = combined_result.get("final_label", "Unknown")
    confidence = combined_result.get("final_confidence", 0)
    risk = combined_result.get("risk_level", "Unknown")

    # Verdict box
    if "AI Generated" in verdict or "Fake" in verdict:
        pdf.set_fill_color(40, 10, 10)
        pdf.set_text_color(255, 59, 48)
    elif "Suspicious" in verdict or "Partially" in verdict:
        pdf.set_fill_color(40, 25, 10)
        pdf.set_text_color(255, 149, 0)
    elif "Inconclusive" in verdict:
        pdf.set_fill_color(30, 30, 10)
        pdf.set_text_color(255, 204, 0)
    else:
        pdf.set_fill_color(10, 40, 20)
        pdf.set_text_color(0, 255, 136)

    # Dynamically shrink font for very long verdicts to prevent crashing
    verdict_text = f"VERDICT: {verdict.upper()}"
    if len(verdict_text) > 30:
        pdf.set_font(default_font, "B", 18)
    elif len(verdict_text) > 20:
        pdf.set_font(default_font, "B", 22)
    else:
        pdf.set_font(default_font, "B", 28)
        
    pdf.multi_cell(0, 20, verdict_text, align="C", fill=True)

    pdf.set_font(default_font, "", 14)
    pdf.set_text_color(160, 174, 192)
    pdf.multi_cell(0, 10, f"Confidence: {confidence*100:.1f}%  |  Risk Level: {risk}", align="C")
    
    # Final Expert Statement (If available)
    expert_statement = combined_result.get("expert_statement")
    if expert_statement:
        pdf.ln(5)
        pdf.set_fill_color(30, 20, 50)
        pdf.set_font(default_font, "B", 11)
        pdf.set_text_color(167, 139, 250)
        pdf.multi_cell(0, 10, "  ⚖️  INVESTIGATIVE CLOSING STATEMENT", fill=True, border=1, align="C")
        pdf.set_fill_color(25, 25, 25)
        pdf.set_font(default_font, "I", 10)
        pdf.set_text_color(226, 232, 240)
        pdf.multi_cell(0, 8, f"\"{expert_statement}\"", fill=True, border=1, align="L")
        
    pdf.ln(8)

    # ─── Input Image ─────────────────────────────────────
    _add_section_header(pdf, "Input Image")
    try:
        img_buffer = io.BytesIO()
        image.resize((400, 400)).save(img_buffer, format="PNG")
        img_buffer.seek(0)
        pdf.image(img_buffer, x=55, w=100)
        pdf.ln(5)
    except Exception:
        pdf.set_text_color(136, 146, 176)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 10, "[Image could not be embedded]")

    # ─── Engine 1: ML Classification ─────────────────────
    _add_section_header(pdf, "Engine 1: ConvNeXtV2 Classification")
    pdf.set_font(default_font, "", 10)
    pdf.set_text_color(226, 232, 240)
    _add_detail(pdf, "Model", "ConvNeXtV2-Base (trained on 400K+ images)")
    _add_detail(pdf, "Label", ml_result.get("label", "N/A"))
    _add_detail(pdf, "Confidence", f"{ml_result.get('confidence', 0)*100:.2f}%")
    _add_detail(pdf, "Real Probability", f"{ml_result.get('real_probability', 0)*100:.2f}%")
    _add_detail(pdf, "Fake Probability", f"{ml_result.get('fake_probability', 0)*100:.2f}%")
    pdf.ln(5)

    # ─── Grad-CAM Heatmap ────────────────────────────────
    if heatmap_b64:
        _add_section_header(pdf, "Grad-CAM Heatmap (ConvNeXtV2)")
        try:
            heatmap_bytes = base64.b64decode(heatmap_b64)
            pdf.image(io.BytesIO(heatmap_bytes), x=55, w=100)
            pdf.ln(3)
        except Exception: pass

    # ─── ViT Regional Heatmap ────────────────────────────
    if vit_heatmap_b64:
        _add_section_header(pdf, "Regional AI Detection Map (ViT)")
        try:
            vit_bytes = base64.b64decode(vit_heatmap_b64)
            pdf.image(io.BytesIO(vit_bytes), x=55, w=100)
            pdf.ln(3)
            pdf.set_font(default_font, "I", 8)
            pdf.set_text_color(113, 128, 150)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, "Vision Transformer patch-grid analysis highlighting AI-generated regions.", align="C")
        except Exception: pass
    pdf.ln(5)

    # ─── Engine 2: Gemini Forensics ──────────────────────
    if gemini_result and gemini_result.get("confidence", 0) > 0:
        _add_section_header(pdf, "Engine 2: Gemini Forensic Analysis")
        _add_detail(pdf, "Verdict", clean_text(gemini_result.get("overall_verdict", "N/A")))
        _add_detail(pdf, "Confidence", f"{gemini_result.get('confidence', 0)*100:.1f}%")
        _add_detail(pdf, "Probable Generator", clean_text(gemini_result.get("probable_generator", "Unknown")))

        pdf.ln(3)
        pdf.set_font(default_font, "I", 9)
        pdf.set_text_color(160, 174, 192)
        explanation = clean_text(gemini_result.get("explanation", ""))
        if explanation:
            pdf.multi_cell(0, 5, explanation)
        pdf.ln(5)

        # Artifact Breakdown
        artifacts = gemini_result.get("artifacts", [])
        if artifacts:
            _add_section_header(pdf, "6-Category Artifact Breakdown")
            for artifact in artifacts:
                category = clean_text(artifact.get("category", "Unknown"))
                score = artifact.get("score", 0)
                desc = clean_text(artifact.get("description", "N/A"))
                severity = clean_text(artifact.get("severity", "low"))

                # Color code by severity
                if severity == "critical":
                    pdf.set_text_color(255, 59, 48)
                elif severity == "high":
                    pdf.set_text_color(255, 149, 0)
                elif severity == "medium":
                    pdf.set_text_color(255, 204, 0)
                else:
                    pdf.set_text_color(0, 255, 136)

                pdf.set_font(default_font, "B", 10)
                safe_cat = category if len(category) < 50 else category[:47] + "..."
                safe_sev = severity[:10]
                pdf.multi_cell(0, 6, f"{safe_cat}: {score}/100 [{safe_sev.upper()}]")
                pdf.set_text_color(160, 174, 192)
                pdf.set_font(default_font, "", 9)
                pdf.multi_cell(0, 5, desc)
                pdf.ln(2)

        # Detailed Analysis
        detailed = clean_text(gemini_result.get("detailed_analysis", ""))
        if detailed:
            pdf.ln(3)
            _add_section_header(pdf, "Detailed Technical Analysis")
            pdf.set_font(default_font, "", 9)
            pdf.set_text_color(160, 174, 192)
            pdf.multi_cell(0, 5, detailed)

    # ─── Advanced Anatomical Check ───────────────────────
    if anatomy_result and anatomy_result.get("overlay_b64"):
        pdf.add_page()
        _add_section_header(pdf, "Anatomical Consistency Check (MediaPipe)")
        try:
            ana_bytes = base64.b64decode(anatomy_result["overlay_b64"])
            pdf.image(io.BytesIO(ana_bytes), x=55, w=100)
            pdf.ln(3)
            f = anatomy_result.get("flags", {})
            pdf.set_font(default_font, "B", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"Hands: {f.get('hand_count', 0)} | Fingers: {f.get('total_fingers', 0)} | Limbs: {f.get('arms_count', 0)+f.get('legs_count', 0)}", align="C")
            pdf.set_font(default_font, "I", 9)
            pdf.multi_cell(0, 5, clean_text(anatomy_result.get("explanation", "")))
        except Exception: pass
        pdf.ln(5)

    # ─── Regional Inconsistency ──────────────────────────
    if regional_result and regional_result.get("composite_b64"):
        _add_section_header(pdf, "Regional Inconsistency (Composite Detection)")
        try:
            reg_bytes = base64.b64decode(regional_result["composite_b64"])
            pdf.image(io.BytesIO(reg_bytes), x=55, w=100)
            pdf.ln(3)
            pdf.set_font(default_font, "B", 9)
            pdf.set_x(pdf.l_margin)
            pdf.multi_cell(0, 5, f"V-Inconsistency: {regional_result['v_inconsistency']} | E-Inconsistency: {regional_result['e_inconsistency']}", align="C")
            pdf.set_font(default_font, "I", 9)
            pdf.multi_cell(0, 5, clean_text(regional_result.get("explanation", "")))
        except Exception: pass
        pdf.ln(5)

    # ─── Metadata Analysis ───────────────────────────────
    if metadata_result:
        pdf.add_page()
        _add_section_header(pdf, "Metadata / EXIF Analysis")
        _add_detail(pdf, "Risk Score", f"{metadata_result.get('risk_score', 0)}/100")
        _add_detail(pdf, "EXIF Fields Found", str(metadata_result.get('exif_count', 0)))
        _add_detail(pdf, "Camera Info", "Yes" if metadata_result.get('has_camera_info') else "No")
        _add_detail(pdf, "GPS Data", "Yes" if metadata_result.get('has_gps') else "No")
        _add_detail(pdf, "Capture Date", "Yes" if metadata_result.get('has_datetime') else "No")

        pdf.ln(3)
        pdf.set_font(default_font, "I", 9)
        pdf.set_text_color(160, 174, 192)
        pdf.multi_cell(0, 5, clean_text(metadata_result.get("summary", "")))
        pdf.ln(5)

    # ─── Frequency Analysis ──────────────────────────────
    if frequency_result:
        _add_section_header(pdf, "Frequency Domain (FFT) Analysis")
        _add_detail(pdf, "Risk Score", f"{frequency_result.get('risk_score', 0)}/100")
        metrics = frequency_result.get("metrics", {})
        _add_detail(pdf, "High-Freq Ratio", f"{metrics.get('high_freq_ratio', 0):.4f}")
        _add_detail(pdf, "Grid Score", f"{metrics.get('grid_score', 0)}/100")
        _add_detail(pdf, "Peak Count", str(metrics.get('peak_count', 0)))

        pdf.ln(3)
        pdf.set_font(default_font, "I", 9)
        pdf.set_text_color(160, 174, 192)
        pdf.multi_cell(0, 5, clean_text(frequency_result.get("summary", "")))

        # FFT Spectrum image
        spectrum_b64 = frequency_result.get("spectrum_base64")
        if spectrum_b64:
            try:
                spec_bytes = base64.b64decode(spectrum_b64)
                spec_buffer = io.BytesIO(spec_bytes)
                pdf.image(spec_buffer, x=55, w=100)
                pdf.ln(3)
                pdf.set_font(default_font, "I", 8)
                pdf.set_text_color(113, 128, 150)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, "FFT Power Spectrum (center = low frequency, edges = high frequency)", align="C")
            except Exception:
                pass
        pdf.ln(5)

    # ─── ELA Analysis ────────────────────────────────────
    if ela_result:
        _add_section_header(pdf, "Error Level Analysis (ELA)")
        _add_detail(pdf, "Risk Score", f"{ela_result.get('risk_score', 0)}/100")
        metrics = ela_result.get("metrics", {})
        _add_detail(pdf, "Mean Error", f"{metrics.get('mean_error', 0):.4f}")
        _add_detail(pdf, "Std Error", f"{metrics.get('std_error', 0):.4f}")
        _add_detail(pdf, "Region Uniformity", f"{metrics.get('uniformity_ratio', 0):.4f}")

        pdf.ln(3)
        pdf.set_font(default_font, "I", 9)
        pdf.set_text_color(160, 174, 192)
        pdf.multi_cell(0, 5, clean_text(ela_result.get("summary", "")))

        ela_heatmap_b64 = ela_result.get("ela_heatmap_base64")
        if ela_heatmap_b64:
            try:
                ela_bytes = base64.b64decode(ela_heatmap_b64)
                ela_buffer = io.BytesIO(ela_bytes)
                pdf.image(ela_buffer, x=55, w=100)
                pdf.ln(3)
                pdf.set_font(default_font, "I", 8)
                pdf.set_text_color(113, 128, 150)
                pdf.set_x(pdf.l_margin)
                pdf.multi_cell(0, 5, "ELA Heatmap — brighter regions indicate compression inconsistencies", align="C")
            except Exception:
                pass

    # ─── Combined Verdict Summary ────────────────────────
    pdf.add_page()
    _add_section_header(pdf, "Combined Analysis Summary")

    agreement = combined_result.get("agreement", True)
    pdf.set_font(default_font, "", 10)
    pdf.set_text_color(226, 232, 240)
    _add_detail(pdf, "Final Verdict", verdict)
    _add_detail(pdf, "Final Confidence", f"{confidence*100:.1f}%")
    _add_detail(pdf, "Risk Level", risk)
    _add_detail(pdf, "ML Weight", f"{combined_result.get('ml_weight', 0.6)*100:.0f}%")
    _add_detail(pdf, "Gemini Weight", f"{combined_result.get('gemini_weight', 0.4)*100:.0f}%")
    _add_detail(pdf, "Engine Agreement", "✓ Yes" if agreement else "✗ No")

    pdf.ln(10)

    # Disclaimer
    pdf.set_font(default_font, "I", 8)
    pdf.set_text_color(74, 85, 104)
    pdf.multi_cell(0, 4,
        "DISCLAIMER: This report is generated by an automated AI system and should be used as a reference tool only. "
        "Results should be verified by a qualified digital forensics professional. False positives and false negatives "
        "are possible. The ConvNeXtV2 model was trained on specific datasets and may not generalize to all types of "
        "AI-generated content. Gemini analysis is contextual and may vary between runs."
    )

    # Output PDF
    output = pdf.output()
    return bytes(output)


def _add_section_header(pdf: FPDF, title: str):
    """Add a styled section header."""
    font_family = pdf.font_family if "DejaVu" in pdf.font_family else "Helvetica"
    pdf.set_font(font_family, "B", 13)
    pdf.set_text_color(0, 212, 255)
    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(0, 10, title)
    pdf.set_draw_color(0, 212, 255)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 120, pdf.get_y())
    pdf.ln(4)


def _add_detail(pdf: FPDF, label: str, value: str):
    """Add a key-value detail line with robust width handling."""
    font_family = pdf.font_family if "DejaVu" in pdf.font_family else "Helvetica"
    pdf.set_font(font_family, "B", 9)
    pdf.set_text_color(136, 146, 176)
    
    # Reset to left margin if not already there
    if pdf.get_x() > pdf.l_margin + 1:
        pdf.ln(5)
        
    pdf.cell(50, 5, f"{label}:")
    pdf.set_font(font_family, "", 9)
    pdf.set_text_color(226, 232, 240)
    
    # Use epw (effective page width) minus the 50 we used for label
    val_width = pdf.epw - 50
    pdf.multi_cell(val_width, 5, str(value))
