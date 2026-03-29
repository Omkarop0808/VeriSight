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


def clean_text(text):
    """Ensure text is a string and handle basic cleaning without stripping Unicode/Emoji."""
    if not isinstance(text, str):
        return str(text)
    
    # We still keep common replacements for semantic clarity in some contexts,
    # but the font will handle the rendering.
    replacements = {
        "🟡": "🟡", "🔴": "🔴", "🔵": "🔵", "🟢": "🟢",
        "✓": "✓", "✗": "✗", "🎯": "🎯", "🧠": "🧠",
        "🔬": "🔬", "🔮": "🔮", "📊": "📊", "📋": "📋",
        "🛡️": "🛡️"
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    
    return text  # No longer encode to ASCII!


class ForensicReport(FPDF):
    """Custom PDF class with header/footer."""
    
    def cell(self, *args, **kwargs):
        """Override cell to ensure all text is cleaned."""
        if 'txt' in kwargs:
            kwargs['txt'] = clean_text(kwargs['txt'])
        elif 'text' in kwargs:
            kwargs['text'] = clean_text(kwargs['text'])
        elif len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = clean_text(args[2])
            args = tuple(args)
        super().cell(*args, **kwargs)

    def multi_cell(self, *args, **kwargs):
        """Override multi_cell to ensure all text is cleaned."""
        if 'txt' in kwargs:
            kwargs['txt'] = clean_text(kwargs['txt'])
        elif 'text' in kwargs:
            kwargs['text'] = clean_text(kwargs['text'])
        elif len(args) >= 3 and isinstance(args[2], str):
            args = list(args)
            args[2] = clean_text(args[2])
            args = tuple(args)
        super().multi_cell(*args, **kwargs)

    def header(self):
        self.set_fill_color(10, 10, 26)
        self.rect(0, 0, 210, 297, 'F')
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(0, 212, 255)
        self.cell(0, 15, "VeriSight AI", align="C", new_x="LMARGIN", new_y="NEXT")
        self.set_font("Helvetica", "", 10)
        self.set_text_color(136, 146, 176)
        self.cell(0, 6, "AI-Generated Image Forensic Analysis Report", align="C", new_x="LMARGIN", new_y="NEXT")
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
        self.cell(0, 10, f"VeriSight AI Report | Generated {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Page {self.page_no()}/{{nb}}", align="C")


def generate_report(
    image: Image.Image,
    combined_result: dict,
    ml_result: dict,
    gemini_result: dict = None,
    heatmap_b64: str = None,
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
    pdf.cell(0, 5, f"File: {filename}  |  Date: {datetime.datetime.now().strftime('%B %d, %Y at %H:%M')}  |  Processing: {processing_time}s", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(5)

    # ─── Verdict Section ─────────────────────────────────
    verdict = combined_result.get("final_label", "Unknown")
    confidence = combined_result.get("final_confidence", 0)
    risk = combined_result.get("risk_level", "Unknown")

    # Verdict box
    if verdict == "Fake":
        pdf.set_fill_color(40, 10, 10)
        pdf.set_text_color(255, 59, 48)
    else:
        pdf.set_fill_color(10, 40, 20)
        pdf.set_text_color(0, 255, 136)

    pdf.set_font(default_font, "B", 28)
    pdf.cell(0, 20, f"VERDICT: {verdict.upper()}", align="C", fill=True, new_x="LMARGIN", new_y="NEXT")

    pdf.set_font(default_font, "", 14)
    pdf.set_text_color(160, 174, 192)
    pdf.cell(0, 10, f"Confidence: {confidence*100:.1f}%  |  Risk Level: {risk}", align="C", new_x="LMARGIN", new_y="NEXT")
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
        pdf.cell(0, 10, "[Image could not be embedded]", new_x="LMARGIN", new_y="NEXT")

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
        _add_section_header(pdf, "Grad-CAM Heatmap (Model Attention)")
        try:
            heatmap_bytes = base64.b64decode(heatmap_b64)
            heatmap_buffer = io.BytesIO(heatmap_bytes)
            pdf.image(heatmap_buffer, x=55, w=100)
            pdf.ln(3)
            pdf.set_font(default_font, "I", 8)
            pdf.set_text_color(113, 128, 150)
            pdf.cell(0, 5, "Brighter regions indicate areas the model found most suspicious", align="C", new_x="LMARGIN", new_y="NEXT")
        except Exception:
            pass
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
                pdf.cell(0, 6, f"{category}: {score}/100 [{severity.upper()}]", new_x="LMARGIN", new_y="NEXT")
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
                pdf.cell(0, 5, "FFT Power Spectrum (center = low frequency, edges = high frequency)", align="C", new_x="LMARGIN", new_y="NEXT")
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
                pdf.cell(0, 5, "ELA Heatmap — brighter regions indicate compression inconsistencies", align="C", new_x="LMARGIN", new_y="NEXT")
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
    pdf.cell(0, 10, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_draw_color(0, 212, 255)
    pdf.set_line_width(0.3)
    pdf.line(10, pdf.get_y(), 120, pdf.get_y())
    pdf.ln(4)


def _add_detail(pdf: FPDF, label: str, value: str):
    """Add a key-value detail line."""
    font_family = pdf.font_family if "DejaVu" in pdf.font_family else "Helvetica"
    pdf.set_font(font_family, "B", 9)
    pdf.set_text_color(136, 146, 176)
    x_start = pdf.get_x()
    pdf.cell(50, 5, f"{label}:")
    pdf.set_font(font_family, "", 9)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(0, 5, value, new_x="LMARGIN", new_y="NEXT")
