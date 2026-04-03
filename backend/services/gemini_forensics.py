"""Gemini 3.1 Flash forensic analysis service — Engine 2 of the dual-engine pipeline.

Uses the new google-genai SDK (replaces deprecated google-generativeai).
Model: gemini-3.1-flash-preview (latest, March 2026)
"""
import os
import json
import base64
import io
from PIL import Image
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ─── Client State ────────────────────────────────────────────
_client = None
_configured = False
MODEL_ID = "gemini-2.5-flash"  # Latest stable flash model for high quota (March 2026)


def configure_gemini(api_key: str = None):
    """Configure the Gemini API with the new google-genai SDK."""
    global _client, _configured
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if key and key != "your_gemini_api_key_here":
        try:
            from google import genai
            _client = genai.Client(api_key=key)
            _configured = True
            print(f"✅ Gemini API configured (model: {MODEL_ID})")
        except Exception as e:
            print(f"⚠️ Gemini SDK init error: {e}")
            _configured = False
    else:
        print("⚠️ Gemini API key not set — forensic analysis will be unavailable")
        print("   Set GEMINI_API_KEY in .env file")


# ─── Forensic Prompt ─────────────────────────────────────────
FORENSIC_PROMPT = """You are VeriSight AI, an expert digital forensics analyst specializing in detecting AI-generated and deepfake images. Analyze this image thoroughly and provide a structured forensic report.

ANALYSIS INSTRUCTIONS:
This image was submitted by a user for authenticity verification. It may be a real photograph taken on a phone, a screenshot, or an AI-generated image. Do not default to calling an image AI-generated simply because it looks high quality or well-composed. Base your judgment solely on specific observable artifacts or anomalies. Examine the image for physical and artifact-level inconsistencies (e.g., FLUX, Midjourney, DALL-E) or organic indicators (e.g., real camera noise, natural proportions).

CRITICAL ANATOMY & SKELETAL CHECK (MANDATORY):
You MUST rigorously scrutinize all human anatomy subject-by-subject:
1. Count EVERY hand, EVERY arm, and EVERY leg. If there are 3 arms in the image, or a hand with 6 fingers, it is 100% FAKE.
2. SKELETAL CONSISTENCY: Look for "ghost hands" or limbs resting on shoulders/backs that do not belong to a visible body. This is a primary indicator of "half real / half fake" compositions.
3. JOINT PHYSICS: Check for impossible bone bends or joints that merge into the background or clothing (melting artifacts).
4. COMPOSITES: Check whether different human subjects in the image appear to have different generation origins — one may be a real photograph while another is AI-generated.
5. If ANY structural mutation is detected, the overall_verdict MUST be "Fake" with 0.95+ confidence. If you detect an extra hand or limb, describe exactly where it is located on the body.

RESPOND IN EXACTLY THIS JSON FORMAT (no markdown code blocks like ```json, no extra text, ONLY valid JSON starting with { and ending with }):
{
    "step_1_describe_content": "Objectively describe what you see in the image before making authentic judgments.",
    "step_2_suspicious_anomalies": ["List any specific AI generation artifacts or anatomical errors."],
    "step_3_organic_indicators": ["List elements that suggest this is a real physical photograph (natural shadows, sensor noise, etc)."],
    "overall_verdict": "Real" or "Fake",
    "confidence": 0.0 to 1.0,
    "explanation": "Based on steps 1-3, provide a 2-3 sentence balanced authenticity assessment.",
    "probable_generator": "Unknown" or the likely AI model (e.g., "DALL-E 3", "Midjourney V6", "Stable Diffusion XL", "FLUX", "StyleGAN"),
    "artifacts": [
        {
            "category": "Texture Analysis",
            "score": 0-100,
            "description": "Specific texture findings (smoothness, patterns, frequency artifacts, plastic skin)",
            "severity": "low/medium/high/critical",
            "regions": "Where in the image these artifacts appear"
        },
        {
            "category": "Lighting & Shadows",
            "score": 0-100,
            "description": "Lighting consistency findings (shadow direction, light sources, reflections, specular highlights)",
            "severity": "low/medium/high/critical",
            "regions": "Where in the image these artifacts appear"
        },
        {
            "category": "Anatomy & Proportions",
            "score": 0-100,
            "description": "Anatomical findings. REQUIRED: explicitly state finger/hand/limb counts. Describe any impossible joints or floating appendages here.",
            "severity": "low/medium/high/critical",
            "regions": "Where in the image these artifacts appear"
        },
        {
            "category": "Text & Symbols",
            "score": 0-100,
            "description": "Text quality (garbled text, impossible fonts, misspellings, gibberish characters, wrong symbols)",
            "severity": "low/medium/high/critical",
            "regions": "Where in the image these artifacts appear"
        },
        {
            "category": "Edges & Boundaries",
            "score": 0-100,
            "description": "Edge/boundary findings (halos, blending artifacts, floating elements, unnatural transitions)",
            "severity": "low/medium/high/critical",
            "regions": "Where in the image these artifacts appear"
        },
        {
            "category": "Physics & Geometry",
            "score": 0-100,
            "description": "Physical plausibility (impossible reflections, wrong gravity, spatial errors, perspective issues)",
            "severity": "low/medium/high/critical",
            "regions": "Where in the image these artifacts appear"
        }
    ],
    "regional_inspection": [
        {
            "subject_id": "e.g. Woman on the left / Man on the right / Foreground Car",
            "is_fake": true/false,
            "confidence": 0-100,
            "explanation": "Explicit explanation detailing exactly why this distinct person or region appears authentic or synthetic."
        }
    ],
    "detailed_analysis": "Comprehensive 4-6 sentence technical analysis covering ALL categories, noting patterns typical of specific AI generators. Explicitly mention anatomy validation.",
    "recommendation": "What further steps a forensic analyst should take"
}

IMPORTANT:
- Carefully examine every human figure in this image. Check for anatomical impossibilities including extra hands, extra arms, limbs in wrong positions, or any appendage appearing in a location that is not anatomically possible.
- Check whether different human subjects in the image appear to have different generation origins — one may be a real photograph while another is AI-generated. Report these findings explicitly and prominently.
- If you detect an extra hand or limb, describe exactly where it is located on the body.
- Focus on AI-SPECIFIC artifacts, not general image quality issues.
- Return ONLY valid JSON, no markdown code blocks.
"""


async def analyze_image_forensically(image: Image.Image) -> Optional[dict]:
    """
    Perform multi-dimensional forensic analysis using Gemini.

    Args:
        image: PIL Image to analyze

    Returns:
        dict with forensic analysis results, or None if Gemini unavailable
    """
    global _client, _configured

    if not _configured or _client is None:
        return _generate_fallback_analysis()

    try:
        # Convert PIL image to bytes for the new SDK
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        from google.genai import types

        # Create image part for the new SDK
        image_part = types.Part.from_bytes(
            data=img_bytes,
            mime_type="image/png"
        )

        # Generate content using the new client-based API
        response = _client.models.generate_content(
            model=MODEL_ID,
            contents=[FORENSIC_PROMPT, image_part],
            config=types.GenerateContentConfig(
                temperature=0.1,  # Low temp for consistent analysis
                max_output_tokens=8192,
            )
        )

        # Parse JSON response
        text = response.text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
        if text.endswith("```"):
            text = text.rsplit("```", 1)[0]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

        result = json.loads(text)
        return result

    except json.JSONDecodeError as e:
        print(f"⚠️ Gemini response parsing error: {e}")
        try:
            import re
            explanation_match = re.search(r'"explanation"\s*:\s*"([^"]+)"', response.text)
            detailed_match = re.search(r'"detailed_analysis"\s*:\s*"([^"]+)"', response.text)
            
            exp_text = explanation_match.group(1) if explanation_match else "Partial forensic analysis recovered. JSON response was truncated."
            det_text = detailed_match.group(1) if detailed_match else "The AI generator returned a complex response that exceeded parsing limits. Structural anomalies may be present. Review regional and anatomical signals."
            verdict = "Fake" if '"overall_verdict": "Fake"' in response.text or '"overall_verdict":"Fake"' in response.text else "Unknown"
            
            return {
                "overall_verdict": verdict,
                "confidence": 0.85 if verdict == "Fake" else 0.5,
                "explanation": exp_text,
                "probable_generator": "Unknown",
                "artifacts": [
                   {"category": "Recovered Diagnostics", "score": 75 if verdict == "Fake" else 50, "description": "Raw data was truncated. Manual review recommended.", "severity": "medium", "regions": "Global"}
                ],
                "detailed_analysis": det_text,
                "recommendation": "Inspect regional anomalies closely. Response was too large to parse completely."
            }
        except Exception as inner_e:
            print(f"⚠️ Regex fallback failed: {inner_e}")
            return _generate_fallback_analysis(reason=f"Failed to parse Gemini JSON output. Underlying error: {str(e)}")
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            reason = "🔴 Gemini API Quota Exhausted (429). Please try another key or wait for the limit to reset."
        else:
            reason = f"⚠️ Gemini Analysis Error: {error_msg}"
        print(f"⚠️ Gemini analysis error: {error_msg}")
        return _generate_fallback_analysis(reason=reason)


def _generate_fallback_analysis(reason: str = "Gemini forensic analysis unavailable. Set GEMINI_API_KEY in .env.") -> dict:
    """Generate a fallback analysis when Gemini is unavailable."""
    return {
        "overall_verdict": "Unknown",
        "confidence": 0.0,
        "explanation": reason,
        "probable_generator": "Unknown",
        "artifacts": [
            {"category": "Texture Analysis", "score": 0, "description": "Analysis unavailable — enable Gemini API", "severity": "low", "regions": "N/A"},
            {"category": "Lighting & Shadows", "score": 0, "description": "Analysis unavailable — enable Gemini API", "severity": "low", "regions": "N/A"},
            {"category": "Anatomy & Proportions", "score": 0, "description": "Analysis unavailable — enable Gemini API", "severity": "low", "regions": "N/A"},
            {"category": "Text & Symbols", "score": 0, "description": "Analysis unavailable — enable Gemini API", "severity": "low", "regions": "N/A"},
            {"category": "Edges & Boundaries", "score": 0, "description": "Analysis unavailable — enable Gemini API", "severity": "low", "regions": "N/A"},
            {"category": "Physics & Geometry", "score": 0, "description": "Analysis unavailable — enable Gemini API", "severity": "low", "regions": "N/A"},
        ],
        "detailed_analysis": "Deep forensic analysis requires Gemini API configuration. Get a free key at aistudio.google.com/apikey",
        "recommendation": "Configure Gemini API key to enable full forensic analysis."
    }

async def generate_expert_summary(signals: dict) -> str:
    """Synthesize all forensic signals into a final expert closing statement."""
    global _client, _configured
    if not _configured or _client is None:
        return "Expert summary unavailable. Please configure Gemini API Key."

    # Extract key signals for the prompt
    ml_prob = signals.get("ml_result", {}).get("fake_probability", 0)
    vit_prob = signals.get("vit_result", {}).get("fake_probability", 0)
    composite_flag = signals.get("regional_result", {}).get("is_inconsistent", False)
    anatomy_anomaly = signals.get("anatomy_result", {}).get("is_suspicious", False)
    meta_risk = signals.get("metadata_result", {}).get("risk_score", 0)
    
    # Construction of a detailed "evidence list" for Gemini
    evidence_context = f"""
    - ConvNeXt Texture Engine: {ml_prob*100:.1f}% AI
    - Regional ViT Engine: {vit_prob*100:.1f}% AI
    - Composite/Regional Consistency: {'FLAGGED' if composite_flag else 'Normal'}
    - Anatomical Check: {'ANOMALY DETECTED' if anatomy_anomaly else 'No visible human anomalies'}
    - Metadata Risk: {meta_risk}/100
    - Combined Forensic Verdict: {signals.get('combined', {}).get('final_label', 'Unknown')}
    """

    prompt = f"""You are a Senior Digital Forensics Lead. Based on the following raw forensic engine results, provide a 3-4 sentence 'Final Closing Statement' for the investigative report.
    
    EVIDENCE SUMMARY:
    {evidence_context}
    
    INSTRUCTIONS:
    - If there is a major conflict (e.g., ML engine says real but ViT says 99% AI fake), explicitly call out the 'Composite Discrepancy'.
    - If anatomical anomalies are present, emphasize them as conclusive evidence of AI generation.
    - Be authoritative but precise. 
    - Mention specific subjects if the data suggests localized manipulation (e.g., 'regional artifacts detected in subject facial structures').
    - Your goal is to tell the investigator EXACTLY why this image is or is not suspicious.
    
    Return ONLY the 3-4 sentence paragraph. No extra text.
    """

    try:
        from google.genai import types
        response = _client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt],
            config=types.GenerateContentConfig(temperature=0.3)
        )
        return response.text.strip()
    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg or "quota" in error_msg.lower():
            return "Expert summary unavailable: Gemini API rate limit/quota exceeded. Please wait 1 minute before analyzing another image."
        # Keep the error message brief so it doesn't break the UI
        return f"Unable to generate expert summary: {error_msg[:100]}..."
