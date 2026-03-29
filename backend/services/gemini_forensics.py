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
MODEL_ID = "gemini-2.5-pro"  # Latest stable pro model for free tier (March 2026)


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
Examine the image for artifacts and inconsistencies that indicate AI generation. Look closely for artifacts typical of state-of-the-art generators (e.g., FLUX, Midjourney V6, DALL-E 3). Score each category from 0-100 (0 = no artifacts detected, 100 = obvious AI artifacts).

CRITICAL ANATOMY & SKELETAL CHECK (MANDATORY):
You MUST rigorously scrutinize all human anatomy subject-by-subject:
1. Count EVERY hand, EVERY arm, and EVERY leg. If there are 3 arms in the image, or a hand with 6 fingers, it is 100% FAKE.
2. SKELETAL CONSISTENCY: Look for "ghost hands" or limbs resting on shoulders/backs that do not belong to a visible body. This is a primary indicator of "half real / half fake" compositions.
3. JOINT PHYSICS: Check for impossible bone bends or joints that merge into the background or clothing (melting artifacts).
4. If ANY such structural mutation is detected, the overall_verdict MUST be "Fake" with 0.95+ confidence, regardless of how photorealistic the skin textures appear. High-quality generators often hide these "mutations" in plain sight.

RESPOND IN EXACTLY THIS JSON FORMAT (no markdown code blocks like ```json, no extra text, ONLY valid JSON starting with { and ending with }):
{
    "overall_verdict": "Real" or "Fake",
    "confidence": 0.0 to 1.0,
    "explanation": "2-3 sentence plain English summary of findings",
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
    "detailed_analysis": "Comprehensive 4-6 sentence technical analysis covering ALL categories, noting patterns typical of specific AI generators. Explicitly mention anatomy validation.",
    "recommendation": "What further steps a forensic analyst should take"
}

IMPORTANT:
- Be thorough but fair -- not all images with minor artifacts are fake.
- If an obvious physiological/anatomical error is found (like an extra hand), the overall_verdict MUST BE "Fake".
- Focus on AI-SPECIFIC artifacts, not general image quality issues.
- Return ONLY valid JSON, no markdown code blocks."""


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
                max_output_tokens=3000,
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
            print(f"   Raw response: {response.text[:300]}...")
        except Exception:
            pass
        return _generate_fallback_analysis(reason=f"Failed to parse Gemini JSON output. Underlying error: {str(e)}")
    except Exception as e:
        print(f"⚠️ Gemini analysis error: {e}")
        return _generate_fallback_analysis(reason=f"Gemini API Error: {str(e)}")


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
