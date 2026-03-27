"""Gemini 2.0 Flash forensic analysis service — Engine 2 of the dual-engine pipeline.
Adapted from awesome-llm-apps/starter_ai_agents/ai_medical_imaging_agent/
"""
import os
import json
import tempfile
from PIL import Image
from typing import Optional
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

# Configure Gemini
_gemini_configured = False


def configure_gemini(api_key: str = None):
    """Configure the Gemini API with the provided key."""
    global _gemini_configured
    key = api_key or os.getenv("GEMINI_API_KEY", "")
    if key and key != "your_gemini_api_key_here":
        genai.configure(api_key=key)
        _gemini_configured = True
        print("✅ Gemini API configured")
    else:
        print("⚠️ Gemini API key not set — forensic analysis will be unavailable")
        print("   Set GEMINI_API_KEY in .env file")


FORENSIC_PROMPT = """You are DeepSight AI, an expert digital forensics analyst specializing in detecting AI-generated and deepfake images. Analyze this image thoroughly and provide a structured forensic report.

ANALYSIS INSTRUCTIONS:
Examine the image for artifacts and inconsistencies that indicate AI generation. Score each category from 0-100 (0 = no artifacts detected, 100 = obvious AI artifacts).

RESPOND IN EXACTLY THIS JSON FORMAT (no markdown, no extra text, ONLY valid JSON):
{
    "overall_verdict": "Real" or "Fake",
    "confidence": 0.0 to 1.0,
    "explanation": "2-3 sentence plain English summary of findings",
    "artifacts": [
        {
            "category": "Texture Analysis",
            "score": 0-100,
            "description": "Specific texture findings (smoothness, patterns, frequency artifacts)",
            "severity": "low/medium/high/critical"
        },
        {
            "category": "Lighting Analysis",
            "score": 0-100,
            "description": "Lighting consistency findings (shadows, light sources, reflections)",
            "severity": "low/medium/high/critical"
        },
        {
            "category": "Anatomy Analysis",
            "score": 0-100,
            "description": "Anatomical findings if humans present (fingers, ears, eyes, proportions)",
            "severity": "low/medium/high/critical"
        },
        {
            "category": "Text Analysis",
            "score": 0-100,
            "description": "Embedded text quality (garbled text, impossible fonts, misspellings)",
            "severity": "low/medium/high/critical"
        },
        {
            "category": "Edge Analysis",
            "score": 0-100,
            "description": "Edge/boundary findings (halos, blending artifacts, floating elements)",
            "severity": "low/medium/high/critical"
        },
        {
            "category": "Physics Analysis",
            "score": 0-100,
            "description": "Physical plausibility (impossible reflections, wrong gravity, spatial errors)",
            "severity": "low/medium/high/critical"
        }
    ],
    "detailed_analysis": "Comprehensive 3-5 sentence technical analysis of ALL findings"
}

IMPORTANT:
- Be thorough but fair — not all images with minor artifacts are fake
- Real photos can have compression artifacts, lens distortion, etc.
- Focus on AI-SPECIFIC artifacts, not general image quality issues
- If the image appears genuinely real with no AI indicators, say so confidently
- Return ONLY valid JSON, no markdown code blocks"""


async def analyze_image_forensically(image: Image.Image) -> Optional[dict]:
    """
    Perform multi-dimensional forensic analysis using Gemini 2.0 Flash.

    Args:
        image: PIL Image to analyze

    Returns:
        dict with forensic analysis results, or None if Gemini unavailable
    """
    global _gemini_configured

    if not _gemini_configured:
        return _generate_fallback_analysis()

    try:
        # Save image to temp file for Gemini
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            image.save(tmp, format="PNG")
            tmp_path = tmp.name

        # Upload to Gemini
        uploaded_file = genai.upload_file(tmp_path, mime_type="image/png")

        # Create model and analyze
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            [FORENSIC_PROMPT, uploaded_file],
            generation_config=genai.GenerationConfig(
                temperature=0.1,  # Low temp for consistent analysis
                max_output_tokens=2000,
            )
        )

        # Clean up temp file
        os.unlink(tmp_path)

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
        print(f"   Raw response: {response.text[:200]}...")
        return _generate_fallback_analysis()
    except Exception as e:
        print(f"⚠️ Gemini analysis error: {e}")
        return _generate_fallback_analysis()


def _generate_fallback_analysis() -> dict:
    """Generate a fallback analysis when Gemini is unavailable."""
    return {
        "overall_verdict": "Unknown",
        "confidence": 0.0,
        "explanation": "Gemini forensic analysis unavailable. Only ML classification results are shown. Set GEMINI_API_KEY in .env to enable deep forensic analysis.",
        "artifacts": [
            {"category": "Texture Analysis", "score": 0, "description": "Analysis unavailable", "severity": "low"},
            {"category": "Lighting Analysis", "score": 0, "description": "Analysis unavailable", "severity": "low"},
            {"category": "Anatomy Analysis", "score": 0, "description": "Analysis unavailable", "severity": "low"},
            {"category": "Text Analysis", "score": 0, "description": "Analysis unavailable", "severity": "low"},
            {"category": "Edge Analysis", "score": 0, "description": "Analysis unavailable", "severity": "low"},
            {"category": "Physics Analysis", "score": 0, "description": "Analysis unavailable", "severity": "low"},
        ],
        "detailed_analysis": "Deep forensic analysis requires Gemini API configuration."
    }
