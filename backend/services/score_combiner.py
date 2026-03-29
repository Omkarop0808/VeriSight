"""Score Combiner — Merges all analysis signals into a final verdict.

Combines:
- Engine 1: ConvNeXtV2 ML classification (60% weight)
- Engine 2: Gemini forensic analysis (40% weight)
- Signal 3: Metadata risk (bonus modifier)
- Signal 4: Frequency analysis (bonus modifier)
- Signal 5: ELA analysis (bonus modifier)
"""


def combine_verdicts(
    ml_result: dict,
    gemini_result: dict = None,
    vit_result: dict = None,
    metadata_result: dict = None,
    frequency_result: dict = None,
    ela_result: dict = None,
) -> dict:
    """
    Combine all analysis results into a final verdict.
    Tri-Engine Ensemble:
    - Engine 1: ConvNeXtV2 (Texture Focus) - 35% weight
    - Engine 2: Gemini Flash (Anatomy Focus) - 30% weight
    - Engine 3: Vision Transformer (Context/Anomaly Focus) - 35% weight
    """
    ml_fake_prob = ml_result.get("fake_probability", 0.0)
    ml_label = ml_result.get("label", "Real")
    
    vit_fake_prob = vit_result.get("fake_probability", ml_fake_prob) if vit_result else ml_fake_prob

    # ─── Fallback Mode (If Gemini Fails) ────────────────────────────────────
    if gemini_result is None or gemini_result.get("confidence", 0) == 0:
        combined_fake_prob = (ml_fake_prob * 0.50) + (vit_fake_prob * 0.50)
        base_result = {
            "final_label": "Fake" if combined_fake_prob > 0.5 else "Real",
            "final_confidence": round(combined_fake_prob if combined_fake_prob > 0.5 else (1 - combined_fake_prob), 4),
            "ml_weight": 0.50,
            "gemini_weight": 0.0,
            "vit_weight": 0.50,
            "agreement": (ml_fake_prob > 0.5) == (vit_fake_prob > 0.5),
            "risk_level": _get_risk_level(combined_fake_prob),
            "analysis_engines": ["ConvNeXtV2", "ViT"] if vit_result else ["ConvNeXtV2"],
        }
        # Apply auxiliary signal bonuses
        base_result = _apply_auxiliary_bonuses(base_result, metadata_result, frequency_result, ela_result)
        return base_result

    # ─── Parse Gemini Verdict ────────────────────────────
    gemini_verdict = gemini_result.get("overall_verdict", "Unknown")
    gemini_confidence = gemini_result.get("confidence", 0.5)

    if gemini_verdict.lower() == "fake":
        gemini_fake_prob = gemini_confidence
    elif gemini_verdict.lower() == "real":
        gemini_fake_prob = 1.0 - gemini_confidence
    else:
        gemini_fake_prob = 0.5

    # ─── Tri-Engine Weighted Combination ─────────────────
    ml_weight = 0.35
    vit_weight = 0.35
    gemini_weight = 0.30

    combined_fake_prob = (ml_fake_prob * ml_weight) + (vit_fake_prob * vit_weight) + (gemini_fake_prob * gemini_weight)

    # ─── Agreement Analysis ──────────────────────────────
    ml_says_fake = ml_fake_prob > 0.5
    vit_says_fake = vit_fake_prob > 0.5
    gemini_says_fake = gemini_fake_prob > 0.5
    
    engines_say_fake = sum([ml_says_fake, vit_says_fake, gemini_says_fake])
    
    if engines_say_fake >= 2:
        # Majority agree it's fake
        combined_fake_prob = max(combined_fake_prob, 0.65)
    elif engines_say_fake == 0:
        # All agree real
        combined_fake_prob = min(combined_fake_prob, 0.35)
    else:
        # 1 says fake, 2 say real. Check Gemini Artifacts to break tie
        artifact_avg = _get_artifact_average(gemini_result.get("artifacts", []))
        if artifact_avg > 60:
             combined_fake_prob = max(combined_fake_prob, 0.55) # Gemini spotted critical artifact

    # ─── Anatomy Override (User request: catch extra hands) ─────────
    # If Gemini is extremely confident about an anatomy error, override weights
    anatomy_score = next((a.get("score", 0) for a in gemini_result.get("artifacts", []) if a.get("category") == "Anatomy & Proportions"), 0)
    if anatomy_score >= 90:
         # Force fake probability higher if blatant anatomy error exists
         combined_fake_prob = max(combined_fake_prob, 0.85)
         gemini_weight = 0.80 # Temporarily boost gemini's influence

    # ─── Final Verdict Logic ────────────────────────────
    if combined_fake_prob >= 0.70:
        final_label = "Fake"
    elif combined_fake_prob >= 0.40:
        final_label = "Potential Fake"
    else:
        final_label = "Real"

    final_confidence = combined_fake_prob if combined_fake_prob >= 0.40 else (1 - combined_fake_prob)

    result = {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 4),
        "combined_fake_probability": round(combined_fake_prob, 4),
        "ml_weight": ml_weight,
        "vit_weight": vit_weight,
        "gemini_weight": gemini_weight,
        "agreement": engines_say_fake in [0, 3], # Absolute agreement
        "risk_level": _get_risk_level(combined_fake_prob),
        "analysis_engines": ["ConvNeXtV2", "ViT", "Gemini"],
        "probable_generator": gemini_result.get("probable_generator", "Unknown"),
    }

    # Apply auxiliary signal bonuses
    result = _apply_auxiliary_bonuses(result, metadata_result, frequency_result, ela_result)

    return result


def _apply_auxiliary_bonuses(
    result: dict,
    metadata_result: dict = None,
    frequency_result: dict = None,
    ela_result: dict = None,
) -> dict:
    """Apply bonus confidence modifiers from auxiliary analysis signals."""
    auxiliary_scores = []
    auxiliary_details = {}

    if metadata_result:
        meta_risk = metadata_result.get("risk_score", 0)
        auxiliary_scores.append(meta_risk)
        auxiliary_details["metadata_risk"] = meta_risk
        if "Metadata" not in result.get("analysis_engines", []):
            result.setdefault("analysis_engines", []).append("Metadata")

    if frequency_result:
        freq_risk = frequency_result.get("risk_score", 0)
        auxiliary_scores.append(freq_risk)
        auxiliary_details["frequency_risk"] = freq_risk
        if "FFT" not in result.get("analysis_engines", []):
            result.setdefault("analysis_engines", []).append("FFT")

    if ela_result:
        ela_risk = ela_result.get("risk_score", 0)
        auxiliary_scores.append(ela_risk)
        auxiliary_details["ela_risk"] = ela_risk
        if "ELA" not in result.get("analysis_engines", []):
            result.setdefault("analysis_engines", []).append("ELA")

    if auxiliary_scores:
        avg_aux = sum(auxiliary_scores) / len(auxiliary_scores)
        auxiliary_details["average_auxiliary_risk"] = round(avg_aux, 2)

        # Moderate confidence adjustment based on auxiliary signals
        current_confidence = result["final_confidence"]
        if avg_aux > 60 and result["final_label"] == "Fake":
            boost = min(0.05, (avg_aux - 60) / 800)
            result["final_confidence"] = round(min(0.99, current_confidence + boost), 4)
            result["combined_fake_probability"] = result["final_confidence"]
        elif avg_aux < 20 and result["final_label"] == "Real":
            boost = min(0.05, (20 - avg_aux) / 800)
            result["final_confidence"] = round(min(0.99, current_confidence + boost), 4)
            result["combined_fake_probability"] = round(1.0 - result["final_confidence"], 4)

    result["auxiliary_analysis"] = auxiliary_details
    return result


def _get_risk_level(fake_probability: float) -> str:
    """Determine risk level based on fake probability."""
    if fake_probability >= 0.85:
        return "Critical"
    elif fake_probability >= 0.65:
        return "High"
    elif fake_probability >= 0.40:
        return "Medium"
    else:
        return "Low"


def _get_artifact_average(artifacts: list) -> float:
    """Calculate average artifact score."""
    if not artifacts:
        return 0.0
    scores = [a.get("score", 0) for a in artifacts]
    return sum(scores) / len(scores) if scores else 0.0
