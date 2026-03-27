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
    metadata_result: dict = None,
    frequency_result: dict = None,
    ela_result: dict = None,
) -> dict:
    """
    Combine all analysis results into a final verdict.

    Uses weighted averaging with agreement bonuses:
    - ML model gets 60% weight (trained on 400K images, reliable)
    - Gemini gets 40% weight (contextual forensic analysis)
    - Metadata/Frequency/ELA provide bonus confidence modifiers
    - If both engines agree: +10% confidence bonus
    - If they disagree: use artifact scores to break tie

    Args:
        ml_result: dict from classifier.classify_image()
        gemini_result: dict from gemini_forensics.analyze_image_forensically()
        metadata_result: dict from metadata_analyzer.analyze_metadata()
        frequency_result: dict from frequency_analyzer.analyze_frequency()
        ela_result: dict from ela_analyzer.analyze_ela()

    Returns:
        dict with final_label, final_confidence, agreement, risk_level, details
    """
    ml_fake_prob = ml_result["fake_probability"]
    ml_label = ml_result["label"]

    # ─── ML-Only Mode ────────────────────────────────────
    if gemini_result is None or gemini_result.get("confidence", 0) == 0:
        base_result = {
            "final_label": ml_label,
            "final_confidence": round(ml_result["confidence"], 4),
            "ml_weight": 1.0,
            "gemini_weight": 0.0,
            "agreement": True,
            "risk_level": _get_risk_level(ml_fake_prob),
            "analysis_engines": ["ConvNeXtV2"],
        }
        # Apply auxiliary signal bonuses even in ML-only mode
        base_result = _apply_auxiliary_bonuses(base_result, metadata_result, frequency_result, ela_result)
        return base_result

    # ─── Parse Gemini Verdict ────────────────────────────
    gemini_verdict = gemini_result.get("overall_verdict", "Unknown")
    gemini_confidence = gemini_result.get("confidence", 0.5)

    # Convert Gemini verdict to fake probability
    if gemini_verdict.lower() == "fake":
        gemini_fake_prob = gemini_confidence
    elif gemini_verdict.lower() == "real":
        gemini_fake_prob = 1.0 - gemini_confidence
    else:
        gemini_fake_prob = 0.5  # Unknown — neutral

    # ─── Weighted Combination ────────────────────────────
    ml_weight = 0.60
    gemini_weight = 0.40

    combined_fake_prob = (ml_fake_prob * ml_weight) + (gemini_fake_prob * gemini_weight)

    # ─── Agreement Analysis ──────────────────────────────
    ml_says_fake = ml_label == "Fake"
    gemini_says_fake = gemini_verdict.lower() == "fake"
    agreement = ml_says_fake == gemini_says_fake

    if agreement:
        # Both agree — boost confidence
        combined_fake_prob = min(1.0, combined_fake_prob * 1.10)
    else:
        # Disagree — use artifact scores to break tie
        artifact_avg = _get_artifact_average(gemini_result.get("artifacts", []))
        if artifact_avg > 50:
            combined_fake_prob = max(combined_fake_prob, 0.60)
        elif artifact_avg < 20:
            combined_fake_prob = min(combined_fake_prob, 0.45)
        else:
            combined_fake_prob = min(combined_fake_prob, 0.50)

    # ─── Final Verdict ───────────────────────────────────
    final_label = "Fake" if combined_fake_prob > 0.5 else "Real"
    final_confidence = combined_fake_prob if final_label == "Fake" else (1 - combined_fake_prob)

    result = {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 4),
        "combined_fake_probability": round(combined_fake_prob, 4),
        "ml_weight": ml_weight,
        "gemini_weight": gemini_weight,
        "agreement": agreement,
        "risk_level": _get_risk_level(combined_fake_prob),
        "analysis_engines": ["ConvNeXtV2", "Gemini"],
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
        elif avg_aux < 20 and result["final_label"] == "Real":
            boost = min(0.05, (20 - avg_aux) / 800)
            result["final_confidence"] = round(min(0.99, current_confidence + boost), 4)

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
