"""Score combiner — merges ML classification and Gemini forensic results into a final verdict."""


def combine_verdicts(ml_result: dict, gemini_result: dict = None) -> dict:
    """
    Combine ML classification and Gemini forensic analysis into a final verdict.

    Uses weighted averaging with agreement bonuses:
    - ML model gets 60% weight (trained on 400K images, reliable)
    - Gemini gets 40% weight (contextual forensic analysis)
    - If both agree: +10% confidence bonus
    - If they disagree: flag for review, use the more confident one

    Args:
        ml_result: dict from classifier.classify_image()
        gemini_result: dict from gemini_forensics.analyze_image_forensically()

    Returns:
        dict with final_label, final_confidence, agreement, risk_level
    """
    ml_fake_prob = ml_result["fake_probability"]
    ml_label = ml_result["label"]

    # If Gemini is not available, use ML result only
    if gemini_result is None or gemini_result.get("confidence", 0) == 0:
        return {
            "final_label": ml_label,
            "final_confidence": round(ml_result["confidence"], 4),
            "ml_weight": 1.0,
            "gemini_weight": 0.0,
            "agreement": True,
            "risk_level": _get_risk_level(ml_fake_prob),
        }

    # Parse Gemini verdict
    gemini_verdict = gemini_result.get("overall_verdict", "Unknown")
    gemini_confidence = gemini_result.get("confidence", 0.5)

    # Convert Gemini verdict to fake probability
    if gemini_verdict.lower() == "fake":
        gemini_fake_prob = gemini_confidence
    elif gemini_verdict.lower() == "real":
        gemini_fake_prob = 1.0 - gemini_confidence
    else:
        gemini_fake_prob = 0.5  # Unknown — neutral

    # Weighted combination
    ml_weight = 0.60
    gemini_weight = 0.40

    combined_fake_prob = (ml_fake_prob * ml_weight) + (gemini_fake_prob * gemini_weight)

    # Check agreement
    ml_says_fake = ml_label == "Fake"
    gemini_says_fake = gemini_verdict.lower() == "fake"
    agreement = ml_says_fake == gemini_says_fake

    # Agreement bonus/penalty
    if agreement:
        # Both agree — boost confidence
        combined_fake_prob = min(1.0, combined_fake_prob * 1.1)
    else:
        # Disagree — use artifact scores to break tie
        artifact_avg = _get_artifact_average(gemini_result.get("artifacts", []))
        if artifact_avg > 50:  # High artifact scores = likely fake
            combined_fake_prob = max(combined_fake_prob, 0.6)
        else:
            combined_fake_prob = min(combined_fake_prob, 0.5)

    # Final verdict
    final_label = "Fake" if combined_fake_prob > 0.5 else "Real"
    final_confidence = combined_fake_prob if final_label == "Fake" else (1 - combined_fake_prob)

    return {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 4),
        "ml_weight": ml_weight,
        "gemini_weight": gemini_weight,
        "agreement": agreement,
        "risk_level": _get_risk_level(combined_fake_prob),
    }


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
