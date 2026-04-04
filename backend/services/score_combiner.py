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
    anatomy_result: dict = None,
    regional_result: dict = None,
    user_threshold: float = 0.50,
) -> dict:
    """
    Combine all analysis results into a final verdict.
    Tri-Engine Ensemble + Advanced Forensics.
    """
    ml_fake_prob = ml_result.get("fake_probability", 0.0)
    vit_fake_prob = vit_result.get("fake_probability", ml_fake_prob) if vit_result else ml_fake_prob
    
    # ─── Parse Gemini Verdict ────────────────────────────
    gemini_weight = 0.40
    ml_weight = 0.30
    vit_weight = 0.30
    
    if gemini_result and gemini_result.get("confidence", 0) > 0:
        gemini_verdict = gemini_result.get("overall_verdict", "Unknown")
        gemini_confidence = gemini_result.get("confidence", 0.5)
        if gemini_verdict.lower() == "fake":
            gemini_fake_prob = gemini_confidence
        elif gemini_verdict.lower() == "real":
            gemini_fake_prob = 1.0 - gemini_confidence
        else:
            gemini_fake_prob = 0.5
    else:
        # If Gemini fails, don't invent a probability. Remove its weight.
        gemini_fake_prob = 0.0
        gemini_weight = 0.0
        ml_weight = 0.50
        vit_weight = 0.50

    # ─── Confidence-Weighted / Max-Pooling Override ───────
    # Standard weighted average
    weighted_avg = (ml_fake_prob * ml_weight) + (vit_fake_prob * vit_weight) + (gemini_fake_prob * gemini_weight)
    
    # Max-Pooling Override (Aggressive Forensic Bias)
    max_conf = max(ml_fake_prob, vit_fake_prob, gemini_fake_prob if gemini_weight > 0 else 0)
    
    if max_conf > 0.95:
        # If any engine is extremely sure, we force the verdict to High Risk
        combined_fake_prob = max(weighted_avg, 0.95 if max_conf > 0.98 else 0.85)
    elif max_conf > 0.85:
        # Significant partial signal
        combined_fake_prob = max(weighted_avg, 0.75)
    else:
        combined_fake_prob = weighted_avg

    # ─── PRE-CLASSIFICATION (Real Photo Indicators) ───────
    has_exif = metadata_result.get("exif_count", 0) > 0 if metadata_result else False
    has_camera_info = metadata_result.get("has_camera_info", False) if metadata_result else False
    
    # Check for natural noise via Frequency Analysis. Low grid score + low high_freq_ratio usually means natural sensor noise.
    freq_grid_score = frequency_result.get("metrics", {}).get("grid_score", 100) if frequency_result else 100
    natural_noise = freq_grid_score < 25 # Natural physics rarely produces checkerboard frequencies.

    is_live_camera = metadata_result.get("is_live_camera", False) if metadata_result else False

    # Identify if it is strongly likely to be a real, physical photograph
    is_strong_real_photo = has_camera_info or (has_exif and natural_noise) or is_live_camera
    
    if natural_noise and gemini_weight == 0.0:
        # If Gemini is offline, visual models often panic over complex organic textures (like leaves/trees).
        # We crush the CNN confidence more aggressively if the Frequency Domain shows NO AI GRID.
        ml_fake_prob *= 0.35 # Penalize hallucination
        vit_fake_prob *= 0.45 
        weighted_avg = (ml_fake_prob * ml_weight) + (vit_fake_prob * vit_weight) + (gemini_fake_prob * gemini_weight)

    if is_live_camera:
        # Hard penalty to AI score. A live hardware feed is definitively a real physical capture.
        combined_fake_prob = min(combined_fake_prob * 0.20, 0.15)
        ml_fake_prob *= 0.10
        vit_fake_prob *= 0.10

    # ─── Conflict Detection: Are engines highly discordant?
    discordant = False
    if (ml_fake_prob < 0.10 and vit_fake_prob > 0.85) or (ml_fake_prob > 0.85 and vit_fake_prob < 0.10):
        discordant = True
        # If highly discordant but it's a strongly proven real photo, trust the metadata instead of defaulting to "Partially Fake"
        if combined_fake_prob < 0.50 and not is_strong_real_photo:
            combined_fake_prob = 0.51 

    # ─── Anomaly Detection (High Priority Overrides) ──────
    has_anatomy_anomaly = anatomy_result.get("is_suspicious", False) if anatomy_result else False
    has_regional_anomaly = regional_result.get("is_inconsistent", False) if regional_result else False
    vit_partial_anomaly = vit_result.get("is_partially_fake", False) if vit_result else False

    # A confirmed camera photo requires exactly +0.20 above the user's threshold to trigger "AI Generated".
    # SAFETY BUFFER: If the score is in the "Low Confidence" zone (0.50 - 0.68) and we have HIGH Natural Noise, 
    # we treat it as a False Positive from the ML models.
    doubt_zone = (0.50 <= combined_fake_prob <= 0.68)
    if doubt_zone and natural_noise and gemini_weight == 0.0:
        combined_fake_prob = min(combined_fake_prob, 0.45)
        final_label = "Likely Real (Filtered Hallucination)"
    
    ai_threshold = min(user_threshold + 0.20, 0.95) if is_strong_real_photo else user_threshold
    inconclusive_threshold = user_threshold if is_strong_real_photo else max(user_threshold - 0.05, 0.10)

    if has_anatomy_anomaly:
        final_label = "Suspicious — Anomaly Detected"
    elif combined_fake_prob >= ai_threshold:
        final_label = "AI Generated"
    elif inconclusive_threshold <= combined_fake_prob < ai_threshold:
        if has_regional_anomaly or vit_partial_anomaly:
            final_label = "Partially AI Generated"
        elif is_strong_real_photo:
            final_label = "Possible False Positive — Manual Review Required"
        else:
            final_label = "Inconclusive"
    else:
        final_label = "Likely Real"

    # Push a flag to the UI indicating if a confirmed camera photo is being penalized heavily by ML/ViT
    false_positive_warning = (is_strong_real_photo and combined_fake_prob >= ai_threshold)

    final_confidence = combined_fake_prob if combined_fake_prob >= 0.5 else (1 - combined_fake_prob)

    # ─── Calculate Agreement ─────────────────────────────
    # High agreement if ML, ViT, and Gemini all favor the same side (>0.5 or <=0.5)
    ml_fake = ml_fake_prob > 0.5
    vit_fake = vit_fake_prob > 0.5
    gem_fake = gemini_fake_prob > 0.5
    agreement = (ml_fake == vit_fake == gem_fake)

    result = {
        "final_label": final_label,
        "final_confidence": round(final_confidence, 4),
        "combined_fake_probability": round(combined_fake_prob, 4),
        "ml_weight": ml_weight,
        "vit_weight": vit_weight,
        "gemini_weight": gemini_weight,
        "risk_level": _get_risk_level(combined_fake_prob, has_anatomy_anomaly),
        "analysis_engines": ["ConvNeXtV2", "ViT", "Gemini"],
        "probable_generator": gemini_result.get("probable_generator", "Unknown") if gemini_result else "Unknown",
        "anomalies_detected": has_anatomy_anomaly or has_regional_anomaly,
        "agreement": agreement,
        "is_discordant": discordant,
        "is_strong_real_photo": is_strong_real_photo,
        "false_positive_warning": false_positive_warning
    }

    # Apply auxiliary signal bonuses
    result = _apply_auxiliary_bonuses(result, metadata_result, frequency_result, ela_result)
    return result


def _get_risk_level(fake_probability: float, has_anomaly: bool = False) -> str:
    """Determine risk level based on fake probability and anomalies."""
    if has_anomaly:
        return "Suspicious"
    if fake_probability >= 0.85:
        return "Critical"
    elif fake_probability >= 0.55:
        return "High"
    elif fake_probability >= 0.35:
        return "Medium"
    else:
        return "Low"


def _get_artifact_average(artifacts: list) -> float:
    """Calculate average artifact score."""
    if not artifacts:
        return 0.0
    scores = [a.get("score", 0) for a in artifacts]
    return sum(scores) / len(scores) if scores else 0.0


def _apply_auxiliary_bonuses(result: dict, metadata: dict, frequency: dict, ela: dict) -> dict:
    """Apply auxiliary signal bonuses to the final result."""
    meta_risk = metadata.get("risk_score", 0) if metadata else 0
    freq_risk = frequency.get("risk_score", 0) if frequency else 0
    ela_risk = ela.get("risk_score", 0) if ela else 0
    
    # average auxiliary risk
    aux_avg = (meta_risk + freq_risk + ela_risk) / 3
    result["auxiliary_risk_score"] = round(aux_avg, 2)
    
    # If auxiliary risks are very high, add a warning flag
    if any(r > 80 for r in [meta_risk, freq_risk, ela_risk]):
        result["high_aux_risk_detected"] = True
        if result["final_label"] == "Inconclusive":
            result["final_label"] = "Partially AI Generated"
            
    return result
