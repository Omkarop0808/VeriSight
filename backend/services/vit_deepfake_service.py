"""Vision Transformer (ViT) deepfake detection service — Engine 3 of the Tri-Engine pipeline."""
import os
import torch
from transformers import pipeline
from PIL import Image

_pipe = None

def load_vit_model():
    """Lazy load the Hugging Face ViT image classification pipeline."""
    global _pipe
    try:
        device = 0 if torch.cuda.is_available() else -1
        print("🧠 Loading ViT Deepfake Detector (Engine 3)...")
        # Load the pre-trained deepfake detection model from HF Hub
        _pipe = pipeline("image-classification", model="prithivMLmods/Deep-Fake-Detector-Model", device=device)
        print("✅ ViT Pipeline loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load ViT pipeline: {e}")

def analyze_vit(image: Image.Image) -> dict:
    """Analyze the image using the ViT model."""
    global _pipe
    if _pipe is None:
        load_vit_model()
    
    if _pipe is None:
        return {"fake_probability": 0.0, "real_probability": 1.0, "label": "Real", "confidence": 1.0}

    try:
        results = _pipe(image)
        
        fake_prob = 0.0
        real_prob = 0.0
        
        # Pipeline output is extremely commonly a list of dicts: [{'label': 'FAKE', 'score': 0.99}]
        # Handle various label formats natively
        for r in results:
            label = r["label"].upper()
            if "FAKE" in label or "DEEPFAKE" in label or label == "1" or "MODIFIED" in label:
                fake_prob = r["score"]
            elif "REAL" in label or "AUTHENTIC" in label or label == "0":
                real_prob = r["score"]
                
        if fake_prob == 0.0 and real_prob == 0.0:
            if "FAKE" in results[0]["label"].upper() or "DEEP" in results[0]["label"].upper():
                fake_prob = results[0]["score"]
                real_prob = 1.0 - fake_prob
            else:
                real_prob = results[0]["score"]
                fake_prob = 1.0 - real_prob
                
        # If it returned only one label, deduce the other
        if fake_prob == 0.0 and real_prob > 0.0:
            fake_prob = 1.0 - real_prob
        elif real_prob == 0.0 and fake_prob > 0.0:
            real_prob = 1.0 - fake_prob

        is_fake = fake_prob >= 0.40
        label = "Fake" if is_fake else "Real"
        # If score is in the 40-50% range, mark as "Suspicious/Potential Partial Fake"
        if 0.40 <= fake_prob < 0.50:
            label = "Potential Fake"
        
        confidence = fake_prob if is_fake else real_prob
        
        return {
            "label": label,
            "confidence": round(confidence, 4),
            "real_probability": round(real_prob, 4),
            "fake_probability": round(fake_prob, 4)
        }
    except Exception as e:
        print(f"ViT Error: {e}")
        return {"fake_probability": 0.0, "real_probability": 1.0, "label": "Real", "confidence": 1.0}
