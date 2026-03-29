"""Vision Transformer (ViT) deepfake detection service with Regional Patch Detection.
Engine 3 (or Engine 2 ML) of the detection pipeline.
"""
import os
import torch
import numpy as np
from transformers import pipeline, ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw
import io
import base64

_model = None
_processor = None
_device = None

def load_vit_model():
    """Load the ViT model and processor."""
    global _model, _processor, _device
    try:
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name = "dima806/ai_vs_real_image_detection"
        print(f"🧠 Loading ViT Regional Detector: {model_name}...")
        
        _processor = ViTImageProcessor.from_pretrained(model_name)
        _model = ViTForImageClassification.from_pretrained(model_name)
        _model.to(_device)
        _model.eval()
        
        print("✅ ViT Regional Detector loaded successfully.")
    except Exception as e:
        print(f"⚠️ Failed to load ViT model: {e}")

def analyze_vit_regional(image: Image.Image, grid_size: int = 4) -> dict:
    """
    Perform regional (patch-based) inference using ViT.
    Splits the image into a grid and classifies each patch.
    """
    global _model, _processor, _device
    if _model is None:
        load_vit_model()
    
    if _model is None:
        return {"overall_fake_prob": 0.0, "patches": [], "heatmap_b64": None}

    # 1. Global classification first
    with torch.no_grad():
        inputs = _processor(images=image, return_tensors="pt").to(_device)
        outputs = _model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Assuming label 1 is AI/Fake (check model config if needed, usually dima806 uses 1 for AI)
        global_fake_prob = probs[0][1].item()

    # 2. Patch-based classification
    w, h = image.size
    patch_w, patch_h = w // grid_size, h // grid_size
    patch_probs = []
    
    # Create Heatmap
    heatmap = Image.new("L", (grid_size, grid_size))
    pixels = heatmap.load()

    for y in range(grid_size):
        row = []
        for x in range(grid_size):
            left = x * patch_w
            top = y * patch_h
            right = (x + 1) * patch_w if x < grid_size - 1 else w
            bottom = (y + 1) * patch_h if y < grid_size - 1 else h
            
            patch = image.crop((left, top, right, bottom))
            
            # Classify patch
            with torch.no_grad():
                p_inputs = _processor(images=patch, return_tensors="pt").to(_device)
                p_outputs = _model(**p_inputs)
                p_probs = torch.nn.functional.softmax(p_outputs.logits, dim=-1)
                p_fake_prob = p_probs[0][1].item()
                
            row.append(p_fake_prob)
            pixels[x, y] = int(p_fake_prob * 255)
            
        patch_probs.append(row)

    # 3. Generate Visual Heatmap Overlay
    # Rescale heatmap to original size
    heatmap_large = heatmap.resize((w, h), Image.NEAREST)
    
    # Convert to RGB heatmap (Red for high risk, Blue for low)
    # Using a simple manual mapping for speed/simplicity
    heatmap_colored = Image.new("RGB", (w, h))
    h_data = np.array(heatmap_large)
    
    # Create colored overlay
    # Red channel = risk, Blue channel = 255-risk
    r = h_data
    g = np.zeros_like(h_data)
    b = 255 - h_data
    rgb_heatmap = np.stack([r, g, b], axis=-1).astype(np.uint8)
    heatmap_img = Image.fromarray(rgb_heatmap)
    
    # Blend with original image (40% opacity)
    blended = Image.blend(image.convert("RGB"), heatmap_img, 0.4)
    
    # Convert to base64
    buffered = io.BytesIO()
    blended.save(buffered, format="PNG")
    heatmap_b64 = base64.b64encode(buffered.getvalue()).decode()

    # Determine regional threat
    max_patch_risk = np.max(patch_probs)
    partially_fake = max_patch_risk > 0.45 and global_fake_prob < 0.55

    return {
        "fake_probability": round(global_fake_prob, 4),
        "max_patch_probability": round(float(max_patch_risk), 4),
        "is_partially_fake": partially_fake,
        "heatmap_b64": heatmap_b64,
        "grid_size": grid_size
    }
