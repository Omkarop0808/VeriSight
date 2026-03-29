"""ConvNeXtV2 classifier service — Engine 1 of the dual-engine pipeline."""
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Tuple
import os

from backend.models.convnext import build_model
from backend.transforms import test_transforms

# Global model instance
_model = None
_device = None
_transform = None


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str = None):
    """Load the ConvNeXtV2 model with trained weights."""
    global _model, _device, _transform

    _device = get_device()
    _model = build_model()

    # Try to load checkpoint
    if checkpoint_path and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=_device)
        _model.load_state_dict(ckpt["model"])
        print(f"✅ Loaded checkpoint from {checkpoint_path}")
    else:
        print("⚠️ No checkpoint found — using ImageNet pretrained weights (demo mode)")
        print(f"   Expected at: {checkpoint_path}")
        print("   Download from: https://huggingface.co/xRayon/convnext-ai-images-detector")

    _model.to(_device)
    _model.eval()
    _transform = test_transforms()
    
    # Warm up GPU memory allocation/compilation
    if _device.type == "cuda":
        print("🔥 Warming up GPU inference engine...")
        dummy_input = torch.zeros(1, 3, 256, 256).to(_device)
        with torch.inference_mode():
            for _ in range(3):
                _model(dummy_input)

    print(f"🧠 Model loaded on {_device}")
    return _model


def get_model():
    """Get the loaded model instance."""
    return _model, _device, _transform


def classify_image(image: Image.Image, threshold: float = 0.50) -> dict:
    """
    Classify an image as Real or Fake using ConvNeXtV2.

    Returns:
        dict with label, confidence, real_prob, fake_prob
    """
    global _model, _device, _transform

    if _model is None:
        # Lazy load with fallback paths
        from backend.services.checkpoint_downloader import is_checkpoint_available
        paths = [
            os.path.join("backend", "checkpoints", "checkpoint_phase2.pth"),
            os.path.join("checkpoints", "checkpoint_phase2.pth"),
        ]
        checkpoint = next((p for p in paths if os.path.exists(p)), None)
        load_model(checkpoint)
        
    if _model is None:
        raise RuntimeError("Model not loaded. Please ensure checkpoints are available.")

    # Preprocess
    img_tensor = _transform(image).unsqueeze(0).to(_device)

    # Inference
    with torch.inference_mode():
        logits = _model(img_tensor)
        probs = F.softmax(logits, dim=1)[0]

    real_prob = probs[0].item()
    fake_prob = probs[1].item()

    is_fake = fake_prob > threshold
    label = "Fake" if is_fake else "Real"
    confidence = fake_prob if is_fake else real_prob

    return {
        "label": label,
        "confidence": round(confidence, 4),
        "real_probability": round(real_prob, 4),
        "fake_probability": round(fake_prob, 4),
        "logits": logits  # Keep for Grad-CAM
    }


def get_model_for_gradcam():
    """Return model, device, and transform for Grad-CAM usage."""
    return _model, _device, _transform
