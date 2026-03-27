"""Grad-CAM heatmap generation service for visual explainability."""
import torch
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import base64
import io

from backend.services.classifier import get_model_for_gradcam
from backend.transforms import test_transforms, get_denormalize_params


def generate_heatmap(
    image: Image.Image,
    target_class: int = 1,  # 1 = "Fake" class
    method: str = "gradcam"
) -> str:
    """
    Generate a Grad-CAM heatmap overlay for the given image.

    Args:
        image: PIL Image to analyze
        target_class: Class to generate heatmap for (1=Fake, 0=Real)
        method: "gradcam", "gradcam++", or "eigencam"

    Returns:
        Base64-encoded PNG image with heatmap overlay
    """
    model, device, transform = get_model_for_gradcam()

    if model is None:
        return _generate_placeholder_heatmap(image)

    # Preprocess image
    img_tensor = transform(image).unsqueeze(0).to(device)

    # Get the target layer for ConvNeXtV2
    # For ConvNeXtV2, the last feature extraction stage is stages[-1]
    try:
        target_layers = [model.stages[-1]]
    except AttributeError:
        # Fallback: try common layer names
        try:
            target_layers = [model.features[-1]]
        except AttributeError:
            target_layers = [list(model.children())[-2]]

    # Select CAM method
    cam_methods = {
        "gradcam": GradCAM,
        "gradcam++": GradCAMPlusPlus,
        "eigencam": EigenCAM,
    }
    CamClass = cam_methods.get(method, GradCAM)

    # Generate heatmap
    targets = [ClassifierOutputTarget(target_class)]

    with CamClass(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        grayscale_cam = grayscale_cam[0, :]  # Get first (only) image

    # Create overlay image
    # Resize original image to match heatmap
    img_resized = image.resize((256, 256))
    img_array = np.array(img_resized).astype(np.float32) / 255.0

    # Create the colored heatmap overlay
    visualization = show_cam_on_image(img_array, grayscale_cam, use_rgb=True)

    # Convert to base64
    result_image = Image.fromarray(visualization)
    return _pil_to_base64(result_image), _cam_to_base64(grayscale_cam)


def _generate_placeholder_heatmap(image: Image.Image) -> tuple:
    """Generate a placeholder heatmap when model is not available."""
    img_resized = image.resize((256, 256))
    # Create a simple red gradient overlay as placeholder
    img_array = np.array(img_resized).astype(np.float32) / 255.0
    fake_cam = np.random.rand(256, 256).astype(np.float32) * 0.3
    visualization = show_cam_on_image(img_array, fake_cam, use_rgb=True)
    result = Image.fromarray(visualization)
    return _pil_to_base64(result), ""


def _pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def _cam_to_base64(cam_array: np.ndarray) -> str:
    """Convert raw CAM grayscale array to base64 heatmap image."""
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_array), cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(heatmap_rgb)
    return _pil_to_base64(img)
