"""ConvNeXtV2-Base model architecture for deepfake detection."""
import timm


def build_model():
    """Build ConvNeXtV2-Base model with 2-class output (Real/Fake)."""
    model = timm.create_model(
        "convnextv2_base",
        pretrained=True,  # ImageNet-1K pretrained
        num_classes=2
    )
    return model
