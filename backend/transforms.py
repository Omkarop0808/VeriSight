"""Image transforms for inference (from existing detector repo)."""
import torchvision.transforms as transforms

# ImageNet mean and std
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)


def test_transforms():
    """Transforms for inference: resize, center crop, normalize."""
    return transforms.Compose([
        transforms.Resize(288),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_denormalize_params():
    """Return mean and std for denormalizing tensor back to image."""
    return mean, std
