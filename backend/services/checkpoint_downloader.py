"""Checkpoint Downloader — Auto-download ConvNeXtV2 weights from HuggingFace."""
import os

REPO_ID = "xRayon/convnext-ai-images-detector"
FILENAME = "AI Images Detector/checkpoints/checkpoint_phase2.pth"
LOCAL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "checkpoints")
LOCAL_PATH = os.path.join(LOCAL_DIR, "checkpoint_phase2.pth")


def is_checkpoint_available() -> bool:
    """Check if the pre-trained checkpoint exists."""
    return os.path.exists(LOCAL_PATH)


def download_checkpoint(force: bool = False) -> str:
    """
    Download checkpoint from HuggingFace if not present.

    Args:
        force: Force re-download even if file exists

    Returns:
        Path to the checkpoint file
    """
    if is_checkpoint_available() and not force:
        print(f"✅ Checkpoint already exists at {LOCAL_PATH}")
        return LOCAL_PATH

    try:
        from huggingface_hub import hf_hub_download

        print(f"📥 Downloading checkpoint from HuggingFace ({REPO_ID})...")
        print("   This is a one-time download (~700MB). Please wait...")

        os.makedirs(LOCAL_DIR, exist_ok=True)

        downloaded_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=FILENAME,
            local_dir=os.path.dirname(LOCAL_DIR),
            local_dir_use_symlinks=False,
        )

        # Move to expected location if needed
        if downloaded_path != LOCAL_PATH:
            import shutil
            if os.path.exists(downloaded_path):
                shutil.copy2(downloaded_path, LOCAL_PATH)
                print(f"   Copied to {LOCAL_PATH}")

        print(f"✅ Checkpoint downloaded successfully!")
        return LOCAL_PATH

    except ImportError:
        print("⚠️ huggingface_hub not installed. Install with: pip install huggingface-hub")
        print(f"   Or manually download from: https://huggingface.co/{REPO_ID}")
        return LOCAL_PATH

    except Exception as e:
        print(f"⚠️ Download failed: {e}")
        print(f"   Please manually download from:")
        print(f"   https://huggingface.co/{REPO_ID}/tree/main/AI%20Images%20Detector/checkpoints")
        print(f"   Place checkpoint_phase2.pth in: {LOCAL_DIR}")
        return LOCAL_PATH


if __name__ == "__main__":
    download_checkpoint()
