#!/usr/bin/env python3
"""
DeepSight AI — Checkpoint Downloader
Run this once to download the pre-trained ConvNeXtV2 model weights.

Usage:
    python download_checkpoint.py
"""
import os
import sys

def main():
    print("=" * 60)
    print("🛡️  DeepSight AI — Checkpoint Downloader")
    print("=" * 60)
    print()

    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend", "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_phase2.pth")

    if os.path.exists(checkpoint_path):
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        print(f"✅ Checkpoint already exists!")
        print(f"   Path: {checkpoint_path}")
        print(f"   Size: {size_mb:.1f} MB")
        print()
        print("To force re-download, delete the file and run again.")
        return

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("❌ huggingface_hub not installed.")
        print("   Install with: pip install huggingface-hub")
        print()
        print("   Or manually download from:")
        print("   https://huggingface.co/xRayon/convnext-ai-images-detector/tree/main/AI%20Images%20Detector/checkpoints")
        print(f"   Place checkpoint_phase2.pth in: {checkpoint_dir}")
        sys.exit(1)

    print("📥 Downloading ConvNeXtV2-Base checkpoint from HuggingFace...")
    print("   Repo: xRayon/convnext-ai-images-detector")
    print("   This is a one-time download (~700MB). Please wait...")
    print()

    os.makedirs(checkpoint_dir, exist_ok=True)

    try:
        downloaded_path = hf_hub_download(
            repo_id="xRayon/convnext-ai-images-detector",
            filename="AI Images Detector/checkpoints/checkpoint_phase2.pth",
            local_dir=os.path.dirname(os.path.abspath(__file__)),
            local_dir_use_symlinks=False,
        )

        # Copy to expected location
        if os.path.exists(downloaded_path) and downloaded_path != checkpoint_path:
            import shutil
            shutil.copy2(downloaded_path, checkpoint_path)

        if os.path.exists(checkpoint_path):
            size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
            print()
            print(f"✅ Download complete!")
            print(f"   Path: {checkpoint_path}")
            print(f"   Size: {size_mb:.1f} MB")
        else:
            print(f"⚠️ Downloaded to {downloaded_path}")
            print(f"   Please copy it to {checkpoint_path}")

    except Exception as e:
        print(f"❌ Download failed: {e}")
        print()
        print("   Please manually download from:")
        print("   https://huggingface.co/xRayon/convnext-ai-images-detector/tree/main/AI%20Images%20Detector/checkpoints")
        print(f"   Place checkpoint_phase2.pth in: {checkpoint_dir}")
        sys.exit(1)

    print()
    print("🚀 You can now run DeepSight AI:")
    print("   streamlit run app.py")


if __name__ == "__main__":
    main()
