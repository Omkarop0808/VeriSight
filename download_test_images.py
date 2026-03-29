import requests
import os

def download_samples():
    os.makedirs("sample_images", exist_ok=True)
    
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    
    print("Downloading Artificial Sample...")
    try:
        # thispersondoesnotexist generates AI faces (StyleGAN)
        r = requests.get("https://thispersondoesnotexist.com", headers=headers, timeout=10)
        r.raise_for_status()
        with open("sample_images/ai_generated_sample.jpg", "wb") as f:
            f.write(r.content)
        print("✅ Saved to sample_images/ai_generated_sample.jpg")
    except Exception as e:
        print(f"Failed to download AI sample: {e}")

    print("\nDownloading Real Human Sample...")
    try:
        real_url = "https://images.unsplash.com/photo-1507003211169-0a1dd7228f2d?q=80&w=600&auto=format&fit=crop"
        r = requests.get(real_url, headers=headers, timeout=10)
        r.raise_for_status()
        with open("sample_images/real_photo_sample.jpg", "wb") as f:
            f.write(r.content)
        print("✅ Saved to sample_images/real_photo_sample.jpg")
    except Exception as e:
        print(f"Failed to download real sample: {e}")

if __name__ == "__main__":
    download_samples()
