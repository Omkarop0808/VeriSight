import requests
import os

fonts = {
    "DejaVuSans.ttf": "https://github.com/prawnpdf/prawn/raw/master/data/fonts/DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf": "https://github.com/prawnpdf/prawn/raw/master/data/fonts/DejaVuSans-Bold.ttf"
}

target_dir = r"c:\dev\hackthon\Neural-Nexus\DeepSight-AI\backend\assets\fonts"
os.makedirs(target_dir, exist_ok=True)

for name, url in fonts.items():
    p = os.path.join(target_dir, name)
    print(f"Downloading {name} from {url} to {p}...")
    try:
        r = requests.get(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        with open(p, "wb") as f:
            f.write(r.content)
        print(f"Successfully downloaded {name}")
    except Exception as e:
        print(f"Failed to download {name}: {e}")
