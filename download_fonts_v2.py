import requests
import os

# URLs encoded as byte strings to avoid any shell interpolation during write_to_file
url1 = b"https://github.com/prawnpdf/prawn/raw/master/data/fonts/DejaVuSans.ttf".decode()
url2 = b"https://github.com/prawnpdf/prawn/raw/master/data/fonts/DejaVuSans-Bold.ttf".decode()

fonts = {
    "DejaVuSans.ttf": url1,
    "DejaVuSans-Bold.ttf": url2
}

target_dir = os.path.join("backend", "assets", "fonts")
os.makedirs(target_dir, exist_ok=True)

for name, url in fonts.items():
    print(f"Downloading {name}...")
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        with open(os.path.join(target_dir, name), "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Done: {name}")
    except Exception as e:
        print(f"Error {name}: {e}")
