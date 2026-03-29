"""Regional Inconsistency Analysis service.
Splits image into a grid and compares noise, ELA, and frequency distributions.
Identifies composites of different generators or mixed real/fake sources.
"""
import numpy as np
import cv2
from PIL import Image, ImageChops, ImageDraw
import io
import base64

def analyze_regional_inconsistency(image: Image.Image, grid_size: int = 4) -> dict:
    """Compare local pixel statistics across regions."""
    # Convert image to grayscale for noise analysis
    img_gray = np.array(image.convert("L"))
    h, w = img_gray.shape
    patch_h, patch_w = h // grid_size, w // grid_size
    
    variances = []
    ela_scores = []
    
    # Re-save at 90 quality to see compression loss
    buffered = io.BytesIO()
    image.convert("RGB").save(buffered, format="JPEG", quality=90)
    buffered.seek(0)
    image_90 = Image.open(buffered)
    image_90.load()  # Ensure data is loaded before buffer is closed
    buffered.close()
    
    # Calculate DIFF
    diff = ImageChops.difference(image, image_90)
    diff_arr = np.array(diff.convert("L"))
    
    for row in range(grid_size):
        v_row = []
        e_row = []
        for col in range(grid_size):
            y_start = row * patch_h
            y_end = (row + 1) * patch_h if row < grid_size - 1 else h
            x_start = col * patch_w
            x_end = (col + 1) * patch_w if col < grid_size - 1 else w
            
            # Patch
            p = img_gray[y_start:y_end, x_start:x_end]
            e = diff_arr[y_start:y_end, x_start:x_end]
            
            # Local Noise Variance
            v_row.append(float(np.var(p)))
            # Local ELA Signal Intensity
            e_row.append(float(np.mean(e)))
            
        variances.append(v_row)
        ela_scores.append(e_row)
        
    # 2. Heuristic: Calculate Inconsistency (Standard Deviation across cells)
    v_arr = np.array(variances)
    e_arr = np.array(ela_scores)
    
    v_inconsistency = np.std(v_arr) / (np.mean(v_arr) + 1e-6)
    e_inconsistency = np.std(e_arr) / (np.mean(e_arr) + 1e-6)
    
    # High score means some regions are vastly different from others
    # High score means some regions are vastly different from others. V-inconsistency fluctuates naturally. ELA is stronger evidence.
    is_inconsistent = e_inconsistency > 0.65 or (v_inconsistency > 1.50 and e_inconsistency > 0.40)
    
    # 3. Highlight Inconsistent Cells
    # Create mask overlay
    mask = Image.new("RGBA", image.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(mask)
    
    max_ela = np.max(e_arr)
    mean_ela = np.mean(e_arr)
    
    suspicious_cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            # If a cell is > 1.5x the mean ELA, it is suspicious
            if e_arr[r, c] > 1.5 * mean_ela:
                y_s = r * patch_h
                x_s = c * patch_w
                y_e = (r + 1) * patch_h
                x_e = (c + 1) * patch_w
                # Draw translucent red box with yellow border
                d.rectangle([x_s, y_s, x_e, y_e], fill=(255, 60, 60, 100), outline=(255, 255, 0, 180), width=4)
                suspicious_cells.append((r, c))

    # Convert overlay to base64
    buffered_mask = io.BytesIO()
    # Composite the mask onto the image
    result_img = Image.alpha_composite(image.convert("RGBA"), mask)
    result_img.save(buffered_mask, format="PNG")
    composite_b64 = base64.b64encode(buffered_mask.getvalue()).decode()

    return {
        "is_inconsistent": is_inconsistent,
        "v_inconsistency": round(float(v_inconsistency), 4),
        "e_inconsistency": round(float(e_inconsistency), 4),
        "suspicious_cells_count": len(suspicious_cells),
        "composite_b64": composite_b64,
        "explanation": "Image regions exhibit consistent noise/compression distributions." if not is_inconsistent else (
            "Detected regions with abnormal compression/noise characteristics — possible composite image."
        )
    }
