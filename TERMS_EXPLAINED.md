# 📖 VeriSight: Terms & Metrics Guide

This document provides a plain English explanation of every indicator, chart, and metric found within the VeriSight Forensic Platform.

---

## AI Probability Index (Verdict Badge)

**What it looks like on screen:** A large, color-coded percentage at the top of the results panel.
- 🟢 **Green (0-35%):** Authentic / Real.
- 🟡 **Yellow (35-55%):** Inconclusive / Manual Review.
- 🟠 **Orange (55-70% / Suspicious):** Potential manipulation or partial AI generation.
- 🔴 **Red (70-100%):** Confirmed AI-Generated.

**What it means in simple words:** This is the system's "confidence" that the image was created using AI software instead of a real camera.

**How to interpret it:** A high value (e.g., 98%) means the image has strong digital signatures of AI models. A low value (e.g., 4%) means the image looks like a natural photograph.

**Example to say out loud during demo:** "As we can see, the Probability Index is at 94%, which triggers a Red 'AI-Generated' verdict."

---

## Grad-CAM Heatmap (ConvNeXtV2)

**What it looks like on screen:** A colorful overlay (red, yellow, blue) on top of the original image.

**What it means in simple words:** This shows exactly "where" the AI was looking when it made its decision. "Hot" red areas are the most suspicious parts of the image.

**How to interpret it:** If red spots appear on unnatural textures or warped edges, the AI model is correctly identifying a synthetic artifact.

**Example to say out loud during demo:** "Notice the red hot-spots on the subject's hair and background—this is where our ConvNext model detected the most AI-related noise."

---

## Regional AI Detection Map (ViT Grid)

**What it looks like on screen:** The image divided into a grid of squares, where some squares are highlighted in red or yellow.

**What it means in simple words:** Instead of looking at the whole image at once, our Vision Transformer model checks the image square-by-square to find small, hidden AI mistakes.

**How to interpret it:** If only a few squares are red (like just the face), it might be a "Face Swap" or a composite fake where only part of the image is AI.

**Example to say out loud during demo:** "Our Grid Analysis highlights specific patches where the AI model struggled with consistency."

---

## 🦴 Anatomical Consistency Check

**What it looks like on screen:** A skeletal overlay (blue lines and points) showing the person's hands, arms, and joints.

**What it means in simple words:** AI often struggles with "human physics" (like giving someone 6 fingers or two elbows). This tool used AI to 'see' the body and flag impossible structures.

**How to interpret it:** If it flags "Extra Finger Detected" or "Impossible Limb Angle," it is a high-priority sign that the image is a fake.

**Example to say out loud during demo:** "The system detected an anatomical anomaly here—AI models often fail to render human hands correctly, as seen in this 6-finger flag."

---

## 📊 Frequency Domain (FFT) Chart

**What it looks like on screen:** A black-and-white grayscale image that looks like a star-field or a grid.

**What it means in simple words:** This looks at the "math" of the pixels. Real cameras produce random noise, but AI programs leave behind a hidden "grid" or "checkerboard" pattern from how they upscale images.

**How to interpret it:** If you see bright dots in a perfect square grid, it is a "smoking gun" for AI generation.

**Example to say out loud during demo:** "The FFT chart reveals a hidden checkerboard grid—a classic signature of modern AI upscaling algorithms."

---

## 🔬 Error Level Analysis (ELA)

**What it looks like on screen:** A dark image with bright white or colored specks.

**What it means in simple words:** Every time a real photo is saved, it compresses evenly. If part of an image is AI-generated and "pasted" in, its compression level will be different from the rest.

**How to interpret it:** Bright white areas in an otherwise dark ELA map suggest that area was modified or generated separately from the original background.

**Example to say out loud during demo:** "Notice how the subject is much brighter in this ELA scan than the background—this tells us they were likely digitally inserted."

---

## 🧩 Composite Image Interpreter

**What it looks like on screen:** A text box below a grid analysis that says "🚨 Composite Synthesis Detected."

**What it means in simple words:** This identifies "Half-Real / Half-Fake" images. It flags if the noise and texture in one part of the photo don't match the other part.

**How to interpret it:** If "Detected," it means someone likely used "Generative Fill" to add a fake object into a real photo.

**Example to say out loud during demo:** "The system has flagged this as a Composite Image, meaning it's a mix of a real photo and AI-generated elements."

---

## 🔮 AI Origin Suggestion

**What it looks like on screen:** A purple box naming a specific software like "Midjourney" or "DALL-E 3" with a percentage match.

**What it means in simple words:** Every AI "artist" has its own unique style and brushstrokes. This guesses which program was likely used to make the fake.

**How to interpret it:** A high match for "Midjourney" means the image has the characteristic "smoothness" and lighting typical of that model.

**Example to say out loud during demo:** "The platform suggests Midjourney as the most likely origin based on the hyper-realistic texture profiling."

---

## ⚖️ Expert Investigative Closing Statement

**What it looks like on screen:** A boxed summary at the very bottom of the report.

**What it means in simple words:** This is our AI "Detective" (Gemini) reading all the technical data and writing a final conclusion in plain English.

**How to interpret it:** This is the most important part for a judge. It explains *why* the image was flagged, summarizing all the complex math into a simple story.

**Example to say out loud during demo:** "Finally, our Expert System synthesizes all signals into this closing report, explaining exactly why this image is a high-risk fake."

---

## Metadata Deep Scan

**What it looks like on screen:** A table showing "Camera Model," "Aperture," "Lens," or a warning that metadata is missing.

**What it means in simple words:** Real photos contain "tags" from the camera that took them. AI images are usually "born" without these tags.

**How to interpret it:** If metadata is "Blank" or "Missing," it is a warning sign. If it shows "iPhone 15 Pro," it is a point in favor of the image being real.

**Example to say out loud during demo:** "The metadata scan confirms this image lacks any physical camera information, which is common in AI-generated files."

---

## edge/texture Map

**What it looks like on screen:** Neon-colored outlines (Canny Edges) or grainy purple maps (Texture Complexity).

**What it means in simple words:** AI models sometimes make edges that are "too perfect" or textures that are "too smooth." These maps highlight those areas.

**How to interpret it:** High complexity in areas that should be simple (like skin) often reveals where AI added fake detail.

**Example to say out loud during demo:** "The Texture Complexity map shows unnaturally smooth patches, indicating algorithmic smoothing often found in deepfakes."
