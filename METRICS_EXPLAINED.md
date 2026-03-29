# 🩺 VeriSight AI Forensic Platform: How to Read Your Results

Welcome to the **VeriSight Forensic Platform**. Think of this platform like a medical diagnostic machine for images. Instead of reading blood test results, we read pixels, data, and light. 

When you upload an image, our engines run a series of "health checks." Here is exactly what those checks mean in simple, plain English—including the "healthy" and "sick" number ranges!

---

## 🔬 Core Diagnostic Engines

### 1. Neural Pattern Scanner (ConvNeXtV2)
**What it does:** It looks for microscopic, repeating "checkerboard" patterns that AI generators accidentally leave behind when drawing pixels.
*   **🩺 The Diagnostic Range:** 
    *   `0% - 30%`: Healthy. Shows natural camera grain.
    *   `31% - 65%`: Suspicious. Could be heavy Photoshop or filters.
    *   `66% - 100%`: 🚨 Critical. Massive synthetic AI patterns detected.

### 2. The Face & Object Tracker (ViT & Regional Inspector)
**What it does:** It chops the image into tiny blocks and analyzes specific people (like "Boy on the left" vs "Girl on the right"). 
*   **Why it matters:** It identifies **Face-Swaps** and **Composite Images**. If the background is 10% AI (Real), but a single face is 99% AI, it flags the image as a manipulated fake!

### 3. Biological Anatomy Check (MediaPipe Holistic)
**What it does:** It acts like an X-Ray for human bones and joints. It actively counts fingers, tracks hand physics, and ensures limbs connect to a torso.
*   **The Golden Rule:** AI (like Midjourney) is terrible at writing human physical laws. If our scanner detects **6 fingers**, a floating ghost hand, or impossible joints, it triggers a **🚨 Mandatory Fake Override**, destroying any "Real" scores from other engines.

---

## 📊 Deep Forensics Lab Tests (The Deep Scan)

When an image looks mildly suspicious, check the Deep Forensics Lab for these specific tests:

### 📸 EXIF Metadata Test (The Digital Fingerprint)
**What it does:** Cameras embed invisible data into photos (GPS location, Lens type, Time of day). AI image generators strip this data out completely.
*   **🩺 Risk Score Range:** 
    *   `< 30/100`: Healthy. (Contains rich camera data like Focal Length and ISO).
    *   `> 70/100`: 🚨 High Risk. (The entire fingerprint was wiped clean, highly typical of DALL-E or Midjourney).

### 🌈 Error Level Analysis (ELA)
**What it does:** It checks if the image was saved multiple times or digitally spliced together. 
*   **🩺 Mean Error Range:**
    *   `0.0 - 1.5`: Healthy. The image is structurally uniform.
    *   `2.0 - 6.0`: 🚨 High Risk. Indicates bright "glowing" areas where someone cut-and-pasted a new object into the photo (like dropping a fake car onto a real street).

### 📡 Frequency Grid Analysis (FFT)
**What it does:** It scans for invisible, perfectly mathematical grids. Nature is chaotic. AI generates perfect math.
*   **🩺 High-Freq Ratio:**
    *   `> 0.20`: Healthy. Chaotic, natural sharpness.
    *   `< 0.15`: 🚨 High Risk. The image is unnaturally smooth and "smeared" by AI generation.

---

## ⚖️ The Final Verdict: "Turnitin" for Images

Because people often mix Real photos with Fake edits, VeriSight is designed to be highly sensitive. 

If VeriSight detects **ANY** critical failure—such as a heavily manipulated patch, impossible anatomy, or a massive spike in AI patterns—it will flag the **entire image** as **Likely AI Generated**. It will then attempt to tell you the *origin* (Who made it: Stable Diffusion, Midjourney, or DALL-E) based on how the pixels were blended!
