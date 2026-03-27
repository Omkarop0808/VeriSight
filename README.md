# 🛡️ DeepSight AI — AI-Generated Image & Deepfake Detector

> **Dual-Engine Forensic Analysis Platform** — Combining a 400K-trained ConvNeXtV2 model with Gemini Vision AI for accurate, explainable deepfake detection.

Built for **Smart India Hackathon 2025 — Problem Statement 7**: Image Classification and Artifact Identification for AI-Generated Images.

---

## 🎯 What It Does

1. **Upload any image** → drag & drop, paste, or use camera
2. **Engine 1 (ML)**: ConvNeXtV2-Base classifies Real/Fake (trained on 400K images)
3. **Grad-CAM heatmap** shows which regions the model found suspicious
4. **Engine 2 (LLM)**: Gemini 2.0 Flash performs 6-category forensic analysis
5. **Combined verdict** merges both engines for higher accuracy
6. **Artifact report** breaks down: texture, lighting, anatomy, text, edges, physics

## 🏗️ Architecture

```
Image → ConvNeXtV2 (ML, 400K trained) ─────┐
     → Gemini 2.0 Flash (Forensics) ───────┤→ Score Combiner → Verdict
     → Grad-CAM (Explainability) ──────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster inference

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model Checkpoint
Download `checkpoint_phase2.pth` from:
[https://huggingface.co/xRayon/convnext-ai-images-detector](https://huggingface.co/xRayon/convnext-ai-images-detector/tree/main/AI%20Images%20Detector/checkpoints)

Place it in: `backend/checkpoints/checkpoint_phase2.pth`

### 3. Set Up Gemini API (Optional, for forensic analysis)
```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get free: https://aistudio.google.com/apikey
```

### 4. Run
```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

### Run FastAPI Backend (Alternative)
```bash
python -m backend.main
# API at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## 🧠 Model Details

| Component | Details |
|---|---|
| **Architecture** | ConvNeXtV2-Base (256×256) |
| **Training** | Phase 1: 400K images, 8 epochs + Phase 2: 20K continual learning |
| **Detects** | DALL-E3, FLUX, Midjourney V6, SDXL, SD3.5, StyleGAN2, BigGAN, and more |
| **OOD Score** | 90.40% fake detection on EvalGen (FLUX, GoT, Infinity, OmniGen, Nova) |
| **Training Techniques** | LLRD, Cosine Annealing, AMP, Gradient Clipping, Continual Learning |

## ✨ Features

- ✅ **Dual-Engine Detection** — ML + LLM cross-validation
- 🎨 **Grad-CAM Heatmaps** — Visual explainability
- 🔬 **6-Category Artifact Analysis** — Texture, Lighting, Anatomy, Text, Edges, Physics
- 📊 **Batch Analysis** — Analyze multiple images at once
- 📋 **Analysis History** — Track your scans
- 📄 **Plain English Explanations** — Gemini explains findings in simple language
- 🎛️ **Adjustable Threshold** — Control detection strictness
- 🌙 **Stunning Dark UI** — Premium glassmorphism design

## 📁 Project Structure
```
DeepSight-AI/
├── app.py                          # Streamlit frontend
├── backend/
│   ├── main.py                     # FastAPI backend
│   ├── transforms.py               # Image transforms
│   ├── models/
│   │   ├── convnext.py             # ConvNeXtV2 architecture
│   │   └── schemas.py              # API schemas
│   ├── services/
│   │   ├── classifier.py           # ML classification (Engine 1)
│   │   ├── gradcam_service.py      # Grad-CAM heatmaps
│   │   ├── gemini_forensics.py     # Gemini analysis (Engine 2)
│   │   └── score_combiner.py       # Dual-engine verdict
│   └── checkpoints/                # Model weights
├── requirements.txt
├── .env.example
└── README.md
```

## 🔑 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analyze` | Analyze single image (ML + Gemini + Grad-CAM) |
| POST | `/api/batch` | Batch analyze multiple images |
| GET | `/api/stats` | Get analysis statistics |
| GET | `/api/health` | Health check |

## 📊 Dataset

Training used CIFAKE (60K) + 11 additional datasets totaling ~400K images:
- DDA Training Set, Defactify (MS COCO AI), genimage_tiny
- DF40 (Deepfake), Gravex200K, StyleGAN2, and more

## 🏆 SIH 2025 Evaluation Criteria

| Criteria | How We Address It |
|---|---|
| Classification Performance | ConvNeXtV2 with 90.4% OOD accuracy |
| Artifact Detection Quality | 6-category Gemini forensic breakdown |
| Explainability | Grad-CAM heatmaps + natural language explanations |
| Innovation | Dual-engine (ML + LLM) pipeline — unique approach |
| Presentation | Premium dark UI with visual forensic reports |

---

Built with ❤️ for Smart India Hackathon 2025
