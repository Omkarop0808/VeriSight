#🚀 You can now run VeriSight AI: v2.0 — AI-Generated Image & Deepfake Forensic Platform

> **5-Engine Forensic Analysis Pipeline** — Combining a 400K-trained ConvNeXtV2 model with Gemini Vision AI, Metadata Analysis, FFT Spectral Analysis, and Error Level Analysis for the most accurate, explainable deepfake detection.

Built for **Smart India Hackathon 2025 — Problem Statement 7**: Image Classification and Artifact Identification for AI-Generated Images.

---

## 🎯 What It Does

1. **Upload any image** → drag & drop, paste from clipboard, or enter URL
2. **Engine 1 (ML)**: ConvNeXtV2-Base classifies Real/Fake (trained on 400K images)
3. **Grad-CAM heatmap** shows exactly which regions the model found suspicious
4. **Engine 2 (LLM)**: Gemini 2.0 Flash performs 6-category forensic analysis
5. **Signal 3 (Metadata)**: EXIF analysis, camera info, AI software detection, C2PA
6. **Signal 4 (FFT)**: Frequency domain analysis for GAN periodic artifacts
7. **Signal 5 (ELA)**: Error Level Analysis for compression inconsistencies
8. **Combined verdict** merges all 5 signals for highest accuracy
9. **PDF Report** — downloadable professional forensic report

## 🏗️ Architecture

```
                        ┌─ ConvNeXtV2 (ML, 400K trained) ──────────┐
                        ├─ Gemini 2.0 Flash (Forensics) ───────────┤
Image → Preprocessing → ├─ Metadata / EXIF Analyzer ───────────────┤→ Score Combiner → Verdict
                        ├─ FFT Spectral Analyzer ──────────────────┤
                        ├─ Error Level Analysis ───────────────────┘
                        └─ Grad-CAM (Explainability) → Heatmap
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- (Optional) NVIDIA GPU with CUDA for faster inference
- (Optional) Gemini API key for forensic analysis

### 1. Clone & Install

```bash
git clone https://github.com/Omkarop0808/VeriSight.git
cd VeriSight
pip install -r requirements.txt
```

### 2. Download Model Checkpoint (Automatic)

The checkpoint downloads automatically on first run. Or download manually:

```bash
python download_checkpoint.py
```

**Manual download:**
[https://huggingface.co/xRayon/convnext-ai-images-detector](https://huggingface.co/xRayon/convnext-ai-images-detector/tree/main/AI%20Images%20Detector/checkpoints)

Place it in: `backend/checkpoints/checkpoint_phase2.pth`

### 3. Set Up Gemini API (Optional)

```bash
cp .env.example .env
# Edit .env and add your Gemini API key
# Get free: https://aistudio.google.com/apikey
```

Or enter the key directly in the sidebar at runtime.

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
| **Architecture** | ConvNeXtV2-Base (256×256 input) |
| **Training** | Phase 1: ~400K images, 8 epochs + Phase 2: 20K continual learning |
| **Detects** | DALL-E3, FLUX, Midjourney V6, SDXL, SD3.5, StyleGAN2, BigGAN, and more |
| **OOD Accuracy** | 90.40% fake detection on EvalGen (FLUX, GoT, Infinity, OmniGen, Nova) |
| **Training Techniques** | LLRD, Cosine Annealing, AMP, Gradient Clipping, Continual Learning |

## ✨ Features

### Detection Engines
- ✅ **ConvNeXtV2 Classification** — ML Engine trained on 400K+ images
- 🔮 **Gemini 2.0 Flash Forensics** — 6-category artifact analysis with AI generator identification
- 📋 **Metadata/EXIF Analysis** — Camera info, GPS, software detection, C2PA credentials
- 📊 **FFT Spectral Analysis** — GAN periodic pattern detection in frequency domain
- 🔬 **Error Level Analysis** — JPEG compression consistency checking

### Explainability
- 🎨 **Grad-CAM Heatmaps** — Visual attention maps showing suspicious regions
- 📝 **Plain English Explanations** — Gemini explains findings in simple language
- 📄 **PDF Forensic Reports** — Professional, downloadable analysis reports

### User Experience
- 📁 **Multi-input support** — File upload, clipboard paste, URL input
- 📊 **Batch Analysis** — Analyze up to 20 images at once
- 📋 **Analysis History** — Track and export your scans
- 🎛️ **Adjustable Threshold** — Fine-tune detection strictness
- 🌙 **Stunning Dark UI** — Premium glassmorphism design with animated gradients
- 📥 **Automatic Checkpoint** — Model weights auto-download on first run

## 📁 Project Structure

```
VeriSight-AI/
├── app.py                              # Streamlit frontend (v2.0)
├── download_checkpoint.py              # One-click checkpoint download
├── backend/
│   ├── main.py                         # FastAPI backend server
│   ├── transforms.py                   # Image transforms (ImageNet norm)
│   ├── models/
│   │   ├── convnext.py                # ConvNeXtV2-Base architecture
│   │   └── schemas.py                 # Pydantic API schemas
│   ├── services/
│   │   ├── classifier.py              # ML classification (Engine 1)
│   │   ├── gemini_forensics.py        # Gemini analysis (Engine 2)
│   │   ├── gradcam_service.py         # Grad-CAM heatmaps
│   │   ├── metadata_analyzer.py       # EXIF/metadata analysis
│   │   ├── frequency_analyzer.py      # FFT spectral analysis
│   │   ├── ela_analyzer.py            # Error Level Analysis
│   │   ├── score_combiner.py          # Multi-engine verdict combiner
│   │   ├── report_generator.py        # PDF report generation
│   │   └── checkpoint_downloader.py   # Auto-download checkpoints
│   └── checkpoints/                    # Model weights (auto-downloaded)
├── requirements.txt
├── .env.example
└── README.md
```

## 🔑 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/analyze` | Full analysis (ML + Gemini + Metadata + FFT + ELA + Grad-CAM) |
| POST | `/api/batch` | Batch analyze multiple images (ML only) |
| POST | `/api/report/pdf` | Generate PDF forensic report |
| GET | `/api/stats` | Get analysis statistics |
| GET | `/api/health` | Health check |

## 📊 Training Dataset

Training used CIFAKE (60K) + 11 additional datasets totaling ~400K images:
- DDA Training Set, Defactify (MS COCO AI), genimage_tiny
- DF40 (Deepfake), Gravex200K, StyleGAN2
- Midjourney, DALL-E3, FLUX, SDXL, SD3.5, BigGAN, and more

## 🏆 SIH 2025 Evaluation Criteria

| Criteria | How We Address It |
|---|---|
| Classification Performance | ConvNeXtV2 with 90.4% OOD accuracy |
| Artifact Detection Quality | 6-category Gemini forensic + FFT + ELA breakdown |
| Explainability | Grad-CAM heatmaps + natural language + PDF reports |
| Innovation | 5-engine pipeline (ML + LLM + Metadata + FFT + ELA) — comprehensive approach |
| Presentation | Premium dark UI with visual forensic reports and deep forensics tab |

## 🔧 Configuration

| Variable | Default | Description |
|---|---|---|
| `GEMINI_API_KEY` | — | Gemini API key for forensic analysis |
| Detection Threshold | 0.50 | Adjustable via sidebar slider (0.0 - 1.0) |
| Gemini Forensics | Enabled | Toggle in sidebar |
| Metadata Analysis | Enabled | Toggle in sidebar |
| FFT Analysis | Enabled | Toggle in sidebar |
| ELA Analysis | Enabled | Toggle in sidebar |

## 📋 Tech Stack

| Component | Technology |
|---|---|
| ML Model | ConvNeXtV2-Base via timm |
| LLM | Gemini 2.0 Flash via google-genai SDK |
| Backend | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Heatmaps | pytorch-grad-cam |
| Reports | fpdf2 |
| Image Processing | Pillow, OpenCV, NumPy |

---

Built with ❤️ for Smart India Hackathon 2025
