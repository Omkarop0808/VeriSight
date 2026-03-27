"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime


class AnalysisRequest(BaseModel):
    threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Fake detection threshold")
    use_gemini: bool = Field(default=True, description="Enable Gemini forensic analysis")


class ArtifactScore(BaseModel):
    category: str
    score: float = Field(ge=0.0, le=100.0)
    description: str
    severity: str  # "low", "medium", "high", "critical"


class MLResult(BaseModel):
    label: str  # "Real" or "Fake"
    confidence: float
    real_probability: float
    fake_probability: float


class GeminiResult(BaseModel):
    overall_verdict: str
    confidence: float
    explanation: str
    artifacts: List[ArtifactScore]
    detailed_analysis: str


class CombinedVerdict(BaseModel):
    final_label: str  # "Real" or "Fake"
    final_confidence: float
    ml_weight: float
    gemini_weight: float
    agreement: bool  # Do both engines agree?
    risk_level: str  # "Low", "Medium", "High", "Critical"


class AnalysisResponse(BaseModel):
    id: str
    filename: str
    timestamp: str
    ml_result: MLResult
    gemini_result: Optional[GeminiResult] = None
    combined_verdict: CombinedVerdict
    heatmap_base64: Optional[str] = None  # Base64 encoded Grad-CAM image
    processing_time: float  # seconds


class BatchAnalysisResponse(BaseModel):
    total_images: int
    results: List[AnalysisResponse]
    summary: dict  # {real_count, fake_count, avg_confidence}


class StatsResponse(BaseModel):
    total_scans: int
    real_count: int
    fake_count: int
    avg_confidence: float
    language_distribution: dict
    recent_scans: List[dict]
