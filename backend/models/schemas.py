"""Pydantic schemas for API request/response models."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict
from datetime import datetime


class AnalysisRequest(BaseModel):
    threshold: float = Field(default=0.50, ge=0.0, le=1.0, description="Fake detection threshold")
    use_gemini: bool = Field(default=True, description="Enable Gemini forensic analysis")
    include_metadata: bool = Field(default=True, description="Enable metadata analysis")
    include_frequency: bool = Field(default=True, description="Enable FFT analysis")
    include_ela: bool = Field(default=True, description="Enable ELA analysis")


class ArtifactScore(BaseModel):
    category: str
    score: float = Field(ge=0.0, le=100.0)
    description: str
    severity: str  # "low", "medium", "high", "critical"
    regions: Optional[str] = None


class MLResult(BaseModel):
    label: str  # "Real" or "Fake"
    confidence: float
    real_probability: float
    fake_probability: float


class GeminiResult(BaseModel):
    overall_verdict: str
    confidence: float
    explanation: str
    probable_generator: Optional[str] = "Unknown"
    artifacts: List[ArtifactScore]
    detailed_analysis: str
    recommendation: Optional[str] = None


class MetadataFinding(BaseModel):
    type: str  # "info", "warning", "critical"
    message: str
    severity: str


class MetadataResult(BaseModel):
    exif_data: Dict = {}
    findings: List[MetadataFinding] = []
    risk_score: int = 0
    image_properties: Dict = {}
    has_camera_info: bool = False
    has_gps: bool = False
    has_datetime: bool = False
    c2pa_detected: bool = False
    exif_count: int = 0
    summary: str = ""


class FrequencyResult(BaseModel):
    risk_score: int = 0
    findings: List[Dict] = []
    metrics: Dict = {}
    summary: str = ""


class ELAResult(BaseModel):
    risk_score: int = 0
    findings: List[Dict] = []
    metrics: Dict = {}
    summary: str = ""


class CombinedVerdict(BaseModel):
    final_label: str  # "Real" or "Fake"
    final_confidence: float
    ml_weight: float
    gemini_weight: float
    agreement: bool
    risk_level: str  # "Low", "Medium", "High", "Critical"
    analysis_engines: List[str] = []
    probable_generator: Optional[str] = "Unknown"
    auxiliary_analysis: Dict = {}


class AnalysisResponse(BaseModel):
    id: str
    filename: str
    timestamp: str
    ml_result: MLResult
    gemini_result: Optional[GeminiResult] = None
    metadata_result: Optional[MetadataResult] = None
    frequency_result: Optional[FrequencyResult] = None
    ela_result: Optional[ELAResult] = None
    combined_verdict: CombinedVerdict
    heatmap_base64: Optional[str] = None
    raw_cam_base64: Optional[str] = None
    processing_time: float


class BatchAnalysisResponse(BaseModel):
    total_images: int
    results: List[dict]
    summary: dict


class StatsResponse(BaseModel):
    total_scans: int
    real_count: int
    fake_count: int
    avg_confidence: float
    recent_scans: List[dict]
