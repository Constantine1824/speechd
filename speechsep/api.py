"""
api.py — FastAPI Server
-----------------------
Wraps the SpeechSep pipeline in a small HTTP API for web frontends.

Run locally:
  uvicorn speechsep.api:app --reload

Endpoints:
  GET  /health      — liveness, package version, resolved device
  POST /transcribe  — multipart audio upload -> speaker-labeled transcript

Environment variables (all optional):
  SPEECHSEP_DEVICE         cpu | cuda | auto   (default: cpu)
  SPEECHSEP_COMPUTE_TYPE   int8 | float16 | float32   (default: int8)
  SPEECHSEP_CORS_ORIGINS   comma-separated origins, or "*"   (default: "*")
"""

import os
import tempfile
import time
from importlib import metadata
from pathlib import Path

import soundfile as sf
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from speechsep.main import run
from speechsep.schemas import PipelineConfig, TranscribedSegment

# Formats soundfile can decode reliably across platforms.
ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}


def _package_version() -> str:
    try:
        return metadata.version("speechsep")
    except metadata.PackageNotFoundError:
        return "unknown"


def _resolve_device() -> str:
    device = os.getenv("SPEECHSEP_DEVICE", "cpu").strip().lower() or "cpu"
    if device == "auto":
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str


class SegmentOut(BaseModel):
    speaker: str
    speaker_id: int
    start: float
    end: float
    text: str
    language: str
    confidence: float


class TranscribeResponse(BaseModel):
    duration_ms: int
    speakers: list[str]
    segments: list[SegmentOut]


app = FastAPI(
    title="SpeechSep API",
    description="Source separation + speaker diarization + transcription",
    version=_package_version(),
)

_cors_env = os.getenv("SPEECHSEP_CORS_ORIGINS", "*").strip()
_cors_origins = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        version=_package_version(),
        device=_resolve_device(),
    )


def _to_response(result: list[TranscribedSegment], duration_ms: int) -> TranscribeResponse:
    return TranscribeResponse(
        duration_ms=duration_ms,
        speakers=sorted({seg.speaker_label for seg in result}),
        segments=[
            SegmentOut(
                speaker=seg.speaker_label,
                speaker_id=seg.speaker_id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=seg.text,
                language=seg.language,
                confidence=round(seg.confidence, 4),
            )
            for seg in result
        ],
    )


# Sync on purpose: the pipeline is CPU/GPU-bound, and sync endpoints run in
# Starlette's threadpool instead of blocking the event loop.
@app.post("/transcribe", response_model=TranscribeResponse)
def transcribe(
    file: UploadFile = File(...),
    speakers: int = Form(2),
    denoise: bool = Form(False),
    auto_speakers: bool = Form(False),
    language: str | None = Form(None),
    whisper_model: str = Form("base"),
) -> TranscribeResponse:
    filename = file.filename or "audio"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    language = language.strip() if language else None
    config = PipelineConfig(
        num_speakers=speakers,
        denoise_only=denoise,
        auto_num_speakers=auto_speakers,
        whisper_model=whisper_model,
        device=_resolve_device(),
        compute_type=os.getenv("SPEECHSEP_COMPUTE_TYPE", "int8"),
        language=language or None,
        print_output=False,
    )
    try:
        config.validate()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    fd, tmp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as out:
            out.write(file.file.read())
        if os.path.getsize(tmp_path) == 0:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")

        started = time.perf_counter()
        try:
            result = run(tmp_path, config=config)
        except sf.LibsndfileError as e:
            raise HTTPException(
                status_code=400,
                detail="Could not decode audio file — it may be corrupt or mislabeled",
            ) from e
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:  # model load / OOM / download failures
            raise HTTPException(status_code=500, detail=str(e)) from e
        elapsed_ms = round((time.perf_counter() - started) * 1000)
    finally:
        os.unlink(tmp_path)

    return _to_response(result, elapsed_ms)
