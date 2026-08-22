from dataclasses import dataclass

import numpy as np


class NonTranscribableAudioError(ValueError):
    """Raised when a speech segment is too short to transcribe reliably."""


@dataclass
class Segment:
    """A contiguous speech region."""

    start: float
    end: float
    audio: np.ndarray
    source_id: int = -1  # index of the separated source this came from (-1 = unknown)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def __repr__(self):
        return f"Segment({self.start:.2f}s → {self.end:.2f}s, dur={self.duration:.2f}s)"


@dataclass
class LabeledSegment:
    """A speech segment with an assigned speaker label."""

    segment: Segment
    speaker_id: int

    @property
    def start(self) -> float:
        return self.segment.start

    @property
    def end(self) -> float:
        return self.segment.end

    @property
    def audio(self) -> np.ndarray:
        return self.segment.audio

    @property
    def source_id(self) -> int:
        return self.segment.source_id

    @property
    def speaker_label(self) -> str:
        return f"SPEAKER_{self.speaker_id:02d}"

    def __repr__(self):
        return (
            f"LabeledSegment({self.speaker_label}, {self.start:.2f}s → {self.end:.2f}s)"
        )


@dataclass
class TranscribedSegment:
    """A speaker-labeled segment with its transcription."""

    start: float
    end: float
    speaker_id: int
    speaker_label: str
    text: str
    language: str = "en"
    confidence: float = 1.0
    source_id: int = -1  # separated source this came from (-1 = unknown)

    def __repr__(self):
        return f'{self.speaker_label}: "{self.text}"'


@dataclass
class PipelineConfig:
    # Stage 1 — Separation
    num_speakers: int = 2  # expected number of speakers
    denoise_only: bool = False  # True = denoise, skip separation

    # Stage 2 — VAD
    vad_threshold: float = 0.5
    min_speech_ms: int = 250
    min_silence_ms: int = 100

    # Stage 3 — Clustering
    auto_num_speakers: bool = False  # True = estimate k automatically
    max_speakers: int = 8

    # Stage 4 — Transcription
    whisper_model: str = "base"  # tiny/base/small/medium/large-v3
    device: str = "cpu"  # cpu or cuda
    compute_type: str = "int8"  # int8 (cpu) or float16 (gpu)
    language: str | None = None  # None = auto-detect

    # Output
    save_path: str | None = None  # if set, saves transcript to file
    save_fmt: str = "json"  # json / plain / rttm / srt
    print_output: bool = True

    def validate(self) -> None:
        """Validate configuration before any model is loaded."""
        if self.denoise_only and self.auto_num_speakers:
            raise ValueError("denoise_only cannot be combined with auto_num_speakers")

        if not self.denoise_only and self.num_speakers not in (2, 3):
            raise ValueError(
                "num_speakers must be 2 or 3 for separation; use denoise_only "
                "for a single-speaker recording"
            )
        if self.max_speakers < 1:
            raise ValueError("max_speakers must be at least 1")
        if not 0.0 <= self.vad_threshold <= 1.0:
            raise ValueError("vad_threshold must be between 0 and 1")
        if self.min_speech_ms <= 0 or self.min_silence_ms <= 0:
            raise ValueError("VAD durations must be positive")

        if self.device not in {"cpu", "cuda", "auto"}:
            raise ValueError("device must be one of: cpu, cuda, auto")
        if self.device == "cuda":
            import torch

            if not torch.cuda.is_available():
                raise ValueError("device='cuda' requested, but CUDA is not available")

        if self.compute_type not in {"int8", "float16", "float32"}:
            raise ValueError("compute_type must be one of: int8, float16, float32")
        if self.whisper_model not in {"tiny", "base", "small", "medium", "large-v3"}:
            raise ValueError(
                "whisper_model must be one of: tiny, base, small, medium, large-v3"
            )
        if self.save_fmt not in {"json", "plain", "rttm", "srt"}:
            raise ValueError("save_fmt must be one of: json, plain, rttm, srt")
