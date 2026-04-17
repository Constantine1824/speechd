from dataclasses import dataclass

import numpy as np


@dataclass
class Segment:
    """A contiguous speech region."""

    start: float
    end: float
    audio: np.ndarray

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
