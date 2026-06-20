import numpy as np
from faster_whisper import WhisperModel

from schemas import LabeledSegment, TranscribedSegment

_whisper_model: WhisperModel | None = None
_whisper_model_size: str = ""


def _load_whisper(model_size: str, device: str, compute_type: str) -> WhisperModel:
    global _whisper_model, _whisper_model_size
    if _whisper_model is None or _whisper_model_size != model_size:
        print(f"Loading faster-whisper ({model_size}) on {device}...")
        _whisper_model = WhisperModel(
            model_size, device=device, compute_type=compute_type
        )
        _whisper_model_size = model_size
    return _whisper_model


WHISPER_SR = 16000  # Whisper expects 16kHz


def transcribe_segment(
    audio: np.ndarray,
    sample_rate: int,
    model: WhisperModel,
    language: str | None = None,
) -> tuple[str, str, float]:
    """
    Transcribe a single audio chunk.

    Returns
    -------
    (text, language, avg_confidence)
    """
    import torch
    import torchaudio

    # Resample to 16kHz if needed
    if sample_rate != WHISPER_SR:
        waveform = torch.tensor(audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sample_rate, WHISPER_SR)
        audio = waveform.squeeze(0).numpy()

    # faster-whisper transcribe
    segments_gen, info = model.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=False,
    )

    text_parts = []
    confidences = []

    for seg in segments_gen:
        text_parts.append(seg.text.strip())
        if seg.avg_logprob is not None:
            # Convert log-prob to rough confidence score (clamped 0–1)
            conf = min(1.0, max(0.0, np.exp(seg.avg_logprob)))
            confidences.append(conf)

    text = " ".join(text_parts).strip()
    avg_conf = float(np.mean(confidences)) if confidences else 1.0
    detected_lang = info.language if info else (language or "en")

    return text, detected_lang, avg_conf


def transcribe_all(
    labeled_segments: list[LabeledSegment],
    sample_rate: int = WHISPER_SR,
    model_size: str = "base",
    device: str = "cpu",
    compute_type: str = "int8",
    language: str | None = None,
    skip_empty: bool = True,
) -> list[TranscribedSegment]:
    """
    Transcribe all speaker-labeled segments.

    Parameters
    ----------
    labeled_segments : list of LabeledSegment from cluster.py
    sample_rate      : sample rate of audio in the segments
    model_size       : whisper model size (tiny/base/small/medium/large-v3)
    device           : "cpu" or "cuda"
    compute_type     : "int8" (CPU), "float16" (GPU), "float32"
    language         : ISO 639-1 code (e.g. "en"); None = auto-detect
    skip_empty       : skip segments that transcribe to empty string

    Returns
    -------
    List of TranscribedSegment objects in chronological order
    """
    model = _load_whisper(model_size, device, compute_type)
    results = []

    for i, ls in enumerate(labeled_segments):
        print(f"Segment {i + 1}/{len(labeled_segments)} ({ls.speaker_label})...")

        text, lang, conf = transcribe_segment(ls.audio, sample_rate, model, language)

        if skip_empty and not text:
            print("Empty transcription, skipping")
            continue

        results.append(
            TranscribedSegment(
                start=ls.start,
                end=ls.end,
                speaker_id=ls.speaker_id,
                speaker_label=ls.speaker_label,
                text=text,
                language=lang,
                confidence=conf,
            )
        )

    print(f"Transcribed {len(results)} segment(s)")
    return results
