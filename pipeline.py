from dataclasses import dataclass

import numpy as np
import torchaudio

from cluster import cluster_speakers
from embed import ECAPA_SR, extract_embeddings
from output import pretty_print, save
from separate import TARGET_SR as SEP_SR
from separate import separate
from transcribe import WHISPER_SR, transcribe_all
from vad import SILERO_SR, detect_speech

from .types import PipelineConfig, TranscribedSegment


def _resample_np(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
    import torch

    if from_sr == to_sr:
        return audio
    waveform = torch.tensor(audio).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, from_sr, to_sr)
    return waveform.squeeze(0).numpy()


def run(
    audio_input: str | np.ndarray,
    sample_rate: int | None = None,
    config: PipelineConfig | None = None,
) -> list[TranscribedSegment]:
    """
    Run the full sound separation + diarization + transcription pipeline.

    Parameters
    ----------
    audio_input  : path to audio file (str) OR raw numpy audio array
    sample_rate  : required if audio_input is a numpy array
    config       : PipelineConfig instance (uses defaults if None)

    Returns
    -------
    List of TranscribedSegment objects in chronological order
    """
    cfg = config or PipelineConfig()

    if isinstance(audio_input, str):
        print(f"Loading audio: {audio_input}")
        waveform, sr = torchaudio.load(audio_input)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        audio_raw = waveform.squeeze(0).numpy()
        sample_rate = sr
    else:
        audio_raw = audio_input
        if sample_rate is None:
            raise ValueError(
                "sample_rate must be provided when audio_input is a numpy array"
            )

    print(f"Audio loaded: {len(audio_raw) / sample_rate:.2f}s at {sample_rate}Hz")

    print("\nStage 1: Source separation / denoising")
    sources = separate(
        audio_raw,
        sample_rate=sample_rate,
        num_speakers=cfg.num_speakers,
        denoise_only=cfg.denoise_only,
    )
    current_sr = SEP_SR

    print("\nStage 2: Voice Activity Detection")
    all_segments = []

    for source_idx, source_audio in enumerate(sources):
        print(f"  → Source {source_idx + 1}/{len(sources)}")

        # Resample from 8kHz to 16kHz for silero-vad
        # audio_16k = _resample_np(source_audio, current_sr, SILERO_SR)

        segs = detect_speech(
            source_audio,
            sample_rate=SILERO_SR,
            threshold=cfg.vad_threshold,
            min_speech_duration_ms=cfg.min_speech_ms,
            min_silence_duration_ms=cfg.min_silence_ms,
        )
        all_segments.extend(segs)

    # Sort all segments by start time across sources
    all_segments.sort(key=lambda s: s.start)
    print(f"Total segments after VAD: {len(all_segments)}")

    if not all_segments:
        print("No speech detected. Exiting.")
        return []

    print("\nStage 3: Speaker embedding + clustering")
    embeddings, valid_segments = extract_embeddings(
        all_segments,
        sample_rate=SILERO_SR,
    )

    k = None if cfg.auto_num_speakers else cfg.num_speakers
    labeled_segments = cluster_speakers(
        embeddings,
        valid_segments,
        num_speakers=k,
        max_speakers=cfg.max_speakers,
    )

    print("\nStage 4: Transcription (faster-whisper)")
    transcribed = transcribe_all(
        labeled_segments,
        sample_rate=SILERO_SR,
        model_size=cfg.whisper_model,
        device=cfg.device,
        compute_type=cfg.compute_type,
        language=cfg.language,
    )

    print("\nStage 5: Output")
    if cfg.print_output:
        pretty_print(transcribed)

    if cfg.save_path:
        save(transcribed, cfg.save_path, fmt=cfg.save_fmt)

    return transcribed
