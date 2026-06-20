import numpy as np
import torch
import torchaudio

from schemas import Segment

_vad_model = None
_get_speech_timestamps = None


def _load_vad():
    global _vad_model, _get_speech_timestamps
    if _vad_model is None:
        print("Loading silero-vad...")
        _vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        _get_speech_timestamps = utils[0]  # get_speech_timestamps
    return _vad_model, _get_speech_timestamps


SILERO_SR = 16000  # silero-vad runs at 16kHz


def detect_speech(
    audio: np.ndarray,
    sample_rate: int,
    threshold: float = 0.5,
    min_speech_duration_ms: int = 250,
    min_silence_duration_ms: int = 100,
    padding_ms: int = 30,
    source_id: int = -1,
) -> list[Segment]:
    """
    Detect speech segments in an audio array.

    Parameters
    ----------
    audio                   : 1-D numpy array of audio samples
    sample_rate             : sample rate of input audio
    threshold               : VAD confidence threshold (0–1); lower = more sensitive
    min_speech_duration_ms  : discard speech chunks shorter than this
    min_silence_duration_ms : fill gaps shorter than this (merges nearby speech)
    padding_ms              : add padding around each detected segment
    source_id               : index of the separated source these segments came
                              from; tagged onto each Segment for later overlap
                              resolution (-1 = unknown / single source)

    Returns
    -------
    List of Segment objects with timestamps and audio slices
    """
    model, get_speech_timestamps = _load_vad()

    # Resample to 16kHz if needed
    if sample_rate != SILERO_SR:
        waveform = torch.tensor(audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sample_rate, SILERO_SR)
        audio_16k = waveform.squeeze(0).numpy()
    else:
        audio_16k = audio

    audio_tensor = torch.tensor(audio_16k, dtype=torch.float32)

    # Get raw timestamp dicts from silero [{start: N, end: N}, ...]
    raw_timestamps = get_speech_timestamps(
        audio_tensor,
        model,
        threshold=threshold,
        sampling_rate=SILERO_SR,
        min_speech_duration_ms=min_speech_duration_ms,
        min_silence_duration_ms=min_silence_duration_ms,
        speech_pad_ms=padding_ms,
        return_seconds=False,  # returns sample indices
    )

    segments = []
    for ts in raw_timestamps:
        start_sample = ts["start"]
        end_sample = ts["end"]
        start_sec = start_sample / SILERO_SR
        end_sec = end_sample / SILERO_SR
        chunk = audio_16k[start_sample:end_sample]
        segments.append(
            Segment(start=start_sec, end=end_sec, audio=chunk, source_id=source_id)
        )

    print(f"[vad] Found {len(segments)} speech segment(s)")
    return segments


def detect_speech_from_file(filepath: str, **kwargs) -> tuple[list[Segment], int]:
    """
    Load audio from file, run VAD.

    Returns
    -------
    (segments, sample_rate) where sample_rate is SILERO_SR (16000)
    """
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    audio = waveform.squeeze(0).numpy()
    segments = detect_speech(audio, sr, **kwargs)
    return segments, SILERO_SR
