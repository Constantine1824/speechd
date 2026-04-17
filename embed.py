import numpy as np
import torch
import torchaudio
from speechbrain.inference.speaker import EncoderClassifier

from .types import Segment

ECAPA_CHECKPOINT = "speechbrain/spkrec-ecapa-voxceleb"
ECAPA_SR = 16000  # ECAPA-TDNN expects 16kHz audio

_embedder = None


def _load_embedder() -> EncoderClassifier:
    global _embedder
    if _embedder is None:
        print(f"[embed] Loading ECAPA-TDNN from {ECAPA_CHECKPOINT}...")
        _embedder = EncoderClassifier.from_hparams(
            source=ECAPA_CHECKPOINT,
            savedir="pretrained_models/ecapa-tdnn",
            run_opts={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        )
    return _embedder


# ---------------------------------------------------------------------------
# Core API
# ---------------------------------------------------------------------------


def extract_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """
    Extract a single speaker embedding from an audio chunk.

    Parameters
    ----------
    audio       : 1-D numpy array of audio samples
    sample_rate : sample rate of the input audio

    Returns
    -------
    embedding : 1-D numpy array of shape (192,), L2-normalized
    """
    embedder = _load_embedder()

    # Resample to 16kHz if needed
    if sample_rate != ECAPA_SR:
        waveform = torch.tensor(audio).unsqueeze(0)
        waveform = torchaudio.functional.resample(waveform, sample_rate, ECAPA_SR)
        audio = waveform.squeeze(0).numpy()

    # ECAPA expects (batch, time) tensor
    audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0)

    with torch.no_grad():
        embedding = embedder.encode_batch(audio_tensor)  # (1, 1, 192)

    embedding = embedding.squeeze().numpy()  # (192,)

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    return embedding


def extract_embeddings(
    segments: list[Segment],
    sample_rate: int = ECAPA_SR,
    min_duration_sec: float = 0.5,
) -> tuple[list[np.ndarray], list[Segment]]:
    """
    Extract embeddings for a list of speech segments.

    Segments shorter than min_duration_sec are skipped (too short to embed
    reliably) and excluded from the returned lists.

    Parameters
    ----------
    segments        : list of Segment objects from vad.py
    sample_rate     : sample rate of audio in the segments
    min_duration_sec: skip segments shorter than this

    Returns
    -------
    (embeddings, valid_segments)
      embeddings     : list of (192,) numpy arrays
      valid_segments : segments that were actually embedded (skips too-short ones)
    """
    embeddings = []
    valid_segments = []

    for i, seg in enumerate(segments):
        if seg.duration < min_duration_sec:
            print(f"[embed] Skipping segment {i} (too short: {seg.duration:.2f}s)")
            continue

        emb = extract_embedding(seg.audio, sample_rate)
        embeddings.append(emb)
        valid_segments.append(seg)

    print(
        f"[embed] Extracted {len(embeddings)} embeddings from {len(segments)} segments"
    )
    return embeddings, valid_segments
