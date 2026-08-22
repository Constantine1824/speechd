import numpy as np
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation
from speechbrain.utils.fetching import LocalStrategy

# Checkpoint names for each use case
DENOISE_MODEL = "speechbrain/sepformer-wham"
SEPARATE_2SPK = "speechbrain/sepformer-libri2mix"
SEPARATE_3SPK = "speechbrain/sepformer-libri3mix"

TARGET_SR = 8000  # SepFormer expects 8kHz audio

# Long inputs are separated in overlapping chunks: SepFormer's attention cost
# grows quadratically with sequence length, so a single full-length pass on a
# multi-minute file exhausts system memory.
CHUNK_SEC = 30
CHUNK_OVERLAP_SEC = 1

_model_cache: dict = {}


def _load_model(checkpoint: str, device: str = "cpu") -> SepformerSeparation:
    cache_key = (checkpoint, device)
    if cache_key not in _model_cache:
        print(f"Loading SepFormer checkpoint: {checkpoint} on {device}")
        _model_cache[cache_key] = SepformerSeparation.from_hparams(
            source=checkpoint,
            savedir=f"pretrained_models/{checkpoint.split('/')[-1]}",
            run_opts={"device": device},
            # Windows: symlinking fetched files requires elevated privileges
            local_strategy=LocalStrategy.COPY,
        )
    return _model_cache[cache_key]


def resample(audio: np.ndarray, orig_sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """Resample audio to the target sample rate SepFormer expects."""
    if orig_sr == target_sr:
        return audio
    waveform = torch.tensor(audio).unsqueeze(0)
    resampled = torchaudio.functional.resample(waveform, orig_sr, target_sr)
    return resampled.squeeze(0).numpy()


def separate(
    audio: np.ndarray,
    sample_rate: int,
    num_speakers: int = 2,
    denoise_only: bool = False,
    device: str = "cpu",
) -> list[np.ndarray]:
    """
    Separate or denoise an audio signal.

    Parameters
    ----------
    audio        : 1-D numpy array of raw audio samples (mono)
    sample_rate  : sample rate of the input audio
    num_speakers : expected number of speakers (used to pick checkpoint)
                   ignored when denoise_only=True
    denoise_only : if True, runs denoising instead of speaker separation
    device       : "cpu" or "cuda" — where to run the SepFormer model

    Returns
    -------
    sources : list of 1-D numpy arrays, one per separated source
              (length = 1 for denoising, = num_speakers for separation)
    """
    # Pick the right checkpoint
    if denoise_only:
        checkpoint = DENOISE_MODEL
    elif num_speakers == 2:
        checkpoint = SEPARATE_2SPK
    elif num_speakers == 3:
        checkpoint = SEPARATE_3SPK
    else:
        raise ValueError(
            f"Separation supports 2 or 3 speakers, got num_speakers={num_speakers}. "
            f"Use --denoise for single-speaker recordings."
        )

    model = _load_model(checkpoint, device=device)

    # SepFormer expects 8kHz mono
    audio_8k = resample(audio, sample_rate, TARGET_SR)

    sources = _separate_chunked(model, audio_8k, device)

    print(f"Produced {len(sources)} source(s) at {TARGET_SR}Hz")
    return sources


def _separate_chunked(
    model: SepformerSeparation, audio_8k: np.ndarray, device: str
) -> list[np.ndarray]:
    """
    Run separation over overlapping chunks and stitch with overlap-add.

    SepFormer's per-chunk output order is arbitrary (source 0/1 may swap
    between chunks); the short crossfade keeps discontinuities minor and the
    downstream VAD/embedding/overlap-resolution stages tolerate occasional
    swaps across chunk boundaries.
    """
    total = len(audio_8k)
    chunk_len = CHUNK_SEC * TARGET_SR
    overlap = CHUNK_OVERLAP_SEC * TARGET_SR
    hop = chunk_len - overlap

    num_chunks = 1 if total <= chunk_len else -(-(total - overlap) // hop)
    if num_chunks > 1:
        print(
            f"Separating in {num_chunks} chunks of {CHUNK_SEC}s "
            f"({CHUNK_OVERLAP_SEC}s overlap)"
        )

    # Determine number of sources from the first chunk
    with torch.no_grad():
        mix = torch.tensor(audio_8k[:chunk_len]).unsqueeze(0).to(device)
        est = model.separate_batch(mix).squeeze(0).cpu().numpy()  # (t, n_src)
    num_src = est.shape[1]

    outputs = [np.zeros(total, dtype=np.float32) for _ in range(num_src)]
    weights = np.zeros(total, dtype=np.float32)

    def accumulate(start: int, est_chunk: np.ndarray) -> None:
        end = start + est_chunk.shape[0]
        for i in range(num_src):
            outputs[i][start:end] += est_chunk[:, i]
        weights[start:end] += 1.0

    accumulate(0, est)

    for start in range(hop, total, hop):
        chunk = audio_8k[start : start + chunk_len]
        with torch.no_grad():
            mix = torch.tensor(chunk).unsqueeze(0).to(device)
            est = model.separate_batch(mix).squeeze(0).cpu().numpy()
        accumulate(start, est)
        print(f"  chunk {start // hop + 1}/{num_chunks} done")

    # Normalize overlap regions
    nonzero = weights > 0
    for i in range(num_src):
        outputs[i][nonzero] /= weights[nonzero]

    return outputs


def separate_from_file(
    filepath: str,
    num_speakers: int = 2,
    denoise_only: bool = False,
) -> tuple[list[np.ndarray], int]:
    """
    Convenience wrapper: load audio from file, run separation.

    Returns
    -------
    (sources, sample_rate) where sample_rate is TARGET_SR (8000)
    """
    waveform, sr = torchaudio.load(filepath)

    # Mix down to mono if stereo
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    audio = waveform.squeeze(0).numpy()
    sources = separate(audio, sr, num_speakers=num_speakers, denoise_only=denoise_only)
    return sources, TARGET_SR
