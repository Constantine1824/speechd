import numpy as np
import torch
import torchaudio
from speechbrain.inference.separation import SepformerSeparation

# Checkpoint names for each use case
DENOISE_MODEL = "speechbrain/sepformer-wham"
SEPARATE_2SPK = "speechbrain/sepformer-libri2mix"
SEPARATE_3SPK = "speechbrain/sepformer-libri3mix"

TARGET_SR = 8000  # SepFormer expects 8kHz audio

_model_cache: dict = {}


def _load_model(checkpoint: str, device: str = "cpu") -> SepformerSeparation:
    cache_key = (checkpoint, device)
    if cache_key not in _model_cache:
        print(f"Loading SepFormer checkpoint: {checkpoint} on {device}")
        _model_cache[cache_key] = SepformerSeparation.from_hparams(
            source=checkpoint,
            savedir=f"pretrained_models/{checkpoint.split('/')[-1]}",
            run_opts={"device": device},
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

    # Convert to tensor shape (1, T) expected by SpeechBrain, on the model's device
    mix_tensor = torch.tensor(audio_8k).unsqueeze(0).to(device)

    # Run separation — returns tensor of shape (T, num_sources)
    with torch.no_grad():
        est_sources = model.separate_batch(mix_tensor)  # (1, T, num_sources)

    # Unpack into list of numpy arrays (move back to CPU before numpy conversion)
    est_sources = est_sources.squeeze(0).cpu()  # (T, num_sources)
    sources = [est_sources[:, i].numpy() for i in range(est_sources.shape[1])]

    print(f"Produced {len(sources)} source(s) at {TARGET_SR}Hz")
    return sources


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
