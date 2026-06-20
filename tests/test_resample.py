import numpy as np

from speechsep.main import _resample_np
from speechsep.pipeline.separate import resample


def test_resample_np_identity_when_rates_match():
    audio = np.linspace(-1, 1, 100, dtype=np.float32)
    out = _resample_np(audio, 16000, 16000)
    # Same rate => returned unchanged.
    assert out is audio


def test_resample_np_upsample_doubles_length():
    audio = np.zeros(8000, dtype=np.float32)  # 1s at 8kHz
    out = _resample_np(audio, 8000, 16000)
    # 1s at 16kHz => ~16000 samples.
    assert abs(len(out) - 16000) <= 2


def test_separate_resample_identity():
    audio = np.ones(50, dtype=np.float32)
    out = resample(audio, 8000, 8000)
    assert out is audio


def test_separate_resample_downsample_halves_length():
    audio = np.zeros(16000, dtype=np.float32)  # 1s at 16kHz
    out = resample(audio, 16000, 8000)
    assert abs(len(out) - 8000) <= 2
