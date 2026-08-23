"""Tests for the FastAPI server. `run` is mocked — no model downloads."""

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import speechsep.api as api
from speechsep.schemas import PipelineConfig, TranscribedSegment


@pytest.fixture
def client():
    return TestClient(api.app)


def _wav_bytes(seconds: float = 0.1, sr: int = 16000) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(sr * seconds), dtype="float32"), sr, format="WAV")
    return buf.getvalue()


def _segments():
    return [
        TranscribedSegment(
            start=0.0,
            end=2.5,
            speaker_id=0,
            speaker_label="SPEAKER_00",
            text="Hello there.",
            language="en",
            confidence=0.91234,
        ),
        TranscribedSegment(
            start=2.6,
            end=5.0,
            speaker_id=1,
            speaker_label="SPEAKER_01",
            text="Hi, how are you?",
            language="en",
            confidence=0.88,
        ),
    ]


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert "version" in body and "device" in body


def test_transcribe_happy_path(client, monkeypatch):
    monkeypatch.setattr(api, "run", lambda path, config: _segments())
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"speakers": "2", "language": "en"},
    )
    assert res.status_code == 200
    body = res.json()

    assert body["speakers"] == ["SPEAKER_00", "SPEAKER_01"]
    assert isinstance(body["duration_ms"], int)
    assert len(body["segments"]) == 2

    first = body["segments"][0]
    assert first["speaker"] == "SPEAKER_00"
    assert first["speaker_id"] == 0
    assert first["start"] == 0.0
    assert first["end"] == 2.5
    assert first["text"] == "Hello there."
    assert first["language"] == "en"
    assert first["confidence"] == 0.9123  # rounded to 4 dp


def test_transcribe_passes_options_to_pipeline(client, monkeypatch):
    captured = {}

    def fake_run(path, config):
        captured["path"] = path
        captured["config"] = config
        return []

    monkeypatch.setattr(api, "run", fake_run)
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={
            "speakers": "3",
            "auto_speakers": "false",
            "language": "fr",
            "whisper_model": "small",
        },
    )
    assert res.status_code == 200

    cfg: PipelineConfig = captured["config"]
    assert cfg.num_speakers == 3
    assert cfg.language == "fr"
    assert cfg.whisper_model == "small"
    assert cfg.print_output is False
    assert cfg.save_path is None
    # device/compute stay server-controlled, not client-controlled
    assert cfg.device in {"cpu", "cuda"}
    assert cfg.compute_type in {"int8", "float16", "float32"}


def test_transcribe_rejects_unsupported_extension(client):
    res = client.post(
        "/transcribe",
        files={"file": ("notes.txt", b"hello", "text/plain")},
    )
    assert res.status_code == 400
    assert "Unsupported file type" in res.json()["detail"]


def test_transcribe_rejects_bad_speaker_count(client):
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"speakers": "4"},
    )
    assert res.status_code == 400


def test_transcribe_rejects_denoise_with_auto_speakers(client):
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"denoise": "true", "auto_speakers": "true"},
    )
    assert res.status_code == 400


def test_transcribe_rejects_bad_whisper_model(client):
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"whisper_model": "gigantic"},
    )
    assert res.status_code == 400


def test_transcribe_rejects_empty_file(client):
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", b"", "audio/wav")},
    )
    assert res.status_code == 400
    assert "empty" in res.json()["detail"]


def test_transcribe_maps_value_error_to_400(client, monkeypatch):
    def fake_run(path, config):
        raise ValueError("segment too short")

    monkeypatch.setattr(api, "run", fake_run)
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
    )
    assert res.status_code == 400
    assert res.json()["detail"] == "segment too short"


def test_transcribe_maps_decode_failure_to_400(client, monkeypatch):
    def fake_run(path, config):
        raise sf.LibsndfileError(1, "System error")

    monkeypatch.setattr(api, "run", fake_run)
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", b"not really audio", "audio/wav")},
    )
    assert res.status_code == 400
    assert "decode" in res.json()["detail"]


def test_transcribe_maps_model_failure_to_500(client, monkeypatch):
    def fake_run(path, config):
        raise RuntimeError("could not download model")

    monkeypatch.setattr(api, "run", fake_run)
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
    )
    assert res.status_code == 500
    assert res.json()["detail"] == "could not download model"


def test_transcribe_treats_empty_language_as_autodetect(client, monkeypatch):
    captured = {}

    def fake_run(path, config):
        captured["config"] = config
        return []

    monkeypatch.setattr(api, "run", fake_run)
    res = client.post(
        "/transcribe",
        files={"file": ("clip.wav", _wav_bytes(), "audio/wav")},
        data={"language": "  "},
    )
    assert res.status_code == 200
    assert captured["config"].language is None
