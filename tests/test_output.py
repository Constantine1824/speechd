import json

import pytest

from speechsep.output import (
    _fmt_time,
    _srt_time,
    save,
    to_json,
    to_plain_text,
    to_rttm,
    to_srt,
)
from speechsep.schemas import TranscribedSegment


@pytest.fixture
def segments():
    return [
        TranscribedSegment(
            start=0.0,
            end=4.21,
            speaker_id=0,
            speaker_label="SPEAKER_00",
            text="Hey, what did you think of the meeting?",
            language="en",
            confidence=0.91,
        ),
        TranscribedSegment(
            start=4.5,
            end=9.1,
            speaker_id=1,
            speaker_label="SPEAKER_01",
            text="It went well.",
            language="en",
            confidence=0.88,
        ),
    ]


def test_fmt_time_under_an_hour():
    assert _fmt_time(0.0) == "00:00.000"
    assert _fmt_time(65.5) == "01:05.500"


def test_fmt_time_over_an_hour():
    # 1h 1m 5.5s
    assert _fmt_time(3665.5) == "1:01:05.500"


def test_srt_time_format():
    assert _srt_time(3661.250) == "01:01:01,250"


def test_to_json_roundtrip(segments):
    data = json.loads(to_json(segments))
    assert len(data) == 2
    assert data[0]["speaker"] == "SPEAKER_00"
    assert data[0]["speaker_id"] == 0
    assert data[0]["start"] == 0.0
    assert data[0]["end"] == 4.21
    assert data[1]["text"] == "It went well."
    assert data[0]["confidence"] == 0.91


def test_to_plain_text_contains_labels_and_text(segments):
    out = to_plain_text(segments)
    assert "SPEAKER_00" in out
    assert "Hey, what did you think of the meeting?" in out


def test_to_rttm_line_shape(segments):
    out = to_rttm(segments, filename="meeting")
    first = out.splitlines()[0].split()
    assert first[0] == "SPEAKER"
    assert first[1] == "meeting"
    assert first[2] == "1"
    assert first[3] == "0.000"  # start
    assert first[4] == "4.210"  # duration
    assert first[7] == "SPEAKER_00"


def test_to_srt_indexed_blocks(segments):
    out = to_srt(segments)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert "-->" in lines[1]
    assert "[SPEAKER_00]" in lines[2]


def test_save_rejects_unknown_format(segments, tmp_path):
    with pytest.raises(ValueError):
        save(segments, str(tmp_path / "x.txt"), fmt="yaml")


@pytest.mark.parametrize("fmt", ["json", "plain", "rttm", "srt"])
def test_save_writes_file(segments, tmp_path, fmt):
    path = tmp_path / f"out.{fmt}"
    save(segments, str(path), fmt=fmt)
    assert path.exists()
    assert path.read_text(encoding="utf-8").strip() != ""
