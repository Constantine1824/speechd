import numpy as np

from speechsep.schemas import LabeledSegment, Segment, TranscribedSegment


def test_segment_duration():
    seg = Segment(start=1.0, end=3.5, audio=np.zeros(10))
    assert seg.duration == 2.5


def test_labeled_segment_proxies_segment_fields():
    seg = Segment(start=2.0, end=4.0, audio=np.arange(5))
    ls = LabeledSegment(segment=seg, speaker_id=3)
    assert ls.start == 2.0
    assert ls.end == 4.0
    assert np.array_equal(ls.audio, np.arange(5))


def test_speaker_label_is_zero_padded():
    seg = Segment(start=0.0, end=1.0, audio=np.zeros(1))
    assert LabeledSegment(segment=seg, speaker_id=0).speaker_label == "SPEAKER_00"
    assert LabeledSegment(segment=seg, speaker_id=7).speaker_label == "SPEAKER_07"
    assert LabeledSegment(segment=seg, speaker_id=12).speaker_label == "SPEAKER_12"


def test_transcribed_segment_defaults():
    ts = TranscribedSegment(
        start=0.0, end=1.0, speaker_id=1, speaker_label="SPEAKER_01", text="hi"
    )
    assert ts.language == "en"
    assert ts.confidence == 1.0
