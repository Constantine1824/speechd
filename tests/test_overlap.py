from speechsep.pipeline.overlap import _overlap_ratio, resolve_overlaps
from speechsep.schemas import TranscribedSegment


def seg(start, end, speaker_id, conf=0.9, source_id=0, text="x"):
    return TranscribedSegment(
        start=start,
        end=end,
        speaker_id=speaker_id,
        speaker_label=f"SPEAKER_{speaker_id:02d}",
        text=text,
        confidence=conf,
        source_id=source_id,
    )


def test_overlap_ratio_full_containment():
    a = seg(0.0, 2.0, 0)
    b = seg(0.5, 1.0, 0)  # fully inside a
    assert _overlap_ratio(a, b) == 1.0


def test_overlap_ratio_disjoint():
    assert _overlap_ratio(seg(0.0, 1.0, 0), seg(2.0, 3.0, 0)) == 0.0


def test_overlap_ratio_touching_is_zero():
    assert _overlap_ratio(seg(0.0, 1.0, 0), seg(1.0, 2.0, 0)) == 0.0


def test_same_speaker_duplicate_drops_lower_confidence():
    a = seg(0.0, 2.0, 0, conf=0.95, source_id=0)
    b = seg(0.05, 2.0, 0, conf=0.40, source_id=1)  # leakage copy
    out = resolve_overlaps([a, b])
    assert len(out) == 1
    assert out[0].confidence == 0.95
    assert out[0].source_id == 0


def test_different_speakers_overlap_kept():
    a = seg(0.0, 2.0, 0, source_id=0)
    b = seg(0.1, 2.1, 1, source_id=1)  # genuine simultaneous speech
    out = resolve_overlaps([a, b])
    assert len(out) == 2


def test_subthreshold_overlap_kept():
    # Same speaker but only a sliver of overlap -> consecutive turns, keep both.
    a = seg(0.0, 2.0, 0)
    b = seg(1.9, 4.0, 0)  # overlap 0.1s over shorter (2.0s) = 0.05 < 0.5
    out = resolve_overlaps([a, b])
    assert len(out) == 2


def test_non_overlapping_unchanged_and_ordered():
    a = seg(4.0, 5.0, 1)
    b = seg(0.0, 1.0, 0)
    c = seg(2.0, 3.0, 0)
    out = resolve_overlaps([a, b, c])
    assert [s.start for s in out] == [0.0, 2.0, 4.0]


def test_confidence_tie_prefers_longer_segment():
    a = seg(0.0, 1.0, 0, conf=0.8, source_id=0)
    b = seg(0.0, 2.0, 0, conf=0.8, source_id=1)  # longer wins on tie
    out = resolve_overlaps([a, b])
    assert len(out) == 1
    assert out[0].end == 2.0


def test_three_way_duplicate_keeps_best_only():
    a = seg(0.0, 2.0, 0, conf=0.5, source_id=0)
    b = seg(0.05, 2.0, 0, conf=0.9, source_id=1)
    c = seg(0.1, 2.0, 0, conf=0.7, source_id=2)
    out = resolve_overlaps([a, b, c])
    assert len(out) == 1
    assert out[0].confidence == 0.9


def test_empty_and_single_inputs():
    assert resolve_overlaps([]) == []
    one = [seg(0.0, 1.0, 0)]
    assert resolve_overlaps(one) == one
