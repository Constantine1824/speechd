"""Overlap resolution for separated-source diarization.

Running VAD independently on each separated source produces duplicate segments
for the same moment of audio: SepFormer leaks each speaker into the other
source channels, so one real utterance is detected once per source. After
clustering and transcription these show up as the same words repeated.

`resolve_overlaps` removes those leakage duplicates while preserving genuine
simultaneous speech:

  * Two segments that overlap substantially in time AND share the same speaker
    are treated as the same utterance captured twice -> the lower-confidence
    copy is dropped.
  * Overlaps between *different* speakers are kept -- that is real overlapped
    speech, which is the whole reason separation runs in the first place.
"""

from speechsep.schemas import TranscribedSegment


def _overlap_ratio(a: TranscribedSegment, b: TranscribedSegment) -> float:
    """Fraction of the shorter segment that overlaps the other (0.0–1.0)."""
    intersection = min(a.end, b.end) - max(a.start, b.start)
    if intersection <= 0:
        return 0.0
    shorter = min(a.end - a.start, b.end - b.start)
    if shorter <= 0:
        return 0.0
    return intersection / shorter


def _keeps_first(a: TranscribedSegment, b: TranscribedSegment) -> bool:
    """Return True if `a` should be kept over `b` when they are duplicates.

    Prefer higher confidence, then longer duration, then earlier start, then
    lower source_id — every tiebreak is deterministic so results are stable.
    """
    if a.confidence != b.confidence:
        return a.confidence > b.confidence
    da, db = a.end - a.start, b.end - b.start
    if da != db:
        return da > db
    if a.start != b.start:
        return a.start < b.start
    return a.source_id <= b.source_id


def resolve_overlaps(
    segments: list[TranscribedSegment],
    overlap_threshold: float = 0.5,
) -> list[TranscribedSegment]:
    """
    Drop same-speaker duplicate segments caused by source-separation leakage.

    Parameters
    ----------
    segments          : transcribed segments (any order)
    overlap_threshold : minimum overlap ratio (relative to the shorter segment)
                        for two same-speaker segments to count as duplicates.
                        Small incidental boundary overlaps between consecutive
                        turns stay below this and are left untouched.

    Returns
    -------
    Surviving segments in chronological order.
    """
    n = len(segments)
    if n <= 1:
        return list(segments)

    # Process in time order; longer segment first on ties so it tends to win.
    order = sorted(
        range(n),
        key=lambda i: (segments[i].start, -(segments[i].end - segments[i].start)),
    )

    removed: set[int] = set()
    for pos, i in enumerate(order):
        if i in removed:
            continue
        a = segments[i]
        for j in order[pos + 1:]:
            b = segments[j]
            # order is sorted by start: once b starts at/after a ends, nothing
            # further can overlap a.
            if b.start >= a.end:
                break
            if j in removed or b.speaker_id != a.speaker_id:
                continue
            if _overlap_ratio(a, b) < overlap_threshold:
                continue
            if _keeps_first(a, b):
                removed.add(j)
            else:
                removed.add(i)
                break  # a is gone; stop comparing against it

    survivors = [segments[i] for i in range(n) if i not in removed]
    survivors.sort(key=lambda s: s.start)

    dropped = n - len(survivors)
    if dropped:
        print(f"[overlap] Dropped {dropped} duplicate segment(s) from source leakage")
    return survivors
