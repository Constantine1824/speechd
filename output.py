import json

from schemas import TranscribedSegment


def _fmt_time(seconds: float) -> str:
    """Format seconds as mm:ss.ms, or hh:mm:ss.ms past one hour."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    if h:
        return f"{h:d}:{m:02d}:{s:06.3f}"
    return f"{m:02d}:{s:06.3f}"


def _srt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS,mmm for SRT files."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def pretty_print(segments: list[TranscribedSegment]) -> None:
    """Print transcript to terminal with timestamps and speaker labels."""
    print("\n" + "=" * 60)
    print("TRANSCRIPT")
    print("=" * 60)

    for seg in segments:
        ts = f"[{_fmt_time(seg.start)} → {_fmt_time(seg.end)}]"
        print(f"\n{ts}  {seg.speaker_label}")
        print(f"  {seg.text}")

    print("\n" + "=" * 60)


def to_plain_text(segments: list[TranscribedSegment]) -> str:
    """
    Format transcript as plain text.

    Example output:
        SPEAKER_00 [00:00.000 → 00:04.210]:
          Hey, what did you think of the meeting?

        SPEAKER_01 [00:04.500 → 00:09.100]:
          It went well, I think we aligned on the roadmap.
    """
    lines = []
    for seg in segments:
        ts = f"{_fmt_time(seg.start)} → {_fmt_time(seg.end)}"
        lines.append(f"{seg.speaker_label} [{ts}]:")
        lines.append(f"  {seg.text}")
        lines.append("")
    return "\n".join(lines)


def to_json(segments: list[TranscribedSegment], indent: int = 2) -> str:
    """
    Format transcript as JSON.

    Useful for passing to downstream APIs or storing results.
    """
    data = []
    for seg in segments:
        data.append(
            {
                "speaker": seg.speaker_label,
                "speaker_id": seg.speaker_id,
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "text": seg.text,
                "language": seg.language,
                "confidence": round(seg.confidence, 4),
            }
        )
    return json.dumps(data, indent=indent, ensure_ascii=False)


def to_rttm(segments: list[TranscribedSegment], filename: str = "audio") -> str:
    """
    Format transcript as RTTM (Rich Transcription Time Mark).

    Standard format for diarization evaluation with tools like pyannote-metrics.

    Format per line:
      SPEAKER <file> <channel> <start> <duration> <NA> <NA> <speaker> <NA> <NA>
    """
    lines = []
    for seg in segments:
        duration = round(seg.end - seg.start, 3)
        lines.append(
            f"SPEAKER {filename} 1 {seg.start:.3f} {duration:.3f} "
            f"<NA> <NA> {seg.speaker_label} <NA> <NA>"
        )
    return "\n".join(lines)


def to_srt(segments: list[TranscribedSegment]) -> str:
    """
    Format transcript as SRT subtitle file.

    Useful for syncing transcript to audio/video playback.
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{_srt_time(seg.start)} --> {_srt_time(seg.end)}")
        lines.append(f"[{seg.speaker_label}] {seg.text}")
        lines.append("")
    return "\n".join(lines)


def save(
    segments: list[TranscribedSegment],
    path: str,
    fmt: str = "json",
    filename: str = "audio",
) -> None:
    """
    Save transcript to file.

    Parameters
    ----------
    segments : transcribed segments from transcribe.py
    path     : output file path
    fmt      : one of "json", "plain", "rttm", "srt"
    filename : used as identifier in RTTM format
    """
    formatters = {
        "json": lambda: to_json(segments),
        "plain": lambda: to_plain_text(segments),
        "rttm": lambda: to_rttm(segments, filename=filename),
        "srt": lambda: to_srt(segments),
    }
    if fmt not in formatters:
        raise ValueError(
            f"Unknown format '{fmt}'. Choose from: {list(formatters.keys())}"
        )

    content = formatters[fmt]()
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"[output] Saved {fmt} transcript to: {path}")
