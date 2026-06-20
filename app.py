"""
app.py — CLI Entrypoint
------------------------
Run the pipeline from the command line.

Usage examples:
  # Transcribe a file with 2 speakers
  python app.py --file meeting.wav --speakers 2

  # Denoise only (single speaker, noisy recording)
  python app.py --file call.wav --denoise

  # Auto-estimate number of speakers, save JSON output
  python app.py --file panel.wav --auto-speakers --save out.json

  # Use large whisper model on GPU
  python app.py --file long_meeting.wav --speakers 3 \
                --whisper large-v3 --device cuda --compute float16

  # Force English transcription
  python app.py --file audio.wav --speakers 2 --language en
"""

import argparse
import sys
from pathlib import Path

from pipeline import PipelineConfig, run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sound separation + speaker diarization + transcription pipeline"
    )

    # Input
    parser.add_argument("--file", required=True, help="Path to input audio file")

    # Stage 1 — Separation
    parser.add_argument(
        "--speakers", type=int, default=2, help="Number of speakers (default: 2)"
    )
    parser.add_argument(
        "--denoise", action="store_true", help="Denoise only (single speaker recording)"
    )

    # Stage 3 — Clustering
    parser.add_argument(
        "--auto-speakers",
        action="store_true",
        help="Automatically estimate number of speakers",
    )
    parser.add_argument(
        "--max-speakers",
        type=int,
        default=8,
        help="Max speakers to consider when auto-estimating (default: 8)",
    )

    # Stage 4 — Whisper
    parser.add_argument(
        "--whisper",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run Whisper on (default: cpu)",
    )
    parser.add_argument(
        "--compute",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type for Whisper (default: int8)",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="Language code e.g. 'en', 'fr' (default: auto-detect)",
    )

    # Output
    parser.add_argument("--save", default=None, help="Save transcript to this path")
    parser.add_argument(
        "--fmt",
        default="json",
        choices=["json", "plain", "rttm", "srt"],
        help="Output format (default: json)",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    audio_path = Path(args.file)
    if not audio_path.is_file():
        sys.exit(f"Error: input file not found: {args.file}")

    if args.compute == "int8" and args.device == "cuda":
        print(
            "Warning: compute_type 'int8' is intended for CPU. "
            "Consider --compute float16 on CUDA."
        )

    config = PipelineConfig(
        num_speakers=args.speakers,
        denoise_only=args.denoise,
        auto_num_speakers=args.auto_speakers,
        max_speakers=args.max_speakers,
        whisper_model=args.whisper,
        device=args.device,
        compute_type=args.compute,
        language=args.language,
        save_path=args.save,
        save_fmt=args.fmt,
        print_output=True,
    )

    results = run(args.file, config=config)
    print(f"\nDone. {len(results)} segment(s) transcribed.")


if __name__ == "__main__":
    main()
