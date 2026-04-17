# SpeechSep

A modular end-to-end pipeline for audio source separation, speaker diarization, and transcription.

## Overview

SpeechSep processes audio files to:
1. Separate speakers or denoise audio
2. Detect speech segments (Voice Activity Detection)
3. Extract speaker embeddings and cluster speakers
4. Transcribe each speaker's segments
5. Output formatted transcripts with speaker labels and timestamps

## Features

- **Source Separation**: Uses SepFormer models for 2/3 speaker separation or denoising
- **Voice Activity Detection**: Powered by silero-vad
- **Speaker Embeddings**: ECAPA-TDNN for robust speaker representation
- **Speaker Clustering**: Spectral clustering with automatic speaker estimation
- **Transcription**: Faster-Whisper with multiple model sizes
- **Flexible Output**: JSON, plain text, RTTM, and SRT formats
- **GPU Acceleration**: CUDA support for compatible models

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd SpeechSep

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

Basic command-line usage:

```bash
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
```

## Output Formats

- **JSON**: Structured data with speaker, timestamps, text, and confidence
- **Plain Text**: Human-readable transcript with speaker labels
- **RTTM**: Standard format for diarization evaluation
- **SRT**: Subtitle format for video synchronization

## Requirements

See `requirements.txt` for detailed dependencies. Key packages include:
- torch, torchaudio
- speechbrain
- silero-vad
- faster-whisper
- scikit-learn
- numpy

## Project Structure

```
SpeechSep/
├── app.py              # CLI entrypoint
├── pipeline.py         # Main orchestration pipeline
├── separate.py         # Source separation/denoising
├── vad.py              # Voice activity detection
├── embed.py            # Speaker embedding extraction
├── cluster.py          # Speaker clustering
├── transcribe.py       # ASR transcription
├── output.py           # Transcript formatting and saving
├── types.py            # Data classes and configuration
└── README.md
```

## Future Improvements

This is a base implementation. Planned enhancements include:
- Real-time audio stream processing
- Improved speaker diarization accuracy
- Additional separation models
- Language identification
- Web API interface
- Docker containerization
- Comprehensive test suite
- Performance benchmarking

## License

MIT