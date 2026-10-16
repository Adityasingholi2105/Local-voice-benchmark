
# Local Voice AI Pipeline and Benchmark

A local voice assistant that converts speech to text, generates a reply with a small language model, and speaks the reply aloud.

## Workflow

Audio → faster-whisper (STT) → Ollama / Qwen2.5 1.5B (LLM) → Piper (TTS) → `data/reply.wav`

## Technology and hardware

- Python 3.13.9
- faster-whisper for speech recognition
- Ollama with Qwen2.5 1.5B for local response generation
- Piper with the `en_US-lessac-medium` voice for text to speech
- JiWER for word error rate (WER)
- RTX 3050 Laptop GPU, 4 GB VRAM

Whisper runs on the CPU using int8. Ollama uses the GPU for the language model.

## Benchmark results

Results are from 10 self-recorded English clips on this laptop. Mean times are calculated from the saved per-clip CSV results.

| Whisper model | Device / compute type | Overall WER ↓ | Mean time per clip ↓ |
| tiny          | CPU / int8            | 12.1%         | 0.424 s              |
| base          | CPU / int8            | 6.9%          | 0.649 s              |

On this evaluation set, `base` had lower WER but took longer per clip. These results apply to this small test set and laptop; they are not general accuracy claims.

GPU memory was observed at 1,203 MiB out of 4,096 MiB during one full-pipeline run. This was a single sample, not a peak measurement.

## Setup

Requirements: Python 3.13.9 and Ollama installed.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
ollama pull qwen2.5:1.5b
python -m piper.download_voices --download-dir voices en_US-lessac-medium
```

## Run the project

Set the input audio path in `main.py`, then run:

```powershell
python main.py
```

The spoken response is saved to `data/reply.wav`.

## Run the STT benchmark

Place your recordings in `data/eval/` as `clip_01.m4a` through `clip_10.m4a`. The matching reference sentences are in `data/eval/references.csv`.

Run:

```powershell
python benchmark.py
```

The script is currently configured for the `base` model and saves to `results/base_int8.csv`. To run `tiny`, change the model name and output filename in `benchmark.py`. Audio recordings and downloaded model files are excluded from this repository; record your own clips to reproduce the evaluation.

## Third-party models and licenses

Piper TTS declares GPL-3.0-or-later. The Lessac voice model card points to the Blizzard 2013 dataset license. Downloaded model and voice files are not included; check their respective terms before redistributing them.

- [Piper license information](https://github.com/OHF-Voice/piper1-gpl/blob/main/setup.py)
- [Lessac voice model card](https://huggingface.co/rhasspy/piper-voices/blob/main/en/en_US/lessac/medium/MODEL_CARD)

This project does not currently include a license for its own code.