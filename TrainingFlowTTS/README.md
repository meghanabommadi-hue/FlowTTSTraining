# FlowTTS Training

Fine-tuning pipeline for training multilingual TTS models (Kannada, Telugu) based on [MiraTTS](https://huggingface.co/YatharthS/MiraTTS) using Unsloth + SFT.

---

## Structure

```
FlowTTSTraining/
├── config.py                  # Paths, hyperparams, env setup
├── download_english_data.py   # Stream & save English TTS data
├── requirements.txt
├── audio/
│   ├── ref/                   # Reference WAVs for voice cloning
│   └── outputs/               # Generated audio (gitignored)
├── utils/
│   ├── codec_utils.py         # TTSCodec setup + audio → token encoding
│   ├── dataset_utils.py       # Dataset loading, mixing, caching
│   ├── infer_utils.py         # Inference helper
│   └── logger.py              # Tee logging to file + stdout
├── training/
│   └── train.py               # Main training entry point
├── inference/
│   └── infer.py               # Run inference from a checkpoint
└── testing/
    └── test_train.py          # Quick end-to-end test on N samples
```

---

## Setup

```bash
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token
```

---

## Training

```bash
# Kannada (default)
python3 training/train.py --lang kannada

# Telugu
python3 training/train.py --lang telugu --model YatharthS/MiraTTS

# Custom base model, limit samples, more epochs
python3 training/train.py \
    --lang kannada \
    --model Shubhangi7/mira-english-1-epoch \
    --num_samples 50000 \
    --epochs 3
```

Checkpoints are saved to `~/data_preparation/flowtts-kannada-training/outputs-<lang>-latest/` and pushed to HuggingFace Hub automatically.

The encoded dataset cache is saved after the first run — subsequent runs skip encoding and go straight to training.

---

## Inference

```bash
# Run all default test sentences
python3 inference/infer.py --lang kannada --checkpoint /path/to/checkpoint

# Single custom text
python3 inference/infer.py \
    --checkpoint /path/to/checkpoint \
    --text "ನಮಸ್ಕಾರ! ಹೇಗಿದ್ದೀರಾ?" \
    --ref-wav audio/ref/simran.wav \
    --out output.wav
```

Output WAVs are saved to `audio/outputs/`.

---

## Quick Test

Verify the full pipeline works before committing to a long training run:

```bash
python3 testing/test_train.py --lang kannada --num_samples 100
```

---

## English Data

The training mix uses 75% target language + 25% English. To download and prepare the English dataset (~3.5 GB):

```bash
python3 download_english_data.py
```

This streams `parler-tts/libritts_r_filtered`, filters to Latin/ASCII only, and saves to `~/data_preparation/english_latin_only/`.

---

## Supported Languages

| Language | Code     | Dataset                        |
|----------|----------|-------------------------------|
| Kannada  | `kannada`| MeghanaKap/kannada_dataset    |
| Telugu   | `telugu` | *(bring your own dataset)*    |

---

## Models

| Model | Description |
|-------|-------------|
| `YatharthS/MiraTTS` | Base MiraTTS model |
| `Shubhangi7/mira-english-1-epoch` | MiraTTS fine-tuned on English (used as Kannada base) |
| `MeghanaKap/FlowTTSKannada` | Output — Kannada fine-tuned model |
