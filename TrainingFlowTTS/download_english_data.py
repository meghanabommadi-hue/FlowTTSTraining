"""
Downloads Indian English TTS data (MeghanaKap/IndianEnglishCleanYoutubeData) via streaming,
caps at ~3.5 GB, and saves to disk in the format expected by train.py.

Output: ~/data_preparation/english_latin_only/
Columns: text (string), audio (dict with bytes + path)
"""

import os
import io
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

import soundfile as sf
import numpy as np
import librosa
from datasets import load_dataset, Dataset, Audio
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
MAX_BYTES  = 3.5 * 1024 ** 3
SAVE_PATH  = os.path.join(os.path.expanduser("~"), "data_preparation", "english_latin_only")
TARGET_SR  = 16_000
DATASET_ID = "MeghanaKap/IndianEnglishCleanYoutubeData"

# ── streaming download ─────────────────────────────────────────────────────────
print(f"Streaming {DATASET_ID} ...")
ds = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{DATASET_ID}/data/*.parquet",
    split="train",
    streaming=True,
    token=HF_TOKEN,
)

rows = []
total_bytes = 0

for sample in tqdm(ds, unit="rows"):
    text = (sample.get("text") or "").strip()
    if not text:
        continue

    audio_bytes = sample["audio"].get("bytes")
    if not audio_bytes or len(audio_bytes) < 100:
        continue

    try:
        audio_array, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32")
    except Exception:
        continue

    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    if sr != TARGET_SR:
        audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)

    buf = io.BytesIO()
    sf.write(buf, audio_array, TARGET_SR, format="WAV", subtype="PCM_16")
    wav_bytes = buf.getvalue()

    rows.append({
        "text":  text,
        "audio": {"bytes": wav_bytes, "path": sample.get("audio_id", "")},
    })
    total_bytes += len(wav_bytes)

    if total_bytes >= MAX_BYTES:
        print(f"\nReached {total_bytes / 1024**3:.2f} GB cap — stopping.")
        break

print(f"\nCollected {len(rows)} rows ({total_bytes / 1024**3:.2f} GB audio)")

# ── save to disk ───────────────────────────────────────────────────────────────
print(f"Saving to {SAVE_PATH} ...")
dataset = Dataset.from_list(rows)
dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR, decode=False))
os.makedirs(SAVE_PATH, exist_ok=True)
dataset.save_to_disk(SAVE_PATH)
print(f"Done. Saved {len(dataset)} rows to {SAVE_PATH}")
os.system(f"du -sh {SAVE_PATH}")
