"""
Downloads English TTS data (parler-tts/libritts_r_filtered) via streaming,
filters to pure ASCII/Latin English only (no Hindi, Devanagari, etc.),
caps at ~3.5 GB, and saves to disk in the format expected by train_flowtts_kannada.py.

Output: ~/data_preparation/english_latin_only/
Columns: text (string), audio (dict with bytes + path)
"""

import os
import io
import re
import sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN

import soundfile as sf
import numpy as np
from datasets import load_dataset, Dataset, Audio, Features, Value
from tqdm import tqdm

# ── config ────────────────────────────────────────────────────────────────────
MAX_BYTES = 3.5 * 1024 ** 3          # 3.5 GB cap
SAVE_PATH = os.path.join(os.path.expanduser("~"), "data_preparation", "english_latin_only")
TARGET_SR = 16_000

# LibriTTS-R filtered: clean-100 + clean-360 splits (both clean English reads)
SPLITS = ["train.clean.100", "train.clean.360"]

# Only keep text that is purely Latin/ASCII — rejects Devanagari, Arabic, etc.
_ALLOWED = re.compile(r"^[\x00-\x7FÀ-ɏḀ-ỿ\s\[\]'\-,\.!?;:\"()]+$")

def is_clean_english(text: str) -> bool:
    return bool(text and _ALLOWED.match(text))

# ── streaming download ─────────────────────────────────────────────────────────
rows = []
total_bytes = 0

for split in SPLITS:
    print(f"\nStreaming {split} ...")
    ds = load_dataset(
        "parler-tts/libritts_r_filtered",
        "clean",
        split=split,
        streaming=True,
        token=HF_TOKEN,
    )

    for sample in tqdm(ds, desc=split, unit="rows"):
        text = sample.get("text_normalized", "") or sample.get("text_original", "")
        if not is_clean_english(text):
            continue

        audio_info = sample["audio"]
        # streaming=True gives decoded numpy array
        audio_array = audio_info["array"].astype(np.float32)
        sr = audio_info["sampling_rate"]

        if sr != TARGET_SR:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=TARGET_SR)

        # encode back to wav bytes (matching Kannada dataset format)
        buf = io.BytesIO()
        sf.write(buf, audio_array, TARGET_SR, format="WAV", subtype="PCM_16")
        audio_bytes = buf.getvalue()

        rows.append({
            "text": text.strip(),
            "audio": {"bytes": audio_bytes, "path": sample.get("path", "")},
        })
        total_bytes += len(audio_bytes)

        if total_bytes >= MAX_BYTES:
            print(f"\nReached {total_bytes / 1024**3:.2f} GB cap — stopping.")
            break

    if total_bytes >= MAX_BYTES:
        break

print(f"\nCollected {len(rows)} rows ({total_bytes / 1024**3:.2f} GB audio)")

# ── save to disk ───────────────────────────────────────────────────────────────
print(f"Saving to {SAVE_PATH} ...")
dataset = Dataset.from_list(rows)
dataset = dataset.cast_column("audio", Audio(sampling_rate=TARGET_SR, decode=False))
os.makedirs(SAVE_PATH, exist_ok=True)
dataset.save_to_disk(SAVE_PATH)
print(f"Done. Saved {len(dataset)} rows to {SAVE_PATH}")
print(f"Disk usage: ", end="")
os.system(f"du -sh {SAVE_PATH}")
