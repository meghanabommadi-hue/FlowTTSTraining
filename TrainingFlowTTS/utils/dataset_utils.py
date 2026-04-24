import os
import glob
import math
import datasets as _datasets
from datasets import load_dataset, Audio, concatenate_datasets
from functools import partial

from config import (
    LANG_SNAPSHOT, LANG_HF_REPO, ENGLISH_FILTERED_PATH, EN_100HOURS_SNAPSHOT, ENGLISH_RATIO,
    ENGLISH_HF_REPO, get_lang_base_path, get_encoded_cache_path,
    get_english_encoded_cache_path,
)
from codec_utils import build_codec, process_wavs


def _encode_and_cache(raw_dataset, cache_path, failed_log_path, desc):
    tts_codec = build_codec()
    encoded = raw_dataset.map(
        partial(process_wavs, tts_codec=tts_codec),
        remove_columns=["audio"],
        with_indices=True,
        desc=desc,
    )
    failed_ids = [int(i) for i in encoded["failed_idx"] if int(i) >= 0]
    os.makedirs(os.path.dirname(failed_log_path), exist_ok=True)
    with open(failed_log_path, "w") as f:
        f.write("# Row indices where encoding failed or sample was skipped\n")
        for i in failed_ids:
            f.write(f"{i}\n")
    if failed_ids:
        print(f"Logged {len(failed_ids)} failed/skipped rows to {failed_log_path}")
    encoded = encoded.filter(lambda x: x["failed_idx"] < 0, desc="Filter bad samples")
    encoded = encoded.remove_columns(["failed_idx"])
    print(f"Saving encoded cache: {cache_path}")
    encoded.save_to_disk(cache_path)
    print("Cache saved.")
    return encoded


def _load_or_encode_english():
    en_cache = get_english_encoded_cache_path()
    if os.path.exists(en_cache):
        print(f"Loading encoded English from cache: {en_cache}")
        ds = _datasets.load_from_disk(en_cache)
        print(f"Loaded {len(ds)} English rows from cache.")
        return ds

    raw = None
    if os.path.exists(ENGLISH_FILTERED_PATH):
        raw = _datasets.load_from_disk(ENGLISH_FILTERED_PATH)
        print("English raw dataset loaded from filtered path:", raw)
    else:
        en_files = glob.glob(EN_100HOURS_SNAPSHOT)
        if en_files:
            raw = load_dataset(
                "parquet",
                data_files={"train": en_files},
                split="train",
                verification_mode="no_checks",
            )
            print("English raw dataset loaded from local snapshot:", raw)
        else:
            print(f"Local English snapshot not found, downloading from HF: {ENGLISH_HF_REPO}")
            from huggingface_hub import snapshot_download as _snap_dl
            en_snap = _snap_dl(ENGLISH_HF_REPO, repo_type="dataset", token=True)
            en_snap_files = glob.glob(os.path.join(en_snap, "data", "*.parquet"))
            if not en_snap_files:
                en_snap_files = glob.glob(os.path.join(en_snap, "**", "*.parquet"), recursive=True)
            if en_snap_files:
                raw = load_dataset(
                    "parquet",
                    data_files={"train": en_snap_files},
                    split="train",
                    verification_mode="no_checks",
                )
                print("English raw dataset loaded from HF snapshot:", raw)
            else:
                print("Warning: could not find English parquet files after download")
                return None

    raw = raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    failed_log = os.path.join(os.path.dirname(en_cache), "english_encoding_failed_rows.log")
    return _encode_and_cache(raw, en_cache, failed_log, "Encoding English audio to TTS tokens")


def _load_or_encode_primary(lang, num_samples=None):
    lang_cache = get_encoded_cache_path(lang)
    base_path = get_lang_base_path(lang)

    if os.path.exists(lang_cache):
        print(f"Loading encoded {lang} from cache: {lang_cache}")
        ds = _datasets.load_from_disk(lang_cache)
        print(f"Loaded {len(ds)} {lang} rows from cache.")
        return ds

    snapshot_files = glob.glob(LANG_SNAPSHOT[lang])
    if snapshot_files:
        raw = load_dataset(
            "parquet",
            data_files={"train": snapshot_files},
            split="train",
            verification_mode="no_checks",
        )
        print(f"{lang.capitalize()} dataset loaded from local snapshot:", raw)
    else:
        hf_repo = LANG_HF_REPO[lang]
        print(f"Local snapshot not found, downloading from HF: {hf_repo}")
        from huggingface_hub import snapshot_download as _snap_dl
        snap = _snap_dl(hf_repo, repo_type="dataset", token=True)
        snap_files = glob.glob(os.path.join(snap, "data", "*.parquet"))
        if not snap_files:
            snap_files = glob.glob(os.path.join(snap, "**", "*.parquet"), recursive=True)
        raw = load_dataset(
            "parquet",
            data_files={"train": snap_files},
            split="train",
            verification_mode="no_checks",
        )
        print(f"{lang.capitalize()} dataset loaded from HF:", raw)
    raw = raw.cast_column("audio", Audio(sampling_rate=16_000, decode=False))

    if num_samples is not None:
        num_samples = min(num_samples, len(raw))
        raw = raw.shuffle(seed=42).select(range(num_samples))
        print(f"Using {num_samples} {lang.capitalize()} samples")

    os.makedirs(base_path, exist_ok=True)
    failed_log = os.path.join(base_path, "encoding_failed_rows.log")
    return _encode_and_cache(raw, lang_cache, failed_log, f"Encoding {lang} audio to TTS tokens")


def load_or_build_dataset(num_samples=None, lang="kannada"):
    primary = _load_or_encode_primary(lang, num_samples)
    english = _load_or_encode_english()

    if english is not None:
        n_primary = len(primary)
        n_english_target = math.ceil(n_primary * ENGLISH_RATIO)
        n_english_target = min(n_english_target, len(english))
        english_subset = english.shuffle(seed=42).select(range(n_english_target))
        dataset = concatenate_datasets([primary, english_subset]).shuffle(seed=42)
        print(f"Mixed dataset: {n_primary} {lang.capitalize()} + {n_english_target} English = {len(dataset)} total")
    else:
        print(f"English dataset not found, training on {lang.capitalize()} only.")
        dataset = primary.shuffle(seed=42)
        print(f"Dataset: {len(primary)} {lang.capitalize()} samples")

    return dataset
