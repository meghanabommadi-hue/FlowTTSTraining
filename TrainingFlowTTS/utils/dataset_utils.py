import os
import math
import datasets as _datasets
from datasets import load_dataset, Audio, concatenate_datasets
from functools import partial

from config import (
    KANNADA_SNAPSHOT, ENGLISH_FILTERED_PATH, ENCODED_CACHE_PATH,
    ENGLISH_RATIO, BASE_DATA_PA,
)
from codec_utils import build_codec, process_wavs


def load_or_build_dataset(num_samples=None):
    if os.path.exists(ENCODED_CACHE_PATH):
        print(f"Loading encoded dataset from cache: {ENCODED_CACHE_PATH}")
        dataset = _datasets.load_from_disk(ENCODED_CACHE_PATH)
        print(f"Loaded {len(dataset)} rows from cache, skipping encoding.")
        return dataset

    kannada_dataset = load_dataset(
        "parquet",
        data_files={"train": KANNADA_SNAPSHOT},
        split="train",
        verification_mode="no_checks",
    )
    kannada_dataset = kannada_dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    print("Kannada dataset loaded:", kannada_dataset)

    if num_samples is not None:
        num_samples = min(num_samples, len(kannada_dataset))
        kannada_dataset = kannada_dataset.shuffle(seed=42).select(range(num_samples))
        print(f"Using {num_samples} Kannada samples")

    n_kannada = len(kannada_dataset)

    if os.path.exists(ENGLISH_FILTERED_PATH):
        english_dataset = _datasets.load_from_disk(ENGLISH_FILTERED_PATH)
        print("English dataset loaded:", english_dataset)
        n_english_target = math.ceil(n_kannada * ENGLISH_RATIO)
        n_english_target = min(n_english_target, len(english_dataset))
        english_subset = english_dataset.shuffle(seed=42).select(range(n_english_target))
        dataset = concatenate_datasets([kannada_dataset, english_subset]).shuffle(seed=42)
        print(f"Mixed dataset: {n_kannada} Kannada + {n_english_target} English = {len(dataset)} total")
    else:
        print("English dataset not found, training on Kannada only.")
        dataset = kannada_dataset.shuffle(seed=42)
        print(f"Dataset: {n_kannada} Kannada samples")

    tts_codec = build_codec()
    dataset = dataset.map(
        partial(process_wavs, tts_codec=tts_codec),
        remove_columns=["audio"],
        with_indices=True,
        desc="Encoding audio to TTS tokens",
    )

    failed_ids = [int(i) for i in dataset["failed_idx"] if int(i) >= 0]
    failed_log_path = os.path.join(BASE_DATA_PA, "encoding_failed_rows.log")
    with open(failed_log_path, "w") as f:
        f.write("# Row indices where encoding failed or sample was skipped\n")
        for i in failed_ids:
            f.write(f"{i}\n")
    if failed_ids:
        print(f"Logged {len(failed_ids)} failed/skipped rows to {failed_log_path}")

    dataset = dataset.filter(lambda x: x["failed_idx"] < 0, desc="Filter bad samples")
    dataset = dataset.remove_columns(["failed_idx"])
    print(f"Saving encoded dataset to cache: {ENCODED_CACHE_PATH}")
    dataset.save_to_disk(ENCODED_CACHE_PATH)
    print("Cache saved.")
    return dataset
