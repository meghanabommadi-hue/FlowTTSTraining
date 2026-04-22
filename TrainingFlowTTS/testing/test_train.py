"""
Language-neutral FlowTTS quick-test training script.
Runs a small sample to verify the pipeline works end-to-end before a full run.

Usage:
  python3 test_train.py --lang kannada --num_samples 100
  python3 test_train.py --lang telugu  --num_samples 50
"""
import os
import sys
import argparse
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
import config
import logger
from dataset_utils import load_or_build_dataset
from codec_utils import build_codec
from infer_utils import run_inference

from huggingface_hub import snapshot_download
from unsloth import FastModel
from trl import SFTTrainer, SFTConfig

LANG_DEFAULTS = {
    "kannada": {
        "model":      "YatharthS/MiraTTS",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "kannada": "ನಮಸ್ಕಾರ! ನಾನು ಬಜಾಜ್ ಫೈನಾನ್ಸ್‌ನಿಂದ ವಾಣಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
        },
    },
    "telugu": {
        "model":      "YatharthS/MiraTTS",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "telugu": "నమస్కారం! నేను బజాజ్ ఫైనాన్స్ నుండి మాట్లాడుతున్నాను.",
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="kannada", choices=list(LANG_DEFAULTS))
    parser.add_argument("--model", default=None, help="Override base model")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples for the test run (default: 100)")
    args = parser.parse_args()

    lang_cfg   = LANG_DEFAULTS[args.lang]
    model_id   = args.model or lang_cfg["model"]
    checkpoint_dir = os.path.join(config.BASE_DATA_PA, f"test-{args.num_samples}-checkpoints")

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.setup(os.path.join(checkpoint_dir, "test_run.log"))
    print(f"=== TEST RUN: {args.num_samples} samples | lang={args.lang} ===")

    model_path = snapshot_download(model_id, repo_type="model", token=config.HF_TOKEN)
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        full_finetuning=True,
        load_in_4bit=False,
        torch_dtype="float32",
    )

    # force rebuild for the small test subset (don't use full encoded cache)
    from datasets import load_dataset, Audio
    import datasets as _datasets
    from codec_utils import process_wavs
    from functools import partial

    kannada_dataset = load_dataset(
        "parquet",
        data_files={"train": config.KANNADA_SNAPSHOT},
        split="train",
        verification_mode="no_checks",
    )
    kannada_dataset = kannada_dataset.cast_column("audio", Audio(sampling_rate=16_000, decode=False))
    kannada_dataset = kannada_dataset.shuffle(seed=42).select(range(args.num_samples))

    tts_codec = build_codec()
    dataset = kannada_dataset.map(
        partial(process_wavs, tts_codec=tts_codec),
        remove_columns=["audio"],
        with_indices=True,
        desc="Encoding test samples",
    )
    dataset = dataset.filter(lambda x: x["failed_idx"] < 0)
    dataset = dataset.remove_columns(["failed_idx"])
    print(f"Test dataset: {len(dataset)} valid samples")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=True,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=2,
            warmup_steps=5,
            num_train_epochs=1,
            learning_rate=2e-4,
            fp16=False,
            bf16=False,
            logging_steps=10,
            optim="adamw_8bit",
            seed=3407,
            output_dir=checkpoint_dir,
            save_strategy="no",
            report_to="none",
        ),
    )

    print(f"GPU = {torch.cuda.get_device_properties(0).name}")
    trainer.train()
    print("Test run complete.")

    run_inference(
        model, tokenizer, tts_codec,
        texts=lang_cfg["infer_texts"],
        ref_wav=lang_cfg["sample_wav"],
        output_dir=config.OUTPUT_AUDIO_DIR,
    )


if __name__ == "__main__":
    main()
