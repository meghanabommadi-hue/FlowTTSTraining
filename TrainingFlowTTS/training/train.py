"""
Language-neutral FlowTTS training script.

Usage:
  python3 train.py --lang kannada --model Shubhangi7/mira-english-1-epoch
  python3 train.py --lang telugu  --model YatharthS/MiraTTS --epochs 3
"""
import os
import sys
import argparse
import torch
import wandb

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
from transformers.trainer_utils import get_last_checkpoint

LANG_DEFAULTS = {
    "kannada": {
        "model":      "Shubhangi7/mira-english-1-epoch",
        "hub_id":     "MeghanaKap/FlowTTSKannada",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "kannada": "ನಮಸ್ಕಾರ! ನಾನು ಬಜಾಜ್ ಫೈನಾನ್ಸ್‌ನಿಂದ ವಾಣಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
            "mixed":   "ನಮಸ್ಕಾರ! ನಾನು big basket ನಿಂದ vaani ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
        },
    },
    "telugu": {
        "model":      "YatharthS/MiraTTS",
        "hub_id":     "MeghanaKap/FlowTTSTelugu",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "telugu": "నమస్కారం! నేను బజాజ్ ఫైనాన్స్ నుండి మాట్లాడుతున్నాను.",
        },
    },
    "malayalam": {
        "model":      "Shubhangi7/mira-english-1-epoch",
        "hub_id":     "MeghanaKap/FlowTTSMalayalam",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "malayalam": "നമസ്കാരം! ഞാൻ ബജാജ് ഫിനാൻസിൽ നിന്ന് സംസാരിക്കുന്നു.",
            "mixed":     "നമസ്കാരം! ഞാൻ big basket-ൽ നിന്ന് vaani ആണ് സംസാരിക്കുന്നത്.",
        },
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", default="kannada", choices=list(LANG_DEFAULTS),
                        help="Language to train (default: kannada; options: kannada, telugu, malayalam)")
    parser.add_argument("--model", default=None, help="Override base model HF id or local path")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--num_samples", type=int, default=None,
                        help="Limit number of primary-language samples (default: all)")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--hub_model_id", default=None, help="Override HF hub model id for pushing")
    args = parser.parse_args()

    lang_cfg = LANG_DEFAULTS[args.lang]
    model_id      = args.model or lang_cfg["model"]
    hub_model_id  = args.hub_model_id or lang_cfg["hub_id"]
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        config.get_lang_base_path(args.lang), f"outputs-{args.lang}-latest"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.setup(os.path.join(checkpoint_dir, "train.log"))

    # model
    model_path = snapshot_download(model_id, repo_type="model", token=config.HF_TOKEN)
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        full_finetuning=True,
        load_in_4bit=False,
        torch_dtype="float32",
    )

    # dataset
    dataset = load_or_build_dataset(num_samples=args.num_samples, lang=args.lang)

    # training
    wandb.init(project="huggingface", name=f"flowtts-{args.lang}-{os.path.basename(checkpoint_dir)}")
    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    print("Last checkpoint:", last_checkpoint)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=True,
        args=SFTConfig(
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            warmup_steps=500,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            fp16=False,
            bf16=False,
            logging_steps=100,
            optim="adamw_8bit",
            weight_decay=0.001,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=checkpoint_dir,
            save_strategy="steps",
            save_steps=5000,
            report_to="wandb",
            save_total_limit=5,
            dataloader_num_workers=4,
            push_to_hub=True,
            hub_model_id=hub_model_id,
            hub_strategy="checkpoint",
        ),
    )

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU = {gpu.name}, Memory = {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")

    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # post-training inference
    tts_codec = build_codec()
    run_inference(
        model, tokenizer, tts_codec,
        texts=lang_cfg["infer_texts"],
        ref_wav=lang_cfg["sample_wav"],
        output_dir=config.OUTPUT_AUDIO_DIR,
    )


if __name__ == "__main__":
    main()
