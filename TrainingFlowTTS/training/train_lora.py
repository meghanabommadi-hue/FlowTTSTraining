"""
LoRA fine-tuning script for multilingual TTS adapters.

Trains a language-specific LoRA adapter on top of the base model.
Each language gets its own adapter — all can be served together on one vLLM instance
using dynamic LoRA loading (--enable-lora).

Usage:
  python3 train_lora.py --lang malayalam
  python3 train_lora.py --lang kannada --epochs 3
  python3 train_lora.py --lang telugu  --lora_r 32 --num_samples 50000

After training, push the adapter to HF:
  The adapter is pushed automatically to MeghanaKap/FlowTTS-<Lang>-LoRA

Serving all adapters on one vLLM:
  vllm serve <base-model> \\
    --enable-lora \\
    --lora-modules kannada=MeghanaKap/FlowTTS-Kannada-LoRA \\
                   malayalam=MeghanaKap/FlowTTS-Malayalam-LoRA \\
                   telugu=MeghanaKap/FlowTTS-Telugu-LoRA \\
    --max-lora-rank 64
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

# ── per-language defaults ─────────────────────────────────────────────────────

HOME = os.path.expanduser("~")

LANG_DEFAULTS = {
    "kannada": {
        # Start from Telugu adapter — closest Dravidian language we have locally
        "model":      os.path.join(HOME, "models", "Shubhangi7-mira_hindi_second_round"),
        "hub_id":     "MeghanaKap/FlowTTS-Kannada-LoRA",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "kannada": "ನಮಸ್ಕಾರ! ನಾನು ಬಜಾಜ್ ಫೈನಾನ್ಸ್‌ನಿಂದ ವಾಣಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
            "mixed":   "ನಮಸ್ಕಾರ! ನಾನು big basket ನಿಂದ vaani ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
        },
    },
    "telugu": {
        "model":      os.path.join(HOME, "models", "Shubhangi7-mira_hindi_second_round"),
        "hub_id":     "MeghanaKap/FlowTTS-Telugu-LoRA",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "telugu": "నమస్కారం! నేను బజాజ్ ఫైనాన్స్ నుండి మాట్లాడుతున్నాను.",
        },
    },
    "malayalam": {
        # Start from Telugu adapter — closest Dravidian language we have locally
        "model":      os.path.join(HOME, "models", "Shubhangi7-mira_hindi_second_round"),
        "hub_id":     "MeghanaKap/FlowTTS-Malayalam-LoRA",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "malayalam": "നമസ്കാരം! ഞാൻ ബജാജ് ഫിനാൻസിൽ നിന്ന് സംസാരിക്കുന്നു.",
            "mixed":     "നമസ്കാരം! ഞാൻ big basket-ൽ നിന്ന് vaani ആണ് സംസാരിക്കുന്നത്.",
        },
    },
    "hindi": {
        "model":      os.path.join(HOME, "models", "Shubhangi7-mira_hindi_second_round"),
        "hub_id":     "MeghanaKap/FlowTTS-Hindi-LoRA",
        "sample_wav": config.SAMPLE_WAV,
        "infer_texts": {
            "hindi": "नमस्ते! मैं बजाज फाइनेंस से वाणी बोल रही हूँ।",
        },
    },
}

# ── LoRA target modules for FlowTTS / Qwen3-TTS architecture ─────────────────
# Targets attention projections + feed-forward in the talker transformer.
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for multilingual TTS adapters")
    parser.add_argument("--lang", default="malayalam", choices=list(LANG_DEFAULTS),
                        help="Language to train")
    parser.add_argument("--model",          default=None, help="Override base model HF id or local path")
    parser.add_argument("--checkpoint_dir", default=None, help="Override checkpoint output directory")
    parser.add_argument("--hub_model_id",   default=None, help="Override HF hub id for the adapter")
    parser.add_argument("--num_samples",    type=int, default=None,
                        help="Limit primary-language samples (default: all)")
    parser.add_argument("--epochs",         type=int,   default=2)
    parser.add_argument("--lora_r",         type=int,   default=16,
                        help="LoRA rank — higher = more capacity, more VRAM (default: 16)")
    parser.add_argument("--lora_alpha",     type=int,   default=32,
                        help="LoRA alpha scaling (default: 32 = 2x rank)")
    parser.add_argument("--lora_dropout",   type=float, default=0.05)
    parser.add_argument("--lr",             type=float, default=2e-4)
    parser.add_argument("--batch_size",     type=int,   default=4)
    parser.add_argument("--grad_accum",     type=int,   default=4)
    parser.add_argument("--save_steps",     type=int,   default=500)
    args = parser.parse_args()

    lang_cfg      = LANG_DEFAULTS[args.lang]
    model_id      = args.model      or lang_cfg["model"]
    hub_model_id  = args.hub_model_id or lang_cfg["hub_id"]
    checkpoint_dir = args.checkpoint_dir or os.path.join(
        config.get_lang_base_path(args.lang), f"lora-{args.lang}-r{args.lora_r}"
    )

    os.makedirs(checkpoint_dir, exist_ok=True)
    logger.setup(os.path.join(checkpoint_dir, "train_lora.log"))

    print(f"Language     : {args.lang}")
    print(f"Base model   : {model_id}")
    print(f"LoRA rank    : {args.lora_r}  alpha={args.lora_alpha}")
    print(f"Output dir   : {checkpoint_dir}")
    print(f"Hub adapter  : {hub_model_id}")

    # ── load base model ───────────────────────────────────────────────────────
    # use local path directly if it exists, otherwise download from HF
    model_path = model_id if os.path.isdir(model_id) else snapshot_download(
        model_id, repo_type="model", token=config.HF_TOKEN
    )
    model, tokenizer = FastModel.from_pretrained(
        model_name=model_path,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        full_finetuning=False,   # LoRA only — base weights stay frozen
        load_in_4bit=False,
        torch_dtype="float32",
    )

    # ── attach LoRA adapter ───────────────────────────────────────────────────
    model = FastModel.get_peft_model(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params : {trainable:,} / {total:,}  ({100*trainable/total:.2f}%)")

    # ── dataset ───────────────────────────────────────────────────────────────
    dataset = load_or_build_dataset(num_samples=args.num_samples, lang=args.lang)

    # ── train ─────────────────────────────────────────────────────────────────
    wandb.init(
        project="flowtts-lora",
        name=f"{args.lang}-r{args.lora_r}-{os.path.basename(checkpoint_dir)}",
    )
    last_checkpoint = get_last_checkpoint(checkpoint_dir)
    print("Resuming from checkpoint:", last_checkpoint or "none")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=config.MAX_SEQ_LENGTH,
        packing=True,
        args=SFTConfig(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            warmup_steps=100,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=False,
            bf16=False,
            logging_steps=50,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=3407,
            output_dir=checkpoint_dir,
            save_strategy="steps",
            save_steps=args.save_steps,
            save_total_limit=3,
            report_to="wandb",
            dataloader_num_workers=4,
            # push only the adapter, not the full model
            push_to_hub=True,
            hub_model_id=hub_model_id,
            hub_strategy="checkpoint",
        ),
    )

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {gpu.name}  |  Reserved: {torch.cuda.max_memory_reserved()/1024**3:.1f} GB")

    if last_checkpoint:
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    # ── save adapter only (not merged weights) ────────────────────────────────
    adapter_path = os.path.join(checkpoint_dir, "adapter_final")
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"Adapter saved to: {adapter_path}")

    # push final adapter to HF
    model.push_to_hub(hub_model_id, token=config.HF_TOKEN)
    tokenizer.push_to_hub(hub_model_id, token=config.HF_TOKEN)
    print(f"Adapter pushed to HF: {hub_model_id}")

    # ── post-training inference check ─────────────────────────────────────────
    tts_codec = build_codec()
    run_inference(
        model, tokenizer, tts_codec,
        texts=lang_cfg["infer_texts"],
        ref_wav=lang_cfg["sample_wav"],
        output_dir=config.OUTPUT_AUDIO_DIR,
    )


if __name__ == "__main__":
    main()
