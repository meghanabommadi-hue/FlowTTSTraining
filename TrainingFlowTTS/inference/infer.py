"""
Language-neutral FlowTTS inference script.

Usage:
  python3 infer.py --checkpoint /path/to/checkpoint --lang kannada
  python3 infer.py --checkpoint /path/to/checkpoint --text "custom text" --ref-wav ref.wav
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
import config
from codec_utils import build_codec
from infer_utils import run_inference

LANG_DEFAULTS = {
    "kannada": {
        "sample_wav": config.SAMPLE_WAV,
        "texts": {
            "greeting": "ನಮಸ್ಕಾರ! ನಾನು ಬಜಾಜ್ ಫೈನಾನ್ಸ್‌ನಿಂದ ವಾಣಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮೊಂದಿಗೆ ಮಾತನಾಡಲು ಇದು ಸರಿಯಾದ ಸಮಯವೇ?",
            "intro":    "ಹೇ, ಹೇಗಿದ್ದೀರಾ? ಒಂದು ನಿಮಿಷ ಮಾತಾಡೋಕೆ ಆಗುತ್ತಾ?",
            "loan":     "ನಿಮ್ಮ loan application ಇನ್ನೂ pending ಲಿದೆ. ಯಾವಾಗ ಬ್ಯಾಂಕಿಗೆ ಬರೋಕೆ ಆಗುತ್ತೆ?",
            "mixed":    "ನಮಸ್ಕಾರ! ನಾನು big basket ನಿಂದ vaani ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ.",
        },
    },
    "telugu": {
        "sample_wav": config.SAMPLE_WAV,
        "texts": {
            "greeting": "నమస్కారం! నేను బజాజ్ ఫైనాన్స్ నుండి మాట్లాడుతున్నాను.",
            "intro":    "హాయ్! మీకు ఒక నిమిషం మాట్లాడటానికి సమయం ఉందా?",
        },
    },
}

DEFAULT_CHECKPOINT = os.path.join(
    config.BASE_DATA_PA, "outputs-second-round", "checkpoint-32540"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint directory")
    parser.add_argument("--lang", default="kannada", choices=list(LANG_DEFAULTS))
    parser.add_argument("--text", default=None,
                        help="Single text to synthesize (runs all default texts if omitted)")
    parser.add_argument("--ref-wav", default=None, help="Reference WAV for voice cloning")
    parser.add_argument("--out", default="output.wav", help="Output path (used with --text)")
    args = parser.parse_args()

    import torch
    from unsloth import FastModel

    if not os.path.isdir(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model, tokenizer = FastModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        full_finetuning=True,
        load_in_4bit=False,
        torch_dtype="float32",
    )

    tts_codec = build_codec()
    lang_cfg = LANG_DEFAULTS[args.lang]
    ref_wav = args.ref_wav or lang_cfg["sample_wav"]

    if args.text:
        texts = {"out": args.text}
        output_dir = os.path.dirname(args.out) or "."
    else:
        texts = lang_cfg["texts"]
        output_dir = config.OUTPUT_AUDIO_DIR

    run_inference(model, tokenizer, tts_codec, texts, ref_wav=ref_wav, output_dir=output_dir)


if __name__ == "__main__":
    main()
