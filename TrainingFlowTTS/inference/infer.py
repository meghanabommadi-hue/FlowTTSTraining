"""
FlowTTS inference script — supports Kannada, Telugu, Malayalam, Tamil, Hindi.

Usage:
  # Run all default sentences for a language
  python3 infer.py --lang kn --checkpoint /path/to/checkpoint

  # Single custom text
  python3 infer.py --lang hi --checkpoint /path/to/checkpoint --text "नमस्ते!"

  # Custom reference audio
  python3 infer.py --lang ta --checkpoint /path/to/checkpoint --ref-wav audio/ref/my_voice.wav

Language codes: kn, te, ml, ta, hi
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "utils"))
import config
from codec_utils import build_codec
from infer_utils import run_inference

# ── test sentences per language ───────────────────────────────────────────────
LANG_TEXTS = {
    "kn": {
        "name": "Kannada",
        "self":    {
            "greeting":  "ನಮಸ್ಕಾರ! ನಾನು ಬಜಾಜ್ ಫೈನಾನ್ಸ್‌ನಿಂದ ವಾಣಿ ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮೊಂದಿಗೆ ಮಾತನಾಡಲು ಇದು ಸರಿಯಾದ ಸಮಯವೇ?",
            "intro":     "ಹೇ, ಹೇಗಿದ್ದೀರಾ? ಒಂದು ನಿಮಿಷ ಮಾತಾಡೋಕೆ ಆಗುತ್ತಾ?",
            "reminder":  "ನಾಳೆ ಬೆಳಿಗ್ಗೆ ಹತ್ತು ಗಂಟೆಗೆ ಡಾಕ್ಟರ್ ಅಪಾಯಿಂಟ್ಮೆಂಟ್ ಇದೆ, ಮರೀಬೇಡಿ.",
        },
        "mixed":   {
            "mixed_1":   "ನಮಸ್ಕಾರ! ನಾನು Big Basket ನಿಂದ vaani ಮಾತನಾಡುತ್ತಿದ್ದೇನೆ. ನಿಮ್ಮ order ready ಆಗಿದೆ.",
            "mixed_2":   "ನಿಮ್ಮ EMI due date ಮುಂದಿನ ವಾರ ಇದೆ. Payment ಮಾಡೋಕೆ ready ಇದ್ದೀರಾ?",
            "mixed_3":   "ನಿಮ್ಮ loan application approve ಆಗಿದೆ. Bank ಗೆ ಬಂದು documents submit ಮಾಡಿ.",
        },
        "english": {
            "english_1": "Hello! I'm calling from Bajaj Finance. Do you have a minute to chat?",
            "english_2": "Your order has been delivered. Please rate your experience.",
        },
    },
    "te": {
        "name": "Telugu",
        "self":    {
            "greeting":  "నమస్కారం! నేను బజాజ్ ఫైనాన్స్ నుండి మాట్లాడుతున్నాను. మీకు మాట్లాడడానికి సమయం ఉందా?",
            "intro":     "హాయ్! మీరు ఎలా ఉన్నారు? ఒక్క నిమిషం మాట్లాడగలరా?",
            "reminder":  "రేపు పదిగంటలకు డాక్టర్ అపాయింట్‌మెంట్ ఉంది, మర్చిపోకండి.",
        },
        "mixed":   {
            "mixed_1":   "నమస్కారం! నేను Big Basket నుండి మాట్లాడుతున్నాను. మీ order ready అయింది.",
            "mixed_2":   "మీ EMI due date వచ్చే వారం ఉంది. Payment చేయడానికి ready గా ఉన్నారా?",
        },
        "english": {
            "english_1": "Hello! I'm calling from Bajaj Finance. Do you have a minute to chat?",
            "english_2": "Your loan application has been approved. Please visit the branch.",
        },
    },
    "ml": {
        "name": "Malayalam",
        "self":    {
            "greeting":  "നമസ്കാരം! ഞാൻ ബജാജ് ഫിനാൻസിൽ നിന്ന് വിളിക്കുകയാണ്. സംസാരിക്കാൻ സമയമുണ്ടോ?",
            "intro":     "ഹേ, സുഖമാണോ? ഒരു മിനിറ്റ് സംസാരിക്കാൻ പറ്റുമോ?",
            "reminder":  "നാളെ രാവിലെ പത്തു മണിക്ക് ഡോക്ടർ അപ്പോയ്ന്റ്മെന്റ് ഉണ്ട്, മറക്കരുത്.",
        },
        "mixed":   {
            "mixed_1":   "നമസ്കാരം! Big Basket-ൽ നിന്ന് വിളിക്കുകയാണ്. നിങ്ങളുടെ order ready ആയി.",
            "mixed_2":   "നിങ്ങളുടെ EMI due date അടുത്ത ആഴ്ചയാണ്. Payment ചെയ്യാൻ ready ആണോ?",
        },
        "english": {
            "english_1": "Hello! I'm calling from Bajaj Finance. Do you have a minute to chat?",
            "english_2": "Your order has been delivered. Please rate your experience.",
        },
    },
    "ta": {
        "name": "Tamil",
        "self":    {
            "greeting":  "வணக்கம்! நான் பஜாஜ் ஃபைனான்ஸிலிருந்து பேசுகிறேன். கொஞ்சம் பேசலாமா?",
            "intro":     "ஹேய்! எப்படி இருக்கீங்க? ஒரு நிமிஷம் பேசலாமா?",
            "reminder":  "நாளை காலை பத்து மணிக்கு டாக்டர் அப்பாயின்ட்மென்ட் இருக்கு, மறக்காதீங்க.",
        },
        "mixed":   {
            "mixed_1":   "வணக்கம்! Big Basket-லிருந்து பேசுகிறேன். உங்கள் order ready ஆகிவிட்டது.",
            "mixed_2":   "உங்கள் EMI due date அடுத்த வாரம் இருக்கு. Payment பண்ண ready-யா?",
        },
        "english": {
            "english_1": "Hello! I'm calling from Bajaj Finance. Do you have a minute to chat?",
            "english_2": "Your loan application has been approved. Please visit the branch.",
        },
    },
    "hi": {
        "name": "Hindi",
        "self":    {
            "greeting":  "नमस्ते! मैं बजाज फाइनेंस से बात कर रही हूँ। क्या आप अभी बात कर सकते हैं?",
            "intro":     "हेलो! कैसे हैं आप? एक मिनट बात कर सकते हैं?",
            "reminder":  "कल सुबह दस बजे डॉक्टर का अपॉइंटमेंट है, याद रखें।",
        },
        "mixed":   {
            "mixed_1":   "नमस्ते! मैं Big Basket से बोल रही हूँ। आपका order ready है।",
            "mixed_2":   "आपकी EMI due date अगले हफ्ते है। Payment के लिए ready हैं?",
            "mixed_3":   "आपका loan application approve हो गया है। Branch में आकर documents submit करें।",
        },
        "english": {
            "english_1": "Hello! I'm calling from Bajaj Finance. Do you have a minute to chat?",
            "english_2": "Your order has been delivered. Please rate your experience.",
        },
    },
}

DEFAULT_CHECKPOINT = os.path.join(
    config.BASE_DATA_PA, "outputs-third-round", "checkpoint-32540"
)


def main():
    parser = argparse.ArgumentParser(description="FlowTTS inference")
    parser.add_argument("--lang", required=True, choices=list(LANG_TEXTS),
                        help="Language code: kn, te, ml, ta, hi")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Path to model checkpoint directory")
    parser.add_argument("--mode", default="all", choices=["all", "self", "mixed", "english"],
                        help="Which sentence set to run (default: all)")
    parser.add_argument("--text", default=None,
                        help="Single custom text to synthesize")
    parser.add_argument("--ref-wav", default=None,
                        help="Reference WAV for voice cloning (default: audio/ref/simran.wav)")
    parser.add_argument("--out", default="output.wav",
                        help="Output path when using --text")
    args = parser.parse_args()

    import torch
    from unsloth import FastModel

    if not os.path.isdir(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    print(f"Loading model from {args.checkpoint} ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=args.checkpoint,
        max_seq_length=config.MAX_SEQ_LENGTH,
        dtype=config.DTYPE,
        full_finetuning=True,
        load_in_4bit=False,
        torch_dtype="float32",
    )

    tts_codec = build_codec()
    ref_wav = args.ref_wav or config.SAMPLE_WAV
    lang_cfg = LANG_TEXTS[args.lang]
    lang_name = lang_cfg["name"]

    if args.text:
        texts = {"custom": args.text}
        output_dir = os.path.dirname(os.path.abspath(args.out))
    else:
        if args.mode == "all":
            texts = {**lang_cfg["self"], **lang_cfg["mixed"], **lang_cfg["english"]}
        else:
            texts = lang_cfg[args.mode]
        output_dir = os.path.join(config.OUTPUT_AUDIO_DIR, args.lang, args.mode)

    print(f"\nLanguage : {lang_name} ({args.lang})")
    print(f"Mode     : {args.mode}")
    print(f"Sentences: {len(texts)}")
    print(f"Output   : {output_dir}\n")

    run_inference(model, tokenizer, tts_codec, texts, ref_wav=ref_wav, output_dir=output_dir)


if __name__ == "__main__":
    main()
