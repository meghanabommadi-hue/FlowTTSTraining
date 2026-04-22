import os
import numpy as np
import torch
import librosa
import soundfile as sf
from datetime import datetime

from config import DTYPE, OUTPUT_AUDIO_DIR


def run_inference(model, tokenizer, tts_codec, texts: dict, ref_wav: str, output_dir: str = None):
    if output_dir is None:
        output_dir = OUTPUT_AUDIO_DIR
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model.eval()

    audio_ref, _ = librosa.load(ref_wav, sr=16000)
    context_tokens = tts_codec.encode(audio_ref)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for name, text in texts.items():
        print(f"\n[{name}] Synthesizing: {text!r}")
        formatted_prompt = tts_codec.format_prompt(text, context_tokens, None)
        model_inputs = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        model_inputs = {
            k: v.to(torch.float32) if v.dtype in (torch.float16, torch.bfloat16) else v
            for k, v in model_inputs.items()
        }
        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.8,
                top_k=50,
                top_p=1.0,
                repetition_penalty=1.2,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=False,
            )
        pred_text = tokenizer.batch_decode(
            generated_ids[:, model_inputs["input_ids"].shape[1]:],
            skip_special_tokens=False,
        )[0]
        audio = tts_codec.decode(pred_text, context_tokens)
        audio = np.asarray(audio).squeeze().astype(np.float32)
        out_path = os.path.join(output_dir, f"flowtts_output_{name}_{ts}.wav")
        sf.write(out_path, audio, samplerate=48000, subtype="PCM_16")
        print(f"Saved: {out_path}")
