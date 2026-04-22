import io
import numpy as np
import soundfile as sf
import librosa
import torch

from ncodec.codec import TTSCodec
from ncodec.encoder.model import audio_volume_normalize
from config import MIN_AUDIO_SAMPLES


def build_codec() -> TTSCodec:
    tts_codec = TTSCodec()

    @torch.inference_mode()
    def encode(audio, encode_semantic=True, duration=8):
        self = tts_codec.audio_encoder
        audio = audio_volume_normalize(audio)
        ref_clip = self.get_ref_clip(audio)
        wav_ref = torch.from_numpy(ref_clip).unsqueeze(0).float()
        mel = self.mel_transformer(wav_ref).squeeze(1)
        new_arr = mel.transpose(1, 2).cpu().numpy()
        global_tokens = self.s_encoder.run(["global_tokens"], {"mel_spectrogram": new_arr})
        context_tokens = "".join(f"<|context_token_{i}|>" for i in global_tokens[0].squeeze())
        if encode_semantic:
            feat = self.extract_wav2vec2_features(audio)
            speech_tokens = self.q_encoder.run(
                ["semantic_tokens"], {"features": feat.cpu().detach().numpy()}
            )
            speech_tokens = "".join(f"<|speech_token_{i}|>" for i in speech_tokens[0][0])
            return speech_tokens, context_tokens
        return context_tokens

    tts_codec.audio_encoder.encode = encode
    tts_codec.audio_encoder.feature_extractor.config.output_hidden_states = True
    return tts_codec


def process_wavs(example, idx, tts_codec):
    text = example["text"]
    audio_bytes = example["audio"]["bytes"]
    if not audio_bytes or len(audio_bytes) < 100:
        return {"text": "", "failed_idx": idx}
    try:
        with io.BytesIO(audio_bytes) as f:
            audio_array, sr = sf.read(f, dtype="float32")
    except Exception:
        return {"text": "", "failed_idx": idx}
    if audio_array is None or audio_array.size == 0:
        return {"text": "", "failed_idx": idx}
    if audio_array.ndim == 2:
        audio_array = audio_array.mean(axis=1)
    if sr != 16000:
        audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=16000)
    if len(audio_array) < MIN_AUDIO_SAMPLES:
        return {"text": "", "failed_idx": idx}
    try:
        semantic_tokens, global_tokens = tts_codec.audio_encoder.encode(audio_array, True, duration=25.0)
    except Exception:
        return {"text": "", "failed_idx": idx}
    prompt = (
        "<|task_tts|><|start_text|>"
        f"{text}"
        "<|end_text|>"
        "<|context_audio_start|>"
        f"{global_tokens}"
        "<|context_audio_end|>"
        "<|prompt_speech_start|>"
        f"{semantic_tokens}"
        "<|end_semantic_token|><|im_end|>"
    )
    return {"text": prompt, "failed_idx": -1}
