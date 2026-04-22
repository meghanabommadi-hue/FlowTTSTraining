# FlowTTS — Commands Reference

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set HuggingFace token
export HF_TOKEN=your_huggingface_token
```

---

## Training

```bash
cd training/

# Kannada — full dataset, default base model
python3 train.py --lang kn

# Telugu — full dataset
python3 train.py --lang te

# Kannada — custom base model
python3 train.py --lang kn --model Shubhangi7/mira-english-1-epoch

# Kannada — limit samples
python3 train.py --lang kn --num_samples 50000

# Kannada — more epochs
python3 train.py --lang kn --epochs 4

# Kannada — custom checkpoint output directory
python3 train.py --lang kn --checkpoint_dir ~/my_checkpoints/round1

# Kannada — custom HuggingFace hub repo to push to
python3 train.py --lang kn --hub_model_id MyOrg/MyKannadaModel

# Resume from existing checkpoint (automatic — just rerun)
python3 train.py --lang kn
```

---

## Inference

```bash
cd inference/

# All sentences — Kannada
python3 infer.py --lang kn --checkpoint /path/to/checkpoint

# All sentences — Telugu
python3 infer.py --lang te --checkpoint /path/to/checkpoint

# All sentences — Malayalam
python3 infer.py --lang ml --checkpoint /path/to/checkpoint

# All sentences — Tamil
python3 infer.py --lang ta --checkpoint /path/to/checkpoint

# All sentences — Hindi
python3 infer.py --lang hi --checkpoint /path/to/checkpoint

# Only native-language sentences
python3 infer.py --lang kn --checkpoint /path/to/checkpoint --mode self

# Only code-mixed sentences
python3 infer.py --lang kn --checkpoint /path/to/checkpoint --mode mixed

# Only English sentences
python3 infer.py --lang kn --checkpoint /path/to/checkpoint --mode english

# Single custom text
python3 infer.py --lang kn --checkpoint /path/to/checkpoint --text "ನಮಸ್ಕಾರ!"

# Single custom text with custom output path
python3 infer.py --lang kn --checkpoint /path/to/checkpoint --text "ನಮಸ್ಕಾರ!" --out my_output.wav

# Custom reference audio for voice cloning
python3 infer.py --lang kn --checkpoint /path/to/checkpoint --ref-wav audio/ref/my_voice.wav
```

**Language codes:** `kn` Kannada · `te` Telugu · `ml` Malayalam · `ta` Tamil · `hi` Hindi

**Output location:** `audio/outputs/<lang>/<mode>/`

---

## Testing

```bash
cd testing/

# Quick test — 100 Kannada samples, 1 epoch
python3 test_train.py --lang kn

# Quick test — 50 Telugu samples
python3 test_train.py --lang te --num_samples 50

# Quick test — custom base model
python3 test_train.py --lang kn --model YatharthS/MiraTTS --num_samples 100
```

---

## Data Preparation

```bash
# Download English TTS data (~3.5 GB, Latin/ASCII only)
python3 download_english_data.py
# Saves to: ~/data_preparation/english_latin_only/
```

---

## Disk Maintenance

```bash
# Check disk usage
du -sh ~/.cache/huggingface/datasets/
du -sh ~/data_preparation/flowtts-kannada-training/

# Safe to delete — HF parquet builder cache (regenerates, not needed after encoding)
rm -rf ~/.cache/huggingface/datasets/parquet/

# Safe to delete — pip/uv install caches
rm -rf ~/.cache/pip/ ~/.cache/uv/

# Safe to delete — old checkpoints (keep latest only)
rm -rf ~/data_preparation/flowtts-kannada-training/outputs-second-round/checkpoint-15000
rm -rf ~/data_preparation/flowtts-kannada-training/outputs-second-round/checkpoint-25000

# DO NOT DELETE — encoded dataset cache (hours to rebuild)
# ~/data_preparation/flowtts-kannada-training/encoded_dataset_cache/
```

---

## Monitoring

```bash
# Watch training log live
tail -f ~/data_preparation/flowtts-kannada-training/outputs-third-round/flowtts_train.log

# Check GPU usage
nvidia-smi

# Check disk usage
df -h /

# Check running training process
ps aux | grep train
```

---

## Git

```bash
# Push changes
git add -A && git commit -m "your message" && git push

# Check what's changed
git status && git diff
```
