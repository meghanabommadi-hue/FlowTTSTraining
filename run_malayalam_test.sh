#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR/TrainingFlowTTS"

# ── HF token ──────────────────────────────────────────────────────────────────
HF_TOKEN="${HF_TOKEN:-$(cat ~/.cache/huggingface/token 2>/dev/null || true)}"
if [[ -z "$HF_TOKEN" ]]; then
    echo "ERROR: HF_TOKEN not set and ~/.cache/huggingface/token not found." >&2
    exit 1
fi
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

# ── download base model ───────────────────────────────────────────────────────
echo "=== Downloading base model: Shubhangi7/mira-english-1-epoch ==="
python3 - <<'PYEOF'
import os
from huggingface_hub import snapshot_download
token = os.environ["HF_TOKEN"]
path = snapshot_download("Shubhangi7/mira-english-1-epoch", repo_type="model", token=token)
print(f"Base model ready at: {path}")
PYEOF

# ── run test_train for 100 Malayalam rows ─────────────────────────────────────
echo ""
echo "=== Running test_train: Malayalam, 100 samples ==="
cd "$REPO_DIR/testing"
python3 test_train.py --lang malayalam --num_samples 100
