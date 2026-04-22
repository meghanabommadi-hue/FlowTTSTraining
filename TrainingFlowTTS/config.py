import os
import torch

HF_TOKEN = os.environ.get("HF_TOKEN", "")

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
os.environ["UNSLOTH_FORCE_FLOAT32"] = "1"
os.environ["UNSLOTH_FORCE_BF16"] = "0"
os.environ["UNSLOTH_DISABLE_BF16"] = "1"
os.environ["DISABLE_TQDM"] = "0"

HOME = os.path.expanduser("~")
BASE_DATA_PA = os.path.join(HOME, "data_preparation", "flowtts-kannada-training") + "/"
BASE_DATA    = os.path.join(HOME, "data_preparation") + "/"

HF_HUB_CACHE          = os.path.join(HOME, ".cache", "huggingface", "hub")
KANNADA_SNAPSHOT      = os.path.join(HF_HUB_CACHE, "datasets--MeghanaKap--kannada_dataset",
                                     "snapshots", "e57baaaab3dc07656337406ff8afae0225928390",
                                     "data", "*.parquet")
ENGLISH_FILTERED_PATH = os.path.join(BASE_DATA, "english_latin_only")
ENCODED_CACHE_PATH    = os.path.join(BASE_DATA_PA, "encoded_dataset_cache")
REPO_ROOT   = os.path.dirname(os.path.abspath(__file__))
REF_AUDIO_DIR    = os.path.join(REPO_ROOT, "audio", "ref")
OUTPUT_AUDIO_DIR = os.path.join(REPO_ROOT, "audio", "outputs")
SAMPLE_WAV       = os.path.join(REF_AUDIO_DIR, "simran.wav")

MAX_SEQ_LENGTH = 100 * 50   # ~30 seconds audio
DTYPE          = torch.float32
MIN_AUDIO_SAMPLES = 8000

ENGLISH_RATIO  = 25 / 75    # 25% English, 75% Kannada
