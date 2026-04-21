from pathlib import Path
import os
from dotenv import load_dotenv

# Load .env from repo root: /home/tu/circuit-tracer/.env
_REPO_ROOT = Path(__file__).resolve().parent
load_dotenv(_REPO_ROOT / ".env", override=False)

def get_env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return value if value is not None else default

def get_required_env(name: str) -> str:
    value = get_env(name, "").strip()
    if not value:
        raise ValueError(f"{name} not set in environment")
    return value

NEURONPEDIA_API_KEY = get_env("NEURONPEDIA_API_KEY", "")
HUGGINGFACE_API_KEY = get_env("HUGGINGFACE_API_KEY", "")
# Support both names for Gemini
GENAI_API_KEY = get_env("GEMINI_API_KEY", "") or get_env("GENAI_API_KEY", "")
OPENAI_API_KEY = get_env("OPENAI_API_KEY", "")