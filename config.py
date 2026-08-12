"""Project configuration for the local-first Damco call-intelligence demo.

Only the LLM layer is remote (Groq). Audio, diarization, speaker embeddings,
transcripts, chunks, and vector indexes are stored and processed locally.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

# Suppress only the known pyannote/TorchCodec import warning. Runtime audio I/O in
# this project reads already-normalized PCM WAV files directly into tensors and passes
# waveform dictionaries to pyannote, so TorchCodec filepath decoding is not required.
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*torchcodec is not installed correctly.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"(?s).*Could not load libtorchcodec.*",
    category=UserWarning,
)

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=False)


def _path_env(name: str, default: Path) -> Path:
    value = str(os.environ.get(name, "")).strip()
    path = Path(value) if value else default
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


DATA_DIR = _path_env("AUDIOBOT_DATA_DIR", PROJECT_ROOT / "data")
AGENTS_DIR = _path_env("AUDIOBOT_AGENTS_DIR", DATA_DIR / "agents")
CALLS_DIR = _path_env("CALLS_DIR", DATA_DIR / "calls")
MODELS_DIR = _path_env("AUDIOBOT_MODELS_DIR", PROJECT_ROOT / "models")
UPLOADS_DIR = _path_env("AUDIOBOT_UPLOADS_DIR", PROJECT_ROOT / "uploads")
SAMPLES_DIR = _path_env("AUDIOBOT_SAMPLES_DIR", PROJECT_ROOT / "samples")

# Keep modules that read environment variables directly on the same absolute paths.
os.environ["AUDIOBOT_DATA_DIR"] = str(DATA_DIR)
os.environ["AUDIOBOT_AGENTS_DIR"] = str(AGENTS_DIR)
os.environ["CALLS_DIR"] = str(CALLS_DIR)
os.environ["AUDIOBOT_MODELS_DIR"] = str(MODELS_DIR)
os.environ["AUDIOBOT_UPLOADS_DIR"] = str(UPLOADS_DIR)
os.environ["AUDIOBOT_SAMPLES_DIR"] = str(SAMPLES_DIR)

# Store Hugging Face caches under models/ by default.
os.environ.setdefault("HF_HOME", str(MODELS_DIR / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(MODELS_DIR / "huggingface" / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "0")

LOCAL_MODELS_ONLY = str(os.environ.get("LOCAL_MODELS_ONLY", "0")).strip().lower() in {
    "1", "true", "yes", "on"
}
if LOCAL_MODELS_ONLY:
    # huggingface_hub reads this flag when resolving cached repositories.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def ensure_directories() -> None:
    for path in (
        DATA_DIR,
        AGENTS_DIR,
        CALLS_DIR,
        MODELS_DIR,
        MODELS_DIR / "whisper",
        MODELS_DIR / "pyannote",
        MODELS_DIR / "sentence_transformers",
        UPLOADS_DIR,
        UPLOADS_DIR / "agents",
        UPLOADS_DIR / "calls",
        SAMPLES_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


ensure_directories()


def env_bool(name: str, default: bool = False) -> bool:
    if name not in os.environ:
        return default
    return str(os.environ[name]).strip().lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = str(os.environ.get(name, "")).strip()
    return float(value) if value else default


def env_int(name: str, default: int) -> int:
    value = str(os.environ.get(name, "")).strip()
    return int(value) if value else default


def is_env_set(name: str) -> bool:
    return bool(str(os.environ.get(name, "")).strip())


def hf_token() -> str | None:
    for name in ("HF_TOKEN", "HUGGINGFACE_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = str(os.environ.get(name, "")).strip()
        if value:
            return value
    return None


def masked_status() -> Dict[str, Any]:
    """Return configuration status without exposing secret values."""
    return {
        "project_root": str(PROJECT_ROOT),
        "env_file_exists": ENV_PATH.exists(),
        "data_dir": str(DATA_DIR),
        "agents_dir": str(AGENTS_DIR),
        "calls_dir": str(CALLS_DIR),
        "models_dir": str(MODELS_DIR),
        "uploads_dir": str(UPLOADS_DIR),
        "local_models_only": LOCAL_MODELS_ONLY,
        "ffmpeg_path_configured": is_env_set("FFMPEG_PATH"),
        "hf_token_configured": hf_token() is not None,
        "groq_api_key_configured": is_env_set("GROQ_API_KEY"),
        "whisper_model": os.environ.get("WHISPER_MODEL", "large-v3"),
        "whisper_device": os.environ.get("WHISPER_DEVICE", "auto"),
        "whisper_word_timestamps": env_bool("WHISPER_WORD_TIMESTAMPS", True),
        "speaker_match_threshold": env_float("SPEAKER_MATCH_THRESHOLD", 0.50),
        "speaker_match_margin": env_float("SPEAKER_MATCH_MARGIN", 0.12),
        "pyannote_model": os.environ.get(
            "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
        ),
        "pyannote_device": os.environ.get("PYANNOTE_DEVICE", "auto"),
        "speaker_embedding_model": os.environ.get(
            "SPEAKER_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
        ),
        "text_embedding_model": os.environ.get(
            "TEXT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        ),
        "groq_model": os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b"),
    }


def runtime_paths() -> Dict[str, Path]:
    return {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "agents_dir": AGENTS_DIR,
        "calls_dir": CALLS_DIR,
        "models_dir": MODELS_DIR,
        "uploads_dir": UPLOADS_DIR,
        "samples_dir": SAMPLES_DIR,
    }
