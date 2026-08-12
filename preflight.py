"""Runtime checks that do not expose secrets."""
from __future__ import annotations

import platform
import sys
from importlib import metadata
from typing import Any, Dict

import config
from agent_enrollment import list_agents, list_incomplete_agent_folders
from audio_function import list_calls
from audio_utils import resolve_ffmpeg

PACKAGE_NAMES = [
    "torch",
    "torchaudio",
    "torchcodec",
    "faster-whisper",
    "ctranslate2",
    "pyannote.audio",
    "sentence-transformers",
    "faiss-cpu",
    "groq",
    "streamlit",
    "python-dotenv",
    "imageio-ffmpeg",
]


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def get_preflight_status() -> Dict[str, Any]:
    packages = {name: _version(name) for name in PACKAGE_NAMES}
    cuda = {"torch_available": False, "cuda_available": False, "device_name": None}
    try:
        import torch
        cuda["torch_available"] = True
        cuda["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            cuda["device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    ctranslate2_status = {"available": False, "cuda_device_count": 0, "cuda_available": False}
    try:
        import ctranslate2
        ctranslate2_status["available"] = True
        ctranslate2_status["cuda_device_count"] = int(ctranslate2.get_cuda_device_count())
        ctranslate2_status["cuda_available"] = ctranslate2_status["cuda_device_count"] > 0
    except Exception:
        pass

    try:
        ffmpeg_path = resolve_ffmpeg()
        ffmpeg_error = None
    except Exception as exc:
        ffmpeg_path = None
        ffmpeg_error = str(exc)

    agents = list_agents()
    incomplete = list_incomplete_agent_folders()
    calls = list_calls()
    return {
        "python": sys.version.split()[0],
        "python_supported": sys.version_info >= (3, 10),
        "platform": platform.platform(),
        "ffmpeg": ffmpeg_path,
        "ffmpeg_available": ffmpeg_path is not None,
        "ffmpeg_error": ffmpeg_error,
        "packages": packages,
        "cuda": cuda,
        "ctranslate2": ctranslate2_status,
        "config": config.masked_status(),
        "agents": {
            "valid_count": len(agents),
            "incomplete_count": len(incomplete),
            "incomplete": incomplete,
        },
        "calls": {
            "total": len(calls),
            "completed": sum(x["status"] == "completed" for x in calls),
            "pending_confirmation": sum(x["status"] == "needs_speaker_confirmation" for x in calls),
            "failed": sum(x["status"] == "failed" for x in calls),
        },
    }
