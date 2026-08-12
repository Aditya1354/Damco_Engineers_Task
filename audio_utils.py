"""Local audio conversion and waveform loading helpers.

FFmpeg resolution is intentionally robust on Windows. The project can use:
1. an explicit ``FFMPEG_PATH`` from ``.env``;
2. ``ffmpeg`` available on the current process PATH;
3. WinGet's portable command alias;
4. the executable bundled by the ``imageio-ffmpeg`` Python wheel.

This avoids making Streamlit depend on whether a terminal was restarted after a
WinGet installation.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import wave
from pathlib import Path
from typing import Any, Dict


def _explicit_ffmpeg_path() -> str | None:
    value = str(os.environ.get("FFMPEG_PATH", "")).strip().strip('"')
    if not value:
        return None
    path = Path(value).expanduser()
    if path.is_dir():
        path = path / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
    if not path.is_absolute():
        # Resolve relative paths from the project folder rather than the caller's CWD.
        path = (Path(__file__).resolve().parent / path).resolve()
    if path.is_file():
        return str(path)
    raise RuntimeError(
        f"FFMPEG_PATH is configured but does not point to an FFmpeg executable: {path}"
    )


def _winget_ffmpeg_alias() -> str | None:
    if os.name != "nt":
        return None
    local_app_data = str(os.environ.get("LOCALAPPDATA", "")).strip()
    if not local_app_data:
        return None
    candidate = Path(local_app_data) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe"
    return str(candidate) if candidate.is_file() else None


def _imageio_ffmpeg_path() -> str | None:
    try:
        import imageio_ffmpeg

        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        return str(candidate) if candidate.is_file() else None
    except Exception:
        return None


def resolve_ffmpeg() -> str | None:
    """Return a usable FFmpeg executable path without exposing shell-specific assumptions."""
    explicit = _explicit_ffmpeg_path()
    if explicit:
        return explicit

    on_path = shutil.which("ffmpeg")
    if on_path:
        return on_path

    winget_alias = _winget_ffmpeg_alias()
    if winget_alias:
        return winget_alias

    bundled = _imageio_ffmpeg_path()
    if bundled:
        return bundled

    return None


def ffmpeg_available() -> bool:
    try:
        return resolve_ffmpeg() is not None
    except RuntimeError:
        return False


def require_ffmpeg() -> str:
    path = resolve_ffmpeg()
    if path:
        return path
    raise RuntimeError(
        "FFmpeg is required to normalize uploaded audio, but no executable was found. "
        "Install project requirements so imageio-ffmpeg is available, install FFmpeg with "
        "`winget install -e --id Gyan.FFmpeg`, or set FFMPEG_PATH in .env to the full "
        "path of ffmpeg.exe. If FFmpeg was installed while Streamlit was already running, "
        "restart the terminal/IDE and Streamlit process."
    )


def convert_to_mono_16k_wav(input_path: str | Path, output_path: str | Path) -> Dict[str, Any]:
    """Convert any FFmpeg-readable audio file to PCM 16-bit mono 16 kHz WAV."""
    src = Path(input_path)
    dst = Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"Audio file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = require_ffmpeg()
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-i", str(src),
        "-vn",
        "-ac", "1",
        "-ar", "16000",
        "-c:a", "pcm_s16le",
        str(dst),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0 or not dst.exists() or dst.stat().st_size == 0:
        raise RuntimeError(
            "FFmpeg audio conversion failed. "
            f"stderr: {(result.stderr or result.stdout or '').strip()}"
        )

    info = wav_info(dst)
    if info["channels"] != 1 or info["sample_rate"] != 16000:
        raise RuntimeError(f"Normalized WAV has unexpected format: {info}")
    return info


def wav_info(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        duration = frames / float(sample_rate) if sample_rate else 0.0
    return {
        "path": str(path),
        "sample_rate": int(sample_rate),
        "channels": int(channels),
        "sample_width_bytes": int(sample_width),
        "frames": int(frames),
        "duration_seconds": round(float(duration), 3),
    }


def load_waveform_dict(path: str | Path, target_sample_rate: int = 16000) -> Dict[str, Any]:
    """Load a PCM WAV into memory without TorchCodec/torchaudio decoding.

    The project always normalizes uploads to uncompressed PCM WAV before this helper is
    used. Reading PCM directly with Python's ``wave`` module keeps pyannote completely
    independent of TorchCodec's file decoder.
    """
    try:
        import numpy as np
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("numpy and torch are required for local waveform loading.") from exc

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    if path.suffix.lower() not in {".wav", ".wave"}:
        raise ValueError(
            f"In-memory audio loading expects a normalized PCM WAV, got: {path.name}. "
            "Call convert_to_mono_16k_wav() first."
        )

    with wave.open(str(path), "rb") as wav:
        channels = int(wav.getnchannels())
        sample_rate = int(wav.getframerate())
        sample_width = int(wav.getsampwidth())
        frame_count = int(wav.getnframes())
        compression = wav.getcomptype()
        raw = wav.readframes(frame_count)

    if compression != "NONE":
        raise RuntimeError(f"Compressed WAV is unsupported for direct PCM loading: {compression}")
    if channels < 1:
        raise RuntimeError(f"Invalid WAV channel count: {channels}")

    if sample_width == 1:
        pcm = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        pcm = (pcm - 128.0) / 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 3:
        packed = np.frombuffer(raw, dtype=np.uint8)
        if packed.size % 3:
            raise RuntimeError("Invalid 24-bit WAV byte length.")
        packed = packed.reshape(-1, 3).astype(np.int32)
        values = packed[:, 0] | (packed[:, 1] << 8) | (packed[:, 2] << 16)
        values = np.where(values & 0x800000, values - 0x1000000, values)
        pcm = values.astype(np.float32) / 8388608.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise RuntimeError(f"Unsupported PCM WAV sample width: {sample_width} bytes")

    if pcm.size % channels:
        raise RuntimeError("PCM sample count is not divisible by the WAV channel count.")
    pcm = pcm.reshape(-1, channels).T
    waveform = torch.from_numpy(pcm.copy()).to(dtype=torch.float32)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    target_sample_rate = int(target_sample_rate)
    if target_sample_rate <= 0:
        raise ValueError("target_sample_rate must be > 0")
    if sample_rate != target_sample_rate:
        raise ValueError(
            f"Expected normalized {target_sample_rate} Hz PCM WAV, got {sample_rate} Hz: {path}. "
            "Call convert_to_mono_16k_wav() first."
        )

    waveform = waveform.clamp(-1.0, 1.0).contiguous()
    return {"waveform": waveform, "sample_rate": sample_rate}


def crop_waveform(audio: Dict[str, Any], start: float, end: float) -> Dict[str, Any]:
    sample_rate = int(audio["sample_rate"])
    waveform = audio["waveform"]
    start_idx = max(0, int(round(float(start) * sample_rate)))
    end_idx = min(int(waveform.shape[-1]), int(round(float(end) * sample_rate)))
    if end_idx <= start_idx:
        raise ValueError(f"Invalid crop interval: start={start}, end={end}")
    return {
        "waveform": waveform[..., start_idx:end_idx].contiguous(),
        "sample_rate": sample_rate,
    }


def concatenate_wav_intervals(
    input_path: str | Path,
    intervals: list[tuple[float, float]],
    output_path: str | Path,
    *,
    max_total_seconds: float = 18.0,
    gap_seconds: float = 0.20,
) -> Dict[str, Any]:
    """Concatenate selected intervals from a normalized PCM WAV into one preview WAV."""
    src, dst = Path(input_path), Path(output_path)
    if not src.is_file():
        raise FileNotFoundError(f"Audio file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)

    with wave.open(str(src), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        sample_rate = reader.getframerate()
        total_frames = reader.getnframes()
        if reader.getcomptype() != "NONE":
            raise RuntimeError("Speaker preview input must be uncompressed PCM WAV.")
        selected_payloads: list[bytes] = []
        used_seconds = 0.0
        for start, end in intervals:
            if used_seconds >= float(max_total_seconds):
                break
            start = max(0.0, float(start))
            end = max(start, float(end))
            remaining = float(max_total_seconds) - used_seconds
            end = min(end, start + remaining)
            start_frame = min(total_frames, max(0, int(round(start * sample_rate))))
            end_frame = min(total_frames, max(start_frame, int(round(end * sample_rate))))
            if end_frame <= start_frame:
                continue
            reader.setpos(start_frame)
            selected_payloads.append(reader.readframes(end_frame - start_frame))
            used_seconds += (end_frame - start_frame) / float(sample_rate)

    if not selected_payloads:
        raise ValueError("No valid audio intervals were supplied for the speaker preview.")

    silence_frames = max(0, int(round(float(gap_seconds) * sample_rate)))
    silence = b"\x00" * silence_frames * channels * sample_width
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        for idx, payload in enumerate(selected_payloads):
            if idx:
                writer.writeframes(silence)
            writer.writeframes(payload)
    return wav_info(dst)
