"""Local speaker diarization with pyannote Community-1.

Important Windows behavior: normalized PCM WAV is read directly into a tensor and
passed to pyannote as an in-memory {waveform, sample_rate} mapping. No pyannote or
torchaudio/TorchCodec filepath decoding is used during diarization.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import config
from audio_utils import load_waveform_dict
from project_utils import utc_now_iso, write_json

log = logging.getLogger("pyannote-diarizer")

DEFAULT_PYANNOTE_MODEL = os.environ.get(
    "PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1"
)
DEFAULT_PYANNOTE_DEVICE = os.environ.get("PYANNOTE_DEVICE", "auto").lower()
DEFAULT_USE_EXCLUSIVE = config.env_bool("PYANNOTE_USE_EXCLUSIVE", True)
DEFAULT_MERGE_GAP_SECONDS = config.env_float("PYANNOTE_MERGE_GAP_SECONDS", 0.00)
DEFAULT_MIN_SEGMENT_SECONDS = config.env_float("PYANNOTE_MIN_SEGMENT_SECONDS", 0.05)
PYANNOTE_CACHE_DIR = config.MODELS_DIR / "pyannote"


def _torch_cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def resolve_device(device: str = "auto") -> str:
    device = str(device or "auto").strip().lower()
    if device == "auto":
        return "cuda" if _torch_cuda_available() else "cpu"
    if device not in {"cpu", "cuda"}:
        raise ValueError("PYANNOTE_DEVICE must be one of: auto, cpu, cuda")
    if device == "cuda" and not _torch_cuda_available():
        raise RuntimeError("PYANNOTE_DEVICE=cuda was requested but torch.cuda.is_available() is False.")
    return device


def _is_local_checkpoint(value: str) -> bool:
    path = Path(value)
    if not path.is_absolute():
        path = config.PROJECT_ROOT / path
    return path.exists()


@lru_cache(maxsize=4)
def _get_pipeline_cached(model_name: str, device: str, token: str | None):
    try:
        import torch
        from pyannote.audio import Pipeline
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyannote.audio is not installed correctly.") from exc

    if not _is_local_checkpoint(model_name) and not token and not config.LOCAL_MODELS_ONLY:
        raise RuntimeError(
            "HF_TOKEN is required the first time Community-1 is downloaded. Accept the "
            "model terms on Hugging Face, place HF_TOKEN in .env, and retry."
        )

    checkpoint: str | Path = model_name
    candidate = Path(model_name)
    if not candidate.is_absolute() and (config.PROJECT_ROOT / candidate).exists():
        checkpoint = (config.PROJECT_ROOT / candidate).resolve()

    log.info("Loading pyannote pipeline=%s device=%s", checkpoint, device)
    try:
        pipeline = Pipeline.from_pretrained(
            checkpoint,
            token=token,
            cache_dir=PYANNOTE_CACHE_DIR,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load pyannote diarization pipeline '{model_name}'. If this is the "
            "first run, verify that you accepted Community-1 terms and HF_TOKEN is valid. "
            "For offline runs, pre-cache the model or set PYANNOTE_MODEL to a local clone."
        ) from exc
    if pipeline is None:
        raise RuntimeError(f"pyannote returned no pipeline for checkpoint: {model_name}")
    pipeline.to(torch.device(device))
    return pipeline


def get_diarization_pipeline(
    model_name: str = DEFAULT_PYANNOTE_MODEL,
    device: str = DEFAULT_PYANNOTE_DEVICE,
):
    resolved_device = resolve_device(device)
    token = None if _is_local_checkpoint(model_name) else config.hf_token()
    return _get_pipeline_cached(model_name, resolved_device, token)


def _annotation_from_output(output: Any, use_exclusive: bool) -> Any:
    if use_exclusive and hasattr(output, "exclusive_speaker_diarization"):
        annotation = output.exclusive_speaker_diarization
        if annotation is not None:
            return annotation
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    return output  # pyannote 3.x legacy Annotation


def _iter_annotation(annotation: Any) -> Iterable[Tuple[float, float, str]]:
    if hasattr(annotation, "itertracks"):
        for segment, _track, label in annotation.itertracks(yield_label=True):
            yield float(segment.start), float(segment.end), str(label)
        return
    # pyannote 4 docs also allow `for turn, speaker in annotation`.
    try:
        for item in annotation:
            if isinstance(item, tuple) and len(item) == 2:
                segment, label = item
                yield float(segment.start), float(segment.end), str(label)
            elif isinstance(item, tuple) and len(item) >= 3:
                segment, _track, label = item[:3]
                yield float(segment.start), float(segment.end), str(label)
    except TypeError as exc:
        raise RuntimeError(f"Unsupported pyannote annotation type: {type(annotation)!r}") from exc


def _merge_segments(segments: List[Dict[str, Any]], max_gap: float) -> List[Dict[str, Any]]:
    if not segments:
        return []
    merged: List[Dict[str, Any]] = [dict(segments[0])]
    for current in segments[1:]:
        previous = merged[-1]
        gap = float(current["start"]) - float(previous["end"])
        if current["speaker"] == previous["speaker"] and gap <= max_gap:
            previous["end"] = max(float(previous["end"]), float(current["end"]))
            previous["duration"] = round(float(previous["end"]) - float(previous["start"]), 3)
        else:
            merged.append(dict(current))
    for idx, item in enumerate(merged):
        item["segment_id"] = idx
    return merged


def annotation_to_segments(
    annotation: Any,
    *,
    min_segment_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
    merge_gap_seconds: float = DEFAULT_MERGE_GAP_SECONDS,
) -> List[Dict[str, Any]]:
    raw = sorted(_iter_annotation(annotation), key=lambda x: (x[0], x[1], x[2]))
    segments: List[Dict[str, Any]] = []
    for start, end, speaker in raw:
        duration = max(0.0, end - start)
        if duration < float(min_segment_seconds):
            continue
        segments.append(
            {
                "segment_id": len(segments),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "speaker": speaker,
            }
        )
    return _merge_segments(segments, float(merge_gap_seconds))


def write_rttm(segments: List[Dict[str, Any]], path: str | Path, uri: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for item in segments:
            start = float(item["start"])
            duration = max(0.0, float(item["end"]) - start)
            speaker = str(item["speaker"])
            file.write(
                f"SPEAKER {uri} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n"
            )


def diarize_audio(
    audio_path: str | Path,
    *,
    output_path: str | Path | None = None,
    rttm_path: str | Path | None = None,
    model_name: str = DEFAULT_PYANNOTE_MODEL,
    device: str = DEFAULT_PYANNOTE_DEVICE,
    num_speakers: int | None = 2,
    min_speakers: int | None = None,
    max_speakers: int | None = None,
    use_exclusive: bool = DEFAULT_USE_EXCLUSIVE,
    uri: str | None = None,
) -> Dict[str, Any]:
    """Diarize a normalized WAV entirely locally."""
    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    resolved_device = resolve_device(device)
    pipeline = get_diarization_pipeline(model_name=model_name, device=resolved_device)
    in_memory_audio = load_waveform_dict(audio_path, target_sample_rate=16000)
    # A URI keeps pyannote metadata/RTTM stable while still avoiding filepath decoding.
    recording_uri = str(uri or audio_path.stem).strip() or audio_path.stem
    pyannote_input = {
        "waveform": in_memory_audio["waveform"],
        "sample_rate": in_memory_audio["sample_rate"],
        "uri": recording_uri,
    }

    kwargs: Dict[str, Any] = {}
    if num_speakers is not None:
        if int(num_speakers) < 1:
            raise ValueError("num_speakers must be >= 1")
        kwargs["num_speakers"] = int(num_speakers)
    else:
        if min_speakers is not None:
            kwargs["min_speakers"] = int(min_speakers)
        if max_speakers is not None:
            kwargs["max_speakers"] = int(max_speakers)

    output = pipeline(pyannote_input, **kwargs)
    annotation = _annotation_from_output(output, bool(use_exclusive))
    segments = annotation_to_segments(annotation)
    speakers = []
    for item in segments:
        if item["speaker"] not in speakers:
            speakers.append(item["speaker"])

    result: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "audio_path": str(audio_path),
        "uri": recording_uri,
        "model": model_name,
        "device": resolved_device,
        "exclusive_diarization": bool(use_exclusive),
        "requested_num_speakers": num_speakers,
        "speakers": speakers,
        "segments": segments,
    }
    if output_path:
        write_json(output_path, result)
    if rttm_path:
        write_rttm(segments, rttm_path, uri=recording_uri)
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Local pyannote speaker diarization")
    parser.add_argument("audio")
    parser.add_argument("--output", default="diarization.json")
    parser.add_argument("--rttm", default="diarization.rttm")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--device", default=DEFAULT_PYANNOTE_DEVICE)
    args = parser.parse_args()
    print(json.dumps(diarize_audio(args.audio, output_path=args.output, rttm_path=args.rttm, num_speakers=args.num_speakers, device=args.device), indent=2))
