"""Robust local speaker-embedding extraction using pyannote WeSpeaker.

The diarization model tells us *when* anonymous speakers talk. This module creates
voice embeddings from multiple clean windows inside those turns. It deliberately:
- reads normalized PCM WAV into memory (no TorchCodec filepath decoding),
- trims turn boundaries to reduce cross-speaker contamination,
- splits very long turns into moderate windows,
- keeps multiple embeddings instead of trusting one long crop,
- removes internally inconsistent windows before building the speaker centroid.
"""
from __future__ import annotations

import logging
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import config
from audio_utils import crop_waveform, load_waveform_dict
from project_utils import slugify, utc_now_iso, write_json

log = logging.getLogger("speaker-embedder")

DEFAULT_MODEL = os.environ.get(
    "SPEAKER_EMBEDDING_MODEL", "pyannote/wespeaker-voxceleb-resnet34-LM"
)
DEFAULT_DEVICE = os.environ.get("SPEAKER_EMBEDDING_DEVICE", "auto").lower()
DEFAULT_MIN_SEGMENT_SECONDS = config.env_float("MIN_SPEAKER_SEGMENT_DURATION_SECONDS", 1.5)
DEFAULT_MAX_SEGMENTS_PER_SPEAKER = config.env_int("MAX_SEGMENTS_PER_SPEAKER", 14)
DEFAULT_MAX_TOTAL_SECONDS_PER_SPEAKER = config.env_float("MAX_TOTAL_SECONDS_PER_SPEAKER", 70.0)
DEFAULT_BOUNDARY_TRIM_SECONDS = config.env_float("SPEAKER_EMBED_BOUNDARY_TRIM_SECONDS", 0.20)
DEFAULT_MAX_WINDOW_SECONDS = config.env_float("SPEAKER_EMBED_MAX_WINDOW_SECONDS", 8.0)
DEFAULT_WEIGHT_CAP_SECONDS = config.env_float("SPEAKER_EMBED_WEIGHT_CAP_SECONDS", 6.0)
DEFAULT_ENROLL_WINDOW_SECONDS = config.env_float("AGENT_ENROLL_WINDOW_SECONDS", 6.0)
DEFAULT_ENROLL_HOP_SECONDS = config.env_float("AGENT_ENROLL_HOP_SECONDS", 4.0)
DEFAULT_ENROLL_MAX_WINDOWS = config.env_int("AGENT_ENROLL_MAX_WINDOWS", 8)
EMBEDDING_CACHE_DIR = config.MODELS_DIR / "pyannote"


def normalize_embedding(embedding: Any) -> np.ndarray:
    arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("Speaker embedding is empty or contains non-finite values.")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError("Speaker embedding has zero norm.")
    return (arr / norm).astype(np.float32)


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
        raise ValueError("SPEAKER_EMBEDDING_DEVICE must be one of: auto, cpu, cuda")
    if device == "cuda" and not _torch_cuda_available():
        raise RuntimeError("Speaker embedding device cuda requested but CUDA is unavailable.")
    return device


def _resolve_checkpoint(model_name: str) -> str | Path:
    candidate = Path(model_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate
    project_candidate = (config.PROJECT_ROOT / candidate).resolve()
    if project_candidate.exists():
        return project_candidate
    return model_name


@lru_cache(maxsize=4)
def _get_inference_cached(model_name: str, device: str, token: str | None):
    try:
        import torch
        from pyannote.audio import Inference, Model
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("pyannote.audio is required for speaker embeddings.") from exc

    checkpoint = _resolve_checkpoint(model_name)
    try:
        model = Model.from_pretrained(checkpoint, token=token, cache_dir=EMBEDDING_CACHE_DIR)
    except Exception as exc:
        raise RuntimeError(
            f"Could not load speaker embedding model '{model_name}'. Verify the model is "
            "cached/accessible and the installed pyannote.audio version is compatible."
        ) from exc
    if model is None:
        raise RuntimeError(f"pyannote returned no speaker embedding model for {model_name}")

    inference = Inference(model, window="whole")
    inference.to(torch.device(device))
    return inference


def get_embedding_inference(model_name: str = DEFAULT_MODEL, device: str = DEFAULT_DEVICE):
    resolved_device = resolve_device(device)
    checkpoint = _resolve_checkpoint(model_name)
    token = None if isinstance(checkpoint, Path) else config.hf_token()
    return _get_inference_cached(model_name, resolved_device, token)


def extract_voice_embedding(
    audio: str | Path | Dict[str, Any],
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
) -> np.ndarray:
    """Extract one L2-normalized embedding from one waveform/file."""
    inference = get_embedding_inference(model_name=model_name, device=device)
    in_memory = audio if isinstance(audio, dict) else load_waveform_dict(audio, target_sample_rate=16000)
    return normalize_embedding(inference(in_memory))


def _audio_duration(audio: Dict[str, Any]) -> float:
    return float(audio["waveform"].shape[-1]) / float(audio["sample_rate"])


def _rms(audio: Dict[str, Any]) -> float:
    waveform = audio["waveform"]
    try:
        return float((waveform.float().pow(2).mean().sqrt()).item())
    except Exception:
        arr = np.asarray(waveform, dtype=np.float32)
        return float(np.sqrt(np.mean(np.square(arr))))


def _trim_interval(start: float, end: float, trim: float, minimum: float) -> Tuple[float, float]:
    start, end = max(0.0, float(start)), max(0.0, float(end))
    if end <= start:
        return start, start
    duration = end - start
    # Only use the full configured trim when enough clean speech remains.
    allowed_trim = min(float(trim), max(0.0, (duration - float(minimum)) / 2.0))
    return start + allowed_trim, end - allowed_trim


def _split_interval(
    start: float,
    end: float,
    *,
    min_seconds: float,
    max_seconds: float,
) -> List[Tuple[float, float]]:
    duration = end - start
    if duration < min_seconds:
        return []
    if duration <= max_seconds:
        return [(start, end)]

    windows: List[Tuple[float, float]] = []
    cursor = start
    while cursor < end:
        win_end = min(end, cursor + max_seconds)
        if win_end - cursor < min_seconds:
            if windows:
                # Keep the remainder by extending the previous window only modestly.
                prev_start, _ = windows[-1]
                windows[-1] = (prev_start, end)
            break
        windows.append((cursor, win_end))
        cursor = win_end
    return windows


def group_diarization_by_speaker(
    diarization_segments: Sequence[Dict[str, Any]],
    *,
    min_duration: float = DEFAULT_MIN_SEGMENT_SECONDS,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for item in diarization_segments:
        speaker = str(item.get("speaker") or "").strip()
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if not speaker or end - start < float(min_duration):
            continue
        grouped.setdefault(speaker, []).append(dict(item))
    for speaker in grouped:
        grouped[speaker].sort(key=lambda x: float(x["end"]) - float(x["start"]), reverse=True)
    return grouped


def build_embedding_windows(
    segments: Sequence[Dict[str, Any]],
    *,
    min_seconds: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_seconds: float = DEFAULT_MAX_WINDOW_SECONDS,
    boundary_trim_seconds: float = DEFAULT_BOUNDARY_TRIM_SECONDS,
    max_windows: int = DEFAULT_MAX_SEGMENTS_PER_SPEAKER,
    max_total_seconds: float = DEFAULT_MAX_TOTAL_SECONDS_PER_SPEAKER,
) -> List[Dict[str, Any]]:
    """Build clean interior windows, preferring longer diarization turns."""
    candidates: List[Dict[str, Any]] = []
    for item in segments:
        start, end = _trim_interval(
            float(item["start"]), float(item["end"]),
            float(boundary_trim_seconds), float(min_seconds),
        )
        for win_start, win_end in _split_interval(
            start, end, min_seconds=float(min_seconds), max_seconds=float(max_seconds)
        ):
            candidates.append(
                {
                    "segment_id": item.get("segment_id"),
                    "start": win_start,
                    "end": win_end,
                    "duration": win_end - win_start,
                }
            )

    # Prefer longer windows, but stop once the configured evidence budget is reached.
    candidates.sort(key=lambda x: float(x["duration"]), reverse=True)
    selected: List[Dict[str, Any]] = []
    total = 0.0
    for item in candidates:
        if len(selected) >= int(max_windows) or total >= float(max_total_seconds):
            break
        record = dict(item)
        remaining = float(max_total_seconds) - total
        if record["duration"] > remaining:
            if remaining < float(min_seconds):
                break
            record["end"] = float(record["start"]) + remaining
            record["duration"] = remaining
        selected.append(record)
        total += float(record["duration"])
    return selected


def robust_aggregate_embeddings(
    embeddings: Sequence[np.ndarray],
    weights: Sequence[float] | None = None,
) -> Tuple[np.ndarray, List[int], List[float]]:
    """Return robust centroid, kept indexes, and per-window consistency scores."""
    if not embeddings:
        raise ValueError("No speaker embeddings supplied for aggregation.")
    matrix = np.vstack([normalize_embedding(x) for x in embeddings]).astype(np.float32)
    count = len(matrix)
    if count == 1:
        return normalize_embedding(matrix[0]), [0], [1.0]

    similarities = np.clip(matrix @ matrix.T, -1.0, 1.0)
    consistency: List[float] = []
    for i in range(count):
        others = np.delete(similarities[i], i)
        consistency.append(float(np.median(others)) if len(others) else 1.0)

    if count <= 2:
        kept = list(range(count))
    else:
        values = np.asarray(consistency, dtype=np.float32)
        median = float(np.median(values))
        mad = float(np.median(np.abs(values - median)))
        adaptive_floor = median - max(0.05, 2.5 * mad)
        kept = [i for i, value in enumerate(consistency) if value >= adaptive_floor]
        minimum_keep = min(count, max(2, int(math.ceil(count * 0.60))))
        if len(kept) < minimum_keep:
            kept = sorted(range(count), key=lambda i: consistency[i], reverse=True)[:minimum_keep]

    kept_matrix = matrix[kept]
    if weights is None:
        centroid = kept_matrix.mean(axis=0)
    else:
        raw_weights = np.asarray(weights, dtype=np.float32).reshape(-1)
        if len(raw_weights) != count:
            raise ValueError("Embedding weights length does not match embeddings length.")
        selected_weights = raw_weights[kept]
        if float(selected_weights.sum()) <= 0:
            selected_weights = np.ones_like(selected_weights)
        selected_weights = selected_weights / selected_weights.sum()
        centroid = np.sum(kept_matrix * selected_weights[:, None], axis=0)
    return normalize_embedding(centroid), kept, consistency


def aggregate_embeddings_weighted(
    embeddings: Sequence[np.ndarray], weights: Sequence[float] | None = None
) -> np.ndarray:
    """Backward-compatible helper; now uses robust aggregation."""
    centroid, _, _ = robust_aggregate_embeddings(embeddings, weights)
    return centroid


def extract_reference_profile(
    audio: str | Path | Dict[str, Any],
    *,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
    window_seconds: float = DEFAULT_ENROLL_WINDOW_SECONDS,
    hop_seconds: float = DEFAULT_ENROLL_HOP_SECONDS,
    max_windows: int = DEFAULT_ENROLL_MAX_WINDOWS,
    min_window_seconds: float = 2.5,
) -> Dict[str, Any]:
    """Create multiple prototypes and a robust centroid for agent enrollment."""
    full_audio = audio if isinstance(audio, dict) else load_waveform_dict(audio, target_sample_rate=16000)
    duration = _audio_duration(full_audio)
    if duration < min_window_seconds:
        embedding = extract_voice_embedding(full_audio, model_name=model_name, device=device)
        return {
            "centroid": embedding,
            "prototypes": np.vstack([embedding]),
            "windows": [{"start": 0.0, "end": duration, "duration": duration, "rms": _rms(full_audio), "used": True}],
        }

    window_seconds = max(float(min_window_seconds), float(window_seconds))
    hop_seconds = max(0.5, min(float(hop_seconds), window_seconds))
    candidates: List[Dict[str, Any]] = []
    start = 0.0
    while start < duration:
        end = min(duration, start + window_seconds)
        if end - start < min_window_seconds:
            if candidates:
                break
            start = max(0.0, duration - min_window_seconds)
            end = duration
        crop = crop_waveform(full_audio, start, end)
        candidates.append({"start": start, "end": end, "duration": end - start, "rms": _rms(crop)})
        if end >= duration:
            break
        start += hop_seconds

    # Low-energy windows are more likely to be silence. Keep the strongest windows while
    # still allowing normal variation in microphone gain.
    candidates.sort(key=lambda x: float(x["rms"]), reverse=True)
    selected = candidates[: max(1, int(max_windows))]
    if selected:
        strongest = float(selected[0]["rms"])
        energy_floor = strongest * 0.12
        filtered = [x for x in selected if float(x["rms"]) >= energy_floor]
        if filtered:
            selected = filtered

    embeddings: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []
    get_embedding_inference(model_name=model_name, device=device)
    for record in selected:
        crop = crop_waveform(full_audio, float(record["start"]), float(record["end"]))
        try:
            emb = extract_voice_embedding(crop, model_name=model_name, device=device)
        except Exception as exc:
            log.warning("Enrollment window %.2f-%.2f skipped: %s", record["start"], record["end"], exc)
            continue
        embeddings.append(emb)
        records.append(dict(record))

    if not embeddings:
        raise RuntimeError("No usable enrollment speaker embeddings could be extracted.")
    weights = [min(float(x["duration"]), DEFAULT_WEIGHT_CAP_SECONDS) for x in records]
    centroid, kept, consistency = robust_aggregate_embeddings(embeddings, weights)
    kept_set = set(kept)
    for idx, record in enumerate(records):
        record["consistency"] = round(float(consistency[idx]), 6)
        record["used"] = idx in kept_set
    prototypes = np.vstack([normalize_embedding(embeddings[i]) for i in kept]).astype(np.float32)
    return {"centroid": centroid, "prototypes": prototypes, "windows": records}


def create_call_speaker_embeddings(
    audio_path: str | Path,
    diarization_segments: Sequence[Dict[str, Any]] | Dict[str, Any],
    output_dir: str | Path,
    *,
    min_segment_duration: float = DEFAULT_MIN_SEGMENT_SECONDS,
    max_segments_per_speaker: int = DEFAULT_MAX_SEGMENTS_PER_SPEAKER,
    max_total_seconds_per_speaker: float = DEFAULT_MAX_TOTAL_SECONDS_PER_SPEAKER,
    model_name: str = DEFAULT_MODEL,
    device: str = DEFAULT_DEVICE,
) -> Dict[str, Dict[str, Any]]:
    """Create robust centroid + segment evidence for every usable diarized speaker."""
    if isinstance(diarization_segments, dict):
        diarization_segments = diarization_segments.get("segments", [])
    audio_path = Path(audio_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    full_audio = load_waveform_dict(audio_path, target_sample_rate=16000)
    grouped = group_diarization_by_speaker(list(diarization_segments), min_duration=min_segment_duration)
    if not grouped:
        raise ValueError(
            "No diarization segments are long enough for speaker embeddings. "
            f"Current minimum is {min_segment_duration:.2f}s."
        )

    get_embedding_inference(model_name=model_name, device=device)
    results: Dict[str, Dict[str, Any]] = {}
    manifest: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "model": model_name,
        "device": resolve_device(device),
        "strategy": "boundary_trimmed_multi_window_robust_centroid",
        "speakers": {},
    }

    for speaker, segments in grouped.items():
        windows = build_embedding_windows(
            segments,
            min_seconds=min_segment_duration,
            max_seconds=DEFAULT_MAX_WINDOW_SECONDS,
            boundary_trim_seconds=DEFAULT_BOUNDARY_TRIM_SECONDS,
            max_windows=max_segments_per_speaker,
            max_total_seconds=max_total_seconds_per_speaker,
        )
        embeddings: List[np.ndarray] = []
        records: List[Dict[str, Any]] = []
        for window in windows:
            start, end = float(window["start"]), float(window["end"])
            try:
                segment_audio = crop_waveform(full_audio, start, end)
                embedding = extract_voice_embedding(segment_audio, model_name=model_name, device=device)
            except Exception as exc:
                log.warning("Speaker embedding skipped for %s %.2f-%.2f: %s", speaker, start, end, exc)
                continue
            embeddings.append(embedding)
            records.append({
                "segment_id": window.get("segment_id"),
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(end - start, 3),
            })

        if not embeddings:
            log.warning("No usable speaker embedding windows for %s", speaker)
            continue

        weights = [min(float(x["duration"]), DEFAULT_WEIGHT_CAP_SECONDS) for x in records]
        centroid, kept, consistency = robust_aggregate_embeddings(embeddings, weights)
        kept_set = set(kept)
        for idx, record in enumerate(records):
            record["consistency"] = round(float(consistency[idx]), 6)
            record["used_in_centroid"] = idx in kept_set

        safe_speaker = slugify(speaker, fallback="speaker")
        embedding_path = output_dir / f"{safe_speaker}_embedding.npy"
        prototypes_path = output_dir / f"{safe_speaker}_prototypes.npy"
        prototypes = np.vstack([normalize_embedding(embeddings[i]) for i in kept]).astype(np.float32)
        np.save(embedding_path, centroid, allow_pickle=False)
        np.save(prototypes_path, prototypes, allow_pickle=False)

        record = {
            "speaker": speaker,
            "embedding": centroid,
            "segment_embeddings": prototypes,
            "embedding_path": str(embedding_path),
            "prototypes_path": str(prototypes_path),
            "embedding_dimension": int(centroid.size),
            "num_windows_extracted": len(embeddings),
            "num_windows_used": len(kept),
            "total_duration": round(float(sum(x["duration"] for x in records if x["used_in_centroid"])), 3),
            "segments_used": records,
            "model": model_name,
            "device": resolve_device(device),
        }
        results[speaker] = record
        manifest["speakers"][speaker] = {
            k: v for k, v in record.items() if k not in {"embedding", "segment_embeddings"}
        }

    if not results:
        raise RuntimeError("No call speaker embeddings could be created.")
    write_json(output_dir / "speaker_embedding_manifest.json", manifest)
    return results


def json_safe_embedding_results(results: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        speaker: {
            k: v for k, v in item.items()
            if k not in {"embedding", "segment_embeddings"}
        }
        for speaker, item in results.items()
    }


if __name__ == "__main__":
    import argparse, json
    from project_utils import read_json
    parser = argparse.ArgumentParser(description="Local speaker embedding utility")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("single")
    one.add_argument("audio")
    one.add_argument("--output", required=True)
    one.add_argument("--device", default=DEFAULT_DEVICE)
    call = sub.add_parser("call")
    call.add_argument("audio")
    call.add_argument("diarization")
    call.add_argument("--output-dir", required=True)
    call.add_argument("--device", default=DEFAULT_DEVICE)
    args = parser.parse_args()
    if args.command == "single":
        emb = extract_voice_embedding(args.audio, device=args.device)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output, emb, allow_pickle=False)
        result = {"embedding_path": args.output, "dimension": int(emb.size)}
    else:
        result = json_safe_embedding_results(
            create_call_speaker_embeddings(
                args.audio, read_json(args.diarization), args.output_dir, device=args.device
            )
        )
    print(json.dumps(result, indent=2))
