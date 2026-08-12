"""Groq Whisper speech-to-text with word-level timestamps.

This module is a drop-in replacement for the previous local faster-whisper
implementation used by the V4 call-intelligence pipeline.

Why word timestamps are required:
    Groq Whisper -> word timestamps -> pyannote speaker intervals
    -> word-level speaker attribution -> reconstructed conversation turns.

The module intentionally keeps the public ``transcribe_audio`` contract and
result shape used by the existing pipeline so ``audio_function.py`` and
``transcript_aligner.py`` do not need to change.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from project_utils import utc_now_iso, write_json

log = logging.getLogger("audio-transcription")

# -----------------------------------------------------------------------------
# Environment-driven configuration
# -----------------------------------------------------------------------------
DEFAULT_GROQ_WHISPER_MODEL = os.environ.get(
    "GROQ_WHISPER_MODEL",
    "whisper-large-v3-turbo",
).strip()
DEFAULT_WHISPER_LANGUAGE = (os.environ.get("WHISPER_LANGUAGE", "") or "").strip() or None
DEFAULT_WORD_TIMESTAMPS = (
    (os.environ.get("WHISPER_WORD_TIMESTAMPS", "1") or "1").strip().lower()
    not in {"0", "false", "no", "off"}
)
DEFAULT_GROQ_WHISPER_PROMPT = (os.environ.get("GROQ_WHISPER_PROMPT", "") or "").strip() or None
DEFAULT_GROQ_WHISPER_TEMPERATURE = float(os.environ.get("GROQ_WHISPER_TEMPERATURE", "0"))

# Backward-compatible aliases. These are retained so any code importing the
# previous constants does not fail after switching away from faster-whisper.
DEFAULT_WHISPER_MODEL = DEFAULT_GROQ_WHISPER_MODEL
DEFAULT_WHISPER_DEVICE = "groq_api"
DEFAULT_WHISPER_COMPUTE_TYPE = "remote"
DEFAULT_BEAM_SIZE = 1
DEFAULT_VAD_FILTER = False
DEFAULT_CONDITION_ON_PREVIOUS = False
DEFAULT_LOCAL_ONLY = False

SUPPORTED_AUDIO_EXTENSIONS = {
    ".flac",
    ".mp3",
    ".mp4",
    ".mpeg",
    ".mpga",
    ".m4a",
    ".ogg",
    ".wav",
    ".webm",
}


# -----------------------------------------------------------------------------
# Generic response helpers
# -----------------------------------------------------------------------------
def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    try:
        return list(value)
    except TypeError:
        return []


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalise_text(text: Any) -> str:
    return " ".join(str(text or "").split()).strip()


def _overlap_duration(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


# -----------------------------------------------------------------------------
# Groq client
# -----------------------------------------------------------------------------
def get_transcription_client():
    """Create a Groq client using GROQ_API_KEY from the environment."""
    api_key = (os.environ.get("GROQ_API_KEY", "") or "").strip()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to your local .env/environment before "
            "using Groq Whisper transcription."
        )

    try:
        from groq import Groq
    except ImportError as exc:
        raise RuntimeError(
            "The `groq` package is not installed. Run `pip install groq` or "
            "`pip install -r requirements.txt`."
        ) from exc

    return Groq(api_key=api_key)


# Backward-compatible name. The old implementation returned a local Whisper
# model; the new implementation returns the Groq client used for transcription.
def get_transcription_model(*args: Any, **kwargs: Any):
    del args, kwargs
    return get_transcription_client()


# -----------------------------------------------------------------------------
# Response normalization
# -----------------------------------------------------------------------------
def _normalise_segments(raw_segments: Sequence[Any]) -> List[Dict[str, Any]]:
    segments: List[Dict[str, Any]] = []

    for index, raw in enumerate(raw_segments):
        start = _safe_float(_get(raw, "start"), 0.0) or 0.0
        end = _safe_float(_get(raw, "end"), start)
        if end is None:
            end = start
        if end < start:
            start, end = end, start

        text = _normalise_text(_get(raw, "text", ""))
        if not text and end <= start:
            continue

        # Use a stable zero-based ID because the word-alignment code depends on
        # segment IDs matching the normalized segment list, not SDK internals.
        segment_id = index
        avg_logprob = _safe_float(_get(raw, "avg_logprob"))
        no_speech_prob = _safe_float(_get(raw, "no_speech_prob"))
        compression_ratio = _safe_float(_get(raw, "compression_ratio"))

        segments.append(
            {
                "segment_id": segment_id,
                "start": round(max(0.0, start), 3),
                "end": round(max(0.0, end), 3),
                "text": text,
                "avg_logprob": round(avg_logprob, 6) if avg_logprob is not None else None,
                "no_speech_prob": (
                    round(no_speech_prob, 6) if no_speech_prob is not None else None
                ),
                "compression_ratio": (
                    round(compression_ratio, 6) if compression_ratio is not None else None
                ),
                "words": [],
            }
        )

    return segments


def _choose_segment_id(
    word_start: float,
    word_end: float,
    segments: Sequence[Dict[str, Any]],
) -> int:
    """Attach a Groq top-level word timestamp to the best transcript segment."""
    if not segments:
        return 0

    best_segment_id = int(segments[0]["segment_id"])
    best_overlap = -1.0

    for segment in segments:
        overlap = _overlap_duration(
            word_start,
            word_end,
            float(segment["start"]),
            float(segment["end"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_segment_id = int(segment["segment_id"])

    if best_overlap > 0:
        return best_segment_id

    midpoint = (word_start + word_end) / 2.0
    nearest = min(
        segments,
        key=lambda segment: abs(
            midpoint
            - ((float(segment["start"]) + float(segment["end"])) / 2.0)
        ),
    )
    return int(nearest["segment_id"])


def _normalise_words(
    raw_words: Sequence[Any],
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    words: List[Dict[str, Any]] = []
    segment_lookup = {int(segment["segment_id"]): segment for segment in segments}

    for word_id, raw in enumerate(raw_words):
        text = str(_get(raw, "word", "") or "")
        if not text.strip():
            continue

        start = _safe_float(_get(raw, "start"))
        end = _safe_float(_get(raw, "end"))
        if start is None or end is None:
            continue
        if end < start:
            start, end = end, start

        segment_id = _choose_segment_id(start, end, segments)

        # Groq's documented word timestamp object contains word/start/end. Keep
        # probability nullable because it is not guaranteed for Groq word output.
        probability = _safe_float(_get(raw, "probability"))
        record = {
            "word_id": len(words),
            "segment_id": segment_id,
            "start": round(max(0.0, start), 3),
            "end": round(max(0.0, end), 3),
            # Preserve the text returned by Groq. transcript_aligner.py uses
            # timing for attribution and can normalize spacing when rebuilding turns.
            "word": text,
            "probability": round(probability, 6) if probability is not None else None,
        }
        words.append(record)

        if segment_id in segment_lookup:
            segment_lookup[segment_id]["words"].append(record)

    return words


def _synthesise_segment_from_words(
    transcription_text: str,
    raw_words: Sequence[Any],
) -> List[Dict[str, Any]]:
    starts = [
        value
        for value in (_safe_float(_get(word, "start")) for word in raw_words)
        if value is not None
    ]
    ends = [
        value
        for value in (_safe_float(_get(word, "end")) for word in raw_words)
        if value is not None
    ]
    if not starts or not ends:
        return []

    return [
        {
            "segment_id": 0,
            "start": round(max(0.0, min(starts)), 3),
            "end": round(max(0.0, max(ends)), 3),
            "text": _normalise_text(transcription_text),
            "avg_logprob": None,
            "no_speech_prob": None,
            "compression_ratio": None,
            "words": [],
        }
    ]


def _duration_from_response(
    response: Any,
    segments: Sequence[Dict[str, Any]],
    words: Sequence[Dict[str, Any]],
) -> float:
    duration = _safe_float(_get(response, "duration"))
    if duration is not None and duration >= 0:
        return duration

    candidates: List[float] = []
    candidates.extend(float(segment["end"]) for segment in segments)
    candidates.extend(float(word["end"]) for word in words)
    return max(candidates, default=0.0)


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def transcribe_audio(
    audio_path: str | Path,
    *,
    output_path: str | Path | None = None,
    model_name_or_path: str = DEFAULT_GROQ_WHISPER_MODEL,
    device: str = DEFAULT_WHISPER_DEVICE,
    compute_type: str = DEFAULT_WHISPER_COMPUTE_TYPE,
    language: str | None = DEFAULT_WHISPER_LANGUAGE,
    beam_size: int = DEFAULT_BEAM_SIZE,
    vad_filter: bool = DEFAULT_VAD_FILTER,
    word_timestamps: bool = DEFAULT_WORD_TIMESTAMPS,
    condition_on_previous_text: bool = DEFAULT_CONDITION_ON_PREVIOUS,
    local_files_only: bool = DEFAULT_LOCAL_ONLY,
    prompt: str | None = DEFAULT_GROQ_WHISPER_PROMPT,
    temperature: float = DEFAULT_GROQ_WHISPER_TEMPERATURE,
) -> Dict[str, Any]:
    """Transcribe audio with Groq Whisper and persist segment + word timestamps.

    Parameters retained from the previous faster-whisper implementation
    (device, compute_type, beam_size, vad_filter, condition_on_previous_text,
    local_files_only) are accepted for drop-in compatibility but are not used by
    Groq's hosted Whisper endpoint.
    """
    del device, compute_type, beam_size, vad_filter, condition_on_previous_text, local_files_only

    audio_path = Path(audio_path)
    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    suffix = audio_path.suffix.lower()
    if suffix not in SUPPORTED_AUDIO_EXTENSIONS:
        raise ValueError(
            f"Unsupported audio format {suffix!r}. Groq transcription supports: "
            + ", ".join(sorted(SUPPORTED_AUDIO_EXTENSIONS))
        )

    model = str(model_name_or_path or DEFAULT_GROQ_WHISPER_MODEL).strip()
    if model not in {"whisper-large-v3", "whisper-large-v3-turbo"}:
        raise ValueError(
            "Groq Whisper model must be `whisper-large-v3` or "
            "`whisper-large-v3-turbo`."
        )

    timestamp_granularities = ["segment"]
    if word_timestamps:
        timestamp_granularities = ["word", "segment"]

    request: Dict[str, Any] = {
        "model": model,
        "response_format": "verbose_json",
        "timestamp_granularities": timestamp_granularities,
        "temperature": float(temperature),
    }
    if language:
        request["language"] = str(language).strip()
    if prompt:
        request["prompt"] = str(prompt).strip()

    client = get_transcription_client()
    log.info(
        "Starting Groq Whisper transcription: file=%s model=%s language=%s word_timestamps=%s",
        audio_path,
        model,
        language or "auto",
        bool(word_timestamps),
    )

    try:
        # Passing an open binary file avoids reading the whole recording into Python
        # memory before the SDK constructs the multipart upload.
        with audio_path.open("rb") as audio_file:
            response = client.audio.transcriptions.create(
                file=audio_file,
                **request,
            )
    except Exception as exc:
        message = str(exc).strip() or exc.__class__.__name__
        raise RuntimeError(
            f"Groq Whisper transcription failed for '{audio_path.name}': {message}"
        ) from exc

    transcription_text = _normalise_text(_get(response, "text", ""))
    raw_segments = _as_list(_get(response, "segments"))
    raw_words = _as_list(_get(response, "words")) if word_timestamps else []

    segments = _normalise_segments(raw_segments)
    if not segments and raw_words:
        segments = _synthesise_segment_from_words(transcription_text, raw_words)

    flat_words = _normalise_words(raw_words, segments) if word_timestamps else []

    # If Groq returned segment timestamps but no top-level text, recover text from
    # segment text so downstream audit/debug output remains useful.
    if not transcription_text and segments:
        transcription_text = _normalise_text(
            " ".join(str(segment.get("text") or "") for segment in segments)
        )

    detected_language = _get(response, "language", language)
    language_probability = _safe_float(_get(response, "language_probability"), 0.0) or 0.0
    duration_seconds = _duration_from_response(response, segments, flat_words)

    result: Dict[str, Any] = {
        "created_at": utc_now_iso(),
        "audio_path": str(audio_path),
        "model": model,
        "provider": "groq",
        "device": "groq_api",
        "compute_type": "remote",
        "language": detected_language,
        "language_probability": float(language_probability),
        "duration_seconds": float(duration_seconds),
        "word_timestamps": bool(word_timestamps),
        "text": transcription_text,
        "segments": segments,
        "words": flat_words,
        "word_count": len(flat_words),
    }

    if word_timestamps and not flat_words:
        log.warning(
            "Groq Whisper returned no usable word timestamps. The speaker aligner "
            "will have to fall back to coarse segment timestamps."
        )

    if not segments and not flat_words and not transcription_text:
        log.warning("Groq Whisper returned an empty transcription for %s", audio_path)

    if output_path:
        write_json(output_path, result)

    log.info(
        "Groq Whisper transcription completed: segments=%d words=%d language=%s duration=%.2fs",
        len(segments),
        len(flat_words),
        detected_language or "unknown",
        duration_seconds,
    )
    return result


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Transcribe an audio file with Groq Whisper and word timestamps"
    )
    parser.add_argument("audio")
    parser.add_argument("--output")
    parser.add_argument(
        "--model",
        default=DEFAULT_GROQ_WHISPER_MODEL,
        choices=["whisper-large-v3", "whisper-large-v3-turbo"],
    )
    parser.add_argument("--language", default=DEFAULT_WHISPER_LANGUAGE)
    parser.add_argument("--prompt", default=DEFAULT_GROQ_WHISPER_PROMPT)
    parser.add_argument("--no-word-timestamps", action="store_true")
    args = parser.parse_args()

    print(
        json.dumps(
            transcribe_audio(
                args.audio,
                output_path=args.output,
                model_name_or_path=args.model,
                language=args.language,
                prompt=args.prompt,
                word_timestamps=not args.no_word_timestamps,
            ),
            indent=2,
            ensure_ascii=False,
        )
    )