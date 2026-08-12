"""Word-level reconciliation of faster-whisper timestamps with pyannote diarization.

The key rule is: never assign a multi-second ASR segment to one speaker when word
timestamps are available. A Whisper segment can contain speech from multiple people.
Each word is assigned independently to the pyannote speaker with the strongest temporal
overlap and only then are adjacent words rebuilt into readable speaker turns.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from project_utils import read_json, utc_now_iso, write_json

UNKNOWN_SPEAKER = "UNKNOWN"


def _segments(value: Any) -> List[Dict[str, Any]]:
    if isinstance(value, dict):
        value = value.get("segments", [])
    if not isinstance(value, list):
        raise TypeError("Expected a list of segments or a dict containing `segments`.")
    return [dict(item) for item in value if isinstance(item, dict)]


def overlap_seconds(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def temporal_gap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    if overlap_seconds(a_start, a_end, b_start, b_end) > 0:
        return 0.0
    if a_end <= b_start:
        return b_start - a_end
    return a_start - b_end


def _speaker_at_midpoint(midpoint: float, diarization: List[Dict[str, Any]]) -> str | None:
    candidates = [
        d for d in diarization
        if float(d.get("start", 0.0)) <= midpoint <= float(d.get("end", 0.0))
    ]
    if not candidates:
        return None
    # Exclusive diarization should usually yield exactly one. If boundaries touch,
    # prefer the interval whose centre is closest to the word centre.
    chosen = min(
        candidates,
        key=lambda d: abs(
            midpoint - (float(d.get("start", 0.0)) + float(d.get("end", 0.0))) / 2.0
        ),
    )
    return str(chosen.get("speaker") or UNKNOWN_SPEAKER)


def _best_speaker(
    t_start: float,
    t_end: float,
    diarization: List[Dict[str, Any]],
    nearest_fallback_seconds: float,
) -> Tuple[str, float, str]:
    """Assign one small timestamp interval to a speaker.

    For non-zero intervals, use accumulated overlap by speaker. For zero/tiny intervals
    or exact boundary ties, the interval midpoint is a useful deterministic tiebreaker.
    """
    t_start = max(0.0, float(t_start))
    t_end = max(t_start, float(t_end))
    duration = t_end - t_start
    midpoint = (t_start + t_end) / 2.0

    overlap_by_speaker: Dict[str, float] = {}
    for d in diarization:
        start, end = float(d["start"]), float(d["end"])
        overlap = overlap_seconds(t_start, t_end, start, end)
        if overlap > 0:
            speaker = str(d.get("speaker") or UNKNOWN_SPEAKER)
            overlap_by_speaker[speaker] = overlap_by_speaker.get(speaker, 0.0) + overlap

    if overlap_by_speaker:
        ordered = sorted(overlap_by_speaker.items(), key=lambda item: item[1], reverse=True)
        best_overlap = ordered[0][1]
        tied = [speaker for speaker, value in ordered if abs(value - best_overlap) <= 1e-6]
        midpoint_speaker = _speaker_at_midpoint(midpoint, diarization)
        if midpoint_speaker in tied:
            speaker = midpoint_speaker
        else:
            speaker = ordered[0][0]
        if duration > 1e-6:
            confidence = min(1.0, overlap_by_speaker[speaker] / duration)
        else:
            confidence = 1.0
        return speaker, confidence, "timestamp_overlap"

    midpoint_speaker = _speaker_at_midpoint(midpoint, diarization)
    if midpoint_speaker:
        return midpoint_speaker, 0.75, "midpoint_inside_turn"

    nearest = None
    for d in diarization:
        gap = temporal_gap(t_start, t_end, float(d["start"]), float(d["end"]))
        if nearest is None or gap < nearest[0]:
            nearest = (gap, str(d.get("speaker") or UNKNOWN_SPEAKER))
    if nearest and nearest[0] <= float(nearest_fallback_seconds):
        confidence = max(
            0.0,
            1.0 - nearest[0] / max(float(nearest_fallback_seconds), 1e-9),
        )
        return nearest[1], confidence * 0.49, "nearest_boundary_fallback"

    return UNKNOWN_SPEAKER, 0.0, "unmatched"


def _flatten_words(transcription: Any) -> List[Dict[str, Any]]:
    """Return validated word records from top-level or per-segment Whisper output."""
    if not isinstance(transcription, dict):
        return []

    raw_words = transcription.get("words")
    candidates: List[Dict[str, Any]] = []
    if isinstance(raw_words, list) and raw_words:
        candidates = [dict(x) for x in raw_words if isinstance(x, dict)]
    else:
        for segment in transcription.get("segments", []) or []:
            if not isinstance(segment, dict):
                continue
            segment_id = segment.get("segment_id")
            for word in segment.get("words", []) or []:
                if not isinstance(word, dict):
                    continue
                record = dict(word)
                record.setdefault("segment_id", segment_id)
                candidates.append(record)

    output: List[Dict[str, Any]] = []
    for idx, word in enumerate(candidates):
        raw = str(word.get("word") or word.get("text") or "")
        if not raw.strip():
            continue
        try:
            start = float(word.get("start"))
            end = float(word.get("end"))
        except (TypeError, ValueError):
            continue
        if end < start:
            start, end = end, start
        output.append(
            {
                **word,
                "word_id": word.get("word_id", idx),
                "start": max(0.0, start),
                "end": max(0.0, end),
                "word": raw,
            }
        )
    output.sort(key=lambda x: (float(x["start"]), float(x["end"]), int(x.get("word_id", 0))))
    return output


def _normalize_spacing(text: str) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\s+([,.;:!?%])", r"\1", text)
    text = re.sub(r"([£$€₹])\s+", r"\1", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text


def _tokens_to_text(tokens: Iterable[str]) -> str:
    values = [str(x) for x in tokens if str(x)]
    if not values:
        return ""
    # faster-whisper normally stores leading whitespace in word.word. Preserve that
    # representation when present. Synthetic/legacy data may not, so join with spaces.
    if any(v[:1].isspace() for v in values[1:]):
        text = "".join(values)
    else:
        text = " ".join(v.strip() for v in values)
    return _normalize_spacing(text)


def _word_assignments(
    words: List[Dict[str, Any]],
    diarization: List[Dict[str, Any]],
    nearest_fallback_seconds: float,
) -> List[Dict[str, Any]]:
    assignments: List[Dict[str, Any]] = []
    for word in words:
        start, end = float(word["start"]), float(word["end"])
        speaker, confidence, method = _best_speaker(
            start, end, diarization, nearest_fallback_seconds
        )
        assignments.append(
            {
                "word_id": word.get("word_id"),
                "segment_id": word.get("segment_id"),
                "start": round(start, 3),
                "end": round(end, 3),
                "word": str(word.get("word") or ""),
                "probability": word.get("probability"),
                "speaker": speaker,
                "alignment_confidence": round(float(confidence), 4),
                "alignment_method": method,
            }
        )
    return assignments


def _turns_from_words(
    assignments: List[Dict[str, Any]],
    *,
    max_gap_seconds: float,
) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    current: Dict[str, Any] | None = None

    def finish() -> None:
        nonlocal current
        if current is None:
            return
        current["text"] = _tokens_to_text(current.pop("_tokens", []))
        confidences = current.pop("_confidences", [])
        current["alignment_confidence"] = round(
            sum(confidences) / len(confidences) if confidences else 0.0, 4
        )
        methods = current.pop("_methods", set())
        current["alignment_method"] = (
            next(iter(methods)) if len(methods) == 1 else "mixed_word_alignment"
        )
        current["source_segment_ids"] = sorted(
            x for x in current["source_segment_ids"] if x is not None
        )
        current["turn_id"] = len(turns)
        current["start"] = round(float(current["start"]), 3)
        current["end"] = round(float(current["end"]), 3)
        if current["text"]:
            turns.append(current)
        current = None

    for item in assignments:
        speaker = str(item.get("speaker") or UNKNOWN_SPEAKER)
        start, end = float(item["start"]), float(item["end"])
        gap = start - float(current["end"]) if current is not None else 0.0
        same_turn = (
            current is not None
            and speaker == current["speaker"]
            and gap <= float(max_gap_seconds)
        )
        if not same_turn:
            finish()
            current = {
                "start": start,
                "end": end,
                "speaker": speaker,
                "_tokens": [],
                "_confidences": [],
                "_methods": set(),
                "source_word_ids": [],
                "source_segment_ids": set(),
            }

        assert current is not None
        current["end"] = max(float(current["end"]), end)
        current["_tokens"].append(str(item.get("word") or ""))
        current["_confidences"].append(float(item.get("alignment_confidence", 0.0)))
        current["_methods"].add(str(item.get("alignment_method") or "unknown"))
        current["source_word_ids"].append(item.get("word_id"))
        current["source_segment_ids"].add(item.get("segment_id"))

    finish()
    return turns


def merge_adjacent_same_speaker_turns(
    turns: List[Dict[str, Any]],
    *,
    max_gap_seconds: float = 0.80,
) -> List[Dict[str, Any]]:
    if not turns:
        return []
    merged: List[Dict[str, Any]] = []
    for turn in turns:
        if not merged:
            merged.append(dict(turn))
            continue
        previous = merged[-1]
        gap = float(turn["start"]) - float(previous["end"])
        if (
            turn.get("speaker") == previous.get("speaker")
            and gap <= max_gap_seconds
            and turn.get("speaker") != UNKNOWN_SPEAKER
        ):
            previous["end"] = max(float(previous["end"]), float(turn["end"]))
            previous["text"] = _normalize_spacing(
                str(previous.get("text", "")).rstrip() + " " + str(turn.get("text", "")).lstrip()
            )
            previous["source_segment_ids"] = sorted(set(
                list(previous.get("source_segment_ids", [])) + list(turn.get("source_segment_ids", []))
            ))
            previous["source_word_ids"] = list(previous.get("source_word_ids", [])) + list(turn.get("source_word_ids", []))
            previous["alignment_confidence"] = round(
                min(
                    float(previous.get("alignment_confidence", 0.0)),
                    float(turn.get("alignment_confidence", 0.0)),
                ),
                4,
            )
            if previous.get("alignment_method") != turn.get("alignment_method"):
                previous["alignment_method"] = "mixed"
        else:
            merged.append(dict(turn))
    for idx, turn in enumerate(merged):
        turn["turn_id"] = idx
        turn["start"] = round(float(turn["start"]), 3)
        turn["end"] = round(float(turn["end"]), 3)
    return merged


def _segment_level_fallback(
    transcription: Any,
    diarization: List[Dict[str, Any]],
    *,
    nearest_fallback_seconds: float,
) -> List[Dict[str, Any]]:
    turns: List[Dict[str, Any]] = []
    for idx, segment in enumerate(sorted(_segments(transcription), key=lambda x: float(x.get("start", 0.0)))):
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        text = _normalize_spacing(str(segment.get("text", "")))
        if not text or end <= start:
            continue
        speaker, confidence, method = _best_speaker(
            start, end, diarization, nearest_fallback_seconds
        )
        turns.append(
            {
                "turn_id": len(turns),
                "start": round(start, 3),
                "end": round(end, 3),
                "speaker": speaker,
                "text": text,
                "alignment_confidence": round(float(confidence), 4),
                "alignment_method": "segment_" + method,
                "source_segment_ids": [segment.get("segment_id", idx)],
                "source_word_ids": [],
            }
        )
    return turns


def align_transcription_with_diarization(
    transcription: Any,
    diarization: Any,
    *,
    nearest_fallback_seconds: float = 0.75,
    max_word_gap_seconds: float = 1.25,
    merge_adjacent: bool = True,
) -> Dict[str, Any]:
    diar_segments = sorted(_segments(diarization), key=lambda x: float(x.get("start", 0.0)))
    if not diar_segments:
        raise ValueError("Diarization contains no segments.")

    words = _flatten_words(transcription)
    assignments: List[Dict[str, Any]] = []
    if words:
        assignments = _word_assignments(words, diar_segments, nearest_fallback_seconds)
        turns = _turns_from_words(assignments, max_gap_seconds=max_word_gap_seconds)
        granularity = "word"
    else:
        if not _segments(transcription):
            raise ValueError("Transcription contains no segments or word timestamps.")
        turns = _segment_level_fallback(
            transcription, diar_segments, nearest_fallback_seconds=nearest_fallback_seconds
        )
        granularity = "segment_fallback"

    if merge_adjacent:
        turns = merge_adjacent_same_speaker_turns(turns)

    labels = list(dict.fromkeys(
        t["speaker"] for t in turns if t.get("speaker") != UNKNOWN_SPEAKER
    ))
    low_confidence_words = sum(
        1 for x in assignments if float(x.get("alignment_confidence", 0.0)) < 0.50
    )
    return {
        "created_at": utc_now_iso(),
        "alignment_granularity": granularity,
        "turns": turns,
        "word_assignments": assignments,
        "speaker_labels": labels,
        "unmatched_turns": sum(1 for t in turns if t.get("speaker") == UNKNOWN_SPEAKER),
        "statistics": {
            "word_count": len(assignments),
            "low_confidence_word_count": low_confidence_words,
            "turn_count": len(turns),
            "diarization_segment_count": len(diar_segments),
        },
    }


def align_files(
    transcription_path: str | Path,
    diarization_path: str | Path,
    output_path: str | Path,
) -> Dict[str, Any]:
    result = align_transcription_with_diarization(
        read_json(transcription_path), read_json(diarization_path)
    )
    write_json(output_path, result)
    return result


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Align faster-whisper words with pyannote diarization")
    parser.add_argument("transcription")
    parser.add_argument("diarization")
    parser.add_argument("--output", default="aligned_transcript.json")
    args = parser.parse_args()
    print(json.dumps(align_files(args.transcription, args.diarization, args.output), indent=2))
