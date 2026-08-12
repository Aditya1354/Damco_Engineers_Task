"""Create deterministic retrieval chunks from final_call_data.json."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Sequence

from project_utils import read_json, utc_now_iso, write_json

DEFAULT_MAX_TURNS_PER_CHUNK = 8
DEFAULT_MAX_WORDS_PER_CHUNK = 700
DEFAULT_OVERLAP_TURNS = 1


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _timestamp(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    minutes, sec = divmod(seconds, 60)
    hours, minutes = divmod(int(minutes), 60)
    return f"{hours:02d}:{minutes:02d}:{sec:06.3f}"


def _turn_line(turn: Dict[str, Any]) -> str:
    speaker = _clean(turn.get("role") or turn.get("speaker_name") or turn.get("speaker") or "Speaker")
    return f"[{_timestamp(float(turn.get('start', 0.0)))}] {speaker}: {_clean(turn.get('text'))}"


def build_summary_chunk(call_data: Dict[str, Any]) -> Dict[str, Any] | None:
    summary = call_data.get("summary") or {}
    if not summary:
        return None
    lines = ["CALL SUMMARY", _clean(summary.get("summary"))]
    for label, key in (
        ("Key points", "key_points"),
        ("Decisions", "decisions"),
        ("Risks/follow-ups", "risks_or_followups"),
    ):
        values = summary.get(key) or []
        if values:
            lines.append(f"{label}: " + "; ".join(_clean(x) for x in values if _clean(x)))
    actions = summary.get("action_items") or []
    if actions:
        action_lines = []
        for item in actions:
            if isinstance(item, dict):
                owner = _clean(item.get("owner")) or "unspecified"
                due = _clean(item.get("due_date")) or "unspecified"
                action_lines.append(f"{_clean(item.get('task'))} (owner: {owner}, due: {due})")
            else:
                action_lines.append(_clean(item))
        lines.append("Action items: " + "; ".join(x for x in action_lines if x))
    text = "\n".join(x for x in lines if x)
    return {
        "chunk_id": "summary-000",
        "chunk_type": "summary",
        "start": None,
        "end": None,
        "turn_ids": [],
        "speakers": [],
        "text": text,
    }


def chunk_transcript_turns(
    transcript: Sequence[Dict[str, Any]],
    *,
    max_turns_per_chunk: int = DEFAULT_MAX_TURNS_PER_CHUNK,
    max_words_per_chunk: int = DEFAULT_MAX_WORDS_PER_CHUNK,
    overlap_turns: int = DEFAULT_OVERLAP_TURNS,
) -> List[Dict[str, Any]]:
    turns = [dict(x) for x in transcript if isinstance(x, dict) and _clean(x.get("text"))]
    chunks: List[Dict[str, Any]] = []
    i = 0
    chunk_number = 0
    while i < len(turns):
        selected: List[Dict[str, Any]] = []
        words = 0
        j = i
        while j < len(turns) and len(selected) < int(max_turns_per_chunk):
            turn = turns[j]
            turn_words = len(_clean(turn.get("text")).split())
            if selected and words + turn_words > int(max_words_per_chunk):
                break
            selected.append(turn)
            words += turn_words
            j += 1
        if not selected:
            selected = [turns[i]]
            j = i + 1

        text = "\n".join(_turn_line(t) for t in selected)
        speakers = list(dict.fromkeys(_clean(t.get("role") or t.get("speaker_name") or t.get("speaker")) for t in selected))
        chunk_number += 1
        chunks.append(
            {
                "chunk_id": f"transcript-{chunk_number:03d}",
                "chunk_type": "transcript",
                "start": round(float(selected[0].get("start", 0.0)), 3),
                "end": round(float(selected[-1].get("end", selected[-1].get("start", 0.0))), 3),
                "turn_ids": [t.get("turn_id") for t in selected],
                "speakers": [x for x in speakers if x],
                "text": text,
            }
        )

        if j >= len(turns):
            break
        next_i = j - max(0, int(overlap_turns))
        i = next_i if next_i > i else j
    return chunks


def build_call_chunks(call_data: Dict[str, Any]) -> Dict[str, Any]:
    chunks: List[Dict[str, Any]] = []
    summary_chunk = build_summary_chunk(call_data)
    if summary_chunk:
        chunks.append(summary_chunk)
    chunks.extend(chunk_transcript_turns(call_data.get("transcript", [])))
    for item in chunks:
        item["call_name"] = call_data.get("call_name")
    return {
        "created_at": utc_now_iso(),
        "call_name": call_data.get("call_name"),
        "chunk_count": len(chunks),
        "chunks": chunks,
    }


def save_call_chunks(call_data: Dict[str, Any], output_path: str | Path) -> Dict[str, Any]:
    result = build_call_chunks(call_data)
    write_json(output_path, result)
    return result


def load_chunks(path: str | Path) -> List[Dict[str, Any]]:
    data = read_json(path)
    if isinstance(data, dict):
        return list(data.get("chunks", []))
    if isinstance(data, list):
        return data
    raise TypeError("chunks.json must contain a list or a dict with a `chunks` list.")


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Build local RAG chunks for a processed call")
    parser.add_argument("final_call_data")
    parser.add_argument("--output", default="chunks.json")
    args = parser.parse_args()
    print(json.dumps(save_call_chunks(read_json(args.final_call_data), args.output), indent=2))
