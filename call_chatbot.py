"""Selected-call chatbot and draft-email interface."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import config
from groq_llm import answer_call_question, generate_email_from_call, is_email_request
from local_vector_store import retrieve_relevant_chunks
from project_utils import append_jsonl, read_json, slugify, utc_now_iso

DEFAULT_CALLS_DIR = config.CALLS_DIR
DEFAULT_TOP_K = int(os.environ.get("CALL_CHAT_TOP_K", "5"))
SAVE_CHAT_HISTORY = config.env_bool("SAVE_CHAT_HISTORY", True)
FINAL_CALL_DATA_FILENAME = "final_call_data.json"
PENDING_CALL_DATA_FILENAME = "pending_call_data.json"
CHAT_HISTORY_FILENAME = "chat_history.jsonl"


def resolve_call_dir(call_name: str, calls_dir: str | Path | None = None) -> Path:
    root = Path(calls_dir) if calls_dir is not None else Path(DEFAULT_CALLS_DIR)
    call_dir = root / slugify(call_name, fallback="call")
    if not call_dir.is_dir():
        raise FileNotFoundError(f"Call folder not found: {call_dir}")
    return call_dir


def _status_from_folder(call_dir: Path) -> str:
    final_path = call_dir / FINAL_CALL_DATA_FILENAME
    pending_path = call_dir / PENDING_CALL_DATA_FILENAME
    error_path = call_dir / "pipeline_error.json"
    status_path = call_dir / "status.json"

    explicit_status = None
    if status_path.is_file():
        try:
            explicit_status = str(read_json(status_path).get("status") or "").strip() or None
        except Exception:
            explicit_status = None

    if pending_path.is_file():
        return "needs_speaker_confirmation"
    if explicit_status == "failed" or error_path.is_file():
        return "failed"
    if explicit_status == "completed" and final_path.is_file():
        return "completed"
    if explicit_status and explicit_status != "completed":
        return explicit_status
    if final_path.is_file():
        return "completed"  # legacy completed folders without a status file
    return "incomplete"


def load_call_data(call_dir: str | Path) -> Dict[str, Any]:
    call_dir = Path(call_dir)
    status = _status_from_folder(call_dir)
    path = call_dir / FINAL_CALL_DATA_FILENAME
    if status != "completed" or not path.is_file():
        raise FileNotFoundError(
            f"Completed call data is unavailable for '{call_dir.name}' (status={status})."
        )
    return read_json(path)


def call_status_from_folder(call_dir: str | Path) -> Dict[str, Any]:
    call_dir = Path(call_dir)
    final_path = call_dir / FINAL_CALL_DATA_FILENAME
    status = _status_from_folder(call_dir)
    return {
        "call_name": call_dir.name,
        "call_dir": str(call_dir),
        "status": status,
        "has_final_data": final_path.is_file() and status == "completed",
        "needs_speaker_confirmation": status == "needs_speaker_confirmation",
    }


def list_available_calls(calls_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    root = Path(calls_dir) if calls_dir is not None else Path(DEFAULT_CALLS_DIR)
    if not root.exists():
        return []
    return [call_status_from_folder(p) for p in sorted(x for x in root.iterdir() if x.is_dir())]


def _save_history(call_dir: Path, record: Dict[str, Any]) -> None:
    if SAVE_CHAT_HISTORY:
        append_jsonl(call_dir / CHAT_HISTORY_FILENAME, record)


def ask_call(
    call_name: str,
    question: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> Dict[str, Any]:
    call_dir = resolve_call_dir(call_name)
    call_data = load_call_data(call_dir)
    chunks = retrieve_relevant_chunks(call_dir, question, top_k=top_k, min_score=min_score)
    response = answer_call_question(question, chunks, call_name=call_dir.name)
    result = {
        "type": "answer",
        "call_name": call_dir.name,
        "question": question,
        "answer": response["answer"],
        "insufficient_context": response["insufficient_context"],
        "cited_chunk_ids": response["cited_chunk_ids"],
        "retrieved_chunks": chunks,
    }
    _save_history(
        call_dir,
        {
            "created_at": utc_now_iso(),
            "mode": "qa",
            "question": question,
            "answer": response["answer"],
            "cited_chunk_ids": response["cited_chunk_ids"],
        },
    )
    return result


def generate_call_email(
    call_name: str,
    request: str,
    *,
    top_k: int = max(DEFAULT_TOP_K, 7),
) -> Dict[str, Any]:
    request = str(request or "").strip()
    if not request:
        raise ValueError("Email draft request cannot be empty.")
    call_dir = resolve_call_dir(call_name)
    call_data = load_call_data(call_dir)
    retrieval_query = (
        f"{request}\nFollow-up commitments, action items, decisions, deadlines, client requests, and next steps"
    )
    chunks = retrieve_relevant_chunks(call_dir, retrieval_query, top_k=top_k)
    email = generate_email_from_call(
        request,
        chunks,
        call_name=call_dir.name,
        call_summary=call_data.get("summary") or {},
    )
    result = {
        "type": "email_draft",
        "draft_only": True,
        "call_name": call_dir.name,
        "request": request,
        "subject": email["subject"],
        "body": email["body"],
        "retrieved_chunks": chunks,
    }
    _save_history(
        call_dir,
        {
            "created_at": utc_now_iso(),
            "mode": "email_draft",
            "request": request,
            "subject": email["subject"],
            "body": email["body"],
        },
    )
    return result


def chat_call_request(call_name: str, message: str, *, top_k: int = DEFAULT_TOP_K) -> Dict[str, Any]:
    if is_email_request(message):
        return generate_call_email(call_name, message, top_k=max(top_k, 7))
    return ask_call(call_name, message, top_k=top_k)
