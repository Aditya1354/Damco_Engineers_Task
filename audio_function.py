"""End-to-end local call processing orchestrator.

Pipeline:
local audio -> mono 16 kHz WAV -> faster-whisper -> pyannote Community-1 ->
overlap alignment -> local speaker embeddings -> agent match/manual confirmation ->
Groq summary -> chunks -> local FAISS -> final_call_data.json.
"""
from __future__ import annotations

import shutil
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import config
from agent_enrollment import load_agent_embeddings
from audio_transcription import transcribe_audio
from audio_utils import concatenate_wav_intervals, convert_to_mono_16k_wav
from groq_llm import summarize_call
from local_vector_store import build_vector_index_for_call
from project_utils import read_json, safe_unlink, slugify, utc_now_iso, write_json
from pyannote_diarizer import diarize_audio
from rag_chunker import save_call_chunks
from speaker_embedder import create_call_speaker_embeddings, json_safe_embedding_results
from speaker_matcher import DEFAULT_MIN_MARGIN, DEFAULT_MIN_SIMILARITY, match_agent_speaker
from transcript_aligner import align_transcription_with_diarization

CALLS_DIR = config.CALLS_DIR
AGENTS_DIR = config.AGENTS_DIR
DEFAULT_NUM_SPEAKERS = config.env_int("DEFAULT_NUM_SPEAKERS", 2)

FILENAMES = {
    "metadata": "call_metadata.json",
    "status": "status.json",
    "transcription": "transcription.json",
    "diarization": "diarization.json",
    "rttm": "diarization.rttm",
    "aligned": "aligned_transcript.json",
    "speaker_resolution": "speaker_resolution.json",
    "pending": "pending_call_data.json",
    "summary": "summary.json",
    "chunks": "chunks.json",
    "final": "final_call_data.json",
    "error": "pipeline_error.json",
}


def _call_dir(call_name: str) -> Path:
    return CALLS_DIR / slugify(call_name, fallback="call")


def _paths(call_dir: Path) -> Dict[str, Path]:
    result = {key: call_dir / name for key, name in FILENAMES.items()}
    result["audio_wav"] = call_dir / "audio_16k.wav"
    result["speaker_embeddings"] = call_dir / "speaker_embeddings"
    result["speaker_previews"] = call_dir / "speaker_previews"
    return result


def create_speaker_previews(
    audio_wav: str | Path,
    diarization: Dict[str, Any],
    output_dir: str | Path,
    *,
    max_segments_per_speaker: int = 3,
    max_total_seconds: float = 15.0,
) -> Dict[str, str]:
    """Create short listenable WAV previews for manual speaker confirmation."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    by_speaker: Dict[str, List[Dict[str, Any]]] = {}
    for segment in diarization.get("segments", []) or []:
        speaker = str(segment.get("speaker") or "").strip()
        if not speaker:
            continue
        by_speaker.setdefault(speaker, []).append(dict(segment))

    previews: Dict[str, str] = {}
    for speaker, segments in by_speaker.items():
        segments.sort(key=lambda x: float(x.get("end", 0.0)) - float(x.get("start", 0.0)), reverse=True)
        intervals: List[tuple[float, float]] = []
        for segment in segments[: max(1, int(max_segments_per_speaker))]:
            start, end = float(segment.get("start", 0.0)), float(segment.get("end", 0.0))
            duration = end - start
            trim = min(0.20, max(0.0, (duration - 0.75) / 2.0))
            if end - start > 0.25:
                intervals.append((start + trim, end - trim))
        if not intervals:
            continue
        preview_path = output_dir / f"{slugify(speaker, fallback='speaker')}.wav"
        try:
            concatenate_wav_intervals(
                audio_wav, intervals, preview_path, max_total_seconds=max_total_seconds
            )
        except Exception:
            continue
        previews[speaker] = str(preview_path)
    return previews


def _set_status(call_dir: Path, status: str, **extra: Any) -> Dict[str, Any]:
    payload = {"status": status, "updated_at": utc_now_iso(), **extra}
    write_json(_paths(call_dir)["status"], payload)
    return payload


def prepare_call_dir(call_name: str, *, overwrite: bool = False) -> Path:
    call_dir = _call_dir(call_name)
    if call_dir.exists() and any(call_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Call folder already exists: {call_dir}. Use overwrite=True to replace it."
            )
        shutil.rmtree(call_dir)
    call_dir.mkdir(parents=True, exist_ok=True)
    return call_dir


def copy_original_audio(audio_path: str | Path, call_dir: Path) -> Path:
    src = Path(audio_path)
    if not src.is_file():
        raise FileNotFoundError(f"Input call audio not found: {src}")
    suffix = src.suffix.lower() or ".audio"
    dst = call_dir / f"original_audio{suffix}"
    shutil.copy2(src, dst)
    return dst


def _client_name(client_info: Optional[Dict[str, Any]]) -> str:
    info = client_info or {}
    return str(
        info.get("client_name")
        or info.get("name")
        or info.get("clientName")
        or "Client"
    ).strip() or "Client"


def _agent_name(agent_info: Optional[Dict[str, Any]]) -> str:
    info = agent_info or {}
    return str(
        info.get("agent_name")
        or info.get("name")
        or info.get("agentName")
        or "Agent"
    ).strip() or "Agent"


def _speaker_labels(aligned: Dict[str, Any]) -> List[str]:
    labels: List[str] = []
    for turn in aligned.get("turns", []):
        label = str(turn.get("speaker") or "").strip()
        if label and label != "UNKNOWN" and label not in labels:
            labels.append(label)
    return labels


def infer_client_speaker(
    aligned_turns: Sequence[Dict[str, Any]],
    speakers: Sequence[str],
    *,
    agent_speaker: str,
) -> str | None:
    """Infer the primary Client among non-Agent speakers.

    This only matters when diarization returns >2 acoustic speakers (for example an IVR
    prompt + Agent + Client). The human Client usually alternates with the Agent, while
    an IVR tends to occur in an isolated block near the start.
    """
    candidates = [speaker for speaker in speakers if speaker != agent_speaker]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    ordered = [
        turn for turn in sorted(aligned_turns, key=lambda x: float(x.get("start", 0.0)))
        if str(turn.get("speaker") or "") in speakers
    ]
    first_agent_index = next(
        (i for i, turn in enumerate(ordered) if str(turn.get("speaker")) == agent_speaker),
        len(ordered),
    )
    scores: Dict[str, float] = {speaker: 0.0 for speaker in candidates}
    for i, turn in enumerate(ordered):
        speaker = str(turn.get("speaker") or "")
        if speaker not in scores:
            continue
        duration = max(0.0, float(turn.get("end", 0.0)) - float(turn.get("start", 0.0)))
        scores[speaker] += min(duration, 8.0) * 0.15
        if i >= first_agent_index:
            scores[speaker] += 1.0
        if i > 0 and str(ordered[i - 1].get("speaker") or "") == agent_speaker:
            scores[speaker] += 4.0
        if i + 1 < len(ordered) and str(ordered[i + 1].get("speaker") or "") == agent_speaker:
            scores[speaker] += 4.0
    return max(candidates, key=lambda speaker: scores.get(speaker, 0.0))


def build_role_mapping(
    speakers: Sequence[str],
    *,
    agent_speaker: str,
    agent_name: str,
    client_name: str,
    aligned_turns: Sequence[Dict[str, Any]] | None = None,
) -> Dict[str, Dict[str, str]]:
    if agent_speaker not in speakers:
        raise ValueError(f"Agent speaker '{agent_speaker}' is not one of {list(speakers)}")
    client_speaker = infer_client_speaker(
        aligned_turns or [], speakers, agent_speaker=agent_speaker
    )
    mapping: Dict[str, Dict[str, str]] = {}
    for speaker in speakers:
        if speaker == agent_speaker:
            mapping[speaker] = {"role": "Agent", "speaker_name": agent_name}
        elif speaker == client_speaker or len(speakers) == 2:
            mapping[speaker] = {"role": "Client", "speaker_name": client_name}
        else:
            mapping[speaker] = {"role": "Other", "speaker_name": speaker}
    return mapping


def apply_role_mapping(
    aligned_turns: Sequence[Dict[str, Any]],
    mapping: Dict[str, Dict[str, str]],
) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for idx, turn in enumerate(aligned_turns):
        item = dict(turn)
        speaker = str(item.get("speaker") or "UNKNOWN")
        mapped = mapping.get(speaker, {"role": "Unknown", "speaker_name": speaker})
        item["turn_id"] = idx
        item["role"] = mapped["role"]
        item["speaker_name"] = mapped["speaker_name"]
        result.append(item)
    return result


def _find_agent_info(agent_id: Optional[str], agents: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    if agent_id and agent_id in agents:
        item = agents[agent_id]
        return {
            "agent_id": item["agent_id"],
            "agent_name": item["agent_name"],
            "profile": item.get("profile", {}),
        }
    return {"agent_id": agent_id, "agent_name": "Agent", "profile": {}}


def finalize_call_with_mapping(
    call_dir: str | Path,
    *,
    role_mapping: Dict[str, Dict[str, str]],
    speaker_resolution: Dict[str, Any],
    client_info: Optional[Dict[str, Any]] = None,
    agent_info: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    call_dir = Path(call_dir)
    paths = _paths(call_dir)
    metadata = read_json(paths["metadata"])
    aligned = read_json(paths["aligned"])
    transcript = apply_role_mapping(aligned.get("turns", []), role_mapping)

    call_data: Dict[str, Any] = {
        "schema_version": 1,
        "call_id": metadata["call_id"],
        "call_name": metadata["call_name"],
        "created_at": metadata["created_at"],
        "finalized_at": utc_now_iso(),
        "audio": metadata.get("audio", {}),
        "models": metadata.get("models", {}),
        "client_info": dict(client_info or metadata.get("client_info") or {}),
        "agent_info": dict(agent_info or metadata.get("agent_info") or {}),
        "speaker_identification": speaker_resolution,
        "role_mapping": role_mapping,
        "transcript": transcript,
    }

    summary = summarize_call(call_data)
    write_json(paths["summary"], summary)
    call_data["summary"] = summary

    # Do not publish final_call_data.json until every downstream artifact succeeds.
    # This prevents a failed vector build (or other downstream step) from leaving a
    # seemingly completed call that the browser/chatbot could accidentally expose.
    chunks_result = save_call_chunks(call_data, paths["chunks"])
    vector_metadata = build_vector_index_for_call(call_dir)
    call_data["artifacts"] = {
        "transcription": FILENAMES["transcription"],
        "diarization": FILENAMES["diarization"],
        "diarization_rttm": FILENAMES["rttm"],
        "aligned_transcript": FILENAMES["aligned"],
        "speaker_resolution": FILENAMES["speaker_resolution"],
        "summary": FILENAMES["summary"],
        "chunks": FILENAMES["chunks"],
        "vector_index": "vector_index.faiss",
        "vector_metadata": "vector_metadata.json",
    }
    call_data["rag"] = {
        "chunk_count": chunks_result.get("chunk_count", 0),
        "vector": vector_metadata,
    }
    write_json(paths["final"], call_data)
    safe_unlink(paths["pending"])
    safe_unlink(paths["error"])
    _set_status(call_dir, "completed", final_call_data=str(paths["final"]))
    return call_data


def process_audio_pipeline(
    audio_path: str | Path,
    *,
    call_name: Optional[str] = None,
    expected_agent_ids: Optional[Sequence[str]] = None,
    client_info: Optional[Dict[str, Any]] = None,
    num_speakers: int | None = DEFAULT_NUM_SPEAKERS,
    min_speakers: int | None = 2,
    max_speakers: int | None = 3,
    overwrite: bool = False,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> Dict[str, Any]:
    """Run the local pipeline until completion or manual speaker confirmation."""
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Call audio not found: {source}")
    safe_call_name = slugify(call_name or source.stem, fallback="call")
    call_dir = prepare_call_dir(safe_call_name, overwrite=overwrite)
    paths = _paths(call_dir)
    stage = "initializing"
    started = time.perf_counter()

    try:
        _set_status(call_dir, "processing", stage=stage)
        stage = "copy_original_audio"
        original = copy_original_audio(source, call_dir)

        stage = "normalize_audio"
        _set_status(call_dir, "processing", stage=stage)
        audio_info = convert_to_mono_16k_wav(original, paths["audio_wav"])

        metadata: Dict[str, Any] = {
            "schema_version": 1,
            "call_id": safe_call_name,
            "call_name": safe_call_name,
            "created_at": utc_now_iso(),
            "source_audio": str(source),
            "audio": {
                "original": original.name,
                "normalized": paths["audio_wav"].name,
                **audio_info,
            },
            "client_info": dict(client_info or {}),
            "expected_agent_ids": list(expected_agent_ids or []),
            "diarization_settings": {
                "num_speakers": num_speakers,
                "min_speakers": min_speakers if num_speakers is None else None,
                "max_speakers": max_speakers if num_speakers is None else None,
            },
            "models": {
                "whisper": config.masked_status()["whisper_model"],
                "diarization": config.masked_status()["pyannote_model"],
                "speaker_embedding": config.masked_status()["speaker_embedding_model"],
                "text_embedding": config.masked_status()["text_embedding_model"],
                "groq": config.masked_status()["groq_model"],
            },
        }
        write_json(paths["metadata"], metadata)

        stage = "transcription"
        _set_status(call_dir, "processing", stage=stage)
        transcription = transcribe_audio(paths["audio_wav"], output_path=paths["transcription"])
        if not transcription.get("segments"):
            raise RuntimeError("faster-whisper produced no transcription segments.")

        stage = "diarization"
        _set_status(call_dir, "processing", stage=stage)
        diarization = diarize_audio(
            paths["audio_wav"],
            output_path=paths["diarization"],
            rttm_path=paths["rttm"],
            num_speakers=int(num_speakers) if num_speakers is not None else None,
            min_speakers=int(min_speakers) if num_speakers is None and min_speakers is not None else None,
            max_speakers=int(max_speakers) if num_speakers is None and max_speakers is not None else None,
            uri=safe_call_name,
        )
        if not diarization.get("segments"):
            raise RuntimeError("pyannote produced no diarization segments.")
        speaker_previews = create_speaker_previews(
            paths["audio_wav"], diarization, paths["speaker_previews"]
        )

        stage = "alignment"
        _set_status(call_dir, "processing", stage=stage)
        aligned = align_transcription_with_diarization(transcription, diarization)
        write_json(paths["aligned"], aligned)
        aligned_speakers = set(_speaker_labels(aligned))
        diarized_speakers = {
            str(x).strip() for x in diarization.get("speakers", []) if str(x).strip()
        }
        if not diarized_speakers:
            diarized_speakers = {
                str(seg.get("speaker", "")).strip()
                for seg in diarization.get("segments", [])
                if str(seg.get("speaker", "")).strip()
            }
        speakers = sorted(aligned_speakers | diarized_speakers)
        expected_minimum = int(num_speakers) if num_speakers is not None else int(min_speakers or 1)
        if len(speakers) < 2 and expected_minimum >= 2:
            # Still continue to manual resolution state; this is more diagnosable than
            # failing before the user can inspect artifacts.
            pass

        stage = "speaker_embeddings"
        _set_status(call_dir, "processing", stage=stage)
        embedding_error: str | None = None
        try:
            call_embeddings = create_call_speaker_embeddings(
                paths["audio_wav"],
                diarization,
                paths["speaker_embeddings"],
            )
            write_json(
                paths["speaker_embeddings"] / "speaker_embeddings.json",
                json_safe_embedding_results(call_embeddings),
            )
        except Exception as exc:
            # Voice matching is an optional confidence layer. If diarization/alignment
            # succeeded but embedding extraction is unavailable (e.g. only very short
            # turns, incompatible speaker model, or a device-specific inference error),
            # preserve the usable call artifacts and route to manual confirmation.
            call_embeddings = {}
            embedding_error = str(exc)
            write_json(
                paths["speaker_embeddings"] / "speaker_embeddings.json",
                {
                    "status": "unavailable",
                    "error": embedding_error,
                    "available_diarized_speakers": speakers,
                },
            )

        stage = "speaker_matching"
        _set_status(call_dir, "processing", stage=stage)
        all_agents = load_agent_embeddings(AGENTS_DIR)
        expected = [slugify(x, fallback="agent") for x in (expected_agent_ids or [])]
        if expected:
            missing_expected = [agent_id for agent_id in expected if agent_id not in all_agents]
            if missing_expected:
                raise ValueError(
                    "Expected agent IDs are not valid enrolled agents: " + ", ".join(missing_expected)
                )
        candidate_agents = (
            {agent_id: all_agents[agent_id] for agent_id in expected}
            if expected
            else all_agents
        )
        if embedding_error is not None:
            resolution = {
                "status": "manual_confirmation_required",
                "reason": "call_speaker_embeddings_unavailable",
                "available_speakers": speakers,
                "scores": [],
                "thresholds": {
                    "min_similarity": float(min_similarity),
                    "min_margin": float(min_margin),
                },
                "speaker_embedding_error": embedding_error,
            }
        else:
            resolution = match_agent_speaker(
                call_embeddings,
                candidate_agents,
                min_similarity=min_similarity,
                min_margin=min_margin,
                expected_agent_ids=expected or None,
            )
            # The matcher reports speakers with usable embeddings; manual confirmation
            # must offer every diarized speaker even when one embedding was unusable.
            resolution["available_speakers"] = speakers
        resolution.update({"method": "automatic_voice_match", "updated_at": utc_now_iso()})
        write_json(paths["speaker_resolution"], resolution)

        if resolution["status"] == "identified":
            agent_id = resolution["agent_id"]
            agent_info = _find_agent_info(agent_id, candidate_agents)
            mapping = build_role_mapping(
                speakers,
                agent_speaker=resolution["agent_speaker"],
                agent_name=agent_info["agent_name"],
                client_name=_client_name(client_info),
                aligned_turns=aligned.get("turns", []),
            )
            final = finalize_call_with_mapping(
                call_dir,
                role_mapping=mapping,
                speaker_resolution=resolution,
                client_info=client_info,
                agent_info=agent_info,
            )
            return {
                "status": "completed",
                "call_name": safe_call_name,
                "call_dir": str(call_dir),
                "automatic_speaker_match": True,
                "final_call_data": final,
                "elapsed_seconds": round(time.perf_counter() - started, 3),
            }

        pending = {
            "status": "needs_speaker_confirmation",
            "created_at": utc_now_iso(),
            "call_name": safe_call_name,
            "call_dir": str(call_dir),
            "available_speakers": speakers,
            "client_info": dict(client_info or {}),
            "expected_agent_ids": expected,
            "candidate_agent_ids": list(candidate_agents),
            "speaker_preview_files": speaker_previews,
            "speaker_resolution": resolution,
        }
        write_json(paths["pending"], pending)
        _set_status(
            call_dir,
            "needs_speaker_confirmation",
            stage="speaker_confirmation",
            available_speakers=speakers,
        )
        return {
            **pending,
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }

    except Exception as exc:
        error = {
            "status": "failed",
            "failed_at": utc_now_iso(),
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(paths["error"], error)
        _set_status(call_dir, "failed", stage=stage, error=str(exc))
        raise


def confirm_agent_speaker(
    call_name: str,
    agent_speaker: str,
    *,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    client_name: Optional[str] = None,
    inherit_expected_agent: bool = True,
) -> Dict[str, Any]:
    """Complete a paused call after the user manually identifies the Agent speaker."""
    call_dir = _call_dir(call_name)
    paths = _paths(call_dir)
    if not paths["pending"].is_file():
        raise FileNotFoundError(
            f"Call '{call_name}' is not waiting for manual speaker confirmation."
        )
    pending = read_json(paths["pending"])
    speakers = list(pending.get("available_speakers", []))
    if agent_speaker not in speakers:
        raise ValueError(f"agent_speaker must be one of {speakers}")

    agents = load_agent_embeddings(AGENTS_DIR)
    selected_agent_id = slugify(agent_id, fallback="agent") if agent_id else None
    if selected_agent_id is None and inherit_expected_agent:
        # Only carry an identity forward when the user explicitly restricted processing
        # to exactly one expected enrolled agent. Never promote a low-confidence voice
        # match (or merely the only enrolled profile) into an identity during manual
        # speaker confirmation.
        expected = [x for x in pending.get("expected_agent_ids", []) if x in agents]
        if len(expected) == 1:
            selected_agent_id = expected[0]

    if selected_agent_id is not None and selected_agent_id not in agents:
        raise ValueError(f"Unknown or incomplete enrolled agent: {selected_agent_id}")

    if selected_agent_id:
        agent_info = _find_agent_info(selected_agent_id, agents)
    else:
        agent_info = {"agent_id": None, "agent_name": str(agent_name or "Agent"), "profile": {}}
    if agent_name:
        agent_info["agent_name"] = str(agent_name).strip() or agent_info["agent_name"]

    client_info = dict(pending.get("client_info") or {})
    if client_name:
        client_info["client_name"] = str(client_name).strip()

    mapping = build_role_mapping(
        speakers,
        agent_speaker=agent_speaker,
        agent_name=_agent_name(agent_info),
        client_name=_client_name(client_info),
        aligned_turns=read_json(paths["aligned"]).get("turns", []),
    )
    resolution = {
        "status": "identified",
        "method": "manual_confirmation",
        "updated_at": utc_now_iso(),
        "agent_speaker": agent_speaker,
        "agent_id": selected_agent_id,
        "agent_name": _agent_name(agent_info),
        "available_speakers": speakers,
        "automatic_match_before_confirmation": pending.get("speaker_resolution"),
    }
    write_json(paths["speaker_resolution"], resolution)
    stage = "finalizing_after_manual_confirmation"
    _set_status(call_dir, "processing", stage=stage)
    try:
        final = finalize_call_with_mapping(
            call_dir,
            role_mapping=mapping,
            speaker_resolution=resolution,
            client_info=client_info,
            agent_info=agent_info,
        )
    except Exception as exc:
        error = {
            "status": "failed",
            "failed_at": utc_now_iso(),
            "stage": stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "recoverable_from_pending_confirmation": True,
        }
        write_json(paths["error"], error)
        # Keep pending_call_data.json so the user can retry finalization after fixing
        # the downstream problem (for example Groq connectivity or model/index setup).
        _set_status(call_dir, "failed", stage=stage, error=str(exc))
        raise
    return {
        "status": "completed",
        "call_name": call_dir.name,
        "call_dir": str(call_dir),
        "automatic_speaker_match": False,
        "final_call_data": final,
    }


def _resolved_call_status(paths: Dict[str, Path]) -> str:
    """Resolve call status without letting stale final artifacts hide failures."""
    explicit_status = None
    if paths["status"].is_file():
        try:
            explicit_status = str(read_json(paths["status"]).get("status") or "").strip() or None
        except Exception:
            explicit_status = None

    if paths["pending"].is_file():
        return "needs_speaker_confirmation"
    if explicit_status == "failed" or paths["error"].is_file():
        return "failed"
    if explicit_status == "completed" and paths["final"].is_file():
        return "completed"
    if explicit_status and explicit_status != "completed":
        return explicit_status
    if paths["final"].is_file():
        # Backward-compatible support for completed folders created before status.json existed.
        return "completed"
    if explicit_status == "completed":
        return "incomplete"
    return "incomplete"


def list_calls() -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not CALLS_DIR.exists():
        return rows
    for folder in sorted(p for p in CALLS_DIR.iterdir() if p.is_dir()):
        paths = _paths(folder)
        status = _resolved_call_status(paths)
        row: Dict[str, Any] = {
            "call_name": folder.name,
            "status": status,
            "call_dir": str(folder),
            "has_final_data": paths["final"].is_file() and status == "completed",
            "needs_speaker_confirmation": status == "needs_speaker_confirmation",
        }
        if paths["pending"].is_file():
            pending = read_json(paths["pending"])
            row["available_speakers"] = pending.get("available_speakers", [])
        if paths["error"].is_file():
            error = read_json(paths["error"])
            row["error"] = error.get("error")
            row["failed_stage"] = error.get("stage")
        rows.append(row)
    return rows


def load_call(call_name: str) -> Dict[str, Any]:
    paths = _paths(_call_dir(call_name))
    status = _resolved_call_status(paths)
    if status != "completed" or not paths["final"].is_file():
        raise FileNotFoundError(
            f"Completed call data not available for '{call_name}' (status={status})."
        )
    return read_json(paths["final"])
