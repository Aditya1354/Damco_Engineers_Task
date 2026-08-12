"""Local agent voice enrollment with transactional folder creation.

Each agent stores a robust centroid embedding plus multiple enrollment prototypes. Older
profiles containing only ``embedding.npy`` remain loadable for backward compatibility.
"""
from __future__ import annotations

import argparse
import logging
import os
import re
import shutil
import uuid
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

import config
from audio_utils import convert_to_mono_16k_wav
from project_utils import read_json, slugify, utc_now_iso, write_json
from speaker_embedder import (
    DEFAULT_MODEL,
    extract_reference_profile,
    normalize_embedding,
)

log = logging.getLogger("agent-enrollment")

DEFAULT_AGENTS_DIR = config.AGENTS_DIR
PROFILE_FILENAME = "profile.json"
EMBEDDING_FILENAME = "embedding.npy"
PROTOTYPES_FILENAME = "prototype_embeddings.npy"
NORMALIZED_SAMPLE_FILENAME = "sample_16k.wav"

# Microphone enrollment uses a fixed phrase so we can verify that the recording
# actually contains the requested speech before generating/storing the voice profile.
DEFAULT_ENROLLMENT_PHRASE = (
    os.environ.get(
        "AGENT_ENROLLMENT_PHRASE",
        "accounts refers to the systematic recording analyzing and reporting of financial transactions of a business or individual it helps track income expenses assets and liabilities to understand financial health tax credits are usually more valuable because they reduce your tax bill directly while tax deductions only lower the amount of income you're taxed on")
    or ""
).strip()
DEFAULT_PHRASE_MATCH_THRESHOLD = float(
    os.environ.get("AGENT_PHRASE_MATCH_THRESHOLD", "0.8")
)


def _normalize_phrase_text(text: str) -> str:
    """Normalize ASR/reference text before phrase comparison."""
    value = str(text or "").lower().replace("’", "'")
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return " ".join(value.split()).strip()


def _phrase_similarity(expected: str, recognized: str) -> float:
    """Return a 0..1 similarity score robust to punctuation/casing differences.

    The score combines token-sequence and normalized-character similarity. This is
    intentionally a phrase-content/liveness check, not biometric verification; the
    actual voice identity is still represented by WeSpeaker embeddings afterwards.
    """
    expected_norm = _normalize_phrase_text(expected)
    recognized_norm = _normalize_phrase_text(recognized)
    if not expected_norm or not recognized_norm:
        return 0.0

    token_score = SequenceMatcher(
        None, expected_norm.split(), recognized_norm.split()
    ).ratio()
    char_score = SequenceMatcher(None, expected_norm, recognized_norm).ratio()
    return max(0.0, min(1.0, (0.70 * token_score) + (0.30 * char_score)))


def verify_enrollment_phrase(
    audio_path: str | Path,
    *,
    expected_phrase: str = DEFAULT_ENROLLMENT_PHRASE,
    min_similarity: float = DEFAULT_PHRASE_MATCH_THRESHOLD,
) -> Dict[str, Any]:
    """Transcribe a microphone recording and verify the required phrase.

    Groq Whisper is used through the project's existing ``transcribe_audio`` API.
    Enrollment is allowed only when similarity is at least ``min_similarity``.
    """
    source = Path(audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Enrollment recording not found: {source}")

    threshold = float(min_similarity)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Phrase similarity threshold must be between 0.0 and 1.0.")

    phrase = str(expected_phrase or "").strip()
    if not phrase:
        raise ValueError("Enrollment phrase cannot be empty.")

    from audio_transcription import transcribe_audio

    # Do not pass the expected phrase as a Whisper prompt here. Prompting with the
    # answer we are trying to verify can bias ASR toward that text and make the
    # validation less meaningful.
    transcription = transcribe_audio(
        source,
        word_timestamps=False,
        prompt=None,
    )
    recognized = str(transcription.get("text") or "").strip()
    if not recognized:
        recognized = " ".join(
            str(segment.get("text") or "").strip()
            for segment in (transcription.get("segments") or [])
            if str(segment.get("text") or "").strip()
        ).strip()

    score = _phrase_similarity(phrase, recognized)
    return {
        "expected_phrase": phrase,
        "recognized_text": recognized,
        "similarity": round(score, 6),
        "similarity_percent": round(score * 100.0, 2),
        "threshold": threshold,
        "threshold_percent": round(threshold * 100.0, 2),
        "passed": bool(score >= threshold),
        "provider": transcription.get("provider"),
        "model": transcription.get("model"),
        "language": transcription.get("language"),
    }


def get_agent_dir(agent_id: str, agents_dir: str | Path = DEFAULT_AGENTS_DIR) -> Path:
    return Path(agents_dir) / slugify(agent_id, fallback="agent")


def _sample_files(agent_dir: Path) -> List[Path]:
    return sorted(
        p for p in agent_dir.glob("sample*")
        if p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}
    )


def _load_prototypes(path: Path, fallback_embedding: np.ndarray) -> np.ndarray:
    if not path.is_file():
        return fallback_embedding.reshape(1, -1).astype(np.float32)
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2 or matrix.shape[0] < 1:
        raise ValueError(f"Invalid prototype embedding matrix shape: {matrix.shape}")
    normalized = np.vstack([normalize_embedding(row) for row in matrix]).astype(np.float32)
    if normalized.shape[1] != fallback_embedding.size:
        raise ValueError(
            f"prototype embedding dimension={normalized.shape[1]} does not match "
            f"embedding.npy dimension={fallback_embedding.size}."
        )
    return normalized


def validate_agent_folder(
    agent_dir: str | Path,
    *,
    expected_agent_id: str | None = None,
) -> Dict[str, Any]:
    path = Path(agent_dir)
    missing: List[str] = []
    profile_path = path / PROFILE_FILENAME
    embedding_path = path / EMBEDDING_FILENAME
    prototypes_path = path / PROTOTYPES_FILENAME
    samples = _sample_files(path) if path.is_dir() else []
    if not profile_path.is_file():
        missing.append(PROFILE_FILENAME)
    if not embedding_path.is_file():
        missing.append(EMBEDDING_FILENAME)
    if not samples:
        missing.append("sample audio")

    valid = not missing
    profile: Dict[str, Any] = {}
    dimension = None
    prototype_count = 0
    error = None
    compatible_with_current_model = False
    if valid:
        try:
            profile = read_json(profile_path)
            if not isinstance(profile, dict):
                raise ValueError("profile.json must contain a JSON object.")
            embedding = normalize_embedding(np.load(embedding_path, allow_pickle=False))
            dimension = int(embedding.size)
            prototypes = _load_prototypes(prototypes_path, embedding)
            prototype_count = int(prototypes.shape[0])
            profile_agent_id = slugify(str(profile.get("agent_id") or path.name), fallback="agent")
            expected_id = (
                slugify(expected_agent_id, fallback="agent")
                if expected_agent_id is not None else path.name
            )
            if profile_agent_id != expected_id:
                location = (
                    f"expected agent_id '{expected_id}'"
                    if expected_agent_id is not None else f"folder '{path.name}'"
                )
                raise ValueError(
                    f"profile agent_id '{profile_agent_id}' does not match {location}."
                )
            declared_dimension = profile.get("embedding_dimension")
            if declared_dimension is not None and int(declared_dimension) != dimension:
                raise ValueError(
                    f"profile embedding_dimension={declared_dimension} does not match "
                    f"embedding.npy dimension={dimension}."
                )
            declared_count = profile.get("prototype_count")
            if declared_count is not None and int(declared_count) != prototype_count:
                raise ValueError(
                    f"profile prototype_count={declared_count} does not match stored "
                    f"prototype count={prototype_count}."
                )
            compatible_with_current_model = str(profile.get("embedding_model") or "") == str(DEFAULT_MODEL)
        except Exception as exc:
            valid = False
            error = str(exc)
    return {
        "valid": valid,
        "path": str(path),
        "missing": missing,
        "error": error,
        "profile": profile,
        "embedding_dimension": dimension,
        "prototype_count": prototype_count,
        "embedding_model": profile.get("embedding_model") if isinstance(profile, dict) else None,
        "compatible_with_current_model": bool(valid and compatible_with_current_model),
        "sample_files": [str(p) for p in samples],
    }


def _copy_original_sample(src: Path, dst_dir: Path) -> Path:
    suffix = src.suffix.lower() or ".audio"
    dst = dst_dir / f"sample_original{suffix}"
    shutil.copy2(src, dst)
    return dst


def enroll_agent(
    agent_id: str,
    agent_name: str,
    sample_audio_path: str | Path,
    *,
    agents_dir: str | Path = DEFAULT_AGENTS_DIR,
    overwrite: bool = False,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Enroll an agent atomically; failed enrollments leave no partial final folder."""
    source = Path(sample_audio_path)
    if not source.is_file():
        raise FileNotFoundError(f"Agent sample audio not found: {source}")

    safe_id = slugify(agent_id, fallback="agent")
    display_name = str(agent_name or safe_id).strip() or safe_id
    root = Path(agents_dir)
    root.mkdir(parents=True, exist_ok=True)
    final_dir = root / safe_id

    if final_dir.exists() and not overwrite:
        validation = validate_agent_folder(final_dir)
        if validation["valid"]:
            raise FileExistsError(f"Agent '{safe_id}' already exists. Enable overwrite to replace it.")
        raise FileExistsError(
            f"Agent folder '{safe_id}' exists but is incomplete. Clean it up first or enable overwrite."
        )

    temp_dir = root / f".__enroll_{safe_id}_{uuid.uuid4().hex[:10]}"
    backup_dir: Path | None = None
    temp_dir.mkdir(parents=False, exist_ok=False)
    try:
        original_copy = _copy_original_sample(source, temp_dir)
        normalized = temp_dir / NORMALIZED_SAMPLE_FILENAME
        audio_info = convert_to_mono_16k_wav(original_copy, normalized)

        reference = extract_reference_profile(normalized)
        embedding = normalize_embedding(reference["centroid"])
        prototypes = np.asarray(reference["prototypes"], dtype=np.float32)
        if prototypes.ndim == 1:
            prototypes = prototypes.reshape(1, -1)
        np.save(temp_dir / EMBEDDING_FILENAME, embedding, allow_pickle=False)
        np.save(temp_dir / PROTOTYPES_FILENAME, prototypes, allow_pickle=False)

        profile = {
            "agent_id": safe_id,
            "agent_name": display_name,
            "created_at": utc_now_iso(),
            "embedding_model": DEFAULT_MODEL,
            "embedding_dimension": int(embedding.size),
            "prototype_count": int(prototypes.shape[0]),
            "prototype_file": PROTOTYPES_FILENAME,
            "enrollment_windows": reference.get("windows", []),
            "sample_original": original_copy.name,
            "sample_normalized": normalized.name,
            "audio": audio_info,
            "metadata": dict(metadata or {}),
        }
        write_json(temp_dir / PROFILE_FILENAME, profile)

        validation = validate_agent_folder(temp_dir, expected_agent_id=safe_id)
        if not validation["valid"]:
            raise RuntimeError(f"Enrollment validation failed: {validation}")

        if final_dir.exists():
            if not overwrite:
                raise FileExistsError(f"Agent '{safe_id}' already exists.")
            backup_dir = root / f".__backup_{safe_id}_{uuid.uuid4().hex[:10]}"
            final_dir.rename(backup_dir)
        try:
            temp_dir.rename(final_dir)
        except Exception:
            if backup_dir and backup_dir.exists() and not final_dir.exists():
                backup_dir.rename(final_dir)
            raise
        if backup_dir and backup_dir.exists():
            shutil.rmtree(backup_dir, ignore_errors=True)

        return {
            "status": "enrolled",
            "agent_id": safe_id,
            "agent_name": display_name,
            "agent_dir": str(final_dir),
            "profile_path": str(final_dir / PROFILE_FILENAME),
            "embedding_path": str(final_dir / EMBEDDING_FILENAME),
            "prototypes_path": str(final_dir / PROTOTYPES_FILENAME),
            "prototype_count": int(prototypes.shape[0]),
            "embedding_dimension": int(embedding.size),
            "sample_path": str(final_dir / original_copy.name),
            "normalized_sample_path": str(final_dir / normalized.name),
        }
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if backup_dir and backup_dir.exists() and not final_dir.exists():
            backup_dir.rename(final_dir)
        raise


def load_agent_embeddings(
    agents_dir: str | Path = DEFAULT_AGENTS_DIR,
    agent_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    root = Path(agents_dir)
    wanted = {slugify(x, fallback="agent") for x in agent_ids} if agent_ids else None
    result: Dict[str, Dict[str, Any]] = {}
    if not root.exists():
        return result

    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".__")):
        if wanted is not None and folder.name not in wanted:
            continue
        validation = validate_agent_folder(folder)
        if not validation["valid"]:
            log.warning("Skipping incomplete agent folder %s: %s", folder, validation["missing"] or validation["error"])
            continue
        if not validation["compatible_with_current_model"]:
            log.warning(
                "Skipping agent folder %s because embedding model %r does not match current model %r. Re-enroll this agent.",
                folder, validation.get("embedding_model"), DEFAULT_MODEL,
            )
            continue
        profile = validation["profile"]
        embedding = normalize_embedding(np.load(folder / EMBEDDING_FILENAME, allow_pickle=False))
        prototypes = _load_prototypes(folder / PROTOTYPES_FILENAME, embedding)
        agent_id = str(profile.get("agent_id") or folder.name)
        result[agent_id] = {
            "agent_id": agent_id,
            "agent_name": str(profile.get("agent_name") or agent_id),
            "profile": profile,
            "embedding": embedding,
            "prototype_embeddings": prototypes,
            "embedding_path": str(folder / EMBEDDING_FILENAME),
            "prototypes_path": str(folder / PROTOTYPES_FILENAME) if (folder / PROTOTYPES_FILENAME).is_file() else None,
            "embedding_dimension": int(embedding.size),
            "prototype_count": int(prototypes.shape[0]),
            "agent_dir": str(folder),
        }
    return result


def list_incomplete_agent_folders(agents_dir: str | Path = DEFAULT_AGENTS_DIR) -> List[Dict[str, Any]]:
    root = Path(agents_dir)
    if not root.exists():
        return []
    rows = []
    for folder in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.startswith(".__")):
        validation = validate_agent_folder(folder)
        if not validation["valid"]:
            rows.append({
                "agent_id": folder.name,
                "agent_dir": str(folder),
                "missing": validation["missing"],
                "error": validation["error"],
            })
    return rows


def cleanup_incomplete_agent_folders(agents_dir: str | Path = DEFAULT_AGENTS_DIR) -> List[str]:
    removed: List[str] = []
    root = Path(agents_dir)
    if not root.exists():
        return removed
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        should_remove = folder.name.startswith(".__") or not validate_agent_folder(folder)["valid"]
        if should_remove:
            shutil.rmtree(folder, ignore_errors=True)
            removed.append(str(folder))
    return removed


def list_agents(agents_dir: str | Path = DEFAULT_AGENTS_DIR) -> List[Dict[str, Any]]:
    agents = load_agent_embeddings(agents_dir)
    return [
        {
            "agent_id": item["agent_id"],
            "agent_name": item["agent_name"],
            "agent_dir": item["agent_dir"],
            "embedding_dimension": item["embedding_dimension"],
            "prototype_count": item.get("prototype_count", 1),
            "embedding_model": item["profile"].get("embedding_model"),
            "created_at": item["profile"].get("created_at"),
            "sample_path": str(Path(item["agent_dir"]) / item["profile"].get("sample_original", "")),
        }
        for item in agents.values()
    ]


def get_agent_profile(agent_id: str, agents_dir: str | Path = DEFAULT_AGENTS_DIR) -> Dict[str, Any]:
    folder = get_agent_dir(agent_id, agents_dir)
    validation = validate_agent_folder(folder)
    if not validation["valid"]:
        raise FileNotFoundError(f"Valid agent profile not found for '{agent_id}': {validation}")
    return validation["profile"]


def delete_agent(agent_id: str, agents_dir: str | Path = DEFAULT_AGENTS_DIR) -> bool:
    folder = get_agent_dir(agent_id, agents_dir)
    if not folder.exists():
        return False
    shutil.rmtree(folder)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Local agent enrollment")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("enroll")
    p.add_argument("agent_id")
    p.add_argument("agent_name")
    p.add_argument("sample")
    p.add_argument("--overwrite", action="store_true")
    sub.add_parser("list")
    sub.add_parser("cleanup")
    d = sub.add_parser("delete")
    d.add_argument("agent_id")
    args = parser.parse_args()
    if args.command == "enroll":
        result = enroll_agent(args.agent_id, args.agent_name, args.sample, overwrite=args.overwrite)
    elif args.command == "list":
        result = list_agents()
    elif args.command == "cleanup":
        result = cleanup_incomplete_agent_folders()
    else:
        result = {"deleted": delete_agent(args.agent_id)}
    import json
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
