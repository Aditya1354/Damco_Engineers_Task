"""Robust cosine matching between diarized call speakers and enrolled agents.

Matching uses both the robust speaker centroid and multiple clean segment/prototype
embeddings when available. This is less sensitive to one contaminated diarization turn
than comparing only two long averaged embeddings.
"""
from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from speaker_embedder import normalize_embedding

DEFAULT_MIN_SIMILARITY = float(os.environ.get("SPEAKER_MATCH_THRESHOLD", "0.50"))
DEFAULT_MIN_MARGIN = float(os.environ.get("SPEAKER_MATCH_MARGIN", "0.12"))
CENTROID_WEIGHT = float(os.environ.get("SPEAKER_MATCH_CENTROID_WEIGHT", "0.55"))


def cosine_similarity(a: Any, b: Any) -> float:
    x = normalize_embedding(a)
    y = normalize_embedding(b)
    if x.size != y.size:
        raise ValueError(f"Embedding dimension mismatch: call={x.size}, agent={y.size}")
    return float(np.dot(x, y))


def _embedding(item: Any) -> np.ndarray:
    if isinstance(item, dict) and "embedding" in item:
        return normalize_embedding(item["embedding"])
    return normalize_embedding(item)


def _matrix_from_item(item: Any, keys: Tuple[str, ...]) -> np.ndarray:
    if isinstance(item, dict):
        for key in keys:
            value = item.get(key)
            if value is None:
                continue
            matrix = np.asarray(value, dtype=np.float32)
            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)
            if matrix.ndim == 2 and matrix.shape[0] > 0:
                return np.vstack([normalize_embedding(row) for row in matrix]).astype(np.float32)
    return _embedding(item).reshape(1, -1).astype(np.float32)


def _pair_score(call_item: Any, agent_item: Any) -> Dict[str, Any]:
    call_centroid = _embedding(call_item)
    agent_centroid = _embedding(agent_item)
    if call_centroid.size != agent_centroid.size:
        raise ValueError(
            f"Embedding dimension mismatch: call={call_centroid.size}, agent={agent_centroid.size}"
        )

    centroid_similarity = float(np.dot(call_centroid, agent_centroid))
    call_windows = _matrix_from_item(call_item, ("segment_embeddings", "prototype_embeddings"))
    agent_prototypes = _matrix_from_item(agent_item, ("prototype_embeddings", "segment_embeddings"))
    if call_windows.shape[1] != agent_prototypes.shape[1]:
        raise ValueError(
            f"Prototype dimension mismatch: call={call_windows.shape[1]}, agent={agent_prototypes.shape[1]}"
        )

    matrix = np.clip(call_windows @ agent_prototypes.T, -1.0, 1.0)
    per_call_window = matrix.max(axis=1)
    # Windows already passed internal-consistency filtering. Median resists one remaining
    # bad crop while still requiring repeated evidence across the speaker's turns.
    segment_similarity = float(np.median(per_call_window))

    weight = min(1.0, max(0.0, float(CENTROID_WEIGHT)))
    combined = weight * centroid_similarity + (1.0 - weight) * segment_similarity
    return {
        "similarity": float(combined),
        "centroid_similarity": centroid_similarity,
        "segment_similarity": segment_similarity,
        "call_evidence_windows": int(call_windows.shape[0]),
        "agent_prototypes": int(agent_prototypes.shape[0]),
        "per_call_window_similarity": [round(float(x), 6) for x in per_call_window.tolist()],
    }


def build_similarity_scores(
    call_speaker_embeddings: Dict[str, Any],
    agent_embeddings: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scores: List[Dict[str, Any]] = []
    for speaker, call_item in call_speaker_embeddings.items():
        for agent_id, agent_item in agent_embeddings.items():
            try:
                details = _pair_score(call_item, agent_item)
                error = None
            except Exception as exc:
                details = {
                    "similarity": float("-inf"),
                    "centroid_similarity": None,
                    "segment_similarity": None,
                    "call_evidence_windows": 0,
                    "agent_prototypes": 0,
                    "per_call_window_similarity": [],
                }
                error = str(exc)
            score = float(details["similarity"])
            scores.append(
                {
                    "speaker": str(speaker),
                    "agent_id": str(agent_id),
                    "agent_name": str(agent_item.get("agent_name") or agent_id),
                    "similarity": None if not np.isfinite(score) else round(score, 6),
                    "centroid_similarity": (
                        round(float(details["centroid_similarity"]), 6)
                        if details["centroid_similarity"] is not None else None
                    ),
                    "segment_similarity": (
                        round(float(details["segment_similarity"]), 6)
                        if details["segment_similarity"] is not None else None
                    ),
                    "call_evidence_windows": details["call_evidence_windows"],
                    "agent_prototypes": details["agent_prototypes"],
                    "per_call_window_similarity": details["per_call_window_similarity"],
                    "error": error,
                }
            )
    scores.sort(
        key=lambda row: float(row["similarity"]) if row["similarity"] is not None else -999.0,
        reverse=True,
    )
    return scores


def match_agent_speaker(
    call_speaker_embeddings: Dict[str, Any],
    agent_embeddings: Dict[str, Dict[str, Any]],
    *,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    min_margin: float = DEFAULT_MIN_MARGIN,
    expected_agent_ids: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Return an automatic Agent match only when cross-speaker evidence is clear."""
    if expected_agent_ids:
        wanted = {str(x) for x in expected_agent_ids}
        agents = {k: v for k, v in agent_embeddings.items() if k in wanted}
    else:
        agents = dict(agent_embeddings)

    speakers = list(call_speaker_embeddings)
    thresholds = {
        "min_similarity": float(min_similarity),
        "min_margin": float(min_margin),
        "centroid_weight": float(CENTROID_WEIGHT),
    }
    if not agents:
        return {
            "status": "manual_confirmation_required",
            "reason": "no_enrolled_agent_embeddings",
            "available_speakers": speakers,
            "scores": [],
            "thresholds": thresholds,
        }
    if len(speakers) < 2:
        return {
            "status": "manual_confirmation_required",
            "reason": "fewer_than_two_usable_call_speaker_embeddings",
            "available_speakers": speakers,
            "scores": build_similarity_scores(call_speaker_embeddings, agents),
            "thresholds": thresholds,
        }

    scores = build_similarity_scores(call_speaker_embeddings, agents)
    valid_scores = [row for row in scores if row["similarity"] is not None]
    if not valid_scores:
        return {
            "status": "manual_confirmation_required",
            "reason": "all_similarity_comparisons_failed",
            "available_speakers": speakers,
            "scores": scores,
            "thresholds": thresholds,
        }

    best = valid_scores[0]
    best_similarity = float(best["similarity"])
    speaker = best["speaker"]
    agent_id = best["agent_id"]

    competing_speakers = [
        float(row["similarity"])
        for row in valid_scores
        if row["agent_id"] == agent_id and row["speaker"] != speaker
    ]
    second_speaker_score = max(competing_speakers) if competing_speakers else None
    speaker_margin = (
        best_similarity - second_speaker_score if second_speaker_score is not None else None
    )

    competing_agents = [
        float(row["similarity"])
        for row in valid_scores
        if row["speaker"] == speaker and row["agent_id"] != agent_id
    ]
    second_agent_score = max(competing_agents) if competing_agents else None
    agent_margin = best_similarity - second_agent_score if second_agent_score is not None else None

    checks = {
        "similarity_ok": best_similarity >= float(min_similarity),
        "speaker_margin_ok": speaker_margin is not None and speaker_margin >= float(min_margin),
        "agent_margin_ok": (
            agent_margin is None if len(agents) == 1
            else agent_margin is not None and agent_margin >= float(min_margin)
        ),
    }
    identified = all(checks.values())

    result: Dict[str, Any] = {
        "status": "identified" if identified else "manual_confirmation_required",
        "reason": "high_confidence_voice_match" if identified else "voice_match_below_confidence_policy",
        "available_speakers": speakers,
        "best_match": {
            "speaker": speaker,
            "agent_id": agent_id,
            "agent_name": best["agent_name"],
            "similarity": round(best_similarity, 6),
            "centroid_similarity": best.get("centroid_similarity"),
            "segment_similarity": best.get("segment_similarity"),
            "call_evidence_windows": best.get("call_evidence_windows"),
            "agent_prototypes": best.get("agent_prototypes"),
            "speaker_margin": round(speaker_margin, 6) if speaker_margin is not None else None,
            "agent_margin": round(agent_margin, 6) if agent_margin is not None else None,
        },
        "checks": checks,
        "scores": scores,
        "thresholds": thresholds,
    }
    if identified:
        result["agent_speaker"] = speaker
        result["agent_id"] = agent_id
        result["agent_name"] = best["agent_name"]
    return result
