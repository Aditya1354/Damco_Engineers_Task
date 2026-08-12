"""Local sentence-transformer embeddings + FAISS retrieval for one call folder."""
from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

import config
from project_utils import read_json, sha256_file, utc_now_iso, write_json
from rag_chunker import load_chunks

log = logging.getLogger("local-vector-store")

DEFAULT_TEXT_EMBEDDING_MODEL = os.environ.get("TEXT_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
DEFAULT_TEXT_EMBEDDING_DEVICE = os.environ.get("TEXT_EMBEDDING_DEVICE", "auto").lower()
DEFAULT_LOCAL_ONLY = config.LOCAL_MODELS_ONLY or config.env_bool("TEXT_EMBEDDING_LOCAL_FILES_ONLY", False)
DEFAULT_BATCH_SIZE = int(os.environ.get("TEXT_EMBEDDING_BATCH_SIZE", "32"))
_CONFIGURED_QUERY_PREFIX = os.environ.get("TEXT_EMBEDDING_QUERY_PREFIX")
DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))
TEXT_CACHE_DIR = config.MODELS_DIR / "sentence_transformers"
INDEX_FILENAME = "vector_index.faiss"
METADATA_FILENAME = "vector_metadata.json"
CHUNKS_FILENAME = "chunks.json"


def resolve_device(device: str = "auto") -> str:
    device = str(device or "auto").strip().lower()
    if device == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("TEXT_EMBEDDING_DEVICE must be auto, cpu, cuda, or mps")
    return device


@lru_cache(maxsize=4)
def _get_model_cached(model_name: str, device: str, local_files_only: bool):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("sentence-transformers is not installed.") from exc
    try:
        return SentenceTransformer(
            model_name,
            device=device,
            cache_folder=str(TEXT_CACHE_DIR),
            local_files_only=bool(local_files_only),
        )
    except Exception as exc:
        mode = "offline/local-only" if local_files_only else "download/cached"
        raise RuntimeError(
            f"Could not load text embedding model '{model_name}' in {mode} mode. "
            "Run `python prepare_models.py` before enabling LOCAL_MODELS_ONLY=1."
        ) from exc


def get_text_embedding_model(
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    device: str = DEFAULT_TEXT_EMBEDDING_DEVICE,
    local_files_only: bool = DEFAULT_LOCAL_ONLY,
):
    return _get_model_cached(model_name, resolve_device(device), bool(local_files_only))


def query_prefix_for_model(model_name: str) -> str:
    """Return the retrieval-query instruction for the configured embedding model."""
    if _CONFIGURED_QUERY_PREFIX is not None:
        value = _CONFIGURED_QUERY_PREFIX
        if value and not value.endswith((" ", "\n", "\t")):
            value += " "
        return value
    normalized = str(model_name or "").strip().lower()
    if normalized.startswith("baai/bge-") and "-en" in normalized:
        return "Represent this sentence for searching relevant passages: "
    return ""


def normalize_embeddings(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.ndim == 1:
        matrix = matrix.reshape(1, -1)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return (matrix / norms).astype(np.float32)


def embed_texts(
    texts: Sequence[str],
    *,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    device: str = DEFAULT_TEXT_EMBEDDING_DEVICE,
    local_files_only: bool = DEFAULT_LOCAL_ONLY,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_text_embedding_model(model_name, device, local_files_only)
    matrix = model.encode(
        list(texts),
        batch_size=int(batch_size),
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return normalize_embeddings(matrix)


def embed_query(
    query: str,
    *,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    device: str = DEFAULT_TEXT_EMBEDDING_DEVICE,
    local_files_only: bool = DEFAULT_LOCAL_ONLY,
) -> np.ndarray:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    prefix = query_prefix_for_model(model_name)
    return embed_texts(
        [prefix + query],
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
    )


def _faiss():
    try:
        import faiss
        return faiss
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("faiss-cpu is not installed or could not be imported.") from exc


def _chunk_text(chunk: Dict[str, Any]) -> str:
    text = str(chunk.get("text", "")).strip()
    if not text:
        return ""
    return f"{chunk.get('chunk_type', 'call')}\n{text}"


def build_vector_index_for_call(
    call_dir: str | Path,
    *,
    model_name: str = DEFAULT_TEXT_EMBEDDING_MODEL,
    device: str = DEFAULT_TEXT_EMBEDDING_DEVICE,
    local_files_only: bool = DEFAULT_LOCAL_ONLY,
) -> Dict[str, Any]:
    call_dir = Path(call_dir)
    chunks_path = call_dir / CHUNKS_FILENAME
    if not chunks_path.is_file():
        raise FileNotFoundError(f"chunks.json not found: {chunks_path}")
    chunks = load_chunks(chunks_path)
    indexed_chunks = [chunk for chunk in chunks if _chunk_text(chunk)]
    if not indexed_chunks:
        raise ValueError("No non-empty chunks are available for indexing.")

    texts = [_chunk_text(chunk) for chunk in indexed_chunks]
    embeddings = embed_texts(
        texts,
        model_name=model_name,
        device=device,
        local_files_only=local_files_only,
    )
    if embeddings.shape[0] != len(indexed_chunks):
        raise RuntimeError("Embedding row count does not match chunk count.")

    faiss = _faiss()
    index = faiss.IndexFlatIP(int(embeddings.shape[1]))
    index.add(np.ascontiguousarray(embeddings, dtype=np.float32))
    index_path = call_dir / INDEX_FILENAME
    faiss.write_index(index, str(index_path))

    metadata = {
        "created_at": utc_now_iso(),
        "call_name": call_dir.name,
        "embedding_model": model_name,
        "embedding_device": resolve_device(device),
        "query_prefix": query_prefix_for_model(model_name),
        "dimension": int(embeddings.shape[1]),
        "vector_count": int(index.ntotal),
        "chunks_sha256": sha256_file(chunks_path),
        "chunk_ids": [str(chunk.get("chunk_id")) for chunk in indexed_chunks],
        "index_filename": INDEX_FILENAME,
        "chunks_filename": CHUNKS_FILENAME,
    }
    write_json(call_dir / METADATA_FILENAME, metadata)
    return metadata


def vector_index_is_current(call_dir: str | Path) -> bool:
    call_dir = Path(call_dir)
    metadata_path = call_dir / METADATA_FILENAME
    index_path = call_dir / INDEX_FILENAME
    chunks_path = call_dir / CHUNKS_FILENAME
    if not (metadata_path.is_file() and index_path.is_file() and chunks_path.is_file()):
        return False
    try:
        metadata = read_json(metadata_path)
        return (
            metadata.get("chunks_sha256") == sha256_file(chunks_path)
            and metadata.get("embedding_model") == DEFAULT_TEXT_EMBEDDING_MODEL
        )
    except Exception:
        return False


def ensure_vector_index(call_dir: str | Path, *, rebuild: bool = False) -> Dict[str, Any]:
    call_dir = Path(call_dir)
    if rebuild or not vector_index_is_current(call_dir):
        return build_vector_index_for_call(call_dir)
    return read_json(call_dir / METADATA_FILENAME)


def retrieve_relevant_chunks(
    call_dir: str | Path,
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float | None = None,
) -> List[Dict[str, Any]]:
    query = str(query or "").strip()
    if not query:
        raise ValueError("Query cannot be empty.")
    call_dir = Path(call_dir)
    metadata = ensure_vector_index(call_dir)
    chunks = load_chunks(call_dir / CHUNKS_FILENAME)
    chunks_by_id = {str(chunk.get("chunk_id")): chunk for chunk in chunks}

    faiss = _faiss()
    index = faiss.read_index(str(call_dir / INDEX_FILENAME))
    q = embed_query(query, model_name=metadata["embedding_model"])
    k = max(1, min(int(top_k), int(index.ntotal)))
    scores, positions = index.search(np.ascontiguousarray(q, dtype=np.float32), k)

    results: List[Dict[str, Any]] = []
    chunk_ids = list(metadata.get("chunk_ids", []))
    for rank, (score, pos) in enumerate(zip(scores[0], positions[0]), start=1):
        if int(pos) < 0 or int(pos) >= len(chunk_ids):
            continue
        score_value = float(score)
        if min_score is not None and score_value < float(min_score):
            continue
        chunk = dict(chunks_by_id.get(chunk_ids[int(pos)], {}))
        if not chunk:
            continue
        chunk["score"] = round(score_value, 6)
        chunk["rank"] = rank
        results.append(chunk)
    return results


if __name__ == "__main__":
    import argparse, json
    parser = argparse.ArgumentParser(description="Local FAISS vector tools")
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("call_dir")
    b.add_argument("--rebuild", action="store_true")
    q = sub.add_parser("query")
    q.add_argument("call_dir")
    q.add_argument("query")
    q.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    i = sub.add_parser("info")
    i.add_argument("call_dir")
    args = parser.parse_args()
    if args.command == "build":
        result = ensure_vector_index(args.call_dir, rebuild=args.rebuild)
    elif args.command == "query":
        result = retrieve_relevant_chunks(args.call_dir, args.query, top_k=args.top_k)
    else:
        result = read_json(Path(args.call_dir) / METADATA_FILENAME)
    print(json.dumps(result, indent=2, default=str))
