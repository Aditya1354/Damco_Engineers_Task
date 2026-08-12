"""Pre-download/cache all local ML models used by the project."""
from __future__ import annotations

import json

import config


def main() -> None:
    if config.LOCAL_MODELS_ONLY:
        print("LOCAL_MODELS_ONLY=1: only already-cached models can be loaded.")

    results = {}

    from audio_transcription import get_transcription_model, DEFAULT_WHISPER_MODEL
    get_transcription_model()
    results["whisper"] = {"status": "ready", "model": DEFAULT_WHISPER_MODEL}

    from pyannote_diarizer import get_diarization_pipeline, DEFAULT_PYANNOTE_MODEL
    get_diarization_pipeline()
    results["diarization"] = {"status": "ready", "model": DEFAULT_PYANNOTE_MODEL}

    from speaker_embedder import get_embedding_inference, DEFAULT_MODEL
    get_embedding_inference()
    results["speaker_embedding"] = {"status": "ready", "model": DEFAULT_MODEL}

    from local_vector_store import get_text_embedding_model, DEFAULT_TEXT_EMBEDDING_MODEL
    get_text_embedding_model()
    results["text_embedding"] = {"status": "ready", "model": DEFAULT_TEXT_EMBEDDING_MODEL}

    print(json.dumps(results, indent=2))
    print("Models are ready. You can now set LOCAL_MODELS_ONLY=1 for offline model loading.")


if __name__ == "__main__":
    main()
