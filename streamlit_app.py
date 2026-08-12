"""Streamlit UI for the complete local-first call-intelligence workflow."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import config
from project_utils import read_json, slugify

try:
    import streamlit as st
except Exception as exc:  # pragma: no cover - only when launched without dependency
    raise RuntimeError("Streamlit is not installed. Run `pip install -r requirements.txt`.") from exc

AUDIO_EXTENSIONS = ["wav", "mp3", "m4a", "flac", "ogg", "aac"]


def save_uploaded_file(uploaded_file, folder: Path, preferred_stem: str | None = None) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    suffix = Path(uploaded_file.name).suffix.lower() or ".audio"
    stem = slugify(preferred_stem or Path(uploaded_file.name).stem, fallback="upload")
    path = folder / f"{stem}{suffix}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def _agents() -> List[Dict[str, Any]]:
    from agent_enrollment import list_agents
    return list_agents()


def _calls() -> List[Dict[str, Any]]:
    
    from audio_function import list_calls
    return list_calls()


def completed_call_names() -> List[str]:
    return [x["call_name"] for x in _calls() if x["status"] == "completed"]


def pending_call_names() -> List[str]:
    return [x["call_name"] for x in _calls() if x["status"] == "needs_speaker_confirmation"]


def show_summary(call_data: Dict[str, Any]) -> None:
    summary = call_data.get("summary") or {}
    if not summary:
        st.info("No summary is stored for this call.")
        return
    st.subheader("Summary")
    st.write(summary.get("summary", ""))
    if summary.get("key_points"):
        st.markdown("**Key points**")
        for item in summary["key_points"]:
            st.markdown(f"- {item}")
    if summary.get("decisions"):
        st.markdown("**Decisions**")
        for item in summary["decisions"]:
            st.markdown(f"- {item}")
    if summary.get("action_items"):
        st.markdown("**Action items**")
        st.dataframe(summary["action_items"], use_container_width=True, hide_index=True)
    if summary.get("risks_or_followups"):
        st.markdown("**Risks / follow-ups**")
        for item in summary["risks_or_followups"]:
            st.markdown(f"- {item}")


def show_transcript(call_data: Dict[str, Any]) -> None:
    transcript = call_data.get("transcript") or []
    if not transcript:
        st.info("No transcript stored.")
        return
    rows = [
        {
            "start": turn.get("start"),
            "end": turn.get("end"),
            "role": turn.get("role"),
            "speaker": turn.get("speaker_name") or turn.get("speaker"),
            "text": turn.get("text"),
        }
        for turn in transcript
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def show_sources(chunks: List[Dict[str, Any]]) -> None:
    if not chunks:
        return
    with st.expander("Retrieved call chunks"):
        for chunk in chunks:
            st.markdown(
                f"**{chunk.get('chunk_id')}** · score `{chunk.get('score')}` · "
                f"{chunk.get('start')} → {chunk.get('end')}"
            )
            st.text(chunk.get("text", ""))


def page_home() -> None:
    from preflight import get_preflight_status
    status = get_preflight_status()
    st.title("Local-First AI Call Intelligence")
    st.caption("Damco Engineer Track demo — all audio/RAG storage is local; only Groq LLM calls are remote.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Valid agents", status["agents"]["valid_count"])
    c2.metric("Completed calls", status["calls"]["completed"])
    c3.metric("Pending speaker map", status["calls"]["pending_confirmation"])
    c4.metric("Failed calls", status["calls"]["failed"])

    st.subheader("Runtime status")
    rows = [
        {"check": "Python >= 3.10", "ok": status["python_supported"], "detail": status["python"]},
        {"check": "FFmpeg", "ok": status["ffmpeg_available"], "detail": status["ffmpeg"] or "not found"},
        {"check": "HF token configured", "ok": status["config"]["hf_token_configured"], "detail": "configured" if status["config"]["hf_token_configured"] else "missing"},
        {"check": "Groq key configured", "ok": status["config"]["groq_api_key_configured"], "detail": "configured" if status["config"]["groq_api_key_configured"] else "missing"},
        {"check": "CUDA visible to PyTorch (pyannote)", "ok": status["cuda"]["cuda_available"], "detail": status["cuda"]["device_name"] or "CPU mode"},
        {"check": "CUDA visible to CTranslate2 (Whisper)", "ok": status["ctranslate2"]["cuda_available"], "detail": f"{status['ctranslate2']['cuda_device_count']} CUDA device(s)" if status["ctranslate2"]["available"] else "CTranslate2 not installed"},
        {"check": "Whisper word timestamps", "ok": bool(status["config"].get("whisper_word_timestamps")), "detail": "required for speaker alignment"},
        {"check": "Agent match threshold", "ok": True, "detail": f"similarity ≥ {status['config'].get('speaker_match_threshold')} · margin ≥ {status['config'].get('speaker_match_margin')}"},
        {"check": "Local-model-only mode", "ok": True, "detail": str(status["config"]["local_models_only"])},
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Configured models")
    st.json({
        "whisper": status["config"]["whisper_model"],
        "diarization": status["config"]["pyannote_model"],
        "speaker_embedding": status["config"]["speaker_embedding_model"],
        "text_embedding": status["config"]["text_embedding_model"],
        "groq": status["config"]["groq_model"],
    })

    if status["agents"]["incomplete_count"]:
        st.warning("Incomplete agent folders were found. Use Agent Enrollment → cleanup before processing calls.")
        st.json(status["agents"]["incomplete"])


def page_agent_enrollment() -> None:
    from agent_enrollment import (
        DEFAULT_ENROLLMENT_PHRASE,
        DEFAULT_PHRASE_MATCH_THRESHOLD,
        cleanup_incomplete_agent_folders,
        enroll_agent,
        list_incomplete_agent_folders,
        verify_enrollment_phrase,
    )

    st.title("Agent Enrollment")
    st.write(
        "Enroll a known agent either by uploading an existing clean voice sample "
        "or by recording the verification phrase directly in the browser."
    )

    incomplete = list_incomplete_agent_folders()
    if incomplete:
        st.warning(f"{len(incomplete)} incomplete agent folder(s) found.")
        st.json(incomplete)
        if st.button("Delete incomplete agent folders"):
            removed = cleanup_incomplete_agent_folders()
            st.success(f"Removed {len(removed)} folder(s).")
            st.rerun()

    agent_id = st.text_input("Agent ID", placeholder="agent001", key="enroll_agent_id")
    agent_name = st.text_input("Agent name", placeholder="Aditya", key="enroll_agent_name")

    enrollment_method = st.radio(
        "Voice sample method",
        ["Upload existing audio", "Record verification phrase"],
        horizontal=True,
        key="enrollment_method",
    )

    sample = None
    if enrollment_method == "Upload existing audio":
        sample = st.file_uploader(
            "Voice sample",
            type=AUDIO_EXTENSIONS,
            key="agent_voice_upload",
        )
        st.caption(
            "The uploaded file is used directly for voice enrollment. Phrase verification "
            "is applied only to recordings made with the microphone option."
        )
    else:
        st.markdown("**Read this exact phrase clearly:**")
        st.info(DEFAULT_ENROLLMENT_PHRASE)
        sample = st.audio_input(
            "Record the enrollment phrase",
            sample_rate=16000,
            key="agent_voice_recording",
            help=(
                "Speak the displayed phrase in a quiet environment. The recording is "
                f"accepted only when transcription similarity is at least "
                f"{DEFAULT_PHRASE_MATCH_THRESHOLD * 100:.0f}%."
            ),
        )
        if sample is not None:
            st.audio(sample)
            st.caption(
                f"Required phrase match: ≥ {DEFAULT_PHRASE_MATCH_THRESHOLD * 100:.0f}%"
            )

    overwrite = st.checkbox(
        "Overwrite existing agent with this ID",
        key="enroll_overwrite",
    )
    submitted = st.button("Enroll agent", type="primary", key="enroll_agent_submit")

    if submitted:
        if not agent_id.strip() or not agent_name.strip() or sample is None:
            st.error("Agent ID, agent name, and a voice sample/recording are required.")
            return

        try:
            is_recording = enrollment_method == "Record verification phrase"
            preferred_stem = (
                f"{agent_id}_recorded" if is_recording else agent_id
            )
            upload_path = save_uploaded_file(
                sample,
                config.UPLOADS_DIR / "agents",
                preferred_stem=preferred_stem,
            )

            enrollment_metadata: Dict[str, Any] = {
                "enrollment_source": "microphone" if is_recording else "uploaded_audio",
            }

            if is_recording:
                with st.spinner("Checking the recorded phrase with Groq Whisper..."):
                    verification = verify_enrollment_phrase(upload_path)

                c1, c2 = st.columns(2)
                c1.metric(
                    "Phrase match",
                    f"{verification['similarity_percent']:.1f}%",
                )
                c2.metric(
                    "Required",
                    f"{verification['threshold_percent']:.0f}%",
                )
                st.caption(
                    "Recognized text: "
                    + (verification.get("recognized_text") or "No speech recognized")
                )

                if not verification["passed"]:
                    st.error(
                        "Enrollment stopped: the recorded speech did not match the "
                        f"required phrase at {verification['threshold_percent']:.0f}% or higher. "
                        "Please record the phrase again clearly."
                    )
                    return

                st.success("Phrase verification passed. Creating the voice embedding...")
                enrollment_metadata["phrase_verification"] = verification

            with st.spinner("Normalizing sample and creating local voice embedding..."):
                result = enroll_agent(
                    agent_id,
                    agent_name,
                    upload_path,
                    overwrite=overwrite,
                    metadata=enrollment_metadata,
                )

            st.success("Agent enrolled successfully.")
            st.json(result)
        except Exception as exc:
            st.exception(exc)

    agents = _agents()
    if agents:
        st.subheader("Enrolled agents")
        st.dataframe(agents, use_container_width=True, hide_index=True)


def page_process_call() -> None:
    from audio_function import process_audio_pipeline
    st.title("Process Call")
    agents = _agents()
    agent_options = ["Auto-detect from all enrolled agents"] + [f"{x['agent_id']} — {x['agent_name']}" for x in agents]

    with st.form("process_call_form"):
        upload = st.file_uploader("Recorded call", type=AUDIO_EXTENSIONS)
        call_name = st.text_input("Call name (optional)", placeholder="client_followup_2026_08_07")
        expected_agent = st.selectbox("Expected agent", agent_options)
        client_name = st.text_input("Client name (optional)")
        speaker_mode = st.selectbox(
            "Diarization speaker count",
            [
                "Exactly 2 speakers (normal Agent/Client call)",
                "Automatic 2-3 speakers (use when IVR/system voice may exist)",
                "Custom exact speaker count",
            ],
        )
        custom_num_speakers = st.number_input(
            "Custom exact speaker count", min_value=1, max_value=10, value=2, step=1,
            disabled=speaker_mode != "Custom exact speaker count",
        )
        overwrite = st.checkbox("Overwrite call folder if it already exists")
        submitted = st.form_submit_button("Process call", type="primary")

    if submitted:
        if upload is None:
            st.error("Upload a call first.")
            return
        effective_call_name = slugify(call_name or Path(upload.name).stem, fallback="call")
        expected_id = None
        if expected_agent != agent_options[0]:
            expected_id = expected_agent.split(" — ", 1)[0]
        try:
            upload_path = save_uploaded_file(upload, config.UPLOADS_DIR / "calls", preferred_stem=effective_call_name)
            with st.spinner(
                        "Running local transcription, diarization, speaker matching, summary, and indexing...\n"
                        "This may take a few minutes the first time you use it as models download in the background...\n"
                        "If it throws an error on the first run, please run it again and it will work smoothly."
                    ):
                if speaker_mode.startswith("Automatic"):
                    exact_speakers, min_speakers, max_speakers = None, 2, 3
                elif speaker_mode == "Custom exact speaker count":
                    exact_speakers, min_speakers, max_speakers = int(custom_num_speakers), None, None
                else:
                    exact_speakers, min_speakers, max_speakers = 2, None, None
                result = process_audio_pipeline(
                    upload_path,
                    call_name=effective_call_name,
                    expected_agent_ids=[expected_id] if expected_id else None,
                    client_info={"client_name": client_name.strip()} if client_name.strip() else {},
                    num_speakers=exact_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                    overwrite=overwrite,
                )
            if result["status"] == "completed":
                st.success("Call processed successfully.")
            else:
                st.warning("Processing paused because the Agent speaker could not be identified with high confidence. Use Speaker Confirmation.")
                st.write("Available speakers:", result.get("available_speakers"))
                st.json(result.get("speaker_resolution", {}))
        except Exception as exc:
            st.exception(exc)


def page_speaker_confirmation() -> None:
    from audio_function import confirm_agent_speaker
    st.title("Speaker Confirmation")
    pending = pending_call_names()
    if not pending:
        st.info("No calls are waiting for speaker confirmation.")
        return

    call_name = st.selectbox("Pending call", pending)
    call_dir = config.CALLS_DIR / call_name
    pending_data = read_json(call_dir / "pending_call_data.json")
    resolution = pending_data.get("speaker_resolution") or {}
    speakers = pending_data.get("available_speakers") or []
    st.write("Automatic matching stopped because the confidence policy was not satisfied.")
    if resolution:
        st.json(resolution)

    preview_files = pending_data.get("speaker_preview_files") or {}
    if speakers:
        st.subheader("Listen before confirming")
        st.caption(
            "These clips come directly from the raw pyannote speaker turns. Use them instead "
            "of judging identity from transcript text alone."
        )
        columns = st.columns(min(len(speakers), 3))
        for idx, speaker_label in enumerate(speakers):
            with columns[idx % len(columns)]:
                st.markdown(f"**{speaker_label}**")
                candidate = Path(preview_files.get(speaker_label, "")) if preview_files.get(speaker_label) else None
                if candidate is None or not candidate.is_file():
                    fallback = call_dir / "speaker_previews" / f"{slugify(speaker_label, fallback='speaker')}.wav"
                    candidate = fallback if fallback.is_file() else None
                if candidate is not None:
                    st.audio(str(candidate))
                else:
                    st.caption("Preview unavailable")

    agents = _agents()
    agent_labels = ["Generic Agent (no enrolled identity)"] + [f"{x['agent_id']} — {x['agent_name']}" for x in agents]
    with st.form("speaker_confirm_form"):
        speaker = st.selectbox("Which speaker is the Agent?", speakers)
        agent_choice = st.selectbox("Agent identity", agent_labels)
        generic_agent_name = st.text_input("Generic Agent display name", value="Agent", disabled=agent_choice != agent_labels[0])
        existing_client = (pending_data.get("client_info") or {}).get("client_name", "")
        client_name = st.text_input("Client display name", value=existing_client or "Client")
        submitted = st.form_submit_button("Confirm and finalize call", type="primary")

    if submitted:
        try:
            selected_id = None if agent_choice == agent_labels[0] else agent_choice.split(" — ", 1)[0]
            with st.spinner("Finalizing summary, chunks, and local FAISS index..."):
                result = confirm_agent_speaker(
                    call_name,
                    speaker,
                    agent_id=selected_id,
                    agent_name=generic_agent_name if selected_id is None else None,
                    client_name=client_name,
                    inherit_expected_agent=selected_id is not None,
                )
            st.success("Speaker mapping confirmed and call finalized.")
            st.json({"status": result["status"], "call_name": result["call_name"]})
        except Exception as exc:
            st.exception(exc)


def page_call_browser() -> None:
    st.title("Call Browser")
    calls = completed_call_names()
    if not calls:
        st.info("No completed calls are available.")
        return
    call_name = st.selectbox("Completed call", calls)
    data = read_json(config.CALLS_DIR / call_name / "final_call_data.json")
    tabs = st.tabs(["Summary", "Transcript", "Speaker resolution", "Raw JSON"])
    with tabs[0]:
        show_summary(data)
    with tabs[1]:
        show_transcript(data)
    with tabs[2]:
        st.json(data.get("speaker_identification") or {})
        st.json(data.get("role_mapping") or {})
    with tabs[3]:
        st.json(data)


def page_chatbot() -> None:
    from call_chatbot import ask_call
    st.title("Call Chatbot")
    calls = completed_call_names()
    if not calls:
        st.info("Process and finalize a call first.")
        return
    call_name = st.selectbox("Selected call", calls, key="chat_call")
    st.caption("Answers are generated only from FAISS-retrieved chunks belonging to this selected call.")

    history_key = f"chat_history::{call_name}"
    if history_key not in st.session_state:
        st.session_state[history_key] = []
    for item in st.session_state[history_key]:
        with st.chat_message(item["role"]):
            st.write(item["content"])

    question = st.chat_input("Ask about this call")
    if question:
        st.session_state[history_key].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        try:
            with st.chat_message("assistant"):
                with st.spinner("Retrieving selected-call context..."):
                    result = ask_call(call_name, question)
                st.write(result["answer"])
                if result["insufficient_context"]:
                    st.info("The retrieved selected-call context was insufficient for a supported answer.")
                show_sources(result.get("retrieved_chunks", []))
            st.session_state[history_key].append({"role": "assistant", "content": result["answer"]})
        except Exception as exc:
            st.exception(exc)


def page_email_generator() -> None:
    from call_chatbot import generate_call_email
    st.title("Email Generator")
    calls = completed_call_names()
    if not calls:
        st.info("Process and finalize a call first.")
        return
    call_name = st.selectbox("Selected call", calls, key="email_call")
    request = st.text_area(
        "Draft request",
        value="Draft a professional follow-up email summarizing the agreed next steps from the call.",
        height=110,
    )
    st.warning("Draft only: this application has no email-send capability.")
    if st.button("Generate email draft", type="primary"):
        try:
            with st.spinner("Retrieving selected-call context and drafting..."):
                result = generate_call_email(call_name, request)
            st.text_input("Subject", value=result["subject"], key=f"subject::{call_name}")
            st.text_area("Body", value=result["body"], height=320, key=f"body::{call_name}")
            show_sources(result.get("retrieved_chunks", []))
        except Exception as exc:
            st.exception(exc)


# def page_vector_tools() -> None:
#     from local_vector_store import ensure_vector_index, retrieve_relevant_chunks
#     st.title("Vector Tools")
#     calls = completed_call_names()
#     if not calls:
#         st.info("No completed calls are available.")
#         return
#     call_name = st.selectbox("Selected call", calls, key="vector_call")
#     call_dir = config.CALLS_DIR / call_name
#     c1, c2 = st.columns(2)
#     if c1.button("Rebuild FAISS index"):
#         try:
#             with st.spinner("Rebuilding local index..."):
#                 st.json(ensure_vector_index(call_dir, rebuild=True))
#         except Exception as exc:
#             st.exception(exc)
#     if c2.button("Show vector metadata"):
#         path = call_dir / "vector_metadata.json"
#         if path.exists():
#             st.json(read_json(path))
#         else:
#             st.info("No vector metadata yet.")

#     query = st.text_input("Test retrieval query")
#     if st.button("Query local index") and query.strip():
#         try:
#             results = retrieve_relevant_chunks(call_dir, query)
#             show_sources(results)
#         except Exception as exc:
#             st.exception(exc)


def main() -> None:
    st.set_page_config(page_title="Damco Local Call Intelligence", page_icon="🎧", layout="wide")
    pages = {
        "Home / Status": page_home,
        "Agent Enrollment": page_agent_enrollment,
        "Process Call": page_process_call,
        "Speaker Confirmation": page_speaker_confirmation,
        "Call Browser": page_call_browser,
        "Chatbot": page_chatbot,
        "Email Generator": page_email_generator,
        # "Vector Tools": page_vector_tools,
    }
    choice = st.sidebar.radio("Workflow", list(pages))
    st.sidebar.caption("Local data: data/ · Uploads: uploads/ · Models: models/")
    pages[choice]()


if __name__ == "__main__":
    main()
