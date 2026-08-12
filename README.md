# Damco Task — AI Call Intelligence

A lightweight AI-powered call intelligence system built for the **Damco Engineer Challenge**.

The project processes recorded calls, separates speakers, identifies the enrolled agent, generates a structured transcript and summary, and allows users to ask questions or create follow-up email drafts from the processed call.

## Objective

The objective of this project is to reduce the manual effort required to review recorded customer calls.

A normal recording is difficult to search and usually does not clearly answer:

- Who said what?
- What were the important points?
- What actions were agreed?
- What information was discussed?
- What should be sent in a follow-up email?

This project converts a raw call recording into structured, searchable call intelligence.

## Why I Chose This Problem

Recorded calls contain useful information, but manually listening to complete conversations is slow and repetitive.

A transcription alone also does not solve the complete problem because the system still needs to determine **which speaker said each part of the conversation**.

This made the problem interesting from an engineering perspective because it required combining speech transcription, speaker diarization, timestamp alignment, speaker verification, local vector retrieval, LLM-based summarization and Q&A, and human confirmation when automatic identification is uncertain.

Instead of building a single AI script, the goal was to build a complete workflow with clear intermediate outputs and fallback behavior.

## Main Features

- Agent voice enrollment using a reference audio sample
- Call audio normalization to mono 16 kHz WAV
- Speech-to-text using **Groq Whisper**
- Word-level transcription timestamps
- Speaker diarization using **Pyannote Community-1**
- Word-to-speaker alignment using timestamp overlap
- Agent identification using **WeSpeaker voice embeddings**
- Manual speaker confirmation when confidence is insufficient
- Call summary and action-item generation
- Local transcript chunking and **FAISS** vector search
- Call-specific chatbot
- Follow-up email draft generation
- Multiple processed-call selection
- Streamlit user interface
- CPU-compatible processing

## Models and Technologies

| Purpose | Technology / Model |
|---|---|
| Speech-to-text | Groq `whisper-large-v3-turbo` |
| Speaker diarization | `pyannote/speaker-diarization-community-1` |
| Speaker verification | `pyannote/wespeaker-voxceleb-resnet34-LM` |
| Text embeddings | `BAAI/bge-small-en-v1.5` |
| Vector search | FAISS |
| Summary / Q&A / Email | Groq LLM |
| UI | Streamlit |

Pyannote diarization, speaker verification, text embeddings, FAISS retrieval, and application data processing run on the application machine. Groq is used for speech transcription and LLM inference.

## Processing Flow

```text
Recorded Call
     |
     v
Audio Normalization
Mono / 16 kHz WAV
     |
     +-----------------------------+
     |                             |
     v                             v
Groq Whisper                 Pyannote Community-1
Word-level transcription     Speaker diarization
     |                             |
     +-------------+---------------+
                   |
                   v
          Word / Speaker Alignment
                   |
                   v
          SPEAKER_00 / SPEAKER_01
                   |
                   v
        WeSpeaker Voice Embeddings
                   |
          Compare with enrolled
             Agent embedding
                   |
        +----------+-----------+
        |                      |
   High confidence         Low confidence
        |                      |
        v                      v
 Agent / Client       Manual Speaker Confirmation
        |                      |
        +----------+-----------+
                   |
                   v
           Final Call Transcript
                   |
          +--------+--------+
          |                 |
          v                 v
      Groq Summary      BGE Embeddings
      Action Items           |
                             v
                         FAISS Index
                             |
                             v
                    Call-specific Chatbot
                             |
                             v
                    Follow-up Email Draft
```

## Important Engineering Decision

One important issue discovered during development was that assigning an entire transcription segment to one diarized speaker produced incorrect results whenever that segment contained speech from both people.

The pipeline was therefore changed to use **word-level timestamps**.

Each transcribed word is now matched against the Pyannote speaker timeline independently. Only after speaker assignment are the words reconstructed into readable conversation turns.

This significantly improves the reliability of determining **who said what**.

Agent identification also uses confidence thresholds. If the system cannot confidently determine which anonymous speaker matches the enrolled agent voice, it does not guess — it asks the user to confirm the speaker manually.

## Project Structure

```text
Damco-Task/
|
|-- streamlit_app.py          # Main Streamlit UI
|-- audio_function.py         # Pipeline orchestration
|-- audio_transcription.py    # Groq Whisper transcription
|-- audio_utils.py            # Audio conversion / normalization
|-- pyannote_diarizer.py      # Speaker diarization
|-- transcript_aligner.py     # Word-to-speaker alignment
|-- agent_enrollment.py       # Agent voice enrollment
|-- speaker_embedder.py       # Voice embedding extraction
|-- speaker_matcher.py        # Agent / speaker matching
|-- groq_llm.py               # Summary, Q&A and email generation
|-- rag_chunker.py            # Transcript chunk creation
|-- local_vector_store.py     # BGE embeddings + FAISS
|-- call_chatbot.py           # Selected-call RAG workflow
|-- config.py                 # Configuration
|-- project_utils.py          # Shared utilities
|-- run.py                    # CLI entry point
|-- requirements.txt
|-- .env.example
|-- Dockerfile
`-- README.md
```

Generated call data and enrolled-agent embeddings are stored under `data/` at runtime and are intentionally excluded from Git.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Create the local environment file:

```bash
cp .env.example .env
```

On Windows:

```powershell
copy .env.example .env
```

Add your own credentials:

```env
GROQ_API_KEY=your_groq_api_key
HF_TOKEN=your_huggingface_token
```

FFmpeg must also be available for audio normalization.

Start the application:

```bash
python -m streamlit run streamlit_app.py
```

## Application Workflow

1. **Agent Enrollment** — upload a clean sample of the agent's voice.
2. **Process Call** — upload and process a recorded conversation.
3. **Speaker Confirmation** — manually identify the agent when automatic matching is uncertain.
4. **Call Browser** — inspect transcript, summary and action items.
5. **Chatbot** — ask questions based only on the selected processed call.
6. **Email Generator** — create a follow-up email draft from call context.
7. **Vector Tools** — inspect or rebuild the local FAISS retrieval index.

## Security

Real credentials are **not stored in the repository**.

Use:

```text
.env.example   -> public configuration template
.env           -> private local credentials
```

The `.env` file, generated call data, uploaded recordings, model caches, FAISS indexes and voice embeddings should remain excluded from Git.

---
