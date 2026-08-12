"""CLI entry point for the local-first call-intelligence project."""
from __future__ import annotations

import argparse
import json
from typing import Any


def _print(data: Any) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False, default=str))


def cmd_status(_args: argparse.Namespace) -> None:
    from preflight import get_preflight_status
    _print(get_preflight_status())


def cmd_enroll(args: argparse.Namespace) -> None:
    from agent_enrollment import enroll_agent
    _print(enroll_agent(args.agent_id, args.agent_name, args.sample, overwrite=args.overwrite))


def cmd_list_agents(_args: argparse.Namespace) -> None:
    from agent_enrollment import list_agents, list_incomplete_agent_folders
    _print({"agents": list_agents(), "incomplete_folders": list_incomplete_agent_folders()})


def cmd_cleanup_agents(_args: argparse.Namespace) -> None:
    from agent_enrollment import cleanup_incomplete_agent_folders
    _print({"removed": cleanup_incomplete_agent_folders()})


def cmd_delete_agent(args: argparse.Namespace) -> None:
    from agent_enrollment import delete_agent
    _print({"agent_id": args.agent_id, "deleted": delete_agent(args.agent_id)})


def cmd_process(args: argparse.Namespace) -> None:
    from audio_function import process_audio_pipeline
    client_info = {"client_name": args.client_name} if args.client_name else {}
    result = process_audio_pipeline(
        args.audio,
        call_name=args.call_name,
        expected_agent_ids=[args.agent_id] if args.agent_id else None,
        client_info=client_info,
        num_speakers=None if args.auto_speakers else args.num_speakers,
        min_speakers=args.min_speakers if args.auto_speakers else None,
        max_speakers=args.max_speakers if args.auto_speakers else None,
        overwrite=args.overwrite,
        min_similarity=args.min_similarity,
        min_margin=args.min_margin,
    )
    _print(result)
    if result.get("status") == "needs_speaker_confirmation":
        speakers = result.get("available_speakers", [])
        if speakers:
            print("\nManual confirmation required. Example:")
            print(
                f"  python run.py confirm-agent {result['call_name']} --speaker {speakers[0]}"
                + (f" --agent-id {args.agent_id}" if args.agent_id else "")
            )


def cmd_confirm(args: argparse.Namespace) -> None:
    from audio_function import confirm_agent_speaker
    _print(
        confirm_agent_speaker(
            args.call_name,
            args.speaker,
            agent_id=args.agent_id,
            agent_name=args.agent_name,
            client_name=args.client_name,
            inherit_expected_agent=not args.generic_agent,
        )
    )


def cmd_list_calls(_args: argparse.Namespace) -> None:
    from audio_function import list_calls
    _print(list_calls())


def cmd_call_info(args: argparse.Namespace) -> None:
    from audio_function import load_call
    _print(load_call(args.call_name))


def cmd_build_index(args: argparse.Namespace) -> None:
    from call_chatbot import resolve_call_dir
    from local_vector_store import ensure_vector_index
    call_dir = resolve_call_dir(args.call_name)
    _print(ensure_vector_index(call_dir, rebuild=args.rebuild))


def cmd_ask(args: argparse.Namespace) -> None:
    from call_chatbot import ask_call
    _print(ask_call(args.call_name, args.question, top_k=args.top_k))


def cmd_email(args: argparse.Namespace) -> None:
    from call_chatbot import generate_call_email
    _print(generate_call_email(args.call_name, args.request, top_k=args.top_k))


def cmd_route(args: argparse.Namespace) -> None:
    from call_chatbot import chat_call_request
    _print(chat_call_request(args.call_name, args.message, top_k=args.top_k))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Damco Engineer Track - local-first AI call intelligence",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("status", help="Show safe local runtime/preflight status")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("enroll-agent", help="Enroll an agent from a local voice sample")
    p.add_argument("agent_id")
    p.add_argument("agent_name")
    p.add_argument("sample")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_enroll)

    p = sub.add_parser("list-agents", help="List valid and incomplete local agent profiles")
    p.set_defaults(func=cmd_list_agents)

    p = sub.add_parser("cleanup-agents", help="Delete incomplete/stale local agent folders")
    p.set_defaults(func=cmd_cleanup_agents)

    p = sub.add_parser("delete-agent", help="Delete one enrolled local agent")
    p.add_argument("agent_id")
    p.set_defaults(func=cmd_delete_agent)

    p = sub.add_parser("process-call", help="Process a local call")
    p.add_argument("audio")
    p.add_argument("--call-name")
    p.add_argument("--agent-id", help="Expected enrolled agent; restricts automatic matching")
    p.add_argument("--client-name")
    count = p.add_mutually_exclusive_group()
    count.add_argument("--num-speakers", type=int, default=2, help="Exact diarized speaker count")
    count.add_argument("--auto-speakers", action="store_true", help="Let pyannote estimate speaker count")
    p.add_argument("--min-speakers", type=int, default=2, help="Minimum count with --auto-speakers")
    p.add_argument("--max-speakers", type=int, default=3, help="Maximum count with --auto-speakers")
    p.add_argument("--min-similarity", type=float, default=0.50)
    p.add_argument("--min-margin", type=float, default=0.12)
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_process)

    p = sub.add_parser("confirm-agent", help="Manually identify which diarized speaker is the Agent")
    p.add_argument("call_name")
    p.add_argument("--speaker", required=True)
    identity = p.add_mutually_exclusive_group()
    identity.add_argument("--agent-id")
    identity.add_argument(
        "--generic-agent",
        action="store_true",
        help="Do not inherit an expected enrolled identity; keep this as a generic Agent",
    )
    p.add_argument("--agent-name")
    p.add_argument("--client-name")
    p.set_defaults(func=cmd_confirm)

    p = sub.add_parser("list-calls", help="List local call folders and states")
    p.set_defaults(func=cmd_list_calls)

    p = sub.add_parser("call-info", help="Print final data for a completed call")
    p.add_argument("call_name")
    p.set_defaults(func=cmd_call_info)

    p = sub.add_parser("build-index", help="Build/rebuild the selected call's FAISS index")
    p.add_argument("call_name")
    p.add_argument("--rebuild", action="store_true")
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("ask", help="Ask a grounded question about one completed call")
    p.add_argument("call_name")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=cmd_ask)

    p = sub.add_parser("email", help="Generate a draft follow-up email from one completed call")
    p.add_argument("call_name")
    p.add_argument("request")
    p.add_argument("--top-k", type=int, default=7)
    p.set_defaults(func=cmd_email)

    p = sub.add_parser("route", help="Automatically route a message to Q&A or email-draft mode")
    p.add_argument("call_name")
    p.add_argument("message")
    p.add_argument("--top-k", type=int, default=5)
    p.set_defaults(func=cmd_route)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
