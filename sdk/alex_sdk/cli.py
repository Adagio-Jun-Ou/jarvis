"""
`alex` — the SDK's command-line entry point.

Usage:
  alex dev                  # run a local governed loop, mock identity
  alex deploy -t prod       # validate, run the constitutional gate, ship
  alex skills               # list/ratify proposed or active skills
  alex cap list             # list effective + delegated capabilities
  alex tools                # table of the ACL-filtered tool manifest
  alex teach "..."          # teach the agent a routine explicitly
  alex chat "..."           # one streamed chat turn (--reasoning shows
                            # the model's reasoning lines dimmed)
  alex workflow             # workflow catalog (not available yet)

The CLI is intentionally small. The SDK is the primary surface; the
CLI is for one-off inspection of the runtime.
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="alex", description=__doc__.splitlines()[0])
    p.add_argument("--app-url", default=os.environ.get("ALEX_APP_URL", "http://localhost:8000"))
    p.add_argument("--token", default=os.environ.get("ALEX_OBO_TOKEN", ""))
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("dev", help="run a local governed loop, mock identity")
    dep = sub.add_parser("deploy", help="validate, gate, and ship")
    dep.add_argument("-t", "--target", default="prod", choices=["dev", "staging", "prod"])
    sub.add_parser("skills", help="list/ratify skills")
    cap = sub.add_parser("cap", help="list capabilities")
    cap_sub = cap.add_subparsers(dest="cap_cmd")
    cap_sub.add_parser("list")
    sub.add_parser("workflow", help="workflow catalog (not available yet)")
    sub.add_parser("tools", help="list the ACL-filtered tool manifest")
    teach = sub.add_parser("teach", help="teach the agent a routine")
    teach.add_argument("prompt", help="the routine, in your own words")
    chat = sub.add_parser("chat", help="one streamed chat turn")
    chat.add_argument("prompt", help="what to ask")
    chat.add_argument(
        "--reasoning",
        action="store_true",
        help="show the model's reasoning lines (dimmed, prefixed)",
    )

    args = p.parse_args(argv)

    if not args.cmd:
        p.print_help()
        return 0

    if args.cmd == "dev":
        print(f"[alex dev] mock-identity loop on {args.app_url}")
        return 0
    if args.cmd == "deploy":
        print(f"[alex deploy] target={args.target} (stubbed in this build)")
        return 0
    if args.cmd == "skills":
        return _list_skills(args)
    if args.cmd == "cap":
        return _capabilities(args)
    if args.cmd == "workflow":
        return _workflows(args)
    if args.cmd == "tools":
        return _tools(args)
    if args.cmd == "teach":
        return _teach(args)
    if args.cmd == "chat":
        return _chat(args)

    p.print_help()
    return 1


def _list_skills(args) -> int:
    import httpx

    with httpx.Client(base_url=args.app_url, headers=_auth(args), timeout=15.0) as c:
        r = c.get("/api/skills")
        r.raise_for_status()
        for s in r.json().get("skills", []):
            print(f"{s['status']:>9s}  {s['skill_id'][:8]}  {s['name']}")
        return 0


def _capabilities(args) -> int:
    if args.cap_cmd == "list":
        import httpx

        with httpx.Client(base_url=args.app_url, headers=_auth(args), timeout=15.0) as c:
            r = c.get("/api/transparency/capabilities")
            r.raise_for_status()
            for cap in r.json().get("effective", []):
                print(
                    f"effective: {cap['capability']} (expires {cap.get('expires_at') or 'forever'})"
                )
            for cap in r.json().get("delegations", []):
                print(f"delegated: {cap['capability']} from {cap['supervisor_email']}")
        return 0
    print("usage: alex cap list")
    print("(capability grants go through the supervisor UI: POST /api/supervisor/delegate)")
    return 2


def _workflows(args) -> int:
    print(
        "the workflow catalog is not available yet: the app backend has no "
        "workflow-listing route. Define workflows in code with "
        "alex_sdk.workflow.workflow(...)."
    )
    return 2


def _client(args):
    from alex_sdk.client import Alex

    return Alex(base_url=args.app_url, token=args.token)


def _tools(args) -> int:
    with _client(args) as alex:
        tools = alex.tools()
        if not tools:
            print("(no tools available to this user)")
            return 0
        width = max(len(t.name) for t in tools)
        for t in tools:
            print(f"{t.name:<{width}}  {t.description}")
    return 0


def _teach(args) -> int:
    with _client(args) as alex:
        r = alex.teach(args.prompt)
        print(f"stored routine {r.get('routine_id', '?')} ({r.get('status', 'unknown')})")
    return 0


DIM = "\x1b[2m"
RESET = "\x1b[0m"


def _chat(args) -> int:
    with _client(args) as alex:
        session = alex.session("cli chat")
        for ev in session.run(args.prompt, show_reasoning=args.reasoning):
            if ev.type == "text":
                print(ev.text, end="", flush=True)
            elif ev.type == "reasoning" and args.reasoning:
                print(f"{DIM}[reasoning] {ev.text}{RESET}")
            elif ev.type == "law_block":
                print(f"[law_block] {ev.article}: {ev.detail}")
            elif ev.type == "error":
                print(f"[error] {ev.code}: {ev.message}")
        print()
    return 0


def _auth(args) -> dict:
    # Single source of truth for auth headers. Both are required: production
    # identity resolution reads X-Forwarded-Access-Token (the Bearer header
    # is ignored outside dev), so sending only Authorization 401s in prod.
    from alex_sdk.client import auth_headers

    return auth_headers(args.token)


if __name__ == "__main__":
    sys.exit(main())
