"""
Session Memory Middleware — per-repo persistent knowledge store.

Two responsibilities:

  1. PRE-PROCESS (Option 1 — knowledge injection):
     On the first API call of a session, load {workspace}/.claude/memory.md
     and append its contents to the system prompt. Claude then has full
     context from all previous sessions in this repo without the user
     having to re-explain anything.

  2. save_session_summary() (Option 2 — session hook):
     Called explicitly at session end (by Agent.save_session_memory()).
     Makes a lightweight Claude API call to summarise what happened, then
     prepends the result to memory.md so the newest session is always first.

Storage layout:
    {workspace}/
    └── .claude/
        └── memory.md      ← human-readable, newest session at top

The .claude/ directory is workspace-local (per-repo), so every project
gets its own isolated memory. Add .claude/memory.md to .gitignore if you
don't want to commit session notes (recommended for personal notes).
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from .base import BaseMiddleware, AgentState


# ─── Prompts ──────────────────────────────────────────────────────────────────

# Injected as a section header when memory.md exists.
_INJECTION_HEADER = (
    "\n\n---\n"
    "## Project Memory (accumulated from past sessions)\n"
    "The following notes were recorded by previous sessions in this repo.\n"
    "Use them to avoid re-doing work, respect established conventions,\n"
    "and recall prior decisions.\n\n"
)

# System prompt for the summarisation call at session end.
_SUMMARY_SYSTEM = """\
You are a memory manager for an AI coding assistant called YACC.
Your job is to distil a session transcript into a concise, structured
summary that will be prepended to a persistent memory file and injected
into future sessions.

Rules:
- Use bullet points, not prose paragraphs
- Focus only on information useful in a FUTURE session
- Include: what was accomplished, key files changed, patterns/conventions
  learned, gotchas or errors fixed, and any pending follow-up work
- Omit: greetings, verbose explanations, anything obvious from the code
- Be terse — the memory file grows over time; every byte counts"""

# Template for the user-turn of the summarisation call.
_SUMMARY_PROMPT = """\
Below is the conversation transcript from this session.
Produce a markdown session block in exactly this format (fill in bullets):

## Session {date}

### Accomplished
-

### Key files modified
-

### Patterns & conventions learned
-

### Gotchas / errors fixed
-

### Pending / follow-up
-

---

Transcript:
{transcript}"""

# Header written once at the top of a new memory.md file.
_FILE_HEADER = (
    "# Project Memory\n\n"
    "<!-- Managed by YACC. Newest sessions appear first.\n"
    "     Add .claude/memory.md to .gitignore to keep notes local. -->\n\n"
)


class SessionMemoryMiddleware(BaseMiddleware):
    """
    Reads project memory before each session and writes a summary at the end.

    Sits first in the middleware chain so that the injected memory is
    part of the system prompt when PromptCachingMiddleware adds its
    cache_control markers.
    """

    def __init__(self, workspace_path: str, enabled: bool = True):
        super().__init__(enabled)
        self.workspace_path = Path(workspace_path)
        # .claude/ mirrors Claude Code's own convention for per-repo config.
        self.memory_dir  = self.workspace_path / ".claude"
        self.memory_file = self.memory_dir / "memory.md"

        # Guard so we inject memory only once per session, not on every turn.
        # (pre_process runs before every API call inside the turn loop.)
        self._injected = False

    # ── Option 1: inject memory into system prompt ────────────────────────────

    def load_memory(self) -> Optional[str]:
        """Return the raw contents of memory.md, or None if it does not exist."""
        if self.memory_file.exists():
            return self.memory_file.read_text(encoding="utf-8").strip() or None
        return None

    def pre_process(self, state: AgentState) -> AgentState:
        """
        Append memory.md to the system prompt on the FIRST call of the session.

        Why only once:
          state.system_prompt persists across all turns of a session.
          Injecting on every pre_process would duplicate the memory block
          once per API call, bloating the prompt and wasting tokens.
        """
        if not self.enabled or self._injected:
            return state

        memory = self.load_memory()
        if memory:
            state.system_prompt = state.system_prompt + _INJECTION_HEADER + memory

        # Mark as injected regardless of whether memory existed — so a file
        # created mid-session isn't picked up half-way through.
        self._injected = True
        return state

    def post_process(
        self, state: AgentState, response: Dict[str, Any]
    ) -> tuple[AgentState, Dict[str, Any]]:
        # Nothing to do after individual API calls.
        return state, response

    # ── Option 2: save session summary at end ─────────────────────────────────

    def save_session_summary(
        self,
        client,                   # anthropic.Anthropic — passed in from Agent
        model: str,
        messages: List[Dict[str, Any]],
    ) -> str:
        """
        Summarise the session and prepend the result to memory.md.

        Steps:
          1. Build a readable transcript from messages (text only — tool
             results are too verbose and would inflate the summary).
          2. Call Claude with a lightweight prompt (max_tokens=600, no tools).
          3. Prepend the returned markdown block to memory.md.

        Returns the summary string so the CLI can display it to the user.
        """
        transcript = self._build_transcript(messages)
        date_str   = datetime.now().strftime("%Y-%m-%d")

        # Separate, minimal API call — outside the normal agent pipeline.
        # We deliberately avoid tools, caching, and middleware here.
        response = client.messages.create(
            model=model,
            max_tokens=600,
            system=_SUMMARY_SYSTEM,
            messages=[{
                "role": "user",
                "content": _SUMMARY_PROMPT.format(
                    date=date_str,
                    transcript=transcript,
                ),
            }],
        )

        summary = response.content[0].text.strip()
        self._write_to_memory(summary)
        return summary

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_transcript(self, messages: List[Dict[str, Any]]) -> str:
        """
        Convert the raw messages list into a concise readable transcript.

        We include:
          - USER: text messages
          - ASSISTANT: text blocks
          - TOOL CALL: tool name + truncated input (so the summary knows
            what files were touched, what commands were run, etc.)

        We skip:
          - tool_result blocks (the raw file contents, command output etc.
            are too long and add no value to the summary)
        """
        lines: List[str] = []

        for msg in messages:
            role    = msg.get("role", "")
            content = msg.get("content", "")

            if isinstance(content, str):
                # Plain string message (user turn at session start)
                lines.append(f"{role.upper()}: {content[:400]}")

            elif isinstance(content, list):
                for block in content:
                    if not isinstance(block, dict):
                        continue
                    btype = block.get("type")

                    if btype == "text":
                        text = block.get("text", "").strip()
                        if text:
                            lines.append(f"{role.upper()}: {text[:400]}")

                    elif btype == "tool_use":
                        name = block.get("name", "")
                        # Truncate large inputs (e.g. write_file content)
                        inp  = json.dumps(block.get("input", {}))[:300]
                        lines.append(f"TOOL → {name}({inp})")

                    # tool_result skipped intentionally

        return "\n".join(lines)

    def _write_to_memory(self, summary: str) -> None:
        """
        Prepend summary to memory.md, creating the file and directory if needed.

        Layout after write:
            # Project Memory
            <!-- header -->

            ## Session YYYY-MM-DD   ← newest (just written)
            ...
            ---

            ## Session YYYY-MM-DD   ← older
            ...
            ---
        """
        self.memory_dir.mkdir(parents=True, exist_ok=True)

        if self.memory_file.exists():
            existing = self.memory_file.read_text(encoding="utf-8")

            # Normalise: ensure the file header is present even if the file
            # was created manually or by an older version without it.
            if not existing.startswith("# Project Memory"):
                existing = _FILE_HEADER + existing

            # Find where the first session block starts so we can insert
            # the new summary before it (keeping the file header intact).
            if "## Session" in existing:
                idx = existing.index("## Session")
                # Preserve everything up to (not including) the first session
                file_header = existing[:idx]
                old_sessions = existing[idx:]
                new_content  = file_header + summary + "\n\n" + old_sessions
            else:
                # No session blocks yet — append after the file header
                new_content = existing.rstrip() + "\n\n" + summary + "\n"
        else:
            # Brand new file
            new_content = _FILE_HEADER + summary + "\n"

        self.memory_file.write_text(new_content, encoding="utf-8")
