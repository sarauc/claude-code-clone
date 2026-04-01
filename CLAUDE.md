# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable)
pip install -e .

# Install with dev dependencies
pip install -e ".[dev]"

# Run the CLI
yacc                          # Interactive mode
yacc "your prompt here"       # Single command
yacc -w /path/to/project      # Specify workspace
yacc --debug                  # Debug mode

# Lint
ruff check src/
black --check src/

# Format
black src/
ruff check --fix src/

# Tests
pytest
pytest tests/test_specific.py::test_name  # Single test
```

## Architecture

The codebase is a Python CLI AI coding assistant (`yacc`) built on the Anthropic SDK without LangChain.

### Request flow

1. **`src/cli/app.py`** — Entry point (`main` -> `YACCCLI`). Handles the REPL loop, streams events from `Agent.chat()`, and renders them via `CLIRenderer`.
2. **`src/agent.py`** — Core `Agent` class. `chat()` is a generator that loops: call API -> yield events -> execute tools -> feed results back. The loop continues until `stop_reason == "end_turn"` or `max_turns` is hit.
3. **Middleware pipeline** (`src/middleware/`) — Applied around every API call:
   - `AnthropicPromptCachingMiddleware` — adds `cache_control` to system prompt and tool definitions (pre-process only)
   - `SummarizationMiddleware` — compresses old messages when estimated tokens exceed 170k (pre-process), tracks token counts (post-process)
   - `PatchToolCallsMiddleware` — fixes dangling tool calls (tool_use with no tool_result) before each API call; runs **last** in pre-process order (always appended last in `Agent.__init__`)
4. **`src/tools/executor.py`** — `ToolExecutor` dispatches tool calls by name to handler methods. `edit_file` requires the file to have been previously read via `read_file` (enforced by `files_read` set).
5. **`src/tools/definitions.py`** — Tool JSON schemas for the Anthropic API.
6. **`src/prompts/system.py`** — Builds the system prompt string.

### Key design points

- `MiddlewareChain.pre_process` runs in **forward** order; `post_process` runs in **reverse** order.
- `AgentState` is the single shared state object passed through all middleware. It holds messages, token counts, tool state, and cache breakpoints.
- Middleware ordering in `Agent.__init__`: `PromptCaching` -> `Summarization` -> `PatchToolCalls`.
- `ToolExecutor` paths: all relative paths are resolved against `workspace_path`. Paths starting with `/workspace` are rewritten to the real workspace root.
- `use_virtual_fs=True` on `ToolExecutor` enables an in-memory filesystem (used for testing).
- Line length is 100 chars (`black` and `ruff` are both configured to 100).
