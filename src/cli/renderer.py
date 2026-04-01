"""
Rich-based renderer for CLI output.
Handles all visual formatting including todos, tool calls, thinking, etc.
"""

import time
from typing import Dict, List, Any, Optional
from rich.console import Console, Group
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.live import Live
from rich.spinner import Spinner
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.rule import Rule
from rich.tree import Tree
from rich.style import Style
from rich.box import ROUNDED, SIMPLE, MINIMAL


console = Console()


# ═══════════════════════════════════════════════════════════════════════════════
# Styles
# ═══════════════════════════════════════════════════════════════════════════════

STYLES = {
    "thinking": Style(color="bright_black", italic=True),
    "tool_name": Style(color="cyan", bold=True),
    "tool_result": Style(color="green"),
    "error": Style(color="red", bold=True),
    "warning": Style(color="yellow"),
    "success": Style(color="green", bold=True),
    "info": Style(color="blue"),
    "muted": Style(color="bright_black"),
    "highlight": Style(color="magenta", bold=True),
    "user": Style(color="bright_blue", bold=True),
    "assistant": Style(color="bright_green"),
    "path": Style(color="cyan", underline=True),
    "number": Style(color="yellow"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Spinner & Progress
# ═══════════════════════════════════════════════════════════════════════════════

SPINNER_STYLES = [
    "dots", "dots2", "dots3", "dots4", "dots5", "dots6", "dots7", "dots8", "dots9",
    "dots10", "dots11", "dots12", "line", "line2", "pipe", "simpleDots",
    "simpleDotsScrolling", "star", "star2", "flip", "hamburger", "growVertical",
    "growHorizontal", "balloon", "balloon2", "noise", "bounce", "boxBounce",
    "boxBounce2", "triangle", "arc", "circle", "squareCorners", "circleQuarters",
    "circleHalves", "squish", "toggle", "toggle2", "toggle3", "toggle4", "toggle5",
    "toggle6", "toggle7", "toggle8", "toggle9", "toggle10", "toggle11", "toggle12",
    "toggle13", "arrow", "arrow2", "arrow3", "bouncingBar", "bouncingBall",
    "smiley", "monkey", "hearts", "clock", "earth", "material", "moon", "runner",
    "pong", "shark", "dqpb", "weather", "christmas", "grenade", "point", "layer",
    "betaWave", "fingerDance", "fistBump", "soccerHeader", "mindblown", "speaker",
    "orangePulse", "bluePulse", "orangeBluePulse", "timeTravel", "aesthetic",
]


class ThinkingSpinner:
    """Animated spinner for thinking/processing states."""
    
    def __init__(self, text: str = "Thinking"):
        self.text = text
        self.spinner_chars = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.idx = 0
    
    def __rich__(self):
        char = self.spinner_chars[self.idx % len(self.spinner_chars)]
        self.idx += 1
        return Text(f"{char} {self.text}...", style=STYLES["thinking"])


class ToolSpinner:
    """Spinner for tool execution."""
    
    def __init__(self, tool_name: str, args_preview: str = ""):
        self.tool_name = tool_name
        self.args_preview = args_preview
        self.frames = ["◐", "◓", "◑", "◒"]
        self.idx = 0
    
    def __rich__(self):
        frame = self.frames[self.idx % len(self.frames)]
        self.idx += 1
        text = Text()
        text.append(f"  {frame} ", style="yellow")
        text.append(self.tool_name, style=STYLES["tool_name"])
        if self.args_preview:
            text.append(f" {self.args_preview}", style=STYLES["muted"])
        return text


# ═══════════════════════════════════════════════════════════════════════════════
# Todo Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_todo_item(todo: Dict[str, Any]) -> Text:
    """Render a single todo item."""
    status = todo.get("status", "pending")
    content = todo.get("content", "")
    
    text = Text()
    
    if status == "completed":
        text.append("  ✓ ", style="green bold")
        text.append(content, style="green dim strike")
    elif status == "in_progress":
        text.append("  ◉ ", style="yellow bold")
        text.append(content, style="yellow")
    else:  # pending
        text.append("  ○ ", style="bright_black")
        text.append(content, style="bright_black")
    
    return text


def render_todos(todos: List[Dict[str, Any]], title: str = "📋 Tasks") -> Panel:
    """Render a todo list panel."""
    if not todos:
        return Panel(
            Text("No tasks", style=STYLES["muted"]),
            title=title,
            border_style="bright_black",
            box=ROUNDED
        )
    
    lines = []
    for todo in todos:
        lines.append(render_todo_item(todo))
    
    # Add progress indicator
    completed = sum(1 for t in todos if t.get("status") == "completed")
    total = len(todos)
    progress_text = Text(f"\n  {completed}/{total} completed", style=STYLES["muted"])
    lines.append(progress_text)
    
    return Panel(
        Group(*lines),
        title=title,
        border_style="blue",
        box=ROUNDED
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Tool Call Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def format_tool_args(args: Dict[str, Any], max_length: int = 60) -> str:
    """Format tool arguments for display."""
    if not args:
        return ""
    
    parts = []
    for key, value in args.items():
        val_str = str(value)
        if len(val_str) > 30:
            val_str = val_str[:27] + "..."
        parts.append(f"{key}={val_str}")
    
    result = ", ".join(parts)
    if len(result) > max_length:
        result = result[:max_length - 3] + "..."
    
    return result


def render_tool_call(tool_name: str, args: Dict[str, Any], result: Optional[str] = None, is_error: bool = False) -> Panel:
    """Render a tool call with optional result."""
    # Icon based on tool type
    icons = {
        "ls": "📁",
        "read_file": "📄",
        "write_file": "✍️",
        "edit_file": "📝",
        "glob": "🔍",
        "grep": "🔎",
        "bash": "💻",
        "write_todos": "📋",
        "task": "🤖",
        "web_search": "🌐",
    }
    icon = icons.get(tool_name, "🔧")
    
    content = Text()
    content.append(f"{icon} ", style="bold")
    content.append(tool_name, style=STYLES["tool_name"])
    
    args_str = format_tool_args(args)
    if args_str:
        content.append(f"\n   {args_str}", style=STYLES["muted"])
    
    if result is not None:
        content.append("\n")
        if is_error:
            content.append(f"   ✗ {result[:200]}", style=STYLES["error"])
        else:
            result_preview = result[:200] + "..." if len(result) > 200 else result
            content.append(f"   → {result_preview}", style=STYLES["tool_result"])
    
    border_style = "red" if is_error else "cyan"
    return Panel(content, border_style=border_style, box=MINIMAL, padding=(0, 1))


def render_tool_start(tool_name: str, args: Dict[str, Any]) -> Text:
    """Render tool execution start."""
    icons = {
        "ls": "📁", "read_file": "📄", "write_file": "✍️", "edit_file": "📝",
        "glob": "🔍", "grep": "🔎", "bash": "💻", "write_todos": "📋",
        "task": "🤖", "web_search": "🌐",
    }
    icon = icons.get(tool_name, "🔧")
    
    text = Text()
    text.append(f"  {icon} ", style="bold")
    text.append(tool_name, style=STYLES["tool_name"])
    
    # Show key args
    if tool_name in ["ls", "read_file", "write_file", "edit_file"]:
        path = args.get("file_path") or args.get("path", "")
        if path:
            text.append(f" {path}", style=STYLES["path"])
    elif tool_name == "grep":
        pattern = args.get("pattern", "")
        text.append(f' "{pattern}"', style="yellow")
    elif tool_name == "glob":
        pattern = args.get("pattern", "")
        text.append(f" {pattern}", style="yellow")
    elif tool_name == "bash":
        cmd = args.get("command", "")
        if len(cmd) > 40:
            cmd = cmd[:37] + "..."
        text.append(f" $ {cmd}", style=STYLES["muted"])
    
    return text


def render_tool_result(result: str, is_error: bool = False) -> Text:
    """Render tool result."""
    text = Text()
    
    if is_error:
        text.append("    ✗ ", style=STYLES["error"])
        preview = result[:150] + "..." if len(result) > 150 else result
        text.append(preview, style=STYLES["error"])
    else:
        text.append("    ✓ ", style=STYLES["success"])
        # Show truncated result
        lines = result.split("\n")
        if len(lines) > 3:
            preview = "\n      ".join(lines[:3]) + f"\n      ... ({len(lines) - 3} more lines)"
        elif len(result) > 150:
            preview = result[:147] + "..."
        else:
            preview = result
        text.append(preview, style=STYLES["muted"])
    
    return text


# ═══════════════════════════════════════════════════════════════════════════════
# Message Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_thinking(text: str) -> Text:
    """Render thinking/reasoning text in gray italic."""
    result = Text()
    result.append("  💭 ", style=STYLES["thinking"])
    # Truncate if too long
    if len(text) > 300:
        text = text[:297] + "..."
    result.append(text, style=STYLES["thinking"])
    return result


def render_assistant_message(content: str) -> Panel:
    """Render assistant response."""
    # Try to render as markdown
    try:
        md = Markdown(content)
        return Panel(
            md,
            title="🤖 Assistant",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1)
        )
    except Exception:
        return Panel(
            Text(content),
            title="🤖 Assistant",
            border_style="green",
            box=ROUNDED,
            padding=(0, 1)
        )


def render_user_message(content: str) -> Text:
    """Render user input."""
    text = Text()
    text.append("❯ ", style=STYLES["user"])
    text.append(content, style="bold")
    return text


def render_code_block(code: str, language: str = "python") -> Syntax:
    """Render a code block with syntax highlighting."""
    return Syntax(code, language, theme="monokai", line_numbers=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Status Indicators
# ═══════════════════════════════════════════════════════════════════════════════

def render_status(message: str, status: str = "info") -> Text:
    """Render a status message."""
    icons = {
        "info": ("ℹ", "blue"),
        "success": ("✓", "green"),
        "warning": ("⚠", "yellow"),
        "error": ("✗", "red"),
        "thinking": ("◌", "bright_black"),
    }
    icon, color = icons.get(status, ("•", "white"))
    
    text = Text()
    text.append(f"  {icon} ", style=f"{color} bold")
    text.append(message, style=color)
    return text


def render_turn_header(turn: int) -> Rule:
    """Render turn separator."""
    return Rule(f"Turn {turn}", style="bright_black")


def render_completion(turns: int, tokens_in: int, tokens_out: int) -> Panel:
    """Render completion summary."""
    content = Text()
    content.append("✨ Completed", style=STYLES["success"])
    content.append(f" in {turns} turn(s)\n", style="white")
    content.append(f"   📊 Tokens: ", style=STYLES["muted"])
    content.append(f"{tokens_in:,}", style=STYLES["number"])
    content.append(" in / ", style=STYLES["muted"])
    content.append(f"{tokens_out:,}", style=STYLES["number"])
    content.append(" out", style=STYLES["muted"])
    
    return Panel(content, border_style="green", box=SIMPLE)


# ═══════════════════════════════════════════════════════════════════════════════
# File Tree Renderer
# ═══════════════════════════════════════════════════════════════════════════════

def render_file_tree(files: List[str], title: str = "Files") -> Tree:
    """Render a file tree."""
    tree = Tree(f"📁 {title}", style="bold")
    
    # Build tree structure
    paths = {}
    for file_path in sorted(files):
        parts = file_path.strip("/").split("/")
        current = paths
        for part in parts:
            if part not in current:
                current[part] = {}
            current = current[part]
    
    def add_to_tree(node: Tree, items: Dict):
        for name, children in sorted(items.items()):
            if children:
                branch = node.add(f"📁 {name}")
                add_to_tree(branch, children)
            else:
                node.add(f"📄 {name}")
    
    add_to_tree(tree, paths)
    return tree


# ═══════════════════════════════════════════════════════════════════════════════
# Main Console Helper
# ═══════════════════════════════════════════════════════════════════════════════

class CLIRenderer:
    """Main renderer class that manages console output."""
    
    def __init__(self):
        self.console = Console()
        self.live = None
        self.total_input_tokens = 0
        self.total_output_tokens = 0
    
    def print(self, *args, **kwargs):
        """Print to console."""
        self.console.print(*args, **kwargs)
    
    def clear(self):
        """Clear the console."""
        self.console.clear()
    
    def print_welcome(self, workspace: str):
        """Print welcome message."""
        self.console.print()
        self.console.print(Panel(
            Text.from_markup(
                "[bold cyan]Yet Another Claude Code[/bold cyan]\n"
                "[dim]A Claude Code-like AI coding assistant[/dim]\n\n"
                f"[bold]Workspace:[/bold] [cyan]{workspace}[/cyan]\n"
                "[dim]Type your request or 'exit' to quit[/dim]"
            ),
            box=ROUNDED,
            border_style="cyan",
            padding=(1, 2)
        ))
        self.console.print()
    
    def print_user_input(self, text: str):
        """Print user input."""
        self.console.print()
        self.console.print(render_user_message(text))
        self.console.print()
    
    def print_thinking(self, text: str):
        """Print thinking text."""
        self.console.print(render_thinking(text))
    
    def print_assistant(self, text: str):
        """Print assistant response."""
        if text.strip():
            self.console.print(render_assistant_message(text))
    
    def print_tool_start(self, tool_name: str, args: Dict[str, Any]):
        """Print tool execution start."""
        self.console.print(render_tool_start(tool_name, args))
    
    def print_tool_result(self, result: str, is_error: bool = False):
        """Print tool result."""
        self.console.print(render_tool_result(result, is_error))
    
    def print_todos(self, todos: List[Dict[str, Any]]):
        """Print todo list."""
        if todos:
            self.console.print(render_todos(todos))
    
    def print_turn(self, turn: int):
        """Print turn header."""
        self.console.print()
        self.console.print(render_turn_header(turn))
    
    def print_status(self, message: str, status: str = "info"):
        """Print status message."""
        self.console.print(render_status(message, status))
    
    def print_completion(self, turns: int):
        """Print completion message."""
        self.console.print()
        self.console.print(render_completion(
            turns,
            self.total_input_tokens,
            self.total_output_tokens
        ))
    
    def print_error(self, message: str):
        """Print error message."""
        self.console.print(render_status(message, "error"))
    
    def start_spinner(self, text: str = "Thinking"):
        """Start a thinking spinner."""
        spinner = ThinkingSpinner(text)
        self.live = Live(spinner, console=self.console, refresh_per_second=10)
        self.live.start()
    
    def stop_spinner(self):
        """Stop the spinner."""
        if self.live:
            self.live.stop()
            self.live = None
    
    def update_tokens(self, input_tokens: int, output_tokens: int):
        """Update token counts."""
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

    # ── Streaming helpers ────────────────────────────────────────────────────
    # Three-phase API called by app.py when enable_streaming=True:
    #   1. print_stream_start()  — called once before the first delta arrives
    #   2. print_stream_delta()  — called per token during streaming
    #   3. print_stream_end()    — called once after the stream closes

    def print_stream_start(self):
        """
        Print the assistant prefix line before the first streaming token.

        We cannot wrap streaming output in a Panel (panels require the full
        text up front), so we print a plain prefix and let deltas follow
        on the same line.
        """
        self.console.print()
        self.console.print("🤖 ", end="", style="bold green")

    def print_stream_delta(self, delta: str):
        """
        Print a single token inline with no trailing newline.

        highlight=False prevents rich from applying syntax colouring to
        partial tokens, which would produce garbled markup mid-stream.
        markup=False prevents rich from misinterpreting [ ] in code as tags.
        """
        self.console.print(delta, end="", highlight=False, markup=False,
                           style=STYLES["assistant"])

    def print_stream_end(self):
        """
        Print a closing newline after the last token.

        Without this the next line of output (tool calls, spinner, prompt)
        would appear on the same line as the last streamed token.
        """
        self.console.print()  # move to next line

