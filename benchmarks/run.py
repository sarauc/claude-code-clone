#!/usr/bin/env python3
"""
YACC Benchmark Script — Hybrid (Mock + Live API)

Mock benchmarks: free, instant, repeatable — measure tool execution,
middleware overhead, and the full agent pipeline with a fake API client.

Live benchmarks: use real Anthropic API — measure true TTFT, latency,
and token usage. Short prompts keep cost under $0.05 total.

Usage:
    python benchmarks/run.py                # all benchmarks (requires ANTHROPIC_API_KEY)
    python benchmarks/run.py --mock-only    # free, no API key needed
    python benchmarks/run.py --live-only    # only live API scenarios
    python benchmarks/run.py --runs 5       # runs per scenario (default: 3)
    python benchmarks/run.py --out results  # output directory (default: benchmarks/results)
"""

import os
import sys
import json
import time
import argparse
import statistics
import tempfile
import shutil
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from unittest.mock import patch, MagicMock

# Add project root to path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box as rbox

console = Console()


# ─── Result types ─────────────────────────────────────────────────────────────

@dataclass
class RunMetrics:
    ttft_ms: float          # time to first assistant_message event
    total_ms: float         # time to complete/end event
    tool_exec_ms: float     # total time inside executor.execute() this run
    input_tokens: int       # from API usage (0 for mock)
    output_tokens: int      # from API usage (0 for mock)
    message_count: int      # len(state.messages) at first API call
    cache_read_tokens: int  # prompt cache hits (0 for mock)


@dataclass
class ScenarioResult:
    name: str
    mode: str               # "mock" or "live"
    description: str
    runs: List[RunMetrics] = field(default_factory=list)
    error: Optional[str] = None

    def valid_runs(self) -> List[RunMetrics]:
        return [r for r in self.runs if r is not None]

    def median(self, attr: str) -> float:
        values = [getattr(r, attr) for r in self.valid_runs()]
        return statistics.median(values) if values else 0.0

    def mean(self, attr: str) -> float:
        values = [getattr(r, attr) for r in self.valid_runs()]
        return statistics.mean(values) if values else 0.0

    def stdev(self, attr: str) -> float:
        values = [getattr(r, attr) for r in self.valid_runs()]
        return statistics.stdev(values) if len(values) > 1 else 0.0


# ─── Mock Anthropic client ────────────────────────────────────────────────────

class _MockUsage:
    def __init__(self, input_tokens=120, output_tokens=40):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0


class _MockBlock:
    """Simulates an Anthropic content block with .model_dump()."""
    def __init__(self, **kwargs):
        self._data = kwargs

    def model_dump(self) -> Dict[str, Any]:
        return self._data


class _MockResponse:
    def __init__(self, content_blocks, stop_reason="end_turn",
                 input_tokens=120, output_tokens=40):
        self.id = "msg_mock_000"
        self.type = "message"
        self.role = "assistant"
        self.model = "claude-sonnet-4-5-20250929"
        self.stop_reason = stop_reason
        self.stop_sequence = None
        self.usage = _MockUsage(input_tokens, output_tokens)
        self.content = [_MockBlock(**b) for b in content_blocks]


class MockAnthropicClient:
    """
    Drop-in replacement for anthropic.Anthropic that returns scripted responses.

    responses: list of response specs consumed in order (cycling if exhausted).
    Each spec is a dict:
        {
            "content": [{"type": "text", "text": "..."}],   # required
            "stop_reason": "end_turn",                       # optional
            "input_tokens": 120,                             # optional
            "output_tokens": 40,                             # optional
            "simulate_latency_ms": 0,                        # optional artificial delay
        }
    """

    def __init__(self, responses: List[Dict[str, Any]]):
        self._responses = responses
        self._call_idx = 0
        self.messages = self  # agent accesses client.messages.create(...)

    def create(self, **kwargs) -> _MockResponse:
        spec = self._responses[self._call_idx % len(self._responses)]
        self._call_idx += 1

        delay = spec.get("simulate_latency_ms", 0) / 1000
        if delay:
            time.sleep(delay)

        return _MockResponse(
            content_blocks=spec["content"],
            stop_reason=spec.get("stop_reason", "end_turn"),
            input_tokens=spec.get("input_tokens", 120),
            output_tokens=spec.get("output_tokens", 40),
        )

    def reset(self):
        self._call_idx = 0


# ─── Timing helpers ───────────────────────────────────────────────────────────

class ToolTimer:
    """Wraps ToolExecutor.execute to record per-call timing."""

    def __init__(self, executor):
        self._executor = executor
        self.exec_times: List[float] = []
        self._original = executor.execute

    def __enter__(self):
        def timed_execute(tool_name, tool_input):
            t0 = time.perf_counter()
            result = self._original(tool_name, tool_input)
            self.exec_times.append((time.perf_counter() - t0) * 1000)
            return result

        self._executor.execute = timed_execute
        return self

    def __exit__(self, *_):
        self._executor.execute = self._original

    @property
    def total_ms(self) -> float:
        return sum(self.exec_times)


def run_agent_timed(agent, message: str, max_turns: int = 10) -> RunMetrics:
    """
    Run agent.chat() and collect timing + token metrics.
    Returns a RunMetrics for one run.
    """
    ttft_ms = 0.0
    total_ms = 0.0
    input_tokens = 0
    output_tokens = 0
    cache_read_tokens = 0
    message_count = 0
    first_event = True

    with ToolTimer(agent.executor) as timer:
        t_start = time.perf_counter()

        for event in agent.chat(message, max_turns=max_turns):
            now_ms = (time.perf_counter() - t_start) * 1000
            etype = event.get("type")

            if etype == "assistant_message" and first_event:
                ttft_ms = now_ms
                first_event = False
                usage = event.get("usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)
                cache_read_tokens += usage.get("cache_read_input_tokens", 0)

            elif etype == "assistant_message":
                usage = event.get("usage", {})
                input_tokens += usage.get("input_tokens", 0)
                output_tokens += usage.get("output_tokens", 0)

            elif etype in ("complete", "max_turns_reached", "error"):
                total_ms = now_ms
                break

        # Capture message count from state (at end of session)
        message_count = len(agent.state.messages)

    return RunMetrics(
        ttft_ms=ttft_ms,
        total_ms=total_ms,
        tool_exec_ms=timer.total_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        message_count=message_count,
        cache_read_tokens=cache_read_tokens,
    )


# ─── Mock scenarios ───────────────────────────────────────────────────────────

def make_mock_agent(responses: List[Dict], workspace: str) -> "Agent":
    """Create an Agent with a mock Anthropic client."""
    from src.agent import Agent, AgentConfig

    config = AgentConfig(
        model="claude-sonnet-4-5-20250929",
        max_tokens=256,
        api_key="mock-key",
        workspace_path=workspace,
        enable_prompt_caching=False,   # keep mock simple
        enable_summarization=False,
    )
    agent = Agent(config)

    mock_client = MockAnthropicClient(responses)
    agent.client = mock_client
    # Also patch the client on executor (not needed, but keeps state clean)
    return agent


def bench_mock_simple_response(workspace: str, n: int) -> ScenarioResult:
    """Mock: single-turn text response, no tools. Measures pure pipeline overhead."""
    result = ScenarioResult(
        name="mock_simple_response",
        mode="mock",
        description="Single turn, text only, no tools — measures pipeline overhead",
    )
    responses = [{"content": [{"type": "text", "text": "The answer is 42."}]}]

    for _ in range(n):
        agent = make_mock_agent(responses, workspace)
        metrics = run_agent_timed(agent, "What is 6 times 7?")
        result.runs.append(metrics)

    return result


def bench_mock_single_tool(workspace: str, n: int) -> ScenarioResult:
    """Mock: one tool call then end_turn. Measures one tool round-trip."""
    result = ScenarioResult(
        name="mock_single_tool",
        mode="mock",
        description="One tool call (read_file) then final answer — measures single tool round-trip",
    )

    # Write a test file into the workspace
    test_file = Path(workspace) / "hello.py"
    test_file.write_text("def greet(name):\n    return f'Hello, {name}!'\n")

    turn1 = {
        "content": [
            {"type": "text", "text": "Let me read the file."},
            {"type": "tool_use", "id": "tu_001", "name": "read_file",
             "input": {"file_path": str(test_file)}},
        ],
        "stop_reason": "tool_use",
    }
    turn2 = {
        "content": [{"type": "text", "text": "The file defines a greet function."}],
        "stop_reason": "end_turn",
    }

    for _ in range(n):
        agent = make_mock_agent([turn1, turn2], workspace)
        metrics = run_agent_timed(agent, f"Summarize {test_file}")
        result.runs.append(metrics)

    return result


def bench_mock_multi_tool(workspace: str, n: int) -> ScenarioResult:
    """Mock: three tool calls in one turn. Measures sequential tool execution cost."""
    result = ScenarioResult(
        name="mock_multi_tool",
        mode="mock",
        description="Three read_file calls in one turn — sequential tool execution baseline",
    )

    files = []
    for i in range(3):
        p = Path(workspace) / f"module_{i}.py"
        p.write_text(f"# Module {i}\n" + "x = 1\n" * 50)
        files.append(str(p))

    turn1 = {
        "content": [
            {"type": "text", "text": "Reading all three files."},
        ] + [
            {"type": "tool_use", "id": f"tu_00{i}", "name": "read_file",
             "input": {"file_path": f}}
            for i, f in enumerate(files)
        ],
        "stop_reason": "tool_use",
    }
    turn2 = {
        "content": [{"type": "text", "text": "Done. All three modules are simple."}],
        "stop_reason": "end_turn",
    }

    for _ in range(n):
        agent = make_mock_agent([turn1, turn2], workspace)
        metrics = run_agent_timed(agent, "Read and compare all three modules.")
        result.runs.append(metrics)

    return result


def bench_mock_growing_history(workspace: str, n: int) -> ScenarioResult:
    """
    Mock: simulate a 10-turn conversation to measure latency growth
    as state.messages accumulates.
    Records per-turn TTFT by running each prefix length separately.
    """
    result = ScenarioResult(
        name="mock_growing_history",
        mode="mock",
        description="10-turn conversation — measures latency change as history grows",
    )

    text_response = {"content": [{"type": "text", "text": "OK."}], "stop_reason": "end_turn"}

    for _ in range(n):
        agent = make_mock_agent([text_response], workspace)
        # Simulate 10 sequential messages on the same agent instance
        for turn_idx in range(10):
            t0 = time.perf_counter()
            for event in agent.chat(f"Message number {turn_idx}", max_turns=2):
                if event.get("type") == "assistant_message":
                    ttft = (time.perf_counter() - t0) * 1000
                elif event.get("type") in ("complete", "error"):
                    total = (time.perf_counter() - t0) * 1000

            result.runs.append(RunMetrics(
                ttft_ms=ttft,
                total_ms=total,
                tool_exec_ms=0.0,
                input_tokens=0,
                output_tokens=0,
                message_count=len(agent.state.messages),
                cache_read_tokens=0,
            ))
            agent.client.reset()

    return result


def bench_mock_middleware_overhead(workspace: str, n: int) -> ScenarioResult:
    """Mock: measure time spent inside the middleware chain per API call."""
    result = ScenarioResult(
        name="mock_middleware_overhead",
        mode="mock",
        description="Time spent in middleware pre_process + post_process per turn",
    )
    from src.agent import Agent, AgentConfig
    from src.middleware.base import AgentState

    config = AgentConfig(
        api_key="mock-key",
        workspace_path=workspace,
        enable_prompt_caching=True,
        enable_summarization=True,
    )
    agent = Agent(config)
    fake_response = {"usage": {"input_tokens": 100, "output_tokens": 50}}

    for _ in range(n):
        state = AgentState(
            system_prompt=agent.system_prompt,
            tools=agent.tools,
            messages=[{"role": "user", "content": "hi"}],
        )
        t0 = time.perf_counter()
        state = agent.middleware.pre_process(state)
        state, _ = agent.middleware.post_process(state, fake_response)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        result.runs.append(RunMetrics(
            ttft_ms=elapsed_ms,
            total_ms=elapsed_ms,
            tool_exec_ms=0.0,
            input_tokens=0,
            output_tokens=0,
            message_count=len(state.messages),
            cache_read_tokens=0,
        ))

    return result


def bench_mock_token_estimation(workspace: str, n: int) -> ScenarioResult:
    """
    Mock: compare SummarizationMiddleware.estimate_tokens() against
    a reference count. Since we have no tokenizer, we test self-consistency
    and measure estimation speed across message sizes.
    """
    result = ScenarioResult(
        name="mock_token_estimation",
        mode="mock",
        description="Token estimator speed + accuracy vs character count reference",
    )
    from src.middleware.summarization import SummarizationMiddleware

    mid = SummarizationMiddleware()
    sizes = [100, 500, 2000, 10000, 50000]  # chars

    for _ in range(n):
        for size in sizes:
            text = "a " * (size // 2)
            msgs = [{"role": "user", "content": text}]

            t0 = time.perf_counter()
            estimated = mid.estimate_message_tokens(msgs)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            # Reference: simple char count / 4 (same formula, used to confirm no drift)
            reference = len(text) // 4
            error_pct = abs(estimated - reference) / max(reference, 1) * 100

            result.runs.append(RunMetrics(
                ttft_ms=elapsed_ms,        # repurposed: estimation time
                total_ms=error_pct,        # repurposed: estimation error %
                tool_exec_ms=float(size),  # repurposed: input size
                input_tokens=estimated,
                output_tokens=reference,
                message_count=1,
                cache_read_tokens=0,
            ))

    return result


# ─── Live scenarios ───────────────────────────────────────────────────────────

def make_live_agent(workspace: str, api_key: str) -> "Agent":
    from src.agent import Agent, AgentConfig
    return Agent(AgentConfig(
        model="claude-sonnet-4-5-20250929",
        max_tokens=128,          # keep output short to minimize cost
        api_key=api_key,
        workspace_path=workspace,
        enable_prompt_caching=True,
        enable_summarization=False,  # disable so history doesn't interfere
        enable_bash=False,           # safer for benchmarking
    ))


def bench_live_simple_response(workspace: str, api_key: str, n: int) -> ScenarioResult:
    """Live: shortest possible response. True TTFT and latency baseline."""
    result = ScenarioResult(
        name="live_simple_response",
        mode="live",
        description="Real API call — text only, no tools. True TTFT baseline.",
    )
    for _ in range(n):
        agent = make_live_agent(workspace, api_key)
        try:
            metrics = run_agent_timed(agent, 'Reply with exactly two words: "benchmark complete"')
            result.runs.append(metrics)
        except Exception as e:
            result.error = str(e)
            break

    return result


def bench_live_single_tool(workspace: str, api_key: str, n: int) -> ScenarioResult:
    """Live: one read_file call. Measures real tool round-trip latency."""
    result = ScenarioResult(
        name="live_single_tool",
        mode="live",
        description="Real API — one read_file call then answer. True tool round-trip.",
    )

    test_file = Path(workspace) / "bench_target.py"
    test_file.write_text(
        "# Benchmark target file\n"
        "ANSWER = 42\n"
        "def compute(): return ANSWER\n"
    )

    for _ in range(n):
        agent = make_live_agent(workspace, api_key)
        try:
            prompt = (
                f"Read the file at {test_file} and tell me the value of ANSWER. "
                "Be brief — one sentence."
            )
            metrics = run_agent_timed(agent, prompt)
            result.runs.append(metrics)
        except Exception as e:
            result.error = str(e)
            break

    return result


def bench_live_multi_tool(workspace: str, api_key: str, n: int) -> ScenarioResult:
    """Live: force three sequential tool calls. Baseline for parallel execution improvement."""
    result = ScenarioResult(
        name="live_multi_tool",
        mode="live",
        description="Real API — reads 3 files in sequence. Baseline for future parallel tool exec.",
    )

    file_paths = []
    for i in range(3):
        p = Path(workspace) / f"bench_file_{i}.txt"
        p.write_text(f"File {i} content. Secret number: {(i+1)*7}.\n")
        file_paths.append(str(p))

    for _ in range(n):
        agent = make_live_agent(workspace, api_key)
        try:
            paths_str = ", ".join(file_paths)
            prompt = (
                f"Read each of these files one by one: {paths_str}. "
                "Then tell me the sum of the three secret numbers. Be brief."
            )
            metrics = run_agent_timed(agent, prompt, max_turns=10)
            result.runs.append(metrics)
        except Exception as e:
            result.error = str(e)
            break

    return result


def bench_live_multi_turn(workspace: str, api_key: str, n: int) -> ScenarioResult:
    """
    Live: 4 sequential user messages on the same agent.
    Measures whether latency grows as state.messages accumulates.
    Each message is logged as a separate RunMetrics entry.
    """
    result = ScenarioResult(
        name="live_multi_turn",
        mode="live",
        description="Real API — 4 sequential messages, same agent. Measures history-growth latency.",
    )

    messages = [
        'Say "one".',
        'Say "two".',
        'Say "three".',
        'Say "four".',
    ]

    for _ in range(n):
        agent = make_live_agent(workspace, api_key)
        try:
            for msg in messages:
                metrics = run_agent_timed(agent, msg)
                result.runs.append(metrics)
        except Exception as e:
            result.error = str(e)
            break

    return result


# ─── Output ───────────────────────────────────────────────────────────────────

def print_results(results: List[ScenarioResult]):
    for r in results:
        if r.error:
            console.print(f"[red]  {r.name}: ERROR — {r.error}[/red]")
            continue

        valid = r.valid_runs()
        if not valid:
            continue

        # Special display for token estimation
        if r.name == "mock_token_estimation":
            console.print(f"\n[bold cyan]{r.name}[/bold cyan]  [dim]{r.description}[/dim]")
            console.print(f"  Estimation time: {r.median('ttft_ms'):.3f} ms median "
                          f"| Error vs reference: {r.median('total_ms'):.1f}%")
            continue

        # Special display for growing history
        if r.name == "mock_growing_history":
            console.print(f"\n[bold cyan]{r.name}[/bold cyan]  [dim]{r.description}[/dim]")
            by_count = {}
            for run in valid:
                by_count.setdefault(run.message_count, []).append(run.total_ms)
            table = Table(box=rbox.SIMPLE_HEAVY)
            table.add_column("msg count", style="dim")
            table.add_column("total_ms median", justify="right")
            for count in sorted(by_count):
                med = statistics.median(by_count[count])
                table.add_row(str(count), f"{med:.1f}")
            console.print(table)
            continue

        table = Table(box=rbox.SIMPLE_HEAVY, show_header=True)
        table.add_column("metric", style="dim")
        table.add_column("median", justify="right")
        table.add_column("mean", justify="right")
        table.add_column("stdev", justify="right")

        rows = [
            ("ttft_ms",         "Time to first token (ms)"),
            ("total_ms",        "Total turn latency (ms)"),
            ("tool_exec_ms",    "Tool exec time (ms)"),
        ]
        if r.mode == "live":
            rows += [
                ("input_tokens",    "Input tokens"),
                ("output_tokens",   "Output tokens"),
                ("cache_read_tokens", "Cache read tokens"),
                ("message_count",   "Messages in state"),
            ]

        color = "cyan" if r.mode == "mock" else "green"
        console.print(f"\n[bold {color}]{r.name}[/bold {color}]  [dim]{r.description}[/dim]")
        for attr, label in rows:
            med = r.median(attr)
            mn  = r.mean(attr)
            sd  = r.stdev(attr)
            if med > 0 or r.mode == "live":
                table.add_row(label, f"{med:.2f}", f"{mn:.2f}", f"{sd:.2f}")
        console.print(table)


def save_results(results: List[ScenarioResult], out_dir: str):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(out_dir) / f"benchmark_{ts}.json"

    payload = {
        "timestamp": ts,
        "scenarios": [
            {
                "name": r.name,
                "mode": r.mode,
                "description": r.description,
                "error": r.error,
                "runs": [asdict(run) for run in r.runs],
                "summary": {
                    "ttft_ms_median": r.median("ttft_ms"),
                    "total_ms_median": r.median("total_ms"),
                    "tool_exec_ms_median": r.median("tool_exec_ms"),
                    "input_tokens_median": r.median("input_tokens"),
                    "output_tokens_median": r.median("output_tokens"),
                    "n_runs": len(r.valid_runs()),
                } if not r.error else {}
            }
            for r in results
        ]
    }

    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="YACC Benchmark Suite")
    parser.add_argument("--mock-only", action="store_true",
                        help="Run only mock benchmarks (no API key needed)")
    parser.add_argument("--live-only", action="store_true",
                        help="Run only live API benchmarks")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of runs per scenario (default: 3)")
    parser.add_argument("--out", default=str(ROOT / "benchmarks" / "results"),
                        help="Output directory for JSON results")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    run_live = not args.mock_only
    run_mock = not args.live_only

    if run_live and not api_key:
        if args.live_only:
            console.print("[red]ANTHROPIC_API_KEY not set. Cannot run live benchmarks.[/red]")
            sys.exit(1)
        console.print("[yellow]ANTHROPIC_API_KEY not set — skipping live benchmarks.[/yellow]")
        run_live = False

    console.print(Panel(
        f"YACC Benchmark Suite\n"
        f"Runs per scenario: [bold]{args.runs}[/bold]   "
        f"Mock: [bold]{'yes' if run_mock else 'no'}[/bold]   "
        f"Live: [bold]{'yes' if run_live else 'no'}[/bold]",
        style="bold blue"
    ))

    workspace = tempfile.mkdtemp(prefix="yacc_bench_")
    results: List[ScenarioResult] = []

    try:
        # ── Mock benchmarks ──────────────────────────────────────────────────
        if run_mock:
            console.print("\n[bold]Running mock benchmarks...[/bold]")
            mock_scenarios = [
                ("Pipeline overhead",       bench_mock_simple_response),
                ("Single tool round-trip",  bench_mock_single_tool),
                ("Multi-tool sequential",   bench_mock_multi_tool),
                ("History growth",          bench_mock_growing_history),
                ("Middleware overhead",     bench_mock_middleware_overhead),
                ("Token estimation",        bench_mock_token_estimation),
            ]
            for label, fn in mock_scenarios:
                console.print(f"  {label}...", end=" ")
                try:
                    r = fn(workspace, args.runs)
                    results.append(r)
                    console.print("[green]done[/green]")
                except Exception as e:
                    console.print(f"[red]ERROR: {e}[/red]")

        # ── Live benchmarks ──────────────────────────────────────────────────
        if run_live:
            console.print("\n[bold]Running live API benchmarks...[/bold]")
            console.print("[dim]  Estimated cost: < $0.05 total[/dim]")
            live_scenarios = [
                ("Simple response (baseline)", bench_live_simple_response),
                ("Single tool call",           bench_live_single_tool),
                ("Multi-tool sequential",      bench_live_multi_tool),
                ("Multi-turn history",         bench_live_multi_turn),
            ]
            for label, fn in live_scenarios:
                console.print(f"  {label}...", end=" ")
                try:
                    r = fn(workspace, api_key, args.runs)
                    results.append(r)
                    if r.error:
                        console.print(f"[red]ERROR: {r.error}[/red]")
                    else:
                        console.print("[green]done[/green]")
                except Exception as e:
                    console.print(f"[red]ERROR: {e}[/red]")

        # ── Print results ────────────────────────────────────────────────────
        console.print("\n" + "─" * 70)
        console.print("[bold]Results[/bold]")
        print_results(results)

        # ── Save results ─────────────────────────────────────────────────────
        out_path = save_results(results, args.out)
        console.print(f"\n[dim]Saved: {out_path}[/dim]")

    finally:
        shutil.rmtree(workspace, ignore_errors=True)


if __name__ == "__main__":
    main()
