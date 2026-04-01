#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Streaming vs Non-Streaming Comparison Benchmark
================================================
Measures the TTFT (Time to First Token) improvement that streaming provides.

Uses a simulated API client — no real API key required.

The simulation models realistic Claude behaviour:
  - first_token_delay_ms  : time before any text starts (network + model startup)
  - token_interval_ms     : time between successive tokens once generation begins
  - n_tokens              : total tokens in the response

  Non-streaming: client.messages.create() blocks for the full duration
                 (first_token_delay + n_tokens * token_interval)

  Streaming:     client.messages.stream() yields the first token after
                 first_token_delay, then one token every token_interval.
                 → TTFT ≈ first_token_delay  (instead of the full wait)

Output:
  • Rich table printed to console
  • PNG bar chart saved to docs/streaming_comparison.png
"""

import os
import sys
import time
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from rich.console import Console
from rich.table import Table
from rich import box as rbox

console = Console()


# ─── Simulated API client ─────────────────────────────────────────────────────

class _MockUsage:
    def __init__(self, inp, out):
        self.input_tokens = inp
        self.output_tokens = out


class _MockBlock:
    def __init__(self, **kwargs):
        self._data = kwargs
    def model_dump(self):
        return self._data


class _MockMessage:
    def __init__(self, text, n_tokens):
        self.id = "sim_msg"
        self.type = "message"
        self.role = "assistant"
        self.model = "claude-sonnet-4-5-20250929"
        self.stop_reason = "end_turn"
        self.stop_sequence = None
        self.usage = _MockUsage(n_tokens * 4, n_tokens)
        self.content = [_MockBlock(type="text", text=text)]


class _MockStreamCtx:
    """
    Context manager returned by SimulatedAPIClient.stream().

    Simulates realistic streaming:
      - Sleeps first_token_delay before yielding the first token.
      - Sleeps token_interval between every subsequent token.
    """
    def __init__(self, tokens: List[str], first_delay_s: float, interval_s: float,
                 full_text: str, n_tokens: int):
        self._tokens = tokens
        self._first_delay = first_delay_s
        self._interval = interval_s
        self._full_text = full_text
        self._n_tokens = n_tokens

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    @property
    def text_stream(self):
        """Yields token strings with simulated network delays between them."""
        for i, token in enumerate(self._tokens):
            # First token waits for model startup; subsequent tokens wait
            # only for the inter-token generation interval.
            time.sleep(self._first_delay if i == 0 else self._interval)
            yield token

    def get_final_message(self) -> _MockMessage:
        """Returns the complete message after the stream is exhausted."""
        return _MockMessage(self._full_text, self._n_tokens)


class SimulatedAPIClient:
    """
    Drop-in replacement for anthropic.Anthropic that simulates realistic
    response timing without making any real API calls.

    Args:
        first_token_delay_ms  : milliseconds before the first token arrives
        token_interval_ms     : milliseconds between subsequent tokens
        n_tokens              : number of tokens in the response
        response_text         : the text content of the response
    """

    def __init__(self, first_token_delay_ms: float, token_interval_ms: float,
                 n_tokens: int, response_text: str = None):
        self._first_delay = first_token_delay_ms / 1000
        self._interval = token_interval_ms / 1000
        self._n_tokens = n_tokens
        self._text = response_text or ("word " * n_tokens).strip()
        # Split text into n_tokens roughly equal chunks to simulate token stream
        words = self._text.split()
        step = max(1, len(words) // n_tokens)
        self._tokens = [" ".join(words[i:i+step]) + " "
                        for i in range(0, len(words), step)][:n_tokens]

        self.messages = self  # agent accesses client.messages.create/stream

    def create(self, **kwargs) -> _MockMessage:
        """
        Blocking call. Waits for the ENTIRE response before returning.
        This is what happens today without streaming.
        Total wait = first_token_delay + n_tokens * token_interval
        """
        total_s = self._first_delay + self._n_tokens * self._interval
        time.sleep(total_s)
        return _MockMessage(self._text, self._n_tokens)

    def stream(self, **kwargs) -> _MockStreamCtx:
        """
        Streaming call. Returns a context manager whose text_stream generator
        yields tokens progressively, sleeping realistically between them.
        TTFT ≈ first_token_delay (vs full total for blocking).
        """
        return _MockStreamCtx(
            tokens=self._tokens,
            first_delay_s=self._first_delay,
            interval_s=self._interval,
            full_text=self._text,
            n_tokens=self._n_tokens,
        )


# ─── Scenario definition ──────────────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    description: str
    first_token_delay_ms: float   # model startup latency
    token_interval_ms: float      # ms per token after first
    n_tokens: int                 # total tokens in response

    @property
    def total_ms(self) -> float:
        return self.first_token_delay_ms + self.n_tokens * self.token_interval_ms


# Scenarios model realistic Claude response profiles:
# Short answers have lower total latency; long ones show the biggest
# relative TTFT improvement from streaming.
SCENARIOS = [
    Scenario(
        name="Short answer",
        description="Quick factual reply\n~10 tokens, 300ms startup",
        first_token_delay_ms=300,
        token_interval_ms=30,
        n_tokens=10,
    ),
    Scenario(
        name="Medium answer",
        description="Explanation or summary\n~40 tokens, 400ms startup",
        first_token_delay_ms=400,
        token_interval_ms=35,
        n_tokens=40,
    ),
    Scenario(
        name="Long answer",
        description="Detailed technical response\n~100 tokens, 500ms startup",
        first_token_delay_ms=500,
        token_interval_ms=40,
        n_tokens=100,
    ),
    Scenario(
        name="Code generation",
        description="Code block output\n~150 tokens, 450ms startup",
        first_token_delay_ms=450,
        token_interval_ms=38,
        n_tokens=150,
    ),
]


# ─── Measurement ─────────────────────────────────────────────────────────────

@dataclass
class ModeResult:
    ttft_samples: List[float]
    total_samples: List[float]

    @property
    def ttft_median(self) -> float:
        return statistics.median(self.ttft_samples)

    @property
    def total_median(self) -> float:
        return statistics.median(self.total_samples)


def measure_blocking(scenario: Scenario, n_runs: int) -> ModeResult:
    """
    Measure TTFT and total latency in NON-STREAMING (blocking) mode.

    With blocking, TTFT == total latency because the client.messages.create()
    call does not return until the full response is ready. The CLI sees
    nothing until that moment.
    """
    from src.agent import Agent, AgentConfig

    ttft_samples, total_samples = [], []

    for _ in range(n_runs):
        agent = Agent(AgentConfig(
            api_key="sim-key",
            enable_streaming=False,      # ← blocking path: _call_api()
            enable_prompt_caching=False,
            enable_summarization=False,
        ))
        agent.client = SimulatedAPIClient(
            first_token_delay_ms=scenario.first_token_delay_ms,
            token_interval_ms=scenario.token_interval_ms,
            n_tokens=scenario.n_tokens,
        )
        agent.client.messages = agent.client  # wire up messages alias

        ttft = None
        t0 = time.perf_counter()

        for event in agent.chat("benchmark", max_turns=2):
            now = (time.perf_counter() - t0) * 1000
            etype = event.get("type")

            if etype == "assistant_message" and ttft is None:
                # Without streaming, assistant_message is the FIRST event that
                # carries any text — so TTFT == time to this event == total wait.
                ttft = now

            elif etype in ("complete", "error"):
                total_samples.append(now)
                ttft_samples.append(ttft if ttft else now)
                break

    return ModeResult(ttft_samples=ttft_samples, total_samples=total_samples)


def measure_streaming(scenario: Scenario, n_runs: int) -> ModeResult:
    """
    Measure TTFT and total latency in STREAMING mode.

    With streaming, the first text_delta event arrives after first_token_delay_ms.
    Total latency is first_delay + n_tokens * interval (same as blocking),
    but the user sees the first character dramatically sooner.
    """
    from src.agent import Agent, AgentConfig

    ttft_samples, total_samples = [], []

    for _ in range(n_runs):
        agent = Agent(AgentConfig(
            api_key="sim-key",
            enable_streaming=True,       # ← streaming path: _call_api_streaming()
            enable_prompt_caching=False,
            enable_summarization=False,
        ))
        agent.client = SimulatedAPIClient(
            first_token_delay_ms=scenario.first_token_delay_ms,
            token_interval_ms=scenario.token_interval_ms,
            n_tokens=scenario.n_tokens,
        )
        agent.client.messages = agent.client

        ttft = None
        t0 = time.perf_counter()

        for event in agent.chat("benchmark", max_turns=2):
            now = (time.perf_counter() - t0) * 1000
            etype = event.get("type")

            if etype == "text_delta" and ttft is None:
                # First token arrives — this is the true TTFT for streaming.
                # The user sees text at this moment.
                ttft = now

            elif etype in ("complete", "error"):
                total_samples.append(now)
                ttft_samples.append(ttft if ttft else now)
                break

    return ModeResult(ttft_samples=ttft_samples, total_samples=total_samples)


# ─── Display ─────────────────────────────────────────────────────────────────

def print_comparison_table(results: list):
    """Print a rich comparison table to the console."""
    table = Table(
        title="Streaming vs Non-Streaming — TTFT Comparison",
        box=rbox.ROUNDED,
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Scenario",           style="bold",        min_width=16)
    table.add_column("Total ms\n(theory)", justify="right",     style="dim")
    table.add_column("TTFT — Blocking",    justify="right",     style="red")
    table.add_column("TTFT — Streaming",   justify="right",     style="green")
    table.add_column("Improvement",        justify="right",     style="bold yellow")
    table.add_column("Total ms\nBlocking", justify="right",     style="dim")
    table.add_column("Total ms\nStreaming", justify="right",    style="dim")

    for scenario, blocking, streaming in results:
        ttft_b = blocking.ttft_median
        ttft_s = streaming.ttft_median
        improvement = ttft_b / ttft_s if ttft_s > 0 else float("inf")

        table.add_row(
            scenario.name,
            f"{scenario.total_ms:.0f}",
            f"{ttft_b:.0f} ms",
            f"{ttft_s:.0f} ms",
            f"{improvement:.1f}×",
            f"{blocking.total_median:.0f} ms",
            f"{streaming.total_median:.0f} ms",
        )

    console.print()
    console.print(table)
    console.print()
    console.print(
        "  [dim]TTFT = Time to First Token  |  "
        "Total ms is similar for both modes — streaming doesn't speed up generation, "
        "it just shows the first token sooner[/dim]"
    )
    console.print()


def save_chart(results: list, out_path: str):
    """Save a grouped bar chart comparing TTFT and total latency."""
    labels = [r[0].name for r in results]
    ttft_blocking  = [r[1].ttft_median  for r in results]
    ttft_streaming = [r[2].ttft_median  for r in results]
    total_blocking = [r[1].total_median for r in results]

    n = len(labels)
    x = np.arange(n)
    bar_w = 0.28

    # ── Colours ──
    C_BG      = "#0d1117"
    C_PANEL   = "#161b22"
    C_BORDER  = "#30363d"
    C_RED     = "#f85149"
    C_GREEN   = "#3fb950"
    C_GRAY    = "#6e7681"
    C_TEXT    = "#e6edf3"
    C_MUTED   = "#8b949e"
    C_YELLOW  = "#d29922"

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor(C_BG)
    fig.suptitle("Streaming vs Non-Streaming  —  TTFT & Latency Comparison",
                 color=C_TEXT, fontsize=13, fontweight="bold", y=1.01)

    # ── Left chart: TTFT comparison ──────────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor(C_PANEL)
    for sp in ax1.spines.values(): sp.set_color(C_BORDER)
    ax1.tick_params(colors=C_MUTED)
    ax1.yaxis.label.set_color(C_MUTED)

    bars_b = ax1.bar(x - bar_w/2, ttft_blocking,  bar_w, label="Non-Streaming TTFT",
                     color=C_RED,   alpha=0.85, zorder=3)
    bars_s = ax1.bar(x + bar_w/2, ttft_streaming, bar_w, label="Streaming TTFT",
                     color=C_GREEN, alpha=0.85, zorder=3)

    # Improvement labels above streaming bars
    for xi, (b, s) in enumerate(zip(ttft_blocking, ttft_streaming)):
        improvement = b / s if s > 0 else 0
        ax1.text(xi + bar_w/2, s + max(ttft_blocking) * 0.02,
                 f"{improvement:.1f}×", ha="center", va="bottom",
                 color=C_YELLOW, fontsize=9, fontweight="bold")

    # Value labels inside bars
    for bar in bars_b:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h/2,
                 f"{h:.0f}", ha="center", va="center",
                 color="white", fontsize=8, fontweight="bold")
    for bar in bars_s:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h/2,
                 f"{h:.0f}", ha="center", va="center",
                 color="white", fontsize=8, fontweight="bold")

    ax1.set_title("Time to First Token (TTFT)", color=C_TEXT, fontsize=11, pad=10)
    ax1.set_ylabel("Milliseconds", color=C_MUTED)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, color=C_TEXT, fontsize=9)
    ax1.yaxis.set_tick_params(labelcolor=C_MUTED)
    ax1.grid(axis="y", color=C_BORDER, linewidth=0.7, zorder=0)
    ax1.set_axisbelow(True)
    ax1.legend(facecolor=C_PANEL, edgecolor=C_BORDER,
               fontsize=9)

    # Annotation box
    ax1.text(0.02, 0.97,
             "Lower is better.\nStreaming shows first token\nmuch sooner.",
             transform=ax1.transAxes, va="top", ha="left",
             color=C_MUTED, fontsize=8,
             bbox=dict(facecolor=C_BG, edgecolor=C_BORDER, boxstyle="round,pad=0.4"))

    # ── Right chart: Total latency vs TTFT breakdown ─────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(C_PANEL)
    for sp in ax2.spines.values(): sp.set_color(C_BORDER)
    ax2.tick_params(colors=C_MUTED)

    # Stacked bars for streaming: TTFT (first token wait) + remaining generation
    ttft_s_arr  = np.array(ttft_streaming)
    total_s_arr = np.array([r[2].total_median for r in results])
    remaining   = total_s_arr - ttft_s_arr

    ax2.bar(x - bar_w/2, total_blocking, bar_w, label="Non-Streaming (full wait)",
            color=C_RED, alpha=0.85, zorder=3)

    ax2.bar(x + bar_w/2, ttft_s_arr,  bar_w, label="Streaming: wait for 1st token",
            color=C_GREEN, alpha=0.85, zorder=3)
    ax2.bar(x + bar_w/2, remaining, bar_w, bottom=ttft_s_arr,
            label="Streaming: remaining generation", color="#238636", alpha=0.55, zorder=3)

    # Total label on top of streaming stacked bar
    for xi, tot in enumerate(total_s_arr):
        ax2.text(xi + bar_w/2, tot + max(total_blocking) * 0.01,
                 f"{tot:.0f}", ha="center", va="bottom",
                 color=C_MUTED, fontsize=8)

    ax2.set_title("Total Latency Breakdown", color=C_TEXT, fontsize=11, pad=10)
    ax2.set_ylabel("Milliseconds", color=C_MUTED)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, color=C_TEXT, fontsize=9)
    ax2.yaxis.set_tick_params(labelcolor=C_MUTED)
    ax2.grid(axis="y", color=C_BORDER, linewidth=0.7, zorder=0)
    ax2.set_axisbelow(True)
    ax2.legend(facecolor=C_PANEL, edgecolor=C_BORDER,
               fontsize=8)

    ax2.text(0.02, 0.97,
             "Total generation time is\nsimilar — streaming just\nreveals it progressively.",
             transform=ax2.transAxes, va="top", ha="left",
             color=C_MUTED, fontsize=8,
             bbox=dict(facecolor=C_BG, edgecolor=C_BORDER, boxstyle="round,pad=0.4"))

    plt.tight_layout(pad=1.5)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=C_BG)
    return out_path


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    N_RUNS = 3  # runs per scenario per mode
    CHART_OUT = str(ROOT / "docs" / "streaming_comparison.png")

    console.print()
    console.print("[bold cyan]Streaming vs Non-Streaming Comparison[/bold cyan]")
    console.print(f"[dim]{N_RUNS} runs per scenario · Simulated API · No real API calls[/dim]")
    console.print()

    results = []
    for scenario in SCENARIOS:
        console.print(f"  [dim]Measuring:[/dim] [bold]{scenario.name}[/bold] "
                      f"[dim]({scenario.n_tokens} tokens, "
                      f"{scenario.first_token_delay_ms:.0f}ms startup, "
                      f"total≈{scenario.total_ms:.0f}ms)[/dim]")

        console.print("    non-streaming...", end=" ")
        blocking = measure_blocking(scenario, N_RUNS)
        console.print(f"[red]TTFT {blocking.ttft_median:.0f}ms[/red]", end="   ")

        console.print("streaming...", end=" ")
        streaming = measure_streaming(scenario, N_RUNS)
        console.print(f"[green]TTFT {streaming.ttft_median:.0f}ms[/green]  "
                      f"[yellow]{blocking.ttft_median/streaming.ttft_median:.1f}× faster[/yellow]")

        results.append((scenario, blocking, streaming))

    print_comparison_table(results)

    out = save_chart(results, CHART_OUT)
    console.print(f"[dim]Chart saved: {out}[/dim]")
    console.print()


if __name__ == "__main__":
    main()
