"""
Generate workflow diagram for YACC architecture.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(20, 28))
ax.set_xlim(0, 20)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')
ax.set_facecolor('#0d1117')

# Colors
C_BG       = '#0d1117'
C_PANEL    = '#161b22'
C_BORDER   = '#30363d'
C_BLUE     = '#58a6ff'
C_GREEN    = '#3fb950'
C_ORANGE   = '#d29922'
C_PURPLE   = '#bc8cff'
C_RED      = '#f85149'
C_TEAL     = '#39d0d8'
C_TEXT     = '#e6edf3'
C_MUTED    = '#8b949e'
C_ARROW    = '#6e7681'


def box(x, y, w, h, color=C_PANEL, border=C_BORDER, radius=0.3):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          facecolor=color, edgecolor=border, linewidth=1.2)
    ax.add_patch(rect)


def label(x, y, text, color=C_TEXT, size=9, ha='center', va='center', bold=False):
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, color=color, fontsize=size, ha=ha, va=va,
            fontweight=weight, fontfamily='monospace')


def arrow(x1, y1, x2, y2, color=C_ARROW):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def section_header(x, y, w, text, color=C_BLUE):
    box(x, y, w, 0.55, color=color + '33', border=color, radius=0.2)
    label(x + w/2, y + 0.27, text, color=color, size=9, bold=True)


# ─── TITLE ───────────────────────────────────────────────────────────────────
label(10, 27.5, 'YACC — Full Workflow: CLI → Middleware → API (Multi-Turn)',
      color=C_TEXT, size=13, bold=True)

# ─── SESSION START ────────────────────────────────────────────────────────────
box(1, 26.3, 18, 0.85, color='#1c2128', border=C_GREEN)
label(10, 26.72, 'SESSION START:  yacc  →  YACCCLI.__init__()  →  initialize_agent()  →  Agent.__init__()',
      color=C_GREEN, size=9)
label(10, 26.35, 'self.agent.state  =  AgentState(messages=[], tools=[...])   ← lives until exit / reset',
      color=C_MUTED, size=8)

arrow(10, 26.3, 10, 25.9)

# ─── USER INPUT ──────────────────────────────────────────────────────────────
box(1, 25.4, 18, 0.45, color='#1c2128', border=C_ORANGE)
label(10, 25.62, 'USER TYPES:  "fix my bug"',
      color=C_ORANGE, size=9, bold=True)

arrow(10, 25.4, 10, 25.0)

# ─── THREE COLUMNS SETUP ─────────────────────────────────────────────────────
# CLI | chat() | _call_api() / Middleware / API
COL_CLI   = (1.0,  3.5)   # x, width
COL_CHAT  = (5.2,  4.2)
COL_API   = (10.1, 9.0)

# Column headers
section_header(COL_CLI[0],  24.5, COL_CLI[1],  'CLI  (app.py)',        color=C_TEAL)
section_header(COL_CHAT[0], 24.5, COL_CHAT[1], 'agent.chat()',         color=C_BLUE)
section_header(COL_API[0],  24.5, COL_API[1],  '_call_api()  /  Middleware  /  Anthropic API', color=C_PURPLE)

# Vertical dividers
for x in [4.9, 9.8]:
    ax.plot([x, x], [0.5, 24.5], color=C_BORDER, lw=0.8, linestyle='--', alpha=0.5)

# ─── STEP 1: Append user message ─────────────────────────────────────────────
y = 23.8
box(COL_CHAT[0], y, COL_CHAT[1], 0.55, border=C_BLUE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.37, 'state.messages.append(', color=C_MUTED, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, '{role:"user", content:"fix my bug"})', color=C_TEXT, size=7.5)

box(COL_CLI[0], y+0.1, COL_CLI[1], 0.35, border=C_TEAL)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.27, 'for event in chat():', color=C_TEAL, size=8)

arrow(4.5, y+0.27, COL_CHAT[0], y+0.27, color=C_TEAL)

# yield user_message
y2 = 23.2
box(COL_CHAT[0], y2, COL_CHAT[1], 0.35, border=C_MUTED)
label(COL_CHAT[0]+COL_CHAT[1]/2, y2+0.17, 'yield {type:"user_message"}', color=C_MUTED, size=7.5)
arrow(COL_CHAT[0], y2+0.17, COL_CLI[0]+COL_CLI[1], y2+0.17, color=C_MUTED)
label(COL_CLI[0]+COL_CLI[1]/2, y2+0.17, '(no render)', color=C_MUTED, size=7.5, bold=False)

arrow(COL_CHAT[0]+COL_CHAT[1]/2, y2, COL_CHAT[0]+COL_CHAT[1]/2, 22.8)

# ─── TURN 1 ──────────────────────────────────────────────────────────────────
y = 22.45
section_header(COL_CHAT[0], y, COL_CHAT[1]+9.0+0.9, 'TURN 1', color=C_ORANGE)

# turn_start yield
y = 22.0
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_ORANGE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'yield {type:"turn_start"}', color=C_ORANGE, size=7.5)
arrow(COL_CHAT[0], y+0.17, COL_CLI[0]+COL_CLI[1], y+0.17, color=C_ORANGE)
box(COL_CLI[0], y, COL_CLI[1], 0.35, border=C_ORANGE)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.17, 'show spinner', color=C_ORANGE, size=7.5)

arrow(COL_CHAT[0]+COL_CHAT[1]/2, y, COL_CHAT[0]+COL_CHAT[1]/2, 21.6)

# _call_api box spanning rest of turn
y_api_top = 21.55
y_api_bot = 18.6
box(COL_API[0], y_api_bot, COL_API[1], y_api_top-y_api_bot, color='#1c2128', border=C_PURPLE, radius=0.3)
label(COL_API[0]+COL_API[1]/2, y_api_top-0.25, '_call_api()', color=C_PURPLE, size=9, bold=True)

# call _call_api
y = 21.2
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_PURPLE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, '_call_api()  →', color=C_PURPLE, size=8)
arrow(COL_CHAT[0]+COL_CHAT[1], y+0.17, COL_API[0], y+0.17, color=C_PURPLE)

# PRE-PROCESS
y = 20.8
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.9, color='#21262d', border=C_GREEN)
label(COL_API[0]+COL_API[1]/2, y+0.72, '① PRE-PROCESS  (forward order)', color=C_GREEN, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.52, 'PromptCachingMiddleware  →  adds cache_control to tools', color=C_MUTED, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.32, 'SummarizationMiddleware  →  checks tokens (<170k), no-op', color=C_MUTED, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.12, 'PatchToolCallsMiddleware  →  no dangling calls yet, no-op', color=C_MUTED, size=7.5)

# BUILD REQUEST
y = 19.9
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.65, color='#21262d', border=C_BLUE)
label(COL_API[0]+COL_API[1]/2, y+0.50, '② BUILD REQUEST  from  state', color=C_BLUE, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.30, 'messages: [U1]    system: "You are..."    tools: [...]', color=C_MUTED, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.12, 'state.messages  ──►  request_params["messages"]', color=C_TEXT, size=7.5)

# ANTHROPIC API
y = 18.8
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.75, color='#21262d', border=C_RED)
label(COL_API[0]+COL_API[1]/2, y+0.60, '③ client.messages.create(...)  →  Anthropic API', color=C_RED, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.40, 'stop_reason: "tool_use"', color=C_ORANGE, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.22, 'content: [TextBlock("Let me look..."), ToolUseBlock(read_file, tu_123)]', color=C_MUTED, size=7)

# POST-PROCESS
y = 17.85
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.7, color='#21262d', border=C_GREEN)
label(COL_API[0]+COL_API[1]/2, y+0.55, '④ POST-PROCESS  (reverse order)', color=C_GREEN, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.35, 'PatchToolCalls  →  SummarizationMiddleware (updates token counts)', color=C_MUTED, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.15, 'PromptCaching  →  tracks cache hits/misses', color=C_MUTED, size=7.5)

# return response_dict
y = 17.5
arrow(COL_API[0], y+0.17, COL_CHAT[0]+COL_CHAT[1], y+0.17, color=C_PURPLE)
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_PURPLE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, '← returns response_dict', color=C_PURPLE, size=7.5)

# append assistant message
y = 17.0
box(COL_CHAT[0], y, COL_CHAT[1], 0.55, border=C_BLUE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.40, 'state.messages.append(', color=C_MUTED, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.22, '{role:"assistant", content:[...]})', color=C_TEXT, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.07, 'messages = [U1, A1]', color=C_GREEN, size=7)

# yield assistant_message
y = 16.35
box(COL_CHAT[0], y, COL_CHAT[1], 0.45, border=C_BLUE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.32, 'yield {type:"assistant_message",', color=C_BLUE, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.14, 'content, tool_calls:[read_file]}', color=C_BLUE, size=7.5)
arrow(COL_CHAT[0], y+0.22, COL_CLI[0]+COL_CLI[1], y+0.22, color=C_BLUE)
box(COL_CLI[0], y+0.05, COL_CLI[1], 0.35, border=C_BLUE)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.22, 'render text + tool', color=C_BLUE, size=7.5)

# stop_reason == tool_use → execute tools
y = 15.75
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_ORANGE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'stop_reason=="tool_use" → execute', color=C_ORANGE, size=7.5)

# execute tools
y = 15.2
box(COL_CHAT[0], y, COL_CHAT[1]+COL_API[1]+0.9, 0.45, color='#21262d', border=C_TEAL)
label((COL_CHAT[0] + 19.1)/2, y+0.32, 'executor.execute("read_file", {path:"x.py"})  →  reads file from disk', color=C_TEAL, size=8)
label((COL_CHAT[0] + 19.1)/2, y+0.12, 'returns: [{type:"tool_result", tool_use_id:"tu_123", content:"def foo():..."}]', color=C_MUTED, size=7.5)

# append tool results
y = 14.65
box(COL_CHAT[0], y, COL_CHAT[1], 0.55, border=C_TEAL)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.40, 'state.messages.append(', color=C_MUTED, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.22, '{role:"user", content:[tool_results]})', color=C_TEXT, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.07, 'messages = [U1, A1, TR1]', color=C_GREEN, size=7)

# yield tool_results
y = 14.05
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_TEAL)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'yield {type:"tool_results"}', color=C_TEAL, size=7.5)
arrow(COL_CHAT[0], y+0.17, COL_CLI[0]+COL_CLI[1], y+0.17, color=C_TEAL)
box(COL_CLI[0], y, COL_CLI[1], 0.35, border=C_TEAL)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.17, 'render tool output', color=C_TEAL, size=7.5)

arrow(COL_CHAT[0]+COL_CHAT[1]/2, 14.05, COL_CHAT[0]+COL_CHAT[1]/2, 13.65)

# ─── TURN 2 ──────────────────────────────────────────────────────────────────
y = 13.3
section_header(COL_CHAT[0], y, COL_CHAT[1]+9.0+0.9, 'TURN 2  (full history re-sent)', color=C_ORANGE)

# turn_start
y = 12.85
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_ORANGE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'yield {type:"turn_start", turn:2}', color=C_ORANGE, size=7.5)
arrow(COL_CHAT[0], y+0.17, COL_CLI[0]+COL_CLI[1], y+0.17, color=C_ORANGE)
box(COL_CLI[0], y, COL_CLI[1], 0.35, border=C_ORANGE)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.17, 'show spinner', color=C_ORANGE, size=7.5)

arrow(COL_CHAT[0]+COL_CHAT[1]/2, y, COL_CHAT[0]+COL_CHAT[1]/2, 12.45)

# _call_api turn 2
y_api2_top = 12.4
y_api2_bot = 10.1
box(COL_API[0], y_api2_bot, COL_API[1], y_api2_top-y_api2_bot, color='#1c2128', border=C_PURPLE, radius=0.3)
label(COL_API[0]+COL_API[1]/2, y_api2_top-0.25, '_call_api()  — Turn 2', color=C_PURPLE, size=9, bold=True)

y = 12.05
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_PURPLE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, '_call_api()  →', color=C_PURPLE, size=8)
arrow(COL_CHAT[0]+COL_CHAT[1], y+0.17, COL_API[0], y+0.17, color=C_PURPLE)

y = 11.65
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.55, color='#21262d', border=C_GREEN)
label(COL_API[0]+COL_API[1]/2, y+0.42, '① PRE-PROCESS', color=C_GREEN, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.22, 'SummarizationMiddleware checks tokens — still < 170k, no-op', color=C_MUTED, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.06, 'PatchToolCalls — no dangling calls, no-op', color=C_MUTED, size=7.5)

y = 11.05
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.45, color='#21262d', border=C_BLUE)
label(COL_API[0]+COL_API[1]/2, y+0.33, '② BUILD REQUEST  —  FULL history re-sent every call', color=C_BLUE, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.13, 'messages: [U1, A1, TR1]    Claude sees everything', color=C_MUTED, size=7.5)

y = 10.3
box(COL_API[0]+0.2, y, COL_API[1]-0.4, 0.55, color='#21262d', border=C_RED)
label(COL_API[0]+COL_API[1]/2, y+0.43, '③ Anthropic API  →  Claude sees full context', color=C_RED, size=8, bold=True)
label(COL_API[0]+COL_API[1]/2, y+0.25, 'stop_reason: "end_turn"', color=C_GREEN, size=7.5)
label(COL_API[0]+COL_API[1]/2, y+0.08, 'content: [TextBlock("The bug is on line 5...")]', color=C_MUTED, size=7.5)

y = 9.7
arrow(COL_API[0], y+0.17, COL_CHAT[0]+COL_CHAT[1], y+0.17, color=C_PURPLE)
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_PURPLE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, '← returns response_dict', color=C_PURPLE, size=7.5)

y = 9.15
box(COL_CHAT[0], y, COL_CHAT[1], 0.55, border=C_BLUE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.40, 'state.messages.append(A2)', color=C_MUTED, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.22, 'messages = [U1, A1, TR1, A2]', color=C_GREEN, size=7.5)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.07, 'stop_reason == "end_turn"  →  break', color=C_ORANGE, size=7)

y = 8.55
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_BLUE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'yield {type:"assistant_message"}', color=C_BLUE, size=7.5)
arrow(COL_CHAT[0], y+0.17, COL_CLI[0]+COL_CLI[1], y+0.17, color=C_BLUE)
box(COL_CLI[0], y, COL_CLI[1], 0.35, border=C_BLUE)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.17, 'render response', color=C_BLUE, size=7.5)

y = 8.0
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_GREEN)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'yield {type:"complete"}', color=C_GREEN, size=7.5)
arrow(COL_CHAT[0], y+0.17, COL_CLI[0]+COL_CLI[1], y+0.17, color=C_GREEN)
box(COL_CLI[0], y, COL_CLI[1], 0.35, border=C_GREEN)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.17, 'show completion', color=C_GREEN, size=7.5)

arrow(COL_CHAT[0]+COL_CHAT[1]/2, 8.0, COL_CHAT[0]+COL_CHAT[1]/2, 7.6)

# ─── NEXT USER INPUT ─────────────────────────────────────────────────────────
y = 7.25
box(1, y, 18, 0.45, color='#1c2128', border=C_ORANGE)
label(10, y+0.22, 'USER TYPES: "now add tests"   ← same Agent instance, state.messages still has [U1, A1, TR1, A2]',
      color=C_ORANGE, size=8.5)

arrow(10, y, 10, 6.6)

y = 6.35
box(COL_CHAT[0], y, COL_CHAT[1], 0.35, border=C_BLUE)
label(COL_CHAT[0]+COL_CHAT[1]/2, y+0.17, 'state.messages.append(U2)', color=C_TEXT, size=7.5)
box(COL_CLI[0], y, COL_CLI[1], 0.35, border=C_TEAL)
label(COL_CLI[0]+COL_CLI[1]/2, y+0.17, 'for event in chat():', color=C_TEAL, size=8)
arrow(4.5, y+0.17, COL_CHAT[0], y+0.17, color=C_TEAL)

y = 5.85
box(COL_CHAT[0]+0.3, y, COL_CHAT[1]-0.3+COL_API[1]+0.9, 0.35, color='#21262d', border=C_MUTED)
label(11, y+0.17,
      'messages = [U1, A1, TR1, A2, U2]  ← grows each session, full history re-sent, Claude has full context of prior fix',
      color=C_MUTED, size=7.5)

arrow(10, 5.85, 10, 5.5)
label(10, 5.3, '... loop repeats for as many turns as needed ...', color=C_MUTED, size=8.5)

# ─── LEGEND ──────────────────────────────────────────────────────────────────
y = 4.4
box(1, 1.0, 18, 3.2, color='#161b22', border=C_BORDER, radius=0.3)
label(10, y+0.55, 'KEY NOTES', color=C_TEXT, size=9, bold=True)

notes = [
    (C_GREEN,  'state.messages  is the single source of truth — all history lives here in RAM'),
    (C_RED,    'Full history re-sent on EVERY API call — no server-side memory in Anthropic API'),
    (C_BLUE,   'chat() is a Python generator — CLI and chat() take turns on ONE thread (no concurrency)'),
    (C_ORANGE, '"tool_use" stop_reason → keep looping;   "end_turn" → break out of while loop'),
    (C_PURPLE, 'Middleware wraps every _call_api() call — runs on every turn, not once per session'),
    (C_TEAL,   'SummarizationMiddleware compresses history at 170k tokens to avoid hitting 200k API limit'),
]
for i, (color, text) in enumerate(notes):
    yy = y + 0.1 - i * 0.42
    ax.plot([1.4, 1.7], [yy-0.05, yy-0.05], color=color, lw=3)
    label(1.85, yy-0.05, text, color=C_TEXT, size=7.8, ha='left', va='center')

plt.tight_layout(pad=0.5)
out = '/Users/mac/Desktop/Projects/yet-another-claude-code/docs/workflow_diagram.png'
plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=C_BG)
print(f"Saved: {out}")
