"""
Main Agent class for the Claude Code-like agent.
Pure Python implementation without LangChain.
"""

import os
import json
from typing import Dict, List, Any, Optional, Generator
from dataclasses import dataclass, field

import anthropic

# Support both relative and absolute imports
try:
    from .tools.definitions import get_tools_for_api, DEFAULT_TOOLS
    from .tools.executor import ToolExecutor
    from .prompts.system import build_system_prompt
    from .middleware.base import AgentState, MiddlewareChain
    from .middleware.summarization import SummarizationMiddleware
    from .middleware.prompt_caching import AnthropicPromptCachingMiddleware
    from .middleware.patch_tool_calls import PatchToolCallsMiddleware
    from .middleware.session_memory import SessionMemoryMiddleware
except ImportError:
    from src.tools.definitions import get_tools_for_api, DEFAULT_TOOLS
    from src.tools.executor import ToolExecutor
    from src.prompts.system import build_system_prompt
    from src.middleware.base import AgentState, MiddlewareChain
    from src.middleware.summarization import SummarizationMiddleware
    from src.middleware.prompt_caching import AnthropicPromptCachingMiddleware
    from src.middleware.patch_tool_calls import PatchToolCallsMiddleware
    from src.middleware.session_memory import SessionMemoryMiddleware


@dataclass 
class AgentConfig:
    """Configuration for the agent."""
    model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 16384
    api_key: Optional[str] = None
    workspace_path: str = "/tmp/workspace"
    
    # Feature toggles
    enable_planning: bool = True
    enable_bash: bool = True
    enable_prompt_caching: bool = True
    enable_summarization: bool = True
    
    # Custom settings
    custom_system_prompt: Optional[str] = None
    enabled_tools: Optional[List[str]] = None
    
    # Streaming
    # When True, chat() yields {"type": "text_delta", "delta": str} events for
    # each token as it arrives, so the CLI can render output in real time.
    # When False, the full response is waited for before any output (original behaviour).
    enable_streaming: bool = True

    # Session memory
    # When True, {workspace}/.claude/memory.md is injected into the system
    # prompt at session start and a summary is saved at session end.
    enable_session_memory: bool = True

    # Debug
    debug: bool = False


class Agent:
    """
    A Claude Code-like agent implementation.
    
    Features:
    - Tool execution (filesystem, bash, planning)
    - Automatic context summarization
    - Prompt caching for efficiency
    - Tool call patching for reliability
    """
    
    def __init__(self, config: Optional[AgentConfig] = None):
        """
        Initialize the agent.
        
        Args:
            config: Agent configuration
        """
        self.config = config or AgentConfig()
        
        # Initialize Anthropic client
        api_key = self.config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is required")
        
        self.client = anthropic.Anthropic(api_key=api_key)
        
        # Initialize tool executor
        self.executor = ToolExecutor(
            workspace_path=self.config.workspace_path,
            use_virtual_fs=False
        )
        
        # Build system prompt
        self.system_prompt = build_system_prompt(
            custom_instructions=self.config.custom_system_prompt,
            workspace_path=self.config.workspace_path,
        )
        
        # Get tool definitions
        tool_names = self.config.enabled_tools or DEFAULT_TOOLS
        if not self.config.enable_planning:
            tool_names = [t for t in tool_names if t != "write_todos"]
        if not self.config.enable_bash:
            tool_names = [t for t in tool_names if t != "bash"]
        
        self.tools = get_tools_for_api(tool_names)
        
        # Initialize middleware
        # Order matters — pre_process runs forward, post_process runs in reverse.
        # SessionMemory goes FIRST so its system_prompt injection is in place
        # before PromptCaching adds cache_control markers to it.
        self.middleware = MiddlewareChain()

        self._session_memory: Optional[SessionMemoryMiddleware] = None
        if self.config.enable_session_memory:
            self._session_memory = SessionMemoryMiddleware(
                workspace_path=self.config.workspace_path
            )
            self.middleware.add(self._session_memory)

        if self.config.enable_prompt_caching:
            self.middleware.add(AnthropicPromptCachingMiddleware())

        if self.config.enable_summarization:
            self.middleware.add(SummarizationMiddleware())

        self.middleware.add(PatchToolCallsMiddleware())
        
        # Conversation state
        self.state = AgentState(
            system_prompt=self.system_prompt,
            tools=self.tools
        )
    
    def _log(self, message: str):
        """Debug logging."""
        if self.config.debug:
            print(f"[DEBUG] {message}")
    
    def _build_request_params(self) -> Dict[str, Any]:
        """
        Build the API request parameters dict from current state.

        Extracted so that both the blocking (_call_api) and streaming
        (_call_api_streaming) paths share identical request construction
        without duplication.
        """
        params: Dict[str, Any] = {
            "model": self.config.model,
            "max_tokens": self.config.max_tokens,
            # The full conversation history is re-sent on every call because
            # the Anthropic API is stateless — there is no server-side memory.
            "messages": self.state.messages,
        }

        # System prompt: use block format when prompt caching is on so that
        # the cache_control marker is included; plain string otherwise.
        if self.config.enable_prompt_caching:
            params["system"] = [{
                "type": "text",
                "text": self.state.system_prompt,
                "cache_control": {"type": "ephemeral"}
            }]
        else:
            params["system"] = self.state.system_prompt

        # Tool definitions (JSON schemas); omitted entirely if none are active.
        if self.state.tools:
            params["tools"] = self.state.tools

        return params

    def _response_to_dict(self, response) -> Dict[str, Any]:
        """
        Convert an Anthropic SDK Message object to a plain dict.

        The SDK returns typed objects (Message, ContentBlock, etc.). We
        normalise them to dicts immediately so every downstream consumer
        (middleware, chat loop, CLI) works with a single consistent format.
        """
        return {
            "id": response.id,
            "type": response.type,
            "role": response.role,
            # model_dump() converts each ContentBlock dataclass → plain dict,
            # giving us {"type": "text", "text": "..."} or
            # {"type": "tool_use", "id": ..., "name": ..., "input": {...}}
            "content": [block.model_dump() for block in response.content],
            "model": response.model,
            "stop_reason": response.stop_reason,
            "stop_sequence": response.stop_sequence,
            "usage": {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        }

    def _call_api(self) -> Dict[str, Any]:
        """
        Blocking API call — waits for the complete response before returning.

        Used when enable_streaming=False, and by the mock benchmark client
        which does not support the streaming interface.

        Returns:
            Response as a plain dict (after post-process middleware).
        """
        # Pre-process middleware runs first: may mutate state (add cache
        # headers, compress history, patch dangling tool calls, etc.)
        self.state = self.middleware.pre_process(self.state)

        request_params = self._build_request_params()
        self._log(f"API call with {len(self.state.messages)} messages")

        # Blocking call: returns only after the full response is generated.
        response = self.client.messages.create(**request_params)

        response_dict = self._response_to_dict(response)

        # Post-process middleware runs after: updates token counts, etc.
        self.state, response_dict = self.middleware.post_process(self.state, response_dict)

        return response_dict

    def _call_api_streaming(self) -> Generator[Dict[str, Any], None, None]:
        """
        Streaming API call — yields tokens as they arrive, then the full response.

        This is a generator that produces two kinds of events:

            {"type": "text_delta", "delta": "<token>"}
                Yielded once per text token as Claude streams them.
                chat() forwards these straight to the CLI so the user sees
                output immediately, rather than waiting for the full response.

            {"type": "_api_response", "response": <dict>}
                Yielded exactly once, after the stream closes. This is an
                INTERNAL event — chat() intercepts it to extract the final
                response dict (which includes tool_use blocks, stop_reason,
                and usage). It is never forwarded to the CLI.

        Why separate the two?
            text_stream only delivers text token deltas. Tool use blocks and
            metadata (stop_reason, usage) are only available via
            stream.get_final_message() after the stream is exhausted.
            We need both: deltas for real-time display, final message for
            tool execution and middleware post-processing.
        """
        # Pre-process middleware must run BEFORE the stream opens so that
        # state mutations (e.g. PatchToolCalls fixing dangling tool_use blocks)
        # are reflected in the request we are about to send.
        self.state = self.middleware.pre_process(self.state)

        request_params = self._build_request_params()
        self._log(f"Streaming API call with {len(self.state.messages)} messages")

        # client.messages.stream() returns a context manager that keeps the
        # HTTP connection open. Tokens arrive via stream.text_stream as the
        # model generates them.
        with self.client.messages.stream(**request_params) as stream:
            for text_delta in stream.text_stream:
                # Forward each token to the caller (chat() → CLI) immediately.
                yield {"type": "text_delta", "delta": text_delta}

            # Block here until the stream is fully consumed, then get the
            # complete Message object — includes content blocks (text + any
            # tool_use), stop_reason, and usage stats.
            final = stream.get_final_message()

        # Convert to the same dict format _call_api() returns so the rest of
        # the chat() loop is identical regardless of which path was taken.
        response_dict = self._response_to_dict(final)

        # Post-process middleware: e.g. SummarizationMiddleware reads
        # response.usage to update the cumulative token count used to decide
        # when to compress conversation history.
        self.state, response_dict = self.middleware.post_process(self.state, response_dict)

        # Signal to chat() that the stream is done and hand over the full
        # response. Using a typed internal event (rather than return) keeps
        # this a generator and avoids StopIteration plumbing.
        yield {"type": "_api_response", "response": response_dict}
    
    def _extract_tool_calls(self, content: List[Dict]) -> List[Dict[str, Any]]:
        """Extract tool calls from response content."""
        tool_calls = []
        for block in content:
            if block.get("type") == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "input": block.get("input", {})
                })
        return tool_calls
    
    def _extract_text(self, content: List[Dict]) -> str:
        """Extract text from response content."""
        texts = []
        for block in content:
            if block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts)
    
    def _execute_tools(self, tool_calls: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Execute tool calls and return results.
        
        Args:
            tool_calls: List of tool calls to execute
            
        Returns:
            List of tool result blocks
        """
        results = []
        
        for tool_call in tool_calls:
            tool_id = tool_call["id"]
            tool_name = tool_call["name"]
            tool_input = tool_call["input"]
            
            self._log(f"Executing tool: {tool_name}")
            
            # Execute the tool
            result = self.executor.execute(tool_name, tool_input)
            
            self._log(f"Tool result: {result[:100]}...")
            
            results.append({
                "type": "tool_result",
                "tool_use_id": tool_id,
                "content": result
            })
        
        return results
    
    def chat(self, message: str, max_turns: int = 50) -> Generator[Dict[str, Any], None, None]:
        """
        Send a message and get responses.
        
        This is a generator that yields events as the agent processes.
        
        Args:
            message: User message
            max_turns: Maximum number of API turns
            
        Yields:
            Event dictionaries with type and data
        """
        # Add user message
        self.state.messages.append({
            "role": "user",
            "content": message
        })
        
        yield {"type": "user_message", "content": message}
        
        turn = 0
        while turn < max_turns:
            turn += 1
            self.state.turn_count += 1
            
            yield {"type": "turn_start", "turn": turn}
            
            # Call API — streaming or blocking depending on config.
            #
            # Streaming path: _call_api_streaming() is a generator. We iterate
            # it here, forwarding text_delta events to the CLI and intercepting
            # the final _api_response sentinel to capture the response dict.
            #
            # Blocking path: _call_api() returns the full response dict at once.
            # No text_delta events are produced; the CLI waits for assistant_message.
            try:
                if self.config.enable_streaming:
                    response = None
                    for event in self._call_api_streaming():
                        if event["type"] == "text_delta":
                            # Pass token straight to the CLI for real-time display.
                            yield event
                        elif event["type"] == "_api_response":
                            # Internal sentinel — capture and stop iterating.
                            response = event["response"]
                else:
                    response = self._call_api()
            except Exception as e:
                yield {"type": "error", "error": str(e)}
                break
            
            # Add assistant response to history
            self.state.messages.append({
                "role": "assistant",
                "content": response["content"]
            })
            
            # Extract text and tool calls
            text = self._extract_text(response["content"])
            tool_calls = self._extract_tool_calls(response["content"])
            
            # Yield assistant response
            yield {
                "type": "assistant_message",
                "content": text,
                "tool_calls": tool_calls,
                "usage": response.get("usage", {}),
                "stop_reason": response.get("stop_reason")
            }
            
            # Check if we're done
            if response.get("stop_reason") == "end_turn":
                yield {"type": "complete", "turn": turn}
                break
            
            # Execute tool calls
            if tool_calls:
                tool_results = self._execute_tools(tool_calls)
                
                # Add tool results to history
                self.state.messages.append({
                    "role": "user",
                    "content": tool_results
                })
                
                yield {
                    "type": "tool_results",
                    "results": tool_results
                }
        else:
            yield {"type": "max_turns_reached", "turn": turn}
    
    def run(self, message: str, max_turns: int = 50) -> str:
        """
        Send a message and get the final response.
        
        This is a blocking call that processes all events and returns
        the final text response.
        
        Args:
            message: User message
            max_turns: Maximum number of API turns
            
        Returns:
            Final text response from the agent
        """
        final_text = ""
        
        for event in self.chat(message, max_turns):
            if event["type"] == "assistant_message":
                if event.get("content"):
                    final_text = event["content"]
        
        return final_text
    
    def get_todos(self) -> List[Dict[str, Any]]:
        """Get current todo list."""
        return self.executor.todos

    def has_conversation(self) -> bool:
        """Return True if at least one full user/assistant exchange has occurred."""
        # state.messages always starts with the user message, so >= 2 means
        # there has been at least one assistant response.
        return len(self.state.messages) >= 2

    def get_memory(self) -> Optional[str]:
        """Return the raw contents of the project memory file, or None."""
        if self._session_memory:
            return self._session_memory.load_memory()
        return None

    def save_session_memory(self) -> Optional[str]:
        """
        Summarise this session and append the result to .claude/memory.md.

        Called explicitly by the CLI at session end (not automatically after
        every chat() call, to avoid saving trivial one-off queries).

        Returns the generated summary string, or None if session memory is
        disabled or the conversation is too short to be worth saving.
        """
        if not self._session_memory:
            return None

        # Skip if nothing meaningful happened (e.g. user typed 'help' and quit).
        if not self.has_conversation():
            return None

        return self._session_memory.save_session_summary(
            client=self.client,
            model=self.config.model,
            messages=self.state.messages,
        )

    def reset(self):
        """Reset conversation state."""
        self.state = AgentState(
            system_prompt=self.system_prompt,
            tools=self.tools
        )
        self.executor.todos = []
        self.executor.files_read = set()
        # Re-enable memory injection for the fresh session.
        if self._session_memory:
            self._session_memory._injected = False

