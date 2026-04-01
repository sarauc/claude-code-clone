"""
Anthropic Prompt Caching middleware.

Implements prompt caching to reduce costs by caching system prompts
and other static content that doesn't change between requests.

See: https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from .base import BaseMiddleware, AgentState


class AnthropicPromptCachingMiddleware(BaseMiddleware):
    """
    Middleware that adds cache control headers for Anthropic's prompt caching.
    
    Prompt caching allows you to cache large contexts (like system prompts)
    and reuse them across multiple requests, reducing latency and costs.
    
    Cache breakpoints are added to:
    - System prompts
    - Tool definitions
    - Large static context that doesn't change between turns
    
    Requirements:
    - Minimum cacheable prompt length: 1024 tokens (Sonnet/Opus) or 2048 (Haiku)
    - Cache TTL: 5 minutes (extended with each use)
    - Max cache breakpoints: 4 per request
    """
    
    # Minimum tokens for caching (Claude Sonnet/Opus)
    MIN_CACHE_TOKENS = 1024
    
    # Maximum cache breakpoints per request
    MAX_BREAKPOINTS = 4
    
    def __init__(
        self,
        cache_system_prompt: bool = True,
        cache_tools: bool = True,
        cache_static_messages: bool = False,
        enabled: bool = True,
    ):
        """
        Initialize the prompt caching middleware.
        
        Args:
            cache_system_prompt: Whether to cache the system prompt
            cache_tools: Whether to cache tool definitions
            cache_static_messages: Whether to cache static messages in history
            enabled: Whether the middleware is active
        """
        super().__init__(enabled)
        self.cache_system_prompt = cache_system_prompt
        self.cache_tools = cache_tools
        self.cache_static_messages = cache_static_messages
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation (4 chars ≈ 1 token)."""
        return len(text) // 4
    
    def add_cache_control(self, content: Any, cache_type: str = "ephemeral") -> Any:
        """
        Add cache_control to a content block or system prompt.
        
        Args:
            content: The content to add caching to
            cache_type: Type of caching ("ephemeral" is the only current option)
            
        Returns:
            Modified content with cache_control
        """
        if isinstance(content, str):
            # Convert string to block format with cache control
            return {
                "type": "text",
                "text": content,
                "cache_control": {"type": cache_type}
            }
        elif isinstance(content, dict):
            # Add cache control to existing block
            content["cache_control"] = {"type": cache_type}
            return content
        elif isinstance(content, list):
            # Add cache control to last block in list
            if content:
                content[-1] = self.add_cache_control(content[-1], cache_type)
            return content
        return content
    
    def prepare_system_prompt_for_caching(self, system_prompt: str) -> List[Dict[str, Any]]:
        """
        Prepare system prompt for caching.
        
        Converts a string system prompt to the block format required for caching.
        
        Args:
            system_prompt: The system prompt string
            
        Returns:
            List of content blocks with cache control
        """
        # Check if prompt is long enough to cache
        if self.estimate_tokens(system_prompt) < self.MIN_CACHE_TOKENS:
            return [{"type": "text", "text": system_prompt}]
        
        # Add cache control to system prompt
        return [{
            "type": "text",
            "text": system_prompt,
            "cache_control": {"type": "ephemeral"}
        }]
    
    def prepare_tools_for_caching(self, tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Prepare tools for caching.
        
        Adds cache control to the last tool definition.
        
        Args:
            tools: List of tool definitions
            
        Returns:
            Modified tools with cache control on last tool
        """
        if not tools:
            return tools
        
        # Check if tools are substantial enough to cache
        tools_str = str(tools)
        if self.estimate_tokens(tools_str) < self.MIN_CACHE_TOKENS:
            return tools
        
        # Add cache control to last tool
        tools = tools.copy()
        tools[-1] = {**tools[-1], "cache_control": {"type": "ephemeral"}}
        
        return tools
    
    def prepare_messages_for_caching(
        self,
        messages: List[Dict[str, Any]],
        breakpoints_used: int = 0
    ) -> List[Dict[str, Any]]:
        """
        Add cache breakpoints to static messages.
        
        Identifies messages that are unlikely to change and adds
        cache breakpoints to them.
        
        Args:
            messages: List of messages
            breakpoints_used: Number of breakpoints already used
            
        Returns:
            Modified messages with cache breakpoints
        """
        if not self.cache_static_messages:
            return messages
        
        remaining_breakpoints = self.MAX_BREAKPOINTS - breakpoints_used
        if remaining_breakpoints <= 0:
            return messages
        
        # Don't cache recent messages (last 4)
        if len(messages) <= 4:
            return messages
        
        messages = messages.copy()
        
        # Find good cache points (large assistant messages with tool results)
        cache_points = []
        for i, msg in enumerate(messages[:-4]):  # Exclude last 4
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, list):
                    # Check for tool results
                    for block in content:
                        if block.get("type") == "tool_result":
                            result_content = block.get("content", "")
                            if self.estimate_tokens(str(result_content)) > 500:
                                cache_points.append(i)
                                break
        
        # Add cache control to selected points
        for idx in cache_points[:remaining_breakpoints]:
            msg = messages[idx]
            content = msg.get("content", [])
            if isinstance(content, list) and content:
                content[-1] = self.add_cache_control(content[-1])
                messages[idx] = {**msg, "content": content}
        
        return messages
    
    def pre_process(self, state: AgentState) -> AgentState:
        """
        Add cache control markers before sending to API.
        
        Args:
            state: Current agent state
            
        Returns:
            Modified state with cache control
        """
        if not self.enabled:
            return state
        
        breakpoints_used = 0
        
        # Cache system prompt
        if self.cache_system_prompt and state.system_prompt:
            # Note: The actual caching is handled when building the API request
            # We mark it in state for the agent to use
            state.cache_breakpoints.append(0)  # System prompt position
            breakpoints_used += 1
        
        # Cache tools
        if self.cache_tools and state.tools:
            state.tools = self.prepare_tools_for_caching(state.tools)
            breakpoints_used += 1
        
        # Cache static messages
        if self.cache_static_messages:
            state.messages = self.prepare_messages_for_caching(
                state.messages,
                breakpoints_used
            )
        
        return state
    
    def post_process(self, state: AgentState, response: Dict[str, Any]) -> tuple[AgentState, Dict[str, Any]]:
        """
        Track cache hits/misses from response.
        
        Args:
            state: Current agent state
            response: API response
            
        Returns:
            Tuple of (state, response)
        """
        # Extract cache usage info if available
        usage = response.get("usage", {})
        cache_creation = usage.get("cache_creation_input_tokens", 0)
        cache_read = usage.get("cache_read_input_tokens", 0)
        
        # Could log or track cache efficiency here
        # print(f"Cache: created={cache_creation}, read={cache_read}")
        
        return state, response


def build_cached_request(
    system_prompt: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    model: str = "claude-sonnet-4-20250514",
    max_tokens: int = 8192,
) -> Dict[str, Any]:
    """
    Build an API request with prompt caching enabled.
    
    This is a helper function that constructs the request in the format
    required for Anthropic's prompt caching.
    
    Args:
        system_prompt: The system prompt
        messages: Conversation messages
        tools: Tool definitions
        model: Model identifier
        max_tokens: Maximum response tokens
        
    Returns:
        Complete API request dictionary
    """
    middleware = AnthropicPromptCachingMiddleware()
    
    request = {
        "model": model,
        "max_tokens": max_tokens,
    }
    
    # Prepare system prompt for caching
    if system_prompt:
        request["system"] = middleware.prepare_system_prompt_for_caching(system_prompt)
    
    # Prepare tools for caching
    if tools:
        request["tools"] = middleware.prepare_tools_for_caching(tools)
    
    # Add messages
    request["messages"] = messages
    
    return request

