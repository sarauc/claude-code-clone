"""
Base middleware classes for the agent pipeline.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field


@dataclass
class AgentState:
    """
    Represents the current state of the agent conversation.
    """
    messages: List[Dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    tools: List[Dict[str, Any]] = field(default_factory=list)
    todos: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    turn_count: int = 0
    
    # Context management
    is_summarized: bool = False
    summary: Optional[str] = None
    
    # Tool state tracking
    pending_tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    files_read: set = field(default_factory=set)
    
    # Cache control
    cache_breakpoints: List[int] = field(default_factory=list)


class BaseMiddleware(ABC):
    """
    Abstract base class for middleware components.
    
    Middleware can:
    - Modify messages before they're sent to the API (pre_process)
    - Modify responses after they're received (post_process)
    - Handle errors during processing (on_error)
    """
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    @abstractmethod
    def pre_process(self, state: AgentState) -> AgentState:
        """
        Process state before sending to the API.
        
        Args:
            state: Current agent state
            
        Returns:
            Modified agent state
        """
        pass
    
    @abstractmethod
    def post_process(self, state: AgentState, response: Dict[str, Any]) -> tuple[AgentState, Dict[str, Any]]:
        """
        Process state and response after receiving from API.
        
        Args:
            state: Current agent state
            response: API response
            
        Returns:
            Tuple of (modified state, modified response)
        """
        pass
    
    def on_error(self, state: AgentState, error: Exception) -> AgentState:
        """
        Handle errors during processing.
        
        Args:
            state: Current agent state
            error: The exception that occurred
            
        Returns:
            Modified agent state
        """
        return state


class MiddlewareChain:
    """
    Chain of middleware components that process requests in order.
    """
    
    def __init__(self, middlewares: Optional[List[BaseMiddleware]] = None):
        self.middlewares = middlewares or []
    
    def add(self, middleware: BaseMiddleware) -> "MiddlewareChain":
        """Add a middleware to the chain."""
        self.middlewares.append(middleware)
        return self
    
    def pre_process(self, state: AgentState) -> AgentState:
        """Run all middleware pre-processing in order."""
        for middleware in self.middlewares:
            if middleware.enabled:
                state = middleware.pre_process(state)
        return state
    
    def post_process(self, state: AgentState, response: Dict[str, Any]) -> tuple[AgentState, Dict[str, Any]]:
        """Run all middleware post-processing in reverse order."""
        for middleware in reversed(self.middlewares):
            if middleware.enabled:
                state, response = middleware.post_process(state, response)
        return state, response
    
    def on_error(self, state: AgentState, error: Exception) -> AgentState:
        """Handle errors through all middleware."""
        for middleware in self.middlewares:
            if middleware.enabled:
                state = middleware.on_error(state, error)
        return state

