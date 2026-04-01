"""
Patch Tool Calls middleware.

Handles fixing dangling or malformed tool calls that can occur from:
- Interrupted responses
- Token limit truncation
- Network issues
- Streaming interruptions

Based on deepagents PatchToolCallsMiddleware pattern.
"""
from __future__ import annotations

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple
from .base import BaseMiddleware, AgentState


class PatchToolCallsMiddleware(BaseMiddleware):
    """
    Middleware that fixes issues with tool calls in the message history.
    
    Common issues handled:
    1. Tool use without corresponding tool result (dangling tool calls)
    2. Tool results without matching tool use
    3. Malformed tool call IDs
    4. Incomplete tool call content
    
    This ensures the message history is always in a valid state for the API.
    """
    
    def __init__(
        self,
        auto_cancel_dangling: bool = True,
        error_message: str = "Tool call was interrupted or failed to complete.",
        enabled: bool = True,
    ):
        """
        Initialize the patch tool calls middleware.
        
        Args:
            auto_cancel_dangling: Automatically add error results for dangling tool calls
            error_message: Error message to use for cancelled tool calls
            enabled: Whether the middleware is active
        """
        super().__init__(enabled)
        self.auto_cancel_dangling = auto_cancel_dangling
        self.error_message = error_message
    
    def find_tool_calls(self, messages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Find all tool calls in the message history.
        
        Args:
            messages: List of messages
            
        Returns:
            Dictionary mapping tool_use_id to tool call info
        """
        tool_calls = {}
        
        for i, msg in enumerate(messages):
            if msg.get("role") != "assistant":
                continue
            
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    if tool_id:
                        tool_calls[tool_id] = {
                            "message_index": i,
                            "name": block.get("name", "unknown"),
                            "input": block.get("input", {}),
                            "has_result": False
                        }
        
        return tool_calls
    
    def find_tool_results(self, messages: List[Dict[str, Any]]) -> set:
        """
        Find all tool result IDs in the message history.
        
        Args:
            messages: List of messages
            
        Returns:
            Set of tool_use_ids that have results
        """
        result_ids = set()
        
        for msg in messages:
            if msg.get("role") != "user":
                continue
            
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    tool_use_id = block.get("tool_use_id")
                    if tool_use_id:
                        result_ids.add(tool_use_id)
        
        return result_ids
    
    def find_dangling_tool_calls(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Tuple[str, str, int]]:
        """
        Find tool calls that don't have corresponding results.
        
        Args:
            messages: List of messages
            
        Returns:
            List of (tool_use_id, tool_name, message_index) for dangling calls
        """
        tool_calls = self.find_tool_calls(messages)
        result_ids = self.find_tool_results(messages)
        
        dangling = []
        for tool_id, info in tool_calls.items():
            if tool_id not in result_ids:
                dangling.append((tool_id, info["name"], info["message_index"]))
        
        return dangling
    
    def create_error_result(
        self,
        tool_use_id: str,
        tool_name: str,
        error_message: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an error tool result block.
        
        Args:
            tool_use_id: The ID of the tool call
            tool_name: Name of the tool
            error_message: Custom error message
            
        Returns:
            Tool result block with error
        """
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": error_message or self.error_message,
            "is_error": True
        }
    
    def patch_dangling_tool_calls(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Add error results for any dangling tool calls.
        
        This ensures the API receives a valid message sequence where
        every tool_use has a corresponding tool_result.
        
        Args:
            messages: List of messages
            
        Returns:
            Modified messages with patched tool calls
        """
        dangling = self.find_dangling_tool_calls(messages)
        
        if not dangling:
            return messages
        
        messages = messages.copy()
        
        # Group dangling calls by whether they need a new user message
        # or can be added to an existing one
        error_results = []
        for tool_id, tool_name, msg_idx in dangling:
            error_results.append(self.create_error_result(tool_id, tool_name))
        
        # Add error results as a new user message at the end
        if error_results:
            messages.append({
                "role": "user",
                "content": error_results
            })
        
        return messages
    
    def fix_malformed_tool_ids(
        self,
        messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fix or regenerate malformed tool IDs.
        
        Args:
            messages: List of messages
            
        Returns:
            Messages with fixed tool IDs
        """
        messages = [msg.copy() for msg in messages]
        id_mapping = {}  # old_id -> new_id
        
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, str):
                continue
            
            new_content = []
            for block in content:
                if not isinstance(block, dict):
                    new_content.append(block)
                    continue
                
                block = block.copy()
                
                if block.get("type") == "tool_use":
                    old_id = block.get("id", "")
                    # Check if ID is valid (should be non-empty string)
                    if not old_id or not isinstance(old_id, str):
                        new_id = f"toolu_{uuid.uuid4().hex[:24]}"
                        id_mapping[old_id] = new_id
                        block["id"] = new_id
                
                elif block.get("type") == "tool_result":
                    old_id = block.get("tool_use_id", "")
                    if old_id in id_mapping:
                        block["tool_use_id"] = id_mapping[old_id]
                
                new_content.append(block)
            
            msg["content"] = new_content
        
        return messages
    
    def validate_message_sequence(
        self,
        messages: List[Dict[str, Any]]
    ) -> Tuple[bool, List[str]]:
        """
        Validate that the message sequence is valid for the API.
        
        Rules:
        1. Messages must alternate between user and assistant (with exceptions for tool results)
        2. Every tool_use must be followed by a tool_result
        3. Tool results must reference valid tool_use IDs
        
        Args:
            messages: List of messages
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        tool_calls = self.find_tool_calls(messages)
        result_ids = self.find_tool_results(messages)
        
        # Check for dangling tool calls
        for tool_id, info in tool_calls.items():
            if tool_id not in result_ids:
                issues.append(f"Tool call '{info['name']}' (id={tool_id}) has no result")
        
        # Check for orphan tool results
        for result_id in result_ids:
            if result_id not in tool_calls:
                issues.append(f"Tool result references unknown tool_use_id: {result_id}")
        
        # Check message alternation (simplified)
        prev_role = None
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if role == prev_role and role != "user":
                # Consecutive assistant messages are invalid
                issues.append(f"Consecutive {role} messages at index {i-1} and {i}")
            prev_role = role
        
        return len(issues) == 0, issues
    
    def pre_process(self, state: AgentState) -> AgentState:
        """
        Fix any issues with tool calls before sending to API.
        
        Args:
            state: Current agent state
            
        Returns:
            Modified state with fixed messages
        """
        if not self.enabled:
            return state
        
        # Fix malformed IDs first
        state.messages = self.fix_malformed_tool_ids(state.messages)
        
        # Patch dangling tool calls
        if self.auto_cancel_dangling:
            state.messages = self.patch_dangling_tool_calls(state.messages)
        
        return state
    
    def post_process(self, state: AgentState, response: Dict[str, Any]) -> tuple[AgentState, Dict[str, Any]]:
        """
        Track pending tool calls from response.
        
        Args:
            state: Current agent state
            response: API response
            
        Returns:
            Tuple of (state, response)
        """
        # Extract any new tool calls from response
        content = response.get("content", [])
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    tool_id = block.get("id")
                    if tool_id:
                        state.pending_tool_calls.append({
                            "id": tool_id,
                            "name": block.get("name"),
                            "input": block.get("input", {})
                        })
        
        return state, response
    
    def on_error(self, state: AgentState, error: Exception) -> AgentState:
        """
        Handle errors by ensuring tool calls are properly closed.
        
        Args:
            state: Current agent state
            error: The exception that occurred
            
        Returns:
            Modified state
        """
        # If there are pending tool calls and we hit an error,
        # we should patch them on the next request
        if state.pending_tool_calls:
            # Mark that we need to patch on next pre_process
            pass
        
        return state

