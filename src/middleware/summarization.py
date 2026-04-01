"""
Summarization middleware for managing context length.

Auto-summarizes conversation when context exceeds token limits.
Based on deepagents SummarizationMiddleware pattern.
"""
from __future__ import annotations

import json
from typing import Dict, List, Any, Optional
from .base import BaseMiddleware, AgentState


class SummarizationMiddleware(BaseMiddleware):
    """
    Middleware that automatically summarizes conversation history
    when the context exceeds a token threshold.
    
    This helps manage long-running conversations by compressing
    older parts of the conversation while preserving key information.
    """
    
    # Default token threshold (170k for Claude models)
    DEFAULT_TOKEN_THRESHOLD = 170_000
    
    # Target tokens after summarization (leave room for response)
    DEFAULT_TARGET_TOKENS = 100_000
    
    def __init__(
        self,
        token_threshold: int = DEFAULT_TOKEN_THRESHOLD,
        target_tokens: int = DEFAULT_TARGET_TOKENS,
        enabled: bool = True,
        summarization_model: Optional[str] = None,
    ):
        """
        Initialize the summarization middleware.
        
        Args:
            token_threshold: Token count that triggers summarization
            target_tokens: Target token count after summarization
            enabled: Whether the middleware is active
            summarization_model: Model to use for summarization (optional)
        """
        super().__init__(enabled)
        self.token_threshold = token_threshold
        self.target_tokens = target_tokens
        self.summarization_model = summarization_model
        
        # Summarization prompt
        self.summarization_prompt = """You are tasked with summarizing a conversation to reduce its length while preserving essential information.

Guidelines:
1. Preserve all important facts, decisions, and context
2. Keep track of completed actions and their results
3. Maintain awareness of pending tasks or goals
4. Preserve code snippets and file paths that may be referenced later
5. Keep user preferences and constraints
6. Remove redundant back-and-forth dialogue
7. Condense tool call results while keeping key information

Format your summary as a concise narrative that captures:
- What the user wanted to accomplish
- What has been done so far
- Key findings or results
- Any pending work or next steps
- Important context for continuing the conversation"""
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for a string.
        
        This is a rough estimate (4 chars ≈ 1 token for English).
        For more accuracy, use a proper tokenizer.
        
        Args:
            text: The text to estimate tokens for
            
        Returns:
            Estimated token count
        """
        return len(text) // 4
    
    def estimate_message_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """
        Estimate total tokens for a list of messages.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            Estimated total token count
        """
        total = 0
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += self.estimate_tokens(content)
            elif isinstance(content, list):
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            total += self.estimate_tokens(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            total += self.estimate_tokens(json.dumps(block.get("input", {})))
                        elif block.get("type") == "tool_result":
                            total += self.estimate_tokens(str(block.get("content", "")))
        return total
    
    def should_summarize(self, state: AgentState) -> bool:
        """
        Check if summarization is needed based on token count.
        
        Args:
            state: Current agent state
            
        Returns:
            True if summarization should be triggered
        """
        # Check total tokens (input + output)
        total_tokens = state.total_input_tokens + state.total_output_tokens
        if total_tokens > self.token_threshold:
            return True
        
        # Also check estimated message tokens
        estimated = self.estimate_message_tokens(state.messages)
        if estimated > self.token_threshold:
            return True
        
        return False
    
    def find_summarization_point(self, messages: List[Dict[str, Any]]) -> int:
        """
        Find the best point to split messages for summarization.
        
        We want to summarize older messages while keeping recent ones intact.
        The split should happen at a natural boundary (after a complete turn).
        
        Args:
            messages: List of messages
            
        Returns:
            Index to split at (messages before this index will be summarized)
        """
        if len(messages) <= 4:
            return 0  # Don't summarize if too few messages
        
        # Keep at least the last 4 messages (2 turns)
        min_keep = 4
        
        # Target: keep messages that fit in target_tokens
        target = self.target_tokens
        
        # Start from the end and work backwards
        tokens_kept = 0
        split_point = len(messages)
        
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            content = msg.get("content", "")
            if isinstance(content, str):
                msg_tokens = self.estimate_tokens(content)
            else:
                msg_tokens = self.estimate_tokens(json.dumps(content))
            
            if tokens_kept + msg_tokens > target and i < len(messages) - min_keep:
                split_point = i + 1
                break
            
            tokens_kept += msg_tokens
            split_point = i
        
        return max(1, split_point)  # Always keep at least first message
    
    def create_summary_message(self, messages_to_summarize: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a summary message from older messages.
        
        In a real implementation, this would call the LLM to generate a summary.
        For now, we create a structured summary placeholder.
        
        Args:
            messages_to_summarize: Messages to be summarized
            
        Returns:
            A summary message dictionary
        """
        # Extract key information
        user_messages = []
        assistant_actions = []
        tool_results = []
        
        for msg in messages_to_summarize:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "user":
                if isinstance(content, str):
                    user_messages.append(content[:200])  # Truncate long messages
            elif role == "assistant":
                if isinstance(content, list):
                    for block in content:
                        if block.get("type") == "text":
                            assistant_actions.append(block.get("text", "")[:100])
                        elif block.get("type") == "tool_use":
                            tool_name = block.get("name", "unknown")
                            assistant_actions.append(f"Used tool: {tool_name}")
                elif isinstance(content, str):
                    assistant_actions.append(content[:100])
        
        # Build summary
        summary_parts = ["[CONVERSATION SUMMARY]"]
        
        if user_messages:
            summary_parts.append("\nUser requests:")
            for msg in user_messages[:5]:  # Limit to 5
                summary_parts.append(f"- {msg}")
        
        if assistant_actions:
            summary_parts.append("\nActions taken:")
            for action in assistant_actions[:10]:  # Limit to 10
                summary_parts.append(f"- {action}")
        
        summary_parts.append("\n[END SUMMARY]")
        
        return {
            "role": "user",
            "content": "\n".join(summary_parts)
        }
    
    def pre_process(self, state: AgentState) -> AgentState:
        """
        Check if summarization is needed and apply it.
        
        Args:
            state: Current agent state
            
        Returns:
            Modified state with summarized messages if needed
        """
        if not self.enabled:
            return state
        
        if not self.should_summarize(state):
            return state
        
        # Find where to split
        split_point = self.find_summarization_point(state.messages)
        
        if split_point <= 1:
            return state  # Nothing to summarize
        
        # Create summary of older messages
        messages_to_summarize = state.messages[:split_point]
        summary_message = self.create_summary_message(messages_to_summarize)
        
        # Replace old messages with summary + recent messages
        state.messages = [summary_message] + state.messages[split_point:]
        state.is_summarized = True
        state.summary = summary_message["content"]
        
        return state
    
    def post_process(self, state: AgentState, response: Dict[str, Any]) -> tuple[AgentState, Dict[str, Any]]:
        """
        Update token counts from response.
        
        Args:
            state: Current agent state
            response: API response
            
        Returns:
            Tuple of (modified state, response)
        """
        # Extract token usage from response
        usage = response.get("usage", {})
        if usage:
            state.total_input_tokens = usage.get("input_tokens", state.total_input_tokens)
            state.total_output_tokens += usage.get("output_tokens", 0)
        
        return state, response

