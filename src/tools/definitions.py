"""
Tool schema definitions for Anthropic Claude API.
Based on deepagents and Cursor tool specifications.
"""
from __future__ import annotations

from typing import Dict, List, Any

# =============================================================================
# Planning Tools
# =============================================================================

WRITE_TODOS_TOOL = {
    "name": "write_todos",
    "description": """Use this tool to create and manage a structured task list for your current work session. This helps you track progress, organize complex tasks, and demonstrate thoroughness to the user.

Only use this tool if you think it will be helpful in staying organized. If the user's request is trivial and takes less than 3 steps, it is better to NOT use this tool and just do the task directly.

## When to Use This Tool
Use this tool in these scenarios:

1. Complex multi-step tasks - When a task requires 3 or more distinct steps or actions
2. Non-trivial and complex tasks - Tasks that require careful planning or multiple operations
3. User explicitly requests todo list - When the user directly asks you to use the todo list
4. User provides multiple tasks - When users provide a list of things to be done (numbered or comma-separated)
5. The plan may need future revisions or updates based on results from the first few steps

## How to Use This Tool
1. When you start working on a task - Mark it as in_progress BEFORE beginning work.
2. After completing a task - Mark it as completed and add any new follow-up tasks discovered during implementation.
3. You can also update future tasks, such as deleting them if they are no longer necessary, or adding new tasks that are necessary. Don't change previously completed tasks.
4. You can make several updates to the todo list at once. For example, when you complete a task, you can mark the next task you need to start as in_progress.

## When NOT to Use This Tool
It is important to skip using this tool when:
1. There is only a single, straightforward task
2. The task is trivial and tracking it provides no benefit
3. The task can be completed in less than 3 trivial steps
4. The task is purely conversational or informational

## Task States and Management

1. **Task States**: Use these states to track progress:
   - pending: Task not yet started
   - in_progress: Currently working on (you can have multiple tasks in_progress at a time if they are not related to each other and can be run in parallel)
   - completed: Task finished successfully

2. **Task Management**:
   - Update task status in real-time as you work
   - Mark tasks complete IMMEDIATELY after finishing (don't batch completions)
   - Complete current tasks before starting new ones
   - Remove tasks that are no longer relevant from the list entirely
   - IMPORTANT: When you write this todo list, you should mark your first task (or tasks) as in_progress immediately!.
   - IMPORTANT: Unless all tasks are completed, you should always have at least one task in_progress to show the user that you are working on something.

3. **Task Completion Requirements**:
   - ONLY mark a task as completed when you have FULLY accomplished it
   - If you encounter errors, blockers, or cannot finish, keep the task as in_progress
   - When blocked, create a new task describing what needs to be resolved
   - Never mark a task as completed if:
     - There are unresolved issues or errors
     - Work is partial or incomplete
     - You encountered blockers that prevent completion
     - You couldn't find necessary resources or dependencies
     - Quality standards haven't been met

4. **Task Breakdown**:
   - Create specific, actionable items
   - Break complex tasks into smaller, manageable steps
   - Use clear, descriptive task names

Being proactive with task management demonstrates attentiveness and ensures you complete all requirements successfully.
Remember: If you only need to make a few tool calls to complete a task, and it is clear what you need to do, it is better to just do the task directly and NOT call this tool at all.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "items": {
                    "type": "object",
                    "description": "A single todo item with content and status.",
                    "properties": {
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for the todo item"
                        },
                        "content": {
                            "type": "string",
                            "maxLength": 100,
                            "description": "The description/content of the todo item"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"],
                            "description": "The current status of the todo item"
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": "Whether to merge the todos with existing todos (true) or replace entirely (false). Default is true.",
                "default": True
            }
        },
        "required": ["todos"]
    }
}

# =============================================================================
# Filesystem Tools
# =============================================================================

LS_TOOL = {
    "name": "ls",
    "description": """Lists all files in the filesystem, filtering by directory.

Usage:
- The path parameter must be an absolute path, not a relative path
- Returns a list of all files in the specified directory.
- This is very useful for exploring the file system and finding the right file to read or edit.
- You should almost ALWAYS use this tool before using the Read or Edit tools.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "The absolute path to the directory to list"
            }
        },
        "required": ["path"]
    }
}

READ_FILE_TOOL = {
    "name": "read_file",
    "description": """Reads a file from the filesystem. You can access any file directly by using this tool.
Assume this tool is able to read all files on the machine. If the User provides a path to a file assume that path is valid. It is okay to read a file that does not exist; an error will be returned.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- By default, it reads up to 500 lines starting from the beginning of the file
- **IMPORTANT for large files and codebase exploration**: Use pagination with offset and limit parameters to avoid context overflow
  - First scan: read_file(path, limit=100) to see file structure
  - Read more sections: read_file(path, offset=100, limit=200) for next 200 lines
  - Only omit limit (read full file) when necessary for editing
- Specify offset and limit: read_file(path, offset=0, limit=100) reads first 100 lines
- Any lines longer than 2000 characters will be truncated
- Results are returned using cat -n format, with line numbers starting at 1
- You have the capability to call multiple tools in a single response. It is always better to speculatively read multiple files as a batch that are potentially useful.
- If you read a file that exists but has empty contents you will receive a system reminder warning in place of file contents.
- You should ALWAYS make sure a file has been read before editing it.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to read"
            },
            "offset": {
                "type": "integer",
                "default": 0,
                "description": "The line number to start reading from (0-indexed)"
            },
            "limit": {
                "type": "integer",
                "default": 500,
                "description": "The maximum number of lines to read"
            }
        },
        "required": ["file_path"]
    }
}

WRITE_FILE_TOOL = {
    "name": "write_file",
    "description": """Writes to a new file in the filesystem.

Usage:
- The file_path parameter must be an absolute path, not a relative path
- The content parameter must be a string
- This tool will create a new file or overwrite an existing file.
- Creates parent directories automatically if they don't exist.
- Prefer to edit existing files over creating new ones when possible.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to write"
            },
            "content": {
                "type": "string",
                "description": "The content to write to the file"
            }
        },
        "required": ["file_path", "content"]
    }
}

EDIT_FILE_TOOL = {
    "name": "edit_file",
    "description": """Performs exact string replacements in files.

Usage:
- You must use your `read_file` tool at least once in the conversation before editing. This tool will error if you attempt an edit without reading the file.
- When editing text from Read tool output, ensure you preserve the exact indentation (tabs/spaces) as it appears AFTER the line number prefix. The line number prefix format is: spaces + line number + tab. Everything after that tab is the actual file content to match. Never include any part of the line number prefix in the old_string or new_string.
- ALWAYS prefer editing existing files. NEVER write new files unless explicitly required.
- Only use emojis if the user explicitly requests it. Avoid adding emojis to files unless asked.
- The edit will FAIL if `old_string` is not unique in the file. Either provide a larger string with more surrounding context to make it unique or use `replace_all` to change every instance of `old_string`.
- Use `replace_all` for replacing and renaming strings across the file. This parameter is useful if you want to rename a variable for instance.""",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "The absolute path to the file to edit"
            },
            "old_string": {
                "type": "string",
                "description": "The text to replace (must be unique in the file unless replace_all is true)"
            },
            "new_string": {
                "type": "string",
                "description": "The text to replace it with"
            },
            "replace_all": {
                "type": "boolean",
                "default": False,
                "description": "Replace all occurrences of old_string (default false)"
            }
        },
        "required": ["file_path", "old_string", "new_string"]
    }
}

GLOB_TOOL = {
    "name": "glob",
    "description": """Find files matching a glob pattern.

Usage:
- The glob tool finds files by matching patterns with wildcards
- Supports standard glob patterns: `*` (any characters), `**` (any directories), `?` (single character)
- Patterns can be absolute (starting with `/`) or relative
- Returns a list of absolute file paths that match the pattern

Examples:
- `**/*.py` - Find all Python files
- `*.txt` - Find all text files in root
- `/subdir/**/*.md` - Find all markdown files under /subdir""",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The glob pattern to match files against"
            },
            "path": {
                "type": "string",
                "default": "/",
                "description": "The base path to search from (default is root)"
            }
        },
        "required": ["pattern"]
    }
}

GREP_TOOL = {
    "name": "grep",
    "description": """Search for a pattern in files.

Usage:
- The grep tool searches for text patterns across files
- The pattern parameter is the text to search for (supports regex)
- The path parameter filters which directory to search in (default is the current working directory)
- The glob parameter accepts a glob pattern to filter which files to search (e.g., `*.py`)
- The output_mode parameter controls the output format:
  - `files_with_matches`: List only file paths containing matches (default)
  - `content`: Show matching lines with file path and line numbers
  - `count`: Show count of matches per file

Examples:
- Search all files: `grep(pattern="TODO")`
- Search Python files only: `grep(pattern="import", glob="*.py")`
- Show matching lines: `grep(pattern="error", output_mode="content")`""",
    "input_schema": {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "The pattern to search for (supports regex)"
            },
            "path": {
                "type": ["string", "null"],
                "default": None,
                "description": "The directory to search in (default is current working directory)"
            },
            "glob": {
                "type": ["string", "null"],
                "default": None,
                "description": "Glob pattern to filter which files to search"
            },
            "output_mode": {
                "type": "string",
                "enum": ["files_with_matches", "content", "count"],
                "default": "files_with_matches",
                "description": "Output format for search results"
            }
        },
        "required": ["pattern"]
    }
}

# =============================================================================
# Shell/Execute Tools
# =============================================================================

BASH_TOOL = {
    "name": "bash",
    "description": """Executes a bash command in a sandboxed environment.

Usage:
- Use this to run shell commands for various tasks like running tests, installing packages, or executing scripts
- Commands run in a sandboxed environment with limited permissions
- Long-running commands should be avoided; use background processes if needed
- For file operations, prefer using the dedicated file tools (read_file, write_file, edit_file) instead
- Output is captured and returned, including both stdout and stderr

Security Notes:
- Network access may be restricted depending on sandbox configuration
- Some system operations may be blocked for security
- File system access is limited to the workspace and allowed paths""",
    "input_schema": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The bash command to execute"
            },
            "timeout": {
                "type": "integer",
                "default": 30,
                "description": "Timeout in seconds for command execution"
            },
            "working_directory": {
                "type": "string",
                "default": None,
                "description": "Working directory for the command (default is workspace root)"
            }
        },
        "required": ["command"]
    }
}

# =============================================================================
# Sub-Agent / Task Delegation Tools
# =============================================================================

TASK_TOOL = {
    "name": "task",
    "description": """Launch an ephemeral subagent to handle complex, multi-step independent tasks with isolated context windows.

Available agent types and the tools they have access to:
- general-purpose: General-purpose agent for researching complex questions, searching for files and content, and executing multi-step tasks. When you are searching for a keyword or file and are not confident that you will find the right match in the first few tries use this agent to perform the search for you. This agent has access to all tools as the main agent.

When using the Task tool, you must specify a subagent_type parameter to select which agent type to use.

## Usage notes:
1. Launch multiple agents concurrently whenever possible, to maximize performance; to do that, use a single message with multiple tool uses
2. When the agent is done, it will return a single message back to you. The result returned by the agent is not visible to the user. To show the user the result, you should send a text message back to the user with a concise summary of the result.
3. Each agent invocation is stateless. You will not be able to send additional messages to the agent, nor will the agent be able to communicate with you outside of its final report. Therefore, your prompt should contain a highly detailed task description for the agent to perform autonomously and you should specify exactly what information the agent should return back to you in its final and only message to you.
4. The agent's outputs should generally be trusted
5. Clearly tell the agent whether you expect it to create content, perform analysis, or just do research (search, file reads, web fetches, etc.), since it is not aware of the user's intent
6. If the agent description mentions that it should be used proactively, then you should try your best to use it without the user having to ask for it first. Use your judgement.
7. When only the general-purpose agent is provided, you should use it for all tasks. It is great for isolating context and token usage, and completing specific, complex tasks, as it has all the same capabilities as the main agent.

### When to use the task tool:
- When a task is complex and multi-step, and can be fully delegated in isolation
- When a task is independent of other tasks and can run in parallel
- When a task requires focused reasoning or heavy token/context usage that would bloat the orchestrator thread
- When sandboxing improves reliability (e.g. code execution, structured searches, data formatting)
- When you only care about the output of the subagent, and not the intermediate steps

### When NOT to use the task tool:
- If you need to see the intermediate reasoning or steps after the subagent has completed (the task tool hides them)
- If the task is trivial (a few tool calls or simple lookup)
- If delegating does not reduce token usage, complexity, or context switching
- If splitting would add latency without benefit

### Subagent lifecycle:
1. **Spawn** → Provide clear role, instructions, and expected output
2. **Run** → The subagent completes the task autonomously
3. **Return** → The subagent provides a single structured result
4. **Reconcile** → Incorporate or synthesize the result into the main thread""",
    "input_schema": {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Detailed task description for the subagent to perform autonomously"
            },
            "subagent_type": {
                "type": "string",
                "description": "The type of subagent to use (e.g., 'general-purpose')"
            }
        },
        "required": ["description", "subagent_type"]
    }
}

# =============================================================================
# Web/Search Tools (Optional)
# =============================================================================

WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": """Search the web for real-time information about any topic.

Use this tool when you need up-to-date information that might not be available in your training data, or when you need to verify current facts. The search results will include relevant snippets and URLs from web pages.

This is particularly useful for:
- Questions about current events
- Technology updates or latest versions
- Any topic that requires recent information
- Verifying facts that may have changed""",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific and include relevant keywords for better results."
            },
            "max_results": {
                "type": "integer",
                "default": 5,
                "description": "Maximum number of search results to return"
            }
        },
        "required": ["query"]
    }
}

# =============================================================================
# All Tool Definitions
# =============================================================================

TOOL_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Planning
    "write_todos": WRITE_TODOS_TOOL,
    
    # Filesystem
    "ls": LS_TOOL,
    "read_file": READ_FILE_TOOL,
    "write_file": WRITE_FILE_TOOL,
    "edit_file": EDIT_FILE_TOOL,
    "glob": GLOB_TOOL,
    "grep": GREP_TOOL,
    
    # Shell
    "bash": BASH_TOOL,
    
    # Sub-agent
    "task": TASK_TOOL,
    
    # Web (optional)
    "web_search": WEB_SEARCH_TOOL,
}

# Default tools to include
DEFAULT_TOOLS = [
    "write_todos",
    "ls",
    "read_file", 
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "bash",
    "task",
]


def get_tool_by_name(name: str) -> Dict[str, Any] | None:
    """Get a tool definition by name."""
    return TOOL_DEFINITIONS.get(name)


def get_all_tools(include_optional: bool = False) -> List[Dict[str, Any]]:
    """
    Get all tool definitions as a list for the API.
    
    Args:
        include_optional: Whether to include optional tools like web_search
        
    Returns:
        List of tool definitions ready for the Anthropic API
    """
    if include_optional:
        return list(TOOL_DEFINITIONS.values())
    
    return [TOOL_DEFINITIONS[name] for name in DEFAULT_TOOLS if name in TOOL_DEFINITIONS]


def get_tools_for_api(tool_names: List[str] | None = None) -> List[Dict[str, Any]]:
    """
    Get tool definitions formatted for the Anthropic API.
    
    Args:
        tool_names: List of tool names to include. If None, returns default tools.
        
    Returns:
        List of tool definitions
    """
    if tool_names is None:
        tool_names = DEFAULT_TOOLS
    
    return [
        TOOL_DEFINITIONS[name] 
        for name in tool_names 
        if name in TOOL_DEFINITIONS
    ]

