"""
sandbox — Optional E2B isolation layer for chart rendering.

NOT used by the MCP server in normal operation.
The MCP server renders charts directly (in-process matplotlib).

To enable sandbox execution, import and use SandboxExecutor:

    from sandbox.executor import SandboxExecutor

    with SandboxExecutor() as executor:
        png_b64 = executor.run_tool("bar_chart", data, title="Chart")
"""

from .executor import SandboxExecutor, SandboxExecutionError

__all__ = ["SandboxExecutor", "SandboxExecutionError"]
