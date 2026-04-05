"""core — base tool decorator, ToolResult, shared helpers."""

from .base_tool import (
    ToolResult,
    ToolInputSpec,
    viz_tool,
    TOOL_REGISTRY,
    PALETTE,
    apply_title,
    style_spines,
    rotate_labels,
)

__all__ = [
    "ToolResult",
    "ToolInputSpec",
    "viz_tool",
    "TOOL_REGISTRY",
    "PALETTE",
    "apply_title",
    "style_spines",
    "rotate_labels",
]
