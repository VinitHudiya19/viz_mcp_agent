"""
core/base_tool.py

Decorator-based foundation for all visualization tools.

Instead of an abstract class, each tool is a plain function decorated with
@viz_tool.  The decorator handles:
  - matplotlib style context
  - exception catching (never crashes the MCP server)
  - automatic Figure → ToolResult conversion
  - auto-registration into TOOL_REGISTRY

Shared helpers (apply_title, style_spines, rotate_labels, PALETTE) are
plain module-level functions imported by the tool functions.
"""

from __future__ import annotations

import base64
import functools
import io
import logging
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no display needed
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result type returned by every tool
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    success: bool
    image_b64: str | None = None        # PNG encoded as base64 string
    image_bytes: bytes | None = None    # raw PNG bytes (for file writing)
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_figure(cls, fig: plt.Figure, metadata: dict | None = None) -> "ToolResult":
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        raw = buf.read()
        return cls(
            success=True,
            image_bytes=raw,
            image_b64=base64.b64encode(raw).decode("utf-8"),
            metadata=metadata or {},
        )

    @classmethod
    def from_error(cls, error: str) -> "ToolResult":
        return cls(success=False, error=error)


# ---------------------------------------------------------------------------
# Input spec — what every tool declares it needs from the caller
# ---------------------------------------------------------------------------

@dataclass
class ToolInputSpec:
    """
    Describes the columns a tool needs.
    Used by the MCP server to validate the LLM's column selections
    before loading anything from Parquet.
    """
    required_columns: list[str]
    optional_columns: list[str] = field(default_factory=list)
    # Column kind requirements: {"x_col": "categorical", "y_col": "numeric"}
    kind_requirements: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Shared style & palette
# ---------------------------------------------------------------------------

STYLE: str = "seaborn-v0_8-whitegrid"

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52",
    "#8172B3", "#937860", "#DA8BC3", "#8C8C8C",
    "#CCB974", "#64B5CD",
]


# ---------------------------------------------------------------------------
# Shared helpers — available to all tool functions
# ---------------------------------------------------------------------------

def apply_title(ax: plt.Axes, title: str) -> None:
    if title:
        ax.set_title(title, fontsize=13, fontweight="semibold", pad=12)


def style_spines(ax: plt.Axes) -> None:
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_linewidth(0.6)


def rotate_labels(ax: plt.Axes, n_labels: int) -> None:
    """Auto-rotate x-axis labels when there are many categories."""
    if n_labels > 8:
        ax.tick_params(axis="x", rotation=45)
    elif n_labels > 4:
        ax.tick_params(axis="x", rotation=30)


# ---------------------------------------------------------------------------
# Tool registry — auto-populated by the @viz_tool decorator
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Callable] = {}


# ---------------------------------------------------------------------------
# @viz_tool decorator
# ---------------------------------------------------------------------------

def viz_tool(
    name: str,
    description: str,
    schema: ToolInputSpec,
) -> Callable:
    """
    Decorator that turns a plain function into a registered visualization tool.

    The decorated function must accept (data, title, **kwargs) and return
    a matplotlib Figure.  The decorator wraps it to:
      1. Apply the shared matplotlib style context
      2. Convert the returned Figure into a ToolResult (base64 PNG)
      3. Catch any exception and return ToolResult.from_error()
      4. Register the function in TOOL_REGISTRY by name

    Usage:
        @viz_tool(
            name="bar_chart",
            description="Draws a vertical bar chart …",
            schema=ToolInputSpec(required_columns=["x_col", "y_col"], …),
        )
        def create_bar_chart(data, title, **kwargs):
            ...
            return fig
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(data: dict[str, Any], title: str = "", **kwargs) -> ToolResult:
            try:
                with plt.style.context(STYLE):
                    fig = func(data, title, **kwargs)
                return ToolResult.from_figure(
                    fig,
                    metadata={"tool": name, "title": title},
                )
            except Exception as exc:
                logger.error(
                    "Tool %s failed: %s\n%s",
                    name, exc, traceback.format_exc(),
                )
                return ToolResult.from_error(
                    f"[{name}] {type(exc).__name__}: {exc}"
                )

        # Attach metadata so the MCP server can read it
        wrapper.tool_name = name
        wrapper.description = description
        wrapper.input_schema = schema

        # Auto-register
        TOOL_REGISTRY[name] = wrapper
        return wrapper

    return decorator
