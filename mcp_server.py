"""
mcp_server.py
────────────────────────────────────────────────────────────────────────────
MCP Server for the Visualization Agent.

Part of a multi-agent system.  This server does NOT decide which chart to
make — the orchestrator agent has already decided.  It receives:
  • tool name  (e.g. "bar_chart")
  • data arrays as JSON  (from the data-profiling agent)
  • title / options

It runs the hardcoded chart function, renders the figure, and returns
the result as a base64-encoded PNG image.

Transport: stdio  (the orchestrator connects via subprocess / MCP client)
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

import numpy as np
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

# ── load environment variables ───────────────────────────────────────────────
load_dotenv()

# ── import all tool functions (this triggers @viz_tool auto-registration) ────
import tools.visualizations  # noqa: F401  — side-effect import fills TOOL_REGISTRY
from core.base_tool import TOOL_REGISTRY, ToolResult

logger = logging.getLogger("viz_mcp_agent")
logging.basicConfig(level=logging.INFO, format="%(name)s | %(levelname)s | %(message)s")


# ── MCP server instance ─────────────────────────────────────────────────────

mcp = FastMCP(
    name="viz-agent",
    version="1.0.0",
)


# ═══════════════════════════════════════════════════════════════════════════
# MCP TOOLS — one per visualization type
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def bar_chart(
    x_values: list,
    y_values: list,
    title: str = "Bar Chart",
    x_label: str = "",
    y_label: str = "",
    color: str | None = None,
    show_values: bool = True,
    sort_bars: bool = False,
    orientation: str = "vertical",
) -> str:
    """
    Draws a vertical or horizontal bar chart comparing a numeric value
    across categories.  Best for: monthly revenue, sales by region,
    count by category.
    """
    data = {
        "x_col": np.array(x_values),
        "y_col": np.array(y_values, dtype=float),
        "x_col_name": x_label,
        "y_col_name": y_label,
    }
    result: ToolResult = TOOL_REGISTRY["bar_chart"](
        data, title,
        color=color, show_values=show_values,
        sort_bars=sort_bars, orientation=orientation,
    )
    return _format_result(result)


@mcp.tool()
def line_chart(
    x_values: list,
    y_values: list,
    title: str = "Line Chart",
    x_label: str = "",
    y_label: str = "",
    group_values: list | None = None,
    show_markers: bool = True,
    show_area: bool = False,
) -> str:
    """
    Draws a line chart showing a numeric trend over time or an ordered axis.
    Supports multiple series via group_values.
    Best for: revenue over months, temperature over days, stock prices.
    """
    data: dict[str, Any] = {
        "x_col": np.array(x_values),
        "y_col": np.array(y_values, dtype=float),
        "x_col_name": x_label,
        "y_col_name": y_label,
    }
    if group_values is not None:
        data["group_col"] = np.array(group_values)

    result: ToolResult = TOOL_REGISTRY["line_chart"](
        data, title,
        show_markers=show_markers, show_area=show_area,
    )
    return _format_result(result)


@mcp.tool()
def scatter_plot(
    x_values: list,
    y_values: list,
    title: str = "Scatter Plot",
    x_label: str = "",
    y_label: str = "",
    color_values: list | None = None,
    size_values: list | None = None,
    show_regression: bool = True,
    show_correlation: bool = True,
) -> str:
    """
    Draws a scatter plot to reveal correlation between two numeric columns.
    Optional color_values to color-code points by group.
    Best for: ad_spend vs revenue, age vs salary, temperature vs sales.
    """
    data: dict[str, Any] = {
        "x_col": np.array(x_values, dtype=float),
        "y_col": np.array(y_values, dtype=float),
        "x_col_name": x_label,
        "y_col_name": y_label,
    }
    if color_values is not None:
        data["color_col"] = np.array(color_values)
    if size_values is not None:
        data["size_col"] = np.array(size_values, dtype=float)

    result: ToolResult = TOOL_REGISTRY["scatter_plot"](
        data, title,
        show_regression=show_regression,
        show_correlation=show_correlation,
    )
    return _format_result(result)


@mcp.tool()
def histogram(
    x_values: list,
    title: str = "Histogram",
    x_label: str = "",
    bins: int = 0,
    show_kde: bool = True,
    show_stats: bool = True,
    show_percentiles: bool = True,
) -> str:
    """
    Draws a histogram showing distribution of a single numeric column.
    Adds KDE curve and summary statistics (mean, median, std).
    Best for: salary distribution, age distribution, response times.
    """
    data = {
        "x_col": np.array(x_values, dtype=float),
        "x_col_name": x_label,
    }
    result: ToolResult = TOOL_REGISTRY["histogram"](
        data, title,
        bins=bins if bins > 0 else "auto",
        show_kde=show_kde, show_stats=show_stats,
        show_percentiles=show_percentiles,
    )
    return _format_result(result)


@mcp.tool()
def box_plot(
    group_values: list,
    value_values: list,
    title: str = "Box Plot",
    group_label: str = "",
    value_label: str = "",
    show_points: bool = True,
    show_means: bool = True,
    orient: str = "v",
) -> str:
    """
    Draws a box plot comparing distribution of a numeric column across groups.
    Best for: salary by department, test scores by grade, response time by region.
    """
    data = {
        "group_col": np.array(group_values),
        "value_col": np.array(value_values, dtype=float),
        "group_col_name": group_label,
        "value_col_name": value_label,
    }
    result: ToolResult = TOOL_REGISTRY["box_plot"](
        data, title,
        show_points=show_points, show_means=show_means, orient=orient,
    )
    return _format_result(result)


@mcp.tool()
def heatmap(
    row_values: list,
    col_values: list,
    value_values: list,
    title: str = "Heatmap",
    row_label: str = "",
    col_label: str = "",
    value_label: str = "",
    aggfunc: str = "mean",
    cmap: str = "YlOrRd",
    annot: bool = True,
) -> str:
    """
    Draws a heatmap showing a numeric metric's intensity across two
    categorical axes.  Aggregates values per cell.
    Best for: sales by region × month, error rate by service × hour.
    """
    data = {
        "row_col": np.array(row_values),
        "col_col": np.array(col_values),
        "value_col": np.array(value_values, dtype=float),
        "row_col_name": row_label,
        "col_col_name": col_label,
        "value_col_name": value_label,
    }
    result: ToolResult = TOOL_REGISTRY["heatmap"](
        data, title,
        aggfunc=aggfunc, cmap=cmap, annot=annot,
    )
    return _format_result(result)


@mcp.tool()
def pie_chart(
    category_values: list,
    title: str = "Pie Chart",
    value_values: list | None = None,
    donut: bool = True,
    show_pct: bool = True,
    show_legend: bool = True,
) -> str:
    """
    Draws a donut/pie chart showing proportional breakdown.
    If value_values is provided, sums per category; otherwise counts.
    Best for: market share, budget allocation, survey responses.
    NOTE: only effective when there are ≤ 8 distinct categories.
    """
    data: dict[str, Any] = {
        "category_col": np.array(category_values),
    }
    if value_values is not None:
        data["value_col"] = np.array(value_values, dtype=float)

    result: ToolResult = TOOL_REGISTRY["pie_chart"](
        data, title,
        donut=donut, show_pct=show_pct, show_legend=show_legend,
    )
    return _format_result(result)


@mcp.tool()
def area_chart(
    x_values: list,
    y_values: list,
    title: str = "Area Chart",
    x_label: str = "",
    y_label: str = "",
    group_values: list | None = None,
    stacked: bool = True,
    show_lines: bool = True,
) -> str:
    """
    Draws an area chart (filled line) showing volume over time.
    Supports stacking multiple series via group_values.
    Best for: cumulative revenue, traffic volume, resource usage over time.
    """
    data: dict[str, Any] = {
        "x_col": np.array(x_values),
        "y_col": np.array(y_values, dtype=float),
        "x_col_name": x_label,
        "y_col_name": y_label,
    }
    if group_values is not None:
        data["group_col"] = np.array(group_values)

    result: ToolResult = TOOL_REGISTRY["area_chart"](
        data, title,
        stacked=stacked, show_lines=show_lines,
    )
    return _format_result(result)


@mcp.tool()
def stacked_bar_chart(
    x_values: list,
    y_values: list,
    group_values: list,
    title: str = "Stacked Bar Chart",
    x_label: str = "",
    y_label: str = "",
    group_label: str = "",
    orientation: str = "vertical",
    show_values: bool = False,
    show_totals: bool = True,
) -> str:
    """
    Draws a stacked bar chart showing how sub-groups contribute to totals.
    Each bar is split into colored segments.
    Best for: revenue by product per quarter, expenses by department per month.
    """
    data = {
        "x_col": np.array(x_values),
        "y_col": np.array(y_values, dtype=float),
        "group_col": np.array(group_values),
        "y_col_name": y_label,
        "group_col_name": group_label,
    }
    result: ToolResult = TOOL_REGISTRY["stacked_bar_chart"](
        data, title,
        orientation=orientation, show_values=show_values, show_totals=show_totals,
    )
    return _format_result(result)


@mcp.tool()
def grouped_bar_chart(
    x_values: list,
    y_values: list,
    group_values: list,
    title: str = "Grouped Bar Chart",
    x_label: str = "",
    y_label: str = "",
    group_label: str = "",
    show_values: bool = True,
    orientation: str = "vertical",
) -> str:
    """
    Draws a grouped (clustered) bar chart with bars side-by-side per category.
    Best for: male vs female salary by department, product A vs B by quarter.
    """
    data = {
        "x_col": np.array(x_values),
        "y_col": np.array(y_values, dtype=float),
        "group_col": np.array(group_values),
        "y_col_name": y_label,
        "group_col_name": group_label,
    }
    result: ToolResult = TOOL_REGISTRY["grouped_bar_chart"](
        data, title,
        show_values=show_values, orientation=orientation,
    )
    return _format_result(result)


@mcp.tool()
def correlation_matrix(
    columns_data: dict[str, list],
    title: str = "Correlation Matrix",
    method: str = "pearson",
    annot: bool = True,
    mask_upper: bool = True,
) -> str:
    """
    Draws a correlation matrix heatmap for multiple numeric columns.
    Pass columns_data as {column_name: [values...]}.
    Best for: feature correlation analysis, multicollinearity check, EDA overview.
    """
    # Convert each list to numpy array
    converted = {k: np.array(v, dtype=float) for k, v in columns_data.items()}
    data = {"columns": converted}
    result: ToolResult = TOOL_REGISTRY["correlation_matrix"](
        data, title,
        method=method, annot=annot, mask_upper=mask_upper,
    )
    return _format_result(result)


@mcp.tool()
def count_plot(
    x_values: list,
    title: str = "Count Plot",
    x_label: str = "",
    group_values: list | None = None,
    group_label: str = "",
    show_values: bool = True,
    show_pct: bool = True,
    sort_by_count: bool = True,
    orientation: str = "vertical",
) -> str:
    """
    Draws a count plot showing frequency of each category.
    Optional group_values for grouped counts.
    Best for: most common categories, class imbalance check, survey responses.
    """
    data: dict[str, Any] = {
        "x_col": np.array(x_values),
        "x_col_name": x_label,
    }
    if group_values is not None:
        data["group_col"] = np.array(group_values)
        data["group_col_name"] = group_label

    result: ToolResult = TOOL_REGISTRY["count_plot"](
        data, title,
        show_values=show_values, show_pct=show_pct,
        sort_by_count=sort_by_count, orientation=orientation,
    )
    return _format_result(result)


@mcp.tool()
def dual_axis_time_series(
    x_values: list,
    y1_values: list,
    y2_values: list,
    title: str = "Dual Axis Time Series",
    x_label: str = "",
    y1_label: str = "Metric 1",
    y2_label: str = "Metric 2",
    y1_style: str = "line",
    y2_style: str = "line",
    show_markers: bool = True,
) -> str:
    """
    Draws a dual-axis time series with two y-axes sharing the same x-axis.
    Best for: revenue vs customers over time, temperature vs humidity,
    stock price vs volume.
    """
    data = {
        "x_col": np.array(x_values),
        "y1_col": np.array(y1_values, dtype=float),
        "y2_col": np.array(y2_values, dtype=float),
        "x_col_name": x_label,
        "y1_col_name": y1_label,
        "y2_col_name": y2_label,
    }
    result: ToolResult = TOOL_REGISTRY["dual_axis_time_series"](
        data, title,
        y1_style=y1_style, y2_style=y2_style, show_markers=show_markers,
    )
    return _format_result(result)


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY TOOL — list available visualizations
# ═══════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_available_charts() -> str:
    """
    Returns a JSON list of all available chart types, their descriptions,
    and required input fields.  The orchestrator can call this to decide
    which chart tool to invoke.
    """
    charts = []
    for name, fn in TOOL_REGISTRY.items():
        charts.append({
            "tool_name": name,
            "description": fn.description,
            "required_columns": fn.input_schema.required_columns,
            "optional_columns": fn.input_schema.optional_columns,
        })
    return json.dumps(charts, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _format_result(result: ToolResult) -> str:
    """
    Convert a ToolResult into the string returned to the MCP caller.
    On success → JSON with base64 image + metadata.
    On failure → JSON with error message.
    """
    if result.success:
        return json.dumps({
            "success": True,
            "image_base64": result.image_b64,
            "metadata": result.metadata,
        })
    else:
        return json.dumps({
            "success": False,
            "error": result.error,
        })


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logger.info("Starting viz-agent MCP server (stdio transport)…")
    logger.info(f"Registered tools: {list(TOOL_REGISTRY.keys())}")
    
    # Current (local only):
    mcp.run(transport="stdio")

    # Cloud (HTTP endpoint):
    # mcp.run(transport="sse", host="0.0.0.0", port=8001)
