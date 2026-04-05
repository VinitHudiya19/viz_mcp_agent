"""tools — All hardcoded visualization tool functions for viz_mcp_agent."""

from .visualizations import (
    create_bar_chart,
    create_box_plot,
    create_heatmap,
    create_histogram,
    create_line_chart,
    create_pie_chart,
    create_scatter_plot,
    create_area_chart,
    create_stacked_bar_chart,
    create_grouped_bar_chart,
    create_correlation_matrix,
    create_count_plot,
    create_dual_axis_time_series,
)

# Re-export the auto-populated registry from base_tool
from core.base_tool import TOOL_REGISTRY

__all__ = [
    "create_bar_chart",
    "create_box_plot",
    "create_heatmap",
    "create_histogram",
    "create_line_chart",
    "create_pie_chart",
    "create_scatter_plot",
    "create_area_chart",
    "create_stacked_bar_chart",
    "create_grouped_bar_chart",
    "create_correlation_matrix",
    "create_count_plot",
    "create_dual_axis_time_series",
    "TOOL_REGISTRY",
]
