"""
tools/visualizations.py
────────────────────────────────────────────────────────────────────────────
Single-file registry of ALL hardcoded visualization tools.

Each function is decorated with @viz_tool which handles error catching,
figure conversion, and auto-registration.  The plotting logic inside each
function never changes — only the column arrays injected at runtime differ.

Tools
─────
  create_bar_chart      – vertical / horizontal bar chart
  create_box_plot       – box plot with optional strip overlay
  create_heatmap        – 2-axis intensity heatmap
  create_histogram      – distribution histogram with KDE + stats
  create_line_chart     – trend line chart (single / multi-series)
  create_pie_chart      – donut / pie with auto-fallback to bar
  create_scatter_plot   – scatter with optional regression + color encoding
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import gaussian_kde

from core.base_tool import (
    ToolInputSpec,
    viz_tool,
    apply_title,
    style_spines,
    rotate_labels,
    PALETTE,
)


# ═══════════════════════════════════════════════════════════════════════════
# 1. BAR CHART
# ═══════════════════════════════════════════════════════════════════════════
# Use when: comparing a numeric value across discrete categories or time periods.
# LLM should call this with x_col (categorical/datetime) + y_col (numeric).

@viz_tool(
    name="bar_chart",
    description=(
        "Draws a vertical bar chart comparing a numeric column across "
        "categories or time periods. "
        "Requires: x_col (categorical or datetime), y_col (numeric). "
        "Best for: monthly revenue, sales by region, count by category."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y_col"],
        kind_requirements={"y_col": "numeric"},
    ),
)
def create_bar_chart(
    data: dict[str, Any],
    title: str,
    color: str | None = None,
    show_values: bool = True,
    sort_bars: bool = False,
    orientation: str = "vertical",   # "vertical" | "horizontal"
    **kwargs,
) -> plt.Figure:

    x = data["x_col"]
    y = data["y_col"].astype(float)

    # Aggregate if x has duplicates (e.g. multiple rows per month)
    series = pd.Series(y, index=x)
    series = series.groupby(level=0, sort=False).sum()

    if sort_bars:
        series = series.sort_values(ascending=(orientation == "horizontal"))

    labels = series.index.astype(str).tolist()
    values = series.values
    n = len(labels)

    bar_color = color or PALETTE[0]

    fig, ax = plt.subplots(figsize=(max(7, n * 0.6), 5))

    if orientation == "horizontal":
        bars = ax.barh(labels, values, color=bar_color, edgecolor="white", linewidth=0.5)
        ax.set_xlabel(data.get("y_col_name", "value"), fontsize=11)
        if show_values:
            for bar in bars:
                w = bar.get_width()
                ax.text(
                    w + max(values) * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{w:,.0f}",
                    va="center", ha="left", fontsize=9,
                )
    else:
        bars = ax.bar(
            np.arange(n), values,
            color=bar_color, edgecolor="white", linewidth=0.5,
            width=0.65,
        )
        ax.set_xticks(np.arange(n))
        ax.set_xticklabels(labels)
        rotate_labels(ax, n)
        ax.set_ylabel(data.get("y_col_name", "value"), fontsize=11)

        if show_values:
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h + max(values) * 0.01,
                    f"{h:,.0f}",
                    ha="center", va="bottom", fontsize=9,
                )

    apply_title(ax, title)
    style_spines(ax)
    ax.yaxis.set_tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 2. BOX PLOT
# ═══════════════════════════════════════════════════════════════════════════
# Use when: comparing distributions of a numeric column across multiple groups,
# especially to spot outliers and spread differences.

_BOX_MAX_GROUPS = 20

@viz_tool(
    name="box_plot",
    description=(
        "Draws a box plot (with optional strip overlay) comparing the distribution "
        "of a numeric column across groups. "
        "Requires: group_col (categorical), value_col (numeric). "
        "Best for: salary by department, test scores by grade, response time by region."
    ),
    schema=ToolInputSpec(
        required_columns=["group_col", "value_col"],
        kind_requirements={"value_col": "numeric"},
    ),
)
def create_box_plot(
    data: dict[str, Any],
    title: str,
    show_points: bool = True,
    show_means: bool = True,
    orient: str = "v",          # "v" vertical | "h" horizontal
    notch: bool = False,
    **kwargs,
) -> plt.Figure:

    groups = data["group_col"].astype(str)
    values = data["value_col"].astype(float)

    df = pd.DataFrame({"group": groups, "value": values})
    df = df[np.isfinite(df["value"])]

    # Order groups by median for readability
    order = (
        df.groupby("group")["value"]
        .median()
        .sort_values(ascending=(orient == "h"))
        .index.tolist()
    )

    if len(order) > _BOX_MAX_GROUPS:
        order = order[:_BOX_MAX_GROUPS]
        df = df[df["group"].isin(order)]

    n_groups = len(order)
    palette = {grp: PALETTE[i % len(PALETTE)] for i, grp in enumerate(order)}

    fig_w = max(6, n_groups * 0.9 + 1.5) if orient == "v" else 8
    fig_h = 5 if orient == "v" else max(4, n_groups * 0.6 + 1)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    x_key = "group" if orient == "v" else "value"
    y_key = "value" if orient == "v" else "group"
    orient_key = orient

    sns.boxplot(
        data=df, x=x_key, y=y_key,
        order=order if orient == "v" else None,
        hue="group",
        palette=palette,
        notch=notch,
        width=0.55,
        flierprops=dict(marker="o", markersize=3, alpha=0.4,
                        markerfacecolor="#888888", markeredgewidth=0),
        medianprops=dict(color="white", linewidth=2),
        boxprops=dict(edgecolor="white", linewidth=0.5),
        whiskerprops=dict(linewidth=0.8),
        capprops=dict(linewidth=0.8),
        legend=False,
        ax=ax,
    )

    # Strip plot (individual points jittered)
    if show_points and len(df) <= 2000:
        sns.stripplot(
            data=df, x=x_key, y=y_key,
            order=order if orient == "v" else None,
            hue="group",
            palette=palette,
            size=2.5, alpha=0.35, jitter=True, dodge=False,
            legend=False,
            ax=ax,
        )

    # Mean markers
    if show_means:
        means = df.groupby("group")["value"].mean()
        for i, grp in enumerate(order):
            if grp in means:
                pos = i
                mean_val = means[grp]
                if orient == "v":
                    ax.plot(pos, mean_val, marker="D", color="white",
                            markersize=5, zorder=5, markeredgecolor="#555")
                else:
                    ax.plot(mean_val, pos, marker="D", color="white",
                            markersize=5, zorder=5, markeredgecolor="#555")

    if orient == "v":
        rotate_labels(ax, n_groups)
        ax.set_xlabel(data.get("group_col_name", ""), fontsize=11)
        ax.set_ylabel(data.get("value_col_name", "value"), fontsize=11)
    else:
        ax.set_ylabel(data.get("group_col_name", ""), fontsize=11)
        ax.set_xlabel(data.get("value_col_name", "value"), fontsize=11)

    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 3. HEATMAP
# ═══════════════════════════════════════════════════════════════════════════
# Use when: showing intensity of a numeric metric across two categorical axes,
# e.g. sales by region × month, errors by service × day-of-week.

_HEAT_MAX_ROWS = 20
_HEAT_MAX_COLS = 20

@viz_tool(
    name="heatmap",
    description=(
        "Draws a heatmap showing a numeric metric's intensity across two categorical axes. "
        "Requires: row_col (categorical), col_col (categorical), value_col (numeric). "
        "Aggregates value_col by mean per (row, col) cell. "
        "Best for: sales by region × month, error rate by service × hour."
    ),
    schema=ToolInputSpec(
        required_columns=["row_col", "col_col", "value_col"],
        kind_requirements={"value_col": "numeric"},
    ),
)
def create_heatmap(
    data: dict[str, Any],
    title: str,
    aggfunc: str = "mean",     # "mean" | "sum" | "count"
    cmap: str = "YlOrRd",
    annot: bool = True,
    fmt: str = ".1f",
    **kwargs,
) -> plt.Figure:

    rows = data["row_col"].astype(str)
    cols = data["col_col"].astype(str)
    vals = data["value_col"].astype(float)

    df = pd.DataFrame({"row": rows, "col": cols, "val": vals})

    # Pivot with aggregation
    agg_fn = {"mean": "mean", "sum": "sum", "count": "count"}.get(aggfunc, "mean")
    pivot = df.pivot_table(
        index="row", columns="col", values="val",
        aggfunc=agg_fn, fill_value=0,
    )

    # Trim if too many categories
    if pivot.shape[0] > _HEAT_MAX_ROWS:
        pivot = pivot.iloc[:_HEAT_MAX_ROWS]
    if pivot.shape[1] > _HEAT_MAX_COLS:
        pivot = pivot.iloc[:, :_HEAT_MAX_COLS]

    n_rows, n_cols = pivot.shape
    fig_w = max(6, n_cols * 0.8 + 2)
    fig_h = max(4, n_rows * 0.55 + 1.5)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Determine annotation: turn off if grid is too large to read
    do_annot = annot and (n_rows * n_cols) <= 200

    sns.heatmap(
        pivot,
        ax=ax,
        cmap=cmap,
        annot=do_annot,
        fmt=fmt,
        linewidths=0.4,
        linecolor="#eeeeee",
        cbar_kws={"shrink": 0.7, "aspect": 20, "pad": 0.02},
        annot_kws={"size": 8},
    )

    ax.set_xlabel(data.get("col_col_name", ""), fontsize=11)
    ax.set_ylabel(data.get("row_col_name", ""), fontsize=11)
    ax.tick_params(axis="x", rotation=30, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)

    # Colorbar label
    ax.collections[0].colorbar.set_label(
        f"{aggfunc}({data.get('value_col_name', 'value')})",
        fontsize=9,
    )

    apply_title(ax, title)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 4. HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════════
# Use when: understanding the distribution of a single numeric column.
# Shows bin counts with optional KDE (kernel density estimate) overlay
# and summary statistics annotation.

@viz_tool(
    name="histogram",
    description=(
        "Draws a histogram showing the distribution of a single numeric column. "
        "Requires: x_col (numeric). "
        "Adds a KDE curve and summary statistics (mean, median, std). "
        "Best for: salary distribution, age distribution, response times."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col"],
        kind_requirements={"x_col": "numeric"},
    ),
)
def create_histogram(
    data: dict[str, Any],
    title: str,
    bins: int | str = "auto",
    show_kde: bool = True,
    show_stats: bool = True,
    show_percentiles: bool = True,
    **kwargs,
) -> plt.Figure:

    x = data["x_col"].astype(float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        raise ValueError("No finite values to plot.")

    # Auto bin count: Sturges capped at 50
    if bins == "auto":
        bins = min(int(np.ceil(np.log2(len(x)) + 1)), 50)
        bins = max(bins, 5)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Main histogram (normalized to density so KDE aligns)
    counts, bin_edges, patches = ax.hist(
        x, bins=bins,
        density=show_kde,       # density=True when KDE will overlay
        color=PALETTE[0],
        edgecolor="white", linewidth=0.5,
        alpha=0.75,
        label="Frequency",
    )

    # KDE overlay
    if show_kde and len(x) > 4:
        kde = gaussian_kde(x, bw_method="scott")
        x_kde = np.linspace(x.min(), x.max(), 400)
        y_kde = kde(x_kde)
        ax.plot(x_kde, y_kde, color=PALETTE[3], linewidth=2,
                label="KDE", zorder=4)

    # Percentile lines
    if show_percentiles:
        for pct, ls, color in [
            (25, ":", PALETTE[2]),
            (50, "--", PALETTE[1]),
            (75, ":", PALETTE[2]),
        ]:
            val = float(np.percentile(x, pct))
            ax.axvline(val, linestyle=ls, color=color, linewidth=1.2,
                       label=f"p{pct}={val:,.1f}", alpha=0.85)

    # Stats annotation box
    if show_stats:
        mean, median, std = x.mean(), float(np.median(x)), x.std()
        skew = float(((x - mean) ** 3).mean() / std ** 3) if std > 0 else 0
        stats_text = (
            f"n = {len(x):,}\n"
            f"mean = {mean:,.2f}\n"
            f"median = {median:,.2f}\n"
            f"std = {std:,.2f}\n"
            f"skew = {skew:.2f}"
        )
        ax.text(
            0.97, 0.97, stats_text,
            transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc", alpha=0.9),
            family="monospace",
        )

    ax.set_xlabel(data.get("x_col_name", "value"), fontsize=11)
    ax.set_ylabel("Density" if show_kde else "Count", fontsize=11)
    ax.legend(fontsize=9, framealpha=0.8)
    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 5. LINE CHART
# ═══════════════════════════════════════════════════════════════════════════
# Use when: showing a trend of a numeric column over time or an ordered sequence.
# LLM should call this with x_col (datetime or ordered categorical) + y_col (numeric).
# Supports an optional group_col for multiple series on one chart.

@viz_tool(
    name="line_chart",
    description=(
        "Draws a line chart showing a numeric trend over time or an ordered axis. "
        "Requires: x_col (datetime or ordered categorical), y_col (numeric). "
        "Optional: group_col (categorical) to draw one line per group. "
        "Best for: revenue over months, temperature over days, stock prices."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y_col"],
        optional_columns=["group_col"],
        kind_requirements={"y_col": "numeric"},
    ),
)
def create_line_chart(
    data: dict[str, Any],
    title: str,
    show_markers: bool = True,
    show_area: bool = False,
    smooth: bool = False,
    **kwargs,
) -> plt.Figure:

    x_raw = data["x_col"]
    y = data["y_col"].astype(float)
    group = data.get("group_col")

    # Try to parse x as datetime for proper axis formatting
    x = _try_parse_datetime(x_raw)

    fig, ax = plt.subplots(figsize=(9, 5))

    is_temporal = _is_temporal(x)

    if group is not None:
        # Multi-series: one line per unique group value
        df = pd.DataFrame({"x": x, "y": y, "g": group})
        if is_temporal:
            df = df.sort_values("x")
        groups = df["g"].unique()

        for i, grp in enumerate(groups):
            subset = df[df["g"] == grp]
            color = PALETTE[i % len(PALETTE)]
            _draw_line(
                ax, subset["x"].values, subset["y"].values,
                label=str(grp), color=color,
                show_markers=show_markers,
                show_area=show_area,
            )
        ax.legend(
            title=str(group[0]) if hasattr(group, "__getitem__") else "",
            fontsize=9,
            framealpha=0.8,
        )
    else:
        # Single series
        df = pd.DataFrame({"x": x, "y": y})
        if is_temporal:
            df = df.sort_values("x")
        _draw_line(
            ax, df["x"].values, df["y"].values,
            label=None, color=PALETTE[0],
            show_markers=show_markers,
            show_area=show_area,
        )

    # X-axis formatting
    if hasattr(x[0], "year") if len(x) > 0 else False:
        fig.autofmt_xdate(rotation=30)
    else:
        n_labels = len(np.unique(x))
        rotate_labels(ax, n_labels)

    ax.set_ylabel(data.get("y_col_name", "value"), fontsize=11)
    ax.set_xlabel(data.get("x_col_name", ""), fontsize=11)
    apply_title(ax, title)
    style_spines(ax)
    ax.yaxis.set_tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 6. PIE CHART
# ═══════════════════════════════════════════════════════════════════════════
# Use when: showing proportions of a whole across a categorical column.
# Renders as a donut (hole in centre) which reads more cleanly than a
# solid pie at small slice counts. Falls back to a bar chart internally
# if there are more than 8 categories (pie charts become unreadable).

_PIE_OTHER_THRESHOLD = 0.02   # 2 %

@viz_tool(
    name="pie_chart",
    description=(
        "Draws a donut/pie chart showing proportional breakdown of a categorical column. "
        "Requires: category_col (categorical). "
        "Optional: value_col (numeric) to sum values per category; "
        "if omitted, counts occurrences. "
        "Best for: market share, budget allocation, survey response breakdown. "
        "NOTE: only use when there are ≤ 8 distinct categories."
    ),
    schema=ToolInputSpec(
        required_columns=["category_col"],
        optional_columns=["value_col"],
        kind_requirements={},
    ),
)
def create_pie_chart(
    data: dict[str, Any],
    title: str,
    donut: bool = True,
    show_pct: bool = True,
    show_legend: bool = True,
    explode_largest: bool = True,
    **kwargs,
) -> plt.Figure:

    categories = data["category_col"].astype(str)
    values_raw = data.get("value_col")

    # Build aggregated series
    if values_raw is not None:
        series = pd.Series(
            values_raw.astype(float), index=categories
        ).groupby(level=0, sort=False).sum()
    else:
        series = pd.Series(categories).value_counts()

    series = series[series > 0].sort_values(ascending=False)

    # Merge tiny slices into "Other"
    total = series.sum()
    small_mask = (series / total) < _PIE_OTHER_THRESHOLD
    if small_mask.any() and small_mask.sum() > 1:
        other_sum = series[small_mask].sum()
        series = series[~small_mask]
        series["Other"] = other_sum

    n = len(series)
    if n > 8:
        # Degrade to a horizontal bar chart with a warning annotation
        return _pie_fallback_bar(series, title)

    colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

    # Explode the largest slice slightly
    explode_arr = np.zeros(n)
    if explode_largest and n > 1:
        explode_arr[0] = 0.05

    wedge_props = {"edgecolor": "white", "linewidth": 1.5}
    text_props = {"fontsize": 9}
    autopct = ("%1.1f%%" if show_pct else None)

    fig, ax = plt.subplots(figsize=(7, 6))

    wedges, texts, autotexts = ax.pie(
        series.values,
        labels=None,                # legend handles labels
        colors=colors,
        autopct=autopct,
        explode=explode_arr,
        wedgeprops=wedge_props,
        textprops=text_props,
        startangle=90,
        counterclock=False,
        pctdistance=0.78,
    )

    # Style percentage labels
    for at in autotexts:
        at.set_fontsize(8.5)
        at.set_color("white")
        at.set_fontweight("semibold")

    # Donut hole
    if donut:
        centre_circle = plt.Circle((0, 0), 0.55, color="white", zorder=10)
        ax.add_patch(centre_circle)
        # Total in centre
        ax.text(
            0, 0, f"Total\n{total:,.0f}",
            ha="center", va="center",
            fontsize=10, fontweight="semibold",
            color="#333333",
        )

    if show_legend:
        ax.legend(
            wedges,
            [f"{lbl}  ({v / total:.1%})" for lbl, v in series.items()],
            loc="lower center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=min(n, 3),
            fontsize=9,
            framealpha=0.8,
        )

    apply_title(ax, title)
    fig.tight_layout()
    return fig


def _pie_fallback_bar(series: pd.Series, title: str) -> plt.Figure:
    """
    Renders a horizontal bar chart when there are too many categories
    for a readable pie/donut.
    """
    fig, ax = plt.subplots(figsize=(8, max(4, len(series) * 0.45)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(series))]
    ax.barh(series.index.astype(str), series.values, color=colors,
            edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Value", fontsize=11)
    total = series.sum()
    for i, (lbl, v) in enumerate(series.items()):
        ax.text(
            v + total * 0.005, i,
            f"{v / total:.1%}",
            va="center", ha="left", fontsize=9,
        )
    annotation = "(too many categories for pie — showing bar chart)"
    ax.text(
        0.98, 0.02, annotation,
        transform=ax.transAxes, fontsize=8,
        ha="right", va="bottom", color="#888888",
    )
    apply_title(ax, title)
    style_spines(ax)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 7. SCATTER PLOT
# ═══════════════════════════════════════════════════════════════════════════
# Use when: exploring correlation or relationship between two numeric columns.
# Supports optional color encoding (3rd categorical column) and a regression line.

@viz_tool(
    name="scatter_plot",
    description=(
        "Draws a scatter plot to reveal correlation between two numeric columns. "
        "Requires: x_col (numeric), y_col (numeric). "
        "Optional: color_col (categorical) to color-code points by group. "
        "Best for: ad_spend vs revenue, age vs salary, temperature vs sales."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y_col"],
        optional_columns=["color_col", "size_col"],
        kind_requirements={"x_col": "numeric", "y_col": "numeric"},
    ),
)
def create_scatter_plot(
    data: dict[str, Any],
    title: str,
    show_regression: bool = True,
    show_correlation: bool = True,
    alpha: float = 0.6,
    point_size: int = 60,
    **kwargs,
) -> plt.Figure:

    x = data["x_col"].astype(float)
    y = data["y_col"].astype(float)
    color_groups = data.get("color_col")
    size_data = data.get("size_col")

    # Remove inf / nan pairs
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if color_groups is not None:
        color_groups = color_groups[mask]
    if size_data is not None:
        size_data = size_data[mask]

    fig, ax = plt.subplots(figsize=(8, 6))

    if color_groups is not None:
        # One scatter call per group
        groups = pd.Categorical(color_groups).categories
        for i, grp in enumerate(groups):
            m = color_groups == grp
            sz = _normalize_sizes(size_data[m], point_size) if size_data is not None else point_size
            ax.scatter(
                x[m], y[m],
                label=str(grp),
                color=PALETTE[i % len(PALETTE)],
                s=sz, alpha=alpha, edgecolors="white", linewidths=0.4,
                zorder=3,
            )
        ax.legend(fontsize=9, framealpha=0.8)
    else:
        sz = _normalize_sizes(size_data, point_size) if size_data is not None else point_size
        ax.scatter(
            x, y,
            color=PALETTE[0],
            s=sz, alpha=alpha, edgecolors="white", linewidths=0.4,
            zorder=3,
        )

    # Optional regression line (on full x/y regardless of grouping)
    if show_regression and len(x) >= 3:
        slope, intercept, r_value, p_value, _ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        y_line = slope * x_line + intercept
        ax.plot(
            x_line, y_line,
            color="#C44E52", linewidth=1.5, linestyle="--",
            label=f"Linear fit", zorder=4,
        )
        if show_correlation:
            pval_str = f"p<0.001" if p_value < 0.001 else f"p={p_value:.3f}"
            ax.text(
                0.03, 0.96,
                f"r = {r_value:.3f}  ({pval_str})",
                transform=ax.transAxes,
                fontsize=9, verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc", alpha=0.85),
            )

    ax.set_xlabel(data.get("x_col_name", "x"), fontsize=11)
    ax.set_ylabel(data.get("y_col_name", "y"), fontsize=11)
    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# SHARED HELPERS (module-level, used by the tool functions above)
# ═══════════════════════════════════════════════════════════════════════════

def _try_parse_datetime(arr: np.ndarray) -> np.ndarray:
    """Attempt to convert string array to datetime; fall back to strings."""
    try:
        parsed = pd.to_datetime(arr, format="mixed", dayfirst=False)
        return parsed.values
    except Exception:
        return arr


def _is_temporal(arr: np.ndarray) -> bool:
    """Check if array is datetime type (should be sorted chronologically)."""
    return np.issubdtype(arr.dtype, np.datetime64)


def _draw_line(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str | None,
    color: str,
    show_markers: bool,
    show_area: bool,
) -> None:
    marker = "o" if show_markers else None
    markersize = 5 if show_markers else 0
    ax.plot(
        x, y,
        color=color, linewidth=2,
        marker=marker, markersize=markersize,
        label=label,
        zorder=3,
    )
    if show_area:
        ax.fill_between(x, y, alpha=0.12, color=color)


def _normalize_sizes(
    size_data: np.ndarray | None,
    base_size: int,
    min_size: int = 20,
    max_size: int = 300,
) -> np.ndarray | int:
    if size_data is None:
        return base_size
    arr = size_data.astype(float)
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max == arr_min:
        return base_size
    normalized = (arr - arr_min) / (arr_max - arr_min)
    return (normalized * (max_size - min_size) + min_size).astype(float)


# ═══════════════════════════════════════════════════════════════════════════
# 8. AREA CHART
# ═══════════════════════════════════════════════════════════════════════════
# Use when: showing volume or magnitude over time.  Like a line chart but
# the filled region emphasises the cumulative quantity.
# Supports stacking multiple series.

@viz_tool(
    name="area_chart",
    description=(
        "Draws an area chart (filled line) showing volume over time or an ordered axis. "
        "Requires: x_col (datetime or ordered categorical), y_col (numeric). "
        "Optional: group_col (categorical) to stack multiple series. "
        "Best for: cumulative revenue, traffic volume, resource usage over time."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y_col"],
        optional_columns=["group_col"],
        kind_requirements={"y_col": "numeric"},
    ),
)
def create_area_chart(
    data: dict[str, Any],
    title: str,
    stacked: bool = True,
    alpha: float = 0.6,
    show_lines: bool = True,
    **kwargs,
) -> plt.Figure:

    x_raw = data["x_col"]
    y = data["y_col"].astype(float)
    group = data.get("group_col")

    x = _try_parse_datetime(x_raw)

    fig, ax = plt.subplots(figsize=(9, 5))

    is_temporal = _is_temporal(x)

    if group is not None:
        df = pd.DataFrame({"x": x, "y": y, "g": group})
        if is_temporal:
            df = df.sort_values("x")
        unique_groups = df["g"].unique()

        if stacked:
            # Pivot for stackplot
            pivot = df.pivot_table(index="x", columns="g", values="y", aggfunc="sum", fill_value=0)
            if is_temporal:
                pivot = pivot.sort_index()
            colors = [PALETTE[i % len(PALETTE)] for i in range(len(pivot.columns))]
            ax.stackplot(
                pivot.index, *[pivot[col].values for col in pivot.columns],
                labels=[str(c) for c in pivot.columns],
                colors=colors, alpha=alpha,
            )
            if show_lines:
                cumulative = np.zeros(len(pivot))
                for i, col in enumerate(pivot.columns):
                    cumulative = cumulative + pivot[col].values
                    ax.plot(pivot.index, cumulative, color=colors[i], linewidth=1.2)
        else:
            for i, grp in enumerate(unique_groups):
                subset = df[df["g"] == grp]
                if is_temporal:
                    subset = subset.sort_values("x")
                color = PALETTE[i % len(PALETTE)]
                ax.fill_between(subset["x"].values, subset["y"].values,
                                alpha=alpha, color=color, label=str(grp))
                if show_lines:
                    ax.plot(subset["x"].values, subset["y"].values,
                            color=color, linewidth=1.5)

        ax.legend(fontsize=9, framealpha=0.8)
    else:
        # Single series
        df = pd.DataFrame({"x": x, "y": y})
        if is_temporal:
            df = df.sort_values("x")
        ax.fill_between(df["x"].values, df["y"].values,
                        alpha=alpha, color=PALETTE[0])
        if show_lines:
            ax.plot(df["x"].values, df["y"].values,
                    color=PALETTE[0], linewidth=2)

    # X-axis formatting
    if len(x) > 0 and hasattr(x[0], "year"):
        fig.autofmt_xdate(rotation=30)
    else:
        rotate_labels(ax, len(np.unique(x)))

    ax.set_ylabel(data.get("y_col_name", "value"), fontsize=11)
    ax.set_xlabel(data.get("x_col_name", ""), fontsize=11)
    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 9. STACKED BAR CHART
# ═══════════════════════════════════════════════════════════════════════════
# Use when: comparing composition of a total across categories.
# Each bar is split into segments showing the contribution of sub-groups.

@viz_tool(
    name="stacked_bar_chart",
    description=(
        "Draws a stacked bar chart showing how sub-groups contribute to the total "
        "across categories. "
        "Requires: x_col (categorical), y_col (numeric), group_col (categorical sub-groups). "
        "Best for: revenue by product per quarter, expenses by department per month."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y_col", "group_col"],
        kind_requirements={"y_col": "numeric"},
    ),
)
def create_stacked_bar_chart(
    data: dict[str, Any],
    title: str,
    orientation: str = "vertical",
    show_values: bool = False,
    show_totals: bool = True,
    **kwargs,
) -> plt.Figure:

    x = data["x_col"]
    y = data["y_col"].astype(float)
    group = data["group_col"]

    df = pd.DataFrame({"x": x, "y": y, "g": group})
    # Use Categorical to preserve original x order (not alphabetical)
    x_order = list(dict.fromkeys(x))  # unique values, input order
    df["x"] = pd.Categorical(df["x"], categories=x_order, ordered=True)
    pivot = df.pivot_table(index="x", columns="g", values="y", aggfunc="sum", fill_value=0, observed=True)
    pivot = pivot.loc[x_order]  # ensure original order

    categories = pivot.index.astype(str).tolist()
    groups = pivot.columns.tolist()
    n_cats = len(categories)
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(groups))]

    fig, ax = plt.subplots(figsize=(max(7, n_cats * 0.8), 5))

    if orientation == "vertical":
        x_pos = np.arange(n_cats)
        bottom = np.zeros(n_cats)
        for i, grp in enumerate(groups):
            vals = pivot[grp].values
            ax.bar(x_pos, vals, bottom=bottom, color=colors[i],
                   edgecolor="white", linewidth=0.5, width=0.65, label=str(grp))
            if show_values:
                for j, v in enumerate(vals):
                    if v > 0:
                        ax.text(x_pos[j], bottom[j] + v / 2, f"{v:,.0f}",
                                ha="center", va="center", fontsize=7.5, color="white")
            bottom += vals

        if show_totals:
            for j, tot in enumerate(bottom):
                ax.text(x_pos[j], tot + max(bottom) * 0.01, f"{tot:,.0f}",
                        ha="center", va="bottom", fontsize=8.5, fontweight="semibold")

        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)
        rotate_labels(ax, n_cats)
        ax.set_ylabel(data.get("y_col_name", "value"), fontsize=11)
    else:
        y_pos = np.arange(n_cats)
        left = np.zeros(n_cats)
        for i, grp in enumerate(groups):
            vals = pivot[grp].values
            ax.barh(y_pos, vals, left=left, color=colors[i],
                    edgecolor="white", linewidth=0.5, height=0.65, label=str(grp))
            if show_values:
                for j, v in enumerate(vals):
                    if v > 0:
                        ax.text(left[j] + v / 2, y_pos[j], f"{v:,.0f}",
                                ha="center", va="center", fontsize=7.5, color="white")
            left += vals

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel(data.get("y_col_name", "value"), fontsize=11)

    ax.legend(fontsize=9, framealpha=0.8, title=data.get("group_col_name", ""))
    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 10. GROUPED BAR CHART
# ═══════════════════════════════════════════════════════════════════════════
# Use when: comparing multiple metrics or groups side-by-side across categories.
# Unlike stacked, the bars sit next to each other for easy comparison.

@viz_tool(
    name="grouped_bar_chart",
    description=(
        "Draws a grouped (clustered) bar chart with bars side-by-side per category. "
        "Requires: x_col (categorical), y_col (numeric), group_col (categorical sub-groups). "
        "Best for: comparing male vs female salary by department, "
        "product A vs B sales by quarter."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y_col", "group_col"],
        kind_requirements={"y_col": "numeric"},
    ),
)
def create_grouped_bar_chart(
    data: dict[str, Any],
    title: str,
    show_values: bool = True,
    orientation: str = "vertical",
    **kwargs,
) -> plt.Figure:

    x = data["x_col"]
    y = data["y_col"].astype(float)
    group = data["group_col"]

    df = pd.DataFrame({"x": x, "y": y, "g": group})
    # Use Categorical to preserve original x order (not alphabetical)
    x_order = list(dict.fromkeys(x))  # unique values, input order
    df["x"] = pd.Categorical(df["x"], categories=x_order, ordered=True)
    pivot = df.pivot_table(index="x", columns="g", values="y", aggfunc="sum", fill_value=0, observed=True)
    pivot = pivot.loc[x_order]  # ensure original order

    categories = pivot.index.astype(str).tolist()
    groups = pivot.columns.tolist()
    n_cats = len(categories)
    n_groups = len(groups)
    colors = [PALETTE[i % len(PALETTE)] for i in range(n_groups)]

    bar_width = 0.8 / n_groups
    fig, ax = plt.subplots(figsize=(max(7, n_cats * (n_groups * 0.4 + 0.5)), 5))

    if orientation == "vertical":
        x_pos = np.arange(n_cats)
        for i, grp in enumerate(groups):
            offset = (i - n_groups / 2 + 0.5) * bar_width
            vals = pivot[grp].values
            bars = ax.bar(x_pos + offset, vals, bar_width * 0.9,
                          color=colors[i], edgecolor="white", linewidth=0.4,
                          label=str(grp))
            if show_values:
                for bar in bars:
                    h = bar.get_height()
                    if h > 0:
                        ax.text(bar.get_x() + bar.get_width() / 2, h,
                                f"{h:,.0f}", ha="center", va="bottom", fontsize=7.5)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)
        rotate_labels(ax, n_cats)
        ax.set_ylabel(data.get("y_col_name", "value"), fontsize=11)
    else:
        y_pos = np.arange(n_cats)
        for i, grp in enumerate(groups):
            offset = (i - n_groups / 2 + 0.5) * bar_width
            vals = pivot[grp].values
            ax.barh(y_pos + offset, vals, bar_width * 0.9,
                    color=colors[i], edgecolor="white", linewidth=0.4,
                    label=str(grp))

        ax.set_yticks(y_pos)
        ax.set_yticklabels(categories)
        ax.set_xlabel(data.get("y_col_name", "value"), fontsize=11)

    ax.legend(fontsize=9, framealpha=0.8, title=data.get("group_col_name", ""))
    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 11. CORRELATION MATRIX
# ═══════════════════════════════════════════════════════════════════════════
# Use when: exploring pairwise correlations between multiple numeric columns.
# This is an EDA staple — every data analyst uses it.

_CORR_MAX_COLS = 20

@viz_tool(
    name="correlation_matrix",
    description=(
        "Draws a correlation matrix heatmap for multiple numeric columns. "
        "Requires: columns (dict of column_name → numeric array). "
        "Best for: feature correlation analysis, multicollinearity check, EDA overview."
    ),
    schema=ToolInputSpec(
        required_columns=["columns"],
        kind_requirements={},
    ),
)
def create_correlation_matrix(
    data: dict[str, Any],
    title: str,
    method: str = "pearson",
    annot: bool = True,
    cmap: str = "RdBu_r",
    mask_upper: bool = True,
    **kwargs,
) -> plt.Figure:

    columns_data = data["columns"]

    # columns_data is {col_name: array}
    df = pd.DataFrame(columns_data)

    # Trim if too many columns
    if df.shape[1] > _CORR_MAX_COLS:
        df = df.iloc[:, :_CORR_MAX_COLS]

    # Compute correlation matrix
    corr = df.corr(method=method)
    n = len(corr)

    fig_size = max(6, n * 0.6 + 2)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Optional upper triangle mask
    mask = None
    if mask_upper:
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    do_annot = annot and (n * n) <= 400

    sns.heatmap(
        corr,
        ax=ax,
        mask=mask,
        cmap=cmap,
        annot=do_annot,
        fmt=".2f",
        vmin=-1, vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        linecolor="#eeeeee",
        cbar_kws={"shrink": 0.7, "aspect": 20, "pad": 0.02,
                  "label": f"{method} correlation"},
        annot_kws={"size": 8},
    )

    ax.tick_params(axis="x", rotation=45, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=9)
    apply_title(ax, title or f"Correlation Matrix ({method})")
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 12. COUNT PLOT
# ═══════════════════════════════════════════════════════════════════════════
# Use when: counting occurrences in a categorical column.
# The simplest and most frequently used chart in basic EDA.

@viz_tool(
    name="count_plot",
    description=(
        "Draws a count plot showing frequency of each category in a column. "
        "Requires: x_col (categorical). "
        "Optional: group_col (categorical) for grouped counts. "
        "Best for: most common categories, class imbalance check, survey responses."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col"],
        optional_columns=["group_col"],
        kind_requirements={},
    ),
)
def create_count_plot(
    data: dict[str, Any],
    title: str,
    show_values: bool = True,
    show_pct: bool = True,
    sort_by_count: bool = True,
    orientation: str = "vertical",
    **kwargs,
) -> plt.Figure:

    x = data["x_col"].astype(str)
    group = data.get("group_col")

    df = pd.DataFrame({"x": x})
    if group is not None:
        df["g"] = group

    # Sort categories by frequency
    if sort_by_count:
        order = df["x"].value_counts().index.tolist()
    else:
        order = sorted(df["x"].unique())

    n = len(order)
    fig, ax = plt.subplots(figsize=(max(7, n * 0.7), 5))

    if group is not None:
        unique_groups = df["g"].unique()
        palette = {str(g): PALETTE[i % len(PALETTE)] for i, g in enumerate(unique_groups)}
        sns.countplot(
            data=df,
            x="x" if orientation == "vertical" else None,
            y="x" if orientation == "horizontal" else None,
            hue="g",
            order=order,
            palette=palette,
            edgecolor="white", linewidth=0.5,
            ax=ax,
        )
        ax.legend(fontsize=9, framealpha=0.8, title=data.get("group_col_name", ""))
    else:
        if orientation == "vertical":
            counts = df["x"].value_counts().reindex(order)
            bars = ax.bar(np.arange(n), counts.values,
                          color=PALETTE[0], edgecolor="white", linewidth=0.5, width=0.65)
            ax.set_xticks(np.arange(n))
            ax.set_xticklabels(order)
            rotate_labels(ax, n)

            if show_values:
                total = counts.sum()
                for i, bar in enumerate(bars):
                    h = bar.get_height()
                    label = f"{h:,.0f}"
                    if show_pct:
                        label += f"\n({h / total:.1%})"
                    ax.text(bar.get_x() + bar.get_width() / 2, h,
                            label, ha="center", va="bottom", fontsize=8)
        else:
            counts = df["x"].value_counts().reindex(order[::-1])
            bars = ax.barh(np.arange(n), counts.values,
                           color=PALETTE[0], edgecolor="white", linewidth=0.5, height=0.65)
            ax.set_yticks(np.arange(n))
            ax.set_yticklabels(counts.index.tolist())

            if show_values:
                total = counts.sum()
                for i, bar in enumerate(bars):
                    w = bar.get_width()
                    label = f"{w:,.0f}"
                    if show_pct:
                        label += f" ({w / total:.1%})"
                    ax.text(w + max(counts.values) * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            label, va="center", ha="left", fontsize=8)

    ax.set_xlabel(data.get("x_col_name", "") if orientation == "vertical" else "Count", fontsize=11)
    ax.set_ylabel("Count" if orientation == "vertical" else data.get("x_col_name", ""), fontsize=11)
    apply_title(ax, title)
    style_spines(ax)
    ax.tick_params(labelsize=10)
    fig.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════
# 13. DUAL-AXIS TIME SERIES
# ═══════════════════════════════════════════════════════════════════════════
# Use when: plotting two metrics with different scales on the same time axis.
# e.g. revenue (₹ lakhs) vs customer count on the same chart.

@viz_tool(
    name="dual_axis_time_series",
    description=(
        "Draws a dual-axis time series chart with two y-axes sharing the same x-axis. "
        "Requires: x_col (datetime or ordered), y1_col (numeric), y2_col (numeric). "
        "Best for: revenue vs customers over time, temperature vs humidity, "
        "stock price vs volume."
    ),
    schema=ToolInputSpec(
        required_columns=["x_col", "y1_col", "y2_col"],
        kind_requirements={"y1_col": "numeric", "y2_col": "numeric"},
    ),
)
def create_dual_axis_time_series(
    data: dict[str, Any],
    title: str,
    y1_style: str = "line",        # "line" | "bar"
    y2_style: str = "line",        # "line" | "bar"
    show_markers: bool = True,
    **kwargs,
) -> plt.Figure:

    x_raw = data["x_col"]
    y1 = data["y1_col"].astype(float)
    y2 = data["y2_col"].astype(float)

    x = _try_parse_datetime(x_raw)

    # Sort only if x is true datetime (preserve original order for strings)
    if _is_temporal(x):
        sort_idx = np.argsort(x)
        x, y1, y2 = x[sort_idx], y1[sort_idx], y2[sort_idx]

    y1_name = data.get("y1_col_name", "Metric 1")
    y2_name = data.get("y2_col_name", "Metric 2")

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    color1 = PALETTE[0]
    color2 = PALETTE[3]
    marker = "o" if show_markers else None
    ms = 5 if show_markers else 0

    # Left y-axis
    if y1_style == "bar":
        ax1.bar(np.arange(len(x)), y1, color=color1, alpha=0.7,
                edgecolor="white", linewidth=0.5, width=0.6, label=y1_name)
        ax1.set_xticks(np.arange(len(x)))
        ax1.set_xticklabels([str(v) for v in x])
    else:
        ax1.plot(x, y1, color=color1, linewidth=2,
                 marker=marker, markersize=ms, label=y1_name, zorder=3)

    ax1.set_ylabel(y1_name, fontsize=11, color=color1)
    ax1.tick_params(axis="y", labelcolor=color1, labelsize=10)

    # Right y-axis
    if y2_style == "bar":
        ax2.bar(np.arange(len(x)), y2, color=color2, alpha=0.5,
                edgecolor="white", linewidth=0.5, width=0.4, label=y2_name)
    else:
        ax2.plot(x, y2, color=color2, linewidth=2, linestyle="--",
                 marker=marker, markersize=ms, label=y2_name, zorder=3)

    ax2.set_ylabel(y2_name, fontsize=11, color=color2)
    ax2.tick_params(axis="y", labelcolor=color2, labelsize=10)

    # X-axis formatting
    if len(x) > 0 and hasattr(x[0], "year"):
        fig.autofmt_xdate(rotation=30)
    else:
        rotate_labels(ax1, len(np.unique(x)))

    ax1.set_xlabel(data.get("x_col_name", ""), fontsize=11)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2,
               fontsize=9, framealpha=0.8, loc="upper left")

    # Style only left axis spines (right axis has its own)
    ax1.spines["top"].set_visible(False)
    ax1.spines["left"].set_linewidth(0.6)
    ax1.spines["bottom"].set_linewidth(0.6)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_linewidth(0.6)

    apply_title(ax1, title)
    fig.tight_layout()
    return fig
