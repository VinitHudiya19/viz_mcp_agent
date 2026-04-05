"""
test_tools.py — Quick smoke test for all 13 visualization tools.
Run from project root:  python test_tools.py

Saves all charts as PNGs in ./test_output/
"""

import os
import sys
import base64
import json
import numpy as np

# ── ensure project root is on path ──────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

from core.base_tool import TOOL_REGISTRY, ToolResult
import tools.visualizations  # triggers @viz_tool registration

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def save_result(name: str, result: ToolResult) -> None:
    if result.success:
        path = os.path.join(OUTPUT_DIR, f"{name}.png")
        with open(path, "wb") as f:
            f.write(base64.b64decode(result.image_b64))
        print(f"  [OK] {name}.png saved")
    else:
        print(f"  [FAIL] {name}: {result.error}")


def main():
    print(f"\nRegistered tools: {list(TOOL_REGISTRY.keys())}")
    print(f"Total: {len(TOOL_REGISTRY)} tools\n")
    print(f"Saving charts to: {OUTPUT_DIR}\n")

    # ── sample data ──────────────────────────────────────────────────────

    months = np.array(["Jan", "Feb", "Mar", "Apr", "May", "Jun"])
    revenue = np.array([12000, 15000, 13500, 17000, 16000, 19500])
    customers = np.array([120, 145, 130, 170, 155, 190])
    departments = np.array(["Sales", "Sales", "Eng", "Eng", "HR", "HR",
                            "Sales", "Eng", "HR", "Sales", "Eng", "HR"])
    salaries = np.array([55000, 62000, 78000, 85000, 48000, 52000,
                         58000, 82000, 50000, 60000, 80000, 51000])
    categories = np.array(["A", "B", "A", "C", "B", "A", "C", "B", "A", "C"])

    # ── 1. Bar Chart ─────────────────────────────────────────────────────
    print("1. Bar Chart")
    r = TOOL_REGISTRY["bar_chart"](
        {"x_col": months, "y_col": revenue, "y_col_name": "Revenue (₹)"},
        "Monthly Revenue"
    )
    save_result("01_bar_chart", r)

    # ── 2. Line Chart ────────────────────────────────────────────────────
    print("2. Line Chart")
    r = TOOL_REGISTRY["line_chart"](
        {"x_col": months, "y_col": revenue, "y_col_name": "Revenue"},
        "Revenue Trend"
    )
    save_result("02_line_chart", r)

    # ── 3. Scatter Plot ──────────────────────────────────────────────────
    print("3. Scatter Plot")
    np.random.seed(42)
    x_scatter = np.random.rand(50) * 100
    y_scatter = x_scatter * 2.5 + np.random.randn(50) * 20 + 10
    r = TOOL_REGISTRY["scatter_plot"](
        {"x_col": x_scatter, "y_col": y_scatter,
         "x_col_name": "Ad Spend", "y_col_name": "Revenue"},
        "Ad Spend vs Revenue"
    )
    save_result("03_scatter_plot", r)

    # ── 4. Histogram ─────────────────────────────────────────────────────
    print("4. Histogram")
    ages = np.random.normal(35, 10, 500).clip(18, 65)
    r = TOOL_REGISTRY["histogram"](
        {"x_col": ages, "x_col_name": "Age"},
        "Employee Age Distribution"
    )
    save_result("04_histogram", r)

    # ── 5. Box Plot ──────────────────────────────────────────────────────
    print("5. Box Plot")
    r = TOOL_REGISTRY["box_plot"](
        {"group_col": departments, "value_col": salaries,
         "group_col_name": "Department", "value_col_name": "Salary (₹)"},
        "Salary by Department"
    )
    save_result("05_box_plot", r)

    # ── 6. Heatmap ───────────────────────────────────────────────────────
    print("6. Heatmap")
    regions = np.array(["North", "South", "East", "West"] * 3)
    qtrs = np.array(["Q1"] * 4 + ["Q2"] * 4 + ["Q3"] * 4)
    sales_vals = np.random.randint(100, 500, 12).astype(float)
    r = TOOL_REGISTRY["heatmap"](
        {"row_col": regions, "col_col": qtrs, "value_col": sales_vals,
         "row_col_name": "Region", "col_col_name": "Quarter",
         "value_col_name": "Sales"},
        "Sales by Region × Quarter"
    )
    save_result("06_heatmap", r)

    # ── 7. Pie Chart ─────────────────────────────────────────────────────
    print("7. Pie Chart")
    r = TOOL_REGISTRY["pie_chart"](
        {"category_col": np.array(["Chrome", "Safari", "Firefox", "Edge", "Other"]),
         "value_col": np.array([65, 18, 8, 5, 4], dtype=float)},
        "Browser Market Share"
    )
    save_result("07_pie_chart", r)

    # ── 8. Area Chart ────────────────────────────────────────────────────
    print("8. Area Chart")
    r = TOOL_REGISTRY["area_chart"](
        {"x_col": months, "y_col": revenue,
         "x_col_name": "Month", "y_col_name": "Revenue"},
        "Revenue Volume"
    )
    save_result("08_area_chart", r)

    # ── 9. Stacked Bar Chart ─────────────────────────────────────────────
    print("9. Stacked Bar Chart")
    sb_x = np.array(["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"])
    sb_y = np.array([30000, 20000, 35000, 25000, 40000, 22000], dtype=float)
    sb_g = np.array(["Product A", "Product B"] * 3)
    r = TOOL_REGISTRY["stacked_bar_chart"](
        {"x_col": sb_x, "y_col": sb_y, "group_col": sb_g,
         "y_col_name": "Revenue", "group_col_name": "Product"},
        "Revenue by Product per Quarter"
    )
    save_result("09_stacked_bar", r)

    # ── 10. Grouped Bar Chart ────────────────────────────────────────────
    print("10. Grouped Bar Chart")
    r = TOOL_REGISTRY["grouped_bar_chart"](
        {"x_col": sb_x, "y_col": sb_y, "group_col": sb_g,
         "y_col_name": "Revenue", "group_col_name": "Product"},
        "Product Comparison by Quarter"
    )
    save_result("10_grouped_bar", r)

    # ── 11. Correlation Matrix ───────────────────────────────────────────
    print("11. Correlation Matrix")
    np.random.seed(7)
    n = 100
    col_data = {
        "Age": np.random.normal(35, 10, n),
        "Salary": np.random.normal(60000, 15000, n),
        "Experience": np.random.normal(10, 5, n),
        "Satisfaction": np.random.uniform(1, 5, n),
    }
    # Add some real correlation
    col_data["Salary"] = col_data["Age"] * 1200 + np.random.randn(n) * 5000
    col_data["Experience"] = col_data["Age"] - 22 + np.random.randn(n) * 3

    r = TOOL_REGISTRY["correlation_matrix"](
        {"columns": col_data},
        "Employee Metrics Correlation"
    )
    save_result("11_correlation_matrix", r)

    # ── 12. Count Plot ───────────────────────────────────────────────────
    print("12. Count Plot")
    survey = np.random.choice(["Excellent", "Good", "Average", "Poor"], 200,
                              p=[0.3, 0.4, 0.2, 0.1])
    r = TOOL_REGISTRY["count_plot"](
        {"x_col": survey, "x_col_name": "Rating"},
        "Customer Satisfaction Survey"
    )
    save_result("12_count_plot", r)

    # ── 13. Dual Axis Time Series ────────────────────────────────────────
    print("13. Dual Axis Time Series")
    r = TOOL_REGISTRY["dual_axis_time_series"](
        {"x_col": months,
         "y1_col": revenue,
         "y2_col": customers,
         "x_col_name": "Month",
         "y1_col_name": "Revenue (₹)",
         "y2_col_name": "Customers"},
        "Revenue vs Customers"
    )
    save_result("13_dual_axis", r)

    print(f"\n{'='*50}")
    print(f"Done! Open {OUTPUT_DIR} to view all charts.")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
