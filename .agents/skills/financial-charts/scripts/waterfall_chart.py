#!/usr/bin/env python3
"""
Financial Waterfall Chart Generator

Creates waterfall charts for visualizing:
- Revenue bridges (period over period changes)
- Profit walkdowns (revenue → net income)
- Variance analysis
"""

import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    from .themes import get_theme
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from themes import get_theme


def create_waterfall_chart(
    categories: list[str],
    values: list[float],
    measure: Optional[list[str]] = None,
    title: str = "Waterfall Chart",
    output_path: str = "waterfall_chart.png",
    theme: str = "default",
    width: int = 1000,
    height: int = 500,
    value_format: str = "${:,.0f}",
    show_connector: bool = True,
    orientation: str = "v",
) -> str:
    """
    Create a waterfall chart.

    Args:
        categories: Category labels
        values: Values (positive = increase, negative = decrease)
        measure: List of "relative", "total", or "absolute" for each bar
                 Default: first and last are "total", rest are "relative"
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        width: Chart width in pixels
        height: Chart height in pixels
        value_format: Format string for values
        show_connector: Whether to show connector lines
        orientation: "v" for vertical, "h" for horizontal

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)

    # Default measure: totals at start and end
    if measure is None:
        measure = ["total"] + ["relative"] * (len(categories) - 2) + ["total"]

    # Calculate text values
    text_values = [value_format.format(abs(v)) for v in values]

    fig = go.Figure(go.Waterfall(
        name="",
        orientation=orientation,
        measure=measure,
        x=categories if orientation == "v" else values,
        y=values if orientation == "v" else categories,
        textposition="outside",
        text=text_values,
        connector={"line": {"color": theme_colors["grid"]}} if show_connector else {"visible": False},
        increasing={"marker": {"color": theme_colors["profit"]}},
        decreasing={"marker": {"color": theme_colors["cost"]}},
        totals={"marker": {"color": theme_colors["revenue"]}},
    ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=theme_colors["text"]),
            x=0.5,
        ),
        font=dict(family="Arial", color=theme_colors["text"]),
        paper_bgcolor=theme_colors["background"],
        plot_bgcolor=theme_colors["background"],
        width=width,
        height=height,
        showlegend=False,
        margin=dict(l=60, r=40, t=80, b=60),
    )

    # Update axes
    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linecolor=theme_colors["grid"],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme_colors["grid"],
        showline=True,
        linecolor=theme_colors["grid"],
        tickformat="$,.0f",
    )

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_profit_walkdown(
    revenue: float,
    cost_of_goods_sold: float,
    operating_expenses: dict[str, float],
    other_items: Optional[dict[str, float]] = None,
    title: str = "Profit Walkdown",
    output_path: str = "profit_walkdown.png",
    theme: str = "default",
) -> str:
    """
    Create a profit walkdown waterfall (Revenue → Net Income).

    Args:
        revenue: Total revenue
        cost_of_goods_sold: COGS (will be shown as negative)
        operating_expenses: Dict of OpEx items {"R&D": 1000, "SG&A": 500}
        other_items: Dict of other items {"Interest": -100, "Tax": -200}
        title: Chart title
        output_path: Output file path
        theme: Color theme name

    Returns:
        Path to saved chart
    """
    categories = ["Revenue"]
    values = [revenue]
    measure = ["total"]

    # COGS
    categories.append("COGS")
    values.append(-cost_of_goods_sold)
    measure.append("relative")

    # Gross Profit subtotal
    gross_profit = revenue - cost_of_goods_sold
    categories.append("Gross Profit")
    values.append(gross_profit)
    measure.append("total")

    # Operating expenses
    for name, value in operating_expenses.items():
        categories.append(name)
        values.append(-value)
        measure.append("relative")

    # Operating Income subtotal
    operating_income = gross_profit - sum(operating_expenses.values())
    categories.append("Operating Income")
    values.append(operating_income)
    measure.append("total")

    # Other items
    if other_items:
        for name, value in other_items.items():
            categories.append(name)
            values.append(value)  # Can be positive or negative
            measure.append("relative")

        net_income = operating_income + sum(other_items.values())
    else:
        net_income = operating_income

    # Net Income
    categories.append("Net Income")
    values.append(net_income)
    measure.append("total")

    return create_waterfall_chart(
        categories=categories,
        values=values,
        measure=measure,
        title=title,
        output_path=output_path,
        theme=theme,
    )


def create_revenue_bridge(
    start_value: float,
    start_label: str,
    changes: dict[str, float],
    end_label: str,
    title: str = "Revenue Bridge",
    output_path: str = "revenue_bridge.png",
    theme: str = "default",
) -> str:
    """
    Create a revenue bridge waterfall (period-over-period analysis).

    Args:
        start_value: Starting value
        start_label: Label for starting value (e.g., "Q1 Revenue")
        changes: Dict of changes {"Price Increase": 500, "Volume Loss": -200}
        end_label: Label for ending value (e.g., "Q2 Revenue")
        title: Chart title
        output_path: Output file path
        theme: Color theme name

    Returns:
        Path to saved chart
    """
    categories = [start_label]
    values = [start_value]
    measure = ["total"]

    for name, value in changes.items():
        categories.append(name)
        values.append(value)
        measure.append("relative")

    end_value = start_value + sum(changes.values())
    categories.append(end_label)
    values.append(end_value)
    measure.append("total")

    return create_waterfall_chart(
        categories=categories,
        values=values,
        measure=measure,
        title=title,
        output_path=output_path,
        theme=theme,
    )


# Example usage
if __name__ == "__main__":
    # Profit walkdown example
    create_profit_walkdown(
        revenue=100e6,
        cost_of_goods_sold=60e6,
        operating_expenses={
            "R&D": 10e6,
            "SG&A": 15e6,
        },
        other_items={
            "Interest": -2e6,
            "Tax": -3e6,
        },
        title="Q4 2024 Profit Walkdown",
        output_path="profit_walkdown_example.png",
    )

    # Revenue bridge example
    create_revenue_bridge(
        start_value=80e6,
        start_label="Q3 Revenue",
        changes={
            "Price Increase": 5e6,
            "New Customers": 12e6,
            "Churn": -7e6,
            "Upsells": 3e6,
            "Currency": -2e6,
        },
        end_label="Q4 Revenue",
        title="Quarterly Revenue Bridge",
        output_path="revenue_bridge_example.png",
    )
