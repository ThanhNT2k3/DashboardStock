#!/usr/bin/env python3
"""
Financial Sankey/Flow Chart Generator

Creates Sankey diagrams for visualizing financial flows like:
- Revenue breakdown (products/services → total revenue)
- Income statement flows (revenue → costs → profit)
- Cash flow visualizations
"""

import plotly.graph_objects as go
from typing import Optional
import json
import sys
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    from .themes import get_theme
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from themes import get_theme


def create_sankey_chart(
    labels: list[str],
    sources: list[int],
    targets: list[int],
    values: list[float],
    colors: Optional[list[str]] = None,
    link_colors: Optional[list[str]] = None,
    title: str = "Financial Flow",
    output_path: str = "sankey_chart.png",
    width: int = 1200,
    height: int = 600,
    font_size: int = 12,
    show_values: bool = True,
) -> str:
    """
    Create a Sankey diagram for financial flows.

    Args:
        labels: Node names (e.g., ["iPhone", "Services", "Revenue", "Profit"])
        sources: Index of source nodes for each link
        targets: Index of target nodes for each link
        values: Flow values for each link
        colors: Node colors (hex or rgba)
        link_colors: Link colors (typically semi-transparent)
        title: Chart title
        output_path: Output file path (.png, .html, .pdf)
        width: Chart width in pixels
        height: Chart height in pixels
        font_size: Label font size
        show_values: Whether to show values in node labels

    Returns:
        Path to saved chart
    """
    # Default color scheme (professional blue-green)
    if colors is None:
        colors = ["#2E86AB"] * len(labels)

    if link_colors is None:
        link_colors = ["rgba(46, 134, 171, 0.4)"] * len(sources)

    # Format labels with values if requested
    if show_values:
        node_values = [0.0] * len(labels)
        for i, (src, val) in enumerate(zip(sources, values)):
            node_values[src] += val
        for i, (tgt, val) in enumerate(zip(targets, values)):
            if node_values[tgt] == 0:
                node_values[tgt] = val

        formatted_labels = [
            f"{label}<br>${val/1e9:.1f}B" if val > 0 else label
            for label, val in zip(labels, node_values)
        ]
    else:
        formatted_labels = labels

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=20,
            thickness=25,
            line=dict(color="white", width=0.5),
            label=formatted_labels,
            color=colors,
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=link_colors,
        )
    )])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color="#333"),
            x=0.5,
        ),
        font=dict(size=font_size, family="Arial"),
        paper_bgcolor="white",
        width=width,
        height=height,
        margin=dict(l=50, r=50, t=80, b=50),
    )

    # Save based on file extension
    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_income_statement_sankey(
    revenue_sources: dict[str, float],
    cost_of_revenue: float,
    operating_expenses: dict[str, float],
    other_expenses: dict[str, float],
    company_name: str = "Company",
    fiscal_period: str = "FY24",
    theme: str = "default",
    output_path: str = "income_statement_sankey.png",
) -> str:
    """
    Create an income statement Sankey diagram (like the Apple example).

    Args:
        revenue_sources: Dict of revenue streams {"iPhone": 205.5e9, "Services": 78.1e9}
        cost_of_revenue: Total cost of revenue/goods sold
        operating_expenses: Dict of OpEx {"R&D": 26.2e9, "SG&A": 25.1e9}
        other_expenses: Dict of other items {"Tax": 19.3e9, "Other": 0.3e9}
        company_name: Company name for title
        fiscal_period: Fiscal period label
        theme: Color theme name
        output_path: Output file path

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)

    # Calculate totals
    total_revenue = sum(revenue_sources.values())
    gross_profit = total_revenue - cost_of_revenue
    total_opex = sum(operating_expenses.values())
    operating_profit = gross_profit - total_opex
    total_other = sum(other_expenses.values())
    net_profit = operating_profit - total_other

    # Build nodes and links
    labels = []
    sources = []
    targets = []
    values = []
    node_colors = []
    link_colors = []

    # Revenue sources
    idx = 0
    revenue_idx = len(revenue_sources)
    for name, value in revenue_sources.items():
        labels.append(name)
        sources.append(idx)
        targets.append(revenue_idx)
        values.append(value)
        node_colors.append(theme_colors["revenue"])
        link_colors.append(theme_colors["revenue_link"])
        idx += 1

    # Revenue node
    labels.append("Revenue")
    node_colors.append(theme_colors["revenue"])
    revenue_idx = idx
    idx += 1

    # Gross profit and cost of revenue
    labels.append("Gross profit")
    gross_profit_idx = idx
    node_colors.append(theme_colors["profit"])
    idx += 1

    labels.append("Cost of revenue")
    cost_idx = idx
    node_colors.append(theme_colors["cost"])
    idx += 1

    # Links from revenue
    sources.extend([revenue_idx, revenue_idx])
    targets.extend([gross_profit_idx, cost_idx])
    values.extend([gross_profit, cost_of_revenue])
    link_colors.extend([theme_colors["profit_link"], theme_colors["cost_link"]])

    # Operating expenses
    labels.append("Operating expenses")
    opex_idx = idx
    node_colors.append(theme_colors["cost"])
    idx += 1

    labels.append("Operating profit")
    op_profit_idx = idx
    node_colors.append(theme_colors["profit"])
    idx += 1

    # Links from gross profit
    sources.extend([gross_profit_idx, gross_profit_idx])
    targets.extend([op_profit_idx, opex_idx])
    values.extend([operating_profit, total_opex])
    link_colors.extend([theme_colors["profit_link"], theme_colors["cost_link"]])

    # OpEx breakdown
    for name, value in operating_expenses.items():
        labels.append(name)
        sources.append(opex_idx)
        targets.append(idx)
        values.append(value)
        node_colors.append(theme_colors["cost"])
        link_colors.append(theme_colors["cost_link"])
        idx += 1

    # Net profit and other expenses
    labels.append("Net profit")
    net_profit_idx = idx
    node_colors.append(theme_colors["profit"])
    idx += 1

    sources.append(op_profit_idx)
    targets.append(net_profit_idx)
    values.append(net_profit)
    link_colors.append(theme_colors["profit_link"])

    for name, value in other_expenses.items():
        labels.append(name)
        sources.append(op_profit_idx)
        targets.append(idx)
        values.append(value)
        node_colors.append(theme_colors["cost"])
        link_colors.append(theme_colors["cost_link"])
        idx += 1

    return create_sankey_chart(
        labels=labels,
        sources=sources,
        targets=targets,
        values=values,
        colors=node_colors,
        link_colors=link_colors,
        title=f"{company_name} Income Statement Flow - {fiscal_period}",
        output_path=output_path,
    )


# Example usage
if __name__ == "__main__":
    # Apple FY22-style example
    chart_path = create_income_statement_sankey(
        revenue_sources={
            "iPhone": 205.5e9,
            "Mac": 40.2e9,
            "iPad": 29.3e9,
            "Wearables": 41.2e9,
            "Services": 78.1e9,
        },
        cost_of_revenue=223.6e9,
        operating_expenses={
            "R&D": 26.2e9,
            "SG&A": 25.1e9,
        },
        other_expenses={
            "Tax": 19.3e9,
            "Other": 0.3e9,
        },
        company_name="Apple",
        fiscal_period="FY22",
        output_path="apple_fy22_sankey.png",
    )
    print(f"Chart saved to: {chart_path}")
