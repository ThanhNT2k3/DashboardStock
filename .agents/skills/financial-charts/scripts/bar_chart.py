#!/usr/bin/env python3
"""
Financial Bar Chart Generator

Creates bar charts for visualizing:
- Revenue/expense comparisons
- Period-over-period comparisons
- Segment breakdowns
- Margin comparisons
"""

import plotly.graph_objects as go
import plotly.express as px
from typing import Optional
import sys
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    from .themes import get_theme, get_palette
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from themes import get_theme, get_palette


def create_bar_chart(
    categories: list[str],
    values: list[float],
    title: str = "Bar Chart",
    output_path: str = "bar_chart.png",
    theme: str = "default",
    width: int = 800,
    height: int = 500,
    value_format: str = "${:,.0f}",
    show_values: bool = True,
    orientation: str = "v",
    color: Optional[str] = None,
) -> str:
    """
    Create a simple bar chart.

    Args:
        categories: Category labels
        values: Values for each category
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        width: Chart width in pixels
        height: Chart height in pixels
        value_format: Format string for values
        show_values: Whether to show value labels
        orientation: "v" for vertical, "h" for horizontal
        color: Override bar color

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)
    bar_color = color or theme_colors["revenue"]

    fig = go.Figure(data=[
        go.Bar(
            x=categories if orientation == "v" else values,
            y=values if orientation == "v" else categories,
            orientation=orientation,
            marker_color=bar_color,
            text=[value_format.format(v) for v in values] if show_values else None,
            textposition="outside" if orientation == "v" else "auto",
        )
    ])

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
        margin=dict(l=60, r=40, t=80, b=60),
    )

    fig.update_xaxes(showgrid=False, showline=True, linecolor=theme_colors["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=theme_colors["grid"], showline=True, linecolor=theme_colors["grid"])

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_grouped_bar_chart(
    categories: list[str],
    series: dict[str, list[float]],
    title: str = "Grouped Bar Chart",
    output_path: str = "grouped_bar_chart.png",
    theme: str = "default",
    palette: str = "default",
    width: int = 900,
    height: int = 500,
    value_format: str = "${:,.0f}",
    show_values: bool = False,
    barmode: str = "group",
) -> str:
    """
    Create a grouped or stacked bar chart.

    Args:
        categories: Category labels (x-axis)
        series: Dict of series name → values {"Q1": [100, 200], "Q2": [120, 180]}
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        palette: Color palette name
        width: Chart width in pixels
        height: Chart height in pixels
        value_format: Format string for values
        show_values: Whether to show value labels
        barmode: "group" for side-by-side, "stack" for stacked

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)
    colors = get_palette(palette)

    fig = go.Figure()

    for i, (name, values) in enumerate(series.items()):
        fig.add_trace(go.Bar(
            name=name,
            x=categories,
            y=values,
            marker_color=colors[i % len(colors)],
            text=[value_format.format(v) for v in values] if show_values else None,
            textposition="outside",
        ))

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=theme_colors["text"]),
            x=0.5,
        ),
        barmode=barmode,
        font=dict(family="Arial", color=theme_colors["text"]),
        paper_bgcolor=theme_colors["background"],
        plot_bgcolor=theme_colors["background"],
        width=width,
        height=height,
        margin=dict(l=60, r=40, t=80, b=60),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=False, showline=True, linecolor=theme_colors["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=theme_colors["grid"], showline=True, linecolor=theme_colors["grid"])

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_margin_comparison_chart(
    companies: list[str],
    gross_margins: list[float],
    operating_margins: list[float],
    net_margins: Optional[list[float]] = None,
    title: str = "Margin Comparison",
    output_path: str = "margin_comparison.png",
    theme: str = "default",
) -> str:
    """
    Create a margin comparison chart for competitive analysis.

    Args:
        companies: Company names
        gross_margins: Gross margin percentages
        operating_margins: Operating margin percentages
        net_margins: Net margin percentages (optional)
        title: Chart title
        output_path: Output file path
        theme: Color theme name

    Returns:
        Path to saved chart
    """
    series = {
        "Gross Margin": gross_margins,
        "Operating Margin": operating_margins,
    }
    if net_margins:
        series["Net Margin"] = net_margins

    return create_grouped_bar_chart(
        categories=companies,
        series=series,
        title=title,
        output_path=output_path,
        theme=theme,
        value_format="{:.1f}%",
        show_values=True,
    )


def create_revenue_segment_chart(
    segments: list[str],
    values: list[float],
    title: str = "Revenue by Segment",
    output_path: str = "revenue_segments.png",
    theme: str = "default",
    show_percentage: bool = True,
) -> str:
    """
    Create a revenue segment breakdown chart.

    Args:
        segments: Segment names
        values: Revenue values per segment
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        show_percentage: Whether to show percentage labels

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)
    palette = get_palette("default")
    total = sum(values)

    if show_percentage:
        labels = [f"${v/1e9:.1f}B ({v/total*100:.1f}%)" for v in values]
    else:
        labels = [f"${v/1e9:.1f}B" for v in values]

    # Sort by value descending
    sorted_data = sorted(zip(segments, values, labels, palette), key=lambda x: x[1], reverse=True)
    segments, values, labels, colors = zip(*sorted_data)

    fig = go.Figure(data=[
        go.Bar(
            x=list(values),
            y=list(segments),
            orientation="h",
            marker_color=list(colors),
            text=list(labels),
            textposition="auto",
        )
    ])

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=theme_colors["text"]),
            x=0.5,
        ),
        font=dict(family="Arial", color=theme_colors["text"]),
        paper_bgcolor=theme_colors["background"],
        plot_bgcolor=theme_colors["background"],
        width=800,
        height=400,
        margin=dict(l=120, r=40, t=80, b=40),
    )

    fig.update_xaxes(showgrid=True, gridcolor=theme_colors["grid"], tickformat="$,.0f")
    fig.update_yaxes(showgrid=False)

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


# Example usage
if __name__ == "__main__":
    # Simple bar chart
    create_bar_chart(
        categories=["Q1", "Q2", "Q3", "Q4"],
        values=[25e6, 28e6, 32e6, 35e6],
        title="Quarterly Revenue",
        output_path="quarterly_revenue.png",
    )

    # Margin comparison
    create_margin_comparison_chart(
        companies=["Apple", "Microsoft", "Google", "Amazon"],
        gross_margins=[43.3, 68.4, 55.3, 47.0],
        operating_margins=[30.3, 41.2, 27.4, 5.3],
        net_margins=[25.3, 34.1, 21.2, 3.2],
        title="Tech Giants Margin Comparison",
        output_path="margin_comparison.png",
    )

    # Revenue segments
    create_revenue_segment_chart(
        segments=["iPhone", "Services", "Mac", "iPad", "Wearables"],
        values=[205.5e9, 78.1e9, 40.2e9, 29.3e9, 41.2e9],
        title="Apple Revenue by Segment (FY22)",
        output_path="apple_segments.png",
    )
