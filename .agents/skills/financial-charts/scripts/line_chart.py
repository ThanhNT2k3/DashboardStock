#!/usr/bin/env python3
"""
Financial Line Chart Generator

Creates line charts for visualizing:
- Revenue/profit trends over time
- Stock price movements
- Margin trends
- Growth rate comparisons
"""

import plotly.graph_objects as go
from typing import Optional
import sys
from pathlib import Path

# Handle imports for both module and standalone execution
try:
    from .themes import get_theme, get_palette
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    from themes import get_theme, get_palette


def create_line_chart(
    x_values: list,
    y_values: list[float],
    title: str = "Line Chart",
    output_path: str = "line_chart.png",
    theme: str = "default",
    width: int = 900,
    height: int = 500,
    x_title: str = "",
    y_title: str = "",
    y_format: str = "${:,.0f}",
    show_markers: bool = True,
    fill: Optional[str] = None,
    line_color: Optional[str] = None,
) -> str:
    """
    Create a simple line chart.

    Args:
        x_values: X-axis values (dates, periods, etc.)
        y_values: Y-axis values
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        width: Chart width in pixels
        height: Chart height in pixels
        x_title: X-axis title
        y_title: Y-axis title
        y_format: Format string for y-axis values
        show_markers: Whether to show data point markers
        fill: Fill area below line ("tozeroy", "tonexty", None)
        line_color: Override line color

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)
    color = line_color or theme_colors["revenue"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x_values,
        y=y_values,
        mode="lines+markers" if show_markers else "lines",
        line=dict(color=color, width=2.5),
        marker=dict(size=8, color=color) if show_markers else None,
        fill=fill,
        fillcolor=f"rgba{tuple(list(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.2])}" if fill else None,
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
        margin=dict(l=70, r=40, t=80, b=60),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor=theme_colors["grid"],
        showline=True,
        linecolor=theme_colors["grid"],
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=theme_colors["grid"],
        showline=True,
        linecolor=theme_colors["grid"],
    )

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_multi_line_chart(
    x_values: list,
    series: dict[str, list[float]],
    title: str = "Multi-Line Chart",
    output_path: str = "multi_line_chart.png",
    theme: str = "default",
    palette: str = "default",
    width: int = 900,
    height: int = 500,
    x_title: str = "",
    y_title: str = "",
    show_markers: bool = True,
) -> str:
    """
    Create a multi-line chart for comparing trends.

    Args:
        x_values: X-axis values (dates, periods, etc.)
        series: Dict of series name → values {"Revenue": [...], "Costs": [...]}
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        palette: Color palette name
        width: Chart width in pixels
        height: Chart height in pixels
        x_title: X-axis title
        y_title: Y-axis title
        show_markers: Whether to show data point markers

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)
    colors = get_palette(palette)

    fig = go.Figure()

    for i, (name, values) in enumerate(series.items()):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=x_values,
            y=values,
            name=name,
            mode="lines+markers" if show_markers else "lines",
            line=dict(color=color, width=2.5),
            marker=dict(size=8, color=color) if show_markers else None,
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
        margin=dict(l=70, r=40, t=80, b=60),
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    fig.update_xaxes(showgrid=True, gridcolor=theme_colors["grid"], showline=True, linecolor=theme_colors["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=theme_colors["grid"], showline=True, linecolor=theme_colors["grid"])

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_trend_chart(
    periods: list[str],
    values: list[float],
    title: str = "Trend Analysis",
    output_path: str = "trend_chart.png",
    theme: str = "default",
    show_growth: bool = True,
    value_format: str = "${:,.0f}",
) -> str:
    """
    Create a trend chart with growth annotations.

    Args:
        periods: Period labels (quarters, years)
        values: Values for each period
        title: Chart title
        output_path: Output file path
        theme: Color theme name
        show_growth: Whether to show growth rate annotations
        value_format: Format string for values

    Returns:
        Path to saved chart
    """
    theme_colors = get_theme(theme)

    fig = go.Figure()

    # Main line
    fig.add_trace(go.Scatter(
        x=periods,
        y=values,
        mode="lines+markers+text",
        line=dict(color=theme_colors["revenue"], width=3),
        marker=dict(size=10, color=theme_colors["revenue"]),
        text=[value_format.format(v) for v in values],
        textposition="top center",
        textfont=dict(size=10),
    ))

    # Add growth annotations
    if show_growth and len(values) > 1:
        annotations = []
        for i in range(1, len(values)):
            growth = (values[i] - values[i-1]) / values[i-1] * 100
            color = theme_colors["profit"] if growth >= 0 else theme_colors["cost"]
            sign = "+" if growth >= 0 else ""
            annotations.append(dict(
                x=periods[i],
                y=values[i],
                text=f"{sign}{growth:.1f}%",
                showarrow=False,
                yshift=30,
                font=dict(size=9, color=color),
            ))
        fig.update_layout(annotations=annotations)

    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=theme_colors["text"]),
            x=0.5,
        ),
        font=dict(family="Arial", color=theme_colors["text"]),
        paper_bgcolor=theme_colors["background"],
        plot_bgcolor=theme_colors["background"],
        width=900,
        height=500,
        margin=dict(l=70, r=40, t=80, b=60),
    )

    fig.update_xaxes(showgrid=False, showline=True, linecolor=theme_colors["grid"])
    fig.update_yaxes(showgrid=True, gridcolor=theme_colors["grid"], showline=True, linecolor=theme_colors["grid"])

    if output_path.endswith(".html"):
        fig.write_html(output_path)
    else:
        fig.write_image(output_path)

    return output_path


def create_margin_trend_chart(
    periods: list[str],
    gross_margins: list[float],
    operating_margins: list[float],
    net_margins: Optional[list[float]] = None,
    title: str = "Margin Trends",
    output_path: str = "margin_trends.png",
    theme: str = "default",
) -> str:
    """
    Create a margin trend chart over time.

    Args:
        periods: Period labels
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

    return create_multi_line_chart(
        x_values=periods,
        series=series,
        title=title,
        output_path=output_path,
        theme=theme,
        y_title="Margin (%)",
    )


# Example usage
if __name__ == "__main__":
    # Simple trend chart
    create_trend_chart(
        periods=["Q1 2023", "Q2 2023", "Q3 2023", "Q4 2023", "Q1 2024", "Q2 2024"],
        values=[85e6, 92e6, 88e6, 105e6, 98e6, 112e6],
        title="Quarterly Revenue Trend",
        output_path="revenue_trend.png",
    )

    # Multi-line comparison
    create_multi_line_chart(
        x_values=["2019", "2020", "2021", "2022", "2023"],
        series={
            "Apple": [260e9, 275e9, 365e9, 394e9, 383e9],
            "Microsoft": [125e9, 143e9, 168e9, 198e9, 211e9],
            "Google": [161e9, 182e9, 257e9, 283e9, 307e9],
        },
        title="Tech Giants Revenue Comparison",
        output_path="tech_revenue_comparison.png",
        y_title="Revenue",
    )

    # Margin trends
    create_margin_trend_chart(
        periods=["Q1", "Q2", "Q3", "Q4"],
        gross_margins=[42.5, 43.1, 43.8, 43.3],
        operating_margins=[29.8, 30.2, 31.5, 30.3],
        net_margins=[24.5, 25.1, 26.2, 25.3],
        title="Apple Margin Trends (FY22)",
        output_path="apple_margin_trends.png",
    )
