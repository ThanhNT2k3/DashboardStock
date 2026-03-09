#!/usr/bin/env python3
"""
Color themes for financial charts.

Provides consistent, professional color schemes optimized for financial data visualization.
"""

THEMES = {
    "default": {
        "revenue": "#2E86AB",
        "revenue_link": "rgba(46, 134, 171, 0.5)",
        "profit": "#28A745",
        "profit_link": "rgba(40, 167, 69, 0.5)",
        "cost": "#DC3545",
        "cost_link": "rgba(220, 53, 69, 0.5)",
        "neutral": "#6C757D",
        "neutral_link": "rgba(108, 117, 125, 0.5)",
        "background": "#FFFFFF",
        "text": "#333333",
        "grid": "#E5E5E5",
        "accent": "#17A2B8",
    },
    "corporate": {
        "revenue": "#1B4F72",
        "revenue_link": "rgba(27, 79, 114, 0.5)",
        "profit": "#145A32",
        "profit_link": "rgba(20, 90, 50, 0.5)",
        "cost": "#922B21",
        "cost_link": "rgba(146, 43, 33, 0.5)",
        "neutral": "#566573",
        "neutral_link": "rgba(86, 101, 115, 0.5)",
        "background": "#FAFAFA",
        "text": "#2C3E50",
        "grid": "#D5D8DC",
        "accent": "#2874A6",
    },
    "dark": {
        "revenue": "#5DADE2",
        "revenue_link": "rgba(93, 173, 226, 0.5)",
        "profit": "#58D68D",
        "profit_link": "rgba(88, 214, 141, 0.5)",
        "cost": "#EC7063",
        "cost_link": "rgba(236, 112, 99, 0.5)",
        "neutral": "#AEB6BF",
        "neutral_link": "rgba(174, 182, 191, 0.5)",
        "background": "#1C2833",
        "text": "#ECF0F1",
        "grid": "#34495E",
        "accent": "#3498DB",
    },
    "apple": {
        # Based on Apple's typical financial presentation style
        "revenue": "#007AFF",
        "revenue_link": "rgba(0, 122, 255, 0.5)",
        "profit": "#34C759",
        "profit_link": "rgba(52, 199, 89, 0.5)",
        "cost": "#FF3B30",
        "cost_link": "rgba(255, 59, 48, 0.5)",
        "neutral": "#8E8E93",
        "neutral_link": "rgba(142, 142, 147, 0.5)",
        "background": "#FFFFFF",
        "text": "#1D1D1F",
        "grid": "#E5E5EA",
        "accent": "#5856D6",
    },
    "tech": {
        # Modern tech company style
        "revenue": "#6366F1",
        "revenue_link": "rgba(99, 102, 241, 0.5)",
        "profit": "#10B981",
        "profit_link": "rgba(16, 185, 129, 0.5)",
        "cost": "#EF4444",
        "cost_link": "rgba(239, 68, 68, 0.5)",
        "neutral": "#9CA3AF",
        "neutral_link": "rgba(156, 163, 175, 0.5)",
        "background": "#F9FAFB",
        "text": "#111827",
        "grid": "#E5E7EB",
        "accent": "#8B5CF6",
    },
    "financial": {
        # Traditional finance/banking style
        "revenue": "#1E3A5F",
        "revenue_link": "rgba(30, 58, 95, 0.5)",
        "profit": "#2D5016",
        "profit_link": "rgba(45, 80, 22, 0.5)",
        "cost": "#7B1818",
        "cost_link": "rgba(123, 24, 24, 0.5)",
        "neutral": "#4A5568",
        "neutral_link": "rgba(74, 85, 104, 0.5)",
        "background": "#FFFFFF",
        "text": "#1A202C",
        "grid": "#CBD5E0",
        "accent": "#2B6CB0",
    },
    "minimal": {
        # Clean grayscale with subtle accents
        "revenue": "#374151",
        "revenue_link": "rgba(55, 65, 81, 0.4)",
        "profit": "#059669",
        "profit_link": "rgba(5, 150, 105, 0.4)",
        "cost": "#DC2626",
        "cost_link": "rgba(220, 38, 38, 0.4)",
        "neutral": "#9CA3AF",
        "neutral_link": "rgba(156, 163, 175, 0.4)",
        "background": "#FFFFFF",
        "text": "#1F2937",
        "grid": "#F3F4F6",
        "accent": "#4B5563",
    },
}

# Color palettes for multi-series charts
PALETTES = {
    "default": ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B", "#95C623"],
    "corporate": ["#1B4F72", "#145A32", "#922B21", "#6C3483", "#7E5109", "#1A5276"],
    "tech": ["#6366F1", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"],
    "pastel": ["#A8DADC", "#F4A261", "#E76F51", "#2A9D8F", "#E9C46A", "#264653"],
    "vibrant": ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"],
}


def get_theme(name: str = "default") -> dict:
    """Get a theme by name."""
    return THEMES.get(name, THEMES["default"])


def get_palette(name: str = "default") -> list[str]:
    """Get a color palette by name."""
    return PALETTES.get(name, PALETTES["default"])


def list_themes() -> list[str]:
    """List available theme names."""
    return list(THEMES.keys())


def list_palettes() -> list[str]:
    """List available palette names."""
    return list(PALETTES.keys())


def create_gradient_colors(
    start_color: str,
    end_color: str,
    steps: int,
    alpha: float = 0.5,
) -> list[str]:
    """
    Create a gradient between two colors for link coloring.

    Args:
        start_color: Starting hex color
        end_color: Ending hex color
        steps: Number of gradient steps
        alpha: Transparency (0-1)

    Returns:
        List of rgba color strings
    """
    def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
        hex_color = hex_color.lstrip("#")
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

    start_rgb = hex_to_rgb(start_color)
    end_rgb = hex_to_rgb(end_color)

    colors = []
    for i in range(steps):
        ratio = i / (steps - 1) if steps > 1 else 0
        r = int(start_rgb[0] + (end_rgb[0] - start_rgb[0]) * ratio)
        g = int(start_rgb[1] + (end_rgb[1] - start_rgb[1]) * ratio)
        b = int(start_rgb[2] + (end_rgb[2] - start_rgb[2]) * ratio)
        colors.append(f"rgba({r}, {g}, {b}, {alpha})")

    return colors
