# Financial Charts Reference Guide

## Chart Selection Matrix

| Data Type | Best Chart | Script |
|-----------|------------|--------|
| Income flow (Revenue→Profit) | Sankey | `sankey_chart.py` |
| Period comparisons | Waterfall | `waterfall_chart.py` |
| Category comparisons | Bar | `bar_chart.py` |
| Trends over time | Line | `line_chart.py` |
| Variance analysis | Waterfall | `waterfall_chart.py` |
| Segment breakdown | Horizontal Bar | `bar_chart.py` |
| Competitive margins | Grouped Bar | `bar_chart.py` |

## Available Themes

| Theme | Use Case |
|-------|----------|
| `default` | General purpose, professional blue |
| `corporate` | Traditional business presentations |
| `dark` | Dark backgrounds, dashboards |
| `apple` | Apple-style clean aesthetics |
| `tech` | Modern tech company style |
| `financial` | Traditional banking/finance |
| `minimal` | Clean grayscale with accents |

## Sankey Charts

### Income Statement Flow
```python
from scripts.sankey_chart import create_income_statement_sankey

create_income_statement_sankey(
    revenue_sources={"Product A": 100e6, "Product B": 50e6},
    cost_of_revenue=90e6,
    operating_expenses={"R&D": 15e6, "SG&A": 10e6},
    other_expenses={"Tax": 8e6},
    company_name="Company",
    fiscal_period="FY24",
    theme="corporate",
    output_path="income_flow.png",
)
```

### Custom Sankey
```python
from scripts.sankey_chart import create_sankey_chart

create_sankey_chart(
    labels=["Source A", "Source B", "Total", "Output"],
    sources=[0, 1, 2],
    targets=[2, 2, 3],
    values=[100, 50, 150],
    title="Custom Flow",
    output_path="custom_sankey.png",
)
```

## Waterfall Charts

### Profit Walkdown
```python
from scripts.waterfall_chart import create_profit_walkdown

create_profit_walkdown(
    revenue=100e6,
    cost_of_goods_sold=60e6,
    operating_expenses={"R&D": 10e6, "SG&A": 15e6},
    other_items={"Tax": -3e6},
    title="Q4 Profit Walkdown",
    output_path="walkdown.png",
)
```

### Revenue Bridge
```python
from scripts.waterfall_chart import create_revenue_bridge

create_revenue_bridge(
    start_value=80e6,
    start_label="Q3",
    changes={"Price": 5e6, "Volume": -2e6, "New Products": 10e6},
    end_label="Q4",
    title="Revenue Bridge",
    output_path="bridge.png",
)
```

## Bar Charts

### Margin Comparison
```python
from scripts.bar_chart import create_margin_comparison_chart

create_margin_comparison_chart(
    companies=["Company A", "Company B", "Company C"],
    gross_margins=[45, 52, 38],
    operating_margins=[20, 28, 15],
    net_margins=[12, 18, 8],
    title="Competitive Margins",
    output_path="margins.png",
)
```

### Revenue Segments
```python
from scripts.bar_chart import create_revenue_segment_chart

create_revenue_segment_chart(
    segments=["North America", "Europe", "Asia", "Other"],
    values=[50e6, 30e6, 40e6, 10e6],
    title="Revenue by Region",
    output_path="segments.png",
)
```

## Line Charts

### Trend with Growth
```python
from scripts.line_chart import create_trend_chart

create_trend_chart(
    periods=["Q1", "Q2", "Q3", "Q4"],
    values=[25e6, 28e6, 32e6, 35e6],
    title="Quarterly Revenue",
    show_growth=True,
    output_path="trend.png",
)
```

### Multi-company Comparison
```python
from scripts.line_chart import create_multi_line_chart

create_multi_line_chart(
    x_values=["2020", "2021", "2022", "2023"],
    series={
        "Company A": [100e6, 120e6, 150e6, 180e6],
        "Company B": [80e6, 95e6, 110e6, 130e6],
    },
    title="Revenue Comparison",
    output_path="comparison.png",
)
```

## Output Formats

All charts support:
- `.png` - Static image (default)
- `.html` - Interactive web version
- `.pdf` - Print-ready PDF

## Dependencies

Required packages:
```
plotly>=5.0.0
kaleido>=0.2.0  # for static image export
```

Install with:
```bash
pip install plotly kaleido
```

## Color Customization

### Using Palettes
```python
from scripts.themes import get_palette

colors = get_palette("tech")  # Returns list of 6 colors
```

### Creating Gradients
```python
from scripts.themes import create_gradient_colors

gradient = create_gradient_colors("#2E86AB", "#DC3545", steps=5, alpha=0.5)
```

### Available Palettes
- `default` - Professional blues and greens
- `corporate` - Subdued business colors
- `tech` - Modern vibrant colors
- `pastel` - Soft muted tones
- `vibrant` - High contrast colors
