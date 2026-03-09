"""
Plotly Chart Builders
Tất cả charts cho Vietnam Stock Market Dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime

# ─── Color Palette ───────────────────────────
COLORS = {
    'advance':  '#00E676',
    'decline':  '#FF1744',
    'neutral':  '#FFD740',
    'bg':       '#080B12',
    'card_bg':  '#0F1420',
    'grid':     'rgba(255,255,255,0.04)',
    'grid2':    'rgba(0,180,216,0.08)',
    'text':     '#C8D8E8',
    'text_dim': '#4A6080',
    'accent':   '#00B4D8',
    'accent2':  '#00D4FF',
    'ma10':     '#F72585',
    'ma20':     '#FFD60A',
    'ma50':     '#FF9A3C',
    'ma200':    '#FF4D6D',
    'border':   'rgba(42,53,80,0.8)',
}

_FONT = 'DM Sans, sans-serif'
_MONO = 'JetBrains Mono, monospace'

BASE_LAYOUT = dict(
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['card_bg'],
    font=dict(color=COLORS['text'], family=_FONT, size=12),
    margin=dict(l=50, r=24, t=44, b=44),
    xaxis=dict(
        gridcolor=COLORS['grid'],
        showgrid=True,
        linecolor=COLORS['border'],
        tickfont=dict(size=10, color=COLORS['text_dim'], family=_MONO),
        showspikes=True,
        spikecolor=COLORS['accent'],
        spikethickness=1,
        spikedash='dot',
    ),
    yaxis=dict(
        gridcolor=COLORS['grid'],
        showgrid=True,
        linecolor=COLORS['border'],
        tickfont=dict(size=10, color=COLORS['text_dim'], family=_MONO),
        showspikes=True,
        spikecolor=COLORS['accent'],
        spikethickness=1,
        spikedash='dot',
    ),
    hoverlabel=dict(
        bgcolor='#111827',
        font_size=12,
        font_family=_MONO,
        bordercolor=COLORS['accent'],
    ),
    hovermode='x unified',
    legend=dict(
        bgcolor='rgba(8,11,18,0.8)',
        bordercolor=COLORS['border'],
        borderwidth=1,
        font=dict(size=11, color=COLORS['text']),
    ),
)


# ─────────────────────────────────────────────
# 1. Sentiment Gauge
# ─────────────────────────────────────────────

def sentiment_gauge(score: float, label: str, color: str) -> go.Figure:
    """Fear/Greed gauge 0–100 — premium style"""
    # Dynamic zone labels
    zone_labels = [
        (0, 25, 'Extreme Fear', '#FF1744'),
        (25, 45, 'Fear', '#FF7043'),
        (45, 55, 'Neutral', '#FFD740'),
        (55, 75, 'Greed', '#69F0AE'),
        (75, 100, 'Extreme Greed', '#00E676'),
    ]
    # Convert hex to rgba for Plotly compatibility with transparency
    steps = []
    for lo, hi, _, c in zone_labels:
        r = int(c[1:3], 16)
        g = int(c[3:5], 16)
        b = int(c[5:7], 16)
        steps.append({'range': [lo, hi], 'color': f'rgba({r},{g},{b},0.1)'})

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={
            'font': {'size': 44, 'color': color, 'family': _MONO},
            'valueformat': '.0f',
            'suffix': ''
        },
        title={
            'text': f'<b style="font-size:16px;">{label}</b>',
            'font': {'size': 16, 'color': color, 'family': _FONT}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': COLORS['text_dim'],
                'tickvals': [0, 25, 50, 75, 100],
                'ticktext': ['0', '25', '50', '75', '100'],
                'tickfont': {'size': 9, 'color': COLORS['text_dim'], 'family': _MONO},
            },
            'bar': {'color': color, 'thickness': 0.18},
            'bgcolor': 'rgba(0,0,0,0)',
            'borderwidth': 0,
            'steps': steps,
            'threshold': {
                'line': {'color': color, 'width': 2},
                'thickness': 0.85,
                'value': score,
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        height=260,
        margin=dict(l=20, r=20, t=30, b=10),
        font=dict(color=COLORS['text'], family=_FONT),
    )
    return fig


def sentiment_history_chart(sent_hist_df: pd.DataFrame) -> go.Figure:
    """Line chart: Sentiment Score Evolution"""
    if sent_hist_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Sentiment History')
        return fig

    fig = go.Figure()
    # Zone backgrounds
    fig.add_hrect(y0=0,  y1=40,  fillcolor='rgba(255,23,68,0.04)',  line_width=0, layer='below')
    fig.add_hrect(y0=60, y1=100, fillcolor='rgba(0,230,118,0.04)',  line_width=0, layer='below')

    fig.add_trace(go.Scatter(
        x=sent_hist_df['date'],
        y=sent_hist_df['sentiment_score'],
        mode='lines',
        name='Sentiment',
        line=dict(color=COLORS['accent2'], width=2.5, shape='spline', smoothing=0.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.06)',
        hovertemplate='<b>%{x}</b><br>Score: <b>%{y:.1f}</b><extra></extra>'
    ))
    fig.add_hline(y=50, line_dash='dot', line_color=COLORS['neutral'],
                  opacity=0.5, annotation_text='50',
                  annotation_font_size=9, annotation_font_color=COLORS['neutral'],
                  annotation_position='right')

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Sentiment History', font=dict(size=12, color=COLORS['text_dim'])),
        height=170,
        margin=dict(l=20, r=50, t=30, b=20),
        xaxis=dict(showgrid=False, tickangle=-30, tickfont=dict(size=9)),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS['grid'],
                   tickfont=dict(size=9)),
        showlegend=False,
        hovermode='x unified',
    )
    return fig


def market_power_chart(power_df: pd.DataFrame) -> go.Figure:
    """Supply/Demand & Power chart"""
    if power_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Market Power (no data)')
        return fig

    fig = go.Figure()
    
    # Supply & Demand bars
    fig.add_trace(go.Bar(
        x=power_df['date'], y=power_df['demand'],
        name='Demand (M)', marker_color=COLORS['advance'],
        opacity=0.6, yaxis='y'
    ))
    fig.add_trace(go.Bar(
        x=power_df['date'], y=-power_df['supply'],
        name='Supply (M)', marker_color=COLORS['decline'],
        opacity=0.6, yaxis='y'
    ))
    
    # Power line
    fig.add_trace(go.Scatter(
        x=power_df['date'], y=power_df['power'],
        name='Power Index', line=dict(color=COLORS['accent'], width=3),
        yaxis='y2'
    ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Market Supply/Demand & Power Index', font=dict(size=14)),
        height=350,
        barmode='overlay',
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(title='Volume (Millions)', gridcolor=COLORS['grid']),
        yaxis2=dict(
            title='Power', overlaying='y', side='right',
            showgrid=False, gridcolor='rgba(0,0,0,0)'
        ),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.1, font=dict(color=COLORS['text'])),
    )
    return fig


def vnindex_chart(prices_dict: dict, lookback: int = 60) -> go.Figure:
    """VNINDEX Chart (if data available)"""
    vni = prices_dict.get('VNINDEX')
    if not vni:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='VNINDEX (no data available yet)')
        return fig

    ts     = vni.get('timestamps', [])[-lookback:]
    closes = vni.get('close', [])[-lookback:]
    dates  = [datetime.fromtimestamp(t).strftime('%Y-%m-%d') for t in ts]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=closes, mode='lines',
        name='VNINDEX', line=dict(color='#FAFAFA', width=3),
        fill='tozeroy', fillcolor='rgba(250, 250, 250, 0.05)'
    ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='VN-INDEX Index', font=dict(size=14)),
        height=350,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], side='right'),
    )
    return fig


# ─────────────────────────────────────────────
# 2. MA Above % — Grouped Bar
# ─────────────────────────────────────────────

def ma_above_bar(ma_stats: dict) -> go.Figure:
    """Grouped bar: % cổ phiếu trên MA20/50/200"""
    periods = sorted(ma_stats.keys())
    colors  = [COLORS['ma20'], COLORS['ma50'], COLORS['ma200']]

    pct_above = [ma_stats[p]['pct_above'] for p in periods]
    pct_below = [100 - x for x in pct_above]
    labels    = [f"MA{p}" for p in periods]

    above_counts = [ma_stats[p]['above'] for p in periods]
    below_counts = [ma_stats[p]['below'] for p in periods]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name='Above MA',
        x=labels,
        y=pct_above,
        marker_color=[COLORS['advance']] * len(periods),
        text=[f"{c}<br>({v:.1f}%)" for c, v in zip(above_counts, pct_above)],
        textposition='inside',
        textfont=dict(size=12, color='black'),
    ))
    fig.add_trace(go.Bar(
        name='Below MA',
        x=labels,
        y=pct_below,
        marker_color=[COLORS['decline']] * len(periods),
        text=[f"{c}<br>({v:.1f}%)" for c, v in zip(below_counts, pct_below)],
        textposition='inside',
        textfont=dict(size=12, color='white'),
    ))
    fig.update_layout(
        BASE_LAYOUT,
        barmode='stack',
        title=dict(text='% Stocks vs MA', font=dict(size=14)),
        height=300,
        showlegend=True,
        legend=dict(
            orientation='h', x=0.5, xanchor='center', y=1.1,
            font=dict(color=COLORS['text'])
        ),
        yaxis=dict(range=[0, 100], ticksuffix='%', gridcolor=COLORS['grid']),
    )
    return fig


def ma_breadth_history_chart(ma_hist_df: pd.DataFrame) -> go.Figure:
    """Line chart: % cổ phiếu trên các đường MA theo lịch sử"""
    if ma_hist_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Breadth History')
        return fig

    fig = go.Figure()

    # Overbought / Oversold zones
    fig.add_hrect(y0=80, y1=100, fillcolor='rgba(255,23,68,0.05)', line_width=0,
                  annotation_text='Overbought Zone', annotation_position='right',
                  annotation_font_size=8, annotation_font_color=COLORS['decline'])
    fig.add_hrect(y0=0, y1=20, fillcolor='rgba(0,230,118,0.05)', line_width=0,
                  annotation_text='Oversold Zone', annotation_position='right',
                  annotation_font_size=8, annotation_font_color=COLORS['advance'])

    ma_cols = [c for c in ma_hist_df.columns if c.startswith('pct_ma')]
    for col in ma_cols:
        period = col.replace('pct_ma', '')
        color = COLORS.get(f'ma{period}', COLORS['accent'])
        fig.add_trace(go.Scatter(
            x=ma_hist_df['date'],
            y=ma_hist_df[col],
            mode='lines',
            name=f'MA{period} Breadth',
            line=dict(width=2, color=color),
            hovertemplate=f'<b>%{{x}}</b><br>MA{period}: <b>%{{y:.1f}}%</b><extra></extra>',
        ))

    # Key levels
    fig.add_hline(y=80, line_dash='dot', line_color=COLORS['decline'], opacity=0.4)
    fig.add_hline(y=50, line_dash='dot', line_color=COLORS['text_dim'], opacity=0.3)
    fig.add_hline(y=20, line_dash='dot', line_color=COLORS['advance'], opacity=0.4)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Market Breadth — % Above MA', font=dict(size=14, color=COLORS['text'])),
        height=360,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], range=[0, 100], ticksuffix='%',
                   title='% Stocks Above MA'),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.08),
    )
    return fig



# ─────────────────────────────────────────────
# 3. Advance / Decline Donut
# ─────────────────────────────────────────────

def advance_decline_donut(ad_stats: dict) -> go.Figure:
    """Donut chart: Advance / Decline / Unchanged"""
    labels = ['Advance', 'Decline', 'Unchanged']
    values = [ad_stats['advances'], ad_stats['declines'], ad_stats['unchanged']]
    colors = [COLORS['advance'], COLORS['decline'], COLORS['neutral']]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS['bg'], width=2)),
        textinfo='label+percent',
        textfont=dict(size=13),
        hovertemplate='%{label}: %{value} stocks (%{percent})<extra></extra>',
    ))
    fig.add_annotation(
        text=f"<b>{ad_stats['advances']}</b><br><span style='font-size:11px'>Advance</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=20, color=COLORS['advance'])
    )
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        font=dict(color=COLORS['text']),
        title=dict(text='Advance / Decline', font=dict(size=14)),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# 4. A/D Line — Cumulative
# ─────────────────────────────────────────────

def ad_line_chart(ad_df: pd.DataFrame) -> go.Figure:
    """Line chart: Cumulative A/D line theo lịch sử"""
    if ad_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='A/D Line')
        return fig

    fig = go.Figure()

    # Positive zone (green fill), negative zone (red fill)
    zero_level = 0
    y_vals = ad_df['cumulative_ad']
    x_vals = ad_df['date']

    # Split into green/red segments
    y_pos = [y if y >= 0 else 0 for y in y_vals]
    y_neg = [y if y < 0 else 0 for y in y_vals]

    fig.add_trace(go.Scatter(
        x=x_vals, y=y_pos,
        mode='none', fill='tozeroy',
        fillcolor='rgba(0,230,118,0.08)', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_vals, y=y_neg,
        mode='none', fill='tozeroy',
        fillcolor='rgba(255,23,68,0.08)', showlegend=False
    ))
    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='lines',
        name='Cumulative A/D',
        line=dict(color=COLORS['accent2'], width=2.5),
        hovertemplate='<b>%{x}</b><br>A/D: <b>%{y:,.0f}</b><extra></extra>',
    ))
    # Zero line
    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['neutral'], opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Advance / Decline Line (Cumulative)', font=dict(size=14)),
        height=300,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], title='Cumulative A/D',
                   tickformat=',.0f'),
    )
    return fig



# ─────────────────────────────────────────────
# 5. Liquidity / Volume Chart
# ─────────────────────────────────────────────

def liquidity_chart(liq_df: pd.DataFrame) -> go.Figure:
    """Area + bar: Daily market liquidity"""
    if liq_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Liquidity (no data)')
        return fig

    fig = go.Figure()

    avg = liq_df['value_bn'].mean()
    # Color bars above/below average
    colors_bar = [
        COLORS['accent'] if v >= avg else COLORS['text_dim']
        for v in liq_df['value_bn']
    ]
    fig.add_trace(go.Bar(
        x=liq_df['date'],
        y=liq_df['value_bn'],
        name='Value (Billion VND)',
        marker_color=colors_bar,
        marker_line_width=0,
        opacity=0.75,
        hovertemplate='<b>%{x}</b><br>Value: <b>%{y:,.0f}B</b><extra></extra>',
    ))

    # MA5 overlay
    ma5_value = liq_df['value_bn'].rolling(5).mean()
    fig.add_trace(go.Scatter(
        x=liq_df['date'],
        y=ma5_value,
        mode='lines',
        name='MA5',
        line=dict(color=COLORS['neutral'], width=2, dash='dot'),
        hovertemplate='<b>%{x}</b><br>MA5: <b>%{y:,.0f}B</b><extra></extra>',
    ))

    fig.add_hline(
        y=avg,
        line_dash='dash', line_color=COLORS['text_dim'], opacity=0.5,
        annotation_text=f'Avg {avg:,.0f}B',
        annotation_font=dict(color=COLORS['text_dim'], size=9),
        annotation_position='right',
    )

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Market Liquidity — Trading Value', font=dict(size=14)),
        height=300,
        barmode='overlay',
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], title='Billion VND'),
        legend=dict(orientation='h', x=0, y=1.08),
    )
    return fig



# ─────────────────────────────────────────────
# 6. New High / Low Bar
# ─────────────────────────────────────────────

def new_high_low_chart(hl_stats: dict) -> go.Figure:
    """Horizontal bar: New 52-week High vs Low"""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=[hl_stats['new_highs']],
        y=['52W High/Low'],
        orientation='h',
        name='New Highs',
        marker_color=COLORS['advance'],
        text=[f"↑ {hl_stats['new_highs']} stocks"],
        textposition='inside',
        textfont=dict(size=13, color='black'),
    ))
    fig.add_trace(go.Bar(
        x=[-hl_stats['new_lows']],
        y=['52W High/Low'],
        orientation='h',
        name='New Lows',
        marker_color=COLORS['decline'],
        text=[f"↓ {hl_stats['new_lows']} stocks"],
        textposition='inside',
        textfont=dict(size=13, color='white'),
    ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='52-Week New High vs Low', font=dict(size=14)),
        height=180,
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.2, font=dict(color=COLORS['text'])),
        xaxis=dict(gridcolor=COLORS['grid'], title='Stocks', zeroline=True, zerolinecolor=COLORS['neutral']),
        yaxis=dict(showticklabels=False),
    )
    return fig


# ─────────────────────────────────────────────
# 7. McClellan Oscillator
# ─────────────────────────────────────────────

def mcclellan_chart(mc_df: pd.DataFrame) -> go.Figure:
    """McClellan Oscillator + Summation line"""
    if mc_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='McClellan (insufficient data)')
        return fig

    fig = go.Figure()

    # Oscillator bars
    bar_colors = [
        COLORS['advance'] if v >= 0 else COLORS['decline']
        for v in mc_df['oscillator']
    ]
    fig.add_trace(go.Bar(
        x=mc_df['date'],
        y=mc_df['oscillator'],
        name='Oscillator',
        marker_color=bar_colors,
        opacity=0.8,
        yaxis='y',
        hovertemplate='%{x}<br>Oscillator: %{y:.1f}<extra></extra>',
    ))

    # Summation line
    fig.add_trace(go.Scatter(
        x=mc_df['date'],
        y=mc_df['summation'],
        mode='lines',
        name='Summation',
        line=dict(color=COLORS['accent'], width=2),
        yaxis='y2',
        hovertemplate='%{x}<br>Summation: %{y:.0f}<extra></extra>',
    ))

    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['neutral'], opacity=0.4)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='McClellan Oscillator & Summation', font=dict(size=14)),
        height=300,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], title='Oscillator', side='left'),
        yaxis2=dict(
            title='Summation',
            overlaying='y', side='right',
            gridcolor='rgba(0,0,0,0)',
            showgrid=False,
        ),
        legend=dict(orientation='h', x=0, y=1.1, font=dict(color=COLORS['text'])),
    )
    return fig


# ─────────────────────────────────────────────
# 8. Change Distribution Histogram
# ─────────────────────────────────────────────

def change_distribution_chart(dist_data: dict) -> go.Figure:
    """Histogram of price change % distribution"""
    raw = dist_data.get('raw', [])
    if not raw:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Price Change Distribution')
        return fig

    colors_hist = [
        COLORS['advance'] if x > 0.5 else (COLORS['decline'] if x < -0.5 else COLORS['neutral'])
        for x in raw
    ]

    fig = go.Figure(go.Histogram(
        x=raw,
        nbinsx=40,
        marker=dict(
            color=raw,
            colorscale=[
                [0.0, COLORS['decline']],
                [0.5, COLORS['neutral']],
                [1.0, COLORS['advance']],
            ],
            cmin=-10, cmax=10,
        ),
        hovertemplate='%{x:.1f}%: %{y} stocks<extra></extra>',
    ))

    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Session Price Change Distribution (%)', font=dict(size=14)),
        height=280,
        xaxis=dict(gridcolor=COLORS['grid'], title='Change (%)', ticksuffix='%'),
        yaxis=dict(gridcolor=COLORS['grid'], title='Number of stocks'),
    )
    return fig


# ─────────────────────────────────────────────
# 9. Sentiment Components Radar
# ─────────────────────────────────────────────

def sentiment_components_chart(components: dict) -> go.Figure:
    """Radar chart: các thành phần Sentiment"""
    categories = list(components.keys())
    values     = list(components.values())
    # Đóng vòng radar
    categories += [categories[0]]
    values     += [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 180, 216, 0.2)',
        line=dict(color=COLORS['accent'], width=2),
        marker=dict(size=6, color=COLORS['accent']),
        hovertemplate='%{theta}: %{r:.1f}<extra></extra>',
    ))

    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        height=280,
        margin=dict(l=20, r=20, t=30, b=20),
        polar=dict(
            bgcolor=COLORS['card_bg'],
            radialaxis=dict(
                visible=True, range=[0, 100],
                color=COLORS['text'], gridcolor=COLORS['grid'],
            ),
            angularaxis=dict(color=COLORS['text'], gridcolor=COLORS['grid']),
        ),
        font=dict(color=COLORS['text']),
        title=dict(text='Sentiment Components Analysis', font=dict(size=14)),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# 10. Advanced Breadth Indicators (RSI, PSY, Thrust)
# ─────────────────────────────────────────────

def rsi_psy_breadth_chart(adv_df: pd.DataFrame, period: int = 20) -> go.Figure:
    """RSI and PSY chart for Market Breadth (usually MA20)"""
    if adv_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='RSI/PSY Breadth (no data)')
        return fig

    fig = go.Figure()
    
    rsi_col = f'rsi_pct_ma{period}'
    psy_col = f'psy_pct_ma{period}'
    
    if rsi_col in adv_df.columns:
        fig.add_trace(go.Scatter(
            x=adv_df['date'], y=adv_df[rsi_col],
            name=f'RSI(MA{period})', line=dict(color='#00B4D8', width=2),
            hovertemplate='%{x}<br>RSI: %{y:.1f}<extra></extra>'
        ))

    if psy_col in adv_df.columns:
        fig.add_trace(go.Scatter(
            x=adv_df['date'], y=adv_df[psy_col],
            name=f'PSY(MA{period})', line=dict(color='#FFD60A', width=2),
            hovertemplate='%{x}<br>PSY: %{y:.1f}<extra></extra>'
        ))

    # Overbought/Oversold levels cho RSI/PSY
    fig.add_hline(y=70, line_dash='dash', line_color=COLORS['decline'], opacity=0.4)
    fig.add_hline(y=30, line_dash='dash', line_color=COLORS['advance'], opacity=0.4)
    fig.add_hline(y=50, line_dash='dot', line_color=COLORS['neutral'], opacity=0.3)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text=f'Market Momentum & Psychological (RSI/PSY of %MA{period})', font=dict(size=14)),
        height=300,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], range=[0, 100]),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.1, font=dict(color=COLORS['text'])),
    )
    return fig


def breadth_thrust_chart(ad_df: pd.DataFrame) -> go.Figure:
    """Biểu đồ Zweig Breadth Thrust"""
    if ad_df.empty: return go.Figure()

    fig = go.Figure()
    
    # Giả định thrust đã được tính trong ad_df
    if 'thrust' in ad_df.columns:
        fig.add_trace(go.Scatter(
            x=ad_df['date'], y=ad_df['thrust'],
            name='Breadth Thrust', line=dict(color='#FF4D6D', width=2.5),
            fill='tozeroy', fillcolor='rgba(255, 77, 109, 0.1)',
            hovertemplate='%{x}<br>Thrust: %{y:.3f}<extra></extra>'
        ))

    # Thrust levels: Thường là vùng từ cực thấp (<0.4) vọt lên cực cao (>0.6) trong thời gian ngắn
    fig.add_hline(y=0.615, line_dash='dash', line_color=COLORS['advance'], opacity=0.6, annotation_text="Bullish Thrust")
    fig.add_hline(y=0.40, line_dash='dash', line_color=COLORS['decline'], opacity=0.6)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Zweig Breadth Thrust Indicator', font=dict(size=14)),
        height=300,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], range=[0.3, 0.7]),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# 11. Backtest Visualizations
# ─────────────────────────────────────────────

def backtest_equity_chart(backtest_results: dict, title: str = "CUMULATIVE PERFORMANCE (TRADE-BY-TRADE)") -> go.Figure:
    """Equity Curve chart for backtest strategy"""
    if not backtest_results or not backtest_results.get('trades'):
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title=title if title else 'Backtest Results (no trades)')
        return fig

    trades = backtest_results['trades']
    pnl_series = [t['pnl'] * 100 for t in trades]
    cumulative_pnl = np.cumsum(pnl_series)
    
    trade_labels = [f"Trade {i+1}" for i in range(len(trades))]

    fig = go.Figure()
    
    # Cumulative PNL line
    fig.add_trace(go.Scatter(
        x=trade_labels, y=cumulative_pnl,
        mode='lines+markers',
        name='Cum. PNL (%)',
        line=dict(color=COLORS['accent'], width=3),
        marker=dict(size=8, color=[COLORS['advance'] if p > 0 else COLORS['decline'] for p in pnl_series]),
        hovertemplate='%{x}<br>Cumulative PNL: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text=title.upper(), font=dict(size=14)),
        height=350,
        xaxis=dict(title='Trades', gridcolor=COLORS['grid']),
        yaxis=dict(title='PNL (%)', gridcolor=COLORS['grid']),
        showlegend=False,
    )
    return fig


    return fig


def professional_candlestick_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Biểu đồ Candlestick chuyên nghiệp với Volume, SMA và Bollinger Bands"""
    if df.empty:
        return go.Figure()

    from plotly.subplots import make_subplots
    
    # Create subplots with shared x-axis
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True, 
        vertical_spacing=0.03, 
        row_heights=[0.75, 0.25]
    )

    # 1. Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name='Price',
        increasing_line_color=COLORS['advance'],
        decreasing_line_color=COLORS['decline']
    ), row=1, col=1)

    # 2. SMA 20, 50
    if 'MA20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA20'], name='SMA 20', line=dict(color=COLORS['ma20'], width=1.5)), row=1, col=1)
    if 'MA50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], name='SMA 50', line=dict(color=COLORS['ma50'], width=1.5)), row=1, col=1)

    # 3. Bollinger Bands
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_upper'], 
            name='BB Upper', 
            line=dict(color='rgba(173, 216, 230, 0.3)', width=1)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=df.index, y=df['bb_lower'], 
            name='BB Lower', 
            line=dict(color='rgba(173, 216, 230, 0.3)', width=1),
            fill='tonexty', fillcolor='rgba(173, 216, 230, 0.05)'
        ), row=1, col=1)

    # 4. Volume
    colors = [COLORS['advance'] if df['close'].iloc[i] >= df['open'].iloc[i] else COLORS['decline'] for i in range(len(df))]
    fig.add_trace(go.Bar(
        x=df.index, y=df['volume'], 
        name='Volume', 
        marker_color=colors,
        opacity=0.6
    ), row=2, col=1)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text=f'Technical Analysis: {ticker}', font=dict(size=16)),
        height=600,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.05),
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    
    return fig


def macd_chart(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ MACD"""
    if df.empty or 'macd' not in df.columns:
        return go.Figure()

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD', line=dict(color='#00B4D8', width=2)))
    fig.add_trace(go.Scatter(x=df.index, y=df['signal'], name='Signal', line=dict(color='#FFD60A', width=2)))
    
    colors = [COLORS['advance'] if v >= 0 else COLORS['decline'] for v in df['hist']]
    fig.add_trace(go.Bar(x=df.index, y=df['hist'], name='Histogram', marker_color=colors, opacity=0.7))

    fig.update_layout(
        BASE_LAYOUT,
        height=250,
        margin=dict(l=20, r=20, t=20, b=20),
        title=dict(text='MACD Indicator', font=dict(size=14)),
        showlegend=True,
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.2)
    )
    return fig


def industry_performance_heatmap(industry_data: dict) -> go.Figure:
    """Heatmap hiệu suất các ngành"""
    if not industry_data:
        return go.Figure()

    sectors = list(industry_data.keys())
    performance = [industry_data[s]['change_pct'] for s in sectors]
    
    fig = go.Figure(data=go.Bar(
        x=performance,
        y=sectors,
        orientation='h',
        marker=dict(
            color=performance,
            colorscale=[[0, COLORS['decline']], [0.5, COLORS['neutral']], [1, COLORS['advance']]],
            cmin=-3,
            cmax=3
        ),
        text=[f"{p:+.2f}%" for p in performance],
        textposition='outside',
        textfont=dict(color=COLORS['text'])
    ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Performance by Sector (%)', font=dict(size=14)),
        height=max(300, len(sectors) * 25),
        xaxis=dict(title='% Change', ticksuffix='%'),
        margin=dict(l=150, r=50, t=50, b=50)
    )
    return fig


# ─────────────────────────────────────────────
# 16. Sector Money Flow (Liquidity)
# ─────────────────────────────────────────────

def sector_liquidity_chart(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ thanh khoản theo nhóm ngành"""
    if df.empty:
        return go.Figure().update_layout(title="No Data")
        
    fig = go.Figure()
    
    # Sort by GTGD
    df = df.sort_values('GTGD (Tỷ)', ascending=True)
    
    fig.add_trace(go.Bar(
        y=df['Ngành'],
        x=df['GTGD (Tỷ)'],
        orientation='h',
        marker=dict(
            color=df['GTGD (Tỷ)'],
            colorscale='Viridis',
            line=dict(color=COLORS['border'], width=1)
        ),
        text=df['GTGD (Tỷ)'].apply(lambda x: f"{x:,.1f} Tỷ"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>GTGD: %{x:,.1f} Tỷ VNĐ<extra></extra>"
    ))
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        title="Money Flow by Sector (Turnover - Billion VND)",
        xaxis_title="Trading Value (Billion VND)",
        yaxis_title=None,
        height=450,
        margin=dict(l=100, r=40, t=60, b=40)
    )
    
    return fig


# ─────────────────────────────────────────────
# 17. Single Stock Money Flow Indicators (MFI, OBV, VWAP)
# ─────────────────────────────────────────────

def single_stock_money_flow_chart(df: pd.DataFrame, mfi: pd.Series, obv: pd.Series, vwap: pd.Series) -> go.Figure:
    """Biểu đồ dòng tiền mã cổ phiếu (MFI, OBV, VWAP)"""
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Price & VWAP", "MFI (Money Flow Index)", "OBV (On-Balance Volume)")
    )
    
    # Row 1: Price & VWAP
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price", showlegend=False
    ), row=1, col=1)
    
    fig.add_trace(go.Scatter(
        x=df.index, y=vwap,
        name="VWAP", line=dict(color='#FFD60A', width=2, dash='dot'),
        mode='lines'
    ), row=1, col=1)
    
    # Row 2: MFI
    fig.add_trace(go.Scatter(
        x=df.index, y=mfi,
        name="MFI", line=dict(color='#00D4FF', width=2),
        fill='tozeroy', fillcolor='rgba(0,212,255,0.1)'
    ), row=2, col=1)
    
    # MFI Levels
    fig.add_hline(y=80, line_dash="dash", line_color="#FF1744", opacity=0.5, row=2, col=1)
    fig.add_hline(y=20, line_dash="dash", line_color="#00E676", opacity=0.5, row=2, col=1)
    
    # Row 3: OBV
    fig.add_trace(go.Scatter(
        x=df.index, y=obv,
        name="OBV", line=dict(color='#F72585', width=2),
    ), row=3, col=1)
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        height=700,
        xaxis_rangeslider_visible=False,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig


# ─────────────────────────────────────────────
# 18. VIX Index Chart
# ─────────────────────────────────────────────

def vix_chart(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ chỉ số VIX (Fear Index)"""
    if df.empty:
        return go.Figure().update_layout(title="No VIX Data")
        
    fig = go.Figure()
    
    # Area chart for VIX
    fig.add_trace(go.Scatter(
        x=df.index, y=df['close'],
        mode='lines',
        name='VIX Index',
        line=dict(color='#FFD740', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(255,215,64,0.1)',
        hovertemplate="<b>VIX Index</b><br>Ngày: %{x}<br>Giá: %{y:.2f}<extra></extra>"
    ))
    
    # Add status lines markers
    fig.add_hline(y=20, line_dash="dash", line_color="#FFA726", opacity=0.5)
    fig.add_hline(y=30, line_dash="dash", line_color="#FF1744", opacity=0.5)
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        title="VIX Fear Index (LiteFinance)",
        yaxis_title="Points",
        height=400,
        showlegend=False,
        margin=dict(l=50, r=20, t=60, b=40)
    )
    return fig


# ─────────────────────────────────────────────
# 16b. Top 10 Money Flow Stocks
# ─────────────────────────────────────────────

def top_money_flow_chart(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ Top 10 cổ phiếu tập trung dòng tiền mạnh nhất"""
    if df.empty:
        return go.Figure().update_layout(title="No Data")
    
    # Sort by value
    df = df.sort_values('Value', ascending=True) # Ascending for horizontal bar display order
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Ticker'],
        x=df['Value'],
        orientation='h',
        marker=dict(
            color=df['Value'],
            colorscale='Turbo',
            line=dict(color=COLORS['border'], width=1)
        ),
        text=df['Value'].apply(lambda x: f"{x:,.1f}B"),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>GTGD: %{x:,.1f} Tỷ VNĐ<extra></extra>"
    ))
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        title="Top 10 Stocks by Money Flow Concentration (Billion VND)",
        xaxis_title="Trading Value (Billion VND)",
        yaxis_title=None,
        height=450,
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig


# ─────────────────────────────────────────────
# 19. Foreign & Proprietary Trading Charts
# ─────────────────────────────────────────────

def org_trading_chart(df: pd.DataFrame, title: str) -> go.Figure:
    """Biểu đồ lịch sử mua bán ròng của khối ngoại hoặc tự doanh"""
    if df.empty:
        return go.Figure().update_layout(title=f"No {title} Data")
        
    # Standardize data: sort by date ascending for the chart
    df = df.copy()
    # Assuming date is in 'dd/mm' format, it might need better sorting logic if crossing years
    # But for a daily chart, we just reverse the API response which usually comes newest first
    df = df.iloc[::-1] 

    fig = go.Figure()
    
    # Colors based on net value
    colors = ['#00E676' if x > 0 else '#FF1744' for x in df['netVal_bn']]
    
    fig.add_trace(go.Bar(
        x=df['date'],
        y=df['netVal_bn'],
        marker_color=colors,
        text=df['netVal_bn'].apply(lambda x: f"{x:+.0f}"),
        textposition='outside',
        textfont=dict(size=10, color='white'),
        name="Net Value (Bn)",
        hovertemplate="<b>Ngày: %{x}</b><br>Mua ròng: %{y:,.1f} Tỷ<extra></extra>"
    ))
    
    # Optional: Line for cumulative flow or just zero line is enough for bar chart
    fig.add_hline(y=0, line_color="white", opacity=0.3)
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        title=title,
        yaxis_title="Net Value (Bn VND)",
        xaxis_title=None,
        height=400,
        margin=dict(l=50, r=20, t=60, b=40),
        showlegend=False
    )
    
    return fig


def top_trading_stocks_chart(df: pd.DataFrame, title: str, color_scale: str = 'Greens') -> go.Figure:
    """Biểu đồ Top 10 cổ phiếu được mua/bán bởi khối ngoại hoặc tự doanh"""
    if df.empty:
        return go.Figure().update_layout(title=f"No {title} Data")
    
    # Sort by Value_bn ascending for horizontal bar chart
    df = df.sort_values('Value_bn', ascending=True)
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df['Symbol'],
        x=df['Value_bn'],
        orientation='h',
        marker=dict(
            color=df['Value_bn'],
            colorscale=color_scale,
            line=dict(color=COLORS['border'], width=1)
        ),
        text=df.apply(lambda r: f"{r['Value_bn']:,.1f}B ({r['ChangePricePercent']:+.1f}%)", axis=1),
        textposition='outside',
        hovertemplate="<b>%{y}</b><br>Giá trị: %{x:,.1f} Tỷ<br>Biến động: %{customdata}%<extra></extra>",
        customdata=df['ChangePricePercent']
    ))
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        title=title,
        xaxis_title="Value (Billion VND)",
        yaxis_title=None,
        height=450,
        margin=dict(l=60, r=40, t=60, b=40)
    )
    return fig


def market_valuation_chart(df: pd.DataFrame) -> go.Figure:
    """Biểu đồ P/E và Index lịch sử của toàn thị trường"""
    if df.empty:
        return go.Figure().update_layout(title="No Valuation Data")
    
    from plotly.subplots import make_subplots
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Trace 1: PE Ratio
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Pe'],
        name="Market P/E",
        mode='lines',
        line=dict(color='#00D4FF', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0,212,255,0.1)',
        hovertemplate="Ngày: %{x}<br>P/E: %{y:.2f}<extra></extra>"
    ), secondary_y=False)
    
    # Trace 2: Index
    fig.add_trace(go.Scatter(
        x=df['Date'], y=df['Index'],
        name="VNINDEX",
        mode='lines',
        line=dict(color='#FFD740', width=1.5, dash='dot'),
        hovertemplate="Ngày: %{x}<br>Index: %{y:,.2f}<extra></extra>"
    ), secondary_y=True)
    
    fig.update_layout(BASE_LAYOUT)
    fig.update_layout(
        title="Market Valuation History (P/E vs. Index)",
        height=500,
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="P/E Ratio", secondary_y=False, gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(title_text="Index Points", secondary_y=True, gridcolor='rgba(255,255,255,0.05)')
    
    return fig


def market_treemap_chart(df: pd.DataFrame, size_col: str = 'Value') -> go.Figure:
    """
    Tạo biểu đồ Treemap (Heatmap) thị trường.
    df columns: ['Ticker', 'Sector', 'Value', 'Volume', 'Change %']
    """
    if df.empty:
        return go.Figure().update_layout(title="No Treemap Data Available", paper_bgcolor='rgba(0,0,0,0)')

    # Sort to ensure layout stability (Plolty handles sizing internally based on values)
    df = df.sort_values(size_col, ascending=False)
    
    import plotly.express as px
    
    # We use plotly.express to generate the structure then customize
    fig = px.treemap(
        df, 
        path=[px.Constant("Thị trường"), 'Sector', 'Ticker'], 
        values=size_col,
        color='Change %',
        range_color=[-7, 7],
        color_continuous_scale=[
            [0, 'rgb(39, 174, 96)'],
            [0, 'rgb(255, 0, 0)'],
            [0.5, 'rgb(255, 255, 0)'],
            [1, 'rgb(0, 255, 0)']
        ],
        hover_data=['Change %', 'Value', 'Volume']
    )
    
    # Add Padding and customize labels
    fig.update_traces(
        marker=dict(
            colorscale=[
                [0.0, 'rgb(0, 191, 255)'],   # Floor (Cyan)
                [0.1, 'rgb(255, 0, 0)'],     # Down (Red)
                [0.5, 'rgb(255, 165, 0)'],   # Ref (Orange/Yellow)
                [0.9, 'rgb(0, 255, 0)'],     # Up (Green)
                [1.0, 'rgb(128, 0, 128)']    # Ceiling (Purple)
            ],
            cmid=0,
            showscale=True,
            pad=dict(b=2, l=2, r=2, t=2), # Padding between boxes
            line=dict(width=1, color='rgba(0,0,0,0.3)')
        ),
        tiling=dict(
            pad=4, # Padding between parent sectors
        ),
        texttemplate="<b>%{label}</b><br>%{customdata[0]:.2f}%",
        hovertemplate="<b>%{label}</b><br>GTGD: %{customdata[1]:,.0f}<br>KLGD: %{customdata[2]:,.0f}<br>Biến động: %{customdata[0]:.2f}%<extra></extra>"
    )

    fig.update_layout(
        margin=dict(t=5, l=5, r=5, b=5),
        height=700,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color="#E0E0E0", size=14),
        coloraxis_showscale=False # Remove duplicate scale if any
    )
    
    return fig
