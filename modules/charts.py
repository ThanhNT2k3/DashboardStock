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
    'bg':       '#0E1117',
    'card_bg':  '#1E2130',
    'grid':     '#2A2D3E',
    'text':     '#FAFAFA',
    'accent':   '#00B4D8',
    'ma20':     '#FFD60A',
    'ma50':     '#FF9A3C',
    'ma200':    '#FF4D6D',
}

BASE_LAYOUT = dict(
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['card_bg'],
    font=dict(color=COLORS['text'], family='IBM Plex Sans, sans-serif'),
    margin=dict(l=40, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
    yaxis=dict(gridcolor=COLORS['grid'], showgrid=True),
)


# ─────────────────────────────────────────────
# 1. Sentiment Gauge
# ─────────────────────────────────────────────

def sentiment_gauge(score: float, label: str, color: str) -> go.Figure:
    """Fear/Greed gauge 0–100"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        number={'font': {'size': 48, 'color': color}},
        title={'text': f"<b>{label}</b>", 'font': {'size': 18, 'color': color}},
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': COLORS['text'],
                'tickvals': [0, 25, 45, 55, 75, 100],
                'ticktext': ['0', '25', '45', '55', '75', '100'],
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': COLORS['card_bg'],
            'borderwidth': 0,
            'steps': [
                {'range': [0,  25], 'color': '#1a0000'},
                {'range': [25, 45], 'color': '#1a0a00'},
                {'range': [45, 55], 'color': '#1a1500'},
                {'range': [55, 75], 'color': '#001a05'},
                {'range': [75,100], 'color': '#001a0a'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 3},
                'thickness': 0.75,
                'value': score,
            }
        }
    ))
    fig.update_layout(
        paper_bgcolor=COLORS['bg'],
        height=280,
        margin=dict(l=20, r=20, t=20, b=10),
        font=dict(color=COLORS['text']),
    )
    return fig


def sentiment_history_chart(sent_hist_df: pd.DataFrame) -> go.Figure:
    """Line chart: Diễn biến Sentiment Score theo thời gian"""
    if sent_hist_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Sentiment History (không có dữ liệu)')
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sent_hist_df['date'],
        y=sent_hist_df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Index',
        line=dict(color=COLORS['accent'], width=2),
        marker=dict(size=4),
        fill='tozeroy',
        fillcolor='rgba(0, 180, 216, 0.1)',
        hovertemplate='%{x}<br>Sentiment: %{y:.1f}<extra></extra>'
    ))

    # Baseline 50
    fig.add_hline(y=50, line_dash='dash', line_color=COLORS['neutral'], opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        height=180,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS['grid']),
        showlegend=False,
    )
    return fig


def market_power_chart(power_df: pd.DataFrame) -> go.Figure:
    """Supply/Demand & Power chart"""
    if power_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Market Power (không có dữ liệu)')
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
    """Biểu đồ VNINDEX (nếu có dữ liệu)"""
    vni = prices_dict.get('VNINDEX')
    if not vni:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='VNINDEX (chưa có dữ liệu)')
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
        title=dict(text='Chỉ số VN-INDEX', font=dict(size=14)),
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
        title=dict(text='% Cổ phiếu so với MA', font=dict(size=14)),
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
        fig.update_layout(BASE_LAYOUT, title='Breadth History (không có dữ liệu)')
        return fig

    fig = go.Figure()
    
    # Lấy các cột pct_ma...
    ma_cols = [c for c in ma_hist_df.columns if c.startswith('pct_ma')]
    
    for col in ma_cols:
        period = col.replace('pct_ma', '')
        color = COLORS.get(f'ma{period}', COLORS['accent'])
        
        fig.add_trace(go.Scatter(
            x=ma_hist_df['date'],
            y=ma_hist_df[col],
            mode='lines',
            name=f'% Trên MA{period}',
            line=dict(width=2, color=color),
            hovertemplate='%{x}<br>' + f'MA{period}: ' + '%{y:.1f}%<extra></extra>',
        ))

    # Vùng Overbought/Oversold
    fig.add_hline(y=80, line_dash='dot', line_color=COLORS['decline'], opacity=0.5)
    fig.add_hline(y=20, line_dash='dot', line_color=COLORS['advance'], opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Diễn biến độ rộng thị trường (% trên MA)', font=dict(size=14)),
        height=350,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], range=[0, 100], ticksuffix='%'),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.1, font=dict(color=COLORS['text'])),
    )
    return fig


# ─────────────────────────────────────────────
# 3. Advance / Decline Donut
# ─────────────────────────────────────────────

def advance_decline_donut(ad_stats: dict) -> go.Figure:
    """Donut chart: Advance / Decline / Unchanged"""
    labels = ['Tăng', 'Giảm', 'Đứng']
    values = [ad_stats['advances'], ad_stats['declines'], ad_stats['unchanged']]
    colors = [COLORS['advance'], COLORS['decline'], COLORS['neutral']]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=COLORS['bg'], width=2)),
        textinfo='label+percent',
        textfont=dict(size=13),
        hovertemplate='%{label}: %{value} mã (%{percent})<extra></extra>',
    ))
    fig.add_annotation(
        text=f"<b>{ad_stats['advances']}</b><br><span style='font-size:11px'>Tăng</span>",
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
        fig.update_layout(BASE_LAYOUT, title='A/D Line (không có dữ liệu)')
        return fig

    fig = go.Figure()

    # Tô màu vùng
    fig.add_trace(go.Scatter(
        x=ad_df['date'],
        y=ad_df['cumulative_ad'],
        mode='lines',
        name='Cumulative A/D',
        line=dict(color=COLORS['accent'], width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 180, 216, 0.1)',
        hovertemplate='%{x}<br>A/D Line: %{y:,.0f}<extra></extra>',
    ))

    # Đường zero
    fig.add_hline(y=0, line_dash='dash', line_color=COLORS['neutral'], opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Advance / Decline Line', font=dict(size=14)),
        height=280,
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], title='A/D Tích lũy'),
    )
    return fig


# ─────────────────────────────────────────────
# 5. Liquidity / Volume Chart
# ─────────────────────────────────────────────

def liquidity_chart(liq_df: pd.DataFrame) -> go.Figure:
    """Area + bar: Thanh khoản toàn thị trường theo ngày"""
    if liq_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Thanh khoản (không có dữ liệu)')
        return fig

    fig = go.Figure()

    # Bar: giá trị giao dịch theo ngày
    colors_bar = [
        COLORS['advance'] if v >= liq_df['value_bn'].mean() else '#444'
        for v in liq_df['value_bn']
    ]
    fig.add_trace(go.Bar(
        x=liq_df['date'],
        y=liq_df['value_bn'],
        name='GTGD (tỷ VNĐ)',
        marker_color=colors_bar,
        opacity=0.7,
        hovertemplate='%{x}<br>GTGD: %{y:,.0f} tỷ<extra></extra>',
    ))

    # Line: MA5
    if 'volume_ma5' in liq_df.columns:
        ma5_value = liq_df['value_bn'].rolling(5).mean()
        fig.add_trace(go.Scatter(
            x=liq_df['date'],
            y=ma5_value,
            mode='lines',
            name='MA5',
            line=dict(color=COLORS['ma20'], width=2),
            hovertemplate='%{x}<br>MA5: %{y:,.0f} tỷ<extra></extra>',
        ))

    # Đường trung bình
    avg = liq_df['value_bn'].mean()
    fig.add_hline(
        y=avg,
        line_dash='dot', line_color=COLORS['neutral'], opacity=0.6,
        annotation_text=f"Avg: {avg:,.0f}B",
        annotation_font=dict(color=COLORS['neutral'])
    )

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Thanh khoản thị trường (Giá trị giao dịch)', font=dict(size=14)),
        height=300,
        barmode='overlay',
        xaxis=dict(gridcolor=COLORS['grid'], tickangle=-30),
        yaxis=dict(gridcolor=COLORS['grid'], title='Tỷ VNĐ'),
        legend=dict(orientation='h', x=0, y=1.1, font=dict(color=COLORS['text'])),
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
        text=[f"↑ {hl_stats['new_highs']} mã"],
        textposition='inside',
        textfont=dict(size=13, color='black'),
    ))
    fig.add_trace(go.Bar(
        x=[-hl_stats['new_lows']],
        y=['52W High/Low'],
        orientation='h',
        name='New Lows',
        marker_color=COLORS['decline'],
        text=[f"↓ {hl_stats['new_lows']} mã"],
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
        xaxis=dict(gridcolor=COLORS['grid'], title='Số mã', zeroline=True, zerolinecolor=COLORS['neutral']),
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
        fig.update_layout(BASE_LAYOUT, title='McClellan (không đủ dữ liệu)')
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
    """Histogram phân phối % thay đổi giá"""
    raw = dist_data.get('raw', [])
    if not raw:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Phân phối thay đổi giá')
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
        hovertemplate='%{x:.1f}%: %{y} mã<extra></extra>',
    ))

    fig.add_vline(x=0, line_dash='dash', line_color='white', opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='Phân phối % thay đổi giá phiên', font=dict(size=14)),
        height=280,
        xaxis=dict(gridcolor=COLORS['grid'], title='% Thay đổi', ticksuffix='%'),
        yaxis=dict(gridcolor=COLORS['grid'], title='Số mã'),
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
        title=dict(text='Phân tích Sentiment Components', font=dict(size=14)),
        showlegend=False,
    )
    return fig


# ─────────────────────────────────────────────
# 10. Advanced Breadth Indicators (RSI, PSY, Thrust)
# ─────────────────────────────────────────────

def rsi_psy_breadth_chart(adv_df: pd.DataFrame, period: int = 20) -> go.Figure:
    """Biểu đồ RSI và PSY của độ rộng thị trường (thường dùng MA20)"""
    if adv_df.empty:
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='RSI/PSY Breadth (không có dữ liệu)')
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

def backtest_equity_chart(backtest_results: dict) -> go.Figure:
    """Biểu đồ Equity Curve của chiến thuật backtest"""
    if not backtest_results or not backtest_results.get('trades'):
        fig = go.Figure()
        fig.update_layout(BASE_LAYOUT, title='Backtest Results (không có lệnh)')
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
        hovertemplate='%{x}<br>Tổng lãi lỗ: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text='CUMULATIVE PERFORMANCE (TRADE-BY-TRADE)', font=dict(size=14)),
        height=350,
        xaxis=dict(title='Trades', gridcolor=COLORS['grid']),
        yaxis=dict(title='PNL (%)', gridcolor=COLORS['grid']),
        showlegend=False,
    )
    return fig


def stochastic_chart(df: pd.DataFrame, ticker: str = "") -> go.Figure:
    """Biểu đồ Stochastic Oscillator (%K, %D)"""
    if df.empty or '%D' not in df.columns:
        return go.Figure()

    fig = go.Figure()
    
    df_plot = df.dropna(subset=['%K', '%D'])
    
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['%K'],
        name='%K', line=dict(color='#00E676', width=2),
        hovertemplate='%{x}<br>%K: %{y:.1f}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=df_plot.index, y=df_plot['%D'],
        name='%D', line=dict(color='#FFD60A', width=2),
        hovertemplate='%{x}<br>%D: %{y:.1f}<extra></extra>'
    ))

    # Overbought/Oversold levels
    fig.add_hline(y=80, line_dash='dash', line_color='#FF1744', opacity=0.5)
    fig.add_hline(y=20, line_dash='dash', line_color='#00E676', opacity=0.5)

    fig.update_layout(
        BASE_LAYOUT,
        title=dict(text=f'Stochastic Oscillator - {ticker}', font=dict(size=14)),
        height=250,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=True, gridcolor=COLORS['grid'], rangeslider=dict(visible=False)),
        yaxis=dict(range=[0, 100], showgrid=True, gridcolor=COLORS['grid']),
        legend=dict(orientation='h', x=0.5, xanchor='center', y=1.2, font=dict(color=COLORS['text'])),
    )
    return fig
