"""
🇻🇳 Vietnam Stock Market Dashboard
Phân tích độ rộng & sentiment thị trường chứng khoán Việt Nam
Data source: VPS API (histdatafeed.vps.com.vn)
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from modules import tickers as tk
from modules import fetcher, calculator, charts

# ─────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="VN Market Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono&display=swap');

    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
    }
    .main { background-color: #0E1117; }
    .stApp { background-color: #0E1117; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #1E2130;
        border: 1px solid #2A2D3E;
        border-radius: 8px;
        padding: 16px;
    }
    [data-testid="metric-container"] > div { color: #FAFAFA; }

    /* Header */
    .dashboard-header {
        background: linear-gradient(135deg, #0E1117 0%, #1E2130 50%, #0E1117 100%);
        border-bottom: 1px solid #2A2D3E;
        padding: 1rem 0;
        margin-bottom: 1.5rem;
    }
    .dashboard-title {
        font-size: 1.8rem;
        font-weight: 600;
        color: #00B4D8;
        letter-spacing: -0.5px;
    }
    .dashboard-subtitle {
        font-size: 0.85rem;
        color: #888;
        margin-top: 4px;
    }

    /* Section headers */
    .section-header {
        font-size: 0.75rem;
        font-weight: 600;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 1.5rem 0 0.5rem;
        border-left: 3px solid #00B4D8;
        padding-left: 8px;
    }

    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'IBM Plex Mono', monospace;
    }
    .status-up   { background: rgba(0,230,118,0.15); color: #00E676; border: 1px solid #00E676; }
    .status-down { background: rgba(255,23,68,0.15);  color: #FF1744; border: 1px solid #FF1744; }
    .status-neu  { background: rgba(255,215,64,0.15); color: #FFD740; border: 1px solid #FFD740; }

    /* Info box */
    .info-box {
        background: #1E2130;
        border: 1px solid #2A2D3E;
        border-radius: 8px;
        padding: 12px 16px;
        margin: 8px 0;
    }

    /* Hide Streamlit default elements */
    /* #MainMenu { visibility: hidden; } */
    /* footer { visibility: hidden; } */
    /* header { visibility: hidden; } */
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <div class="dashboard-title">📊 VN Market Dashboard</div>
    <div class="dashboard-subtitle">Phân tích độ rộng & sentiment thị trường chứng khoán Việt Nam · Data: VPS API</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar Filters
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Bộ lọc")

    exchange = st.selectbox(
        "Sàn giao dịch",
        options=["HOSE", "HNX", "UPCOM", "VN30", "VN100", "ALL"],
        index=0,
        help="Chọn sàn hoặc rổ chỉ số để phân tích"
    )

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Từ ngày",
            value=date.today() - timedelta(days=90),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )
    with col_d2:
        end_date = st.date_input(
            "Đến ngày",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )

    st.markdown("---")

    ma_periods = st.multiselect(
        "MA Periods",
        options=[10, 20, 50],
        default=[10, 20, 50],
        help="Chọn các đường MA để phân tích"
    )
    if not ma_periods:
        ma_periods = [10, 20, 50]

    st.markdown("---")

    lookback_days = st.slider(
        "Lịch sử hiển thị (ngày)",
        min_value=20,
        max_value=200,
        value=60,
        step=10,
    )

    st.markdown("---")

    min_liq = st.sidebar.slider(
        "Thanh khoản tối thiểu (Tỷ VNĐ/phiên)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Lọc các mã có GTGD trung bình 20 phiên gần nhất lớn hơn X tỷ"
    )

    st.markdown("---")

    run_btn = st.button("🔄 Tải dữ liệu", use_container_width=True, type="primary")

    st.markdown("---")
    ticker_list = tk.get_tickers(exchange)
    st.markdown(f"""
    <div class="info-box">
        <div style="color:#888;font-size:0.75rem">THÔNG TIN</div>
        <div style="margin-top:6px">
            <b style="color:#00B4D8">{exchange}</b><br>
            <span style="color:#FAFAFA;font-size:0.9rem">{len(ticker_list):,} mã cổ phiếu</span><br>
            <span style="color:#888;font-size:0.75rem">{start_date} → {end_date}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading with Cache
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(exchange: str, start: str, end: str) -> dict:
    """Fetch & process toàn bộ dữ liệu thị trường"""
    ticker_list = tk.get_tickers(exchange)
    if "VNINDEX" not in ticker_list:
        ticker_list.append("VNINDEX")

    # Limit cho demo: lấy mẫu nếu quá nhiều
    # Production: bỏ giới hạn này
    MAX_TICKERS = 5000
    if len(ticker_list) > MAX_TICKERS:
        import random
        # Luôn giữ VN30 + random sample phần còn lại
        vn30 = [t for t in tk.VN30 if t in ticker_list]
        rest = [t for t in ticker_list if t not in vn30]
        sample_rest = random.sample(rest, min(MAX_TICKERS - len(vn30), len(rest)))
        ticker_list = vn30 + sample_rest

    raw_results = fetcher.batch_fetch(ticker_list, start, end)
    return fetcher.parse_results(raw_results)


def compute_all_stats(prices_dict: dict, ma_periods: list, lookback: int) -> dict:
    """Tính tất cả chỉ số từ dữ liệu giá"""
    ma_stats     = calculator.compute_ma_stats(prices_dict, ma_periods)
    ma_history   = calculator.compute_ma_history(prices_dict, ma_periods, lookback)
    ad_stats     = calculator.compute_advance_decline(prices_dict)
    ad_history   = calculator.compute_ad_history(prices_dict, lookback)
    hl_stats     = calculator.compute_new_high_low(prices_dict)
    liq_history  = calculator.compute_liquidity_history(prices_dict, lookback)
    vol_momentum = calculator.compute_volume_momentum(prices_dict)
    dist_data    = calculator.compute_change_distribution(prices_dict)
    mc_df        = calculator.compute_mcclellan(ad_history) if not ad_history.empty else pd.DataFrame()
    power_hist   = calculator.compute_market_power_history(prices_dict, lookback)
    
    # Advanced Analytics
    adv_analytics = calculator.compute_advanced_analytics(ma_history)
    ad_history['thrust'] = calculator.compute_breadth_thrust(ad_history)
    
    sentiment_history = calculator.compute_sentiment_history(
        prices_dict, 
        ma_period=50 if 50 in ma_periods else ma_periods[0],
        lookback=lookback
    )
    sentiment = calculator.compute_sentiment_score(
        ma_stats, ad_stats, hl_stats, vol_momentum,
        ma_period=50 if 50 in ma_periods else ma_periods[0]
    )

    return {
        'ma_stats':     ma_stats,
        'ma_history':   ma_history,
        'ad_stats':     ad_stats,
        'ad_history':   ad_history,
        'hl_stats':     hl_stats,
        'liq_history':  liq_history,
        'vol_momentum': vol_momentum,
        'dist_data':    dist_data,
        'mc_df':        mc_df,
        'power_hist':   power_hist,
        'adv_analytics': adv_analytics,
        'prices_raw':   prices_dict, # Để vẽ VNINDEX chart riêng
        'sentiment':    sentiment,
        'sentiment_history': sentiment_history,
        'total_tickers': len(prices_dict),
    }


# ─────────────────────────────────────────────
# Session State init
# ─────────────────────────────────────────────
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'stats' not in st.session_state:
    st.session_state.stats = None


# ─────────────────────────────────────────────
# Trigger Load
# ─────────────────────────────────────────────
if run_btn or not st.session_state.data_loaded:
    with st.spinner(f"⏳ Đang tải dữ liệu {exchange} ({len(tk.get_tickers(exchange)):,} mã)..."):
        try:
            prices_dict = load_market_data(
                exchange,
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d')
            )

            if not prices_dict:
                st.error("❌ Không tải được dữ liệu. Kiểm tra kết nối mạng hoặc thử lại.")
                st.stop()

            # Apply Liquidity Filter
            prices_dict = calculator.filter_by_liquidity(prices_dict, min_liq)

            if not prices_dict:
                st.warning(f"⚠️ Không có mã nào thỏa mãn điều kiện thanh khoản > {min_liq} tỷ.")
                st.stop()

            st.session_state.stats = compute_all_stats(prices_dict, ma_periods, lookback_days)
            st.session_state.data_loaded = True
            st.session_state.exchange = exchange
            st.success(f"✅ Đã tải {st.session_state.stats['total_tickers']:,} mã thành công!")

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            st.stop()


# ─────────────────────────────────────────────
# Render Dashboard
# ─────────────────────────────────────────────
if not st.session_state.data_loaded or st.session_state.stats is None:
    st.info("👈 Nhấn **Tải dữ liệu** ở sidebar để bắt đầu phân tích")
    st.stop()

s = st.session_state.stats
sentiment = s['sentiment']
ad = s['ad_stats']
hl = s['hl_stats']


# ══════════════════════════════════════════════
# ROW 1: KPIs
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">TỔNG QUAN THỊ TRƯỜNG</div>', unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)

with k1:
    st.metric(
        "🟢 Tăng",
        f"{ad['advances']:,}",
        f"{ad['pct_advance']:.1f}%",
    )
with k2:
    st.metric(
        "🔴 Giảm",
        f"{ad['declines']:,}",
        f"{100 - ad['pct_advance'] - (ad['unchanged']/ad['total']*100 if ad['total'] else 0):.1f}%",
    )
with k3:
    st.metric(
        "⚪ Đứng giá",
        f"{ad['unchanged']:,}",
    )
with k4:
    st.metric(
        "📊 Tổng mã",
        f"{ad['total']:,}",
    )
with k5:
    st.metric(
        "⬆️ 52W High",
        f"{hl['new_highs']:,}",
        delta=f"vs {hl['new_lows']} lows",
    )
with k6:
    vol_mom = s['vol_momentum']
    st.metric(
        "💧 Volume Mom",
        f"{vol_mom:.2f}x",
        delta="↑ tăng" if vol_mom > 1.0 else "↓ giảm",
        delta_color="normal" if vol_mom > 1.0 else "inverse",
    )


# ══════════════════════════════════════════════
# ROW 2: Sentiment + A/D Donut + AD Line
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">SENTIMENT & ADVANCE/DECLINE</div>', unsafe_allow_html=True)

col_gauge, col_donut, col_adline = st.columns([1, 1, 2])

with col_gauge:
    st.markdown(f"""
    <div style="text-align:center; padding: 8px 0 4px;">
        <span style="font-size:0.75rem; color:#888; text-transform:uppercase; letter-spacing:1px;">Fear/Greed Index</span>
    </div>
    """, unsafe_allow_html=True)
    fig_gauge = charts.sentiment_gauge(
        sentiment['score'], sentiment['label'], sentiment['color']
    )
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    # Sentiment History link
    fig_sent_hist = charts.sentiment_history_chart(s['sentiment_history'])
    st.plotly_chart(fig_sent_hist, use_container_width=True)

with col_donut:
    fig_donut = charts.advance_decline_donut(ad)
    st.plotly_chart(fig_donut, use_container_width=True)

with col_adline:
    fig_adline = charts.ad_line_chart(s['ad_history'])
    st.plotly_chart(fig_adline, use_container_width=True)


# ══════════════════════════════════════════════
# ROW 3: MA Analysis + Distribution
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">PHÂN TÍCH MOVING AVERAGE</div>', unsafe_allow_html=True)

col_ma, col_dist = st.columns([1, 1])

with col_ma:
    fig_ma = charts.ma_above_bar(s['ma_stats'])
    st.plotly_chart(fig_ma, use_container_width=True)

    # MA table
    ma_rows = []
    for p, data in sorted(s['ma_stats'].items()):
        ma_rows.append({
            'Period': f'MA{p}',
            'Số lượng (Trên/Dưới)': f"{data['above']} / {data['below']}",
            '% Trên MA': f"{data['pct_above']}%",
            'Tổng mã đạt chuẩn': data['total'],
        })
    if ma_rows:
        st.dataframe(
            pd.DataFrame(ma_rows),
            hide_index=True,
            use_container_width=True,
        )

with col_dist:
    fig_dist = charts.change_distribution_chart(s['dist_data'])
    st.plotly_chart(fig_dist, use_container_width=True)

# Historical Breadth
st.markdown('<div class="section-header">DIỄN BIẾN ĐỘ RỘNG THỊ TRƯỜNG</div>', unsafe_allow_html=True)
fig_ma_hist = charts.ma_breadth_history_chart(s['ma_history'])
st.plotly_chart(fig_ma_hist, use_container_width=True)

# Historical Table Summary
if not s['ma_history'].empty:
    with st.expander("📊 Bảng thống kê độ rộng lịch sử", expanded=True):
        df_hist = s['ma_history'].copy().sort_values('date', ascending=False)
        
        # Rename columns for display
        cols_map = {'date': 'Ngày', 'VNINDEX': 'VNINDEX'}
        for col in df_hist.columns:
            if col.startswith('count_ma'):
                p = col.replace('count_ma', '')
                cols_map[col] = f'MA{p} (Mã)'
            elif col.startswith('pct_ma'):
                p = col.replace('pct_ma', '')
                cols_map[col] = f'MA{p} (%)'
        
        df_display = df_hist[list(cols_map.keys())].rename(columns=cols_map)
        
        # Apply Styling
        def style_vnindex(val, prev_val):
            if pd.isna(prev_val) or val == prev_val: return ""
            color = "#00C853" if val > prev_val else "#FF1744"
            return f"color: {color}; font-weight: bold"

        def style_ma_pct(val):
            if val >= 80: color = "rgba(255, 23, 68, 0.3)" # Overbought
            elif val <= 20: color = "rgba(0, 230, 118, 0.3)" # Oversold
            elif val > 55: color = "rgba(105, 240, 174, 0.1)"
            else: color = "transparent"
            return f"background-color: {color}"

        # Prepare styler
        def make_styler(df_to_style):
            # Safe copy
            df = df_to_style.copy()
            df = df.sort_values('Ngày') # Sort for diff calculation
            
            # Use 'VNINDEX' if exists
            use_vni = 'VNINDEX' in df.columns
            if use_vni:
                df['VNI_Prev'] = df['VNINDEX'].shift(1)
            
            def vni_color(row):
                colors = [''] * len(row)
                if use_vni and 'VNI_Prev' in row.index and not pd.isna(row['VNI_Prev']):
                    vni_idx = row.index.get_loc('VNINDEX')
                    if row['VNINDEX'] > row['VNI_Prev']: colors[vni_idx] = 'color: #00E676; font-weight: bold'
                    elif row['VNINDEX'] < row['VNI_Prev']: colors[vni_idx] = 'color: #FF1744; font-weight: bold'
                return colors

            def ma_color(val):
                if isinstance(val, (int, float)):
                    if val >= 80: return 'background-color: rgba(255, 23, 68, 0.4); color: white'
                    if val >= 55: return 'background-color: rgba(0, 200, 83, 0.3); color: white'
                    if val <= 20: return 'background-color: rgba(170, 0, 255, 0.3); color: white'
                return ''

            styler = df.sort_values('Ngày', ascending=False).style
            styler = styler.apply(vni_color, axis=1)
            
            if use_vni:
                styler = styler.hide(subset=['VNI_Prev'], axis="columns")
            
            # Apply background to pct columns
            pct_cols = [c for c in df_display.columns if '%' in c]
            styler = styler.applymap(ma_color, subset=pct_cols)
            
            # Add % to pct columns
            format_dict = {c: "{:.1f}%" for c in pct_cols}
            styler = styler.format(format_dict)
            return styler

        st.dataframe(
            make_styler(df_display),
            hide_index=True,
            use_container_width=True
        )


# ══════════════════════════════════════════════
# ROW 4: Liquidity
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">THANH KHOẢN THỊ TRƯỜNG</div>', unsafe_allow_html=True)

fig_liq = charts.liquidity_chart(s['liq_history'])
st.plotly_chart(fig_liq, use_container_width=True)


# ══════════════════════════════════════════════
# ROW 5: Market Power & VNINDEX
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">MARKET SENTIMENT ANALYSIS (SUPPLY/DEMAND/POWER)</div>', unsafe_allow_html=True)

col_power, col_vni = st.columns([1, 1])

with col_power:
    fig_power = charts.market_power_chart(s['power_hist'])
    st.plotly_chart(fig_power, use_container_width=True)

with col_vni:
    fig_vni = charts.vnindex_chart(s['prices_raw'], lookback_days)
    st.plotly_chart(fig_vni, use_container_width=True)


# ══════════════════════════════════════════════
# ROW 6: Advanced Analytics (RSI, PSY, Thrust)
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">ADVANCED MARKET BREADTH ANALYTICS (MOMENTUM & PSYCHOLOGICAL)</div>', unsafe_allow_html=True)

col_mom_psy, col_thrust = st.columns([1, 1])

with col_mom_psy:
    # Mặc định lấy RSI/PSY của MA20 (hoặc MA đầu tiên trong list)
    p_ref = ma_periods[0]
    fig_mom_psy = charts.rsi_psy_breadth_chart(s['adv_analytics'], period=p_ref)
    st.plotly_chart(fig_mom_psy, use_container_width=True)

with col_thrust:
    fig_thrust = charts.breadth_thrust_chart(s['ad_history'])
    st.plotly_chart(fig_thrust, use_container_width=True)


# ══════════════════════════════════════════════
# ROW 7: New High/Low + McClellan + Sentiment Radar
# ══════════════════════════════════════════════
st.markdown('<div class="section-header">BREADTH INDICATORS</div>', unsafe_allow_html=True)

col_hl, col_mc, col_radar = st.columns([1, 2, 1])

with col_hl:
    fig_hl = charts.new_high_low_chart(s['hl_stats'])
    st.plotly_chart(fig_hl, use_container_width=True)

    st.markdown(f"""
    <div class="info-box">
        <div style="font-size:0.75rem; color:#888">HỆ SỐ H/L</div>
        <div style="font-size:1.5rem; font-weight:600; color:#00B4D8">{hl['hl_ratio']:.2f}</div>
        <div style="font-size:0.8rem; color:#888">
            {hl['new_highs']} highs · {hl['new_lows']} lows<br>
            trên {hl['total']} mã phân tích
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_mc:
    fig_mc = charts.mcclellan_chart(s['mc_df'])
    st.plotly_chart(fig_mc, use_container_width=True)

with col_radar:
    fig_radar = charts.sentiment_components_chart(sentiment['components'])
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════
# Footer
# ══════════════════════════════════════════════
st.markdown("---")
st.markdown(f"""
<div style="text-align:center; color:#444; font-size:0.75rem; padding: 8px;">
    VN Market Dashboard  · 
    Cập nhật: {datetime.now().strftime('%d/%m/%Y %H:%M')} · 
    Sàn: <b style="color:#00B4D8">{st.session_state.get('exchange', exchange)}</b> · 
    Mã phân tích: <b style="color:#00B4D8">{s['total_tickers']:,}</b>
</div>
""", unsafe_allow_html=True)
