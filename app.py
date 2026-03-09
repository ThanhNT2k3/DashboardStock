"""
🇻🇳 Vietnam Stock Market Dashboard
Phân tích độ rộng & sentiment thị trường chứng khoán Việt Nam

"""

import streamlit as st
import pandas as pd
from datetime import datetime, date, timedelta
import sys, os

sys.path.insert(0, os.path.dirname(__file__))

from modules import tickers as tk
from modules import fetcher, calculator, charts, notifier, keep_alive

# Start Keep-Alive thread (Stay Awake)
keep_alive.start_keep_alive()

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
# Custom CSS — Premium Trading Terminal
# ─────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* ── Base ─────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }
    .main, .stApp {
        background: #080B12 !important;
    }

    /* ── Sidebar ──────────────────────────── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D1117 0%, #0A0E18 100%) !important;
        border-right: 1px solid rgba(0, 180, 216, 0.12) !important;
    }
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #00B4D8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 2px;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }

    /* ── Tabs ─────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(15,18,30,0.9) !important;
        border-bottom: 1px solid rgba(0,180,216,0.15);
        padding: 0 8px;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        font-size: 0.82rem;
        color: #666 !important;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
        border: none !important;
        background: transparent !important;
    }
    .stTabs [aria-selected="true"] {
        color: #00D4FF !important;
        background: rgba(0,180,216,0.08) !important;
        border-bottom: 2px solid #00D4FF !important;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #00D4FF !important;
        background: rgba(0,180,216,0.05) !important;
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: transparent !important;
        padding-top: 1rem;
    }

    /* ── Metric Cards ─────────────────────── */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #111520 0%, #0F1420 100%);
        border: 1px solid rgba(42,45,62,0.8);
        border-top: 2px solid rgba(0,180,216,0.3);
        border-radius: 10px;
        padding: 16px 20px;
        transition: border-color 0.2s, transform 0.15s;
        position: relative;
        overflow: hidden;
    }
    [data-testid="metric-container"]::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,180,216,0.5), transparent);
    }
    [data-testid="metric-container"]:hover {
        border-color: rgba(0,212,255,0.35);
        transform: translateY(-1px);
    }
    [data-testid="metric-container"] > div { color: #FAFAFA !important; }
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 1.5rem !important;
        font-weight: 600 !important;
        letter-spacing: -0.5px;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.7rem !important;
        font-weight: 500 !important;
        text-transform: uppercase !important;
        letter-spacing: 1.2px !important;
        color: #5E7291 !important;
    }
    [data-testid="stMetricDelta"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 0.8rem !important;
    }

    /* ── Header ───────────────────────────── */
    .dashboard-header {
        background: linear-gradient(135deg, 
            rgba(0,180,216,0.04) 0%, 
            rgba(0,212,255,0.02) 50%, 
            transparent 100%);
        border-bottom: 1px solid rgba(0,180,216,0.12);
        border-radius: 0 0 12px 12px;
        padding: 1.2rem 0 1rem;
        margin-bottom: 1.5rem;
        position: relative;
    }
    .dashboard-header::after {
        content: '';
        position: absolute;
        bottom: -1px; left: 0; right: 0;
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(0,180,216,0.5) 50%, transparent);
    }
    .dashboard-title {
        font-size: 1.7rem;
        font-weight: 700;
        background: linear-gradient(135deg, #00D4FF 0%, #0090B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        letter-spacing: -0.8px;
    }
    .dashboard-subtitle {
        font-size: 0.8rem;
        color: #4A5568;
        margin-top: 4px;
        letter-spacing: 0.2px;
    }
    .live-badge {
        display: inline-flex;
        align-items: center;
        gap: 5px;
        background: rgba(0,230,118,0.08);
        border: 1px solid rgba(0,230,118,0.25);
        color: #00E676;
        font-size: 0.65rem;
        font-weight: 600;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        padding: 3px 10px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        margin-left: 10px;
    }
    .live-badge::before {
        content: '';
        width: 5px;
        height: 5px;
        background: #00E676;
        border-radius: 50%;
        animation: blink 1.5s infinite;
    }
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.2; }
    }

    /* ── Section Headers ──────────────────── */
    .section-header {
        font-size: 0.68rem;
        font-weight: 600;
        color: #4A6080;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin: 1.8rem 0 0.8rem;
        padding-left: 10px;
        border-left: 3px solid #00B4D8;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* ── Info Cards ───────────────────────── */
    .info-box {
        background: linear-gradient(135deg, #111520, #0F1420);
        border: 1px solid #1E2538;
        border-radius: 10px;
        padding: 14px 18px;
        margin: 8px 0;
        position: relative;
        overflow: hidden;
    }
    .info-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: linear-gradient(180deg, #00B4D8, transparent);
        border-radius: 3px;
    }

    /* ── Status Badges ────────────────────── */
    .status-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }
    .status-up   { background: rgba(0,230,118,0.12); color: #00E676; border: 1px solid rgba(0,230,118,0.3); }
    .status-down { background: rgba(255,23,68,0.12);  color: #FF1744; border: 1px solid rgba(255,23,68,0.3); }
    .status-neu  { background: rgba(255,215,64,0.12); color: #FFD740; border: 1px solid rgba(255,215,64,0.3); }

    /* ── Buttons ──────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #005F73, #0077A8) !important;
        border: 1px solid rgba(0,180,216,0.4) !important;
        border-radius: 8px !important;
        color: #E0F7FF !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        letter-spacing: 0.3px !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #0077A8, #009CC5) !important;
        border-color: rgba(0,212,255,0.6) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 15px rgba(0,180,216,0.2) !important;
    }
    .stButton > button {
        background: rgba(20,25,40,0.8) !important;
        border: 1px solid rgba(60,70,100,0.6) !important;
        border-radius: 8px !important;
        color: #8BA0BC !important;
        font-weight: 500 !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        border-color: rgba(0,180,216,0.4) !important;
        color: #00D4FF !important;
    }

    /* ── Selectbox & Inputs ───────────────── */
    .stSelectbox [data-baseweb="select"] > div {
        background: #0F1420 !important;
        border-color: #1E2538 !important;
        border-radius: 8px !important;
        color: #C8D8E8 !important;
    }
    .stNumberInput > div > div > input,
    .stTextInput > div > div > input {
        background: #0F1420 !important;
        border: 1px solid #1E2538 !important;
        border-radius: 8px !important;
        color: #C8D8E8 !important;
        font-family: 'JetBrains Mono', monospace !important;
    }
    .stNumberInput > div > div > input:focus,
    .stTextInput > div > div > input:focus {
        border-color: rgba(0,180,216,0.5) !important;
        box-shadow: 0 0 0 2px rgba(0,180,216,0.1) !important;
    }

    /* ── Slider ───────────────────────────── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00B4D8, #0077A8) !important;
    }

    /* ── Dataframe ────────────────────────── */
    .stDataFrame {
        border: 1px solid #1E2538 !important;
        border-radius: 10px !important;
        overflow: hidden;
    }
    .stDataFrame thead th {
        background: #0A0E1A !important;
        color: #4A6080 !important;
        font-size: 0.7rem !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        font-weight: 600 !important;
    }

    /* ── Expander ─────────────────────────── */
    .streamlit-expanderHeader {
        background: #0F1420 !important;
        border: 1px solid #1E2538 !important;
        border-radius: 8px !important;
        color: #8BA0BC !important;
        font-size: 0.85rem !important;
    }

    /* ── Spinner ──────────────────────────── */
    .stSpinner > div {
        border-top-color: #00B4D8 !important;
    }

    /* ── Alert boxes ──────────────────────── */
    .stAlert {
        border-radius: 8px !important;
        border: none !important;
    }
    .stSuccess { background: rgba(0,230,118,0.08) !important; border-left: 3px solid #00E676 !important; }
    .stInfo    { background: rgba(0,180,216,0.08) !important; border-left: 3px solid #00B4D8 !important; }
    .stWarning { background: rgba(255,215,64,0.08) !important; border-left: 3px solid #FFD740 !important; }
    .stError   { background: rgba(255,23,68,0.08) !important; border-left: 3px solid #FF1744 !important; }

    /* ── Divider ──────────────────────────── */
    hr {
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, rgba(0,180,216,0.2) 50%, transparent) !important;
        margin: 1rem 0 !important;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.markdown(f"""
<div class="dashboard-header">
    <div style="display:flex; align-items:center; gap:0;">
        <span class="dashboard-title">⟨ VN Market Terminal ⟩</span>
        <span class="live-badge">LIVE</span>
    </div>
    <div class="dashboard-subtitle">
        Vietnam Stock Market Intelligence &nbsp;·&nbsp; Breadth · Sentiment · AI-Powered Analysis
        &nbsp;&nbsp;<span style="color:#2A3D55; font-family:'JetBrains Mono',monospace;">v3.1</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Sidebar Filters
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;padding-bottom:12px;border-bottom:1px solid rgba(0,180,216,0.12);">
        <span style="font-size:1.2rem;">📊</span>
        <span style="font-weight:700;font-size:0.95rem;color:#E0F0FF;">VN Terminal</span>
        <span style="font-size:0.65rem;color:#4A6080;font-family:JetBrains Mono,monospace;">v3.0</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("### ⚙️ Configuration")

    exchange = st.selectbox(
        "Exchange Selection",
        options=["HOSE", "HNX", "UPCOM", "VN30", "VN100", "ALL"],
        index=0,
        help="Select exchange or index bucket to analyze"
    )

    st.markdown("---")

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        start_date = st.date_input(
            "Start Date",
            value=date.today() - timedelta(days=720),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )
    with col_d2:
        end_date = st.date_input(
            "End Date",
            value=date.today(),
            min_value=date(2020, 1, 1),
            max_value=date.today(),
        )

    st.markdown("---")

    ma_periods = st.multiselect(
        "MA Periods",
        options=[10, 20, 50],
        default=[10, 20, 50],
        help="Select MA periods to analyze"
    )
    if not ma_periods:
        ma_periods = [10, 20, 50]

    st.markdown("---")

    lookback_days = st.slider(
        "Historical Lookback (Days)",
        min_value=20,
        max_value=200,
        value=60,
        step=10,
    )

    st.markdown("---")

    min_liq = st.sidebar.slider(
        "Min Liquidity (Bn VND/Session)",
        min_value=0.0,
        max_value=100.0,
        value=10.0,
        step=1.0,
        help="Filter stocks based on turnover"
    )

    liq_mode = st.sidebar.selectbox(
        "Liquidity Metric",
        options=["Mean (Average)", "Median", "Min", "Last Session"],
        index=0
    )
    liq_mode_map = {
        "Mean (Average)": "mean", 
        "Median": "median", 
        "Min": "min", 
        "Last Session": "last"
    }

    st.markdown("---")

    run_btn = st.button("🔄 Run Analysis", use_container_width=True, type="primary")

    st.markdown("---")
    st.markdown("### 🧪 Stochastic Backtest")
    bt_ticker = st.text_input("Symbol", value="SSI", help="Enter symbol to run Stochastic crossover backtest")
    
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        bt_k = st.number_input("%K Period", value=14, min_value=1)
        bt_oversold = st.number_input("Oversold", value=20, min_value=1)
    with col_bt2:
        bt_d = st.number_input("%D Period", value=3, min_value=1)
        bt_overbought = st.number_input("Overbought", value=80, max_value=100)
        
    bt_run = st.button("🚀 Run Backtest", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 Stock Signal Scanner")
    scanner_run = st.button("📊 Scan VN30 Signals", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 AI Multi-Factor Analysis")
    ai_ticker = st.text_input("Symbol (AI)", value="SSI", key="ai_ticker_input", help="AI Multi-factor analysis combined with Macro data")
    ai_split = st.slider("Train/Test Split", 0.5, 0.9, 0.7, 0.1)
    ai_run = st.button("🤖 Run AI Analysis", use_container_width=True)
    ai_scanner_run = st.button("🔍 AI VN100 Scanner", use_container_width=True, help="Scan VN100 bucket using AI multi-factor models")

    st.markdown("---")
    ticker_list = tk.get_tickers(exchange)
    st.markdown(f"""
    <div class="info-box">
        <div style="color:#888;font-size:0.75rem">INFORMATION</div>
        <div style="margin-top:6px">
            <b style="color:#00B4D8">{exchange}</b><br>
            <span style="color:#FAFAFA;font-size:0.9rem">{len(ticker_list):,} tickers</span><br>
            <span style="color:#888;font-size:0.75rem">{start_date} → {end_date}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Data Loading with Cache
# ─────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def load_market_data(exchange: str, start: str, end: str, custom_tickers: list = None) -> dict:
    """Fetch & process toàn bộ dữ liệu thị trường"""
    if custom_tickers:
        ticker_list = custom_tickers
    else:
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


def compute_all_stats(prices_dict: dict, ma_periods: list, lookback: int, agg_liq_map: dict = {}) -> dict:
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
    sector_liq   = calculator.compute_sector_liquidity(prices_dict, agg_liq_map)
    
    # Advanced Analytics
    adv_analytics = calculator.compute_advanced_analytics(ma_history)
    ad_history['thrust'] = calculator.compute_breadth_thrust(ad_history)
    influence = calculator.compute_index_influence(prices_dict)
    
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
        'sector_liq':   sector_liq,
        'index_influence': influence,
    }


# ─────────────────────────────────────────────
# Session State init
# ─────────────────────────────────────────────
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'stats' not in st.session_state:
    st.session_state.stats = None
if 'bt_results' not in st.session_state:
    st.session_state.bt_results = None
if 'bt_target' not in st.session_state:
    st.session_state.bt_target = ""
if 'agg_liq_map' not in st.session_state:
    st.session_state.agg_liq_map = fetcher.load_liquidity_cache()
if 'agg_results' not in st.session_state:
    st.session_state.agg_results = {}
if 'last_discord_send' not in st.session_state:
    st.session_state.last_discord_send = None
if 'last_market_date' not in st.session_state:
    st.session_state.last_market_date = None


# ─────────────────────────────────────────────
# Cache Liquidity Refresh
# ─────────────────────────────────────────────
refresh_liq = st.sidebar.button("🔄 Refresh Liquidity Cache", help="Fetch real aggressive trading data from VPS to apply accurate filters")

if refresh_liq:
    with st.spinner("⏳ Scanning real market liquidity..."):
        all_tk = tk.get_tickers("ALL")
        agg_res = fetcher.batch_fetch_aggressive(all_tk)
        st.session_state.agg_results = agg_res
        liq_map = calculator.compute_liquidity_map(agg_res)
        if liq_map:
            fetcher.save_liquidity_cache(liq_map)
            st.session_state.agg_liq_map = liq_map
            st.success("✅ Update successful!")
        else:
            st.sidebar.error("❌ Failed to fetch liquidity data")
            
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🔔 NOTIFICATION SYSTEM")
    
    # Auto Discord logic at 15:00
    now = datetime.now()
    today_str = now.strftime('%Y-%m-%d')
    can_notify = st.session_state.get('data_loaded') and st.session_state.stats is not None
    
    if now.hour >= 15:
        if st.session_state.last_discord_send != today_str:
            if can_notify:
                if st.sidebar.button("📤 Close Session & Send to Discord", type="primary", use_container_width=True):
                    with st.sidebar:
                        with st.spinner("🚀 Sending summary to Discord..."):
                            success = notifier.send_market_summary_sync(st.session_state.stats)
                            if success:
                                st.session_state.last_discord_send = today_str
                                st.success("✅ Discord notification sent successfully!")
                            else:
                                st.error("❌ Failed to send Discord notification. Check Webhook!")
            else:
                st.sidebar.warning("⚠️ Please click 'Run Analysis' before sending notification.")
        else:
            st.sidebar.success(f"✅ Report sent for {today_str}")
    else:
        st.sidebar.info("⏰ Wait until 15:00 to close session and send report.")


# ─────────────────────────────────────────────
# Trigger Backtest (Independent)
# ─────────────────────────────────────────────
if bt_run:
    with st.spinner(f"🧪 Running backtest for {bt_ticker}..."):
        try:
            # Load specific data for backtest (extra lookback for MA/Stoch stability)
            bt_start = (start_date - timedelta(days=200)).strftime('%Y-%m-%d')
            raw_bt = fetcher.batch_fetch([bt_ticker], bt_start, end_date.strftime('%Y-%m-%d'))
            dict_bt = fetcher.parse_results(raw_bt)
            
            if bt_ticker in dict_bt:
                t_data = dict_bt[bt_ticker]
                df_bt = pd.DataFrame({
                    'close': t_data['close'],
                    'high': t_data['high'],
                    'low': t_data['low'],
                    'volume': t_data['volume']
                }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
                
                results = calculator.run_backtest_stochastic(df_bt, bt_k, bt_d, bt_oversold, bt_overbought)
                st.session_state.bt_results = results
                st.session_state.bt_target = bt_ticker
            else:
                st.error(f"❌ Symbol {bt_ticker} not found")
        except Exception as e:
            st.error(f"❌ Backtest error: {e}")


# ─────────────────────────────────────────────
# Trigger Scanner (VN30 Batch)
# ─────────────────────────────────────────────
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = None

if scanner_run:
    with st.spinner("🔍 Scanning VN100 for Stochastic signals..."):
        try:
            # Use VN100 as implied by label
            vn_list = tk.VN100
            scanner_start = (date.today() - timedelta(days=200)).strftime('%Y-%m-%d')
            raw_scan = fetcher.batch_fetch(vn_list, scanner_start, date.today().strftime('%Y-%m-%d'))
            dict_scan = fetcher.parse_results(raw_scan)
            
            # Apply liquidity filter
            dict_scan = calculator.filter_by_liquidity(dict_scan, min_liq, mode=liq_mode_map[liq_mode])
            
            scan_rows = []
            for ticker in dict_scan.keys():
                t_data = dict_scan[ticker]
                df_t = pd.DataFrame({
                    'close': t_data['close'],
                    'high': t_data['high'],
                    'low': t_data['low']
                }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
                    
                # 1. Run historical backtest
                bt = calculator.run_backtest_stochastic(df_t)
                
                # 2. Get current state
                df_stoch = calculator.compute_stochastic(df_t)
                if not df_stoch.empty:
                    last_k = df_stoch.iloc[-1]['%K']
                    last_d = df_stoch.iloc[-1]['%D']
                    prev_k = df_stoch.iloc[-2]['%K']
                    prev_d = df_stoch.iloc[-2]['%D']
                    
                    signal = "Neutral"
                    if prev_k <= prev_d and last_k > last_d:
                        signal = "BUY (Cross Up)" if last_k < 30 else "Potential Up"
                    elif prev_k >= prev_d and last_k < last_d:
                        signal = "SELL (Cross Down)" if last_k > 70 else "Potential Down"

                    wr_val = float(bt.get('win_rate', 0))
                    total_ret_val = float(bt.get('total_return', 0))
                    
                    recommendation = "Hold"
                    if signal == "BUY (Cross Up)":
                        recommendation = "STRONG BUY 🔥" if wr_val > 55 else "BUY ✅"
                    elif signal == "SELL (Cross Down)":
                        recommendation = "STRONG SELL ⚠️" if wr_val > 55 else "SELL 🔻"
                    elif signal == "Potential Up":
                        recommendation = "Watch BUY 👀"
                    elif signal == "Potential Down":
                        recommendation = "Watch SELL 📉"

                    scan_rows.append({
                        'Symbol': ticker,
                        'Price': f"{t_data['close'][-1]:,.2f}",
                        '%K': round(last_k, 1),
                        '%D': round(last_d, 1),
                        'Signal': signal,
                        'Recommendation': recommendation,
                        'Win Rate': f"{wr_val}%",
                        'Total Return': f"{total_ret_val}%",
                        'Trades': bt.get('total_trades', 0)
                    })

            
            st.session_state.scanner_results = pd.DataFrame(scan_rows)
        except Exception as e:
            st.error(f"❌ VN100 Scan Error: {e}")


# ─────────────────────────────────────────────
# Trigger AI Backtest
# ─────────────────────────────────────────────
if 'ai_results' not in st.session_state:
    st.session_state.ai_results = None

if 'ai_run' in locals() and ai_run:
    with st.spinner(f"🤖 Training AI for {ai_ticker}..."):
        try:
            # Fetch long history (need at least 2 years for AI model)
            ai_start = (date.today() - timedelta(days=730)).strftime('%Y-%m-%d')
            ai_end_s = end_date.strftime('%Y-%m-%d')
            
            raw_ai = fetcher.batch_fetch([ai_ticker], ai_start, ai_end_s)
            dict_ai = fetcher.parse_results(raw_ai)
            
            if ai_ticker in dict_ai:
                t_data = dict_ai[ai_ticker]
                df_ticker_raw = pd.DataFrame({
                    'close': t_data['close'],
                    'open': t_data['open'],
                    'high': t_data['high'],
                    'low': t_data['low'],
                    'volume': t_data['volume']
                }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
                
                # Run AI Backtest
                ai_res = calculator.run_backtest_ai(ai_ticker, df_ticker_raw, ai_start, ai_end_s, ai_split)
                st.session_state.ai_results = ai_res
                st.session_state.ai_target = ai_ticker
            else:
                st.error(f"❌ No data found for symbol {ai_ticker}")
        except Exception as e:
            st.error(f"❌ AI Engine Error: {e}")


# ─────────────────────────────────────────────
# Trigger AI Scanner (VN30)
# ─────────────────────────────────────────────
if 'ai_scan_results' not in st.session_state:
    st.session_state.ai_scan_results = None

if 'ai_scanner_run' in locals() and ai_scanner_run:
    with st.spinner("🤖 Scanning VN100 with AI (Technical + Macro + Foreign)..."):
        try:
            vn30_list = tk.VN100
            ai_start = (date.today() - timedelta(days=730)).strftime('%Y-%m-%d')
            ai_end_s = end_date.strftime('%Y-%m-%d')
            
            # Fetch data for all VN100
            raw_ai_scan = fetcher.batch_fetch(vn30_list, ai_start, ai_end_s)
            dict_ai_scan = fetcher.parse_results(raw_ai_scan)
            
            # Apply liquidity filter
            dict_ai_scan = calculator.filter_by_liquidity(dict_ai_scan, min_liq, mode=liq_mode_map[liq_mode])
            
            # Fetch Macro & Foreign Flow data (Optimized)
            from modules.ai_engine import AIEngine
            engine = AIEngine()
            macro_df = engine.fetch_macro_data(ai_start, ai_end_s)
            foreign_cache = engine.fetch_foreign_flow_batch(vn30_list, ai_start, ai_end_s)
            
            ai_scan_rows = []
            for ticker in vn30_list:
                if ticker in dict_ai_scan:
                    t_data = dict_ai_scan[ticker]
                    df_t = pd.DataFrame({
                        'close': t_data['close'],
                        'open': t_data['open'],
                        'high': t_data['high'],
                        'low': t_data['low'],
                        'volume': t_data['volume']
                    }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
                    
                    foreign_df = foreign_cache.get(ticker, pd.DataFrame())
                    full_df = engine.prepare_features(df_t, macro_df, foreign_df)
                    
                    if not full_df.empty:
                        # Quick train for this ticker
                        engine.train(full_df)
                        signal = engine.predict(full_df)

                        # Win/Loss logic (historical backtest on last 25%)

                        # Tính tỉ lệ Win/Loss lịch sử (backtest nhanh trên 25% dữ liệu cuối)
                        win_rate_pct, win_count, loss_count = None, 0, 0
                        test_size = max(20, int(len(full_df) * 0.25))
                        test_df = full_df.tail(test_size)
                        pos, entry_p, entry_d = None, 0.0, None
                        for i in range(len(test_df)):
                            row = test_df.iloc[i:i+1]
                            pred = engine.predict(row)
                            price = row['close'].values[0]
                            dt = test_df.index[i]
                            days_held = (dt - entry_d).days if entry_d is not None else 0
                            if pos is None and pred == 2:
                                pos, entry_p, entry_d = 'long', price, dt
                            elif pos == 'long' and (pred == 0 or days_held >= 10):
                                pnl = (price - entry_p) / entry_p
                                if pnl > 0: win_count += 1
                                else: loss_count += 1
                                pos = None
                        if pos == 'long':  # Close any open trade at the end
                            pnl = (test_df['close'].iloc[-1] - entry_p) / entry_p
                            if pnl > 0: win_count += 1
                            else: loss_count += 1
                        total = win_count + loss_count
                        if total > 0:
                            win_rate_pct = round(win_count / total * 100, 1)
                        
                        # Metadata for display
                        last_close = t_data['close'][-1]
                        change = t_data['change_pct']
                        
                        # Predicted Return: average TARGET_RET of last 20 sessions (last row is usually NaN)
                        valid_ret = full_df['TARGET_RET'].dropna()
                        pred_ret = valid_ret.tail(20).mean() * 100 if len(valid_ret) > 0 else 0.0
                        
                        # Only consider BUY if expected return is sufficient
                        MIN_STRONG_RET = 3.0  # 3% for next 5 sessions
                        if signal == 2 and pred_ret < MIN_STRONG_RET:
                            signal = 1  # Downgrade to HOLD if signal strength is weak
                        
                        label = "BUY 🚀" if signal == 2 else ("SELL ⚠️" if signal == 0 else "HOLD ⏳")
                        
                        # Collect conditions met (from last row of full_df)
                        last_row = full_df.iloc[-1]
                        prev_row = full_df.iloc[-2] if len(full_df) > 1 else None
                        conditions = []
                        if signal == 2:
                            conditions.append("✓ AI Signal: BUY")
                        elif signal == 0:
                            conditions.append("⚠ AI Signal: SELL")
                        else:
                            conditions.append("○ AI Signal: HOLD")
                        if pred_ret >= 3.0:
                            conditions.append(f"✓ Predicted Return ≥ 3% ({pred_ret:+.1f}%)")
                        elif pred_ret > 0:
                            conditions.append(f"✓ Predicted Return Positive ({pred_ret:+.1f}%)")
                        elif pred_ret < -3:
                            conditions.append(f"⚠ Predicted Return Negative ({pred_ret:+.1f}%)")
                        rsi = last_row.get('RSI', 50)
                        if rsi < 30:
                            conditions.append(f"✓ RSI Oversold ({rsi:.0f}) - Buying Opportunity")
                        elif rsi >= 70:
                            conditions.append(f"⚠ RSI Overbought ({rsi:.0f})")
                        elif rsi < 70:
                            conditions.append(f"✓ RSI Not Overbought ({rsi:.0f})")
                        ma20 = last_row.get('MA20')
                        if ma20 and last_close > ma20:
                            conditions.append("✓ Price above MA20 (Momentum)")
                        elif ma20:
                            conditions.append("○ Price below MA20")
                        ma50 = last_row.get('MA50')
                        if ma50 and last_close > ma50:
                            conditions.append("✓ Price above MA50 (Trend)")
                        elif ma50:
                            conditions.append("○ Price below MA50")
                        ff = last_row.get('foreignNetValue', 0)
                        if ff > 0:
                            conditions.append("✓ Foreign Net Buying")
                        elif ff < 0:
                            conditions.append("○ Foreign Net Selling")
                        ret1d = last_row.get('RETURNS_1D', 0)
                        if ret1d is not None and ret1d > 0:
                            conditions.append("✓ Last session Price Increased")
                        elif ret1d is not None and ret1d < 0:
                            conditions.append("○ Last session Price Decreased")

                        # Điều kiện vĩ mô & giá hàng hóa toàn cầu (so với phiên trước)
                        if prev_row is not None:
                            def _macro_change(col_name: str) -> float | None:
                                if col_name not in last_row or col_name not in prev_row:
                                    return None
                                cur = last_row.get(col_name)
                                prev = prev_row.get(col_name)
                                if cur is None or prev is None or prev == 0:
                                    return None
                                try:
                                    return (cur / prev - 1.0) * 100.0
                                except Exception:
                                    return None

                            gold_chg = _macro_change('GOLD')
                            oil_chg = _macro_change('OIL')
                            dxy_chg = _macro_change('DXY')
                            us10y_chg = _macro_change('US10Y')

                            if gold_chg is not None and abs(gold_chg) >= 1.0:
                                direction = "up" if gold_chg > 0 else "down"
                                conditions.append(f"○ Gold price {direction} approx {gold_chg:+.1f}% today")

                            if oil_chg is not None and abs(oil_chg) >= 1.0:
                                direction = "up" if oil_chg > 0 else "down"
                                conditions.append(f"○ Crude Oil {direction} approx {oil_chg:+.1f}% today")

                            if dxy_chg is not None and abs(dxy_chg) >= 0.5:
                                direction = "up" if dxy_chg > 0 else "down"
                                conditions.append(f"○ USD Index (DXY) {direction} approx {dxy_chg:+.1f}%")

                            if us10y_chg is not None and abs(us10y_chg) >= 0.5:
                                direction = "up" if us10y_chg > 0 else "down"
                                conditions.append(f"○ US 10Y Treasury Yield {direction} approx {us10y_chg:+.1f} bps relative")

                        if len(conditions) <= 1:
                            conditions.append("— No significant conditions")

                        # Điểm chất lượng tín hiệu: dựa trên số điều kiện tích cực / tiêu cực
                        positives = sum(1 for c in conditions if c.startswith("✓"))
                        negatives = sum(1 for c in conditions if c.startswith("⚠"))
                        raw_score = positives * 15 - negatives * 10
                        quality_score = max(0, min(100, raw_score))

                        conditions_str = "\n".join(conditions)
                        
                        # Recommended Buy Price (0.5% below current for optimization)
                        buy_price = last_close * 0.995 if signal == 2 else None
                        # Default Stop Loss & Take Profit (-3%, +8%) - only for BUY signals
                        stop_loss_price = buy_price * 0.97 if buy_price else None
                        take_profit_price = buy_price * 1.08 if buy_price else None
                        
                        win_loss_str = f"{win_rate_pct}% ({win_count}W/{loss_count}L)" if win_rate_pct is not None else "-"

                        ai_scan_rows.append({
                            'Symbol': ticker,
                            'Last Price': f"{last_close:,.2f}",
                            '% Change': f"{change}%",
                            'Liquidity (Bn)': round(t_data.get('avg_liquidity_bn', 0), 1),
                            'AI Forecast': label,
                            'Buy Price': f"{buy_price:,.2f}" if buy_price else "-",
                            'Stop Loss': f"{stop_loss_price:,.2f}" if stop_loss_price else "-",
                            'Take Profit': f"{take_profit_price:,.2f}" if take_profit_price else "-",
                            'Exp. Return (%)': f"{pred_ret:+.2f}%",
                            'Win/Loss': win_loss_str,
                            'Signal Score': quality_score,
                            'Conditions Met': conditions_str,
                            'Signal': signal,
                            '_pred_ret': pred_ret
                        })
            
            # Sort by predicted return descending
            ai_scan_rows.sort(key=lambda r: -r['_pred_ret'])
            for r in ai_scan_rows:
                r.pop('_pred_ret', None)
            
            cols = ['Symbol', 'Last Price', '% Change', 'Liquidity (Bn)', 'AI Forecast', 'Buy Price', 'Stop Loss', 'Take Profit', 'Exp. Return (%)', 'Win/Loss', 'Signal Score', 'Conditions Met', 'Signal']
            st.session_state.ai_scan_results = pd.DataFrame(ai_scan_rows, columns=cols)

        except Exception as e:
            st.error(f"❌ VN30 AI Scan Error: {e}")


# ─────────────────────────────────────────────
# Trigger Load (OPTIMIZED: Filter by Aggressive first)
# ─────────────────────────────────────────────
if run_btn or not st.session_state.data_loaded:
    tickers_to_scan = tk.get_tickers(exchange)
    
    with st.spinner(f"⏳ Scanning real-time liquidity for {exchange} ({len(tickers_to_scan)} symbols)..."):
        try:
            # 1. Fetch Aggressive data cho toàn bộ danh sách để lọc
            agg_results = fetcher.batch_fetch_aggressive(tickers_to_scan)
            st.session_state.agg_results = agg_results
            agg_liq_map = calculator.compute_liquidity_map(agg_results)
            st.session_state.agg_liq_map = agg_liq_map
            fetcher.save_liquidity_cache(agg_liq_map) # Cache lại
            
            # 2. Lọc danh sách mã thỏa mãn điều kiện
            winners = []
            for t in tickers_to_scan:
                if calculator.is_index_ticker(t):
                    winners.append(t)
                    continue
                liq = agg_liq_map.get(t, 0)
                if liq >= min_liq:
                    winners.append(t)
            
            full_count = len(tickers_to_scan)
            filtered_count = len(winners)
            
            if not winners or (len(winners) == 1 and calculator.is_index_ticker(winners[0])):
                st.warning(f"⚠️ No symbols found matching liquidity >= {min_liq} Bn VND (Based on actual volume).")
                st.stop()

            # 3. Tải dữ liệu lịch sử CHỈ cho các mã đã lọc
            st.info(f"📊 Found **{filtered_count}** tickers. Loading technical data...")
            
            prices_dict = load_market_data(
                "ALL", # Đã có danh sách winners cụ thể
                start_date.strftime('%Y-%m-%d'),
                end_date.strftime('%Y-%m-%d'),
                custom_tickers=winners
            )

            if not prices_dict:
                st.error("❌ Failed to load technical data.")
                st.stop()

            # Gán lại thông tin thanh khoản chuẩn vào dict để hiển thị ở các bảng
            for t, data in prices_dict.items():
                if t in agg_liq_map:
                    data['avg_liquidity_bn'] = agg_liq_map[t]

            st.session_state.stats = compute_all_stats(prices_dict, ma_periods, lookback_days, agg_liq_map=st.session_state.get('agg_liq_map', {}))
            st.session_state.data_loaded = True
            st.session_state.exchange = exchange
            st.success(f"✅ Successfully loaded {len(prices_dict)} tickers!")

        except Exception as e:
            st.error(f"❌ System Error: {e}")
            st.stop()


# ─────────────────────────────────────────────
# Render Dashboard
# ─────────────────────────────────────────────
# ─────────────────────────────────────────────
# Main Dashboard Tabs
# ─────────────────────────────────────────────
tabs = st.tabs([
    "🌍 Market Overview", 
    "📈 Stock Analysis", 
    "🔍 Scanners",
    "🤖 AI Engine",
    "⚖️ Tools"
])

# ─────────────────────────────────────────────
# TAB 1: THỊ TRƯỜNG (MARKET OVERVIEW)
# ─────────────────────────────────────────────
with tabs[0]:
    s = st.session_state.stats
    sentiment = s['sentiment']
    ad = s['ad_stats']
    hl = s['hl_stats']
    vol_mom = s['vol_momentum']

    # ── Index & Liquidity Row ───────────────────────────────────
    st.markdown('<div class="section-header">INDEX & MARKET LIQUIDITY</div>', unsafe_allow_html=True)
    col_vni, col_liq = st.columns(2)
    with col_vni:
        st.plotly_chart(charts.vnindex_chart(s['prices_raw'], lookback=lookback_days), use_container_width=True)
    with col_liq:
        st.plotly_chart(charts.liquidity_chart(s['liq_history']), use_container_width=True)

    # ── Market Status Banner ──────────────────────────────────────
    pct_adv = ad['pct_advance']
    dom_trend = "🟢 BULLISH" if pct_adv >= 55 else ("🔴 BEARISH" if pct_adv <= 40 else "🟡 NEUTRAL")
    dom_color = "#00E676" if pct_adv >= 55 else ("#FF1744" if pct_adv <= 40 else "#FFD740")
    hl_ratio  = hl.get('hl_ratio', 0)

    st.markdown(f"""
    <div style="background:linear-gradient(135deg,rgba(0,180,216,0.05) 0%,rgba(0,212,255,0.02) 100%);
                border:1px solid rgba(0,180,216,0.1); border-radius:12px;
                padding:14px 22px; margin-bottom:16px; display:flex; align-items:center; gap:32px; flex-wrap:wrap;">
        <div>
            <div style="font-size:0.62rem;color:#4A6080;text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;">Market Trend</div>
            <div style="font-size:1.2rem;font-weight:700;color:{dom_color};margin-top:3px;">{dom_trend}</div>
        </div>
        <div style="width:1px;height:40px;background:rgba(255,255,255,0.05);"></div>
        <div>
            <div style="font-size:0.62rem;color:#4A6080;text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;">Fear/Greed Score</div>
            <div style="font-size:1.2rem;font-weight:700;color:{sentiment['color']};margin-top:3px;">{sentiment['score']:.0f} · {sentiment['label']}</div>
        </div>
        <div style="width:1px;height:40px;background:rgba(255,255,255,0.05);"></div>
        <div>
            <div style="font-size:0.62rem;color:#4A6080;text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;">H/L Ratio (52W)</div>
            <div style="font-size:1.2rem;font-weight:700;color:{'#00E676' if hl_ratio > 1 else '#FF1744'};margin-top:3px;">{hl_ratio:.2f}x &nbsp;<span style="font-size:0.8rem;color:#4A6080;">({hl['new_highs']} highs / {hl['new_lows']} lows)</span></div>
        </div>
        <div style="width:1px;height:40px;background:rgba(255,255,255,0.05);"></div>
        <div>
            <div style="font-size:0.62rem;color:#4A6080;text-transform:uppercase;letter-spacing:2px;font-family:'JetBrains Mono',monospace;">Volume Momentum</div>
            <div style="font-size:1.2rem;font-weight:700;color:{'#00E676' if vol_mom > 1 else '#FF7043'};margin-top:3px;">{vol_mom:.2f}x &nbsp;<span style="font-size:0.8rem;color:#4A6080;">vs avg</span></div>
        </div>
        <div style="margin-left:auto;text-align:right;">
            <div style="font-size:0.62rem;color:#4A6080;font-family:'JetBrains Mono',monospace;">Total Analyzed</div>
            <div style="font-size:1.5rem;font-weight:700;color:#00D4FF;font-family:'JetBrains Mono',monospace;">{ad['total']:,}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── KPI Row ────────────────────────────────────────────────
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    pct_dec = 100 - pct_adv - (ad['unchanged']/ad['total']*100 if ad['total'] else 0)
    with k1: st.metric("🟢 Advance",     f"{ad['advances']:,}",      f"{pct_adv:.1f}%")
    with k2: st.metric("🔴 Decline",     f"{ad['declines']:,}",      f"{pct_dec:.1f}%")
    with k3: st.metric("⚪ Unchanged",   f"{ad['unchanged']:,}")
    with k4: st.metric("📊 Total",       f"{ad['total']:,}")
    with k5: st.metric("⬆️ 52W High",    f"{hl['new_highs']:,}",     delta=f"vs {hl['new_lows']} lows")
    with k6: st.metric("💧 Volume Mom",  f"{vol_mom:.2f}x",          delta="↑ Higher" if vol_mom > 1.0 else "↓ Lower",
                        delta_color="normal" if vol_mom > 1.0 else "inverse")

    # ── Sentiment Row ─────────────────────────────────────────
    st.markdown('<div class="section-header">SENTIMENT & ADVANCE/DECLINE</div>', unsafe_allow_html=True)
    col_gauge, col_donut, col_adline = st.columns([1, 1, 2])
    with col_gauge:
        st.plotly_chart(charts.sentiment_gauge(sentiment['score'], sentiment['label'], sentiment['color']), use_container_width=True)
        st.plotly_chart(charts.sentiment_history_chart(s['sentiment_history']), use_container_width=True)
    with col_donut:
        st.plotly_chart(charts.advance_decline_donut(ad), use_container_width=True)
    with col_adline:
        st.plotly_chart(charts.ad_line_chart(s['ad_history']), use_container_width=True)

    # ── Market Breadth Depth Row ─────────────────────────────
    st.markdown('<div class="section-header">MARKET BREADTH DEPTH</div>', unsafe_allow_html=True)
    col_ma_bar, col_hl_bar = st.columns(2)
    with col_ma_bar:
        st.plotly_chart(charts.ma_above_bar(s['ma_stats']), use_container_width=True)
    with col_hl_bar:
        st.plotly_chart(charts.new_high_low_chart(hl), use_container_width=True)

    # ── Breadth & Power Row ──────────────────────────────────
    st.markdown('<div class="section-header">MARKET BREADTH TRENDS & POWER INDEX</div>', unsafe_allow_html=True)
    col_breadth, col_power = st.columns(2)
    with col_breadth:
        st.plotly_chart(charts.ma_breadth_history_chart(s['ma_history']), use_container_width=True)
    with col_power:
        st.plotly_chart(charts.market_power_chart(s['power_hist']), use_container_width=True)

    # ── Advanced Breadth Row ─────────────────────────────────
    st.markdown('<div class="section-header">ADVANCED BREADTH ANALYTICS</div>', unsafe_allow_html=True)
    col_rsi_psy, col_thrust = st.columns(2)
    with col_rsi_psy:
        st.plotly_chart(charts.rsi_psy_breadth_chart(s['adv_analytics']), use_container_width=True)
    with col_thrust:
        st.plotly_chart(charts.breadth_thrust_chart(s['ad_history']), use_container_width=True)

    # ── Leaders & Liquidity ──────────────────────────────────
    st.markdown('<div class="section-header">MARKET LEADERS & MONEY FLOW</div>', unsafe_allow_html=True)
    col_influence, col_top_flow = st.columns([3, 2])
    with col_influence:
        st.markdown("### 🏆 Index Influence (Leaders & Laggards)")
        st.dataframe(s['index_influence'], use_container_width=True, hide_index=True)
    with col_top_flow:
        # Prepare Top Flow Data
        liq_map = st.session_state.agg_liq_map
        if liq_map:
            top_liq_df = pd.DataFrame([{'Ticker': k, 'Value': v} for k, v in liq_map.items()])
            top_liq_df = top_liq_df.sort_values('Value', ascending=False).head(10)
            st.plotly_chart(charts.top_money_flow_chart(top_liq_df), use_container_width=True)
        else:
            st.info("💡 Run 'Refresh Liquidity' or 'Run Analysis' to see top stock flows.")

    # ── Historical Summary Table ─────────────────────────────
    st.markdown('<div class="section-header">MARKET HISTORY & BREADTH SUMMARY</div>', unsafe_allow_html=True)
    
    with st.spinner("📊 Compiling historical data..."):
        # 1. Real-time Calculation (Today)
        live_calc = calculator.compute_market_history_combined(
            s['prices_raw'], 
            ma_periods, 
            lookback_days,
            agg_results=st.session_state.get('agg_results', {})
        )
        
        # Take today from live_calc
        if not live_calc.empty:
            today_df = live_calc.iloc[[0]] 
            # Save snapshot for EOD
            fetcher.save_market_summary_snapshot(today_df)
        else:
            today_df = pd.DataFrame()
        
        # 3. Load history from file (Previous sessions)
        persistent_history = fetcher.load_market_summary_history()
        
        # Merge: Today (Live) + All Past (File)
        if not persistent_history.empty:
            hist_df = pd.concat([today_df, persistent_history], ignore_index=True)
            hist_df = hist_df.drop_duplicates(subset=['Date'], keep='first')
        else:
            hist_df = today_df

        if not hist_df.empty:
            hist_df = hist_df.sort_values('Date', ascending=False)
            st.info(f"📁 Merged current session with {len(persistent_history)} historical records.")

        if not hist_df.empty:
            # Formatting logic
            def color_change(val):
                if pd.isna(val): return ""
                color = "#00E676" if val > 0 else ("#FF1744" if val < 0 else "#888")
                return f'color: {color}; font-weight: 600;'

            def color_index(val, df_col):
                if pd.isna(val) or df_col.empty: return ""
                min_v, max_v = df_col.min(), df_col.max()
                if max_v == min_v: return "background-color: rgba(0,212,255,0.1);"
                alpha = 0.05 + 0.25 * (val - min_v) / (max_v - min_v)
                return f'background-color: rgba(0,212,255,{alpha}); color: #E0F0FF; font-weight: 600;'
            
            def color_ma(val, df_col):
                if pd.isna(val) or df_col.empty: return ""
                min_v, max_v = df_col.min(), df_col.max()
                if max_v == min_v: return ""
                rel = (val - min_v) / (max_v - min_v)
                if rel < 0.5:
                    alpha = 0.3 * (1 - rel*2)
                    return f'background-color: rgba(255,23,68,{alpha});'
                else:
                    alpha = 0.3 * (rel-0.5)*2
                    return f'background-color: rgba(0,230,118,{alpha});'

            def color_sd(val):
                if pd.isna(val) or val == 0: return ""
                alpha = min(0.6, abs(val) / 200)
                if val > 0:
                    return f'background-color: rgba(0,230,118,{alpha}); color: #E0F0FF; font-weight: 600;'
                else:
                    return f'background-color: rgba(255,23,68,{alpha}); color: #E0F0FF; font-weight: 600;'

            def color_rsi(val):
                if pd.isna(val) or val == 0: return ""
                if val >= 70: return 'background-color: rgba(255,23,68,0.2); color: #FF1744; font-weight: 700;'
                if val <= 30: return 'background-color: rgba(0,230,118,0.2); color: #00E676; font-weight: 700;'
                return 'color: #C8D8E8;'

            # Apply styler
            styler = hist_df.style.format({
                'Change': '{:+.2f}',
                'VNINDEX': '{:,.2f}',
                'RSI': '{:.1f}',
                'Net S/D': '{:,.1f}M',
                'Demand (M)': '{:,.1f}M',
                'Supply (M)': '{:,.1f}M',
                'Power': '{:+.2f}'
            })
            
            styler = styler.applymap(color_change, subset=['Change', 'Power'])
            styler = styler.apply(lambda x: [color_index(v, x) for v in x], subset=['VNINDEX'])
            if 'RSI' in hist_df.columns:
                styler = styler.applymap(color_rsi, subset=['RSI'])
            for p in ma_periods:
                ma_col = f'MA{p}'
                if ma_col in hist_df.columns:
                    styler = styler.apply(lambda x: [color_ma(v, x) for v in x], subset=[ma_col])
            
            if 'Net S/D' in hist_df.columns:
                styler = styler.applymap(color_sd, subset=['Net S/D'])
            
            if 'Demand (M)' in hist_df.columns:
                styler = styler.applymap(lambda v: 'color: #00E676; font-weight: 500;' if v > 0 else '', subset=['Demand (M)'])
            if 'Supply (M)' in hist_df.columns:
                styler = styler.applymap(lambda v: 'color: #FF1744; font-weight: 500;' if v > 0 else '', subset=['Supply (M)'])

            if 'Notes' in hist_df.columns:
                styler = styler.applymap(lambda v: 'color: #FFD740; font-weight: 700;' if any(x in str(v) for x in ['PUMP', 'CRASH', 'RSI', 'STRONG']) else '', subset=['Notes'])

            st.dataframe(
                styler,
                use_container_width=True,
                height=500,
                hide_index=True
            )
            st.caption("💡 History: MA (number of stocks above MA), Net S/D (Million shares), Auto-notes based on Index & Breadth.")
        else:
            st.warning("⚠️ Insufficient data for historical table.")

    # ── Market Momentum & Distribution ───────────────────────
    st.markdown('<div class="section-header">MARKET MOMENTUM & DISTRIBUTION</div>', unsafe_allow_html=True)
    c_mc, c_dist, c_radar = st.columns([2, 2, 1])
    with c_mc:
        st.plotly_chart(charts.mcclellan_chart(s['mc_df']), use_container_width=True)
    with c_dist:
        st.plotly_chart(charts.change_distribution_chart(s['dist_data']), use_container_width=True)
    with c_radar:
        st.plotly_chart(charts.sentiment_components_chart(sentiment['components']), use_container_width=True)

    # ── International (VIX) Row ──────────────────────────────
    st.markdown('<div class="section-header">GLOBAL FEAR INDEX (VIX)</div>', unsafe_allow_html=True)
    try:
        vix_df = fetcher.fetch_vix()
        if not vix_df.empty:
            st.plotly_chart(charts.vix_chart(vix_df), use_container_width=True)
        else:
            st.warning("Could not fetch VIX data from LiteFinance. Check API status.")
    except Exception as e:
        st.error(f"VIX Display Error: {e}")


# ─────────────────────────────────────────────
# TAB 2: PHÂN TÍCH CỔ PHIẾU (SINGLE STOCK)
# ─────────────────────────────────────────────
with tabs[1]:
    st.markdown('<div class="section-header">TECHNICAL STOCK ANALYSIS</div>', unsafe_allow_html=True)
    
    # Ticker Selection
    all_tickers = tk.get_tickers("ALL")
    selected_ticker = st.selectbox("Select Ticker", options=all_tickers, index=all_tickers.index("SSI") if "SSI" in all_tickers else 0)
    
    if selected_ticker:
        with st.spinner(f"⏳ Loading analysis for {selected_ticker}..."):
            # Fetch data for technical analysis
            ta_start = (date.today() - timedelta(days=365)).strftime('%Y-%m-%d')
            raw_ta = fetcher.batch_fetch([selected_ticker], ta_start, date.today().strftime('%Y-%m-%d'))
            dict_ta = fetcher.parse_results(raw_ta)
            
            if selected_ticker in dict_ta:
                t_data = dict_ta[selected_ticker]
                df_ta = pd.DataFrame({
                    'open': t_data['open'],
                    'high': t_data['high'],
                    'low': t_data['low'],
                    'close': t_data['close'],
                    'volume': t_data['volume']
                }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
                
                # Calculate indicators
                df_ta['MA20'] = df_ta['close'].rolling(20).mean()
                df_ta['MA50'] = df_ta['close'].rolling(50).mean()
                bb = calculator.compute_bollinger_bands(df_ta['close'])
                df_ta['bb_upper'] = bb['upper']
                df_ta['bb_lower'] = bb['lower']
                
                # Render Professional Chart
                st.plotly_chart(charts.professional_candlestick_chart(df_ta, selected_ticker), use_container_width=True)
                
                # Render MACD Chart
                df_macd = calculator.compute_macd(df_ta['close'])
                st.plotly_chart(charts.macd_chart(df_macd), use_container_width=True)
                
                # Render Money Flow Indicators
                mfi = calculator.compute_mfi(df_ta)
                obv = calculator.compute_obv(df_ta)
                vwap = calculator.compute_vwap(df_ta)
                st.plotly_chart(charts.single_stock_money_flow_chart(df_ta, mfi, obv, vwap), use_container_width=True)
                
                # Stats Metrics
                c1, c2, c3, c4 = st.columns(4)
                last_price = df_ta['close'].iloc[-1]
                prev_price = df_ta['close'].iloc[-2]
                chg = (last_price - prev_price) / prev_price * 100
                with c1: st.metric("Current Price", f"{last_price:,.1f}", f"{chg:+.2f}%")
                with c2: st.metric("RSI (14)", f"{calculator.compute_rsi(df_ta['close']).iloc[-1]:.1f}")
                with c3: st.metric("MA20", f"{df_ta['MA20'].iloc[-1]:,.1f}", delta=f"{((last_price/df_ta['MA20'].iloc[-1])-1)*100:+.1f}%")
                with c4: st.metric("Volume", f"{df_ta['volume'].iloc[-1]/1e6:.1f}M")
                
                # ── Aggressive Trading Analysis ──────────────────────
                st.markdown('<div class="section-header">LIQUIDITY & AGGRESSIVE TRADING (REAL-TIME)</div>', unsafe_allow_html=True)
                
                agg_data = fetcher.fetch_aggressive_trading(selected_ticker)
                # Đảm bảo agg_data là list và không trống
                if isinstance(agg_data, list) and len(agg_data) > 0:
                    agg_stats = calculator.compute_aggressive_stats(agg_data)
                    
                    # Layout metrics
                    m1, m2, m3, m4 = st.columns(4)
                    with m1:
                        st.metric("Total Turnover", f"{agg_stats['total_value_bn']:,.2f} Bn", help="Total traded value calculated from executed price levels")
                    with m2:
                        st.metric("Aggressive Buying", f"{agg_stats['buy_value']/1e9:,.2f} Bn", delta=f"{agg_stats['buy_volume']/1e6:,.1f}M shares", delta_color="normal")
                    with m3:
                        st.metric("Aggressive Selling", f"{agg_stats['sell_value']/1e9:,.2f} Bn", delta=f"-{agg_stats['sell_volume']/1e6:,.1f}M shares", delta_color="inverse")
                    with m4:
                        net_val = agg_stats['net_value_bn']
                        st.metric("Net Flow", f"{net_val:+.2f} Bn", delta="Demand > Supply" if net_val > 0 else "Supply > Demand", delta_color="normal" if net_val > 0 else "inverse")
                    
                    # Price Level Table
                    expected_cols = ['Price', 'TotalVolume', 'AggressiveBuyingVolume', 'AggressiveSellingVolume', 'OtherVolume']
                    
                    # Trích xuất list từ dict {"data": [...]}
                    agg_list = agg_data.get('data', []) if isinstance(agg_data, dict) else agg_data
                    df_agg = pd.DataFrame(agg_list)
                    
                    # Kiểm tra xem có đủ cột cần thiết không
                    if not df_agg.empty and all(c in df_agg.columns for c in expected_cols):
                        with st.expander("Price Level Trade Details"):
                            df_agg = df_agg[expected_cols]
                            df_agg.columns = ['Price', 'Total Vol', 'Agg Buy', 'Agg Sell', 'Others']
                            df_agg = df_agg.sort_values('Price', ascending=False)
                            st.dataframe(df_agg.style.format({
                                'Price': '{:,.0f}',
                                'Total Vol': '{:,.0f}',
                                'Agg Buy': '{:,.0f}',
                                'Agg Sell': '{:,.0f}',
                                'Others': '{:,.0f}'
                            }), use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ Detailed trade data format incorrect.")
                else:
                    st.info("💡 Information: Aggressive trade data will appear when the market is open.")
            else:
                st.error(f"❌ No data found for symbol {selected_ticker}")

    # ── Stochastic Backtest Results (If exists for selected ticker) ──
    if st.session_state.bt_results and st.session_state.bt_target == selected_ticker:
        st.markdown("---")
        st.markdown(f'<div class="section-header">STOCHASTIC BACKTEST RESULTS: {selected_ticker}</div>', unsafe_allow_html=True)
        
        res = st.session_state.bt_results
        
        # Performance Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Return", f"{res['total_return']:.2f}%")
        m2.metric("Win Rate", f"{res['win_rate']:.1f}%")
        m3.metric("Total Trades", res['total_trades'])
        m4.metric("Avg PnL/Trade", f"{res['avg_pnl']:.2f}%")
        
        # Equity Curve
        st.plotly_chart(charts.backtest_equity_chart(res), use_container_width=True)
        
        # Trade History Table
        with st.expander("Detailed Trade History"):
            if res['trades']:
                trades_df = pd.DataFrame(res['trades'])
                # Format trades table
                trades_df['entry_price'] = trades_df['entry_price'].map('{:,.2f}'.format)
                trades_df['exit_price'] = trades_df['exit_price'].map('{:,.2f}'.format)
                trades_df['pnl'] = (trades_df['pnl'] * 100).map('{:+.2f}%'.format)
                st.dataframe(trades_df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────
# TAB 3: TRÌNH QUÉT (SCANNERS)
# ─────────────────────────────────────────────
with tabs[2]:
    st.markdown('<div class="section-header">STOCK SIGNAL SCANNERS</div>', unsafe_allow_html=True)
    
    if st.session_state.scanner_results is not None:
        st.markdown("### 📊 Stochastic Signal Scanner (VN100)")
        df_scan = st.session_state.scanner_results
        
        # Styling the scanner table
        def color_signal(val):
            if 'BUY' in str(val): return 'background-color: rgba(0, 230, 118, 0.2); color: #00E676; font-weight: 700;'
            if 'SELL' in str(val): return 'background-color: rgba(255, 23, 68, 0.2); color: #FF1744; font-weight: 700;'
            return ''
            
        st.dataframe(
            df_scan.style.applymap(color_signal, subset=['Signal', 'Recommendation']),
            use_container_width=True,
            height=600,
            hide_index=True
        )
    else:
        st.info("💡 Click **'Scan VN30 Signals'** in the sidebar to run the Stochastic scanner.")

# ─────────────────────────────────────────────
# TAB 4: AI ENGINE
# ─────────────────────────────────────────────
with tabs[3]:
    st.markdown('<div class="section-header">AI MULTI-FACTOR ANALYSIS</div>', unsafe_allow_html=True)
    
    # ── AI Scanner Results ──
    if st.session_state.ai_scan_results is not None:
        st.markdown("### 🔍 AI VN100 Scanner Results")
        ai_df = st.session_state.ai_scan_results
        
        def color_ai(val):
            if 'BUY' in str(val): return 'background-color: rgba(0, 230, 118, 0.2); color: #00E676; font-weight: 700;'
            if 'SELL' in str(val): return 'background-color: rgba(255, 23, 68, 0.2); color: #FF1744; font-weight: 700;'
            return ''

        st.dataframe(
            ai_df.style.applymap(color_ai, subset=['AI Forecast']),
            use_container_width=True,
            height=600,
            hide_index=True
        )
    
    # ── AI Individual Results ──
    if st.session_state.get('ai_results') is not None:
        st.markdown("---")
        target = st.session_state.get('ai_target', 'Unknown')
        st.markdown(f"### 🤖 AI Individual Analysis: {target}")
        ai_res = st.session_state.ai_results
        
        # Display AI metrics
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Predicted Return", f"{ai_res.get('predicted_return', 0):+.2f}%")
        c2.metric("Model Accuracy", f"{ai_res.get('accuracy', 0)*100:.1f}%")
        c3.metric("Signal", ai_res.get('signal', 'HOLD'))
        c4.metric("Sharpe Ratio", f"{ai_res.get('sharpe_ratio', 0):.2f}")
        
        # Feature Importance or other AI charts
        if 'importance_fig' in ai_res:
             st.plotly_chart(ai_res['importance_fig'], use_container_width=True)

    if st.session_state.ai_scan_results is None and st.session_state.get('ai_results') is None:
        st.info("💡 Run **'AI Multi-Factor Analysis'** or **'AI VN100 Scanner'** from the sidebar to see AI insights.")

# ─────────────────────────────────────────────
# TAB 5: TOOLS
# ─────────────────────────────────────────────
with tabs[4]:
    st.markdown('<div class="section-header">INVESTMENT TOOLS</div>', unsafe_allow_html=True)
    
    col_calc1, col_calc2 = st.columns(2)
    with col_calc1:
        st.markdown("### ⚖️ Position Sizing Calculator")
        capital = st.number_input("Total Capital (VND)", value=100000000, step=10000000)
        risk_pct = st.slider("Risk per Trade (%)", 1.0, 5.0, 2.0, 0.5)
        entry_p = st.number_input("Entry Price (VND)", value=30000, step=100)
        stop_l = st.number_input("Stop Loss (VND)", value=28500, step=100)
        
        if entry_p > stop_l:
            risk_per_share = entry_p - stop_l
            max_risk_amt = capital * (risk_pct / 100)
            shares = int(max_risk_amt / risk_per_share)
            total_val = shares * entry_p
            
            st.success(f"""
            **Calculation Results:**
            - Shares to buy: **{shares:,}**
            - Total Investment Value: **{total_val:,} VND**
            - Capital Allocation Ratio: **{total_val/capital*100:.1f}%**
            """)
        else:
            st.warning("Stop Loss must be lower than Entry Price.")

    with col_calc2:
        st.markdown("### 💹 Expected ROI")
        shares_owned = st.number_input("Quantity Owned", value=1000, step=100)
        avg_price = st.number_input("Average Price", value=30000, step=100)
        target_p = st.number_input("Target Price", value=35000, step=100)
        
        profit = (target_p - avg_price) * shares_owned
        profit_pct = (target_p / avg_price - 1) * 100
        st.info(f"""
        **If target price is reached:**
        - Expected Profit: **{profit:,} VND**
        - Return Rate: **{profit_pct:+.1f}%**
        """)

# ─────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────
st.markdown("---")
st.markdown(f'<div style="text-align:center; color:#444; font-size:0.75rem;">VN Market Dashboard • {datetime.now().strftime("%d/%m/%Y %H:%M")}</div>', unsafe_allow_html=True)
