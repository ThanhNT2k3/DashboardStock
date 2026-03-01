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
    <div class="dashboard-subtitle">Phân tích độ rộng & sentiment thị trường chứng khoán Việt Nam </div>
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
    st.markdown("### 🧪 Backtest Stochastic")
    bt_ticker = st.text_input("Mã cổ phiếu", value="SSI", help="Nhập mã CP để chạy backtest tín hiệu Stochastic")
    
    col_bt1, col_bt2 = st.columns(2)
    with col_bt1:
        bt_k = st.number_input("%K Period", value=14, min_value=1)
        bt_oversold = st.number_input("Oversold", value=20, min_value=1)
    with col_bt2:
        bt_d = st.number_input("%D Period", value=3, min_value=1)
        bt_overbought = st.number_input("Overbought", value=80, max_value=100)
        
    bt_run = st.button("🚀 Chạy Backtest", use_container_width=True)
    
    st.markdown("---")
    st.markdown("### 🔍 VN30 Stochastic Scanner")
    scanner_run = st.button("📊 Quét tín hiệu VN30", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🧠 AI Multi-Factor Backtest")
    ai_ticker = st.text_input("Mã CP (AI)", value="SSI", key="ai_ticker_input", help="Phân tích mã CP kết hợp Vàng, Dầu, Lãi suất, Khối ngoại")
    ai_split = st.slider("Train/Test Split", 0.5, 0.9, 0.7, 0.1)
    ai_run = st.button("🤖 Chạy AI Backtest", use_container_width=True)
    ai_scanner_run = st.button("🔍 Quét AI VN30", use_container_width=True, help="Quét toàn bộ rổ VN30 bằng mô hình AI đa nhân tố")

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
if 'bt_results' not in st.session_state:
    st.session_state.bt_results = None
if 'bt_target' not in st.session_state:
    st.session_state.bt_target = ""


# ─────────────────────────────────────────────
# Trigger Backtest (Independent)
# ─────────────────────────────────────────────
if bt_run:
    with st.spinner(f"🧪 Đang backtest mã {bt_ticker}..."):
        try:
            # Tải dữ liệu riêng cho ticker này (start_date lùi xa hơn để có dữ liệu SMA/Stoch chuẩn)
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
                st.error(f"❌ Không tìm thấy dữ liệu cho mã {bt_ticker}")
        except Exception as e:
            st.error(f"❌ Lỗi backtest: {e}")


# ─────────────────────────────────────────────
# Trigger Scanner (VN30 Batch)
# ─────────────────────────────────────────────
if 'scanner_results' not in st.session_state:
    st.session_state.scanner_results = None

if scanner_run:
    with st.spinner("🔍 Đang quét tín hiệu Stochastic cho rổ VN30..."):
        try:
            vn30_list = tk.VN30
            # Tải dữ liệu 200 ngày để Stoch chuẩn
            scanner_start = (date.today() - timedelta(days=200)).strftime('%Y-%m-%d')
            raw_scan = fetcher.batch_fetch(vn30_list, scanner_start, date.today().strftime('%Y-%m-%d'))
            dict_scan = fetcher.parse_results(raw_scan)
            
            scan_rows = []
            for ticker in vn30_list:
                if ticker in dict_scan:
                    t_data = dict_scan[ticker]
                    df_t = pd.DataFrame({
                        'close': t_data['close'],
                        'high': t_data['high'],
                        'low': t_data['low']
                    }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
                    
                    # 1. Chạy Backtest lịch sử
                    bt = calculator.run_backtest_stochastic(df_t)
                    
                    # 2. Lấy trạng thái hiện tại
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
                        
                        recommendation = "Nắm giữ"
                        if signal == "BUY (Cross Up)":
                            recommendation = "MUA MẠNH 🔥" if wr_val > 55 else "MUA ✅"
                        elif signal == "SELL (Cross Down)":
                            recommendation = "BÁN MẠNH ⚠️" if wr_val > 55 else "BÁN 🔻"
                        elif signal == "Potential Up":
                            recommendation = "Theo dõi MUA 👀"
                        elif signal == "Potential Down":
                            recommendation = "Theo dõi BÁN 📉"

                        scan_rows.append({
                            'Mã': ticker,
                            'Giá hiện tại': f"{t_data['close'][-1]:,.2f}",
                            '%K': round(last_k, 1),
                            '%D': round(last_d, 1),
                            'Tín hiệu hiện tại': signal,
                            'Khuyến nghị': recommendation,
                            'Win Rate': f"{wr_val}%",
                            'Total Return': f"{total_ret_val}%",
                            'Số lệnh': bt.get('total_trades', 0)
                        })
            
            st.session_state.scanner_results = pd.DataFrame(scan_rows)
        except Exception as e:
            st.error(f"❌ Lỗi quét VN30: {e}")


# ─────────────────────────────────────────────
# Trigger AI Backtest
# ─────────────────────────────────────────────
if 'ai_results' not in st.session_state:
    st.session_state.ai_results = None

if 'ai_run' in locals() and ai_run:
    with st.spinner(f"🤖 Đang huấn luyện AI cho {ai_ticker}..."):
        try:
            # Tải dữ liệu lịch sử dài (cần ít nhất 2 năm để model AI học tốt)
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
                
                # Chạy AI Backtest
                ai_res = calculator.run_backtest_ai(ai_ticker, df_ticker_raw, ai_start, ai_end_s, ai_split)
                st.session_state.ai_results = ai_res
                st.session_state.ai_target = ai_ticker
            else:
                st.error(f"❌ Không tìm thấy dữ liệu cho mã {ai_ticker}")
        except Exception as e:
            st.error(f"❌ Lỗi AI Engine: {e}")


# ─────────────────────────────────────────────
# Trigger AI Scanner (VN30)
# ─────────────────────────────────────────────
if 'ai_scan_results' not in st.session_state:
    st.session_state.ai_scan_results = None

if 'ai_scanner_run' in locals() and ai_scanner_run:
    with st.spinner("🤖 Đang quét rổ VN30 bằng AI (kỹ thuật + vĩ mô + khối ngoại)..."):
        try:
            vn30_list = tk.VN30
            ai_start = (date.today() - timedelta(days=730)).strftime('%Y-%m-%d')
            ai_end_s = end_date.strftime('%Y-%m-%d')
            
            # Tải dữ liệu toàn bộ VN30
            raw_ai_scan = fetcher.batch_fetch(vn30_list, ai_start, ai_end_s)
            dict_ai_scan = fetcher.parse_results(raw_ai_scan)
            
            # Tải dữ liệu vĩ mô & khối ngoại batch (tối ưu tốc độ)
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
                        # Train mô hình nhanh cho mã này
                        engine.train(full_df)
                        signal = engine.predict(full_df)
                        
                        # Metadata cho hiển thị
                        last_close = t_data['close'][-1]
                        change = t_data['change_pct']
                        
                        # Lợi nhuận dự báo: trung bình TARGET_RET gần nhất (dòng cuối thường NaN)
                        valid_ret = full_df['TARGET_RET'].dropna()
                        pred_ret = valid_ret.tail(20).mean() * 100 if len(valid_ret) > 0 else 0.0
                        
                        # Chỉ coi là BUY khi lợi nhuận kỳ vọng đủ cao
                        MIN_STRONG_RET = 3.0  # 3% cho 5 phiên tới
                        if signal == 2 and pred_ret < MIN_STRONG_RET:
                            signal = 1  # chuyển về HOLD nếu tín hiệu yếu
                        
                        label = "BUY 🚀" if signal == 2 else ("SELL ⚠️" if signal == 0 else "HOLD ⏳")
                        
                        # Thu thập điều kiện cổ phiếu đáp ứng (từ dòng cuối full_df)
                        last_row = full_df.iloc[-1]
                        prev_row = full_df.iloc[-2] if len(full_df) > 1 else None
                        conditions = []
                        if signal == 2:
                            conditions.append("✓ Tín hiệu AI: BUY")
                        elif signal == 0:
                            conditions.append("⚠ Tín hiệu AI: SELL")
                        else:
                            conditions.append("○ Tín hiệu AI: HOLD")
                        if pred_ret >= 3.0:
                            conditions.append(f"✓ Lợi nhuận dự báo ≥ 3% ({pred_ret:+.1f}%)")
                        elif pred_ret > 0:
                            conditions.append(f"✓ Lợi nhuận dự báo dương ({pred_ret:+.1f}%)")
                        elif pred_ret < -3:
                            conditions.append(f"⚠ Lợi nhuận dự báo âm ({pred_ret:+.1f}%)")
                        rsi = last_row.get('RSI', 50)
                        if rsi < 30:
                            conditions.append(f"✓ RSI oversold ({rsi:.0f}) - cơ hội mua")
                        elif rsi >= 70:
                            conditions.append(f"⚠ RSI quá mua ({rsi:.0f})")
                        elif rsi < 70:
                            conditions.append(f"✓ RSI không quá mua ({rsi:.0f})")
                        ma20 = last_row.get('MA20')
                        if ma20 and last_close > ma20:
                            conditions.append("✓ Giá trên MA20 (momentum)")
                        elif ma20:
                            conditions.append("○ Giá dưới MA20")
                        ma50 = last_row.get('MA50')
                        if ma50 and last_close > ma50:
                            conditions.append("✓ Giá trên MA50 (xu hướng)")
                        elif ma50:
                            conditions.append("○ Giá dưới MA50")
                        ff = last_row.get('foreignNetValue', 0)
                        if ff > 0:
                            conditions.append("✓ Khối ngoại mua ròng")
                        elif ff < 0:
                            conditions.append("○ Khối ngoại bán ròng")
                        ret1d = last_row.get('RETURNS_1D', 0)
                        if ret1d is not None and ret1d > 0:
                            conditions.append("✓ Phiên gần nhất tăng giá")
                        elif ret1d is not None and ret1d < 0:
                            conditions.append("○ Phiên gần nhất giảm giá")

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
                                direction = "tăng" if gold_chg > 0 else "giảm"
                                conditions.append(f"○ Giá vàng thế giới {direction} khoảng {gold_chg:+.1f}% hôm nay")

                            if oil_chg is not None and abs(oil_chg) >= 1.0:
                                direction = "tăng" if oil_chg > 0 else "giảm"
                                conditions.append(f"○ Giá dầu thô {direction} khoảng {oil_chg:+.1f}% hôm nay")

                            if dxy_chg is not None and abs(dxy_chg) >= 0.5:
                                direction = "tăng" if dxy_chg > 0 else "giảm"
                                conditions.append(f"○ Chỉ số USD (DXY) {direction} khoảng {dxy_chg:+.1f}%")

                            if us10y_chg is not None and abs(us10y_chg) >= 0.5:
                                direction = "tăng" if us10y_chg > 0 else "giảm"
                                conditions.append(f"○ Lợi suất TPCP Mỹ 10Y {direction} khoảng {us10y_chg:+.1f} điểm bps tương đối")

                        if len(conditions) <= 1:
                            conditions.append("— Không đủ điều kiện nổi bật")

                        # Điểm chất lượng tín hiệu: dựa trên số điều kiện tích cực / tiêu cực
                        positives = sum(1 for c in conditions if c.startswith("✓"))
                        negatives = sum(1 for c in conditions if c.startswith("⚠"))
                        raw_score = positives * 15 - negatives * 10
                        quality_score = max(0, min(100, raw_score))

                        conditions_str = "\n".join(conditions)
                        
                        # Giá mua khuyến nghị (ví dụ: thấp hơn giá hiện tại 0.5% để tối ưu)
                        buy_price = last_close * 0.995 if signal == 2 else None
                        # Giá cắt lỗ & chốt bán mặc định (-3%, +8%) - chỉ khi có tín hiệu BUY
                        stop_loss_price = buy_price * 0.97 if buy_price else None
                        take_profit_price = buy_price * 1.08 if buy_price else None
                        
                        ai_scan_rows.append({
                            'Mã': ticker,
                            'Giá hiện tại': f"{last_close:,.2f}",
                            '% Thay đổi': f"{change}%",
                            'Dự báo AI': label,
                            'Giá mua': f"{buy_price:,.2f}" if buy_price else "-",
                            'Giá cắt lỗ': f"{stop_loss_price:,.2f}" if stop_loss_price else "-",
                            'Giá chốt bán': f"{take_profit_price:,.2f}" if take_profit_price else "-",
                            'Lợi nhuận dự báo (%)': f"{pred_ret:+.2f}%",
                            'Điểm tín hiệu': quality_score,
                            'Điều kiện đáp ứng': conditions_str,
                            'Tín hiệu': signal,
                            '_pred_ret': pred_ret
                        })
            
            # Sắp xếp theo lợi nhuận dự báo giảm dần → mã có khả năng lợi nhuận cao hiển thị trước
            ai_scan_rows.sort(key=lambda r: -r['_pred_ret'])
            for r in ai_scan_rows:
                r.pop('_pred_ret', None)
            
            cols = ['Mã', 'Giá hiện tại', '% Thay đổi', 'Dự báo AI', 'Giá mua', 'Giá cắt lỗ', 'Giá chốt bán', 'Lợi nhuận dự báo (%)', 'Điểm tín hiệu', 'Điều kiện đáp ứng', 'Tín hiệu']
            st.session_state.ai_scan_results = pd.DataFrame(ai_scan_rows, columns=cols)
        except Exception as e:
            st.error(f"❌ Lỗi quét AI VN30: {e}")


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
# ROW 8: Backtest Results (If triggered)
# ══════════════════════════════════════════════
if 'bt_results' in st.session_state and st.session_state.bt_results:
    st.markdown('<div class="section-header">SIGNAL BACKTEST RESULTS: STOCHASTIC CROSSOVER</div>', unsafe_allow_html=True)
    res = st.session_state.bt_results
    ticker_name = st.session_state.bt_target
    
    # 1. Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Return", f"{res['total_return']}%", delta=None)
    with m2:
        st.metric("Win Rate", f"{res['win_rate']}%", delta=None)
    with m3:
        st.metric("Total Trades", res['total_trades'])
    with m4:
        st.metric("Asset", ticker_name)
        
    # 2. Charts
    tab_equity, tab_signals = st.tabs(["📈 Equity Curve", "📉 Stochastic Signal"])
    
    with tab_equity:
        fig_equity = charts.backtest_equity_chart(res)
        st.plotly_chart(fig_equity, use_container_width=True)
        
    with tab_signals:
        # Lấy dữ liệu raw của ticker để vẽ signal
        t_data = s['prices_raw'].get(ticker_name, {})
        if t_data:
            # Reconstruct DataFrame for chart
            df_t = pd.DataFrame({
                'close': t_data['close'],
                'high': t_data['high'],
                'low': t_data['low']
            }, index=pd.to_datetime([datetime.fromtimestamp(t) for t in t_data['timestamps']]))
            df_stoch = calculator.compute_stochastic(df_t)
            fig_stoch = charts.stochastic_chart(df_stoch, ticker_name)
            st.plotly_chart(fig_stoch, use_container_width=True)
    
    # 3. Trade List
    if res['trades']:
        with st.expander("📜 Chi tiết lịch sử lệnh"):
            df_trades = pd.DataFrame(res['trades'])
            df_trades['pnl'] = (df_trades['pnl'] * 100).map("{:.2f}%".format)
            st.dataframe(df_trades, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════
# ROW 9: Scanner Results
# ══════════════════════════════════════════════
if st.session_state.scanner_results is not None:
    st.markdown('<div class="section-header">VN30 STOCHASTIC SCANNER & PERFORMANCE SUMMARY</div>', unsafe_allow_html=True)
    df_scan = st.session_state.scanner_results
    
    def style_scanner(row):
        cols = [''] * len(row)
        sig_idx = row.index.get_loc('Tín hiệu hiện tại')
        wr_idx = row.index.get_loc('Win Rate')
        rec_idx = row.index.get_loc('Khuyến nghị')
        
        # Color for Signal
        if "BUY" in str(row['Tín hiệu hiện tại']): cols[sig_idx] = 'color: #00E676; font-weight: bold'
        elif "SELL" in str(row['Tín hiệu hiện tại']): cols[sig_idx] = 'color: #FF1744; font-weight: bold'
        
        # Color for Recommendation
        rec = str(row['Khuyến nghị'])
        if "MUA MẠNH" in rec: cols[rec_idx] = 'background-color: rgba(0, 230, 118, 0.4); color: white; font-weight: bold'
        elif "MUA" in rec: cols[rec_idx] = 'background-color: rgba(0, 230, 118, 0.15); color: #00E676'
        elif "BÁN MẠNH" in rec: cols[rec_idx] = 'background-color: rgba(255, 23, 68, 0.4); color: white; font-weight: bold'
        elif "BÁN" in rec: cols[rec_idx] = 'background-color: rgba(255, 23, 68, 0.15); color: #FF1744'
        elif "Theo dõi" in rec: cols[rec_idx] = 'color: #FFD740; font-style: italic'
        
        wr_val = float(str(row['Win Rate']).replace('%', ''))
        if wr_val > 60: cols[wr_idx] = 'color: #00E676; font-weight: bold'
        elif wr_val < 45: cols[wr_idx] = 'color: #FF1744'
        
        return cols

    st.dataframe(
        df_scan.style.apply(style_scanner, axis=1),
        use_container_width=True,
        hide_index=True
    )


# ══════════════════════════════════════════════
# ROW 10: AI Backtest Results
# ══════════════════════════════════════════════
if st.session_state.ai_results:
    st.markdown('<div class="section-header">🤖 AI MULTI-FACTOR PREDICTION & BACKTEST (XGBOOST)</div>', unsafe_allow_html=True)
    res_ai = st.session_state.ai_results
    ai_target = st.session_state.ai_target
    
    if 'summary' in res_ai:
        st.warning(res_ai['summary'])
    else:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("AI Total Return", f"{res_ai['total_return']}%")
        with m2:
            st.metric("Win Rate", f"{res_ai['win_rate']}%")
        with m3:
            st.metric("Total Trades", res_ai['total_trades'])
        with m4:
            st.metric("Asset", ai_target)
            
        st.info(f"💡 {res_ai.get('train_info', '')}")
        
        tab_ai_eq, tab_ai_trades = st.tabs(["📈 Equity Curve", "📜 Trade List"])
        
        with tab_ai_eq:
            fig_ai_eq = charts.backtest_equity_chart(res_ai, title=f"AI Backtest Equity: {ai_target}")
            st.plotly_chart(fig_ai_eq, use_container_width=True)
            
        with tab_ai_trades:
            if res_ai['trades']:
                df_ai_trades = pd.DataFrame(res_ai['trades'])
                df_ai_trades['pnl'] = (df_ai_trades['pnl'] * 100).map("{:.2f}%".format)
                st.dataframe(df_ai_trades, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════
# ROW 11: AI Scanner Results
# ══════════════════════════════════════════════
if st.session_state.ai_scan_results is not None:
    st.markdown('<div class="section-header">🤖 AI VN30 OPPORTUNITY SCANNER (MULTI-FACTOR)</div>', unsafe_allow_html=True)
    df_ai_scan = st.session_state.ai_scan_results
    
    # Cơ hội BUY
    buy_list = df_ai_scan[df_ai_scan['Tín hiệu'] == 2]['Mã'].tolist()
    if buy_list:
        st.success(f"🔥 **Cơ hội tiềm năng (BUY):** {', '.join(buy_list)}")
    else:
        st.info("💡 Chưa tìm thấy cơ hội mua mạnh trong VN30 hiện tại theo mô hình AI.")
    
    # Top mã có khả năng lợi nhuận cao (đã sắp xếp theo Lợi nhuận dự báo giảm dần)
    top5 = df_ai_scan.head(5)
    top5_str = ", ".join([f"{r['Mã']} ({r['Lợi nhuận dự báo (%)']})" for _, r in top5.iterrows()])
    st.info(f"📈 **Top 5 khả năng lợi nhuận cao:** {top5_str}")

    # Chi tiết điều kiện từng mã (expander cho trực quan)
    with st.expander("📋 **Chi tiết điều kiện đáp ứng theo mã**", expanded=False):
        for _, row in df_ai_scan.iterrows():
            cond = row.get('Điều kiện đáp ứng', '')
            if pd.isna(cond) or not str(cond).strip():
                continue
            sig_label = "🟢" if row['Tín hiệu'] == 2 else ("🔴" if row['Tín hiệu'] == 0 else "🟡")
            st.markdown(f"**{sig_label} {row['Mã']}** — {row['Dự báo AI']} | LN dự báo: {row['Lợi nhuận dự báo (%)']}")
            for line in str(cond).strip().split("\n"):
                st.markdown(f"- {line}")
            st.markdown("---")

    def style_ai_scanner(row):
        cols = [''] * len(row)
        sig_idx = row.index.get_loc('Dự báo AI')
        ret_idx = row.index.get_loc('Lợi nhuận dự báo (%)')
        
        if row['Tín hiệu'] == 2: 
            cols[sig_idx] = 'background-color: rgba(0, 230, 118, 0.4); color: white; font-weight: bold'
            cols[ret_idx] = 'color: #00E676; font-weight: bold'
        elif row['Tín hiệu'] == 0: 
            cols[sig_idx] = 'background-color: rgba(255, 23, 68, 0.4); color: white; font-weight: bold'
            cols[ret_idx] = 'color: #FF1744'
        
        return cols

    st.dataframe(
        df_ai_scan.style.apply(style_ai_scanner, axis=1).hide(subset=['Tín hiệu'], axis='columns'),
        use_container_width=True,
        hide_index=True,
        column_config={
            "Điều kiện đáp ứng": st.column_config.TextColumn("Điều kiện đáp ứng", width="large", help="Các điều kiện cổ phiếu đáp ứng")
        }
    )


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
