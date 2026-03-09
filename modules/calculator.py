"""
Market Breadth & Sentiment Calculator
Tính toán các chỉ số phân tích độ rộng thị trường
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# Import AIEngine for AI backtesting
try:
    from modules.ai_engine import AIEngine
except ImportError:
    AIEngine = None


# ─────────────────────────────────────────────
# Thống kê Dòng tiền (Money Flow)
# ─────────────────────────────────────────────

SECTOR_MAP = {
    'Banking': ['VCB', 'BID', 'CTG', 'TCB', 'MBB', 'VPB', 'ACB', 'HDB', 'LPB', 'TPB', 'MSB', 'STB', 'OCB', 'VIB', 'SSB', 'NAB', 'EIB', 'BAB', 'NVB', 'KLB', 'SGB', 'ABB', 'VBB'],
    'Real Estate': ['VIC', 'VHM', 'VRE', 'NVL', 'PDR', 'DIG', 'DXG', 'NLG', 'KBC', 'KDH', 'HDC', 'CEO', 'L14', 'IDC', 'SZC', 'VGC', 'HTN', 'DXS', 'KHG', 'CRE', 'SCR', 'TCH', 'HQC', 'LDG', 'KOS', 'BCM', 'SJS', 'VPI', 'VPH', 'AGG', 'NBB', 'NRC', 'HDG', 'ITA', 'FLC', 'ROS'],
    'Securities': ['SSI', 'VND', 'VCI', 'HCM', 'SHS', 'FTS', 'BSI', 'ORS', 'VIX', 'CTS', 'AGR', 'MBS', 'VDS', 'TVS', 'BVS', 'PSI', 'WSS', 'APS', 'EVS', 'FTS', 'BSI', 'TVB', 'TVC'],
    'Steel & Resources': ['HPG', 'HSG', 'NKG', 'VGS', 'TVN', 'SMC', 'TLH', 'POM', 'TIS', 'VCA', 'KSA', 'KSB', 'DHB'],
    'Oil & Gas': ['GAS', 'PVD', 'PVS', 'PVT', 'BSR', 'PLX', 'OIL', 'PVC', 'PVB', 'PVG', 'PVH', 'POS', 'PVO', 'PEQ'],
    'Industrial Zones': ['IDC', 'KBC', 'SZC', 'VGC', 'LH', 'TIP', 'D2D', 'NTC', 'ITA', 'HPI', 'VRG'],
    'Construction & Materials': ['HHV', 'VCG', 'LCG', 'FCN', 'C4G', 'HT1', 'BCC', 'PC1', 'DHA', 'CTD', 'HBC', 'REZ', 'ACC', 'DAH'],
    'Retail & E-commerce': ['MWG', 'FRT', 'DGW', 'PET', 'MSN', 'PNJ', 'ABS', 'SVW'],
    'Technology': ['FPT', 'CMG', 'ELC', 'ITD', 'SAM', 'ONE', 'DST'],
    'Utilities': ['POW', 'GAS', 'TDM', 'BWE', 'REE', 'NT2', 'PPC', 'GEG', 'VSH', 'SJD', 'TMP', 'HNA', 'VPD', 'GHC', 'AVC', 'TBC', 'SBA'],
    'Consumer Staples': ['VNM', 'SAB', 'MSN', 'BHN', 'KDC', 'VHC', 'ANV', 'IDI', 'DBC', 'HAG', 'HNG', 'MCH', 'VOC', 'QNS', 'CLX', 'NLS', 'MML', 'VHE'],
    'Chemicals & Fertilizer': ['DGC', 'DCM', 'DPM', 'CSV', 'LAS', 'BFC', 'DDV', 'DNP', 'PHR', 'DPR', 'TRC', 'GVR'],
    'Logistics & Transport': ['GMD', 'HAH', 'VSC', 'PVT', 'VNA', 'VTO', 'VIP', 'TCW', 'SGP', 'VND', 'PHP', 'TCL', 'CDN', 'ILB', 'VJC', 'HVN'],
    'Insurance': ['BVH', 'MIG', 'BMI', 'PGI', 'PTI', 'AIC', 'BIC', 'VNR'],
    'Health & Pharma': ['DMC', 'DHG', 'TRA', 'IMP', 'DP3', 'DBT', 'AMV', 'JVC'],
}
# ─────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────
def is_index_ticker(ticker: str) -> bool:
    """Kiểm tra xem ticker có phải là chỉ số thị trường không"""
    return ticker in ['VNINDEX', 'HNXINDEX', 'UPINDEX', 'VN30', 'VN100', 'HNX30']

# ─────────────────────────────────────────────
# MA Analysis
# ─────────────────────────────────────────────

def compute_ma_stats(prices_dict: dict, periods: list = [20, 50, 200]) -> dict:
    """
    Đếm số cổ phiếu đang giao dịch TRÊN / DƯỚI MA(period)
    
    Returns:
        {
            20: {'above': 120, 'below': 80, 'total': 200, 'pct_above': 60.0},
            50: {...},
            200: {...}
        }
    """
    result = {}
    for period in periods:
        above = below = total = 0
        for ticker, data in prices_dict.items():
            if is_index_ticker(ticker):
                continue
            closes = data.get('close', [])
            if len(closes) >= period + 1:
                ma = float(np.mean(closes[-period:]))
                last = closes[-1]
                if last > ma:
                    above += 1
                elif last < ma:
                    below += 1
                total += 1
        result[period] = {
            'above': above,
            'below': below,
            'total': total,
            'pct_above': round(above / total * 100, 1) if total else 0.0,
        }
    return result


def compute_market_power_history(prices_dict: dict, lookback: int = 60) -> pd.DataFrame:
    """
    Tính Supply, Demand, Power theo lịch sử (toàn thị trường)
    1. Supply: Total volume where Close < Open (in Millions)
    2. Demand: Total volume where Close > Open (in Millions)
    3. Supply_Demand: Demand - Supply
    4. Power: Sum((Close - Open) * Volume) / 1,000,000
    """
    all_ts = set()
    for data in prices_dict.values():
        all_ts.update(data.get('timestamps', []))
    
    sorted_ts = sorted(list(all_ts))[-lookback:]
    if not sorted_ts:
        return pd.DataFrame()

    history = []
    for t in sorted_ts:
        supply = 0.0
        demand = 0.0
        power  = 0.0
        
        for ticker, data in prices_dict.items():
            # Bỏ qua các chỉ số để tránh double counting Supply/Demand
            if ticker in ['VNINDEX', 'HNXINDEX', 'UPINDEX', 'VN30', 'HNX30']:
                continue

            ts      = data.get('timestamps', [])
            closes  = data.get('close', [])
            opens   = data.get('open', [])
            volumes = data.get('volume', [])
            
            if t in ts:
                idx = ts.index(t)
                c = closes[idx]
                o = opens[idx]
                v = volumes[idx] if volumes[idx] else 0
                
                if c < o:
                    supply += v
                elif c > o:
                    demand += v
                
                power += (c - o) * v
        
        history.append({
            'date':          datetime.fromtimestamp(t).strftime('%Y-%m-%d'),
            'supply':        round(supply / 1e6, 2),
            'demand':        round(demand / 1e6, 2),
            'supply_demand': round((demand - supply) / 1e6, 2),
            'power':         round(power / 1e6, 2),
        })

    return pd.DataFrame(history)


def compute_ma_history(prices_dict: dict, periods: list = [20, 50, 200], lookback: int = 60) -> pd.DataFrame:
    """
    Tính % cổ phiếu trên MA theo lịch sử (mỗi ngày)
    Trả về DataFrame: date, pct_ma20, pct_ma50, pct_ma200...
    """
    # Tìm tất cả timestamps
    all_ts = set()
    for data in prices_dict.values():
        all_ts.update(data.get('timestamps', []))
    
    sorted_ts = sorted(list(all_ts))[-lookback:]
    if not sorted_ts:
        return pd.DataFrame()

    history = []
    
    # Để tối ưu, tính MA cho toàn bộ chuỗi của mỗi ticker trước
    ticker_mas = {}
    for ticker, data in prices_dict.items():
        closes = np.array(data.get('close', []))
        ts     = np.array(data.get('timestamps', []))
        if len(closes) < 10: continue
        
        mas = {}
        for p in periods:
            if len(closes) >= p:
                # Simple Moving Average
                ma_vals = pd.Series(closes).rolling(window=p).mean().values
                mas[p] = dict(zip(ts, ma_vals))
        ticker_mas[ticker] = {'closes': dict(zip(ts, closes)), 'mas': mas}

    for t in sorted_ts:
        row = {'date': datetime.fromtimestamp(t).strftime('%Y-%m-%d')}
        # Thêm VNINDEX nếu có
        if 'VNINDEX' in prices_dict:
            vni_data = prices_dict['VNINDEX']
            vni_ts = vni_data.get('timestamps', [])
            vni_closes = vni_data.get('close', [])
            if t in vni_ts:
                vni_idx = vni_ts.index(t)
                row['VNINDEX'] = round(vni_closes[vni_idx], 2)

        for p in periods:
            above = total = 0
            for ticker, info in ticker_mas.items():
                if ticker == 'VNINDEX': continue # Bỏ qua VNINDEX trong đếm breadth
                ma_map = info['mas'].get(p, {})
                close_map = info['closes']
                if t in ma_map and t in close_map:
                    if close_map[t] > ma_map[t]:
                        above += 1
                    total += 1
            row[f'count_ma{p}'] = above
            row[f'pct_ma{p}'] = round(above / total * 100, 1) if total else 0.0
        history.append(row)

    return pd.DataFrame(history)


# ─────────────────────────────────────────────
# Advanced Technical Indicators
# ─────────────────────────────────────────────

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Tính Relative Strength Index (RSI)"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / (loss.replace(0, 0.001))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def compute_psy(series: pd.Series, period: int = 12) -> pd.Series:
    """Tính Psychological Line (PSY) - Tâm lý thị trường"""
    # % số phiên tăng điểm trong N phiên gần nhất
    rising = (series.diff() > 0).rolling(window=period).sum()
    psy = (rising / period) * 100
    return psy.fillna(50)


def compute_breadth_thrust(ad_df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Zweig Breadth Thrust Indicator
    = EMA(advances / (advances + declines), period)
    """
    if ad_df.empty: return pd.Series()
    ratio = ad_df['advances'] / (ad_df['advances'] + ad_df['declines']).replace(0, 1)
    thrust = ratio.ewm(span=period, adjust=False).mean()
    return thrust


def compute_advanced_analytics(ma_history: pd.DataFrame) -> pd.DataFrame:
    """
    Tính toán RSI và PSY cho các đường Breadth (ví dụ % Above MA20)
    """
    if ma_history.empty:
        return pd.DataFrame()

    df = ma_history.copy()
    for col in df.columns:
        if col.startswith('pct_ma'):
            df[f'rsi_{col}'] = compute_rsi(df[col], period=14)
            df[f'psy_{col}'] = compute_psy(df[col], period=12)

    return df


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.DataFrame:
    """Tính MACD (Moving Average Convergence Divergence)"""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    hist = macd - signal
    return pd.DataFrame({'macd': macd, 'signal': signal, 'hist': hist})


def compute_bollinger_bands(series: pd.Series, period: int = 20, std_dev: int = 2) -> pd.DataFrame:
    """Tính Bollinger Bands (BBands)"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return pd.DataFrame({'middle': sma, 'upper': upper, 'lower': lower})




# ─────────────────────────────────────────────
# Advance / Decline
# ─────────────────────────────────────────────

def compute_advance_decline(prices_dict: dict) -> dict:
    """
    Tính A/D stats cho phiên gần nhất
    
    Returns:
        {
            'advances': 234,
            'declines': 156,
            'unchanged': 30,
            'total': 420,
            'ratio': 1.5,
            'ad_line': 78,          # advances - declines
            'pct_advance': 55.7,
        }
    """
    advances = declines = unchanged = 0

    for ticker, data in prices_dict.items():
        if is_index_ticker(ticker):
            continue
        chg = data.get('change_pct', 0)
        if chg > 0.01:
            advances += 1
        elif chg < -0.01:
            declines += 1
        else:
            unchanged += 1

    total = advances + declines + unchanged
    return {
        'advances':    advances,
        'declines':    declines,
        'unchanged':   unchanged,
        'total':       total,
        'ratio':       round(advances / max(declines, 1), 2),
        'ad_line':     advances - declines,
        'pct_advance': round(advances / total * 100, 1) if total else 0.0,
    }


def compute_ad_history(prices_dict: dict, lookback: int = 60) -> pd.DataFrame:
    """
    Tính A/D line theo lịch sử (mỗi ngày)
    Trả về DataFrame với index là ngày, columns: advances, declines, ad_line, cumulative_ad
    """
    # Tìm timestamp chung nhất
    all_ts = {}
    for ticker, data in prices_dict.items():
        ts = data.get('timestamps', [])
        closes = data.get('close', [])
        if len(ts) >= 2 and len(closes) >= 2:
            for i, t in enumerate(ts[-lookback:]):
                if t not in all_ts:
                    all_ts[t] = []
                # Tính change so với ngày trước (nếu có)
                idx = len(ts) - lookback + i
                if idx > 0 and closes[idx - 1] != 0:
                    chg = (closes[idx] - closes[idx - 1]) / closes[idx - 1] * 100
                    all_ts[t].append(chg)

    rows = []
    for ts_val in sorted(all_ts.keys()):
        changes = all_ts[ts_val]
        if not changes:
            continue
        adv = sum(1 for c in changes if c > 0.01)
        dec = sum(1 for c in changes if c < -0.01)
        rows.append({
            'date':     datetime.fromtimestamp(ts_val).strftime('%Y-%m-%d'),
            'advances': adv,
            'declines': dec,
            'ad_net':   adv - dec,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df['cumulative_ad'] = df['ad_net'].cumsum()
    return df


# ─────────────────────────────────────────────
# New High / Low
# ─────────────────────────────────────────────

def compute_new_high_low(prices_dict: dict, period_weeks: int = 52) -> dict:
    """
    Tính số mã đạt 52-week high/low
    period_weeks=52 → lookback ~260 ngày giao dịch
    
    Returns:
        {'new_highs': 45, 'new_lows': 12, 'total': 420, 'hl_ratio': 0.79}
    """
    trading_days = period_weeks * 5
    new_highs = new_lows = total = 0

    for ticker, data in prices_dict.items():
        if is_index_ticker(ticker):
            continue
        closes = data.get('close', [])
        if len(closes) >= 20:  # cần ít nhất 20 ngày
            window = closes[-min(trading_days, len(closes)):]
            last = closes[-1]
            if last >= max(window) * 0.99:   # trong 1% của high
                new_highs += 1
            elif last <= min(window) * 1.01:  # trong 1% của low
                new_lows += 1
            total += 1

    hl_ratio = new_highs / max(new_highs + new_lows, 1)
    return {
        'new_highs': new_highs,
        'new_lows':  new_lows,
        'total':     total,
        'hl_ratio':  round(hl_ratio, 3),
    }


# ─────────────────────────────────────────────
# Volume / Liquidity
# ─────────────────────────────────────────────

def compute_liquidity_history(prices_dict: dict, lookback: int = 60) -> pd.DataFrame:
    """
    Tổng giá trị giao dịch (volume * close) theo ngày
    Trả về DataFrame: date, total_volume, total_value, avg_volume_ma5
    """
    ts_data = {}  # {timestamp: {volume, value}}

    for ticker, data in prices_dict.items():
        ts_list = data.get('timestamps', [])
        closes  = data.get('close', [])
        volumes = data.get('volume', [])

        n = min(len(ts_list), len(closes), len(volumes), lookback)
        for i in range(-n, 0):
            t = ts_list[i]
            v = volumes[i] if volumes[i] else 0
            c = closes[i]  if closes[i]  else 0
            val = v * c

            if t not in ts_data:
                ts_data[t] = {'volume': 0, 'value': 0}
            ts_data[t]['volume'] += v
            ts_data[t]['value']  += val

    if not ts_data:
        return pd.DataFrame()

    rows = []
    for ts_val in sorted(ts_data.keys()):
        rows.append({
            'date':   datetime.fromtimestamp(ts_val).strftime('%Y-%m-%d'),
            'volume': ts_data[ts_val]['volume'],
            'value':  ts_data[ts_val]['value'],
        })

    df = pd.DataFrame(rows)
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['value_bn']   = df['value'] / 1e9   # tỷ VNĐ
    return df


def filter_by_liquidity(prices_dict: dict, min_value_bn: float = 0.0, days: int = 20, mode: str = 'mean', **kwargs) -> dict:
    """
    Lọc danh sách tickers theo GTGD (Giá trị giao dịch)
    min_value_bn: Tỷ VNĐ
    mode: 'mean', 'median', 'min', 'last'
    """
    if min_value_bn <= 0:
        return prices_dict

    filtered = {}
    
    # 1. Nếu có aggressive liquidity map, ưu tiên dùng cái này để lọc
    # (vì dữ liệu aggressive phản ánh chính xác GTGD thực tế của phiên hiện tại)
    agg_liq_map = kwargs.get('agg_liq_map', {})

    for ticker, data in prices_dict.items():
        # Luôn giữ các chỉ số index
        if is_index_ticker(ticker):
            filtered[ticker] = data
            continue
            
        # Ưu tiên lấy từ cache aggressive
        if ticker in agg_liq_map:
            metric = agg_liq_map[ticker]
            data['avg_liquidity_bn'] = metric
            if metric >= min_value_bn:
                filtered[ticker] = data
            continue
            
        # Nếu không có trong cache, tính từ dữ liệu lịch sử
        closes = np.array(data.get('close', []))
        volumes = np.array(data.get('volume', []))
        
        # Đảm bảo có đủ dữ liệu
        n = min(len(closes), len(volumes), days)
        if n < 1:
            continue

        # Lấy n phiên gần nhất
        c_recent = closes[-n:]
        v_recent = volumes[-n:]
        
        # Giá trị giao dịch từng phiên (tỷ đồng): (Giá_1000đ * Khối_lượng) / 10^6
        # VPS API trả giá dạng đơn vị 1000đ (ví dụ: 30.2 = 30,200 VNĐ)
        daily_values = (c_recent * v_recent) / 1e6
        
        metric = 0.0
        if mode == 'mean':
            metric = np.mean(daily_values)
        elif mode == 'median':
            metric = np.median(daily_values)
        elif mode == 'min':
            metric = np.min(daily_values)
        elif mode == 'last':
            metric = daily_values[-1]
        else:
            metric = np.mean(daily_values)

        if metric >= min_value_bn:
            # Lưu lại giá trị thanh khoản để hiển thị sau này if needed
            data['avg_liquidity_bn'] = metric
            filtered[ticker] = data
            
    return filtered


def compute_volume_momentum(prices_dict: dict, short: int = 5, long: int = 20) -> float:
    """
    Tính Volume Momentum = SMA(5) / SMA(20) trung bình toàn thị trường
    > 1.0: volume tăng (tích cực)
    < 1.0: volume giảm
    """
    ratios = []
    for ticker, data in prices_dict.items():
        if is_index_ticker(ticker):
            continue
        vols = data.get('volume', [])
        if len(vols) >= long:
            sma_short = np.mean(vols[-short:])
            sma_long  = np.mean(vols[-long:])
            if sma_long > 0:
                ratios.append(sma_short / sma_long)
    return round(float(np.mean(ratios)), 3) if ratios else 1.0


# ─────────────────────────────────────────────
# Price Distribution
# ─────────────────────────────────────────────

def compute_change_distribution(prices_dict: dict) -> dict:
    """
    Phân phối % thay đổi giá phiên cuối
    Returns: {'bins': [...], 'counts': [...]}
    """
    changes = [
        data['change_pct']
        for data in prices_dict.values()
        if 'change_pct' in data
    ]
    if not changes:
        return {'bins': [], 'counts': []}

    bins = [-10, -7, -5, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 7, 10]
    counts = [0] * (len(bins) + 1)
    for c in changes:
        placed = False
        for i, b in enumerate(bins):
            if c <= b:
                counts[i] += 1
                placed = True
                break
        if not placed:
            counts[-1] += 1

    labels = [f"< {bins[0]}"] + [f"{bins[i-1]}~{bins[i]}" for i in range(1, len(bins))] + [f"> {bins[-1]}"]
    return {'bins': labels, 'counts': counts, 'raw': changes}


# ─────────────────────────────────────────────
# Market Sentiment Score (Fear/Greed)
# ─────────────────────────────────────────────

def compute_sentiment_score(
    ma_stats: dict,
    ad_stats: dict,
    hl_stats: dict,
    vol_momentum: float,
    ma_period: int = 50
) -> dict:
    """
    Fear/Greed Score 0–100
    
    Weights:
        25% — % stocks above MA50
        25% — A/D pct advance
        20% — New High ratio
        15% — Volume momentum
        15% — A/D ratio (normalized)
    
    Returns:
        {'score': 62.5, 'label': 'Greed', 'color': '#90EE90', 'components': {...}}
    """
    # Component 1: % above MA50 (0-100)
    pct_ma = ma_stats.get(ma_period, {}).get('pct_above', 50.0)
    s1 = float(pct_ma)

    # Component 2: % advance (0-100)
    pct_adv = ad_stats.get('pct_advance', 50.0)
    s2 = float(pct_adv)

    # Component 3: New High ratio (0-100)
    hl_ratio = hl_stats.get('hl_ratio', 0.5)
    s3 = float(hl_ratio) * 100

    # Component 4: Volume momentum (0.5-2.0 → 0-100)
    s4 = max(0, min(100, (float(vol_momentum) - 0.5) / 1.5 * 100))

    # Component 5: A/D ratio (0-3 → 0-100)
    ad_ratio = float(ad_stats.get('ratio', 1.0))
    s5 = max(0, min(100, ad_ratio / 3.0 * 100))

    score = (
        0.25 * s1 +
        0.25 * s2 +
        0.20 * s3 +
        0.15 * s4 +
        0.15 * s5
    )
    score = max(0.0, min(100.0, score))

    # Label và màu
    if score >= 75:
        label, color = "Extreme Greed", "#00C853"
    elif score >= 55:
        label, color = "Greed",         "#69F0AE"
    elif score >= 45:
        label, color = "Neutral",       "#FFD740"
    elif score >= 25:
        label, color = "Fear",          "#FF6D00"
    else:
        label, color = "Extreme Fear",  "#FF1744"

    return {
        'score': round(score, 1),
        'label': label,
        'color': color,
        'components': {
            'MA50 above %':       round(s1, 1),
            'Advance %':          round(s2, 1),
            'New High ratio':     round(s3, 1),
            'Volume momentum':    round(s4, 1),
            'A/D ratio':          round(s5, 1),
        }
    }


def compute_sentiment_history(prices_dict: dict, ma_period: int = 50, lookback: int = 60) -> pd.DataFrame:
    """
    Tính Sentiment Score theo lịch sử
    """
    ad_hist = compute_ad_history(prices_dict, lookback)
    ma_hist = compute_ma_history(prices_dict, [ma_period], lookback)
    
    if ad_hist.empty or ma_hist.empty:
        return pd.DataFrame()
        
    combined = pd.merge(ad_hist, ma_hist, on='date')
    
    # Giả định volume momentum và hl_stats là hằng số hoặc trung bình cho lịch sử (vì tính historical volume momentum hơi phức tạp ở đây)
    # Tuy nhiên, để chính xác nhất ta nên có Volume và H/L history.
    # Để nhanh, ta dùng MA Breadth + A/D Breadth làm sentiment chính cho lịch sử
    
    results = []
    for _, row in combined.iterrows():
        # Dùng công thức đơn giản hóa cho lịch sử: 50% MA breadth, 50% A/D breadth
        s1 = float(row.get(f'pct_ma{ma_period}', 50))
        
        total_ad = row['advances'] + row['declines']
        s2 = (row['advances'] / total_ad * 100) if total_ad > 0 else 50
        
        score = 0.5 * s1 + 0.5 * s2
        results.append({
            'date': row['date'],
            'sentiment_score': round(score, 1)
        })
        
    return pd.DataFrame(results)


# ─────────────────────────────────────────────
# McClellan Oscillator
# ─────────────────────────────────────────────

def compute_mcclellan(ad_df: pd.DataFrame) -> pd.DataFrame:
    """
    McClellan Oscillator = EMA(19) - EMA(39) của A/D net
    McClellan Summation = cumsum of oscillator
    """
    if ad_df.empty or len(ad_df) < 20:
        return pd.DataFrame()

    df = ad_df.copy()
    df['ema19'] = df['ad_net'].ewm(span=19, adjust=False).mean()
    df['ema39'] = df['ad_net'].ewm(span=39, adjust=False).mean()
    df['oscillator']  = df['ema19'] - df['ema39']
    df['summation']   = df['oscillator'].cumsum()
    return df


def compute_stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3, slowing: int = 3) -> pd.DataFrame:
    """Tính Stochastic Oscillator (%K, %D)"""
    if df.empty or len(df) < k_period:
        return pd.DataFrame()
    
    # Đảm bảo index là datetime nếu chưa có
    low_min = df['low'].rolling(window=k_period).min()
    high_max = df['high'].rolling(window=k_period).max()
    
    # %K raw
    df['k_raw'] = 100 * ((df['close'] - low_min) / (high_max - low_min).replace(0, 0.001))
    # %K slowed (SMA)
    df['%K'] = df['k_raw'].rolling(window=slowing).mean()
    # %D (SMA of %K)
    df['%D'] = df['%K'].rolling(window=d_period).mean()
    
    return df.drop(columns=['k_raw'])


def run_backtest_stochastic(
    data: pd.DataFrame, 
    k_period: int = 14, 
    d_period: int = 3, 
    oversold: int = 20, 
    overbought: int = 80
) -> dict:
    """
    Backtest chiến thuật Stochastic Crossover
    Buy: %K cắt LÊN %D và %K < oversold
    Sell: %K cắt XUỐNG %D và %K > overbought
    """
    if data.empty: return {}
    
    df = compute_stochastic(data.copy(), k_period, d_period)
    if '%D' not in df.columns: return {}

    df = df.dropna(subset=['%K', '%D'])
    
    trades = []
    position = None # None, 'long'
    entry_price = 0
    entry_date = None
    
    for i in range(1, len(df)):
        current_k = df.iloc[i]['%K']
        current_d = df.iloc[i]['%D']
        prev_k = df.iloc[i-1]['%K']
        prev_d = df.iloc[i-1]['%D']
        price = df.iloc[i]['close']
        date = df.iloc[i].name
        
        # Tín hiệu Crossover
        cross_up = prev_k <= prev_d and current_k > current_d
        cross_down = prev_k >= prev_d and current_k < current_d
        
        # BUY Logic
        if position is None and cross_up and current_k < oversold:
            position = 'long'
            entry_price = price
            entry_date = date
            
        # SELL Logic
        elif position == 'long' and cross_down and current_k > overbought:
            pnl = (price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': price,
                'pnl': pnl,
                'status': 'Win' if pnl > 0 else 'Loss'
            })
            position = None

    if not trades:
        return {'total_return': 0, 'win_rate': 0, 'total_trades': 0, 'trades': [], 'summary': "No trades executed"}

    # Tính toán performance
    df_trades = pd.DataFrame(trades)
    total_return = (1 + df_trades['pnl']).prod() - 1
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    
    return {
        'total_return': round(total_return * 100, 2),
        'win_rate': round(win_rate, 2),
        'total_trades': len(trades),
        'trades': trades,
        'equity_curve': (1 + df_trades['pnl']).cumprod()
    }


def run_backtest_ai(
    ticker: str,
    ticker_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    train_split: float = 0.7
) -> dict:
    """
    Backtest chiến thuật dựa trên mô hình AI (Multi-Factor)
    1. Huấn luyện mô hình trên `train_split` đầu tiên của dữ liệu.
    2. Dự báo trên phần còn lại.
    3. Mô phỏng mua/bán.
    """
    if AIEngine is None or ticker_df.empty:
        return {'total_return': 0, 'win_rate': 0, 'total_trades': 0, 'trades': [], 'summary': "AI module or data missing"}

    engine = AIEngine()
    
    # 1. Thu thập dữ liệu bổ sung (Macro, Foreign)
    macro_df = engine.fetch_macro_data(start_date, end_date)
    foreign_df = engine.fetch_foreign_flow(ticker, start_date, end_date)
    
    # 2. Xây dựng bộ Feature đầy đủ
    full_df = engine.prepare_features(ticker_df, macro_df, foreign_df)
    if len(full_df) < 50:
        return {'total_return': 0, 'win_rate': 0, 'total_trades': 0, 'trades': [], 'summary': f"Insufficient data for AI (need >50 sessions, current: {len(full_df)})"}

    # 3. Chia tập Train/Test theo thời gian (Tránh data leakage)
    split_idx = int(len(full_df) * train_split)
    train_data = full_df.iloc[:split_idx]
    test_data = full_df.iloc[split_idx:]
    
    # 4. Huấn luyện mô hình trên tập Train
    engine.train(train_data)
    
    # 5. Backtest trên tập Test
    trades = []
    position = None
    entry_price = 0
    entry_date = None
    
    # Tên cột feature đầu tiên để lấy date (test_data.index)
    test_dates = test_data.index

    for i in range(len(test_data)):
        current_row = test_data.iloc[i:i+1] # Lấy 1 dòng làm DF để giữ feature names
        pred_signal = engine.predict(current_row) # predict signal: 2: BUY, 1: HOLD, 0: SELL
        
        price = current_row['close'].values[0]
        date = test_dates[i]
        
        # BUY Logic: AI báo mua (2) & chưa có lệnh
        if position is None and pred_signal == 2:
            position = 'long'
            entry_price = price
            entry_date = date
            
        # SELL Logic: AI báo bán (0) & đang có lệnh (hoặc tín hiệu HOLD/Neutral sau 10 ngày)
        elif position == 'long' and (pred_signal == 0 or (date - entry_date).days >= 10):
            pnl = (price - entry_price) / entry_price
            trades.append({
                'entry_date': entry_date,
                'exit_date': date,
                'entry_price': entry_price,
                'exit_price': price,
                'pnl': pnl,
                'status': 'Win' if pnl > 0 else 'Loss'
            })
            position = None

    if not trades:
        return {'total_return': 0, 'win_rate': 0, 'total_trades': 0, 'trades': [], 'summary': "Hệ thống AI chưa tìm thấy cơ hội giao dịch phù hợp trong giai đoạn này."}

    # Tính performance
    df_trades = pd.DataFrame(trades)
    total_return = (1 + df_trades['pnl']).prod() - 1
    win_rate = (df_trades['pnl'] > 0).mean() * 100
    
    return {
        'total_return': round(total_return * 100, 2),
        'win_rate': round(win_rate, 2),
        'total_trades': len(trades),
        'trades': trades,
        'equity_curve': (1 + df_trades['pnl']).cumprod(),
        'train_info': f"Đã huấn luyện trên {len(train_data)} phiên, Test trên {len(test_data)} phiên."
    }


# ─────────────────────────────────────────────
# MA Detail Table (Above/Below MA10, MA20, MA50)
# ─────────────────────────────────────────────

def compute_ma_detail_table(prices_dict: dict, periods: list = [10, 20, 50]) -> pd.DataFrame:
    """
    Bảng chi tiết cổ phiếu nằm trên/dưới các đường MA.
    Mỗi cổ phiếu 1 dòng, có cột:
      - Mã, Giá, % thay đổi
      - MA10, MA20, MA50 (giá trị)
      - Trên_MA10, Trên_MA20, Trên_MA50 (True/False)
      - Khoảng cách (%) so với từng MA
      - Tổng MA trên (0-3)
      - Thanh khoản TB (Tỷ)
    """
    rows = []
    for ticker, data in prices_dict.items():
        # Bỏ qua chỉ số
        if ticker in ['VNINDEX', 'HNXINDEX', 'UPINDEX', 'VN30', 'VN100', 'HNX30']:
            continue
        
        closes = data.get('close', [])
        volumes = data.get('volume', [])
        if len(closes) < max(periods) + 1:
            continue
        
        last_close = closes[-1]
        change_pct = data.get('change_pct', 0)
        
        # Tính thanh khoản TB 20 phiên (tỷ VNĐ): (Giá * KL) / 1e6
        n = min(len(closes), len(volumes), 20)
        c_arr = np.array(closes[-n:])
        v_arr = np.array(volumes[-n:])
        avg_liq = float(np.mean(c_arr * v_arr) / 1e6)
        
        row = {
            'Symbol': ticker,
            'Price': last_close,
            '% Change': change_pct,
            'Avg Turnover (B)': round(avg_liq, 2),
        }
        
        total_above = 0
        for p in periods:
            if len(closes) >= p:
                ma_val = float(np.mean(closes[-p:]))
                above = last_close > ma_val
                dist_pct = (last_close - ma_val) / ma_val * 100 if ma_val != 0 else 0
                row[f'MA{p}'] = round(ma_val, 2)
                row[f'Above MA{p}'] = above
                row[f'%vs MA{p}'] = round(dist_pct, 2)
                if above:
                    total_above += 1
            else:
                row[f'MA{p}'] = None
                row[f'Trên MA{p}'] = None
                row[f'%vs MA{p}'] = None
        
        row['Total MAs Above'] = total_above
        rows.append(row)
    
    if not rows:
        return pd.DataFrame()
    
    df = pd.DataFrame(rows)
    # Sắp xếp: trên cả 3 MA trước, rồi theo % thay đổi
    df = df.sort_values(['Tổng MA trên', '% Thay đổi'], ascending=[False, False])
    return df.reset_index(drop=True)


# ─────────────────────────────────────────────
# Index Influence — Cổ phiếu ảnh hưởng chỉ số
# ─────────────────────────────────────────────

def compute_index_influence(prices_dict: dict, top_n: int = 30) -> pd.DataFrame:
    """
    Ước tính mức ảnh hưởng của từng cổ phiếu đến chỉ số VNINDEX.
    
    Logic:
    - Dùng GTGD trung bình 20 phiên làm proxy cho trọng số vốn hóa
      (cổ phiếu thanh khoản lớn → vốn hóa lớn → ảnh hưởng nhiều đến chỉ số)
    - Đóng góp điểm = (trọng số vốn hóa) x (% thay đổi giá) / 100
    - Sắp xếp theo |đóng góp| giảm dần
    
    Trả về DataFrame:
        Mã, Giá, % Thay đổi, GTGD TB (Tỷ), Vốn hóa ước tính %, Đóng góp (điểm),
        Loại tác động (Tích cực/Tiêu cực)
    """
    # Bước 1: Thu thập thông tin và tính GTGD trung bình
    stock_info = []
    for ticker, data in prices_dict.items():
        if ticker in ['VNINDEX', 'HNXINDEX', 'UPINDEX', 'VN30', 'VN100', 'HNX30']:
            continue
        
        closes = data.get('close', [])
        volumes = data.get('volume', [])
        change_pct = data.get('change_pct', 0)
        
        if len(closes) < 2 or len(volumes) < 2:
            continue
        
        last_close = closes[-1]
        
        # GTGD trung bình 20 phiên gần nhất (tỷ VNĐ)
        n = min(len(closes), len(volumes), 20)
        c_arr = np.array(closes[-n:])
        v_arr = np.array(volumes[-n:])
        avg_value_bn = float(np.mean(c_arr * v_arr) / 1e6)
        
        stock_info.append({
            'ticker': ticker,
            'close': last_close,
            'change_pct': change_pct,
            'avg_value_bn': avg_value_bn,
        })
    
    if not stock_info:
        return pd.DataFrame()
    
    df = pd.DataFrame(stock_info)
    
    # Bước 2: Trọng số ước tính (dựa trên GTGD)
    total_value = df['avg_value_bn'].sum()
    if total_value == 0:
        return pd.DataFrame()
    
    df['weight_pct'] = df['avg_value_bn'] / total_value * 100
    
    # Bước 3: Đóng góp điểm (đơn vị tương đối)
    # Nếu VNINDEX có dữ liệu, dùng giá trị VNI để quy đổi
    vnindex_data = prices_dict.get('VNINDEX', {})
    vnindex_close = vnindex_data.get('close', [0])[-1] if vnindex_data.get('close') else 1000
    
    df['contribution'] = df['weight_pct'] * df['change_pct'] / 100
    # Quy đổi thành "điểm" VNIndex tương đối
    df['contribution_pts'] = df['contribution'] * vnindex_close / 100
    
    df['abs_contribution'] = df['contribution_pts'].abs()
    df = df.sort_values('abs_contribution', ascending=False)
    
    # Step 4: Format output
    result_rows = []
    for _, row in df.head(top_n).iterrows():
        impact_type = '🟢 Positive' if row['change_pct'] > 0 else ('🔴 Negative' if row['change_pct'] < 0 else '⚪ Unchanged')
        result_rows.append({
            'Symbol': row['ticker'],
            'Price': f"{row['close']:,.2f}",
            '% Change': f"{row['change_pct']:+.2f}%",
            'Avg Turnover (B)': round(row['avg_value_bn'], 1),
            'Weight (%)': round(row['weight_pct'], 2),
            'Contribution (pts)': round(row['contribution_pts'], 3),
            'Impact Type': impact_type,
        })
    
    return pd.DataFrame(result_rows)


# ─────────────────────────────────────────────
# Market History Combined Table (The image request)
# ─────────────────────────────────────────────

def compute_market_history_combined(prices_dict: dict, periods: list = [10, 20, 50], lookback: int = 40, **kwargs) -> pd.DataFrame:
    """
    Gộp dữ liệu MA, A/D, Supply/Demand vào một bảng lịch sử.
    Thêm cột nhận xét (Nhận xét/Label) dựa trên tương quan Index và Độ rộng.
    """
    # 1. Lấy dữ liệu thành phần
    ma_hist = compute_ma_history(prices_dict, periods, lookback)
    ad_hist = compute_ad_history(prices_dict, lookback)
    pw_hist = compute_market_power_history(prices_dict, lookback)
    
    # 2. Lấy dữ liệu Aggressive thực tế (nếu có)
    agg_results = kwargs.get('agg_results', {})
    
    if ma_hist.empty:
        return pd.DataFrame()
        
    # 2. Merge dữ liệu
    df = ma_hist.copy()
    if not ad_hist.empty:
        df = pd.merge(df, ad_hist, on='date', how='left')
    if not pw_hist.empty:
        df = pd.merge(df, pw_hist, on='date', how='left')
        
    # 2b. Nếu có agg_results, ghi đè giá trị Supply/Demand/Power của ngày cuối cùng bằng dữ liệu chuẩn
    if agg_results and not df.empty:
        total_buy_vol = 0
        total_sell_vol = 0
        total_power = 0.0
        
        for ticker, raw_data in agg_results.items():
             stats = compute_aggressive_stats(raw_data)
             total_buy_vol += stats.get('buy_volume', 0)
             total_sell_vol += stats.get('sell_volume', 0)
             total_power += stats.get('net_value', 0) # proxy for power
        
        # Ghi đè vào dòng cuối (phiên hiện tại) trước khi sort DESC
        today_idx = df.index[-1]
        df.at[today_idx, 'supply'] = total_sell_vol / 1e6
        df.at[today_idx, 'demand'] = total_buy_vol / 1e6
        df.at[today_idx, 'supply_demand'] = (total_buy_vol - total_sell_vol) / 1e6 # Triệu CP
        df.at[today_idx, 'power'] = total_power / 1e6 # proxy unit

    # 3. Tính toán cột Change (Index) và RSI
    if 'VNINDEX' in df.columns:
        df['Change'] = df['VNINDEX'].diff()
        df['rsi'] = compute_rsi(df['VNINDEX'], period=14)
    else:
        df['Change'] = 0
        df['rsi'] = 0
        
    # 4. Logic Nhận xét (Labels)
    def label_market(row):
        try:
            chg = row.get('Change', 0)
            adv = row.get('advances', 0)
            dec = row.get('declines', 0)
            net_sd = row.get('supply_demand', 0)
            pct_ma20 = row.get('pct_ma20', 50)
            rsi = row.get('rsi', 50)
            
            labels = []
            # RSI Signals
            if rsi >= 70: labels.append("OVERBOUGHT (RSI)")
            if rsi <= 30: labels.append("OVERSOLD (RSI)")

            # Market Breath Signals
            if chg > 3 and adv < dec:
                labels.append("BLUECHIP PUMP")
            elif chg < -3 and adv > dec:
                labels.append("BLUECHIP DUMP")
            
            if chg > 5 and adv > dec * 1.5:
                labels.append("BREADTH EXPANSION 🟢")
            elif chg < -10 and dec > adv * 2:
                labels.append("BREADTH CRASH 🔴")
            
            if net_sd > 200: labels.append("STRONG BUYING")
            if net_sd < -200: labels.append("STRONG SELLING")

            return " | ".join(labels) if labels else ""
        except:
            return ""

    df['Notes'] = df.apply(label_market, axis=1)
    
    # 5. Format lại thứ tự cột giống ảnh mẫu
    # DATE | Change | VNIND | MA10 | MA20 | MA50 | S/D (Net) | Demand | Supply | Power
    cols_to_keep = ['date', 'Change', 'VNINDEX', 'rsi', 'Notes']
    for p in periods:
        cols_to_keep.append(f'count_ma{p}')
    
    for c in ['supply_demand', 'demand', 'supply', 'power']:
        if c in df.columns:
            cols_to_keep.append(c)
        
    df = df[cols_to_keep].sort_values('date', ascending=False)
    
    # Friendly column names
    rename_map = {
        'date': 'Date',
        'VNINDEX': 'VNINDEX',
        'rsi': 'RSI',
        'supply_demand': 'Net S/D',
        'demand': 'Demand (M)',
        'supply': 'Supply (M)',
        'power': 'Power'
    }
    for p in periods:
        rename_map[f'count_ma{p}'] = f'MA{p}'
        
    df = df.rename(columns=rename_map)
    return df


def compute_aggressive_stats(data: list) -> dict:
    """
    Tính toán tổng các loại thanh khoản từ dữ liệu Aggressive Trading
    - Tổng GTGD (Turnover): sum(Price * TotalVolume)
    - Tổng Mua Chủ động (Agg Buy Value): sum(Price * AggressiveBuyingVolume)
    - Tổng Bán Chủ động (Agg Sell Value): sum(Price * AggressiveSellingVolume)
    """
    # Xử lý trường hợp data là dict {"data": [...]} từ API
    if isinstance(data, dict) and 'data' in data:
        data = data['data']

    if not data or not isinstance(data, list):
        return {
            'total_value': 0,
            'buy_value': 0,
            'sell_value': 0,
            'net_value': 0,
            'buy_volume': 0,
            'sell_volume': 0,
            'total_value_bn': 0,
            'net_value_bn': 0
        }
    
    total_val = 0.0
    buy_val = 0.0
    sell_val = 0.0
    buy_vol = 0
    sell_vol = 0
    
    for item in data:
        if not isinstance(item, dict):
            continue
            
        p = float(item.get('Price', 0))
        tv = float(item.get('TotalVolume', 0))
        bv = float(item.get('AggressiveBuyingVolume', 0))
        sv = float(item.get('AggressiveSellingVolume', 0))
        
        # VPS API: Giá chia cho đơn vị VND? Thường là VND.
        # Nếu Price là 25350 = 25,350 VND.
        # GTGD = Price * Volume
        total_val += p * tv
        buy_val += p * bv
        sell_val += p * sv
        buy_vol += bv
        sell_vol += sv
        
    return {
        'total_value': total_val,
        'buy_value': buy_val,
        'sell_value': sell_val,
        'net_value': buy_val - sell_val,
        'buy_volume': buy_vol,
        'sell_volume': sell_vol,
        'total_value_bn': total_val / 1e9, # Tỷ VNĐ
        'net_value_bn': (buy_val - sell_val) / 1e9
    }


def compute_liquidity_map(agg_results: dict) -> dict:
    """Xử lý kết quả batch fetch aggressive → map {ticker: turnover_bn}"""
    liq_map = {}
    for ticker, raw_data in agg_results.items():
        # raw_data có thể là dict {"data": [...]}
        data_list = raw_data.get('data', []) if isinstance(raw_data, dict) else raw_data
        stats = compute_aggressive_stats(data_list)
        if stats['total_value_bn'] > 0:
            liq_map[ticker] = round(stats['total_value_bn'], 3)
    return liq_map


def compute_sector_liquidity(prices_dict: dict, agg_liq_map: dict = {}) -> pd.DataFrame:
    """Tính thanh khoản theo dòng tiền ngành"""
    sector_data = []
    
    # 1. Tính mapping ngược (Ticker -> Sector)
    ticker_to_sector = {}
    for sector, tickers in SECTOR_MAP.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    # 2. Gom nhóm thanh khoản
    stats = {} # {sector: {'value': 0, 'count': 0}}
    
    for ticker, data in prices_dict.items():
        sector = ticker_to_sector.get(ticker)
        if not sector: continue
        
        # Lấy thanh khoản (ưu tiên aggressive)
        liq = agg_liq_map.get(ticker)
        if liq is None:
            # Tính trung bình 5 phiên gần nhất
            vals = np.array(data.get('close', [])) * np.array(data.get('volume', [])) / 1e6 # Tỷ VNĐ
            liq = np.mean(vals[-5:]) if len(vals) > 0 else 0
            
        if sector not in stats:
            stats[sector] = {'value': 0, 'count': 0}
        stats[sector]['value'] += liq
        stats[sector]['count'] += 1
        
    for sector, s in stats.items():
        sector_data.append({
            'Ngành': sector,
            'GTGD (Tỷ)': round(s['value'], 1),
            'Số mã': s['count'],
            'Bình quân/Mã': round(s['value'] / s['count'], 1) if s['count'] > 0 else 0
        })
        
    return pd.DataFrame(sector_data).sort_values('GTGD (Tỷ)', ascending=False)


    return pd.DataFrame(sector_data).sort_values('GTGD (Tỷ)', ascending=False)


def compute_market_treemap(prices_dict: dict) -> pd.DataFrame:
    """
    Chuẩn bị dữ liệu cho Treemap
    Returns DataFrame: [Ticker, Sector, Value, Change %]
    """
    ticker_to_sector = {}
    for sector, tickers in SECTOR_MAP.items():
        for t in tickers:
            ticker_to_sector[t] = sector
            
    treemap_data = []
    for ticker, data in prices_dict.items():
        if is_index_ticker(ticker):
            continue
            
        closes = data.get('close', [])
        volumes = data.get('volume', [])
        
        if len(closes) < 2:
            continue
            
        last_price = closes[-1]
        prev_price = closes[-2]
        change_pct = ((last_price - prev_price) / prev_price * 100) if prev_price else 0
        
        # Value (Size) can be (Last Price * Last Volume) or just Last Volume
        # For treemap usually trading value is better
        last_vol = volumes[-1] if volumes else 0
        value = last_price * last_vol
        
        sector = ticker_to_sector.get(ticker, "Others")
        
        if value > 0: # Only show active ones
            treemap_data.append({
                'Ticker': ticker,
                'Sector': sector,
                'Value': value,
                'Volume': last_vol,
                'Change %': round(change_pct, 2)
            })
            
    return pd.DataFrame(treemap_data)


def compute_mfi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Money Flow Index (MFI)"""
    tp = (df['high'] + df['low'] + df['close']) / 3
    rmf = tp * df['volume']
    
    pos_mf = []
    neg_mf = []
    
    for i in range(len(tp)):
        if i == 0:
            pos_mf.append(0)
            neg_mf.append(0)
        else:
            if tp[i] > tp[i-1]:
                pos_mf.append(rmf[i])
                neg_mf.append(0)
            elif tp[i] < tp[i-1]:
                pos_mf.append(0)
                neg_mf.append(rmf[i])
            else:
                pos_mf.append(0)
                neg_mf.append(0)
                
    pos_mf_sum = pd.Series(pos_mf).rolling(window=period).sum()
    neg_mf_sum = pd.Series(neg_mf).rolling(window=period).sum()
    
    mfr = pos_mf_sum / neg_mf_sum
    mfi = 100 - (100 / (1 + mfr))
    return mfi


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume (OBV)"""
    return (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()


def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """Volume Weighted Average Price (VWAP)"""
    return (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
