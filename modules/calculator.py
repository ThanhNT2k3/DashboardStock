"""
Market Breadth & Sentiment Calculator
Tính toán các chỉ số phân tích độ rộng thị trường
"""

import numpy as np
import pandas as pd
from datetime import datetime


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
    if ma_history.empty: return pd.DataFrame()
    
    df = ma_history.copy()
    # Tính RSI cho các cột % MA
    for col in df.columns:
        if col.startswith('pct_ma'):
            df[f'rsi_{col}'] = compute_rsi(df[col], period=14)
            df[f'psy_{col}'] = compute_psy(df[col], period=12)
            
    return df


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


def filter_by_liquidity(prices_dict: dict, min_value_bn: float = 0.0, days: int = 20) -> dict:
    """
    Lọc danh sách tickers theo GTGD trung bình `days` phiên gần nhất
    min_value_bn: Tỷ VNĐ
    """
    if min_value_bn <= 0:
        return prices_dict

    filtered = {}
    for ticker, data in prices_dict.items():
        # Luôn giữ các chỉ số index
        if ticker in ['VNINDEX', 'HNXINDEX', 'UPINDEX', 'VN30', 'HNX30']:
            filtered[ticker] = data
            continue

        closes  = data.get('close', [])
        volumes = data.get('volume', [])
        
        n = min(len(closes), len(volumes), days)
        if n > 0:
            # Giá (nghìn đồng) * Khối lượng / 1.000.000 = Tỷ đồng
            avg_val = np.mean([closes[i] * volumes[i] for i in range(-n, 0)])
            if avg_val / 1e6 >= min_value_bn:
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
