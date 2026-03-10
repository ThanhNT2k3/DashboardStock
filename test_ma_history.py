import time
import pandas as pd
import numpy as np
from datetime import datetime

def is_index_ticker(ticker):
    return ticker in ['VNINDEX', 'HNXINDEX', 'UPINDEX', 'VN30', 'VN100', 'HNX30']

def compute_ma_history_new(prices_dict, periods, lookback):
    all_ts = set()
    for data in prices_dict.values():
        all_ts.update(data.get('timestamps', []))
    sorted_ts = sorted(list(all_ts))[-lookback:]
    if not sorted_ts:
        return pd.DataFrame()

    closes_list = {}
    ma_lists = {p: {} for p in periods}

    for ticker, data in prices_dict.items():
        if is_index_ticker(ticker): continue
        close = data.get('close', [])
        ts = data.get('timestamps', [])
        if len(close) < min(periods): continue
        
        c_series = pd.Series(close, index=ts)
        closes_list[ticker] = c_series
        
        for p in periods:
            if len(close) >= p:
                ma_series = c_series.rolling(window=p).mean()
                ma_lists[p][ticker] = ma_series

    if not closes_list:
        return pd.DataFrame()

    df_closes = pd.DataFrame(closes_list)
    df_closes.sort_index(inplace=True)
    df_closes.ffill(inplace=True)

    df_mas = {}
    for p in periods:
        if ma_lists[p]:
            df_m = pd.DataFrame(ma_lists[p])
            df_m.sort_index(inplace=True)
            df_m.ffill(inplace=True)
            df_mas[p] = df_m
        else:
            df_mas[p] = pd.DataFrame()

    history = []
    for t in sorted_ts:
        if t not in df_closes.index:
            continue
        row = {'date': datetime.fromtimestamp(t).strftime('%Y-%m-%d')}
                
        c_series = df_closes.loc[t]
        for p in periods:
            if df_mas[p].empty or t not in df_mas[p].index:
                row[f'count_ma{p}'] = 0
                row[f'pct_ma{p}'] = 0.0
                continue
                
            m_series = df_mas[p].loc[t]
            valid_mask = m_series.notna() & c_series.notna()
            total = valid_mask.sum()
            above = (c_series[valid_mask] > m_series[valid_mask]).sum()
            
            row[f'count_ma{p}'] = int(above)
            row[f'pct_ma{p}'] = round(above / total * 100, 1) if total else 0.0
        history.append(row)
     
    return pd.DataFrame(history)

# mockup data
prices_dict = {}
np.random.seed(42)
ts_base = list(range(1000, 1000 + 86400*200, 86400)) # 200 days
for i in range(1000):
    # simulate missing days randomly
    mask = np.random.rand(len(ts_base)) > 0.1
    ts_partial = np.array(ts_base)[mask].tolist()
    closes_partial = (np.random.rand(len(ts_partial)) * 100).tolist()
    
    prices_dict[f'TICKER_{i}'] = {
        'timestamps': ts_partial,
        'close': closes_partial
    }

st = time.time()
df = compute_ma_history_new(prices_dict, [10, 20, 50], 60)
print("new speed:", time.time() - st)
# Output last row
print(df.tail(1))
