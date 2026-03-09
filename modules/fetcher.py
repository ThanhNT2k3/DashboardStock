"""
VPS API Fetcher
Async batch fetch với concurrency control
API: https://histdatafeed.vps.com.vn/tradingview/history
"""

import asyncio
import httpx
import pandas as pd
import os
from datetime import datetime
from typing import Optional

BASE_URL = "https://histdatafeed.vps.com.vn/tradingview/history"
AGGRESSIVE_BASE_URL = "https://histdatafeed.vps.com.vn/volumeaggressivetrading"
VIX_API_URL = "https://litefinance.vn/trading/trading-instruments/chart/?symbol=VIX&period=1440&date_start=24"
GLOBAL_INDICES_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxchisothegioi.ashx"
GOODS_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxhanghoa.ashx?type=1"
PROP_BUY_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxgiaodichtudoanh.ashx?type=BUYVALUE"
PROP_SELL_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxgiaodichtudoanh.ashx?type=SELLVALUE"
FOREIGN_BUY_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxkhoingoai.ashx?type=buy"
FOREIGN_SELL_URL = "https://cafef.vn/du-lieu/ajax/mobile/smart/ajaxkhoingoai.ashx?type=sell"
MARKET_VALUATION_URL = "https://cafef.vn/du-lieu/Ajax/PageNew/FinanceData/GetDataChartPE.ashx"
CONCURRENCY = 20  # max concurrent requests
TIMEOUT = 15      # seconds per request


async def fetch_one(
    client: httpx.AsyncClient,
    ticker: str,
    start_date: str,
    end_date: str,
    sem: asyncio.Semaphore
) -> dict:
    """Fetch OHLCV data cho 1 mã"""
    async with sem:
        try:
            t_end   = int(datetime.strptime(end_date,   '%Y-%m-%d').timestamp())
            t_start = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp())
            url = (
                f"{BASE_URL}"
                f"?symbol={ticker}&resolution=1D"
                f"&from={t_start}&to={t_end}&countback=1000"
            )
            r = await client.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            raw = r.json()

            # VPS trả về: {s, t, o, h, l, c, v}
            # s = "ok" nếu có data
            if raw.get('s') != 'ok' or not raw.get('t'):
                return {"ticker": ticker, "ok": False, "data": None}

            return {
                "ticker": ticker,
                "ok": True,
                "data": {
                    "timestamps": raw['t'],
                    "open":       raw['o'],
                    "high":       raw['h'],
                    "low":        raw['l'],
                    "close":      raw['c'],
                    "volume":     raw['v'],
                }
            }
        except Exception as e:
            return {"ticker": ticker, "ok": False, "error": str(e), "data": None}


async def _batch_fetch_async(
    tickers: list,
    start_date: str,
    end_date: str,
    progress_callback=None
) -> list:
    """Internal async batch fetch"""
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []

    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_one(client, ticker, start_date, end_date, sem)
            for ticker in tickers
        ]

        completed = 0
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1
            if progress_callback:
                progress_callback(completed, len(tickers))

    return results


def batch_fetch(
    tickers: list,
    start_date: str,
    end_date: str,
    progress_callback=None
) -> list:
    """
    Sync wrapper — gọi từ Streamlit
    Trả về list[dict] với keys: ticker, ok, data
    """
    return asyncio.run(
        _batch_fetch_async(tickers, start_date, end_date, progress_callback)
    )

def fetch_vix() -> pd.DataFrame:
    """Fetch VIX data from LiteFinance API"""
    import requests
    try:
        r = requests.get(VIX_API_URL, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if raw.get('status') == 'success' and 'content' in raw:
            df = pd.DataFrame(raw['content'], columns=['timestamp', 'open', 'high', 'low', 'close'])
            df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('date', inplace=True)
            return df
    except Exception as e:
        print(f"Error fetching VIX: {e}")
    return pd.DataFrame()


def parse_results(raw_results: list) -> dict:
    """
    Convert list of raw results → dict[ticker] = processed data
    
    Returns:
        {
            ticker: {
                'close':  [float, ...],    # giá đóng cửa
                'volume': [float, ...],    # khối lượng
                'timestamps': [int, ...],  # unix timestamp
                'change_pct': float,       # % thay đổi phiên cuối
                'last_close': float,
                'last_volume': float,
            }
        }
    """
    processed = {}

    for item in raw_results:
        if not item.get('ok') or not item.get('data'):
            continue

        d = item['data']
        closes  = d.get('close', [])
        volumes = d.get('volume', [])
        timestamps = d.get('timestamps', [])

        if len(closes) < 2:
            continue

        change_pct = 0.0
        if closes[-2] and closes[-2] != 0:
            change_pct = (closes[-1] - closes[-2]) / closes[-2] * 100

        processed[item['ticker']] = {
            'close':      closes,
            'volume':     volumes,
            'timestamps': timestamps,
            'open':       d.get('open', []),
            'high':       d.get('high', []),
            'low':        d.get('low', []),
            'change_pct': round(change_pct, 2),
            'last_close': closes[-1],
            'last_volume': volumes[-1] if volumes else 0,
        }

    return processed


def fetch_aggressive_trading(ticker: str) -> list:
    """Fetch dữ liệu lệnh mua/bán chủ động theo giá của phiên hiện tại"""
    import requests
    try:
        url = f"{AGGRESSIVE_BASE_URL}/{ticker}"
        r = requests.get(url, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"Error fetching aggressive data for {ticker}: {e}")
        return []


async def fetch_aggressive_one(client: httpx.AsyncClient, ticker: str, sem: asyncio.Semaphore) -> dict:
    """Fetch aggressive trading cho 1 mã (Async)"""
    async with sem:
        try:
            url = f"{AGGRESSIVE_BASE_URL}/{ticker}"
            r = await client.get(url, timeout=TIMEOUT)
            r.raise_for_status()
            return {"ticker": ticker, "ok": True, "data": r.json()}
        except Exception as e:
            return {"ticker": ticker, "ok": False, "error": str(e)}


async def _batch_fetch_aggressive_async(tickers: list) -> list:
    sem = asyncio.Semaphore(CONCURRENCY)
    async with httpx.AsyncClient() as client:
        tasks = [fetch_aggressive_one(client, t, sem) for t in tickers]
        return await asyncio.gather(*tasks)


def batch_fetch_aggressive(tickers: list) -> dict:
    """Sync wrapper để lấy dữ liệu Aggressive cho hàng loạt mã"""
    raw_results = asyncio.run(_batch_fetch_aggressive_async(tickers))
    processed = {}
    for item in raw_results:
        if item.get('ok'):
            processed[item['ticker']] = item.get('data', [])
    return processed


def save_liquidity_cache(liquidity_map: dict, filename: str = "daily_liquidity.json"):
    """Lưu cache thanh khoản vào file để dùng cho bộ lọc"""
    import json, os
    data = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "values": liquidity_map
    }
    # Đảm bảo thư mục tồn tại nếu là đường dẫn
    os.makedirs(os.path.dirname(os.path.abspath(filename)) if os.path.dirname(filename) else ".", exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_liquidity_cache(filename: str = "daily_liquidity.json") -> dict:
    """Tải cache thanh khoản từ file"""
    import json, os
    if not os.path.exists(filename):
        return {}
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Chỉ lấy cache nếu trùng ngày hiện tại
            if data.get("date") == datetime.now().strftime("%Y-%m-%d"):
                return data.get("values", {})
    except:
        pass
    return {}


def save_market_summary_snapshot(df: pd.DataFrame, filename: str = "market_summary_history.csv"):
    """Lưu snapshot dữ liệu cuối ngày vào file lịch sử"""
    import os
    if df.empty:
        return
    
    # Chỉ lấy dòng đầu tiên (thường là ngày gần nhất)
    latest_row = df.iloc[[0]].copy()
    
    if not os.path.exists(filename):
        latest_row.to_csv(filename, index=False, encoding='utf-8-sig')
    else:
        # Đọc dữ liệu cũ để tránh trùng lặp ngày
        old_df = pd.read_csv(filename, encoding='utf-8-sig')
        
        # Hỗ trợ chuyển đổi từ file cũ (Ngày -> Date)
        if 'Ngày' in old_df.columns and 'Date' not in old_df.columns:
            old_df = old_df.rename(columns={'Ngày': 'Date'})
            
        # Xóa các dòng trùng Date trong file cũ
        # Kiểm tra xem latest_row có cột 'Date' hay 'Ngày' không (phòng hờ)
        date_col = 'Date' if 'Date' in latest_row.columns else ('Ngày' if 'Ngày' in latest_row.columns else None)
        
        if date_col:
            date_str = latest_row[date_col].iloc[0]
            # Đảm bảo old_df cũng so sánh đúng cột
            if 'Date' in old_df.columns:
                old_df = old_df[old_df['Date'] != date_str]
            elif 'Ngày' in old_df.columns:
                old_df = old_df[old_df['Ngày'] != date_str]
        
        # Nối dòng mới và lưu lại
        new_history = pd.concat([latest_row, old_df], ignore_index=True)
        # Giới hạn lưu 200 ngày gần nhất
        new_history.head(200).to_csv(filename, index=False, encoding='utf-8-sig')


def load_market_summary_history(filename: str = "market_summary_history.csv") -> pd.DataFrame:
    """Tải lịch sử snapshot đã được chốt từ trước"""
    import os
    if not os.path.exists(filename):
        return pd.DataFrame()
    try:
        df = pd.read_csv(filename, encoding='utf-8-sig')
        # Hỗ trợ chuyển đổi từ file cũ (Ngày -> Date)
        if 'Ngày' in df.columns and 'Date' not in df.columns:
            df = df.rename(columns={'Ngày': 'Date'})
        return df
    except:
        return pd.DataFrame()


def fetch_global_indices() -> pd.DataFrame:
    """Fetch global indices from CafeF API"""
    import requests
    try:
        r = requests.get(GLOBAL_INDICES_URL, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if raw.get('Success') and 'Data' in raw:
            df = pd.DataFrame(raw['Data'])
            # Clean up and rename
            df = df[['index', 'last', 'change', 'changePercent', 'lastUpdate']]
            df.columns = ['Index', 'Last', 'Change', '% Change', 'Updated']
            return df
    except Exception as e:
        print(f"Error fetching global indices: {e}")
    return pd.DataFrame()


def fetch_commodities() -> pd.DataFrame:
    """Fetch commodity prices from CafeF API"""
    import requests
    try:
        r = requests.get(GOODS_URL, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if raw.get('Success') and 'Data' in raw:
            df = pd.DataFrame(raw['Data'])
            # Clean up and rename
            df = df[['goods', 'last', 'change', 'changePercent', 'last_update']]
            df.columns = ['Commodity', 'Last', 'Change', '% Change', 'Updated']
            return df
    except Exception as e:
        print(f"Error fetching commodities: {e}")
    return pd.DataFrame()


def fetch_org_trading(org_type: int, symbol: str = "VNINDEX", limit: int = 20) -> pd.DataFrame:
    """
    Fetch Foreign or Proprietary trading data.
    org_type: 0 for Foreign, 1 for Proprietary
    """
    import requests
    from datetime import datetime
    today_str = datetime.now().strftime('%Y%m%d')
    url = f"https://msh-appdata.cafef.vn/rest-api/api/v1/OverviewOrgnizaztion/{org_type}/{today_str}/{limit}?symbol={symbol}"
    
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            df = pd.DataFrame(data)
            # Process date
            df['date'] = pd.to_datetime(df['date']).dt.strftime('%d/%m')
            # Convert values to Billion VND
            df['netVal_bn'] = df['netVal'] / 1e9
            # Keep only necessary columns for the chart/table
            return df[['date', 'buyVal', 'sellVal', 'netVal', 'netVal_bn', 'buyVol', 'sellVol', 'netVol']]
    except Exception as e:
        print(f"Error fetching org trading (type={org_type}): {e}")
    return pd.DataFrame()


def fetch_top_trading_stocks(url: str) -> pd.DataFrame:
    """Fetch top stocks from a given CafeF trading API URL"""
    import requests
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if raw.get('Success') and 'Data' in raw:
            df = pd.DataFrame(raw['Data'])
            if not df.empty:
                # Standardize columns
                df = df[['Symbol', 'Value', 'CurrentPrice', 'ChangePricePercent']]
                df['Value_bn'] = df['Value'] / 1e9
                return df
    except Exception as e:
        print(f"Error fetching top trading stocks: {e}")
    return pd.DataFrame()


def fetch_market_valuation() -> dict:
    """Fetch market valuation data (P/E, P/B, Cap) and historical chart data"""
    import requests
    try:
        r = requests.get(MARKET_VALUATION_URL, timeout=10)
        r.raise_for_status()
        raw = r.json()
        if raw.get('Data'):
            data = raw['Data']
            # NowDataFinance, PastDataFinance, DataChart
            if 'DataChart' in data:
                df = pd.DataFrame(data['DataChart'])
                # Convert TimeStamp to datetime
                df['Date'] = pd.to_datetime(df['TimeStamp'], unit='s')
                df = df.sort_values('Date')
                data['history_df'] = df
            return data
    except Exception as e:
        print(f"Error fetching market valuation: {e}")
    return {}
