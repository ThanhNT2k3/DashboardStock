"""
VPS API Fetcher
Async batch fetch với concurrency control
API: https://histdatafeed.vps.com.vn/tradingview/history
"""

import asyncio
import httpx
from datetime import datetime
from typing import Optional

BASE_URL = "https://histdatafeed.vps.com.vn/tradingview/history"
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
