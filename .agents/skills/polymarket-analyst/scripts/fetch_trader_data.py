#!/usr/bin/env python3
"""
fetch_trader_data.py
====================
Fetches all public Polymarket data for a given trader wallet address
and saves it to a structured JSON output directory.

Usage:
    python3 fetch_trader_data.py --address 0xABCD... --output /tmp/poly_data/0xABCD
    python3 fetch_trader_data.py --address 0xABCD... --output /tmp/out --days 90
    python3 fetch_trader_data.py --leaderboard --limit 50 --output /tmp/poly_leaderboard

Requirements:
    pip install requests python-dateutil
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: 'requests' not installed. Run: pip install requests")
    sys.exit(1)

# ── API base URLs ────────────────────────────────────────────────────────────
DATA_API  = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API  = "https://clob.polymarket.com"

# ── Default settings ─────────────────────────────────────────────────────────
PAGE_SIZE      = 500     # max records per paginated call
RETRY_ATTEMPTS = 3
RETRY_BACKOFF  = 1.5     # seconds; doubles on each retry
REQUEST_DELAY  = 0.6     # polite delay between requests (stay under rate limit)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def fetch_with_retry(url: str, params: dict = None) -> dict | list:
    """GET a URL with retry logic. Returns parsed JSON or raises."""
    for attempt in range(RETRY_ATTEMPTS):
        try:
            resp = requests.get(url, params=params, timeout=20)
            if resp.status_code == 429:
                wait = RETRY_BACKOFF * (2 ** attempt)
                log(f"Rate limited — waiting {wait:.1f}s then retrying…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == RETRY_ATTEMPTS - 1:
                raise
            wait = RETRY_BACKOFF * (2 ** attempt)
            log(f"Request error ({exc}) — retrying in {wait:.1f}s…")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {url} after {RETRY_ATTEMPTS} attempts")


def paginate(url: str, params: dict, key: str = None) -> list:
    """
    Paginate through all records using offset-based pagination.
    If `key` is given, extract that key from each response dict.
    """
    results = []
    offset = 0
    while True:
        p = {**params, "limit": PAGE_SIZE, "offset": offset}
        data = fetch_with_retry(url, p)
        # Data API returns a list directly; Gamma may return a dict with a key
        if isinstance(data, dict):
            records = data.get(key, data.get("results", data.get("data", [])))
        else:
            records = data
        results.extend(records)
        if len(records) < PAGE_SIZE:
            break
        offset += PAGE_SIZE
        time.sleep(REQUEST_DELAY)
    return results


# ── Fetchers ─────────────────────────────────────────────────────────────────

def fetch_portfolio(address: str) -> dict:
    log(f"Fetching portfolio summary for {address[:10]}…")
    url = f"{DATA_API}/portfolio"
    return fetch_with_retry(url, {"user": address})


def fetch_activity(address: str, start_ts: int = None, end_ts: int = None) -> list:
    log(f"Fetching activity for {address[:10]}…")
    params = {"user": address}
    if start_ts:
        params["startTs"] = start_ts
    if end_ts:
        params["endTs"] = end_ts
    return paginate(f"{DATA_API}/activity", params)


def fetch_positions(address: str) -> list:
    log(f"Fetching positions for {address[:10]}…")
    params = {"user": address, "sizeThreshold": 0}
    all_positions = paginate(f"{DATA_API}/positions", params)
    return all_positions


def fetch_market_metadata(condition_ids: list) -> list:
    """Batch-fetch market metadata from Gamma API (up to 50 IDs at a time)."""
    if not condition_ids:
        return []
    log(f"Fetching metadata for {len(condition_ids)} markets…")
    markets = []
    batch_size = 50
    for i in range(0, len(condition_ids), batch_size):
        batch = condition_ids[i : i + batch_size]
        params = {"condition_ids": ",".join(batch)}
        data = fetch_with_retry(f"{GAMMA_API}/markets", params)
        if isinstance(data, list):
            markets.extend(data)
        elif isinstance(data, dict):
            markets.extend(data.get("markets", []))
        time.sleep(REQUEST_DELAY)
    return markets


def fetch_leaderboard(metric: str = "profit", window: str = "alltime", limit: int = 100) -> list:
    log(f"Fetching leaderboard (metric={metric}, window={window}, limit={limit})…")
    params = {"metric": metric, "window": window, "limit": limit}
    data = fetch_with_retry(f"{GAMMA_API}/leaderboard", params)
    if isinstance(data, list):
        return data
    return data.get("data", data.get("traders", []))


def fetch_price_history(condition_id: str, start_ts: int, end_ts: int,
                         fidelity: int = 60) -> list:
    """Fetch OHLC-style price history for a market token."""
    params = {
        "market": condition_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }
    try:
        data = fetch_with_retry(f"{CLOB_API}/prices-history", params)
        return data.get("history", [])
    except Exception as exc:
        log(f"  Price history unavailable for {condition_id[:12]}: {exc}")
        return []


# ── Orchestration ─────────────────────────────────────────────────────────────

def fetch_trader(address: str, output_dir: Path, days: int = None) -> None:
    """
    Full data-fetch pipeline for a single trader address.
    Results are saved as JSON files in output_dir.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Time window
    now_ts = int(datetime.now(timezone.utc).timestamp())
    start_ts = (now_ts - days * 86400) if days else None

    # 1. Portfolio summary
    portfolio = fetch_portfolio(address)
    _save(output_dir / "portfolio.json", portfolio)

    # 2. Activity (all trades)
    activity = fetch_activity(address, start_ts=start_ts)
    _save(output_dir / "activity.json", activity)
    log(f"  {len(activity)} activity records fetched")

    # 3. Positions
    positions = fetch_positions(address)
    _save(output_dir / "positions.json", positions)
    log(f"  {len(positions)} positions fetched")

    # 4. Market metadata (deduplicated)
    condition_ids = list({
        rec.get("market") or rec.get("conditionId", "")
        for rec in activity + positions
        if rec.get("market") or rec.get("conditionId")
    })
    condition_ids = [c for c in condition_ids if c]  # remove empties
    markets = fetch_market_metadata(condition_ids)
    _save(output_dir / "markets.json", markets)
    log(f"  {len(markets)} market metadata records fetched")

    # 5. Price history (lazy — only most recent 20 most-active markets)
    prices_dir = output_dir / "prices"
    prices_dir.mkdir(exist_ok=True)
    active_markets = _most_active_markets(activity, n=20)
    for cid in active_markets:
        fname = prices_dir / f"{cid[:16]}.json"
        if fname.exists():
            continue  # skip already fetched
        hist = fetch_price_history(
            cid,
            start_ts=start_ts or (now_ts - 365 * 86400),
            end_ts=now_ts,
        )
        if hist:
            _save(fname, hist)
        time.sleep(REQUEST_DELAY)

    log(f"Done. Data saved to: {output_dir}")


def _most_active_markets(activity: list, n: int = 20) -> list:
    from collections import Counter
    counts = Counter(rec.get("market", "") for rec in activity if rec.get("market"))
    return [cid for cid, _ in counts.most_common(n) if cid]


def _save(path: Path, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)


def fetch_leaderboard_cmd(output_dir: Path, metric: str, limit: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    traders = fetch_leaderboard(metric=metric, limit=limit)
    _save(output_dir / "leaderboard.json", traders)
    log(f"Saved {len(traders)} trader profiles to {output_dir / 'leaderboard.json'}")
    # Print address list for easy downstream processing
    print("\nTop trader addresses:")
    for i, t in enumerate(traders[:20], 1):
        addr = t.get("address", "unknown")
        profit = t.get("profit", 0)
        print(f"  {i:2d}. {addr}  — profit: ${profit:,.0f}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Fetch Polymarket trader data to local JSON files"
    )
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--address", help="Polygon wallet address (0x…)")
    group.add_argument("--leaderboard", action="store_true",
                       help="Fetch the Polymarket leaderboard instead of a single trader")

    p.add_argument("--output", required=True,
                   help="Output directory for JSON files")
    p.add_argument("--days", type=int, default=None,
                   help="Limit to last N days (default: all-time)")
    p.add_argument("--limit", type=int, default=100,
                   help="Number of leaderboard entries (default: 100)")
    p.add_argument("--metric", default="profit",
                   choices=["profit", "volume", "winrate"],
                   help="Leaderboard sort metric (default: profit)")
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output)

    if args.leaderboard:
        fetch_leaderboard_cmd(out, metric=args.metric, limit=args.limit)
    else:
        if not args.address.startswith("0x"):
            print(f"WARNING: Address '{args.address}' doesn't start with '0x'. "
                  "Double-check it's a Polygon address.")
        fetch_trader(args.address.lower(), out, days=args.days)


if __name__ == "__main__":
    main()
