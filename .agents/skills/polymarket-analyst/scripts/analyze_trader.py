#!/usr/bin/env python3
"""
analyze_trader.py
=================
Computes all performance metrics and strategy attribution signals for a
Polymarket trader from pre-fetched JSON data (produced by fetch_trader_data.py).

Usage:
    python3 analyze_trader.py \
        --data-dir /tmp/poly_data/0xABCD \
        --output   /tmp/poly_data/0xABCD/analysis.json \
        [--print]

Output:
    analysis.json — structured dict of all metrics and strategy signals
    (if --print) — a human-readable summary printed to stdout

Requirements:
    pip install python-dateutil
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_json(path: Path, default=None):
    if not path.exists():
        return default if default is not None else []
    with open(path) as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return default if default is not None else []


def ts_to_dt(ts) -> Optional[datetime]:
    """Convert a unix timestamp or ISO string to a UTC datetime."""
    if ts is None:
        return None
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts, tz=timezone.utc)
    if isinstance(ts, str):
        try:
            from dateutil.parser import parse as dp
            dt = dp(ts)
            if dt.tzinfo is None:
                return dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception:
            return None
    return None


def safe_div(num, den, default=0.0):
    return num / den if den else default


def safe_mean(lst):
    return sum(lst) / len(lst) if lst else 0.0


def safe_std(lst):
    if len(lst) < 2:
        return 0.0
    m = safe_mean(lst)
    variance = sum((x - m) ** 2 for x in lst) / len(lst)
    return math.sqrt(variance)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_dir: Path) -> dict:
    portfolio = load_json(data_dir / "portfolio.json", {})
    activity  = load_json(data_dir / "activity.json", [])
    positions = load_json(data_dir / "positions.json", [])
    markets_raw = load_json(data_dir / "markets.json", [])

    # Build market lookup by conditionId
    markets = {}
    for m in markets_raw:
        cid = m.get("conditionId") or m.get("id") or m.get("condition_id", "")
        if cid:
            markets[cid] = m

    # Load price histories
    prices = {}  # conditionId → list of {t, p}
    prices_dir = data_dir / "prices"
    if prices_dir.exists():
        for f in prices_dir.glob("*.json"):
            hist = load_json(f, [])
            if hist:
                prices[f.stem] = hist  # stem is first 16 chars of conditionId

    return {
        "portfolio": portfolio,
        "activity": activity,
        "positions": positions,
        "markets": markets,
        "prices": prices,
    }


# ── Core computations ─────────────────────────────────────────────────────────

def compute_performance(portfolio: dict, positions: list, activity: list) -> dict:
    """Performance metrics from references/metrics.md §1."""

    # Use portfolio API summary as the primary source
    total_pnl = portfolio.get("profit", 0.0) or 0.0
    win_rate   = portfolio.get("winRate", None)
    pos_won    = portfolio.get("positionsWon", 0)
    pos_lost   = portfolio.get("positionsLost", 0)
    pos_pending = portfolio.get("positionsPending", 0)
    trades_count = portfolio.get("tradesCount", len(activity))
    total_volume_api = portfolio.get("volume", 0.0)

    # Augment from positions list (cashPnl field)
    closed_pnls = [
        p.get("cashPnl", 0.0) or 0.0
        for p in positions
        if p.get("redeemed") or p.get("closed")
    ]
    if not closed_pnls:
        # fall back to any position with a non-zero cashPnl
        closed_pnls = [p.get("cashPnl", 0.0) or 0.0 for p in positions if p.get("cashPnl")]

    wins   = [p for p in closed_pnls if p > 0]
    losses = [p for p in closed_pnls if p < 0]

    total_pnl_computed = sum(closed_pnls) if closed_pnls else total_pnl
    profit_factor = safe_div(sum(wins), abs(sum(losses)))

    # Volume from activity
    buy_trades = [a for a in activity if str(a.get("side","")).upper() == "BUY"
                  and str(a.get("type","")).upper() in ("TRADE","")]
    sell_trades = [a for a in activity if str(a.get("side","")).upper() == "SELL"
                   and str(a.get("type","")).upper() in ("TRADE","")]

    buy_usdc = [
        float(a.get("usdcSize") or a.get("amount") or 0)
        for a in buy_trades
    ]
    total_volume_computed = sum(buy_usdc)
    total_volume = total_volume_api or total_volume_computed

    sizes = [s for s in buy_usdc if s > 0]
    avg_position_size = safe_mean(sizes)
    position_size_cv  = safe_div(safe_std(sizes), avg_position_size)

    largest_win  = max(wins)  if wins  else 0.0
    largest_loss = min(losses) if losses else 0.0
    expectancy   = safe_div(total_pnl_computed, len(closed_pnls))
    roi          = safe_div(total_pnl_computed, total_volume)

    return {
        "total_realized_pnl":    round(total_pnl_computed, 2),
        "win_rate":               round(win_rate or safe_div(len(wins), len(wins) + len(losses)), 4),
        "positions_won":         pos_won  or len(wins),
        "positions_lost":        pos_lost or len(losses),
        "positions_pending":     pos_pending,
        "total_closed_positions": len(closed_pnls) or pos_won + pos_lost,
        "total_volume":          round(total_volume, 2),
        "roi_on_volume":         round(roi, 4),
        "profit_factor":         round(profit_factor, 2),
        "expectancy":            round(expectancy, 2),
        "largest_win":           round(largest_win, 2),
        "largest_loss":          round(largest_loss, 2),
        "avg_position_size":     round(avg_position_size, 2),
        "position_size_cv":      round(position_size_cv, 3),
        "total_buy_trades":      len(buy_trades),
        "total_sell_trades":     len(sell_trades),
    }


def compute_behavior(activity: list, positions: list, markets: dict) -> dict:
    """Behavioral metrics from references/metrics.md §2."""

    buy_trades = [a for a in activity if str(a.get("side","")).upper() == "BUY"
                  and str(a.get("type","")).upper() in ("TRADE","")]
    prices_paid = [float(a.get("price", 0)) for a in buy_trades if a.get("price")]
    avg_entry_prob = safe_mean(prices_paid)

    # YES vs NO volume
    yes_vol = sum(
        float(a.get("usdcSize") or 0)
        for a in buy_trades
        if str(a.get("outcome","")).upper() == "YES"
    )
    no_vol  = sum(
        float(a.get("usdcSize") or 0)
        for a in buy_trades
        if str(a.get("outcome","")).upper() == "NO"
    )
    yes_no_ratio = safe_div(yes_vol, yes_vol + no_vol, default=0.5)

    # Hold times from positions
    hold_days = []
    for p in positions:
        if not (p.get("redeemed") or p.get("closed")):
            continue
        first_buy_ts = p.get("firstBuyTimestamp") or p.get("createdAt")
        resolve_ts   = p.get("resolvedAt") or p.get("endDate")
        dt_buy     = ts_to_dt(first_buy_ts)
        dt_resolve = ts_to_dt(resolve_ts)
        if dt_buy and dt_resolve and dt_resolve > dt_buy:
            hold_days.append((dt_resolve - dt_buy).total_seconds() / 86400)
    avg_hold_time_days = safe_mean(hold_days)

    # Repeat market rate
    market_counts = Counter(
        a.get("market", "") or a.get("conditionId", "")
        for a in activity
        if a.get("market") or a.get("conditionId")
    )
    markets_traded = len(market_counts)
    repeat_markets = sum(1 for c in market_counts.values() if c > 1)
    repeat_market_rate = safe_div(repeat_markets, markets_traded)

    return {
        "avg_entry_probability": round(avg_entry_prob, 4),
        "yes_volume":            round(yes_vol, 2),
        "no_volume":             round(no_vol, 2),
        "yes_no_ratio":          round(yes_no_ratio, 3),
        "avg_hold_time_days":    round(avg_hold_time_days, 2),
        "total_markets_traded":  markets_traded,
        "repeat_market_rate":    round(repeat_market_rate, 3),
    }


def compute_market_concentration(activity: list, positions: list, markets: dict) -> dict:
    """Market concentration metrics from references/metrics.md §3."""

    # Category breakdown by volume and P&L
    cat_volume: dict[str, float] = defaultdict(float)
    cat_pnl:    dict[str, float] = defaultdict(float)
    cat_wins:   dict[str, int]   = defaultdict(int)
    cat_total:  dict[str, int]   = defaultdict(int)

    for a in activity:
        cid = a.get("market") or a.get("conditionId", "")
        meta = markets.get(cid, {})
        tags = meta.get("tags") or meta.get("categories") or ["unknown"]
        if isinstance(tags, list) and tags:
            cat = tags[0] if isinstance(tags[0], str) else tags[0].get("slug", "unknown")
        else:
            cat = str(tags) if tags else "unknown"
        size = float(a.get("usdcSize") or 0)
        if str(a.get("side","")).upper() == "BUY":
            cat_volume[cat] += size

    for p in positions:
        cid = p.get("market") or p.get("conditionId", "")
        meta = markets.get(cid, {})
        tags = meta.get("tags") or meta.get("categories") or ["unknown"]
        if isinstance(tags, list) and tags:
            cat = tags[0] if isinstance(tags[0], str) else tags[0].get("slug", "unknown")
        else:
            cat = "unknown"
        pnl = float(p.get("cashPnl") or 0)
        cat_pnl[cat] += pnl
        if p.get("redeemed") or p.get("closed"):
            cat_total[cat] += 1
            if pnl > 0:
                cat_wins[cat] += 1

    total_vol = sum(cat_volume.values()) or 1.0
    hhi = sum((v / total_vol) ** 2 for v in cat_volume.values())

    top5_vol = sorted(cat_volume.items(), key=lambda x: -x[1])[:5]
    top5_pnl = sorted(cat_pnl.items(), key=lambda x: -x[1])[:5]

    category_breakdown = {
        cat: {
            "volume":   round(vol, 2),
            "pct_volume": round(100 * vol / total_vol, 1),
            "pnl":      round(cat_pnl.get(cat, 0), 2),
            "win_rate": round(safe_div(cat_wins[cat], cat_total[cat]), 3)
                        if cat_total[cat] else None,
        }
        for cat, vol in top5_vol
    }

    # Top category for specialist detection
    top_cat = top5_vol[0][0] if top5_vol else "unknown"
    top_cat_win_rate = safe_div(cat_wins[top_cat], cat_total[top_cat]) if cat_total[top_cat] else None

    return {
        "herfindahl_index":       round(hhi, 4),
        "top_category":           top_cat,
        "top_category_win_rate":  round(top_cat_win_rate, 3) if top_cat_win_rate else None,
        "category_breakdown_vol": category_breakdown,
        "top5_by_pnl":            [{"category": c, "pnl": round(p, 2)} for c, p in top5_pnl],
    }


def compute_timing(activity: list, markets: dict) -> dict:
    """Timing metrics from references/metrics.md §4."""

    hour_dist:   list[int] = [0] * 24
    dow_dist:    list[int] = [0] * 7  # Mon=0 … Sun=6
    entry_lags:  list[float] = []

    for a in activity:
        if str(a.get("side","")).upper() != "BUY":
            continue

        ts = a.get("timestamp") or a.get("createdAt")
        dt = ts_to_dt(ts)
        if dt:
            hour_dist[dt.hour] += 1
            dow_dist[dt.weekday()] += 1

        # Entry lag vs. market creation
        cid  = a.get("market") or a.get("conditionId", "")
        meta = markets.get(cid, {})
        start = meta.get("startDate") or meta.get("createdAt")
        dt_market = ts_to_dt(start)
        if dt and dt_market:
            lag = (dt - dt_market).total_seconds() / 86400
            if 0 <= lag < 365:
                entry_lags.append(lag)

    # Uniformity of hour distribution (bot detection)
    total_trades = sum(hour_dist)
    if total_trades > 0:
        expected = total_trades / 24
        chi2 = sum((h - expected) ** 2 / expected for h in hour_dist if expected > 0)
        # Low chi2 → uniform → bot-like; high chi2 → clustered → human-like
        hour_uniformity_chi2 = round(chi2, 2)
    else:
        hour_uniformity_chi2 = None

    return {
        "avg_entry_lag_days":     round(safe_mean(entry_lags), 2) if entry_lags else None,
        "median_entry_lag_days":  round(sorted(entry_lags)[len(entry_lags)//2], 2) if entry_lags else None,
        "hour_distribution_utc":  hour_dist,
        "day_of_week_distribution": dow_dist,
        "hour_uniformity_chi2":   hour_uniformity_chi2,  # low = bot-like
    }


def compute_risk(positions: list, activity: list) -> dict:
    """Risk metrics from references/metrics.md §5."""

    # Cumulative P&L over time → max drawdown
    closed_with_ts = []
    for p in positions:
        if not (p.get("redeemed") or p.get("closed")):
            continue
        pnl = float(p.get("cashPnl") or 0)
        ts  = p.get("resolvedAt") or p.get("endDate")
        dt  = ts_to_dt(ts)
        if dt:
            closed_with_ts.append((dt, pnl))
    closed_with_ts.sort(key=lambda x: x[0])

    cum_pnl  = 0.0
    peak     = 0.0
    max_dd   = 0.0
    for _, pnl in closed_with_ts:
        cum_pnl += pnl
        if cum_pnl > peak:
            peak = cum_pnl
        if peak > 0:
            dd = (cum_pnl - peak) / peak
            if dd < max_dd:
                max_dd = dd

    # Sharpe (monthly buckets)
    monthly: dict[str, float] = defaultdict(float)
    for dt, pnl in closed_with_ts:
        key = dt.strftime("%Y-%m")
        monthly[key] += pnl
    monthly_values = list(monthly.values())
    if len(monthly_values) >= 3:
        mean_m = safe_mean(monthly_values)
        std_m  = safe_std(monthly_values)
        sharpe = safe_div(mean_m, std_m) * math.sqrt(12) if std_m else None
    else:
        sharpe = None

    # Max concurrent open positions (approximate from activity)
    # Sort buy/sell events by time; track open count
    events = []
    for a in activity:
        ts  = a.get("timestamp") or a.get("createdAt")
        dt  = ts_to_dt(ts)
        side = str(a.get("side","")).upper()
        if dt and side in ("BUY", "SELL"):
            events.append((dt, 1 if side == "BUY" else -1))
    events.sort(key=lambda x: x[0])
    open_count = 0
    max_open   = 0
    for _, delta in events:
        open_count = max(0, open_count + delta)
        if open_count > max_open:
            max_open = open_count

    return {
        "max_drawdown_pct":             round(max_dd * 100, 2),
        "sharpe_ratio_approx":          round(sharpe, 2) if sharpe else None,
        "max_concurrent_open_positions": max_open,
        "monthly_pnl_buckets":          {k: round(v, 2) for k, v in sorted(monthly.items())},
    }


def compute_strategy_signals(perf: dict, behavior: dict, concentration: dict,
                              timing: dict, risk: dict) -> dict:
    """
    Compute all boolean strategy signals and derive a strategy attribution.
    See references/strategies.md for the scoring logic.
    """

    # ── Raw signals ──────────────────────────────────────────────────────────
    avg_entry_prob  = behavior["avg_entry_probability"]
    hhi             = concentration["herfindahl_index"]
    top_cat_wr      = concentration.get("top_category_win_rate") or 0.0
    hold            = behavior["avg_hold_time_days"]
    yes_no          = behavior["yes_no_ratio"]
    win_rate        = perf["win_rate"]
    n_closed        = perf["total_closed_positions"]
    pf              = perf["profit_factor"]
    size_cv         = perf["position_size_cv"]
    entry_lag       = timing.get("avg_entry_lag_days") or 999
    hour_chi2       = timing.get("hour_uniformity_chi2") or 9999
    max_dd          = abs(risk.get("max_drawdown_pct") or 0)

    bond_signal              = avg_entry_prob > 0.88
    early_entry_signal       = entry_lag < 2
    domain_specialist_signal = hhi > 0.45 and (top_cat_wr or 0.0) > 0.60
    market_maker_signal      = (0.40 <= yes_no <= 0.60) and hold < 1.0
    size_consistency_signal  = size_cv < 0.40
    bot_signal               = hour_chi2 < 10.0 and hold < 0.5

    # Warnings
    lucky_variance_flag = n_closed < 30 and win_rate > 0.80
    insider_timing_flag = entry_lag < 0.1 and win_rate > 0.70

    # ── Strategy scoring ─────────────────────────────────────────────────────
    scores = {
        "Information Arbitrage":      (30 * int(early_entry_signal)
                                       + 20 * int(win_rate > 0.65)
                                       + 15 * int(not domain_specialist_signal)),
        "Cross-Platform Arbitrage":   (30 * int(pf > 3.0 and max_dd < 5)
                                       + 25 * int(size_consistency_signal)
                                       + 20 * int(hold < 3.0)),
        "High-Probability Bond":      (35 * int(bond_signal)
                                       + 20 * int(win_rate > 0.80)
                                       + 10 * int(hold > 7)),
        "Market Making":              (35 * int(market_maker_signal)
                                       + 20 * int(bot_signal)
                                       + 15 * int(hold < 0.5)),
        "Domain Specialist":          (35 * int(domain_specialist_signal)
                                       + 25 * int(hhi > 0.45)
                                       + 20 * int((top_cat_wr or 0) > 0.60)),
        "Speed / News Trading":       (30 * int(early_entry_signal and entry_lag < 0.5)
                                       + 25 * int(hold < 5)
                                       + 15 * int(not bot_signal)),
    }

    primary   = max(scores, key=scores.get)
    primary_score = scores[primary]
    secondaries = [k for k, v in scores.items() if v > 25 and k != primary]

    confidence = "High" if primary_score > 70 else ("Medium" if primary_score > 50 else "Low")

    return {
        "bond_signal":              bond_signal,
        "early_entry_signal":       early_entry_signal,
        "domain_specialist_signal": domain_specialist_signal,
        "market_maker_signal":      market_maker_signal,
        "size_consistency_signal":  size_consistency_signal,
        "bot_signal":               bot_signal,
        "lucky_variance_flag":      lucky_variance_flag,
        "insider_timing_flag":      insider_timing_flag,
        "strategy_scores":          scores,
        "primary_strategy":         primary,
        "primary_strategy_score":   primary_score,
        "strategy_confidence":      confidence,
        "secondary_strategies":     secondaries,
    }


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze(data_dir: Path, address_hint: str = None) -> dict:
    d = load_data(data_dir)

    address = (
        d["portfolio"].get("address")
        or address_hint
        or data_dir.name
    )

    perf    = compute_performance(d["portfolio"], d["positions"], d["activity"])
    behav   = compute_behavior(d["activity"], d["positions"], d["markets"])
    conc    = compute_market_concentration(d["activity"], d["positions"], d["markets"])
    timing  = compute_timing(d["activity"], d["markets"])
    risk    = compute_risk(d["positions"], d["activity"])
    signals = compute_strategy_signals(perf, behav, conc, timing, risk)

    return {
        "address":        address,
        "snapshot_date":  datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "performance":    perf,
        "behavior":       behav,
        "market_concentration": conc,
        "timing":         timing,
        "risk":           risk,
        "strategy_signals": signals,
    }


def print_summary(result: dict) -> None:
    p = result["performance"]
    b = result["behavior"]
    c = result["market_concentration"]
    r = result["risk"]
    s = result["strategy_signals"]

    print("\n" + "=" * 60)
    print(f"  POLYMARKET TRADER ANALYSIS")
    print(f"  Address: {result['address']}")
    print(f"  Date:    {result['snapshot_date']}")
    print("=" * 60)

    print("\n── PERFORMANCE ────────────────────────────────────────────")
    print(f"  Total Realized P&L:     ${p['total_realized_pnl']:>12,.2f}")
    print(f"  Win Rate:               {p['win_rate']*100:>8.1f}%")
    print(f"  Profit Factor:          {p['profit_factor']:>8.2f}")
    print(f"  Expectancy / trade:     ${p['expectancy']:>12,.2f}")
    print(f"  Total Volume:           ${p['total_volume']:>12,.2f}")
    print(f"  ROI on Volume:          {p['roi_on_volume']*100:>8.2f}%")
    print(f"  Closed Positions:       {p['total_closed_positions']:>8}")
    print(f"  Largest Win:            ${p['largest_win']:>12,.2f}")
    print(f"  Largest Loss:           ${p['largest_loss']:>12,.2f}")
    print(f"  Avg. Position Size:     ${p['avg_position_size']:>12,.2f}")

    print("\n── BEHAVIOR ────────────────────────────────────────────────")
    print(f"  Avg. Entry Probability: {b['avg_entry_probability']*100:>8.1f}%")
    print(f"  YES / NO Ratio:         {b['yes_no_ratio']:>8.3f}")
    print(f"  Avg. Hold Time:         {b['avg_hold_time_days']:>8.1f} days")
    print(f"  Markets Traded:         {b['total_markets_traded']:>8}")
    print(f"  Repeat Market Rate:     {b['repeat_market_rate']*100:>8.1f}%")

    print("\n── MARKET CONCENTRATION ────────────────────────────────────")
    print(f"  Herfindahl Index:       {c['herfindahl_index']:>8.4f}  (1=monopoly)")
    print(f"  Top Category:           {c['top_category']}")
    print(f"  Top Cat Win Rate:       "
          + (f"{c['top_category_win_rate']*100:.1f}%" if c['top_category_win_rate'] else "N/A"))
    if c.get("category_breakdown_vol"):
        print("  By Volume (top 5):")
        for cat, data in list(c["category_breakdown_vol"].items())[:5]:
            print(f"    {cat:<18} ${data['volume']:>10,.0f}  ({data['pct_volume']:.0f}%)  "
                  f"WR: {(data['win_rate']*100):.0f}%" if data['win_rate'] is not None
                  else f"    {cat:<18} ${data['volume']:>10,.0f}  ({data['pct_volume']:.0f}%)")

    print("\n── RISK ────────────────────────────────────────────────────")
    print(f"  Max Drawdown:           {r['max_drawdown_pct']:>8.2f}%")
    print(f"  Sharpe Ratio (approx):  "
          + (f"{r['sharpe_ratio_approx']:>8.2f}" if r['sharpe_ratio_approx'] else "   N/A (< 3 months)"))
    print(f"  Max Concurrent Open:    {r['max_concurrent_open_positions']:>8}")

    print("\n── STRATEGY ATTRIBUTION ────────────────────────────────────")
    print(f"  Primary Strategy:  {s['primary_strategy']}")
    print(f"  Confidence:        {s['strategy_confidence']}  (score: {s['primary_strategy_score']})")
    if s["secondary_strategies"]:
        print(f"  Secondary:         {', '.join(s['secondary_strategies'])}")
    print()

    if s["lucky_variance_flag"]:
        print("  ⚠  WARNING: Fewer than 30 closed positions — results may reflect luck.")
    if s["insider_timing_flag"]:
        print("  ⚠  NOTE: Entry timing pattern warrants attention (see report).")

    print("\n── ALL STRATEGY SCORES ─────────────────────────────────────")
    for strat, score in sorted(s["strategy_scores"].items(), key=lambda x: -x[1]):
        bar = "█" * (score // 5)
        print(f"  {strat:<30} {score:>3}  {bar}")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Analyze Polymarket trader data")
    p.add_argument("--data-dir", required=True, help="Directory with fetched JSON files")
    p.add_argument("--output",   required=True, help="Output path for analysis.json")
    p.add_argument("--print",    action="store_true", help="Print human-readable summary")
    p.add_argument("--address",  default=None, help="Trader address (if not in portfolio.json)")
    return p.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: data directory not found: {data_dir}")
        raise SystemExit(1)

    result = analyze(data_dir, address_hint=args.address)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)

    if args.print:
        print_summary(result)
    else:
        print(f"Analysis saved to: {out}")
        print(f"Primary strategy: {result['strategy_signals']['primary_strategy']} "
              f"({result['strategy_signals']['strategy_confidence']})")


if __name__ == "__main__":
    main()
