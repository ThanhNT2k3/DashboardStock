# Polymarket Analyst — Metrics Reference

This file defines every metric computed by `scripts/analyze_trader.py`.
All monetary values are in USD unless stated otherwise.

---

## Table of Contents

1. [Performance Metrics](#1-performance-metrics)
2. [Behavioral Metrics](#2-behavioral-metrics)
3. [Market Concentration](#3-market-concentration)
4. [Timing Metrics](#4-timing-metrics)
5. [Risk Metrics](#5-risk-metrics)
6. [Strategy Signals](#6-strategy-signals)

---

## 1. Performance Metrics

These measure raw profitability and trading effectiveness.

### total_realized_pnl
**Definition:** Sum of `cashPnl` across all *redeemed* (closed) positions.
**Formula:** `Σ (redeem_value – cost_basis)` for each closed position
**Unit:** USD
**Interpretation:** The ground truth of how much real money the trader made.
Only counts settled markets; open positions are excluded to avoid mark-to-market
noise.

---

### win_rate
**Definition:** Fraction of closed positions where cashPnl > 0.
**Formula:** `positions_won / (positions_won + positions_lost)`
**Range:** 0–1 (report as %)
**Caveat:** A 90% win rate on tiny wins and 10% loss rate on huge losses is
actually a losing strategy — always pair with profit_factor.

---

### profit_factor
**Definition:** Gross profit divided by gross loss (absolute).
**Formula:** `Σ max(cashPnl, 0) / |Σ min(cashPnl, 0)|`
**Interpretation:**
- < 1.0 → losing overall
- 1.0–1.5 → marginal edge
- 1.5–2.5 → solid edge
- > 2.5 → exceptional or concentrated

---

### expectancy
**Definition:** Average P&L per closed position.
**Formula:** `total_realized_pnl / total_closed_positions`
**Unit:** USD per trade
**Use:** Quickly see if the trader makes money on average per bet.

---

### total_volume
**Definition:** Sum of all USDC deployed across all BUY trades.
**Formula:** `Σ usdcSize` for `type=TRADE, side=BUY`
**Unit:** USD

---

### roi_on_volume
**Definition:** Return on invested capital.
**Formula:** `total_realized_pnl / total_volume`
**Range:** typically –1 to +∞

---

### largest_win / largest_loss
**Definition:** Single position with highest / lowest cashPnl.
**Unit:** USD
**Use:** Reveals if the trader bets big on conviction or spreads risk.

---

### avg_position_size
**Definition:** Mean USDC deployed per BUY trade.
**Formula:** `total_volume / total_buy_trades`

---

### position_size_cv (coefficient of variation)
**Definition:** `std(position_sizes) / mean(position_sizes)`
**Interpretation:**
- Low CV (< 0.5) → consistent sizing (disciplined Kelly-style)
- High CV (> 1.5) → sizing varies wildly (opportunistic or emotional)

---

## 2. Behavioral Metrics

These reveal *how* the trader operates, not just their results.

### avg_hold_time_days
**Definition:** Mean time (days) between first BUY and position resolution.
**Formula:** `mean(resolved_ts – first_buy_ts)` per position
**Interpretation:**
- < 1 day → intraday / news trader
- 1–7 days → short-term
- 7–30 days → medium-term
- > 30 days → long-term / bond-style

---

### markets_per_category
**Definition:** Count and P&L per Gamma category tag.
**Categories (Polymarket taxonomy):** politics, sports, crypto, finance,
entertainment, science, world-events
**Use:** Reveals specialization vs. diversification.

---

### herfindahl_index (category concentration)
**Definition:** Sum of squared market-share fractions by category volume.
**Formula:** `Σ (vol_category_i / total_volume)²`
**Range:** 0–1
**Interpretation:**
- < 0.15 → highly diversified
- 0.15–0.4 → moderate concentration
- > 0.4 → specialist (single-domain dominance)

---

### repeat_market_rate
**Definition:** Fraction of markets traded more than once.
**Formula:** `markets_with_multiple_trades / total_markets_traded`
**Interpretation:** High repeat rate → systematic re-entry, scale-in, or
averaging. Low repeat rate → opportunistic one-shot bets.

---

### avg_entry_probability
**Definition:** Mean price paid on BUY orders (proxy for implied probability).
**Formula:** `mean(price)` across BUY trades
**Interpretation:**
- < 0.15 → long-shot gambler
- 0.15–0.4 → contrarian / value hunter
- 0.4–0.6 → near-50/50 (news-sensitive)
- 0.6–0.85 → probability-weighted bets
- > 0.85 → high-probability bond strategy

---

### yes_no_ratio
**Definition:** Fraction of BUY volume on YES tokens vs. NO tokens.
**Formula:** `vol_yes_buys / (vol_yes_buys + vol_no_buys)`
**Interpretation:** Strongly biased toward YES may mean the trader lacks
experience with NO tokens; a balanced ratio suggests strategic flexibility.

---

### exit_timing_score
**Definition:** How close to the peak price did the trader sell?
**Formula:** `mean(sell_price / max_price_before_sell)` across SELL trades
**Range:** 0–1 (1 = perfect top)
**Caveat:** Requires price-history data. Report as N/A if unavailable.

---

## 3. Market Concentration

### top_5_markets_by_pnl
**Definition:** The five individual markets that contributed most to total P&L.
**Fields:** market title, category, P&L contribution, % of total P&L.
**Use:** Identify whether the trader has a "hero trade" that explains most of
their success (concentration risk).

---

### top_category_win_rate
**Definition:** Win rate in the trader's top category by volume.
**Use:** Check whether their edge is real (high win rate in specialty) or they
just happened to be in a big market.

---

### geographic_political_bias
**Definition:** If politics is a major category, break down by country/region
(US, UK, EU, etc.) to detect country specialization.

---

## 4. Timing Metrics

### entry_lag_vs_market_creation_days
**Definition:** Mean days after market creation date that the trader first buys.
**Interpretation:**
- Near 0 → early entry / information advantage
- Days to weeks → research-driven
- Near market end → bond or near-sure-thing strategy

---

### news_correlation_score
**Definition:** Fraction of entries that occur within 24 hours of a major
price move (>10% in one hour) in that market.
**Formula:** requires price-history data
**Interpretation:** High score → news trader or insider-like timing.

---

### time_of_day_distribution
**Definition:** Distribution of trade timestamps by hour (UTC).
**Use:** Night-heavy trading may indicate automated bots; trading around 9 am ET
or 4 pm ET may correlate with US news cycles.

---

### day_of_week_distribution
**Definition:** Trade count per weekday (Mon–Sun).
**Use:** Bots trade flat; humans show weekday bias.

---

## 5. Risk Metrics

### max_drawdown_pct
**Definition:** Largest peak-to-trough decline in cumulative realized P&L over time.
**Formula:** `min((pnl_t – running_max_pnl_t) / running_max_pnl_t)` across time
**Use:** Reveals whether the trader blew up during a bad stretch.

---

### sharpe_ratio_approx
**Definition:** Approximation of risk-adjusted return using monthly P&L buckets.
**Formula:** `mean(monthly_pnl) / std(monthly_pnl) * sqrt(12)`
**Caveat:** Prediction market P&L is lumpy and non-Gaussian. Treat as
directional indicator only, not a precise financial metric.
**Interpretation:**
- < 0 → risk-adjusted losing
- 0–1 → modest edge
- 1–2 → solid edge
- > 2 → exceptional (or too few observations)

---

### max_concurrent_open_positions
**Definition:** Maximum number of unresolved positions open at the same time.
**Interpretation:** High concurrency → diversification; low concurrency →
concentrated conviction bets.

---

### kelly_fraction_estimate
**Definition:** Estimated Kelly criterion bet size fraction for this trader.
**Formula:** `expectancy / (avg_win_size / avg_loss_size)` (simplified Kelly)
**Use:** Compare to actual avg_position_size / bankroll to detect over/under
betting relative to Kelly.

---

## 6. Strategy Signals

These are computed signals — not raw metrics — that feed strategy attribution
in references/strategies.md.

| Signal | Formula / Source | Type |
|--------|-----------------|------|
| `bond_signal` | avg_entry_probability > 0.88 | bool |
| `early_entry_signal` | entry_lag < 2 days AND news_correlation > 0.4 | bool |
| `domain_specialist_signal` | herfindahl_index > 0.45 AND top_category_win_rate > 0.6 | bool |
| `market_maker_signal` | yes_no_ratio between 0.4–0.6 AND avg_hold_time < 1 day | bool |
| `arb_signal` | multiple platforms in activity OR very consistent small gains | bool |
| `size_consistency_signal` | position_size_cv < 0.4 | bool |
| `bot_signal` | time_of_day distribution is near-uniform OR avg_hold < 1 hr | bool |
| `lucky_variance_flag` | total_closed_positions < 30 AND win_rate > 0.8 | warn |
| `insider_timing_flag` | news_correlation_score > 0.7 AND entry_lag < 1 hr | warn |

Signals labelled `warn` are reported as caution flags in the output, not as
strategy labels.
