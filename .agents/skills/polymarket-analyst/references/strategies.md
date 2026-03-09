# Polymarket Trading Strategy Classification

This file documents the six canonical Polymarket profit strategies, their
theoretical basis, the metric thresholds used to detect them, and the evidence
quality expected in the analyst report.

---

## How Classification Works

After computing the metric signals from references/metrics.md:

1. Score each strategy (0–100) using the weighted signal table below.
2. The strategy with the highest score > 40 becomes the **primary** label.
3. Any strategy scoring > 25 is listed as a **secondary signal**.
4. If no strategy scores above 40, label the trader **Unclassified** and note
   insufficient data or mixed behavior.
5. Always report a confidence level: **High** (>70), **Medium** (50–70),
   **Low** (<50).

---

## Strategy 1 — Information Arbitrage

**Core Idea:** The trader holds information the market hasn't yet priced in —
a private poll, an expert network, early access to data. They enter before the
market moves and exit when the crowd catches up.

**Real example:** A French trader made ~$85M in 2024 by commissioning a
"neighbor effect" poll during the US election that showed a different result
than public polls. He bought YES on Trump at ~$0.47 before the market moved
to >$0.90.

### Detection Signals

| Signal | Weight | Direction |
|--------|--------|-----------|
| `early_entry_signal` (enters < 2 days after market creation) | 30 | + |
| `news_correlation_score` > 0.5 | 25 | + |
| `entry_lag_vs_market_creation_days` < 3 | 20 | + |
| Large single positions on `politics` or `world-events` | 15 | + |
| `win_rate` > 0.65 in those specific markets | 10 | + |

### Report Narrative Template
"This trader consistently enters markets significantly before any major price
movement, suggesting access to private information or early expert analysis.
Their largest trades cluster around [category] and precede price moves of
[X]% on average. This is consistent with an information arbitrage strategy."

---

## Strategy 2 — Cross-Platform Arbitrage

**Core Idea:** The same event is priced differently on Polymarket vs. Kalshi,
Manifold, PredictIt, or other platforms. The trader buys cheap on one and
sells (or bets opposite) on the other, locking in near-riskless profit.

**Risk warning:** Resolution criteria often differ across platforms, making
supposed arbitrage positions unexpectedly directional. The 2024 US shutdown
example (Polymarket vs. Kalshi different resolution standards) is a canonical
cautionary tale.

### Detection Signals

| Signal | Weight | Direction |
|--------|--------|-----------|
| `profit_factor` > 3 with low `max_drawdown` (< 5%) | 30 | + |
| Very consistent small gains (low variance of position P&L) | 25 | + |
| `avg_hold_time_days` < 3 | 20 | + |
| `position_size_cv` < 0.3 (systematic sizing) | 15 | + |
| Active on many markets simultaneously | 10 | + |

### Report Narrative Template
"This trader shows hallmarks of systematic arbitrage: highly consistent small
gains with very low drawdown, mechanical position sizing, and brief hold times
across many markets. Their returns resemble a yield strategy rather than
directional betting, consistent with cross-platform or statistical arbitrage."

---

## Strategy 3 — High-Probability Bond Strategy

**Core Idea:** Buy contracts already trading at >$0.88–$0.95 (near-certainty)
and collect the residual yield as they drift to $1.00 at resolution. Low
upside per trade, but extremely high win rate and near-zero variance.

**Why it works:** Markets often underprice near-certainty due to liquidity
constraints and attention scarcity. A $0.93 contract on a 99% likely outcome
offers a ~7.5% yield with minimal risk — better than T-bills when scaled.

### Detection Signals

| Signal | Weight | Direction |
|--------|--------|-----------|
| `bond_signal` = true (`avg_entry_probability` > 0.88) | 35 | + |
| `win_rate` > 0.85 | 20 | + |
| `avg_hold_time_days` > 7 (wait for resolution) | 10 | + |
| Low `largest_win` relative to total volume (small per-trade yields) | 10 | + |

### Report Narrative Template
"The overwhelming majority of this trader's positions are entered at prices
above $0.88, indicating a high-probability 'bond' strategy. They accept small
yields (~5–12% per trade) in exchange for near-certain resolution, producing
a very high win rate with low variance — a capital-preservation approach more
akin to yield farming than speculative trading."

---

## Strategy 4 — Market Making / Liquidity Provision

**Core Idea:** Post bid and ask orders simultaneously, earn the spread.
Market makers don't predict outcomes — they profit from transaction flow.
Requires significant capital, tight risk controls, and automated tooling.

**Since early 2026:** Polymarket introduced post-only rebates on select
markets, making official liquidity provision more attractive.

### Detection Signals

| Signal | Weight | Direction |
|--------|--------|-----------|
| `market_maker_signal` (yes_no_ratio ≈ 0.5) | 35 | + |
| `avg_hold_time_days` < 1 (constant entry/exit) | 25 | + |
| `bot_signal` = true (near-uniform time distribution) | 20 | + |
| High `total_volume` relative to `total_realized_pnl` (low margin %) | 15 | + |
| Many markets with small positions (diversified spread capture) | 5 | + |

### Report Narrative Template
"This trader's activity pattern — balanced YES/NO exposure, very short hold
times, and near-uniform trading across the day — is consistent with automated
market making. Their revenue comes from capturing bid-ask spreads rather than
directional bets. Volume is very high relative to net profit, characteristic
of a market-making operation with tight margins."

---

## Strategy 5 — Domain Specialist

**Core Idea:** Deep expertise in one domain (baseball statistics, EU politics,
crypto protocol developments) allows the trader to model outcomes more
accurately than the crowd. They focus narrowly and build an information edge
through expertise rather than proprietary data.

**Real examples:**
- 1j59y6nk ($1.4M P&L) — baseball and football specialist
- Erasmus ($1.3M P&L) — US political polling specialist
- WindWalk3 ($1.1M P&L) — RFK/health policy specialist

### Detection Signals

| Signal | Weight | Direction |
|--------|--------|-----------|
| `domain_specialist_signal` = true | 35 | + |
| `herfindahl_index` > 0.45 | 25 | + |
| `top_category_win_rate` > 0.60 | 20 | + |
| Consistent long-term performance in that category | 15 | + |
| Below-average win rate in categories outside specialty | 5 | + |

### Report Narrative Template
"This trader concentrates [X]% of their volume in [category], where they
achieve a win rate of [Y]% — well above their [Z]% rate in other categories.
This divergence strongly suggests domain expertise rather than general market
skill. Their positions in [category] also show systematic entry patterns
consistent with a structured analytical framework."

---

## Strategy 6 — Speed / News Trading

**Core Idea:** Be the first to trade on breaking news. When a major event
happens, markets lag for minutes or hours. Fast traders (human or bot) capture
the gap between the new reality and the old price.

**Tools of the trade:** Custom news feeds, Twitter/X alerting, Telegram bots,
pre-staged limit orders, fast infrastructure near Polygon RPC nodes.

### Detection Signals

| Signal | Weight | Direction |
|--------|--------|-----------|
| `entry_lag_vs_market_creation_days` < 0.5 (enters within hours) | 30 | + |
| `news_correlation_score` > 0.6 | 25 | + |
| `avg_hold_time_days` < 5 (exits after market catches up) | 20 | + |
| `bot_signal` partially true (fast but not 24/7 uniform) | 15 | + |
| Spikes in `time_of_day_distribution` around US news hours | 10 | + |

### Report Narrative Template
"This trader's entry timestamps cluster within hours of breaking news or major
announcements. Their hold time is short — they appear to enter before the
market prices in news and exit once the price converges. Combined with the
timing patterns, this is consistent with a speed-trading or news-trading
operation, possibly partially automated."

---

## Edge Cases & Caveats

### Lucky Variance
If `lucky_variance_flag` is triggered (< 30 positions, high win rate), write:

> "Warning: This account has fewer than 30 closed positions. The high win
> rate may reflect luck rather than skill. Strategy classification is
> preliminary and should be revisited after more trading history accumulates."

### Insider Timing
If `insider_timing_flag` is triggered:

> "Note: Several trades entered within 1 hour of a significant price move,
> and before public information was widely available. While this may reflect
> exceptional monitoring or analytical speed, it is also consistent with
> material non-public information. This observation is included for
> completeness; it does not constitute an accusation of wrongdoing."

### Mixed / Hybrid Strategies
Many sophisticated traders combine strategies across market types. For example:
- Bond strategy on US politics (near-certain outcomes)
- Domain specialist on sports (where they have actual expertise)
- News trading on crypto (fast markets)

In this case, report all detected strategies with their respective category
concentrations and explain the hybrid nature clearly.
