# polymarket-analyst-skills


> A Claude skill for deep analysis of Polymarket prediction market trader accounts.
> Fetches live on-chain data, computes performance metrics, and classifies trading strategies.

---

## What This Skill Does

Give it a Polymarket wallet address and it will:

1. **Fetch** all trading history via the public Polymarket API (no key required)
2. **Compute** 25+ performance, behavioral, timing, and risk metrics
3. **Classify** the trader's strategy (one of 6 canonical Polymarket strategies)
4. **Generate** a structured analyst report with evidence-backed conclusions

It can also scan the leaderboard and compare multiple accounts side by side.

---

## The Six Strategies It Detects

| Strategy | Description |
|----------|-------------|
| **Information Arbitrage** | Enters before the market prices in private/early information |
| **Cross-Platform Arbitrage** | Exploits price gaps between Polymarket and other platforms |
| **High-Probability Bond** | Buys near-certain outcomes (>88¢) for yield |
| **Market Making** | Posts bid/ask to capture spread, often automated |
| **Domain Specialist** | Deep expertise in one category (sports, politics…) |
| **Speed / News Trading** | Enters within minutes of breaking news |

---

## Installation

### Option A — Claude Code (recommended)
Drop the `polymarket-analyst-skills/` folder into your Claude skills directory:
```
~/.claude/skills/polymarket-analyst-skills/
```

### Option B — Cowork Mode
Same as above — place the folder under `~/.claude/skills/` and Claude Code will
pick it up automatically on next startup.

### Option C — Manual usage
Run the scripts directly:
```bash
# 1. Fetch data for a trader
python3 scripts/fetch_trader_data.py \
  --address 0xYourAddressHere \
  --output  /tmp/poly_data/trader1

# 2. Analyze and get strategy attribution
python3 scripts/analyze_trader.py \
  --data-dir /tmp/poly_data/trader1 \
  --output   /tmp/poly_data/trader1/analysis.json \
  --print

# 3. Fetch the leaderboard
python3 scripts/fetch_trader_data.py \
  --leaderboard \
  --limit 100 \
  --output /tmp/poly_leaderboard
```

**Requirements:** Python 3.9+ with `pip install requests python-dateutil`

---

## Example Prompts (when used as a Claude skill)

```
Analyze this Polymarket wallet: 0xabc123...
```
```
Who are the top 10 traders on Polymarket and what's their strategy?
```
```
Compare these two wallets and tell me which has better risk-adjusted returns.
```
```
Is this trader using insider information? Their win rate seems too high.
```
```
Find me Polymarket accounts that specialize in political markets.
```

---

## File Structure

```
polymarket-analyst-skills/
├── SKILL.md                    ← Main skill instructions (for Claude)
├── README.md                   ← This file (for humans)
├── references/
│   ├── api.md                  ← Full Polymarket API documentation
│   ├── metrics.md              ← Definition of all 25+ metrics
│   └── strategies.md           ← Strategy classification logic & thresholds
├── scripts/
│   ├── fetch_trader_data.py    ← Fetches data from Polymarket APIs
│   ├── analyze_trader.py       ← Computes metrics & produces analysis.json
│   └── requirements.txt        ← Python dependencies (requests, python-dateutil)
└── evals/
    └── evals.json              ← 5 test cases for skill evaluation
```

---

## How It Works

### Data Sources (all public, no authentication)

| Source | URL | What it provides |
|--------|-----|-----------------|
| Data API | `data-api.polymarket.com` | Activity, positions, portfolio P&L |
| Gamma API | `gamma-api.polymarket.com` | Market metadata, leaderboard |
| CLOB API | `clob.polymarket.com` | Price history, order book data |

All data lives on Polygon (chain ID 137). You can verify everything on
[Polygonscan](https://polygonscan.com) or [Dune Analytics](https://dune.com).

### Metrics Computed

Performance: total P&L, win rate, profit factor, expectancy, ROI, largest win/loss, position size coefficient of variation.

Behavioral: avg entry probability, YES/NO ratio, hold time, repeat market rate, market concentration (Herfindahl index).

Timing: entry lag vs. market creation, hour-of-day distribution, day-of-week distribution, bot detection (chi-squared uniformity test).

Risk: max drawdown, approximate Sharpe ratio, max concurrent open positions, monthly P&L buckets.

### Strategy Scoring

Each of the 6 strategies receives a score (0–100) based on weighted signal combinations. The highest-scoring strategy becomes the primary label. A confidence level (High/Medium/Low) and any secondary strategies are also reported.

---

## Caveats & Ethics

- This skill uses **public on-chain data only**. No private APIs or proprietary data.
- Strategy classification is **probabilistic**, not definitive. A high win rate alone does not prove insider trading.
- The skill includes a **lucky variance warning** for accounts with fewer than 30 closed positions.
- When the insider timing flag triggers, the report presents the observation **without accusation** — on-chain data cannot prove intent.
- Copy-trading based on this analysis carries significant risk. Past performance does not guarantee future results.

---

## Contributing

Pull requests are welcome. Key areas for improvement:

- Cross-platform arbitrage detection (requires Kalshi / Manifold API integration)
- News event correlation (requires external news API)
- Portfolio-level Kelly fraction estimation
- Multi-wallet portfolio consolidation (for traders using multiple addresses)

Please open an issue before starting major work.

---

## License

MIT — see [LICENSE](LICENSE)

---

## Background: Polymarket Market Mechanics

Polymarket is a decentralized prediction market on Polygon where users trade binary outcome contracts priced between $0 and $1. A $0.72 YES contract implies a 72% chance of the outcome. All trades settle in USDC via a hybrid Central Limit Order Book (CLOB): orders are matched off-chain, settled on-chain.

As of 2026, only ~7.6% of wallets are profitable, and the top 0.51% account for the vast majority of gains. The six strategies in this skill are derived from an analysis of 95 million on-chain transactions by Polymarket's own research team.
