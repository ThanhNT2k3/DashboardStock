# Polymarket API Reference

All endpoints are **public** (no API key required for read-only market and
trader data). Collateral is USDC on Polygon (PoS). All monetary amounts in
API responses are in **USDC** (e.g. `"profit": 125000.0` means $125,000).

---

## Base URLs

| Service | Base URL |
|---------|----------|
| **Data API** | `https://data-api.polymarket.com` |
| **CLOB API** | `https://clob.polymarket.com` |
| **Gamma API** | `https://gamma-api.polymarket.com` |
| **WebSocket** | `wss://ws-subscriptions-clob.polymarket.com` |

---

## Data API — Trader Endpoints

### GET /activity

Returns all on-chain trading activity for a wallet.

**URL:** `https://data-api.polymarket.com/activity`

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `user` | string | required | Wallet address (0x…) |
| `limit` | int | 100 | Max records per page (max 500) |
| `offset` | int | 0 | Pagination offset |
| `startTs` | int | — | Unix timestamp (start of window) |
| `endTs` | int | — | Unix timestamp (end of window) |
| `type` | string | — | Filter: `TRADE`, `REDEEM`, `SPLIT`, `MERGE` |

**Response fields (per record):**

```json
{
  "id": "...",
  "type": "TRADE",
  "timestamp": 1700000000,
  "market": "0xMARKET...",
  "asset": "TOKEN_ID",
  "side": "BUY" | "SELL",
  "price": 0.72,
  "size": 1000,
  "usdcSize": 720.0,
  "outcome": "Yes" | "No",
  "transactionHash": "0xTX..."
}
```

**Pagination:** Increment `offset` by `limit` until fewer records than `limit`
are returned.

---

### GET /positions

Returns open and closed positions for a wallet.

**URL:** `https://data-api.polymarket.com/positions`

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `user` | string | required | Wallet address |
| `sizeThreshold` | float | 0 | Min position size in USDC |
| `redeemed` | bool | — | Filter to redeemed (closed) only |
| `market` | string | — | Filter to specific market ID |
| `limit` | int | 100 | Max records (max 500) |
| `offset` | int | 0 | Pagination offset |

**Response fields (per position):**

```json
{
  "market": "0xMARKET...",
  "asset": "TOKEN_ID",
  "outcome": "Yes",
  "size": 5000,
  "avgPrice": 0.68,
  "initialValue": 3400.0,
  "currentValue": 4500.0,
  "cashPnl": 1100.0,
  "percentPnl": 32.35,
  "redeemed": true,
  "redeemedSize": 5000,
  "title": "Will X happen?",
  "endDate": "2025-11-05"
}
```

---

### GET /portfolio

Returns aggregated portfolio statistics for a wallet.

**URL:** `https://data-api.polymarket.com/portfolio`

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `user` | string | Wallet address (required) |

**Response fields:**

```json
{
  "address": "0x...",
  "profit": 125000.0,
  "volume": 980000.0,
  "tradesCount": 342,
  "marketsTraded": 89,
  "winRate": 0.613,
  "positionsWon": 54,
  "positionsLost": 34,
  "positionsPending": 1
}
```

---

### GET /leaderboard (via Gamma)

**URL:** `https://gamma-api.polymarket.com/leaderboard`

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `window` | string | `monthly` \| `alltime` |
| `limit` | int | Number of results (default 100) |
| `offset` | int | Pagination |
| `metric` | string | `profit` \| `volume` \| `winrate` |

**Response:** Array of `{ address, username, profit, volume, winRate, tradesCount }`

---

## CLOB API — Order & Trade Endpoints

### GET /data/trades

Returns individual matched trades (more granular than activity).

**URL:** `https://clob.polymarket.com/data/trades`

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `maker_address` | string | Filter by maker wallet |
| `taker_address` | string | Filter by taker wallet |
| `market` | string | Market condition ID |
| `before` | string | ISO timestamp upper bound |
| `after` | string | ISO timestamp lower bound |
| `limit` | int | Max records (default 100, max 500) |
| `cursor` | string | Pagination cursor from previous response |

**Response fields (per trade):**

```json
{
  "id": "...",
  "taker_order_id": "...",
  "maker_order_id": "...",
  "market": "CONDITION_ID",
  "asset_id": "TOKEN_ID",
  "side": "BUY",
  "size": "1000",
  "fee_rate_bps": "0",
  "price": "0.72",
  "status": "MATCHED",
  "match_time": "2025-10-01T12:00:00Z",
  "last_update": "2025-10-01T12:00:01Z",
  "outcome": "Yes",
  "maker_address": "0x...",
  "transaction_hash": "0x..."
}
```

---

### GET /prices-history

Returns price time-series for a specific outcome token.

**URL:** `https://clob.polymarket.com/prices-history`

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `market` | string | Condition ID |
| `startTs` | int | Unix timestamp |
| `endTs` | int | Unix timestamp |
| `fidelity` | int | Data resolution in minutes (default 60) |

**Response:** `{ history: [ { t: unix_ts, p: 0.72 }, ... ] }`

---

## Gamma API — Market Metadata

### GET /markets

Fetch metadata for markets (title, category, end date, resolution).

**URL:** `https://gamma-api.polymarket.com/markets`

**Query Parameters:**

| Param | Type | Description |
|-------|------|-------------|
| `condition_ids` | string | Comma-separated list of condition IDs |
| `tag_slug` | string | Category filter (e.g., `politics`, `sports`, `crypto`) |
| `active` | bool | Only active markets |
| `limit` | int | Max results (default 100) |
| `offset` | int | Pagination |

**Key response fields:**

```json
{
  "id": "...",
  "conditionId": "0x...",
  "question": "Will X happen?",
  "description": "...",
  "startDate": "2025-01-01",
  "endDate": "2025-11-05",
  "resolvedPrice": 1,
  "volume": 2500000,
  "liquidity": 45000,
  "tags": ["politics", "us-elections"],
  "active": false,
  "closed": true
}
```

---

## Rate Limits

| Tier | Limit | Notes |
|------|-------|-------|
| Public (no auth) | ~100 req/min | Sufficient for analysis scripts |
| Authenticated | ~1000 req/min | For trading bots |

**Best practice:** Add `time.sleep(0.6)` between paginated requests to stay
well within public limits. Use exponential back-off (1 s, 2 s, 4 s) on 429s.

---

## On-Chain Fallback (Polygon)

If API data is incomplete, raw trades live on Polygon (chain ID 137).

- **CTF Exchange contract:** `0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E`
- **USDC (PoS):** `0x2791Bca1f2de4661ED88A30C99A7a9449Aa84174`

Tools: Polygonscan API, Dune Analytics (`polymarket` dataset), The Graph.

---

## Useful Third-Party Analytics Endpoints

| Tool | URL | Notes |
|------|-----|-------|
| PolymarketAnalytics | `https://polymarketanalytics.com` | Leaderboard, charts |
| Dune Analytics | `https://dune.com/browse/dashboards?q=polymarket` | SQL over raw chain data |
| PolyTrack | `https://polytrackhq.app` | Whale tracker, copy-trade alerts |
