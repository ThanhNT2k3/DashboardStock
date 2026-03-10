import httpx
import pandas as pd
from datetime import datetime

WEBHOOK_URL = "https://discord.com/api/webhooks/1480606106689601587/cW0S1S9Rx013Hnuvt4ie-uSY3bu8lwh2-oZRnO9d0lLZ4QuwxMdEMmZSTNRYQ-pD6K-n"

async def send_market_summary_to_discord(stats: dict):
    """Gửi tổng kết thị trường cuối ngày qua Discord Webhook"""
    
    sentiment = stats.get('sentiment', {})
    ad = stats.get('ad_stats', {})
    hl = stats.get('hl_stats', {})
    vol_mom = stats.get('vol_momentum', 0)
    
    # Format message
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    embed = {
        "title": f"🚀 MARKET SUMMARY {datetime.now().strftime('%d/%m/%Y')}",
        "description": f"Automated EOD report at 15:00\nUpdated at: {now_str}",
        "color": 0x00B4D8 if stats.get('dom_trend') != "BEARISH" else 0xFF1744,
        "fields": [
            {
                "name": "📊 General Metrics",
                "value": f"**Fear/Greed:** {sentiment.get('score', 0):.0f} ({sentiment.get('label', 'N/A')})\n**Volume Mom:** {vol_mom:.2f}x\n**H/L Ratio:** {hl.get('hl_ratio', 0):.2f}x",
                "inline": False
            },
            {
                "name": "📈 Market Breadth",
                "value": f"🟢 Advance: {ad.get('advances', 0)} ({ad.get('pct_advance', 0):.1f}%)\n🔴 Decline: {ad.get('declines', 0)}\n⚪ Unchanged: {ad.get('unchanged', 0)}",
                "inline": True
            },
            {
                "name": "🏢 Sector Flow",
                "value": "View full details on Dashboard",
                "inline": True
            }
        ],
        "footer": {
            "text": "Antigravity Trading System • Vietnam Stock Market"
        },
        "timestamp": datetime.utcnow().isoformat()
    }

    # If sector data exists, list top 3
    if 'sector_liq' in stats:
        sec_df = stats['sector_liq'].sort_values('GTGD (Tỷ)', ascending=False).head(3)
        sec_text = "\n".join([f"• **{r['Ngành']}**: {r['GTGD (Tỷ)']:,.1f}B" for _, r in sec_df.iterrows()])
        embed['fields'][2]['value'] = sec_text

    payload = {
        "username": "Market Guardian",
        "avatar_url": "https://cdn-icons-png.flaticon.com/512/2585/2585141.png",
        "embeds": [embed]
    }

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(WEBHOOK_URL, json=payload)
            return response.status_code == 204
    except Exception as e:
        print(f"Error sending to Discord: {e}")
        return False

def send_market_summary_sync(stats: dict):
    """Bản đồng bộ cho Streamlit"""
    import requests
    
    sentiment = stats.get('sentiment', {})
    ad = stats.get('ad_stats', {})
    hl = stats.get('hl_stats', {})
    vol_mom = stats.get('vol_momentum', 0)
    
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # ── Lấy dữ liệu cho Discord embed ──────────────────────────────────────────
    ma_row = {}
    today_str_date = datetime.now().strftime('%Y-%m-%d')

    # 1. MA counts từ ma_history
    ma_history = stats.get('ma_history')
    if ma_history is not None and not ma_history.empty:
        row = ma_history[ma_history['date'] == today_str_date]
        if row.empty:
            row = ma_history.iloc[[-1]]
        r = row.iloc[0]
        ma_row['MA10'] = int(r.get('count_ma10', r.get('MA10', 0)))
        ma_row['MA20'] = int(r.get('count_ma20', r.get('MA20', 0)))
        ma_row['MA50'] = int(r.get('count_ma50', r.get('MA50', 0)))

    # 2. Net S/D, Demand, Supply, Power từ power_hist
    power_hist = stats.get('power_hist')
    if power_hist is not None and not power_hist.empty:
        row = power_hist[power_hist['date'] == today_str_date] if 'date' in power_hist.columns else power_hist.iloc[[-1]]
        if row.empty:
            row = power_hist.iloc[[-1]]
        r = row.iloc[0]
        ma_row['net_sd'] = float(r.get('supply_demand', 0))
        ma_row['demand'] = float(r.get('demand', 0))
        ma_row['supply'] = float(r.get('supply', 0))
        ma_row['power']  = float(r.get('power', 0))

    # 3. VNINDEX price, Change, RSI — fetch trực tiếp từ VPS API
    try:
        import requests as _req
        from datetime import timedelta as _td
        _end   = datetime.now()
        _start = _end - _td(days=60)
        _t_end   = int(_end.timestamp())
        _t_start = int(_start.timestamp())
        _url = (f"https://histdatafeed.vps.com.vn/tradingview/history"
                f"?symbol=VNINDEX&resolution=1D&from={_t_start}&to={_t_end}&countback=100")
        _r = _req.get(_url, timeout=10)
        _raw = _r.json()
        vni_closes = _raw.get('c', [])
        if vni_closes:
            ma_row['vnindex'] = float(vni_closes[-1])
            ma_row['change']  = float(vni_closes[-1] - vni_closes[-2]) if len(vni_closes) >= 2 else 0.0
        else:
            ma_row['vnindex'] = 0.0
            ma_row['change']  = 0.0
    except Exception as _e:
        print(f"[NOTIFIER] VNINDEX fetch error: {_e}")
        vni_closes = []
        ma_row['vnindex'] = 0.0
        ma_row['change']  = 0.0

    # 4. RSI tính trên VNINDEX closes
    try:
        import pandas as _pd
        if len(vni_closes) >= 15:
            _s = _pd.Series(vni_closes, dtype=float)
            _delta = _s.diff()
            _gain = _delta.clip(lower=0).rolling(14).mean()
            _loss = (-_delta.clip(upper=0)).rolling(14).mean()
            _rs = _gain / _loss.replace(0, float('nan'))
            _rsi_val = 100 - (100 / (1 + _rs))
            ma_row['rsi'] = round(float(_rsi_val.iloc[-1]), 1)
        else:
            ma_row['rsi'] = 0.0
    except Exception:
        ma_row['rsi'] = 0.0

    # 5. Notes / label
    rsi_v = ma_row.get('rsi', 50)
    net_v = ma_row.get('net_sd', 0)
    if rsi_v < 30:
        ma_row['notes'] = 'OVERSOLD (RSI)'
    elif rsi_v > 70:
        ma_row['notes'] = 'OVERBOUGHT (RSI)'
    elif net_v < -100:
        ma_row['notes'] = 'STRONG SELLING'
    elif net_v > 100:
        ma_row['notes'] = 'STRONG BUYING'
    else:
        ma_row['notes'] = 'NEUTRAL'

    # ── Build embed ──────────────────────────────────────────────────────────
    # Màu theo xu hướng
    net_sd_val = ma_row.get('net_sd', 0)
    color = 0x00E676 if net_sd_val > 0 else (0xFF1744 if net_sd_val < 0 else 0x00B4D8)

    # Format Net S/D
    net_sd_fmt = f"{net_sd_val:+.1f}M" if net_sd_val else "N/A"
    change_fmt = f"{ma_row.get('change', 0):+.2f}" if ma_row.get('change') else "+0.00"

    embed = {
        "title": f"📊 MARKET SUMMARY {datetime.now().strftime('%d/%m/%Y')}",
        "description": (
            f"**{ma_row.get('notes', 'N/A')}** • RSI: {ma_row.get('rsi', 0):.1f}\n"
            f"VNINDEX: `{ma_row.get('vnindex', 0):.2f}` ({change_fmt})"
        ),
        "color": color,
        "fields": [
            {
                "name": "📈 Market Breadth (MA Above)",
                "value": (
                    f"🟢 **MA10:** {ma_row.get('MA10', 0)} | "
                    f"**MA20:** {ma_row.get('MA20', 0)} | "
                    f"**MA50:** {ma_row.get('MA50', 0)}"
                ),
                "inline": False
            },
            {
                "name": "💰 Flow",
                "value": (
                    f"**Net S/D:** {net_sd_fmt}\n"
                    f"🟢 Demand: {ma_row.get('demand', 0):.1f}M\n"
                    f"🔴 Supply: {ma_row.get('supply', 0):.1f}M\n"
                    f"⚡ Power: {ma_row.get('power', 0):.0f}"
                ),
                "inline": True
            },
            {
                "name": "📊 Breadth",
                "value": (
                    f"🟢 Advance: {ad.get('advances', 0)} ({ad.get('pct_advance', 0):.1f}%)\n"
                    f"🔴 Decline: {ad.get('declines', 0)}\n"
                    f"⚪ Unchanged: {ad.get('unchanged', 0)}\n"
                    f"**Fear/Greed:** {sentiment.get('score', 0):.0f} ({sentiment.get('label', 'N/A')})"
                ),
                "inline": True
            },
            {
                "name": "🏢 Top Sector Flow",
                "value": "N/A",
                "inline": False
            }
        ],
        "footer": {
            "text": "Antigravity Trading System • Final Snapshot"
        },
        "timestamp": datetime.now().astimezone().isoformat()
    }

    if 'sector_liq' in stats:
        sec_df = stats['sector_liq'].sort_values('GTGD (Tỷ)', ascending=False).head(3)
        sec_text = "\n".join([f"• **{r['Ngành']}**: {r['GTGD (Tỷ)']:,.1f}B" for _, r in sec_df.iterrows()])
        embed['fields'][2]['value'] = sec_text

    payload = {
        "username": "Market Guardian",
        "embeds": [embed]
    }

    try:
        res = requests.post(WEBHOOK_URL, json=payload)
        return res.status_code == 204
    except Exception as e:
        return False