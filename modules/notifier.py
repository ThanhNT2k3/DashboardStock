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
        sec_df = stats['sector_liq'].sort_values('Value (B)', ascending=False).head(3)
        sec_text = "\n".join([f"• **{r['Sector']}**: {r['Value (B)']:,.1f}B" for _, r in sec_df.iterrows()])
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
    
    embed = {
        "title": f"🚀 MARKET SUMMARY {datetime.now().strftime('%d/%m/%Y')}",
        "description": f"Automated EOD report (Antigravity System)\nUpdated at: {now_str}",
        "color": 0x00B4D8, 
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
                "name": "🏢 Top Sector Flow",
                "value": "N/A",
                "inline": True
            }
        ],
        "footer": {
            "text": "Antigravity Trading System • Final Snapshot"
        },
        "timestamp": datetime.now().astimezone().isoformat()
    }

    if 'sector_liq' in stats:
        sec_df = stats['sector_liq'].sort_values('Value (B)', ascending=False).head(3)
        sec_text = "\n".join([f"• **{r['Sector']}**: {r['Value (B)']:,.1f}B" for _, r in sec_df.iterrows()])
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
