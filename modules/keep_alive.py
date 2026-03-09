import threading
import time
import requests
import os

def ping_site(url, interval=600):
    """
    Background loop to ping a URL to keep it awake.
    Default interval is 600 seconds (10 minutes).
    """
    print(f"  - Starting Keep-Alive thread for: {url}")
    while True:
        try:
            # We use a timeout to prevent hanging
            response = requests.get(url, timeout=10)
            # print(f"  - Keep-Alive Ping sent to {url} [Status: {response.status_code}]")
        except Exception as e:
            # print(f"  - Keep-Alive Ping failed: {e}")
            pass
        time.sleep(interval)

def start_keep_alive(url=None):
    """
    Starts a background thread to ping the application.
    If url is not provided, it tries to detect from RENDER_EXTERNAL_URL.
    """
    if url is None:
        # Render provides this environment variable
        url = os.environ.get('RENDER_EXTERNAL_URL')
        
    if not url:
        # Fallback to the known name if on Render
        # Usually: https://vn-stock-dashboard.onrender.com
        url = "https://dashboardstock.onrender.com/"
        
    if url:
        t = threading.Thread(target=ping_site, args=(url,), daemon=True)
        t.start()
        return True
    return False
