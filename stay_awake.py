import requests
import time
import sys
from datetime import datetime

URL = "https://vn-stock-dashboard.onrender.com"
INTERVAL = 600 # 10 seconds

def stay_awake():
    print(f"🚀 Stay-Awake Service started for {URL}")
    print(f"⏰ Ping interval: {INTERVAL}s")
    
    count = 0
    while True:
        try:
            now = datetime.now().strftime("%H:%M:%S")
            res = requests.get(URL, timeout=15)
            count += 1
            print(f"[{now}] Ping #{count}: Status {res.status_code}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Ping failed: {e}")
        
        time.sleep(INTERVAL)

if __name__ == "__main__":
    try:
        stay_awake()
    except KeyboardInterrupt:
        print("\n👋 Stay-Awake Service stopped.")
        sys.exit(0)
