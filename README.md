# 🇻🇳 Vietnam Stock Market Dashboard

Phân tích độ rộng & sentiment thị trường chứng khoán Việt Nam.

## Features

- **Market Breadth**: Advance/Decline Line, A/D Ratio
- **MA Analysis**: % cổ phiếu trên MA20/50/200
- **Sentiment**: Fear/Greed Index tổng hợp từ 5 chỉ số
- **Liquidity**: Thanh khoản thị trường theo ngày
- **New High/Low**: 52-week breakouts
- **McClellan Oscillator**: Breadth momentum
- **Price Distribution**: Phân phối % thay đổi giá

## Filters

- Sàn: HOSE / HNX / UPCOM / VN30 / VN100 / ALL
- Khoảng thời gian tùy chọn
- MA Periods: 20, 50, 200

## Tech Stack

- **Python 3.11**
- **Streamlit** — UI framework
- **Plotly** — Interactive charts
- **httpx** — Async HTTP client
- **Data**: VPS API (`histdatafeed.vps.com.vn`)

## Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run app
streamlit run app.py
```

Truy cập: http://localhost:8501

## Deploy lên Render

1. Push code lên GitHub
2. Vào [render.com](https://render.com) → **New Web Service**
3. Connect GitHub repo
4. Render tự detect `render.yaml` và deploy

**Build command:** `pip install -r requirements.txt`  
**Start command:** `streamlit run app.py --server.port $PORT --server.address 0.0.0.0 --server.headless true`

## Cấu trúc

```
vietnam-stock-dashboard/
├── app.py                  # Main Streamlit app
├── requirements.txt
├── render.yaml             # Render deploy config
├── .streamlit/
│   └── config.toml         # Dark theme
└── modules/
    ├── tickers.py          # Danh sách mã theo sàn
    ├── fetcher.py          # VPS API async fetcher
    ├── calculator.py       # Breadth & sentiment logic
    └── charts.py           # Plotly chart builders
```

## Lưu ý

- Cache dữ liệu 1 giờ (`@st.cache_data(ttl=3600)`)
- Batch fetch 20 concurrent requests (Semaphore)
- VN30 luôn được ưu tiên trong sample
- Free tier Render: 512MB RAM, đủ cho VN30/VN100
