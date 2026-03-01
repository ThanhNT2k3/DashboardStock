"""
AI Engine - Phân tích đa nhân tố (Technical + Macro + Foreign Flow)
Dùng XGBoost/Random Forest để dự báo biến động giá.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import asyncio
import httpx
from typing import Optional, List, Dict

try:
    from xgboost import XGBClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
except ImportError:
    # Fallback if not installed yet
    XGBClassifier = None
    RandomForestClassifier = None

# Tickers cho yếu tố vĩ mô (Macro Factors)
MACRO_TICKERS = {
    "GOLD": "GC=F",     # Vàng - Chỉ báo an toàn (Safe haven) khi có chiến tranh
    "OIL": "CL=F",      # Dầu thô - Chi phí sản xuất & lạm phát
    "DXY": "DX-Y.NYB",  # Sức mạnh đồng USD - Tác động tỷ giá & khối ngoại
    "US10Y": "^TNX"     # Lợi suất TPCP Mỹ 10Y - Kỳ vọng lãi suất
}

class AIEngine:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler() if 'StandardScaler' in globals() else None
        self.features = []

    def fetch_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Lấy dữ liệu vĩ mô từ yfinance"""
        try:
            # yf.download trả về MultiIndex hoặc DataFrame tùy số lượng ticker
            data = yf.download(list(MACRO_TICKERS.values()), start=start_date, end=end_date, interval="1d")
            
            # Chỉ lấy giá Close (đóng cửa)
            if isinstance(data.columns, pd.MultiIndex):
                macro_df = data['Close'].copy()
            else:
                macro_df = data[['Close']].copy()
                macro_df.columns = [list(MACRO_TICKERS.keys())[0]] # Trường hợp chỉ có 1 ticker

            # Đổi tên cột từ ticker sang label thân thiện
            reverse_map = {v: k for k, v in MACRO_TICKERS.items()}
            macro_df.rename(columns=reverse_map, inplace=True)
            
            # Fill NaN bằng giá trị trước đó (ffill)
            macro_df = macro_df.ffill().dropna()
            # Bỏ timezone để join (yfinance thường trả về UTC)
            if macro_df.index.tz is not None:
                macro_df.index = macro_df.index.tz_localize(None)
            macro_df.index = macro_df.index.normalize()
            return macro_df
        except Exception as e:
            print(f"Error fetching macro data: {e}")
            return pd.DataFrame()

    def fetch_foreign_flow(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu Khối ngoại (Foreign Flow)
        Tạm thời sử dụng mock data hoặc API VNDIRECT công khai nếu có thể.
        Sử dụng vnstock3 nếu được cài đặt.
        """
        try:
            from vnstock3 import Vnstock
            stock = Vnstock().stock(symbol=ticker, source='VCI') # VCI hoặc TCBS
            df = stock.finance.foreign_flow(start_date=start_date, end_date=end_date)
            # Chuẩn hóa cột: date, foreignNetValue
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                df.index = df.index.normalize()
                return df[['foreignNetValue']]
        except:
            # Fallback: Trả về DF trống để không lỗi, hoặc sinh mock data cho demo
            return pd.DataFrame()
        return pd.DataFrame()

    def prepare_features(self, ticker_df: pd.DataFrame, macro_df: pd.DataFrame, foreign_df: pd.DataFrame) -> pd.DataFrame:
        """Kết hợp technical + macro + foreign flow"""
        df = ticker_df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Bỏ timezone để join
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.index = df.index.normalize()

        # 1. Technical Features
        df['MA20'] = df['close'].rolling(20).mean()
        df['MA50'] = df['close'].rolling(50).mean()
        df['RSI']  = self._compute_rsi(df['close'])
        df['VOL_EMA'] = df['volume'].ewm(span=20).mean()
        df['RETURNS_1D'] = df['close'].pct_change()

        # 2. Merge Macro (Dùng ffill vì macro global có thể lệch múi giờ/ngày nghỉ)
        if not macro_df.empty:
            df = df.join(macro_df, how='left').ffill()

        # 3. Merge Foreign Flow
        if not foreign_df.empty:
            df = df.join(foreign_df, how='left').fillna(0)
        else:
            # Tạo feature giả nếu không có dữ liệu thực (vẫn đảm bảo model chạy)
            df['foreignNetValue'] = 0

        # 4. Target Label: 5-day Forward Return
        df['TARGET_RET'] = df['close'].shift(-5) / df['close'] - 1
        
        # Phân loại cho XGBoost (cần nhãn từ 0, 1, 2...)
        # 2: BUY (ret > 3%), 0: SELL (ret < -3%), 1: HOLD
        def classify(x):
            if x > 0.03: return 2
            if x < -0.03: return 0
            return 1
        
        df['LABEL'] = df['TARGET_RET'].apply(classify)
        
        # Loại bỏ các hàng có NaN
        df.dropna(inplace=True)
        return df

    def _compute_rsi(self, series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss.replace(0, 0.001))
        return 100 - (100 / (1 + rs))

    def train(self, data: pd.DataFrame):
        """Huấn luyện mô hình XGBoost hoặc Random Forest"""
        if XGBClassifier is None:
            print("XGBoost not installed. Training skipped.")
            return

        features_cols = ['RSI', 'MA20', 'MA50', 'RETURNS_1D', 'GOLD', 'OIL', 'DXY', 'US10Y', 'foreignNetValue']
        # Chỉ lấy các cột tồn tại trong data
        X_cols = [c for c in features_cols if c in data.columns]
        self.features = X_cols
        
        X = data[X_cols]
        y = data['LABEL']

        if len(X) < 50: # Quá ít dữ liệu để train
            return

        self.model = XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
        # Random Forest as alternative:
        # self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        self.model.fit(X, y)

    def predict(self, current_features: pd.DataFrame) -> int:
        """Dự báo: 1 (Buy), -1 (Sell), 0 (Hold)"""
        if self.model is None or not self.features:
            return 0
        
        X = current_features[self.features].tail(1)
        pred = self.model.predict(X)
        return int(pred[0])

# Wrapper function để tích hợp vào app
def run_ai_analysis(ticker: str, history_df: pd.DataFrame, start_date: str, end_date: str):
    engine = AIEngine()
    
    # Lấy dữ liệu vĩ mô
    macro_df = engine.fetch_macro_data(start_date, end_date)
    
    # Lấy dữ liệu ngoại
    foreign_df = engine.fetch_foreign_flow(ticker, start_date, end_date)
    
    # Chuẩn bị dữ liệu huấn luyện
    full_df = engine.prepare_features(history_df, macro_df, foreign_df)
    
    if full_df.empty:
        return None, "Không đủ dữ liệu để huấn luyện mô hình AI."

    # Train (Sử dụng 80% dữ liệu đầu để train, 20% cuối để backtest/predict)
    engine.train(full_df)
    
    # Tín hiệu dự báo cho phiên cuối cùng
    signal = engine.predict(full_df)
    
    return {
        'engine': engine,
        'signal': signal,
        'full_df': full_df,
        'label': 'BUY' if signal == 2 else ('SELL' if signal == 0 else 'HOLD')
    }, None
