"""
AI Engine - Phân tích đa nhân tố (Technical + Macro + Foreign Flow)
Dùng XGBoost để dự báo biến động giá. Tối ưu hyperparameters & features.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
except ImportError:
    RandomForestClassifier = None
    StandardScaler = None

# Tickers cho yếu tố vĩ mô & hàng hóa bên ngoài (ảnh hưởng thị trường VN)
# Nguồn: Yahoo Finance (yfinance) - giá realtime
MACRO_TICKERS = {
    # Tài chính & tiền tệ
    "GOLD": "GC=F",       # Vàng - Safe haven, vàng trang sức
    "OIL": "CL=F",        # Dầu thô WTI - Chi phí vận tải, nhiên liệu
    "DXY": "DX-Y.NYB",    # Chỉ số USD - Tỷ giá, khối ngoại
    "US10Y": "^TNX",      # Lợi suất TPCP Mỹ 10Y - Lãi suất toàn cầu
    # Kim loại & năng lượng
    "SILVER": "SI=F",     # Bạc
    "COPPER": "HG=F",     # Đồng - Kim loại công nghiệp
    "NATGAS": "NG=F",     # Khí đốt - Chi phí phân bón, điện
    # Nông sản (ảnh hưởng VN: lúa gạo, cà phê, cao su, đường)
    "CORN": "ZC=F",       # Ngô - Thức ăn chăn nuôi
    "WHEAT": "ZW=F",      # Lúa mì
    "SOYBEANS": "ZS=F",   # Đậu tương
    "SUGAR": "SB=F",      # Đường
    "COFFEE": "KC=F",     # Cà phê - Rất quan trọng với VN
    "COTTON": "CT=F",     # Bông
    # Phân bón (ETF theo dõi ngành phân bón/potash)
    "FERTILIZER": "SOIL", # Global X Fertilizers/Potash ETF
}

# Feature columns (model dùng các cột có sẵn trong data)
FEATURE_COLS = [
    'RSI', 'MA20', 'MA50', 'RETURNS_1D', 'VOLATILITY_10D', 'PRICE_MA20_RATIO',
    'GOLD', 'OIL', 'DXY', 'US10Y',
    'SILVER', 'COPPER', 'NATGAS',
    'CORN', 'WHEAT', 'SOYBEANS', 'SUGAR', 'COFFEE', 'COTTON', 'FERTILIZER',
    'foreignNetValue'
]


class AIEngine:
    def __init__(
        self,
        model_type: str = "ensemble",  # "xgb", "rf", "ensemble"
        buy_threshold: float = 0.6,
        sell_threshold: float = 0.6,
    ):
        self.model_type = model_type
        self.model_xgb: Optional[XGBClassifier] = None
        self.model_rf: Optional["RandomForestClassifier"] = None
        self.scaler = StandardScaler() if "StandardScaler" in globals() else None
        self.features: List[str] = []
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def _fetch_single_macro(self, label: str, symbol: str, start_date: str, end_date: str) -> Optional[pd.Series]:
        """Fetch 1 ticker macro, trả về Series hoặc None nếu lỗi"""
        try:
            data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False, threads=False)
            if data.empty or len(data) < 5:
                return None
            close = data['Close'].copy() if 'Close' in data.columns else data.iloc[:, 0]
            close.name = label
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close.index = close.index.normalize()
            return close
        except Exception:
            return None

    def fetch_macro_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Lấy dữ liệu vĩ mô & hàng hóa từ yfinance (song song).
        Gồm: vàng, dầu, USD, lãi suất, bạc, đồng, khí đốt, ngô, lúa mì,
        đậu tương, đường, cà phê, bông, ETF phân bón.
        """
        dfs = []
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(self._fetch_single_macro, label, symbol, start_date, end_date): label
                for label, symbol in MACRO_TICKERS.items()
            }
            for future in as_completed(futures):
                series = future.result()
                if series is not None:
                    dfs.append(series)

        if not dfs:
            return pd.DataFrame()
        macro_df = pd.concat(dfs, axis=1, join='outer')
        macro_df = macro_df.ffill().bfill()
        return macro_df

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
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame()

    def fetch_foreign_flow_batch(self, tickers: List[str], start_date: str, end_date: str, max_workers: int = 5) -> Dict[str, pd.DataFrame]:
        """Lấy dữ liệu khối ngoại cho nhiều mã song song (tối ưu cho AI Scanner)"""
        result = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.fetch_foreign_flow, t, start_date, end_date): t for t in tickers}
            for future in as_completed(futures):
                ticker = futures[future]
                try:
                    result[ticker] = future.result()
                except Exception:
                    result[ticker] = pd.DataFrame()
        return result

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
        df['RSI'] = self._compute_rsi(df['close']).clip(0, 100).fillna(50)  # RSI ổn định
        df['VOL_EMA'] = df['volume'].ewm(span=20).mean()
        df['RETURNS_1D'] = df['close'].pct_change()
        # Volatility 10 ngày (độ biến động giá)
        df['VOLATILITY_10D'] = df['close'].pct_change().rolling(10).std().fillna(0)
        # Tỷ lệ giá/MA20 - momentum tương đối
        df['PRICE_MA20_RATIO'] = (df['close'] / df['MA20']).fillna(1.0)

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
        rs = gain / loss.replace(0, np.finfo(float).eps)
        return 100 - (100 / (1 + rs))

    def train(self, data: pd.DataFrame):
        """Huấn luyện mô hình AI (XGBoost, RandomForest hoặc Ensemble)"""
        if XGBClassifier is None and RandomForestClassifier is None:
            return

        X_cols = [c for c in FEATURE_COLS if c in data.columns]
        self.features = X_cols
        if not X_cols:
            return

        X = data[X_cols]
        y = data["LABEL"]
        if len(X) < 50:
            return

        # Cân bằng lớp (BUY/SELL thường ít hơn HOLD)
        class_counts = y.value_counts()
        scale_pos_weight = (class_counts.get(1, 1) / max(class_counts.get(2, 1), 1))

        # XGBoost model
        if XGBClassifier is not None and self.model_type in ("xgb", "ensemble"):
            self.model_xgb = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
            )
            self.model_xgb.fit(X, y)

        # RandomForest model (tập trung vào độ ổn định)
        if RandomForestClassifier is not None and self.model_type in ("rf", "ensemble"):
            self.model_rf = RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=3,
                class_weight="balanced_subsample",
                n_jobs=-1,
                random_state=42,
            )
            self.model_rf.fit(X, y)

    def predict(self, current_features: pd.DataFrame) -> int:
        """
        Dự báo: 2 (BUY), 1 (HOLD), 0 (SELL)
        Sử dụng ensemble XGBoost + RandomForest (nếu có) và ngưỡng xác suất.
        """
        if not self.features:
            return 0

        X = current_features[self.features].tail(1)

        prob_sum = None
        prob_count = 0

        # XGBoost probability
        if self.model_xgb is not None and self.model_type in ("xgb", "ensemble"):
            try:
                proba_xgb = self.model_xgb.predict_proba(X)[0]
                prob_sum = proba_xgb if prob_sum is None else prob_sum + proba_xgb
                prob_count += 1
            except Exception:
                pass

        # RandomForest probability
        if self.model_rf is not None and self.model_type in ("rf", "ensemble"):
            try:
                proba_rf = self.model_rf.predict_proba(X)[0]
                prob_sum = proba_rf if prob_sum is None else prob_sum + proba_rf
                prob_count += 1
            except Exception:
                pass

        # Nếu không lấy được xác suất, fallback về predict() của model có sẵn
        if prob_sum is None or prob_count == 0:
            if self.model_xgb is not None:
                return int(self.model_xgb.predict(X)[0])
            if self.model_rf is not None:
                return int(self.model_rf.predict(X)[0])
            return 0

        avg_proba = prob_sum / prob_count  # trung bình xác suất các model

        # Giả định thứ tự lớp [0, 1, 2] tương ứng SELL, HOLD, BUY
        proba_sell = float(avg_proba[0])
        proba_hold = float(avg_proba[1])
        proba_buy = float(avg_proba[2])

        # Áp dụng ngưỡng: chỉ BUY/SELL khi xác suất đủ cao, còn lại HOLD
        if proba_buy >= self.buy_threshold and proba_buy > proba_sell and proba_buy > proba_hold:
            return 2
        if proba_sell >= self.sell_threshold and proba_sell > proba_buy and proba_sell > proba_hold:
            return 0
        return 1

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
