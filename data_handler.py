import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime

def initialize_mt5():
    """Khởi tạo kết nối đến MetaTrader 5."""
    if not mt5.initialize():
        print(f"Khởi tạo MetaTrader 5 thất bại, mã lỗi = {mt5.last_error()}")
        return False
    return True

def get_mt5_data(symbol, timeframe, start_date, end_date):
    """Lấy dữ liệu lịch sử từ MT5 và trả về DataFrame."""
    timeframe_map = {
        'D1': mt5.TIMEFRAME_D1, 'W1': mt5.TIMEFRAME_W1, 'MN1': mt5.TIMEFRAME_MN1,
        'H1': mt5.TIMEFRAME_H1, 'M30': mt5.TIMEFRAME_M30, 'M15': mt5.TIMEFRAME_M15,
        'M5': mt5.TIMEFRAME_M5, 'M1': mt5.TIMEFRAME_M1
    }
    
    if not initialize_mt5():
        return None

    try:
        rates = mt5.copy_rates_range(symbol, timeframe_map[timeframe], start_date, end_date)
        mt5.shutdown()
        
        if rates is None or len(rates) == 0:
            print("Không có dữ liệu.")
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.rename(columns={
            'time': 'Date', 'open': 'Open', 'high': 'High', 'low': 'Low', 
            'close': 'Close', 'tick_volume': 'Volume'
        }, inplace=True)
        df.set_index('Date', inplace=True)
        return df[['Open', 'High', 'Low', 'Close', 'Volume']]
        
    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu: {e}")
        mt5.shutdown()
        return None

def extract_features(df):
    """Trích xuất các đặc trưng thị trường và kỹ thuật như trong bài báo."""
    features = pd.DataFrame(index=df.index)
    
    # 1. Đặc trưng thị trường
    features['Open'] = df['Open']
    features['High'] = df['High']
    features['Low'] = df['Low']
    features['Close'] = df['Close']
    features['Volume'] = df['Volume']
    features['Market Spread'] = df['High'] - df['Low']
    features['Change'] = df['Close'].pct_change()

    # 2. Chỉ số kỹ thuật
    # MA
    for n in [5, 20, 50, 88, 100, 200]:
        features[f'MA({n}days)'] = df['Close'].rolling(window=n).mean()
    
    # STD và EMA
    for n in [5, 20, 50, 100]:
        features[f'EMA({n}days)'] = df['Close'].ewm(span=n, adjust=False).mean()
        features[f'STD({n}days)'] = df['Close'].rolling(window=n).std()

    # MACD 
    # ema_a = df['Close'].ewm(span=12, adjust=False).mean()
    # ema_b = df['Close'].ewm(span=26, adjust=False).mean()
    # features['MACD'] = ema_a - ema_b
    # for k in range(6):                       # MACD0 … MACD5
    #     features[f'MACD{k}'] = features['MACD'].shift(k)
    # features.dropna(inplace=True)

    macd_pairs = {
        0: (5, 20),
        1: (5, 50),
        2: (5, 100),
        3: (20, 50),
        4: (20, 100),
        5: (50, 100),
    }

    for k, (a, b) in macd_pairs.items():
        ema_fast = df['Close'].ewm(span=a, adjust=False).mean()
        ema_slow = df['Close'].ewm(span=b, adjust=False).mean()
        features[f'MACD{k}'] = ema_fast - ema_slow


    # Bollinger Bands (ví dụ với 20 ngày)
    ma_20 = df['Close'].rolling(window=20).mean()
    std_20 = df['Close'].rolling(window=20).std()
    features['Bollinger Upper'] = ma_20 + (std_20 * 2)
    features['Bollinger Lower'] = ma_20 - (std_20 * 2)

    features.dropna(inplace=True)
    return features