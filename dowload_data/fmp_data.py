import os
import requests
import pandas as pd
from datetime import datetime

# Lấy API key từ biến môi trường (hoặc gán trực tiếp tại đây)
FMP_KEY = 'ApLQ3DwmFvvKRHD7gZQTdCOhzZgqgJOG'


def get_fmp_data(symbol: str,
                 start_date: str = "2022-06-01",
                 timeframe: str = "D1",
                 end_date: str | None = None) -> pd.DataFrame:
    """Lấy dữ liệu OHLCV hàng ngày từ Financial Modeling Prep (FMP).

    Parameters
    ----------
    symbol : str
        Mã chứng khoán/Crypto cần lấy (ví dụ: "AAPL", "BTCUSD").
    start_date : str
        Ngày bắt đầu theo định dạng YYYY-MM-DD.
    end_date : str | None
        Ngày kết thúc theo định dạng YYYY-MM-DD. Mặc định là ngày hiện tại (UTC).

    Returns
    -------
    pd.DataFrame
        DataFrame với index là cột Date và các cột
        ['Open', 'High', 'Low', 'Close', 'Volume'].
    """
    if end_date is None:
        end_date = datetime.utcnow().strftime("%Y-%m-%d")

    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/"
        f"{symbol}?from={start_date}&to={end_date}&apikey={FMP_KEY}&timeframe={timeframe}"
    )

    try:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        data = r.json()

        # Kiểm tra dữ liệu phản hồi
        if "historical" not in data or not data["historical"]:
            raise ValueError("Không tìm thấy trường 'historical' hoặc dữ liệu trống.")

        # Chuyển đổi sang DataFrame và chuẩn hóa tên cột
        df = pd.DataFrame(data["historical"])[
            ["date", "open", "high", "low", "close", "volume"]
        ]
        df.rename(
            columns={
                "date": "Date",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

        # Định dạng cột Date và sắp xếp tăng dần
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values("Date", inplace=True)
        df.set_index("Date", inplace=True)

        # Bảo đảm tần suất hàng ngày, điền forward‑fill cho ngày nghỉ
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq="D")
        df = df.reindex(full_range, method="ffill")

        return df[["Open", "High", "Low", "Close", "Volume"]]

    except Exception as e:
        print(f"Lỗi khi lấy dữ liệu FMP cho {symbol}: {e}")
        return pd.DataFrame()


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Trích xuất các đặc trưng thị trường và kỹ thuật."""
    features = pd.DataFrame(index=df.index)

    # 1. Đặc trưng thị trường cơ bản
    features["Open"] = df["Open"]
    features["High"] = df["High"]
    features["Low"] = df["Low"]
    features["Close"] = df["Close"]
    features["Volume"] = df["Volume"]
    features["Market Spread"] = df["High"] - df["Low"]
    features["Change"] = df["Close"].pct_change()

    # 2. Chỉ báo kỹ thuật
    # Moving Averages (MA)
    for n in [5, 20, 50, 88, 100, 200]:
        features[f"MA({n}days)"] = df["Close"].rolling(window=n).mean()

    # Exponential Moving Average (EMA) và Standard Deviation (STD)
    for n in [5, 20, 50, 100]:
        features[f"EMA({n}days)"] = df["Close"].ewm(span=n, adjust=False).mean()
        features[f"STD({n}days)"] = df["Close"].rolling(window=n).std()

    # MACD với các cặp khoảng thời gian khác nhau
    macd_pairs = {
        0: (5, 20),
        1: (5, 50),
        2: (5, 100),
        3: (20, 50),
        4: (20, 100),
        5: (50, 100),
    }

    for k, (a, b) in macd_pairs.items():
        ema_fast = df["Close"].ewm(span=a, adjust=False).mean()
        ema_slow = df["Close"].ewm(span=b, adjust=False).mean()
        features[f"MACD{k}"] = ema_fast - ema_slow

    # Bollinger Bands (20 ngày)
    ma_20 = df["Close"].rolling(window=20).mean()
    std_20 = df["Close"].rolling(window=20).std()
    features["Bollinger Upper"] = ma_20 + 2 * std_20
    features["Bollinger Lower"] = ma_20 - 2 * std_20

    # Loại bỏ các hàng NaN phát sinh do rolling window
    features.dropna(inplace=True)

    return features
