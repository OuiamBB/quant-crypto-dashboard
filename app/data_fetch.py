import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf


def _download_data(tickers, start, end):
    """
    Internal helper to call yfinance safely.
    Returns a DataFrame or an empty DataFrame on error.
    """
    try:
        data = yf.download(tickers, start=start, end=end, progress=False)
        return data
    except Exception as e:
        print(f"[data_fetch] Error while downloading data for {tickers}: {e}")
        return pd.DataFrame()


def get_price_history(ticker: str, days: int = 90) -> pd.DataFrame:
    """
    Download historical daily prices for a single asset.
    Returns a DataFrame with a 'price' column indexed by date.
    This version is robust to different yfinance output shapes.
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    data = _download_data(ticker, start, end)

    # Always return a DataFrame with the right column, even on error
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["price"])

    # Case 1: data is a Series (rare but possible)
    if isinstance(data, pd.Series):
        close = data

    # Case 2: data is a DataFrame (most common)
    elif isinstance(data, pd.DataFrame):
        # Standard yfinance case: OHLCV columns with 'Close'
        if "Close" in data.columns:
            close = data["Close"]

        # If there's only one numeric column, use it as price
        elif data.select_dtypes(include="number").shape[1] == 1:
            col = data.select_dtypes(include="number").columns[0]
            close = data[col]

        # Fallback: take the first column
        else:
            close = data.iloc[:, 0]
    else:
        # Unknown type → return empty price series
        return pd.DataFrame(columns=["price"])

    # At this point, close should be a Series
    close = pd.Series(close.squeeze()).dropna()
    df = close.to_frame(name="price")
    df.index.name = "date"
    return df




def get_multi_price_history(tickers, days: int = 90) -> pd.DataFrame:
    """
    Download historical daily close prices for several assets.
    Returns a DataFrame where columns are tickers and rows are dates.
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    data = _download_data(tickers, start, end)
    if data.empty:
        return pd.DataFrame()

    # yfinance returns a multi-indexed DataFrame when there are several tickers
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    close.index.name = "date"
    return close
