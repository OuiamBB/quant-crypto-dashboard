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
    """
    end = datetime.today()
    start = end - timedelta(days=days)

    data = _download_data(ticker, start, end)
    if data is None or len(data) == 0:
        return pd.DataFrame(columns=["price"])

    if isinstance(data, pd.Series):
        close = data

    elif isinstance(data, pd.DataFrame):
        if "Close" in data.columns:
            close = data["Close"]
        elif data.select_dtypes(include="number").shape[1] == 1:
            col = data.select_dtypes(include="number").columns[0]
            close = data[col]
        else:
            close = data.iloc[:, 0]

    else:
        return pd.DataFrame(columns=["price"])

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

    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"]
    else:
        close = data

    close.index.name = "date"
    return close
