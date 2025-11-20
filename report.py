import pandas as pd
import numpy as np
from datetime import datetime
from app.data_fetch import get_multi_price_history

TICKERS = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]

def max_drawdown(series):
    cummax = series.cummax()
    dd = (series - cummax) / cummax
    return dd.min()

def main():
    today = datetime.now().strftime("%Y-%m-%d")

    prices = get_multi_price_history(TICKERS, days=365)
    returns = prices.pct_change().dropna()

    report = {}

    for t in TICKERS:
        asset = prices[t]
        mdd = max_drawdown(asset.astype(float))
        daily_vol = returns[t].std()
        open_price = asset.iloc[0]
        close_price = asset.iloc[-1]

        report[t] = {
            "open": float(open_price),
            "close": float(close_price),
            "volatility": float(daily_vol),
            "max_drawdown": float(mdd),
        }

    df = pd.DataFrame(report).T
    df.to_csv(f"/home/tkg/daily_report_{today}.csv")

if __name__ == "__main__":
    main()
