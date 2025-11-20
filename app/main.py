import streamlit as st
import numpy as np
import pandas as pd
from data_fetch import get_price_history, get_multi_price_history


# Max drawdown helper
def max_drawdown(series):
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()


# Streamlit page config
st.set_page_config(page_title="Crypto Quant Dashboard", layout="wide")

st.title("Crypto Quant Dashboard")
st.caption("A4 IF - Python, Linux & Git project")

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Single Asset (Ouiam)", "Portfolio (Erian)"]
)

# ----- HOME PAGE -----
if page == "Home":
    st.subheader("Welcome 👋")
    st.write(
        "This dashboard is part of the *Python, Linux and Git* course at ESILV. "
        "It provides single-asset and multi-asset crypto analytics, with basic backtests and risk metrics."
    )

# ----- SINGLE ASSET PAGE -----
elif page == "Single Asset (Ouiam)":
    st.subheader("Single Asset Crypto Analysis")

    asset = st.selectbox("Select crypto asset:", ["BTC-USD", "ETH-USD"])
    days = st.slider("Time window (days):", 30, 365, 180, step=10)

    prices = get_price_history(asset, days=days)

    if prices.empty:
        st.warning("No data available.")
    else:
        st.subheader(f"Daily close price — {asset}")
        st.line_chart(prices["price"])

        # Returns
        returns = prices["price"].pct_change().dropna()

        st.subheader("Performance metrics")
        trading_days_per_year = 252
        avg_daily_ret = returns.mean()
        daily_vol = returns.std()

        ann_return = (1 + avg_daily_ret) ** trading_days_per_year - 1
        ann_vol = daily_vol * np.sqrt(trading_days_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average daily return", f"{avg_daily_ret:.4%}")
            st.metric("Daily volatility", f"{daily_vol:.4%}")
            st.metric("Annualized return", f"{ann_return:.2%}")
        with col2:
            st.metric("Annualized volatility", f"{ann_vol:.2%}")
            st.metric("Sharpe ratio", f"{sharpe:.2f}")
            st.metric("Days", days)

# ----- PORTFOLIO PAGE -----
elif page == "Portfolio (Erian)":
    st.subheader("Multi-Asset Crypto Portfolio")

    all_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
    tickers = st.multiselect("Select crypto assets:", all_tickers, default=all_tickers)

    days = st.slider("Time window (days):", 30, 365, 180, step=10)

    if len(tickers) == 0:
        st.warning("Please select at least one asset.")
    else:
        # Load multi prices
        prices = get_multi_price_history(tickers, days=days)

        st.subheader("Daily close prices")
        st.line_chart(prices)

        # Daily returns
        returns = prices.pct_change().dropna()

        # ----- CUSTOM WEIGHTS -----
        st.subheader("Portfolio weights")

        weights_input = {}
        for t in tickers:
            weights_input[t] = st.number_input(
                f"Weight for {t} (%)",
                min_value=0.0, max_value=100.0,
                value=100.0 / len(tickers), step=1.0
            )

        weights = np.array(list(weights_input.values()))
        weights = weights / weights.sum()

        st.write("Normalized weights:", {t: f"{w:.1%}" for t, w in zip(tickers, weights)})

        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod()

        st.subheader("Portfolio value (base = 1.0)")
        st.line_chart(portfolio_value)

        # ----- METRICS -----
        st.subheader("Portfolio performance metrics")

        trading_days_per_year = 252
        avg_daily_ret = portfolio_returns.mean()
        daily_vol = portfolio_returns.std()

        ann_return = (1 + avg_daily_ret) ** trading_days_per_year - 1
        ann_vol = daily_vol * np.sqrt(trading_days_per_year)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan
        mdd = max_drawdown(portfolio_value)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Average daily return", f"{avg_daily_ret:.4%}")
            st.metric("Daily volatility", f"{daily_vol:.4%}")
            st.metric("Annualized return", f"{ann_return:.2%}")
        with col2:
            st.metric("Annualized volatility", f"{ann_vol:.2%}")
            st.metric("Sharpe ratio (annualized)", f"{sharpe:.2f}")
            st.metric("Max drawdown", f"{mdd:.2%}")

        # ----- COMPARISON -----
        st.subheader("Comparison: Portfolio vs BTC")

        if "BTC-USD" in prices.columns:
            btc_norm = prices["BTC-USD"] / prices["BTC-USD"].iloc[0]
            comp = pd.DataFrame({"Portfolio": portfolio_value, "BTC-USD": btc_norm})
            st.line_chart(comp)
        else:
            st.info("BTC-USD not selected → comparison unavailable.")

        # ----- CORRELATION -----
        st.subheader("Correlation matrix between assets")
        st.dataframe(returns.corr().style.background_gradient(cmap="coolwarm").format("{:.2f}"))
