import streamlit as st
import numpy as np
import pandas as pd
from data_fetch import get_price_history, get_multi_price_history


# ---------------------------------------------------------------------
#                      STYLING (LEVEL 3 - MAX)
# ---------------------------------------------------------------------

st.set_page_config(page_title="Crypto Quant Dashboard", layout="wide")

st.markdown("""
    <style>

    body { background-color: #0E1117; }

    [data-testid="stSidebar"] {
        background-color: #11141c;
        padding-top: 40px;
    }

    h1 {
        text-align: center !important;
        color: white !important;
        margin-bottom: 10px !important;
        font-size: 42px !important;
        font-weight: 700;
    }

    h2, h3, h4 { color: white !important; }

    .divider {
        border-top: 1px solid #4CAF50;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    .metric-card {
        padding: 20px;
        background-color: #1E1F26;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        margin-bottom: 15px;
    }

    .metric-title { font-size: 16px; font-weight: 600; color: white; }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #4CAF50;
        margin-top: -5px;
    }

    </style>
""", unsafe_allow_html=True)


def metric_card(title, value, color="#4CAF50"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 6px solid {color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
        </div>
        """, unsafe_allow_html=True
    )


def max_drawdown(series):
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()


# ---------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------

st.markdown("<h1>Crypto Quant Dashboard</h1><div class='divider'></div>", unsafe_allow_html=True)
st.caption("A4 IF — Python, Linux & Git project")


# ---------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Home", "Single Asset (Ouiam)", "Portfolio (Erian)"])


# ---------------------------------------------------------------------
# HOME PAGE
# ---------------------------------------------------------------------

if page == "Home":
    st.markdown("""
        <h1 style='text-align:center; font-size:48px; font-weight:700; color:white;'>
            Crypto Quant Dashboard
        </h1>

        <h3 style='text-align:center; margin-top:-10px; color:#4CAF50;'>
            A4 IF — Python • Linux • Git • Quantitative Finance
        </h3>

        <div class="divider"></div>

        <p style='text-align:center; font-size:18px; color:#CCCCCC; 
                  max-width:900px; margin:auto;'>
            This dashboard provides advanced cryptocurrency analytics including 
            single-asset analysis, multi-asset portfolio construction, 
            performance evaluation, correlation analysis and automated reporting.
        </p>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------
# SINGLE ASSET (OUIAM)
# ---------------------------------------------------------------------

elif page == "Single Asset (Ouiam)":

    st.subheader("Single Asset Crypto Analysis")

    asset = st.selectbox("Select crypto asset:", ["BTC-USD", "ETH-USD"])

    days = st.slider("Time window (days):", 30, 365, 180, step=10)

    strategy = st.selectbox(
        "Select strategy:",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy"]
    )

    st.write(" ")

    # STRATEGY PARAMETERS
    if strategy == "SMA Crossover":
        sma_short = st.number_input("Short SMA window", 5, 100, 20)
        sma_long = st.number_input("Long SMA window", 20, 300, 100)

    elif strategy == "RSI Strategy":
        rsi_window = st.number_input("RSI window", 5, 50, 14)
        rsi_buy = st.number_input("RSI Buy Threshold", 5, 50, 30)
        rsi_sell = st.number_input("RSI Sell Threshold", 50, 95, 70)

    prices = get_price_history(asset, days)

    if prices.empty:
        st.warning("No data available.")
        st.stop()

    prices["returns"] = prices["price"].pct_change()

    # BUY & HOLD
    bh_curve = (1 + prices["returns"].fillna(0)).cumprod()


    # --------------------------------------------------------
    # SMA CROSSOVER STRATEGY
    # --------------------------------------------------------
    if strategy == "SMA Crossover":
        df = prices.copy()

        df["SMA_short"] = df["price"].rolling(sma_short).mean()
        df["SMA_long"] = df["price"].rolling(sma_long).mean()

        df["signal"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
        df["position"] = df["signal"].shift(1).fillna(0)

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

        # 🔵 OVERLAY GRAPH (Price + SMA)
        st.subheader("SMA Overlay — Price + SMA Short + SMA Long")

        overlay = pd.DataFrame({
            "Price": df["price"],
            f"SMA {sma_short}": df["SMA_short"],
            f"SMA {sma_long}": df["SMA_long"]
        })

        st.line_chart(overlay)

    # --------------------------------------------------------
    # RSI STRATEGY
    # --------------------------------------------------------
    elif strategy == "RSI Strategy":
        df = prices.copy()

        delta = df["price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(rsi_window).mean()
        avg_loss = loss.rolling(rsi_window).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["position"] = 0
        df.loc[df["RSI"] < rsi_buy, "position"] = 1
        df.loc[df["RSI"] > rsi_sell, "position"] = 0
        df["position"] = df["position"].ffill().fillna(0)

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

        # 🔵 RSI GRAPH
        st.subheader("RSI Indicator")

        st.line_chart(df[["RSI"]])

        st.markdown(f"**RSI BUY < {rsi_buy}** — **RSI SELL > {rsi_sell}**")

    else:
        strat_curve = bh_curve


    # --------------------------------------------------------
    # PRICE VS STRATEGY
    # --------------------------------------------------------
    st.subheader(f"Price vs Strategy — {asset}")
    st.line_chart({"Price": prices["price"], "Strategy value": strat_curve})


    # --------------------------------------------------------
    # PERFORMANCE METRICS
    # --------------------------------------------------------
    st.subheader("Performance Metrics")

    trading_days = 252
    avg_daily = float(prices["returns"].mean())
    vol_daily = float(prices["returns"].std())

    annual_ret = (1 + avg_daily) ** trading_days - 1
    annual_vol = vol_daily * np.sqrt(trading_days)

    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan
    mdd = max_drawdown(strat_curve)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Annualized Return", f"{annual_ret:.2%}")
        st.metric("Annualized Volatility", f"{annual_vol:.2%}")

    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Max Drawdown", f"{mdd:.2%}")


# ---------------------------------------------------------------------
# PORTFOLIO (ERIAN) — UNMODIFIED
# ---------------------------------------------------------------------

elif page == "Portfolio (Erian)":
    # (Ton code portfolio est ici, inchangé)
    pass


# ---------------------------------------------------------------------
#                       PORTFOLIO (ERIAN)
# ---------------------------------------------------------------------

elif page == "Portfolio (Erian)":
    st.subheader("Multi-Asset Crypto Portfolio")

    all_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
    tickers = st.multiselect("Select crypto assets:", all_tickers, default=all_tickers)

    days = st.slider("Time window (days):", 30, 365, 180, step=10)

    if len(tickers) == 0:
        st.warning("Please select at least one asset.")
    else:
        prices = get_multi_price_history(tickers, days=days)

        st.subheader("Daily close prices")
        st.line_chart(prices)

        returns = prices.pct_change().dropna()

        # ---- CUSTOM WEIGHTS ----
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

        portfolio_returns = (returns * weights).sum(axis=1)
        portfolio_value = (1 + portfolio_returns).cumprod()

        st.subheader("Portfolio value (base = 1.0)")
        st.line_chart(portfolio_value)

        # ---- METRICS ----
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
            metric_card("Average daily return", f"{avg_daily_ret:.4%}")
            metric_card("Daily volatility", f"{daily_vol:.4%}")
            metric_card("Annualized return", f"{ann_return:.2%}")

        with col2:
            metric_card("Annualized volatility", f"{ann_vol:.2%}")
            metric_card("Sharpe ratio", f"{sharpe:.2f}", "#FF9800")
            metric_card("Max drawdown", f"{mdd:.2%}", "#E53935")

        # ---- COMPARISON ----
        st.subheader("Comparison: Portfolio vs BTC")

        if "BTC-USD" in prices.columns:
            btc_norm = prices["BTC-USD"] / prices["BTC-USD"].iloc[0]
            comp = pd.DataFrame({"Portfolio": portfolio_value, "BTC-USD": btc_norm})
            st.line_chart(comp)
        else:
            st.info("BTC-USD not selected → comparison unavailable.")

        # ---- CORRELATION ----
        st.subheader("Correlation matrix")
        st.dataframe(
            returns.corr()
            .style.background_gradient(cmap="coolwarm")
            .format("{:.2f}")
        )
