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

    /* GLOBAL BACKGROUND */
    body {
        background-color: #0E1117;
    }

    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #11141c;
        padding-top: 40px;
    }

    /* SIDEBAR TITLE */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white;
    }

    /* TITLES */
    h1 {
        text-align: center !important;
        color: white !important;
        margin-bottom: 10px !important;
        font-size: 42px !important;
        font-weight: 700;
    }

    h2, h3, h4 {
        color: white !important;
    }

    /* DIVIDER */
    .divider {
        border-top: 1px solid #4CAF50;
        margin-top: 10px;
        margin-bottom: 20px;
    }

    /* CARDS */
    .metric-card {
        padding: 20px;
        background-color: #1E1F26;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        margin-bottom: 15px;
    }

    .metric-title {
        font-size: 16px;
        font-weight: 600;
        color: white;
    }

    .metric-value {
        font-size: 30px;
        font-weight: 700;
        color: #4CAF50;
        margin-top: -5px;
    }

    </style>
""", unsafe_allow_html=True)


# Function to create cards
def metric_card(title, value, color="#4CAF50"):
    st.markdown(
        f"""
        <div class="metric-card" style="border-left: 6px solid {color};">
            <div class="metric-title">{title}</div>
            <div class="metric-value" style="color:{color};">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


# Max drawdown helper
def max_drawdown(series):
    cummax = series.cummax()
    drawdown = (series - cummax) / cummax
    return drawdown.min()


# ---------------------------------------------------------------------
#                             HEADER
# ---------------------------------------------------------------------

st.markdown("""
    <h1>Crypto Quant Dashboard</h1>
    <div class="divider"></div>
""", unsafe_allow_html=True)

st.caption("A4 IF — Python, Linux & Git project")


# ---------------------------------------------------------------------
#                             SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Single Asset (Ouiam)", "Portfolio (Erian)"]
)


# ---------------------------------------------------------------------
#                             HOME PAGE
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
            It was developed as part of the A4 IF course at ESILV.
        </p>

        <br><br>

        <h3 style='text-align:center; color:white;'>Team Members</h3>

        <div style='display:flex; justify-content:center; gap:40px; margin-top:20px;'>

            <div class="metric-card" style="width:350px; border-left:6px solid #4CAF50;">
                <h3 style='color:white; text-align:center;'>Erian STANLEY YOGARAJ</h3>
                <p style='color:#AAAAAA; text-align:center;'>
                    Developer — Portfolio Analysis (Quant B)<br>
                    Multi-asset modeling, risk metrics,<br>
                    correlations and Linux automation.
                </p>
            </div>

            <div class="metric-card" style="width:350px; border-left:6px solid #2196F3;">
                <h3 style='color:white; text-align:center;'>Ouiam BOUSSAID BENCHAARA</h3>
                <p style='color:#AAAAAA; text-align:center;'>
                    Developer — Single Asset Analysis (Quant A)<br>
                    Technical indicators, BTC analytics,<br>
                    performance computation and strategies.
                </p>
            </div>

        </div>

        <br><br>

        <h3 style='color:white; text-align:center;'>Project Overview</h3>

        <div style='display:flex; justify-content:center; gap:40px; margin-top:20px;'>

            <div class="metric-card" style="width:350px; border-left:6px solid #4CAF50;">
                <h3 style='color:white;'>Quant A — Single Asset</h3>
                <p style='color:#AAAAAA;'>
                    • BTC-USD analysis<br>
                    • Technical indicators (MAs)<br>
                    • Volatility & Sharpe ratio<br>
                    • Performance metrics
                </p>
            </div>

            <div class="metric-card" style="width:350px; border-left:6px solid #2196F3;">
                <h3 style='color:white;'>Quant B — Portfolio</h3>
                <p style='color:#AAAAAA;'>
                    • Multi-asset portfolio (BTC, ETH, BNB, SOL)<br>
                    • Risk & diversification<br>
                    • Correlation matrix<br>
                    • Drawdown & Sharpe ratio
                </p>
            </div>

        </div>

        <br><br>

        <p style='text-align:center; color:#777777; font-size:15px;'>
            Developed by Erian & Ouiam — ESILV A4 IF — 2024/2025
        </p>

    """, unsafe_allow_html=True)




# ---------------------------------------------------------------------
#                       SINGLE ASSET (OUIAM)
# ---------------------------------------------------------------------

elif page == "Single Asset (Ouiam)":
    st.subheader("Single Asset Crypto Analysis")

    # --- Inputs ---
    asset = st.selectbox("Select crypto asset:", ["BTC-USD", "ETH-USD"], index=0)

    days = st.slider(
        "Time window (days):",
        min_value=30, max_value=365, value=180, step=10
    )

    strategy = st.selectbox(
        "Select strategy:",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy"]
    )

    st.write(" ")

    # Strategy parameters
    if strategy == "SMA Crossover":
        sma_short = st.number_input("Short SMA window", min_value=5, max_value=100, value=20)
        sma_long = st.number_input("Long SMA window", min_value=20, max_value=300, value=100)

    elif strategy == "RSI Strategy":
        rsi_window = st.number_input("RSI window", min_value=5, max_value=50, value=14)
        rsi_buy = st.number_input("RSI Buy Threshold", min_value=5, max_value=50, value=30)
        rsi_sell = st.number_input("RSI Sell Threshold", min_value=50, max_value=95, value=70)

    # --- Load data ---
    prices = get_price_history(asset, days=days)

    if prices.empty:
        st.warning("No data available for this asset.")
        st.stop()

    prices["returns"] = prices["price"].pct_change()

    # ------------------------------
    # 🔥 Strategy Implementations
    # ------------------------------

    # BUY & HOLD
    bh_curve = (1 + prices["returns"].fillna(0)).cumprod()

    # SMA CROSSOVER
    if strategy == "SMA Crossover":
        df = prices.copy()
        df["SMA_short"] = df["price"].rolling(sma_short).mean()
        df["SMA_long"] = df["price"].rolling(sma_long).mean()

        df["signal"] = 0
        df["signal"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
        df["position"] = df["signal"].shift(1).fillna(0)

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

    # RSI STRATEGY
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
        df.loc[df["RSI"] < rsi_buy, "position"] = 1      # BUY
        df.loc[df["RSI"] > rsi_sell, "position"] = 0     # SELL
        df["position"] = df["position"].ffill().fillna(0)

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

    else:
        strat_curve = bh_curve  # Buy & Hold by default

    # ------------------------------
    # 📉 Max Drawdown Function
    # ------------------------------
    def max_drawdown(series):
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        return drawdown.min()

    mdd = max_drawdown(strat_curve)

    # ------------------------------
    # 📊 DISPLAY CHART
    # ------------------------------
    st.subheader(f"Price vs Strategy — {asset}")
    st.line_chart(
        {
            "Price": prices["price"],
            "Strategy value": strat_curve
        }
    )

    # ------------------------------
    # 📈 Performance Metrics
    # ------------------------------
    st.subheader("Performance Metrics")

    trading_days = 252
    avg_daily = float(prices["returns"].mean())
    vol_daily = float(prices["returns"].std())

    annual_ret = (1 + avg_daily) ** trading_days - 1
    annual_vol = vol_daily * np.sqrt(trading_days)

    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Annualized Return", f"{annual_ret:.2%}")
        st.metric("Annualized Volatility", f"{annual_vol:.2%}")

    with col2:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
        st.metric("Max Drawdown", f"{mdd:.2%}")



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
