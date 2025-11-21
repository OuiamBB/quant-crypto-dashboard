import streamlit as st
import numpy as np
import pandas as pd
from data_fetch import get_price_history, get_multi_price_history
import time
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------
#                      STYLING (LEVEL 3 - MAX)
# ---------------------------------------------------------------------

st.set_page_config(page_title="Crypto Quant Dashboard", layout="wide")
# Auto-refresh every 5 minutes (300 sec)
st.markdown("""
    <meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)




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




# ---------------------------------------------------------------------
#                             SIDEBAR
# ---------------------------------------------------------------------

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Single Asset (Ouiam)", "Portfolio (Erian)", "Daily Reports (Auto)"]
)


# ---------------------------------------------------------------------
#                               HOME PAGE
# ---------------------------------------------------------------------

if page == "Home":

    st.markdown("""
        <h1 style='text-align:center; color:white; font-size:50px; font-weight:700;'>
            🚀 Crypto Quant Dashboard
        </h1>

        <h3 style='text-align:center; color:#4CAF50; margin-top:-10px;'>
            Real-time Crypto Analytics • Machine Learning • Quant Finance
        </h3>

        <br>

        <p style='text-align:center; color:#CCCCCC; font-size:18px; max-width:850px; margin:auto;'>
            Welcome to the Crypto Quant Dashboard — a complete analytics platform 
            developed for the A4 IF Python/Linux/Git project. 
            Explore single-asset strategies, build optimized portfolios, analyze correlations, 
            and access automatically generated daily reports.
        </p>

        <div class="divider"></div>
    """, unsafe_allow_html=True)

    # -----------------------------------------------------------------
    # 🔍 Quick Overview Cards
    # -----------------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        metric_card("Quant A", "Single Asset Analysis", color="#2196F3")
    with col2:
        metric_card("Quant B", "Portfolio Optimization", color="#4CAF50")
    with col3:
        metric_card("Daily Reports", "Auto-generated", color="#E53935")

    st.write("")

    # -----------------------------------------------------------------
    # 📄 Latest Daily Report Preview
    # -----------------------------------------------------------------

    st.markdown("<h3 style='color:white;'>📄 Latest Daily Report</h3>", unsafe_allow_html=True)

    import os
    reports_dir = "/home/obous/quant-crypto-dashboard/reports"

    if os.path.exists(reports_dir):
        files = sorted(os.listdir(reports_dir))
        if len(files) > 0:
            last_report = files[-1]
            df = pd.read_csv(f"{reports_dir}/{last_report}")

            st.write(f"**Latest file:** `{last_report}`")

            # Download button
            with open(f"{reports_dir}/{last_report}", "rb") as f:
                st.download_button(
                    "⬇️ Download latest report",
                    data=f,
                    file_name=last_report,
                    mime="text/csv"
                )

            # Preview
            st.dataframe(df.head(), use_container_width=True)
        else:
            st.info("No reports available yet.")
    else:
        st.warning("Reports folder not found.")

    st.write("")
    st.write("")

    # -----------------------------------------------------------------
    # 👥 Team Section
    # -----------------------------------------------------------------

    st.markdown("<h3 style='text-align:center; color:white;'>👥 Project Team</h3>", unsafe_allow_html=True)

    colA, colB = st.columns(2)

    with colA:
        metric_card(
            "Ouiam BOUSSAID BENCHAARA",
            "Single Asset Analysis (Quant A)",
            color="#2196F3"
        )

    with colB:
        metric_card(
            "Erian STANLEY YOGARAJ",
            "Portfolio Optimization (Quant B)",
            color="#4CAF50"
        )

    st.write("")
    st.markdown("<p style='text-align:center; color:#777; font-size:15px;'>ESILV A4 IF — 2024/2025</p>", unsafe_allow_html=True)



# ---------------------------------------------------------------------
#                       DAILY REPORTS (AUTO)
# ---------------------------------------------------------------------

elif page == "Daily Reports (Auto)":
    st.subheader("📄 Daily Automated Reports")

    import os

    reports_dir = "/home/obous/quant-crypto-dashboard/reports"

    if not os.path.exists(reports_dir):
        st.warning("No reports directory found.")
        st.stop()

    files = sorted(os.listdir(reports_dir))

    if len(files) == 0:
        st.info("No reports have been generated yet.")
        st.stop()

    # List reports
    st.write("### Available Reports:")
    for f in files:
        file_path = f"{reports_dir}/{f}"

        with open(file_path, "rb") as report_file:
            st.download_button(
                label=f"Download {f}",
                data=report_file,
                file_name=f,
                mime="text/csv"
            )

    # Show last report
    last_report = files[-1]
    st.write("### 📊 Latest Report")
    st.write(f"**File:** {last_report}")

    df = pd.read_csv(f"{reports_dir}/{last_report}")
    st.dataframe(df)


# ---------------------------------------------------------------------
#                       SINGLE ASSET (OUIAM)
# ---------------------------------------------------------------------

elif page == "Single Asset (Ouiam)":
    st.subheader("Single Asset Crypto Analysis")

    # --- Inputs ---
    asset = st.selectbox("Select crypto asset:", ["BTC-USD", "ETH-USD"], index=0)

    days = st.slider(
        "Time window (days):",
        min_value=30,
        max_value=365,
        value=180,
        step=10,
    )

    strategy = st.selectbox(
        "Select strategy:",
        ["Buy & Hold", "SMA Crossover", "RSI Strategy"],
    )

    st.write(" ")

    # --- Extra parameters depending on strategy ---
    sma_short = sma_long = None
    rsi_window = rsi_buy = rsi_sell = None

    if strategy == "SMA Crossover":
        sma_short = st.number_input(
            "Short SMA window",
            min_value=5,
            max_value=100,
            value=20,
        )
        sma_long = st.number_input(
            "Long SMA window",
            min_value=20,
            max_value=300,
            value=100,
        )

    elif strategy == "RSI Strategy":
        rsi_window = st.number_input(
            "RSI window",
            min_value=5,
            max_value=50,
            value=14,
        )
        rsi_buy = st.number_input(
            "RSI Buy Threshold",
            min_value=5,
            max_value=50,
            value=30,
        )
        rsi_sell = st.number_input(
            "RSI Sell Threshold",
            min_value=50,
            max_value=95,
            value=70,
        )

    # --- Load data ---
    prices = get_price_history(asset, days=days)

    if prices.empty:
        st.warning("No data available for this asset.")
        st.stop()

    prices["returns"] = prices["price"].pct_change()

    # ------------------------------
    # 🔥 Strategy Implementations
    # ------------------------------

    df = prices.copy()

    # BUY & HOLD baseline
    bh_curve = (1 + df["returns"].fillna(0)).cumprod()
    strat_curve = bh_curve.copy()  # default

    indicator_df = None  # for SMA/RSI visualisation

    # SMA CROSSOVER
    if strategy == "SMA Crossover":
        df["SMA_short"] = df["price"].rolling(sma_short).mean()
        df["SMA_long"] = df["price"].rolling(sma_long).mean()

        df["signal"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
        df["position"] = df["signal"].shift(1).fillna(0)  # trade next day

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

        indicator_df = df[["price", "SMA_short", "SMA_long"]]

    # RSI STRATEGY
    elif strategy == "RSI Strategy":
        delta = df["price"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(rsi_window).mean()
        avg_loss = loss.rolling(rsi_window).mean()

        rs = avg_gain / avg_loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df["position"] = 0
        df.loc[df["RSI"] < rsi_buy, "position"] = 1   # BUY
        df.loc[df["RSI"] > rsi_sell, "position"] = 0  # SELL
        df["position"] = df["position"].ffill().fillna(0)

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

        indicator_df = df[["RSI"]]

    # ------------------------------
    # 📉 Max Drawdown Function
    # ------------------------------
    def max_drawdown(series):
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        return drawdown.min()

    mdd = max_drawdown(strat_curve)

    # ------------------------------
    # 📊 MAIN CHART: Price vs Strategy
    # ------------------------------
    st.subheader(f"Price vs Strategy — {asset}")
    st.line_chart(
        {
            "Price": df["price"],
            "Strategy value": strat_curve,
        }
    )

    # ------------------------------
    # 📈 INDICATOR CHARTS (SMA / RSI)
    # ------------------------------
    if strategy == "SMA Crossover" and indicator_df is not None:
        st.subheader("Moving Averages (SMA)")
        st.line_chart(
            {
                "Price": indicator_df["price"],
                "SMA Short": indicator_df["SMA_short"],
                "SMA Long": indicator_df["SMA_long"],
            }
        )

    elif strategy == "RSI Strategy" and indicator_df is not None:
        st.subheader("RSI Indicator")
        st.line_chart(indicator_df["RSI"])

        st.caption(
            f"RSI Strategy: BUY when RSI < {rsi_buy}, SELL when RSI > {rsi_sell}."
        )

    # ------------------------------
    # 📈 Performance Metrics (cards)
    # ------------------------------
    st.subheader("Performance Metrics")

    trading_days = 252
    avg_daily = float(df["returns"].mean())
    vol_daily = float(df["returns"].std())

    annual_ret = (1 + avg_daily) ** trading_days - 1
    annual_vol = vol_daily * np.sqrt(trading_days)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan

    col1, col2 = st.columns(2)

    with col1:
        metric_card("Annualized Return", f"{annual_ret:.2%}")
        metric_card("Annualized Volatility", f"{annual_vol:.2%}")

    with col2:
        metric_card("Sharpe Ratio", f"{sharpe:.2f}", color="#FF9800")
        metric_card("Max Drawdown", f"{mdd:.2%}", color="#E53935")

    # (Optionnel) afficher les dernières lignes de la stratégie
    with st.expander("Show last signals / data"):
        st.dataframe(df.tail(10))

    
    # --- Prediction Model (BONUS) ---
    n_days_future = 14
    model = LinearRegression()

    X = np.arange(len(df)).reshape(-1,1)
    y = df["price"].values

    model.fit(X, y)

    future_X = np.arange(len(df), len(df)+n_days_future).reshape(-1,1)
    preds = model.predict(future_X)

    df.index = df.index.date
    st.subheader("14-day Price Prediction")
    st.line_chart(
        {
            "Historical price": df["price"],
            "Predicted price": pd.Series(preds, index=pd.RangeIndex(start=len(df), stop=len(df)+n_days_future)),
        }
    )





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
