import streamlit as st
import numpy as np
import pandas as pd
from data_fetch import get_price_history, get_multi_price_history
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import time
from sklearn.linear_model import LinearRegression


def compute_drawdown_series(series):
    """
    Computes the drawdown series from a portfolio value time series.
    Drawdown = (value / rolling max) - 1
    """
    rolling_max = series.cummax()
    drawdown = (series / rolling_max) - 1
    return drawdown
    
st.set_page_config(page_title="Crypto Quant Dashboard", layout="wide")
st.markdown("""
    <meta http-equiv="refresh" content="300">
""", unsafe_allow_html=True)




st.markdown("""
    <style>

    /* MAIN BACKGROUND */
    [data-testid="stAppViewContainer"] {
        background-color: #000000 !important;
    }

    /* MAIN CONTENT BACKGROUND */
    [data-testid="stAppViewContainer"] > .main {
        background-color: #000000 !important;
    }

    /* SIDEBAR BACKGROUND */
    [data-testid="stSidebar"] {
        background-color: #000000 !important;
    }


    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background-color: #000000;
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

    /* --------------------------------------------- */
    /* SIDEBAR NAVIGATION BUTTONS (BLACK & WHITE)    */
    /* --------------------------------------------- */

    .sidebar-button {
        display: block;
        width: 100%;
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 6px;

        background-color: #000000 !important;   /* BLACK BUTTON */
        color: #FFFFFF !important;              /* WHITE TEXT */
        text-align: left;
        font-weight: 600;
        border: 1px solid #333333;

        cursor: pointer;
        transition: all 0.2s ease-in-out;
    }

    /* Hover effect */
    .sidebar-button:hover {
        background-color: #1a1a1a !important;  /* Slightly lighter black */
        border-color: #D9A94C !important;      /* GOLD BORDER on hover */
    }

    /* Active button */
    .sidebar-button-active {
        background-color: #1f1f1f !important;
        border-left: 4px solid #D9A94C !important;  /* GOLD indicator */
        color: #FFFFFF !important;
    }

    .button-nav {
        background-color: #000000;
        color: black;
        padding: 12px 18px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 10px;
        font-weight: 600;
        cursor: pointer;
        border: 2px solid #a77f2c;
        transition: 0.2s;
    }

    .button-nav:hover {
        background-color: #e8c879;
        color: black;
        transform: translateX(5px);
    }

    .button-nav-active {
        background-color: #a77f2c;
        color: black;
        border: 2px solid #000000;
        padding: 12px 18px;
        border-radius: 10px;
        font-weight: 700;
        margin-bottom: 10px;
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

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Home", "Single Asset (Ouiam)", "Portfolio (Erian)", "Daily Reports (Auto)"]
)

if page == "Home":

    st.markdown(
        "<h1 style='text-align:center; color:white; font-size:48px; font-weight:700;'>"
        "Crypto Quant Dashboard"
        "</h1>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<p style='text-align:center; color:#9E9E9E; font-size:18px; margin-top:-10px;'>"
        "Crypto analytics • Quant strategies • Portfolio optimization"
        "</p>",
        unsafe_allow_html=True
    )

    

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    st.markdown("<h3 style='color:white; margin-top:10px;'>Quick Access</h3>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card("Quant A", "Single Asset Strategies (SMA / RSI)", color="#2196F3")
        st.caption("Go to: **Single Asset (Ouiam)**")

    with c2:
        metric_card("Quant B", "Portfolio • Correlation • Markowitz", color="#4CAF50")
        st.caption("Go to: **Portfolio (Erian)**")

    with c3:
        metric_card("Daily Reports", "CSV reports generated automatically", color="#E53935")
        st.caption("Go to: **Daily Reports (Auto)**")

    st.write("")

    st.markdown("<h3 style='color:white;'>Latest Daily Report</h3>", unsafe_allow_html=True)

    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../quant-crypto-dashboard
    reports_dir = os.path.join(BASE_DIR, "reports")


    if os.path.exists(reports_dir):
        files = sorted([f for f in os.listdir(reports_dir) if f.endswith(".csv")])
        if len(files) > 0:
            last_report = files[-1]
            df = pd.read_csv(f"{reports_dir}/{last_report}")

            top_left, top_right = st.columns([3, 1])
            with top_left:
                st.markdown(f"<span style='color:#9E9E9E;'>Latest file:</span> <span style='color:#E0E0E0; font-weight:700;'>{last_report}</span>", unsafe_allow_html=True)
            with top_right:
                with open(f"{reports_dir}/{last_report}", "rb") as f:
                    st.download_button(
                        "⬇️ Download",
                        data=f,
                        file_name=last_report,
                        mime="text/csv",
                        use_container_width=True
                    )

            preview = df.copy()
            preview = preview.drop(columns=["Unnamed: 0"], errors="ignore")
            st.dataframe(preview, use_container_width=True, height=220)

        else:
            st.info("No reports available yet.")
    else:
        st.warning("Reports folder not found.")

    st.write("")

    st.markdown("<h3 style='text-align:center; color:white; margin-top:20px;'>Project Team</h3>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)

    with t1:
        metric_card("Ouiam BOUSSAID BENCHAARA", "Quant A — Single Asset", color="#2196F3")

    with t2:
        metric_card("Erian STANLEY YOGARAJ", "Quant B — Portfolio", color="#4CAF50")

    st.markdown("<p style='text-align:center; color:#777; font-size:14px; margin-top:10px;'>ESILV A4 IF — 2024/2025</p>", unsafe_allow_html=True)

# Single Asset (Ouiam)

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

    prices = get_price_history(asset, days=days)

    if prices.empty:
        st.warning("No data available for this asset.")
        st.stop()

    prices["returns"] = prices["price"].pct_change()

    df = prices.copy()
    bh_curve = (1 + df["returns"].fillna(0)).cumprod()
    strat_curve = bh_curve.copy()  

    indicator_df = None  
    if strategy == "SMA Crossover":
        df["SMA_short"] = df["price"].rolling(sma_short).mean()
        df["SMA_long"] = df["price"].rolling(sma_long).mean()

        df["signal"] = (df["SMA_short"] > df["SMA_long"]).astype(int)
        df["position"] = df["signal"].shift(1).fillna(0)  

        df["strategy_returns"] = df["position"] * df["returns"]
        strat_curve = (1 + df["strategy_returns"].fillna(0)).cumprod()

        indicator_df = df[["price", "SMA_short", "SMA_long"]]

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

    def max_drawdown(series):
        rolling_max = series.cummax()
        drawdown = (series - rolling_max) / rolling_max
        return drawdown.min()

    mdd = max_drawdown(strat_curve)

    st.subheader(f"Price vs Strategy — {asset}")
    st.line_chart(
        {
            "Price": df["price"],
            "Strategy value": strat_curve,
        }
    )

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

    st.subheader("Performance Metrics")

    trading_days = 252

    if strategy == "Buy & Hold":
        used_returns = df["returns"]
    else:
        used_returns = df["strategy_returns"]

    avg_daily = float(used_returns.mean())
    vol_daily = float(used_returns.std())

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

    with st.expander("Show last signals / data"):
        st.dataframe(df.tail(10))

    
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


# Portfolio (Erian)

elif page == "Portfolio (Erian)":

    st.header(" Multi-Asset Crypto Portfolio Analysis")

    all_tickers = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"]
    tickers = st.multiselect("Select crypto assets:", all_tickers, default=all_tickers)

    days = st.slider("Time window (days):", 30, 365, 180, step=10)

    rebal_freq = st.selectbox(
        "Rebalancing frequency:",
        ["None", "Weekly", "Monthly"]
    )

    vol_target = st.checkbox("Enable volatility targeting (Risk-based weights)")

    risk_parity = st.checkbox("Enable Risk-Parity weights")

    if len(tickers) == 0:
        st.warning("Please select at least one asset.")

    else:
        prices = get_multi_price_history(tickers, days=days)
        returns = prices.pct_change().dropna()

        st.subheader(" Daily Close Prices")
        st.line_chart(prices)

        st.subheader("Portfolio Weights")

        if risk_parity:
            vols = returns.std()
            inv_vol = 1 / vols
            weights = inv_vol / inv_vol.sum()
            st.info("Weights computed using **risk parity (inverse volatility)**.")

        elif vol_target:
            vols = returns.std()
            inv_vol = 1 / vols
            weights = inv_vol / inv_vol.sum()
            st.info("Weights computed automatically based on **inverse volatility**.")

        else:
            st.write("Custom weights (%):")
            weights_input = {
                t: st.number_input(
                    f"Weight for {t} (%)",
                    min_value=0.0, max_value=100.0,
                    value=100.0 / len(tickers), step=1.0
                )
                for t in tickers
            }
            weights = np.array(list(weights_input.values()))
            weights = weights / weights.sum()

        st.write("**Normalized weights:**")
        st.json({t: f"{w:.1%}" for t, w in zip(tickers, weights)})

        portfolio_returns = (returns * weights).sum(axis=1)

        if rebal_freq == "Weekly":
            portfolio_returns = portfolio_returns.resample("W").mean().reindex(portfolio_returns.index, method="pad")
        elif rebal_freq == "Monthly":
            portfolio_returns = portfolio_returns.resample("M").mean().reindex(portfolio_returns.index, method="pad")

        portfolio_value = (1 + portfolio_returns).cumprod()

        st.subheader(" Portfolio Value (base = 1.0)")
        st.line_chart(portfolio_value)

        st.subheader(" Portfolio Performance Metrics")

        trading_days = 252
        avg_daily_ret = portfolio_returns.mean()
        daily_vol = portfolio_returns.std()
        ann_return = (1 + avg_daily_ret) ** trading_days - 1
        ann_vol = daily_vol * np.sqrt(trading_days)
        sharpe = ann_return / ann_vol if ann_vol > 0 else np.nan

        neg_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = neg_returns.std() if len(neg_returns) > 0 else np.nan
        sortino = ann_return / (downside_std * np.sqrt(252)) if downside_std > 0 else np.nan

        mdd = max_drawdown(portfolio_value)
        skew = returns.skew().mean()
        kurt = returns.kurt().mean()

        col1, col2 = st.columns(2)

        with col1:
            metric_card("Average daily return", f"{avg_daily_ret:.4%}")
            metric_card("Daily volatility", f"{daily_vol:.4%}")
            metric_card("Annualized return", f"{ann_return:.2%}")

        with col2:
            metric_card("Annualized volatility", f"{ann_vol:.2%}")
            metric_card("Sharpe ratio", f"{sharpe:.2f}", "#FF9800")
            metric_card("Sortino ratio", f"{sortino:.2f}")
            metric_card("Max drawdown", f"{mdd:.2%}", "#E53935")

        metric_card("Skewness", f"{skew:.2f}")
        metric_card("Kurtosis", f"{kurt:.2f}")

        st.subheader(" Comparison: Portfolio vs BTC")

        if "BTC-USD" in prices.columns:
            btc_norm = prices["BTC-USD"] / prices["BTC-USD"].iloc[0]
            comp = pd.DataFrame({"Portfolio": portfolio_value, "BTC-USD": btc_norm})
            st.line_chart(comp)
        else:
            st.info("BTC-USD not selected → comparison unavailable.")

        st.subheader(" Correlation Matrix")
        st.dataframe(
            returns.corr().style.background_gradient(cmap="coolwarm").format("{:.2f}")
        )

        st.subheader(" Drawdown Analysis")
        dd = compute_drawdown_series(portfolio_value)
        st.line_chart(dd)

        st.subheader(" Markowitz Portfolio Optimization")

        def markowitz_opt(returns):
            mean = returns.mean()
            cov = returns.cov()
            n = len(mean)

            def portfolio_vol(w):
                return np.sqrt(w.T @ cov @ w)

            cons = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            bounds = [(0, 1) for _ in range(n)]

            w0 = np.ones(n) / n

            res = minimize(portfolio_vol, w0, bounds=bounds, constraints=cons)
            return res.x

        opt_weights = markowitz_opt(returns)
        st.write("Optimal minimum-variance weights:")
        st.json({t: f"{w:.1%}" for t, w in zip(tickers, opt_weights)})

        st.subheader(" Forecast using ARIMA")

        ts = portfolio_value.copy()
        ts.index = pd.to_datetime(ts.index)
        ts = ts.asfreq("D").ffill()

        try:
            model = ARIMA(ts, order=(1,1,1))
            fitted = model.fit()
            forecast = fitted.forecast(steps=7)

            forecast_df = pd.DataFrame({
                "date": pd.date_range(ts.index[-1], periods=8, freq="D")[1:],
                "forecast": forecast
            }).set_index("date")

            st.line_chart(forecast_df)
            st.success("ARIMA forecast generated successfully.")

        except Exception as e:
            st.error(f"ARIMA failed: {e}")

        st.download_button(
            "Download portfolio data (CSV)",
            prices.to_csv().encode(),
            "portfolio_data.csv",
            "text/csv"
        )

# Daily Reports 

elif page == "Daily Reports (Auto)":
    st.subheader(" Daily Automated Reports")

    import os
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(BASE_DIR, "reports")


    if not os.path.exists(reports_dir):
        st.warning("No reports directory found.")
        st.stop()

    files = sorted(os.listdir(reports_dir))

    if len(files) == 0:
        st.info("No reports have been generated yet.")
        st.stop()

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

    last_report = files[-1]
    st.write("###  Latest Report")
    st.write(f"**File:** {last_report}")

    df = pd.read_csv(f"{reports_dir}/{last_report}")
    st.dataframe(df)

