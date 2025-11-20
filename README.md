# Crypto Quant Dashboard

A quantitative cryptocurrency analysis dashboard built with Python, Streamlit, Git, and Linux.

This project was developed as part of the **A4 IF – Python, Linux & Git** module at ESILV.
It consists of two main analytical components:

- **Quant A – Single Asset Analysis (Ouiam)**
- **Quant B – Multi-Asset Portfolio Analysis (Erian)**

---

## 1. Project Objectives

- Retrieve historical cryptocurrency data using the yfinance API.
- Build an interactive dashboard with Streamlit.
- Implement single-asset analytics and trading strategies.
- Implement a multi-asset portfolio model with performance metrics.
- Compute return, volatility, Sharpe ratio, correlation, and max drawdown.
- Generate an automated daily financial report using Linux cron.
- Ensure proper Git workflow (branches, commits, collaboration).
- Deploy the dashboard on a Linux environment with 24/7 uptime.

---

## 2. Features

### Quant A – Single Asset (Ouiam)
- Asset analyzed: BTC-USD
- Daily close price visualization
- Daily and annualized performance metrics
- Technical strategies (Buy & Hold, Moving Average Crossover)
- Metrics:
  - Average daily return
  - Daily/annualized volatility
  - Annualized return
  - Sharpe ratio
  - Max drawdown

### Quant B – Multi-Asset Portfolio (Erian)
- Assets: BTC-USD, ETH-USD, BNB-USD, SOL-USD
- Multi-asset price visualization
- Portfolio returns based on user-defined weights
- Daily and annualized volatility
- Annualized return
- Sharpe ratio
- Max drawdown
- Correlation matrix between assets
- Portfolio vs BTC normalized comparison
- Dynamic weight adjustment in real-time

---

## 3. Project Structure

```
quant-crypto-dashboard/
│
├── app/
│   ├── main.py              # Streamlit dashboard
│   ├── data_fetch.py        # Data retrieval logic
│
├── report.py                # Daily automated report script
├── run.sh                   # Background execution script (Linux/WSL)
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 4. Installation

### 4.1 Clone the project

```bash
git clone https://github.com/Erian15/quant-crypto-dashboard.git
cd quant-crypto-dashboard
```

### 4.2 Create and activate virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4.3 Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Running the Dashboard

### Standard execution

```bash
streamlit run app/main.py
```

### 24/7 execution on Linux/WSL (using nohup)

1. Make the script executable:

```bash
chmod +x run.sh
```

2. Run it in the background:

```bash
./run.sh
```

The dashboard now runs continuously in background mode.

### Verifying that Streamlit is running

```bash
ps aux | grep streamlit
```

### Stopping the dashboard

```bash
pkill -f streamlit
```

---

## 6. Daily Automated Report (Cron Job)

A daily report is automatically generated at **20:00** using cron.

### Cron configuration:

```
0 20 * * * /home/tkg/quant-crypto-dashboard/venv/bin/python3 /home/tkg/quant-crypto-dashboard/report.py >> /home/tkg/cron.log 2>&1
```

This creates files such as:

```
~/daily_report_2025-11-20.csv
```

The report includes for each asset:

- Open price
- Close price
- Daily volatility
- Max drawdown

---

## 7. Deployment Verification

### Check process

```bash
ps aux | grep streamlit
```

### Check port 8501

```bash
netstat -tulnp | grep 8501
```

### Check logs

```bash
tail -f streamlit.log
```

If Streamlit is running and the port is active, the dashboard is running 24/7.

---

## 8. Team Roles

### Ouiam – Quant A
- BTC-USD analysis
- Technical strategies
- Single-asset performance calculations
- Streamlit integration of the single-asset module

### Erian – Quant B
- Multi-asset portfolio design
- Portfolio performance metrics
- Correlation analysis
- Linux deployment (nohup + cron)
- Automation of daily reporting

---

## 9. Summary

This project combines:

- Python programming
- Financial data processing
- Portfolio analytics
- Streamlit dashboard creation
- Git collaboration workflow
- Linux automation (shell + cron)

The result is a complete educational project demonstrating both technical and quantitative finance skills.

