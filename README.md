# Exploring the Bitcoin Cryptocurrency Market
**Short description**: Data cleaning, feature engineering, statistical
comparison and visualization of Bitcoin vs major cryptocurrencies (ETH, BNB,
SOL, XRP). Includes scripts to clean raw data, compute returns, run statistical
tests, and produce publication-ready charts.
## Contents
- `scripts/` — ETL & plotting scripts
- `notebooks/` — exploratory notebook with walkthrough and visuals
- `data/` — sample/placeholder
- `artifacts/` — generated CSVs and figures
- `requirements.txt` — Python dependencies
## Quick start
1. Clone the repo
2. Create virtual env and install deps:
 ```bash
 python -m venv venv
 source venv/bin/activate # or venv\Scripts\activate on Windows
 pip install -r requirements.txt
Plot price trend, rolling volatility, and return distribution.

Run normality and mean return t-tests.

Output summary statistics including Sharpe ratio and annualized return
