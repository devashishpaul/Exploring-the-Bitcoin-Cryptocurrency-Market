import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# --- Load BTC daily data ---
df = pd.read_csv("BTC-Daily.csv", parse_dates=["date"])
df = df.sort_values("date")

# --- Basic clean ---
# Ensure correct numeric types
df["close"] = pd.to_numeric(df["close"], errors="coerce")
df["Volume USD"] = pd.to_numeric(df["Volume USD"], errors="coerce")
df = df.dropna(subset=["close", "Volume USD"])

# --- Feature engineering ---
# Daily returns
df["return"] = df["close"].pct_change().replace([np.inf, -np.inf], np.nan)
# Log returns
df["log_return"] = np.log(df["close"] / df["close"].shift(1))
# 30-day rolling volatility (annualized from daily)
ROLL = 30
df["roll_vol_30d"] = df["return"].rolling(ROLL).std() * np.sqrt(365)

# --- Visuals ---
# 1) Price over time
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["close"], label="BTC Price")
plt.title("BTC Price Over Time")
plt.xlabel("Date"); plt.ylabel("Price (USD)")
plt.legend()
plt.tight_layout()
plt.show()

# 2) 30-day rolling volatility
plt.figure(figsize=(10,5))
plt.plot(df["date"], df["roll_vol_30d"], color="orange", label="30D Annualized Volatility")
plt.title("BTC Rolling Volatility (30D)")
plt.xlabel("Date"); plt.ylabel("Volatility")
plt.legend()
plt.tight_layout()
plt.show()

# 3) Distribution of daily returns
plt.figure(figsize=(8,4))
plt.hist(df["return"].dropna(), bins=60, alpha=0.8)
plt.title("BTC Daily Returns Distribution")
plt.xlabel("Daily Return"); plt.ylabel("Count")
plt.tight_layout()
plt.show()

# --- Statistical tests ---
btc_ret = df["return"].dropna()

# Normality test (Jarque–Bera)
jb_stat, jb_p = stats.jarque_bera(btc_ret)
print(f"Jarque–Bera: stat={jb_stat:.3f}, p={jb_p:.3e}")
if jb_p < 0.05:
    print("Reject normality → returns not normally distributed.")
else:
    print("Fail to reject normality.")

# Mean return significance test (one-sample t-test against 0)
t_stat, t_p = stats.ttest_1samp(btc_ret, 0.0, nan_policy="omit")
print(f"One-sample t-test (mean=0): t={t_stat:.3f}, p={t_p:.3e}")

# --- Summary stats ---
summary = {
    "Mean daily return": btc_ret.mean(),
    "Median daily return": btc_ret.median(),
    "Annualized volatility": btc_ret.std() * np.sqrt(365),
    "Annualized return (approx)": (1 + btc_ret.mean())**365 - 1,
    "Sharpe ratio (daily)": btc_ret.mean() / btc_ret.std()
}
print("\n=== BTC Summary Stats ===")
for k,v in summary.items():
    print(f"{k}: {v:.6f}")
