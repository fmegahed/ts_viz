"""Generate demo CSV datasets for the time-series visualization app."""

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Reproducibility
np.random.seed(42)

# Resolve paths relative to the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. data/demo_single.csv  --  Monthly retail sales (Jan 2014 - Dec 2023)
# ---------------------------------------------------------------------------
def generate_single_series() -> pd.DataFrame:
    n = 120  # 10 years * 12 months
    dates = pd.date_range(start="2014-01-01", periods=n, freq="MS")

    months = np.arange(n)

    # Upward trend: start ~50 000, grow ~200 per month
    trend = 50_000 + 200 * months

    # Seasonal component: sin wave peaking in December (month index 11)
    # sin peaks at pi/2; December is month 11 (0-indexed within each year).
    # Shift so that sin(...) = 1 when month-of-year == 11 (December).
    month_of_year = months % 12
    seasonal = 8_000 * np.sin(2 * np.pi * (month_of_year - 2) / 12)

    # Random noise
    noise = np.random.normal(0, 2_000, size=n)

    sales = trend + seasonal + noise

    # Inject 2-3 anomaly spikes
    for idx in [36, 72, 100]:
        sales[idx] += 15_000

    df = pd.DataFrame({"date": dates, "sales": np.round(sales, 2)})
    return df


# ---------------------------------------------------------------------------
# 2. data/demo_multi_wide.csv  --  Quarterly revenue by region (Q1 2017 - Q4 2023)
# ---------------------------------------------------------------------------
def generate_multi_wide() -> pd.DataFrame:
    n = 28  # 7 years * 4 quarters
    dates = pd.date_range(start="2017-01-01", periods=n, freq="QS")

    quarters = np.arange(n)
    quarter_of_year = quarters % 4  # 0=Q1 .. 3=Q4

    regions = {
        "North": 100_000,
        "South": 80_000,
        "East": 120_000,
        "West": 90_000,
    }

    data: dict[str, object] = {"date": dates}

    for name, base in regions.items():
        trend = base + 800 * quarters
        seasonal = 5_000 * np.sin(2 * np.pi * quarter_of_year / 4)
        noise = np.random.normal(0, 3_000, size=n)
        data[name] = np.round(trend + seasonal + noise, 2)

    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# 3. data/demo_multi_long.csv  --  Daily stock prices for 20 tickers
#    (2022-01-03 to 2023-12-29, business days only)
# ---------------------------------------------------------------------------
def generate_multi_long() -> pd.DataFrame:
    trading_days = pd.bdate_range(start="2022-01-03", end="2023-12-29")

    # 20 simple four-letter tickers: AAAA, BBBB, ..., TTTT
    tickers = [chr(ord("A") + i) * 4 for i in range(20)]

    daily_drift = 0.0002
    daily_vol = 0.02

    frames: list[pd.DataFrame] = []

    for ticker in tickers:
        start_price = np.random.uniform(50, 500)
        n_days = len(trading_days)

        # Geometric Brownian Motion: S_t = S_0 * exp(cumsum(log returns))
        log_returns = np.random.normal(
            daily_drift - 0.5 * daily_vol**2, daily_vol, size=n_days
        )
        log_returns[0] = 0  # first day: price = start_price
        prices = start_price * np.exp(np.cumsum(log_returns))

        frames.append(
            pd.DataFrame(
                {
                    "date": trading_days,
                    "ticker": ticker,
                    "price": np.round(prices, 2),
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    single = generate_single_series()
    single.to_csv(DATA_DIR / "demo_single.csv", index=False)
    print(f"Wrote {len(single)} rows -> {DATA_DIR / 'demo_single.csv'}")

    wide = generate_multi_wide()
    wide.to_csv(DATA_DIR / "demo_multi_wide.csv", index=False)
    print(f"Wrote {len(wide)} rows -> {DATA_DIR / 'demo_multi_wide.csv'}")

    long = generate_multi_long()
    long.to_csv(DATA_DIR / "demo_multi_long.csv", index=False)
    print(f"Wrote {len(long)} rows -> {DATA_DIR / 'demo_multi_long.csv'}")


if __name__ == "__main__":
    main()
