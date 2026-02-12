"""Download real FRED datasets for the time-series visualization app.

Uses only ``urllib.request`` (stdlib) — no extra dependencies.

Series
------
* OHUR  — Ohio Unemployment Rate (%, Monthly, 1976–present)
* OHMFG — Ohio Manufacturing Employment (1000s, SA, Monthly)
* INMFG — Indiana Manufacturing Employment
* MIMFG — Michigan Manufacturing Employment
* TXMFG — Texas Manufacturing Employment
* CAMFG — California Manufacturing Employment

Run once locally, then commit the resulting CSVs::

    python scripts/download_fred_data.py
"""

from __future__ import annotations

import io
import urllib.request
from pathlib import Path

import pandas as pd

# Resolve paths relative to the project root (parent of scripts/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

_FRED_CSV_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def _fetch_fred(series_id: str) -> pd.DataFrame:
    """Download a single FRED series and return a two-column DataFrame."""
    url = _FRED_CSV_URL.format(series_id=series_id)
    print(f"  Downloading {series_id} …")
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
        raw = resp.read().decode("utf-8")
    df = pd.read_csv(io.StringIO(raw))
    # FRED uses "." for missing values — coerce to NaN
    df.columns = ["date", series_id]
    df[series_id] = pd.to_numeric(df[series_id], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    df = df.dropna()
    return df


# ---------------------------------------------------------------------------
# 1. Single series: Ohio Unemployment Rate
# ---------------------------------------------------------------------------
def build_ohio_unemployment() -> pd.DataFrame:
    df = _fetch_fred("OHUR")
    df = df.rename(columns={"OHUR": "unemployment_rate"})
    return df


# ---------------------------------------------------------------------------
# 2. Multi-series wide: Manufacturing Employment by State
# ---------------------------------------------------------------------------
_MFG_SERIES = {
    "OHMFG": "Ohio",
    "INMFG": "Indiana",
    "MIMFG": "Michigan",
    "TXMFG": "Texas",
    "CAMFG": "California",
}


def build_manufacturing_wide() -> pd.DataFrame:
    frames = []
    for sid, state_name in _MFG_SERIES.items():
        df = _fetch_fred(sid)
        df = df.rename(columns={sid: state_name})
        frames.append(df)

    # Inner-join on date so all states share the same date range
    wide = frames[0]
    for f in frames[1:]:
        wide = wide.merge(f, on="date", how="inner")

    wide = wide.sort_values("date").reset_index(drop=True)
    return wide


# ---------------------------------------------------------------------------
# 3. Multi-series long: same data melted
# ---------------------------------------------------------------------------
def build_manufacturing_long(wide: pd.DataFrame) -> pd.DataFrame:
    long = wide.melt(
        id_vars="date",
        var_name="state",
        value_name="manufacturing_employment",
    )
    long = long.sort_values(["date", "state"]).reset_index(drop=True)
    return long


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Downloading FRED data ===\n")

    # 1. Ohio Unemployment
    single = build_ohio_unemployment()
    out = DATA_DIR / "demo_ohio_unemployment.csv"
    single.to_csv(out, index=False)
    print(f"  -> {len(single)} rows  =>  {out}\n")

    # 2. Manufacturing wide
    wide = build_manufacturing_wide()
    out = DATA_DIR / "demo_manufacturing_wide.csv"
    wide.to_csv(out, index=False)
    print(f"  -> {len(wide)} rows  =>  {out}\n")

    # 3. Manufacturing long
    long = build_manufacturing_long(wide)
    out = DATA_DIR / "demo_manufacturing_long.csv"
    long.to_csv(out, index=False)
    print(f"  -> {len(long)} rows  =>  {out}\n")

    print("Done.")


if __name__ == "__main__":
    main()
