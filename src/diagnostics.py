"""Time-series diagnostics utilities.

Provides summary statistics, stationarity tests, trend estimation,
autocorrelation analysis, seasonal decomposition, rolling statistics,
year-over-year change computation, and multi-series summaries.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy import stats
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose, DecomposeResult


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SummaryStats:
    """Container for univariate time-series summary statistics."""

    count: int
    missing_count: int
    missing_pct: float
    min_val: float
    max_val: float
    mean_val: float
    median_val: float
    std_val: float
    p25: float
    p75: float
    date_start: pd.Timestamp
    date_end: pd.Timestamp
    date_span_days: int
    trend_slope: float
    trend_pvalue: float
    adf_statistic: float
    adf_pvalue: float


# ---------------------------------------------------------------------------
# Core helper functions
# ---------------------------------------------------------------------------

def compute_adf_test(series: pd.Series) -> tuple[float, float]:
    """Run the Augmented Dickey-Fuller test for stationarity.

    Parameters
    ----------
    series : pd.Series
        The time-series values (NaNs are dropped automatically).

    Returns
    -------
    tuple[float, float]
        ``(adf_statistic, p_value)``.  Returns ``(np.nan, np.nan)`` when the
        test cannot be performed (e.g. too few observations or constant data).
    """
    clean = series.dropna()
    if len(clean) < 2:
        return np.nan, np.nan
    try:
        result = adfuller(clean, autolag="AIC")
        return float(result[0]), float(result[1])
    except Exception:
        return np.nan, np.nan


def compute_trend_slope(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
) -> tuple[float, float]:
    """Estimate a linear trend via OLS on a numeric index.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain *date_col* and *y_col*.
    date_col : str
        Column with datetime-like values.
    y_col : str
        Column with numeric values.

    Returns
    -------
    tuple[float, float]
        ``(slope, p_value)`` from ``scipy.stats.linregress``.
        Returns ``(np.nan, np.nan)`` when the regression cannot be computed.
    """
    subset = df[[date_col, y_col]].dropna()
    if len(subset) < 2:
        return np.nan, np.nan
    try:
        x = np.arange(len(subset), dtype=float)
        y = subset[y_col].astype(float).values
        result = stats.linregress(x, y)
        return float(result.slope), float(result.pvalue)
    except Exception:
        return np.nan, np.nan


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_stats(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
) -> SummaryStats:
    """Compute a comprehensive set of summary statistics for a time series.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    date_col : str
        Name of the datetime column.
    y_col : str
        Name of the numeric value column.

    Returns
    -------
    SummaryStats
        Dataclass instance containing descriptive stats, date range info,
        trend slope / p-value, and ADF test results.
    """
    series = df[y_col]
    dates = pd.to_datetime(df[date_col])

    count = int(series.notna().sum())
    missing_count = int(series.isna().sum())
    total = len(series)
    missing_pct = (missing_count / total * 100.0) if total > 0 else 0.0

    min_val = float(series.min())
    max_val = float(series.max())
    mean_val = float(series.mean())
    median_val = float(series.median())
    std_val = float(series.std())
    p25 = float(series.quantile(0.25))
    p75 = float(series.quantile(0.75))

    date_start = dates.min()
    date_end = dates.max()
    date_span_days = int((date_end - date_start).days)

    trend_slope, trend_pvalue = compute_trend_slope(df, date_col, y_col)
    adf_statistic, adf_pvalue = compute_adf_test(series)

    return SummaryStats(
        count=count,
        missing_count=missing_count,
        missing_pct=missing_pct,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        median_val=median_val,
        std_val=std_val,
        p25=p25,
        p75=p75,
        date_start=date_start,
        date_end=date_end,
        date_span_days=date_span_days,
        trend_slope=trend_slope,
        trend_pvalue=trend_pvalue,
        adf_statistic=adf_statistic,
        adf_pvalue=adf_pvalue,
    )


# ---------------------------------------------------------------------------
# Autocorrelation / partial autocorrelation
# ---------------------------------------------------------------------------

def compute_acf_pacf(
    series: pd.Series,
    nlags: int = 40,
) -> tuple[NDArray, NDArray, NDArray, NDArray]:
    """Compute ACF and PACF with confidence intervals.

    Parameters
    ----------
    series : pd.Series
        The time-series values (NaNs are dropped automatically).
    nlags : int, optional
        Maximum number of lags (default 40).  Automatically reduced when the
        series is shorter than ``nlags + 1``.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, ndarray]
        ``(acf_values, acf_confint, pacf_values, pacf_confint)``

        * ``acf_values``  -- shape ``(nlags + 1,)``
        * ``acf_confint`` -- shape ``(nlags + 1, 2)``
        * ``pacf_values`` -- shape ``(nlags + 1,)``
        * ``pacf_confint`` -- shape ``(nlags + 1, 2)``
    """
    clean = series.dropna().values.astype(float)

    # Ensure nlags does not exceed what the data can support.
    max_possible = len(clean) - 1
    if max_possible < 1:
        raise ValueError(
            "Series has fewer than 2 non-NaN observations; "
            "cannot compute ACF/PACF."
        )
    nlags = min(nlags, max_possible)

    acf_values, acf_confint = acf(clean, nlags=nlags, alpha=0.05)
    pacf_values, pacf_confint = pacf(clean, nlags=nlags, alpha=0.05)

    return acf_values, acf_confint, pacf_values, pacf_confint


# ---------------------------------------------------------------------------
# Seasonal decomposition
# ---------------------------------------------------------------------------

def _infer_period(df: pd.DataFrame, date_col: str) -> int:
    """Best-effort period inference from the date column's frequency.

    Returns a sensible integer period or raises ``ValueError`` when the
    frequency cannot be determined.
    """
    dates = pd.to_datetime(df[date_col])
    freq = pd.infer_freq(dates)
    if freq is None:
        raise ValueError(
            "Cannot infer a regular frequency from the date column. "
            "Please supply an explicit 'period' argument or resample the "
            "data to a regular frequency before calling compute_decomposition."
        )

    # Map common frequency strings to typical seasonal periods.
    freq_upper = freq.upper()
    period_map: dict[str, int] = {
        "D": 365,
        "B": 252,       # business days in a year
        "W": 52,
        "SM": 24,       # semi-monthly
        "BMS": 12,
        "BM": 12,
        "MS": 12,
        "M": 12,        # calendar month end
        "ME": 12,       # month-end (pandas >= 2.2)
        "QS": 4,
        "Q": 4,
        "QE": 4,
        "BQ": 4,
        "AS": 1,
        "A": 1,
        "YS": 1,
        "Y": 1,
        "YE": 1,
        "H": 24,
        "T": 60,
        "MIN": 60,
        "S": 60,
    }

    # Strip leading digits (e.g. "2W" -> "W") to normalise anchored offsets.
    stripped = freq_upper.lstrip("0123456789")
    # Also strip any anchor suffix like "W-SUN" -> "W".
    base = stripped.split("-")[0]

    if base in period_map:
        return period_map[base]

    raise ValueError(
        f"Unable to map inferred frequency '{freq}' to a seasonal period. "
        "Please provide an explicit 'period' argument."
    )


def compute_decomposition(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    model: str = "additive",
    period: Optional[int] = None,
) -> DecomposeResult:
    """Decompose a time series into trend, seasonal, and residual components.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    date_col : str
        Datetime column name.
    y_col : str
        Numeric value column name.
    model : str, optional
        ``"additive"`` (default) or ``"multiplicative"``.
    period : int or None, optional
        Seasonal period.  When *None* the period is inferred from the date
        column's frequency.

    Returns
    -------
    statsmodels.tsa.seasonal.DecomposeResult

    Raises
    ------
    ValueError
        If a regular frequency cannot be inferred and *period* is not given.
    """
    ts = (
        df[[date_col, y_col]]
        .copy()
        .set_index(date_col)
        .sort_index()
    )
    ts.index = pd.to_datetime(ts.index)

    # Forward-fill / back-fill small gaps so decomposition doesn't fail on
    # a handful of interior NaNs.
    ts[y_col] = ts[y_col].ffill().bfill()

    if period is None:
        period = _infer_period(df, date_col)

    # Attempt to set a frequency on the index so that seasonal_decompose is
    # happy; fall back to the explicit period if this fails.
    if ts.index.freq is None:
        inferred = pd.infer_freq(ts.index)
        if inferred is not None:
            ts = ts.asfreq(inferred)
            ts[y_col] = ts[y_col].ffill().bfill()

    return seasonal_decompose(ts[y_col], model=model, period=period)


# ---------------------------------------------------------------------------
# Rolling statistics
# ---------------------------------------------------------------------------

def compute_rolling_stats(
    df: pd.DataFrame,
    y_col: str,
    window: int = 12,
) -> pd.DataFrame:
    """Add rolling mean and rolling standard deviation columns to *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Source data (not mutated).
    y_col : str
        Column over which rolling statistics are calculated.
    window : int, optional
        Rolling window size (default 12).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with two extra columns: ``rolling_mean`` and
        ``rolling_std``.
    """
    out = df.copy()
    out["rolling_mean"] = out[y_col].rolling(window=window, min_periods=1).mean()
    out["rolling_std"] = out[y_col].rolling(window=window, min_periods=1).std()
    return out


# ---------------------------------------------------------------------------
# Year-over-year change
# ---------------------------------------------------------------------------

def _offset_for_frequency(df: pd.DataFrame, date_col: str) -> pd.DateOffset:
    """Return a 1-year ``DateOffset`` appropriate to the series frequency."""
    dates = pd.to_datetime(df[date_col])
    freq = pd.infer_freq(dates)

    if freq is not None:
        freq_upper = freq.upper().lstrip("0123456789").split("-")[0]
        # For sub-monthly frequencies we shift by 365 days / 52 weeks etc.
        if freq_upper in {"D", "B"}:
            return pd.DateOffset(days=365)
        if freq_upper in {"W"}:
            return pd.DateOffset(weeks=52)
        if freq_upper in {"H", "T", "MIN", "S"}:
            return pd.DateOffset(days=365)

    # Default: shift by 12 months (works for M, Q, and annual data).
    return pd.DateOffset(months=12)


def compute_yoy_change(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
) -> pd.DataFrame:
    """Compute year-over-year absolute and percentage change.

    The number of periods to shift is determined from the inferred frequency
    of the date column.

    Parameters
    ----------
    df : pd.DataFrame
        Source data (not mutated).
    date_col : str
        Datetime column name.
    y_col : str
        Numeric value column name.

    Returns
    -------
    pd.DataFrame
        Copy of *df* sorted by *date_col* with additional columns
        ``yoy_abs_change`` and ``yoy_pct_change``.
    """
    out = df.copy().sort_values(date_col).reset_index(drop=True)
    out[date_col] = pd.to_datetime(out[date_col])

    # Determine the number of rows that correspond to ~1 year.
    freq = pd.infer_freq(out[date_col])
    if freq is not None:
        freq_upper = freq.upper().lstrip("0123456789").split("-")[0]
        period_map: dict[str, int] = {
            "D": 365,
            "B": 252,
            "W": 52,
            "SM": 24,
            "BMS": 12,
            "BM": 12,
            "MS": 12,
            "M": 12,
            "ME": 12,
            "QS": 4,
            "Q": 4,
            "QE": 4,
            "BQ": 4,
            "AS": 1,
            "A": 1,
            "YS": 1,
            "Y": 1,
            "YE": 1,
            "H": 8760,
            "T": 525600,
            "MIN": 525600,
            "S": 31536000,
        }
        base = freq_upper
        shift_periods = period_map.get(base, 12)
    else:
        # Fallback: assume monthly data.
        shift_periods = 12

    shifted = out[y_col].shift(shift_periods)
    out["yoy_abs_change"] = out[y_col] - shifted
    out["yoy_pct_change"] = out["yoy_abs_change"] / shifted.abs().replace(0, np.nan) * 100.0

    return out


# ---------------------------------------------------------------------------
# Multi-series summary
# ---------------------------------------------------------------------------

def compute_multi_series_summary(
    df: pd.DataFrame,
    date_col: str,
    y_cols: list[str],
) -> pd.DataFrame:
    """Produce a summary DataFrame with one row per value column.

    Parameters
    ----------
    df : pd.DataFrame
        Source data.
    date_col : str
        Datetime column name.
    y_cols : list[str]
        List of numeric column names to summarise.

    Returns
    -------
    pd.DataFrame
        Columns: ``variable``, ``count``, ``mean``, ``std``, ``min``,
        ``max``, ``trend_slope``, ``adf_pvalue``.
    """
    rows: list[dict] = []
    for col in y_cols:
        series = df[col]
        slope, _ = compute_trend_slope(df, date_col, col)
        _, adf_p = compute_adf_test(series)
        rows.append(
            {
                "variable": col,
                "count": int(series.notna().sum()),
                "mean": float(series.mean()),
                "std": float(series.std()),
                "min": float(series.min()),
                "max": float(series.max()),
                "trend_slope": slope,
                "adf_pvalue": adf_p,
            }
        )

    return pd.DataFrame(rows)
