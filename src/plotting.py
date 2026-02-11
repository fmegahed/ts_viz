"""
plotting.py
-----------
Chart-generation functions for time-series visualisation.

Every public function returns a :class:`matplotlib.figure.Figure` object.
Callers (e.g. Streamlit pages) can pass the figure to ``st.pyplot(fig)``
or convert it to PNG bytes via :func:`fig_to_png_bytes`.

All functions accept an optional *style_dict* (typically from
:func:`ui_theme.get_miami_mpl_style`) and an optional *palette_colors*
list so that colours stay consistent across the application.
"""

from __future__ import annotations

import io
import math
from typing import Dict, List, Optional, Sequence

# CRITICAL: set the non-interactive backend before any other mpl import.
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Brand defaults (mirrors ui_theme.py)
# ---------------------------------------------------------------------------
MIAMI_RED: str = "#C41230"
_DEFAULT_FIG_SIZE = (10, 6)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def fig_to_png_bytes(fig: matplotlib.figure.Figure, dpi: int = 150) -> bytes:
    """Render *fig* to an in-memory PNG and return the raw bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

class _StyleContext:
    """Context manager that temporarily applies *style_dict* to rcParams.

    On exit the previous values are restored so that other figures are not
    affected.
    """

    def __init__(self, style_dict: Optional[Dict[str, object]]):
        self._style = style_dict
        self._saved: Dict[str, object] = {}

    def __enter__(self) -> "_StyleContext":
        if self._style:
            for key, value in self._style.items():
                self._saved[key] = plt.rcParams.get(key)
                try:
                    plt.rcParams[key] = value
                except (KeyError, ValueError):
                    pass
        return self

    def __exit__(self, *exc_info: object) -> None:
        for key, value in self._saved.items():
            try:
                plt.rcParams[key] = value
            except (KeyError, ValueError):
                pass


def _default_color(palette_colors: Optional[List[str]], idx: int = 0) -> str:
    """Pick a colour from *palette_colors* or fall back to MIAMI_RED."""
    if palette_colors and len(palette_colors) > idx:
        return palette_colors[idx % len(palette_colors)]
    return MIAMI_RED


def _finish_figure(fig: matplotlib.figure.Figure) -> matplotlib.figure.Figure:
    """Apply common finishing touches and return the figure."""
    fig.tight_layout()
    return fig


def _auto_date_axis(ax: plt.Axes) -> None:
    """Auto-format and rotate date tick labels."""
    ax.xaxis.set_major_formatter(mdates.AutoDateFormatter(mdates.AutoDateLocator()))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")


def _grid_dims(n: int) -> tuple[int, int]:
    """Return (nrows, ncols) for a compact grid of *n* panels."""
    ncols = min(n, 3)
    nrows = math.ceil(n / ncols)
    return nrows, ncols


# ===================================================================
# 1. Line with markers
# ===================================================================

def plot_line_with_markers(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
    palette_colors: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """Simple line plot with small circle markers.

    Uses the first palette colour or *MIAMI_RED* as the default.
    """
    with _StyleContext(style_dict):
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE)
        color = _default_color(palette_colors, 0)
        ax.plot(
            df[date_col], df[y_col],
            marker="o", markersize=4, linewidth=1.5,
            color=color, label=y_col,
        )
        ax.set_xlabel(date_col)
        ax.set_ylabel(y_col)
        if title:
            ax.set_title(title)
        _auto_date_axis(ax)
        ax.legend(loc="best")
    return _finish_figure(fig)


# ===================================================================
# 2. Line with coloured markers
# ===================================================================

def plot_line_colored_markers(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    color_by: str,
    palette_colors: List[str],
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
) -> matplotlib.figure.Figure:
    """Line plot where marker colour varies by a categorical column.

    A legend is added mapping each unique value of *color_by* to its
    colour.
    """
    with _StyleContext(style_dict):
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE)

        # Draw the connecting line in a neutral grey
        ax.plot(
            df[date_col], df[y_col],
            linewidth=1.0, color="#AAAAAA", zorder=1,
        )

        # Map categories to colours
        categories = df[color_by].unique()
        n_cats = len(categories)
        if len(palette_colors) < n_cats:
            # cycle palette to cover all categories
            import itertools
            palette_colors = list(itertools.islice(
                itertools.cycle(palette_colors), n_cats
            ))

        color_map = {cat: palette_colors[i] for i, cat in enumerate(categories)}

        for cat in categories:
            mask = df[color_by] == cat
            ax.scatter(
                df.loc[mask, date_col], df.loc[mask, y_col],
                c=color_map[cat], label=str(cat),
                s=30, zorder=2, edgecolors="white", linewidths=0.3,
            )

        ax.set_xlabel(date_col)
        ax.set_ylabel(y_col)
        if title:
            ax.set_title(title)
        _auto_date_axis(ax)
        ax.legend(title=color_by, loc="best", fontsize=8, ncol=max(1, n_cats // 8))
    return _finish_figure(fig)


# ===================================================================
# 3. Seasonal plot
# ===================================================================

def plot_seasonal(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    period: str,
    palette_name_colors: List[str],
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
) -> matplotlib.figure.Figure:
    """Seasonal plot: one line per year/cycle, x-axis is within-period position.

    Parameters
    ----------
    period:
        ``"month"`` (x = month 1-12) or ``"quarter"`` (x = quarter 1-4).
    palette_name_colors:
        List of hex colours; one per cycle/year.
    """
    with _StyleContext(style_dict):
        tmp = df[[date_col, y_col]].copy()
        tmp["_year"] = tmp[date_col].dt.year

        if period.lower().startswith("q"):
            tmp["_period_pos"] = tmp[date_col].dt.quarter
            x_label = "Quarter"
        else:
            tmp["_period_pos"] = tmp[date_col].dt.month
            x_label = "Month"

        years = sorted(tmp["_year"].unique())
        n_years = len(years)
        if len(palette_name_colors) < n_years:
            import itertools
            palette_name_colors = list(itertools.islice(
                itertools.cycle(palette_name_colors), n_years
            ))

        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE)
        for i, year in enumerate(years):
            sub = tmp[tmp["_year"] == year].sort_values("_period_pos")
            ax.plot(
                sub["_period_pos"], sub[y_col],
                marker="o", markersize=4, linewidth=1.4,
                color=palette_name_colors[i], label=str(year),
            )

        ax.set_xlabel(x_label)
        ax.set_ylabel(y_col)
        if title:
            ax.set_title(title)
        ax.legend(title="Year", loc="best", fontsize=8, ncol=max(1, n_years // 6))
    return _finish_figure(fig)


# ===================================================================
# 4. Seasonal sub-series
# ===================================================================

def plot_seasonal_subseries(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    period: str,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
    palette_colors: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """Subseries plot with vertical panels for each season and horizontal mean lines.

    Parameters
    ----------
    period:
        ``"month"`` or ``"quarter"``.
    """
    with _StyleContext(style_dict):
        tmp = df[[date_col, y_col]].copy()

        if period.lower().startswith("q"):
            tmp["_season"] = tmp[date_col].dt.quarter
            labels = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
        else:
            tmp["_season"] = tmp[date_col].dt.month
            labels = {
                1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
                5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
                9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
            }

        seasons = sorted(tmp["_season"].unique())
        n = len(seasons)
        fig_w = max(10, n * 1.3)
        fig, axes = plt.subplots(1, n, figsize=(fig_w, 5), sharey=True)
        if n == 1:
            axes = [axes]

        color = _default_color(palette_colors, 0)

        for idx, season in enumerate(seasons):
            ax = axes[idx]
            sub = tmp[tmp["_season"] == season].sort_values(date_col)
            x_positions = range(len(sub))
            ax.plot(x_positions, sub[y_col].values, marker="o", markersize=3,
                    linewidth=1.2, color=color)

            mean_val = sub[y_col].mean()
            ax.axhline(mean_val, color=MIAMI_RED, linewidth=1.8, linestyle="--", alpha=0.8)

            ax.set_title(labels.get(season, str(season)), fontsize=10)
            ax.set_xticks([])
            ax.tick_params(axis="y", labelsize=8)
            if idx == 0:
                ax.set_ylabel(y_col)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    return _finish_figure(fig)


# ===================================================================
# 5. ACF / PACF
# ===================================================================

def plot_acf_pacf(
    acf_vals: np.ndarray,
    acf_ci: np.ndarray,
    pacf_vals: np.ndarray,
    pacf_ci: np.ndarray,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
) -> matplotlib.figure.Figure:
    """Side-by-side ACF and PACF bar plots with confidence-interval bands.

    Parameters
    ----------
    acf_vals, pacf_vals:
        1-D arrays of autocorrelation values (lag 0, 1, ...).
    acf_ci, pacf_ci:
        Arrays of shape ``(n_lags, 2)`` giving the lower and upper CI bounds.
    """
    with _StyleContext(style_dict):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        for ax, vals, ci, sub_title in [
            (ax1, acf_vals, acf_ci, "ACF"),
            (ax2, pacf_vals, pacf_ci, "PACF"),
        ]:
            lags = np.arange(len(vals))
            ax.bar(lags, vals, width=0.3, color=MIAMI_RED, alpha=0.85, zorder=2)

            # Confidence band
            lower = ci[:, 0]
            upper = ci[:, 1]
            ax.fill_between(lags, lower, upper, color="#C41230", alpha=0.12, zorder=1)
            ax.axhline(0, color="black", linewidth=0.8)

            ax.set_xlabel("Lag")
            ax.set_ylabel("Correlation")
            ax.set_title(sub_title)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    return _finish_figure(fig)


# ===================================================================
# 6. Decomposition
# ===================================================================

def plot_decomposition(
    decomposition_result,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
) -> matplotlib.figure.Figure:
    """4-panel plot: observed, trend, seasonal, residual.

    Parameters
    ----------
    decomposition_result:
        An object with ``.observed``, ``.trend``, ``.seasonal``, and
        ``.resid`` attributes (e.g. from ``statsmodels.tsa.seasonal_decompose``).
    """
    with _StyleContext(style_dict):
        components = [
            ("Observed", decomposition_result.observed),
            ("Trend", decomposition_result.trend),
            ("Seasonal", decomposition_result.seasonal),
            ("Residual", decomposition_result.resid),
        ]
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)

        for ax, (label, series) in zip(axes, components):
            ax.plot(series.index, series.values, linewidth=1.2, color=MIAMI_RED)
            ax.set_ylabel(label, fontsize=10)
            ax.tick_params(axis="both", labelsize=9)

        # Date formatting on the shared x-axis (bottom panel)
        _auto_date_axis(axes[-1])

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.01)
    return _finish_figure(fig)


# ===================================================================
# 7. Rolling overlay
# ===================================================================

def plot_rolling_overlay(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    window: int,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
    palette_colors: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """Original series (light) with rolling-mean overlay (bold) and +/-1 std band."""
    with _StyleContext(style_dict):
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE)

        raw_color = _default_color(palette_colors, 0)
        mean_color = _default_color(palette_colors, 1) if palette_colors and len(palette_colors) > 1 else "#333333"

        dates = df[date_col]
        vals = df[y_col]
        rolling_mean = vals.rolling(window=window, center=True).mean()
        rolling_std = vals.rolling(window=window, center=True).std()

        # Original series (light)
        ax.plot(dates, vals, linewidth=0.8, alpha=0.4, color=raw_color, label="Original")

        # Rolling mean (bold)
        ax.plot(dates, rolling_mean, linewidth=2.2, color=mean_color,
                label=f"{window}-pt Rolling Mean")

        # +/- 1 std band
        ax.fill_between(
            dates,
            rolling_mean - rolling_std,
            rolling_mean + rolling_std,
            alpha=0.15, color=mean_color, label="\u00b11 Std Dev",
        )

        ax.set_xlabel(date_col)
        ax.set_ylabel(y_col)
        if title:
            ax.set_title(title)
        _auto_date_axis(ax)
        ax.legend(loc="best")
    return _finish_figure(fig)


# ===================================================================
# 8. Year-over-Year change
# ===================================================================

def plot_yoy_change(
    df: pd.DataFrame,
    date_col: str,
    y_col: str,
    yoy_df: pd.DataFrame,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
) -> matplotlib.figure.Figure:
    """Two-subplot bar chart: absolute YoY change (top) and percentage YoY change (bottom).

    Parameters
    ----------
    yoy_df:
        DataFrame with columns ``"date"``, ``"abs_change"``, ``"pct_change"``.
    """
    with _StyleContext(style_dict):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        dates = yoy_df["date"]
        abs_change = yoy_df["abs_change"]
        pct_change = yoy_df["pct_change"]

        # Colours: green for positive, red for negative
        abs_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in abs_change]
        pct_colors = ["#2ca02c" if v >= 0 else "#d62728" for v in pct_change]

        ax1.bar(dates, abs_change, color=abs_colors, width=20, edgecolor="white", linewidth=0.3)
        ax1.axhline(0, color="black", linewidth=0.6)
        ax1.set_ylabel("Absolute Change")
        ax1.set_title("Year-over-Year Absolute Change")

        ax2.bar(dates, pct_change, color=pct_colors, width=20, edgecolor="white", linewidth=0.3)
        ax2.axhline(0, color="black", linewidth=0.6)
        ax2.set_ylabel("% Change")
        ax2.set_title("Year-over-Year Percentage Change")

        _auto_date_axis(ax2)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    return _finish_figure(fig)


# ===================================================================
# 9. Lag plot
# ===================================================================

def plot_lag(
    series: pd.Series,
    lag: int = 1,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
) -> matplotlib.figure.Figure:
    """Scatter plot of y(t) vs y(t-lag) with correlation-coefficient annotation."""
    with _StyleContext(style_dict):
        y = series.dropna().values
        y_t = y[lag:]
        y_lag = y[:-lag]

        corr = np.corrcoef(y_t, y_lag)[0, 1]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(y_lag, y_t, alpha=0.5, s=20, color=MIAMI_RED, edgecolors="white", linewidths=0.3)

        # Annotation
        ax.annotate(
            f"r = {corr:.3f}",
            xy=(0.05, 0.95), xycoords="axes fraction",
            fontsize=12, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC", alpha=0.9),
            verticalalignment="top",
        )

        ax.set_xlabel(f"y(t\u2212{lag})")
        ax.set_ylabel("y(t)")
        if title:
            ax.set_title(title)
        else:
            ax.set_title(f"Lag-{lag} Plot")
    return _finish_figure(fig)


# ===================================================================
# 10. Panel (small multiples)
# ===================================================================

def plot_panel(
    df: pd.DataFrame,
    date_col: str,
    y_cols: List[str],
    chart_type: str = "line",
    shared_y: bool = True,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
    palette_colors: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """Small multiples: one subplot per *y_col* arranged in a grid.

    Parameters
    ----------
    chart_type:
        ``"line"`` or ``"bar"``.
    shared_y:
        If ``True`` all panels share the same y-axis limits.
    """
    with _StyleContext(style_dict):
        n = len(y_cols)
        nrows, ncols = _grid_dims(n)
        fig_h = max(4, nrows * 3.5)
        fig_w = max(8, ncols * 4.5)
        fig, axes = plt.subplots(
            nrows, ncols, figsize=(fig_w, fig_h),
            sharey=shared_y, squeeze=False,
        )
        flat_axes = axes.flatten()

        for i, col in enumerate(y_cols):
            ax = flat_axes[i]
            color = _default_color(palette_colors, i)

            if chart_type == "bar":
                ax.bar(df[date_col], df[col], color=color, width=2, edgecolor="white", linewidth=0.3)
            else:
                ax.plot(df[date_col], df[col], linewidth=1.3, color=color)

            ax.set_title(col, fontsize=10)
            _auto_date_axis(ax)

        # Hide unused subplots
        for j in range(n, len(flat_axes)):
            flat_axes[j].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    return _finish_figure(fig)


# ===================================================================
# 11. Spaghetti plot
# ===================================================================

def plot_spaghetti(
    df: pd.DataFrame,
    date_col: str,
    y_cols: List[str],
    alpha: float = 0.15,
    highlight_col: Optional[str] = None,
    top_n: Optional[int] = None,
    show_median_band: bool = False,
    title: Optional[str] = None,
    style_dict: Optional[Dict[str, object]] = None,
    palette_colors: Optional[List[str]] = None,
) -> matplotlib.figure.Figure:
    """All series on one plot at low opacity, with optional highlighting.

    Parameters
    ----------
    alpha:
        Opacity for the background spaghetti lines.
    highlight_col:
        Column name to draw with full opacity and thicker line.
    top_n:
        If set, highlight the *top_n* series by maximum value.
    show_median_band:
        If ``True``, overlay the median line and shade the IQR.
    """
    with _StyleContext(style_dict):
        fig, ax = plt.subplots(figsize=_DEFAULT_FIG_SIZE)

        dates = df[date_col]

        # Determine which columns to highlight
        highlight_set: set[str] = set()
        if highlight_col and highlight_col in y_cols:
            highlight_set.add(highlight_col)
        if top_n:
            max_vals = {col: df[col].max() for col in y_cols}
            sorted_cols = sorted(max_vals, key=max_vals.get, reverse=True)  # type: ignore[arg-type]
            highlight_set.update(sorted_cols[:top_n])

        # Draw all series
        for i, col in enumerate(y_cols):
            color = _default_color(palette_colors, i)
            if col in highlight_set:
                ax.plot(dates, df[col], linewidth=2.0, alpha=0.9,
                        color=color, label=col, zorder=3)
            else:
                ax.plot(dates, df[col], linewidth=0.8, alpha=alpha,
                        color=color, zorder=1)

        # Median + IQR band
        if show_median_band:
            numeric_data = df[y_cols]
            median_line = numeric_data.median(axis=1)
            q1 = numeric_data.quantile(0.25, axis=1)
            q3 = numeric_data.quantile(0.75, axis=1)

            ax.plot(dates, median_line, linewidth=2.2, color="#333333",
                    label="Median", zorder=4)
            ax.fill_between(dates, q1, q3, alpha=0.2, color="#333333",
                            label="IQR", zorder=2)

        ax.set_xlabel(date_col)
        ax.set_ylabel("Value")
        if title:
            ax.set_title(title)
        _auto_date_axis(ax)

        # Only add legend if there are labelled items
        handles, labels = ax.get_legend_handles_labels()
        if labels:
            ax.legend(loc="best", fontsize=8)
    return _finish_figure(fig)
