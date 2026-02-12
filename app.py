"""
Time Series Visualizer + AI Chart Interpreter
=============================================
Main Gradio application.  Run with:

    python app.py
"""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import gradio as gr

from src.ui_theme import (
    MiamiTheme,
    get_miami_css,
    get_miami_mpl_style,
    get_palette_colors,
    render_palette_preview,
)
from src.cleaning import (
    detect_delimiter,
    suggest_date_columns,
    suggest_numeric_columns,
    clean_dataframe,
    detect_frequency,
    add_time_features,
    detect_long_format,
    pivot_long_to_wide,
    CleaningReport,
    FrequencyInfo,
)
from src.diagnostics import (
    compute_summary_stats,
    compute_acf_pacf,
    compute_decomposition,
    compute_yoy_change,
    compute_multi_series_summary,
)
from src.plotting import (
    fig_to_png_bytes,
    plot_line_with_markers,
    plot_line_colored_markers,
    plot_seasonal,
    plot_seasonal_subseries,
    plot_acf_pacf,
    plot_decomposition,
    plot_rolling_overlay,
    plot_yoy_change,
    plot_lag,
    plot_panel,
    plot_spaghetti,
)
from src.ai_interpretation import (
    check_api_key_available,
    interpret_chart,
    render_interpretation_markdown,
)
from src.querychat_helpers import (
    check_querychat_available,
    create_querychat,
    get_filtered_pandas_df,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DATA_DIR = Path(__file__).parent / "data"
_DEMO_FILES = {
    "Ohio Unemployment Rate (single, monthly)": _DATA_DIR / "demo_ohio_unemployment.csv",
    "Manufacturing Employment by State (wide, monthly)": _DATA_DIR / "demo_manufacturing_wide.csv",
    "Manufacturing Employment by State (long, monthly)": _DATA_DIR / "demo_manufacturing_long.csv",
}
_DEMO_CHOICES = ["(none)"] + list(_DEMO_FILES.keys())

_CHART_TYPES = [
    "Line with Markers",
    "Line \u2013 Colored Markers",
    "Seasonal Plot",
    "Seasonal Sub-series",
    "ACF / PACF",
    "Decomposition",
    "Rolling Mean Overlay",
    "Year-over-Year Change",
    "Lag Plot",
]

_PALETTE_NAMES = ["Set2", "Dark2", "Set1", "Paired", "Pastel1", "Pastel2", "Accent"]
_STYLE_DICT = get_miami_mpl_style()
_MODE_SINGLE = "Single Series"
_MODE_PANEL = "Compare Few (Panel)"
_MODE_SPAG = "Compare Many (Spaghetti)"
_DATE_HINT_TOKENS = ("date", "time", "year", "month", "day", "period")

# ---------------------------------------------------------------------------
# State helpers
# ---------------------------------------------------------------------------

def _make_empty_state() -> dict:
    return {
        "raw_df_original": None,
        "cleaned_df": None,
        "cleaning_report": None,
        "freq_info": None,
        "date_col": None,
        "y_cols": None,
        "setup_applied": False,
        "single_png": None,
        "panel_png": None,
        "spag_png": None,
        "qc": None,
        "mode_choices": [_MODE_SINGLE],
        "recommended_mode": _MODE_SINGLE,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _format_cleaning_report_md(report: CleaningReport) -> str:
    lines = [
        "| Metric | Value |", "|:--|:--|",
        f"| **Rows before** | {report.rows_before:,} |",
        f"| **Rows after** | {report.rows_after:,} |",
        f"| **Duplicates found** | {report.duplicates_found:,} |",
    ]
    if report.missing_before:
        lines += ["", "**Missing values:**", "| Column | Before | After |", "|:--|:--|:--|"]
        for col in report.missing_before:
            lines.append(f"| {col} | {report.missing_before[col]} | {report.missing_after.get(col, 0)} |")
    if report.parsing_warnings:
        lines += ["", "**Warnings:**"]
        for w in report.parsing_warnings:
            lines.append(f"- {w}")
    return "\n".join(lines)


def _fmt(val, fmt_str):
    return fmt_str.format(val) if pd.notna(val) else "N/A"


def _format_summary_stats_md(stats) -> str:
    lines = [
        "| Statistic | Value |", "|:--|:--|",
        f"| **Count** | {stats.count:,} |",
        f"| **Missing** | {stats.missing_count} ({stats.missing_pct:.1f}%) |",
        f"| **Mean** | {stats.mean_val:,.2f} |",
        f"| **Std Dev** | {stats.std_val:,.2f} |",
        f"| **Min** | {stats.min_val:,.2f} |",
        f"| **25th %ile** | {stats.p25:,.2f} |",
        f"| **Median** | {stats.median_val:,.2f} |",
        f"| **75th %ile** | {stats.p75:,.2f} |",
        f"| **Max** | {stats.max_val:,.2f} |",
        f"| **Trend slope** | {_fmt(stats.trend_slope, '{:,.4f}')} |",
        f"| **Trend p-value** | {_fmt(stats.trend_pvalue, '{:.4f}')} |",
        f"| **ADF statistic** | {_fmt(stats.adf_statistic, '{:.4f}')} |",
        f"| **ADF p-value** | {_fmt(stats.adf_pvalue, '{:.4f}')} |",
        "",
        f"*Date range: {stats.date_start.date()} to {stats.date_end.date()} ({stats.date_span_days:,} days)*",
    ]
    return "\n".join(lines)


def _format_multi_summary_md(summary_df: pd.DataFrame) -> str:
    lines = [
        "| Variable | Count | Mean | Std | Min | Max | Trend Slope | ADF p |",
        "|:--|--:|--:|--:|--:|--:|--:|--:|",
    ]
    for _, row in summary_df.iterrows():
        adf = f"{row['adf_pvalue']:.4f}" if pd.notna(row['adf_pvalue']) else "N/A"
        slope = f"{row['trend_slope']:,.4f}" if pd.notna(row['trend_slope']) else "N/A"
        lines.append(
            f"| {row['variable']} | {row['count']:,} | {row['mean']:,.2f} | "
            f"{row['std']:,.2f} | {row['min']:,.2f} | {row['max']:,.2f} | "
            f"{slope} | {adf} |"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# UX helpers
# ---------------------------------------------------------------------------

def _preview_df(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    return df.head(n).copy()


def _format_sidebar_status_md(df: pd.DataFrame | None, date_col: str | None = None,
                              data_format: str | None = None, y_count: int | None = None,
                              freq_label: str | None = None, cleaned_rows: int | None = None) -> str:
    if df is None:
        return "*No data loaded yet.*"

    row_count = cleaned_rows if cleaned_rows is not None else len(df)
    col_count = len(df.columns)
    parts = [
        "### Dataset Status",
        f"- Rows: **{row_count:,}**",
        f"- Columns: **{col_count}**",
    ]
    if date_col:
        parts.append(f"- Date column: **{date_col}**")
    if data_format:
        parts.append(f"- Structure: **{data_format}**")
    if y_count is not None:
        parts.append(f"- Value series selected: **{y_count}**")
    if freq_label:
        parts.append(f"- Frequency: **{freq_label}**")
    return "\n".join(parts)


def _format_raw_profile_md(df: pd.DataFrame, date_col: str, data_format: str,
                           y_cols: list[str]) -> str:
    numeric_cols = int(df.select_dtypes(include=[np.number]).shape[1])
    object_cols = int(df.select_dtypes(include=["object"]).shape[1])
    return "\n".join([
        "### Dataset Profile",
        "| Metric | Value |",
        "|:--|:--|",
        f"| Rows | {len(df):,} |",
        f"| Columns | {len(df.columns)} |",
        f"| Suggested date column | {date_col} |",
        f"| Detected structure | {data_format} |",
        f"| Numeric columns | {numeric_cols} |",
        f"| Text columns | {object_cols} |",
        f"| Value series selected | {len(y_cols)} |",
    ])


def _get_mode_config(y_count: int) -> tuple[list[str], str, str]:
    if y_count <= 1:
        return (
            [_MODE_SINGLE],
            _MODE_SINGLE,
            "Single series detected. Multi-series comparison modes are hidden.",
        )

    if y_count <= 8:
        return (
            [_MODE_SINGLE, _MODE_PANEL],
            _MODE_PANEL,
            "Best fit: compare a few series in panel view. Spaghetti is hidden to reduce clutter.",
        )

    return (
        [_MODE_SINGLE, _MODE_PANEL, _MODE_SPAG],
        _MODE_SPAG,
        "Many series detected. Spaghetti is the recommended default.",
    )


def _chart_availability(df_plot: pd.DataFrame, date_col: str, y_col: str,
                        freq_info: FrequencyInfo | None) -> dict[str, str]:
    blocked: dict[str, str] = {}

    if y_col not in df_plot.columns:
        return {name: "Value column not found in active data." for name in _CHART_TYPES}

    n_obs = int(df_plot[y_col].dropna().shape[0])
    if n_obs <= 1:
        return {name: "Need at least 2 non-missing observations." for name in _CHART_TYPES}

    date_series = pd.to_datetime(df_plot[date_col], errors="coerce").dropna()
    span_days = int((date_series.max() - date_series.min()).days) if len(date_series) >= 2 else 0
    has_time_features = "month" in df_plot.columns
    freq_label = freq_info.label if freq_info else "Unknown"
    period_map = {"Monthly": 12, "Quarterly": 4, "Weekly": 52, "Daily": 365}
    period = period_map.get(freq_label)

    if not has_time_features:
        blocked["Line – Colored Markers"] = "Calendar features unavailable."
        blocked["Seasonal Plot"] = "Calendar features unavailable."
        blocked["Seasonal Sub-series"] = "Calendar features unavailable."

    if has_time_features and n_obs < 12:
        blocked["Seasonal Plot"] = "Need at least 12 observations."
        blocked["Seasonal Sub-series"] = "Need at least 12 observations."

    if n_obs < 8:
        blocked["ACF / PACF"] = "Need at least 8 observations."

    if period is None:
        blocked["Decomposition"] = "Requires Daily/Weekly/Monthly/Quarterly frequency."
    elif n_obs < max(8, period * 2):
        blocked["Decomposition"] = f"Need at least {max(8, period * 2)} observations."

    if span_days < 365:
        blocked["Year-over-Year Change"] = "Need at least one year of coverage."

    if n_obs < 3:
        blocked["Lag Plot"] = "Need at least 3 observations."

    return blocked


def _available_chart_choices(df_plot: pd.DataFrame, date_col: str, y_col: str,
                             freq_info: FrequencyInfo | None) -> tuple[list[str], str]:
    blocked = _chart_availability(df_plot, date_col, y_col, freq_info)
    available = [name for name in _CHART_TYPES if name not in blocked]
    if not available:
        available = ["Line with Markers"]
    notes = ["**Chart availability (auto-gated):**"]
    if blocked:
        for chart_name, reason in blocked.items():
            notes.append(f"- {chart_name}: {reason}")
    else:
        notes.append("- All chart types are available.")
    return available, "\n".join(notes)


def _mode_visibility(mode: str) -> tuple[bool, bool, bool]:
    return (
        mode == _MODE_SINGLE,
        mode == _MODE_PANEL,
        mode == _MODE_SPAG,
    )


def _choose_default_date_col(df: pd.DataFrame) -> str | None:
    cols = list(df.columns)
    if not cols:
        return None

    for col in cols:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    for col in cols:
        name = str(col).lower()
        if any(tok in name for tok in _DATE_HINT_TOKENS):
            return col

    suggestions = suggest_date_columns(df)
    if suggestions:
        return suggestions[0]
    return cols[0]


def _derive_setup_options(df: pd.DataFrame, date_col: str | None, data_format: str,
                          group_col: str | None = None, value_col: str | None = None,
                          current_y: list[str] | None = None) -> dict:
    all_cols = list(df.columns)
    resolved_date = date_col if date_col in all_cols else (all_cols[0] if all_cols else None)

    if not resolved_date:
        return {
            "resolved_date": None,
            "string_cols": [],
            "value_options": [],
            "group_default": None,
            "value_default": None,
            "available_y": [],
            "default_y": [],
        }

    other_cols = [c for c in all_cols if c != resolved_date]
    string_cols = [
        c for c in other_cols
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
    ]
    numeric_suggest = suggest_numeric_columns(df)

    group_default = (
        group_col if group_col and group_col in string_cols
        else (string_cols[0] if string_cols else None)
    )
    value_options = [c for c in numeric_suggest if c != resolved_date and c != group_default]
    value_default = (
        value_col if value_col and value_col in value_options
        else (value_options[0] if value_options else None)
    )

    if data_format == "Long" and group_default and value_default:
        try:
            effective = pivot_long_to_wide(df, resolved_date, group_default, value_default)
            available_y = [c for c in effective.columns if c != resolved_date]
        except Exception:
            available_y = value_options.copy()
    else:
        available_y = value_options.copy()

    kept = [c for c in (current_y or []) if c in available_y]
    default_y = kept if kept else available_y[:4]

    return {
        "resolved_date": resolved_date,
        "string_cols": string_cols,
        "value_options": value_options,
        "group_default": group_default,
        "value_default": value_default,
        "available_y": available_y,
        "default_y": default_y,
    }


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _read_file_to_df(file_path: str) -> tuple[pd.DataFrame, str]:
    with open(file_path, "rb") as f:
        raw = f.read()
    delim = detect_delimiter(raw)
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=delim)
    return df, delim


def _apply_date_filter(df, date_col, mode, n_years, custom_start, custom_end):
    if mode == "Last N years" and n_years:
        cutoff = df[date_col].max() - pd.DateOffset(years=int(n_years))
        df = df[df[date_col] >= cutoff]
    elif mode == "Custom":
        try:
            if custom_start and str(custom_start).strip():
                df = df[df[date_col] >= pd.to_datetime(custom_start)]
        except (ValueError, TypeError):
            pass
        try:
            if custom_end and str(custom_end).strip():
                df = df[df[date_col] <= pd.to_datetime(custom_end)]
        except (ValueError, TypeError):
            pass
    return df


def _generate_single_chart(df_plot, date_col, active_y, chart_type, palette_colors,
                           color_by, period_label, window_size, lag_val, decomp_model,
                           freq_info):
    """Generate a single chart figure.  Returns ``(fig, error_msg)``."""
    try:
        if chart_type == "Line with Markers":
            return plot_line_with_markers(
                df_plot, date_col, active_y,
                title=f"{active_y} over Time",
                style_dict=_STYLE_DICT, palette_colors=palette_colors,
            ), None

        elif "Colored Markers" in chart_type and color_by:
            return plot_line_colored_markers(
                df_plot, date_col, active_y,
                color_by=color_by, palette_colors=palette_colors,
                title=f"{active_y} colored by {color_by}",
                style_dict=_STYLE_DICT,
            ), None

        elif chart_type == "Seasonal Plot":
            return plot_seasonal(
                df_plot, date_col, active_y,
                period=period_label or "month",
                palette_name_colors=palette_colors,
                title=f"Seasonal Plot - {active_y}",
                style_dict=_STYLE_DICT,
            ), None

        elif chart_type == "Seasonal Sub-series":
            return plot_seasonal_subseries(
                df_plot, date_col, active_y,
                period=period_label or "month",
                title=f"Seasonal Sub-series - {active_y}",
                style_dict=_STYLE_DICT, palette_colors=palette_colors,
            ), None

        elif chart_type == "ACF / PACF":
            series = df_plot[active_y].dropna()
            acf_vals, acf_ci, pacf_vals, pacf_ci = compute_acf_pacf(series)
            return plot_acf_pacf(
                acf_vals, acf_ci, pacf_vals, pacf_ci,
                title=f"ACF / PACF - {active_y}",
                style_dict=_STYLE_DICT,
            ), None

        elif chart_type == "Decomposition":
            period_int = None
            if freq_info:
                period_int = {"Monthly": 12, "Quarterly": 4, "Weekly": 52, "Daily": 365}.get(freq_info.label)
            result = compute_decomposition(
                df_plot, date_col, active_y,
                model=decomp_model or "additive", period=period_int,
            )
            return plot_decomposition(
                result,
                title=f"Decomposition - {active_y} ({decomp_model})",
                style_dict=_STYLE_DICT,
            ), None

        elif chart_type == "Rolling Mean Overlay":
            w = int(window_size) if window_size else 12
            return plot_rolling_overlay(
                df_plot, date_col, active_y,
                window=w,
                title=f"Rolling {w}-pt Mean - {active_y}",
                style_dict=_STYLE_DICT, palette_colors=palette_colors,
            ), None

        elif chart_type == "Year-over-Year Change":
            yoy_result = compute_yoy_change(df_plot, date_col, active_y)
            yoy_df = pd.DataFrame({
                "date": yoy_result[date_col],
                "abs_change": yoy_result["yoy_abs_change"],
                "pct_change": yoy_result["yoy_pct_change"],
            }).dropna()
            return plot_yoy_change(
                df_plot, date_col, active_y, yoy_df,
                title=f"Year-over-Year Change - {active_y}",
                style_dict=_STYLE_DICT,
            ), None

        elif chart_type == "Lag Plot":
            lag = int(lag_val) if lag_val else 1
            return plot_lag(
                df_plot[active_y],
                lag=lag,
                title=f"Lag-{lag} Plot - {active_y}",
                style_dict=_STYLE_DICT,
            ), None

    except Exception as exc:
        return None, str(exc)

    return None, "Unknown chart type"


# ---------------------------------------------------------------------------
# HTML fragments
# ---------------------------------------------------------------------------

_DEVELOPER_CARD = """
<div class="dev-card">
    <div class="dev-row">
        <svg class="dev-avatar" viewBox="0 0 16 16" aria-hidden="true">
            <path d="M11 6a3 3 0 1 1-6 0 3 3 0 0 1 6 0"/>
            <path fill-rule="evenodd" d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m8-7a7 7 0 0 0-5.468 11.37c.69-1.198 1.97-2.015 3.526-2.015h3.884c1.556 0 2.835.817 3.526 2.014A7 7 0 0 0 8 1"/>
        </svg>
        <div>
            <div class="dev-name">Fadel M. Megahed</div>
            <div class="dev-role">
                Raymond E. Glos Professor, Farmer School of Business<br>
                Miami University
            </div>
        </div>
    </div>
    <div class="dev-links">
        <a class="dev-link" href="mailto:fmegahed@miamioh.edu">
            <svg viewBox="0 0 16 16"><path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4zm2-1a1 1 0 0 0-1 1v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2zm13 2.383-4.708 2.825L15 11.105zM14.247 12.6 9.114 8.98 8 9.67 6.886 8.98 1.753 12.6A1 1 0 0 0 2 13h12a1 1 0 0 0 .247-.4zM1 11.105l4.708-2.897L1 5.383z"/></svg>
            Email</a>
        <a class="dev-link" href="https://www.linkedin.com/in/fadel-megahed-289046b4/" target="_blank">
            <svg viewBox="0 0 16 16"><path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.475 0 16 .513 16 1.146v13.708c0 .633-.525 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zM4.943 13.5V6H2.542v7.5zM3.743 4.927c.837 0 1.358-.554 1.358-1.248-.015-.709-.521-1.248-1.342-1.248-.821 0-1.358.54-1.358 1.248 0 .694.521 1.248 1.327 1.248zm4.908 8.573V9.359c0-.22.016-.44.08-.598.176-.44.576-.897 1.248-.897.88 0 1.232.676 1.232 1.667v4.0h2.401V9.247c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193h.016V6H6.35c.03.7 0 7.5 0 7.5z"/></svg>
            LinkedIn</a>
        <a class="dev-link" href="https://miamioh.edu/fsb/directory/?up=/directory/megahefm" target="_blank">
            <svg viewBox="0 0 16 16"><path d="M8 0a8 8 0 1 0 0 16A8 8 0 0 0 8 0M1.018 7.5h2.49a14 14 0 0 1 .535-3.55A6 6 0 0 0 1.018 7.5m0 1h2.49c.05 1.24.217 2.44.535 3.55a6 6 0 0 1-3.025-3.55m11.964 0a6 6 0 0 1-3.025 3.55c.318-1.11.485-2.31.535-3.55zm0-1a6 6 0 0 0-3.025-3.55c.318 1.11.485 2.31.535 3.55zM8 1.016q.347.372.643.812C9.157 2.6 9.545 3.71 9.757 5H6.243c.212-1.29.6-2.4 1.114-3.172Q7.653 1.388 8 1.016M8 15q-.347-.372-.643-.812C6.843 13.4 6.455 12.29 6.243 11h3.514c-.212 1.29-.6 2.4-1.114 3.172A6 6 0 0 1 8 14.984M5.494 7.5a13 13 0 0 0 0 1h5.012a13 13 0 0 0 0-1z"/></svg>
            Website</a>
        <a class="dev-link" href="https://github.com/fmegahed/" target="_blank">
            <svg viewBox="0 0 16 16"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/></svg>
            GitHub</a>
    </div>
</div>
"""

_WELCOME_MD = """
# Time Series Visualizer
*ISA 444 \u00b7 Miami University \u00b7 Farmer School of Business*

---

### Guided Workflow

<div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(150px, 1fr)); gap:0.75rem; margin:1rem 0;">
<div class="step-card">
  <div class="step-number">1</div>
  <div class="step-title">Load Data</div>
  <div class="step-desc">Upload your CSV or load a demo dataset from the sidebar.</div>
</div>
<div class="step-card">
  <div class="step-number">2</div>
  <div class="step-title">Understand</div>
  <div class="step-desc">Review auto-detected structure and inspect a raw-data preview.</div>
</div>
<div class="step-card">
  <div class="step-number">3</div>
  <div class="step-title">Prepare</div>
  <div class="step-desc">Set date/value columns and apply cleaning options in the main canvas.</div>
</div>
<div class="step-card">
  <div class="step-number">4</div>
  <div class="step-title">Visualize</div>
  <div class="step-desc">Only relevant visualization modes are shown based on series count.</div>
</div>
<div class="step-card">
  <div class="step-number">5</div>
  <div class="step-title">Interpret</div>
  <div class="step-desc">Generate AI interpretation when your chart is ready.</div>
</div>
</div>

---

### Features

| | |
|:--|:--|
| **Smart Import** | Auto-detect delimiters, dates, and numeric formats |
| **9+ Chart Types** | Line, seasonal, ACF/PACF, decomposition, lag, and more |
| **Multi-Series** | Panel (small multiples) and spaghetti plots |
| **AI Insights** | GPT vision analyzes your charts and returns structured interpretation |
| **QueryChat** | Natural-language data filtering powered by DuckDB |

### Good to Know

**Privacy** \u2014 All data processing happens in-memory.
No data is stored on disk. Only chart images (never raw data) are sent to
OpenAI when you click *Interpret Chart with AI*.

**Demo Datasets** \u2014 Three built-in FRED datasets are available in the
sidebar: Ohio Unemployment Rate (single series), Manufacturing Employment
for five states in wide format, and the same data in long/stacked format.
"""


# ---------------------------------------------------------------------------
# Event handlers
# ---------------------------------------------------------------------------

def _process_new_data(df: pd.DataFrame, delim: str | None = None):
    """Shared logic for file upload and demo select.

    Returns a tuple of values matching ``_DATA_LOAD_OUTPUTS``.
    """
    state = _make_empty_state()
    state["raw_df_original"] = df

    all_cols = list(df.columns)
    default_date = _choose_default_date_col(df)
    infer_date = default_date if default_date in all_cols else (all_cols[0] if all_cols else None)
    if infer_date is None:
        return (
            state, gr.Column(visible=False), gr.Dropdown(), gr.Radio(), gr.Column(visible=False),
            gr.Dropdown(), gr.Dropdown(), gr.CheckboxGroup(choices=[], value=[]), "",
            _format_sidebar_status_md(None), "", pd.DataFrame(), gr.Dropdown(), "",
            gr.Column(visible=True), gr.Column(visible=False),
            gr.Radio(choices=[_MODE_SINGLE], value=_MODE_SINGLE), "",
            gr.Dropdown(choices=_CHART_TYPES, value=_CHART_TYPES[0]), "",
            gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False),
            pd.DataFrame(),
        )

    is_long, _, _ = detect_long_format(df, infer_date)
    fmt = "Long" if is_long else "Wide"
    setup_opts = _derive_setup_options(
        df,
        date_col=infer_date,
        data_format=fmt,
        group_col=None,
        value_col=None,
        current_y=None,
    )
    resolved_date = setup_opts["resolved_date"]
    group_default = setup_opts["group_default"]
    value_default = setup_opts["value_default"]
    available_y = setup_opts["available_y"]
    default_y = setup_opts["default_y"]
    delim_text = f"Detected delimiter: `{repr(delim)}`" if delim else ""
    profile_md = _format_raw_profile_md(df, resolved_date, fmt, default_y)
    status_md = _format_sidebar_status_md(
        df=df,
        date_col=resolved_date,
        data_format=fmt,
        y_count=len(default_y),
    )

    return (
        state,                                                           # app_state
        gr.Column(visible=True),                                         # setup_col
        gr.Dropdown(choices=all_cols, value=resolved_date),              # date_col_dd
        gr.Radio(value=fmt),                                             # format_radio
        gr.Column(visible=is_long),                                      # long_col
        gr.Dropdown(choices=setup_opts["string_cols"], value=group_default),     # group_col_dd
        gr.Dropdown(choices=setup_opts["value_options"], value=value_default),    # value_col_dd
        gr.CheckboxGroup(choices=available_y, value=default_y),          # y_cols_cbg
        delim_text,                                                      # delim_md
        status_md,                                                       # status_md
        profile_md,                                                      # raw_profile_md
        _preview_df(df),                                                 # raw_preview_df
        gr.Dropdown(choices=all_cols, value=resolved_date),              # cast_col_dd
        "",                                                              # cast_status_md
        gr.Column(visible=False),                                        # welcome_col
        gr.Column(visible=False),                                        # analysis_col
        gr.Radio(choices=[_MODE_SINGLE], value=_MODE_SINGLE),            # viz_mode_radio
        "Apply setup to unlock visualization modes.",                    # mode_hint_md
        gr.Dropdown(choices=_CHART_TYPES, value=_CHART_TYPES[0]),        # single_chart_dd
        "*Apply setup to tailor chart options to your data.*",           # single_gate_md
        gr.Column(visible=False),                                        # single_mode_col
        gr.Column(visible=False),                                        # panel_mode_col
        gr.Column(visible=False),                                        # spag_mode_col
        pd.DataFrame(),                                                  # cleaned_preview_df
    )


def on_file_upload(file_obj, state):
    if file_obj is None:
        empty = _make_empty_state()
        return (
            empty,
            gr.Column(visible=False), gr.Dropdown(), gr.Radio(),
            gr.Column(visible=False), gr.Dropdown(), gr.Dropdown(),
            gr.CheckboxGroup(choices=[], value=[]), "",
            "*No data loaded yet.*",
            "",
            pd.DataFrame(),
            gr.Dropdown(),
            "",
            gr.Column(visible=True), gr.Column(visible=False),
            gr.Radio(choices=[_MODE_SINGLE], value=_MODE_SINGLE),
            "",
            gr.Dropdown(choices=_CHART_TYPES, value=_CHART_TYPES[0]),
            "",
            gr.Column(visible=False), gr.Column(visible=False), gr.Column(visible=False),
            pd.DataFrame(),
        )
    path = file_obj if isinstance(file_obj, str) else str(file_obj)
    df, delim = _read_file_to_df(path)
    return _process_new_data(df, delim)


def on_demo_select(choice, state):
    if choice == "(none)" or choice is None:
        return (
            state,
            gr.Column(), gr.Dropdown(), gr.Radio(),
            gr.Column(), gr.Dropdown(), gr.Dropdown(),
            gr.CheckboxGroup(), "",
            gr.Markdown(), gr.Markdown(), gr.Dataframe(), gr.Dropdown(), gr.Markdown(),
            gr.Column(), gr.Column(),
            gr.Radio(), gr.Markdown(), gr.Dropdown(), gr.Markdown(),
            gr.Column(), gr.Column(), gr.Column(),
            gr.Dataframe(),
        )
    demo_path = _DEMO_FILES[choice]
    df = pd.read_csv(demo_path)
    return _process_new_data(df, None)


def on_format_change(fmt):
    return gr.Column(visible=(fmt == "Long"))


def on_setup_inputs_change(date_col, data_format, group_col, value_col, current_y, state):
    raw_df = state.get("raw_df_original")
    if raw_df is None:
        return (
            gr.Column(visible=(data_format == "Long")),
            gr.Dropdown(), gr.Dropdown(), gr.CheckboxGroup(),
            "", _format_sidebar_status_md(None),
        )

    opts = _derive_setup_options(
        raw_df,
        date_col=date_col,
        data_format=data_format,
        group_col=group_col,
        value_col=value_col,
        current_y=list(current_y) if current_y else [],
    )
    resolved_date = opts["resolved_date"]
    profile_md = _format_raw_profile_md(raw_df, resolved_date, data_format, opts["default_y"])
    status_md = _format_sidebar_status_md(
        raw_df,
        date_col=resolved_date,
        data_format=data_format,
        y_count=len(opts["default_y"]),
    )
    return (
        gr.Column(visible=(data_format == "Long")),
        gr.Dropdown(choices=opts["string_cols"], value=opts["group_default"]),
        gr.Dropdown(choices=opts["value_options"], value=opts["value_default"]),
        gr.CheckboxGroup(choices=opts["available_y"], value=opts["default_y"]),
        profile_md,
        status_md,
    )


def on_long_cols_change(date_col, group_col, value_col, state):
    raw_df = state.get("raw_df_original")
    if raw_df is None or not group_col or not value_col:
        return gr.CheckboxGroup()
    try:
        effective = pivot_long_to_wide(raw_df, date_col, group_col, value_col)
        available = [c for c in effective.columns if c != date_col]
        return gr.CheckboxGroup(choices=available, value=available[:4])
    except Exception:
        return gr.CheckboxGroup(choices=[], value=[])


def on_y_selection_change(date_col, data_format, y_cols, state):
    raw_df = state.get("raw_df_original")
    if raw_df is None:
        return "", _format_sidebar_status_md(None)

    all_cols = list(raw_df.columns)
    resolved_date = date_col if date_col in all_cols else (all_cols[0] if all_cols else "")
    y_list = list(y_cols) if y_cols else []
    profile_md = _format_raw_profile_md(raw_df, resolved_date, data_format, y_list)
    status_md = _format_sidebar_status_md(
        raw_df,
        date_col=resolved_date,
        data_format=data_format,
        y_count=len(y_list),
    )
    return profile_md, status_md


def on_cast_apply(state, cast_col, cast_type, date_col, data_format, group_col, value_col, y_cols):
    raw_df = state.get("raw_df_original")
    if raw_df is None or not cast_col or cast_col not in raw_df.columns:
        return (
            state,
            gr.Dataframe(),
            "",
            _format_sidebar_status_md(None),
            gr.Dropdown(),
            gr.Column(visible=(data_format == "Long")),
            gr.Dropdown(),
            gr.Dropdown(),
            gr.CheckboxGroup(),
            gr.Dropdown(),
            "*Select a valid column to cast.*",
        )

    updated = raw_df.copy()
    try:
        if cast_type == "Numeric (coerce)":
            updated[cast_col] = pd.to_numeric(updated[cast_col], errors="coerce")
        elif cast_type == "Datetime (coerce)":
            updated[cast_col] = pd.to_datetime(updated[cast_col], errors="coerce")
        else:
            updated[cast_col] = updated[cast_col].astype(str)
    except Exception as exc:
        return (
            state,
            gr.Dataframe(value=_preview_df(raw_df)),
            "",
            _format_sidebar_status_md(raw_df, date_col=date_col, data_format=data_format),
            gr.Dropdown(choices=list(raw_df.columns), value=date_col),
            gr.Column(visible=(data_format == "Long")),
            gr.Dropdown(),
            gr.Dropdown(),
            gr.CheckboxGroup(),
            gr.Dropdown(choices=list(raw_df.columns), value=cast_col),
            f"*Type cast failed: {exc}*",
        )

    state["raw_df_original"] = updated
    all_cols = list(updated.columns)
    next_date = date_col if date_col in all_cols else _choose_default_date_col(updated)

    opts = _derive_setup_options(
        updated,
        date_col=next_date,
        data_format=data_format,
        group_col=group_col,
        value_col=value_col,
        current_y=list(y_cols) if y_cols else [],
    )

    profile_md = _format_raw_profile_md(updated, opts["resolved_date"], data_format, opts["default_y"])
    status_md = _format_sidebar_status_md(
        updated,
        date_col=opts["resolved_date"],
        data_format=data_format,
        y_count=len(opts["default_y"]),
    )

    return (
        state,
        gr.Dataframe(value=_preview_df(updated)),
        profile_md,
        status_md,
        gr.Dropdown(choices=all_cols, value=opts["resolved_date"]),
        gr.Column(visible=(data_format == "Long")),
        gr.Dropdown(choices=opts["string_cols"], value=opts["group_default"]),
        gr.Dropdown(choices=opts["value_options"], value=opts["value_default"]),
        gr.CheckboxGroup(choices=opts["available_y"], value=opts["default_y"]),
        gr.Dropdown(choices=all_cols, value=cast_col),
        f"*Applied cast: `{cast_col}` -> {cast_type}*",
    )


def on_apply_setup(state, date_col, data_format, group_col, value_col,
                   y_cols, dup_action, missing_action, freq_override):
    def _error_return(message: str):
        return (
            state,                                      # 0  app_state
            gr.Column(visible=False),                  # 1  welcome_col
            gr.Column(visible=True),                   # 2  analysis_col
            message,                                   # 3  quality_md
            "",                                        # 4  freq_info_md
            gr.Dropdown(),                             # 5  single_y_dd
            gr.Dropdown(),                             # 6  color_by_dd
            None,                                      # 7  single_plot
            "",                                        # 8  single_stats_md
            "",                                        # 9  single_interp_md
            gr.CheckboxGroup(),                        # 10 panel_cols_cbg
            None,                                      # 11 panel_plot
            "",                                        # 12 panel_summary_md
            "",                                        # 13 panel_interp_md
            gr.CheckboxGroup(),                        # 14 spag_cols_cbg
            gr.Dropdown(),                             # 15 spag_highlight_dd
            None,                                      # 16 spag_plot
            "",                                        # 17 spag_summary_md
            "",                                        # 18 spag_interp_md
            gr.Radio(choices=[_MODE_SINGLE], value=_MODE_SINGLE),  # 19 viz_mode_radio
            "Load data and apply setup.",              # 20 mode_hint_md
            gr.Dropdown(choices=_CHART_TYPES, value=_CHART_TYPES[0]),  # 21 single_chart_dd
            "",                                        # 22 single_gate_md
            gr.Column(visible=False),                  # 23 single_mode_col
            gr.Column(visible=False),                  # 24 panel_mode_col
            gr.Column(visible=False),                  # 25 spag_mode_col
            pd.DataFrame(),                            # 26 cleaned_preview_df
            _format_sidebar_status_md(None),           # 27 status_md
        )

    if not y_cols:
        return _error_return("*Select at least one value column.*")

    raw_df = state.get("raw_df_original")
    if raw_df is None:
        return _error_return("*No data loaded.*")

    # Pivot if long format
    if data_format == "Long" and group_col and value_col:
        effective_df = pivot_long_to_wide(raw_df, date_col, group_col, value_col)
    else:
        effective_df = raw_df

    # Clean
    cleaned, report = clean_dataframe(
        effective_df, date_col, list(y_cols),
        dup_action=dup_action, missing_action=missing_action,
    )
    freq = detect_frequency(cleaned, date_col)
    cleaned = add_time_features(cleaned, date_col)

    if freq_override and freq_override.strip():
        freq = FrequencyInfo(
            label=freq_override.strip(),
            median_delta=freq.median_delta,
            is_regular=freq.is_regular,
        )

    state["cleaned_df"] = cleaned
    state["cleaning_report"] = report
    state["freq_info"] = freq
    state["date_col"] = date_col
    state["y_cols"] = list(y_cols)
    state["setup_applied"] = True
    state["single_png"] = None
    state["panel_png"] = None
    state["spag_png"] = None

    # Create QueryChat instance if available
    if check_querychat_available():
        try:
            state["qc"] = create_querychat(
                cleaned, name="uploaded_data",
                date_col=date_col, y_cols=list(y_cols),
                freq_label=freq.label,
            )
        except Exception:
            state["qc"] = None
    else:
        state["qc"] = None

    quality_md = _format_cleaning_report_md(report)
    freq_text = f"Frequency: **{freq.label}** ({'regular' if freq.is_regular else 'irregular'})"

    # Color-by choices
    color_by_choices = []
    if "month" in cleaned.columns:
        color_by_choices = ["month", "quarter", "year", "day_of_week"]

    y_list = list(y_cols)
    panel_default = y_list[:4] if len(y_list) >= 2 else y_list
    highlight_choices = ["(none)"] + y_list
    mode_choices, recommended_mode, mode_hint = _get_mode_config(len(y_list))
    single_visible, panel_visible, spag_visible = _mode_visibility(recommended_mode)
    state["mode_choices"] = mode_choices
    state["recommended_mode"] = recommended_mode

    chart_choices, chart_gate_md = _available_chart_choices(
        cleaned, date_col, y_list[0], freq
    )

    status_md = _format_sidebar_status_md(
        df=cleaned,
        date_col=date_col,
        data_format=data_format,
        y_count=len(y_list),
        freq_label=freq.label,
        cleaned_rows=report.rows_after,
    )

    return (
        state,                                                              # 0  app_state
        gr.Column(visible=False),                                           # 1  welcome_col
        gr.Column(visible=True),                                            # 2  analysis_col
        quality_md,                                                         # 3  quality_md
        freq_text,                                                          # 4  freq_info_md
        gr.Dropdown(choices=y_list, value=y_list[0]),                       # 5  single_y_dd
        gr.Dropdown(choices=color_by_choices,
                    value=color_by_choices[0] if color_by_choices else None),# 6  color_by_dd
        None,                                                               # 7  single_plot
        "",                                                                 # 8  single_stats_md
        "",                                                                 # 9  single_interp_md
        gr.CheckboxGroup(choices=y_list, value=panel_default),              # 10 panel_cols_cbg
        None,                                                               # 11 panel_plot
        "",                                                                 # 12 panel_summary_md
        "",                                                                 # 13 panel_interp_md
        gr.CheckboxGroup(choices=y_list, value=y_list),                     # 14 spag_cols_cbg
        gr.Dropdown(choices=highlight_choices, value="(none)"),             # 15 spag_highlight_dd
        None,                                                               # 16 spag_plot
        "",                                                                 # 17 spag_summary_md
        "",                                                                 # 18 spag_interp_md
        gr.Radio(choices=mode_choices, value=recommended_mode),             # 19 viz_mode_radio
        mode_hint,                                                          # 20 mode_hint_md
        gr.Dropdown(choices=chart_choices, value=chart_choices[0]),         # 21 single_chart_dd
        chart_gate_md,                                                      # 22 single_gate_md
        gr.Column(visible=single_visible),                                  # 23 single_mode_col
        gr.Column(visible=panel_visible),                                   # 24 panel_mode_col
        gr.Column(visible=spag_visible),                                    # 25 spag_mode_col
        _preview_df(cleaned),                                               # 26 cleaned_preview_df
        status_md,                                                          # 27 status_md
    )


# ---- Visibility toggles ----

def on_dr_mode_change(mode):
    return (
        gr.Column(visible=(mode == "Last N years")),
        gr.Column(visible=(mode == "Custom")),
    )


def on_viz_mode_change(mode):
    single_visible, panel_visible, spag_visible = _mode_visibility(mode)
    return (
        gr.Column(visible=single_visible),
        gr.Column(visible=panel_visible),
        gr.Column(visible=spag_visible),
    )


def on_single_y_change(state, y_col, current_chart):
    cleaned_df = state.get("cleaned_df")
    date_col = state.get("date_col")
    freq_info = state.get("freq_info")
    if cleaned_df is None or not y_col or not date_col:
        return gr.Dropdown(choices=_CHART_TYPES, value=_CHART_TYPES[0]), ""

    chart_choices, chart_gate_md = _available_chart_choices(cleaned_df, date_col, y_col, freq_info)
    next_chart = current_chart if current_chart in chart_choices else chart_choices[0]
    return gr.Dropdown(choices=chart_choices, value=next_chart), chart_gate_md


def on_chart_type_change(chart_type):
    return (
        gr.Column(visible=("Colored Markers" in chart_type)),
        gr.Column(visible=(chart_type in ("Seasonal Plot", "Seasonal Sub-series"))),
        gr.Column(visible=(chart_type == "Rolling Mean Overlay")),
        gr.Column(visible=(chart_type == "Lag Plot")),
        gr.Column(visible=(chart_type == "Decomposition")),
    )


def on_palette_change(pal_name):
    colors = get_palette_colors(pal_name, 8)
    return render_palette_preview(colors)


# ---- Single series ----

def on_single_update(state, y_col, dr_mode, dr_n, dr_start, dr_end,
                     chart_type, palette_name, color_by, period,
                     window, lag, decomp_model):
    cleaned_df = state.get("cleaned_df")
    date_col = state.get("date_col")
    freq_info = state.get("freq_info")

    if cleaned_df is None or not y_col:
        return state, None, "*No data. Apply setup first.*"

    palette_colors = get_palette_colors(palette_name, 12)
    df_plot = _apply_date_filter(cleaned_df.copy(), date_col, dr_mode, dr_n, dr_start, dr_end)

    if df_plot.empty:
        return state, None, "*No data in selected range.*"

    blocked = _chart_availability(df_plot, date_col, y_col, freq_info)
    if chart_type in blocked:
        return state, None, f"*{chart_type} unavailable: {blocked[chart_type]}*"

    fig, err = _generate_single_chart(
        df_plot, date_col, y_col, chart_type, palette_colors,
        color_by, period, window, lag, decomp_model, freq_info,
    )

    if err:
        return state, None, f"**Chart error:** {err}"

    # Summary stats
    stats = compute_summary_stats(df_plot, date_col, y_col)
    stats_md = _format_summary_stats_md(stats)

    # Store PNG for AI interpretation
    state["single_png"] = fig_to_png_bytes(fig) if fig else None

    return state, fig, stats_md


def on_single_interpret(state):
    png = state.get("single_png")
    if not png:
        return "*Generate a chart first, then click Interpret.*"
    if not check_api_key_available():
        return "*Set `OPENAI_API_KEY` to enable AI interpretation.*"

    freq_info = state.get("freq_info")
    metadata = {
        "chart_type": "single series",
        "frequency_label": freq_info.label if freq_info else "Unknown",
        "y_column": state.get("y_cols", [""])[0],
    }
    interp = interpret_chart(png, metadata)
    return render_interpretation_markdown(interp)


# ---- Panel ----

def on_panel_update(state, panel_cols, panel_chart, shared_y, palette_name):
    cleaned_df = state.get("cleaned_df")
    date_col = state.get("date_col")

    if cleaned_df is None or not panel_cols or len(panel_cols) < 2:
        return state, None, "*Select 2+ columns and apply setup first.*"

    palette_colors = get_palette_colors(palette_name, len(panel_cols))

    try:
        fig = plot_panel(
            cleaned_df, date_col, list(panel_cols),
            chart_type=panel_chart, shared_y=shared_y,
            title="Panel Comparison",
            style_dict=_STYLE_DICT, palette_colors=palette_colors,
        )
        summary_df = compute_multi_series_summary(cleaned_df, date_col, list(panel_cols))
        summary_md = _format_multi_summary_md(summary_df)
        state["panel_png"] = fig_to_png_bytes(fig)
        return state, fig, summary_md
    except Exception as exc:
        return state, None, f"**Panel chart error:** {exc}"


def on_panel_interpret(state):
    png = state.get("panel_png")
    if not png:
        return "*Generate a panel chart first, then click Interpret.*"
    if not check_api_key_available():
        return "*Set `OPENAI_API_KEY` to enable AI interpretation.*"

    freq_info = state.get("freq_info")
    metadata = {
        "chart_type": "panel (small multiples)",
        "frequency_label": freq_info.label if freq_info else "Unknown",
        "y_column": ", ".join(state.get("y_cols", [])),
    }
    interp = interpret_chart(png, metadata)
    return render_interpretation_markdown(interp)


# ---- Spaghetti ----

def on_spag_update(state, spag_cols, alpha, topn, highlight, show_median, palette_name):
    cleaned_df = state.get("cleaned_df")
    date_col = state.get("date_col")

    if cleaned_df is None or not spag_cols or len(spag_cols) < 2:
        return state, None, "*Select 2+ columns and apply setup first.*"

    highlight_col = highlight if highlight and highlight != "(none)" else None
    top_n = int(topn) if topn and int(topn) > 0 else None
    palette_colors = get_palette_colors(palette_name, len(spag_cols))

    try:
        fig = plot_spaghetti(
            cleaned_df, date_col, list(spag_cols),
            alpha=float(alpha),
            highlight_col=highlight_col,
            top_n=top_n,
            show_median_band=bool(show_median),
            title="Spaghetti Plot",
            style_dict=_STYLE_DICT, palette_colors=palette_colors,
        )
        summary_df = compute_multi_series_summary(cleaned_df, date_col, list(spag_cols))
        summary_md = _format_multi_summary_md(summary_df)
        state["spag_png"] = fig_to_png_bytes(fig)
        return state, fig, summary_md
    except Exception as exc:
        return state, None, f"**Spaghetti chart error:** {exc}"


def on_spag_interpret(state):
    png = state.get("spag_png")
    if not png:
        return "*Generate a spaghetti chart first, then click Interpret.*"
    if not check_api_key_available():
        return "*Set `OPENAI_API_KEY` to enable AI interpretation.*"

    freq_info = state.get("freq_info")
    metadata = {
        "chart_type": "spaghetti (overlay)",
        "frequency_label": freq_info.label if freq_info else "Unknown",
        "y_column": ", ".join(state.get("y_cols", [])),
    }
    interp = interpret_chart(png, metadata)
    return render_interpretation_markdown(interp)


def on_auto_generate(state, viz_mode,
                     single_y, dr_mode, dr_n, dr_start, dr_end,
                     single_chart, single_pal, color_by, period, window, lag, decomp_model,
                     panel_cols, panel_chart, panel_shared, panel_pal,
                     spag_cols, spag_alpha, spag_topn, spag_highlight, spag_median, spag_pal):
    if viz_mode == _MODE_PANEL:
        next_state, fig, summary_md = on_panel_update(
            state, panel_cols, panel_chart, panel_shared, panel_pal
        )
        return next_state, None, "", fig, summary_md, None, ""

    if viz_mode == _MODE_SPAG:
        next_state, fig, summary_md = on_spag_update(
            state, spag_cols, spag_alpha, spag_topn, spag_highlight, spag_median, spag_pal
        )
        return next_state, None, "", None, "", fig, summary_md

    next_state, fig, stats_md = on_single_update(
        state, single_y, dr_mode, dr_n, dr_start, dr_end,
        single_chart, single_pal, color_by, period, window, lag, decomp_model
    )
    return next_state, fig, stats_md, None, "", None, ""


# ---------------------------------------------------------------------------
# Build the Gradio app
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="Time Series Visualizer",
) as demo:

    app_state = gr.State(_make_empty_state())

    # ===================================================================
    # Sidebar
    # ===================================================================
    with gr.Sidebar():
        gr.HTML(
            '<div class="app-title">'
            '<span class="title-text">Time Series Visualizer</span><br>'
            '<span class="subtitle-text">ISA 444 &middot; Miami University</span>'
            '</div>'
        )
        gr.Markdown("### Data Input")

        file_upload = gr.File(
            label="Upload a CSV file",
            file_types=[".csv", ".tsv", ".txt"],
            type="filepath",
        )
        demo_dd = gr.Dropdown(
            label="Or load a demo dataset",
            choices=_DEMO_CHOICES,
            value="(none)",
        )
        reset_btn = gr.Button("Reset all", variant="secondary", size="sm")
        delim_md = gr.Markdown("")
        status_md = gr.Markdown("*No data loaded yet.*")

        with gr.Accordion("About", open=False):
            gr.Markdown("**Vibe-Coded By**")
            gr.HTML(_DEVELOPER_CARD)
            gr.Markdown("v0.2.0 &middot; Last updated Feb 2026", elem_classes=["caption"])
            gr.Markdown("---")
            gr.Markdown("### QueryChat")
            if check_querychat_available():
                gr.Markdown(
                    "QueryChat natural-language filtering is available. "
                    "Use the chat below to filter your dataset."
                )
            else:
                gr.Markdown(
                    "*Set `OPENAI_API_KEY` and install `querychat[gradio]` "
                    "to enable natural-language data filtering.*"
                )

    # ===================================================================
    # Welcome screen
    # ===================================================================
    with gr.Column(visible=True) as welcome_col:
        gr.Markdown(_WELCOME_MD)

    # ===================================================================
    # Setup panel (hidden until data loaded)
    # ===================================================================
    with gr.Column(visible=False) as setup_col:
        gr.Markdown("## Step 1. Understand Data")
        gr.Markdown("*Check inferred structure and preview your raw file before cleaning.*")
        raw_profile_md = gr.Markdown("")
        raw_preview_df = gr.Dataframe(
            label="Raw data preview (first 10 rows)",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("## Step 2. Prepare Data")
        gr.Markdown("*If the date guess is wrong, change the date column - value choices update automatically.*")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Structure")
                date_col_dd = gr.Dropdown(label="Date column", choices=[])
                format_radio = gr.Radio(
                    label="Data format", choices=["Wide", "Long"], value="Wide",
                )
                with gr.Column(visible=False) as long_col:
                    group_col_dd = gr.Dropdown(label="Group column", choices=[])
                    value_col_dd = gr.Dropdown(label="Value column", choices=[])
                y_cols_cbg = gr.CheckboxGroup(label="Value column(s)", choices=[])

            with gr.Column(scale=1, min_width=300):
                gr.Markdown("### Cleaning")
                dup_dd = gr.Dropdown(
                    label="Duplicate dates",
                    choices=["keep_last", "keep_first", "drop_all"],
                    value="keep_last",
                )
                missing_dd = gr.Dropdown(
                    label="Missing values",
                    choices=["interpolate", "ffill", "drop"],
                    value="interpolate",
                )
                freq_tb = gr.Textbox(
                    label="Override frequency label (optional)",
                    placeholder="e.g. Daily, Weekly, Monthly",
                )
                apply_btn = gr.Button("Apply setup", variant="primary")
                freq_info_md = gr.Markdown("")

                with gr.Accordion("Type Casting (optional)", open=False):
                    gr.Markdown("*Use this when a column is read with the wrong dtype.*")
                    cast_col_dd = gr.Dropdown(label="Column", choices=[])
                    cast_type_dd = gr.Dropdown(
                        label="Cast to",
                        choices=["Numeric (coerce)", "Datetime (coerce)", "String"],
                        value="Numeric (coerce)",
                    )
                    cast_apply_btn = gr.Button("Apply cast", variant="secondary", size="sm")
                    cast_status_md = gr.Markdown("")

    # ===================================================================
    # Analysis panel (hidden until setup applied)
    # ===================================================================
    with gr.Column(visible=False) as analysis_col:
        gr.Markdown("## Step 3. Visualize")
        mode_hint_md = gr.Markdown("")
        viz_mode_radio = gr.Radio(
            label="Visualization mode",
            choices=[_MODE_SINGLE],
            value=_MODE_SINGLE,
        )

        with gr.Accordion("Cleaned Data Preview", open=False):
            cleaned_preview_df = gr.Dataframe(
                label="Cleaned data preview (first 10 rows)",
                interactive=False,
                wrap=True,
            )

        with gr.Accordion("Data Quality Report", open=False):
            quality_md = gr.Markdown("")

        with gr.Column(visible=False) as single_mode_col:
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    single_y_dd = gr.Dropdown(label="Value column", choices=[])
                    dr_mode_radio = gr.Radio(
                        label="Date range",
                        choices=["All", "Last N years", "Custom"],
                        value="All",
                    )
                    with gr.Column(visible=False) as dr_n_col:
                        dr_n_slider = gr.Slider(
                            label="Years", minimum=1, maximum=20,
                            value=5, step=1,
                        )
                    with gr.Column(visible=False) as dr_custom_col:
                        dr_start_tb = gr.Textbox(label="Start date", placeholder="YYYY-MM-DD")
                        dr_end_tb = gr.Textbox(label="End date", placeholder="YYYY-MM-DD")

                    single_chart_dd = gr.Dropdown(
                        label="Chart type", choices=_CHART_TYPES,
                        value=_CHART_TYPES[0],
                    )
                    single_gate_md = gr.Markdown("")
                    single_pal_dd = gr.Dropdown(
                        label="Color palette", choices=_PALETTE_NAMES,
                        value=_PALETTE_NAMES[0],
                    )
                    single_swatch = gr.Plot(label="Palette preview", show_label=False)

                    with gr.Column(visible=False) as color_by_col:
                        color_by_dd = gr.Dropdown(
                            label="Color by",
                            choices=["month", "quarter", "year", "day_of_week"],
                        )
                    with gr.Column(visible=False) as period_col:
                        period_dd = gr.Dropdown(
                            label="Period", choices=["month", "quarter"],
                            value="month",
                        )
                    with gr.Column(visible=False) as window_col:
                        window_slider = gr.Slider(
                            label="Window", minimum=2, maximum=52,
                            value=12, step=1,
                        )
                    with gr.Column(visible=False) as lag_col:
                        lag_slider = gr.Slider(
                            label="Lag", minimum=1, maximum=52,
                            value=1, step=1,
                        )
                    with gr.Column(visible=False) as decomp_col:
                        decomp_dd = gr.Dropdown(
                            label="Model",
                            choices=["additive", "multiplicative"],
                            value="additive",
                        )
                    single_update_btn = gr.Button("Update chart", variant="primary")

                with gr.Column(scale=3):
                    single_plot = gr.Plot(label="Chart")
                    with gr.Accordion("Summary Statistics", open=False):
                        single_stats_md = gr.Markdown("")
                    with gr.Accordion("AI Chart Interpretation", open=False):
                        gr.Markdown(
                            "*The chart image (PNG) is sent to OpenAI for "
                            "interpretation. Do not include sensitive data.*"
                        )
                        single_interp_btn = gr.Button(
                            "Interpret Chart with AI", variant="secondary",
                        )
                        single_interp_md = gr.Markdown("")

        with gr.Column(visible=False) as panel_mode_col:
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    panel_cols_cbg = gr.CheckboxGroup(
                        label="Columns to plot", choices=[],
                    )
                    panel_chart_dd = gr.Dropdown(
                        label="Chart type", choices=["line", "bar"],
                        value="line",
                    )
                    panel_shared_cb = gr.Checkbox(
                        label="Shared Y axis", value=True,
                    )
                    panel_pal_dd = gr.Dropdown(
                        label="Color palette", choices=_PALETTE_NAMES,
                        value=_PALETTE_NAMES[0],
                    )
                    panel_update_btn = gr.Button("Update chart", variant="primary")

                with gr.Column(scale=3):
                    panel_plot = gr.Plot(label="Panel Chart")
                    with gr.Accordion("Per-series Summary", open=False):
                        panel_summary_md = gr.Markdown("")
                    with gr.Accordion("AI Chart Interpretation", open=False):
                        gr.Markdown(
                            "*The chart image (PNG) is sent to OpenAI for "
                            "interpretation. Do not include sensitive data.*"
                        )
                        panel_interp_btn = gr.Button(
                            "Interpret Chart with AI", variant="secondary",
                        )
                        panel_interp_md = gr.Markdown("")

        with gr.Column(visible=False) as spag_mode_col:
            with gr.Row():
                with gr.Column(scale=1, min_width=280):
                    spag_cols_cbg = gr.CheckboxGroup(
                        label="Columns to include", choices=[],
                    )
                    spag_alpha_slider = gr.Slider(
                        label="Alpha (opacity)",
                        minimum=0.05, maximum=1.0, value=0.15, step=0.05,
                    )
                    spag_topn_num = gr.Number(
                        label="Highlight top N (0 = none)", value=0,
                        minimum=0, precision=0,
                    )
                    spag_highlight_dd = gr.Dropdown(
                        label="Highlight series",
                        choices=["(none)"], value="(none)",
                    )
                    spag_median_cb = gr.Checkbox(
                        label="Show Median + IQR band", value=False,
                    )
                    spag_pal_dd = gr.Dropdown(
                        label="Color palette", choices=_PALETTE_NAMES,
                        value=_PALETTE_NAMES[0],
                    )
                    spag_update_btn = gr.Button("Update chart", variant="primary")

                with gr.Column(scale=3):
                    spag_plot = gr.Plot(label="Spaghetti Chart")
                    with gr.Accordion("Per-series Summary", open=False):
                        spag_summary_md = gr.Markdown("")
                    with gr.Accordion("AI Chart Interpretation", open=False):
                        gr.Markdown(
                            "*The chart image (PNG) is sent to OpenAI for "
                            "interpretation. Do not include sensitive data.*"
                        )
                        spag_interp_btn = gr.Button(
                            "Interpret Chart with AI", variant="secondary",
                        )
                        spag_interp_md = gr.Markdown("")

    # ===================================================================
    # Event wiring
    # ===================================================================

    _DATA_LOAD_OUTPUTS = [
        app_state, setup_col, date_col_dd, format_radio, long_col,
        group_col_dd, value_col_dd, y_cols_cbg, delim_md,
        status_md, raw_profile_md, raw_preview_df,
        cast_col_dd, cast_status_md,
        welcome_col, analysis_col,
        viz_mode_radio, mode_hint_md, single_chart_dd, single_gate_md,
        single_mode_col, panel_mode_col, spag_mode_col, cleaned_preview_df,
    ]

    file_upload.change(
        on_file_upload,
        inputs=[file_upload, app_state],
        outputs=_DATA_LOAD_OUTPUTS,
    )

    demo_dd.change(
        on_demo_select,
        inputs=[demo_dd, app_state],
        outputs=_DATA_LOAD_OUTPUTS,
    )

    # Reset via page reload
    reset_btn.click(fn=None, js="() => { window.location.reload(); }")

    date_col_dd.change(
        on_setup_inputs_change,
        inputs=[date_col_dd, format_radio, group_col_dd, value_col_dd, y_cols_cbg, app_state],
        outputs=[long_col, group_col_dd, value_col_dd, y_cols_cbg, raw_profile_md, status_md],
    )

    format_radio.change(
        on_setup_inputs_change,
        inputs=[date_col_dd, format_radio, group_col_dd, value_col_dd, y_cols_cbg, app_state],
        outputs=[long_col, group_col_dd, value_col_dd, y_cols_cbg, raw_profile_md, status_md],
    )

    # Long-format column changes update y_cols
    for _comp in [group_col_dd, value_col_dd]:
        _comp.change(
            on_long_cols_change,
            inputs=[date_col_dd, group_col_dd, value_col_dd, app_state],
            outputs=[y_cols_cbg],
        )

    y_cols_cbg.change(
        on_y_selection_change,
        inputs=[date_col_dd, format_radio, y_cols_cbg, app_state],
        outputs=[raw_profile_md, status_md],
    )

    cast_apply_btn.click(
        on_cast_apply,
        inputs=[app_state, cast_col_dd, cast_type_dd, date_col_dd, format_radio, group_col_dd, value_col_dd, y_cols_cbg],
        outputs=[
            app_state, raw_preview_df, raw_profile_md, status_md, date_col_dd,
            long_col, group_col_dd, value_col_dd, y_cols_cbg, cast_col_dd, cast_status_md,
        ],
    )

    # Apply setup
    _APPLY_OUTPUTS = [
        app_state,             # 0
        welcome_col,           # 1
        analysis_col,          # 2
        quality_md,            # 3
        freq_info_md,          # 4
        # Single
        single_y_dd,           # 5
        color_by_dd,           # 6
        single_plot,           # 7
        single_stats_md,       # 8
        single_interp_md,      # 9
        # Panel
        panel_cols_cbg,        # 10
        panel_plot,            # 11
        panel_summary_md,      # 12
        panel_interp_md,       # 13
        # Spaghetti
        spag_cols_cbg,         # 14
        spag_highlight_dd,     # 15
        spag_plot,             # 16
        spag_summary_md,       # 17
        spag_interp_md,        # 18
        viz_mode_radio,        # 19
        mode_hint_md,          # 20
        single_chart_dd,       # 21
        single_gate_md,        # 22
        single_mode_col,       # 23
        panel_mode_col,        # 24
        spag_mode_col,         # 25
        cleaned_preview_df,    # 26
        status_md,             # 27
    ]

    apply_btn.click(
        on_apply_setup,
        inputs=[
            app_state, date_col_dd, format_radio, group_col_dd,
            value_col_dd, y_cols_cbg, dup_dd, missing_dd, freq_tb,
        ],
        outputs=_APPLY_OUTPUTS,
    ).then(
        on_auto_generate,
        inputs=[
            app_state, viz_mode_radio,
            single_y_dd, dr_mode_radio, dr_n_slider, dr_start_tb, dr_end_tb,
            single_chart_dd, single_pal_dd, color_by_dd, period_dd, window_slider, lag_slider, decomp_dd,
            panel_cols_cbg, panel_chart_dd, panel_shared_cb, panel_pal_dd,
            spag_cols_cbg, spag_alpha_slider, spag_topn_num, spag_highlight_dd, spag_median_cb, spag_pal_dd,
        ],
        outputs=[app_state, single_plot, single_stats_md, panel_plot, panel_summary_md, spag_plot, spag_summary_md],
    )

    # Date range mode visibility
    dr_mode_radio.change(
        on_dr_mode_change,
        inputs=[dr_mode_radio],
        outputs=[dr_n_col, dr_custom_col],
    )

    viz_mode_radio.change(
        on_viz_mode_change,
        inputs=[viz_mode_radio],
        outputs=[single_mode_col, panel_mode_col, spag_mode_col],
    )

    single_y_dd.change(
        on_single_y_change,
        inputs=[app_state, single_y_dd, single_chart_dd],
        outputs=[single_chart_dd, single_gate_md],
    )

    # Chart type conditional controls
    single_chart_dd.change(
        on_chart_type_change,
        inputs=[single_chart_dd],
        outputs=[color_by_col, period_col, window_col, lag_col, decomp_col],
    )

    # Palette swatch preview
    single_pal_dd.change(on_palette_change, [single_pal_dd], [single_swatch])

    # Initialise swatch on load
    demo.load(on_palette_change, [single_pal_dd], [single_swatch])

    # ---- Single series chart + stats ----
    single_update_btn.click(
        on_single_update,
        inputs=[
            app_state, single_y_dd, dr_mode_radio, dr_n_slider,
            dr_start_tb, dr_end_tb, single_chart_dd, single_pal_dd,
            color_by_dd, period_dd, window_slider, lag_slider, decomp_dd,
        ],
        outputs=[app_state, single_plot, single_stats_md],
    )

    single_interp_btn.click(
        on_single_interpret,
        inputs=[app_state],
        outputs=[single_interp_md],
    )

    # ---- Panel chart + stats ----
    panel_update_btn.click(
        on_panel_update,
        inputs=[app_state, panel_cols_cbg, panel_chart_dd, panel_shared_cb, panel_pal_dd],
        outputs=[app_state, panel_plot, panel_summary_md],
    )

    panel_interp_btn.click(
        on_panel_interpret,
        inputs=[app_state],
        outputs=[panel_interp_md],
    )

    # ---- Spaghetti chart + stats ----
    spag_update_btn.click(
        on_spag_update,
        inputs=[
            app_state, spag_cols_cbg, spag_alpha_slider, spag_topn_num,
            spag_highlight_dd, spag_median_cb, spag_pal_dd,
        ],
        outputs=[app_state, spag_plot, spag_summary_md],
    )

    spag_interp_btn.click(
        on_spag_interpret,
        inputs=[app_state],
        outputs=[spag_interp_md],
    )


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=MiamiTheme(),
        css=get_miami_css(),
    )
