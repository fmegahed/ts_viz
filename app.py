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

### Get Started in 3 Steps

<div style="display:grid; grid-template-columns:repeat(3, 1fr); gap:1rem; margin:1rem 0;">
<div class="step-card">
  <div class="step-number">1</div>
  <div class="step-title">Load Data</div>
  <div class="step-desc">Upload a CSV from the sidebar or pick one of the built-in demo datasets.</div>
</div>
<div class="step-card">
  <div class="step-number">2</div>
  <div class="step-title">Pick Columns</div>
  <div class="step-desc">Select a date column and one or more numeric value columns. The app auto-detects sensible defaults.</div>
</div>
<div class="step-card">
  <div class="step-number">3</div>
  <div class="step-title">Explore</div>
  <div class="step-desc">Choose from 9+ chart types, view summary statistics, and get AI-powered chart interpretation.</div>
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
    date_suggestions = suggest_date_columns(df)
    default_date = date_suggestions[0] if date_suggestions else all_cols[0]

    is_long, auto_group, auto_value = detect_long_format(df, default_date)
    fmt = "Long" if is_long else "Wide"

    other_cols = [c for c in all_cols if c != default_date]
    string_cols = [
        c for c in other_cols
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c])
    ]
    numeric_cols = [
        c for c in other_cols if pd.api.types.is_numeric_dtype(df[c])
    ]

    group_default = (
        auto_group if auto_group and auto_group in string_cols
        else (string_cols[0] if string_cols else None)
    )
    value_options = [c for c in numeric_cols if c != group_default] if group_default else numeric_cols
    value_default = (
        auto_value if auto_value and auto_value in value_options
        else (value_options[0] if value_options else None)
    )

    # Compute initial y_cols
    if is_long and group_default and value_default:
        try:
            effective = pivot_long_to_wide(df, default_date, group_default, value_default)
            available_y = [c for c in effective.columns if c != default_date]
        except Exception:
            available_y = list(numeric_cols)
    else:
        numeric_suggest = suggest_numeric_columns(df)
        available_y = [c for c in numeric_suggest if c != default_date]

    default_y = available_y[:4] if available_y else []
    delim_text = f"Detected delimiter: `{repr(delim)}`" if delim else ""

    return (
        state,                                                           # app_state
        gr.Column(visible=True),                                         # setup_col
        gr.Dropdown(choices=all_cols, value=default_date),               # date_col_dd
        gr.Radio(value=fmt),                                             # format_radio
        gr.Column(visible=is_long),                                      # long_col
        gr.Dropdown(choices=string_cols, value=group_default),           # group_col_dd
        gr.Dropdown(choices=value_options, value=value_default),         # value_col_dd
        gr.CheckboxGroup(choices=available_y, value=default_y),          # y_cols_cbg
        delim_text,                                                      # delim_md
        gr.Column(visible=True),                                         # welcome_col
        gr.Column(visible=False),                                        # analysis_col
    )


def on_file_upload(file_obj, state):
    if file_obj is None:
        empty = _make_empty_state()
        return (
            empty,
            gr.Column(visible=False), gr.Dropdown(), gr.Radio(),
            gr.Column(visible=False), gr.Dropdown(), gr.Dropdown(),
            gr.CheckboxGroup(choices=[], value=[]), "",
            gr.Column(visible=True), gr.Column(visible=False),
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
            gr.Column(), gr.Column(),
        )
    demo_path = _DEMO_FILES[choice]
    df = pd.read_csv(demo_path)
    return _process_new_data(df, None)


def on_format_change(fmt):
    return gr.Column(visible=(fmt == "Long"))


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


def on_apply_setup(state, date_col, data_format, group_col, value_col,
                   y_cols, dup_action, missing_action, freq_override):
    if not y_cols:
        return (
            state,
            gr.Column(visible=True), gr.Column(visible=False),
            "*Select at least one value column.*", "",
            gr.Dropdown(), gr.Dropdown(),
            None, "", "",
            gr.CheckboxGroup(), None, "", "",
            gr.CheckboxGroup(), gr.Dropdown(), None, "", "",
        )

    raw_df = state.get("raw_df_original")
    if raw_df is None:
        return (
            state,
            gr.Column(visible=True), gr.Column(visible=False),
            "*No data loaded.*", "",
            gr.Dropdown(), gr.Dropdown(),
            None, "", "",
            gr.CheckboxGroup(), None, "", "",
            gr.CheckboxGroup(), gr.Dropdown(), None, "", "",
        )

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

    return (
        state,                                                              # 0  app_state
        gr.Column(visible=False),                                           # 1  welcome_col
        gr.Column(visible=True),                                            # 2  analysis_col
        quality_md,                                                         # 3  quality_md
        freq_text,                                                          # 4  freq_info_md
        # Single series tab
        gr.Dropdown(choices=y_list, value=y_list[0]),                       # 5  single_y_dd
        gr.Dropdown(choices=color_by_choices,
                    value=color_by_choices[0] if color_by_choices else None),# 6  color_by_dd
        None,                                                               # 7  single_plot
        "",                                                                 # 8  single_stats_md
        "",                                                                 # 9  single_interp_md
        # Panel tab
        gr.CheckboxGroup(choices=y_list, value=panel_default),              # 10 panel_cols_cbg
        None,                                                               # 11 panel_plot
        "",                                                                 # 12 panel_summary_md
        "",                                                                 # 13 panel_interp_md
        # Spaghetti tab
        gr.CheckboxGroup(choices=y_list, value=y_list),                     # 14 spag_cols_cbg
        gr.Dropdown(choices=highlight_choices, value="(none)"),              # 15 spag_highlight_dd
        None,                                                               # 16 spag_plot
        "",                                                                 # 17 spag_summary_md
        "",                                                                 # 18 spag_interp_md
    )


# ---- Visibility toggles ----

def on_dr_mode_change(mode):
    return (
        gr.Column(visible=(mode == "Last N years")),
        gr.Column(visible=(mode == "Custom")),
    )


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
        gr.Markdown("**Vibe-Coded By**")
        gr.HTML(_DEVELOPER_CARD)
        gr.Markdown("v0.2.0 &middot; Last updated Feb 2026", elem_classes=["caption"])

        gr.Markdown("---")
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

        # ---- Setup controls (hidden until data loaded) ----
        with gr.Column(visible=False) as setup_col:
            gr.Markdown("---")
            gr.Markdown("### Column & Cleaning Setup")
            gr.Markdown("*Configure below, then click **Apply setup**.*")

            date_col_dd = gr.Dropdown(label="Date column", choices=[])
            format_radio = gr.Radio(
                label="Data format", choices=["Wide", "Long"], value="Wide",
            )

            with gr.Column(visible=False) as long_col:
                group_col_dd = gr.Dropdown(label="Group column", choices=[])
                value_col_dd = gr.Dropdown(label="Value column", choices=[])

            y_cols_cbg = gr.CheckboxGroup(label="Value column(s)", choices=[])

            gr.Markdown("**Cleaning options**")
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

        # ---- QueryChat placeholder ----
        with gr.Column(visible=False) as qc_col:
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
    # Analysis panel (hidden until setup applied)
    # ===================================================================
    with gr.Column(visible=False) as analysis_col:
        with gr.Accordion("Data Quality Report", open=False):
            quality_md = gr.Markdown("")

        with gr.Tabs():
            # ---------------------------------------------------------------
            # Tab: Single Series
            # ---------------------------------------------------------------
            with gr.Tab("Single Series"):
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

            # ---------------------------------------------------------------
            # Tab: Few Series (Panel)
            # ---------------------------------------------------------------
            with gr.Tab("Few Series (Panel)"):
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

            # ---------------------------------------------------------------
            # Tab: Many Series (Spaghetti)
            # ---------------------------------------------------------------
            with gr.Tab("Many Series (Spaghetti)"):
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
        welcome_col, analysis_col,
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

    # Format toggle
    format_radio.change(
        on_format_change,
        inputs=[format_radio],
        outputs=[long_col],
    )

    # Long-format column changes update y_cols
    for _comp in [group_col_dd, value_col_dd]:
        _comp.change(
            on_long_cols_change,
            inputs=[date_col_dd, group_col_dd, value_col_dd, app_state],
            outputs=[y_cols_cbg],
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
    ]

    apply_btn.click(
        on_apply_setup,
        inputs=[
            app_state, date_col_dd, format_radio, group_col_dd,
            value_col_dd, y_cols_cbg, dup_dd, missing_dd, freq_tb,
        ],
        outputs=_APPLY_OUTPUTS,
    )

    # Date range mode visibility
    dr_mode_radio.change(
        on_dr_mode_change,
        inputs=[dr_mode_radio],
        outputs=[dr_n_col, dr_custom_col],
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
