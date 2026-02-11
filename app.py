"""
Time Series Visualizer + AI Chart Interpreter
=============================================
Main Streamlit application.  Run with:

    streamlit run app.py --server.port=7860
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import streamlit as st

from src.ui_theme import (
    apply_miami_theme,
    get_miami_mpl_style,
    get_palette_colors,
    render_palette_preview,
)
from src.cleaning import (
    read_csv_upload,
    suggest_date_columns,
    suggest_numeric_columns,
    clean_dataframe,
    detect_frequency,
    add_time_features,
    CleaningReport,
    FrequencyInfo,
)
from src.diagnostics import (
    compute_summary_stats,
    compute_acf_pacf,
    compute_decomposition,
    compute_rolling_stats,
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
    render_interpretation,
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
    "Monthly Retail Sales (single)": _DATA_DIR / "demo_single.csv",
    "Quarterly Revenue by Region (wide)": _DATA_DIR / "demo_multi_wide.csv",
    "Daily Stock Prices – 20 Tickers (long)": _DATA_DIR / "demo_multi_long.csv",
}

_CHART_TYPES = [
    "Line with Markers",
    "Line – Colored Markers",
    "Seasonal Plot",
    "Seasonal Sub-series",
    "ACF / PACF",
    "Decomposition",
    "Rolling Mean Overlay",
    "Year-over-Year Change",
    "Lag Plot",
]

_PALETTE_NAMES = ["Set2", "Dark2", "Set1", "Paired", "Pastel1", "Pastel2", "Accent"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _df_hash(df: pd.DataFrame) -> str:
    """Fast hash of a DataFrame for cache-key / change-detection."""
    return hashlib.md5(
        pd.util.hash_pandas_object(df).values.tobytes()
    ).hexdigest()


def _load_demo(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _clean_pipeline(_raw_hash, raw_df, date_col, y_cols, dup_action, missing_action):
    cleaned, report = clean_dataframe(raw_df, date_col, list(y_cols),
                                       dup_action=dup_action,
                                       missing_action=missing_action)
    freq = detect_frequency(cleaned, date_col)
    cleaned = add_time_features(cleaned, date_col)
    return cleaned, report, freq


@st.fragment
def _querychat_fragment(cleaned_df, date_col, y_cols, freq_label):
    current_hash = _df_hash(cleaned_df) + str(y_cols)
    if st.session_state.qc_hash != current_hash:
        st.session_state.qc = create_querychat(
            cleaned_df, name="uploaded_data",
            date_col=date_col, y_cols=y_cols,
            freq_label=freq_label,
        )
        st.session_state.qc_hash = current_hash
    st.session_state.qc.ui()


def _render_cleaning_report(report: CleaningReport) -> None:
    """Show a data-quality card."""
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows before", f"{report.rows_before:,}")
    c2.metric("Rows after", f"{report.rows_after:,}")
    c3.metric("Duplicates found", f"{report.duplicates_found:,}")

    if report.missing_before:
        with st.expander("Missing values"):
            cols = list(report.missing_before.keys())
            mc1, mc2 = st.columns(2)
            with mc1:
                st.write("**Before cleaning**")
                for c in cols:
                    st.write(f"- {c}: {report.missing_before[c]}")
            with mc2:
                st.write("**After cleaning**")
                for c in cols:
                    st.write(f"- {c}: {report.missing_after.get(c, 0)}")

    if report.parsing_warnings:
        with st.expander("Parsing warnings"):
            for w in report.parsing_warnings:
                st.warning(w)


def _render_summary_stats(stats) -> None:
    """Render SummaryStats as metric cards + expander."""
    row1 = st.columns(4)
    row1[0].metric("Count", f"{stats.count:,}")
    row1[1].metric("Missing", f"{stats.missing_count} ({stats.missing_pct:.1f}%)")
    row1[2].metric("Mean", f"{stats.mean_val:,.2f}")
    row1[3].metric("Std Dev", f"{stats.std_val:,.2f}")

    row2 = st.columns(4)
    row2[0].metric("Min", f"{stats.min_val:,.2f}")
    row2[1].metric("25th %ile", f"{stats.p25:,.2f}")
    row2[2].metric("Median", f"{stats.median_val:,.2f}")
    row2[3].metric("75th %ile / Max", f"{stats.p75:,.2f} / {stats.max_val:,.2f}")

    with st.expander("Trend & Stationarity"):
        tc1, tc2 = st.columns(2)
        tc1.metric(
            "Trend slope (per period)",
            f"{stats.trend_slope:,.4f}" if pd.notna(stats.trend_slope) else "N/A",
            help="Slope from OLS on a numeric index.",
        )
        tc2.metric(
            "Trend p-value",
            f"{stats.trend_pvalue:.4f}" if pd.notna(stats.trend_pvalue) else "N/A",
        )
        ac1, ac2 = st.columns(2)
        ac1.metric(
            "ADF statistic",
            f"{stats.adf_statistic:.4f}" if pd.notna(stats.adf_statistic) else "N/A",
            help="Augmented Dickey-Fuller test statistic.",
        )
        ac2.metric(
            "ADF p-value",
            f"{stats.adf_pvalue:.4f}" if pd.notna(stats.adf_pvalue) else "N/A",
            help="p < 0.05 suggests the series is stationary.",
        )
        st.caption(
            f"Date range: {stats.date_start.date()} to {stats.date_end.date()} "
            f"({stats.date_span_days:,} days)"
        )


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Time Series Visualizer",
    page_icon="\U0001f4c8",
    layout="wide",
)
apply_miami_theme()
style_dict = get_miami_mpl_style()

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key in [
    "raw_df", "cleaned_df", "cleaning_report", "freq_info",
    "date_col", "y_cols", "qc", "qc_hash",
    "_upload_id", "_upload_delim", "cleaned_df_hash",
]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------------------------------------
# Sidebar — Data input
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center; margin-bottom:0.5rem;">
            <span style="font-size:1.6rem; font-weight:800; color:#C41230;">
                Time Series Visualizer
            </span><br>
            <span style="font-size:0.82rem; color:#000;">
                ISA 444 &middot; Miami University
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()
    st.header("Data Input")

    uploaded = st.file_uploader("Upload a CSV file", type=["csv", "tsv", "txt"], key="csv_upload")

    demo_choice = st.selectbox(
        "Or load a demo dataset",
        ["(none)"] + list(_DEMO_FILES.keys()),
        key="demo_select",
    )

    # Load data
    if uploaded is not None:
        file_id = (uploaded.name, uploaded.size)
        if st.session_state.get("_upload_id") != file_id:
            df_raw, delim = read_csv_upload(uploaded)
            st.session_state.raw_df = df_raw
            st.session_state._upload_delim = delim
            st.session_state._upload_id = file_id
        st.caption(f"Detected delimiter: `{repr(st.session_state._upload_delim)}`")
    elif demo_choice != "(none)":
        st.session_state.raw_df = _load_demo(_DEMO_FILES[demo_choice])
    # else: keep whatever was already in session state

    raw_df: pd.DataFrame | None = st.session_state.raw_df

    if raw_df is not None:
        st.divider()
        st.subheader("Column Selection")

        # Auto-suggest
        date_suggestions = suggest_date_columns(raw_df)
        numeric_suggestions = suggest_numeric_columns(raw_df)

        all_cols = list(raw_df.columns)
        default_date_idx = all_cols.index(date_suggestions[0]) if date_suggestions else 0

        date_col = st.selectbox("Date column", all_cols, index=default_date_idx, key="sidebar_date_col")

        remaining = [c for c in all_cols if c != date_col]
        default_y = [c for c in numeric_suggestions if c != date_col]
        y_cols = st.multiselect(
            "Value column(s)",
            remaining,
            default=default_y[:4] if default_y else [],
            key="sidebar_y_cols",
        )

        st.session_state.date_col = date_col
        st.session_state.y_cols = y_cols

        st.divider()
        st.subheader("Cleaning Options")
        dup_action = st.selectbox(
            "Duplicate dates",
            ["keep_last", "keep_first", "drop_all"],
            key="sidebar_dup_action",
        )
        missing_action = st.selectbox(
            "Missing values",
            ["interpolate", "ffill", "drop"],
            key="sidebar_missing_action",
        )

        # Clean
        if y_cols:
            cleaned_df, report, freq_info = _clean_pipeline(
                _df_hash(raw_df), raw_df, date_col, tuple(y_cols),
                dup_action, missing_action,
            )
            st.session_state.cleaned_df = cleaned_df
            st.session_state.cleaning_report = report
            st.session_state.freq_info = freq_info

            st.caption(f"Frequency: **{freq_info.label}** "
                       f"({'regular' if freq_info.is_regular else 'irregular'})")

            # Frequency override
            freq_override = st.text_input(
                "Override frequency label (optional)",
                value="",
                help="e.g. Daily, Weekly, Monthly, Quarterly, Yearly",
                key="sidebar_freq_override",
            )
            if freq_override.strip():
                st.session_state.freq_info = FrequencyInfo(
                    label=freq_override.strip(),
                    median_delta=freq_info.median_delta,
                    is_regular=freq_info.is_regular,
                )

            # ------ QueryChat ------
            if check_querychat_available():
                st.divider()
                st.subheader("QueryChat")
                _querychat_fragment(cleaned_df, date_col, y_cols,
                                     st.session_state.freq_info.label)
            else:
                st.divider()
                st.info(
                    "Set `OPENAI_API_KEY` to enable QueryChat "
                    "(natural-language data filtering)."
                )

        # Reset button
        st.divider()
        if st.button("Reset all"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()

    st.divider()
    st.markdown(
        """
        <div style="text-align:center; padding:0.5rem 0;">
            <span style="font-size:0.75rem; color:#000;">
                Developed by <strong>Fadel M. Megahed</strong><br>
                for <strong>ISA 444</strong> &middot; Miami University<br>
                Version <strong>0.1.0</strong>
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(
        "**Privacy:** All processing is in-memory. "
        "If you click **Interpret Chart with AI**, the chart image is sent to OpenAI — "
        "do not include sensitive data in your charts. "
        "QueryChat protects your privacy by only passing metadata (not your data) to OpenAI."
    )

# ---------------------------------------------------------------------------
# Main area — guard
# ---------------------------------------------------------------------------
cleaned_df: pd.DataFrame | None = st.session_state.cleaned_df
date_col: str | None = st.session_state.date_col
y_cols: list[str] | None = st.session_state.y_cols
freq_info: FrequencyInfo | None = st.session_state.freq_info
report: CleaningReport | None = st.session_state.cleaning_report

if cleaned_df is None or not y_cols:
    st.title("Time Series Visualizer")
    st.write(
        "Upload a CSV or choose a demo dataset from the sidebar to get started."
    )
    st.stop()

# If QueryChat is active, use its filtered df
if st.session_state.qc is not None:
    working_df = get_filtered_pandas_df(st.session_state.qc)
    if working_df.empty:
        working_df = cleaned_df
else:
    working_df = cleaned_df

# Data quality report
if report is not None:
    with st.expander("Data Quality Report", expanded=False):
        _render_cleaning_report(report)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_single, tab_few, tab_many = st.tabs([
    "Single Series",
    "Few Series (Panel)",
    "Many Series (Spaghetti)",
])

# ===================================================================
# Tab A — Single Series
# ===================================================================
with tab_single:
    if len(y_cols) == 1:
        active_y = y_cols[0]
    else:
        active_y = st.selectbox("Select value column", y_cols, key="tab_a_y")

    # ---- Date range filter ------------------------------------------------
    dr_mode = st.radio(
        "Date range",
        ["All", "Last N years", "Custom"],
        horizontal=True,
        key="dr_mode",
    )
    df_plot = working_df.copy()
    if dr_mode == "Last N years":
        n_years = st.slider("Years", 1, 20, 5, key="dr_n")
        cutoff = df_plot[date_col].max() - pd.DateOffset(years=n_years)
        df_plot = df_plot[df_plot[date_col] >= cutoff]
    elif dr_mode == "Custom":
        d_min = df_plot[date_col].min().date()
        d_max = df_plot[date_col].max().date()
        sel = st.slider("Date range", d_min, d_max, (d_min, d_max), key="dr_custom")
        df_plot = df_plot[
            (df_plot[date_col].dt.date >= sel[0])
            & (df_plot[date_col].dt.date <= sel[1])
        ]

    if df_plot.empty:
        st.warning("No data in selected range.")
        st.stop()

    # ---- Chart controls ---------------------------------------------------
    col_chart, col_opts = st.columns([2, 1])
    with col_opts:
        chart_type = st.selectbox("Chart type", _CHART_TYPES, key="chart_type_a")

        palette_name = st.selectbox("Color palette", _PALETTE_NAMES, key="pal_a")
        n_colors = max(12, len(y_cols))
        palette_colors = get_palette_colors(palette_name, n_colors)
        swatch_fig = render_palette_preview(palette_colors[:8])
        st.pyplot(swatch_fig)

        # Color-by control (for colored markers chart)
        color_by = None
        if chart_type == "Line – Colored Markers":
            if "month" in working_df.columns:
                color_by = st.selectbox(
                    "Color by",
                    ["month", "quarter", "year", "day_of_week"],
                    key="color_by_a",
                )
            else:
                other_cols = [
                    c for c in working_df.columns
                    if c not in (date_col, active_y)
                ][:5]
                if other_cols:
                    color_by = st.selectbox(
                        "Color by", other_cols, key="color_by_a",
                    )

        # Chart-specific controls
        period_label = "month"
        window_size = 12
        lag_val = 1
        decomp_model = "additive"

        if chart_type in ("Seasonal Plot", "Seasonal Sub-series"):
            period_label = st.selectbox("Period", ["month", "quarter"], key="period_a")

        if chart_type == "Rolling Mean Overlay":
            window_size = st.slider("Window", 2, 52, 12, key="window_a")

        if chart_type == "Lag Plot":
            lag_val = st.slider("Lag", 1, 52, 1, key="lag_a")

        if chart_type == "Decomposition":
            decomp_model = st.selectbox("Model", ["additive", "multiplicative"], key="decomp_a")

    # ---- Render chart -----------------------------------------------------
    with col_chart:
        fig = None
        try:
            if chart_type == "Line with Markers":
                fig = plot_line_with_markers(
                    df_plot, date_col, active_y,
                    title=f"{active_y} over Time",
                    style_dict=style_dict, palette_colors=palette_colors,
                )

            elif chart_type == "Line – Colored Markers" and color_by is not None:
                fig = plot_line_colored_markers(
                    df_plot, date_col, active_y,
                    color_by=color_by, palette_colors=palette_colors,
                    title=f"{active_y} colored by {color_by}",
                    style_dict=style_dict,
                )

            elif chart_type == "Seasonal Plot":
                fig = plot_seasonal(
                    df_plot, date_col, active_y,
                    period=period_label,
                    palette_name_colors=palette_colors,
                    title=f"Seasonal Plot – {active_y}",
                    style_dict=style_dict,
                )

            elif chart_type == "Seasonal Sub-series":
                fig = plot_seasonal_subseries(
                    df_plot, date_col, active_y,
                    period=period_label,
                    title=f"Seasonal Sub-series – {active_y}",
                    style_dict=style_dict, palette_colors=palette_colors,
                )

            elif chart_type == "ACF / PACF":
                series = df_plot[active_y].dropna()
                acf_vals, acf_ci, pacf_vals, pacf_ci = compute_acf_pacf(series)
                fig = plot_acf_pacf(
                    acf_vals, acf_ci, pacf_vals, pacf_ci,
                    title=f"ACF / PACF – {active_y}",
                    style_dict=style_dict,
                )

            elif chart_type == "Decomposition":
                period_int = None
                if freq_info and freq_info.label == "Monthly":
                    period_int = 12
                elif freq_info and freq_info.label == "Quarterly":
                    period_int = 4
                elif freq_info and freq_info.label == "Weekly":
                    period_int = 52
                elif freq_info and freq_info.label == "Daily":
                    period_int = 365

                result = compute_decomposition(
                    df_plot, date_col, active_y,
                    model=decomp_model, period=period_int,
                )
                fig = plot_decomposition(
                    result,
                    title=f"Decomposition – {active_y} ({decomp_model})",
                    style_dict=style_dict,
                )

            elif chart_type == "Rolling Mean Overlay":
                fig = plot_rolling_overlay(
                    df_plot, date_col, active_y,
                    window=window_size,
                    title=f"Rolling {window_size}-pt Mean – {active_y}",
                    style_dict=style_dict, palette_colors=palette_colors,
                )

            elif chart_type == "Year-over-Year Change":
                yoy_result = compute_yoy_change(df_plot, date_col, active_y)
                yoy_df = pd.DataFrame({
                    "date": yoy_result[date_col],
                    "abs_change": yoy_result["yoy_abs_change"],
                    "pct_change": yoy_result["yoy_pct_change"],
                }).dropna()
                fig = plot_yoy_change(
                    df_plot, date_col, active_y, yoy_df,
                    title=f"Year-over-Year Change – {active_y}",
                    style_dict=style_dict,
                )

            elif chart_type == "Lag Plot":
                fig = plot_lag(
                    df_plot[active_y],
                    lag=lag_val,
                    title=f"Lag-{lag_val} Plot – {active_y}",
                    style_dict=style_dict,
                )

        except Exception as exc:
            st.error(f"Chart error: {exc}")

        if fig is not None:
            st.pyplot(fig)

    # ---- Summary stats expander -------------------------------------------
    with st.expander("Summary Statistics", expanded=False):
        stats = compute_summary_stats(df_plot, date_col, active_y)
        _render_summary_stats(stats)

    # ---- AI Interpretation ------------------------------------------------
    with st.expander("AI Chart Interpretation", expanded=False):
        st.caption(
            "The chart image (PNG) is sent to OpenAI for interpretation. "
            "Do not include sensitive data in your charts."
        )
        if not check_api_key_available():
            st.warning("Set `OPENAI_API_KEY` to enable AI interpretation.")
        elif fig is not None:
            if st.button("Interpret Chart with AI", key="interpret_a"):
                with st.spinner("Analyzing chart..."):
                    png = fig_to_png_bytes(fig)
                    date_range_str = (
                        f"{df_plot[date_col].min().date()} to "
                        f"{df_plot[date_col].max().date()}"
                    )
                    metadata = {
                        "chart_type": chart_type,
                        "frequency_label": freq_info.label if freq_info else "Unknown",
                        "date_range": date_range_str,
                        "y_column": active_y,
                    }
                    interp = interpret_chart(png, metadata)
                    render_interpretation(interp)

# ===================================================================
# Tab B — Few Series (Panel)
# ===================================================================
with tab_few:
    if len(y_cols) < 2:
        st.info("Select 2+ value columns in the sidebar to use panel plots.")
    else:
        st.subheader("Panel Plot (Small Multiples)")

        panel_cols = st.multiselect(
            "Columns to plot",
            y_cols,
            default=y_cols[:4],
            key="panel_cols",
        )

        if panel_cols:
            pc1, pc2 = st.columns(2)
            with pc1:
                panel_chart = st.selectbox(
                    "Chart type", ["line", "bar"], key="panel_chart"
                )
            with pc2:
                shared_y = st.checkbox("Shared Y axis", value=True, key="panel_shared")

            palette_name_b = st.selectbox("Color palette", _PALETTE_NAMES, key="pal_b")
            palette_b = get_palette_colors(palette_name_b, len(panel_cols))

            try:
                fig_panel = plot_panel(
                    working_df, date_col, panel_cols,
                    chart_type=panel_chart,
                    shared_y=shared_y,
                    title="Panel Comparison",
                    style_dict=style_dict,
                    palette_colors=palette_b,
                )
                st.pyplot(fig_panel)
            except Exception as exc:
                st.error(f"Panel chart error: {exc}")

            # Per-series summary table
            with st.expander("Per-series Summary", expanded=False):
                summary_df = compute_multi_series_summary(
                    working_df, date_col, panel_cols,
                )
                st.dataframe(
                    summary_df.style.format({
                        "mean": "{:,.2f}",
                        "std": "{:,.2f}",
                        "min": "{:,.2f}",
                        "max": "{:,.2f}",
                        "trend_slope": "{:,.4f}",
                        "adf_pvalue": "{:.4f}",
                    }),
                    width="stretch",
                )

# ===================================================================
# Tab C — Many Series (Spaghetti)
# ===================================================================
with tab_many:
    if len(y_cols) < 2:
        st.info("Select 2+ value columns in the sidebar to use spaghetti plots.")
    else:
        st.subheader("Spaghetti Plot")

        spag_cols = st.multiselect(
            "Columns to include",
            y_cols,
            default=y_cols,
            key="spag_cols",
        )

        if spag_cols:
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                alpha_val = st.slider("Alpha", 0.05, 1.0, 0.15, 0.05, key="spag_alpha")
            with sc2:
                top_n = st.number_input("Highlight top N", 0, len(spag_cols), 0, key="spag_topn")
                top_n = top_n if top_n > 0 else None
            with sc3:
                highlight = st.selectbox(
                    "Highlight series",
                    ["(none)"] + spag_cols,
                    key="spag_highlight",
                )
                highlight_col = highlight if highlight != "(none)" else None

            show_median = st.checkbox("Show Median + IQR band", value=False, key="spag_median")

            palette_name_c = st.selectbox("Color palette", _PALETTE_NAMES, key="pal_c")
            palette_c = get_palette_colors(palette_name_c, len(spag_cols))

            try:
                fig_spag = plot_spaghetti(
                    working_df, date_col, spag_cols,
                    alpha=alpha_val,
                    highlight_col=highlight_col,
                    top_n=top_n,
                    show_median_band=show_median,
                    title="Spaghetti Plot",
                    style_dict=style_dict,
                    palette_colors=palette_c,
                )
                st.pyplot(fig_spag)
            except Exception as exc:
                st.error(f"Spaghetti chart error: {exc}")
