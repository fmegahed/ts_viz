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


@st.fragment
def _data_quality_fragment(report: CleaningReport | None) -> None:
    if report is None:
        return
    with st.expander("Data Quality Report", expanded=False):
        _render_cleaning_report(report)


@st.fragment
def _single_chart_fragment(working_df, date_col, y_cols, freq_info, style_dict):
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
        st.session_state["_single_df_plot"] = None
        st.session_state["_single_fig"] = None
        st.session_state["_single_active_y"] = None
        st.session_state["_single_chart_type"] = None
        return

    # ---- Chart controls ---------------------------------------------------
    col_chart, col_opts = st.columns([2, 1])
    with col_opts:
        chart_type = st.selectbox("Chart type", _CHART_TYPES, key="chart_type_a")

        palette_name = st.selectbox("Color palette", _PALETTE_NAMES, key="pal_a")
        n_colors = max(12, len(y_cols))
        palette_colors = get_palette_colors(palette_name, n_colors)
        swatch_fig = render_palette_preview(palette_colors[:8])
        st.pyplot(swatch_fig, use_container_width=True)

        # Color-by control (for colored markers chart)
        color_by = None
        if chart_type == "Line â€“ Colored Markers":
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

            elif chart_type == "Line â€“ Colored Markers" and color_by is not None:
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
                    title=f"Seasonal Plot â€“ {active_y}",
                    style_dict=style_dict,
                )

            elif chart_type == "Seasonal Sub-series":
                fig = plot_seasonal_subseries(
                    df_plot, date_col, active_y,
                    period=period_label,
                    title=f"Seasonal Sub-series â€“ {active_y}",
                    style_dict=style_dict, palette_colors=palette_colors,
                )

            elif chart_type == "ACF / PACF":
                series = df_plot[active_y].dropna()
                acf_vals, acf_ci, pacf_vals, pacf_ci = compute_acf_pacf(series)
                fig = plot_acf_pacf(
                    acf_vals, acf_ci, pacf_vals, pacf_ci,
                    title=f"ACF / PACF â€“ {active_y}",
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
                    title=f"Decomposition â€“ {active_y} ({decomp_model})",
                    style_dict=style_dict,
                )

            elif chart_type == "Rolling Mean Overlay":
                fig = plot_rolling_overlay(
                    df_plot, date_col, active_y,
                    window=window_size,
                    title=f"Rolling {window_size}-pt Mean â€“ {active_y}",
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
                    title=f"Year-over-Year Change â€“ {active_y}",
                    style_dict=style_dict,
                )

            elif chart_type == "Lag Plot":
                fig = plot_lag(
                    df_plot[active_y],
                    lag=lag_val,
                    title=f"Lag-{lag_val} Plot â€“ {active_y}",
                    style_dict=style_dict,
                )

        except Exception as exc:
            st.error(f"Chart error: {exc}")

        if fig is not None:
            st.pyplot(fig, use_container_width=True)

    st.session_state["_single_df_plot"] = df_plot
    st.session_state["_single_fig"] = fig
    st.session_state["_single_active_y"] = active_y
    st.session_state["_single_chart_type"] = chart_type


@st.fragment
def _single_insights_fragment(freq_info, date_col):
    df_plot = st.session_state.get("_single_df_plot")
    active_y = st.session_state.get("_single_active_y")
    chart_type = st.session_state.get("_single_chart_type")
    fig = st.session_state.get("_single_fig")

    if df_plot is None or active_y is None:
        return

    # ---- Summary stats expander -------------------------------------------
    with st.expander("Summary Statistics", expanded=False):
        stats = compute_summary_stats(df_plot, date_col, active_y)
        _render_summary_stats(stats)

    # ---- AI Interpretation ------------------------------------------------
    _render_ai_interpretation(
        fig, chart_type, freq_info, df_plot, date_col, active_y, "interpret_a",
    )


@st.fragment
def _panel_chart_fragment(working_df, date_col, y_cols, style_dict):
    if len(y_cols) < 2:
        st.info("Select 2+ value columns in the sidebar to use panel plots.")
        st.session_state["_panel_fig"] = None
        return

    st.subheader("Panel Plot (Small Multiples)")

    if "panel_cols" not in st.session_state:
        st.session_state["panel_cols"] = y_cols[:4]
    else:
        st.session_state["panel_cols"] = [
            c for c in st.session_state["panel_cols"] if c in y_cols
        ]
    panel_cols = st.multiselect("Columns to plot", y_cols, key="panel_cols")

    if panel_cols:
        pc1, pc2 = st.columns(2)
        with pc1:
            panel_chart = st.selectbox(
                "Chart type", ["line", "bar"], key="panel_chart"
            )
        with pc2:
            if "panel_shared" not in st.session_state:
                st.session_state["panel_shared"] = True
            shared_y = st.checkbox("Shared Y axis", key="panel_shared")

        palette_name_b = st.selectbox("Color palette", _PALETTE_NAMES, key="pal_b")
        palette_b = get_palette_colors(palette_name_b, len(panel_cols))

        fig_panel = None
        try:
            fig_panel = plot_panel(
                working_df, date_col, panel_cols,
                chart_type=panel_chart,
                shared_y=shared_y,
                title="Panel Comparison",
                style_dict=style_dict,
                palette_colors=palette_b,
            )
            st.pyplot(fig_panel, use_container_width=True)
        except Exception as exc:
            st.error(f"Panel chart error: {exc}")

        st.session_state["_panel_fig"] = fig_panel
    else:
        st.session_state["_panel_fig"] = None


@st.fragment
def _panel_insights_fragment(working_df, date_col, freq_info):
    panel_cols = st.session_state.get("panel_cols") or []
    fig_panel = st.session_state.get("_panel_fig")
    panel_chart = st.session_state.get("panel_chart", "line")

    if not panel_cols:
        return

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
            use_container_width=True,
        )

    # AI Interpretation
    _render_ai_interpretation(
        fig_panel, f"Panel ({panel_chart})", freq_info,
        working_df, date_col, ", ".join(panel_cols), "interpret_b",
    )


@st.fragment
def _spaghetti_chart_fragment(working_df, date_col, y_cols, style_dict):
    if len(y_cols) < 2:
        st.info("Select 2+ value columns in the sidebar to use spaghetti plots.")
        st.session_state["_spag_fig"] = None
        return

    st.subheader("Spaghetti Plot")

    if "spag_cols" not in st.session_state:
        st.session_state["spag_cols"] = list(y_cols)
    else:
        st.session_state["spag_cols"] = [
            c for c in st.session_state["spag_cols"] if c in y_cols
        ]
    spag_cols = st.multiselect("Columns to include", y_cols, key="spag_cols")

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

        show_median = st.checkbox("Show Median + IQR band", key="spag_median")

        palette_name_c = st.selectbox("Color palette", _PALETTE_NAMES, key="pal_c")
        palette_c = get_palette_colors(palette_name_c, len(spag_cols))

        fig_spag = None
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
            st.pyplot(fig_spag, use_container_width=True)
        except Exception as exc:
            st.error(f"Spaghetti chart error: {exc}")

        st.session_state["_spag_fig"] = fig_spag
    else:
        st.session_state["_spag_fig"] = None


@st.fragment
def _spaghetti_insights_fragment(working_df, date_col, freq_info):
    spag_cols = st.session_state.get("spag_cols") or []
    fig_spag = st.session_state.get("_spag_fig")

    if not spag_cols:
        return

    # Per-series summary table
    with st.expander("Per-series Summary", expanded=False):
        spag_summary = compute_multi_series_summary(
            working_df, date_col, spag_cols,
        )
        st.dataframe(
            spag_summary.style.format({
                "mean": "{:,.2f}",
                "std": "{:,.2f}",
                "min": "{:,.2f}",
                "max": "{:,.2f}",
                "trend_slope": "{:,.4f}",
                "adf_pvalue": "{:.4f}",
            }),
            use_container_width=True,
        )

    # AI Interpretation
    _render_ai_interpretation(
        fig_spag, "Spaghetti Plot", freq_info,
        working_df, date_col, ", ".join(spag_cols), "interpret_c",
    )


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
    """Render SummaryStats as metric cards (flat, no nesting)."""
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

    row3 = st.columns(4)
    row3[0].metric(
        "Trend slope",
        f"{stats.trend_slope:,.4f}" if pd.notna(stats.trend_slope) else "N/A",
        help="Slope from OLS on a numeric index.",
    )
    row3[1].metric(
        "Trend p-value",
        f"{stats.trend_pvalue:.4f}" if pd.notna(stats.trend_pvalue) else "N/A",
    )
    row3[2].metric(
        "ADF statistic",
        f"{stats.adf_statistic:.4f}" if pd.notna(stats.adf_statistic) else "N/A",
        help="Augmented Dickey-Fuller test statistic.",
    )
    row3[3].metric(
        "ADF p-value",
        f"{stats.adf_pvalue:.4f}" if pd.notna(stats.adf_pvalue) else "N/A",
        help="p < 0.05 suggests the series is stationary.",
    )
    st.caption(
        f"Date range: {stats.date_start.date()} to {stats.date_end.date()} "
        f"({stats.date_span_days:,} days)"
    )


def _render_ai_interpretation(fig, chart_type_label, freq_info, df_plot,
                               date_col, y_label, button_key):
    """Reusable AI Chart Interpretation block for any tab."""
    with st.expander("AI Chart Interpretation", expanded=False):
        st.caption(
            "The chart image (PNG) is sent to OpenAI for interpretation. "
            "Do not include sensitive data in your charts."
        )
        if not check_api_key_available():
            st.warning("Set `OPENAI_API_KEY` to enable AI interpretation.")
        elif fig is not None:
            if st.button("Interpret Chart with AI", key=button_key):
                with st.spinner("Analyzing chart..."):
                    png = fig_to_png_bytes(fig)
                    date_range_str = (
                        f"{df_plot[date_col].min().date()} to "
                        f"{df_plot[date_col].max().date()}"
                    )
                    metadata = {
                        "chart_type": chart_type_label,
                        "frequency_label": freq_info.label if freq_info else "Unknown",
                        "date_range": date_range_str,
                        "y_column": y_label,
                    }
                    interp = interpret_chart(png, metadata)
                    render_interpretation(interp)


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
    "_upload_id", "_upload_delim", "_clean_key",
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
    st.subheader("Vibe-Coded by:")
    st.markdown(
        """
        <div class="dev-card">
            <div class="dev-row">
                <svg class="dev-avatar" viewBox="0 0 16 16" aria-hidden="true" focusable="false">
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
                    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                        <path d="M0 4a2 2 0 0 1 2-2h12a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V4zm2-1a1 1 0 0 0-1 1v.217l7 4.2 7-4.2V4a1 1 0 0 0-1-1H2zm13 2.383-4.708 2.825L15 11.105zM14.247 12.6 9.114 8.98 8 9.67 6.886 8.98 1.753 12.6A1 1 0 0 0 2 13h12a1 1 0 0 0 .247-.4zM1 11.105l4.708-2.897L1 5.383z"/>
                    </svg>
                    Email
                </a>
                <a class="dev-link" href="https://www.linkedin.com/in/fadel-megahed-289046b4/" target="_blank">
                    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                        <path d="M0 1.146C0 .513.526 0 1.175 0h13.65C15.475 0 16 .513 16 1.146v13.708c0 .633-.525 1.146-1.175 1.146H1.175C.526 16 0 15.487 0 14.854zM4.943 13.5V6H2.542v7.5zM3.743 4.927c.837 0 1.358-.554 1.358-1.248-.015-.709-.521-1.248-1.342-1.248-.821 0-1.358.54-1.358 1.248 0 .694.521 1.248 1.327 1.248zm4.908 8.573V9.359c0-.22.016-.44.08-.598.176-.44.576-.897 1.248-.897.88 0 1.232.676 1.232 1.667v4.0h2.401V9.247c0-2.22-1.184-3.252-2.764-3.252-1.274 0-1.845.7-2.165 1.193h.016V6H6.35c.03.7 0 7.5 0 7.5z"/>
                    </svg>
                    LinkedIn
                </a>
                <a class="dev-link" href="https://miamioh.edu/fsb/directory/?up=/directory/megahefm" target="_blank">
                    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                        <path d="M0 8a8 8 0 1 1 16 0A8 8 0 0 1 0 8m7-7a7 7 0 0 0-2.468.45c.303.393.58.825.82 1.3A5.5 5.5 0 0 1 7 3.5zm2 0v2.5a5.5 5.5 0 0 1 1.648-.75 7 7 0 0 0-.82-1.3A7 7 0 0 0 9 1m3.97 3.06a6.5 6.5 0 0 0-1.71-.9c.21.53.36 1.1.44 1.69h2.21a7 7 0 0 0-.94-.79M15 8a7 7 0 0 0-.33-2h-2.34a6.5 6.5 0 0 1 0 4h2.34c.22-.64.33-1.32.33-2m-1.03 3.94a7 7 0 0 0 .94-.79h-2.21a6.5 6.5 0 0 1-.44 1.69c.62-.22 1.2-.53 1.71-.9M9 15a7 7 0 0 0 1.648-.75c.24-.48.517-.91.82-1.3A7 7 0 0 0 9 15m-2 0v-2.5a5.5 5.5 0 0 1-1.648.75c.24.48.517.91.82 1.3A7 7 0 0 0 7 15M4.03 11.94a6.5 6.5 0 0 0 1.71.9A6.5 6.5 0 0 1 5.3 11.15H3.09c.25.3.58.57.94.79M1 8c0 .68.11 1.36.33 2h2.34a6.5 6.5 0 0 1 0-4H1.33A7 7 0 0 0 1 8m1.03-3.94c.36.37.78.68 1.24.9a6.5 6.5 0 0 1 .44-1.69H2.06a7 7 0 0 0-.03.79"/>
                    </svg>
                    Website
                </a>
                <a class="dev-link" href="https://github.com/fmegahed/" target="_blank">
                    <svg viewBox="0 0 16 16" aria-hidden="true" focusable="false">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8"/>
                    </svg>
                    GitHub
                </a>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v0.2.0 &middot; Last updated Feb 2026")
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
        demo_key = ("demo", demo_choice)
        if st.session_state.get("_upload_id") != demo_key:
            st.session_state.raw_df = _load_demo(_DEMO_FILES[demo_choice])
            st.session_state._upload_id = demo_key
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

        if "sidebar_date_col" not in st.session_state:
            st.session_state["sidebar_date_col"] = all_cols[default_date_idx]
        date_col = st.selectbox("Date column", all_cols, key="sidebar_date_col")

        remaining = [c for c in all_cols if c != date_col]
        default_y = [c for c in numeric_suggestions if c != date_col]
        if "sidebar_y_cols" not in st.session_state:
            st.session_state["sidebar_y_cols"] = default_y[:4] if default_y else []
        y_cols = st.multiselect("Value column(s)", remaining, key="sidebar_y_cols")

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

        # Clean — only recompute and write session_state when inputs change
        if y_cols:
            _key = (date_col, tuple(y_cols), dup_action, missing_action,
                    st.session_state._upload_id)
            if st.session_state.get("_clean_key") != _key:
                cleaned_df, report, freq_info = _clean_pipeline(
                    _df_hash(raw_df), raw_df, date_col, tuple(y_cols),
                    dup_action, missing_action,
                )
                st.session_state.cleaned_df = cleaned_df
                st.session_state.cleaning_report = report
                st.session_state.freq_info = freq_info
                st.session_state._clean_key = _key

            cleaned_df = st.session_state.cleaned_df
            freq_info = st.session_state.freq_info
            st.caption(f"Frequency: **{freq_info.label}** "
                       f"({'regular' if freq_info.is_regular else 'irregular'})")

            # Frequency override
            freq_override = st.text_input(
                "Override frequency label (optional)",
                help="e.g. Daily, Weekly, Monthly, Quarterly, Yearly",
                key="sidebar_freq_override",
            )
            if freq_override.strip():
                st.session_state.freq_info = FrequencyInfo(
                    label=freq_override.strip(),
                    median_delta=freq_info.median_delta,
                    is_regular=freq_info.is_regular,
                )
                freq_info = st.session_state.freq_info

            # ------ QueryChat ------
            if check_querychat_available():
                st.divider()
                st.subheader("QueryChat")
                if cleaned_df is not None:
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
    st.caption("ISA 444 · Miami University · Farmer School of Business")

    st.markdown("")  # spacer

    # --- Getting started steps as visual cards ---
    st.markdown("#### Get Started in 3 Steps")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            '<div style="background:#F5F5F5; border-radius:8px; padding:1rem; '
            'border-left:4px solid #C41230; height:100%;">'
            '<div style="font-size:1.6rem; font-weight:700; color:#C41230;">1</div>'
            '<div style="font-weight:600; margin:0.3rem 0 0.2rem;">Load Data</div>'
            '<div style="font-size:0.82rem; color:#444;">'
            'Upload a CSV from the sidebar or pick one of the built-in demo datasets.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            '<div style="background:#F5F5F5; border-radius:8px; padding:1rem; '
            'border-left:4px solid #C41230; height:100%;">'
            '<div style="font-size:1.6rem; font-weight:700; color:#C41230;">2</div>'
            '<div style="font-weight:600; margin:0.3rem 0 0.2rem;">Pick Columns</div>'
            '<div style="font-size:0.82rem; color:#444;">'
            'Select a date column and one or more numeric value columns. '
            'The app auto-detects sensible defaults.'
            '</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            '<div style="background:#F5F5F5; border-radius:8px; padding:1rem; '
            'border-left:4px solid #C41230; height:100%;">'
            '<div style="font-size:1.6rem; font-weight:700; color:#C41230;">3</div>'
            '<div style="font-weight:600; margin:0.3rem 0 0.2rem;">Explore</div>'
            '<div style="font-size:0.82rem; color:#444;">'
            'Choose from 9+ chart types, view summary statistics, '
            'and get AI-powered chart interpretation.'
            '</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")  # spacer

    # --- Features and privacy ---
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("#### Features")
        st.markdown(
            "| | |\n"
            "|:--|:--|\n"
            "| **Smart Import** | Auto-detect delimiters, dates, and numeric formats |\n"
            "| **9+ Chart Types** | Line, seasonal, ACF/PACF, decomposition, lag, and more |\n"
            "| **Multi-Series** | Panel (small multiples) and spaghetti plots |\n"
            "| **AI Insights** | GPT vision analyzes your charts and returns structured interpretation |\n"
            "| **QueryChat** | Natural-language data filtering powered by DuckDB |"
        )
    with f2:
        st.markdown("#### Good to Know")
        st.info(
            "**Privacy** — All data processing happens in-memory. "
            "No data is stored on disk. Only chart images (never raw data) "
            "are sent to OpenAI when you click *Interpret Chart with AI*.",
            icon="\U0001f512",
        )
        st.info(
            "**Demo Datasets** — Three built-in datasets are available in the sidebar: "
            "monthly retail sales (single series), quarterly revenue by region (wide), "
            "and daily stock prices with 20 tickers (long).",
            icon="\U0001f4ca",
        )

    st.stop()

# If QueryChat is active, use its filtered df
if st.session_state.qc is not None:
    working_df = get_filtered_pandas_df(st.session_state.qc)
    # Fall back if filtered df is empty or missing expected columns
    required = {date_col} | set(y_cols)
    if working_df.empty or not required.issubset(working_df.columns):
        working_df = cleaned_df
else:
    working_df = cleaned_df

# Data quality report
_data_quality_fragment(report)

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
    _single_chart_fragment(working_df, date_col, y_cols, freq_info, style_dict)
    _single_insights_fragment(freq_info, date_col)

# ===================================================================
# Tab B — Few Series (Panel)
# ===================================================================
with tab_few:
    _panel_chart_fragment(working_df, date_col, y_cols, style_dict)
    _panel_insights_fragment(working_df, date_col, freq_info)

# ===================================================================
# Tab C — Many Series (Spaghetti)
# ===================================================================
with tab_many:
    _spaghetti_chart_fragment(working_df, date_col, y_cols, style_dict)
    _spaghetti_insights_fragment(working_df, date_col, freq_info)
