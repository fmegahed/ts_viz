"""
QueryChat initialization and filtered DataFrame helpers.

Provides convenience wrappers around the ``querychat`` library for
natural-language filtering of time-series DataFrames inside a Streamlit
app.  All functions degrade gracefully when the package or an API key
is unavailable.
"""

from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import streamlit as st

try:
    from querychat.streamlit import QueryChat as _QueryChat

    _QUERYCHAT_AVAILABLE = True
except ImportError:  # pragma: no cover
    _QUERYCHAT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Availability check
# ---------------------------------------------------------------------------

def check_querychat_available() -> bool:
    """Return ``True`` when both *querychat* is installed and an API key is set.

    QueryChat requires an ``OPENAI_API_KEY`` environment variable.  This
    helper lets callers gate UI elements behind a simple boolean.
    """
    if not _QUERYCHAT_AVAILABLE:
        return False
    return bool(os.environ.get("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# QueryChat factory
# ---------------------------------------------------------------------------

def create_querychat(
    df: pd.DataFrame,
    name: str = "dataset",
    date_col: str = "date",
    y_cols: Optional[List[str]] = None,
    freq_label: str = "",
):
    """Create and return a QueryChat instance bound to *df*.

    Parameters
    ----------
    df:
        The pandas DataFrame to expose to the chat interface.
    name:
        A human-readable name for the dataset (used in the description).
    date_col:
        Name of the date/time column.
    y_cols:
        Names of the value (numeric) columns.  If ``None``, an empty
        list is used in the description.
    freq_label:
        Optional frequency label (e.g. ``"Monthly"``, ``"Daily"``).

    Returns
    -------
    QueryChat instance
        The object returned by ``QueryChat()``.

    Raises
    ------
    RuntimeError
        If querychat is not installed.
    """
    if not _QUERYCHAT_AVAILABLE:
        raise RuntimeError(
            "The 'querychat' package is not installed. "
            "Install it with: pip install 'querychat[streamlit]'"
        )

    if y_cols is None:
        y_cols = []

    value_cols_str = ", ".join(y_cols) if y_cols else "none specified"
    freq_part = f"  Frequency: {freq_label}." if freq_label else ""

    data_description = (
        f"This dataset is named '{name}'.  "
        f"It contains {len(df):,} rows.  "
        f"The date column is '{date_col}'.  "
        f"Value columns: {value_cols_str}."
        f"{freq_part}"
    )

    greeting = (
        f"Hi! I can help you filter and explore the **{name}** dataset.  "
        "Try asking me something like:\n"
        '- "Show only 2023 data"\n'
        '- "Filter where sales > 60000"\n'
        '- "Show rows from January to March"'
    )

    qc = _QueryChat(
        data_source=df,
        table_name=name.replace(" ", "_"),
        client="openai/gpt-5.2-2025-12-11",
        data_description=data_description,
        greeting=greeting,
    )

    return qc


# ---------------------------------------------------------------------------
# Filtered DataFrame extraction
# ---------------------------------------------------------------------------

def get_filtered_pandas_df(qc) -> pd.DataFrame:
    """Extract the currently filtered DataFrame from a QueryChat instance.

    The underlying ``qc.df()`` may return a *narwhals* DataFrame rather
    than a pandas one.  This helper transparently converts when needed
    and falls back to the original frame on any error.

    Parameters
    ----------
    qc:
        A QueryChat instance previously created via :func:`create_querychat`.

    Returns
    -------
    pd.DataFrame
        The filtered data as a pandas DataFrame.
    """
    try:
        result = qc.df()

        # narwhals (or polars) DataFrames expose .to_pandas()
        if hasattr(result, "to_pandas"):
            return result.to_pandas()

        # Already a pandas DataFrame
        if isinstance(result, pd.DataFrame):
            return result

        # Unknown type -- attempt conversion as a last resort
        return pd.DataFrame(result)
    except Exception:  # noqa: BLE001
        # If anything goes wrong, surface the unfiltered data so the app
        # can continue to function.
        try:
            raw = qc.df()
            if isinstance(raw, pd.DataFrame):
                return raw
        except Exception:  # noqa: BLE001
            pass

        return pd.DataFrame()
