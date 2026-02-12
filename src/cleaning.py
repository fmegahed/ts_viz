"""
CSV ingest and auto-clean pipeline for time-series data.

Provides delimiter detection, date/numeric column suggestion,
numeric cleaning (currency, commas, percentages, parenthesised negatives),
duplicate and missing-value handling, frequency detection, and
calendar-feature extraction.
"""

import csv
import io
import re
import warnings
from dataclasses import dataclass, field
from datetime import timedelta

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class CleaningReport:
    """Summary produced by :func:`clean_dataframe`."""

    rows_before: int = 0
    rows_after: int = 0
    duplicates_found: int = 0
    duplicates_action: str = ""
    missing_before: dict = field(default_factory=dict)
    missing_after: dict = field(default_factory=dict)
    parsing_warnings: list = field(default_factory=list)


@dataclass
class FrequencyInfo:
    """Result of :func:`detect_frequency`."""

    label: str = "Unknown"
    median_delta: timedelta = timedelta(0)
    is_regular: bool = False


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------

def detect_delimiter(file_bytes: bytes) -> str:
    """Return the most likely CSV delimiter for *file_bytes*.

    Uses :class:`csv.Sniffer` on the first 8 KB of text.  Falls back to a
    comma if the sniffer cannot decide.
    """
    try:
        sample = file_bytes[:8192].decode("utf-8", errors="replace")
        dialect = csv.Sniffer().sniff(sample)
        return dialect.delimiter
    except csv.Error:
        return ","


# ---------------------------------------------------------------------------
# Reading uploads
# ---------------------------------------------------------------------------

def read_csv_upload(uploaded_file) -> tuple[pd.DataFrame, str]:
    """Read a Streamlit ``UploadedFile`` and return ``(df, delimiter)``.

    The file position is rewound so the object can be re-read later if
    needed.
    """
    raw = uploaded_file.getvalue()
    delimiter = detect_delimiter(raw)
    text = raw.decode("utf-8", errors="replace")
    df = pd.read_csv(io.StringIO(text), sep=delimiter)
    # Rewind in case the caller wants to read again
    uploaded_file.seek(0)
    return df, delimiter


# ---------------------------------------------------------------------------
# Column suggestion helpers
# ---------------------------------------------------------------------------

_DATE_NAME_TOKENS = re.compile(r"(date|time|year|month|day|period)", re.IGNORECASE)


def suggest_date_columns(df: pd.DataFrame) -> list[str]:
    """Return column names that are likely to contain date/time values.

    Checks are applied in order:

    1. Column already has a datetime dtype.
    2. :func:`pd.to_datetime` succeeds on the first non-null values.
    3. The column *name* contains a date-related keyword.
    """
    candidates: list[str] = []

    for col in df.columns:
        # 1. Already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            if col not in candidates:
                candidates.append(col)
            continue

        # 2. Parseable as datetime (check up to first 5 non-null values)
        sample = df[col].dropna().head(5)
        if not sample.empty:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    pd.to_datetime(sample)
                if col not in candidates:
                    candidates.append(col)
                continue
            except (ValueError, TypeError, OverflowError):
                pass

        # 3. Column name heuristic
        if _DATE_NAME_TOKENS.search(str(col)):
            if col not in candidates:
                candidates.append(col)

    return candidates


def suggest_numeric_columns(df: pd.DataFrame) -> list[str]:
    """Return columns that are numeric or could be cleaned to numeric.

    A non-numeric column qualifies if, after stripping common formatting
    characters (currency symbols, commas, ``%``, parentheses), at least half
    of its non-null values can be converted to a number.
    """
    candidates: list[str] = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            candidates.append(col)
            continue

        # Attempt lightweight cleaning on a sample
        sample = df[col].dropna().head(50).astype(str)
        if sample.empty:
            continue

        cleaned = (
            sample
            .str.replace(r"[\$\u20ac\u00a3,% ]", "", regex=True)
            .str.replace(r"^\((.+)\)$", r"-\1", regex=True)
        )
        numeric = pd.to_numeric(cleaned, errors="coerce")
        if numeric.notna().sum() >= max(1, len(sample) * 0.5):
            candidates.append(col)

    return candidates


# ---------------------------------------------------------------------------
# Numeric cleaning
# ---------------------------------------------------------------------------

def clean_numeric_series(series: pd.Series) -> pd.Series:
    """Clean a series into proper numeric values.

    Handles:
    * Currency symbols: ``$``, ``EUR`` (U+20AC), ``GBP`` (U+00A3)
    * Thousands separators (commas)
    * Percentage signs
    * Parenthesised negatives, e.g. ``(123)`` becomes ``-123``
    """
    s = series.astype(str)

    # Strip currency symbols, commas, percent signs, and whitespace
    s = s.str.replace(r"[\$\u20ac\u00a3,%\s]", "", regex=True)

    # Convert accounting-style negatives: (123.45) -> -123.45
    s = s.str.replace(r"^\((.+)\)$", r"-\1", regex=True)

    return pd.to_numeric(s, errors="coerce")


# ---------------------------------------------------------------------------
# Full cleaning pipeline
# ---------------------------------------------------------------------------

def clean_dataframe(
    df: pd.DataFrame,
    date_col: str,
    y_cols: list[str],
    dup_action: str = "keep_last",
    missing_action: str = "interpolate",
) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the full cleaning pipeline and return ``(cleaned_df, report)``.

    Parameters
    ----------
    df:
        Input dataframe (will not be mutated).
    date_col:
        Name of the column to parse as dates.
    y_cols:
        Names of the value columns to clean to numeric.
    dup_action:
        How to handle duplicate dates: ``"keep_first"``, ``"keep_last"``,
        or ``"drop_all"``.
    missing_action:
        How to handle missing values in *y_cols*: ``"interpolate"``,
        ``"ffill"``, or ``"drop"``.
    """
    df = df.copy()
    report = CleaningReport()
    report.rows_before = len(df)

    # --- Parse date column ------------------------------------------------
    try:
        df[date_col] = pd.to_datetime(df[date_col])
    except Exception as exc:  # noqa: BLE001
        report.parsing_warnings.append(
            f"Date parsing issue in column '{date_col}': {exc}"
        )
        # Coerce individually so partial failures become NaT
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

    nat_count = int(df[date_col].isna().sum())
    if nat_count > 0:
        report.parsing_warnings.append(
            f"{nat_count} value(s) in '{date_col}' could not be parsed as dates."
        )
        df = df.dropna(subset=[date_col])

    # --- Clean numeric columns --------------------------------------------
    for col in y_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = clean_numeric_series(df[col])

    # Record missing values *before* imputation
    report.missing_before = {
        col: int(df[col].isna().sum()) for col in y_cols
    }

    # --- Handle duplicates on date column ---------------------------------
    dup_mask = df.duplicated(subset=[date_col], keep=False)
    report.duplicates_found = int(dup_mask.sum())
    report.duplicates_action = dup_action

    if report.duplicates_found > 0:
        if dup_action == "keep_first":
            df = df.drop_duplicates(subset=[date_col], keep="first")
        elif dup_action == "keep_last":
            df = df.drop_duplicates(subset=[date_col], keep="last")
        elif dup_action == "drop_all":
            df = df[~dup_mask]

    # --- Sort by date -----------------------------------------------------
    df = df.sort_values(date_col).reset_index(drop=True)

    # --- Handle missing values --------------------------------------------
    if missing_action == "interpolate":
        df[y_cols] = df[y_cols].interpolate(method="linear", limit_direction="both")
    elif missing_action == "ffill":
        df[y_cols] = df[y_cols].ffill().bfill()
    elif missing_action == "drop":
        df = df.dropna(subset=y_cols)

    report.missing_after = {
        col: int(df[col].isna().sum()) for col in y_cols
    }
    report.rows_after = len(df)

    return df, report


# ---------------------------------------------------------------------------
# Frequency detection
# ---------------------------------------------------------------------------

def detect_frequency(df: pd.DataFrame, date_col: str) -> FrequencyInfo:
    """Classify the time-series frequency based on median time delta.

    Returns a :class:`FrequencyInfo` with a human-readable label, the
    computed median delta, and whether the series is *regular* (the
    standard deviation of deltas is less than 20 % of the median).
    """
    dates = df[date_col].dropna().sort_values()
    if len(dates) < 2:
        return FrequencyInfo(label="Unknown", median_delta=timedelta(0), is_regular=False)

    deltas = dates.diff().dropna()
    median_delta = deltas.median()

    # Regularity: std < 20% of median
    std_delta = deltas.std()
    is_regular = bool(std_delta <= median_delta * 0.2) if median_delta > timedelta(0) else False

    # Classify by median days
    days = median_delta.days

    if days <= 1:
        label = "Daily"
    elif 5 <= days <= 9:
        label = "Weekly"
    elif 25 <= days <= 35:
        label = "Monthly"
    elif 85 <= days <= 100:
        label = "Quarterly"
    elif 350 <= days <= 380:
        label = "Yearly"
    else:
        label = "Irregular"

    return FrequencyInfo(label=label, median_delta=median_delta, is_regular=is_regular)


# ---------------------------------------------------------------------------
# Calendar feature extraction
# ---------------------------------------------------------------------------

def detect_long_format(
    df: pd.DataFrame,
    date_col: str,
) -> tuple[bool, str | None, str | None]:
    """Heuristic: detect whether *df* is in long (stacked) format.

    Returns ``(is_long, group_col, value_col)``.

    A DataFrame is flagged as *long* when the date column contains
    duplicate values **and** there is at least one string/object column
    among the remaining columns (the likely group identifier).
    """
    if date_col not in df.columns:
        return False, None, None

    dates = df[date_col]
    if dates.nunique() >= len(dates):
        # Every date is unique → wide format
        return False, None, None

    remaining = [c for c in df.columns if c != date_col]

    # Find first string/object column → candidate group column
    group_col: str | None = None
    for c in remaining:
        if df[c].dtype == object or pd.api.types.is_string_dtype(df[c]):
            group_col = c
            break

    if group_col is None:
        return False, None, None

    # Find first numeric column (excluding the group column) → candidate value
    value_col: str | None = None
    for c in remaining:
        if c == group_col:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            value_col = c
            break

    if value_col is None:
        return False, None, None

    return True, group_col, value_col


def pivot_long_to_wide(
    df: pd.DataFrame,
    date_col: str,
    group_col: str,
    value_col: str,
) -> pd.DataFrame:
    """Pivot a long-format DataFrame to wide format.

    Parameters
    ----------
    df:
        Long-format dataframe.
    date_col:
        Column with date values (becomes the index/row key).
    group_col:
        Column whose unique values become the new column headers.
    value_col:
        Column with the numeric values to spread.

    Returns
    -------
    pd.DataFrame
        Wide dataframe with *date_col* as a regular column and one
        column per unique value in *group_col*.
    """
    wide = df.pivot_table(
        index=date_col,
        columns=group_col,
        values=value_col,
        aggfunc="first",
    )
    # Flatten MultiIndex column names to plain strings
    wide.columns = [str(c) for c in wide.columns]
    wide = wide.reset_index()
    return wide


# ---------------------------------------------------------------------------
# Calendar feature extraction
# ---------------------------------------------------------------------------

def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add calendar columns derived from *date_col*.

    New columns: ``year``, ``quarter``, ``month``, ``day_of_week``.
    The dataframe is returned (not copied) with new columns appended.
    """
    dt = df[date_col].dt
    df["year"] = dt.year
    df["quarter"] = dt.quarter
    df["month"] = dt.month
    df["day_of_week"] = dt.dayofweek
    return df
