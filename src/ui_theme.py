"""
ui_theme.py
-----------
Miami University branded theme and styling utilities for Streamlit apps.

Provides:
    - CSS injection for Streamlit components (buttons, sidebar, metrics, cards)
    - Matplotlib rcParams styled with Miami branding
    - ColorBrewer palette loading via palettable with graceful fallback
    - Color-swatch preview figure generation
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import matplotlib.figure
import matplotlib.pyplot as plt
import streamlit as st

# ---------------------------------------------------------------------------
# Brand constants — Miami University (Ohio) official palette
# ---------------------------------------------------------------------------
MIAMI_RED: str = "#C41230"
MIAMI_BLACK: str = "#000000"
MIAMI_WHITE: str = "#FFFFFF"

# Secondary palette tokens used only inside the CSS below.
_WHITE = "#FFFFFF"
_BLACK = "#000000"
_LIGHT_GRAY = "#F5F5F5"
_BORDER_GRAY = "#E0E0E0"
_DARK_TEXT = "#000000"
_HOVER_RED = "#9E0E26"


# ---------------------------------------------------------------------------
# Streamlit CSS injection
# ---------------------------------------------------------------------------
def apply_miami_theme() -> None:
    """Inject Miami-branded CSS into the active Streamlit page.

    Styles affected:
        * Primary buttons  -- Miami Red background with white text
        * Card containers  -- subtle border and rounded corners
        * Sidebar header   -- Miami Red accent bar
        * Metric cards     -- light background with left red accent
    """
    css = f"""
    <style>
        /* ---- Primary buttons ---- */
        .stButton > button[kind="primary"],
        .stButton > button {{
            background-color: {MIAMI_RED};
            color: {_WHITE};
            border: none;
            border-radius: 6px;
            padding: 0.5rem 1.25rem;
            font-weight: 600;
            transition: background-color 0.2s ease;
        }}
        .stButton > button:hover {{
            background-color: {_HOVER_RED};
            color: {_WHITE};
            border: none;
        }}
        .stButton > button:active,
        .stButton > button:focus {{
            background-color: {_HOVER_RED};
            color: {_WHITE};
            box-shadow: none;
        }}

        /* ---- Card borders ---- */
        div[data-testid="stExpander"],
        div[data-testid="stHorizontalBlock"] > div {{
            border: 1px solid {_BORDER_GRAY};
            border-radius: 8px;
            padding: 0.75rem;
        }}

        /* ---- Sidebar header accent ---- */
        section[data-testid="stSidebar"] > div:first-child {{
            border-top: 4px solid {MIAMI_RED};
        }}
        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {{
            color: {MIAMI_RED};
        }}

        /* ---- Metric cards ---- */
        div[data-testid="stMetric"] {{
            background-color: {_LIGHT_GRAY};
            border-left: 4px solid {MIAMI_RED};
            border-radius: 6px;
            padding: 0.75rem 1rem;
        }}
        div[data-testid="stMetric"] label {{
            color: {_BLACK};
            font-size: 0.85rem;
        }}
        div[data-testid="stMetric"] div[data-testid="stMetricValue"] {{
            color: {_BLACK};
            font-weight: 700;
        }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Matplotlib style dictionary
# ---------------------------------------------------------------------------
def get_miami_mpl_style() -> Dict[str, object]:
    """Return a dictionary of matplotlib rcParams for Miami branding.

    Usage::

        import matplotlib as mpl
        mpl.rcParams.update(get_miami_mpl_style())

    Or apply to a single figure::

        with mpl.rc_context(get_miami_mpl_style()):
            fig, ax = plt.subplots()
            ...
    """
    return {
        # Figure
        "figure.facecolor": _WHITE,
        "figure.edgecolor": _WHITE,
        "figure.figsize": (10, 5),
        "figure.dpi": 100,
        # Axes
        "axes.facecolor": _WHITE,
        "axes.edgecolor": _BLACK,
        "axes.labelcolor": _BLACK,
        "axes.titlecolor": MIAMI_RED,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.prop_cycle": plt.cycler(
            color=[MIAMI_RED, _BLACK, "#4E79A7", "#F28E2B", "#76B7B2"]
        ),
        # Grid
        "axes.grid": True,
        "grid.color": _BORDER_GRAY,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.7,
        # Ticks
        "xtick.color": _BLACK,
        "ytick.color": _BLACK,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        # Legend
        "legend.fontsize": 10,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.edgecolor": _BORDER_GRAY,
        # Font
        "font.size": 11,
        "font.family": "sans-serif",
        # Savefig
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }


# ---------------------------------------------------------------------------
# ColorBrewer palette loading
# ---------------------------------------------------------------------------

# Mapping of short friendly names to palettable module paths.
_PALETTE_MAP: Dict[str, str] = {
    "Set1": "colorbrewer.qualitative.Set1",
    "Set2": "colorbrewer.qualitative.Set2",
    "Set3": "colorbrewer.qualitative.Set3",
    "Dark2": "colorbrewer.qualitative.Dark2",
    "Paired": "colorbrewer.qualitative.Paired",
    "Pastel1": "colorbrewer.qualitative.Pastel1",
    "Pastel2": "colorbrewer.qualitative.Pastel2",
    "Accent": "colorbrewer.qualitative.Accent",
    "Tab10": "colorbrewer.qualitative.Set1",  # fallback alias
}

_FALLBACK_COLORS: List[str] = [
    MIAMI_RED,
    MIAMI_BLACK,
    "#4E79A7",
    "#F28E2B",
    "#76B7B2",
    "#E15759",
    "#59A14F",
    "#EDC948",
]


def _resolve_palette(name: str) -> Optional[List[str]]:
    """Dynamically import a palettable ColorBrewer palette by *name*.

    Palettable organises palettes by maximum number of classes, e.g.
    ``colorbrewer.qualitative.Set2_8``.  We find the variant with the
    most colours available so the caller gets the richest palette.
    """
    import importlib

    module_path = _PALETTE_MAP.get(name)
    if module_path is None:
        # Try a direct guess: colorbrewer.qualitative.<Name>
        module_path = f"colorbrewer.qualitative.{name}"

    # palettable stores each size variant as <Name>_<N> inside the module.
    try:
        mod = importlib.import_module(f"palettable.{module_path}")
    except (ImportError, ModuleNotFoundError):
        return None

    # Discover the variant with the most colours.
    best = None
    best_n = 0
    base = name.split(".")[-1] if "." in name else name
    for attr_name in dir(mod):
        if not attr_name.startswith(base + "_"):
            continue
        try:
            suffix = int(attr_name.split("_")[-1])
        except ValueError:
            continue
        if suffix > best_n:
            best_n = suffix
            best = attr_name

    if best is None:
        return None

    palette_obj = getattr(mod, best, None)
    if palette_obj is None:
        return None

    return [
        "#{:02X}{:02X}{:02X}".format(*rgb) for rgb in palette_obj.colors
    ]


def get_palette_colors(name: str = "Set2", n: int = 8) -> List[str]:
    """Load *n* hex colour strings from a ColorBrewer palette.

    Parameters
    ----------
    name:
        Friendly palette name such as ``"Set2"``, ``"Dark2"``, ``"Paired"``.
    n:
        Number of colours required.  If *n* exceeds the palette length the
        colours are cycled.

    Returns
    -------
    list[str]
        List of *n* hex colour strings (e.g. ``["#66C2A5", ...]``).

    Notes
    -----
    If the requested palette cannot be found, a sensible fallback list is
    returned so that calling code never receives an empty list.
    """
    n = max(1, n)
    colors = _resolve_palette(name)
    if colors is None:
        colors = _FALLBACK_COLORS

    # Cycle if the caller needs more colours than the palette provides.
    cycled = list(itertools.islice(itertools.cycle(colors), n))
    return cycled


# ---------------------------------------------------------------------------
# Palette preview swatch
# ---------------------------------------------------------------------------
def render_palette_preview(
    colors: List[str],
    swatch_width: float = 1.0,
    swatch_height: float = 0.4,
) -> matplotlib.figure.Figure:
    """Create a small matplotlib figure showing colour swatches.

    Parameters
    ----------
    colors:
        List of hex colour strings to display.
    swatch_width:
        Width of each individual swatch in inches.
    swatch_height:
        Height of the swatch strip in inches.

    Returns
    -------
    matplotlib.figure.Figure
        A Figure instance ready to be passed to ``st.pyplot()`` or saved.
    """
    n = len(colors)
    fig_width = max(swatch_width * n, 2.0)
    fig, ax = plt.subplots(
        figsize=(fig_width, swatch_height + 0.3), dpi=100
    )

    for i, colour in enumerate(colors):
        ax.add_patch(
            plt.Rectangle(
                (i, 0),
                width=1,
                height=1,
                facecolor=colour,
                edgecolor=_WHITE,
                linewidth=1.5,
            )
        )

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.close(fig)  # prevent display in non-Streamlit contexts
    return fig
