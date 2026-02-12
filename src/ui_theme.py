"""
ui_theme.py
-----------
Miami University branded theme and styling utilities.

Provides:
    - Gradio theme subclass (MiamiTheme) with Miami branding
    - Custom CSS string for elements beyond theme control
    - Matplotlib rcParams styled with Miami branding
    - ColorBrewer palette loading via palettable with graceful fallback
    - Color-swatch preview figure generation
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Optional

import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import matplotlib.figure
import matplotlib.pyplot as plt

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
# Gradio theme
# ---------------------------------------------------------------------------

_miami_red_palette = colors.Color(
    c50="#fff5f6",
    c100="#ffe0e4",
    c200="#ffc7ce",
    c300="#ffa3ad",
    c400="#ff6b7d",
    c500="#C41230",
    c600="#a30f27",
    c700="#850c1f",
    c800="#6b0a19",
    c900="#520714",
    c950="#3d0510",
    name="miami_red",
)


class MiamiTheme(Base):
    """Gradio theme subclass with Miami University branding."""

    def __init__(self, **kwargs):
        super().__init__(
            primary_hue=_miami_red_palette,
            secondary_hue=colors.gray,
            neutral_hue=colors.gray,
            spacing_size=sizes.spacing_md,
            radius_size=sizes.radius_sm,
            text_size=sizes.text_md,
            font=(
                fonts.GoogleFont("Source Sans Pro"),
                fonts.Font("ui-sans-serif"),
                fonts.Font("system-ui"),
                fonts.Font("sans-serif"),
            ),
            font_mono=(
                fonts.Font("ui-monospace"),
                fonts.Font("SFMono-Regular"),
                fonts.Font("monospace"),
            ),
            **kwargs,
        )
        super().set(
            # Buttons
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_700",
            button_primary_text_color="white",
            button_primary_border_color="*primary_500",
            # Block titles
            block_title_text_weight="600",
            block_title_text_color="*primary_500",
            # Body
            body_text_color="*neutral_900",
            # Sidebar accent
            block_border_width="1px",
            block_border_color="*neutral_200",
            # Checkbox / Radio
            checkbox_background_color_selected="*primary_500",
            checkbox_border_color_selected="*primary_500",
        )


def get_miami_css() -> str:
    """Return custom CSS for elements that ``gr.themes.Base`` cannot control.

    This string is passed to ``gr.Blocks(css=...)`` alongside the
    :class:`MiamiTheme`.
    """
    return f"""
    /* ---- Sidebar header accent ---- */
    .sidebar > .panel {{
        border-top: 4px solid {MIAMI_RED} !important;
    }}

    /* ---- Developer card ---- */
    .dev-card {{
        padding: 0;
        background: transparent;
    }}
    .dev-row {{
        display: flex;
        gap: 0.5rem;
        align-items: flex-start;
    }}
    .dev-avatar {{
        width: 28px;
        height: 28px;
        min-width: 28px;
        fill: {_BLACK};
    }}
    .dev-name {{
        font-weight: 600;
        color: {_BLACK};
        font-size: 0.82rem;
        line-height: 1.3;
    }}
    .dev-role {{
        font-size: 0.7rem;
        color: #6c757d;
        line-height: 1.3;
    }}
    .dev-links {{
        display: flex;
        gap: 0.3rem;
        flex-wrap: wrap;
        margin-top: 0.35rem;
    }}
    .dev-link,
    .dev-link:visited,
    .dev-link:link {{
        display: inline-flex;
        align-items: center;
        gap: 0.2rem;
        padding: 0.15rem 0.4rem;
        border: 1px solid {MIAMI_RED};
        border-radius: 4px;
        font-size: 0.65rem;
        color: {MIAMI_RED} !important;
        text-decoration: none;
        background: {_WHITE};
        line-height: 1.4;
        white-space: nowrap;
    }}
    .dev-link svg {{
        width: 11px;
        height: 11px;
        fill: {MIAMI_RED};
    }}
    .dev-link:hover {{
        background-color: {MIAMI_RED};
        color: {_WHITE} !important;
    }}
    .dev-link:hover svg {{
        fill: {_WHITE};
    }}

    /* ---- Metric-like stat cards ---- */
    .stat-card {{
        background-color: {_LIGHT_GRAY};
        box-shadow: inset 4px 0 0 0 {MIAMI_RED};
        border-radius: 6px;
        padding: 0.6rem 0.75rem 0.6rem 1rem;
    }}
    .stat-card .stat-label {{
        color: {_BLACK};
        font-size: 0.78rem;
    }}
    .stat-card .stat-value {{
        color: {_BLACK};
        font-weight: 700;
        font-size: 0.95rem;
    }}

    /* ---- Step cards on welcome screen ---- */
    .step-card {{
        background: {_LIGHT_GRAY};
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid {MIAMI_RED};
        height: 100%;
    }}
    .step-card .step-number {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {MIAMI_RED};
    }}
    .step-card .step-title {{
        font-weight: 600;
        margin: 0.3rem 0 0.2rem;
    }}
    .step-card .step-desc {{
        font-size: 0.82rem;
        color: #444;
    }}

    /* ---- App title in sidebar ---- */
    .app-title {{
        text-align: center;
        margin-bottom: 0.5rem;
    }}
    .app-title .title-text {{
        font-size: 1.6rem;
        font-weight: 800;
        color: {MIAMI_RED};
    }}
    .app-title .subtitle-text {{
        font-size: 0.82rem;
        color: {_BLACK};
    }}
    """


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
        A Figure instance ready to be passed to ``gr.Plot`` or saved.
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
    plt.close(fig)  # prevent display in non-Gradio contexts
    return fig
