"""Shared matplotlib style of the documentation figures."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# one colour per implementation (fixed order; validated for colour-vision deficiencies)
ORIGINAL = "#2a78d6"   # blue
SNOWAGG = "#eb6834"    # orange
SNOWAGG_MT = "#1baf7a"  # aqua: snowagg with 8 threads

# column density of the projected elements: one hue, light to dark
DENSITY = LinearSegmentedColormap.from_list(
    "density", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

IMG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "img")


def apply():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": "sans-serif", "font.size": 9,
        "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 9,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.axisbelow": True,
        "lines.linewidth": 2, "lines.solid_capstyle": "round",
        "legend.frameon": False, "legend.fontsize": 9,
    })


def plain_log(axis, subs=(1, 2, 5)):
    """Log axis with ticks at subs x 10^k, written as plain numbers."""
    axis.set_major_locator(LogLocator(base=10, subs=subs))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    axis.set_minor_formatter(NullFormatter())


def save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print("wrote", path)
