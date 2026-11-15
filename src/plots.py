"""Shared plotting helpers. Figures are saved under results/figures/."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIGURES_DIR = Path(__file__).resolve().parents[1] / "results" / "figures"
TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save `fig` as results/figures/<name>.png and close it. Returns the path."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
