"""Reaction distance distribution analysis.

Plots histogram of reaction distances with optional reference distribution
overlay and KS statistics.
"""

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from scipy import stats

from .loader import load_reaction_details, load_reference_distance
from .plot import (
    setup_style, save_figure,
    PlotConfig, HistogramStyle, get_config, set_config,
)


def analyze_distance(
    df: Optional[pd.DataFrame] = None,
    input_dir: Optional[str | Path] = None,
    ref_csv: Optional[str | Path] = None,
    bins: Optional[int] = None,
    output_dir: Optional[str | Path] = None,
    plot_config: Optional[str | Path] = None,
) -> None:
    """Analyze reaction distance distribution and plot histogram.

    Parameters
    ----------
    df : reaction_details DataFrame (loaded from input_dir if None)
    input_dir : simulation output directory
    ref_csv : reference distance CSV (single column or bin_center,count)
    bins : histogram bin count (CLI override)
    output_dir : output directory for plots (default: input_dir/analysis/)
    plot_config : path to plot_config.yaml
    """
    # --- Config ---
    config = PlotConfig.from_yaml(plot_config)
    set_config(config)
    chart = config.get_chart("distance")
    if bins is not None:
        if chart.histogram is None:
            chart.histogram = HistogramStyle()
        chart.histogram.bins = bins
    chart = chart.fill_defaults()
    setup_style(chart)

    if df is None:
        if input_dir is None:
            raise ValueError("Must provide either df or input_dir")
        df = load_reaction_details(input_dir)

    if output_dir is None and input_dir is not None:
        output_dir = Path(input_dir) / "analysis"
    elif output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reaction_types = df["reaction_type"].unique()
    n_types = len(reaction_types)

    # Reference distribution
    ref_data = None
    if ref_csv is not None:
        try:
            ref_data = load_reference_distance(ref_csv)
        except FileNotFoundError:
            print(f"WARNING: reference file not found: {ref_csv}, skipping")

    # ---- Plot ----
    import matplotlib.pyplot as plt
    fw, fh = chart.figure.figsize
    fig, axes = plt.subplots(1, n_types, figsize=(fw * n_types, fh))
    if n_types == 1:
        axes = [axes]

    hist_style = chart.histogram
    ref_style = chart.reference

    for idx, rxn_type in enumerate(reaction_types):
        ax = axes[idx]
        distances = df[df["reaction_type"] == rxn_type]["distance"].dropna().values

        if len(distances) == 0:
            ax.text(0.5, 0.5, f"{rxn_type}: no data", transform=ax.transAxes,
                    ha="center", va="center")
            ax.set_title(rxn_type)
            continue

        # Simulation histogram
        ax.hist(
            distances, bins=hist_style.bins, alpha=hist_style.alpha,
            density=True, color=hist_style.color,
            edgecolor=hist_style.edge_color,
            label=f"Simulation (n={len(distances)})"
        )

        # Reference overlay
        if ref_data is not None and len(ref_data) > 0:
            ax.hist(ref_data, bins=hist_style.bins, alpha=ref_style.alpha,
                    density=True, color=ref_style.color,
                    edgecolor=ref_style.edge_color,
                    label=f"Reference (n={len(ref_data)})")

            ks_stat, ks_pval = stats.ks_2samp(distances, ref_data)
            ax.set_title(f"{rxn_type}  (KS={ks_stat:.3f}, p={ks_pval:.3e})", fontsize=12)
        else:
            ax.set_title(f"{rxn_type}  (n={len(distances)})", fontsize=12)

        # Mean/std markers
        mean_d = np.mean(distances)
        std_d = np.std(distances)
        ml = chart.mean_line
        ax.axvline(mean_d, color=ml.color, linestyle=ml.linestyle, alpha=ml.alpha,
                   label=f"Mean={mean_d:.2f}")
        ax.axvline(mean_d - std_d, color="gray", linestyle=":", alpha=0.5)
        ax.axvline(mean_d + std_d, color="gray", linestyle=":", alpha=0.5)

        # Axis labels
        axis_cfg = chart.axis
        ax.set_xlabel(axis_cfg.xlabel or "Distance (Angstrom)",
                      fontsize=axis_cfg.label_fontsize, labelpad=axis_cfg.label_pad)
        ax.set_ylabel(axis_cfg.ylabel or "Probability Density",
                      fontsize=axis_cfg.label_fontsize, labelpad=axis_cfg.label_pad)

        lg = chart.legend
        ax.legend(loc=lg.loc, fontsize=lg.font_size, framealpha=lg.frame_alpha)

    title = chart.axis.title or "Reaction Distance Distribution"
    fig.suptitle(title, fontsize=chart.axis.title_fontsize, y=1.02)
    fig.tight_layout()

    output_path = output_dir / "distance_distribution.png"
    saved_path = save_figure(fig, output_path, dpi=chart.figure.dpi,
                             save_format=chart.figure.save_format)
    print(f"[distance] Saved to: {saved_path}")

    # Summary
    print("\n=== Distance Distribution ===")
    for rxn_type in reaction_types:
        distances = df[df["reaction_type"] == rxn_type]["distance"].dropna().values
        if len(distances) == 0:
            continue
        print(f"  {rxn_type}: n={len(distances)}, "
              f"min={distances.min():.2f}, max={distances.max():.2f}, "
              f"mean={distances.mean():.2f}, std={distances.std():.2f}")
    if ref_data is not None and len(ref_data) > 0:
        print(f"  Reference: n={len(ref_data)}, "
              f"min={ref_data.min():.2f}, max={ref_data.max():.2f}, "
              f"mean={ref_data.mean():.2f}, std={ref_data.std():.2f}")
