"""Reaction statistics module.

Computes reaction counts, ratios, per-cycle statistics and generates
stacked bar charts and pie charts.
"""

from pathlib import Path
from typing import Optional, Dict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .loader import load_reaction_details
from .plot import (
    setup_style, save_figure,
    PlotConfig, get_config, set_config,
)


def analyze_reaction_stats(
    df: Optional[pd.DataFrame] = None,
    input_dir: Optional[str | Path] = None,
    output_dir: Optional[str | Path] = None,
    plot_config: Optional[str | Path] = None,
) -> Dict:
    """Compute reaction counts and ratios.

    Parameters
    ----------
    df : reaction_details DataFrame
    input_dir : simulation output directory
    output_dir : output directory
    plot_config : path to plot_config.yaml

    Returns
    -------
    dict with total, by_type, by_pair_type, by_cycle
    """
    config = PlotConfig.from_yaml(plot_config)
    set_config(config)
    chart = config.get_chart("reaction_stats").fill_defaults()
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

    # --- Overall statistics ---
    total = len(df)
    by_type = df["reaction_type"].value_counts()
    by_type_pct = (by_type / total * 100).round(1)

    print("\n=== Reaction Statistics ===")
    print(f"  Total reactions: {total}")
    print(f"  Cycles with reactions: {df['cycle'].nunique()}")
    print(f"  By type:")
    for rxn_type in by_type.index:
        ratio = ""
        if len(by_type) > 1:
            others = [t for t in by_type.index if t != rxn_type]
            for other in others:
                r = by_type[rxn_type] / max(by_type[other], 1)
                ratio += f", vs {other}={r:.2f}:1"
        print(f"    {rxn_type}: {by_type[rxn_type]} ({by_type_pct[rxn_type]}%){ratio}")

    pair_type_counts = df["pair_type"].value_counts()
    print(f"  By bead pair type:")
    for pt, count in pair_type_counts.items():
        print(f"    {pt}: {count} ({count/total*100:.1f}%)")

    # --- Per-cycle ---
    by_cycle = df.groupby("cycle")["reaction_type"].value_counts().unstack(fill_value=0)
    by_cycle.index.name = "cycle"

    csv_path = output_dir / "reaction_counts.csv"
    by_cycle.to_csv(str(csv_path))
    print(f"\n  Per-cycle stats saved to: {csv_path}")

    # --- Plot ---
    fig_size = chart.figure.figsize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=fig_size)
    axis_cfg = chart.axis

    # Left: pie chart
    colors = plt.cm.Set3(np.linspace(0, 1, max(len(by_type), 1)))
    wedges, texts, autotexts = ax1.pie(
        by_type.values, labels=by_type.index, autopct="%1.1f%%",
        colors=colors, startangle=90
    )
    pie_title = axis_cfg.title or f"Reaction Type Distribution (total={total})"
    ax1.set_title(pie_title, fontsize=axis_cfg.title_fontsize,
                  pad=axis_cfg.title_pad)

    # Right: per-cycle stacked bar
    if len(by_cycle) > 0:
        reaction_types_list = by_cycle.columns.tolist()
        bottom = np.zeros(len(by_cycle))
        for rxn_type in reaction_types_list:
            values = by_cycle[rxn_type].values
            ax2.bar(by_cycle.index, values, bottom=bottom, label=rxn_type,
                    alpha=0.85, width=0.8)
            bottom += values

        ax2.set_xlabel(axis_cfg.xlabel or "Cycle",
                       fontsize=axis_cfg.label_fontsize,
                       labelpad=axis_cfg.label_pad)
        ax2.set_ylabel(axis_cfg.ylabel or "Reaction Count",
                       fontsize=axis_cfg.label_fontsize,
                       labelpad=axis_cfg.label_pad)
        ax2.set_title("Reactions per Cycle", fontsize=axis_cfg.title_fontsize,
                      pad=axis_cfg.title_pad)
        lg = chart.legend
        ax2.legend(fontsize=lg.font_size, loc=lg.loc, framealpha=lg.frame_alpha)
        ax2.set_xticks(by_cycle.index)
        ax2.set_xticklabels(by_cycle.index, rotation=45, ha="right", fontsize=8)

    fig.tight_layout()
    output_path = output_dir / "reaction_stats.png"
    saved_path = save_figure(fig, output_path, dpi=chart.figure.dpi,
                             save_format=chart.figure.save_format)
    print(f"  Figure saved to: {saved_path}")

    return {
        "total": total,
        "by_type": by_type.to_dict(),
        "by_pair_type": pair_type_counts.to_dict(),
        "by_cycle": by_cycle,
    }
