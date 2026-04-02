"""
Report generation functions for distribution smoothing.
分布平滑模块的报告生成函数
"""

import os
import numpy as np
from typing import Tuple, Dict, List, Any

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Relative imports for package
from .constants import (
    DEFAULT_TEMPERATURE, WEIGHT_PEAK_POS, WEIGHT_FORCE_SMOOTH, WEIGHT_PEAK_HEIGHT,
    FORCE_CONSISTENCY_THRESHOLD
)
from .quality import (
    calculate_potential_and_force, check_force_consistency, match_peaks
)


def generate_comparison_plot(x: np.ndarray, P_raw: np.ndarray, P_smooth: np.ndarray,
                            dist_type: str,
                            peak_positions_raw: np.ndarray, peak_heights_raw: np.ndarray,
                            peak_positions_smooth: np.ndarray, peak_heights_smooth: np.ndarray,
                            output_path: str,
                            temperature: float = DEFAULT_TEMPERATURE) -> str:
    """
    Generate comparison plot showing original vs smoothed distribution.

    Creates a 3-panel figure:
    1. Distribution comparison with peaks marked
    2. Potential energy curves
    3. Force curves

    Args:
        x: Coordinate array
        P_raw: Original probability density
        P_smooth: Smoothed probability density
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        peak_positions_raw, peak_heights_raw: Original peaks
        peak_positions_smooth, peak_heights_smooth: Smoothed peaks
        output_path: Path for saving the figure
        temperature: Temperature in Kelvin

    Returns:
        Path to saved figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Get axis labels based on distribution type
    xlabel, ylabel, title = _get_axis_labels(dist_type)

    # ===== Panel 1: Distribution Comparison =====
    ax1 = axes[0]

    # Plot distributions
    ax1.plot(x, P_raw, 'b-', alpha=0.5, linewidth=1, label='Original')
    ax1.plot(x, P_smooth, 'r-', linewidth=1.5, label='Smoothed')

    # Mark peaks
    if len(peak_positions_raw) > 0:
        ax1.scatter(peak_positions_raw, peak_heights_raw, c='blue', marker='o',
                   s=60, label='Original peaks', zorder=5, edgecolors='darkblue')
    if len(peak_positions_smooth) > 0:
        ax1.scatter(peak_positions_smooth, peak_heights_smooth, c='red', marker='x',
                   s=60, label='Smoothed peaks', zorder=5, linewidths=2)

    ax1.set_xlabel(xlabel, fontsize=11)
    ax1.set_ylabel(ylabel, fontsize=11)
    ax1.set_title(f'{title}: Original vs Smoothed', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)

    # ===== Panel 2: Potential Energy =====
    ax2 = axes[1]

    # Calculate potentials
    U_raw, _ = calculate_potential_and_force(x, P_raw, temperature)
    U_smooth, _ = calculate_potential_and_force(x, P_smooth, temperature)

    # Plot only finite values and limit range
    max_U = 10.0

    finite_raw = np.isfinite(U_raw) & (U_raw < max_U)
    finite_smooth = np.isfinite(U_smooth) & (U_smooth < max_U)

    if np.any(finite_raw):
        ax2.plot(x[finite_raw], U_raw[finite_raw], 'b-', alpha=0.5, linewidth=1, label='Original U')
    if np.any(finite_smooth):
        ax2.plot(x[finite_smooth], U_smooth[finite_smooth], 'r-', linewidth=1.5, label='Smoothed U')

    ax2.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax2.set_xlabel(xlabel, fontsize=11)
    ax2.set_ylabel('U (kcal/mol)', fontsize=11)
    ax2.set_title('Potential Energy (Boltzmann Inversion)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-1, max_U)

    # ===== Panel 3: Force Curve =====
    ax3 = axes[2]

    _, F_raw = calculate_potential_and_force(x, P_raw, temperature)
    _, F_smooth = calculate_potential_and_force(x, P_smooth, temperature)

    # Limit force range for display
    max_F = 50.0
    min_F = -50.0

    finite_raw_F = np.isfinite(F_raw) & (F_raw < max_F) & (F_raw > min_F)
    finite_smooth_F = np.isfinite(F_smooth) & (F_smooth < max_F) & (F_smooth > min_F)

    if np.any(finite_raw_F):
        ax3.plot(x[finite_raw_F], F_raw[finite_raw_F], 'b-', alpha=0.5, linewidth=1, label='Original F')
    if np.any(finite_smooth_F):
        ax3.plot(x[finite_smooth_F], F_smooth[finite_smooth_F], marker='o', color='red',
                label='Smoothed F', markersize=1)

    ax3.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax3.set_xlabel(xlabel, fontsize=11)
    force_unit = _get_force_unit(dist_type)
    ax3.set_ylabel(f'F ({force_unit})', fontsize=11)
    ax3.set_title('Force Curve: F = -dU/dx', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10, loc='upper right')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    return output_path


def generate_text_report(filepath: str,
                        x: np.ndarray, P_raw: np.ndarray, P_smooth: np.ndarray,
                        dist_type: str, dr: float,
                        peak_positions_raw: np.ndarray, peak_heights_raw: np.ndarray,
                        peak_positions_smooth: np.ndarray, peak_heights_smooth: np.ndarray,
                        metrics: Dict[str, float],
                        params: Dict[str, Any],
                        optimization_log: List[Dict],
                        output_path: str,
                        temperature: float = DEFAULT_TEMPERATURE,
                        idx_valid_start: int = None, idx_valid_end: int = None) -> str:
    """
    Generate text report with quality metrics and optimization details.

    Args:
        filepath: Original input file path
        x, P_raw, P_smooth: Distribution data
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width
        peak_positions_raw, peak_heights_raw: Original peaks
        peak_positions_smooth, peak_heights_smooth: Smoothed peaks
        metrics: Quality metrics dictionary
        params: Smoothing parameters used
        optimization_log: List of optimization iterations
        output_path: Path for saving report
        temperature: Temperature in Kelvin
        idx_valid_start, idx_valid_end: Valid data range for force consistency check

    Returns:
        Path to saved report
    """
    # Check force consistency (within valid range)
    n_inconsistent, total_points, _ = check_force_consistency(
        x, P_smooth, temperature, FORCE_CONSISTENCY_THRESHOLD,
        idx_valid_start, idx_valid_end
    )
    consistency_rate = 100 * (1 - n_inconsistent / total_points) if total_points > 0 else 100

    # Match peaks for detailed comparison
    peak_matches = match_peaks(
        peak_positions_raw, peak_heights_raw,
        peak_positions_smooth, peak_heights_smooth
    )

    # Get unit string based on distribution type
    unit_str = _get_unit_string(dist_type)

    with open(output_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("DISTRIBUTION SMOOTHING QUALITY REPORT\n")
        f.write("=" * 70 + "\n\n")

        # File information
        f.write("--- Input Information ---\n")
        f.write(f"  Input file: {filepath}\n")
        f.write(f"  Distribution type: {dist_type}\n")
        f.write(f"  Data points: {len(x)}\n")
        f.write(f"  Bin width: {dr:.6f} {unit_str}\n")
        f.write(f"  Data range: [{x[0]:.4f}, {x[-1]:.4f}] {unit_str}\n")
        f.write(f"  Temperature: {temperature} K\n\n")

        # Smoothing parameters
        f.write("--- Smoothing Parameters ---\n")
        if params:
            for key, value in params.items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
        f.write("\n")

        # Quality metrics
        f.write("--- Quality Metrics ---\n")
        f.write(f"  Total Score: {metrics.get('total_score', 0):.4f}\n\n")

        f.write(f"  Peak Position Fidelity:\n")
        f.write(f"    Score: {metrics.get('score_peak_position', 0):.4f} (weight: {WEIGHT_PEAK_POS})\n")
        f.write(f"    Average error: {metrics.get('peak_position_error_absolute', 0):.4f} {unit_str} ")
        f.write(f"({metrics.get('peak_position_error_bins', 0):.2f} bins)\n\n")

        f.write(f"  Force Smoothness:\n")
        f.write(f"    Score: {metrics.get('score_force_smooth', 0):.4f} (weight: {WEIGHT_FORCE_SMOOTH})\n")
        f.write(f"    Roughness (raw): {metrics.get('force_roughness_raw', 0):.2e}\n")
        f.write(f"    Roughness (smoothed): {metrics.get('force_roughness', 0):.2e}\n")
        f.write(f"    Roughness reduction: {metrics.get('force_roughness_reduction_ratio', 0)*100:.2f}%\n\n")

        f.write(f"  Peak Height Fidelity:\n")
        f.write(f"    Score: {metrics.get('score_peak_height', 0):.4f} (weight: {WEIGHT_PEAK_HEIGHT})\n")
        f.write(f"    Average relative error: {metrics.get('peak_height_error_relative', 0):.4f}\n\n")

        # Force consistency check
        f.write("--- Force Consistency Check ---\n")
        f.write(f"  Inconsistent points: {n_inconsistent} / {total_points}\n")
        f.write(f"  Consistency rate: {consistency_rate:.1f}%\n")
        f.write(f"  Threshold used: {FORCE_CONSISTENCY_THRESHOLD}\n\n")

        # Peak comparison
        f.write("--- Peak Statistics ---\n")
        f.write(f"  Original peaks detected: {len(peak_positions_raw)}\n")
        f.write(f"  Smoothed peaks detected: {len(peak_positions_smooth)}\n\n")

        if len(peak_matches) > 0:
            f.write("  Peak-by-peak comparison:\n")
            f.write("  " + "-" * 66 + "\n")
            f.write(f"  {'Peak':<5} {'Orig Pos':<10} {'Smooth Pos':<12} {'Shift':<10} {'Height Δ':<12}\n")
            f.write("  " + "-" * 66 + "\n")

            for match in peak_matches:
                peak_num = match['raw_index'] + 1
                orig_pos = f"{match['raw_position']:.4f}"
                smooth_pos = f"{match['smooth_position']:.4f}" if match['smooth_position'] else "N/A"
                shift = f"{match['position_shift']:.4f}" if match['position_shift'] else "N/A"
                h_change = f"{match['height_change']*100:+.1f}%" if match['height_change'] else "N/A"

                f.write(f"  {peak_num:<5} {orig_pos:<10} {smooth_pos:<12} {shift:<10} {h_change:<12}\n")

            f.write("  " + "-" * 66 + "\n\n")

        # Optimization log (top 10)
        f.write("--- Optimization Log (Top 10 by Score) ---\n")
        if optimization_log:
            sorted_log = sorted(optimization_log, key=lambda x: x.get('score', 0), reverse=True)[:10]

            for entry in sorted_log:
                score = entry.get('score', 0)
                ws = entry.get('actual_window_small', 'N/A')
                wl = entry.get('actual_window_large', 'N/A')
                po = entry.get('params', {}).get('polyorder', 'N/A')
                pwf = entry.get('params', {}).get('peak_width_factor', 'N/A')

                f.write(f"  Score {score:.4f}: ws={ws}, wl={wl}, po={po}, pwf={pwf}\n")
        else:
            f.write("  No optimization log available (manual parameters used)\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 70 + "\n")

    return output_path


def generate_report(filepath: str,
                   x: np.ndarray, P_raw: np.ndarray, P_smooth: np.ndarray,
                   dist_type: str, dr: float,
                   peak_positions_raw: np.ndarray, peak_heights_raw: np.ndarray,
                   peak_positions_smooth: np.ndarray, peak_heights_smooth: np.ndarray,
                   metrics: Dict[str, float],
                   params: Dict[str, Any],
                   optimization_log: List[Dict],
                   output_dir: str,
                   temperature: float = DEFAULT_TEMPERATURE,
                   idx_valid_start: int = None, idx_valid_end: int = None) -> Tuple[str, str]:
    """
    Generate complete quality report (PNG + TXT).

    Args:
        filepath: Original input file path
        x, P_raw, P_smooth: Distribution data
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width
        peak_positions_raw, peak_heights_raw: Original peaks
        peak_positions_smooth, peak_heights_smooth: Smoothed peaks
        metrics: Quality metrics dictionary
        params: Smoothing parameters used
        optimization_log: List of optimization iterations
        output_dir: Output directory
        temperature: Temperature in Kelvin
        idx_valid_start, idx_valid_end: Valid data range

    Returns:
        Tuple of (plot_path, report_path)
    """
    os.makedirs(output_dir, exist_ok=True)

    basename = os.path.splitext(os.path.basename(filepath))[0]

    # Generate comparison plot
    plot_path = os.path.join(output_dir, f'{basename}_comparison.png')
    generate_comparison_plot(
        x, P_raw, P_smooth, dist_type,
        peak_positions_raw, peak_heights_raw,
        peak_positions_smooth, peak_heights_smooth,
        plot_path, temperature
    )

    # Generate text report
    report_path = os.path.join(output_dir, f'{basename}_report.txt')
    generate_text_report(
        filepath, x, P_raw, P_smooth, dist_type, dr,
        peak_positions_raw, peak_heights_raw,
        peak_positions_smooth, peak_heights_smooth,
        metrics, params, optimization_log,
        report_path, temperature,
        idx_valid_start, idx_valid_end
    )

    return plot_path, report_path


def _get_axis_labels(dist_type: str) -> Tuple[str, str, str]:
    """
    Get axis labels and title based on distribution type.

    Args:
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'

    Returns:
        Tuple of (xlabel, ylabel, title)
    """
    if dist_type == 'angle':
        return 'θ (deg)', 'P(θ)', 'Angle Distribution'
    elif dist_type == 'dihedral':
        return 'φ (deg)', 'P(φ)', 'Dihedral Distribution'
    elif dist_type == 'rdf':
        return 'r (Å)', 'g(r)', 'Radial Distribution Function'
    else:  # bond
        return 'r (Å)', 'P(r)', 'Bond Distribution'


def _get_force_unit(dist_type: str) -> str:
    """
    Get force unit based on distribution type.

    Args:
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'

    Returns:
        Force unit string
    """
    if dist_type in ['angle', 'dihedral']:
        return 'kcal/mol/deg'
    else:
        return 'kcal/mol/Å'


def _get_unit_string(dist_type: str) -> str:
    """
    Get coordinate unit string based on distribution type.

    Args:
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'

    Returns:
        Unit string
    """
    if dist_type in ['angle', 'dihedral']:
        return 'deg'
    else:
        return 'Å'