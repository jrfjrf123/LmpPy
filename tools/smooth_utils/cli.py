#!/usr/bin/env python3
"""
Distribution Smoothing for All Bonded Types (Bond, Angle, Dihedral, RDF)
所有键合类型分布平滑主脚本

Features:
- Automatic detection of distribution type (bond/angle/dihedral/rdf)
- Harmonic boundary smoothing for bond and angle distributions
- Periodic Gaussian smoothing for dihedral distributions
- RDF-specific smoothing with harmonic left boundary
- Quality report generation

Usage:
    python -m LmpPy.tools.smooth_utils.cli input_file.txt -o output_dir
    python -m LmpPy.tools.smooth_utils.cli -d input_directory/ -o output_dir
    python -m LmpPy.tools.smooth_utils.cli --types bond angle dihedral -d input_dir/
"""

import numpy as np
import os
import argparse
from typing import Tuple, Dict, List, Any, Optional

# Import from submodules (relative imports for package)
from .constants import DEFAULT_TEMPERATURE
from .io import load_distribution, get_adaptive_window_base
from .preprocess import find_valid_range, interpolate_zeros
from .peaks import detect_peaks, detect_significant_peaks
from .zones import should_use_gaussian
from .core import (
    smooth_bond_with_harmonic_boundary,
    smooth_rdf_with_harmonic_left_boundary,
    gaussian_smooth_with_constraint
)
from .angle_dihedral import (
    smooth_angle_with_harmonic_boundary,
    smooth_dihedral_periodic
)
from .quality import calculate_quality_metrics, check_force_consistency
from .report import generate_report


def smooth_distribution(filepath: str,
                        output_dir: str,
                        temperature: float = DEFAULT_TEMPERATURE,
                        verbose: bool = True) -> Tuple[str, Dict[str, Any]]:
    """
    Main function to smooth a distribution file.

    Automatically detects distribution type and applies appropriate smoothing:
    - bond: harmonic boundary smoothing
    - rdf: harmonic left boundary + Gaussian core + decay right boundary
    - angle: harmonic boundary smoothing (same as bond)
    - dihedral: periodic Gaussian smoothing

    Args:
        filepath: Path to input distribution file
        output_dir: Directory for output files
        temperature: Temperature in Kelvin
        verbose: Whether to print progress information

    Returns:
        Tuple of (smoothed_file_path, result_dict)
        result_dict contains: metrics, params, peak info, etc.
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {filepath}")
        print(f"{'='*60}")

    # Step 1: Load distribution
    if verbose:
        print("Step 1: Loading distribution data...")
    x, P, dist_type, dr = load_distribution(filepath)
    n_bins = len(x)

    if verbose:
        print(f"  Type: {dist_type}")
        print(f"  Data points: {n_bins}")
        print(f"  Bin width: {dr:.6f}")
        print(f"  Range: [{x[0]:.4f}, {x[-1]:.4f}]")

    # Keep raw data for comparison
    P_raw = P.copy()

    # Step 2: Find valid range
    if verbose:
        print("Step 2: Finding valid data range...")
    idx_valid_start, idx_valid_end, _, _ = find_valid_range(x, P)

    if verbose:
        print(f"  Valid range: [{x[idx_valid_start]:.4f}, {x[idx_valid_end]:.4f}]")

    # Step 3: Interpolate zeros
    if verbose:
        print("Step 3: Interpolating zero values...")
    P = interpolate_zeros(x, P, idx_valid_start, idx_valid_end)

    # Step 4: Detect peaks
    if verbose:
        print("Step 4: Detecting peaks...")
    peak_indices_raw, peak_positions_raw, peak_heights_raw = detect_peaks(x, P)

    if verbose:
        print(f"  Found {len(peak_indices_raw)} peaks")
        for i, (pos, height) in enumerate(zip(peak_positions_raw, peak_heights_raw)):
            print(f"    Peak {i+1}: pos={pos:.4f}, height={height:.6f}")

    # Step 5: Calculate base window
    base_window = get_adaptive_window_base(n_bins)
    if verbose:
        print(f"Step 5: Base window size: {base_window}")

    # Step 6: Apply smoothing based on distribution type
    if verbose:
        print(f"Step 6: Applying {dist_type} smoothing...")

    if dist_type == 'bond':
        # Bond: harmonic boundary smoothing
        adaptive_sg_window = max(7, min(21, base_window * 2 + 1))
        if adaptive_sg_window % 2 == 0:
            adaptive_sg_window += 1

        P_smooth, metrics = smooth_bond_with_harmonic_boundary(
            x, P, idx_valid_start, idx_valid_end,
            sg_window=adaptive_sg_window,
            sg_poly=3,
            threshold=0.001,
            F_max=500.0,
            temperature=temperature,
            verbose=verbose
        )
        params = {
            'method': 'harmonic_boundary',
            'sg_window': adaptive_sg_window,
            'dist_type': 'bond'
        }

    elif dist_type == 'rdf':
        # RDF: harmonic left boundary + Gaussian core + decay right boundary
        best_sigma = 3.0
        best_score = -float('inf')
        best_result = None
        best_metrics = None

        for sigma in [2.0, 3.0, 5.0, 7.0]:
            g_smooth_trial, metrics_trial = smooth_rdf_with_harmonic_left_boundary(
                x, P, idx_valid_start, idx_valid_end,
                sigma=sigma,
                threshold=0.05,
                F_max=500.0,
                temperature=temperature,
                verbose=False
            )

            reduction = metrics_trial.get('roughness_reduction_ratio', 0)
            oscillations = metrics_trial.get('left_oscillations', 0)
            decreasing = metrics_trial.get('g_decreasing_points', 0)
            score = reduction - 0.1 * oscillations - 0.1 * decreasing

            if score > best_score:
                best_score = score
                best_sigma = sigma
                best_result = g_smooth_trial
                best_metrics = metrics_trial

        P_smooth = best_result
        metrics = best_metrics
        params = {
            'method': 'harmonic_left_rdf',
            'sigma': best_sigma,
            'dist_type': 'rdf'
        }

        if verbose:
            print(f"  Selected sigma: {best_sigma}")
            print(f"  Roughness reduction: {metrics.get('roughness_reduction_ratio', 0)*100:.1f}%")

    elif dist_type == 'angle':
        # Angle: harmonic boundary smoothing (same as bond)
        adaptive_sg_window = max(7, min(21, base_window * 2 + 1))
        if adaptive_sg_window % 2 == 0:
            adaptive_sg_window += 1

        P_smooth, metrics = smooth_angle_with_harmonic_boundary(
            x, P, idx_valid_start, idx_valid_end,
            sg_window=adaptive_sg_window,
            sg_poly=3,
            threshold=0.001,
            F_max=500.0,
            temperature=temperature,
            verbose=verbose
        )
        params = {
            'method': 'harmonic_boundary_angle',
            'sg_window': adaptive_sg_window,
            'dist_type': 'angle'
        }

    elif dist_type == 'dihedral':
        # Dihedral: periodic Gaussian smoothing
        P_smooth, metrics = smooth_dihedral_periodic(
            x, P,
            sigma=3.0,
            sigma_range=[2.0, 3.0, 5.0, 7.0],
            temperature=temperature,
            verbose=verbose
        )
        params = {
            'method': 'periodic_gaussian',
            'sigma': metrics.get('sigma', 3.0),
            'dist_type': 'dihedral'
        }

    else:
        # Unknown type: default to bond smoothing
        if verbose:
            print(f"  Warning: Unknown type '{dist_type}', using bond smoothing")
        adaptive_sg_window = max(7, min(21, base_window * 2 + 1))
        if adaptive_sg_window % 2 == 0:
            adaptive_sg_window += 1

        P_smooth, metrics = smooth_bond_with_harmonic_boundary(
            x, P, idx_valid_start, idx_valid_end,
            sg_window=adaptive_sg_window,
            temperature=temperature,
            verbose=verbose
        )
        params = {
            'method': 'harmonic_boundary',
            'sg_window': adaptive_sg_window,
            'dist_type': dist_type
        }

    # Step 7: Detect peaks in smoothed data
    if verbose:
        print("Step 7: Detecting peaks in smoothed data...")
    peak_indices_smooth, peak_positions_smooth, peak_heights_smooth = detect_peaks(x, P_smooth)

    if verbose:
        print(f"  Found {len(peak_indices_smooth)} peaks in smoothed data")

    # Step 8: Calculate quality metrics
    if verbose:
        print("Step 8: Calculating quality metrics...")
    quality_metrics, total_score = calculate_quality_metrics(
        x, P_raw, P_smooth,
        peak_positions_raw, peak_heights_raw,
        peak_positions_smooth, peak_heights_smooth,
        dr, temperature,
        idx_valid_start, idx_valid_end
    )

    if verbose:
        print(f"  Total score: {total_score:.4f}")

    # Step 9: Check force consistency
    if verbose:
        print("Step 9: Checking force consistency...")
    n_inconsistent, total_points, _ = check_force_consistency(
        x, P_smooth, temperature,
        threshold=1.0,
        idx_valid_start=idx_valid_start,
        idx_valid_end=idx_valid_end
    )
    consistency_rate = 100 * (1 - n_inconsistent / total_points) if total_points > 0 else 100

    if verbose:
        print(f"  Consistency rate: {consistency_rate:.1f}%")

    # Step 10: Save smoothed distribution
    if verbose:
        print("Step 10: Saving output files...")

    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(filepath))[0]
    # smoothed_path = os.path.join(output_dir, f'{basename}_smoothed.txt')
    smoothed_path = os.path.join(output_dir, f'{basename}_dist.txt')

    # Write smoothed data
    with open(smoothed_path, 'w') as f:
        f.write(f"# Smoothed {dist_type} distribution\n")
        f.write(f"# Original file: {filepath}\n")
        f.write(f"# Method: {params.get('method', 'unknown')}\n")
        f.write(f"# Temperature: {temperature} K\n")
        f.write("# x P_smoothed\n")
        for i in range(len(x)):
            f.write(f"{x[i]:.6e} {P_smooth[i]:.6e}\n")

    # Generate report
    plot_path, report_path = generate_report(
        filepath, x, P_raw, P_smooth, dist_type, dr,
        peak_positions_raw, peak_heights_raw,
        peak_positions_smooth, peak_heights_smooth,
        quality_metrics, params, [],
        output_dir, temperature,
        idx_valid_start, idx_valid_end
    )

    if verbose:
        print(f"  Saved: {smoothed_path}")
        print(f"  Plot: {plot_path}")
        print(f"  Report: {report_path}")

    result = {
        'smoothed_path': smoothed_path,
        'plot_path': plot_path,
        'report_path': report_path,
        'dist_type': dist_type,
        'metrics': quality_metrics,
        'params': params,
        'force_consistency': {
            'n_inconsistent': n_inconsistent,
            'total_points': total_points,
            'consistency_rate': consistency_rate
        },
        'peaks': {
            'original': len(peak_indices_raw),
            'smoothed': len(peak_indices_smooth)
        }
    }

    return smoothed_path, result


def smooth_all_distributions(input_dir: str,
                             output_dir: str,
                             dist_types: List[str] = None,
                             temperature: float = DEFAULT_TEMPERATURE,
                             verbose: bool = True) -> Dict[str, Any]:
    """
    Batch process all distribution files in a directory.

    Automatically discovers and processes:
    - bond_type*_dist.txt
    - angle_type*_dist.txt
    - dihedral_type*_dist.txt
    - rdf_type*_*.txt

    Args:
        input_dir: Directory containing distribution files
        output_dir: Directory for output files
        dist_types: List of types to process (None = all)
        temperature: Temperature in Kelvin
        verbose: Whether to print progress

    Returns:
        Dictionary with processing results
    """
    if dist_types is None:
        dist_types = ['bond', 'angle', 'dihedral', 'rdf']

    # Discover files
    files_by_type = {'bond': [], 'angle': [], 'dihedral': [], 'rdf': []}

    for f in os.listdir(input_dir):
        if not f.endswith('_dist.txt'):
            continue

        filepath = os.path.join(input_dir, f)
        f_lower = f.lower()

        if 'dihedral' in f_lower:
            files_by_type['dihedral'].append(filepath)
        elif 'angle' in f_lower:
            files_by_type['angle'].append(filepath)
        elif 'bond' in f_lower:
            files_by_type['bond'].append(filepath)
        elif 'rdf' in f_lower:
            files_by_type['rdf'].append(filepath)

    # Process each type
    results = {}
    total_success = 0
    total_fail = 0

    for dtype in dist_types:
        if dtype not in files_by_type:
            continue

        files = files_by_type[dtype]
        if verbose and files:
            print(f"\n{'='*60}")
            print(f"Processing {dtype} distributions ({len(files)} files)")
            print(f"{'='*60}")

        for filepath in files:
            try:
                smoothed_path, result = smooth_distribution(
                    filepath, output_dir, temperature, verbose
                )
                results[filepath] = {'status': 'success', 'result': result}
                total_success += 1
            except Exception as e:
                if verbose:
                    print(f"Error processing {filepath}: {e}")
                results[filepath] = {'status': 'error', 'error': str(e)}
                total_fail += 1

    return {
        'results': results,
        'total_success': total_success,
        'total_fail': total_fail
    }


def main():
    """Command line interface."""
    parser = argparse.ArgumentParser(
        description='Smooth distribution curves (bond/angle/dihedral/rdf) for IBM potential calculation.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single file
  python smooth_all_bonded_distribution.py bond_type1_dist.txt -o smoothed_output/

  # Process all files in a directory
  python smooth_all_bonded_distribution.py -d distributions/ -o smoothed_output/

  # Process only angle and dihedral files
  python smooth_all_bonded_distribution.py -d distributions/ -o smoothed_output/ --types angle dihedral

  # Specify temperature
  python smooth_all_bonded_distribution.py bond_type1_dist.txt -T 300 -o smoothed_output/
"""
    )

    # Input options
    parser.add_argument('files', nargs='*',
                       help='Input distribution file(s) to process')
    parser.add_argument('-d', '--directory',
                       help='Process all *_dist.txt files in this directory')

    # Output options
    parser.add_argument('-o', '--output', default='smoothed_output',
                       help='Output directory (default: smoothed_output)')

    # Type filtering
    parser.add_argument('--types', nargs='+',
                        choices=['bond', 'angle', 'dihedral', 'rdf', 'all'],
                        default=['all'],
                        help='Distribution types to process (default: all)')

    # Other options
    parser.add_argument('-T', '--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'Temperature in Kelvin (default: {DEFAULT_TEMPERATURE})')
    parser.add_argument('-q', '--quiet', action='store_true',
                       help='Suppress progress output')

    args = parser.parse_args()

    # Process type filter
    if 'all' in args.types:
        dist_types = None  # Process all
    else:
        dist_types = args.types

    verbose = not args.quiet

    # Collect input files
    input_files = []

    if args.files:
        for f in args.files:
            if os.path.isfile(f):
                input_files.append(f)
            else:
                print(f"Warning: File not found: {f}")

    if args.directory:
        if os.path.isdir(args.directory):
            for f in os.listdir(args.directory):
                if f.endswith('_dist.txt'):
                    input_files.append(os.path.join(args.directory, f))
        else:
            print(f"Error: Directory not found: {args.directory}")
            return 1

    if not input_files:
        print("Error: No input files specified.")
        print("Use -h for help.")
        return 1

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# Distribution Smoothing Tool (Bond/Angle/Dihedral/RDF)")
        print(f"# Input files: {len(input_files)}")
        print(f"# Output directory: {args.output}")
        print(f"# Temperature: {args.temperature} K")
        print(f"{'#'*70}")

    # Process files
    results = []
    success_count = 0
    fail_count = 0

    for i, filepath in enumerate(input_files):
        if verbose:
            print(f"\n[{i+1}/{len(input_files)}] Processing: {filepath}")

        try:
            smoothed_path, result = smooth_distribution(
                filepath=filepath,
                output_dir=args.output,
                temperature=args.temperature,
                verbose=verbose
            )
            results.append({'input': filepath, 'status': 'success', 'result': result})
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"Error: {e}")
            results.append({'input': filepath, 'status': 'error', 'error': str(e)})
            fail_count += 1

    # Print summary
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# PROCESSING SUMMARY")
        print(f"#   Total files: {len(input_files)}")
        print(f"#   Successful: {success_count}")
        print(f"#   Failed: {fail_count}")
        print(f"#   Output directory: {os.path.abspath(args.output)}")
        print(f"{'#'*70}")

    return 0 if fail_count == 0 else 1


if __name__ == '__main__':
    import sys
    sys.exit(main())