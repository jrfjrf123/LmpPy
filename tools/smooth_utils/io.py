"""
Data loading functions for distribution smoothing.
分布平滑模块的数据加载函数
"""

import numpy as np
import os
from typing import Tuple

# Relative imports for package
from .constants import DEFAULT_THRESHOLD


def load_distribution(filepath: str) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Load distribution data from file.

    Automatically detects distribution type (bond/rdf/angle/dihedral) from filename.
    Handles comment lines starting with '#'.

    Args:
        filepath: Path to the distribution file

    Returns:
        x: Coordinate array (distance in Angstrom or angle in degrees)
        P: Probability density array
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    # Detect distribution type from filename
    filename = os.path.basename(filepath).lower()
    if 'rdf' in filename:
        dist_type = 'rdf'
    elif 'dihedral' in filename:
        dist_type = 'dihedral'
    elif 'angle' in filename:
        dist_type = 'angle'
    elif 'bond' in filename:
        dist_type = 'bond'
    else:
        # Try to infer from file header
        with open(filepath, 'r') as f:
            header = f.readline().lower()
            if 'rdf' in header or 'g(r)' in header:
                dist_type = 'rdf'
            elif 'dihedral' in header or 'phi' in header:
                dist_type = 'dihedral'
            elif 'angle' in header or 'theta' in header:
                dist_type = 'angle'
            elif 'bond' in header or 'p(r)' in header:
                dist_type = 'bond'
            else:
                dist_type = 'unknown'
                print(f"  Warning: Could not determine distribution type for {filepath}")
                print(f"           Defaulting to 'bond' type processing")
                dist_type = 'bond'

    # Load data (skip comment lines)
    try:
        data = np.loadtxt(filepath, comments='#')
    except Exception as e:
        raise ValueError(f"Error reading file {filepath}: {e}")

    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Invalid data format in {filepath}. Expected at least 2 columns.")

    x = data[:, 0]
    P = data[:, 1]

    # Calculate bin width
    if len(x) > 1:
        dr = x[1] - x[0]
    else:
        dr = 1.0
        print(f"  Warning: Only one data point in {filepath}")

    # Validate data
    if np.any(np.isnan(x)) or np.any(np.isnan(P)):
        print(f"  Warning: NaN values detected in {filepath}")
        # Replace NaN with 0
        P = np.nan_to_num(P, nan=0.0)

    if np.any(P < 0):
        print(f"  Warning: Negative values detected in {filepath}, setting to 0")
        P = np.maximum(P, 0.0)

    return x, P, dist_type, dr


def get_adaptive_window_base(n_bins: int) -> int:
    """
    Calculate adaptive base window size based on number of bins.

    Formula: window = max(5, int(n_bins / 100))
    Ensures window is always odd.

    Args:
        n_bins: Number of data points

    Returns:
        Base window size (odd number >= 5)
    """
    window = max(5, int(n_bins / 100))
    # Ensure odd number
    if window % 2 == 0:
        window += 1
    return window