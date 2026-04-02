"""
Zone definition functions for distribution smoothing.
分布平滑模块的区域定义函数
"""

import numpy as np
from typing import Tuple

# Relative imports for package
from .constants import MAX_PEAKS, DEFAULT_PEAK_WIDTH_FACTOR, BOUNDARY_RATIO
from .peaks import estimate_peak_width


def define_zones(x: np.ndarray, P: np.ndarray,
                 peak_indices: np.ndarray, peak_positions: np.ndarray,
                 dist_type: str, idx_valid_start: int, idx_valid_end: int,
                 peak_width_factor: float = DEFAULT_PEAK_WIDTH_FACTOR,
                 boundary_ratio: float = BOUNDARY_RATIO) -> np.ndarray:
    """
    Define processing zones for adaptive smoothing.

    Zone definitions:
        0: Left zero region (keep as-is)
        1: Peak region (small smoothing window)
        2: Valley region (large smoothing window)
        3: RDF boundary region (apply g(r) → 1⁻ constraint)
        4: Right zero region (keep as-is)

    Args:
        x: Coordinate array
        P: Probability density array
        peak_indices: Indices of detected peaks
        peak_positions: X coordinates of peaks
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        idx_valid_start: Start index of valid data range
        idx_valid_end: End index of valid data range
        peak_width_factor: Multiplier for peak protection width
        boundary_ratio: Fraction of r_max where boundary constraint starts

    Returns:
        zone_mask: Array indicating zone for each point
    """
    n = len(x)
    zone_mask = np.zeros(n, dtype=int)

    # Zone 0: Left zero region (before valid data)
    zone_mask[:idx_valid_start] = 0

    # Estimate peak width for zone assignment
    if len(peak_indices) > 0:
        avg_peak_width = estimate_peak_width(x, P, peak_indices) * peak_width_factor
    else:
        avg_peak_width = (x[-1] - x[0]) / 20

    # Handle right side based on distribution type
    if dist_type in ['bond', 'angle']:
        # Zone 4: Right zero region for bond/angle distribution
        zone_mask[idx_valid_end + 1:] = 4
    elif dist_type == 'rdf':
        # Zone 3: Boundary region for RDF
        r_max = x[-1]

        # Transition point: max of (last significant peak position, boundary_ratio * r_max)
        if len(peak_positions) >= MAX_PEAKS:
            r_transition = max(peak_positions[MAX_PEAKS - 1], boundary_ratio * r_max)
        elif len(peak_positions) > 0:
            r_transition = max(peak_positions[-1] + avg_peak_width, boundary_ratio * r_max)
        else:
            r_transition = boundary_ratio * r_max

        idx_transition = np.searchsorted(x, r_transition)
        idx_transition = min(idx_transition, n - 1)
        zone_mask[idx_transition:] = 3
    elif dist_type == 'dihedral':
        # Dihedral is periodic, no zero region on right
        # All points are valid
        pass

    # Within valid region: distinguish peak and valley zones
    half_width = avg_peak_width / 2

    # First, mark everything in valid region as valley (zone 2)
    for i in range(idx_valid_start, min(idx_valid_end + 1, n)):
        if zone_mask[i] == 0:  # Only if not already assigned
            zone_mask[i] = 2

    # Then, mark peak regions (zone 1)
    for peak_idx in peak_indices:
        if peak_idx < idx_valid_start or peak_idx > idx_valid_end:
            continue

        peak_x = x[peak_idx]

        # Mark points within half_width of peak as peak region
        for i in range(idx_valid_start, min(idx_valid_end + 1, n)):
            if zone_mask[i] in [3, 4]:  # Don't override boundary or right-zero
                continue
            if abs(x[i] - peak_x) <= half_width:
                zone_mask[i] = 1

    return zone_mask


def should_use_gaussian(n_peaks: int, threshold: int = 2, dist_type: str = 'bond') -> bool:
    """
    Determine whether to use Gaussian smoothing based on peak count and distribution type.

    Gaussian smoothing is preferred for:
    1. Simple distributions (≤ threshold peaks)
    2. RDF distributions (to avoid Gibbs ringing at zero boundary)

    Args:
        n_peaks: Number of peaks in original distribution
        threshold: Maximum peaks for using Gaussian (default: 2)
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'

    Returns:
        True if Gaussian smoothing should be used
    """
    # For RDF, always prefer Gaussian to avoid boundary artifacts
    if dist_type == 'rdf':
        return True
    return n_peaks <= threshold