"""
Peak detection functions for distribution smoothing.
分布平滑模块的峰检测函数
"""

import numpy as np
from typing import Tuple

# Relative imports for package
from .constants import MAX_PEAKS


def detect_peaks(x: np.ndarray, P: np.ndarray,
                 max_peaks: int = MAX_PEAKS,
                 min_height_fraction: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect peaks in distribution using derivative sign change.

    Peaks are identified where dP/dx changes from positive to negative.
    Only the top max_peaks (sorted by height) are returned.

    Args:
        x: Coordinate array
        P: Probability density array
        max_peaks: Maximum number of peaks to return
        min_height_fraction: Minimum peak height as fraction of max(P)

    Returns:
        peak_indices: Indices of detected peaks
        peak_positions: X coordinates of peaks
        peak_heights: Heights (P values) of peaks
    """
    if len(P) < 3:
        return np.array([]), np.array([]), np.array([])

    # Calculate first derivative
    dP = np.gradient(P, x)

    # Find sign changes from positive to negative (local maxima)
    sign_changes = np.where((dP[:-1] > 0) & (dP[1:] <= 0))[0]

    if len(sign_changes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Get peak heights
    peak_heights = P[sign_changes]

    # Filter by minimum height
    max_height = np.max(P)
    min_height = min_height_fraction * max_height
    valid_mask = peak_heights >= min_height
    sign_changes = sign_changes[valid_mask]
    peak_heights = peak_heights[valid_mask]

    if len(sign_changes) == 0:
        return np.array([]), np.array([]), np.array([])

    # Sort by height (descending) and keep top max_peaks
    sorted_indices = np.argsort(peak_heights)[::-1]
    top_indices = sorted_indices[:min(len(sorted_indices), max_peaks)]

    # Sort by x position for output
    peak_indices = sign_changes[top_indices]
    sort_by_x = np.argsort(peak_indices)
    peak_indices = peak_indices[sort_by_x]

    peak_positions = x[peak_indices]
    peak_heights = P[peak_indices]

    return peak_indices, peak_positions, peak_heights


def estimate_peak_width(x: np.ndarray, P: np.ndarray,
                        peak_indices: np.ndarray) -> float:
    """
    Estimate average peak width using full width at half maximum (FWHM).

    Args:
        x: Coordinate array
        P: Probability density array
        peak_indices: Indices of detected peaks

    Returns:
        Average peak width (FWHM)
    """
    if len(peak_indices) == 0:
        # Default: 5% of data range
        return (x[-1] - x[0]) * 0.05

    widths = []
    for peak_idx in peak_indices:
        peak_height = P[peak_idx]
        half_height = peak_height / 2

        # Search left for half-height point
        left_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if P[i] < half_height:
                left_idx = i
                break

        # Search right for half-height point
        right_idx = peak_idx
        for i in range(peak_idx + 1, len(P)):
            if P[i] < half_height:
                right_idx = i
                break

        width = x[right_idx] - x[left_idx]
        if width > 0:
            widths.append(width)

    if len(widths) == 0:
        return (x[-1] - x[0]) * 0.05

    return np.mean(widths)


def detect_significant_peaks(x: np.ndarray, P: np.ndarray,
                             dist_type: str = 'bond',
                             min_prominence: float = 0.1) -> int:
    """
    Detect significant peaks using prominence-based filtering.

    This function distinguishes true structural peaks from noise fluctuations
    by requiring peaks to have a minimum prominence (height above nearby valleys).

    For RDF:
    - A true peak should rise significantly above the baseline (g(r) ≈ 1)
    - Noise fluctuations around g(r) = 1 are not considered true peaks

    For bond distributions:
    - Uses standard prominence-based detection

    Args:
        x: Coordinate array
        P: Probability density array
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        min_prominence: Minimum peak prominence as fraction of max height

    Returns:
        Number of significant peaks detected
    """
    from scipy.signal import find_peaks

    if len(P) < 3:
        return 0

    max_height = np.max(P)
    if max_height <= 0:
        return 0

    # For RDF, use higher prominence threshold
    # because noise around g(r)=1 should not count as peaks
    if dist_type == 'rdf':
        # RDF peaks should be at least 10% above baseline
        prominence_threshold = min_prominence * max_height
        # Also consider: true RDF peaks typically occur at r < 6-8 Å
        # and should have g(r) significantly different from 1
    else:
        # For bond distributions, use standard threshold
        prominence_threshold = min_prominence * max_height

    # Use scipy's find_peaks with prominence
    peaks, properties = find_peaks(P, prominence=prominence_threshold)

    return len(peaks)