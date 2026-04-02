"""
Preprocessing functions for distribution smoothing.
分布平滑模块的预处理函数
"""

import numpy as np
from typing import Tuple

# Relative imports for package
from .constants import DEFAULT_CDF_PERCENT, DEFAULT_THRESHOLD


def find_valid_range(x: np.ndarray, P: np.ndarray,
                     cdf_percent: float = DEFAULT_CDF_PERCENT) -> Tuple[int, int, float, float]:
    """
    Find valid data range based on cumulative distribution.

    Identifies the interval containing cdf_percent (default 99%) of the data.

    Args:
        x: Coordinate array
        P: Probability density array
        cdf_percent: Fraction of data to include (0 < cdf_percent <= 1)

    Returns:
        idx_start: Start index of valid range
        idx_end: End index of valid range
        r_start: Start coordinate value
        r_end: End coordinate value
    """
    if len(P) == 0:
        return 0, 0, 0.0, 0.0

    # Handle case where all values are zero
    total = np.sum(P)
    if total <= 0:
        # Find first and last non-zero indices
        nonzero_idx = np.where(P > 0)[0]
        if len(nonzero_idx) == 0:
            return 0, len(x) - 1, x[0], x[-1]
        return nonzero_idx[0], nonzero_idx[-1], x[nonzero_idx[0]], x[nonzero_idx[-1]]

    # Calculate cumulative distribution
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    cdf = np.cumsum(P) * dx
    cdf_normalized = cdf / cdf[-1]

    # Find interval containing cdf_percent of data
    lower_bound = (1 - cdf_percent) / 2  # e.g., 0.005 for 99%
    upper_bound = 1 - lower_bound         # e.g., 0.995 for 99%

    idx_start = np.searchsorted(cdf_normalized, lower_bound)
    idx_end = np.searchsorted(cdf_normalized, upper_bound)

    # Ensure indices are valid
    idx_start = max(0, min(idx_start, len(x) - 1))
    idx_end = max(idx_start, min(idx_end, len(x) - 1))

    return idx_start, idx_end, x[idx_start], x[idx_end]


def interpolate_zeros(x: np.ndarray, P: np.ndarray,
                      idx_start: int, idx_end: int,
                      threshold: float = DEFAULT_THRESHOLD) -> np.ndarray:
    """
    Interpolate zero/near-zero values within valid data range.

    Rules:
    - Values outside valid range [idx_start, idx_end] are unchanged
    - Within valid range, if P[i] < threshold and there are valid values
      on both sides, linear interpolation is applied

    Args:
        x: Coordinate array
        P: Probability density array
        idx_start: Start index of valid range
        idx_end: End index of valid range
        threshold: Threshold below which values are considered "zero"

    Returns:
        P_interp: Interpolated probability density array
    """
    P_interp = P.copy()

    if idx_end <= idx_start:
        return P_interp

    # Find all indices within valid range that need interpolation
    for i in range(idx_start, idx_end + 1):
        if P_interp[i] < threshold:
            # Search left for valid value
            left_idx = None
            left_val = None
            for j in range(i - 1, idx_start - 1, -1):
                if P_interp[j] >= threshold:
                    left_idx = j
                    left_val = P_interp[j]
                    break

            # Search right for valid value
            right_idx = None
            right_val = None
            for j in range(i + 1, idx_end + 1):
                if P_interp[j] >= threshold:
                    right_idx = j
                    right_val = P_interp[j]
                    break

            # If both sides have valid values, interpolate
            if left_idx is not None and right_idx is not None:
                t = (x[i] - x[left_idx]) / (x[right_idx] - x[left_idx])
                P_interp[i] = left_val + t * (right_val - left_val)

    return P_interp