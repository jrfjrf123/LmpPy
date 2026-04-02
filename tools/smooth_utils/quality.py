"""
Quality assessment functions for distribution smoothing.
分布平滑模块的质量评估函数
"""

import numpy as np
from typing import Tuple, Dict, List, Any

# Relative imports for package
from .constants import (
    KB, DEFAULT_TEMPERATURE,
    WEIGHT_PEAK_POS, WEIGHT_FORCE_SMOOTH, WEIGHT_PEAK_HEIGHT,
    FORCE_CONSISTENCY_THRESHOLD
)


def calculate_potential_and_force(x: np.ndarray, P: np.ndarray,
                                  temperature: float = DEFAULT_TEMPERATURE) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate potential energy and force from distribution.

    Uses Boltzmann inversion: U(r) = -kT * ln(P(r))
    Force: F(r) = -dU/dr

    Args:
        x: Coordinate array
        P: Probability density array
        temperature: Temperature in Kelvin

    Returns:
        U: Potential energy array (kcal/mol)
        F: Force array (kcal/mol/Angstrom or kcal/mol/degree)
    """
    # Avoid log(0) by setting minimum value
    P_safe = np.maximum(P, 1e-15)

    # Boltzmann inversion
    U = -KB * temperature * np.log(P_safe)

    # Normalize so minimum is 0
    finite_mask = np.isfinite(U)
    if np.any(finite_mask):
        U = U - np.min(U[finite_mask])

    # Calculate force as negative derivative of potential
    F = -np.gradient(U, x)

    return U, F


def calculate_quality_metrics(x: np.ndarray, P_raw: np.ndarray, P_smooth: np.ndarray,
                             peak_positions_raw: np.ndarray, peak_heights_raw: np.ndarray,
                             peak_positions_smooth: np.ndarray, peak_heights_smooth: np.ndarray,
                             dr: float, temperature: float = DEFAULT_TEMPERATURE,
                             idx_valid_start: int = None, idx_valid_end: int = None) -> Tuple[Dict[str, float], float]:
    """
    Calculate quality metrics for smoothed distribution.

    Metrics:
    1. Peak position fidelity (weight: 0.50)
    2. Force curve smoothness (weight: 0.35)
    3. Peak height fidelity (weight: 0.15)

    Args:
        x: Coordinate array
        P_raw: Original probability density
        P_smooth: Smoothed probability density
        peak_positions_raw: Peak positions in original data
        peak_heights_raw: Peak heights in original data
        peak_positions_smooth: Peak positions in smoothed data
        peak_heights_smooth: Peak heights in smoothed data
        dr: Bin width
        temperature: Temperature in Kelvin
        idx_valid_start: Start index of valid range (if None, use full range)
        idx_valid_end: End index of valid range (if None, use full range)

    Returns:
        metrics: Dictionary of individual metrics
        total_score: Weighted sum of scores (0-1)
    """
    metrics = {}

    # Set default valid range if not provided
    if idx_valid_start is None:
        idx_valid_start = 0
    if idx_valid_end is None:
        idx_valid_end = len(x) - 1

    # 1. Peak Position Fidelity
    if len(peak_positions_raw) > 0 and len(peak_positions_smooth) > 0:
        pos_errors = []
        for pos_raw in peak_positions_raw:
            # Find closest peak in smoothed data
            distances = np.abs(peak_positions_smooth - pos_raw)
            min_dist = np.min(distances)
            pos_errors.append(min_dist)

        avg_pos_error = np.mean(pos_errors) / dr  # Normalize to bin widths
        score_pos = 1.0 / (1.0 + avg_pos_error)
    else:
        avg_pos_error = 0.0
        score_pos = 1.0 if len(peak_positions_raw) == 0 else 0.5

    metrics['peak_position_error_bins'] = avg_pos_error
    metrics['peak_position_error_absolute'] = avg_pos_error * dr
    metrics['score_peak_position'] = score_pos

    # 2. Force Curve Smoothness (only within valid range)
    # Calculate roughness for both raw and smoothed data
    U_raw, F_raw = calculate_potential_and_force(x, P_raw, temperature)
    U_smooth, F_smooth = calculate_potential_and_force(x, P_smooth, temperature)

    # Calculate second derivative of force (roughness measure)
    d2F_raw = np.gradient(np.gradient(F_raw, x), x)
    d2F_smooth = np.gradient(np.gradient(F_smooth, x), x)

    # Create mask for valid range AND finite values
    valid_range_mask = np.zeros(len(x), dtype=bool)
    valid_range_mask[idx_valid_start:idx_valid_end+1] = True
    finite_mask_raw = np.isfinite(d2F_raw) & valid_range_mask
    finite_mask_smooth = np.isfinite(d2F_smooth) & valid_range_mask

    # Calculate roughness for raw data
    if np.any(finite_mask_raw):
        roughness_raw = np.sum(d2F_raw[finite_mask_raw]**2) * dr
    else:
        roughness_raw = 0.0

    # Calculate roughness for smoothed data
    if np.any(finite_mask_smooth):
        roughness_smooth = np.sum(d2F_smooth[finite_mask_smooth]**2) * dr
    else:
        roughness_smooth = 0.0

    # Calculate score using relative improvement ratio
    if roughness_raw + roughness_smooth > 0:
        score_force = roughness_raw / (roughness_raw + roughness_smooth)
    else:
        # Both are zero - perfect case
        score_force = 1.0

    # Also calculate reduction ratio for reporting
    if roughness_raw > 0:
        roughness_reduction_ratio = (roughness_raw - roughness_smooth) / roughness_raw
    else:
        roughness_reduction_ratio = 0.0 if roughness_smooth == 0 else -1.0

    metrics['force_roughness_raw'] = roughness_raw
    metrics['force_roughness'] = roughness_smooth
    metrics['force_roughness_reduction_ratio'] = roughness_reduction_ratio
    metrics['score_force_smooth'] = score_force

    # 3. Peak Height Fidelity
    if len(peak_heights_raw) > 0 and len(peak_heights_smooth) > 0:
        height_errors = []
        n_peaks = min(len(peak_heights_raw), len(peak_heights_smooth))

        for i in range(n_peaks):
            h_raw = peak_heights_raw[i]
            h_smooth = peak_heights_smooth[i]
            if h_raw > 0:
                rel_error = np.abs(h_smooth - h_raw) / h_raw
                height_errors.append(rel_error)

        avg_height_error = np.mean(height_errors) if height_errors else 0.0
        score_height = 1.0 / (1.0 + avg_height_error)
    else:
        avg_height_error = 0.0
        score_height = 1.0 if len(peak_heights_raw) == 0 else 0.5

    metrics['peak_height_error_relative'] = avg_height_error
    metrics['score_peak_height'] = score_height

    # Calculate weighted total score
    total_score = (WEIGHT_PEAK_POS * score_pos +
                   WEIGHT_FORCE_SMOOTH * score_force +
                   WEIGHT_PEAK_HEIGHT * score_height)

    metrics['total_score'] = total_score

    return metrics, total_score


def check_force_consistency(x: np.ndarray, P_smooth: np.ndarray,
                           temperature: float = DEFAULT_TEMPERATURE,
                           threshold: float = FORCE_CONSISTENCY_THRESHOLD,
                           idx_valid_start: int = None, idx_valid_end: int = None) -> Tuple[int, int, List[int]]:
    """
    Check consistency between force values and -dU/dr.

    Simulates LAMMPS's validation logic to identify points where
    the second derivative of force is too large (indicating inconsistency).

    Args:
        x: Coordinate array
        P_smooth: Smoothed probability density
        temperature: Temperature in Kelvin
        threshold: Maximum allowed |d²F/dr²| before flagging inconsistency
        idx_valid_start: Start index of valid range (if None, use full range)
        idx_valid_end: End index of valid range (if None, use full range)

    Returns:
        n_inconsistent: Number of inconsistent points
        total_points: Total number of points checked
        inconsistent_indices: List of indices with inconsistent force
    """
    # Set default valid range if not provided
    if idx_valid_start is None:
        idx_valid_start = 0
    if idx_valid_end is None:
        idx_valid_end = len(x) - 1

    U, F = calculate_potential_and_force(x, P_smooth, temperature)

    # Calculate second derivative of force
    d2F = np.gradient(np.gradient(F, x), x)

    inconsistent_indices = []
    total_checked = 0

    # Only check within valid range
    check_start = max(1, idx_valid_start)
    check_end = min(len(F) - 1, idx_valid_end + 1)

    for i in range(check_start, check_end):
        if np.isfinite(F[i-1]) and np.isfinite(F[i]) and np.isfinite(F[i+1]):
            total_checked += 1

            # Check if second derivative exceeds threshold
            if np.isfinite(d2F[i]) and np.abs(d2F[i]) > threshold:
                inconsistent_indices.append(i)

    return len(inconsistent_indices), total_checked, inconsistent_indices


def match_peaks(peak_positions_raw: np.ndarray, peak_heights_raw: np.ndarray,
                peak_positions_smooth: np.ndarray, peak_heights_smooth: np.ndarray,
                max_distance: float = None) -> List[Dict[str, Any]]:
    """
    Match peaks between raw and smoothed distributions.

    Args:
        peak_positions_raw: Peak positions in original data
        peak_heights_raw: Peak heights in original data
        peak_positions_smooth: Peak positions in smoothed data
        peak_heights_smooth: Peak heights in smoothed data
        max_distance: Maximum distance for matching (None = no limit)

    Returns:
        List of dicts with matched peak information
    """
    matches = []
    used_smooth = set()

    for i, (pos_r, h_r) in enumerate(zip(peak_positions_raw, peak_heights_raw)):
        best_match = None
        best_dist = float('inf')
        best_j = None

        for j, (pos_s, h_s) in enumerate(zip(peak_positions_smooth, peak_heights_smooth)):
            if j in used_smooth:
                continue

            dist = np.abs(pos_s - pos_r)
            if max_distance is not None and dist > max_distance:
                continue

            if dist < best_dist:
                best_dist = dist
                best_match = (pos_s, h_s)
                best_j = j

        match_info = {
            'raw_index': i,
            'raw_position': pos_r,
            'raw_height': h_r,
            'smooth_position': best_match[0] if best_match else None,
            'smooth_height': best_match[1] if best_match else None,
            'position_shift': best_dist if best_match else None,
            'height_change': (best_match[1] - h_r) / h_r if best_match and h_r > 0 else None
        }
        matches.append(match_info)

        if best_j is not None:
            used_smooth.add(best_j)

    return matches