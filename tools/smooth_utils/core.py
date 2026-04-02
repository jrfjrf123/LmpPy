"""
Core smoothing functions for distribution smoothing.
分布平滑模块的核心平滑函数
"""

import numpy as np
from typing import Tuple, Dict, List
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d

# Relative imports for package
from .constants import (
    KB, DEFAULT_TEMPERATURE, DEFAULT_POLYORDER,
    BOUNDARY_RATIO, BOUNDARY_DECAY_LENGTH, EPSILON
)
from .peaks import detect_peaks
from .quality import calculate_potential_and_force


def adaptive_smooth(x: np.ndarray, P: np.ndarray, zone_mask: np.ndarray,
                    window_small: int = 11, window_large: int = 31,
                    polyorder: int = DEFAULT_POLYORDER) -> np.ndarray:
    """
    Apply adaptive smoothing based on zone assignment.

    Uses smaller window for peak regions to preserve peak shape,
    and larger window for valley regions to reduce noise.

    Args:
        x: Coordinate array
        P: Probability density array
        zone_mask: Zone assignment for each point
        window_small: Window size for peak regions
        window_large: Window size for valley regions
        polyorder: Polynomial order for Savitzky-Golay filter

    Returns:
        P_smooth: Smoothed probability density array
    """
    P_smooth = P.copy()
    n = len(P)

    # Ensure window sizes are odd and larger than polyorder
    window_small = max(polyorder + 2, window_small)
    window_large = max(polyorder + 2, window_large)
    if window_small % 2 == 0:
        window_small += 1
    if window_large % 2 == 0:
        window_large += 1

    # Apply Savitzky-Golay filter to entire array with different windows
    # Then selectively use based on zone

    # Small window smoothing (for peaks)
    if n >= window_small:
        try:
            P_small = savgol_filter(P, window_small, polyorder)
        except Exception:
            P_small = P.copy()
    else:
        P_small = P.copy()

    # Large window smoothing (for valleys)
    if n >= window_large:
        try:
            P_large = savgol_filter(P, window_large, polyorder)
        except Exception:
            P_large = P.copy()
    else:
        P_large = P.copy()

    # Assign smoothed values based on zone
    for i in range(n):
        zone = zone_mask[i]

        if zone == 0 or zone == 4:
            # Zero regions: keep original (usually zero or near-zero)
            P_smooth[i] = P[i]
        elif zone == 1:
            # Peak region: use small window
            P_smooth[i] = P_small[i]
        elif zone == 2:
            # Valley region: use large window
            P_smooth[i] = P_large[i]
        elif zone == 3:
            # RDF boundary: will be handled separately
            P_smooth[i] = P_large[i]  # Start with smoothed value

    # Ensure non-negative values
    P_smooth = np.maximum(P_smooth, 0.0)

    return P_smooth


def gaussian_smooth_with_constraint(x: np.ndarray, P: np.ndarray,
                                    original_n_peaks: int,
                                    sigma_range: List[float] = None,
                                    verbose: bool = False) -> Tuple[np.ndarray, float, int]:
    """
    Apply Gaussian smoothing with peak count constraint.

    For single-peak or dual-peak distributions, Gaussian smoothing produces
    smoother results without Gibbs ringing artifacts. This function automatically
    searches for the minimum sigma that ensures the smoothed distribution
    has at most the same number of peaks as the original.

    Args:
        x: Coordinate array
        P: Probability density array
        original_n_peaks: Number of peaks in original distribution
        sigma_range: List of sigma values to try (default: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])
        verbose: Whether to print debug information

    Returns:
        P_smooth: Smoothed probability density array
        sigma_used: The sigma value that was used
        n_peaks_smooth: Number of peaks in smoothed distribution
    """
    if sigma_range is None:
        sigma_range = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]

    best_sigma = sigma_range[-1]
    best_P_smooth = None
    best_n_peaks = float('inf')

    for sigma in sigma_range:
        P_smooth = gaussian_filter1d(P, sigma=sigma)

        # Ensure non-negative values
        P_smooth = np.maximum(P_smooth, 0.0)

        # Detect peaks in smoothed distribution
        peak_indices, _, _ = detect_peaks(x, P_smooth)
        n_peaks_smooth = len(peak_indices)

        if verbose:
            print(f"    Gaussian sigma={sigma:.1f}: {n_peaks_smooth} peaks (target ≤ {original_n_peaks})")

        # Check if peak constraint is satisfied
        if n_peaks_smooth <= original_n_peaks:
            return P_smooth, sigma, n_peaks_smooth

        # Keep track of best result in case no sigma satisfies constraint
        if n_peaks_smooth < best_n_peaks:
            best_n_peaks = n_peaks_smooth
            best_sigma = sigma
            best_P_smooth = P_smooth

    # If no sigma satisfies constraint, return the one with fewest peaks
    if best_P_smooth is None:
        best_P_smooth = gaussian_filter1d(P, sigma=sigma_range[-1])
        best_P_smooth = np.maximum(best_P_smooth, 0.0)

    if verbose:
        print(f"    Warning: Could not achieve peak count constraint, using sigma={best_sigma}")

    return best_P_smooth, best_sigma, best_n_peaks


def gaussian_smooth_for_rdf(x: np.ndarray, P: np.ndarray,
                            idx_valid_start: int, idx_valid_end: int,
                            sigma_range: List[float] = None,
                            temperature: float = DEFAULT_TEMPERATURE,
                            verbose: bool = False) -> Tuple[np.ndarray, float, Dict]:
    """
    Apply Gaussian smoothing optimized for RDF distributions with boundary protection.

    For RDF, the goal is to maximize roughness reduction while preserving
    the physical shape of g(r) AND protecting the zero-to-nonzero boundary.

    Boundary Protection Strategy:
    1. Only smooth the valid data region (idx_valid_start to idx_valid_end)
    2. Keep zero values in the excluded region unchanged
    3. Apply smooth transition blending at the boundary to avoid artifacts

    Args:
        x: Coordinate array
        P: Probability density array
        idx_valid_start: Start index of valid data range
        idx_valid_end: End index of valid data range
        sigma_range: List of sigma values to try
        temperature: Temperature in Kelvin
        verbose: Whether to print debug information

    Returns:
        P_smooth: Smoothed probability density array
        sigma_used: The sigma value that was used
        metrics: Dictionary with optimization metrics
    """
    if sigma_range is None:
        # Use moderate sigma range to balance smoothness and boundary preservation
        sigma_range = [1.0, 2.0, 3.0, 5.0, 7.0, 10.0]

    dr = x[1] - x[0] if len(x) > 1 else 1.0
    n = len(P)

    # Find the actual first non-zero point (more reliable than idx_valid_start for boundary)
    first_nonzero_idx = 0
    for i in range(n):
        if P[i] > 1e-10:
            first_nonzero_idx = i
            break

    if verbose:
        print(f"    Boundary protection: first non-zero at idx={first_nonzero_idx}, r={x[first_nonzero_idx]:.4f} Å")

    # Calculate raw roughness as baseline (only in valid range, excluding boundary)
    eval_start = min(first_nonzero_idx + 20, idx_valid_end)

    U_raw, F_raw = calculate_potential_and_force(x, P, temperature)
    d2F_raw = np.gradient(np.gradient(F_raw, x), x)

    valid_mask = np.zeros(n, dtype=bool)
    valid_mask[eval_start:idx_valid_end+1] = True
    finite_mask_raw = np.isfinite(d2F_raw) & valid_mask

    if np.any(finite_mask_raw):
        roughness_raw = np.sum(d2F_raw[finite_mask_raw]**2) * dr
    else:
        roughness_raw = 1.0

    best_sigma = sigma_range[0]
    best_P_smooth = None
    best_score = -float('inf')
    best_metrics = {}

    for sigma in sigma_range:
        # Strategy: smooth the entire array, then restore boundary region
        P_smooth_full = gaussian_filter1d(P, sigma=sigma)
        P_smooth_full = np.maximum(P_smooth_full, 0.0)

        # Create the final smoothed array with boundary protection
        P_smooth = P.copy()

        # Define the transition region width
        transition_width = int(3 * sigma)

        blend_start = first_nonzero_idx
        blend_end = min(first_nonzero_idx + transition_width, idx_valid_end)

        # Transition region: linear blend
        for i in range(blend_start, blend_end + 1):
            t = (i - blend_start) / max(1, blend_end - blend_start)
            blend_weight = t * t * (3 - 2 * t)  # Smoothstep function
            P_smooth[i] = P[i] * (1 - blend_weight) + P_smooth_full[i] * blend_weight

        # After transition: use smoothed values
        P_smooth[blend_end + 1:] = P_smooth_full[blend_end + 1:]

        # Ensure non-negative
        P_smooth = np.maximum(P_smooth, 0.0)

        # Calculate smoothed roughness
        U_smooth, F_smooth = calculate_potential_and_force(x, P_smooth, temperature)
        d2F_smooth = np.gradient(np.gradient(F_smooth, x), x)
        finite_mask_smooth = np.isfinite(d2F_smooth) & valid_mask

        if np.any(finite_mask_smooth):
            roughness_smooth = np.sum(d2F_smooth[finite_mask_smooth]**2) * dr
        else:
            roughness_smooth = roughness_raw

        reduction_ratio = (roughness_raw - roughness_smooth) / roughness_raw if roughness_raw > 0 else 0.0

        # Calculate shape fidelity
        eval_region = slice(blend_end + 1, idx_valid_end + 1)
        P_eval = P[eval_region]
        P_smooth_eval = P_smooth[eval_region]

        if len(P_eval) > 10 and np.std(P_eval) > 0 and np.std(P_smooth_eval) > 0:
            correlation = np.corrcoef(P_eval, P_smooth_eval)[0, 1]
        else:
            correlation = 1.0

        # Check boundary preservation
        boundary_check_region = slice(first_nonzero_idx, min(first_nonzero_idx + 10, n))
        boundary_error = np.mean(np.abs(P_smooth[boundary_check_region] - P[boundary_check_region]))
        boundary_penalty = 1.0 / (1.0 + boundary_error * 100)

        score = reduction_ratio * (correlation ** 2) * boundary_penalty

        if verbose:
            print(f"      sigma={sigma:5.1f}: reduction={reduction_ratio*100:+.1f}%, score={score:.4f}")

        if score > best_score:
            best_score = score
            best_sigma = sigma
            best_P_smooth = P_smooth.copy()
            best_metrics = {
                'roughness_raw': roughness_raw,
                'roughness_smooth': roughness_smooth,
                'reduction_ratio': reduction_ratio,
                'correlation': correlation,
                'boundary_error': boundary_error,
                'score': score
            }

    if best_P_smooth is None:
        best_P_smooth = P.copy()

    return best_P_smooth, best_sigma, best_metrics


def smooth_bond_with_harmonic_boundary(x: np.ndarray, P: np.ndarray,
                                        idx_valid_start: int, idx_valid_end: int,
                                        sg_window: int = 11, sg_poly: int = 3,
                                        threshold: float = 0.0015,
                                        F_max: float = 500.0,
                                        temperature: float = DEFAULT_TEMPERATURE,
                                        verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Apply smoothing to bond distributions with harmonic boundary potential.

    This simplified approach directly constructs smooth potential energy U(r) in
    boundary regions (where P < threshold), then reverse-engineers P via Boltzmann
    relation. This eliminates force oscillations completely at boundaries.

    Args:
        x: Coordinate array
        P: Probability density array
        idx_valid_start: Start index of valid data range
        idx_valid_end: End index of valid data range
        sg_window: Savitzky-Golay filter window size
        sg_poly: Savitzky-Golay polynomial order
        threshold: P threshold for boundary definition
        F_max: Maximum force at boundary extremes (kcal/mol/Å)
        temperature: Temperature in Kelvin
        verbose: Whether to print debug information

    Returns:
        P_smooth: Smoothed probability density array
        metrics: Dictionary with smoothing metrics
    """
    n = len(P)
    dr = x[1] - x[0] if len(x) > 1 else 1.0
    kT = KB * temperature

    # Find the actual first and last non-zero points
    nonzero_idx = np.where(P > 1e-10)[0]
    if len(nonzero_idx) == 0:
        return P.copy(), {'method': 'none', 'reason': 'all zeros'}

    first_nz = nonzero_idx[0]
    last_nz = nonzero_idx[-1]

    # Ensure S-G window is valid
    data_length = last_nz - first_nz + 1
    actual_sg_window = min(sg_window, data_length // 2 * 2 - 1)
    if actual_sg_window < sg_poly + 2:
        actual_sg_window = sg_poly + 2
    if actual_sg_window % 2 == 0:
        actual_sg_window += 1

    # Step 1: Apply S-G filter to entire P
    P_sg = savgol_filter(P, actual_sg_window, sg_poly)
    P_sg = np.maximum(P_sg, 1e-15)

    # Step 2: Find core region where P_sg >= threshold
    core_mask = P_sg >= threshold
    core_indices = np.where(core_mask)[0]

    if len(core_indices) == 0:
        if verbose:
            print("    No core region found (all P < threshold), using pure S-G")
        return P_sg, {'method': 'sg_only', 'reason': 'no_core_region'}

    left_core = core_indices[0]
    right_core = core_indices[-1]

    if verbose:
        print(f"    Harmonic boundary: threshold={threshold}, core region idx={left_core}-{right_core}")

    # Step 3: Compute U for core region
    U_core = -kT * np.log(P_sg)

    # Get connection point values
    U_left = U_core[left_core]
    F_all = -np.gradient(U_core, x)
    F_left = F_all[left_core] if left_core > 0 else 0.0

    U_right = U_core[right_core]
    F_right = F_all[right_core] if right_core < n - 1 else 0.0

    # Step 4: Build result array
    P_result = np.zeros(n)

    # Core region: use S-G smoothed P
    P_result[left_core:right_core+1] = P_sg[left_core:right_core+1]

    # === Left boundary: Harmonic potential ===
    if left_core > first_nz:
        r_left = x[left_core]
        r_first = x[first_nz]
        delta_r_left = r_left - r_first

        if delta_r_left > 0:
            k_left = (F_max - F_left) / delta_r_left

            for i in range(first_nz, left_core):
                r = x[i]
                dr_from_left = r_left - r
                U_harmonic = 0.5 * k_left * dr_from_left**2 + F_left * dr_from_left + U_left
                P_result[i] = np.exp(-U_harmonic / kT)

    # === Right boundary: Harmonic potential ===
    if right_core < last_nz:
        r_right = x[right_core]
        r_last = x[last_nz]
        delta_r_right = r_last - r_right

        if delta_r_right > 0:
            k_right = (F_right + F_max) / delta_r_right

            for i in range(right_core + 1, last_nz + 1):
                r = x[i]
                dr_from_right = r - r_right
                U_harmonic = 0.5 * k_right * dr_from_right**2 - F_right * dr_from_right + U_right
                P_result[i] = np.exp(-U_harmonic / kT)

    # Zero regions
    P_result[:first_nz] = 0.0
    P_result[last_nz+1:] = 0.0

    # Ensure non-negative
    P_result = np.maximum(P_result, 0.0)

    # Calculate metrics
    eval_start = left_core + 5
    eval_end = right_core - 5

    if eval_end > eval_start:
        U_raw, F_raw = calculate_potential_and_force(x, P, temperature)
        U_smooth, F_smooth = calculate_potential_and_force(x, P_result, temperature)

        d2F_raw = np.gradient(np.gradient(F_raw, x), x)
        d2F_smooth = np.gradient(np.gradient(F_smooth, x), x)

        eval_mask = np.zeros(n, dtype=bool)
        eval_mask[eval_start:eval_end+1] = True
        finite_mask_raw = np.isfinite(d2F_raw) & eval_mask
        finite_mask_smooth = np.isfinite(d2F_smooth) & eval_mask

        if np.any(finite_mask_raw):
            roughness_raw = np.sum(d2F_raw[finite_mask_raw]**2) * dr
        else:
            roughness_raw = 1.0

        if np.any(finite_mask_smooth):
            roughness_smooth = np.sum(d2F_smooth[finite_mask_smooth]**2) * dr
        else:
            roughness_smooth = roughness_raw

        reduction_ratio = (roughness_raw - roughness_smooth) / roughness_raw if roughness_raw > 0 else 0.0
    else:
        roughness_raw = 0.0
        roughness_smooth = 0.0
        reduction_ratio = 0.0

    metrics = {
        'method': 'harmonic_boundary',
        'sg_window': actual_sg_window,
        'sg_poly': sg_poly,
        'threshold': threshold,
        'F_max': F_max,
        'first_nonzero': first_nz,
        'last_nonzero': last_nz,
        'left_core': left_core,
        'right_core': right_core,
        'roughness_raw': roughness_raw,
        'roughness_smooth': roughness_smooth,
        'roughness_reduction_ratio': reduction_ratio
    }

    return P_result, metrics


def smooth_rdf_with_harmonic_left_boundary(x: np.ndarray, g: np.ndarray,
                                            idx_valid_start: int, idx_valid_end: int,
                                            sigma: float = 3.0,
                                            threshold: float = 0.1,
                                            F_max: float = 500.0,
                                            temperature: float = DEFAULT_TEMPERATURE,
                                            boundary_ratio: float = BOUNDARY_RATIO,
                                            decay_length: float = BOUNDARY_DECAY_LENGTH,
                                            epsilon: float = EPSILON,
                                            verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Apply smoothing to RDF with harmonic left boundary and g(r)→1⁻ right boundary.

    Args:
        x: Coordinate array
        g: RDF g(r) array
        idx_valid_start: Start index of valid data range
        idx_valid_end: End index of valid data range
        sigma: Gaussian smoothing sigma for core region
        threshold: g(r) threshold for left boundary definition
        F_max: Maximum force at left boundary extreme (kcal/mol/Å)
        temperature: Temperature in Kelvin
        boundary_ratio: Fraction of r_max where right boundary starts
        decay_length: Decay length for right boundary (Angstrom)
        epsilon: Target offset for right boundary (g → 1 - epsilon)
        verbose: Whether to print debug information

    Returns:
        g_smooth: Smoothed RDF array
        metrics: Dictionary with smoothing metrics
    """
    n = len(g)
    dr = x[1] - x[0] if len(x) > 1 else 1.0
    kT = KB * temperature

    # Find the actual first and last non-zero points
    nonzero_idx = np.where(g > 1e-10)[0]
    if len(nonzero_idx) == 0:
        return g.copy(), {'method': 'harmonic_left_rdf', 'error': 'all zeros'}

    first_nz = nonzero_idx[0]
    last_nz = nonzero_idx[-1]

    if verbose:
        print(f"    RDF data range: idx {first_nz}-{last_nz}, r = [{x[first_nz]:.3f}, {x[last_nz]:.3f}] Å")

    # Step 1: Apply Gaussian smoothing to entire valid region
    g_gaussian = gaussian_filter1d(g, sigma=sigma)
    g_gaussian = np.maximum(g_gaussian, 1e-15)

    # Step 2: Find left core region where g_gaussian >= threshold
    core_mask = g_gaussian >= threshold
    core_indices = np.where(core_mask)[0]

    if len(core_indices) == 0:
        if verbose:
            print(f"    Warning: No points above threshold {threshold}, using pure Gaussian")
        return g_gaussian, {'method': 'gaussian_fallback', 'sigma': sigma}

    left_core = core_indices[0]

    # Right boundary starts at boundary_ratio * r_max
    r_max = x[last_nz]
    r_boundary_start = boundary_ratio * r_max
    right_boundary_idx = np.searchsorted(x, r_boundary_start)
    right_boundary_idx = min(right_boundary_idx, last_nz)

    if verbose:
        print(f"    Left boundary: idx 0-{left_core}, r < {x[left_core]:.3f} Å")
        print(f"    Core region: idx {left_core}-{right_boundary_idx}")
        print(f"    Right boundary: idx {right_boundary_idx}-{last_nz}")

    # Step 3: Compute U for Gaussian-smoothed core region
    U_gaussian = -kT * np.log(g_gaussian)

    # Get connection point values at left core boundary
    U_left = U_gaussian[left_core]
    if left_core > 0 and left_core < n - 1:
        F_left = -(U_gaussian[left_core + 1] - U_gaussian[left_core - 1]) / (2 * dr)
    elif left_core > 0:
        F_left = -(U_gaussian[left_core] - U_gaussian[left_core - 1]) / dr
    else:
        F_left = 0.0

    # Step 4: Build result array
    g_result = np.zeros(n)

    # === Left boundary: Harmonic potential ===
    if left_core > first_nz:
        r_left = x[left_core]
        r_first = x[first_nz]
        delta_r = r_left - r_first

        if delta_r > 1e-6:
            k_left = (F_max - F_left) / delta_r
        else:
            k_left = 0.0

        for i in range(first_nz, left_core):
            dr_i = r_left - x[i]
            U_i = 0.5 * k_left * dr_i**2 + F_left * dr_i + U_left
            g_result[i] = np.exp(-U_i / kT)

    # === Core region: use Gaussian smoothed g(r) ===
    g_result[left_core:right_boundary_idx+1] = g_gaussian[left_core:right_boundary_idx+1]

    # === Right boundary: exponential decay to (1 - epsilon) ===
    target = 1.0 - epsilon
    r_transition = x[right_boundary_idx]

    for i in range(right_boundary_idx + 1, last_nz + 1):
        dist_from_transition = x[i] - r_transition
        weight = np.exp(-dist_from_transition / decay_length)
        g_result[i] = g_gaussian[i] * weight + target * (1 - weight)

    # Zero regions
    g_result[:first_nz] = 0.0
    g_result[last_nz+1:] = 0.0

    # Ensure non-negative
    g_result = np.maximum(g_result, 0.0)

    # Calculate metrics
    eval_start = left_core + 5
    eval_end = right_boundary_idx - 5

    if eval_end > eval_start:
        U_raw, F_raw = calculate_potential_and_force(x, g, temperature)
        U_smooth, F_smooth = calculate_potential_and_force(x, g_result, temperature)

        d2F_raw = np.gradient(np.gradient(F_raw, x), x)
        d2F_smooth = np.gradient(np.gradient(F_smooth, x), x)

        eval_mask = np.zeros(n, dtype=bool)
        eval_mask[eval_start:eval_end+1] = True
        finite_raw = np.isfinite(d2F_raw) & eval_mask
        finite_smooth = np.isfinite(d2F_smooth) & eval_mask

        if np.any(finite_raw):
            roughness_raw = np.sum(d2F_raw[finite_raw]**2) * dr
        else:
            roughness_raw = 1.0

        if np.any(finite_smooth):
            roughness_smooth = np.sum(d2F_smooth[finite_smooth]**2) * dr
        else:
            roughness_smooth = roughness_raw

        reduction_ratio = (roughness_raw - roughness_smooth) / roughness_raw if roughness_raw > 0 else 0.0
    else:
        roughness_raw = 1.0
        roughness_smooth = 1.0
        reduction_ratio = 0.0

    metrics = {
        'method': 'harmonic_left_rdf',
        'sigma': sigma,
        'threshold': threshold,
        'F_max': F_max,
        'first_nonzero': first_nz,
        'last_nonzero': last_nz,
        'left_core': left_core,
        'right_boundary_idx': right_boundary_idx,
        'roughness_raw': roughness_raw,
        'roughness_smooth': roughness_smooth,
        'roughness_reduction_ratio': reduction_ratio
    }

    return g_result, metrics


def apply_rdf_boundary(x: np.ndarray, g_r: np.ndarray, zone_mask: np.ndarray,
                       decay_length: float = BOUNDARY_DECAY_LENGTH,
                       epsilon: float = EPSILON) -> np.ndarray:
    """
    Apply boundary constraint for RDF: g(r) → (1 - ε) at large r.

    Args:
        x: Coordinate array (radial distance)
        g_r: Radial distribution function values
        zone_mask: Zone assignment array
        decay_length: Characteristic length for exponential decay (Angstrom)
        epsilon: Small positive value (g → 1 - epsilon)

    Returns:
        g_smooth: RDF with boundary constraint applied
    """
    g_smooth = g_r.copy()

    # Find boundary region (zone 3)
    boundary_mask = zone_mask == 3
    if not np.any(boundary_mask):
        return g_smooth

    # Find transition point
    boundary_indices = np.where(boundary_mask)[0]
    if len(boundary_indices) == 0:
        return g_smooth

    r_transition = x[boundary_indices[0]]

    # Target value: 1 - epsilon
    target = 1.0 - epsilon

    # Apply exponential decay toward target
    for i in boundary_indices:
        dist_from_transition = x[i] - r_transition
        weight = np.exp(-dist_from_transition / decay_length)
        g_smooth[i] = g_r[i] * weight + target * (1 - weight)

    return g_smooth


def smooth_zone_transitions(P_smooth: np.ndarray, zone_mask: np.ndarray,
                            transition_width: int = 3) -> np.ndarray:
    """
    Smooth transitions between zones to avoid discontinuities.

    Args:
        P_smooth: Smoothed probability density array
        zone_mask: Zone assignment array
        transition_width: Number of points for transition smoothing

    Returns:
        P_smoothed: Array with smoothed zone transitions
    """
    P_result = P_smooth.copy()
    n = len(P_smooth)

    # Find zone boundaries
    for i in range(1, n):
        if zone_mask[i] != zone_mask[i-1]:
            # Apply local averaging around the boundary
            for j in range(max(1, i-1), min(n-1, i+2)):
                P_result[j] = (P_smooth[j-1] + P_smooth[j] + P_smooth[j+1]) / 3

    return P_result