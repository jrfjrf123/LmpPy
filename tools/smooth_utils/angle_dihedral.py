"""
Angle and Dihedral distribution smoothing functions.
Angle 和 Dihedral 分布平滑函数

Key differences:
- Angle: 非周期性, 范围 [0°, 180°], 可复用 bond 的 harmonic boundary 方法
- Dihedral: 周期性边界, 范围 [-180°, +180°], 需要特殊的周期性处理
"""

import numpy as np
from typing import Tuple, Dict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter

# Relative imports for package
from .constants import KB, DEFAULT_TEMPERATURE
from .quality import calculate_potential_and_force


def smooth_angle_with_harmonic_boundary(x: np.ndarray, P: np.ndarray,
                                         idx_valid_start: int, idx_valid_end: int,
                                         sg_window: int = 11, sg_poly: int = 3,
                                         threshold: float = 0.001,
                                         F_max: float = 500.0,
                                         temperature: float = DEFAULT_TEMPERATURE,
                                         verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Apply smoothing to angle distributions with harmonic boundary potential.

    Angle 分布特点:
    - 单位是角度 (degrees) 而非 Å
    - 范围: [0°, 180°]
    - 边界: θ → 0° 时 P → 0 (类似 bond 左边界)
    - 边界: θ → 180° 时 P → 0 (类似 bond 右边界)

    复用 smooth_bond_with_harmonic_boundary 的逻辑,
    仅在报告和度量中体现角度单位.

    Args:
        x: Coordinate array (degrees)
        P: Probability density array
        idx_valid_start: Start index of valid data range
        idx_valid_end: End index of valid data range
        sg_window: Savitzky-Golay filter window size
        sg_poly: Savitzky-Golay polynomial order
        threshold: P threshold for boundary definition
        F_max: Maximum force at boundary extremes (kcal/mol/degree)
        temperature: Temperature in Kelvin
        verbose: Whether to print debug information

    Returns:
        P_smooth: Smoothed probability density array
        metrics: Dictionary with smoothing metrics
    """
    n = len(P)
    dtheta = x[1] - x[0] if len(x) > 1 else 1.0  # Bin width in degrees
    kT = KB * temperature

    # Find the actual first and last non-zero points
    nonzero_idx = np.where(P > 1e-10)[0]
    if len(nonzero_idx) == 0:
        return P.copy(), {'method': 'none', 'reason': 'all zeros', 'dist_type': 'angle'}

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
            print("    No core region found, using pure S-G")
        return P_sg, {'method': 'sg_only', 'dist_type': 'angle'}

    left_core = core_indices[0]
    right_core = core_indices[-1]

    if verbose:
        print(f"    Angle harmonic boundary: threshold={threshold}")
        print(f"    Core region: θ = [{x[left_core]:.1f}°, {x[right_core]:.1f}°]")

    # Step 3: Compute U for core region
    U_core = -kT * np.log(P_sg)

    # Get connection point values
    U_left = U_core[left_core]
    F_all = -np.gradient(U_core, x)  # Force in kcal/mol/degree
    F_left = F_all[left_core] if left_core > 0 else 0.0

    U_right = U_core[right_core]
    F_right = F_all[right_core] if right_core < n - 1 else 0.0

    # Step 4: Build result array
    P_result = np.zeros(n)

    # Core region: use S-G smoothed P
    P_result[left_core:right_core+1] = P_sg[left_core:right_core+1]

    # === Left boundary: Harmonic potential ===
    if left_core > first_nz:
        theta_left = x[left_core]
        theta_first = x[first_nz]
        delta_theta = theta_left - theta_first

        if delta_theta > 0:
            k_left = (F_max - F_left) / delta_theta

            for i in range(first_nz, left_core):
                dtheta_from_left = theta_left - x[i]
                U_harmonic = 0.5 * k_left * dtheta_from_left**2 + F_left * dtheta_from_left + U_left
                P_result[i] = np.exp(-U_harmonic / kT)

    # === Right boundary: Harmonic potential ===
    if right_core < last_nz:
        theta_right = x[right_core]
        theta_last = x[last_nz]
        delta_theta = theta_last - theta_right

        if delta_theta > 0:
            k_right = (F_right + F_max) / delta_theta

            for i in range(right_core + 1, last_nz + 1):
                dtheta_from_right = x[i] - theta_right
                U_harmonic = 0.5 * k_right * dtheta_from_right**2 - F_right * dtheta_from_right + U_right
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
            roughness_raw = np.sum(d2F_raw[finite_mask_raw]**2) * dtheta
        else:
            roughness_raw = 1.0

        if np.any(finite_mask_smooth):
            roughness_smooth = np.sum(d2F_smooth[finite_mask_smooth]**2) * dtheta
        else:
            roughness_smooth = roughness_raw

        reduction_ratio = (roughness_raw - roughness_smooth) / roughness_raw if roughness_raw > 0 else 0.0
    else:
        roughness_raw = 0.0
        roughness_smooth = 0.0
        reduction_ratio = 0.0

    metrics = {
        'method': 'harmonic_boundary_angle',
        'dist_type': 'angle',
        'sg_window': actual_sg_window,
        'sg_poly': sg_poly,
        'threshold': threshold,
        'F_max': F_max,
        'first_nonzero': first_nz,
        'last_nonzero': last_nz,
        'left_core': left_core,
        'right_core': right_core,
        'theta_range_deg': (x[first_nz], x[last_nz]),
        'roughness_raw': roughness_raw,
        'roughness_smooth': roughness_smooth,
        'roughness_reduction_ratio': reduction_ratio
    }

    if verbose:
        print(f"    Roughness reduction: {reduction_ratio*100:.1f}%")

    return P_result, metrics


def smooth_dihedral_periodic(phi: np.ndarray, P: np.ndarray,
                              sigma: float = 3.0,
                              sigma_range: list = None,
                              temperature: float = DEFAULT_TEMPERATURE,
                              verbose: bool = False) -> Tuple[np.ndarray, Dict]:
    """
    Apply periodic Gaussian smoothing to dihedral distributions.

    Dihedral 分布特点:
    - 周期性边界: φ = -180° 和 φ = +180° 是同一个物理点
    - 范围: [-180°, +180°]
    - 必须满足: P(-180°) = P(+180°)

    方法步骤:
    1. 周期性扩展: P_extended = [P[-n_pad:], P, P[:n_pad]]
    2. 高斯平滑扩展数据
    3. 提取中心部分
    4. 强制边界连续: P_smooth[0] = P_smooth[-1]
    5. 归一化确保非负

    Args:
        phi: Coordinate array (degrees), range [-180, 180]
        P: Probability density array
        sigma: Gaussian smoothing sigma (in bins, not degrees)
        sigma_range: List of sigma values to try (for optimization)
        temperature: Temperature in Kelvin
        verbose: Whether to print debug information

    Returns:
        P_smooth: Smoothed probability density array
        metrics: Dictionary with smoothing metrics
    """
    n = len(P)
    dphi = phi[1] - phi[0] if len(phi) > 1 else 1.0

    # Ensure P is non-negative
    P = np.maximum(P, 0.0)

    # If sigma_range provided, try multiple values
    if sigma_range is not None:
        best_sigma = sigma_range[0]
        best_score = -float('inf')
        best_P_smooth = None

        for s in sigma_range:
            P_trial, _ = _apply_periodic_gaussian(P, s)
            score = _evaluate_periodic_smooth(P, P_trial, phi, temperature)

            if score > best_score:
                best_score = score
                best_sigma = s
                best_P_smooth = P_trial

        P_smooth = best_P_smooth
        sigma_used = best_sigma
    else:
        P_smooth, sigma_used = _apply_periodic_gaussian(P, sigma)

    # Calculate metrics
    U_raw, F_raw = calculate_potential_and_force(phi, P, temperature)
    U_smooth, F_smooth = calculate_potential_and_force(phi, P_smooth, temperature)

    d2F_raw = np.gradient(np.gradient(F_raw, phi), phi)
    d2F_smooth = np.gradient(np.gradient(F_smooth, phi), phi)

    finite_mask_raw = np.isfinite(d2F_raw)
    finite_mask_smooth = np.isfinite(d2F_smooth)

    if np.any(finite_mask_raw):
        roughness_raw = np.sum(d2F_raw[finite_mask_raw]**2) * dphi
    else:
        roughness_raw = 1.0

    if np.any(finite_mask_smooth):
        roughness_smooth = np.sum(d2F_smooth[finite_mask_smooth]**2) * dphi
    else:
        roughness_smooth = roughness_raw

    reduction_ratio = (roughness_raw - roughness_smooth) / roughness_raw if roughness_raw > 0 else 0.0

    # Check boundary continuity
    boundary_diff = abs(P_smooth[0] - P_smooth[-1])
    force_boundary_diff = abs(F_smooth[0] - F_smooth[-1]) if np.isfinite(F_smooth[0]) and np.isfinite(F_smooth[-1]) else float('inf')

    metrics = {
        'method': 'periodic_gaussian',
        'dist_type': 'dihedral',
        'sigma': sigma_used,
        'boundary_continuity_P': boundary_diff,
        'boundary_continuity_F': force_boundary_diff,
        'roughness_raw': roughness_raw,
        'roughness_smooth': roughness_smooth,
        'roughness_reduction_ratio': reduction_ratio
    }

    if verbose:
        print(f"    Periodic Gaussian sigma={sigma_used}")
        print(f"    Boundary continuity: ΔP={boundary_diff:.2e}, ΔF={force_boundary_diff:.2e}")
        print(f"    Roughness reduction: {reduction_ratio*100:.1f}%")

    return P_smooth, metrics


def _apply_periodic_gaussian(P: np.ndarray, sigma: float) -> Tuple[np.ndarray, float]:
    """
    Apply periodic Gaussian smoothing to a distribution.

    Internal function that performs the actual smoothing.

    Args:
        P: Probability density array
        sigma: Gaussian sigma (in bins)

    Returns:
        P_smooth: Smoothed array
        sigma_used: The sigma value used
    """
    n = len(P)
    pad_width = int(3 * sigma) + 1  # Ensure enough padding

    # Periodic extension
    P_extended = np.concatenate([P[-pad_width:], P, P[:pad_width]])

    # Gaussian smoothing
    P_smooth_ext = gaussian_filter1d(P_extended, sigma=sigma)

    # Extract center part
    P_smooth = P_smooth_ext[pad_width:pad_width + n]

    # Force boundary continuity by averaging
    boundary_avg = (P_smooth[0] + P_smooth[-1]) / 2
    P_smooth[0] = boundary_avg
    P_smooth[-1] = boundary_avg

    # Ensure non-negative
    P_smooth = np.maximum(P_smooth, 0.0)

    return P_smooth, sigma


def _evaluate_periodic_smooth(P_raw: np.ndarray, P_smooth: np.ndarray,
                              x: np.ndarray, temperature: float) -> float:
    """
    Evaluate the quality of periodic smoothing.

    Internal function for sigma optimization.

    Args:
        P_raw: Original distribution
        P_smooth: Smoothed distribution
        x: Coordinate array
        temperature: Temperature in Kelvin

    Returns:
        Quality score (higher is better)
    """
    # Calculate roughness reduction
    U_raw, F_raw = calculate_potential_and_force(x, P_raw, temperature)
    U_smooth, F_smooth = calculate_potential_and_force(x, P_smooth, temperature)

    d2F_raw = np.gradient(np.gradient(F_raw, x), x)
    d2F_smooth = np.gradient(np.gradient(F_smooth, x), x)

    dx = x[1] - x[0] if len(x) > 1 else 1.0

    finite_raw = np.isfinite(d2F_raw)
    finite_smooth = np.isfinite(d2F_smooth)

    if np.any(finite_raw):
        roughness_raw = np.sum(d2F_raw[finite_raw]**2) * dx
    else:
        roughness_raw = 1.0

    if np.any(finite_smooth):
        roughness_smooth = np.sum(d2F_smooth[finite_smooth]**2) * dx
    else:
        roughness_smooth = roughness_raw

    reduction = (roughness_raw - roughness_smooth) / roughness_raw if roughness_raw > 0 else 0.0

    # Penalize boundary discontinuity
    boundary_diff = abs(P_smooth[0] - P_smooth[-1])
    boundary_penalty = 1.0 / (1.0 + boundary_diff * 1000)

    # Check shape preservation (correlation in valid region)
    valid_mask = P_raw > 1e-10
    if np.sum(valid_mask) > 10:
        correlation = np.corrcoef(P_raw[valid_mask], P_smooth[valid_mask])[0, 1]
        if np.isnan(correlation):
            correlation = 0.5
    else:
        correlation = 0.5

    # Combined score
    score = reduction * boundary_penalty * (correlation ** 2)

    return score