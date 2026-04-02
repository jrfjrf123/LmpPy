"""
Optimization functions for distribution smoothing.
分布平滑模块的优化函数
"""

import numpy as np
from typing import Tuple, Dict, List, Any

# Relative imports for package
from .constants import (
    DEFAULT_TEMPERATURE, DEFAULT_POLYORDER, DEFAULT_PEAK_WIDTH_FACTOR,
    PARAM_GRID, BAYESIAN_N_CALLS
)
from .zones import define_zones
from .core import adaptive_smooth, apply_rdf_boundary, smooth_zone_transitions
from .peaks import detect_peaks
from .quality import calculate_quality_metrics


def _evaluate_smoothing_params(params: Dict[str, Any],
                               x: np.ndarray, P: np.ndarray,
                               dist_type: str, dr: float,
                               idx_valid_start: int, idx_valid_end: int,
                               peak_indices_raw: np.ndarray,
                               peak_positions_raw: np.ndarray,
                               peak_heights_raw: np.ndarray,
                               base_window: int,
                               temperature: float = DEFAULT_TEMPERATURE) -> Tuple[float, np.ndarray, Dict]:
    """
    Evaluate a set of smoothing parameters.

    Internal function used by optimization routines.

    Args:
        params: Dictionary with smoothing parameters
        x, P: Distribution data
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width
        idx_valid_start, idx_valid_end: Valid data range
        peak_indices_raw, peak_positions_raw, peak_heights_raw: Original peaks
        base_window: Base window size from adaptive calculation
        temperature: Temperature in Kelvin

    Returns:
        score: Quality score (higher is better)
        P_smooth: Smoothed distribution
        metrics: Quality metrics dictionary
    """
    # Extract parameters
    window_small_factor = params.get('window_small_factor', 1.0)
    window_large_factor = params.get('window_large_factor', 3.0)
    polyorder = params.get('polyorder', DEFAULT_POLYORDER)
    peak_width_factor = params.get('peak_width_factor', DEFAULT_PEAK_WIDTH_FACTOR)

    # Calculate actual window sizes
    window_small = max(polyorder + 2, int(base_window * window_small_factor))
    window_large = max(polyorder + 2, int(base_window * window_large_factor))

    # Ensure odd
    if window_small % 2 == 0:
        window_small += 1
    if window_large % 2 == 0:
        window_large += 1

    try:
        # Define zones
        zone_mask = define_zones(
            x, P, peak_indices_raw, peak_positions_raw,
            dist_type, idx_valid_start, idx_valid_end,
            peak_width_factor=peak_width_factor
        )

        # Apply adaptive smoothing
        P_smooth = adaptive_smooth(
            x, P, zone_mask,
            window_small=window_small,
            window_large=window_large,
            polyorder=polyorder
        )

        # Apply RDF boundary constraint if needed
        if dist_type == 'rdf':
            P_smooth = apply_rdf_boundary(x, P_smooth, zone_mask)

        # Smooth zone transitions
        P_smooth = smooth_zone_transitions(P_smooth, zone_mask)

        # Detect peaks in smoothed distribution
        peak_idx_s, peak_pos_s, peak_h_s = detect_peaks(x, P_smooth)

        # Calculate quality metrics (only within valid range)
        metrics, score = calculate_quality_metrics(
            x, P, P_smooth,
            peak_positions_raw, peak_heights_raw,
            peak_pos_s, peak_h_s,
            dr, temperature,
            idx_valid_start, idx_valid_end
        )

        return score, P_smooth, metrics

    except Exception as e:
        # Return low score on error
        return 0.0, P.copy(), {'error': str(e)}


def grid_search_optimize(x: np.ndarray, P: np.ndarray,
                         dist_type: str, dr: float,
                         idx_valid_start: int, idx_valid_end: int,
                         peak_indices_raw: np.ndarray,
                         peak_positions_raw: np.ndarray,
                         peak_heights_raw: np.ndarray,
                         base_window: int,
                         temperature: float = DEFAULT_TEMPERATURE,
                         verbose: bool = True) -> Tuple[Dict, np.ndarray, List[Dict]]:
    """
    Grid search optimization for smoothing parameters.

    Searches over predefined parameter grid to find optimal settings.

    Args:
        x, P: Distribution data
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width
        idx_valid_start, idx_valid_end: Valid data range
        peak_indices_raw, peak_positions_raw, peak_heights_raw: Original peaks
        base_window: Base window size
        temperature: Temperature in Kelvin
        verbose: Whether to print progress

    Returns:
        best_params: Best parameter set
        best_P_smooth: Best smoothed distribution
        optimization_log: List of all evaluations
    """
    best_score = -1
    best_params = None
    best_P_smooth = None
    optimization_log = []

    # Calculate total combinations
    total = (len(PARAM_GRID['window_small_factors']) *
             len(PARAM_GRID['window_large_factors']) *
             len(PARAM_GRID['polyorder']) *
             len(PARAM_GRID['peak_width_factor']))

    iteration = 0
    for wsf in PARAM_GRID['window_small_factors']:
        for wlf in PARAM_GRID['window_large_factors']:
            for po in PARAM_GRID['polyorder']:
                for pwf in PARAM_GRID['peak_width_factor']:
                    iteration += 1

                    params = {
                        'window_small_factor': wsf,
                        'window_large_factor': wlf,
                        'polyorder': po,
                        'peak_width_factor': pwf
                    }

                    score, P_smooth, metrics = _evaluate_smoothing_params(
                        params, x, P, dist_type, dr,
                        idx_valid_start, idx_valid_end,
                        peak_indices_raw, peak_positions_raw, peak_heights_raw,
                        base_window, temperature
                    )

                    # Calculate actual window sizes for logging
                    ws = max(po + 2, int(base_window * wsf))
                    wl = max(po + 2, int(base_window * wlf))
                    if ws % 2 == 0:
                        ws += 1
                    if wl % 2 == 0:
                        wl += 1

                    log_entry = {
                        'iteration': iteration,
                        'params': params,
                        'actual_window_small': ws,
                        'actual_window_large': wl,
                        'score': score,
                        'metrics': metrics
                    }
                    optimization_log.append(log_entry)

                    if score > best_score:
                        best_score = score
                        best_params = params.copy()
                        best_params['actual_window_small'] = ws
                        best_params['actual_window_large'] = wl
                        best_P_smooth = P_smooth.copy()

                        if verbose:
                            print(f"    [Grid {iteration}/{total}] New best: {score:.4f} "
                                  f"(ws={ws}, wl={wl}, po={po}, pwf={pwf:.1f})")

    return best_params, best_P_smooth, optimization_log


def bayesian_optimize(x: np.ndarray, P: np.ndarray,
                      dist_type: str, dr: float,
                      idx_valid_start: int, idx_valid_end: int,
                      peak_indices_raw: np.ndarray,
                      peak_positions_raw: np.ndarray,
                      peak_heights_raw: np.ndarray,
                      base_window: int,
                      n_calls: int = BAYESIAN_N_CALLS,
                      temperature: float = DEFAULT_TEMPERATURE,
                      verbose: bool = True) -> Tuple[Dict, np.ndarray, List[Dict]]:
    """
    Bayesian optimization for smoothing parameters.

    Uses scikit-optimize for efficient parameter search.
    Falls back to grid search if skopt is not available.

    Args:
        x, P: Distribution data
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width
        idx_valid_start, idx_valid_end: Valid data range
        peak_indices_raw, peak_positions_raw, peak_heights_raw: Original peaks
        base_window: Base window size
        n_calls: Number of optimization iterations
        temperature: Temperature in Kelvin
        verbose: Whether to print progress

    Returns:
        best_params: Best parameter set
        best_P_smooth: Best smoothed distribution
        optimization_log: List of all evaluations
    """
    try:
        from skopt import gp_minimize
        HAS_SKOPT = True
    except ImportError:
        HAS_SKOPT = False
        if verbose:
            print("    Note: scikit-optimize not installed, using grid search")
        return grid_search_optimize(
            x, P, dist_type, dr, idx_valid_start, idx_valid_end,
            peak_indices_raw, peak_positions_raw, peak_heights_raw,
            base_window, temperature, verbose
        )

    optimization_log = []
    best_result = {'score': -1, 'params': None, 'P_smooth': None}

    # Define search space
    space = [
        (0.5, 1.5),  # window_small_factor
        (1.5, 5.1),  # window_large_factor
        (2, 4),      # polyorder (integer)
        (0.5, 2.0)   # peak_width_factor
    ]

    iteration_counter = [0]

    def objective(params):
        iteration_counter[0] += 1
        wsf, wlf, po, pwf = params

        param_dict = {
            'window_small_factor': wsf,
            'window_large_factor': wlf,
            'polyorder': int(po),
            'peak_width_factor': pwf
        }

        score, P_smooth, metrics = _evaluate_smoothing_params(
            param_dict, x, P, dist_type, dr,
            idx_valid_start, idx_valid_end,
            peak_indices_raw, peak_positions_raw, peak_heights_raw,
            base_window, temperature
        )

        # Calculate actual windows
        ws = max(int(po) + 2, int(base_window * wsf))
        wl = max(int(po) + 2, int(base_window * wlf))
        if ws % 2 == 0:
            ws += 1
        if wl % 2 == 0:
            wl += 1

        log_entry = {
            'iteration': iteration_counter[0],
            'params': param_dict,
            'actual_window_small': ws,
            'actual_window_large': wl,
            'score': score,
            'metrics': metrics
        }
        optimization_log.append(log_entry)

        if score > best_result['score']:
            best_result['score'] = score
            best_result['params'] = param_dict.copy()
            best_result['params']['actual_window_small'] = ws
            best_result['params']['actual_window_large'] = wl
            best_result['P_smooth'] = P_smooth.copy()

            if verbose:
                print(f"    [Bayes {iteration_counter[0]}/{n_calls}] New best: {score:.4f}")

        # Return negative because gp_minimize minimizes
        return -score

    # Run optimization
    if verbose:
        print(f"    Running Bayesian optimization ({n_calls} iterations)...")

    result = gp_minimize(
        objective,
        space,
        n_calls=n_calls,
        random_state=42,
        verbose=False
    )

    return best_result['params'], best_result['P_smooth'], optimization_log


def auto_optimize(x: np.ndarray, P: np.ndarray,
                  dist_type: str, dr: float,
                  idx_valid_start: int, idx_valid_end: int,
                  peak_indices_raw: np.ndarray,
                  peak_positions_raw: np.ndarray,
                  peak_heights_raw: np.ndarray,
                  base_window: int,
                  use_bayesian: bool = True,
                  temperature: float = DEFAULT_TEMPERATURE,
                  verbose: bool = True) -> Tuple[Dict, np.ndarray, List[Dict]]:
    """
    Automatic parameter optimization wrapper.

    Uses Bayesian optimization if available and enabled,
    otherwise falls back to grid search.

    Args:
        x, P: Distribution data
        dist_type: 'bond', 'rdf', 'angle', or 'dihedral'
        dr: Bin width
        idx_valid_start, idx_valid_end: Valid data range
        peak_indices_raw, peak_positions_raw, peak_heights_raw: Original peaks
        base_window: Base window size
        use_bayesian: Whether to try Bayesian optimization
        temperature: Temperature in Kelvin
        verbose: Whether to print progress

    Returns:
        best_params: Best parameter set
        best_P_smooth: Best smoothed distribution
        optimization_log: List of all evaluations
    """
    if use_bayesian:
        return bayesian_optimize(
            x, P, dist_type, dr, idx_valid_start, idx_valid_end,
            peak_indices_raw, peak_positions_raw, peak_heights_raw,
            base_window, BAYESIAN_N_CALLS, temperature, verbose
        )
    else:
        return grid_search_optimize(
            x, P, dist_type, dr, idx_valid_start, idx_valid_end,
            peak_indices_raw, peak_positions_raw, peak_heights_raw,
            base_window, temperature, verbose
        )