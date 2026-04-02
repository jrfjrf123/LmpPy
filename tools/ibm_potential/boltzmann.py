"""
玻尔兹曼反演模块

通过玻尔兹曼反演从分布计算势能。

公式:
- U(r) = -kB * T * ln(P(r))
- 对于RDF: U(r) = -kB * T * ln(g(r))

作者: 整合自 md_base_on_ml/calc_ibm_pot/calculate_potentials.py
"""

import numpy as np
from typing import Dict, Tuple, Optional
from scipy.signal import savgol_filter

# 玻尔兹曼常数
KB = {
    'kcal/mol': 0.0019872041,  # kcal/mol/K
    'kJ/mol': 0.0083144626,    # kJ/mol/K
    'eV': 8.617333262e-5       # eV/K
}


def calculate_bond_potential(r: np.ndarray, hist: np.ndarray,
                             temperature: float = 400.0,
                             units: str = 'kcal/mol',
                             jacobian_correction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    从键长分布计算势能。

    Args:
        r: 距离数组
        hist: 分布数组
        temperature: 温度 (K)
        units: 能量单位 ('kcal/mol', 'kJ/mol', 'eV')
        jacobian_correction: 是否进行Jacobian校正

    Returns:
        (r, potential): 距离和势能数组
    """
    kB = KB.get(units, KB['kcal/mol'])
    kT = kB * temperature

    # Jacobian校正: P(r) ∝ hist(r) / r²
    if jacobian_correction and len(r) > 1:
        dr = r[1] - r[0]
        prob_dist = hist / (4 * np.pi * r**2 * dr + 1e-9)
    else:
        prob_dist = hist

    # 玻尔兹曼反演
    potential = np.full_like(prob_dist, np.inf)
    valid = prob_dist > 0
    potential[valid] = -kT * np.log(prob_dist[valid])

    # 归一化（最小值设为0）
    finite = np.isfinite(potential)
    if np.any(finite):
        potential -= np.min(potential[finite])

    # 插值填补inf
    for i in range(1, len(potential) - 1):
        if np.isinf(potential[i]) and np.isfinite(potential[i-1]) and np.isfinite(potential[i+1]):
            potential[i] = (potential[i-1] + potential[i+1]) / 2

    return r, potential


def calculate_angle_potential(theta: np.ndarray, hist: np.ndarray,
                              temperature: float = 400.0,
                              units: str = 'kcal/mol',
                              jacobian_correction: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    从角度分布计算势能。

    Args:
        theta: 角度数组 (degrees)
        hist: 分布数组
        temperature: 温度 (K)
        units: 能量单位
        jacobian_correction: 是否进行Jacobian校正

    Returns:
        (theta, potential): 角度和势能数组
    """
    kB = KB.get(units, KB['kcal/mol'])
    kT = kB * temperature

    # Jacobian校正: P(θ) ∝ hist(θ) / sin(θ)
    if jacobian_correction:
        sin_theta = np.sin(np.deg2rad(theta)) + 1e-9
        prob_dist = hist / sin_theta
    else:
        prob_dist = hist

    potential = np.full_like(prob_dist, np.inf)
    valid = prob_dist > 0
    potential[valid] = -kT * np.log(prob_dist[valid])

    finite = np.isfinite(potential)
    if np.any(finite):
        potential -= np.min(potential[finite])

    for i in range(1, len(potential) - 1):
        if np.isinf(potential[i]) and np.isfinite(potential[i-1]) and np.isfinite(potential[i+1]):
            potential[i] = (potential[i-1] + potential[i+1]) / 2

    return theta, potential


def calculate_dihedral_potential(phi: np.ndarray, hist: np.ndarray,
                                 temperature: float = 400.0,
                                 units: str = 'kcal/mol') -> Tuple[np.ndarray, np.ndarray]:
    """
    从二面角分布计算势能。

    Args:
        phi: 二面角数组 (degrees)
        hist: 分布数组
        temperature: 温度 (K)
        units: 能量单位

    Returns:
        (phi, potential): 二面角和势能数组
    """
    kB = KB.get(units, KB['kcal/mol'])
    kT = kB * temperature

    potential = np.full_like(hist, np.inf)
    valid = hist > 0
    potential[valid] = -kT * np.log(hist[valid])

    finite = np.isfinite(potential)
    if np.any(finite):
        potential -= np.min(potential[finite])

    for i in range(1, len(potential) - 1):
        if np.isinf(potential[i]) and np.isfinite(potential[i-1]) and np.isfinite(potential[i+1]):
            potential[i] = (potential[i-1] + potential[i+1]) / 2

    return phi, potential


def calculate_pair_potential(r: np.ndarray, g_r: np.ndarray,
                             temperature: float = 400.0,
                             units: str = 'kcal/mol') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从RDF计算非键合势能。

    Args:
        r: 距离数组
        g_r: RDF数组
        temperature: 温度 (K)
        units: 能量单位

    Returns:
        (r, potential, g_r): 距离、势能和RDF数组
    """
    kB = KB.get(units, KB['kcal/mol'])
    kT = kB * temperature

    potential = np.full_like(g_r, np.inf)
    valid = g_r > 0
    potential[valid] = -kT * np.log(g_r[valid])

    for i in range(1, len(potential) - 1):
        if np.isinf(potential[i]) and np.isfinite(potential[i-1]) and np.isfinite(potential[i+1]):
            potential[i] = (potential[i-1] + potential[i+1]) / 2

    return r, potential, g_r


def extrapolate_and_smooth(x: np.ndarray, potential: np.ndarray,
                           smooth_window: int = 21,
                           smooth_polyorder: int = 3,
                           extrap_points: int = 5,
                           extrap_method: str = 'linear') -> Tuple[np.ndarray, np.ndarray]:
    """
    平滑势能并外推未采样区域。

    Args:
        x: 坐标数组
        potential: 势能数组
        smooth_window: Savitzky-Golay窗口大小
        smooth_polyorder: 多项式阶数
        extrap_points: 用于拟合外推的点数
        extrap_method: 外推方法 ('linear', 'exponential')

    Returns:
        (x, potential_processed): 处理后的坐标和势能
    """
    V_processed = np.copy(potential)
    finite_indices = np.where(np.isfinite(potential))[0]

    if len(finite_indices) == 0:
        return x, V_processed

    # 平滑
    if smooth_window > 0 and len(finite_indices) > smooth_window:
        if smooth_window % 2 == 0:
            smooth_window += 1
        V_processed[finite_indices] = savgol_filter(potential[finite_indices], smooth_window, smooth_polyorder)

    # 外推
    if extrap_points == 0 or len(finite_indices) < extrap_points:
        return x, V_processed

    first_finite, last_finite = finite_indices[0], finite_indices[-1]
    min_potential = np.min(V_processed[finite_indices])

    # 左侧外推
    if first_finite > 0:
        fit_x = x[first_finite:first_finite + extrap_points]
        fit_y = V_processed[first_finite:first_finite + extrap_points]
        extrap_x = x[:first_finite]

        if extrap_method == 'linear':
            coeffs = np.polyfit(fit_x, fit_y, 1)
            V_processed[:first_finite] = np.polyval(coeffs, extrap_x)
        elif extrap_method == 'exponential':
            safe_fit_y = fit_y - min_potential + 1e-9
            coeffs = np.polyfit(fit_x, np.log(safe_fit_y), 1)
            V_processed[:first_finite] = np.exp(np.polyval(coeffs, extrap_x)) + min_potential - 1e-9

    # 右侧外推
    if last_finite < len(x) - 1:
        fit_x = x[last_finite - extrap_points + 1:last_finite + 1]
        fit_y = V_processed[last_finite - extrap_points + 1:last_finite + 1]
        extrap_x = x[last_finite + 1:]

        if extrap_method == 'linear':
            coeffs = np.polyfit(fit_x, fit_y, 1)
            V_processed[last_finite + 1:] = np.polyval(coeffs, extrap_x)
        elif extrap_method == 'exponential':
            safe_fit_y = fit_y - min_potential + 1e-9
            coeffs = np.polyfit(fit_x, np.log(safe_fit_y), 1)
            V_processed[last_finite + 1:] = np.exp(np.polyval(coeffs, extrap_x)) + min_potential - 1e-9

    return x, V_processed


def calculate_force(x: np.ndarray, potential: np.ndarray) -> np.ndarray:
    """计算力 (-dU/dx)"""
    return -np.gradient(potential, x)


def save_potential(filename: str, x: np.ndarray, potential: np.ndarray,
                   potential_type: str, bead_types: tuple = None):
    """保存势能到文件"""
    header = f"{potential_type} potential"
    if bead_types:
        header += f" (bead types {bead_types})"

    np.savetxt(filename, np.column_stack([x, potential]),
               header=header + f"\nx potential")


if __name__ == "__main__":
    print("玻尔兹曼反演模块")
    print("使用方法:")
    print("  from LmpPy.tools.ibm_potential import boltzmann")
    print("  r, U = boltzmann.calculate_bond_potential(r, hist, temperature=400)")