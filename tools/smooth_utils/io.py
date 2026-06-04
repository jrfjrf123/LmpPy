"""
Data loading functions for distribution smoothing.
分布平滑模块的数据加载函数
"""

import numpy as np
import os
import re
from typing import Tuple

# Relative imports for package
from .constants import DEFAULT_THRESHOLD


def load_distribution(filepath: str) -> Tuple[np.ndarray, np.ndarray, str, float]:
    """
    Load distribution data from file.

    Automatically detects distribution type (bond/rdf/angle/dihedral) from filename.
    Supports both LAMMPS/IBI format (*_dist.txt) and VOTCA format (*.dist.tgt).
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

    # VOTCA 格式检测 (如 bond_type1.dist.tgt, angle_type1.dist.tgt)
    votca_match = re.match(r'(bond|angle|dihedral|pair)_type\d+(_\d+)?\.dist\.tgt', filename)
    if votca_match:
        dist_type = votca_match.group(1)
        if dist_type == 'pair':
            dist_type = 'rdf'
    # 有序 bond 格式 (如 bond_ordered_1_2.dist.tgt)
    elif re.match(r'bond_ordered_\d+_\d+\.dist\.tgt', filename):
        dist_type = 'bond'
    # 通用 bond/angle/dihedral/pair/rdf 关键字检测
    elif 'rdf' in filename or 'pair' in filename:
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
    # VOTCA 格式每行有 3 列: <value> <probability> i
    # 我们只读取前两列
    try:
        # 首先尝试只读取前两列（适用于 VOTCA 格式和其他格式）
        data = np.loadtxt(filepath, comments='#', usecols=(0, 1))
    except Exception:
        # 如果失败，尝试读取所有列（适用于 LAMMPS/IBI 格式）
        try:
            data = np.loadtxt(filepath, comments='#')
            if data.shape[1] > 2:
                data = data[:, :2]  # 只取前两列
        except Exception as e:
            raise ValueError(f"Error reading file {filepath}: {e}")

    if data.ndim == 1:
        # 单列数据，无法处理
        raise ValueError(f"Invalid data format in {filepath}. Expected at least 2 columns, got 1.")
    elif data.ndim != 2 or data.shape[1] < 2:
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


def convert_distribution_units(
    x: np.ndarray,
    P: np.ndarray,
    dist_type: str,
    from_unit: str,
    to_unit: str
) -> tuple:
    """
    转换分布数据的单位，同步缩放 x 和 P 以保持积分守恒 ∫P(x)dx。

    距离单位: 'A' (Angstrom) 或 'nm' (nanometer)
    角度单位: 'deg' (degree) 或 'rad' (radian)

    当 x → α·x 时，P → P/α，保证 ∫P(x)dx = ∫P'(x')dx'。

    Args:
        x: x 轴数组（距离或角度）
        P: 概率密度数组
        dist_type: 分布类型 ('bond', 'rdf', 'angle', 'dihedral')
        from_unit: 输入单位
        to_unit: 输出单位

    Returns:
        (x_new, P_new): 转换后的数组
    """
    if from_unit == to_unit:
        return x.copy(), P.copy()

    if dist_type in ('bond', 'rdf'):
        # 距离: 以 Å 为基准
        factor_map = {'A': 1.0, 'nm': 10.0}
    else:
        # 角度: 以 deg 为基准
        factor_map = {'deg': 1.0, 'rad': 180.0 / np.pi}

    if from_unit not in factor_map:
        raise ValueError(f"不支持的单位 '{from_unit}'，dist_type={dist_type}。"
                         f"支持的单位: {list(factor_map.keys())}")
    if to_unit not in factor_map:
        raise ValueError(f"不支持的单位 '{to_unit}'，dist_type={dist_type}。"
                         f"支持的单位: {list(factor_map.keys())}")

    scale = factor_map[from_unit] / factor_map[to_unit]
    return x * scale, P / scale