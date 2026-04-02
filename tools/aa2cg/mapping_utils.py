"""
CG映射工具函数

提供AA到CG映射的通用函数，包括：
- 映射文件加载
- 坐标转换（质心计算）
- PBC处理

作者: 整合自 md_base_on_ml/AA_trj2CG_trj/AA_trj2CG_trj.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def load_aa_to_cg_mapping(csv_file: str) -> Dict:
    """
    从CSV文件加载AA到CG的映射关系。

    Args:
        csv_file: CSV文件路径，格式: bead_id, mol_id, bead_type, AA_id, mass

    Returns:
        mapping_dict: 映射字典
            {bead_id: {'bead_type': int, 'mol_id': int,
                       'aa_atoms': ndarray, 'aa_masses': ndarray, 'aa_indices': ndarray}}
    """
    df = pd.read_csv(csv_file)

    mapping_dict = {}
    for bead_id, group in df.groupby('bead_id'):
        # 预转换为numpy数组，提高效率
        aa_atoms = group['AA_id'].values.astype(np.int32)
        aa_masses = group['mass'].values.astype(np.float64)

        mapping_dict[bead_id] = {
            'bead_type': int(group['bead_type'].iloc[0]),
            'mol_id': int(group['mol_id'].iloc[0]),
            'aa_atoms': aa_atoms,
            'aa_masses': aa_masses,
            'aa_indices': aa_atoms - 1  # 预计算索引 (0-based)
        }

    return mapping_dict


def convert_aa_to_cg_frame(aa_coords: np.ndarray, aa_ids: np.ndarray,
                           mapping_dict: Dict, box: np.ndarray) -> Dict:
    """
    将单帧AA坐标转换为CG坐标。

    Args:
        aa_coords: AA坐标数组，形状 (n_aa_atoms, 3)
        aa_ids: AA原子ID数组
        mapping_dict: 从 load_aa_to_cg_mapping() 获取的映射字典
        box: 盒子尺寸数组，形状 (3, 2) 或 (3,)

    Returns:
        cg_data: dict，包含:
            - 'ids': CG bead ID数组
            - 'mol_ids': CG分子ID数组
            - 'types': CG bead类型数组
            - 'coords': CG坐标数组，形状 (n_beads, 3)
            - 'box': 盒子尺寸
            - 'natoms': CG bead数量
    """
    # 计算CG bead位置（质心）
    n_beads = len(mapping_dict)
    cg_ids = np.zeros(n_beads, dtype=np.int32)
    cg_mol_ids = np.zeros(n_beads, dtype=np.int32)
    cg_types = np.zeros(n_beads, dtype=np.int32)
    cg_coords = np.zeros((n_beads, 3), dtype=np.float64)

    for i, (bead_id, bead_info) in enumerate(sorted(mapping_dict.items())):
        cg_ids[i] = bead_id
        cg_mol_ids[i] = bead_info['mol_id']
        cg_types[i] = bead_info['bead_type']

        # 获取该bead对应的AA原子索引和质量
        aa_indices = bead_info['aa_indices']

        if len(aa_indices) > 0:
            # 计算质心（质量加权平均）
            bead_coords = aa_coords[aa_indices]
            masses = bead_info['aa_masses']

            total_mass = np.sum(masses)
            if total_mass > 0:
                cg_coords[i] = np.sum(bead_coords * masses[:, np.newaxis], axis=0) / total_mass
            else:
                # 如果质量为零，使用几何中心
                cg_coords[i] = np.mean(bead_coords, axis=0)
        else:
            print(f"警告: Bead {bead_id} 没有对应的AA原子")

    # 处理盒子格式
    if box.ndim == 1:
        box = np.array([[0.0, box[0]], [0.0, box[1]], [0.0, box[2]]], dtype=np.float64)

    cg_data = {
        'ids': cg_ids,
        'mol_ids': cg_mol_ids,
        'types': cg_types,
        'coords': cg_coords,
        'box': box,
        'natoms': n_beads
    }

    return cg_data


def pbc_distance(pos1: np.ndarray, pos2: np.ndarray, box_length: np.ndarray) -> np.ndarray:
    """
    计算PBC-aware距离。

    Args:
        pos1: 位置数组1，形状 (..., 3)
        pos2: 位置数组2，形状 (..., 3)
        box_length: 盒子长度，形状 (3,)

    Returns:
        距离数组
    """
    delta = pos2 - pos1
    delta -= box_length * np.round(delta / box_length)
    return np.linalg.norm(delta, axis=-1)


def wrap_coords(coords: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    将坐标包装到盒子内。

    Args:
        coords: 坐标数组，形状 (n_atoms, 3)
        box: 盒子尺寸，形状 (3, 2) 或 (3,)

    Returns:
        包装后的坐标数组
    """
    if box.ndim == 1:
        box_lo = np.zeros(3)
        box_hi = box
    else:
        box_lo = box[:, 0]
        box_hi = box[:, 1]

    box_length = box_hi - box_lo

    # 包装坐标到 [box_lo, box_hi)
    wrapped = coords.copy()
    for dim in range(3):
        wrapped[:, dim] = wrapped[:, dim] - box_lo[dim]
        wrapped[:, dim] = wrapped[:, dim] - np.floor(wrapped[:, dim] / box_length[dim]) * box_length[dim]
        wrapped[:, dim] = wrapped[:, dim] + box_lo[dim]

    return wrapped


def calculate_com(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    计算质心。

    Args:
        coords: 坐标数组，形状 (n_atoms, 3)
        masses: 质量数组，形状 (n_atoms,)

    Returns:
        质心坐标，形状 (3,)
    """
    total_mass = np.sum(masses)
    if total_mass > 0:
        return np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass
    else:
        return np.mean(coords, axis=0)


if __name__ == "__main__":
    # 测试映射加载
    import sys
    if len(sys.argv) > 1:
        mapping_file = sys.argv[1]
    else:
        mapping_file = "AtomId_BeadId_compare_list.csv"

    try:
        mapping = load_aa_to_cg_mapping(mapping_file)
        print(f"加载映射成功: {len(mapping)} beads")
        print(f"示例 (bead 1): {mapping[1]}")
    except Exception as e:
        print(f"错误: {e}")