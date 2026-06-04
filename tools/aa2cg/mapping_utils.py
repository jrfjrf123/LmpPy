"""
CG映射工具函数

提供AA到CG映射的通用函数，包括：
- 映射文件加载
- 坐标转换（质心计算）
- PBC处理
- 预缓存结构（性能优化）

作者: 整合自 md_base_on_ml/AA_trj2CG_trj/AA_trj2CG_trj.py
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

# 尝试导入 Numba（用于 JIT 加速）
try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


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
    # 处理盒子格式
    if box.ndim == 1:
        box = np.array([[0.0, box[0]], [0.0, box[1]], [0.0, box[2]]], dtype=np.float64)

    box_length = box[:, 1] - box[:, 0]

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
            # 获取AA原子坐标
            bead_coords = aa_coords[aa_indices]
            masses = bead_info['aa_masses']

            # PBC-aware 质心计算
            # 将所有原子展开到相对于第一个原子的位置
            if len(bead_coords) > 1:
                ref_coord = bead_coords[0]
                delta = bead_coords - ref_coord
                # 应用最小图像约定，确保所有原子在相对位置上连续
                delta -= box_length * np.round(delta / box_length)
                # 展开后的坐标
                unfolded_coords = ref_coord + delta

                # 计算质心（质量加权平均）
                total_mass = np.sum(masses)
                if total_mass > 0:
                    com = np.sum(unfolded_coords * masses[:, np.newaxis], axis=0) / total_mass
                else:
                    com = np.mean(unfolded_coords, axis=0)
            else:
                # 单个原子，直接使用其坐标
                com = bead_coords[0]

            cg_coords[i] = com
        else:
            print(f"警告: Bead {bead_id} 没有对应的AA原子")

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


@dataclass
class CGMappingCache:
    """
    预缓存的 CG 映射数据结构

    CSR-like 格式存储 bead-atom 关系，用于 Numba JIT 加速。

    属性:
        bead_ids: 预排序的 bead IDs 数组
        mol_ids: 每个 bead 的分子 ID
        bead_types: 每个 bead 的类型
        bead_atom_indices: CSR-like 索引数组，-1 表示无效位置
        bead_masses: CSR-like 质量数组
        bead_mass_totals: 预计算的总质量（避免每帧重复计算）
        bead_atom_counts: 每个 bead 的原子数
        n_beads: bead 总数
        max_atoms_per_bead: 每个 bead 最大原子数
    """
    bead_ids: np.ndarray
    mol_ids: np.ndarray
    bead_types: np.ndarray
    bead_atom_indices: np.ndarray
    bead_masses: np.ndarray
    bead_mass_totals: np.ndarray
    bead_atom_counts: np.ndarray
    n_beads: int
    max_atoms_per_bead: int


def prepare_mapping_cache(mapping_dict: Dict) -> CGMappingCache:
    """
    预处理映射字典，构建可复用的缓存结构。

    在 convert_trajectory_to_cg() 开始时调用一次，
    返回的 cache 可用于所有帧的转换，避免每帧重复处理 mapping_dict。

    Args:
        mapping_dict: 从 load_aa_to_cg_mapping() 获取的映射字典

    Returns:
        CGMappingCache: 预缓存的映射数据结构
    """
    # 预排序 bead IDs（一次性处理）
    sorted_bead_ids = sorted(mapping_dict.keys())
    n_beads = len(sorted_bead_ids)

    # 统计每个 bead 的原子数
    bead_atom_counts_list = []
    for bead_id in sorted_bead_ids:
        bead_atom_counts_list.append(len(mapping_dict[bead_id]['aa_indices']))

    max_atoms_per_bead = max(bead_atom_counts_list) if bead_atom_counts_list else 1

    # 构建 CSR-like 数组（固定大小，便于 Numba 处理）
    bead_atom_indices = np.full((n_beads, max_atoms_per_bead), -1, dtype=np.int32)
    bead_masses = np.zeros((n_beads, max_atoms_per_bead), dtype=np.float64)
    bead_mass_totals = np.zeros(n_beads, dtype=np.float64)

    # 填充 bead IDs, mol_ids, bead_types
    bead_ids = np.array(sorted_bead_ids, dtype=np.int32)
    mol_ids = np.zeros(n_beads, dtype=np.int32)
    bead_types = np.zeros(n_beads, dtype=np.int32)

    for i, bead_id in enumerate(sorted_bead_ids):
        info = mapping_dict[bead_id]
        n_atoms = len(info['aa_indices'])

        # 填充原子索引和质量
        bead_atom_indices[i, :n_atoms] = info['aa_indices']
        bead_masses[i, :n_atoms] = info['aa_masses']
        bead_mass_totals[i] = np.sum(info['aa_masses'])

        mol_ids[i] = info['mol_id']
        bead_types[i] = info['bead_type']

    bead_atom_counts = np.array(bead_atom_counts_list, dtype=np.int32)

    return CGMappingCache(
        bead_ids=bead_ids,
        mol_ids=mol_ids,
        bead_types=bead_types,
        bead_atom_indices=bead_atom_indices,
        bead_masses=bead_masses,
        bead_mass_totals=bead_mass_totals,
        bead_atom_counts=bead_atom_counts,
        n_beads=n_beads,
        max_atoms_per_bead=max_atoms_per_bead
    )


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _convert_aa_to_cg_frame_jit(
        aa_coords: np.ndarray,
        bead_atom_indices: np.ndarray,
        bead_masses: np.ndarray,
        bead_mass_totals: np.ndarray,
        bead_atom_counts: np.ndarray,
        box_length: np.ndarray,
        n_beads: int,
        max_atoms_per_bead: int
    ) -> np.ndarray:
        """
        JIT 版本的 AA 到 CG 坐标转换

        核心优化：
        - 使用预缓存的 CSR-like 数组，避免 Python 循环和字典访问
        - 内联 PBC 计算，避免函数调用
        - 预计算总质量，避免循环内 sum

        Args:
            aa_coords: AA 坐标数组 (n_aa_atoms, 3)
            bead_atom_indices: CSR-like 索引数组 (n_beads, max_atoms_per_bead)
            bead_masses: CSR-like 质量数组 (n_beads, max_atoms_per_bead)
            bead_mass_totals: 预计算的总质量 (n_beads,)
            bead_atom_counts: 每个 bead 的原子数 (n_beads,)
            box_length: 盒子尺寸 (3,)
            n_beads: bead 数量
            max_atoms_per_bead: 每个 bead 最大原子数

        Returns:
            cg_coords: CG 坐标数组 (n_beads, 3)
        """
        cg_coords = np.zeros((n_beads, 3), dtype=np.float64)

        for i in range(n_beads):
            n_atoms = bead_atom_counts[i]

            if n_atoms == 0:
                continue

            if n_atoms == 1:
                # 单原子 bead：直接使用其坐标
                aa_idx = bead_atom_indices[i, 0]
                if aa_idx >= 0:
                    cg_coords[i] = aa_coords[aa_idx]
            else:
                # 多原子 bead：PBC-aware 质心计算
                # 获取第一个原子坐标作为参考点
                ref_idx = bead_atom_indices[i, 0]
                if ref_idx < 0:
                    continue

                ref_coord = aa_coords[ref_idx]
                total_mass = bead_mass_totals[i]

                if total_mass <= 0:
                    continue

                # 累加质量加权坐标
                com_x = 0.0
                com_y = 0.0
                com_z = 0.0

                for j in range(n_atoms):
                    aa_idx = bead_atom_indices[i, j]
                    if aa_idx < 0:
                        continue

                    mass = bead_masses[i, j]

                    # 计算相对于参考点的最小镜像位移
                    dx = aa_coords[aa_idx, 0] - ref_coord[0]
                    dy = aa_coords[aa_idx, 1] - ref_coord[1]
                    dz = aa_coords[aa_idx, 2] - ref_coord[2]

                    # 应用最小图像约定
                    dx = dx - box_length[0] * round(dx / box_length[0])
                    dy = dy - box_length[1] * round(dy / box_length[1])
                    dz = dz - box_length[2] * round(dz / box_length[2])

                    # 展开后的坐标，累加质量加权
                    com_x += mass * (ref_coord[0] + dx)
                    com_y += mass * (ref_coord[1] + dy)
                    com_z += mass * (ref_coord[2] + dz)

                cg_coords[i, 0] = com_x / total_mass
                cg_coords[i, 1] = com_y / total_mass
                cg_coords[i, 2] = com_z / total_mass

        return cg_coords


def _convert_aa_to_cg_frame_python_with_cache(
    aa_coords: np.ndarray,
    mapping_cache: CGMappingCache,
    box_length: np.ndarray
) -> np.ndarray:
    """
    Python 版本的 AA 到 CG 坐标转换（使用预缓存）

    当 Numba 不可用时使用此版本，仍然比原版快（因为有预缓存）。

    Args:
        aa_coords: AA 坐标数组
        mapping_cache: 预缓存的映射数据
        box_length: 盒子尺寸

    Returns:
        cg_coords: CG 坐标数组
    """
    n_beads = mapping_cache.n_beads
    cg_coords = np.zeros((n_beads, 3), dtype=np.float64)

    for i in range(n_beads):
        n_atoms = mapping_cache.bead_atom_counts[i]

        if n_atoms == 0:
            continue

        # 获取原子索引
        aa_indices = mapping_cache.bead_atom_indices[i, :n_atoms]

        if n_atoms == 1:
            cg_coords[i] = aa_coords[aa_indices[0]]
        else:
            # PBC-aware 质心计算
            ref_coord = aa_coords[aa_indices[0]]
            bead_coords = aa_coords[aa_indices]
            masses = mapping_cache.bead_masses[i, :n_atoms]

            # 最小图像约定
            delta = bead_coords - ref_coord
            delta -= box_length * np.round(delta / box_length)
            unfolded_coords = ref_coord + delta

            # 质量加权质心
            total_mass = mapping_cache.bead_mass_totals[i]
            if total_mass > 0:
                cg_coords[i] = np.sum(unfolded_coords * masses[:, np.newaxis], axis=0) / total_mass
            else:
                cg_coords[i] = np.mean(unfolded_coords, axis=0)

    return cg_coords


def convert_aa_to_cg_frame_optimized(
    aa_coords: np.ndarray,
    mapping_cache: CGMappingCache,
    box: np.ndarray
) -> Dict:
    """
    优化的 AA 到 CG 坐标转换

    使用预缓存 + Numba JIT 加速，相比原版可提升 20-50x 性能。

    Args:
        aa_coords: AA 坐标数组 (n_aa_atoms, 3)
        mapping_cache: 预处理的映射缓存（从 prepare_mapping_cache() 获取）
        box: 盒子尺寸 (3, 2) 或 (3,)

    Returns:
        cg_data: CG 数据字典，包含 ids, mol_ids, types, coords, box, natoms
    """
    # 处理盒子格式
    if box.ndim == 1:
        box_length = box.astype(np.float64)
        box_out = np.array([[0.0, box[0]], [0.0, box[1]], [0.0, box[2]]], dtype=np.float64)
    else:
        box_length = (box[:, 1] - box[:, 0]).astype(np.float64)
        box_out = box

    if NUMBA_AVAILABLE:
        cg_coords = _convert_aa_to_cg_frame_jit(
            aa_coords.astype(np.float64),
            mapping_cache.bead_atom_indices,
            mapping_cache.bead_masses,
            mapping_cache.bead_mass_totals,
            mapping_cache.bead_atom_counts,
            box_length,
            mapping_cache.n_beads,
            mapping_cache.max_atoms_per_bead
        )
    else:
        # 回退到 Python + 预缓存版本
        cg_coords = _convert_aa_to_cg_frame_python_with_cache(
            aa_coords, mapping_cache, box_length
        )

    return {
        'ids': mapping_cache.bead_ids,
        'mol_ids': mapping_cache.mol_ids,
        'types': mapping_cache.bead_types,
        'coords': cg_coords,
        'box': box_out,
        'natoms': mapping_cache.n_beads
    }


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

        # 测试预缓存
        cache = prepare_mapping_cache(mapping)
        print(f"\n预缓存成功:")
        print(f"  n_beads: {cache.n_beads}")
        print(f"  max_atoms_per_bead: {cache.max_atoms_per_bead}")
        print(f"  bead_atom_indices shape: {cache.bead_atom_indices.shape}")

        # 测试 JIT 函数（如果 Numba 可用）
        if NUMBA_AVAILABLE:
            print(f"\nNumba 可用，测试 JIT 函数...")
            dummy_coords = np.random.rand(1000, 3).astype(np.float64)
            dummy_box = np.array([10.0, 10.0, 10.0], dtype=np.float64)
            cg_coords = _convert_aa_to_cg_frame_jit(
                dummy_coords,
                cache.bead_atom_indices,
                cache.bead_masses,
                cache.bead_mass_totals,
                cache.bead_atom_counts,
                dummy_box,
                cache.n_beads,
                cache.max_atoms_per_bead
            )
            print(f"  JIT 输出 shape: {cg_coords.shape}")
            print(f"  ✓ JIT 测试成功")
        else:
            print(f"\nNumba 不可用，使用 Python 版本")

    except Exception as e:
        print(f"错误: {e}")