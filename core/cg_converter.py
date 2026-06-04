"""
CG坐标转换器模块

功能:
- 全原子坐标 → 粗粒化坐标
- 惰性索引缓存优化
- 质心计算
- CG映射一致性验证

优化:
- 无反应时: O(n_beads × n_atoms) → O(n_atoms)
- 有反应时: O(n_beads × n_atoms) → O(n_atoms log n_atoms)

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np

# 支持两种导入方式
try:
    from .cg_mapper import CGMapping
except ImportError:
    from cg_mapper import CGMapping


class CGMappingValidationError(Exception):
    """CG映射验证错误"""
    pass


def validate_cg_mapping_consistency(cg_compare_list: np.ndarray,
                                     raise_error: bool = True) -> Dict[int, int]:
    """
    验证 CG 映射的一致性：确保每个 bead_id 的所有原子有相同的 bead_type

    使用排序 + 相邻比较代替逐掩码扫描

    Args:
        cg_compare_list: CG映射 (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]
        raise_error: 如果验证失败是否抛出错误

    Returns:
        Dict[int, int]: {bead_id: bead_type} 映射

    Raises:
        CGMappingValidationError: 当同一 bead_id 有不同的 bead_type 时
    """
    bead_ids = cg_compare_list[:, 0].astype(np.int32)
    bead_types = cg_compare_list[:, 2].astype(np.int32)

    # 排序后相邻比较
    sort_idx = np.argsort(bead_ids, kind='mergesort')
    sorted_beads = bead_ids[sort_idx]
    sorted_types = bead_types[sort_idx]

    # 找分组边界
    unique_bead_ids, first_idx, counts = np.unique(
        sorted_beads, return_index=True, return_counts=True
    )

    errors = []
    bead_id_to_type = {}

    for i, bead_id in enumerate(unique_bead_ids):
        start = first_idx[i]
        end = start + counts[i]
        unique_types = np.unique(sorted_types[start:end])
        if len(unique_types) > 1:
            error_msg = (
                f"Bead {bead_id} 有 {len(unique_types)} 个不同的 bead_type: {unique_types.tolist()}. "
                f"涉及的原子索引: {np.where(sorted_beads[start:end] == bead_id)[0].tolist()[:10]}..."
            )
            errors.append(error_msg)
        else:
            bead_id_to_type[bead_id] = int(unique_types[0])

    if errors:
        error_msg = "CG映射一致性验证失败:\n" + "\n".join(errors)
        if raise_error:
            raise CGMappingValidationError(error_msg)
        else:
            print(f"警告: {error_msg}")

    return bead_id_to_type


@dataclass
class CGConverter:
    """
    CG坐标转换器

    功能:
    - 高效的全原子坐标到粗粒化坐标转换
    - 惰性索引缓存
    - CG映射一致性验证
    """

    # 缓存的bead到原子的映射
    bead_to_atoms: Dict[int, np.ndarray] = field(default_factory=dict)

    # 缓存的bead_type (避免每次重新扫描 cg_compare_list)
    bead_to_type: Dict[int, int] = field(default_factory=dict)

    # 预计算的扁平化原子索引（加速向量化转换）
    _flat_atom_indices: Optional[np.ndarray] = None  # 所有原子的 0-based 索引
    _flat_bead_indices: Optional[np.ndarray] = None  # 每个原子对应的 bead 行索引
    _bead_ids_sorted: Optional[np.ndarray] = None    # 排序后的 bead_id 数组
    _n_beads: int = 0

    # 缓存的有效性标志
    cache_valid: bool = False

    # 上次的cg_compare_list哈希
    last_hash: Optional[int] = None

    # 验证过的 bead_id -> bead_type 映射
    bead_id_to_type: Dict[int, int] = field(default_factory=dict)

    def convert(self, atom_coords: np.ndarray,
                cg_compare_list: np.ndarray,
                mass_list: Dict[int, float],
                validate: bool = True) -> np.ndarray:
        """
        转换全原子坐标到粗粒化坐标

        Args:
            atom_coords: 原子坐标 (n_atoms, 3+) 或 (n_atoms, 5+) [id, type, x, y, z, ...]
            cg_compare_list: CG映射 (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]
            mass_list: 原子类型到质量的映射
            validate: 是否验证 CG 映射一致性 (默认 True)

        Returns:
            CG坐标 (n_beads, 3+) [bead_id, bead_type, x, y, z]

        Raises:
            CGMappingValidationError: 当 validate=True 且 CG 映射不一致时
        """
        # 检查缓存是否有效
        current_hash = hash(cg_compare_list.tobytes())
        if self.cache_valid and self.last_hash == current_hash:
            # 使用缓存的索引
            return self._convert_with_cache(atom_coords, cg_compare_list, mass_list)
        else:
            # 验证 CG 映射一致性
            if validate:
                self.bead_id_to_type = validate_cg_mapping_consistency(cg_compare_list)

            # 重建索引
            self._build_index(cg_compare_list)
            self.cache_valid = True
            self.last_hash = current_hash
            return self._convert_with_cache(atom_coords, cg_compare_list, mass_list)

    def invalidate_cache(self):
        """使缓存失效"""
        self.cache_valid = False
        self.last_hash = None
        self.bead_to_atoms.clear()
        self.bead_to_type.clear()
        self._flat_atom_indices = None
        self._flat_bead_indices = None
        self._bead_ids_sorted = None
        self._n_beads = 0

    def _build_index(self, cg_compare_list: np.ndarray):
        """
        构建bead到原子的索引
        使用排序分组代替逐掩码扫描，并预计算扁平化索引
        """
        self.bead_to_atoms.clear()
        self.bead_to_type.clear()

        # 提取列
        bead_ids_col = cg_compare_list[:, 0].astype(np.int32)
        bead_types_col = cg_compare_list[:, 2].astype(np.int32)
        atom_ids_col = cg_compare_list[:, 3].astype(np.int32)

        # 按 bead_id 排序
        sort_idx = np.argsort(bead_ids_col, kind='mergesort')
        sorted_beads = bead_ids_col[sort_idx]
        sorted_atoms = atom_ids_col[sort_idx]
        sorted_types = bead_types_col[sort_idx]

        # 找分组边界
        unique_bead_ids, first_idx, counts = np.unique(
            sorted_beads, return_index=True, return_counts=True
        )

        self._n_beads = len(unique_bead_ids)
        self._bead_ids_sorted = unique_bead_ids

        # 预计算扁平化索引
        all_atom_ids = []
        all_bead_rows = []
        for i, bead_id in enumerate(unique_bead_ids):
            start = first_idx[i]
            end = start + counts[i]
            self.bead_to_atoms[bead_id] = sorted_atoms[start:end]
            self.bead_to_type[bead_id] = int(sorted_types[start])
            all_atom_ids.append(sorted_atoms[start:end])
            all_bead_rows.append(np.full(counts[i], i, dtype=np.int32))

        all_atom_ids_flat = np.concatenate(all_atom_ids)
        self._flat_atom_indices = all_atom_ids_flat - 1  # 转 0-based
        self._flat_bead_indices = np.concatenate(all_bead_rows)

    def _convert_with_cache(self, atom_coords: np.ndarray,
                            cg_compare_list: np.ndarray,
                            mass_list: Dict[int, float]) -> np.ndarray:
        """
        使用缓存的索引进行转换（全向量化，无 Python 循环）
        """
        n_beads = self._n_beads
        has_masses = atom_coords.shape[1] >= 5

        # 批量获取所有原子坐标和类型
        atom_indices_0based = self._flat_atom_indices

        if has_masses:
            atom_xyz = atom_coords[atom_indices_0based, 2:5]
            atom_types = atom_coords[atom_indices_0based, 1].astype(np.int32)
        else:
            atom_xyz = atom_coords[atom_indices_0based, :3]
            atom_types = np.ones(len(atom_indices_0based), dtype=np.int32)

        # 向量化质量获取
        unique_types = np.unique(atom_types)
        type_to_mass = {int(t): mass_list.get(int(t), 1.0) for t in unique_types}
        mass_values = np.array([type_to_mass[int(t)] for t in atom_types], dtype=np.float64)

        # 分组质心计算: np.add.at 做 scatter-add
        weighted = mass_values[:, np.newaxis] * atom_xyz  # (n_atoms, 3)

        sum_weighted = np.zeros((n_beads, 3), dtype=np.float64)
        sum_masses = np.zeros(n_beads, dtype=np.float64)
        np.add.at(sum_weighted, self._flat_bead_indices, weighted)
        np.add.at(sum_masses, self._flat_bead_indices, mass_values)

        # 归一化
        valid = sum_masses > 0
        centroids = np.zeros((n_beads, 3), dtype=np.float64)
        centroids[valid] = sum_weighted[valid] / sum_masses[valid, np.newaxis]

        # 构建输出
        cg_trj = np.zeros((n_beads, 5), dtype=np.float64)
        cg_trj[:, 0] = self._bead_ids_sorted
        cg_trj[:, 1] = np.array([self.bead_to_type[bid] for bid in self._bead_ids_sorted], dtype=np.float64)
        cg_trj[:, 2:5] = centroids

        return cg_trj

    def _calculate_central_mass(self, coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """
        计算质心

        Args:
            coords: 坐标 (n, 3)
            masses: 质量 (n,)

        Returns:
            质心坐标 (3,)
        """
        total_mass = np.sum(masses)
        if total_mass == 0:
            return np.mean(coords, axis=0)

        weighted_coords = masses[:, np.newaxis] * coords
        central_mass = np.sum(weighted_coords, axis=0) / total_mass
        return central_mass


def lammpstrj2cg(atom_coords: np.ndarray,
                 cg_compare_list: np.ndarray,
                 mass_list: Dict[int, float],
                 converter: Optional[CGConverter] = None) -> np.ndarray:
    """
    便捷函数：全原子坐标转粗粒化坐标 (兼容原run.py接口)

    Args:
        atom_coords: 原子坐标 (n_atoms, 5+) [id, type, x, y, z, ...]
        cg_compare_list: CG映射 (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]
        mass_list: 原子类型到质量的映射
        converter: 可选的CGConverter实例 (用于缓存)

    Returns:
        CG坐标 (n_beads, 5) [bead_id, bead_type, x, y, z]
    """
    if converter is None:
        converter = CGConverter()

    return converter.convert(atom_coords, cg_compare_list, mass_list)


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("测试CG坐标转换器")
    print("=" * 60)

    # 创建测试数据
    n_atoms = 1000
    n_beads = 100

    # 原子坐标
    np.random.seed(42)
    atom_coords = np.zeros((n_atoms, 5), dtype=np.float64)
    atom_coords[:, 0] = np.arange(1, n_atoms + 1)  # id
    atom_coords[:, 1] = np.random.randint(1, 5, n_atoms)  # type
    atom_coords[:, 2:5] = np.random.rand(n_atoms, 3) * 50  # coords

    # CG映射
    atoms_per_bead = n_atoms // n_beads
    cg_compare_list = []
    for bead_id in range(1, n_beads + 1):
        start_atom = (bead_id - 1) * atoms_per_bead + 1
        for i in range(atoms_per_bead):
            atom_id = start_atom + i
            mass = 12.0
            cg_compare_list.append([bead_id, bead_id % 10 + 1, bead_id % 10 + 1, atom_id, mass])

    cg_compare_list = np.array(cg_compare_list, dtype=np.float64)

    # 质量映射
    mass_list = {1: 12.0, 2: 14.0, 3: 16.0, 4: 1.0}

    # 测试转换
    print("\n首次转换 (构建索引):")
    converter = CGConverter()
    start = time.time()
    cg_coords = converter.convert(atom_coords, cg_compare_list, mass_list)
    time1 = time.time() - start
    print(f"  耗时: {time1*1000:.2f} ms")
    print(f"  结果形状: {cg_coords.shape}")
    print(f"  Bead数: {len(cg_coords)}")

    # 测试缓存效果
    print("\n第二次转换 (使用缓存):")
    start = time.time()
    cg_coords2 = converter.convert(atom_coords, cg_compare_list, mass_list)
    time2 = time.time() - start
    print(f"  耗时: {time2*1000:.2f} ms")
    print(f"  加速比: {time1/time2:.2f}x")

    # 验证结果一致
    if np.allclose(cg_coords, cg_coords2):
        print("  ✅ 结果一致")
    else:
        print("  ❌ 结果不一致")

    # 测试缓存失效
    print("\n测试缓存失效:")
    converter.invalidate_cache()
    start = time.time()
    cg_coords3 = converter.convert(atom_coords, cg_compare_list, mass_list)
    time3 = time.time() - start
    print(f"  耗时: {time3*1000:.2f} ms")

    print("\n✅ 测试完成!")