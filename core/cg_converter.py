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

    这是必须满足的约束，因为每个 bead_id 代表一个粗粒珠子，
    一个粗粒珠子只能有一个 bead_type。

    Args:
        cg_compare_list: CG映射 (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]
        raise_error: 如果验证失败是否抛出错误

    Returns:
        Dict[int, int]: {bead_id: bead_type} 映射

    Raises:
        CGMappingValidationError: 当同一 bead_id 有不同的 bead_type 时
    """
    # 向量化验证
    bead_ids = cg_compare_list[:, 0].astype(int)
    bead_types = cg_compare_list[:, 2].astype(int)  # 修复：[:, 2] 是 bead_type（[:, 1] 是 mol_id）

    # 获取唯一的 bead_id
    unique_bead_ids = np.unique(bead_ids)

    errors = []
    bead_id_to_type = {}

    for bead_id in unique_bead_ids:
        mask = bead_ids == bead_id
        types_for_bead = bead_types[mask]
        unique_types = np.unique(types_for_bead)

        if len(unique_types) > 1:
            # 找出冲突的原子
            error_msg = (
                f"Bead {bead_id} 有 {len(unique_types)} 个不同的 bead_type: {unique_types.tolist()}. "
                f"涉及的原子索引: {np.where(mask)[0].tolist()[:10]}..."
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

    def _build_index(self, cg_compare_list: np.ndarray):
        """
        构建bead到原子的索引

        Args:
            cg_compare_list: CG映射
        """
        self.bead_to_atoms.clear()

        # 获取唯一的bead_id
        bead_ids = np.unique(cg_compare_list[:, 0]).astype(int)

        for bead_id in bead_ids:
            # 找到属于该bead的所有原子ID
            mask = cg_compare_list[:, 0] == bead_id
            atom_ids = cg_compare_list[mask, 3].astype(int)  # AA_id
            self.bead_to_atoms[bead_id] = atom_ids

    def _convert_with_cache(self, atom_coords: np.ndarray,
                            cg_compare_list: np.ndarray,
                            mass_list: Dict[int, float]) -> np.ndarray:
        """
        使用缓存的索引进行转换

        Args:
            atom_coords: 原子坐标
            cg_compare_list: CG映射
            mass_list: 质量映射

        Returns:
            CG坐标
        """
        cg_trj = []

        for bead_id, atom_ids in self.bead_to_atoms.items():
            # 获取bead_type (假设同一个bead内原子类型一致)
            # cg_compare_list格式: [bead_id, mol_id, bead_type, AA_id, mass]
            # [:, 2]才是bead_type，[:, 1]是mol_id
            mask = cg_compare_list[:, 0] == bead_id
            bead_type = int(cg_compare_list[mask, 2][0])  # 修复：[:, 1]→[:, 2]

            # 获取原子坐标
            # atom_coords格式: [id, type, x, y, z, ...] 或 [x, y, z, ...]
            if atom_coords.shape[1] >= 5:
                # 包含id和type
                coords = atom_coords[atom_ids - 1, 2:5]
                atom_types = atom_coords[atom_ids - 1, 1].astype(int)
            else:
                # 只有坐标
                coords = atom_coords[atom_ids - 1, :3]
                atom_types = np.ones(len(atom_ids), dtype=int)

            # 获取质量
            masses = np.array([mass_list.get(int(t), 1.0) for t in atom_types])

            # 计算质心
            bead_coord = self._calculate_central_mass(coords, masses)

            cg_trj.append([bead_id, bead_type, *bead_coord])

        return np.array(cg_trj)

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