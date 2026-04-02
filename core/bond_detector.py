"""
键变化检测器模块

功能:
- 检测反应前后的键变化
- 向量化键编码实现高效比较
- 返回创建的键和删除的键

优化:
- 使用整数编码代替元组比较
- 向量化实现避免Python循环

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass
from typing import Set, Tuple, Optional
import numpy as np


@dataclass
class BondChanges:
    """键变化结果"""
    created_bonds: np.ndarray   # 创建的键 (n_created, 3) [bond_type, atom1, atom2]
    deleted_bonds: np.ndarray   # 删除的键 (n_deleted, 3)
    created_codes: Set[int]     # 创建键的编码集合
    deleted_codes: Set[int]     # 删除键的编码集合

    @property
    def n_created(self) -> int:
        return len(self.created_bonds)

    @property
    def n_deleted(self) -> int:
        return len(self.deleted_bonds)

    @property
    def has_changes(self) -> bool:
        return self.n_created > 0 or self.n_deleted > 0


class BondDetector:
    """
    键变化检测器

    功能:
    - 向量化键编码
    - 高效比较键变化
    """

    def __init__(self, n_atoms: int):
        """
        初始化检测器

        Args:
            n_atoms: 体系中的原子总数
        """
        self.n_atoms = n_atoms
        self.encoder = n_atoms + 1  # 用于编码键

    def encode_bonds(self, bonds: np.ndarray) -> Set[int]:
        """
        将键编码为整数集合

        Args:
            bonds: 键数组 (n_bonds, 3) [bond_type, atom1, atom2]
                   或 (n_bonds, 2) [atom1, atom2]

        Returns:
            编码后的整数集合
        """
        if len(bonds) == 0:
            return set()

        # 提取原子ID
        if bonds.shape[1] == 3:
            atom_pairs = bonds[:, 1:3]
        else:
            atom_pairs = bonds

        # 规范化: atom1 < atom2
        sorted_pairs = np.sort(atom_pairs, axis=1)

        # 编码: code = atom1 * encoder + atom2
        codes = sorted_pairs[:, 0] * self.encoder + sorted_pairs[:, 1]

        return set(codes.tolist())

    def encode_bond(self, atom1: int, atom2: int) -> int:
        """
        编码单个键

        Args:
            atom1: 第一个原子ID
            atom2: 第二个原子ID

        Returns:
            编码后的整数
        """
        a1, a2 = min(atom1, atom2), max(atom1, atom2)
        return a1 * self.encoder + a2

    def decode_code(self, code: int) -> Tuple[int, int]:
        """
        解码单个编码

        Args:
            code: 编码后的整数

        Returns:
            (atom1, atom2) 元组
        """
        atom1 = code // self.encoder
        atom2 = code % self.encoder
        return (atom1, atom2)

    def detect(self, bonds_before: np.ndarray, bonds_after: np.ndarray) -> BondChanges:
        """
        检测键变化

        Args:
            bonds_before: 反应前的键 (n_bonds, 3) [bond_type, atom1, atom2]
            bonds_after: 反应后的键 (n_bonds, 3)

        Returns:
            BondChanges: 键变化结果
        """
        # 编码键
        codes_before = self.encode_bonds(bonds_before)
        codes_after = self.encode_bonds(bonds_after)

        # 计算差异
        created_codes = codes_after - codes_before
        deleted_codes = codes_before - codes_after

        # 解码回键数组
        created_bonds = self._codes_to_bonds(created_codes, bonds_after)
        deleted_bonds = self._codes_to_bonds(deleted_codes, bonds_before)

        return BondChanges(
            created_bonds=created_bonds,
            deleted_bonds=deleted_bonds,
            created_codes=created_codes,
            deleted_codes=deleted_codes
        )

    def _codes_to_bonds(self, codes: Set[int], original_bonds: np.ndarray) -> np.ndarray:
        """
        将编码集合转换回键数组 (包含键类型)

        Args:
            codes: 编码集合
            original_bonds: 原始键数组 (用于获取键类型)

        Returns:
            键数组 (n_bonds, 3) 或空数组
        """
        if not codes:
            return np.array([]).reshape(0, 3)

        # 解码原子对
        bonds = []
        for code in codes:
            atom1, atom2 = self.decode_code(code)
            # 查找键类型
            bond_type = self._find_bond_type(atom1, atom2, original_bonds)
            bonds.append([bond_type, atom1, atom2])

        return np.array(bonds, dtype=np.int32)

    def _find_bond_type(self, atom1: int, atom2: int, bonds: np.ndarray) -> int:
        """
        在键数组中查找键类型

        Args:
            atom1: 第一个原子ID
            atom2: 第二个原子ID
            bonds: 键数组

        Returns:
            键类型 (默认1)
        """
        if len(bonds) == 0:
            return 1

        # 规范化
        a1, a2 = min(atom1, atom2), max(atom1, atom2)

        for bond in bonds:
            b1, b2 = min(bond[1], bond[2]), max(bond[1], bond[2])
            if b1 == a1 and b2 == a2:
                return int(bond[0])

        return 1  # 默认键类型


def compare_two_bonds(bonds1: np.ndarray, bonds2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    比较两组键的差异 (兼容原run.py接口)

    Args:
        bonds1: 第一组键 (n_bonds, 3)
        bonds2: 第二组键 (n_bonds, 3)

    Returns:
        (bonds1_only, bonds2_only): 仅在bonds1中的键和仅在bonds2中的键
    """
    if len(bonds1) == 0 and len(bonds2) == 0:
        return np.array([]).reshape(0, 3), np.array([]).reshape(0, 3)

    # 估算原子数
    if len(bonds1) > 0:
        n_atoms = max(bonds1[:, 1:3].max(), bonds2[:, 1:3].max() if len(bonds2) > 0 else 0)
    else:
        n_atoms = bonds2[:, 1:3].max()

    detector = BondDetector(int(n_atoms))
    changes = detector.detect(bonds1, bonds2)

    return changes.deleted_bonds, changes.created_bonds


def get_changed_atoms(bond_changes: BondChanges) -> Set[int]:
    """
    从键变化中提取参与反应的原子

    Args:
        bond_changes: 键变化结果

    Returns:
        参与反应的原子ID集合
    """
    atoms = set()

    for bond in bond_changes.created_bonds:
        atoms.add(int(bond[1]))
        atoms.add(int(bond[2]))

    for bond in bond_changes.deleted_bonds:
        atoms.add(int(bond[1]))
        atoms.add(int(bond[2]))

    return atoms


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("测试键变化检测器")
    print("=" * 60)

    # 创建测试数据
    np.random.seed(42)
    n_atoms = 10000
    n_bonds = 5000

    bonds1 = np.zeros((n_bonds, 3), dtype=np.int32)
    for i in range(n_bonds):
        a1 = np.random.randint(1, n_atoms + 1)
        a2 = np.random.randint(1, n_atoms + 1)
        if a1 != a2:
            bonds1[i] = [1, a1, a2]

    bonds2 = bonds1.copy()

    # 修改一些键
    bonds2[0] = [1, 9999, 9998]  # 删除原键，创建新键
    bonds2[1] = [1, 8888, 8887]
    bonds2[2] = [2, 7777, 7776]

    print("\n测试向量化键编码:")
    detector = BondDetector(n_atoms)

    # 编码测试
    start = time.time()
    codes1 = detector.encode_bonds(bonds1)
    codes2 = detector.encode_bonds(bonds2)
    encode_time = time.time() - start
    print(f"  编码时间: {encode_time*1000:.3f} ms")
    print(f"  编码数量: {len(codes1)}, {len(codes2)}")

    # 检测变化
    print("\n测试键变化检测:")
    start = time.time()
    changes = detector.detect(bonds1, bonds2)
    detect_time = time.time() - start
    print(f"  检测时间: {detect_time*1000:.3f} ms")
    print(f"  创建的键: {changes.n_created}")
    print(f"  删除的键: {changes.n_deleted}")
    print(f"  有变化: {changes.has_changes}")

    if changes.n_created > 0:
        print(f"  创建键示例: {changes.created_bonds[:3]}")
    if changes.n_deleted > 0:
        print(f"  删除键示例: {changes.deleted_bonds[:3]}")

    # 测试参与反应的原子
    changed_atoms = get_changed_atoms(changes)
    print(f"\n参与反应的原子数: {len(changed_atoms)}")

    # 测试兼容接口
    print("\n测试兼容接口 compare_two_bonds:")
    x_only, y_only = compare_two_bonds(bonds1, bonds2)
    print(f"  仅在bonds1中: {len(x_only)} 个键")
    print(f"  仅在bonds2中: {len(y_only)} 个键")

    print("\n✅ 测试完成!")