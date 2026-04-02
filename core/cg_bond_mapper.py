"""
粗粒键映射器模块

功能:
- 原子键 → 粗粒键转换
- 基于bead_id判断是否创建粗粒键
- 向量化实现

核心规则:
- 新键连接不同bead: 创建粗粒键
- 新键连接同一bead: 忽略

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional
import numpy as np

# 支持两种导入方式
try:
    from .cg_mapper import CGMapping
except ImportError:
    from cg_mapper import CGMapping


@dataclass
class CGBond:
    """粗粒键数据"""
    bead1: int
    bead2: int
    bond_type: int


class CGBondMapper:
    """
    粗粒键映射器

    功能:
    - 将原子键转换为粗粒键
    - 忽略同一bead内的键
    """

    def __init__(self, cg_mapping: CGMapping):
        """
        初始化映射器

        Args:
            cg_mapping: CG映射对象
        """
        self.cg_mapping = cg_mapping

    def atom_bonds_to_cg_bonds(self, atom_bonds: np.ndarray,
                               default_bond_type: int = 1) -> np.ndarray:
        """
        将原子键转换为粗粒键

        Args:
            atom_bonds: 原子键数组 (n_bonds, 3) [bond_type, atom1, atom2]
            default_bond_type: 默认粗粒键类型

        Returns:
            粗粒键数组 (n_cg_bonds, 3) [bond_type, bead1, bead2]
        """
        if len(atom_bonds) == 0:
            return np.array([]).reshape(0, 3)

        # 获取每个键连接的原子对的bead_id
        atom1_ids = atom_bonds[:, 1].astype(int)
        atom2_ids = atom_bonds[:, 2].astype(int)

        # 获取bead_id
        bead1_ids = self.cg_mapping.data[atom1_ids - 1, 0]
        bead2_ids = self.cg_mapping.data[atom2_ids - 1, 0]

        # 筛选跨bead的键
        mask = bead1_ids != bead2_ids

        if not mask.any():
            return np.array([]).reshape(0, 3)

        # 提取有效的粗粒键
        valid_bead1 = bead1_ids[mask]
        valid_bead2 = bead2_ids[mask]
        valid_types = atom_bonds[mask, 0]

        # 规范化: bead1 < bead2
        sorted_beads = np.sort(np.column_stack([valid_bead1, valid_bead2]), axis=1)

        # 去重
        cg_bonds_set = set()
        for i in range(len(sorted_beads)):
            b1, b2 = int(sorted_beads[i, 0]), int(sorted_beads[i, 1])
            cg_bonds_set.add((default_bond_type, b1, b2))

        # 转换为数组
        if cg_bonds_set:
            return np.array(list(cg_bonds_set), dtype=np.int32)
        else:
            return np.array([]).reshape(0, 3)

    def update_cg_bonds(self, existing_cg_bonds: np.ndarray,
                        new_atom_bonds: np.ndarray,
                        deleted_atom_bonds: np.ndarray,
                        default_bond_type: int = 1) -> np.ndarray:
        """
        更新粗粒键

        Args:
            existing_cg_bonds: 现有的粗粒键
            new_atom_bonds: 新创建的原子键
            deleted_atom_bonds: 删除的原子键
            default_bond_type: 默认粗粒键类型

        Returns:
            更新后的粗粒键
        """
        # 添加新键
        new_cg_bonds = self.atom_bonds_to_cg_bonds(new_atom_bonds, default_bond_type)

        # 删除键
        deleted_cg_bonds = self.atom_bonds_to_cg_bonds(deleted_atom_bonds, default_bond_type)

        # 合并
        if len(existing_cg_bonds) == 0:
            return new_cg_bonds

        # 转换为集合操作
        existing_set = set(map(tuple, existing_cg_bonds))
        new_set = set(map(tuple, new_cg_bonds))
        deleted_set = set(map(tuple, deleted_cg_bonds))

        # 更新
        result_set = existing_set - deleted_set
        result_set.update(new_set)

        if result_set:
            return np.array(list(result_set), dtype=np.int32)
        else:
            return np.array([]).reshape(0, 3)


def atom_bonds_to_cg_bonds(atom_bonds: np.ndarray,
                           cg_mapping_data: np.ndarray,
                           default_bond_type: int = 1) -> np.ndarray:
    """
    便捷函数：原子键转粗粒键

    Args:
        atom_bonds: 原子键数组 (n_bonds, 3)
        cg_mapping_data: CG映射数据 (n_atoms, 2) [bead_id, bead_type]
        default_bond_type: 默认键类型

    Returns:
        粗粒键数组 (n_cg_bonds, 3)

    Raises:
        ValueError: 如果有原子未被映射（bead_id 为 0）
    """
    if len(atom_bonds) == 0:
        return np.array([]).reshape(0, 3)

    # 获取bead_id
    atom1_ids = atom_bonds[:, 1].astype(int)
    atom2_ids = atom_bonds[:, 2].astype(int)

    # 检查原子是否在映射范围内
    max_atom_id = len(cg_mapping_data)
    if atom1_ids.max() > max_atom_id or atom2_ids.max() > max_atom_id:
        unmapped_atoms = set()
        unmapped_atoms.update(atom1_ids[atom1_ids > max_atom_id])
        unmapped_atoms.update(atom2_ids[atom2_ids > max_atom_id])
        raise ValueError(
            f"CG 映射不完整: 发现 {len(unmapped_atoms)} 个原子未被映射。"
            f"示例: {sorted(list(unmapped_atoms))[:10]}... "
            f"请检查 mapping 配置是否与实际体系一致。"
        )

    bead1_ids = cg_mapping_data[atom1_ids - 1, 0]
    bead2_ids = cg_mapping_data[atom2_ids - 1, 0]

    # 检查是否有 bead_id 为 0 的原子（未被映射）
    unmapped_mask = (bead1_ids == 0) | (bead2_ids == 0)
    if unmapped_mask.any():
        # 找出未映射的原子
        unmapped_atoms = set()
        for i in range(len(atom_bonds)):
            if bead1_ids[i] == 0:
                unmapped_atoms.add(int(atom1_ids[i]))
            if bead2_ids[i] == 0:
                unmapped_atoms.add(int(atom2_ids[i]))
        raise ValueError(
            f"CG 映射不完整: 发现 {len(unmapped_atoms)} 个原子未被映射（bead_id=0）。"
            f"示例: {sorted(list(unmapped_atoms))[:10]}... "
            f"请检查 mapping 配置是否与实际体系一致。"
        )

    # 筛选跨bead的键
    mask = bead1_ids != bead2_ids

    if not mask.any():
        return np.array([]).reshape(0, 3)

    valid_bead1 = bead1_ids[mask]
    valid_bead2 = bead2_ids[mask]

    # 规范化并去重
    sorted_beads = np.sort(np.column_stack([valid_bead1, valid_bead2]), axis=1)
    cg_bonds_set = set()
    for i in range(len(sorted_beads)):
        b1, b2 = int(sorted_beads[i, 0]), int(sorted_beads[i, 1])
        cg_bonds_set.add((default_bond_type, b1, b2))

    if cg_bonds_set:
        return np.array(list(cg_bonds_set), dtype=np.int32)
    else:
        return np.array([]).reshape(0, 3)


if __name__ == "__main__":
    print("=" * 60)
    print("测试粗粒键映射器")
    print("=" * 60)

    # 创建测试数据
    n_atoms = 20

    # CG映射: 每4个原子一个bead
    cg_mapping_data = np.zeros((n_atoms, 2), dtype=np.int32)
    for i in range(n_atoms):
        bead_id = (i // 4) + 1
        bead_type = bead_id
        cg_mapping_data[i] = [bead_id, bead_type]

    cg_mapping = CGMapping(data=cg_mapping_data, n_atoms=n_atoms)

    # 原子键
    atom_bonds = np.array([
        [1, 1, 2],    # 同bead (bead 1)
        [1, 1, 5],    # 跨bead (bead 1 → bead 2)
        [1, 4, 5],    # 跨bead (bead 1 → bead 2)
        [1, 8, 9],    # 跨bead (bead 2 → bead 3)
        [1, 10, 11],  # 同bead (bead 3)
    ], dtype=np.int32)

    print("\n原子键:")
    for bond in atom_bonds:
        a1, a2 = bond[1], bond[2]
        b1 = cg_mapping_data[a1 - 1, 0]
        b2 = cg_mapping_data[a2 - 1, 0]
        print(f"  原子 {a1}-{a2} → Bead {b1}-{b2} {'(同bead, 忽略)' if b1 == b2 else '(跨bead)'}")

    # 转换
    mapper = CGBondMapper(cg_mapping)
    cg_bonds = mapper.atom_bonds_to_cg_bonds(atom_bonds)

    print(f"\n粗粒键数量: {len(cg_bonds)}")
    print("粗粒键:")
    for bond in cg_bonds:
        print(f"  Bead {bond[1]}-{bond[2]}")

    # 测试便捷函数
    print("\n测试便捷函数:")
    cg_bonds2 = atom_bonds_to_cg_bonds(atom_bonds, cg_mapping_data)
    print(f"  粗粒键数量: {len(cg_bonds2)}")

    print("\n✅ 测试完成!")