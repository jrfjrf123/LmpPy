"""
CG 拓扑生成器模块

功能:
- 从 CG bonds 推导 angles 和 dihedrals
- CG 拓扑数据结构

移植自: md_base_on_ml/AA_data2CG_data/PIP_polymerization/create_cg_info.py

作者: Claude
日期: 2026-03-27
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np


@dataclass
class CGTopology:
    """
    CG 拓扑数据结构

    存储 bonds, angles, dihedrals
    """
    bonds: np.ndarray      # (n_bonds, 3) [bond_type, bead1, bead2]
    angles: np.ndarray     # (n_angles, 4) [angle_type, bead1, bead2, bead3]
    dihedrals: np.ndarray  # (n_dihedrals, 5) [dihedral_type, bead1, bead2, bead3, bead4]

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    @property
    def n_dihedrals(self) -> int:
        return len(self.dihedrals)

    def to_files(self, prefix: str):
        """
        保存到文件

        Args:
            prefix: 文件名前缀，如 "cg" 生成 cg_bonds.txt, cg_angles.txt, cg_dihedrals.txt
        """
        if len(self.bonds) > 0:
            np.savetxt(f'{prefix}_bonds.txt', self.bonds, fmt='%d')
        if len(self.angles) > 0:
            np.savetxt(f'{prefix}_angles.txt', self.angles, fmt='%d')
        if len(self.dihedrals) > 0:
            np.savetxt(f'{prefix}_dihedrals.txt', self.dihedrals, fmt='%d')


def derive_angles_from_bonds(cg_bonds: np.ndarray,
                             default_angle_type: int = 1) -> np.ndarray:
    """
    从键列表推导角度列表

    角度存在条件: 三个珠子连接 bead1-bead2-bead3

    Args:
        cg_bonds: CG键数组 (n_bonds, 3) [bond_type, bead1, bead2]
        default_angle_type: 默认角度类型

    Returns:
        angles: 角度数组 (n_angles, 4) [angle_type, bead1, bead2, bead3]
                bead2 是中心原子
    """
    if len(cg_bonds) == 0:
        return np.array([], dtype=np.int32).reshape(0, 4)

    # 从键构建邻接表
    adjacency = defaultdict(set)
    for bond in cg_bonds:
        bead1, bead2 = bond[1], bond[2]
        adjacency[bead1].add(bead2)
        adjacency[bead2].add(bead1)

    # 找出所有角度
    angles_set = set()

    for center_bead in adjacency:
        neighbors = list(adjacency[center_bead])

        # 对于连接到中心原子的每对邻居
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                bead1 = neighbors[i]
                bead3 = neighbors[j]

                # 创建角度，中心原子在中间
                # 对 bead1 和 bead3 排序以避免重复
                if bead1 > bead3:
                    bead1, bead3 = bead3, bead1

                angles_set.add((default_angle_type, bead1, center_bead, bead3))

    # 转换为排序的 numpy 数组
    if angles_set:
        angles = np.array(sorted(list(angles_set)), dtype=np.int32)
    else:
        angles = np.array([], dtype=np.int32).reshape(0, 4)

    return angles


def derive_dihedrals_from_bonds(cg_bonds: np.ndarray,
                                default_dihedral_type: int = 1) -> np.ndarray:
    """
    从键列表推导二面角列表

    二面角存在条件: 四个珠子连接 bead1-bead2-bead3-bead4

    Args:
        cg_bonds: CG键数组 (n_bonds, 3) [bond_type, bead1, bead2]
        default_dihedral_type: 默认二面角类型

    Returns:
        dihedrals: 二面角数组 (n_dihedrals, 5) [dihedral_type, bead1, bead2, bead3, bead4]
    """
    if len(cg_bonds) == 0:
        return np.array([], dtype=np.int32).reshape(0, 5)

    # 从键构建邻接表
    adjacency = defaultdict(set)
    for bond in cg_bonds:
        bead1, bead2 = bond[1], bond[2]
        adjacency[bead1].add(bead2)
        adjacency[bead2].add(bead1)

    # 找出所有二面角
    dihedrals_set = set()

    # 对于每个键 (b2-b3)，找到 b2 和 b3 的邻居
    for bond in cg_bonds:
        bead2, bead3 = bond[1], bond[2]

        # 获取 bead2 的邻居 (排除 bead3)
        neighbors_b2 = adjacency[bead2] - {bead3}
        # 获取 bead3 的邻居 (排除 bead2)
        neighbors_b3 = adjacency[bead3] - {bead2}

        # 创建二面角: bead1-bead2-bead3-bead4
        for bead1 in neighbors_b2:
            for bead4 in neighbors_b3:
                # 避免创建重复的二面角
                # 使用规范形式: 如果 bead1 > bead4，交换顺序
                if bead1 < bead4:
                    dihedrals_set.add((default_dihedral_type, bead1, bead2, bead3, bead4))
                elif bead1 > bead4:
                    dihedrals_set.add((default_dihedral_type, bead4, bead3, bead2, bead1))
                # 如果 bead1 == bead4，跳过 (形成3元环)

    # 转换为排序的 numpy 数组
    if dihedrals_set:
        dihedrals = np.array(sorted(list(dihedrals_set)), dtype=np.int32)
    else:
        dihedrals = np.array([], dtype=np.int32).reshape(0, 5)

    return dihedrals


def create_cg_bead_info(cg_compare_list: np.ndarray) -> np.ndarray:
    """
    从 CG 映射创建 CG 珠子信息

    Args:
        cg_compare_list: CG映射数组 (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]

    Returns:
        cg_bead_info: 珠子信息数组 (n_beads, 4) [bead_id, mol_id, bead_type, total_mass]
    """
    import pandas as pd

    # 创建 DataFrame
    df = pd.DataFrame(cg_compare_list, columns=['bead_id', 'mol_id', 'bead_type', 'AA_id', 'mass'])

    # 按 bead_id 分组并聚合
    bead_agg = df.groupby('bead_id').agg({
        'mol_id': 'first',
        'bead_type': 'first',
        'mass': 'sum'
    }).reset_index()

    bead_agg = bead_agg.sort_values('bead_id')

    # 创建 bead info 数组
    n_beads = len(bead_agg)
    cg_bead_info = np.zeros((n_beads, 4), dtype=np.float64)

    cg_bead_info[:, 0] = bead_agg['bead_id'].to_numpy()
    cg_bead_info[:, 1] = bead_agg['mol_id'].to_numpy()
    cg_bead_info[:, 2] = bead_agg['bead_type'].to_numpy()
    cg_bead_info[:, 3] = bead_agg['mass'].to_numpy()

    return cg_bead_info


def verify_molecule_ids_consistency(cg_compare_list: np.ndarray,
                                    cg_bonds: np.ndarray) -> Tuple[bool, List]:
    """
    验证分子ID与键连关系一致

    同一 CG bond 的两个珠子应有相同的分子ID

    Args:
        cg_compare_list: CG映射数组
        cg_bonds: CG键数组

    Returns:
        (is_consistent, inconsistent_bonds):
            - is_consistent: 是否一致
            - inconsistent_bonds: 不一致的键列表 [(bond, mol1, mol2), ...]
    """
    import pandas as pd

    # 创建映射: bead_id -> mol_id
    df = pd.DataFrame(cg_compare_list, columns=['bead_id', 'mol_id', 'bead_type', 'AA_id', 'mass'])
    bead_mol_df = df[['bead_id', 'mol_id']].drop_duplicates()
    bead_to_mol = dict(zip(bead_mol_df['bead_id'], bead_mol_df['mol_id']))

    inconsistent_bonds = []
    for bond in cg_bonds:
        bead1, bead2 = bond[1], bond[2]
        mol1 = bead_to_mol.get(bead1)
        mol2 = bead_to_mol.get(bead2)

        if mol1 != mol2:
            inconsistent_bonds.append((tuple(bond), mol1, mol2))

    is_consistent = len(inconsistent_bonds) == 0
    return is_consistent, inconsistent_bonds


def derive_cg_topology_from_bonds(cg_bonds: np.ndarray,
                                  default_angle_type: int = 1,
                                  default_dihedral_type: int = 1) -> CGTopology:
    """
    从 CG 键推导完整的 CG 拓扑

    Args:
        cg_bonds: CG键数组 (n_bonds, 3) [bond_type, bead1, bead2]
        default_angle_type: 默认角度类型
        default_dihedral_type: 默认二面角类型

    Returns:
        CGTopology: 完整的 CG 拓扑
    """
    angles = derive_angles_from_bonds(cg_bonds, default_angle_type)
    dihedrals = derive_dihedrals_from_bonds(cg_bonds, default_dihedral_type)

    return CGTopology(bonds=cg_bonds, angles=angles, dihedrals=dihedrals)


if __name__ == "__main__":
    print("=" * 60)
    print("测试 CG 拓扑生成器")
    print("=" * 60)

    # 创建测试数据 - 线性链
    cg_bonds = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [1, 3, 4],
        [1, 4, 5],
    ], dtype=np.int32)

    print("\n输入 CG 键:")
    print(cg_bonds)

    # 推导角度
    angles = derive_angles_from_bonds(cg_bonds)
    print(f"\n推导的角度 ({len(angles)} 个):")
    print(angles)

    # 推导二面角
    dihedrals = derive_dihedrals_from_bonds(cg_bonds)
    print(f"\n推导的二面角 ({len(dihedrals)} 个):")
    print(dihedrals)

    # 创建完整拓扑
    topology = derive_cg_topology_from_bonds(cg_bonds)
    print(f"\n完整拓扑:")
    print(f"  键: {topology.n_bonds}")
    print(f"  角度: {topology.n_angles}")
    print(f"  二面角: {topology.n_dihedrals}")

    print("\n✅ 测试完成!")