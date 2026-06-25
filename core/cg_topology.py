"""
CG 拓扑生成器模块

功能:
- 从 CG bonds 推导 angles 和 dihedrals
- CG 拓扑数据结构
- 根据 bead type 组合分配拓扑类型

移植自: md_base_on_ml/AA_data2CG_data/PIP_polymerization/create_cg_info.py

作者: Claude
日期: 2026-03-27
"""

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict
import numpy as np


def _assign_topology_types_local(topology_array: np.ndarray,
                                  bead_types: Dict[int, int],
                                  topology_kind: str,
                                  type_mapping: Optional[Dict[Tuple, int]] = None) -> np.ndarray:
    """
    根据 bead type 组合分配拓扑类型 ID（本地辅助函数）。

    回环结构的 angle 和 dihedral 合并为同一类型：
    - Angle: (1,1,2) 和 (2,1,1) 合并
    - Dihedral: (1,1,2,2) 和 (2,2,1,1) 合并

    Args:
        topology_array: 拓扑数组 [type, bead1, bead2, ...]
        bead_types: {bead_id: bead_type}
        topology_kind: 'bond', 'angle', 'dihedral'
        type_mapping: YAML 提供的组合→类型ID预填表（规范化后的组合为key），
                      未列出的组合自动分配ID，从 max(yaml_types)+1 开始

    Returns:
        更新 type 列后的拓扑数组
    """
    if len(topology_array) == 0:
        return topology_array

    # 收集所有 bead type 组合
    type_combinations = {}

    # 预填充：YAML 中的组合优先占据类型 ID
    if type_mapping:
        for combo, tid in type_mapping.items():
            type_combinations[combo] = tid
        next_type_id = max(type_mapping.values()) + 1
    else:
        next_type_id = 1

    for topo in topology_array:
        bead_ids = topo[1:]  # 获取 bead IDs
        bead_type_combo = tuple(bead_types.get(int(bid), 1) for bid in bead_ids)

        # 规范化 bead type 组合
        if topology_kind == 'bond':
            # Bond: (1,2) 和 (2,1) 合并
            bead_type_combo = tuple(sorted(bead_type_combo))
        elif topology_kind == 'angle':
            # Angle: 首尾对称合并，(1,1,2) 和 (2,1,1) 合并
            # 格式: (type1, type_center, type3)
            if bead_type_combo[0] > bead_type_combo[2]:
                bead_type_combo = (bead_type_combo[2], bead_type_combo[1], bead_type_combo[0])
        elif topology_kind == 'dihedral':
            # Dihedral: 首尾对称合并，(1,1,2,2) 和 (2,2,1,1) 合并
            # 格式: (type1, type2, type3, type4)
            # 先比较首尾，再比较中间
            reversed_combo = (bead_type_combo[3], bead_type_combo[2], bead_type_combo[1], bead_type_combo[0])
            if bead_type_combo > reversed_combo:
                bead_type_combo = reversed_combo

        # 分配 type ID（查表优先，预填充的组合已存在）
        if bead_type_combo not in type_combinations:
            type_combinations[bead_type_combo] = next_type_id
            next_type_id += 1

        topo[0] = type_combinations[bead_type_combo]

    return topology_array


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
                             bead_types: Optional[Dict[int, int]] = None,
                             default_angle_type: int = 1,
                             type_mapping: Optional[Dict[Tuple, int]] = None) -> np.ndarray:
    """
    从键列表推导角度列表，并根据 bead type 分配 angle type。

    角度存在条件: 三个珠子连接 bead1-bead2-bead3

    Args:
        cg_bonds: CG键数组 (n_bonds, 3) [bond_type, bead1, bead2]
        bead_types: {bead_id: bead_type} 映射（可选，用于分配正确的 angle type）
        default_angle_type: 默认角度类型（当 bead_types 未提供时使用）
        type_mapping: YAML 提供的 angle 组合→类型ID预填表

    Returns:
        angles: 角度数组 (n_angles, 4) [angle_type, bead1, bead2, bead3]
                bead2 是中心原子
    """
    if len(cg_bonds) == 0:
        return np.array([], dtype=np.int32).reshape(0, 4)

    # 从键构建邻接表
    adjacency = defaultdict(set)
    for bond in cg_bonds:
        bead1, bead2 = int(bond[1]), int(bond[2])
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

                # 暂时使用 default_angle_type，后续根据 bead_types 分配
                angles_set.add((default_angle_type, bead1, center_bead, bead3))

    # 转换为排序的 numpy 数组
    if angles_set:
        angles = np.array(sorted(list(angles_set)), dtype=np.int32)
    else:
        angles = np.array([], dtype=np.int32).reshape(0, 4)

    # 根据 bead_types 分配正确的 angle type
    if bead_types and len(angles) > 0:
        angles = _assign_topology_types_local(angles, bead_types, 'angle',
                                               type_mapping=type_mapping)

    return angles


def derive_dihedrals_from_bonds(cg_bonds: np.ndarray,
                                bead_types: Optional[Dict[int, int]] = None,
                                default_dihedral_type: int = 1,
                                type_mapping: Optional[Dict[Tuple, int]] = None) -> np.ndarray:
    """
    从键列表推导二面角列表，并根据 bead type 分配 dihedral type。

    二面角存在条件: 四个珠子连接 bead1-bead2-bead3-bead4

    Args:
        cg_bonds: CG键数组 (n_bonds, 3) [bond_type, bead1, bead2]
        bead_types: {bead_id: bead_type} 映射（可选，用于分配正确的 dihedral type）
        default_dihedral_type: 默认二面角类型（当 bead_types 未提供时使用）
        type_mapping: YAML 提供的 dihedral 组合→类型ID预填表

    Returns:
        dihedrals: 二面角数组 (n_dihedrals, 5) [dihedral_type, bead1, bead2, bead3, bead4]
    """
    if len(cg_bonds) == 0:
        return np.array([], dtype=np.int32).reshape(0, 5)

    # 从键构建邻接表
    adjacency = defaultdict(set)
    for bond in cg_bonds:
        bead1, bead2 = int(bond[1]), int(bond[2])
        adjacency[bead1].add(bead2)
        adjacency[bead2].add(bead1)

    # 找出所有二面角
    dihedrals_set = set()

    # 对于每个键 (b2-b3)，找到 b2 和 b3 的邻居
    for bond in cg_bonds:
        bead2, bead3 = int(bond[1]), int(bond[2])

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

    # 根据 bead_types 分配正确的 dihedral type
    if bead_types and len(dihedrals) > 0:
        dihedrals = _assign_topology_types_local(dihedrals, bead_types, 'dihedral',
                                                  type_mapping=type_mapping)

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
                                  bead_types: Optional[Dict[int, int]] = None,
                                  default_angle_type: int = 1,
                                  default_dihedral_type: int = 1,
                                  angle_type_mapping: Optional[Dict[Tuple, int]] = None,
                                  dihedral_type_mapping: Optional[Dict[Tuple, int]] = None) -> CGTopology:
    """
    从 CG 键推导完整的 CG 拓扑

    Args:
        cg_bonds: CG键数组 (n_bonds, 3) [bond_type, bead1, bead2]
        bead_types: {bead_id: bead_type} 映射（可选，用于分配正确的拓扑类型）
        default_angle_type: 默认角度类型（当 bead_types 未提供时使用）
        default_dihedral_type: 默认二面角类型（当 bead_types 未提供时使用）
        angle_type_mapping: YAML 提供的 angle 组合→类型ID预填表
        dihedral_type_mapping: YAML 提供的 dihedral 组合→类型ID预填表

    Returns:
        CGTopology: 完整的 CG 拓扑
    """
    angles = derive_angles_from_bonds(cg_bonds, bead_types, default_angle_type,
                                       type_mapping=angle_type_mapping)
    dihedrals = derive_dihedrals_from_bonds(cg_bonds, bead_types, default_dihedral_type,
                                             type_mapping=dihedral_type_mapping)

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