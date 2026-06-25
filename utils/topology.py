"""
拓扑文件读写工具

提供统一的拓扑文件读写接口，支持：
- LAMMPS格式键/角度/二面角文件
- 类型字典文件

作者: Claude
日期: 2026-04-02
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class TopologyData:
    """拓扑数据结构"""
    bonds: np.ndarray      # (nbonds, 3) [type, atom1, atom2]
    angles: np.ndarray     # (nangles, 4) [type, atom1, atom2, atom3]
    dihedrals: np.ndarray  # (ndihedrals, 5) [type, atom1, atom2, atom3, atom4]

    @property
    def n_bonds(self) -> int:
        return len(self.bonds)

    @property
    def n_angles(self) -> int:
        return len(self.angles)

    @property
    def n_dihedrals(self) -> int:
        return len(self.dihedrals)


def read_topology_files(bonds_file: str, angles_file: str = None,
                        dihedrals_file: str = None) -> TopologyData:
    """
    读取拓扑文件。

    Args:
        bonds_file: 键文件路径
        angles_file: 角度文件路径（可选）
        dihedrals_file: 二面角文件路径（可选）

    Returns:
        TopologyData: 拓扑数据对象
    """
    # 读取键
    if Path(bonds_file).exists():
        bonds_df = pd.read_csv(bonds_file, sep=r'\s+', comment='#')
        bonds = bonds_df[['bond_type', 'atom1_id', 'atom2_id']].values.astype(np.int32)
    else:
        bonds = np.array([], dtype=np.int32).reshape(0, 3)

    # 读取角度
    if angles_file and Path(angles_file).exists():
        angles_df = pd.read_csv(angles_file, sep=r'\s+', comment='#')
        angles = angles_df[['angle_type', 'atom1_id', 'atom2_id', 'atom3_id']].values.astype(np.int32)
    else:
        angles = np.array([], dtype=np.int32).reshape(0, 4)

    # 读取二面角
    if dihedrals_file and Path(dihedrals_file).exists():
        dihedrals_df = pd.read_csv(dihedrals_file, sep=r'\s+', comment='#')
        dihedrals = dihedrals_df[['dihedral_type', 'atom1_id', 'atom2_id', 'atom3_id', 'atom4_id']].values.astype(np.int32)
    else:
        dihedrals = np.array([], dtype=np.int32).reshape(0, 5)

    return TopologyData(bonds=bonds, angles=angles, dihedrals=dihedrals)


def read_type_dict(file_path: str, n_beads: int) -> Dict[int, Tuple]:
    """
    读取类型字典文件。

    Args:
        file_path: 类型字典文件路径
        n_beads: bead数量 (2 for bond, 3 for angle, 4 for dihedral)

    Returns:
        dict: {type_id: (bead_type1, bead_type2, ...)}
    """
    type_dict = {}

    if not Path(file_path).exists():
        return type_dict

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) >= n_beads + 1:
                type_id = int(parts[0])
                bead_types = tuple(map(int, parts[1:n_beads + 1]))
                type_dict[type_id] = bead_types

    return type_dict


def write_topology_file(output_file: str, topology: np.ndarray,
                        topology_type: str, comment: str = ""):
    """
    写入拓扑文件。

    Args:
        output_file: 输出文件路径
        topology: 拓扑数组
        topology_type: 类型 ('bond', 'angle', 'dihedral')
        comment: 注释
    """
    if topology_type == 'bond':
        columns = ['bond_type', 'atom1_id', 'atom2_id']
    elif topology_type == 'angle':
        columns = ['angle_type', 'atom1_id', 'atom2_id', 'atom3_id']
    elif topology_type == 'dihedral':
        columns = ['dihedral_type', 'atom1_id', 'atom2_id', 'atom3_id', 'atom4_id']
    else:
        raise ValueError(f"未知拓扑类型: {topology_type}")

    df = pd.DataFrame(topology, columns=columns)
    df.to_csv(output_file, sep=' ', index=False, header=True)

    print(f"写入 {topology_type} 文件: {output_file} ({len(df)} {topology_type}s)")


def assign_topology_types(topology_array: np.ndarray,
                          bead_types: Dict[int, int],
                          topology_kind: str,
                          type_mapping: Optional[Dict[Tuple, int]] = None) -> np.ndarray:
    """
    根据 bead type 组合分配拓扑类型 ID。

    回环结构的 angle 和 dihedral 合并为同一类型：
    - Bond: (1,2) 和 (2,1) 合并
    - Angle: (1,1,2) 和 (2,1,1) 合并
    - Dihedral: (1,1,2,2) 和 (2,2,1,1) 合并

    Args:
        topology_array: 拓扑数组 [type, bead1, bead2, ...]
        bead_types: {bead_id: bead_type}
        topology_kind: 'bond', 'angle', 'dihedral'
        type_mapping: YAML 提供的组合→类型ID预填表（规范化后的组合为key），
                      未列出的组合自动分配ID

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
            # Angle: 首尾对称合并
            if bead_type_combo[0] > bead_type_combo[2]:
                bead_type_combo = (bead_type_combo[2], bead_type_combo[1], bead_type_combo[0])
        elif topology_kind == 'dihedral':
            # Dihedral: 首尾对称合并
            reversed_combo = (bead_type_combo[3], bead_type_combo[2], bead_type_combo[1], bead_type_combo[0])
            if bead_type_combo > reversed_combo:
                bead_type_combo = reversed_combo

        # 分配 type ID
        if bead_type_combo not in type_combinations:
            type_combinations[bead_type_combo] = next_type_id
            next_type_id += 1

        topo[0] = type_combinations[bead_type_combo]

    return topology_array


def derive_cg_bonds_from_aa(aa_bonds: np.ndarray,
                            aa_to_cg_mapping: Dict,
                            bead_types: Optional[Dict[int, int]] = None,
                            type_mapping: Optional[Dict[Tuple, int]] = None) -> np.ndarray:
    """
    从AA键推导CG键，并根据 bead type 分配 bond type。

    Args:
        aa_bonds: AA键数组 [type, atom1, atom2]
        aa_to_cg_mapping: AA到CG的映射字典
        bead_types: {bead_id: bead_type} 映射（可选，用于分配正确的 bond type）
        type_mapping: YAML 提供的 bond 组合→类型ID预填表

    Returns:
        CG键数组 [type, bead1, bead2]
    """
    # 构建AA原子到CG bead的映射
    aa_to_cg = {}
    for bead_id, bead_info in aa_to_cg_mapping.items():
        for aa_id in bead_info['aa_atoms']:
            aa_to_cg[int(aa_id)] = int(bead_id)

    # 推导CG键
    cg_bonds = []
    bond_type = 1

    for bond in aa_bonds:
        aa1, aa2 = int(bond[1]), int(bond[2])
        if aa1 in aa_to_cg and aa2 in aa_to_cg:
            bead1, bead2 = aa_to_cg[aa1], aa_to_cg[aa2]
            if bead1 != bead2:  # 排除同一bead内的键
                cg_bonds.append([bond_type, min(bead1, bead2), max(bead1, bead2)])

    # 去重
    cg_bonds = list(set(map(tuple, cg_bonds)))
    cg_bonds = np.array(sorted(cg_bonds), dtype=np.int32)

    # 分配 bond type
    if len(cg_bonds) > 0:
        if bead_types:
            # 根据 bead type 组合分配正确的 bond type（透传 type_mapping）
            cg_bonds = assign_topology_types(cg_bonds, bead_types, 'bond',
                                              type_mapping=type_mapping)
        else:
            # 兼容旧逻辑：所有 bond type 为 1
            cg_bonds[:, 0] = 1

    return cg_bonds


if __name__ == "__main__":
    print("拓扑文件工具模块")
    print("使用方法:")
    print("  from LmpPy.utils.topology import read_topology_files, read_type_dict")
    print("  topo = read_topology_files('cg_bonds.txt', 'cg_angles.txt', 'cg_dihedrals.txt')")