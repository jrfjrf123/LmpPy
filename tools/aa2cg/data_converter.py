"""
LAMMPS Data文件转换器 - AA data到CG data的转换

功能:
- 读取LAMMPS data文件（全原子）
- 使用CG映射转换为粗粒化格式
- 写入CG LAMMPS data文件

作者: 整合自 md_base_on_ml/AA_data2CG_data/DGEBA_PPB_sys/AA_data2CG_data.py
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple

# 尝试导入MDAnalysis（可选依赖）
try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from .mapping_utils import (
    load_aa_to_cg_mapping,
    convert_aa_to_cg_frame,
    wrap_coords
)


def read_lammps_data(data_file: str) -> Dict:
    """
    读取LAMMPS data文件。

    Args:
        data_file: data文件路径

    Returns:
        dict with keys:
            - 'ids': 原子ID数组
            - 'mol_ids': 分子ID数组
            - 'types': 原子类型数组
            - 'coords': 坐标数组 (natoms, 3)
            - 'bonds': 键数组 (nbonds, 3) [type, atom1, atom2]
            - 'angles': 角度数组
            - 'dihedrals': 二面角数组
            - 'mass_list': 质量字典 {type: mass}
            - 'box': 盒子尺寸 (3, 2)
    """
    if not HAS_MDA:
        raise ImportError("需要安装MDAnalysis: pip install MDAnalysis")

    # 尝试两种 atom_style: 7列(含charge) 和 6列(无charge)
    for atom_style in ['id resid type charge x y z', 'id resid type x y z']:
        try:
            u = mda.Universe(data_file, format='DATA', atom_style=atom_style)
            break
        except Exception:
            continue
    else:
        raise ValueError(f"无法解析LAMMPS data文件: {data_file}，支持 'id resid type charge x y z' 或 'id resid type x y z' 格式")

    # 提取原子信息
    atoms = u.atoms
    ids = atoms.ids.copy()
    mol_ids = atoms.resids.copy()
    types = atoms.types.astype(int).copy()
    coords = atoms.positions.copy()

    # 盒子尺寸
    box = u.dimensions[0:3].copy()
    box = np.array([[0.0, box[0]], [0.0, box[1]], [0.0, box[2]]], dtype=np.float64)

    # 质量信息
    unique_types = np.unique(types)
    mass_dict = {}
    for atype in unique_types:
        mask = types == atype
        mass_dict[int(atype)] = float(atoms.masses[mask][0])

    # 键信息
    if len(u.bonds) > 0:
        bonds_list = []
        for bond in u.bonds:
            atom1_id = bond.atoms[0].id
            atom2_id = bond.atoms[1].id
            bonds_list.append([1, min(atom1_id, atom2_id), max(atom1_id, atom2_id)])
        bonds = np.array(bonds_list, dtype=np.int32)
    else:
        bonds = np.array([], dtype=np.int32).reshape(0, 3)

    # 角度信息
    if len(u.angles) > 0:
        angles_list = []
        for angle in u.angles:
            atom1_id = angle.atoms[0].id
            atom2_id = angle.atoms[1].id
            atom3_id = angle.atoms[2].id
            angles_list.append([1, atom1_id, atom2_id, atom3_id])
        angles = np.array(angles_list, dtype=np.int32)
    else:
        angles = np.array([], dtype=np.int32).reshape(0, 4)

    # 二面角信息
    if len(u.dihedrals) > 0:
        dihedrals_list = []
        for dihedral in u.dihedrals:
            atom1_id = dihedral.atoms[0].id
            atom2_id = dihedral.atoms[1].id
            atom3_id = dihedral.atoms[2].id
            atom4_id = dihedral.atoms[3].id
            dihedrals_list.append([1, atom1_id, atom2_id, atom3_id, atom4_id])
        dihedrals = np.array(dihedrals_list, dtype=np.int32)
    else:
        dihedrals = np.array([], dtype=np.int32).reshape(0, 5)

    return {
        'ids': ids,
        'mol_ids': mol_ids,
        'types': types,
        'coords': coords,
        'bonds': bonds,
        'angles': angles,
        'dihedrals': dihedrals,
        'mass_list': mass_dict,
        'box': box
    }


def write_cg_data_file(filename: str, cg_data: Dict, cg_bonds: np.ndarray = None,
                       cg_angles: np.ndarray = None, cg_dihedrals: np.ndarray = None,
                       mass_list: Dict = None):
    """
    写入粗粒化LAMMPS data文件。

    Args:
        filename: 输出文件名
        cg_data: CG数据字典，包含 ids, mol_ids, types, coords, box
        cg_bonds: CG键数组 (nbonds, 3) [type, bead1, bead2]
        cg_angles: CG角度数组
        cg_dihedrals: CG二面角数组
        mass_list: 质量字典 {type: mass}
    """
    ids = cg_data['ids'].astype(int)
    mol_ids = cg_data['mol_ids'].astype(int)
    types = cg_data['types'].astype(int)
    coords = cg_data['coords']
    box = cg_data['box']

    n_beads = len(ids)
    n_bonds = len(cg_bonds) if cg_bonds is not None else 0
    n_angles = len(cg_angles) if cg_angles is not None else 0
    n_dihedrals = len(cg_dihedrals) if cg_dihedrals is not None else 0

    # 获取唯一类型
    unique_types = np.unique(types)
    n_types = len(unique_types)

    n_bond_types = len(np.unique(cg_bonds[:, 0])) if n_bonds > 0 else 0
    n_angle_types = len(np.unique(cg_angles[:, 0])) if n_angles > 0 else 0
    n_dihedral_types = len(np.unique(cg_dihedrals[:, 0])) if n_dihedrals > 0 else 0

    with open(filename, 'w') as f:
        # 头部
        f.write("LAMMPS data file - Coarse-grained system\n\n")

        # 计数
        f.write(f"{n_beads} atoms\n")
        f.write(f"{n_bonds} bonds\n")
        f.write(f"{n_angles} angles\n")
        f.write(f"{n_dihedrals} dihedrals\n")
        f.write(f"0 impropers\n\n")

        # 类型数
        f.write(f"{n_types} atom types\n")
        if n_bond_types > 0:
            f.write(f"{n_bond_types} bond types\n")
        if n_angle_types > 0:
            f.write(f"{n_angle_types} angle types\n")
        if n_dihedral_types > 0:
            f.write(f"{n_dihedral_types} dihedral types\n")
        f.write("\n")

        # 盒子边界
        f.write(f"{box[0, 0]:.6f} {box[0, 1]:.6f} xlo xhi\n")
        f.write(f"{box[1, 0]:.6f} {box[1, 1]:.6f} ylo yhi\n")
        f.write(f"{box[2, 0]:.6f} {box[2, 1]:.6f} zlo zhi\n\n")

        # 质量
        if mass_list:
            f.write("Masses\n\n")
            for bead_type in sorted(unique_types):
                mass = mass_list.get(bead_type, 1.0)
                f.write(f"{bead_type} {mass:.6f}\n")
            f.write("\n")

        # 原子部分
        f.write("Atoms # molecular\n\n")
        for i in range(n_beads):
            f.write(f"{ids[i]} {mol_ids[i]} {types[i]} {coords[i, 0]:.6f} {coords[i, 1]:.6f} {coords[i, 2]:.6f}\n")
        f.write("\n")

        # 键部分
        if n_bonds > 0:
            f.write("Bonds\n\n")
            for i, bond in enumerate(cg_bonds):
                f.write(f"{i+1} {bond[0]} {bond[1]} {bond[2]}\n")
            f.write("\n")

        # 角度部分
        if n_angles > 0:
            f.write("Angles\n\n")
            for i, angle in enumerate(cg_angles):
                f.write(f"{i+1} {angle[0]} {angle[1]} {angle[2]} {angle[3]}\n")
            f.write("\n")

        # 二面角部分
        if n_dihedrals > 0:
            f.write("Dihedrals\n\n")
            for i, dihedral in enumerate(cg_dihedrals):
                f.write(f"{i+1} {dihedral[0]} {dihedral[1]} {dihedral[2]} {dihedral[3]} {dihedral[4]}\n")
            f.write("\n")

    print(f"CG data文件已写入: {filename}")
    print(f"  Beads: {n_beads}")
    print(f"  Bonds: {n_bonds}")
    print(f"  Angles: {n_angles}")
    print(f"  Dihedrals: {n_dihedrals}")


def derive_cg_topology(aa_data: Dict, mapping_csv: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 AA data + mapping CSV 自动推导 CG bonds/angles/dihedrals。

    内部复用:
      - LmpPy.utils.topology.derive_cg_bonds_from_aa() 做键映射
      - LmpPy.core.cg_topology.derive_cg_topology_from_bonds() 做 angle/dihedral 推导

    Args:
        aa_data: 从read_lammps_data()获取的AA数据，必须包含'bonds'键
        mapping_csv: CG映射CSV文件路径 (bead_id, mol_id, bead_type, AA_id, mass)

    Returns:
        (cg_bonds, cg_angles, cg_dihedrals):
            - cg_bonds: np.ndarray (nbonds, 3) [type, bead1, bead2]
            - cg_angles: np.ndarray (nangles, 4) [type, bead1, center, bead3]
            - cg_dihedrals: np.ndarray (ndihedrals, 5) [type, b1, b2, b3, b4]
    """
    from LmpPy.utils.topology import derive_cg_bonds_from_aa
    from LmpPy.core.cg_topology import derive_cg_topology_from_bonds

    # 加载映射
    mapping_dict = load_aa_to_cg_mapping(mapping_csv)

    # 构建 bead_types 映射 {bead_id: bead_type}
    bead_types = {bead_id: info['bead_type'] for bead_id, info in mapping_dict.items()}

    # 从 AA bonds 推导 CG bonds（传入 bead_types）
    cg_bonds = derive_cg_bonds_from_aa(aa_data['bonds'], mapping_dict, bead_types)

    # 从 CG bonds 推导 angles 和 dihedrals（传入 bead_types）
    if len(cg_bonds) > 0:
        topology = derive_cg_topology_from_bonds(cg_bonds, bead_types=bead_types)
        return cg_bonds, topology.angles, topology.dihedrals
    else:
        return (
            cg_bonds,
            np.array([], dtype=np.int32).reshape(0, 4),
            np.array([], dtype=np.int32).reshape(0, 5),
        )


def export_cg_topology(cg_bonds: np.ndarray, cg_angles: np.ndarray,
                       cg_dihedrals: np.ndarray, output_dir: str = ".",
                       prefix: str = "cg"):
    """
    将推导的 CG 拓扑保存到文本文件。

    创建文件: {prefix}_bonds.txt, {prefix}_angles.txt, {prefix}_dihedrals.txt

    Args:
        cg_bonds: CG bonds 数组
        cg_angles: CG angles 数组
        cg_dihedrals: CG dihedrals 数组
        output_dir: 输出目录
        prefix: 文件名前缀
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if len(cg_bonds) > 0:
        np.savetxt(out / f"{prefix}_bonds.txt", cg_bonds, fmt='%d')
    if len(cg_angles) > 0:
        np.savetxt(out / f"{prefix}_angles.txt", cg_angles, fmt='%d')
    if len(cg_dihedrals) > 0:
        np.savetxt(out / f"{prefix}_dihedrals.txt", cg_dihedrals, fmt='%d')

    print(f"CG topology 已导出到: {out}")
    if len(cg_bonds) > 0:
        print(f"  Bonds: {len(cg_bonds)} -> {prefix}_bonds.txt")
    if len(cg_angles) > 0:
        print(f"  Angles: {len(cg_angles)} -> {prefix}_angles.txt")
    if len(cg_dihedrals) > 0:
        print(f"  Dihedrals: {len(cg_dihedrals)} -> {prefix}_dihedrals.txt")


def export_cg_bead_info(cg_data: Dict, mapping_dict: Dict,
                        output_dir: str = ".", filename: str = "cg_bead_info.txt"):
    """
    导出 CG bead 信息文件。

    格式: bead_id mol_id bead_type mass

    Args:
        cg_data: CG数据字典，包含 ids, mol_ids, types
        mapping_dict: 映射字典，包含 aa_masses
        output_dir: 输出目录
        filename: 输出文件名
    """
    out_path = Path(output_dir) / filename

    with open(out_path, 'w') as f:
        f.write("# bead_id mol_id bead_type mass\n")
        for i, bead_id in enumerate(cg_data['ids']):
            mol_id = cg_data['mol_ids'][i]
            bead_type = cg_data['types'][i]
            # 计算bead总质量（AA原子质量之和）
            bead_mass = np.sum(mapping_dict[bead_id]['aa_masses'])
            f.write(f"{bead_id} {mol_id} {bead_type} {bead_mass:.6f}\n")

    print(f"CG bead info 已导出: {out_path} ({len(cg_data['ids'])} beads)")


def convert_data_to_cg(aa_data: Dict, mapping_csv: str,
                       cg_bonds_file: str = None,
                       cg_angles_file: str = None,
                       cg_dihedrals_file: str = None,
                       derive_topology: bool = False,
                       output_cg_topology_dir: str = None,
                       export_bead_info: bool = True) -> Tuple[Dict, Dict]:
    """
    将AA data转换为CG data。

    Args:
        aa_data: 从read_lammps_data()获取的AA数据
        mapping_csv: CG映射CSV文件路径
        cg_bonds_file: CG键文件路径（可选，优先使用）
        cg_angles_file: CG角度文件路径（可选）
        cg_dihedrals_file: CG二面角文件路径（可选）
        derive_topology: 是否从AA data + mapping自动推导CG拓扑
        output_cg_topology_dir: 导出推导的CG拓扑到文件（仅在derive_topology=True时有效）
        export_bead_info: 是否导出cg_bead_info.txt（当output_cg_topology_dir存在时自动导出）

    Returns:
        (cg_data, mapping_dict): CG数据字典和映射字典
    """
    # 加载映射
    mapping_dict = load_aa_to_cg_mapping(mapping_csv)

    # 转换坐标
    cg_data = convert_aa_to_cg_frame(
        aa_coords=aa_data['coords'],
        aa_ids=aa_data['ids'],
        mapping_dict=mapping_dict,
        box=aa_data['box']
    )

    # 加载或推导 CG 拓扑
    cg_bonds = None
    cg_angles = None
    cg_dihedrals = None

    if cg_bonds_file and Path(cg_bonds_file).exists():
        # 优先级1: 显式文件路径
        print(f"从文件加载 CG topology: {cg_bonds_file}")
        cg_bonds = np.loadtxt(cg_bonds_file, dtype=int)
        if cg_bonds.ndim == 1:
            cg_bonds = cg_bonds.reshape(1, -1)

        if cg_angles_file and Path(cg_angles_file).exists():
            cg_angles = np.loadtxt(cg_angles_file, dtype=int)
            if cg_angles.ndim == 1:
                cg_angles = cg_angles.reshape(1, -1)

        if cg_dihedrals_file and Path(cg_dihedrals_file).exists():
            cg_dihedrals = np.loadtxt(cg_dihedrals_file, dtype=int)
            if cg_dihedrals.ndim == 1:
                cg_dihedrals = cg_dihedrals.reshape(1, -1)
    elif derive_topology:
        # 优先级2: 自动推导
        print("从 AA bonds + mapping 推导 CG topology...")
        cg_bonds, cg_angles, cg_dihedrals = derive_cg_topology(aa_data, mapping_csv)
        print(f"  CG bonds: {len(cg_bonds)}, angles: {len(cg_angles)}, dihedrals: {len(cg_dihedrals)}")

        # 可选: 导出推导的拓扑
        if output_cg_topology_dir:
            export_cg_topology(cg_bonds, cg_angles, cg_dihedrals,
                               output_dir=output_cg_topology_dir)
            # 自动导出 cg_bead_info.txt
            if export_bead_info:
                export_cg_bead_info(cg_data, mapping_dict,
                                    output_dir=output_cg_topology_dir)

    cg_data['bonds'] = cg_bonds
    cg_data['angles'] = cg_angles
    cg_data['dihedrals'] = cg_dihedrals

    return cg_data, mapping_dict


def unwrap_coords(ids: np.ndarray, types: np.ndarray, coords: np.ndarray,
                  box: np.ndarray, bonds: np.ndarray) -> np.ndarray:
    """
    解缠分子坐标（处理PBC边界跨越）。

    Args:
        ids: 原子ID数组
        types: 原子类型数组
        coords: 坐标数组
        box: 盒子尺寸 (3, 2)
        bonds: 键数组 [type, atom1, atom2]

    Returns:
        解缠后的坐标数组
    """
    from collections import defaultdict, deque

    box_length = box[:, 1] - box[:, 0]
    unwrapped = coords.copy()

    # 构建邻接表
    adjacency = defaultdict(list)
    id_to_idx = {id_: i for i, id_ in enumerate(ids)}

    for bond in bonds:
        atom1_id = bond[1]
        atom2_id = bond[2]
        if atom1_id in id_to_idx and atom2_id in id_to_idx:
            idx1 = id_to_idx[atom1_id]
            idx2 = id_to_idx[atom2_id]
            adjacency[idx1].append(idx2)
            adjacency[idx2].append(idx1)

    # BFS解缠
    visited = set()
    for start_idx in range(len(ids)):
        if start_idx in visited:
            continue

        # 对每个连通分量进行解缠
        queue = deque([start_idx])
        visited.add(start_idx)

        while queue:
            current_idx = queue.popleft()

            for neighbor_idx in adjacency[current_idx]:
                if neighbor_idx not in visited:
                    visited.add(neighbor_idx)
                    queue.append(neighbor_idx)

                    # 解缠相邻原子
                    delta = unwrapped[neighbor_idx] - unwrapped[current_idx]
                    for dim in range(3):
                        if delta[dim] > box_length[dim] / 2:
                            unwrapped[neighbor_idx, dim] -= box_length[dim]
                        elif delta[dim] < -box_length[dim] / 2:
                            unwrapped[neighbor_idx, dim] += box_length[dim]

    return unwrapped


def unwrap_by_molecule(coords: np.ndarray, mol_ids: np.ndarray,
                       ids: np.ndarray, box: np.ndarray) -> np.ndarray:
    """
    基于 mol_id 对分子坐标进行 unwrap（二次校正）。

    用于处理原始 GRO 文件中分子已被 PBC wrap 分割的情况。
    此函数在 unwrap_coords() 之后调用，确保同一分子内所有原子连续。

    算法：
    1. 按 mol_id 分组原子
    2. 对每个 molecule：
       a. 使用第一个原子作为参考点
       b. 将所有原子展开到参考点附近（最小图像约定）
       c. 计算分子质心
       d. 以质心为基准重新展开所有原子

    Args:
        coords: 坐标数组 (natoms, 3)
        mol_ids: 分子ID数组 (natoms,)
        ids: 原子ID数组 (natoms,)
        box: 盒子尺寸 (3, 2)

    Returns:
        解缠后的坐标数组 (natoms, 3)
    """
    box_length = box[:, 1] - box[:, 0]
    unwrapped = coords.copy()

    # 按 mol_id 分组
    unique_mols = np.unique(mol_ids)

    for mol_id in unique_mols:
        # 获取该分子的所有原子索引
        mol_mask = mol_ids == mol_id
        mol_indices = np.where(mol_mask)[0]

        if len(mol_indices) < 2:
            # 单原子分子，无需处理
            continue

        mol_coords = coords[mol_indices]

        # 步骤1：使用第一个原子作为参考点
        ref = mol_coords[0]
        delta = mol_coords - ref

        # 步骤2：应用最小图像约定，将所有原子展开到参考点附近
        for dim in range(3):
            shift = np.round(delta[:, dim] / box_length[dim])
            delta[:, dim] -= shift * box_length[dim]

        unfolded = ref + delta

        # 步骤3：计算分子质心
        com = np.mean(unfolded, axis=0)

        # 步骤4：以质心为基准重新展开所有原子
        delta2 = coords[mol_indices] - com
        for dim in range(3):
            shift2 = np.round(delta2[:, dim] / box_length[dim])
            delta2[:, dim] -= shift2 * box_length[dim]

        unwrapped[mol_indices] = com + delta2

    return unwrapped


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print("用法: python data_converter.py <data_file> <mapping_csv> [cg_bonds_file] [cg_angles_file] [cg_dihedrals_file]")
        sys.exit(1)

    data_file = sys.argv[1]
    mapping_csv = sys.argv[2]
    cg_bonds_file = sys.argv[3] if len(sys.argv) > 3 else None
    cg_angles_file = sys.argv[4] if len(sys.argv) > 4 else None
    cg_dihedrals_file = sys.argv[5] if len(sys.argv) > 5 else None

    # 读取AA data
    print(f"读取AA data文件: {data_file}")
    aa_data = read_lammps_data(data_file)
    print(f"  原子数: {len(aa_data['ids'])}")
    print(f"  键数: {len(aa_data['bonds'])}")

    # 解缠坐标
    print("解缠分子坐标...")
    unwrapped_coords = unwrap_coords(
        aa_data['ids'], aa_data['types'], aa_data['coords'],
        aa_data['box'], aa_data['bonds']
    )
    aa_data['coords'] = unwrapped_coords

    # 转换为CG
    print(f"转换为CG...")
    cg_data, mapping = convert_data_to_cg(
        aa_data, mapping_csv,
        cg_bonds_file, cg_angles_file, cg_dihedrals_file
    )

    # 写入CG data
    output_file = Path(data_file).stem + "_cg.data"
    write_cg_data_file(
        output_file, cg_data,
        cg_data.get('bonds'), cg_data.get('angles'), cg_data.get('dihedrals'),
        mass_list=aa_data['mass_list']
    )

    print(f"\n✓ CG data文件创建成功: {output_file}")