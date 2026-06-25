#!/usr/bin/env python
"""
根据单分子拓扑文件和用户指定的分子数量，构建完整多分子体系的 LAMMPS .data 文件。

用法:
    # 提供 GRO 文件（坐标和 atom type 从 GRO 读取）
    python build_cg_system.py \
        --mol initiator:cg_6units_initiator_bonds.txt:1000 \
        --mol IP::100000 \
        --masses 1:12.01,2:12.01,3:12.01 \
        --gro system.gro \
        -o system.data

    # 不提供 GRO 文件（通过 --types 指定 bead 类型）
    python build_cg_system.py \
        --mol initiator:cg_6units_initiator_bonds.txt:1000 \
        --mol IP::100000 \
        --types initiator:2,1,1,1,1,1 \
        --types IP:3 \
        --masses 1:12.01,2:12.01,3:12.01 \
        -o system.data

说明:
    - --mol 格式: 名称:bonds文件路径:数量 (bonds 文件可为空表示单 bead 分子)
    - --types 格式: 名称:type1,type2,... (不提供 GRO 时必需)
    - 自动从 bonds 推导 angles 和 dihedrals
    - 输出 atom_style molecular 的 LAMMPS data 文件
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np

# 复用 cg_topology 中的拓扑推导逻辑
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from LmpPy.core.cg_topology import (
    derive_cg_topology_from_bonds,
    derive_angles_from_bonds,
    derive_dihedrals_from_bonds,
    CGTopology,
)

# YAML 配置解析模块
from LmpPy.scripts.build_cg_config import (
    BuildCGConfig,
    read_bonds as _read_bonds_new,
    read_angles,
    read_dihedrals,
)


def parse_gro(filepath):
    """
    解析 GROMACS .gro 文件，返回原子信息和盒子尺寸。
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    title = lines[0].strip()
    n_atoms = int(lines[1].strip())
    box_idx = len(lines) - 1
    while box_idx >= 0 and not lines[box_idx].strip():
        box_idx -= 1
    box_line = lines[box_idx].strip()
    box_parts = box_line.split()
    box = {
        "x": float(box_parts[0]),
        "y": float(box_parts[1]),
        "z": float(box_parts[2]) if len(box_parts) > 2 else 0.0,
    }

    atoms = []
    for i in range(2, 2 + n_atoms):
        line = lines[i]
        if len(line) < 44:
            continue
        res_id = int(line[0:5].strip())
        res_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        atom_id = int(line[15:20].strip())
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])

        # 从 atom name 提取类型编号，如 T1->1, T2->2
        m = re.search(r"(\d+)$", atom_name)
        atom_type = int(m.group(1)) if m else 1

        atoms.append({
            "res_id": res_id,
            "res_name": res_name,
            "atom_name": atom_name,
            "atom_id": atom_id,
            "x": x,
            "y": y,
            "z": z,
            "atom_type": atom_type,
        })

    return atoms, box, title


def read_bonds(filepath):
    """
    读取 bonds 文件。支持 3 列 (bond_type atom1 atom2) 或 4 列 (bond_id bond_type atom1 atom2)。
    若文件为空或不存在，返回空数组（单 bead 分子）。

    (委托给 build_cg_config.read_bonds)
    """
    return _read_bonds_new(filepath)


def parse_mol_spec(spec_str):
    """解析 --mol 参数: name:bonds_file:count"""
    parts = spec_str.split(":")
    if len(parts) != 3:
        print(f"错误: --mol 格式应为 'name:bonds_file:count'，得到 '{spec_str}'", file=sys.stderr)
        sys.exit(1)
    name = parts[0]
    bonds_file = parts[1] if parts[1] else None
    try:
        count = int(parts[2])
    except ValueError:
        print(f"错误: 分子数量应为整数，得到 '{parts[2]}'", file=sys.stderr)
        sys.exit(1)
    return name, bonds_file, count


def parse_types_spec(spec_str):
    """解析 --types 参数: name:type1,type2,..."""
    parts = spec_str.split(":")
    if len(parts) != 2:
        print(f"错误: --types 格式应为 'name:type1,type2,...'，得到 '{spec_str}'", file=sys.stderr)
        sys.exit(1)
    name = parts[0]
    try:
        types = [int(t.strip()) for t in parts[1].split(",")]
    except ValueError:
        print(f"错误: --types 类型应为整数，得到 '{parts[1]}'", file=sys.stderr)
        sys.exit(1)
    return name, types


def parse_masses(masses_str):
    """解析质量字符串 "1:12.01,2:12.01" -> {1: 12.01, 2: 12.01}"""
    masses = {}
    for part in masses_str.split(","):
        part = part.strip()
        if ":" in part:
            atype, mass = part.split(":", 1)
            masses[int(atype.strip())] = float(mass.strip())
        else:
            print(f"错误: 质量格式不正确 '{part}'，应为 'type:mass'", file=sys.stderr)
            sys.exit(1)
    return masses


def build_system(mol_specs, type_specs, masses_dict, gro_path, output_path,
                 mol_angles_list=None, mol_dihedrals_list=None):
    """
    构建完整体系并输出 LAMMPS .data 文件。

    Args:
        mol_specs: list of (name, bonds_file, count)
        type_specs: dict {name: [atom_types]} (从 --types 解析)
        masses_dict: dict {atom_type: mass}
        gro_path: GRO 文件路径（可选）
        output_path: 输出路径
    """
    # 1. 解析 GRO 文件（可选）
    gro_atoms = None
    gro_box = None
    if gro_path:
        gro_atoms, gro_box, gro_title = parse_gro(gro_path)
        print(f"GRO 文件: {gro_path}, 原子数: {len(gro_atoms)}")

    # 2. 读取每种分子的拓扑，计算总原子数
    mol_topologies = []
    total_atoms = 0

    for name, bonds_file, count in mol_specs:
        mol_bonds = read_bonds(bonds_file)
        # 推断每分子的 bead 数
        if len(mol_bonds) > 0:
            beads_per_mol = max(mol_bonds[:, 1].max(), mol_bonds[:, 2].max())
        else:
            beads_per_mol = 1  # 单 bead 分子

        n_atoms_type = beads_per_mol * count
        total_atoms += n_atoms_type
        mol_topologies.append({
            "name": name,
            "bonds": mol_bonds,
            "count": count,
            "beads_per_mol": beads_per_mol,
            "n_atoms": n_atoms_type,
        })
        print(f"分子 {name}: bonds={len(mol_bonds)}, beads/mol={beads_per_mol}, count={count}, total_atoms={n_atoms_type}")

    print(f"体系总原子数: {total_atoms}")

    # 3. 验证 GRO 原子数
    if gro_atoms is not None and len(gro_atoms) != total_atoms:
        print(f"警告: GRO 文件原子数 ({len(gro_atoms)}) 与体系总原子数 ({total_atoms}) 不一致", file=sys.stderr)
        print("  将仅使用 GRO 的前 {} 个原子".format(min(len(gro_atoms), total_atoms)), file=sys.stderr)

    # 4. 验证 --types 与 --mol 名称匹配
    for name, _, _ in mol_specs:
        if gro_path is None and name not in type_specs:
            print(f"错误: 分子 '{name}' 未指定 --types，不提供 GRO 时必需", file=sys.stderr)
            sys.exit(1)
        if name in type_specs:
            expected_beads = max(mol_topologies[i]["beads_per_mol"]
                                 for i, (n, _, _) in enumerate(mol_specs) if n == name)
            if len(type_specs[name]) != expected_beads:
                print(f"错误: --types {name} 类型数 ({len(type_specs[name])}) "
                      f"与 bead 数 ({expected_beads}) 不匹配", file=sys.stderr)
                sys.exit(1)

    # 5. 复制拓扑 N 次
    all_bonds = []
    all_angles = []
    all_dihedrals = []
    atom_rows = []
    gro_idx = 0
    global_mol_id = 1          # 跨分子类型的全局分子编号
    global_atom_offset = 0     # 跨分子类型的全局原子偏移量

    for mol_idx_global, mol_info in enumerate(mol_topologies):
        name = mol_info["name"]
        mol_bonds = mol_info["bonds"]
        count = mol_info["count"]
        beads_per_mol = mol_info["beads_per_mol"]
        atom_types = type_specs.get(name, None)

        # 获取预读取的 angles / dihedrals（YAML 模式传入）
        mol_angles = (mol_angles_list[mol_idx_global]
                      if mol_angles_list and mol_idx_global < len(mol_angles_list)
                      else np.array([], dtype=np.int32).reshape(0, 4))
        mol_dihedrals = (mol_dihedrals_list[mol_idx_global]
                         if mol_dihedrals_list and mol_idx_global < len(mol_dihedrals_list)
                         else np.array([], dtype=np.int32).reshape(0, 5))

        # 逐分子兜底推导：未显式指定的从该分子 bonds 自动推导
        if len(mol_angles) == 0 and len(mol_bonds) > 0:
            mol_angles = derive_angles_from_bonds(mol_bonds)
        if len(mol_dihedrals) == 0 and len(mol_bonds) > 0:
            mol_dihedrals = derive_dihedrals_from_bonds(mol_bonds)

        for mol_idx in range(count):
            mol_id = global_mol_id
            atom_offset = global_atom_offset + mol_idx * beads_per_mol
            global_mol_id += 1

            for local_idx in range(beads_per_mol):
                global_atom_id = gro_idx + 1

                if gro_atoms and gro_idx < len(gro_atoms):
                    gro_atom = gro_atoms[gro_idx]
                    x = gro_atom["x"] * 10.0
                    y = gro_atom["y"] * 10.0
                    z = gro_atom["z"] * 10.0
                    atom_type = gro_atom["atom_type"]
                elif atom_types:
                    x, y, z = 0.0, 0.0, 0.0
                    atom_type = atom_types[local_idx]
                else:
                    x, y, z = 0.0, 0.0, 0.0
                    atom_type = 1

                atom_rows.append([global_atom_id, mol_id, atom_type, x, y, z])
                gro_idx += 1

            # 复制 bonds
            for bond in mol_bonds:
                all_bonds.append([bond[0], bond[1] + atom_offset, bond[2] + atom_offset])

            # 复制 angles
            for angle in mol_angles:
                all_angles.append([angle[0], angle[1] + atom_offset,
                                   angle[2] + atom_offset, angle[3] + atom_offset])

            # 复制 dihedrals
            for dihedral in mol_dihedrals:
                all_dihedrals.append([dihedral[0], dihedral[1] + atom_offset,
                                      dihedral[2] + atom_offset, dihedral[3] + atom_offset,
                                      dihedral[4] + atom_offset])

        # 当前分子类型处理完毕，累加全局原子偏移量
        global_atom_offset += count * beads_per_mol

    if all_bonds:
        all_bonds = np.array(all_bonds, dtype=np.int32)
    else:
        all_bonds = np.array([], dtype=np.int32).reshape(0, 3)

    if all_angles:
        all_angles = np.array(all_angles, dtype=np.int32)
    else:
        all_angles = np.array([], dtype=np.int32).reshape(0, 4)

    if all_dihedrals:
        all_dihedrals = np.array(all_dihedrals, dtype=np.int32)
    else:
        all_dihedrals = np.array([], dtype=np.int32).reshape(0, 5)

    # 6. 组装拓扑（angles/dihedrals 已在逐分子循环中推导完成）
    topology = CGTopology(bonds=all_bonds, angles=all_angles, dihedrals=all_dihedrals)

    print(f"总 bonds: {topology.n_bonds}, angles: {topology.n_angles}, dihedrals: {topology.n_dihedrals}")

    # 7. 写 data 文件
    n_atom_types = len(masses_dict)
    bond_types = len(set(b[0] for b in all_bonds)) if topology.n_bonds > 0 else 0
    angle_types = len(set(a[0] for a in topology.angles)) if topology.n_angles > 0 else 0
    dihedral_types = len(set(d[0] for d in topology.dihedrals)) if topology.n_dihedrals > 0 else 0

    with open(output_path, "w") as f:
        # 头信息
        f.write("LAMMPS data file - Coarse-grained system\n")
        f.write("\n")
        f.write(f"{len(atom_rows)} atoms\n")
        f.write(f"{topology.n_bonds} bonds\n")
        f.write(f"{topology.n_angles} angles\n")
        f.write(f"{topology.n_dihedrals} dihedrals\n")
        f.write("\n")
        f.write(f"{n_atom_types} atom types\n")
        if bond_types > 0:
            f.write(f"{bond_types} bond types\n")
        if angle_types > 0:
            f.write(f"{angle_types} angle types\n")
        if dihedral_types > 0:
            f.write(f"{dihedral_types} dihedral types\n")
        f.write("\n")

        # 盒子信息 (Å)
        if gro_box:
            box_x = gro_box["x"] * 10.0
            box_y = gro_box["y"] * 10.0
            box_z = gro_box["z"] * 10.0
        else:
            box_x = box_y = box_z = 100.0
        f.write(f"0.0 {box_x:.6f} xlo xhi\n")
        f.write(f"0.0 {box_y:.6f} ylo yhi\n")
        f.write(f"0.0 {box_z:.6f} zlo zhi\n")

        # Masses
        f.write("\nMasses\n\n")
        for atype in sorted(masses_dict.keys()):
            f.write(f"{atype} {masses_dict[atype]}\n")

        # Atoms
        f.write("\nAtoms # molecular\n\n")
        for row in atom_rows:
            f.write(f"{int(row[0]):7d}{int(row[1]):7d}{int(row[2]):7d}"
                    f"{row[3]:12.6f}{row[4]:12.6f}{row[5]:12.6f}\n")

        # Bonds
        if topology.n_bonds > 0:
            f.write("\nBonds\n\n")
            for i, bond in enumerate(all_bonds, 1):
                f.write(f"{i:7d}{bond[0]:7d}{bond[1]:7d}{bond[2]:7d}\n")

        # Angles
        if topology.n_angles > 0:
            f.write("\nAngles\n\n")
            for i, angle in enumerate(topology.angles, 1):
                f.write(f"{i:7d}{angle[0]:7d}{angle[1]:7d}{angle[2]:7d}{angle[3]:7d}\n")

        # Dihedrals
        if topology.n_dihedrals > 0:
            f.write("\nDihedrals\n\n")
            for i, dihedral in enumerate(topology.dihedrals, 1):
                f.write(f"{i:7d}{dihedral[0]:7d}{dihedral[1]:7d}{dihedral[2]:7d}"
                        f"{dihedral[3]:7d}{dihedral[4]:7d}\n")

    print(f"已写入 {output_path}")


def build_system_from_config(config: BuildCGConfig, output_path: str):
    """
    从 BuildCGConfig 对象构建体系（YAML 模式入口）。

    Args:
        config: BuildCGConfig 实例（已通过 validate() 校验）
        output_path: 输出 .data 文件路径
    """
    # 读取每种分子的拓扑文件
    mol_specs = []           # list of (name, bonds_file, count)，兼容 build_system
    mol_angles_list = []     # 每个分子预读取的 angles
    mol_dihedrals_list = []  # 每个分子预读取的 dihedrals

    for mol in config.molecules:
        # 读取 bonds
        mol_bonds = read_bonds(mol.bonds_file)

        # 使用 dataclass 中已推断的 beads_per_mol
        beads_per_mol = mol.beads_per_mol
        n_atoms_type = beads_per_mol * mol.count

        mol_specs.append((mol.name, mol.bonds_file, mol.count))

        # 读取 angles（指定文件则读取，未指定则空数组 → 后续自动推导）
        mol_angles = read_angles(mol.angles_file)
        mol_angles_list.append(mol_angles)

        # 读取 dihedrals（指定文件则读取，未指定则空数组 → 后续自动推导）
        mol_dihedrals = read_dihedrals(mol.dihedrals_file)
        mol_dihedrals_list.append(mol_dihedrals)

        print(f"分子 {mol.name}: bonds={len(mol_bonds)}, "
              f"angles={len(mol_angles)}{' (指定文件)' if mol.angles_file else ' (自动推导)'}, "
              f"dihedrals={len(mol_dihedrals)}{' (指定文件)' if mol.dihedrals_file else ' (自动推导)'}, "
              f"beads/mol={beads_per_mol}, count={mol.count}, total_atoms={n_atoms_type}")

    # 调用 build_system
    build_system(
        mol_specs=mol_specs,
        type_specs=config.types,
        masses_dict=config.masses,
        gro_path=config.gro,
        output_path=output_path,
        mol_angles_list=mol_angles_list,
        mol_dihedrals_list=mol_dihedrals_list,
    )


def main():
    parser = argparse.ArgumentParser(
        description="根据单分子拓扑和分子数量构建完整多分子体系的 LAMMPS .data 文件"
    )

    # 两种模式（互斥）
    parser.add_argument(
        "--yaml", default=None,
        help="YAML 配置文件路径（与 --mol 互斥，包含所有构建参数）"
    )
    parser.add_argument(
        "--mol", action="append", default=None,
        help="分子规格: name:bonds_file:count (与 --yaml 互斥)"
    )

    # 命令行模式的辅助参数
    parser.add_argument(
        "--types", action="append", default=[],
        help="分子 bead 类型: name:type1,type2,... (不提供 GRO 时必需)"
    )
    parser.add_argument("--masses", default=None, help="原子类型质量，如 '1:12.01,2:12.01,3:12.01'")
    parser.add_argument("--gro", default=None, help="GRO 文件路径（提供坐标和类型，可选）")
    parser.add_argument("-o", "--output", required=True, help="输出 LAMMPS .data 文件路径")
    args = parser.parse_args()

    # --- 互斥检查 ---
    if args.yaml and args.mol:
        parser.error("--yaml 与 --mol 不能同时使用，请选择一种模式")

    # --- YAML 模式 ---
    if args.yaml:
        config = BuildCGConfig.from_yaml(args.yaml)
        output = args.output  # 命令行 -o 优先于 YAML 中的 output
        build_system_from_config(config, output)
        return

    # --- 命令行模式（原有逻辑，完全不变）---
    if not args.mol:
        parser.error("必须提供 --mol 或 --yaml 参数")

    mol_specs = [parse_mol_spec(s) for s in args.mol]
    type_specs = {}
    for t in args.types:
        name, types = parse_types_spec(t)
        type_specs[name] = types
    masses = parse_masses(args.masses)
    build_system(mol_specs, type_specs, masses, args.gro, args.output)


if __name__ == "__main__":
    main()
