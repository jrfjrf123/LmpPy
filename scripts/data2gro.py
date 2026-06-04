#!/usr/bin/env python
"""
将 LAMMPS data 文件（atom_style molecular/full）转换为 GROMACS .gro 文件。

用法:
    python data2gro.py <input.data> <output.gro> [--residue-name RESNAME] [--unit angstrom]

说明:
    - 解析 LAMMPS data 文件的 Atoms 段和盒子信息
    - LAMMPS 坐标单位通常为 Å，GRO 文件坐标单位为 nm，脚本默认除以 10 转换
    - 使用 --unit angstrom 可显式指定输入单位（默认就是 angstrom）
    - 使用 --residue-name 指定残基名称（默认按 molecule-ID 命名为 MOLxx）
"""

import argparse
import re
import sys


def parse_lammps_data(filepath):
    """解析 LAMMPS data 文件，返回原子列表和盒子信息。"""
    with open(filepath, "r") as f:
        lines = f.readlines()

    # 跳过第一行标题
    header_lines = []
    atom_style = "molecular"  # 默认
    section_name = None
    in_section = False

    atoms = []
    box = {}
    masses = {}

    # 第一阶段：解析头信息和盒子
    i = 1  # 跳过标题行
    while i < len(lines):
        line = lines[i].strip()
        i += 1

        # 空行或注释跳过（在头信息阶段）
        if not line or line.startswith("#"):
            continue

        # 检测段落标题
        if re.match(r"^(Masses|Atoms|Bonds|Angles|Dihedrals|Impropers)", line, re.IGNORECASE):
            section_name = line.split()[0]
            in_section = True
            # 检查是否有 # style 后缀，如 "Atoms # full"
            if "# full" in line.lower():
                atom_style = "full"
            elif "# molecular" in line.lower():
                atom_style = "molecular"
            break

        # 数量头信息
        m = re.match(r"(\d+)\s+atoms", line)
        if m:
            continue
        m = re.match(r"(\d+)\s+atom\s+types", line)
        if m:
            continue

        # 盒子信息
        m = re.match(r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+xlo\s+xhi", line)
        if m:
            box["xlo"] = float(m.group(1))
            box["xhi"] = float(m.group(2))
            continue
        m = re.match(r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+ylo\s+yhi", line)
        if m:
            box["ylo"] = float(m.group(1))
            box["yhi"] = float(m.group(2))
            continue
        m = re.match(r"([-\d.eE+]+)\s+([-\d.eE+]+)\s+zlo\s+zhi", line)
        if m:
            box["zlo"] = float(m.group(1))
            box["zhi"] = float(m.group(2))
            continue
        # 倾斜因子（暂不支持，仅跳过）
        if re.match(r".*\bxy\s+xz\s+yz", line):
            continue

    # 第二阶段：解析 Masses 段
    if section_name and section_name.lower() == "masses":
        i += 1  # 跳过 Masses 标题后的空行
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith("#"):
                if not masses:
                    continue
                break
            parts = line.split("#")[0].split()
            if len(parts) >= 2:
                masses[int(parts[0])] = float(parts[1])

        # 找下一个段落
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            if re.match(r"^(Masses|Atoms|Bonds|Angles|Dihedrals|Impropers)", line, re.IGNORECASE):
                section_name = line.split()[0]
                if "# full" in line.lower():
                    atom_style = "full"
                elif "# molecular" in line.lower():
                    atom_style = "molecular"
                break
            break

    # 第三阶段：解析 Atoms 段
    if section_name and section_name.lower() == "atoms":
        i += 1  # 跳过 Atoms 标题后的空行
        while i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line or line.startswith("#"):
                if not atoms:
                    continue
                break
            parts = line.split("#")[0].split()
            if len(parts) < 6:
                continue

            atom_id = int(parts[0])
            mol_id = int(parts[1])
            atom_type = int(parts[2])

            if atom_style == "full":
                # full: atom-ID mol-ID atom-type q x y z [nx ny nz]
                q = float(parts[3])
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
            else:
                # molecular: atom-ID mol-ID atom-type x y z [nx ny nz]
                q = 0.0
                x = float(parts[3])
                y = float(parts[4])
                z = float(parts[5])

            mass = masses.get(atom_type, 0.0)
            atoms.append({
                "atom_id": atom_id,
                "mol_id": mol_id,
                "atom_type": atom_type,
                "q": q,
                "x": x,
                "y": y,
                "z": z,
                "mass": mass,
            })

    return atoms, box, atom_style


def write_gro(filepath, atoms, box, residue_name_template="MOL{:03d}"):
    """将原子数据写入 GROMACS .gro 文件。"""
    # 构建分子 ID 到残基名称的映射
    mol_ids = sorted(set(a["mol_id"] for a in atoms))
    mol_to_resname = {}
    for idx, mid in enumerate(mol_ids, 1):
        mol_to_resname[mid] = residue_name_template.format(idx)

    with open(filepath, "w") as f:
        # 标题行
        f.write("Converted from LAMMPS data file\n")
        # 原子数
        f.write(f"{len(atoms)}\n")
        # 每个原子一行
        for atom in atoms:
            resname = mol_to_resname[atom["mol_id"]][:5]
            # 根据原子类型推断原子名（简化：用类型编号）
            atom_name = f"T{atom['atom_type']}"
            # 坐标从 Å 转换为 nm
            x_nm = atom["x"] / 10.0
            y_nm = atom["y"] / 10.0
            z_nm = atom["z"] / 10.0

            # GRO 固定格式: %5d%-5s%5s%5d%8.3f%8.3f%8.3f
            f.write(
                f"{atom['mol_id']:5d}{resname:<5s}{atom_name:>5s}"
                f"{atom['atom_id']:5d}"
                f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}"
                f"\n"
            )
        # 盒子向量（nm）
        vx = (box.get("xhi", 0) - box.get("xlo", 0)) / 10.0
        vy = (box.get("yhi", 0) - box.get("ylo", 0)) / 10.0
        vz = (box.get("zhi", 0) - box.get("zlo", 0)) / 10.0
        f.write(f"{vx:8.3f}{vy:8.3f}{vz:8.3f}\n")


def main():
    parser = argparse.ArgumentParser(
        description="将 LAMMPS data 文件转换为 GROMACS .gro 文件"
    )
    parser.add_argument("input", help="输入 LAMMPS data 文件路径")
    parser.add_argument("output", help="输出 .gro 文件路径")
    parser.add_argument(
        "--residue-name",
        default=None,
        help="残基名称模板，例如 'WAT' 或 'MOL'（默认按 molecule-ID 自动生成）",
    )
    parser.add_argument(
        "--unit",
        choices=["angstrom", "nm"],
        default="angstrom",
        help="LAMMPS data 文件中的坐标单位（默认 angstrom）",
    )
    args = parser.parse_args()

    atoms, box, atom_style = parse_lammps_data(args.input)

    if not atoms:
        print("错误: 未找到原子数据，请检查文件格式和 atom_style", file=sys.stderr)
        sys.exit(1)

    print(f"解析到 {len(atoms)} 个原子，atom_style={atom_style}")
    print(f"盒子: x=[{box.get('xlo', 0):.3f}, {box.get('xhi', 0):.3f}] "
          f"y=[{box.get('ylo', 0):.3f}, {box.get('yhi', 0):.3f}] "
          f"z=[{box.get('zlo', 0):.3f}, {box.get('zhi', 0):.3f}]")

    unit_scale = 10.0 if args.unit == "angstrom" else 1.0

    if args.residue_name:
        resname = args.residue_name[:5]
        # 所有原子使用同一残基名
        mol_ids = sorted(set(a["mol_id"] for a in atoms))
        mol_to_resname = {mid: resname for mid in mol_ids}
    else:
        mol_ids = sorted(set(a["mol_id"] for a in atoms))
        mol_to_resname = {}
        for idx, mid in enumerate(mol_ids, 1):
            mol_to_resname[mid] = f"M{idx:04d}"

    with open(args.output, "w") as f:
        f.write("Converted from LAMMPS data file\n")
        f.write(f"{len(atoms)}\n")
        for atom in atoms:
            resname = mol_to_resname[atom["mol_id"]]
            atom_name = f"T{atom['atom_type']}"
            x_nm = atom["x"] / unit_scale
            y_nm = atom["y"] / unit_scale
            z_nm = atom["z"] / unit_scale
            f.write(
                f"{atom['mol_id']:5d}{resname:<5s}{atom_name:>5s}"
                f"{atom['atom_id']:5d}"
                f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}"
                f"\n"
            )
        vx = (box.get("xhi", 0) - box.get("xlo", 0)) / unit_scale
        vy = (box.get("yhi", 0) - box.get("ylo", 0)) / unit_scale
        vz = (box.get("zhi", 0) - box.get("zlo", 0)) / unit_scale
        f.write(f"{vx:8.3f}{vy:8.3f}{vz:8.3f}\n")

    print(f"已写入 {args.output}")


if __name__ == "__main__":
    main()
