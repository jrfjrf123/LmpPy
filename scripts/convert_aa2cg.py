#!/usr/bin/env python
"""
AA到CG转换工具 CLI

用法:
    # 转换轨迹
    python -m LmpPy.scripts.convert_aa2cg trj --tpr topol.tpr --trr traj.trr --mapping mapping.csv -o cg_traj.pkl

    # 转换data文件（使用预计算的拓扑文件）
    python -m LmpPy.scripts.convert_aa2cg data --input system.data --mapping mapping.csv --bonds cg_bonds.txt

    # 转换data文件（自动推导拓扑）
    python -m LmpPy.scripts.convert_aa2cg data --input system.data --mapping mapping.csv --derive-topology --output-cg-topology .

    # 转换data文件（自动推导拓扑 + YAML类型映射）
    python -m LmpPy.scripts.convert_aa2cg data --input system.data --mapping mapping.csv \\
        --derive-topology --type-mapping type_mapping.yaml

    # 转换data文件并输出XYZ（unwrap格式）
    python -m LmpPy.scripts.convert_aa2cg data --input system.data --mapping mapping.csv --xyz

作者: Claude
日期: 2026-04-02
"""

import sys
import numpy as np
from pathlib import Path
from typing import Dict

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.aa2cg import (
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
    load_cg_trajectory_pickle,
    write_trajectory_to_xyz,
    read_lammps_data,
    convert_data_to_cg,
    write_cg_data_file,
    unwrap_coords,
    derive_cg_topology,
    export_cg_topology,
)

# 导入 GRO 文件读取（项目根目录）
try:
    import sys
    _project_root = Path(__file__).resolve().parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from gmx_file_io import get_gro_pandas_data
    HAS_GMX_IO = True
except ImportError:
    HAS_GMX_IO = False

# 导入 MDAnalysis（用于 TPR 拓扑）
try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False


def build_cg_mass_dict(cg_data: Dict, mapping_dict: Dict) -> Dict[int, float]:
    """
    从 mapping_dict 构建 CG bead 质量字典 {bead_type: mass}.

    Args:
        cg_data: CG数据字典，包含 ids, types
        mapping_dict: 映射字典，包含 aa_masses

    Returns:
        {bead_type: mass} 字典，每个 bead_type 的质量为该类型 bead 的平均质量
    """
    mass_dict = {}
    type_masses = {}  # {bead_type: [list of masses]}

    for i, bead_id in enumerate(cg_data['ids']):
        bead_type = int(cg_data['types'][i])
        bead_mass = np.sum(mapping_dict[bead_id]['aa_masses'])

        if bead_type not in type_masses:
            type_masses[bead_type] = []
        type_masses[bead_type].append(bead_mass)

    # 计算每个 bead_type 的平均质量
    for bead_type, masses in type_masses.items():
        mass_dict[bead_type] = np.mean(masses)

    return mass_dict


def write_single_frame_xyz(coords: np.ndarray, types: np.ndarray, output_file: str,
                           comment: str = "", box: np.ndarray = None):
    """
    将单帧坐标写入XYZ文件。

    Args:
        coords: 坐标数组 (natoms, 3)
        types: 类型数组 (natoms,)
        output_file: 输出文件路径
        comment: 注释行内容
        box: 盒子尺寸（可选）
    """
    natoms = len(coords)

    with open(output_file, 'w') as f:
        f.write(f"{natoms}\n")
        f.write(f"{comment}\n")
        for i in range(natoms):
            atom_type = types[i]
            x, y, z = coords[i]
            if isinstance(atom_type, (int, np.integer)):
                element = f"T{atom_type}"
            else:
                element = str(atom_type)
            f.write(f"{element:4s} {x:12.6f} {y:12.6f} {z:12.6f}\n")

    print(f"写入 {natoms} 个原子到 {output_file}")


def cmd_convert_trajectory(args):
    """轨迹转换子命令"""
    print("="*60)
    print("AA轨迹 → CG轨迹转换")
    print("="*60)
    print(f"TPR文件: {args.tpr}")
    print(f"TRR文件: {args.trr}")
    print(f"映射文件: {args.mapping}")
    print(f"输出文件: {args.output}")
    print(f"帧间隔: {args.stride}")

    # 读取AA轨迹
    print("\n读取AA轨迹...")
    aa_frames = read_gromacs_trr_all_frames(args.tpr, args.trr, stride=args.stride)
    print(f"  共 {len(aa_frames)} 帧")

    # 转换为CG
    print("\n转换为CG...")
    cg_traj = convert_trajectory_to_cg(aa_frames, args.mapping)
    print(f"  共 {len(cg_traj)} 帧")

    # 保存
    save_cg_trajectory_pickle(cg_traj, args.output)

    # 可选：输出XYZ（最后一帧，AA unwrap + CG）
    if args.xyz:
        print("\n输出XYZ文件（最后一帧）...")
        base_name = Path(args.output).stem

        # 获取最后一帧
        last_aa_frame = aa_frames[-1]
        last_cg_frame = cg_traj[-1]

        # 输出 CG 最后一帧
        cg_xyz_file = f"{base_name}_last_CG.xyz"
        cg_box = last_cg_frame.get('box', np.array([0, 0, 0]))
        if cg_box.ndim == 2:
            cg_box_str = f"box=[{cg_box[0,1]-cg_box[0,0]:.3f}, {cg_box[1,1]-cg_box[1,0]:.3f}, {cg_box[2,1]-cg_box[2,0]:.3f}]"
        else:
            cg_box_str = f"box=[{cg_box[0]:.3f}, {cg_box[1]:.3f}, {cg_box[2]:.3f}]"
        write_single_frame_xyz(
            last_cg_frame['coords'], last_cg_frame['types'],
            cg_xyz_file,
            comment=f"CG last frame {cg_box_str}"
        )
        print(f"  ✓ CG last: {cg_xyz_file}")

        # 输出 AA 最后一帧（原始坐标，轨迹文件本身通常已是连续坐标）
        aa_xyz_file = f"{base_name}_last_AA.xyz"
        aa_box = last_aa_frame.get('box', np.array([0, 0, 0]))
        if aa_box.ndim == 2:
            aa_box_str = f"box=[{aa_box[0,1]-aa_box[0,0]:.3f}, {aa_box[1,1]-aa_box[1,0]:.3f}, {aa_box[2,1]-aa_box[2,0]:.3f}]"
        else:
            aa_box_str = f"box=[{aa_box[0]:.3f}, {aa_box[1]:.3f}, {aa_box[2]:.3f}]"
        write_single_frame_xyz(
            last_aa_frame['coords'], last_aa_frame['types'],
            aa_xyz_file,
            comment=f"AA last frame {aa_box_str}"
        )
        print(f"  ✓ AA last: {aa_xyz_file}")

    print("\n✓ 转换完成!")


def cmd_convert_data(args):
    """Data文件转换子命令"""
    print("="*60)
    print("AA data → CG data 转换")
    print("="*60)
    print(f"输入文件: {args.input}")
    print(f"映射文件: {args.mapping}")
    print(f"输出文件: {args.output}")

    # 读取AA data
    print("\n读取AA data...")
    aa_data = read_lammps_data(args.input)
    print(f"  原子数: {len(aa_data['ids'])}")
    print(f"  键数: {len(aa_data['bonds'])}")

    # 保存原始坐标（用于XYZ输出）
    original_coords = aa_data['coords'].copy()

    # 解缠分子坐标（处理PBC边界跨越）
    print("\n解缠分子坐标...")
    unwrapped_coords = unwrap_coords(
        aa_data['ids'], aa_data['types'], aa_data['coords'],
        aa_data['box'], aa_data['bonds']
    )
    aa_data['coords'] = unwrapped_coords
    print(f"  ✓ 坐标已解缠")

    # 转换为CG
    print("\n转换为CG...")
    cg_data, mapping = convert_data_to_cg(
        aa_data, args.mapping,
        args.bonds, args.angles, args.dihedrals,
        derive_topology=args.derive_topology,
        output_cg_topology_dir=args.output_cg_topology,
        type_mapping_yaml=args.type_mapping
    )
    print(f"  Beads: {cg_data['natoms']}")

    # 写入CG data
    write_cg_data_file(
        args.output, cg_data,
        cg_data.get('bonds'), cg_data.get('angles'), cg_data.get('dihedrals'),
        mass_list=aa_data['mass_list']
    )

    # 可选：输出XYZ（unwrap格式）
    if args.xyz:
        print("\n输出XYZ文件（unwrap格式）...")
        base_name = Path(args.input).stem

        # 输出AA unwrap坐标
        aa_xyz_file = f"{base_name}_AA_unwrap.xyz"
        box = aa_data['box']
        if box.ndim == 2:
            box_str = f"box=[{box[0,1]-box[0,0]:.3f}, {box[1,1]-box[1,0]:.3f}, {box[2,1]-box[2,0]:.3f}]"
        else:
            box_str = f"box=[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}]"
        write_single_frame_xyz(
            unwrapped_coords, aa_data['types'],
            aa_xyz_file,
            comment=f"AA unwrapped coordinates {box_str}"
        )

        # 输出CG坐标
        cg_xyz_file = f"{base_name}_CG.xyz"
        cg_box = cg_data['box']
        if cg_box.ndim == 2:
            cg_box_str = f"box=[{cg_box[0,1]-cg_box[0,0]:.3f}, {cg_box[1,1]-cg_box[1,0]:.3f}, {cg_box[2,1]-cg_box[2,0]:.3f}]"
        else:
            cg_box_str = f"box=[{cg_box[0]:.3f}, {cg_box[1]:.3f}, {cg_box[2]:.3f}]"
        write_single_frame_xyz(
            cg_data['coords'], cg_data['types'],
            cg_xyz_file,
            comment=f"CG bead coordinates {cg_box_str}"
        )

        print(f"  ✓ AA unwrap: {aa_xyz_file}")
        print(f"  ✓ CG: {cg_xyz_file}")

    print("\n✓ 转换完成!")


def cmd_convert_gro(args):
    """GRO文件转换子命令"""
    if not HAS_GMX_IO:
        print("错误: 需要安装 gmx_file_io.py（项目根目录）")
        return 1
    if not HAS_MDA:
        print("错误: 需要安装 MDAnalysis")
        print("安装方法: pip install MDAnalysis")
        return 1

    print("="*60)
    print("GRO结构文件 → CG data 转换")
    print("="*60)
    print(f"GRO文件: {args.gro}")
    print(f"TPR文件: {args.tpr}")
    print(f"映射文件: {args.mapping}")
    print(f"输出文件: {args.output}")

    # 1. 读取 GRO 坐标
    print("\n读取GRO坐标...")
    gro_df, box_nm = get_gro_pandas_data(args.gro)
    coords_nm = gro_df[['x', 'y', 'z']].values  # 单位 nm
    coords_A = coords_nm * 10.0  # 转换为 Å
    box_A = box_nm * 10.0  # 转换为 Å，形状 (3,)
    print(f"  原子数: {len(gro_df)}")
    print(f"  盒子: [{box_A[0]:.3f}, {box_A[1]:.3f}, {box_A[2]:.3f}] Å")

    # 2. 从 TPR 读取拓扑和原子信息
    print("\n从TPR读取拓扑...")
    u = mda.Universe(args.tpr)

    # 获取原子信息
    # 注意：MDAnalysis从TPR读取时，atoms.ids是0-based，需要转换为1-based
    atom_ids = u.atoms.ids.copy() + 1  # 转换为1-based，与mapping.csv的AA_id对应
    mol_ids = u.atoms.resids.copy()  # 分子ID (GROMACS resid通常是1-based)
    atom_types = u.atoms.types.copy()
    masses = u.atoms.masses.copy()

    # 尝试将类型转换为整数
    try:
        atom_types = np.array([int(t) for t in atom_types], dtype=np.int32)
    except (ValueError, TypeError):
        atom_types = np.array(atom_types, dtype=str)

    print(f"  原子数: {len(atom_ids)}")
    print(f"  分子数: {len(np.unique(mol_ids))}")

    # 构建质量字典 {type: mass}
    # 注意：原子类型可能是字符串（如 GROMACS 的 'c3'）或整数
    unique_atom_types = np.unique(atom_types)
    mass_dict = {}
    for atype in unique_atom_types:
        mask = atom_types == atype
        # 使用原始类型作为 key（字符串或整数）
        key = str(atype) if isinstance(atype, (str, np.str_)) else int(atype)
        mass_dict[key] = float(masses[mask][0])
    print(f"  原子类型数: {len(unique_atom_types)}")

    # 3. 提取 AA bonds
    aa_bonds = []
    if hasattr(u, 'bonds') and len(u.bonds) > 0:
        for bond in u.bonds:
            # 注意：bond.atoms[i].id是0-based，需要转换为1-based
            aa_bonds.append([1, bond.atoms[0].id + 1, bond.atoms[1].id + 1])
        print(f"  键数: {len(aa_bonds)}")
    else:
        print("  键数: 0 (TPR中无键信息)")
    aa_bonds = np.array(aa_bonds, dtype=np.int32).reshape(-1, 3) if aa_bonds else np.array([], dtype=np.int32).reshape(0, 3)

    # 4. 构建 aa_data 结构
    # 注意：unwrap_coords 需要 (3, 2) 格式的盒子
    aa_data = {
        'ids': atom_ids,
        'mol_ids': mol_ids,
        'types': atom_types,
        'coords': coords_A.astype(np.float64),
        'box': np.array([[0.0, box_A[0]], [0.0, box_A[1]], [0.0, box_A[2]]], dtype=np.float64),  # (3, 2) 格式
        'bonds': aa_bonds,
        'natoms': len(atom_ids),
        'mass_list': mass_dict  # 使用字典格式 {type: mass}
    }

    # 5. 解缠分子坐标（使用 TPR bonds）
    if len(aa_bonds) > 0:
        print("\n解缠分子坐标（基于 bonds）...")
        unwrapped_coords = unwrap_coords(
            aa_data['ids'], aa_data['types'], aa_data['coords'],
            aa_data['box'], aa_data['bonds']
        )
        aa_data['coords'] = unwrapped_coords
        print(f"  ✓ 坐标已解缠")
    else:
        print("\n警告: 无键信息，跳过基于 bonds 的解缠")
        unwrapped_coords = coords_A.copy()

    # 保存原始坐标（用于XYZ输出）
    original_coords = coords_A.copy()

    # 6. CG 转换 + 拓扑推导
    print("\n转换为CG...")
    cg_data, mapping = convert_data_to_cg(
        aa_data, args.mapping,
        None, None, None,  # 不提供预计算拓扑文件
        derive_topology=args.derive_topology,
        output_cg_topology_dir=args.output_cg_topology,
        type_mapping_yaml=args.type_mapping
    )
    print(f"  Beads: {cg_data['natoms']}")

    # 7. 写入 CG data
    # 构建 CG bead 质量字典
    cg_mass_dict = build_cg_mass_dict(cg_data, mapping)
    write_cg_data_file(
        args.output, cg_data,
        cg_data.get('bonds'), cg_data.get('angles'), cg_data.get('dihedrals'),
        mass_list=cg_mass_dict
    )

    # 8. 可选：输出 XYZ
    if args.xyz:
        print("\n输出XYZ文件...")
        base_name = Path(args.gro).stem

        # 输出 AA unwrap 坐标
        aa_xyz_file = f"{base_name}_AA_unwrap.xyz"
        box_str = f"box=[{box_A[0]:.3f}, {box_A[1]:.3f}, {box_A[2]:.3f}]"
        write_single_frame_xyz(
            unwrapped_coords, aa_data['types'],
            aa_xyz_file,
            comment=f"AA unwrapped coordinates {box_str}"
        )

        # 输出 CG 坐标
        cg_xyz_file = f"{base_name}_CG.xyz"
        cg_box = cg_data['box']
        if cg_box.ndim == 2:
            cg_box_str = f"box=[{cg_box[0,1]-cg_box[0,0]:.3f}, {cg_box[1,1]-cg_box[1,0]:.3f}, {cg_box[2,1]-cg_box[2,0]:.3f}]"
        else:
            cg_box_str = f"box=[{cg_box[0]:.3f}, {cg_box[1]:.3f}, {cg_box[2]:.3f}]"
        write_single_frame_xyz(
            cg_data['coords'], cg_data['types'],
            cg_xyz_file,
            comment=f"CG bead coordinates {cg_box_str}"
        )

        print(f"  ✓ AA unwrap: {aa_xyz_file}")
        print(f"  ✓ CG: {cg_xyz_file}")

    print("\n✓ 转换完成!")


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='AA到CG转换工具',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='子命令')

    # 轨迹转换子命令
    trj_parser = subparsers.add_parser('trj', help='轨迹转换')
    trj_parser.add_argument('--tpr', required=True, help='TPR拓扑文件')
    trj_parser.add_argument('--trr', required=True, help='TRR轨迹文件')
    trj_parser.add_argument('--mapping', '-m', required=True, help='CG映射CSV文件')
    trj_parser.add_argument('--output', '-o', default='cg_trajectory.pkl', help='输出文件')
    trj_parser.add_argument('--stride', type=int, default=1, help='帧间隔')
    trj_parser.add_argument('--xyz', action='store_true',
                            help='输出最后一帧XYZ文件：last_AA.xyz 和 last_CG.xyz')

    # Data转换子命令
    data_parser = subparsers.add_parser('data', help='Data文件转换')
    data_parser.add_argument('--input', '-i', required=True, help='输入LAMMPS data文件')
    data_parser.add_argument('--mapping', '-m', required=True, help='CG映射CSV文件')
    data_parser.add_argument('--output', '-o', default='cg.data', help='输出文件')
    data_parser.add_argument('--bonds', help='CG键文件（如果提供，优先使用，忽略--derive-topology）')
    data_parser.add_argument('--angles', help='CG角度文件')
    data_parser.add_argument('--dihedrals', help='CG二面角文件')
    data_parser.add_argument('--derive-topology', action='store_true',
                             help='自动从AA data + mapping推导CG拓扑（bonds/angles/dihedrals）')
    data_parser.add_argument('--output-cg-topology', metavar='DIR',
                             help='目录路径，保存推导的CG拓扑文件（cg_bonds.txt, cg_angles.txt, '
                                  'cg_dihedrals.txt, cg_bead_info.txt）')
    data_parser.add_argument('--xyz', action='store_true',
                             help='输出单帧XYZ文件：AA_unwrap.xyz（解缠坐标）和CG.xyz')
    data_parser.add_argument('--type-mapping', metavar='YAML',
                             help='YAML 文件路径，bead type 组合 → 拓扑类型映射表'
                                  '（仅 --derive-topology 模式下生效）')

    # GRO文件转换子命令
    gro_parser = subparsers.add_parser('gro', help='GRO结构文件转换（需TPR拓扑）')
    gro_parser.add_argument('--gro', '-g', required=True, help='GRO坐标文件')
    gro_parser.add_argument('--tpr', '-t', required=True, help='TPR拓扑文件')
    gro_parser.add_argument('--mapping', '-m', required=True, help='CG映射CSV文件')
    gro_parser.add_argument('--output', '-o', default='cg.data', help='输出CG data文件')
    gro_parser.add_argument('--derive-topology', action='store_true',
                            help='自动从AA bonds + mapping推导CG拓扑（bonds/angles/dihedrals）')
    gro_parser.add_argument('--output-cg-topology', metavar='DIR',
                            help='目录路径，保存推导的CG拓扑文件（cg_bonds.txt, cg_angles.txt, '
                                 'cg_dihedrals.txt, cg_bead_info.txt）')
    gro_parser.add_argument('--xyz', action='store_true',
                            help='输出XYZ文件：AA_unwrap.xyz（解缠坐标）和CG.xyz')
    gro_parser.add_argument('--type-mapping', metavar='YAML',
                            help='YAML 文件路径，bead type 组合 → 拓扑类型映射表'
                                 '（仅 --derive-topology 模式下生效）')

    args = parser.parse_args()

    if args.command == 'trj':
        return cmd_convert_trajectory(args)
    elif args.command == 'data':
        return cmd_convert_data(args)
    elif args.command == 'gro':
        return cmd_convert_gro(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())