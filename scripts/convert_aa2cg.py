#!/usr/bin/env python
"""
AA到CG转换工具 CLI

用法:
    # 转换轨迹
    python -m LmpPy.scripts.convert_aa2cg trj --tpr topol.tpr --trr traj.trr --mapping mapping.csv -o cg_traj.pkl

    # 转换data文件
    python -m LmpPy.scripts.convert_aa2cg data --input system.data --mapping mapping.csv --bonds cg_bonds.txt

作者: Claude
日期: 2026-04-02
"""

import sys
from pathlib import Path

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
    unwrap_coords
)


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

    # 可选：输出XYZ
    if args.xyz:
        xyz_file = Path(args.output).stem + ".xyz"
        write_trajectory_to_xyz(cg_traj, xyz_file)
        print(f"\nXYZ轨迹: {xyz_file}")

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
        args.bonds, args.angles, args.dihedrals
    )
    print(f"  Beads: {cg_data['natoms']}")

    # 写入CG data
    write_cg_data_file(
        args.output, cg_data,
        cg_data.get('bonds'), cg_data.get('angles'), cg_data.get('dihedrals'),
        mass_list=aa_data['mass_list']
    )

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
    trj_parser.add_argument('--xyz', action='store_true', help='同时输出XYZ格式')

    # Data转换子命令
    data_parser = subparsers.add_parser('data', help='Data文件转换')
    data_parser.add_argument('--input', '-i', required=True, help='输入LAMMPS data文件')
    data_parser.add_argument('--mapping', '-m', required=True, help='CG映射CSV文件')
    data_parser.add_argument('--output', '-o', default='cg.data', help='输出文件')
    data_parser.add_argument('--bonds', help='CG键文件')
    data_parser.add_argument('--angles', help='CG角度文件')
    data_parser.add_argument('--dihedrals', help='CG二面角文件')

    args = parser.parse_args()

    if args.command == 'trj':
        return cmd_convert_trajectory(args)
    elif args.command == 'data':
        return cmd_convert_data(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())