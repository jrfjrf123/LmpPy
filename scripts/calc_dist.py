#!/usr/bin/env python
"""
分布计算脚本 - 计算键/角度/二面角/Pair分布并输出 VOTCA 格式

用法:
    python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./
    python -m LmpPy.scripts.calc_dist --traj cg_trajectory.lammpstrj --top-dir ./ --output-dir dist_output
    python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --calc-pairs --n-jobs 8

输入文件格式:
    cg_bonds.txt:       bond_type atom1 atom2
    cg_angles.txt:      angle_type atom1 atom2 atom3
    cg_dihedrals.txt:   dihedral_type atom1 atom2 atom3 atom4
    cg_bead_info.txt:   bead_id mol_id bead_type mass

输出格式 (VOTCA .dist.tgt):
    r/theta/phi  probability  i

作者: Claude
日期: 2026-04-05
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.ibm_potential.distribution import (
    load_cg_trajectory,
    calculate_bond_distribution,
    calculate_angle_distribution,
    calculate_dihedral_distribution,
    calculate_rdf,
    calculate_all_distributions
)
from LmpPy.tools.ibm_potential.dist_config import DistributionConfig


def save_votca_dist(filename: str, x: np.ndarray, hist: np.ndarray):
    """
    保存为 VOTCA 目标分布格式。

    Args:
        filename: 输出文件路径
        x: 距离/角度/二面角数组
        hist: 概率分布数组
    """
    with open(filename, 'w') as f:
        for xi, hi in zip(x, hist):
            f.write(f"{xi:.6f} {hi:.6e} i\n")


def load_topology_files(top_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载拓扑文件。

    Args:
        top_dir: 拓扑文件目录

    Returns:
        (bonds_df, angles_df, dihedrals_df, bead_info_df)
    """
    top_path = Path(top_dir)

    bonds_file = top_path / "cg_bonds.txt"
    angles_file = top_path / "cg_angles.txt"
    dihedrals_file = top_path / "cg_dihedrals.txt"
    bead_info_file = top_path / "cg_bead_info.txt"

    # 加载键文件
    if bonds_file.exists():
        bonds_df = pd.read_csv(bonds_file, sep=r'\s+', comment='#')
        # 统一列名
        if 'atom1' in bonds_df.columns and 'atom1_id' not in bonds_df.columns:
            bonds_df.rename(columns={'atom1': 'atom1_id', 'atom2': 'atom2_id'}, inplace=True)
    else:
        bonds_df = pd.DataFrame()

    # 加载角度文件
    if angles_file.exists():
        angles_df = pd.read_csv(angles_file, sep=r'\s+', comment='#')
        if 'atom1' in angles_df.columns and 'atom1_id' not in angles_df.columns:
            angles_df.rename(columns={'atom1': 'atom1_id', 'atom2': 'atom2_id', 'atom3': 'atom3_id'}, inplace=True)
    else:
        angles_df = pd.DataFrame()

    # 加载二面角文件
    if dihedrals_file.exists():
        dihedrals_df = pd.read_csv(dihedrals_file, sep=r'\s+', comment='#')
        if 'atom1' in dihedrals_df.columns and 'atom1_id' not in dihedrals_df.columns:
            dihedrals_df.rename(columns={'atom1': 'atom1_id', 'atom2': 'atom2_id', 'atom3': 'atom3_id', 'atom4': 'atom4_id'}, inplace=True)
    else:
        dihedrals_df = pd.DataFrame()

    # 加载 bead_info 文件
    if bead_info_file.exists():
        bead_info_df = pd.read_csv(bead_info_file, sep=r'\s+', comment='#')
    else:
        bead_info_df = pd.DataFrame()

    return bonds_df, angles_df, dihedrals_df, bead_info_df


def run_pipeline(args):
    """
    运行分布计算流程。

    Args:
        args: 命令行参数
    """
    print("=" * 60)
    print("分布计算脚本")
    print("=" * 60)
    print(f"并行进程数: {args.n_jobs}")

    # 1. 加载轨迹
    print("\n加载轨迹...")
    cg_data = load_cg_trajectory(args.traj)
    print(f"  文件: {args.traj}")
    print(f"  共 {len(cg_data['R'])} 帧")

    # 2. 加载拓扑
    print("\n加载拓扑...")
    bonds_df, angles_df, dihedrals_df, bead_info_df = load_topology_files(args.top_dir)
    print(f"  目录: {args.top_dir}")
    print(f"  键: {len(bonds_df)} ({bonds_df['bond_type'].nunique() if len(bonds_df) > 0 else 0} 类型)")
    print(f"  角度: {len(angles_df)} ({angles_df['angle_type'].nunique() if len(angles_df) > 0 else 0} 类型)")
    print(f"  二面角: {len(dihedrals_df)} ({dihedrals_df['dihedral_type'].nunique() if len(dihedrals_df) > 0 else 0} 类型)")
    print(f"  Bead信息: {len(bead_info_df)} beads ({bead_info_df['bead_type'].nunique() if len(bead_info_df) > 0 else 0} 类型)")

    # 3. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 计算键距离分布
    if len(bonds_df) > 0:
        print("\n计算键距离分布...")
        atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']

        for bond_type in sorted(bonds_df['bond_type'].unique()):
            bonds_of_type = bonds_df[bonds_df['bond_type'] == bond_type]
            bond_pairs = bonds_of_type[atom_cols].values.T

            r, hist = calculate_bond_distribution(
                cg_data, bond_pairs,
                n_bins=args.n_bins,
                custom_range=tuple(args.bond_range)
            )

            output_file = output_dir / f"bond_type{bond_type}.dist.tgt"
            save_votca_dist(str(output_file), r, hist)
            print(f"  bond_type{bond_type}: {len(bonds_of_type)} 键 -> {output_file.name}")

    # 5. 计算角度分布
    if len(angles_df) > 0:
        print("\n计算角度分布...")
        atom_cols = ['atom1_id', 'atom2_id', 'atom3_id'] if 'atom1_id' in angles_df.columns else ['atom1', 'atom2', 'atom3']

        for angle_type in sorted(angles_df['angle_type'].unique()):
            angles_of_type = angles_df[angles_df['angle_type'] == angle_type]
            angle_triplets = angles_of_type[atom_cols].values.T

            theta, hist = calculate_angle_distribution(
                cg_data, angle_triplets,
                n_bins=args.n_bins,
                custom_range=tuple(args.angle_range)
            )

            output_file = output_dir / f"angle_type{angle_type}.dist.tgt"
            save_votca_dist(str(output_file), theta, hist)
            print(f"  angle_type{angle_type}: {len(angles_of_type)} 角度 -> {output_file.name}")

    # 6. 计算二面角分布
    if len(dihedrals_df) > 0:
        print("\n计算二面角分布...")
        atom_cols = ['atom1_id', 'atom2_id', 'atom3_id', 'atom4_id'] if 'atom1_id' in dihedrals_df.columns else ['atom1', 'atom2', 'atom3', 'atom4']

        for dihedral_type in sorted(dihedrals_df['dihedral_type'].unique()):
            dihedrals_of_type = dihedrals_df[dihedrals_df['dihedral_type'] == dihedral_type]
            dihedral_quads = dihedrals_of_type[atom_cols].values.T

            phi, hist = calculate_dihedral_distribution(
                cg_data, dihedral_quads,
                n_bins=args.n_bins,
                custom_range=tuple(args.dihedral_range)
            )

            output_file = output_dir / f"dihedral_type{dihedral_type}.dist.tgt"
            save_votca_dist(str(output_file), phi, hist)
            print(f"  dihedral_type{dihedral_type}: {len(dihedrals_of_type)} 二面角 -> {output_file.name}")

    # 7. 计算 Pair 分布（RDF）
    if args.calc_pairs and len(bead_info_df) > 0:
        print("\n计算 Pair 距离分布（排除 1-2/1-3/1-4）...")

        # 获取所有 bead 类型
        unique_types = sorted(bead_info_df['bead_type'].unique())
        print(f"  Bead类型: {unique_types}")

        # 计算每对 bead type 的 RDF
        for i, type1 in enumerate(unique_types):
            for type2 in unique_types[i:]:
                r, g_r = calculate_rdf(
                    cg_data, bead_info_df, type1, type2,
                    bonds_df=bonds_df,
                    angles_df=angles_df,
                    dihedrals_df=dihedrals_df,
                    exclude_bonds=args.exclude_12,
                    exclude_angles=args.exclude_13,
                    exclude_dihedrals=args.exclude_14,
                    n_bins=args.n_bins,
                    custom_range=tuple(args.pair_range),
                    n_jobs=args.n_jobs,
                    show_progress=True
                )

                if len(r) > 0:
                    output_file = output_dir / f"pair_type{type1}_{type2}.dist.tgt"
                    save_votca_dist(str(output_file), r, g_r)
                    print(f"  pair type{type1}-{type2} -> {output_file.name}")

    print("\n" + "=" * 60)
    print(f"✓ 完成! 输出目录: {output_dir}")
    print("=" * 60)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='计算键/角度/二面角/Pair分布并输出 VOTCA 格式\n\n'
                    '注意: 默认假设轨迹距离单位为 Å (埃)。如果轨迹是 nm 单位，请使用 --unit nm。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（轨迹单位为 Å）
  python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./

  # 如果轨迹单位是 nm
  python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./ --unit nm

  # 自定义范围（Å 单位）
  python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --bond-range 2.0 6.0

  # 计算 pair 分布
  python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --calc-pairs --n-jobs 8
"""
    )

    parser.add_argument('--traj', required=True,
                        help='CG轨迹文件 (.pkl 或 .lammpstrj)')
    parser.add_argument('--top-dir', default='.',
                        help='拓扑文件目录 (包含 cg_bonds.txt, cg_angles.txt, cg_dihedrals.txt, cg_bead_info.txt)')
    parser.add_argument('--output-dir', default='./dist_output',
                        help='输出目录 (默认: ./dist_output)')
    parser.add_argument('--n-bins', type=int, default=100,
                        help='直方图bins数 (默认: 100)')
    parser.add_argument('--unit', choices=['A', 'nm'], default='A',
                        help='轨迹距离单位 (默认: A, 埃)')
    parser.add_argument('--bond-range', nargs=2, type=float, default=[2.0, 6.0],
                        help='键距离范围 (与轨迹单位一致), 默认: 2.0 6.0 (Å)')
    parser.add_argument('--angle-range', nargs=2, type=float, default=[0, 180],
                        help='角度范围 (度), 默认: 0 180')
    parser.add_argument('--dihedral-range', nargs=2, type=float, default=[-180, 180],
                        help='二面角范围 (度), 默认: -180 180')
    parser.add_argument('--calc-pairs', action='store_true',
                        help='计算所有 bead type 对的距离分布（排除 1-2/1-3/1-4）')
    parser.add_argument('--pair-range', nargs=2, type=float, default=[3.0, 15.0],
                        help='Pair 距离范围 (与轨迹单位一致), 默认: 3.0 15.0 (Å)')
    parser.add_argument('--exclude-12', action='store_true', default=True,
                        help='排除 1-2 对 (键)')
    parser.add_argument('--exclude-13', action='store_true', default=True,
                        help='排除 1-3 对 (角度两端)')
    parser.add_argument('--exclude-14', action='store_true', default=True,
                        help='排除 1-4 对 (二面角两端)')
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='并行进程数（默认: 1，串行模式；建议 < 10）')

    args = parser.parse_args()

    try:
        run_pipeline(args)
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())