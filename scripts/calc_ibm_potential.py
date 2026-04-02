#!/usr/bin/env python
"""
IBM势能计算工具 CLI

用法:
    python -m LmpPy.scripts.calc_ibm_potential -c ibm_potential.yaml
    python -m LmpPy.scripts.calc_ibm_potential --traj cg_trajectory.pkl --mapping mapping.csv

作者: Claude
日期: 2026-04-02
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.ibm_potential import (
    load_ibm_config,
    load_cg_trajectory,
    load_topology,
    load_type_dict,
    calculate_bond_distribution,
    calculate_angle_distribution,
    calculate_dihedral_distribution,
    calculate_rdf,
    calculate_bond_potential,
    calculate_angle_potential,
    calculate_dihedral_potential,
    calculate_pair_potential,
    extrapolate_and_smooth,
    save_potential,
    create_lammps_table_files,
    create_reference_file
)


def run_ibm_pipeline(config_path: str = None, verbose: bool = True):
    """
    运行IBM势能计算流程。

    Args:
        config_path: 配置文件路径
        verbose: 是否打印详细信息
    """
    # 加载配置
    config = load_ibm_config(config_path)

    if verbose:
        print("="*70)
        print("IBM势能计算流程")
        print("="*70)
        print(f"温度: {config.simulation.temperature} K")
        print(f"轨迹: {config.trajectory.path}")
        print(f"输出目录: {config.output.potentials_dir}")

    # 创建输出目录
    os.makedirs(config.output.distributions_dir, exist_ok=True)
    os.makedirs(config.output.potentials_dir, exist_ok=True)

    # 加载轨迹
    if verbose:
        print("\n加载CG轨迹...")
    cg_data = load_cg_trajectory(
        config.resolve_path(config.trajectory.path),
        format=config.trajectory.format
    )
    n_frames = len(cg_data['R'])
    if verbose:
        print(f"  共 {n_frames} 帧")

    # 加载拓扑
    if verbose:
        print("\n加载拓扑文件...")

    # 这里需要配置中的拓扑文件路径
    # 使用默认文件名
    config_dir = config.config_dir
    bonds_file = config_dir / "cg_bonds.txt"
    angles_file = config_dir / "cg_angles.txt"
    dihedrals_file = config_dir / "cg_dihedrals.txt"
    bead_info_file = config_dir / "cg_bead_info.txt"

    bonds_df, angles_df, dihedrals_df = load_topology(
        str(bonds_file), str(angles_file), str(dihedrals_file)
    )

    if verbose:
        print(f"  键: {len(bonds_df)} ({bonds_df['bond_type'].nunique() if len(bonds_df) > 0 else 0} 类型)")
        print(f"  角度: {len(angles_df)} ({angles_df['angle_type'].nunique() if len(angles_df) > 0 else 0} 类型)")
        print(f"  二面角: {len(dihedrals_df)} ({dihedrals_df['dihedral_type'].nunique() if len(dihedrals_df) > 0 else 0} 类型)")

    # 加载bead信息
    import pandas as pd
    if bead_info_file.exists():
        bead_info = pd.read_csv(bead_info_file, sep=r'\s+', comment='#')
    else:
        bead_info = None

    # 计算分布和势能
    temperature = config.simulation.temperature
    n_bins = config.distribution.n_bins

    # 键分布和势能
    if verbose:
        print("\n计算键分布...")
    if len(bonds_df) > 0:
        for bond_type in sorted(bonds_df['bond_type'].unique()):
            bonds_of_type = bonds_df[bonds_df['bond_type'] == bond_type]
            bond_pairs = bonds_of_type[['atom1_id', 'atom2_id']].values.T

            dist_file = os.path.join(config.output.distributions_dir, f'bond_type{bond_type}_dist.txt')
            r, hist = calculate_bond_distribution(
                cg_data, bond_pairs,
                n_bins=n_bins,
                custom_range=config.distribution.bond_range,
                output_file=dist_file
            )

            # 计算势能
            r, U = calculate_bond_potential(r, hist, temperature=temperature)
            r, U = extrapolate_and_smooth(r, U,
                                          smooth_window=config.smoothing.sg_window,
                                          smooth_polyorder=config.smoothing.sg_polyorder)

            pot_file = os.path.join(config.output.potentials_dir, f'bond_type{bond_type}_potential.txt')
            save_potential(pot_file, r, U, 'Bond')

            if verbose:
                print(f"  键类型 {bond_type}: {len(bonds_of_type)} 键")

    # 角度分布和势能
    if verbose:
        print("\n计算角度分布...")
    if len(angles_df) > 0:
        for angle_type in sorted(angles_df['angle_type'].unique()):
            angles_of_type = angles_df[angles_df['angle_type'] == angle_type]
            angle_triplets = angles_of_type[['atom1_id', 'atom2_id', 'atom3_id']].values.T

            dist_file = os.path.join(config.output.distributions_dir, f'angle_type{angle_type}_dist.txt')
            theta, hist = calculate_angle_distribution(
                cg_data, angle_triplets,
                n_bins=n_bins,
                custom_range=config.distribution.angle_range,
                output_file=dist_file
            )

            theta, U = calculate_angle_potential(theta, hist, temperature=temperature)
            theta, U = extrapolate_and_smooth(theta, U,
                                               smooth_window=config.smoothing.sg_window,
                                               smooth_polyorder=config.smoothing.sg_polyorder)

            pot_file = os.path.join(config.output.potentials_dir, f'angle_type{angle_type}_potential.txt')
            save_potential(pot_file, theta, U, 'Angle')

    # 二面角分布和势能
    if verbose:
        print("\n计算二面角分布...")
    if len(dihedrals_df) > 0:
        for dihedral_type in sorted(dihedrals_df['dihedral_type'].unique()):
            dihedrals_of_type = dihedrals_df[dihedrals_df['dihedral_type'] == dihedral_type]
            dihedral_quads = dihedrals_of_type[['atom1_id', 'atom2_id', 'atom3_id', 'atom4_id']].values.T

            dist_file = os.path.join(config.output.distributions_dir, f'dihedral_type{dihedral_type}_dist.txt')
            phi, hist = calculate_dihedral_distribution(
                cg_data, dihedral_quads,
                n_bins=n_bins,
                custom_range=config.distribution.dihedral_range,
                output_file=dist_file
            )

            phi, U = calculate_dihedral_potential(phi, hist, temperature=temperature)
            phi, U = extrapolate_and_smooth(phi, U,
                                             smooth_window=config.smoothing.sg_window,
                                             smooth_polyorder=config.smoothing.sg_polyorder)

            pot_file = os.path.join(config.output.potentials_dir, f'dihedral_type{dihedral_type}_potential.txt')
            save_potential(pot_file, phi, U, 'Dihedral')

    # RDF和pair势能
    if verbose:
        print("\n计算RDF...")
    if bead_info is not None:
        unique_types = sorted(bead_info['bead_type'].unique())

        for i, type1 in enumerate(unique_types):
            for type2 in unique_types[i:]:
                r, g_r = calculate_rdf(
                    cg_data, bead_info, type1, type2,
                    bonds_df=bonds_df, angles_df=angles_df, dihedrals_df=dihedrals_df,
                    exclude_bonds=config.distribution.rdf_exclude_bonds,
                    exclude_angles=config.distribution.rdf_exclude_angles,
                    exclude_dihedrals=config.distribution.rdf_exclude_dihedrals,
                    n_bins=n_bins,
                    custom_range=config.distribution.pair_range,
                    dr=config.distribution.rdf_dr,
                    output_file=os.path.join(config.output.distributions_dir, f'rdf_type{type1}_{type2}.txt')
                )

                if len(r) > 0:
                    r, U, g_r = calculate_pair_potential(r, g_r, temperature=temperature)
                    r, U = extrapolate_and_smooth(r, U,
                                                   smooth_window=config.smoothing.sg_window,
                                                   smooth_polyorder=config.smoothing.sg_polyorder)

                    pot_file = os.path.join(config.output.potentials_dir, f'pair_type{type1}_{type2}_potential.txt')
                    save_potential(pot_file, r, U, 'Pair')

                    if verbose:
                        print(f"  Pair {type1}-{type2}")

    # 生成LAMMPS table文件
    if verbose:
        print("\n生成LAMMPS table文件...")
    output_files = create_lammps_table_files(
        config.output.potentials_dir,
        config.output.table_dir
    )

    if output_files:
        create_reference_file(config.output.table_dir)

    if verbose:
        print("\n" + "="*70)
        print("✓ IBM势能计算完成!")
        print("="*70)


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='IBM势能计算工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用配置文件
  python -m LmpPy.scripts.calc_ibm_potential -c ibm_potential.yaml

  # 使用默认配置
  python -m LmpPy.scripts.calc_ibm_potential
"""
    )

    parser.add_argument('-c', '--config', help='配置文件路径 (ibm_potential.yaml)')
    parser.add_argument('-q', '--quiet', action='store_true', help='安静模式')

    args = parser.parse_args()

    try:
        run_ibm_pipeline(args.config, verbose=not args.quiet)
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())