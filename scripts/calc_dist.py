#!/usr/bin/env python
"""
分布计算脚本 - 计算键/角度/二面角/Pair分布并输出 VOTCA 格式

用法:
    # LAMMPS 格式
    python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./
    python -m LmpPy.scripts.calc_dist --traj cg_trajectory.lammpstrj --top-dir ./ --output-dir dist_output

    # GROMACS 格式（支持大轨迹文件）
    python -m LmpPy.scripts.calc_dist --tpr topol.tpr --xtc traj.xtc --top-dir ./ --stride 10
    python -m LmpPy.scripts.calc_dist --tpr topol.tpr --trr traj.trr --stride 5 --parallel 4

输入文件格式:
    cg_bonds.txt:       bond_type atom1 atom2
    cg_angles.txt:      angle_type atom1 atom2 atom3
    cg_dihedrals.txt:   dihedral_type atom1 atom2 atom3 atom4
    cg_bead_info.txt:   bead_id mol_id bead_type mass

    GROMACS 格式:
    TPR: 拓扑文件（包含 bonds/angles/dihedrals）
    XTC/TRR: 轨迹文件

输出格式 (VOTCA .dist.tgt):
    r/theta/phi  probability  i

作者: Claude
日期: 2026-04-05 (更新: 2026-05-19 添加 GROMACS 支持)
"""

import sys
import argparse
import re
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

# 可选导入 GROMACS 加载模块
try:
    from LmpPy.tools.ibm_potential.gromacs_loader import (
        TrajectoryCache,
        TopologyIndex,
        calculate_bond_distribution_vectorized,
        calculate_angle_distribution_vectorized,
        calculate_dihedral_distribution_vectorized,
        calculate_rdf_vectorized,
        HAS_MDA
    )
    HAS_GROMACS_SUPPORT = HAS_MDA
except ImportError:
    HAS_GROMACS_SUPPORT = False

# 可选导入 Pickle 向量化加载模块
try:
    from LmpPy.tools.ibm_potential.pickle_loader import (
        PickleTrajectoryCache,
        load_pickle_vectorized,
        calculate_bond_distribution_vectorized as calc_bond_pickle,
        calculate_angle_distribution_vectorized as calc_angle_pickle,
        calculate_dihedral_distribution_vectorized as calc_dihedral_pickle,
        calculate_rdf_vectorized as calc_rdf_pickle,
        calculate_rdf_frame_by_frame as calc_rdf_pickle_fbf,
        check_existing_distributions,
        print_existing_summary
    )
    HAS_PICKLE_VECTORIZED = True
except ImportError:
    HAS_PICKLE_VECTORIZED = False


def normalize_distribution(x: np.ndarray, hist: np.ndarray, mode: str = None) -> np.ndarray:
    """
    对分布进行归一化处理。

    Args:
        x: 距离/角度/二面角数组（用于area模式积分）
        hist: 概率分布数组
        mode: 'max' 最大值为1, 'area' 积分面积为1, None 不处理

    Returns:
        归一化后的分布数组
    """
    if mode == 'max':
        max_val = np.max(hist)
        if max_val > 0:
            return hist / max_val
    elif mode == 'area':
        # 兼容 NumPy 新旧版本：优先使用 trapezoid，回退到 trapz
        if hasattr(np, 'trapezoid'):
            area = np.trapezoid(hist, x)
        else:
            area = np.trapz(hist, x)
        if area > 0:
            return hist / area
    return hist


def save_votca_dist(filename: str, x: np.ndarray, hist: np.ndarray, normalize_mode: str = None):
    """
    保存为 VOTCA 目标分布格式，支持可选归一化。

    Args:
        filename: 输出文件路径
        x: 距离/角度/二面角数组
        hist: 概率分布数组
        normalize_mode: 'max' | 'area' | None
    """
    if normalize_mode:
        hist = normalize_distribution(x, hist, normalize_mode)
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
        bonds_df = pd.read_csv(bonds_file, sep=r'\s+', comment='#', header=None)
        # 统一列名：支持多种格式
        # 格式1: 无头部，列名为数字 0,1,2 -> bond_type, atom1_id, atom2_id
        # 格式2: 有头部，列名为 atom1, atom2 -> atom1_id, atom2_id
        if bonds_df.columns[0] == 0:  # 无头部，数字列名
            bonds_df.columns = ['bond_type', 'atom1_id', 'atom2_id']
        elif 'atom1' in bonds_df.columns and 'atom1_id' not in bonds_df.columns:
            bonds_df.rename(columns={'atom1': 'atom1_id', 'atom2': 'atom2_id'}, inplace=True)
    else:
        bonds_df = pd.DataFrame()

    # 加载角度文件
    if angles_file.exists():
        angles_df = pd.read_csv(angles_file, sep=r'\s+', comment='#', header=None)
        # 统一列名
        if angles_df.columns[0] == 0:  # 无头部，数字列名
            angles_df.columns = ['angle_type', 'atom1_id', 'atom2_id', 'atom3_id']
        elif 'atom1' in angles_df.columns and 'atom1_id' not in angles_df.columns:
            angles_df.rename(columns={'atom1': 'atom1_id', 'atom2': 'atom2_id', 'atom3': 'atom3_id'}, inplace=True)
    else:
        angles_df = pd.DataFrame()

    # 加载二面角文件
    if dihedrals_file.exists():
        dihedrals_df = pd.read_csv(dihedrals_file, sep=r'\s+', comment='#', header=None)
        # 统一列名
        if dihedrals_df.columns[0] == 0:  # 无头部，数字列名
            dihedrals_df.columns = ['dihedral_type', 'atom1_id', 'atom2_id', 'atom3_id', 'atom4_id']
        elif 'atom1' in dihedrals_df.columns and 'atom1_id' not in dihedrals_df.columns:
            dihedrals_df.rename(columns={'atom1': 'atom1_id', 'atom2': 'atom2_id', 'atom3': 'atom3_id', 'atom4': 'atom4_id'}, inplace=True)
    else:
        dihedrals_df = pd.DataFrame()

    # 加载 bead_info 文件
    if bead_info_file.exists():
        bead_info_df = pd.read_csv(bead_info_file, sep=r'\s+', comment='#', header=None)
        # 统一列名
        if bead_info_df.columns[0] == 0:  # 无头部，数字列名
            bead_info_df.columns = ['bead_id', 'mol_id', 'bead_type', 'mass']
    else:
        bead_info_df = pd.DataFrame()

    return bonds_df, angles_df, dihedrals_df, bead_info_df


def assign_ordered_bond_types(bonds_df: pd.DataFrame, bead_info_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    为 bond 分配有序的 type（不合并回环结构）。

    测试模式：区分 (type1, type2) 和 (type2, type1) 为不同类型。

    Args:
        bonds_df: cg_bonds.txt 数据 [bond_type, atom1_id, atom2_id]
        bead_info_df: cg_bead_info.txt 数据 [bead_id, mol_id, bead_type, mass]

    Returns:
        bonds_df: 新增 'ordered_bond_type' 列
        ordered_type_map: {(type1, type2): ordered_type_id}
    """
    if len(bonds_df) == 0 or len(bead_info_df) == 0:
        return bonds_df, {}

    # 构建 bead_id -> bead_type 映射
    bead_type_map = dict(zip(bead_info_df['bead_id'], bead_info_df['bead_type']))

    # 收集有序 bead type 组合（不排序）
    ordered_type_map = {}
    next_type_id = 1

    # 添加新列
    bonds_df['ordered_bond_type'] = 0

    atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']

    for idx, row in bonds_df.iterrows():
        bead1 = int(row[atom_cols[0]])
        bead2 = int(row[atom_cols[1]])

        type1 = bead_type_map.get(bead1, 1)
        type2 = bead_type_map.get(bead2, 1)

        # 关键：不排序，保持原始顺序
        ordered_combo = (type1, type2)

        if ordered_combo not in ordered_type_map:
            ordered_type_map[ordered_combo] = next_type_id
            next_type_id += 1

        bonds_df.loc[idx, 'ordered_bond_type'] = ordered_type_map[ordered_combo]

    return bonds_df, ordered_type_map


def run_gromacs_pipeline(args):
    """
    运行 GROMACS 轨迹分布计算流程（激进内存策略）。

    Args:
        args: 命令行参数
    """
    if not HAS_GROMACS_SUPPORT:
        print("错误: 需要 MDAnalysis 支持")
        print("安装方法: pip install MDAnalysis")
        return 1

    print("=" * 60)
    print("GROMACS 轨迹分布计算（激进内存模式）")
    print("=" * 60)
    print(f"TPR 文件: {args.tpr}")
    print(f"轨迹文件: {args.trj}")
    print(f"帧间隔: {args.stride}")

    # 1. 加载全轨迹到内存
    print("\n[Step 1] 加载轨迹...")
    trj_cache = TrajectoryCache()
    trj_cache.load_from_gromacs(
        args.tpr, args.trj,
        stride=args.stride,
        verbose=True
    )

    # 2. 加载拓扑
    print("\n[Step 2] 加载拓扑...")
    topo_idx = TopologyIndex()
    topo_idx.load_from_gromacs(args.tpr, verbose=True)

    # 3. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # 4. 向量化计算键分布
    if topo_idx.n_bonds > 0:
        print("\n[Step 3] 计算键分布（向量化）...")
        r_bond, hist_bond, _ = calculate_bond_distribution_vectorized(
            trj_cache.coords_all, trj_cache.box_all, topo_idx.bond_pairs,
            n_bins=args.n_bins,
            bond_range=tuple(args.bond_range),
            normalize=True
        )
        results['bond'] = (r_bond, hist_bond)
        output_file = output_dir / "bond.dist.tgt"
        save_votca_dist(str(output_file), r_bond, hist_bond, args.normalize_mode)
        print(f"  键分布: {topo_idx.n_bonds} 键 -> {output_file.name}")

    # 5. 向量化计算角度分布
    if topo_idx.n_angles > 0:
        print("\n[Step 4] 计算角度分布（向量化）...")
        r_angle, hist_angle, _ = calculate_angle_distribution_vectorized(
            trj_cache.coords_all, trj_cache.box_all, topo_idx.angle_triples,
            n_bins=args.n_bins,
            angle_range=tuple(args.angle_range),
            normalize=True
        )
        results['angle'] = (r_angle, hist_angle)
        output_file = output_dir / "angle.dist.tgt"
        save_votca_dist(str(output_file), r_angle, hist_angle, args.normalize_mode)
        print(f"  角度分布: {topo_idx.n_angles} 角度 -> {output_file.name}")

    # 6. 向量化计算二面角分布
    if topo_idx.n_dihedrals > 0:
        print("\n[Step 5] 计算二面角分布（向量化）...")
        r_dih, hist_dih, _ = calculate_dihedral_distribution_vectorized(
            trj_cache.coords_all, trj_cache.box_all, topo_idx.dihedral_quads,
            n_bins=args.n_bins,
            dihedral_range=tuple(args.dihedral_range),
            normalize=True
        )
        results['dihedral'] = (r_dih, hist_dih)
        output_file = output_dir / "dihedral.dist.tgt"
        save_votca_dist(str(output_file), r_dih, hist_dih, args.normalize_mode)
        print(f"  二面角分布: {topo_idx.n_dihedrals} 二面角 -> {output_file.name}")

    # 7. 计算 RDF（可选）
    if args.calc_pairs:
        print("\n[Step 6] 计算 RDF...")
        # 从拓扑文件读取 bead_info
        _, _, _, bead_info_df = load_topology_files(args.top_dir)
        if len(bead_info_df) > 0:
            unique_types = sorted(bead_info_df['bead_type'].unique())
            print(f"  Bead类型: {unique_types}")

            for i, type1 in enumerate(unique_types):
                for type2 in unique_types[i:]:
                    # 构建 bead_info 数组
                    bead_info_arr = bead_info_df[['bead_id', 'bead_type']].values

                    r, g_r = calculate_rdf_vectorized(
                        trj_cache.coords_all, trj_cache.box_all, bead_info_arr,
                        type1, type2,
                        exclusion_12=topo_idx.exclusion_12,
                        exclusion_13=topo_idx.exclusion_13,
                        exclusion_14=topo_idx.exclusion_14,
                        n_bins=args.n_bins,
                        r_range=tuple(args.pair_range),
                        normalize=True,
                        verbose=True
                    )

                    if len(r) > 0:
                        results[f'pair_{type1}_{type2}'] = (r, g_r)
                        output_file = output_dir / f"pair_type{type1}_{type2}.dist.tgt"
                        save_votca_dist(str(output_file), r, g_r, args.normalize_mode)
                        print(f"  RDF {type1}-{type2} -> {output_file.name}")

    print("\n" + "=" * 60)
    print(f"✓ 完成! 输出目录: {output_dir}")
    print("=" * 60)

    return 0


def run_pickle_vectorized_pipeline(args):
    """
    运行 Pickle 格式向量化分布计算流程。

    Args:
        args: 命令行参数
    """
    if not HAS_PICKLE_VECTORIZED:
        print("错误: 无法导入 pickle_loader 模块")
        return 1

    print("=" * 60)
    print("Pickle 轨迹向量化分布计算")
    print("=" * 60)

    # 1. 加载轨迹
    print("\n[Step 1] 加载轨迹...")
    cache = load_pickle_vectorized(args.traj, verbose=True)

    # 2. 加载拓扑
    print("\n[Step 2] 加载拓扑...")
    bonds_df, angles_df, dihedrals_df, bead_info_df = load_topology_files(args.top_dir)

    bond_types = sorted(bonds_df['bond_type'].unique()) if len(bonds_df) > 0 else []
    angle_types = sorted(angles_df['angle_type'].unique()) if len(angles_df) > 0 else []
    dihedral_types = sorted(dihedrals_df['dihedral_type'].unique()) if len(dihedrals_df) > 0 else []
    bead_types = sorted(bead_info_df['bead_type'].unique()) if len(bead_info_df) > 0 else []

    print(f"  键: {len(bonds_df)} ({len(bond_types)} 类型)")
    print(f"  角度: {len(angles_df)} ({len(angle_types)} 类型)")
    print(f"  二面角: {len(dihedrals_df)} ({len(dihedral_types)} 类型)")
    print(f"  Beads: {len(bead_info_df)} ({len(bead_types)} 类型)")

    # 3. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 检查已存在文件（如果启用 --skip-existing）
    skip_bond_types = set()
    skip_angle_types = set()
    skip_dihedral_types = set()
    skip_pair_types = set()

    if args.skip_existing:
        existing = check_existing_distributions(output_dir, bond_types, angle_types, dihedral_types, bead_types)
        print_existing_summary(existing)

        # 解析需要跳过的类型
        for f in existing['bond']:
            # bond_type1.dist.tgt -> type 1
            match = re.match(r'bond_type(\d+)\.dist\.tgt', f)
            if match:
                skip_bond_types.add(int(match.group(1)))

        for f in existing['angle']:
            match = re.match(r'angle_type(\d+)\.dist\.tgt', f)
            if match:
                skip_angle_types.add(int(match.group(1)))

        for f in existing['dihedral']:
            match = re.match(r'dihedral_type(\d+)\.dist\.tgt', f)
            if match:
                skip_dihedral_types.add(int(match.group(1)))

        for f in existing['pair']:
            match = re.match(r'pair_type(\d+)_(\d+)\.dist\.tgt', f)
            if match:
                skip_pair_types.add((int(match.group(1)), int(match.group(2))))

        if any([skip_bond_types, skip_angle_types, skip_dihedral_types, skip_pair_types]):
            print("  将跳过已存在的计算")

    # 5. 向量化计算键分布
    if len(bonds_df) > 0:
        atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']

        if args.ordered_bond_type:
            # === 有序 bond type 测试模式 ===
            print("\n[Step 3] 计算键分布（有序类型测试）...")
            bonds_df, ordered_type_map = assign_ordered_bond_types(bonds_df, bead_info_df)

            print(f"  有序类型映射: {ordered_type_map}")
            print(f"  原合并类型数: {len(bond_types)}, 有序类型数: {len(ordered_type_map)}")

            for ordered_combo, ordered_type_id in sorted(ordered_type_map.items(), key=lambda x: x[1]):
                type1, type2 = ordered_combo
                bonds_of_type = bonds_df[bonds_df['ordered_bond_type'] == ordered_type_id]

                if len(bonds_of_type) == 0:
                    continue

                # 转换为 0-based 索引
                bond_pairs = bonds_of_type[atom_cols].values - 1  # (n_bonds, 2)

                r, hist, _ = calc_bond_pickle(
                    cache.coords_all, cache.box_all, bond_pairs,
                    n_bins=args.n_bins,
                    bond_range=tuple(args.bond_range),
                    normalize=True
                )

                output_file = output_dir / f"bond_ordered_{type1}_{type2}.dist.tgt"
                save_votca_dist(str(output_file), r, hist, args.normalize_mode)
                print(f"  bond_ordered_{type1}_{type2}: {len(bonds_of_type)} 键 -> {output_file.name}")
        else:
            # === 原合并模式 ===
            print("\n[Step 3] 计算键分布（向量化）...")

            for bond_type in bond_types:
                if bond_type in skip_bond_types:
                    print(f"  bond_type{bond_type}: 已存在，跳过")
                    continue

                bonds_of_type = bonds_df[bonds_df['bond_type'] == bond_type]
                # 转换为 0-based 索引
                bond_pairs = bonds_of_type[atom_cols].values - 1  # (n_bonds, 2)

                r, hist, _ = calc_bond_pickle(
                    cache.coords_all, cache.box_all, bond_pairs,
                    n_bins=args.n_bins,
                    bond_range=tuple(args.bond_range),
                    normalize=True
                )

                output_file = output_dir / f"bond_type{bond_type}.dist.tgt"
                save_votca_dist(str(output_file), r, hist, args.normalize_mode)
                print(f"  bond_type{bond_type}: {len(bonds_of_type)} 键 -> {output_file.name}")

    # 6. 向量化计算角度分布
    if len(angles_df) > 0:
        print("\n[Step 4] 计算角度分布（向量化）...")
        atom_cols = ['atom1_id', 'atom2_id', 'atom3_id'] if 'atom1_id' in angles_df.columns else ['atom1', 'atom2', 'atom3']

        for angle_type in angle_types:
            if angle_type in skip_angle_types:
                print(f"  angle_type{angle_type}: 已存在，跳过")
                continue

            angles_of_type = angles_df[angles_df['angle_type'] == angle_type]
            angle_triples = angles_of_type[atom_cols].values - 1  # (n_angles, 3)

            theta, hist, _ = calc_angle_pickle(
                cache.coords_all, cache.box_all, angle_triples,
                n_bins=args.n_bins,
                angle_range=tuple(args.angle_range),
                normalize=True
            )

            output_file = output_dir / f"angle_type{angle_type}.dist.tgt"
            save_votca_dist(str(output_file), theta, hist, args.normalize_mode)
            print(f"  angle_type{angle_type}: {len(angles_of_type)} 角度 -> {output_file.name}")

    # 7. 向量化计算二面角分布
    if len(dihedrals_df) > 0:
        print("\n[Step 5] 计算二面角分布（向量化）...")
        atom_cols = ['atom1_id', 'atom2_id', 'atom3_id', 'atom4_id'] if 'atom1_id' in dihedrals_df.columns else ['atom1', 'atom2', 'atom3', 'atom4']

        for dihedral_type in dihedral_types:
            if dihedral_type in skip_dihedral_types:
                print(f"  dihedral_type{dihedral_type}: 已存在，跳过")
                continue

            dihedrals_of_type = dihedrals_df[dihedrals_df['dihedral_type'] == dihedral_type]
            dihedral_quads = dihedrals_of_type[atom_cols].values - 1  # (n_dihedrals, 4)

            phi, hist, _ = calc_dihedral_pickle(
                cache.coords_all, cache.box_all, dihedral_quads,
                n_bins=args.n_bins,
                dihedral_range=tuple(args.dihedral_range),
                normalize=True
            )

            output_file = output_dir / f"dihedral_type{dihedral_type}.dist.tgt"
            save_votca_dist(str(output_file), phi, hist, args.normalize_mode)
            print(f"  dihedral_type{dihedral_type}: {len(dihedrals_of_type)} 二面角 -> {output_file.name}")

    # 8. 向量化计算 RDF（可选）
    if args.calc_pairs and len(bead_info_df) > 0:
        print("\n[Step 6] 计算 RDF（向量化）...")

        # 构建 bead_info 数组
        bead_info_arr = bead_info_df[['bead_id', 'bead_type']].values

        # 构建排除列表
        exclusion_12 = set()
        exclusion_13 = set()
        exclusion_14 = set()

        if args.exclude_12 and len(bonds_df) > 0:
            atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']
            for _, row in bonds_df.iterrows():
                a1, a2 = int(row[atom_cols[0]]), int(row[atom_cols[1]])
                exclusion_12.add((min(a1, a2), max(a1, a2)))

        if args.exclude_13 and len(angles_df) > 0:
            atom_cols = ['atom1_id', 'atom3_id'] if 'atom1_id' in angles_df.columns else ['atom1', 'atom3']
            for _, row in angles_df.iterrows():
                a1, a3 = int(row[atom_cols[0]]), int(row[atom_cols[1]])
                exclusion_13.add((min(a1, a3), max(a1, a3)))

        if args.exclude_14 and len(dihedrals_df) > 0:
            atom_cols = ['atom1_id', 'atom4_id'] if 'atom1_id' in dihedrals_df.columns else ['atom1', 'atom4']
            for _, row in dihedrals_df.iterrows():
                a1, a4 = int(row[atom_cols[0]]), int(row[atom_cols[1]])
                exclusion_14.add((min(a1, a4), max(a1, a4)))

        for i, type1 in enumerate(bead_types):
            for type2 in bead_types[i:]:
                if (type1, type2) in skip_pair_types:
                    print(f"  RDF {type1}-{type2}: 已存在，跳过")
                    continue

                r, g_r = calc_rdf_pickle(
                    cache.coords_all, cache.box_all, bead_info_arr,
                    type1, type2,
                    exclusion_12=exclusion_12 if args.exclude_12 else None,
                    exclusion_13=exclusion_13 if args.exclude_13 else None,
                    exclusion_14=exclusion_14 if args.exclude_14 else None,
                    n_bins=args.n_bins,
                    r_range=tuple(args.pair_range),
                    normalize=True,
                    verbose=True
                )

                if len(r) > 0:
                    output_file = output_dir / f"pair_type{type1}_{type2}.dist.tgt"
                    save_votca_dist(str(output_file), r, g_r, args.normalize_mode)
                    print(f"  RDF {type1}-{type2} -> {output_file.name}")

    print("\n" + "=" * 60)
    print(f"✓ 完成! 输出目录: {output_dir}")
    print("=" * 60)

    return 0


def run_pickle_pipeline(args):
    """
    运行 Pickle 格式逐帧分布计算流程（内存友好）。

    与 run_pickle_vectorized_pipeline 使用相同的数据加载，
    但 RDF 计算使用逐帧模式，内存占用 O(n_pairs) 而非 O(n_frames × n_pairs)。

    Args:
        args: 命令行参数
    """
    if not HAS_PICKLE_VECTORIZED:
        print("错误: 无法导入 pickle_loader 模块")
        return 1

    print("=" * 60)
    print("Pickle 轨迹逐帧分布计算")
    print("=" * 60)

    # 1. 加载轨迹
    print("\n[Step 1] 加载轨迹...")
    cache = load_pickle_vectorized(args.traj, verbose=True)

    # 2. 加载拓扑
    print("\n[Step 2] 加载拓扑...")
    bonds_df, angles_df, dihedrals_df, bead_info_df = load_topology_files(args.top_dir)

    bond_types = sorted(bonds_df['bond_type'].unique()) if len(bonds_df) > 0 else []
    angle_types = sorted(angles_df['angle_type'].unique()) if len(angles_df) > 0 else []
    dihedral_types = sorted(dihedrals_df['dihedral_type'].unique()) if len(dihedrals_df) > 0 else []
    bead_types = sorted(bead_info_df['bead_type'].unique()) if len(bead_info_df) > 0 else []

    print(f"  键: {len(bonds_df)} ({len(bond_types)} 类型)")
    print(f"  角度: {len(angles_df)} ({len(angle_types)} 类型)")
    print(f"  二面角: {len(dihedrals_df)} ({len(dihedral_types)} 类型)")
    print(f"  Beads: {len(bead_info_df)} ({len(bead_types)} 类型)")

    # 3. 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 4. 检查已存在文件
    skip_bond_types = set()
    skip_angle_types = set()
    skip_dihedral_types = set()
    skip_pair_types = set()

    if args.skip_existing:
        print("\n[Step 3] 检查已存在文件...")
        existing = check_existing_distributions(
            output_dir, bond_types, angle_types, dihedral_types, bead_types
        )
        print_existing_summary(existing)

        import re
        for f in existing['bond']:
            match = re.match(r'bond_type(\d+)\.dist\.tgt', f)
            if match:
                skip_bond_types.add(int(match.group(1)))
        for f in existing['angle']:
            match = re.match(r'angle_type(\d+)\.dist\.tgt', f)
            if match:
                skip_angle_types.add(int(match.group(1)))
        for f in existing['dihedral']:
            match = re.match(r'dihedral_type(\d+)\.dist\.tgt', f)
            if match:
                skip_dihedral_types.add(int(match.group(1)))
        for f in existing['pair']:
            match = re.match(r'pair_type(\d+)_(\d+)\.dist\.tgt', f)
            if match:
                skip_pair_types.add((int(match.group(1)), int(match.group(2))))

        if any([skip_bond_types, skip_angle_types, skip_dihedral_types, skip_pair_types]):
            print("  将跳过已存在的计算")

    # 5. 键分布（向量化，键数量小不会内存爆炸）
    if len(bonds_df) > 0:
        atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']

        if args.ordered_bond_type:
            print("\n[Step 4] 计算键分布（有序类型测试）...")
            bonds_df, ordered_type_map = assign_ordered_bond_types(bonds_df, bead_info_df)
            print(f"  有序类型映射: {ordered_type_map}")
            print(f"  原合并类型数: {len(bond_types)}, 有序类型数: {len(ordered_type_map)}")

            for ordered_combo, ordered_type_id in sorted(ordered_type_map.items(), key=lambda x: x[1]):
                type1, type2 = ordered_combo
                bonds_of_type = bonds_df[bonds_df['ordered_bond_type'] == ordered_type_id]
                if len(bonds_of_type) == 0:
                    continue
                bond_pairs = bonds_of_type[atom_cols].values - 1
                r, hist, _ = calc_bond_pickle(
                    cache.coords_all, cache.box_all, bond_pairs,
                    n_bins=args.n_bins, bond_range=tuple(args.bond_range), normalize=True
                )
                name1 = bead_info_df[bead_info_df['bead_type'] == type1]['bead_name'].iloc[0]
                name2 = bead_info_df[bead_info_df['bead_type'] == type2]['bead_name'].iloc[0]
                output_file = output_dir / f"bond_type{ordered_type_id}_{name1}_{name2}.dist.tgt"
                save_votca_dist(str(output_file), r, hist, args.normalize_mode)
                print(f"  bond_type{ordered_type_id}({name1}-{name2}): {len(bonds_of_type)} 键 -> {output_file.name}")
        else:
            print("\n[Step 4] 计算键分布（向量化）...")
            for bond_type in bond_types:
                if bond_type in skip_bond_types:
                    print(f"  bond_type{bond_type}: 已存在，跳过")
                    continue
                bonds_of_type = bonds_df[bonds_df['bond_type'] == bond_type]
                bond_pairs = bonds_of_type[atom_cols].values - 1  # (n_bonds, 2)
                r, hist, _ = calc_bond_pickle(
                    cache.coords_all, cache.box_all, bond_pairs,
                    n_bins=args.n_bins, bond_range=tuple(args.bond_range), normalize=True
                )
                output_file = output_dir / f"bond_type{bond_type}.dist.tgt"
                save_votca_dist(str(output_file), r, hist, args.normalize_mode)
                print(f"  bond_type{bond_type}: {len(bonds_of_type)} 键 -> {output_file.name}")

    # 6. 角度分布（向量化，角度数量小不会内存爆炸）
    if len(angles_df) > 0:
        print("\n[Step 5] 计算角度分布（向量化）...")
        for angle_type in angle_types:
            if angle_type in skip_angle_types:
                print(f"  angle_type{angle_type}: 已存在，跳过")
                continue
            angles_of_type = angles_df[angles_df['angle_type'] == angle_type]
            atom_cols = ['atom1_id', 'atom2_id', 'atom3_id'] if 'atom1_id' in angles_df.columns else ['atom1', 'atom2', 'atom3']
            angle_triples = angles_of_type[atom_cols].values - 1  # (n_angles, 3)
            r, hist, _ = calc_angle_pickle(
                cache.coords_all, cache.box_all, angle_triples,
                n_bins=args.n_bins, angle_range=tuple(args.angle_range), normalize=True
            )
            output_file = output_dir / f"angle_type{angle_type}.dist.tgt"
            save_votca_dist(str(output_file), r, hist, args.normalize_mode)
            print(f"  angle_type{angle_type}: {len(angles_of_type)} 角度 -> {output_file.name}")

    # 7. 二面角分布（向量化）
    if len(dihedrals_df) > 0:
        print("\n[Step 6] 计算二面角分布（向量化）...")
        for dihedral_type in dihedral_types:
            if dihedral_type in skip_dihedral_types:
                print(f"  dihedral_type{dihedral_type}: 已存在，跳过")
                continue
            dihedrals_of_type = dihedrals_df[dihedrals_df['dihedral_type'] == dihedral_type]
            atom_cols = ['atom1_id', 'atom2_id', 'atom3_id', 'atom4_id'] if 'atom1_id' in dihedrals_df.columns else ['atom1', 'atom2', 'atom3', 'atom4']
            dihedral_quads = dihedrals_of_type[atom_cols].values - 1  # (n_dihedrals, 4)
            phi, hist, _ = calc_dihedral_pickle(
                cache.coords_all, cache.box_all, dihedral_quads,
                n_bins=args.n_bins, dihedral_range=tuple(args.dihedral_range), normalize=True
            )
            output_file = output_dir / f"dihedral_type{dihedral_type}.dist.tgt"
            save_votca_dist(str(output_file), phi, hist, args.normalize_mode)
            print(f"  dihedral_type{dihedral_type}: {len(dihedrals_of_type)} 二面角 -> {output_file.name}")

    # 8. RDF（逐帧模式，内存 O(n_pairs)）
    if args.calc_pairs and len(bead_info_df) > 0:
        print("\n[Step 7] 计算 RDF（逐帧模式，内存友好）...")

        bead_info_arr = bead_info_df[['bead_id', 'bead_type']].values

        exclusion_12 = set()
        exclusion_13 = set()
        exclusion_14 = set()

        if args.exclude_12 and len(bonds_df) > 0:
            atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']
            for _, row in bonds_df.iterrows():
                a1, a2 = int(row[atom_cols[0]]), int(row[atom_cols[1]])
                exclusion_12.add((min(a1, a2), max(a1, a2)))

        if args.exclude_13 and len(angles_df) > 0:
            atom_cols = ['atom1_id', 'atom3_id'] if 'atom1_id' in angles_df.columns else ['atom1', 'atom3']
            for _, row in angles_df.iterrows():
                a1, a3 = int(row[atom_cols[0]]), int(row[atom_cols[1]])
                exclusion_13.add((min(a1, a3), max(a1, a3)))

        if args.exclude_14 and len(dihedrals_df) > 0:
            atom_cols = ['atom1_id', 'atom4_id'] if 'atom1_id' in dihedrals_df.columns else ['atom1', 'atom4']
            for _, row in dihedrals_df.iterrows():
                a1, a4 = int(row[atom_cols[0]]), int(row[atom_cols[1]])
                exclusion_14.add((min(a1, a4), max(a1, a4)))

        for i, type1 in enumerate(bead_types):
            for type2 in bead_types[i:]:
                if (type1, type2) in skip_pair_types:
                    print(f"  RDF {type1}-{type2}: 已存在，跳过")
                    continue

                r, g_r = calc_rdf_pickle_fbf(
                    cache.coords_all, cache.box_all, bead_info_arr,
                    type1, type2,
                    exclusion_12=exclusion_12 if args.exclude_12 else None,
                    exclusion_13=exclusion_13 if args.exclude_13 else None,
                    exclusion_14=exclusion_14 if args.exclude_14 else None,
                    n_bins=args.n_bins,
                    r_range=tuple(args.pair_range),
                    normalize=True,
                    verbose=True
                )

                output_file = output_dir / f"pair_type{type1}_{type2}.dist.tgt"
                save_votca_dist(str(output_file), r, g_r, args.normalize_mode)
                print(f"  RDF {type1}-{type2} -> {output_file.name}")

    print("\n" + "=" * 60)
    print(f"✓ 完成! 输出目录: {output_dir}")
    print("=" * 60)

    return 0


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

    # 3.5 检查已存在文件（--skip-existing 时跳过对应计算）
    skip_bond_types, skip_angle_types, skip_dihedral_types, skip_pair_types = set(), set(), set(), set()
    if args.skip_existing:
        for f in output_dir.glob("*.dist.tgt"):
            m = re.match(r'bond_type(\d+)\.dist\.tgt', f.name)
            if m:
                skip_bond_types.add(int(m.group(1))); continue
            m = re.match(r'angle_type(\d+)\.dist\.tgt', f.name)
            if m:
                skip_angle_types.add(int(m.group(1))); continue
            m = re.match(r'dihedral_type(\d+)\.dist\.tgt', f.name)
            if m:
                skip_dihedral_types.add(int(m.group(1))); continue
            m = re.match(r'pair_type(\d+)_(\d+)\.dist\.tgt', f.name)
            if m:
                skip_pair_types.add((int(m.group(1)), int(m.group(2))))
        if any([skip_bond_types, skip_angle_types, skip_dihedral_types, skip_pair_types]):
            print(f"\n--skip-existing: 跳过 bond {sorted(skip_bond_types)}, "
                  f"angle {sorted(skip_angle_types)}, "
                  f"dihedral {sorted(skip_dihedral_types)}, "
                  f"pair {sorted(skip_pair_types)}")

    # 4. 计算键距离分布
    if len(bonds_df) > 0:
        print("\n计算键距离分布...")
        atom_cols = ['atom1_id', 'atom2_id'] if 'atom1_id' in bonds_df.columns else ['atom1', 'atom2']

        for bond_type in sorted(bonds_df['bond_type'].unique()):
            if bond_type in skip_bond_types:
                print(f"  bond_type{bond_type}: 已存在，跳过")
                continue
            bonds_of_type = bonds_df[bonds_df['bond_type'] == bond_type]
            bond_pairs = bonds_of_type[atom_cols].values.T

            r, hist = calculate_bond_distribution(
                cg_data, bond_pairs,
                n_bins=args.n_bins,
                custom_range=tuple(args.bond_range)
            )

            output_file = output_dir / f"bond_type{bond_type}.dist.tgt"
            save_votca_dist(str(output_file), r, hist, args.normalize_mode)
            print(f"  bond_type{bond_type}: {len(bonds_of_type)} 键 -> {output_file.name}")

    # 5. 计算角度分布
    if len(angles_df) > 0:
        print("\n计算角度分布...")
        atom_cols = ['atom1_id', 'atom2_id', 'atom3_id'] if 'atom1_id' in angles_df.columns else ['atom1', 'atom2', 'atom3']

        for angle_type in sorted(angles_df['angle_type'].unique()):
            if angle_type in skip_angle_types:
                print(f"  angle_type{angle_type}: 已存在，跳过")
                continue
            angles_of_type = angles_df[angles_df['angle_type'] == angle_type]
            angle_triplets = angles_of_type[atom_cols].values.T

            theta, hist = calculate_angle_distribution(
                cg_data, angle_triplets,
                n_bins=args.n_bins,
                custom_range=tuple(args.angle_range)
            )

            output_file = output_dir / f"angle_type{angle_type}.dist.tgt"
            save_votca_dist(str(output_file), theta, hist, args.normalize_mode)
            print(f"  angle_type{angle_type}: {len(angles_of_type)} 角度 -> {output_file.name}")

    # 6. 计算二面角分布
    if len(dihedrals_df) > 0:
        print("\n计算二面角分布...")
        atom_cols = ['atom1_id', 'atom2_id', 'atom3_id', 'atom4_id'] if 'atom1_id' in dihedrals_df.columns else ['atom1', 'atom2', 'atom3', 'atom4']

        for dihedral_type in sorted(dihedrals_df['dihedral_type'].unique()):
            if dihedral_type in skip_dihedral_types:
                print(f"  dihedral_type{dihedral_type}: 已存在，跳过")
                continue
            dihedrals_of_type = dihedrals_df[dihedrals_df['dihedral_type'] == dihedral_type]
            dihedral_quads = dihedrals_of_type[atom_cols].values.T

            phi, hist = calculate_dihedral_distribution(
                cg_data, dihedral_quads,
                n_bins=args.n_bins,
                custom_range=tuple(args.dihedral_range)
            )

            output_file = output_dir / f"dihedral_type{dihedral_type}.dist.tgt"
            save_votca_dist(str(output_file), phi, hist, args.normalize_mode)
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
                if (type1, type2) in skip_pair_types:
                    print(f"  pair type{type1}-{type2}: 已存在，跳过")
                    continue
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
                    save_votca_dist(str(output_file), r, g_r, args.normalize_mode)
                    print(f"  pair type{type1}-{type2} -> {output_file.name}")

    print("\n" + "=" * 60)
    print(f"✓ 完成! 输出目录: {output_dir}")
    print("=" * 60)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='计算键/角度/二面角/Pair分布并输出 VOTCA 格式\n\n'
                    '注意: 默认假设轨迹距离单位为 Å (埃)。如果轨迹是 nm 单位，请使用 --unit nm。\n'
                    'GROMACS 轨迹支持: 使用 --tpr 和 --xtc/--trr 参数处理 GROMACS 格式轨迹。',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # LAMMPS 格式（轨迹单位为 Å）
  python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./

  # 如果轨迹单位是 nm
  python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./ --unit nm

  # 自定义范围（Å 单位）
  python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --bond-range 2.0 6.0

  # 计算 pair 分布
  python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --calc-pairs --n-jobs 8

  # GROMACS 格式（大轨迹文件）
  python -m LmpPy.scripts.calc_dist --tpr topol.tpr --xtc traj.xtc --top-dir ./ --stride 10

  # GROMACS TRR 格式（包含力信息）
  python -m LmpPy.scripts.calc_dist --tpr topol.tpr --trr traj.trr --stride 5 --n-jobs 4
"""
    )

    # === 格式选择参数 ===
    # LAMMPS/Pickle 格式
    parser.add_argument('--traj', default=None,
                        help='CG轨迹文件 (.pkl 或 .lammpstrj)')

    # GROMACS 格式参数
    parser.add_argument('--tpr', type=str, default=None,
                        help='GROMACS TPR 拓扑文件')
    parser.add_argument('--xtc', '--trr', dest='trj', type=str, default=None,
                        help='GROMACS XTC/TRR 轨迹文件')

    # 大轨迹处理参数
    parser.add_argument('--stride', type=int, default=1,
                        help='帧间隔（跳帧，减少数据量），默认: 1')

    # === 通用参数 ===
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
                        help='并行进程数（用于 RDF 计算），默认: 1（串行）')
    parser.add_argument('--skip-existing', action='store_true',
                        help='跳过已存在的输出文件')
    parser.add_argument('--vectorized', action='store_true',
                        help='使用全向量化计算（仅对 .pkl 格式有效，速度快但内存消耗大，默认关闭）')
    parser.add_argument('--ordered-bond-type', action='store_true',
                        help='测试模式：区分 bond 方向性，不合并 (type1,type2) 和 (type2,type1)')

    # === 归一化参数（互斥） ===
    norm_group = parser.add_mutually_exclusive_group()
    norm_group.add_argument('--norm-max', action='store_true',
                            help='对输出分布进行最大值为1的归一化')
    norm_group.add_argument('--norm-area', action='store_true',
                            help='对输出分布进行积分面积为1的归一化（概率密度归一化）')

    args = parser.parse_args()

    # 确定归一化模式
    args.normalize_mode = None
    if args.norm_max:
        args.normalize_mode = 'max'
    elif args.norm_area:
        args.normalize_mode = 'area'

    # 判断使用哪种格式
    if args.tpr and args.trj:
        # GROMACS 格式
        if not HAS_GROMACS_SUPPORT:
            print("错误: 需要 MDAnalysis 支持")
            print("安装方法: pip install MDAnalysis")
            return 1
        return run_gromacs_pipeline(args)
    elif args.traj:
        # LAMMPS/Pickle 格式
        traj_path = Path(args.traj)

        if traj_path.suffix.lower() == '.pkl' and HAS_PICKLE_VECTORIZED:
            # Pickle 格式：默认逐帧处理，--vectorized 启用全向量化
            if args.vectorized:
                try:
                    return run_pickle_vectorized_pipeline(args)
                except Exception as e:
                    print(f"\n错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1
            else:
                try:
                    return run_pickle_pipeline(args)
                except Exception as e:
                    print(f"\n错误: {e}")
                    import traceback
                    traceback.print_exc()
                    return 1

        # 传统方法（LAMMPS dump 或 fallback）
        try:
            run_pipeline(args)
            return 0
        except Exception as e:
            print(f"\n错误: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("错误: 需要指定轨迹文件")
        print("  Pickle 格式 (推荐): --traj <轨迹文件.pkl> [--vectorized]")
        print("  LAMMPS dump 格式: --traj <轨迹文件.lammpstrj>")
        print("  GROMACS 格式: --tpr <TPR文件> --xtc/--trr <轨迹文件>")
        return 1


if __name__ == "__main__":
    sys.exit(main())