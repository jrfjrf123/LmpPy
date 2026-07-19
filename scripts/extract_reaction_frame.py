#!/usr/bin/env python3
"""随机抽取反应帧并输出详细信息。

功能:
1. 读取指定 chunk/md 的 cg_trajectory.lammpstrj 和 reaction_frames.npz
2. 随机抽取一个反应发生帧，输出该帧和前一帧 (lammpstrj 格式)
3. 按 AtomId_BeadId_compare_list.csv 格式输出这两帧的 cg mapping
4. 打印该反应帧发生的反应 (全原子和 CG 信息)

用法:
    python LmpPy/scripts/extract_reaction_frame.py <chunk/md 目录> [--seed SEED]

示例:
    python LmpPy/scripts/extract_reaction_frame.py \
        data/LmpPy_original/epr/multi_chunk_md/with_constrain2/chunk1/md1
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录到路径
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.utils.file_utils import write_lammps_dump_file
from SOAP_calc_and_Feature_select.utils import iter_lammpstrj_frames


def load_reaction_frames(npz_path: Path) -> Dict:
    """加载 reaction_frames.npz。"""
    data = np.load(npz_path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def find_cg_frame_at_timestep(
    traj_path: Path, target_ts: int
) -> Optional[Dict]:
    """在 cg_trajectory.lammpstrj 中查找指定时间步的帧。"""
    for frame in iter_lammpstrj_frames(str(traj_path)):
        if frame['timestep'] == target_ts:
            return frame
    return None


def find_previous_cg_frame(
    traj_path: Path, target_ts: int
) -> Optional[Dict]:
    """在 cg_trajectory.lammpstrj 中查找指定时间步之前的最近一帧。"""
    prev_frame = None
    for frame in iter_lammpstrj_frames(str(traj_path)):
        if frame['timestep'] >= target_ts:
            break
        prev_frame = frame
    return prev_frame


def write_cg_mapping_csv(
    output_path: Path,
    cg_mapping: np.ndarray,
) -> None:
    """按 AtomId_BeadId_compare_list.csv 格式输出 cg mapping。"""
    df = pd.DataFrame({
        'bead_id': cg_mapping[:, 0].astype(int),
        'mol_id': cg_mapping[:, 1].astype(int),
        'bead_type': cg_mapping[:, 2].astype(int),
        'AA_id': cg_mapping[:, 3].astype(int),
        'mass': cg_mapping[:, 4],
    })
    df.to_csv(output_path, index=False)
    print(f"  已写出: {output_path}")


def analyze_reactions(
    frame_idx: int,
    reaction_data: Dict,
    atom_types: np.ndarray,
    cg_mapping_before: np.ndarray,
    cg_mapping_after: np.ndarray,
) -> List[Dict]:
    """分析反应帧中发生的反应。

    返回:
        反应列表，每个反应包含:
        - new_bonds: 新建 AA 键
        - bead_type_changes: bead 类型变化
        - atom_info: 涉及原子的详细信息
    """
    aa_bonds_before = reaction_data['aa_bonds_before'][frame_idx]
    aa_bonds_after = reaction_data['aa_bonds_after'][frame_idx]

    # 编码键为集合进行比较
    def encode_bonds(bonds):
        codes = set()
        for b in bonds:
            a1, a2 = min(int(b[1]), int(b[2])), max(int(b[1]), int(b[2]))
            codes.add((int(b[0]), a1, a2))
        return codes

    before_codes = encode_bonds(aa_bonds_before)
    after_codes = encode_bonds(aa_bonds_after)

    new_bond_codes = after_codes - before_codes
    deleted_bond_codes = before_codes - after_codes

    # 构建 atom_id -> bead_id 映射
    atom_to_bead = {}
    bead_info = {}
    for row in cg_mapping_before:
        bead_id = int(row[0])
        aa_id = int(row[3])
        if bead_id > 0:
            atom_to_bead[aa_id] = bead_id
            if bead_id not in bead_info:
                bead_info[bead_id] = {
                    'mol_id': int(row[1]),
                    'type_before': int(row[2]),
                    'type_after': int(row[2]),
                }

    # 更新 bead 类型 (after)
    for row in cg_mapping_after:
        bead_id = int(row[0])
        if bead_id > 0 and bead_id in bead_info:
            bead_info[bead_id]['type_after'] = int(row[2])

    # 分析新键
    reactions = []
    for btype, a1, a2 in new_bond_codes:
        bead1 = atom_to_bead.get(a1)
        bead2 = atom_to_bead.get(a2)

        reaction = {
            'bond_type': btype,
            'atom1': a1,
            'atom2': a2,
            'bead1': bead1,
            'bead2': bead2,
            'bead1_type_before': bead_info.get(bead1, {}).get('type_before'),
            'bead1_type_after': bead_info.get(bead1, {}).get('type_after'),
            'bead2_type_before': bead_info.get(bead2, {}).get('type_before'),
            'bead2_type_after': bead_info.get(bead2, {}).get('type_after'),
        }
        reactions.append(reaction)

    return reactions


def print_reaction_details(
    frame_idx: int,
    timestep: int,
    reaction_data: Dict,
    reactions: List[Dict],
    cg_mapping_before: np.ndarray,
    cg_mapping_after: np.ndarray,
) -> None:
    """打印反应帧的详细信息。"""
    print(f"\n{'='*70}")
    print(f"反应帧详细信息 (帧索引: {frame_idx}, 时间步: {timestep})")
    print(f"{'='*70}")

    # 基本信息
    aa_coords_before = reaction_data['aa_coords_before'][frame_idx]
    aa_coords_after = reaction_data['aa_coords_after'][frame_idx]
    aa_ids = reaction_data['aa_ids']
    if aa_ids.ndim == 2:
        aa_ids = aa_ids[frame_idx]

    print(f"\n总原子数: {len(aa_ids)}")
    print(f"新建 AA 键数: {len(reactions)}")

    # 统计 bead 类型变化
    type_changes = 0
    for i in range(len(cg_mapping_before)):
        if cg_mapping_before[i, 2] != cg_mapping_after[i, 2]:
            type_changes += 1
    print(f"Bead 类型变化数: {type_changes}")

    # 打印每个反应的详细信息
    print(f"\n{'-'*70}")
    print("反应列表:")
    print(f"{'-'*70}")

    for i, rxn in enumerate(reactions, 1):
        a1, a2 = rxn['atom1'], rxn['atom2']
        b1, b2 = rxn['bead1'], rxn['bead2']

        # 获取原子坐标 (反应后)
        idx1 = np.where(aa_ids == a1)[0]
        idx2 = np.where(aa_ids == a2)[0]

        coord1 = aa_coords_after[idx1[0]] if len(idx1) > 0 else [0, 0, 0]
        coord2 = aa_coords_after[idx2[0]] if len(idx2) > 0 else [0, 0, 0]

        print(f"\n反应 {i}:")
        print(f"  新 AA 键: {a1} - {a2} (键类型: {rxn['bond_type']})")
        print(f"    原子 {a1}: 坐标 ({coord1[0]:.3f}, {coord1[1]:.3f}, {coord1[2]:.3f})")
        print(f"    原子 {a2}: 坐标 ({coord2[0]:.3f}, {coord2[1]:.3f}, {coord2[2]:.3f})")

        if b1 is not None:
            print(f"    CG bead {b1}: type {rxn['bead1_type_before']} -> {rxn['bead1_type_after']}")
        if b2 is not None:
            print(f"    CG bead {b2}: type {rxn['bead2_type_before']} -> {rxn['bead2_type_after']}")


def main():
    parser = argparse.ArgumentParser(
        description='随机抽取反应帧并输出详细信息'
    )
    parser.add_argument(
        'md_dir',
        type=Path,
        help='包含 cg_trajectory.lammpstrj 和 reaction_frames.npz 的目录'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='随机种子（用于可重复的随机选择）'
    )
    parser.add_argument(
        '--output-prefix',
        type=str,
        default='reaction_frame',
        help='输出文件前缀'
    )

    args = parser.parse_args()
    md_dir = args.md_dir

    if args.seed is not None:
        np.random.seed(args.seed)

    # 检查文件
    traj_path = md_dir / 'cg_trajectory.lammpstrj'
    npz_path = md_dir / 'reaction_frames.npz'

    if not traj_path.exists():
        print(f"错误: 找不到 {traj_path}")
        return 1
    if not npz_path.exists():
        print(f"错误: 找不到 {npz_path}")
        return 1

    print(f"加载: {md_dir}")

    # 加载 reaction_frames.npz
    print("加载 reaction_frames.npz...")
    reaction_data = load_reaction_frames(npz_path)
    n_frames = len(reaction_data['timestep'])
    print(f"  反应帧数: {n_frames}")

    # 随机选择一个反应帧
    frame_idx = np.random.randint(0, n_frames)
    timestep = int(reaction_data['timestep'][frame_idx])
    print(f"随机选择: 帧索引 {frame_idx}, 时间步 {timestep}")

    # 获取 cg_mapping
    cg_mapping_before = reaction_data['cg_mapping_before'][frame_idx]
    cg_mapping_after = reaction_data['cg_mapping_after'][frame_idx]

    # 从 cg_trajectory.lammpstrj 中找到对应帧
    print(f"\n在 cg_trajectory.lammpstrj 中查找时间步 {timestep}...")
    cg_frame = find_cg_frame_at_timestep(traj_path, timestep)

    if cg_frame is None:
        print(f"警告: 在轨迹中找不到时间步 {timestep} 的帧")
        # 尝试找前一帧
        cg_frame = find_previous_cg_frame(traj_path, timestep)
        if cg_frame is not None:
            print(f"  使用前一帧: 时间步 {cg_frame['timestep']}")

    if cg_frame is None:
        print("错误: 无法找到对应的 CG 帧")
        return 1

    # 查找前一帧
    prev_timestep = None
    if frame_idx > 0:
        prev_timestep = int(reaction_data['timestep'][frame_idx - 1])
    prev_frame = find_previous_cg_frame(traj_path, timestep) if prev_timestep is None else find_cg_frame_at_timestep(traj_path, prev_timestep)

    # 输出 lammpstrj 格式
    output_dir = md_dir
    prefix = args.output_prefix

    # 当前帧 (反应后)
    curr_output = output_dir / f"{prefix}_current.lammpstrj"
    print(f"\n写出当前帧 (反应后): {curr_output}")
    write_lammps_dump_file(
        str(curr_output),
        cg_frame['timestep'],
        cg_frame['box'],
        cg_frame['atoms'][:, 0].astype(int),
        cg_frame['atoms'][:, 1].astype(int),
        cg_frame['atoms'][:, 2:5],
        ixyz=cg_frame.get('ixyz'),
    )

    # 前一帧 (反应前)
    if prev_frame is not None:
        prev_output = output_dir / f"{prefix}_previous.lammpstrj"
        print(f"写出前一帧 (反应前): {prev_output}")
        write_lammps_dump_file(
            str(prev_output),
            prev_frame['timestep'],
            prev_frame['box'],
            prev_frame['atoms'][:, 0].astype(int),
            prev_frame['atoms'][:, 1].astype(int),
            prev_frame['atoms'][:, 2:5],
            ixyz=prev_frame.get('ixyz'),
        )

    # 输出 cg mapping CSV
    print(f"\n写出 CG mapping:")
    write_cg_mapping_csv(
        output_dir / f"{prefix}_mapping_before.csv",
        cg_mapping_before,
    )
    write_cg_mapping_csv(
        output_dir / f"{prefix}_mapping_after.csv",
        cg_mapping_after,
    )

    # 分析并打印反应
    reactions = analyze_reactions(
        frame_idx,
        reaction_data,
        reaction_data['aa_types'],
        cg_mapping_before,
        cg_mapping_after,
    )

    print_reaction_details(
        frame_idx,
        timestep,
        reaction_data,
        reactions,
        cg_mapping_before,
        cg_mapping_after,
    )

    print(f"\n{'='*70}")
    print("完成!")
    print(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
