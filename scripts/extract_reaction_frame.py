#!/usr/bin/env python3
"""随机抽取反应帧并输出详细信息。

功能:
1. 读取指定 chunk/md 的 cg_trajectory.lammpstrj 和 reaction_frames.npz
2. 随机抽取一个反应发生帧，输出该帧和前一帧 (lammpstrj 格式，含 unwrap 坐标)
3. 输出 AA 坐标 (反应前后)
4. 按 AtomId_BeadId_compare_list.csv 格式输出这两帧的 cg mapping
5. 打印该反应帧发生的反应 (全原子和 CG 信息，含 bead 反应对距离)
6. 所有输出同时写入 log 文件

用法:
    python LmpPy/scripts/extract_reaction_frame.py <chunk/md 目录> [--seed SEED]

示例:
    python LmpPy/scripts/extract_reaction_frame.py \
        data/LmpPy_original/epr/multi_chunk_md/with_constrain2/chunk1/md1 \
        --output-dir /path/to/output --seed 42
"""

import argparse
import logging
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


def setup_logging(output_dir: Path, prefix: str) -> logging.Logger:
    """设置日志，同时输出到控制台和文件。"""
    log_file = output_dir / f"{prefix}.log"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger(__name__)
    logger.info(f"日志文件: {log_file}")
    return logger


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


def unwrap_coordinates(
    coords: np.ndarray,
    ixyz: Optional[np.ndarray],
    box: np.ndarray,
) -> np.ndarray:
    """将 wrapped 坐标转换为 unwrapped 坐标。

    unwrap_coord = coord + image_flag * box_length

    Args:
        coords: (n_atoms, 3) wrapped 坐标
        ixyz: (n_atoms, 3) image flags，可为 None
        box: (3, 2) 盒子边界 [[xlo, xhi], [ylo, yhi], [zlo, zhi]]

    Returns:
        (n_atoms, 3) unwrapped 坐标
    """
    if ixyz is None:
        return coords.copy()

    box_lengths = box[:, 1] - box[:, 0]
    unwrapped = coords + ixyz * box_lengths
    return unwrapped


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


def write_aa_coords(
    output_path: Path,
    aa_ids: np.ndarray,
    aa_types: np.ndarray,
    aa_coords: np.ndarray,
    timestep: int,
) -> None:
    """输出 AA 坐标到 lammpstrj 格式文件。"""
    # 创建简单的盒子（根据坐标范围）
    margin = 5.0
    xlo, xhi = aa_coords[:, 0].min() - margin, aa_coords[:, 0].max() + margin
    ylo, yhi = aa_coords[:, 1].min() - margin, aa_coords[:, 1].max() + margin
    zlo, zhi = aa_coords[:, 2].min() - margin, aa_coords[:, 2].max() + margin
    box = np.array([[xlo, xhi], [ylo, yhi], [zlo, zhi]])

    write_lammps_dump_file(
        str(output_path),
        timestep,
        box,
        aa_ids.astype(int),
        aa_types.astype(int),
        aa_coords,
    )


def analyze_reactions(
    frame_idx: int,
    reaction_data: Dict,
    cg_mapping_before: np.ndarray,
    cg_mapping_after: np.ndarray,
) -> Tuple[List[Dict], Dict[int, Tuple[int, int]]]:
    """分析反应帧中发生的反应。

    返回:
        reactions: 反应列表
        bead_type_changes: {bead_id: (type_before, type_after)}
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
    bead_type_changes = {}

    for btype, a1, a2 in new_bond_codes:
        bead1 = atom_to_bead.get(a1)
        bead2 = atom_to_bead.get(a2)

        # 记录 bead 类型变化
        if bead1 is not None and bead1 not in bead_type_changes:
            bead_type_changes[bead1] = (
                bead_info.get(bead1, {}).get('type_before'),
                bead_info.get(bead1, {}).get('type_after'),
            )
        if bead2 is not None and bead2 not in bead_type_changes:
            bead_type_changes[bead2] = (
                bead_info.get(bead2, {}).get('type_before'),
                bead_info.get(bead2, {}).get('type_after'),
            )

        reaction = {
            'bond_type': btype,
            'atom1': a1,
            'atom2': a2,
            'bead1': bead1,
            'bead2': bead2,
        }
        reactions.append(reaction)

    return reactions, bead_type_changes


def compute_bead_distance(
    bead1: int,
    bead2: int,
    cg_frame_atoms: np.ndarray,
) -> float:
    """计算两个 CG bead 之间的距离。

    Args:
        bead1: 第一个 bead 的 ID
        bead2: 第二个 bead 的 ID
        cg_frame_atoms: CG 轨迹帧的原子数组，列为 [id, type, x, y, z, ...]

    Returns:
        距离，如果 bead 不存在则返回 -1.0
    """
    # 从 CG 轨迹中找到 bead 坐标 (atom id 是 1-based)
    idx1 = bead1 - 1  # 转换为 0-based 索引
    idx2 = bead2 - 1

    if idx1 < 0 or idx1 >= len(cg_frame_atoms):
        return -1.0
    if idx2 < 0 or idx2 >= len(cg_frame_atoms):
        return -1.0

    # 验证 atom id 是否匹配
    if int(cg_frame_atoms[idx1, 0]) != bead1:
        # 如果不匹配，搜索正确的索引
        mask1 = cg_frame_atoms[:, 0].astype(int) == bead1
        if not mask1.any():
            return -1.0
        idx1 = np.where(mask1)[0][0]

    if int(cg_frame_atoms[idx2, 0]) != bead2:
        mask2 = cg_frame_atoms[:, 0].astype(int) == bead2
        if not mask2.any():
            return -1.0
        idx2 = np.where(mask2)[0][0]

    coord1 = cg_frame_atoms[idx1, 2:5]
    coord2 = cg_frame_atoms[idx2, 2:5]

    return float(np.linalg.norm(coord1 - coord2))


def print_reaction_details(
    logger: logging.Logger,
    frame_idx: int,
    timestep: int,
    reaction_data: Dict,
    reactions: List[Dict],
    bead_type_changes: Dict[int, Tuple[int, int]],
    cg_frame_atoms: np.ndarray,
) -> None:
    """打印反应帧的详细信息。"""
    logger.info(f"\n{'='*70}")
    logger.info(f"反应帧详细信息 (帧索引: {frame_idx}, 时间步: {timestep})")
    logger.info(f"{'='*70}")

    # 基本信息
    aa_coords_after = reaction_data['aa_coords_after'][frame_idx]
    aa_ids = reaction_data['aa_ids']
    if aa_ids.ndim == 2:
        aa_ids = aa_ids[frame_idx]

    logger.info(f"\n总原子数: {len(aa_ids)}")
    logger.info(f"新建 AA 键数: {len(reactions)}")
    logger.info(f"Bead 类型变化数: {len(bead_type_changes)}")

    # 打印每个反应的详细信息
    logger.info(f"\n{'-'*70}")
    logger.info("反应列表 (atom IDs 为 1-based):")
    logger.info(f"{'-'*70}")

    for i, rxn in enumerate(reactions, 1):
        a1, a2 = rxn['atom1'], rxn['atom2']  # 1-based atom IDs
        b1, b2 = rxn['bead1'], rxn['bead2']

        # 获取原子坐标 (反应后)，aa_ids 是 1-based，数组索引是 0-based
        idx1 = a1 - 1  # 转换为 0-based 索引
        idx2 = a2 - 1

        coord1 = aa_coords_after[idx1] if 0 <= idx1 < len(aa_coords_after) else [0, 0, 0]
        coord2 = aa_coords_after[idx2] if 0 <= idx2 < len(aa_coords_after) else [0, 0, 0]

        logger.info(f"\n反应 {i}:")
        logger.info(f"  新 AA 键: {a1} - {a2} (键类型: {rxn['bond_type']})")
        logger.info(f"    原子 {a1}: 坐标 ({coord1[0]:.3f}, {coord1[1]:.3f}, {coord1[2]:.3f})")
        logger.info(f"    原子 {a2}: 坐标 ({coord2[0]:.3f}, {coord2[1]:.3f}, {coord2[2]:.3f})")

        if b1 is not None and b2 is not None:
            # 计算 bead 间距离
            distance = compute_bead_distance(b1, b2, cg_frame_atoms)
            logger.info(f"    CG bead 对: {b1} <-> {b2}")
            logger.info(f"    Bead 反应对距离: {distance:.3f}")
        elif b1 is not None:
            logger.info(f"    -> CG bead {b1}")
        elif b2 is not None:
            logger.info(f"    -> CG bead {b2}")

    # 单独打印 bead 类型变化（去重后）
    if bead_type_changes:
        logger.info(f"\n{'-'*70}")
        logger.info("Bead 类型变化 (去重):")
        logger.info(f"{'-'*70}")
        for bead_id, (t_before, t_after) in sorted(bead_type_changes.items()):
            logger.info(f"  Bead {bead_id}: type {t_before} -> {t_after}")


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
        '--output-dir',
        type=Path,
        default=None,
        help='输出目录（默认为 md_dir）'
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

    # 设置输出目录和日志
    output_dir = args.output_dir if args.output_dir is not None else md_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_prefix
    logger = setup_logging(output_dir, prefix)

    logger.info(f"加载: {md_dir}")

    # 加载 reaction_frames.npz
    logger.info("加载 reaction_frames.npz...")
    reaction_data = load_reaction_frames(npz_path)
    n_frames = len(reaction_data['timestep'])
    logger.info(f"  反应帧数: {n_frames}")

    # 随机选择一个反应帧
    frame_idx = np.random.randint(0, n_frames)
    timestep = int(reaction_data['timestep'][frame_idx])
    logger.info(f"随机选择: 帧索引 {frame_idx}, 时间步 {timestep}")

    # 获取 cg_mapping
    cg_mapping_before = reaction_data['cg_mapping_before'][frame_idx]
    cg_mapping_after = reaction_data['cg_mapping_after'][frame_idx]

    # 从 cg_trajectory.lammpstrj 中找到对应帧
    logger.info(f"\n在 cg_trajectory.lammpstrj 中查找时间步 {timestep}...")
    cg_frame = find_cg_frame_at_timestep(traj_path, timestep)

    if cg_frame is None:
        logger.warning(f"在轨迹中找不到时间步 {timestep} 的帧")
        cg_frame = find_previous_cg_frame(traj_path, timestep)
        if cg_frame is not None:
            logger.info(f"  使用前一帧: 时间步 {cg_frame['timestep']}")

    if cg_frame is None:
        logger.error("无法找到对应的 CG 帧")
        return 1

    # 查找前一帧
    prev_timestep = None
    if frame_idx > 0:
        prev_timestep = int(reaction_data['timestep'][frame_idx - 1])
    prev_frame = find_previous_cg_frame(traj_path, timestep) if prev_timestep is None else find_cg_frame_at_timestep(traj_path, prev_timestep)

    # 获取 CG 坐标并 unwrap
    cg_coords_wrapped = cg_frame['atoms'][:, 2:5]
    cg_ixyz = cg_frame.get('ixyz')
    cg_box = cg_frame['box']
    cg_coords_unwrapped = unwrap_coordinates(cg_coords_wrapped, cg_ixyz, cg_box)

    # 输出当前帧 (反应后) - unwrapped 坐标
    curr_output = output_dir / f"{prefix}_current.lammpstrj"
    logger.info(f"\n写出当前帧 (反应后, unwrapped): {curr_output}")
    write_lammps_dump_file(
        str(curr_output),
        cg_frame['timestep'],
        cg_box,
        cg_frame['atoms'][:, 0].astype(int),
        cg_frame['atoms'][:, 1].astype(int),
        cg_coords_unwrapped,
        ixyz=cg_ixyz,
    )

    # 输出前一帧 (反应前)
    prev_coords_unwrapped = None
    if prev_frame is not None:
        prev_coords_wrapped = prev_frame['atoms'][:, 2:5]
        prev_ixyz = prev_frame.get('ixyz')
        prev_coords_unwrapped = unwrap_coordinates(prev_coords_wrapped, prev_ixyz, prev_frame['box'])

        prev_output = output_dir / f"{prefix}_previous.lammpstrj"
        logger.info(f"写出前一帧 (反应前, unwrapped): {prev_output}")
        write_lammps_dump_file(
            str(prev_output),
            prev_frame['timestep'],
            prev_frame['box'],
            prev_frame['atoms'][:, 0].astype(int),
            prev_frame['atoms'][:, 1].astype(int),
            prev_coords_unwrapped,
            ixyz=prev_ixyz,
        )

    # 输出 AA 坐标 (反应前后)
    aa_ids = reaction_data['aa_ids']
    if aa_ids.ndim == 2:
        aa_ids_frame = aa_ids[frame_idx]
    else:
        aa_ids_frame = aa_ids

    aa_types = reaction_data['aa_types']
    if aa_types.ndim == 2:
        aa_types_frame = aa_types[frame_idx]
    else:
        aa_types_frame = aa_types

    aa_coords_before = reaction_data['aa_coords_before'][frame_idx]
    aa_coords_after = reaction_data['aa_coords_after'][frame_idx]

    aa_before_output = output_dir / f"{prefix}_aa_before.lammpstrj"
    aa_after_output = output_dir / f"{prefix}_aa_after.lammpstrj"

    logger.info(f"\n写出 AA 坐标 (反应前): {aa_before_output}")
    write_aa_coords(aa_before_output, aa_ids_frame, aa_types_frame, aa_coords_before, timestep)

    logger.info(f"写出 AA 坐标 (反应后): {aa_after_output}")
    write_aa_coords(aa_after_output, aa_ids_frame, aa_types_frame, aa_coords_after, timestep)

    # 输出 cg mapping CSV
    logger.info(f"\n写出 CG mapping:")
    write_cg_mapping_csv(
        output_dir / f"{prefix}_mapping_before.csv",
        cg_mapping_before,
    )
    logger.info(f"  已写出: {output_dir / f'{prefix}_mapping_before.csv'}")
    write_cg_mapping_csv(
        output_dir / f"{prefix}_mapping_after.csv",
        cg_mapping_after,
    )
    logger.info(f"  已写出: {output_dir / f'{prefix}_mapping_after.csv'}")

    # 分析反应
    reactions, bead_type_changes = analyze_reactions(
        frame_idx,
        reaction_data,
        cg_mapping_before,
        cg_mapping_after,
    )

    # 打印反应详细信息
    print_reaction_details(
        logger,
        frame_idx,
        timestep,
        reaction_data,
        reactions,
        bead_type_changes,
        cg_frame['atoms'],
    )

    logger.info(f"\n{'='*70}")
    logger.info("完成!")
    logger.info(f"{'='*70}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
