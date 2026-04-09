"""
轨迹转换器 - AA轨迹到CG轨迹的转换

支持格式:
- 输入: GROMACS TPR/TRR, LAMMPS dump
- 输出: pickle格式, LAMMPS dump格式, XYZ格式

用法:
    # 作为模块运行（推荐）
    python -m LmpPy.tools.aa2cg.trj_converter <tpr_file> <trr_file> <mapping_csv> [output_pickle]

    # 直接运行（需要正确设置 PYTHONPATH）
    python trj_converter.py <tpr_file> <trr_file> <mapping_csv> [output_pickle]

作者: 整合自 md_base_on_ml/AA_trj2CG_trj/AA_trj2CG_trj.py
"""

import sys
from pathlib import Path

# 支持直接运行脚本
if __name__ == "__main__":
    _project_root = Path(__file__).resolve().parent.parent.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

import numpy as np
import pickle
from typing import Dict, List, Optional, Tuple

# 尝试导入MDAnalysis（可选依赖）
try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

from LmpPy.tools.aa2cg.mapping_utils import (
    load_aa_to_cg_mapping,
    convert_aa_to_cg_frame,
    wrap_coords
)


def read_lammps_dump(dump_file: str, frame_stride: int = 1) -> Dict:
    """
    读取LAMMPS dump轨迹文件。

    Args:
        dump_file: dump文件路径
        frame_stride: 帧间隔

    Returns:
        dict with keys:
            - 'R': {frame_idx: coords}
            - 'cell': {frame_idx: box}
    """
    frames = {'R': {}, 'cell': {}}
    frame_idx = 0
    actual_frame = 0

    with open(dump_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            if 'ITEM: TIMESTEP' in line:
                # 检查是否跳过该帧
                if actual_frame % frame_stride != 0:
                    # 跳过该帧
                    actual_frame += 1
                    # 跳过该帧的所有数据
                    for _ in range(8):  # 大约需要跳过的行数
                        f.readline()
                    # 跳过原子数据
                    line = f.readline()
                    if 'ITEM: NUMBER OF ATOMS' in line:
                        natoms = int(f.readline())
                        for _ in range(natoms + 4):  # 原子数据 + box行
                            f.readline()
                    continue

                timestep = int(f.readline())

                # 读取原子数
                f.readline()  # ITEM: NUMBER OF ATOMS
                natoms = int(f.readline())

                # 读取盒子
                f.readline()  # ITEM: BOX BOUNDS
                box = np.zeros((3, 2))
                for i in range(3):
                    parts = f.readline().split()
                    box[i] = [float(parts[0]), float(parts[1])]

                # 读取原子数据
                f.readline()  # ITEM: ATOMS
                coords = np.zeros((natoms, 3))
                ids = np.zeros(natoms, dtype=int)
                types = np.zeros(natoms, dtype=int)

                for i in range(natoms):
                    parts = f.readline().split()
                    ids[i] = int(parts[0])
                    types[i] = int(parts[1])
                    coords[i] = [float(parts[2]), float(parts[3]), float(parts[4])]

                # 按ID排序
                sort_idx = np.argsort(ids - 1)
                coords = coords[sort_idx]

                frames['R'][frame_idx] = coords
                frames['cell'][frame_idx] = box[:, 1] - box[:, 0]

                frame_idx += 1
                actual_frame += 1

    return frames


def read_gromacs_trr(tpr_file: str, trr_file: str, frame_idx: int = 0,
                     make_whole: bool = False) -> Dict:
    """
    读取GROMACS TRR轨迹文件的单帧。

    Args:
        tpr_file: TPR拓扑文件路径
        trr_file: TRR轨迹文件路径
        frame_idx: 帧索引
        make_whole: 是否解缠分子

    Returns:
        dict with atom data
    """
    if not HAS_MDA:
        raise ImportError("需要安装MDAnalysis: pip install MDAnalysis")

    u = mda.Universe(tpr_file, trr_file)

    if frame_idx >= len(u.trajectory):
        raise ValueError(f"帧索引 {frame_idx} 超出范围")

    u.trajectory[frame_idx]

    atoms = u.atoms
    ids = atoms.ids.copy()
    types = atoms.types.copy()

    try:
        types = np.array([int(t) for t in types], dtype=np.int32)
    except (ValueError, TypeError):
        types = np.array(types, dtype=str)

    coords = atoms.positions.copy()
    box_dimensions = u.dimensions

    box = np.array([
        [0.0, box_dimensions[0]],
        [0.0, box_dimensions[1]],
        [0.0, box_dimensions[2]]
    ], dtype=np.float64)

    mol_ids = atoms.resids.copy() if hasattr(atoms, 'resids') else np.ones(len(atoms), dtype=np.int32)
    time = u.trajectory.time

    return {
        'ids': ids,
        'mol_ids': mol_ids,
        'types': types,
        'coords': coords,
        'box': box,
        'natoms': len(atoms),
        'time': time
    }


def read_gromacs_trr_all_frames(tpr_file: str, trr_file: str,
                                make_whole: bool = False,
                                stride: int = 1) -> List[Dict]:
    """
    读取GROMACS TRR轨迹的所有帧。

    Args:
        tpr_file: TPR拓扑文件路径
        trr_file: TRR轨迹文件路径
        make_whole: 是否解缠分子
        stride: 帧间隔

    Returns:
        帧数据列表
    """
    if not HAS_MDA:
        raise ImportError("需要安装MDAnalysis: pip install MDAnalysis")

    u = mda.Universe(tpr_file, trr_file)

    frames_list = []
    total_frames = len(u.trajectory)

    print(f"读取 {total_frames} 帧 (stride={stride})...")

    for frame_idx, ts in enumerate(u.trajectory[::stride]):
        atoms = u.atoms
        ids = atoms.ids.copy()
        types = atoms.types.copy()

        try:
            types = np.array([int(t) for t in types], dtype=np.int32)
        except (ValueError, TypeError):
            types = np.array(types, dtype=str)

        coords = atoms.positions.copy()
        box_dimensions = u.dimensions
        box = np.array([
            [0.0, box_dimensions[0]],
            [0.0, box_dimensions[1]],
            [0.0, box_dimensions[2]]
        ], dtype=np.float64)

        mol_ids = atoms.resids.copy() if hasattr(atoms, 'resids') else np.ones(len(atoms), dtype=np.int32)

        data = {
            'ids': ids,
            'mol_ids': mol_ids,
            'types': types,
            'coords': coords,
            'box': box,
            'natoms': len(atoms),
            'time': ts.time,
            'frame': frame_idx
        }

        frames_list.append(data)

        if (frame_idx + 1) % 100 == 0:
            print(f"  已处理 {frame_idx + 1} 帧")

    print(f"完成！共读取 {len(frames_list)} 帧")
    return frames_list


def convert_trajectory_to_cg(aa_frames: List[Dict], mapping_csv: str) -> List[Dict]:
    """
    将AA轨迹转换为CG轨迹。

    Args:
        aa_frames: AA帧数据列表
        mapping_csv: CG映射CSV文件路径

    Returns:
        CG帧数据列表
    """
    print(f"加载CG映射: {mapping_csv}")
    mapping_dict = load_aa_to_cg_mapping(mapping_csv)
    print(f"  CG beads数量: {len(mapping_dict)}")

    cg_trajectory = []

    for frame_idx, aa_frame in enumerate(aa_frames):
        cg_frame = convert_aa_to_cg_frame(
            aa_coords=aa_frame['coords'],
            aa_ids=aa_frame['ids'],
            mapping_dict=mapping_dict,
            box=aa_frame['box']
        )

        # 添加时间和帧信息
        if 'time' in aa_frame:
            cg_frame['time'] = aa_frame['time']
        if 'frame' in aa_frame:
            cg_frame['frame'] = aa_frame['frame']
        else:
            cg_frame['frame'] = frame_idx

        cg_trajectory.append(cg_frame)

        if (frame_idx + 1) % 100 == 0:
            print(f"  已转换 {frame_idx + 1}/{len(aa_frames)} 帧")

    return cg_trajectory


def save_cg_trajectory_pickle(cg_trajectory: List[Dict], output_file: str):
    """
    保存CG轨迹到pickle文件。

    Args:
        cg_trajectory: CG帧数据列表
        output_file: 输出文件路径
    """
    with open(output_file, 'wb') as f:
        pickle.dump(cg_trajectory, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"CG轨迹已保存: {output_file}")
    print(f"  总帧数: {len(cg_trajectory)}")
    print(f"  每帧beads: {cg_trajectory[0]['natoms']}")


def load_cg_trajectory_pickle(pickle_file: str) -> List[Dict]:
    """
    从pickle文件加载CG轨迹。

    Args:
        pickle_file: pickle文件路径

    Returns:
        CG帧数据列表
    """
    with open(pickle_file, 'rb') as f:
        cg_trajectory = pickle.load(f)
    return cg_trajectory


def write_frame_to_xyz(data_dict: Dict, output_file: str, comment: str = ""):
    """
    将单帧写入XYZ格式文件。

    Args:
        data_dict: 帧数据字典
        output_file: 输出文件路径
        comment: 注释行
    """
    natoms = data_dict['natoms']
    coords = data_dict['coords']
    types = data_dict['types']

    # 生成注释行
    if not comment:
        comment_parts = []
        if 'time' in data_dict:
            comment_parts.append(f"time={data_dict['time']:.4f}")
        if 'box' in data_dict:
            box = data_dict['box']
            if box.ndim == 2:
                box_str = f"box=[{box[0,1]-box[0,0]:.3f}, {box[1,1]-box[1,0]:.3f}, {box[2,1]-box[2,0]:.3f}]"
            else:
                box_str = f"box=[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}]"
            comment_parts.append(box_str)
        if 'frame' in data_dict:
            comment_parts.append(f"frame={data_dict['frame']}")
        comment = " ".join(comment_parts) if comment_parts else "Generated by LmpPy"

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

    print(f"写入 {natoms} 原子到 {output_file}")


def write_trajectory_to_xyz(trajectory_data: List[Dict], output_file: str,
                            frame_indices: Optional[List[int]] = None):
    """
    将多帧写入XYZ轨迹文件。

    Args:
        trajectory_data: 帧数据列表
        output_file: 输出文件路径
        frame_indices: 要写入的帧索引列表（默认全部）
    """
    if frame_indices is None:
        frame_indices = range(len(trajectory_data))

    with open(output_file, 'w') as f:
        for idx in frame_indices:
            if idx >= len(trajectory_data):
                print(f"警告: 帧索引 {idx} 超出范围，跳过")
                continue

            frame = trajectory_data[idx]
            natoms = frame['natoms']
            coords = frame['coords']
            types = frame['types']

            # 生成注释行
            comment_parts = []
            if 'time' in frame:
                comment_parts.append(f"time={frame['time']:.4f}")
            if 'frame' in frame:
                comment_parts.append(f"frame={frame['frame']}")
            else:
                comment_parts.append(f"frame={idx}")
            if 'box' in frame:
                box = frame['box']
                if box.ndim == 2:
                    box_str = f"box=[{box[0,1]-box[0,0]:.3f}, {box[1,1]-box[1,0]:.3f}, {box[2,1]-box[2,0]:.3f}]"
                else:
                    box_str = f"box=[{box[0]:.3f}, {box[1]:.3f}, {box[2]:.3f}]"
                comment_parts.append(box_str)
            comment = " ".join(comment_parts)

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

    print(f"写入 {len(list(frame_indices))} 帧到 {output_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 4:
        print("用法: python trj_converter.py <tpr_file> <trr_file> <mapping_csv> [output_pickle]")
        sys.exit(1)

    tpr_file = sys.argv[1]
    trr_file = sys.argv[2]
    mapping_csv = sys.argv[3]
    output_pickle = sys.argv[4] if len(sys.argv) > 4 else "cg_trajectory.pkl"

    # 读取AA轨迹
    aa_frames = read_gromacs_trr_all_frames(tpr_file, trr_file)

    # 转换为CG
    cg_trajectory = convert_trajectory_to_cg(aa_frames, mapping_csv)

    # 保存
    save_cg_trajectory_pickle(cg_trajectory, output_pickle)

    # 验证
    print("\n验证保存的轨迹...")
    cg_traj = load_cg_trajectory_pickle(output_pickle)
    print(f"加载 {len(cg_traj)} 帧")
    print(f"第一帧:")
    print(f"  时间: {cg_traj[0].get('time', 'N/A')} ps")
    print(f"  Beads: {cg_traj[0]['natoms']}")