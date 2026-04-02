"""
文件I/O工具函数

功能:
- write_lammps_dump_file: 写入LAMMPS轨迹文件
- write_cg_trajectory: 写入CG轨迹文件

作者: Claude
日期: 2026-03-26
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional


def write_lammps_dump_file(path: str, timestep: int, box: np.ndarray,
                           id_arr: np.ndarray, atype_arr: np.ndarray,
                           coord_arr: np.ndarray,
                           ixyz: Optional[np.ndarray] = None,
                           append: bool = True) -> None:
    """
    写入LAMMPS dump格式的轨迹文件

    参数:
        path: 文件路径
        timestep: 时间步
        box: 盒子边界，shape=(3, 2)
        id_arr: 原子ID数组
        atype_arr: 原子类型数组
        coord_arr: 坐标数组，shape=(n_atoms, 3)
        ixyz: image flags数组，shape=(n_atoms, 3) (可选)
        append: 是否追加模式
    """
    id_arr = id_arr.astype(int)
    atype_arr = atype_arr.astype(int)

    # 确定打开模式
    if append and os.path.exists(path):
        open_mode = "a"
    else:
        open_mode = "w"

    with open(path, open_mode) as f:
        # 写入头部
        f.write("ITEM: TIMESTEP\n")
        f.write(f"{int(timestep)}\n")
        f.write("ITEM: NUMBER OF ATOMS\n")
        f.write(f"{int(len(id_arr))}\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n")

        for i in range(3):
            f.write(f"{box[i, 0]:.6f} {box[i, 1]:.6f}\n")

        # 写入原子数据
        if ixyz is not None:
            ixyz = ixyz.astype(int)
            f.write("ITEM: ATOMS id type x y z ix iy iz\n")
            for i in range(len(id_arr)):
                f.write(f"{id_arr[i]:>5} {atype_arr[i]:>5} "
                        f"{coord_arr[i, 0]:>8.3f} {coord_arr[i, 1]:>8.3f} {coord_arr[i, 2]:>8.3f} "
                        f"{ixyz[i, 0]:>5} {ixyz[i, 1]:>5} {ixyz[i, 2]:>5}\n")
        else:
            f.write("ITEM: ATOMS id type x y z\n")
            for i in range(len(id_arr)):
                f.write(f"{id_arr[i]:>5} {atype_arr[i]:>5} "
                        f"{coord_arr[i, 0]:>8.3f} {coord_arr[i, 1]:>8.3f} {coord_arr[i, 2]:>8.3f}\n")


def write_cg_trajectory(path: str, timestep: int, box: np.ndarray,
                        bead_ids: np.ndarray, bead_types: np.ndarray,
                        bead_coords: np.ndarray, append: bool = True) -> None:
    """
    写入CG轨迹文件

    参数:
        path: 文件路径
        timestep: 时间步
        box: 盒子边界，shape=(3, 2)
        bead_ids: 珠子ID数组
        bead_types: 珠子类型数组
        bead_coords: 珠子坐标数组，shape=(n_beads, 3)
        append: 是否追加模式
    """
    write_lammps_dump_file(
        path, timestep, box,
        bead_ids, bead_types, bead_coords,
        ixyz=None, append=append
    )


def read_lammps_dump_file(path: str, max_frames: Optional[int] = None) -> list:
    """
    读取LAMMPS dump格式的轨迹文件

    参数:
        path: 文件路径
        max_frames: 最大读取帧数 (None表示读取所有)

    返回:
        frames: 帧列表，每个元素为字典
                {'timestep': int, 'box': np.ndarray, 'ids': np.ndarray,
                 'types': np.ndarray, 'coords': np.ndarray}
    """
    frames = []

    with open(path, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        if max_frames and len(frames) >= max_frames:
            break

        line = lines[i].strip()

        if line == "ITEM: TIMESTEP":
            timestep = int(lines[i + 1].strip())
            i += 2

            # 读取原子数
            n_atoms = int(lines[i + 1].strip())
            i += 3  # 跳过 "ITEM: NUMBER OF ATOMS" 和 原子数 和 "ITEM: BOX BOUNDS"

            # 读取盒子
            box = np.zeros((3, 2), dtype=np.float64)
            for j in range(3):
                parts = lines[i + j].strip().split()
                box[j] = [float(parts[0]), float(parts[1])]
            i += 3

            # 跳过 "ITEM: ATOMS ..." 行
            i += 1

            # 读取原子数据
            ids = np.zeros(n_atoms, dtype=np.int32)
            types = np.zeros(n_atoms, dtype=np.int32)
            coords = np.zeros((n_atoms, 3), dtype=np.float64)

            for j in range(n_atoms):
                parts = lines[i + j].strip().split()
                ids[j] = int(parts[0])
                types[j] = int(parts[1])
                coords[j] = [float(parts[2]), float(parts[3]), float(parts[4])]

            i += n_atoms

            frames.append({
                'timestep': timestep,
                'box': box,
                'ids': ids,
                'types': types,
                'coords': coords
            })
        else:
            i += 1

    return frames


if __name__ == "__main__":
    import tempfile

    print("=" * 60)
    print("测试文件I/O工具函数")
    print("=" * 60)

    # 创建测试数据
    box = np.array([[0, 50], [0, 50], [0, 50]], dtype=np.float64)
    ids = np.array([1, 2, 3, 4, 5])
    types = np.array([1, 1, 2, 2, 3])
    coords = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0]
    ])

    # 测试写入
    with tempfile.NamedTemporaryFile(suffix='.lammpstrj', delete=False) as tmp:
        tmp_path = tmp.name

    try:
        print("\n测试写入LAMMPS dump文件:")
        write_lammps_dump_file(tmp_path, 0, box, ids, types, coords, append=False)
        write_lammps_dump_file(tmp_path, 100, box, ids, types, coords, append=True)
        print(f"  文件写入成功: {tmp_path}")

        # 测试读取
        print("\n测试读取LAMMPS dump文件:")
        frames = read_lammps_dump_file(tmp_path)
        print(f"  读取帧数: {len(frames)}")
        print(f"  第一帧时间步: {frames[0]['timestep']}")
        print(f"  原子数: {len(frames[0]['ids'])}")

        print("\n✅ 测试完成!")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)