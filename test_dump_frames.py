#!/usr/bin/env python3
"""read_lammps_dump_frames 的 pytest 测试（trj 子命令 --dump 的底层适配函数）。"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from LmpPy.tools.aa2cg.trj_converter import read_lammps_dump_frames


def _write_dump(path, frames):
    """frames: list of (ids, types, coords(N,3))，原子行按给定顺序写出。"""
    with open(path, "w") as f:
        for t, (ids, types, coords) in enumerate(frames):
            f.write(f"ITEM: TIMESTEP\n{t * 100}\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{len(ids)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n0.0 50.0\n0.0 60.0\n0.0 70.0\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for i in range(len(ids)):
                f.write(f"{ids[i]} {types[i]} "
                        f"{coords[i,0]:.4f} {coords[i,1]:.4f} {coords[i,2]:.4f}\n")


def _frame(n, shuffle=False):
    """生成一帧；坐标由原子 id 决定（coord(id=k) = [10k, 10k+1, 10k+2]）。

    shuffle=True 时行顺序与坐标一起打乱，保持 dump 内部 id→坐标 对应关系。
    """
    ids = np.arange(1, n + 1, dtype=np.int64)
    types = np.ones(n, dtype=np.int32)
    coords = (np.arange(1, n + 1)[:, None] * 10.0
              + np.arange(3)[None, :]).astype(np.float64)
    if shuffle:
        rng = np.random.default_rng(n)
        perm = rng.permutation(n)
        ids = ids[perm]
        coords = coords[perm]
    return ids, types, coords


def test_basic_frames_and_box(tmp_path):
    dump = tmp_path / "traj.lammpstrj"
    frames_data = [_frame(5), _frame(5)]
    _write_dump(dump, frames_data)

    frames = read_lammps_dump_frames(str(dump))
    assert len(frames) == 2
    fr = frames[0]
    assert fr["natoms"] == 5
    assert fr["frame"] == 0
    assert frames[1]["frame"] == 1
    # box 保留 (3,2) 边界格式，与 GROMACS 路径输出一致
    np.testing.assert_allclose(fr["box"], [[0, 50], [0, 60], [0, 70]])
    np.testing.assert_array_equal(fr["ids"], np.arange(1, 6))
    np.testing.assert_allclose(fr["coords"], frames_data[0][2])


def test_shuffled_atom_order_sorted_by_id(tmp_path):
    """dump 原子行乱序时，输出坐标必须按 id 重排（convert 按 AA_id-1 索引）。"""
    dump = tmp_path / "traj.lammpstrj"
    ids, types, coords = _frame(6, shuffle=True)
    _write_dump(dump, [(ids, types, coords)])

    (fr,) = read_lammps_dump_frames(str(dump))
    np.testing.assert_array_equal(fr["ids"], np.arange(1, 7))
    expected = (np.arange(1, 7)[:, None] * 10.0 + np.arange(3)[None, :])
    np.testing.assert_allclose(fr["coords"], expected)


def test_stride(tmp_path):
    dump = tmp_path / "traj.lammpstrj"
    _write_dump(dump, [_frame(3) for _ in range(6)])

    frames = read_lammps_dump_frames(str(dump), stride=2)
    assert len(frames) == 3
    assert [f["frame"] for f in frames] == [0, 1, 2]


def test_non_contiguous_ids_raise(tmp_path):
    dump = tmp_path / "traj.lammpstrj"
    ids = np.array([2, 3, 4], dtype=np.int64)  # 缺 id=1
    _write_dump(dump, [(ids, np.ones(3, dtype=np.int32),
                        np.ones((3, 3)))])

    with pytest.raises(ValueError, match="非连续"):
        read_lammps_dump_frames(str(dump))


def test_triclinic_rejected(tmp_path):
    dump = tmp_path / "traj.lammpstrj"
    with open(dump, "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n")
        f.write("ITEM: BOX BOUNDS xy xz yz pp pp pp\n")
        f.write("0.0 50.0 1.0\n0.0 60.0 2.0\n0.0 70.0 3.0\n")
        f.write("ITEM: ATOMS id type x y z\n1 1 1.0 2.0 3.0\n")

    with pytest.raises(NotImplementedError, match="三斜"):
        read_lammps_dump_frames(str(dump))
