"""竞聚率分析模块测试。"""
import io
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from LmpPy.utils.file_utils import iter_lammps_dump_frames


def _write_tiny_dump(path: Path) -> None:
    """写 2 帧 × 3 原子的微型 dump 文件。"""
    with open(path, "w") as f:
        for ts in (0, 100):
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{ts}\n")
            f.write("ITEM: NUMBER OF ATOMS\n3\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write("0.0 10.0\n0.0 10.0\n0.0 10.0\n")
            f.write("ITEM: ATOMS id type x y z ix iy iz\n")
            f.write(f"1 3 1.0 1.0 1.0 0 0 0\n")
            f.write(f"2 5 2.0 1.0 1.0 0 0 0\n")
            f.write(f"3 6 8.0 8.0 8.0 0 0 0\n")


def test_iter_lammps_dump_frames(tmp_path):
    dump = tmp_path / "tiny.lammpstrj"
    _write_tiny_dump(dump)
    frames = list(iter_lammps_dump_frames(dump))
    assert len(frames) == 2
    f0 = frames[0]
    assert f0["timestep"] == 0
    assert f0["box"].shape == (3, 2)
    assert list(f0["types"]) == [3, 5, 6]
    assert f0["coords"].shape == (3, 3)
    assert frames[1]["timestep"] == 100


def test_iter_lammps_dump_frames_max_frames(tmp_path):
    dump = tmp_path / "tiny.lammpstrj"
    _write_tiny_dump(dump)
    frames = list(iter_lammps_dump_frames(dump, max_frames=1))
    assert len(frames) == 1
