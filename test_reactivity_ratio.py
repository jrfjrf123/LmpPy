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


from LmpPy.output_analysis.reactivity_ratio import (
    CHANNELS,
    channel_of_pair,
    count_events,
)


def test_channel_of_pair():
    assert channel_of_pair(3, 5) == "11"
    assert channel_of_pair(5, 3) == "11"  # 无序
    assert channel_of_pair(3, 6) == "12"
    assert channel_of_pair(4, 5) == "21"
    assert channel_of_pair(4, 6) == "22"
    assert channel_of_pair(1, 2) is None  # 非反应通道


def test_count_events_zero_fill():
    df = pd.DataFrame({
        "cycle": [1, 3, 3],
        "atom1_type_before": [3, 4, 3],
        "atom2_type_before": [5, 6, 6],
    })
    out = count_events(df)
    assert list(out["cycle"]) == [1, 2, 3]
    assert int(out.loc[out["cycle"] == 1, "N11"].iloc[0]) == 1
    assert int(out.loc[out["cycle"] == 2, ["N11", "N12", "N21", "N22"]].sum().sum()) == 0
    assert int(out.loc[out["cycle"] == 3, "N12"].iloc[0]) == 1
    assert int(out.loc[out["cycle"] == 3, "N22"].iloc[0]) == 1


from LmpPy.output_analysis.reactivity_ratio import (
    count_candidates_frame,
    count_types_frame,
)


def test_count_types_frame():
    types = np.array([1, 1, 3, 5, 5, 5, 6])
    out = count_types_frame(types)
    assert out["n1"] == 2
    assert out["n3"] == 1
    assert out["n5"] == 3
    assert out["n6"] == 1
    assert out["n2"] == 0
    assert out["n4"] == 0


def test_count_candidates_frame_pbc():
    # 10 Å 立方盒子; 1 个 type3 末端, 两个 type5 单体(一个跨 PBC 边界), 一个 type6 远处
    types = np.array([3, 5, 5, 6])
    coords = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],   # 距末端 1.0 Å
        [9.5, 1.0, 1.0],   # PBC 最小镜像下距末端 1.5 Å
        [6.0, 6.0, 6.0],   # 远
    ])
    box_lengths = np.array([10.0, 10.0, 10.0])
    out = count_candidates_frame(types, coords, box_lengths, cutoff=1.2)
    assert out["E35"] == 1  # 只有 1.0 Å 的那对
    assert out["E36"] == 0
    out2 = count_candidates_frame(types, coords, box_lengths, cutoff=2.0)
    assert out2["E35"] == 2  # PBC 对也计入
