"""竞聚率分析模块测试。"""
import io
import json
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


from LmpPy.output_analysis.reactivity_ratio import (
    block_bootstrap,
    mayo_lewis_fit,
    point_estimates,
    window_estimates,
)


def _synthetic_counts(n_cycles=4000, r1=2.0, seed=0):
    """注入已知 r1 的合成 per-cycle 计数表（等浓度、等暴露）。

    单体随 cycle 线性消耗,保证转化率 X 有跨度（窗口按转化率划分）。
    """
    rng = np.random.default_rng(seed)
    E = 200  # 每 cycle 每通道候选对数
    p12 = 0.02
    p11 = r1 * p12
    cycle = np.arange(1, n_cycles + 1)
    n5 = np.round(50000 * (1 - 0.4 * cycle / n_cycles)).astype(int)
    n6 = np.round(50000 * (1 - 0.4 * cycle / n_cycles)).astype(int)
    return pd.DataFrame({
        "N11": rng.binomial(E, p11, n_cycles),
        "N12": rng.binomial(E, p12, n_cycles),
        "N21": rng.binomial(E, 0.03, n_cycles),
        "N22": rng.binomial(E, 0.03, n_cycles),
        "E35": E, "E36": E, "E45": E, "E46": E,
        "n5": n5, "n6": n6,
    })


def test_point_estimates_recovers_known_ratio():
    df = _synthetic_counts(r1=2.0)
    est = point_estimates(df)
    assert est["r1_cand"] == pytest.approx(2.0, rel=0.1)
    assert est["r1_bulk"] == pytest.approx(2.0, rel=0.1)
    assert est["r2_cand"] == pytest.approx(1.0, rel=0.15)
    assert est["r2_bulk"] == pytest.approx(1.0, rel=0.15)


def test_point_estimates_nan_on_zero_events():
    df = pd.DataFrame({
        "N11": [0], "N12": [0], "N21": [5], "N22": [5],
        "E35": [10], "E36": [10], "E45": [10], "E46": [10],
        "n5": [100], "n6": [100],
    })
    est = point_estimates(df)
    assert math.isnan(est["r1_bulk"])
    assert math.isnan(est["r1_cand"])
    assert est["r2_bulk"] == pytest.approx(1.0)


def test_window_estimates_covers_all_cycles():
    df = _synthetic_counts(n_cycles=1000)
    win = window_estimates(df, n_windows=10)
    assert len(win) == 10
    assert int(win["n_events"].sum()) == int(
        df[["N11", "N12", "N21", "N22"]].sum().sum()
    )


def test_block_bootstrap_ci_contains_truth():
    df = _synthetic_counts(r1=2.0)
    ci = block_bootstrap(df, n_blocks=20, n_boot=500, seed=1)
    lo, hi = ci["r1_cand_boot_ci"]
    assert lo < 2.0 < hi


def test_mayo_lewis_fit_recovers_known_r():
    r1_true, r2_true = 2.0, 0.5
    win = _simulate_mayo_lewis_trajectory(r1_true, r2_true, f1_0=0.3, x_end=0.85)
    fit = mayo_lewis_fit(win, f1_0=0.3)
    assert fit["r1_ml"] == pytest.approx(r1_true, rel=0.03)
    assert fit["r2_ml"] == pytest.approx(r2_true, rel=0.05)


def _simulate_mayo_lewis_trajectory(r1, r2, f1_0, x_end=0.6, n=20):
    """从已知 (r1, r2) 数值积分瞬时组成方程, 生成 (X, f1) 窗口数据。"""
    from scipy.integrate import solve_ivp

    def ode(x, y):
        f1 = y[0]
        f2 = 1.0 - f1
        F1 = (r1 * f1**2 + f1 * f2) / (r1 * f1**2 + 2 * f1 * f2 + r2 * f2**2)
        return [(f1 - F1) / (1.0 - x)]

    X = np.linspace(0.001, x_end, n)
    sol = solve_ivp(ode, [0.001, x_end], [f1_0], t_eval=X, rtol=1e-8, atol=1e-10)
    return pd.DataFrame({"X_mid": X, "f1": sol.y[0]})


def test_mayo_lewis_integral_recovers_known_r():
    """积分形式(Meyer-Lowry): 窄 f1 域内也应能从 X 轨迹 recover 真值。"""
    r1_true, r2_true = 2.0, 0.5
    win = _simulate_mayo_lewis_trajectory(r1_true, r2_true, f1_0=0.3)
    fit = mayo_lewis_fit(win, f1_0=0.3)
    assert fit["r1_ml"] == pytest.approx(r1_true, rel=0.03)
    assert fit["r2_ml"] == pytest.approx(r2_true, rel=0.05)


def test_mayo_lewis_identifiability_flag():
    """f1 动态范围过窄时应标记不可辨识。"""
    win = _simulate_mayo_lewis_trajectory(1.0, 1.0, f1_0=0.5, x_end=0.1)
    fit = mayo_lewis_fit(win, f1_0=0.5)
    assert fit["f1_delta"] < 0.05
    assert fit["identifiable"] is False


from LmpPy.output_analysis.reactivity_ratio import collect_per_cycle_counts


def test_collect_per_cycle_counts(tmp_path):
    """两个微型 process 目录,验证拼接、cycle 偏移与帧数校验。"""
    for proc, n_frames, events in [
        ("process1", 3, [(1, 3, 5), (2, 4, 6)]),
        ("process2", 2, [(1, 3, 6)]),
    ]:
        ml_output = tmp_path / proc / "ml_output"
        ml_output.mkdir(parents=True)
        # 微型轨迹（20 Å 盒子: 确保 (1,1,1)-(8,8,8) 在 PBC 最小镜像下仍 > 10 Å 截断）
        with open(ml_output / "cg_trajectory.lammpstrj", "w") as f:
            for i in range(n_frames):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{i * 100}\n")
                f.write("ITEM: NUMBER OF ATOMS\n4\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0.0 20.0\n0.0 20.0\n0.0 20.0\n")
                f.write("ITEM: ATOMS id type x y z\n")
                f.write("1 3 1.0 1.0 1.0\n2 5 2.0 1.0 1.0\n3 6 8.0 8.0 8.0\n4 1 5.0 5.0 5.0\n")
        # 微型 reaction_details.csv
        cols = ("cycle,atom1_id,atom2_id,atom1_type_before,atom2_type_before,"
                "atom1_type_after,atom2_type_after,distance,reaction_type,n_angles,n_dihedrals\n")
        with open(ml_output / "reaction_details.csv", "w") as f:
            f.write(cols)
            for cyc, t1, t2 in events:
                f.write(f"{cyc},1,2,{t1},{t2},1,3,3.5,rxn_X,1,0\n")

    df = collect_per_cycle_counts(
        [tmp_path / "process1", tmp_path / "process2"], pair_cutoff=10.0, verbose=False
    )
    assert list(df["cycle"]) == [1, 2, 3, 4, 5]  # 全局连续编号
    # process1 cycle1 有一个 3+5 事件
    row1 = df[df["cycle"] == 1].iloc[0]
    assert row1["N11"] == 1 and row1["N12"] == 0
    assert row1["n5"] == 1 and row1["n6"] == 1 and row1["n3"] == 1
    assert row1["E35"] == 1  # 末端(1,1,1)与单体5(2,1,1)距离 1 < 10
    assert row1["E36"] == 0  # 单体6 距离 ~12 > 10
    # process2 的唯一事件落在全局 cycle 4
    assert df[df["cycle"] == 4].iloc[0]["N12"] == 1


def test_collect_per_cycle_counts_frame_mismatch(tmp_path):
    """帧数与事件 cycle 数不一致时应报错。"""
    ml_output = tmp_path / "process1" / "ml_output"
    ml_output.mkdir(parents=True)
    with open(ml_output / "cg_trajectory.lammpstrj", "w") as f:
        f.write("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n")
        f.write("ITEM: BOX BOUNDS pp pp pp\n0.0 10.0\n0.0 10.0\n0.0 10.0\n")
        f.write("ITEM: ATOMS id type x y z\n1 1 1.0 1.0 1.0\n")
    with open(ml_output / "reaction_details.csv", "w") as f:
        f.write("cycle,atom1_id,atom2_id,atom1_type_before,atom2_type_before,"
                "atom1_type_after,atom2_type_after,distance,reaction_type,n_angles,n_dihedrals\n")
        f.write("5,1,2,3,5,1,3,3.5,rxn_X,1,0\n")  # cycle 5 > 帧数 1
    with pytest.raises(ValueError, match="不一致"):
        collect_per_cycle_counts([tmp_path / "process1"], pair_cutoff=10.0, verbose=False)


from LmpPy.output_analysis.reactivity_ratio import analyze_reactivity_ratio


def test_analyze_from_counts(tmp_path):
    """从 counts CSV 走完整估计流程,验证输出文件与 JSON 结构。"""
    df = _synthetic_counts(n_cycles=2000, r1=2.0)
    df.insert(0, "cycle", range(1, len(df) + 1))
    df.insert(1, "timestep", df["cycle"] * 1000)
    for t in (1, 2, 3, 4):
        df[f"n{t}"] = 50
    counts_csv = tmp_path / "per_cycle_counts.csv"
    df.to_csv(counts_csv, index=False)
    out_dir = tmp_path / "analysis"
    analyze_reactivity_ratio(
        from_counts=str(counts_csv), output_dir=str(out_dir),
        windows=10, blocks=10, n_boot=200, seed=0,
    )
    summary = json.loads((out_dir / "reactivity_ratio_summary.json").read_text())
    assert summary["global"]["r1_cand"] == pytest.approx(2.0, rel=0.1)
    assert "r1_cand_boot_ci" in summary["bootstrap_ci"]
    assert "r1_ml" in summary["mayo_lewis_fit"]
    assert (out_dir / "r_vs_conversion.png").exists()
    assert (out_dir / "composition_mayo_lewis.png").exists()
    assert (out_dir / "window_estimates.csv").exists()
    diag = summary["diagnostics"]
    assert diag["events_per_active_center"] > 0
    assert diag["candidate_pairs_per_cycle"] > 0
