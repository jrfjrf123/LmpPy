"""竞聚率(reactivity ratio)统计分析。

从 mlcgsim ML 驱动 CG 模拟输出直接统计共聚竞聚率 r1/r2。
设计文档: docs/superpowers/specs/2026-08-09-reactivity-ratio-analysis-design.md

通道映射(末端+单体 -> 通道):
    3+5 -> "11" (k11), 3+6 -> "12" (k12)
    4+5 -> "21" (k21), 4+6 -> "22" (k22)
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

# 通道 -> (末端 type, 单体 type)
CHANNELS: Dict[str, tuple] = {
    "11": (3, 5),
    "12": (3, 6),
    "21": (4, 5),
    "22": (4, 6),
}

# 通道 -> 候选对暴露列名
CHANNEL_EXPOSURE: Dict[str, str] = {
    "11": "E35",
    "12": "E36",
    "21": "E45",
    "22": "E46",
}

# 单体 type（用于转化率与浓度）
MONOMER_TYPES = (5, 6)


def channel_of_pair(t1: int, t2: int) -> Optional[str]:
    """无序类型对 -> 通道名；非反应通道返回 None。"""
    pair = (min(t1, t2), max(t1, t2))
    for name, types in CHANNELS.items():
        if pair == types:
            return name
    return None


def count_events(details_df: pd.DataFrame) -> pd.DataFrame:
    """单 process 的 reaction_details -> 每 cycle 四通道事件计数。

    参数
    ----
    details_df : 含 cycle, atom1_type_before, atom2_type_before 列

    返回
    ----
    pd.DataFrame: cycle(1..max, 缺失补 0), N11, N12, N21, N22
    """
    channels = [
        channel_of_pair(a, b)
        for a, b in zip(details_df["atom1_type_before"], details_df["atom2_type_before"])
    ]
    d = details_df.assign(channel=channels)
    grp = d.groupby(["cycle", "channel"]).size().unstack(fill_value=0)
    max_cycle = int(details_df["cycle"].max())
    idx = pd.Index(range(1, max_cycle + 1), name="cycle")
    grp = grp.reindex(idx, fill_value=0)
    out = pd.DataFrame(index=idx)
    for name in CHANNELS:
        out[f"N{name}"] = grp[name] if name in grp.columns else 0
    return out.reset_index()


def count_types_frame(types: np.ndarray, n_types: int = 6) -> Dict[str, int]:
    """统计单帧各 bead type 数量,返回 {'n1': ..., 'n6': ...}。"""
    counts = np.bincount(types, minlength=n_types + 1)
    return {f"n{t}": int(counts[t]) for t in range(1, n_types + 1)}


def count_candidates_frame(
    types: np.ndarray,
    coords: np.ndarray,
    box_lengths: np.ndarray,
    cutoff: float,
    block: int = 20000,
) -> Dict[str, int]:
    """统计单帧内四通道候选对数（PBC 最小镜像, 正交盒子）。

    只计算 末端(type 3/4) × 单体(type 5/6) 对, 与 ML 候选对搜索同定义。
    分块计算防内存爆。

    返回: {'E35': ..., 'E36': ..., 'E45': ..., 'E46': ...}
    """
    out = {}
    for a, b in [(3, 5), (3, 6), (4, 5), (4, 6)]:
        ends = coords[types == a]
        monos = coords[types == b]
        cnt = 0
        for s in range(0, len(monos), block):
            d = monos[s : s + block, None, :] - ends[None, :, :]
            d -= box_lengths * np.round(d / box_lengths)
            cnt += int((np.einsum("ijk,ijk->ij", d, d) < cutoff * cutoff).sum())
        out[f"E{a}{b}"] = cnt
    return out


# ---------------------------------------------------------------------------
# per-cycle 计数收集（事件 + 类型 + 候选对暴露,多 process 拼接）
# ---------------------------------------------------------------------------


def collect_per_cycle_counts(
    run_dirs: Sequence,
    pair_cutoff: float = 10.0,
    max_frames: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """从多个 process 运行目录收集 per-cycle 计数表。

    参数
    ----
    run_dirs : process 目录列表（各含 ml_output/reaction_details.csv
        与 ml_output/cg_trajectory.lammpstrj）,按时间顺序给出
    pair_cutoff : 候选对距离截断 (Å),需与模拟的 pair_cutoff 一致
    max_frames : 每个 process 最多读取的帧数（调试用;设置时跳过帧数校验）
    verbose : 打印进度

    返回
    ----
    pd.DataFrame: cycle(全局连续), timestep, n1..n6, E35,E36,E45,E46,
        N11,N12,N21,N22
    """
    from LmpPy.output_analysis.loader import load_reaction_details
    from LmpPy.utils.file_utils import iter_lammps_dump_frames

    all_rows = []
    cycle_offset = 0
    for run_dir in run_dirs:
        ml_output = Path(run_dir) / "ml_output"
        details = load_reaction_details(ml_output)
        events = count_events(details)
        traj = ml_output / "cg_trajectory.lammpstrj"
        if not traj.exists():
            raise FileNotFoundError(f"未找到轨迹文件: {traj}")

        rows_by_cycle: Dict[int, dict] = {}
        for i, frame in enumerate(iter_lammps_dump_frames(str(traj), max_frames=max_frames)):
            cyc = i + 1
            row = {"cycle": cyc, "timestep": int(frame["timestep"])}
            row.update(count_types_frame(frame["types"]))
            row.update(count_candidates_frame(
                frame["types"], frame["coords"],
                frame["box"][:, 1] - frame["box"][:, 0], pair_cutoff,
            ))
            rows_by_cycle[cyc] = row
            if verbose and cyc % 100 == 0:
                print(f"  [{Path(run_dir).name}] 已处理 {cyc} 帧")

        n_frames = len(rows_by_cycle)
        n_event_cycles = int(events["cycle"].max())
        if max_frames is None and n_event_cycles > n_frames:
            raise ValueError(
                f"{run_dir}: 事件 cycle 数 {n_event_cycles} 超出轨迹帧数 {n_frames} 不一致"
            )

        for cyc in range(1, n_frames + 1):
            row = rows_by_cycle[cyc]
            ev = events[events["cycle"] == cyc]
            for ch in CHANNELS:
                row[f"N{ch}"] = int(ev[f"N{ch}"].iloc[0]) if len(ev) else 0
            row["cycle"] = cyc + cycle_offset
            all_rows.append(row)
        cycle_offset += n_frames
        if verbose:
            print(f"[collect] {Path(run_dir).name}: {n_frames} 帧完成")

    return pd.DataFrame(all_rows)


def collect_per_cycle_counts_aa(
    aa_dirs: Sequence,
    pair_cutoff: float = 10.0,
    bond_react_check_step: int = 1,
    verbose: bool = True,
) -> pd.DataFrame:
    """从 AA bond/react 模拟输出收集 per-frame 计数表。

    数据源为 LmpPy bond/react 后处理的产物（如
    data/LmpPy_original/*/multi_chunk_md/*/）：
    - reaction_frames.npz: 反应帧,含 cg_bonds_before/after 与 cg_mapping_before,
      每帧一个反应事件(新键类型对 -> 通道)
    - cg_trajectory.lammpstrj: 事件驱动轨迹(每个事件前后各一帧)

    **轨迹结构**：每个 md 编号是一条跨 chunk 的连续模拟
    （chunk1/mdN -> chunk2/mdN -> ...），按 (md, chunk) 分组串联为
    多条独立轨迹,输出带 traj 列（md 名）。切勿按 rglob 目录序交错拼接。

    与 collect_per_cycle_counts 的区别: 无固定 cycle 网格,行 = 事件帧,
    cycle 为轨迹内连续帧序号。

    **反应前帧**：事件帧取轨迹中 ts - bond_react_check_step 帧（反应前构型,
    单体未消耗、类型未变,与 reaction_frames.npz 的 cg_mapping_before 一致;
    LmpPy 写出事件驱动轨迹时,反应前帧即按此间隔标注 timestep），
    n5/n6 与候选对暴露 E 均为事件发生时刻的浓度/暴露。

    参数
    ----
    aa_dirs : 根目录列表(各含 chunk*/md*/reaction_frames.npz),
        或直接传 md 目录(含 reaction_frames.npz 的目录)
    pair_cutoff : 候选对距离截断 (Å)
    bond_react_check_step : 该数据 bond/react 的检查间隔 (MD 步),
        需与 lammps_params.yaml 的 steps.bond_react_check 一致
    verbose : 打印进度

    返回
    ----
    pd.DataFrame: traj, cycle(轨迹内连续帧序号), timestep, n1..n6,
        E35,E36,E45,E46, N11,N12,N21,N22
    """
    from LmpPy.utils.file_utils import iter_lammps_dump_frames

    md_dirs: List[Path] = []
    for root in aa_dirs:
        root = Path(root)
        npz = root / "reaction_frames.npz"
        if npz.exists():
            md_dirs.append(root)
        else:
            found = sorted(p.parent for p in root.rglob("reaction_frames.npz"))
            if not found:
                raise FileNotFoundError(f"目录下未找到任何 reaction_frames.npz: {root}")
            md_dirs.extend(found)
    if not md_dirs:
        raise ValueError("未找到任何含 reaction_frames.npz 的目录")

    # 按 md 编号分组(chunkN/mdM -> md=M),组内按 chunk 数字序串联
    def _md_key(d: Path) -> str:
        return d.parent.name if d.name.startswith("chunk") else d.name

    def _chunk_key(d: Path) -> int:
        try:
            return int(d.parent.name.removeprefix("chunk"))
        except ValueError:
            return 0

    traj_groups: Dict[str, List[Path]] = {}
    for md_dir in md_dirs:
        key = _md_key(md_dir)
        traj_groups.setdefault(key, []).append(md_dir)
    for key in traj_groups:
        traj_groups[key].sort(key=_chunk_key)

    all_rows = []
    for traj_id, dirs in traj_groups.items():
        n_traj_rows = 0
        for md_dir in dirs:
            npz_path = md_dir / "reaction_frames.npz"
            traj = md_dir / "cg_trajectory.lammpstrj"
            if not traj.exists():
                raise FileNotFoundError(f"未找到轨迹文件: {traj}")

            d = np.load(npz_path, allow_pickle=True)
            # timestep -> 该帧新增键的通道列表(每帧通常恰好一个事件)
            events_by_ts: Dict[int, List[str]] = {}
            for i in range(len(d["timestep"])):
                ts = int(d["timestep"][i])
                m = d["cg_mapping_before"][i]
                bt = {int(r[0]): int(r[2]) for r in m}
                sb = set(map(tuple, np.sort(d["cg_bonds_before"][i][:, 1:3], axis=1)))
                sa = set(map(tuple, np.sort(d["cg_bonds_after"][i][:, 1:3], axis=1)))
                chans = []
                for a, b in (sa - sb):
                    t1, t2 = bt.get(a), bt.get(b)
                    if t1 is None or t2 is None:
                        continue
                    ch = channel_of_pair(t1, t2)
                    if ch:
                        chans.append(ch)
                events_by_ts[ts] = chans

            # 事件帧取 ts - bond_react_check_step(反应前构型,与 LmpPy
            # run_refactored.py 写出反应前帧的 timestep 标注间隔一致);
            # 缺失则直接报错(不静默回退),提示核对检查间隔配置
            rows_by_ts: Dict[int, dict] = {}
            for frame in iter_lammps_dump_frames(str(traj)):
                ts = int(frame["timestep"])
                ev_ts = ts + bond_react_check_step
                if ev_ts not in events_by_ts:
                    continue
                row = {"timestep": ev_ts}
                row.update(count_types_frame(frame["types"]))
                row.update(count_candidates_frame(
                    frame["types"], frame["coords"],
                    frame["box"][:, 1] - frame["box"][:, 0], pair_cutoff,
                ))
                rows_by_ts[ev_ts] = row

            missing_ts = [ts for ts in events_by_ts if ts not in rows_by_ts]
            if missing_ts:
                raise ValueError(
                    f"{md_dir}: 事件反应前帧(ts-{bond_react_check_step})"
                    f"未在轨迹中找到: {missing_ts}; 请确认 bond_react_check_step "
                    f"与该数据 lammps_params.yaml 的 steps.bond_react_check 一致"
                )

            for ts in sorted(events_by_ts):
                row = rows_by_ts[ts]
                for ch in CHANNELS:
                    row[f"N{ch}"] = int(events_by_ts[ts].count(ch))
                row["traj"] = traj_id
                n_traj_rows += 1
                row["cycle"] = n_traj_rows
                all_rows.append(row)
            if verbose:
                print(f"[collect_aa] {md_dir}: {len(events_by_ts)} 事件帧完成")

    return pd.DataFrame(all_rows)


# ---------------------------------------------------------------------------
# r 值估计
# ---------------------------------------------------------------------------


def _safe_ratio(num: float, den: float) -> float:
    """分子或分母为 0 时返回 NaN。"""
    if num <= 0 or den <= 0:
        return float("nan")
    return num / den


def point_estimates(sub: pd.DataFrame) -> Dict[str, float]:
    """对任意 cycle 子集计算双归一化 r 点估计。

    sub 需含列: N11,N12,N21,N22, E35,E36,E45,E46, n5,n6
    """
    N = {ch: float(sub[f"N{ch}"].sum()) for ch in CHANNELS}
    E = {col: float(sub[col].sum()) for col in CHANNEL_EXPOSURE.values()}
    n5 = float(sub["n5"].mean())
    n6 = float(sub["n6"].mean())
    if n5 <= 0 or n6 <= 0:
        raise ValueError("单体计数非正，无法归一化")
    return {
        "r1_bulk": _safe_ratio(N["11"], N["12"]) * (n6 / n5),
        "r2_bulk": _safe_ratio(N["22"], N["21"]) * (n5 / n6),
        "r1_cand": _safe_ratio(
            _safe_ratio(N["11"], E["E35"]), _safe_ratio(N["12"], E["E36"])
        ) if N["11"] > 0 and N["12"] > 0 else float("nan"),
        "r2_cand": _safe_ratio(
            _safe_ratio(N["22"], E["E46"]), _safe_ratio(N["21"], E["E45"])
        ) if N["22"] > 0 and N["21"] > 0 else float("nan"),
    }


def conversion_series(df: pd.DataFrame) -> pd.Series:
    """总转化率序列 X(c) = 1 - (n5+n6)(c)/(n5+n6)[轨迹首帧]。

    含 traj 列时按轨迹分组,每条轨迹以自身首帧单体数为 n0
    (AA 数据为跨 chunk 连续的多条独立轨迹);
    无 traj 列时退化为全表首帧(mlcgsim 数据)。
    """
    if "traj" in df.columns:
        out = np.empty(len(df))
        for _, sub in df.groupby("traj", sort=False):
            n0 = float(sub["n5"].iloc[0] + sub["n6"].iloc[0])
            out[sub.index] = 1.0 - (sub["n5"] + sub["n6"]) / n0
        return pd.Series(out, index=df.index, name="X")
    n0 = float(df["n5"].iloc[0] + df["n6"].iloc[0])
    return 1.0 - (df["n5"] + df["n6"]) / n0


def window_estimates(df: pd.DataFrame, n_windows: int) -> pd.DataFrame:
    """按转化率等分窗口,逐窗输出 r 估计与组成量。"""
    X = conversion_series(df)
    edges = np.linspace(0.0, float(X.iloc[-1]), n_windows + 1)
    win_id = np.clip(np.searchsorted(edges, X.values, side="right") - 1, 0, n_windows - 1)
    rows = []
    for w in range(n_windows):
        mask = win_id == w
        sub = df[mask]
        if len(sub) == 0:
            continue
        n_tot = sum(float(sub[f"N{ch}"].sum()) for ch in CHANNELS)
        row = {
            "window": w,
            "X_mid": float(X.values[mask].mean()),
            "f1": float((sub["n5"] / (sub["n5"] + sub["n6"])).mean()),
            "F1": (float(sub["N11"].sum()) + float(sub["N21"].sum())) / n_tot
            if n_tot > 0 else float("nan"),
            "n_events": int(n_tot),
        }
        row.update(point_estimates(sub))
        rows.append(row)
    return pd.DataFrame(rows)


def block_bootstrap(
    df: pd.DataFrame, n_blocks: int = 20, n_boot: int = 1000, seed: int = 0
) -> Dict[str, list]:
    """按连续 cycle 块重抽样,输出各 r 估计的 95% 置信区间。

    含 traj 列时块在**轨迹内**连续(每条轨迹切 n_blocks/轨迹数 块,
    块池混合后重抽样),避免块横跨独立轨迹;
    无 traj 列时按全表连续行切块(mlcgsim 数据)。
    """
    rng = np.random.default_rng(seed)
    if "traj" in df.columns:
        n_traj = int(df["traj"].nunique())
        k = max(1, int(np.ceil(n_blocks / n_traj)))
        blocks = []
        for _, sub in df.groupby("traj", sort=False):
            blocks.extend(b for b in np.array_split(sub, k) if len(b) > 0)
    else:
        blocks = [b for b in np.array_split(df, n_blocks) if len(b) > 0]
    keys = ["r1_bulk", "r1_cand", "r2_bulk", "r2_cand"]
    samples: Dict[str, List[float]] = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, len(blocks), size=len(blocks))
        sub = pd.concat([blocks[i] for i in idx])
        est = point_estimates(sub)
        for k in keys:
            samples[k].append(est[k])
    out = {}
    for k in keys:
        arr = np.asarray(samples[k], dtype=float)
        arr = arr[~np.isnan(arr)]
        if len(arr) == 0:
            out[f"{k}_boot_ci"] = [None, None]
        else:
            lo, hi = np.percentile(arr, [2.5, 97.5])
            out[f"{k}_boot_ci"] = [float(lo), float(hi)]
    return out


def beta_intervals(df: pd.DataFrame) -> Dict[str, list]:
    """Beta 后验解析区间（辅助对照,非主 CI——事件间有相关性）。"""
    from scipy.stats import beta as beta_dist

    n5 = float(df["n5"].mean())
    n6 = float(df["n6"].mean())
    E = {col: float(df[col].sum()) for col in CHANNEL_EXPOSURE.values()}
    specs = [
        ("r1", "N11", "N12", n6 / n5, E["E36"] / E["E35"]),
        ("r2", "N22", "N21", n5 / n6, E["E45"] / E["E46"]),
    ]
    out = {}
    for name, num_col, den_col, f_bulk, f_cand in specs:
        a = float(df[num_col].sum()) + 1.0
        b = float(df[den_col].sum()) + 1.0
        lo, hi = beta_dist.ppf([0.025, 0.975], a, b)
        r_lo = lo / (1.0 - lo)
        r_hi = hi / (1.0 - hi)
        out[f"{name}_bulk_beta_ci"] = [float(r_lo * f_bulk), float(r_hi * f_bulk)]
        out[f"{name}_cand_beta_ci"] = [float(r_lo * f_cand), float(r_hi * f_cand)]
    return out


def _mayo_lewis_ode(x: float, y, r1: float, r2: float):
    """瞬时共聚组成方程(Mayo-Lewis)的转化率积分形式 ODE。

    df1/dX = (f1 - F1(f1; r1, r2)) / (1 - X)
    """
    f1 = y[0]
    f2 = 1.0 - f1
    F1 = (r1 * f1**2 + f1 * f2) / (r1 * f1**2 + 2 * f1 * f2 + r2 * f2**2)
    return [(f1 - F1) / (1.0 - x)]


def mayo_lewis_fit(win_df: pd.DataFrame, f1_0: Optional[float] = None) -> Dict[str, float]:
    """Meyer-Lowry 积分形式拟合（累计单体消耗轨迹）。

    对窗口化 (X_mid, f1) 数据数值积分瞬时组成方程 ODE,最小二乘拟合 (r1, r2)。
    相比微分形式(F1 vs f1),积分形式利用整个转化率轨迹,
    在 f1 动态范围窄时仍保有可辨识性;f1 范围过窄时标记 identifiable=False。
    """
    from scipy.integrate import solve_ivp
    from scipy.optimize import least_squares

    d = win_df.dropna(subset=["f1"]).sort_values("X_mid")
    if len(d) < 3 or "X_mid" not in d.columns:
        return {"r1_ml": float("nan"), "r2_ml": float("nan"),
                "f1_delta": float("nan"), "identifiable": False}
    X = d["X_mid"].to_numpy()
    f1_obs = d["f1"].to_numpy()
    f1_delta = float(f1_obs.max() - f1_obs.min())
    if f1_0 is None:
        f1_0 = float(f1_obs[0])

    def resid(params):
        r1, r2 = params
        try:
            sol = solve_ivp(
                lambda x, y: _mayo_lewis_ode(x, y, r1, r2),
                [0.0, X[-1]], [f1_0], t_eval=X, rtol=1e-6, atol=1e-9,
            )
            return sol.y[0] - f1_obs
        except Exception:
            return np.full_like(f1_obs, 1e6)

    best = None
    for p0 in ([1.0, 1.0], [0.5, 0.5], [2.0, 2.0], [0.3, 3.0]):
        try:
            cand = least_squares(
                resid, p0, bounds=([1e-3, 1e-3], [100.0, 100.0]),
                xtol=1e-10, ftol=1e-10, gtol=1e-10,
            )
        except Exception:
            continue
        if best is None or cand.cost < best.cost:
            best = cand
    if best is None:
        return {"r1_ml": float("nan"), "r2_ml": float("nan"),
                "f1_delta": f1_delta, "identifiable": False}
    return {
        "r1_ml": float(best.x[0]),
        "r2_ml": float(best.x[1]),
        "f1_delta": f1_delta,
        "identifiable": bool(f1_delta >= 0.1),
    }


# ---------------------------------------------------------------------------
# 编排入口与绘图
# ---------------------------------------------------------------------------


def _nan_to_none(obj):
    """递归把 NaN 替换为 None,便于 JSON 序列化。"""
    if isinstance(obj, dict):
        return {k: _nan_to_none(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_nan_to_none(v) for v in obj]
    if isinstance(obj, float) and math.isnan(obj):
        return None
    return obj


def plot_r_vs_conversion(
    win_tables: Dict[int, pd.DataFrame],
    global_est: Dict[str, float],
    boot_ci: Dict[str, list],
    out_path: Path,
) -> None:
    """r1(X)、r2(X) 轨迹图:双归一化 + 窗宽扫描 + 全局 CI 带。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    primary = max(win_tables.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, rkey, title in [
        (axes[0], "r1", "r1 (E chain end)"),
        (axes[1], "r2", "r2 (P chain end)"),
    ]:
        for nw, wdf in sorted(win_tables.items()):
            alpha = 1.0 if nw == primary else 0.35
            lw = 1.8 if nw == primary else 1.0
            label_suffix = f" (windows={nw})"
            ax.plot(wdf["X_mid"], wdf[f"{rkey}_bulk"], "o-", alpha=alpha, lw=lw,
                    color="tab:blue", label="bulk" + label_suffix)
            ax.plot(wdf["X_mid"], wdf[f"{rkey}_cand"], "s--", alpha=alpha, lw=lw,
                    color="tab:orange", label="cand" + label_suffix)
        ci = boot_ci.get(f"{rkey}_bulk_boot_ci")
        if ci and ci[0] is not None:
            ax.axhspan(ci[0], ci[1], color="tab:blue", alpha=0.08)
        ax.axhline(global_est[f"{rkey}_bulk"], color="tab:blue", ls=":", alpha=0.7)
        ax.axhline(1.0, color="gray", ls="-", lw=0.5)
        ax.set_xlabel("conversion X")
        ax.set_ylabel(rkey)
        ax.set_title(title)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_channel_rates(df: pd.DataFrame, out_path: Path) -> None:
    """四通道细粒度柱状图。

    左图 bulk 口径: rho_ij = N_ij / <n_j>（每单体每 cycle 事件率,
    r1_bulk = rho11/rho12, r2_bulk = rho22/rho21 直接成立）;
    右图 cand 口径: pi_ij = N_ij / E_ij（每候选对每 cycle 接受率,
    r1_cand = pi11/pi12, r2_cand = pi22/pi21）。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    N = {ch: float(df[f"N{ch}"].sum()) for ch in CHANNELS}
    E = {col: float(df[col].sum()) for col in CHANNEL_EXPOSURE.values()}
    n5 = float(df["n5"].mean())
    n6 = float(df["n6"].mean())
    n_mono = {"11": n5, "12": n6, "21": n5, "22": n6}

    channels = ["11", "12", "21", "22"]
    labels = ["E+E\n(k11)", "E+P\n(k12)", "P+E\n(k21)", "P+P\n(k22)"]
    rho = [N[c] / n_mono[c] for c in channels]
    pi = [N[c] / E[CHANNEL_EXPOSURE[c]] for c in channels]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    bars1 = ax1.bar(labels, rho, color="tab:blue", alpha=0.85)
    ax1.set_ylabel("N_ij / <n_j>  (per monomer per cycle)")
    ax1.set_title("Bulk-normalized channel rates")
    for b, v in zip(bars1, rho):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.3f}",
                 ha="center", fontsize=9)
    ax1.axhline(0, color="gray", lw=0.5)

    bars2 = ax2.bar(labels, pi, color="tab:orange", alpha=0.85)
    ax2.set_ylabel("N_ij / E_ij  (per candidate pair per cycle)")
    ax2.set_title("Exposure-normalized channel rates")
    for b, v in zip(bars2, pi):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.02, f"{v:.5f}",
                 ha="center", fontsize=9)
    ax2.axhline(0, color="gray", lw=0.5)

    fig.suptitle(
        f"Channel rates:  r1_bulk={rho[0]/rho[1]:.3f}, r1_cand={pi[0]/pi[1]:.3f} | "
        f"r2_bulk={rho[3]/rho[2]:.3f}, r2_cand={pi[3]/pi[2]:.3f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_summary_dashboard(
    global_est: Dict[str, float],
    boot_ci: Dict[str, list],
    out_path: Path,
) -> None:
    """r1/r2 详细柱状图（双归一化 + 95% CI + 术语解释）。

    横轴明确标注 r1、r2；每组两根柱：bulk（浓度归一化）、cand（暴露归一化）;
    误差棒为 block bootstrap 95% CI; 右下角注释框解释全部专有名词。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))

    r_names = ["r1", "r2"]
    x = np.arange(len(r_names))
    width = 0.32
    for i, mode in enumerate(["bulk", "cand"]):
        vals = np.array([global_est[f"{rk}_{mode}"] for rk in r_names])
        errs = np.zeros((2, len(r_names)))
        for j, rk in enumerate(r_names):
            ci = boot_ci[f"{rk}_{mode}_boot_ci"]
            if ci[0] is not None:
                errs[0, j] = vals[j] - ci[0]
                errs[1, j] = ci[1] - vals[j]
        bars = ax.bar(
            x + (i - 0.5) * width, vals, width,
            yerr=errs, capsize=5, label=f"{mode}",
            color="tab:blue" if mode == "bulk" else "tab:orange", alpha=0.9,
        )
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                    ha="center", fontsize=10, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["r1", "r2"], fontsize=13)
    ax.set_xlabel("reactivity ratio", fontsize=12)
    ax.set_ylabel("reactivity ratio r", fontsize=12)
    ax.set_title("Reactivity ratios r1, r2 with 95% block-bootstrap CI",
                 fontsize=13)
    ax.axhline(1.0, color="gray", ls=":", lw=1.0)
    ax.legend(fontsize=10, loc="upper left")
    ymax = max(global_est["r1_bulk"], global_est["r1_cand"],
               global_est["r2_bulk"], global_est["r2_cand"]) * 1.35
    ax.set_ylim(0, ymax)

    # 右下角术语解释框
    r1p = global_est["r1_bulk"] * global_est["r2_bulk"]
    r1p_c = global_est["r1_cand"] * global_est["r2_cand"]
    glossary = (
        "Notation:\n"
        "  r1 = k11/k12 : E chain-end selectivity (E vs P monomer)\n"
        "  r2 = k22/k21 : P chain-end selectivity (P vs E monomer)\n"
        "  k11: E end + E mono,  k12: E end + P mono,\n"
        "  k21: P end + E mono,  k22: P end + P mono\n"
        "  bulk : normalized by global monomer concentration\n"
        "         (Mayo-Lewis literature convention)\n"
        "  cand : normalized by 10 A candidate-pair exposure\n"
        "         (local environment convention)\n"
        "  error bar : 95% block-bootstrap CI\n"
        f"  r1*r2 = {r1p:.2f} (bulk) / {r1p_c:.2f} (cand)"
    )
    ax.text(
        0.99, 0.02, glossary, transform=ax.transAxes, ha="right", va="bottom",
        family="monospace", fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow",
                  edgecolor="gray", alpha=0.9),
    )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mayo_lewis(
    win_df: pd.DataFrame, fit: Dict[str, float], out_path: Path
) -> None:
    """瞬时共聚组成 F1 vs f1 散点 + Mayo-Lewis 拟合曲线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    r1_ml, r2_ml = fit["r1_ml"], fit["r2_ml"]
    f1_grid = np.linspace(0.01, 0.99, 200)
    f2_grid = 1.0 - f1_grid
    F1_grid = ((r1_ml * f1_grid**2 + f1_grid * f2_grid)
               / (r1_ml * f1_grid**2 + 2 * f1_grid * f2_grid + r2_ml * f2_grid**2))

    fig, ax = plt.subplots(figsize=(6, 6))
    d = win_df.dropna(subset=["F1"])
    ax.plot(d["f1"], d["F1"], "o", label="simulation (windowed)")
    ax.plot(f1_grid, F1_grid, "-",
            label=f"Mayo-Lewis fit: r1={r1_ml:.3f}, r2={r2_ml:.3f}")
    ax.plot([0, 1], [0, 1], ":", color="gray", label="F1 = f1")
    ax.set_xlabel("f1 (feed, E fraction)")
    ax.set_ylabel("F1 (instantaneous copolymer, E fraction)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def analyze_reactivity_ratio(
    run_dirs: Optional[Sequence] = None,
    output_dir: Optional[str] = None,
    windows: int = 20,
    pair_cutoff: float = 10.0,
    blocks: int = 20,
    n_boot: int = 1000,
    seed: int = 0,
    max_frames: Optional[int] = None,
    from_counts: Optional[str] = None,
    counts_out: Optional[str] = None,
    from_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """竞聚率分析主入口。

    四种用法:
    1. 全流程: 传 run_dirs + output_dir
    2. 只统计计数: 传 run_dirs + counts_out(不传 output_dir)
    3. 只做估计: 传 from_counts + output_dir
    4. 直接传计数表: 传 from_df + output_dir(AA 数据入口内部复用)
    """
    if from_counts is not None:
        df = pd.read_csv(from_counts)
    elif from_df is not None:
        df = from_df
    else:
        if not run_dirs:
            raise ValueError("必须提供 run_dirs、from_counts 或 from_df")
        df = collect_per_cycle_counts(run_dirs, pair_cutoff, max_frames)
        if counts_out:
            Path(counts_out).parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(counts_out, index=False)
            print(f"[输出] per-cycle 计数表: {counts_out}")
    required = ({"cycle", "n5", "n6", "E35", "E36", "E45", "E46"}
                | {f"N{ch}" for ch in CHANNELS})
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"counts 数据缺少列: {sorted(missing)}")

    if output_dir is None:
        return df  # counts-only 模式

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_est = point_estimates(df)
    boot = block_bootstrap(df, n_blocks=blocks, n_boot=n_boot, seed=seed)
    beta_ci = beta_intervals(df)

    # 窗宽敏感性:主窗宽 + 10/40
    widths = sorted({10, windows, 40})
    win_tables = {w: window_estimates(df, w) for w in widths}
    win_primary = win_tables[windows]
    ml_fit = mayo_lewis_fit(win_primary)

    win_primary.to_csv(out_dir / "window_estimates.csv", index=False)
    X = conversion_series(df)
    n_events_total = int(sum(int(df[f"N{ch}"].sum()) for ch in CHANNELS))
    n_cycles = int(len(df))
    active_centers = float(df["n3"].mean()) + float(df["n4"].mean())
    candidate_pairs = float(
        sum(float(df[col].mean()) for col in CHANNEL_EXPOSURE.values())
    )
    summary_diag = {
        "n_cycles": n_cycles,
        "events_per_channel": {ch: int(df[f"N{ch}"].sum()) for ch in CHANNELS},
        "events_per_active_center": (
            (n_events_total / n_cycles) / active_centers if active_centers > 0 else None
        ),
        "candidate_pairs_per_cycle": candidate_pairs,
        "conversion_final": (
            float(X.groupby(df["traj"], sort=False).last().mean())
            if "traj" in df.columns else float(X.iloc[-1])
        ),
        "zero_event_windows": int((win_primary["n_events"] == 0).sum()),
    }
    plot_r_vs_conversion(win_tables, global_est, boot, out_dir / "r_vs_conversion.png")
    plot_mayo_lewis(win_primary, ml_fit, out_dir / "composition_mayo_lewis.png")
    plot_channel_rates(df, out_dir / "channel_rates.png")
    plot_summary_dashboard(global_est, boot, out_dir / "summary_dashboard.png")
    summary = {
        "global": global_est,
        "bootstrap_ci": boot,
        "beta_ci": beta_ci,
        "mayo_lewis_fit": ml_fit,
        "diagnostics": summary_diag,
        "params": {
            "windows": windows, "pair_cutoff": pair_cutoff,
            "blocks": blocks, "n_boot": n_boot, "seed": seed,
        },
    }
    with open(out_dir / "reactivity_ratio_summary.json", "w") as f:
        json.dump(_nan_to_none(summary), f, indent=2, ensure_ascii=False)

    # 输出扩展: 浓度/通道事件演化数据与图、详细报告
    source_desc = None
    if from_counts is not None:
        source_desc = f"from_counts: {from_counts}"
    elif run_dirs:
        source_desc = "run_dirs: " + ", ".join(str(d) for d in run_dirs)
    write_evolution_outputs(df, out_dir)
    write_detailed_report(
        out_dir=out_dir,
        df=df,
        global_est=global_est,
        boot=boot,
        beta_ci=beta_ci,
        ml_fit=ml_fit,
        summary_diag=summary_diag,
        params=summary["params"],
        source_desc=source_desc,
    )

    print(f"[输出] 分析结果目录: {out_dir}")
    print(f"  r1: bulk={global_est['r1_bulk']:.3f}, cand={global_est['r1_cand']:.3f}, ML={ml_fit['r1_ml']:.3f}")
    print(f"  r2: bulk={global_est['r2_bulk']:.3f}, cand={global_est['r2_cand']:.3f}, ML={ml_fit['r2_ml']:.3f}")
    return df


def write_evolution_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    """输出浓度与通道事件的变化数据（evolution_data.csv）与演化图。

    - 浓度: n5, n6, 单体总数, n6/n5, f1, 转化率 X 随 cycle 的变化
    - 事件: N11/N12/N21/N22 每 cycle 值与累积值（含 traj 列时按轨迹分组累积）
    """
    out = pd.DataFrame({"cycle": df["cycle"]})
    if "timestep" in df.columns:
        out["timestep"] = df["timestep"]
    if "traj" in df.columns:
        out["traj"] = df["traj"]
    out["n5"] = df["n5"].astype(float)
    out["n6"] = df["n6"].astype(float)
    out["n_monomer_total"] = out["n5"] + out["n6"]
    out["n6_over_n5"] = out["n6"] / out["n5"].replace(0, np.nan)
    out["f1"] = out["n5"] / out["n_monomer_total"].replace(0, np.nan)
    out["X"] = conversion_series(df)
    for ch in CHANNELS:
        out[f"N{ch}"] = df[f"N{ch}"].astype(int)
    out["N_total"] = sum(out[f"N{ch}"] for ch in CHANNELS)
    if "traj" in df.columns:
        for ch in CHANNELS:
            out[f"N{ch}_cum"] = df.groupby("traj", sort=False)[f"N{ch}"].cumsum()
        out["N_total_cum"] = out.groupby("traj", sort=False)["N_total"].cumsum()
    else:
        for ch in CHANNELS:
            out[f"N{ch}_cum"] = df[f"N{ch}"].cumsum()
        out["N_total_cum"] = out["N_total"].cumsum()
    out.to_csv(out_dir / "evolution_data.csv", index=False)
    print(f"[输出] 演化数据: {out_dir / 'evolution_data.csv'}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    ax = axes[0]
    ax.plot(out["cycle"], out["n5"], label="n5", lw=1.2)
    ax.plot(out["cycle"], out["n6"], label="n6", lw=1.2)
    ax.set_ylabel("monomer count")
    ax.set_title("Monomer concentration evolution")
    ax2 = ax.twinx()
    ax2.plot(out["cycle"], out["X"], "k--", alpha=0.6, label="conversion X")
    ax2.set_ylabel("conversion X")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc="upper right")

    ax = axes[1]
    for ch in CHANNELS:
        ax.plot(out["cycle"], out[f"N{ch}_cum"], label=f"N{ch} (cum)", lw=1.2)
    ax.plot(out["cycle"], out["N_total_cum"], "k--", label="total (cum)", lw=1.5)
    ax.set_xlabel("cycle")
    ax.set_ylabel("cumulative events")
    ax.set_title("Channel event evolution (cumulative)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "evolution.png", dpi=150)
    plt.close(fig)
    print(f"[输出] 演化图: {out_dir / 'evolution.png'}")


def write_detailed_report(
    out_dir: Path,
    df: pd.DataFrame,
    global_est: Dict[str, float],
    boot: Dict[str, list],
    beta_ci: Dict[str, list],
    ml_fit: Dict[str, float],
    summary_diag: Dict,
    params: Dict,
    source_desc: Optional[str] = None,
) -> Path:
    """输出详细 report.txt: 数据源、设置、事件/浓度统计、估计与 CI 的完整总结。"""
    report_file = out_dir / "report.txt"
    n_events = {ch: int(df[f"N{ch}"].sum()) for ch in CHANNELS}
    n_total = sum(n_events.values())
    X = conversion_series(df)

    lines = []
    lines.append("=" * 70)
    lines.append("竞聚率分析详细报告 (reactivity ratio analysis report)")
    lines.append("=" * 70)

    lines.append("\n## 数据源")
    if source_desc:
        lines.append(f"  {source_desc}")
    if "traj" in df.columns:
        lines.append(f"  轨迹数: {df['traj'].nunique()}")
        for t, n in df.groupby("traj", sort=False).size().items():
            lines.append(f"    {t}: {n} 行 (事件帧)")
    lines.append(f"  总行数(cycle): {len(df)}")

    lines.append("\n## 参数设置")
    for k, v in params.items():
        lines.append(f"  {k}: {v}")

    lines.append("\n## 事件统计")
    for ch in CHANNELS:
        pct = n_events[ch] / n_total * 100 if n_total else 0.0
        lines.append(f"  N{ch} ({CHANNELS[ch][0]}+{CHANNELS[ch][1]}): "
                     f"{n_events[ch]} ({pct:.2f}%)")
    lines.append(f"  事件总数: {n_total}")
    if n_total:
        lines.append(f"  N11/N12 = {n_events['11'] / n_events['12']:.4f}  "
                     f"N22/N21 = {n_events['22'] / n_events['21']:.4f}")

    lines.append("\n## 浓度统计 (每 cycle 帧)")
    for col, label in [("n5", "n5 (A 单体)"), ("n6", "n6 (B 单体)")]:
        s = df[col].astype(float)
        lines.append(f"  {label}: 首={s.iloc[0]:.1f} 末={s.iloc[-1]:.1f} "
                     f"min={s.min():.1f} max={s.max():.1f} mean={s.mean():.1f}")
    r = df["n6"].astype(float) / df["n5"].astype(float).replace(0, np.nan)
    lines.append(f"  n6/n5: 首={r.iloc[0]:.4f} 末={r.iloc[-1]:.4f} mean={r.mean():.4f}")
    f1 = df["n5"].astype(float) / (df["n5"].astype(float) + df["n6"].astype(float))
    lines.append(f"  f1 (A 单体摩尔分率): 首={f1.iloc[0]:.4f} 末={f1.iloc[-1]:.4f} "
                 f"mean={f1.mean():.4f}")
    lines.append(f"  转化率 X: 首={X.iloc[0]:.4f} 末={X.iloc[-1]:.4f} "
                 f"max={X.max():.4f}")
    lines.append(f"  转化率最终(按轨迹平均): {summary_diag.get('conversion_final')}")

    lines.append("\n## 竞聚率点估计")
    for k, v in global_est.items():
        lines.append(f"  {k} = {v:.4f}" if isinstance(v, float) else f"  {k} = {v}")

    lines.append("\n## 置信区间 (95%)")
    for k, v in boot.items():
        if v and v[0] is not None:
            lines.append(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            lines.append(f"  {k}: 不可用")
    lines.append("  (beta 解析区间)")
    for k, v in beta_ci.items():
        if v and v[0] is not None:
            lines.append(f"  {k}: [{v[0]:.4f}, {v[1]:.4f}]")
        else:
            lines.append(f"  {k}: 不可用")

    lines.append("\n## Mayo-Lewis 组成法拟合 (Meyer-Lowry 积分形式)")
    for k, v in ml_fit.items():
        lines.append(f"  {k} = {v}")

    lines.append("\n## 诊断量")
    for k, v in summary_diag.items():
        lines.append(f"  {k}: {v}")

    lines.append("\n## 输出文件清单")
    for name in ["reactivity_ratio_summary.json", "window_estimates.csv",
                 "evolution_data.csv", "evolution.png",
                 "r_vs_conversion.png", "channel_rates.png",
                 "composition_mayo_lewis.png", "summary_dashboard.png"]:
        lines.append(f"  - {name}")
    lines.append("=" * 70)

    report_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[输出] 详细报告: {report_file}")
    return report_file


def analyze_reactivity_ratio_aa(
    aa_dirs: Optional[Sequence] = None,
    output_dir: Optional[str] = None,
    windows: int = 20,
    pair_cutoff: float = 10.0,
    bond_react_check_step: int = 1,
    blocks: int = 20,
    n_boot: int = 1000,
    seed: int = 0,
    counts_out: Optional[str] = None,
) -> pd.DataFrame:
    """AA bond/react 数据竞聚率分析入口(复用 analyze_reactivity_ratio 估计逻辑)。

    三种用法:
    1. 全流程: 传 aa_dirs + output_dir
    2. 只统计计数: 传 aa_dirs + counts_out(不传 output_dir)
    3. 只做估计: 用 reactivity-ratio --from-counts(counts_out 产物)

    bond_react_check_step 需与该数据 lammps_params.yaml 的
    steps.bond_react_check 一致(默认 1)。
    """
    if not aa_dirs:
        raise ValueError("必须提供 aa_dirs")
    df = collect_per_cycle_counts_aa(
        aa_dirs, pair_cutoff, bond_react_check_step=bond_react_check_step
    )
    if counts_out:
        Path(counts_out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(counts_out, index=False)
        print(f"[输出] per-frame 计数表: {counts_out}")
    if output_dir is None:
        return df
    return analyze_reactivity_ratio(
        output_dir=output_dir,
        windows=windows,
        blocks=blocks,
        n_boot=n_boot,
        seed=seed,
        from_df=df,
    )
