"""竞聚率(reactivity ratio)统计分析。

从 mlcgsim ML 驱动 CG 模拟输出直接统计共聚竞聚率 r1/r2。
设计文档: docs/superpowers/specs/2026-08-09-reactivity-ratio-analysis-design.md

通道映射(末端+单体 -> 通道):
    3+5 -> "11" (k11), 3+6 -> "12" (k12)
    4+5 -> "21" (k21), 4+6 -> "22" (k22)
"""

from __future__ import annotations

import json
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
    """总转化率序列 X(cycle) = 1 - (n5+n6)/(n5[0]+n6[0])。"""
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
    """按连续 cycle 块重抽样,输出各 r 估计的 95% 置信区间。"""
    rng = np.random.default_rng(seed)
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


def mayo_lewis_fit(win_df: pd.DataFrame) -> Dict[str, float]:
    """对窗口化 (f1, F1) 数据做微分 Mayo-Lewis 方程 NLLS 拟合。"""
    from scipy.optimize import curve_fit

    d = win_df.dropna(subset=["F1"])
    f1 = d["f1"].to_numpy()
    F1 = d["F1"].to_numpy()

    def model(f1, r1, r2):
        f2 = 1.0 - f1
        return (r1 * f1**2 + f1 * f2) / (r1 * f1**2 + 2 * f1 * f2 + r2 * f2**2)

    popt, _ = curve_fit(model, f1, F1, p0=[1.0, 1.0], bounds=(0, np.inf))
    return {"r1_ml": float(popt[0]), "r2_ml": float(popt[1])}
