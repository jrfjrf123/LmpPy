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
