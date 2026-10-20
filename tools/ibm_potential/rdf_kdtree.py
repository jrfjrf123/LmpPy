# -*- coding: utf-8 -*-
"""KDTree 加速的 RDF（pair 分布）计算 —— 与原版全对枚举逐 bin 等价。

供 LmpPy/scripts/calc_dist.py 的 --rdf-engine kdtree 使用。
键/角/二面角分布不涉及，仅替换 RDF 的 O(n²) 全对枚举为 cKDTree 周期性近邻查询。

精确等价设计（非近似，目标逐 bin 计数完全相同）：
1. 树仅用于候选对选择；查询半径 r_query = r_max + 1e-4 Å
   （远大于坐标包裹引入的浮点偏差 ~1e-5 Å，远小于 bin 宽 0.12 Å），
   保证候选集合是「原版会进入直方图的对」的超集；
2. 包裹坐标（pos - L*floor(pos/L)，float64）仅供 cKDTree(boxsize) 建树；
   距离用原始未包裹坐标、按原版同一 numpy 表达式重算
   （f32 版对齐 pickle_loader.calculate_rdf_frame_by_frame: float32, sqrt(sum(delta**2))；
   f64 版对齐 distribution.calculate_rdf: float64, np.linalg.norm(delta, axis=-1)），
   再经同一个 np.histogram(range/edges) 过滤 → 逐 bin 计数与原版相同；
3. 同类型对：query_pairs 每对返回一次（原版有序对双重计数）→ 直方图 ×2；
   跨类型对：query_ball_tree 每对正好一次（与原版一致）；
   归一化代码逐字对齐各自原版（total_pairs / avg_volume / ideal gas）；
4. 排除集与原版同一构造函数（1-based (min,max) 元组集合 → 0-based），
   候选对打包 int64 键后用 np.isin 过滤（替代原版的 Python 逐对循环，
   结果相同，速度快数千倍）。

限制：cKDTree boxsize 仅支持正交盒（立方/长方）；r_max 须 < min(L)/2
（与最小影像约定无歧义的条件相同，原版亦然）。串行实现，不支持多进程
（已足够快，原 --n-jobs 在此引擎下无效）。

作者: Claude (等价改造)
日期: 2026-10-08（2026-10-09 自 scripts/calc_dist_kdtree.py 迁入本模块）
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from .distribution import build_exclusion_pairs

try:
    from scipy.spatial import cKDTree
    HAS_KDTREE = True
except ImportError:
    HAS_KDTREE = False

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x


def _wrap_to_box(pos: np.ndarray, L64: np.ndarray) -> np.ndarray:
    """包裹进 [0, L)（仅供 cKDTree 建树；距离重算仍用原始坐标）。"""
    pos64 = pos.astype(np.float64)
    return pos64 - L64 * np.floor(pos64 / L64)


def _kdtree_candidates(coords_frame: np.ndarray, box: np.ndarray,
                       beads1: np.ndarray, beads2: np.ndarray,
                       same_type: bool, r_query: float,
                       excl_keys: np.ndarray, n_atoms: int):
    """用 cKDTree 选出候选对（排除键合对后），返回 0-based 全局索引 (gi, gj)。

    同类型：query_pairs，每对一次且局部 i<j（beads 升序 → 全局 gi<gj）；
    跨类型：tree1.query_ball_tree(tree2)，每跨类对正好一次。
    excl_keys: 0-based (min,max) 打包键 a*(n_atoms+1)+b；None 表示无排除。
    """
    L64 = box.astype(np.float64)
    tree1 = cKDTree(_wrap_to_box(coords_frame[beads1], L64), boxsize=L64)

    if same_type:
        pairs = tree1.query_pairs(r_query, output_type='ndarray')
        if len(pairs) == 0:
            return None
        gi = beads1[pairs[:, 0]]
        gj = beads1[pairs[:, 1]]
    else:
        tree2 = cKDTree(_wrap_to_box(coords_frame[beads2], L64), boxsize=L64)
        lists = tree1.query_ball_tree(tree2, r_query)
        counts = np.fromiter((len(l) for l in lists), dtype=np.int64, count=len(lists))
        if counts.sum() == 0:
            return None
        ii = np.repeat(np.arange(len(lists), dtype=np.int64), counts)
        jj = np.fromiter((j for l in lists for j in l), dtype=np.int64, count=int(counts.sum()))
        gi = beads1[ii]
        gj = beads2[jj]

    if excl_keys is not None and len(excl_keys) > 0:
        kmin = np.minimum(gi, gj)
        kmax = np.maximum(gi, gj)
        keys = kmin * (n_atoms + 1) + kmax
        keep = ~np.isin(keys, excl_keys)
        gi = gi[keep]
        gj = gj[keep]

    if len(gi) == 0:
        return None
    return gi, gj


def calculate_rdf_kdtree_f32(
    coords_all: np.ndarray,
    box_all: np.ndarray,
    bead_info_arr: np.ndarray,
    type1: int,
    type2: int,
    exclusion_12=None,
    exclusion_13=None,
    exclusion_14=None,
    n_bins: int = 200,
    r_range: Tuple[float, float] = (0.0, 15.0),
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """KDTree 版逐帧 RDF —— 与 pickle_loader.calculate_rdf_frame_by_frame 逐 bin 等价。

    数学口径（dtype、表达式、归一化）逐字对齐原版 fbf 实现，
    仅 pair 生成由全对枚举替换为 cKDTree 候选 + 原样重算。
    """
    n_frames, n_atoms, _ = coords_all.shape

    # 获取指定类型的 bead（0-based 索引），与原版相同
    beads_type1 = bead_info_arr[bead_info_arr[:, 1] == type1, 0].astype(np.int64) - 1
    beads_type2 = bead_info_arr[bead_info_arr[:, 1] == type2, 0].astype(np.int64) - 1

    if len(beads_type1) == 0 or len(beads_type2) == 0:
        r = np.linspace(r_range[0], r_range[1], n_bins)
        return r, np.zeros(n_bins)

    n_ref = len(beads_type1)
    n_target = len(beads_type2)
    same_type = (type1 == type2)

    # 排除集：与原版相同的构造与 0-based 转换
    excl_keys = None
    if exclusion_12 or exclusion_13 or exclusion_14:
        all_exclusions = set()
        if exclusion_12:
            all_exclusions.update(exclusion_12)
        if exclusion_13:
            all_exclusions.update(exclusion_13)
        if exclusion_14:
            all_exclusions.update(exclusion_14)
        exclusions_0based = set((a - 1, b - 1) if a < b else (b - 1, a - 1)
                                for a, b in all_exclusions)
        excl_keys = np.array(sorted(a * (n_atoms + 1) + b for a, b in exclusions_0based),
                             dtype=np.int64)

    # 安全余量：>> 包裹浮点偏差，<< bin 宽
    r_query = r_range[1] + 1e-4

    if verbose:
        print(f"  RDF {type1}-{type2}: KDTree 模式 (n_ref={n_ref}, n_target={n_target}, "
              f"r_query={r_query:.4f})")

    hist_sum = np.zeros(n_bins, dtype=np.float64)
    frame_iterator = range(n_frames)
    if verbose and HAS_TQDM:
        frame_iterator = tqdm(frame_iterator, desc=f"  RDF {type1}-{type2}", unit="frame")

    for frame in frame_iterator:
        coords_frame = coords_all[frame]
        box = box_all[frame]

        cand = _kdtree_candidates(coords_frame, box, beads_type1, beads_type2,
                                  same_type, r_query, excl_keys, n_atoms)
        if cand is None:
            continue
        gi, gj = cand

        # 与原版 fbf 完全相同的距离表达式（float32）
        pos1 = coords_frame[gi, :]
        pos2 = coords_frame[gj, :]
        delta = pos2 - pos1
        delta -= box * np.round(delta / box)
        distances = np.sqrt(np.sum(delta ** 2, axis=1))

        hist, _ = np.histogram(distances, bins=n_bins, range=r_range)
        hist_sum += hist

    # 同类型：原版有序对双重计数，query_pairs 每对一次 → ×2
    if same_type:
        hist_sum *= 2

    # 以下归一化与 pickle_loader.calculate_rdf_frame_by_frame 逐字一致
    r = (np.linspace(r_range[0], r_range[1], n_bins + 1)[:-1] +
         np.linspace(r_range[0], r_range[1], n_bins + 1)[1:]) / 2
    dr = (r_range[1] - r_range[0]) / n_bins

    if normalize:
        avg_volume = np.mean(np.prod(box_all, axis=1))

        if type1 == type2:
            total_pairs = n_ref * (n_ref - 1)
        else:
            total_pairs = n_ref * n_target

        pair_density = total_pairs / avg_volume
        shell_volume = 4 * np.pi * r ** 2 * dr
        ideal_pairs = shell_volume * pair_density

        g_r = np.zeros_like(hist_sum, dtype=np.float64)
        valid_mask = ideal_pairs > 1e-9
        g_r[valid_mask] = (hist_sum[valid_mask] / n_frames) / ideal_pairs[valid_mask]
    else:
        g_r = hist_sum

    return r, g_r


def calculate_rdf_kdtree_f64(
    cg_data: Dict,
    bead_info: pd.DataFrame,
    type1: int,
    type2: int,
    bonds_df: pd.DataFrame = None,
    angles_df: pd.DataFrame = None,
    dihedrals_df: pd.DataFrame = None,
    exclude_bonds: bool = True,
    exclude_angles: bool = True,
    exclude_dihedrals: bool = True,
    n_bins: int = 100,
    custom_range: Tuple[float, float] = None,
    show_progress: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """KDTree 版 RDF —— 与 distribution.calculate_rdf 逐 bin 等价（lammpstrj 路径）。

    数学口径（float64、np.linalg.norm、edges、归一化）逐字对齐原版，
    仅 pair 生成替换为 cKDTree 候选 + 原样重算；不支持多进程（串行已足够快）。
    """
    beads_type1 = bead_info[bead_info['bead_type'] == type1]['bead_id'].values.astype(np.int64)
    beads_type2 = bead_info[bead_info['bead_type'] == type2]['bead_id'].values.astype(np.int64)

    if len(beads_type1) == 0 or len(beads_type2) == 0:
        return np.array([]), np.array([])

    n_ref = len(beads_type1)
    n_target = len(beads_type2)
    same_type = (type1 == type2)
    n_atoms = int(bead_info['bead_id'].max())

    # 排除集：与原版同一构造函数（1-based (min,max)），转 0-based 打包键
    excluded_pairs = build_exclusion_pairs(
        bonds_df if bonds_df is not None else pd.DataFrame(),
        angles_df if angles_df is not None else pd.DataFrame(),
        dihedrals_df if dihedrals_df is not None else pd.DataFrame(),
        exclude_bonds, exclude_angles, exclude_dihedrals
    )
    excl_keys = None
    if excluded_pairs:
        excl_keys = np.array(sorted((a - 1) * (n_atoms + 1) + (b - 1) for a, b in excluded_pairs),
                             dtype=np.int64)

    # bins 设置与原版相同
    first_frame = list(cg_data['cell'].keys())[0]
    box_lengths = cg_data['cell'][first_frame]
    r_max = np.min(box_lengths) / 2.0 if custom_range is None else custom_range[1]
    r_min = 0.0 if custom_range is None else custom_range[0]
    edges = np.linspace(r_min, r_max, n_bins + 1)

    r_query = r_max + 1e-4

    frame_keys = sorted(cg_data['R'].keys())
    num_frames = len(frame_keys)
    pair_desc = f"RDF {type1}-{type2} (KDTree)"

    idx1 = beads_type1 - 1  # 0-based
    idx2 = beads_type2 - 1

    hist_sum = None
    frame_iter = tqdm(frame_keys, desc=pair_desc, unit="frame") if show_progress and HAS_TQDM else frame_keys

    for frame_idx in frame_iter:
        coords = cg_data['R'][frame_idx]
        box = cg_data['cell'][frame_idx]

        cand = _kdtree_candidates(coords, box, idx1, idx2,
                                  same_type, r_query, excl_keys, n_atoms)
        if cand is None:
            continue
        gi, gj = cand

        # 与原版 distribution.calculate_rdf 完全相同的距离表达式（float64）
        pos1 = coords[gi]
        pos2 = coords[gj]
        delta = pos2 - pos1
        delta -= box * np.round(delta / box)
        distances = np.linalg.norm(delta, axis=-1)

        hist, _ = np.histogram(distances, bins=edges)
        if hist_sum is None:
            hist_sum = np.zeros_like(hist, dtype=np.float64)
        hist_sum += hist

    if hist_sum is None:
        hist_sum = np.zeros(n_bins, dtype=np.float64)

    if same_type:
        hist_sum *= 2

    # 以下归一化与 distribution.calculate_rdf 逐字一致
    r = (edges[:-1] + edges[1:]) / 2
    dr = edges[1] - edges[0]

    avg_volume = np.mean([np.prod(cg_data['cell'][i]) for i in cg_data['cell'].keys()])

    if type1 == type2:
        total_pairs = n_ref * (n_ref - 1)
    else:
        total_pairs = n_ref * n_target

    pair_density = total_pairs / avg_volume
    shell_volume = 4 * np.pi * r ** 2 * dr
    ideal_pairs = shell_volume * pair_density

    g_r = np.zeros_like(hist_sum, dtype=np.float64)
    valid_mask = ideal_pairs > 1e-9
    g_r[valid_mask] = (hist_sum[valid_mask] / num_frames) / ideal_pairs[valid_mask]

    return r, g_r
