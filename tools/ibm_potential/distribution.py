"""
分布计算模块

计算键长、角度、二面角和RDF分布。

功能:
- 从CG轨迹计算各种分布
- 支持LAMMPS dump和pickle格式轨迹
- 支持排除1-2/1-3/1-4键合对
- 支持多进程并行计算
- 向量化计算提高性能

作者: 整合自 md_base_on_ml/calc_ibm_pot/300DGEBA_150PPD/calculate_all_bonded_dist_dump.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from collections import defaultdict
from functools import partial
import warnings

# 从 coordinate_utils 导入 pbc_distance，避免重复实现
from LmpPy.utils.coordinate_utils import pbc_distance

# 可选导入 tqdm 和 multiprocessing
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

try:
    from multiprocessing import Pool
    HAS_MULTIPROCESSING = True
except ImportError:
    HAS_MULTIPROCESSING = False


def load_cg_trajectory(data_path: str, format: str = 'auto') -> Dict:
    """
    加载CG轨迹数据。

    Args:
        data_path: 轨迹文件路径
        format: 格式 ('pickle', 'lammpsdump', 'auto')

    Returns:
        dict with keys:
            - 'R': {frame_idx: coords}
            - 'cell': {frame_idx: box}
    """
    data_path = Path(data_path)

    if format == 'auto':
        suffix = data_path.suffix.lower()
        if suffix == '.pkl':
            format = 'pickle'
        elif suffix in ['.lammpstrj', '.dump']:
            format = 'lammpsdump'
        else:
            raise ValueError(f"无法自动识别格式: {suffix}")

    if format == 'pickle':
        return _load_pickle_trajectory(data_path)
    elif format == 'lammpsdump':
        return _load_lammps_dump(data_path)
    else:
        raise ValueError(f"不支持的格式: {format}")


def _load_pickle_trajectory(pickle_file: str) -> Dict:
    """加载pickle格式轨迹"""
    import pickle

    with open(pickle_file, 'rb') as f:
        cg_trajectory = pickle.load(f)

    data = {'R': {}, 'cell': {}}

    for frame_data in cg_trajectory:
        frame_idx = frame_data.get('frame', len(data['R']))
        data['R'][frame_idx] = frame_data['coords']

        box_bounds = frame_data['box']
        if box_bounds.ndim == 2:
            box_lengths = box_bounds[:, 1] - box_bounds[:, 0]
        else:
            box_lengths = box_bounds
        data['cell'][frame_idx] = box_lengths

    print(f"从 {pickle_file} 加载 {len(data['R'])} 帧")
    return data


def _load_lammps_dump(dump_file: str, stride: int = 1) -> Dict:
    """加载LAMMPS dump格式轨迹"""
    frames = {'R': {}, 'cell': {}}
    frame_idx = 0
    actual_frame = 0

    with open(dump_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break

            if 'ITEM: TIMESTEP' in line:
                if actual_frame % stride != 0:
                    actual_frame += 1
                    for _ in range(8):
                        f.readline()
                    line = f.readline()
                    if 'ITEM: NUMBER OF ATOMS' in line:
                        natoms = int(f.readline())
                        for _ in range(natoms + 4):
                            f.readline()
                    continue

                f.readline()  # timestep
                f.readline()  # ITEM: NUMBER OF ATOMS
                natoms = int(f.readline())

                f.readline()  # ITEM: BOX BOUNDS
                box = np.zeros((3, 2))
                for i in range(3):
                    parts = f.readline().split()
                    box[i] = [float(parts[0]), float(parts[1])]

                f.readline()  # ITEM: ATOMS
                coords = np.zeros((natoms, 3))
                ids = np.zeros(natoms, dtype=int)

                for i in range(natoms):
                    parts = f.readline().split()
                    ids[i] = int(parts[0])
                    coords[i] = [float(parts[2]), float(parts[3]), float(parts[4])]

                sort_idx = np.argsort(ids - 1)
                coords = coords[sort_idx]

                frames['R'][frame_idx] = coords
                frames['cell'][frame_idx] = box[:, 1] - box[:, 0]

                frame_idx += 1
                actual_frame += 1

    return frames


def load_topology(bonds_file: str, angles_file: str = None,
                  dihedrals_file: str = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    加载拓扑文件。

    Args:
        bonds_file: 键文件路径
        angles_file: 角度文件路径
        dihedrals_file: 二面角文件路径

    Returns:
        (bonds_df, angles_df, dihedrals_df)
    """
    bonds_df = pd.read_csv(bonds_file, sep=r'\s+', comment='#') if Path(bonds_file).exists() else pd.DataFrame()
    angles_df = pd.read_csv(angles_file, sep=r'\s+', comment='#') if angles_file and Path(angles_file).exists() else pd.DataFrame()
    dihedrals_df = pd.read_csv(dihedrals_file, sep=r'\s+', comment='#') if dihedrals_file and Path(dihedrals_file).exists() else pd.DataFrame()

    return bonds_df, angles_df, dihedrals_df


def load_type_dict(file_path: str, n_cols: int) -> Dict:
    """加载类型字典文件"""
    type_dict = {}
    if not Path(file_path).exists():
        return type_dict

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) >= n_cols:
                type_id = int(parts[0])
                bead_types = tuple(map(int, parts[1:n_cols]))
                type_dict[type_id] = bead_types

    return type_dict


def calculate_bond_distribution(cg_data: Dict, bond_pairs: np.ndarray,
                                n_bins: int = 100,
                                custom_range: Tuple[float, float] = None,
                                output_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算键长分布。

    Args:
        cg_data: CG轨迹数据
        bond_pairs: 键原子对，形状 (2, nbonds)
        n_bins: 直方图bins数
        custom_range: 范围 (r_min, r_max)
        output_file: 输出文件路径

    Returns:
        (r, hist): 距离数组和分布数组
    """
    all_lengths = []

    for i in sorted(cg_data['R'].keys()):
        coords = cg_data['R'][i]
        box = cg_data['cell'][i]

        if box.ndim == 1:
            box = np.column_stack((np.zeros(3), box))
        elif box.ndim > 2:
            raise ValueError(f"无效的box格式: {box.shape}")

        bead1 = coords[bond_pairs[0] - 1]
        bead2 = coords[bond_pairs[1] - 1]

        delta = pbc_distance(bead1, bead2, box)
        distances = np.linalg.norm(delta, axis=-1)
        all_lengths.extend(distances)

    hist, edges = np.histogram(all_lengths, bins=n_bins, range=custom_range, density=True)
    r = (edges[:-1] + edges[1:]) / 2

    if output_file:
        np.savetxt(output_file, np.column_stack([r, hist]),
                   header=f"Bond length distribution\nr(A) P(r)")

    return r, hist


def calculate_angle_distribution(cg_data: Dict, angle_triplets: np.ndarray,
                                 n_bins: int = 100,
                                 custom_range: Tuple[float, float] = (0, 180),
                                 output_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算角度分布。

    Args:
        cg_data: CG轨迹数据
        angle_triplets: 角度原子三元组，形状 (3, nangles)
        n_bins: 直方图bins数
        custom_range: 范围 (theta_min, theta_max)
        output_file: 输出文件路径

    Returns:
        (theta, hist): 角度数组和分布数组
    """
    all_angles = []

    for i in sorted(cg_data['R'].keys()):
        coords = cg_data['R'][i]
        box = cg_data['cell'][i]

        pA = coords[angle_triplets[0] - 1]
        pB = coords[angle_triplets[1] - 1]
        pC = coords[angle_triplets[2] - 1]

        v1 = pA - pB
        v1 -= box * np.round(v1 / box)
        v2 = pC - pB
        v2 -= box * np.round(v2 / box)

        cos_theta = np.einsum('ij,ij->i', v1, v2) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        angles = np.degrees(np.arccos(cos_theta))
        all_angles.extend(angles)

    hist, edges = np.histogram(all_angles, bins=n_bins, range=custom_range, density=True)
    theta = (edges[:-1] + edges[1:]) / 2

    if output_file:
        np.savetxt(output_file, np.column_stack([theta, hist]),
                   header=f"Angle distribution\ntheta(deg) P(theta)")

    return theta, hist


def calculate_dihedral_distribution(cg_data: Dict, dihedral_quads: np.ndarray,
                                    n_bins: int = 100,
                                    custom_range: Tuple[float, float] = (-180, 180),
                                    output_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算二面角分布。

    Args:
        cg_data: CG轨迹数据
        dihedral_quads: 二面角原子四元组，形状 (4, ndihedrals)
        n_bins: 直方图bins数
        custom_range: 范围 (phi_min, phi_max)
        output_file: 输出文件路径

    Returns:
        (phi, hist): 二面角数组和分布数组
    """
    all_phis = []

    for i in sorted(cg_data['R'].keys()):
        coords = cg_data['R'][i]
        box = cg_data['cell'][i]

        p1 = coords[dihedral_quads[0] - 1]
        p2 = coords[dihedral_quads[1] - 1]
        p3 = coords[dihedral_quads[2] - 1]
        p4 = coords[dihedral_quads[3] - 1]

        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3

        for b in [b1, b2, b3]:
            b -= box * np.round(b / box)

        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)

        cos_phi = np.einsum('ij,ij->i', n1, n2) / (np.linalg.norm(n1, axis=1) * np.linalg.norm(n2, axis=1))
        cos_phi = np.clip(cos_phi, -1.0, 1.0)
        phi = np.degrees(np.arccos(cos_phi))

        sign = np.einsum('ij,ij->i', n1, b3)
        phi[sign < 0] = -phi[sign < 0]

        all_phis.extend(phi)

    hist, edges = np.histogram(all_phis, bins=n_bins, range=custom_range, density=True)
    phi = (edges[:-1] + edges[1:]) / 2

    if output_file:
        np.savetxt(output_file, np.column_stack([phi, hist]),
                   header=f"Dihedral distribution\nphi(deg) P(phi)")

    return phi, hist


def build_exclusion_pairs(bonds_df: pd.DataFrame, angles_df: pd.DataFrame,
                          dihedrals_df: pd.DataFrame,
                          exclude_bonds: bool = True,
                          exclude_angles: bool = True,
                          exclude_dihedrals: bool = True) -> Set[Tuple[int, int]]:
    """
    构建排除对列表（1-2, 1-3, 1-4键合对）。

    Args:
        bonds_df: 键 DataFrame，需包含 'atom1_id', 'atom2_id' 列
        angles_df: 角度 DataFrame，需包含 'atom1_id', 'atom3_id' 列
        dihedrals_df: 二面角 DataFrame，需包含 'atom1_id', 'atom4_id' 列
        exclude_bonds: 是否排除1-2键合对
        exclude_angles: 是否排除1-3键合对
        exclude_dihedrals: 是否排除1-4键合对

    Returns:
        排除对集合，每个元素为 (atom_id1, atom_id2) 排序后的元组
    """
    excluded_pairs = set()

    # 自动检测列名（兼容不同格式）
    def get_atom_ids(df, cols):
        if df.empty or len(df) == 0:
            return []
        # 尝试不同的列名格式
        for col_variants in cols:
            if all(c in df.columns for c in col_variants):
                return [(int(row[col_variants[0]]), int(row[col_variants[1]]))
                        for _, row in df.iterrows()]
        return []

    # 1-2 排除（键两端的原子）
    if exclude_bonds and bonds_df is not None and len(bonds_df) > 0:
        bond_pairs = get_atom_ids(bonds_df, [['atom1_id', 'atom2_id'], ['atom1', 'atom2']])
        for a1, a2 in bond_pairs:
            excluded_pairs.add(tuple(sorted((a1, a2))))

    # 1-3 排除（角度两端的原子）
    if exclude_angles and angles_df is not None and len(angles_df) > 0:
        angle_pairs = get_atom_ids(angles_df, [['atom1_id', 'atom3_id'], ['atom1', 'atom3']])
        for a1, a3 in angle_pairs:
            excluded_pairs.add(tuple(sorted((a1, a3))))

    # 1-4 排除（二面角两端的原子）
    if exclude_dihedrals and dihedrals_df is not None and len(dihedrals_df) > 0:
        dihedral_pairs = get_atom_ids(dihedrals_df, [['atom1_id', 'atom4_id'], ['atom1', 'atom4']])
        for a1, a4 in dihedral_pairs:
            excluded_pairs.add(tuple(sorted((a1, a4))))

    return excluded_pairs


def _process_frame_batch(
    frame_indices: List[int],
    cg_data: Dict,
    beads_type1: np.ndarray,
    beads_type2: np.ndarray,
    ref_indices: np.ndarray,
    target_indices: np.ndarray,
    edges: np.ndarray
) -> np.ndarray:
    """
    处理一批帧用于 RDF 计算（多进程 worker 函数）。

    Args:
        frame_indices: 帧索引列表
        cg_data: CG 轨迹数据
        beads_type1: type1 bead ID 数组
        beads_type2: type2 bead ID 数组
        ref_indices: 参考索引数组
        target_indices: 目标索引数组
        edges: 直方图 bin 边界

    Returns:
        这些帧的直方图累加结果
    """
    hist_sum = None

    for frame_idx in frame_indices:
        coords = cg_data['R'][frame_idx]
        box = cg_data['cell'][frame_idx]

        # 获取所有 bead 的坐标
        coords_type1 = coords[beads_type1 - 1]
        coords_type2 = coords[beads_type2 - 1]

        # 使用索引提取有效 pair 的坐标
        pos1 = coords_type1[ref_indices]
        pos2 = coords_type2[target_indices]

        # 向量化计算距离（PBC-aware）
        delta = pos2 - pos1
        delta -= box * np.round(delta / box)
        distances = np.linalg.norm(delta, axis=-1)

        # 累积直方图
        hist, _ = np.histogram(distances, bins=edges)

        if hist_sum is None:
            hist_sum = np.zeros_like(hist, dtype=np.float64)
        hist_sum += hist

    return hist_sum


def calculate_rdf(
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
    n_jobs: int = 1,
    show_progress: bool = True,
    output_file: str = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算径向分布函数(RDF) - 优化版本。

    使用向量化计算和多进程并行支持，性能显著提升。

    Args:
        cg_data: CG轨迹数据，格式 {'R': {frame: coords}, 'cell': {frame: box}}
        bead_info: bead信息DataFrame，需包含 'bead_id', 'bead_type' 列
        type1, type2: bead类型
        bonds_df: 键拓扑DataFrame
        angles_df: 角度拓扑DataFrame
        dihedrals_df: 二面角拓扑DataFrame
        exclude_bonds: 是否排除1-2键合对
        exclude_angles: 是否排除1-3键合对
        exclude_dihedrals: 是否排除1-4键合对
        n_bins: 直方图bins数
        custom_range: 距离范围 (r_min, r_max)
        n_jobs: 并行进程数（默认1，串行）
        show_progress: 是否显示进度条
        output_file: 输出文件路径

    Returns:
        (r, g_r): 距离数组和RDF数组
    """
    # 获取指定类型的 bead ID
    beads_type1 = bead_info[bead_info['bead_type'] == type1]['bead_id'].values.astype(np.int32)
    beads_type2 = bead_info[bead_info['bead_type'] == type2]['bead_id'].values.astype(np.int32)

    if len(beads_type1) == 0 or len(beads_type2) == 0:
        return np.array([]), np.array([])

    n_ref = len(beads_type1)
    n_target = len(beads_type2)

    # === 向量化构建所有 pair ===
    # 使用 np.repeat 和 np.tile 构建所有可能的 pair 组合
    ref_repeated = np.repeat(beads_type1, n_target)
    target_tiled = np.tile(beads_type2, n_ref)
    all_pairs = np.column_stack((ref_repeated, target_tiled))

    # 排序 pair 以便比较（确保小的 ID 在前）
    all_pairs.sort(axis=1)

    # 获取索引（用于后续坐标提取）
    ref_indices = np.repeat(np.arange(n_ref), n_target)
    target_indices = np.tile(np.arange(n_target), n_ref)

    # === 过滤无效 pair ===
    # 1. 排除自身配对 (i == j)
    mask_self = all_pairs[:, 0] != all_pairs[:, 1]

    filtered_ref_i = ref_indices[mask_self]
    filtered_target_j = target_indices[mask_self]
    filtered_pairs = all_pairs[mask_self]

    # 2. 排除键合对 (1-2, 1-3, 1-4)
    excluded_pairs = build_exclusion_pairs(
        bonds_df if bonds_df is not None else pd.DataFrame(),
        angles_df if angles_df is not None else pd.DataFrame(),
        dihedrals_df if dihedrals_df is not None else pd.DataFrame(),
        exclude_bonds, exclude_angles, exclude_dihedrals
    )

    if excluded_pairs:
        # 使用结构化数组视图进行高效过滤
        filtered_pairs_cont = np.ascontiguousarray(filtered_pairs)
        excluded_arr = np.array(sorted(list(excluded_pairs)), dtype=np.int32)
        excluded_cont = np.ascontiguousarray(excluded_arr)

        # 创建结构化视图用于高效比较
        dtype_str = f'V{filtered_pairs_cont.dtype.itemsize * filtered_pairs_cont.shape[1]}'
        filtered_view = filtered_pairs_cont.view(dtype_str).flatten()
        excluded_view = excluded_cont.view(dtype_str).flatten()

        mask_excluded = ~np.isin(filtered_view, excluded_view)

        final_ref_i = filtered_ref_i[mask_excluded]
        final_target_j = filtered_target_j[mask_excluded]
    else:
        final_ref_i = filtered_ref_i
        final_target_j = filtered_target_j

    if len(final_ref_i) == 0:
        return np.array([]), np.array([])

    # === 设置 bins ===
    first_frame = list(cg_data['cell'].keys())[0]
    box_lengths = cg_data['cell'][first_frame]

    r_max = np.min(box_lengths) / 2.0 if custom_range is None else custom_range[1]
    r_min = 0.0 if custom_range is None else custom_range[0]

    edges = np.linspace(r_min, r_max, n_bins + 1)

    # === 计算 RDF ===
    frame_keys = sorted(cg_data['R'].keys())
    num_frames = len(frame_keys)

    # 构建进度条描述
    pair_desc = f"RDF {type1}-{type2}"

    # 判断是否使用并行
    use_parallel = n_jobs > 1 and HAS_MULTIPROCESSING and num_frames >= n_jobs

    if use_parallel:
        # 并行处理
        batch_size = max(1, num_frames // n_jobs)
        frame_batches = [frame_keys[i:i + batch_size] for i in range(0, num_frames, batch_size)]

        process_func = partial(
            _process_frame_batch,
            cg_data=cg_data,
            beads_type1=beads_type1,
            beads_type2=beads_type2,
            ref_indices=final_ref_i,
            target_indices=final_target_j,
            edges=edges
        )

        with Pool(processes=n_jobs) as pool:
            if show_progress and HAS_TQDM:
                results = list(tqdm(
                    pool.imap(process_func, frame_batches),
                    total=len(frame_batches),
                    desc=pair_desc,
                    unit="batch"
                ))
            else:
                results = pool.map(process_func, frame_batches)

        hist_sum = np.sum(results, axis=0)
    else:
        # 串行处理
        hist_sum = None
        frame_iter = tqdm(frame_keys, desc=pair_desc, unit="frame") if show_progress and HAS_TQDM else frame_keys

        for frame_idx in frame_iter:
            coords = cg_data['R'][frame_idx]
            box = cg_data['cell'][frame_idx]

            # 获取坐标
            coords_type1 = coords[beads_type1 - 1]
            coords_type2 = coords[beads_type2 - 1]

            pos1 = coords_type1[final_ref_i]
            pos2 = coords_type2[final_target_j]

            # 向量化计算距离
            delta = pos2 - pos1
            delta -= box * np.round(delta / box)
            distances = np.linalg.norm(delta, axis=-1)

            hist, _ = np.histogram(distances, bins=edges)

            if hist_sum is None:
                hist_sum = np.zeros_like(hist, dtype=np.float64)
            hist_sum += hist

    # === 归一化 ===
    r = (edges[:-1] + edges[1:]) / 2
    dr = edges[1] - edges[0]

    # 计算理想气体分布
    avg_volume = np.mean([np.prod(cg_data['cell'][i]) for i in cg_data['cell'].keys()])

    if type1 == type2:
        total_pairs = n_ref * (n_ref - 1)
    else:
        total_pairs = n_ref * n_target

    pair_density = total_pairs / avg_volume
    shell_volume = 4 * np.pi * r**2 * dr
    ideal_pairs = shell_volume * pair_density

    # 计算 g(r)
    g_r = np.zeros_like(hist_sum, dtype=np.float64)
    valid_mask = ideal_pairs > 1e-9
    g_r[valid_mask] = (hist_sum[valid_mask] / num_frames) / ideal_pairs[valid_mask]

    if output_file:
        np.savetxt(output_file, np.column_stack([r, g_r]),
                   header=f"RDF for bead types {type1}-{type2}\nr(A) g(r)")

    return r, g_r


def calculate_all_distributions(
    cg_data: Dict,
    topology: Dict,
    bead_info: pd.DataFrame,
    n_bins: int = 100,
    bond_range: Tuple[float, float] = (2.0, 6.0),
    angle_range: Tuple[float, float] = (0, 180),
    dihedral_range: Tuple[float, float] = (-180, 180),
    pair_range: Tuple[float, float] = (3.0, 15.0),
    calc_pairs: bool = False,
    exclude_12: bool = True,
    exclude_13: bool = True,
    exclude_14: bool = True,
    n_jobs: int = 1,
    show_progress: bool = True
) -> Dict[str, Dict]:
    """
    计算所有类型的分布（bond, angle, dihedral, pair）。

    Args:
        cg_data: CG轨迹数据
        topology: 拓扑信息字典 {'bonds': df, 'angles': df, 'dihedrals': df}
        bead_info: bead信息DataFrame
        n_bins: 直方图bins数
        bond_range: 键距离范围 (Å)
        angle_range: 角度范围 (deg)
        dihedral_range: 二面角范围 (deg)
        pair_range: Pair距离范围 (Å)
        calc_pairs: 是否计算非键合 pair 分布
        exclude_12/13/14: 排除1-2/1-3/1-4键合对
        n_jobs: 并行进程数
        show_progress: 是否显示进度

    Returns:
        {
            'bond': {type_id: (r, hist)},
            'angle': {type_id: (theta, hist)},
            'dihedral': {type_id: (phi, hist)},
            'pair': {(type1, type2): (r, g_r)}  # 仅当 calc_pairs=True
        }
    """
    results = {
        'bond': {},
        'angle': {},
        'dihedral': {},
        'pair': {}
    }

    bonds_df = topology.get('bonds', pd.DataFrame())
    angles_df = topology.get('angles', pd.DataFrame())
    dihedrals_df = topology.get('dihedrals', pd.DataFrame())

    # 计算键分布
    if bonds_df is not None and len(bonds_df) > 0:
        # 自动检测列名
        atom_cols = None
        for col_set in [['atom1_id', 'atom2_id'], ['atom1', 'atom2']]:
            if all(c in bonds_df.columns for c in col_set):
                atom_cols = col_set
                break

        if atom_cols:
            for bond_type in sorted(bonds_df['bond_type'].unique()):
                bonds_of_type = bonds_df[bonds_df['bond_type'] == bond_type]
                bond_pairs = bonds_of_type[atom_cols].values.T

                r, hist = calculate_bond_distribution(
                    cg_data, bond_pairs,
                    n_bins=n_bins,
                    custom_range=bond_range
                )
                results['bond'][bond_type] = (r, hist)

    # 计算角度分布
    if angles_df is not None and len(angles_df) > 0:
        atom_cols = None
        for col_set in [['atom1_id', 'atom2_id', 'atom3_id'], ['atom1', 'atom2', 'atom3']]:
            if all(c in angles_df.columns for c in col_set):
                atom_cols = col_set
                break

        if atom_cols:
            for angle_type in sorted(angles_df['angle_type'].unique()):
                angles_of_type = angles_df[angles_df['angle_type'] == angle_type]
                angle_triplets = angles_of_type[atom_cols].values.T

                theta, hist = calculate_angle_distribution(
                    cg_data, angle_triplets,
                    n_bins=n_bins,
                    custom_range=angle_range
                )
                results['angle'][angle_type] = (theta, hist)

    # 计算二面角分布
    if dihedrals_df is not None and len(dihedrals_df) > 0:
        atom_cols = None
        for col_set in [['atom1_id', 'atom2_id', 'atom3_id', 'atom4_id'], ['atom1', 'atom2', 'atom3', 'atom4']]:
            if all(c in dihedrals_df.columns for c in col_set):
                atom_cols = col_set
                break

        if atom_cols:
            for dihedral_type in sorted(dihedrals_df['dihedral_type'].unique()):
                dihedrals_of_type = dihedrals_df[dihedrals_df['dihedral_type'] == dihedral_type]
                dihedral_quads = dihedrals_of_type[atom_cols].values.T

                phi, hist = calculate_dihedral_distribution(
                    cg_data, dihedral_quads,
                    n_bins=n_bins,
                    custom_range=dihedral_range
                )
                results['dihedral'][dihedral_type] = (phi, hist)

    # 计算 pair 分布 (RDF)
    if calc_pairs and bead_info is not None and len(bead_info) > 0:
        unique_types = sorted(bead_info['bead_type'].unique())

        for i, t1 in enumerate(unique_types):
            for t2 in unique_types[i:]:
                r, g_r = calculate_rdf(
                    cg_data, bead_info, t1, t2,
                    bonds_df=bonds_df,
                    angles_df=angles_df,
                    dihedrals_df=dihedrals_df,
                    exclude_bonds=exclude_12,
                    exclude_angles=exclude_13,
                    exclude_dihedrals=exclude_14,
                    n_bins=n_bins,
                    custom_range=pair_range,
                    n_jobs=n_jobs,
                    show_progress=show_progress
                )

                if len(r) > 0:
                    results['pair'][(t1, t2)] = (r, g_r)

    return results


if __name__ == "__main__":
    print("分布计算模块")
    print("使用方法:")
    print("  from LmpPy.tools.ibm_potential import distribution")
    print("  cg_data = distribution.load_cg_trajectory('traj.pkl')")
    print("  r, hist = distribution.calculate_bond_distribution(cg_data, bond_pairs)")
    print("")
    print("新增功能:")
    print("  results = distribution.calculate_all_distributions(cg_data, topology, bead_info)")
    print("  r, g_r = distribution.calculate_rdf(cg_data, bead_info, type1, type2, n_jobs=4)")