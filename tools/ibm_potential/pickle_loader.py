"""
Pickle 轨迹向量化加载模块

特性:
- 全轨迹预加载到大数组
- 向量化批量计算分布
- 支持 --skip-existing 跳过已完成计算

适用场景:
- Pickle 格式的 CG 轨迹（由 convert_aa2cg.py 生成）
- 大轨迹文件（>1000 帧）
- 需要高效批量处理

使用示例:
    from LmpPy.tools.ibm_potential.pickle_loader import (
        PickleTrajectoryCache,
        load_pickle_vectorized,
        calculate_bond_distribution_vectorized,
        check_existing_distributions
    )

    # 加载轨迹
    cache = load_pickle_vectorized('cg_trajectory.pkl')

    # 计算键分布
    r, hist = calculate_bond_distribution_vectorized(
        cache.coords_all, cache.box_all, bond_pairs
    )

作者: Claude
日期: 2026-05-20
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict

# 尝试导入 tqdm
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x

# 模块可用标志
HAS_PICKLE_VECTORIZED = True


@dataclass
class PickleTrajectoryCache:
    """
    Pickle 轨迹缓存 - 向量化存储

    内存策略：一次性加载全部帧，利用充足内存优势

    属性:
        coords_all: (n_frames, n_atoms, 3) 坐标数组，单位 Å，float32
        box_all: (n_frames, 3) 盒子尺寸数组，单位 Å，float32
        time_all: (n_frames,) 时间数组，单位 ps，float64
        n_frames: 帧数
        n_atoms: 原子数
        source_file: 源文件路径
    """
    coords_all: np.ndarray = field(default_factory=lambda: np.array([]))
    box_all: np.ndarray = field(default_factory=lambda: np.array([]))
    time_all: np.ndarray = field(default_factory=lambda: np.array([]))
    n_frames: int = 0
    n_atoms: int = 0
    source_file: str = ""

    def get_memory_info(self) -> Dict:
        """获取内存占用信息"""
        coords_bytes = self.coords_all.nbytes if len(self.coords_all) > 0 else 0
        box_bytes = self.box_all.nbytes if len(self.box_all) > 0 else 0
        time_bytes = self.time_all.nbytes if len(self.time_all) > 0 else 0

        return {
            'coords_mb': coords_bytes / 1e6,
            'box_mb': box_bytes / 1e6,
            'time_mb': time_bytes / 1e6,
            'total_mb': (coords_bytes + box_bytes + time_bytes) / 1e6
        }


def load_pickle_vectorized(
    pickle_file: str,
    verbose: bool = True
) -> PickleTrajectoryCache:
    """
    高效加载 pickle 轨迹为向量化数组

    策略：
    1. pickle.load 一次性读取
    2. 预分配大数组（避免增量 append）
    3. 单次遍历填充数组（无字典构建）

    Args:
        pickle_file: pickle 文件路径
        verbose: 是否输出进度信息

    Returns:
        PickleTrajectoryCache 对象

    注意：
    - 不验证数据一致性（假设 pickle 格式正确）
    - 不支持 stride（用户在生成 pickle 时控制帧数）
    """
    pickle_path = Path(pickle_file)

    if not pickle_path.exists():
        raise FileNotFoundError(f"Pickle 文件不存在: {pickle_file}")

    # 加载 pickle
    with open(pickle_file, 'rb') as f:
        cg_trajectory = pickle.load(f)

    # 获取基本信息
    n_frames = len(cg_trajectory)
    n_atoms = len(cg_trajectory[0]['coords'])

    if verbose:
        print(f"加载 Pickle 轨迹:")
        print(f"  文件: {pickle_file}")
        print(f"  帧数: {n_frames}")
        print(f"  原子数: {n_atoms}")
        print(f"  预估内存占用: {n_frames * n_atoms * 3 * 4 / 1e6:.2f} MB (坐标)")

    # 预分配大数组（关键优化）
    coords_all = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
    box_all = np.empty((n_frames, 3), dtype=np.float32)
    time_all = np.empty((n_frames,), dtype=np.float64)

    # 单次遍历填充
    iterator = tqdm(enumerate(cg_trajectory), total=n_frames,
                    desc="转换轨迹", unit="frame") if verbose and HAS_TQDM else enumerate(cg_trajectory)

    for i, frame_data in iterator:
        # 坐标转换
        coords_all[i] = frame_data['coords'].astype(np.float32)

        # 盒子尺寸
        box = frame_data['box']
        if box.ndim == 2:
            box_all[i] = (box[:, 1] - box[:, 0]).astype(np.float32)
        else:
            box_all[i] = box.astype(np.float32)

        # 时间
        time_all[i] = frame_data.get('time', i)

    if verbose:
        cache = PickleTrajectoryCache(
            coords_all=coords_all,
            box_all=box_all,
            time_all=time_all,
            n_frames=n_frames,
            n_atoms=n_atoms,
            source_file=str(pickle_path)
        )
        mem_info = cache.get_memory_info()
        print(f"加载完成:")
        print(f"  实际内存占用: ~{mem_info['total_mb']:.2f} MB")
        print(f"  时间范围: {time_all[0]:.2f} - {time_all[-1]:.2f} ps")

    return PickleTrajectoryCache(
        coords_all=coords_all,
        box_all=box_all,
        time_all=time_all,
        n_frames=n_frames,
        n_atoms=n_atoms,
        source_file=str(pickle_path)
    )


def check_existing_distributions(
    output_dir: Path,
    bond_types: List[int],
    angle_types: List[int],
    dihedral_types: List[int],
    bead_types: List[int]
) -> Dict[str, List[str]]:
    """
    检查输出目录已有的分布文件

    Args:
        output_dir: 输出目录路径
        bond_types: 键类型列表
        angle_types: 角度类型列表
        dihedral_types: 二面角类型列表
        bead_types: Bead 类型列表

    Returns:
        {'bond': ['bond_type1.dist.tgt', ...],
         'angle': [...],
         'dihedral': [...],
         'pair': [...]}
    """
    existing = defaultdict(list)

    # Bond 文件
    for bond_type in bond_types:
        file_path = output_dir / f"bond_type{bond_type}.dist.tgt"
        if file_path.exists():
            existing['bond'].append(file_path.name)

    # Angle 文件
    for angle_type in angle_types:
        file_path = output_dir / f"angle_type{angle_type}.dist.tgt"
        if file_path.exists():
            existing['angle'].append(file_path.name)

    # Dihedral 文件
    for dihedral_type in dihedral_types:
        file_path = output_dir / f"dihedral_type{dihedral_type}.dist.tgt"
        if file_path.exists():
            existing['dihedral'].append(file_path.name)

    # Pair 文件（bead type 组合）
    for i, t1 in enumerate(bead_types):
        for t2 in bead_types[i:]:
            file_path = output_dir / f"pair_type{t1}_{t2}.dist.tgt"
            if file_path.exists():
                existing['pair'].append(file_path.name)

    return existing


def print_existing_summary(existing: Dict[str, List[str]]) -> None:
    """
    打印已存在文件摘要

    Args:
        existing: check_existing_distributions 返回的字典
    """
    if not any(len(v) > 0 for v in existing.values()):
        return

    print("\n检测到已存在的分布文件:")
    for calc_type, files in existing.items():
        if files:
            print(f"  {calc_type}: {len(files)} 个文件")
            for f in files[:3]:  # 最多显示3个
                print(f"    - {f}")
            if len(files) > 3:
                print(f"    ... 还有 {len(files)-3} 个")


# ============== 向量化计算函数 ==============

def calculate_bond_distribution_vectorized(
    coords_all: np.ndarray,
    box_all: np.ndarray,
    bond_pairs: np.ndarray,
    n_bins: int = 200,
    bond_range: Tuple[float, float] = (0.5, 6.0),
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    向量化批量计算键长分布

    Args:
        coords_all: (n_frames, n_atoms, 3) 坐标数组，单位 Å
        box_all: (n_frames, 3) 盒子尺寸数组，单位 Å
        bond_pairs: (n_bonds, 2) 键原子索引，0-based
        n_bins: 直方图 bins 数
        bond_range: 距离范围 (Å)
        normalize: 是否归一化

    Returns:
        (bin_centers, hist_normalized, distances):
        - bin_centers: bin 中心位置
        - hist_normalized: 归一化分布
        - distances: (n_frames, n_bonds) 距离数组（可选保留用于后续分析）
    """
    n_frames, n_atoms, _ = coords_all.shape
    n_bonds = len(bond_pairs)

    if n_bonds == 0:
        bin_centers = np.linspace(bond_range[0], bond_range[1], n_bins)
        return bin_centers, np.zeros(n_bins), None

    # 高级索引 - 向量化提取所有帧、所有键原子坐标
    atom1_coords = coords_all[:, bond_pairs[:, 0], :]  # (n_frames, n_bonds, 3)
    atom2_coords = coords_all[:, bond_pairs[:, 1], :]

    # 批量 PBC 距离计算
    delta = atom1_coords - atom2_coords

    # 应用 PBC（广播 box）
    box_expand = box_all[:, np.newaxis, :]  # (n_frames, 1, 3)
    delta -= box_expand * np.round(delta / box_expand)

    # 计算距离
    distances = np.sqrt(np.sum(delta**2, axis=2))  # (n_frames, n_bonds)

    # 扁平化并计算直方图
    distances_flat = distances.ravel()
    hist, bin_edges = np.histogram(distances_flat, bins=n_bins, range=bond_range)

    # 计算 bin 中心
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    if normalize:
        # 归一化为概率密度
        norm = n_frames * n_bonds * bin_width
        hist_normalized = hist / norm if norm > 0 else hist
    else:
        hist_normalized = hist

    return bin_centers, hist_normalized, distances


def calculate_angle_distribution_vectorized(
    coords_all: np.ndarray,
    box_all: np.ndarray,
    angle_triples: np.ndarray,
    n_bins: int = 180,
    angle_range: Tuple[float, float] = (0, 180),
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    向量化批量计算角度分布

    Args:
        coords_all: (n_frames, n_atoms, 3) 坐标数组
        box_all: (n_frames, 3) 盒子尺寸
        angle_triples: (n_angles, 3) 角原子索引 [a, b, c]，0-based
        n_bins: 直方图 bins 数
        angle_range: 角度范围 (度)
        normalize: 是否归一化

    Returns:
        (bin_centers, hist_normalized, angles):
        - bin_centers: bin 中心位置
        - hist_normalized: 归一化分布
        - angles: (n_frames, n_angles) 角度数组
    """
    n_frames = coords_all.shape[0]
    n_angles = len(angle_triples)

    if n_angles == 0:
        bin_centers = np.linspace(angle_range[0], angle_range[1], n_bins)
        return bin_centers, np.zeros(n_bins), None

    # 高级索引提取坐标
    atom_a = coords_all[:, angle_triples[:, 0], :]  # (n_frames, n_angles, 3)
    atom_b = coords_all[:, angle_triples[:, 1], :]
    atom_c = coords_all[:, angle_triples[:, 2], :]

    # PBC 矢量计算
    box_expand = box_all[:, np.newaxis, :]

    def pbc_vector(v1, v2):
        delta = v1 - v2
        delta -= box_expand * np.round(delta / box_expand)
        return delta

    vec_ab = pbc_vector(atom_a, atom_b)
    vec_cb = pbc_vector(atom_c, atom_b)

    # 批量角度计算
    dot = np.sum(vec_ab * vec_cb, axis=2)
    norm_ab = np.linalg.norm(vec_ab, axis=2)
    norm_cb = np.linalg.norm(vec_cb, axis=2)

    # 数值稳定性处理
    cos_angle = dot / (norm_ab * norm_cb + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    angles = np.degrees(np.arccos(cos_angle))  # (n_frames, n_angles)

    # 直方图
    angles_flat = angles.ravel()
    hist, bin_edges = np.histogram(angles_flat, bins=n_bins, range=angle_range)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    if normalize:
        norm = n_frames * n_angles * bin_width
        hist_normalized = hist / norm if norm > 0 else hist
    else:
        hist_normalized = hist

    return bin_centers, hist_normalized, angles


def calculate_dihedral_distribution_vectorized(
    coords_all: np.ndarray,
    box_all: np.ndarray,
    dihedral_quads: np.ndarray,
    n_bins: int = 360,
    dihedral_range: Tuple[float, float] = (-180, 180),
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    向量化批量计算二面角分布

    Args:
        coords_all: (n_frames, n_atoms, 3) 坐标数组
        box_all: (n_frames, 3) 盒子尺寸
        dihedral_quads: (n_dihedrals, 4) 二面角原子索引 [a, b, c, d]，0-based
        n_bins: 直方图 bins 数
        dihedral_range: 二面角范围 (度)
        normalize: 是否归一化

    Returns:
        (bin_centers, hist_normalized, dihedrals):
        - bin_centers: bin 中心位置
        - hist_normalized: 归一化分布
        - dihedrals: (n_frames, n_dihedrals) 二面角数组
    """
    n_frames = coords_all.shape[0]
    n_dihedrals = len(dihedral_quads)

    if n_dihedrals == 0:
        bin_centers = np.linspace(dihedral_range[0], dihedral_range[1], n_bins)
        return bin_centers, np.zeros(n_bins), None

    # 高级索引提取坐标
    atom_a = coords_all[:, dihedral_quads[:, 0], :]
    atom_b = coords_all[:, dihedral_quads[:, 1], :]
    atom_c = coords_all[:, dihedral_quads[:, 2], :]
    atom_d = coords_all[:, dihedral_quads[:, 3], :]

    # PBC 矢量计算
    box_expand = box_all[:, np.newaxis, :]

    def pbc_vector(v1, v2):
        delta = v1 - v2
        delta -= box_expand * np.round(delta / box_expand)
        return delta

    ba = pbc_vector(atom_b, atom_a)
    bc = pbc_vector(atom_b, atom_c)
    cd = pbc_vector(atom_c, atom_d)

    # 法向量计算
    n1 = np.cross(ba, bc)
    n2 = np.cross(bc, cd)

    # 批量二面角计算（使用 atan2 方法）
    m1 = np.cross(n1, bc)

    norm_bc = np.linalg.norm(bc, axis=2)
    norm_n1 = np.linalg.norm(n1, axis=2)
    norm_n2 = np.linalg.norm(n2, axis=2)

    # 数值稳定性
    # 注意：atan2 的 y/x 必须同一量级（同除 |n1||n2|）。
    # 2026-08-18 修复：原实现 y 只除了 |bc|（量级 |n1||n2|）而 x 除了 |n1||n2|，
    # 等效把 tan(phi) 放大 |n1||n2| 倍，分布被系统性压向 ±90°
    # （CG 键长下 |n1||n2| 可达数百，几乎全部样本塌缩到 ±90°）
    y = np.sum(m1 * n2, axis=2) / (norm_bc * norm_n1 * norm_n2 + 1e-10)
    x = np.sum(n1 * n2, axis=2) / (norm_n1 * norm_n2 + 1e-10)

    dihedrals = np.degrees(np.arctan2(y, x))  # (n_frames, n_dihedrals)

    # 直方图
    dihedrals_flat = dihedrals.ravel()
    hist, bin_edges = np.histogram(dihedrals_flat, bins=n_bins, range=dihedral_range)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    if normalize:
        norm = n_frames * n_dihedrals * bin_width
        hist_normalized = hist / norm if norm > 0 else hist
    else:
        hist_normalized = hist

    return bin_centers, hist_normalized, dihedrals


def calculate_rdf_vectorized(
    coords_all: np.ndarray,
    box_all: np.ndarray,
    bead_info_arr: np.ndarray,
    type1: int,
    type2: int,
    exclusion_12: Optional[Set[Tuple[int, int]]] = None,
    exclusion_13: Optional[Set[Tuple[int, int]]] = None,
    exclusion_14: Optional[Set[Tuple[int, int]]] = None,
    n_bins: int = 200,
    r_range: Tuple[float, float] = (0.0, 15.0),
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化批量计算 RDF

    Args:
        coords_all: (n_frames, n_atoms, 3) 坐标数组
        box_all: (n_frames, 3) 盒子尺寸
        bead_info_arr: (n_beads, 2) [bead_id, bead_type]，1-based bead_id
        type1, type2: 要计算的 bead 类型
        exclusion_12/13/14: 排除列表（1-2, 1-3, 1-4 键合对）
        n_bins: 直方图 bins 数
        r_range: 距离范围 (Å)
        normalize: 是否归一化
        verbose: 是否输出进度

    Returns:
        (r, g_r): 距离数组和 RDF
    """
    n_frames, n_atoms, _ = coords_all.shape

    # 获取指定类型的 bead（0-based 索引）
    beads_type1 = bead_info_arr[bead_info_arr[:, 1] == type1, 0] - 1  # 转为 0-based
    beads_type2 = bead_info_arr[bead_info_arr[:, 1] == type2, 0] - 1

    if len(beads_type1) == 0 or len(beads_type2) == 0:
        r = np.linspace(r_range[0], r_range[1], n_bins)
        return r, np.zeros(n_bins)

    n_ref = len(beads_type1)
    n_target = len(beads_type2)

    # 构建所有 pair 组合
    ref_repeated = np.repeat(beads_type1, n_target)
    target_tiled = np.tile(beads_type2, n_ref)

    # 排除自身配对
    if type1 == type2:
        mask_self = ref_repeated != target_tiled
        valid_ref = ref_repeated[mask_self]
        valid_target = target_tiled[mask_self]
    else:
        valid_ref = ref_repeated
        valid_target = target_tiled

    # 排除键合对（转换为 0-based 比较）
    if exclusion_12 or exclusion_13 or exclusion_14:
        all_exclusions = set()
        if exclusion_12:
            all_exclusions.update(exclusion_12)
        if exclusion_13:
            all_exclusions.update(exclusion_13)
        if exclusion_14:
            all_exclusions.update(exclusion_14)

        # 转换为 0-based 排除列表
        exclusions_0based = set((a-1, b-1) if a < b else (b-1, a-1) for a, b in all_exclusions)

        # 检查每个 pair 是否在排除列表中
        mask_excluded = np.ones(len(valid_ref), dtype=bool)
        for i in range(len(valid_ref)):
            pair = (min(valid_ref[i], valid_target[i]), max(valid_ref[i], valid_target[i]))
            if pair in exclusions_0based:
                mask_excluded[i] = False

        final_ref = valid_ref[mask_excluded]
        final_target = valid_target[mask_excluded]
    else:
        final_ref = valid_ref
        final_target = valid_target

    if len(final_ref) == 0:
        r = np.linspace(r_range[0], r_range[1], n_bins)
        return r, np.zeros(n_bins)

    if verbose:
        print(f"  RDF {type1}-{type2}: {len(final_ref)} 有效 pairs")

    # 高级索引提取坐标
    coords_type1 = coords_all[:, beads_type1, :]  # (n_frames, n_ref, 3)
    coords_type2 = coords_all[:, beads_type2, :]  # (n_frames, n_target, 3)

    # 使用索引提取有效 pair 坐标
    # 需要将 final_ref/final_target 映射回 beads_type1/beads_type2 中的位置
    ref_pos = np.searchsorted(beads_type1, final_ref)
    target_pos = np.searchsorted(beads_type2, final_target)

    pos1 = coords_type1[:, ref_pos, :]  # (n_frames, n_valid_pairs, 3)
    pos2 = coords_type2[:, target_pos, :]

    # 批量距离计算
    delta = pos2 - pos1
    box_expand = box_all[:, np.newaxis, :]
    delta -= box_expand * np.round(delta / box_expand)
    distances = np.sqrt(np.sum(delta**2, axis=2))  # (n_frames, n_valid_pairs)

    # 直方图
    distances_flat = distances.ravel()
    hist, bin_edges = np.histogram(distances_flat, bins=n_bins, range=r_range)

    r = (bin_edges[:-1] + bin_edges[1:]) / 2
    dr = bin_edges[1] - bin_edges[0]

    if normalize:
        # RDF 归一化
        avg_volume = np.mean(np.prod(box_all, axis=1))

        if type1 == type2:
            total_pairs = n_ref * (n_ref - 1)
        else:
            total_pairs = n_ref * n_target

        pair_density = total_pairs / avg_volume
        shell_volume = 4 * np.pi * r**2 * dr
        ideal_pairs = shell_volume * pair_density

        g_r = np.zeros_like(hist, dtype=np.float64)
        valid_mask = ideal_pairs > 1e-9
        g_r[valid_mask] = (hist[valid_mask] / n_frames) / ideal_pairs[valid_mask]
    else:
        g_r = hist

    return r, g_r


def calculate_rdf_frame_by_frame(
    coords_all: np.ndarray,
    box_all: np.ndarray,
    bead_info_arr: np.ndarray,
    type1: int,
    type2: int,
    exclusion_12: Optional[Set[Tuple[int, int]]] = None,
    exclusion_13: Optional[Set[Tuple[int, int]]] = None,
    exclusion_14: Optional[Set[Tuple[int, int]]] = None,
    n_bins: int = 200,
    r_range: Tuple[float, float] = (0.0, 15.0),
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    逐帧计算 RDF（内存友好版本）。

    与 calculate_rdf_vectorized 使用相同的 pair 构建逻辑，
    但逐帧处理而非一次性物化所有帧的所有 pair 坐标。
    内存占用 O(n_pairs) vs 向量化的 O(n_frames × n_pairs)。

    Args:
        coords_all: (n_frames, n_atoms, 3) 坐标数组
        box_all: (n_frames, 3) 盒子尺寸
        bead_info_arr: (n_beads, 2) [bead_id, bead_type]，1-based bead_id
        type1, type2: 要计算的 bead 类型
        exclusion_12/13/14: 排除列表（1-2, 1-3, 1-4 键合对）
        n_bins: 直方图 bins 数
        r_range: 距离范围 (Å)
        normalize: 是否归一化
        verbose: 是否输出进度

    Returns:
        (r, g_r): 距离数组和 RDF
    """
    n_frames, n_atoms, _ = coords_all.shape

    # 获取指定类型的 bead（0-based 索引）
    beads_type1 = bead_info_arr[bead_info_arr[:, 1] == type1, 0] - 1
    beads_type2 = bead_info_arr[bead_info_arr[:, 1] == type2, 0] - 1

    if len(beads_type1) == 0 or len(beads_type2) == 0:
        r = np.linspace(r_range[0], r_range[1], n_bins)
        return r, np.zeros(n_bins)

    n_ref = len(beads_type1)
    n_target = len(beads_type2)

    # 构建所有 pair 组合（与向量化版本相同）
    ref_repeated = np.repeat(beads_type1, n_target)
    target_tiled = np.tile(beads_type2, n_ref)

    # 排除自身配对
    if type1 == type2:
        mask_self = ref_repeated != target_tiled
        valid_ref = ref_repeated[mask_self]
        valid_target = target_tiled[mask_self]
    else:
        valid_ref = ref_repeated
        valid_target = target_tiled

    # 排除键合对
    if exclusion_12 or exclusion_13 or exclusion_14:
        all_exclusions = set()
        if exclusion_12:
            all_exclusions.update(exclusion_12)
        if exclusion_13:
            all_exclusions.update(exclusion_13)
        if exclusion_14:
            all_exclusions.update(exclusion_14)

        exclusions_0based = set((a-1, b-1) if a < b else (b-1, a-1) for a, b in all_exclusions)

        mask_excluded = np.ones(len(valid_ref), dtype=bool)
        for i in range(len(valid_ref)):
            pair = (min(valid_ref[i], valid_target[i]), max(valid_ref[i], valid_target[i]))
            if pair in exclusions_0based:
                mask_excluded[i] = False

        final_ref = valid_ref[mask_excluded]
        final_target = valid_target[mask_excluded]
    else:
        final_ref = valid_ref
        final_target = valid_target

    if len(final_ref) == 0:
        r = np.linspace(r_range[0], r_range[1], n_bins)
        return r, np.zeros(n_bins)

    if verbose:
        print(f"  RDF {type1}-{type2}: {len(final_ref)} 有效 pairs (逐帧模式)")

    # final_ref/final_target 已经是 0-based 全局 bead 索引
    # 逐帧累计直方图
    hist_sum = np.zeros(n_bins, dtype=np.float64)
    frame_iterator = range(n_frames)
    if verbose and HAS_TQDM:
        from tqdm import tqdm
        frame_iterator = tqdm(frame_iterator, desc=f"  RDF {type1}-{type2}", unit="frame")

    for frame in frame_iterator:
        pos1 = coords_all[frame, final_ref, :]  # (n_pairs, 3)
        pos2 = coords_all[frame, final_target, :]

        delta = pos2 - pos1
        box = box_all[frame]
        delta -= box * np.round(delta / box)
        distances = np.sqrt(np.sum(delta**2, axis=1))

        hist, _ = np.histogram(distances, bins=n_bins, range=r_range)
        hist_sum += hist

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
        shell_volume = 4 * np.pi * r**2 * dr
        ideal_pairs = shell_volume * pair_density

        g_r = np.zeros_like(hist_sum, dtype=np.float64)
        valid_mask = ideal_pairs > 1e-9
        g_r[valid_mask] = (hist_sum[valid_mask] / n_frames) / ideal_pairs[valid_mask]
    else:
        g_r = hist_sum

    return r, g_r


if __name__ == "__main__":
    # 测试加载
    import sys
    if len(sys.argv) > 1:
        pickle_file = sys.argv[1]
        cache = load_pickle_vectorized(pickle_file)
        print(f"\n缓存信息:")
        print(f"  帧数: {cache.n_frames}")
        print(f"  原子数: {cache.n_atoms}")
        print(f"  内存: {cache.get_memory_info()['total_mb']:.2f} MB")