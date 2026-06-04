"""
GROMACS 轨迹加载模块 - 针对大轨迹文件的激进内存策略

特性:
- 全轨迹预加载到内存
- 向量化批量计算
- 单位转换 (nm → Å)
- 预计算拓扑索引

适用场景:
- 大轨迹文件 (~10GB)
- 充足内存可用 (~100GB)
- 需要高效批量处理

使用示例:
    from LmpPy.tools.ibm_potential.gromacs_loader import TrajectoryCache, TopologyIndex

    # 加载全轨迹
    trj_cache = TrajectoryCache()
    trj_cache.load_from_gromacs('topol.tpr', 'traj.xtc', stride=10)

    # 加载拓扑
    topo_idx = TopologyIndex()
    topo_idx.load_from_gromacs('topol.tpr')

    # 向量化计算分布
    r_bond, hist_bond = calculate_bond_distribution_vectorized(
        trj_cache.coords_all, trj_cache.box_all, topo_idx.bond_pairs
    )

作者: Claude
日期: 2026-05-19
"""

import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, List
from dataclasses import dataclass, field

# 尝试导入 MDAnalysis（可选依赖）
try:
    import MDAnalysis as mda
    HAS_MDA = True
except ImportError:
    HAS_MDA = False

# 尝试导入 tqdm（可选）
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = lambda x, **kwargs: x


@dataclass
class TrajectoryCache:
    """
    轨迹缓存 - 全量预加载

    内存策略：一次性加载全部帧，利用充足内存优势
    预估：10GB 轨迹 → 30-50GB 内存占用

    属性:
        coords_all: (n_frames, n_atoms, 3) 坐标数组，单位 Å
        box_all: (n_frames, 3) 盒子尺寸数组，单位 Å
        time_all: (n_frames,) 时间数组，单位 ps
        atom_ids: (n_atoms,) 原子 ID
        atom_types: (n_atoms,) 原子类型
        mol_ids: (n_atoms,) 分子 ID (resid)
        masses: (n_atoms,) 原子质量
    """
    coords_all: np.ndarray = field(default_factory=lambda: np.array([]))
    box_all: np.ndarray = field(default_factory=lambda: np.array([]))
    time_all: np.ndarray = field(default_factory=lambda: np.array([]))
    atom_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    atom_types: np.ndarray = field(default_factory=lambda: np.array([]))
    mol_ids: np.ndarray = field(default_factory=lambda: np.array([]))
    masses: np.ndarray = field(default_factory=lambda: np.array([]))

    n_frames: int = 0
    n_atoms: int = 0
    source_file: str = ""

    def load_from_gromacs(
        self,
        tpr_file: str,
        trj_file: str,
        stride: int = 1,
        verbose: bool = True,
        unit_conversion: float = 10.0  # nm → Å
    ) -> 'TrajectoryCache':
        """
        从 GROMACS 文件加载全轨迹

        Args:
            tpr_file: TPR 拓扑文件路径
            trj_file: XTC/TRR 轨迹文件路径
            stride: 帧间隔（跳帧），用于减少数据量
            verbose: 是否输出进度信息
            unit_conversion: 单位转换因子（默认 10.0，nm → Å）

        Returns:
            self（支持链式调用）

        Raises:
            ImportError: 如果 MDAnalysis 未安装
            FileNotFoundError: 如果文件不存在
        """
        if not HAS_MDA:
            raise ImportError(
                "需要安装 MDAnalysis: pip install MDAnalysis\n"
                "或完整安装: pip install MDAnalysis MDAnalysisTests"
            )

        tpr_path = Path(tpr_file)
        trj_path = Path(trj_file)

        if not tpr_path.exists():
            raise FileNotFoundError(f"TPR 文件不存在: {tpr_file}")
        if not trj_path.exists():
            raise FileNotFoundError(f"轨迹文件不存在: {trj_file}")

        # 创建 Universe
        u = mda.Universe(str(tpr_path), str(trj_path))

        # 计算实际帧数
        trajectory_slice = u.trajectory[::stride]
        n_frames = len(trajectory_slice)
        n_atoms = len(u.atoms)

        if verbose:
            print(f"加载 GROMACS 轨迹:")
            print(f"  TPR: {tpr_file}")
            print(f"  轨迹: {trj_file}")
            print(f"  帧数: {n_frames} (stride={stride})")
            print(f"  原子数: {n_atoms}")
            print(f"  预估内存占用: {n_frames * n_atoms * 3 * 4 / 1e9:.2f} GB (坐标)")

        # 预分配大数组
        self.coords_all = np.empty((n_frames, n_atoms, 3), dtype=np.float32)
        self.box_all = np.empty((n_frames, 3), dtype=np.float32)
        self.time_all = np.empty((n_frames,), dtype=np.float64)

        # 原子静态信息
        self.atom_ids = u.atoms.indices + 1  # 1-based
        self.atom_types = np.array(
            u.atoms.types if hasattr(u.atoms, 'types') else u.atoms.names,
            dtype=str
        )
        self.mol_ids = u.atoms.resids
        self.masses = u.atoms.masses

        self.n_frames = n_frames
        self.n_atoms = n_atoms
        self.source_file = str(trj_path)

        # 批量读取所有帧
        iterator = tqdm(enumerate(trajectory_slice),
                       total=n_frames,
                       desc="加载轨迹",
                       unit="frame") if verbose and HAS_TQDM else enumerate(trajectory_slice)

        for i, ts in iterator:
            # 坐标转换（GROMACS 使用 nm，转换为 Å）
            self.coords_all[i] = ts.positions * unit_conversion

            # 盒子尺寸（仅取长度）
            self.box_all[i] = ts.dimensions[:3] * unit_conversion

            # 时间
            self.time_all[i] = ts.time

        if verbose:
            actual_memory = self.coords_all.nbytes / 1e9
            print(f"加载完成:")
            print(f"  实际内存占用: ~{actual_memory:.2f} GB")
            print(f"  时间范围: {self.time_all[0]:.2f} - {self.time_all[-1]:.2f} ps")

        return self

    def get_frame_slice(
        self,
        frame_indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取指定帧的坐标和盒子切片

        Args:
            frame_indices: 帧索引数组

        Returns:
            (coords_slice, box_slice): 坐标和盒子切片
        """
        return self.coords_all[frame_indices], self.box_all[frame_indices]

    def get_memory_info(self) -> Dict:
        """
        获取内存占用信息

        Returns:
            内存信息字典
        """
        coords_mb = self.coords_all.nbytes / 1e6 if len(self.coords_all) > 0 else 0
        box_mb = self.box_all.nbytes / 1e6 if len(self.box_all) > 0 else 0
        time_mb = self.time_all.nbytes / 1e6 if len(self.time_all) > 0 else 0

        return {
            'coords_mb': coords_mb,
            'box_mb': box_mb,
            'time_mb': time_mb,
            'total_mb': coords_mb + box_mb + time_mb,
            'n_frames': self.n_frames,
            'n_atoms': self.n_atoms
        }


@dataclass
class TopologyIndex:
    """
    拓扑索引 - 预计算原子对映射

    内存策略：拓扑静态不变，提前构建高效查找结构

    属性:
        bond_pairs: (n_bonds, 2) 键原子索引，0-based
        angle_triples: (n_angles, 3) 角原子索引
        dihedral_quads: (n_dihedrals, 4) 二面角原子索引
        exclusion_12: 1-2 排除对集合
        exclusion_13: 1-3 排除对集合
        exclusion_14: 1-4 排除对集合
    """
    bond_pairs: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32).reshape(0, 2))
    angle_triples: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32).reshape(0, 3))
    dihedral_quads: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int32).reshape(0, 4))

    exclusion_12: Set[Tuple[int, int]] = field(default_factory=set)
    exclusion_13: Set[Tuple[int, int]] = field(default_factory=set)
    exclusion_14: Set[Tuple[int, int]] = field(default_factory=set)

    n_bonds: int = 0
    n_angles: int = 0
    n_dihedrals: int = 0
    source_file: str = ""

    def load_from_gromacs(self, tpr_file: str, verbose: bool = True) -> 'TopologyIndex':
        """
        从 TPR 文件提取拓扑信息

        Args:
            tpr_file: TPR 拓扑文件路径
            verbose: 是否输出进度信息

        Returns:
            self（支持链式调用）

        Raises:
            ImportError: 如果 MDAnalysis 未安装
            FileNotFoundError: 如果文件不存在
        """
        if not HAS_MDA:
            raise ImportError("需要安装 MDAnalysis: pip install MDAnalysis")

        tpr_path = Path(tpr_file)
        if not tpr_path.exists():
            raise FileNotFoundError(f"TPR 文件不存在: {tpr_file}")

        # 创建 Universe（仅拓扑）
        u = mda.Universe(str(tpr_path))

        self.source_file = str(tpr_path)

        # Bonds
        if hasattr(u, 'bonds') and len(u.bonds) > 0:
            self.bond_pairs = np.array(
                [b.atoms.indices for b in u.bonds],
                dtype=np.int32
            )
            self.n_bonds = len(self.bond_pairs)
        else:
            self.bond_pairs = np.array([], dtype=np.int32).reshape(0, 2)
            self.n_bonds = 0

        # Angles
        if hasattr(u, 'angles') and len(u.angles) > 0:
            self.angle_triples = np.array(
                [a.atoms.indices for a in u.angles],
                dtype=np.int32
            )
            self.n_angles = len(self.angle_triples)
        else:
            self.angle_triples = np.array([], dtype=np.int32).reshape(0, 3)
            self.n_angles = 0

        # Dihedrals
        if hasattr(u, 'dihedrals') and len(u.dihedrals) > 0:
            self.dihedral_quads = np.array(
                [d.atoms.indices for d in u.dihedrals],
                dtype=np.int32
            )
            self.n_dihedrals = len(self.dihedral_quads)
        else:
            self.dihedral_quads = np.array([], dtype=np.int32).reshape(0, 4)
            self.n_dihedrals = 0

        # 构建排除列表
        self._build_exclusions()

        if verbose:
            print(f"从 TPR 提取拓扑:")
            print(f"  文件: {tpr_file}")
            print(f"  键: {self.n_bonds}")
            print(f"  角度: {self.n_angles}")
            print(f"  二面角: {self.n_dihedrals}")
            print(f"  1-2 排除对: {len(self.exclusion_12)}")
            print(f"  1-3 排除对: {len(self.exclusion_13)}")
            print(f"  1-4 排除对: {len(self.exclusion_14)}")

        return self

    def _build_exclusions(self):
        """预计算 1-2/1-3/1-4 排除对"""
        # 1-2 排除（键两端的原子）
        self.exclusion_12 = set()
        for a, b in self.bond_pairs:
            self.exclusion_12.add((min(a, b), max(a, b)))

        # 1-3 排除（角度两端的原子）
        self.exclusion_13 = set()
        for a, b, c in self.angle_triples:
            self.exclusion_13.add((min(a, c), max(a, c)))

        # 1-4 排除（二面角两端的原子）
        self.exclusion_14 = set()
        for a, b, c, d in self.dihedral_quads:
            self.exclusion_14.add((min(a, d), max(a, d)))

    def get_topology_summary(self) -> Dict:
        """
        获取拓扑摘要信息

        Returns:
            拓扑信息字典
        """
        return {
            'n_bonds': self.n_bonds,
            'n_angles': self.n_angles,
            'n_dihedrals': self.n_dihedrals,
            'n_exclusion_12': len(self.exclusion_12),
            'n_exclusion_13': len(self.exclusion_13),
            'n_exclusion_14': len(self.exclusion_14),
        }


def create_exclusion_mask(
    atom_pairs: np.ndarray,
    exclusion_set: Set[Tuple[int, int]]
) -> np.ndarray:
    """
    创建排除掩码数组

    Args:
        atom_pairs: (n_pairs, 2) 原子对数组
        exclusion_set: 排除对集合

    Returns:
        (n_pairs,) bool 数组，True 表示需要保留（不排除）
    """
    if not exclusion_set:
        return np.ones(len(atom_pairs), dtype=bool)

    # 排序原子对以便比较
    sorted_pairs = np.sort(atom_pairs, axis=1)

    # 创建掩码
    mask = np.ones(len(atom_pairs), dtype=bool)
    for i, pair in enumerate(sorted_pairs):
        if tuple(pair) in exclusion_set:
            mask[i] = False

    return mask


# 向量化计算函数
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

    # 提取键原子坐标 - 向量化索引
    # 使用 0-based 索引直接访问
    atom1_coords = coords_all[:, bond_pairs[:, 0], :]  # (n_frames, n_bonds, 3)
    atom2_coords = coords_all[:, bond_pairs[:, 1], :]

    # 批量 PBC 距离计算
    delta = atom1_coords - atom2_coords

    # 应用 PBC（broadcast box）
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

    # 提取三原子坐标
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

    # 防止除零
    cos_angle = dot / (norm_ab * norm_cb + 1e-10)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    # 转换为角度（度）
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
        dihedral_quads: (n_dihedrals, 4) 二面角原子索引，0-based
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

    # 提取四原子坐标
    atom_a = coords_all[:, dihedral_quads[:, 0], :]
    atom_b = coords_all[:, dihedral_quads[:, 1], :]
    atom_c = coords_all[:, dihedral_quads[:, 2], :]
    atom_d = coords_all[:, dihedral_quads[:, 3], :]

    box_expand = box_all[:, np.newaxis, :]

    def pbc_vector(v1, v2):
        delta = v1 - v2
        delta -= box_expand * np.round(delta / box_expand)
        return delta

    # 四矢量
    ba = pbc_vector(atom_b, atom_a)
    bc = pbc_vector(atom_b, atom_c)
    cd = pbc_vector(atom_c, atom_d)

    # 法向量
    n1 = np.cross(ba, bc)
    n2 = np.cross(bc, cd)

    # 批量二面角计算（使用 atan2 方法）
    m1 = np.cross(n1, bc)

    dot_n1n2 = np.sum(n1 * n2, axis=2)
    dot_m1n2 = np.sum(m1 * n2, axis=2)

    norm_n1 = np.linalg.norm(n1, axis=2) + 1e-10
    norm_n2 = np.linalg.norm(n2, axis=2) + 1e-10
    norm_bc = np.linalg.norm(bc, axis=2) + 1e-10

    y = dot_m1n2 / norm_bc
    x = dot_n1n2 / (norm_n1 * norm_n2)

    # 二面角（度）
    dihedrals = np.degrees(np.arctan2(y, x))  # (n_frames, n_dihedrals)

    # 直方图
    dih_flat = dihedrals.ravel()
    hist, bin_edges = np.histogram(dih_flat, bins=n_bins, range=dihedral_range)

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
    bead_info: np.ndarray,
    type1: int,
    type2: int,
    exclusion_12: Set[Tuple[int, int]] = None,
    exclusion_13: Set[Tuple[int, int]] = None,
    exclusion_14: Set[Tuple[int, int]] = None,
    n_bins: int = 100,
    r_range: Tuple[float, float] = None,
    normalize: bool = True,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    向量化批量计算 RDF（径向分布函数）

    Args:
        coords_all: (n_frames, n_atoms, 3) 坐标数组
        box_all: (n_frames, 3) 盒子尺寸
        bead_info: (n_atoms, 2) bead 信息 [bead_id, bead_type] 或类似结构
        type1, type2: bead 类型
        exclusion_12/13/14: 排除对集合
        n_bins: bins 数
        r_range: 距离范围（默认取最小盒子边长的一半）
        normalize: 是否归一化为 g(r)
        verbose: 是否输出进度

    Returns:
        (r, g_r): 距离数组和 RDF 数组
    """
    n_frames, n_atoms, _ = coords_all.shape

    # 获取指定类型的 bead（假设 bead_info 包含 bead_type）
    # 这里需要根据实际 bead_info 格式调整
    # 简化版：假设 bead_info[:, 1] 是 bead_type
    mask_type1 = bead_info[:, 1] == type1
    mask_type2 = bead_info[:, 1] == type2

    beads_type1 = bead_info[mask_type1, 0].astype(np.int32) - 1  # 0-based
    beads_type2 = bead_info[mask_type2, 0].astype(np.int32) - 1

    n_ref = len(beads_type1)
    n_target = len(beads_type2)

    if n_ref == 0 or n_target == 0:
        return np.array([]), np.array([])

    # 构建所有 pair 组合
    ref_repeated = np.repeat(beads_type1, n_target)
    target_tiled = np.tile(beads_type2, n_ref)
    all_pairs = np.column_stack((ref_repeated, target_tiled))
    all_pairs.sort(axis=1)  # 排序

    # 索引映射
    ref_indices = np.repeat(np.arange(n_ref), n_target)
    target_indices = np.tile(np.arange(n_target), n_ref)

    # 排除自身配对
    mask_self = all_pairs[:, 0] != all_pairs[:, 1]

    # 排除键合对
    excluded_pairs = set()
    if exclusion_12:
        excluded_pairs.update(exclusion_12)
    if exclusion_13:
        excluded_pairs.update(exclusion_13)
    if exclusion_14:
        excluded_pairs.update(exclusion_14)

    if excluded_pairs:
        mask_excluded = create_exclusion_mask(all_pairs[mask_self], excluded_pairs)
        final_mask = np.zeros(len(all_pairs), dtype=bool)
        final_mask[mask_self] = mask_excluded
    else:
        final_mask = mask_self

    final_ref_i = ref_indices[final_mask]
    final_target_j = target_indices[final_mask]

    if len(final_ref_i) == 0:
        return np.array([]), np.array([])

    # 设置 bins
    if r_range is None:
        # 取平均盒子最小边长的一半
        avg_box_min = np.mean(np.min(box_all, axis=1))
        r_max = avg_box_min / 2.0
        r_range = (0.0, r_max)

    edges = np.linspace(r_range[0], r_range[1], n_bins + 1)

    # 向量化计算距离
    # 提取 bead 坐标
    coords_type1 = coords_all[:, beads_type1, :]  # (n_frames, n_ref, 3)
    coords_type2 = coords_all[:, beads_type2, :]  # (n_frames, n_target, 3)

    # 使用索引提取有效 pair 的坐标
    pos1 = coords_type1[:, final_ref_i, :]  # (n_frames, n_valid_pairs, 3)
    pos2 = coords_type2[:, final_target_j, :]

    # 批量计算距离（PBC-aware）
    delta = pos2 - pos1
    box_expand = box_all[:, np.newaxis, :]
    delta -= box_expand * np.round(delta / box_expand)
    distances = np.linalg.norm(delta, axis=2)  # (n_frames, n_valid_pairs)

    # 直方图
    distances_flat = distances.ravel()
    hist, _ = np.histogram(distances_flat, bins=edges)

    # 归一化
    r = (edges[:-1] + edges[1:]) / 2
    dr = edges[1] - edges[0]

    if normalize:
        # 计算理想气体分布
        avg_volume = np.mean(np.prod(box_all, axis=1))

        if type1 == type2:
            total_pairs = n_ref * (n_ref - 1)
        else:
            total_pairs = n_ref * n_target

        pair_density = total_pairs / avg_volume
        shell_volume = 4 * np.pi * r**2 * dr
        ideal_pairs = shell_volume * pair_density

        # 计算 g(r)
        g_r = np.zeros_like(hist, dtype=np.float64)
        valid_mask = ideal_pairs > 1e-9
        g_r[valid_mask] = (hist[valid_mask] / n_frames) / ideal_pairs[valid_mask]
    else:
        g_r = hist

    return r, g_r


def calculate_all_distributions_from_gromacs(
    tpr_file: str,
    trj_file: str,
    stride: int = 1,
    n_bins: int = 200,
    bond_range: Tuple[float, float] = (0.5, 6.0),
    angle_range: Tuple[float, float] = (0, 180),
    dihedral_range: Tuple[float, float] = (-180, 180),
    calc_rdf: bool = False,
    bead_types: List[int] = None,
    verbose: bool = True
) -> Dict:
    """
    从 GROMACS 文件计算所有分布（一站式函数）

    Args:
        tpr_file: TPR 拓扑文件
        trj_file: XTC/TRR 轨迹文件
        stride: 帧间隔
        n_bins: bins 数
        bond_range: 键距离范围 (Å)
        angle_range: 角度范围 (度)
        dihedral_range: 二面角范围 (度)
        calc_rdf: 是否计算 RDF
        bead_types: 要计算 RDF 的 bead 类型列表
        verbose: 是否输出进度

    Returns:
        分布结果字典
    """
    # 加载轨迹
    trj_cache = TrajectoryCache()
    trj_cache.load_from_gromacs(tpr_file, trj_file, stride=stride, verbose=verbose)

    # 加载拓扑
    topo_idx = TopologyIndex()
    topo_idx.load_from_gromacs(tpr_file, verbose=verbose)

    results = {}

    # 键分布
    if topo_idx.n_bonds > 0:
        if verbose:
            print("计算键分布...")
        r_bond, hist_bond, _ = calculate_bond_distribution_vectorized(
            trj_cache.coords_all, trj_cache.box_all, topo_idx.bond_pairs,
            n_bins=n_bins, bond_range=bond_range
        )
        results['bond'] = (r_bond, hist_bond)

    # 角度分布
    if topo_idx.n_angles > 0:
        if verbose:
            print("计算角度分布...")
        r_angle, hist_angle, _ = calculate_angle_distribution_vectorized(
            trj_cache.coords_all, trj_cache.box_all, topo_idx.angle_triples,
            n_bins=n_bins, angle_range=angle_range
        )
        results['angle'] = (r_angle, hist_angle)

    # 二面角分布
    if topo_idx.n_dihedrals > 0:
        if verbose:
            print("计算二面角分布...")
        r_dih, hist_dih, _ = calculate_dihedral_distribution_vectorized(
            trj_cache.coords_all, trj_cache.box_all, topo_idx.dihedral_quads,
            n_bins=n_bins, dihedral_range=dihedral_range
        )
        results['dihedral'] = (r_dih, hist_dih)

    return results


if __name__ == "__main__":
    print("GROMACS 轨迹加载模块 - 激进内存策略")
    print("")
    print("功能:")
    print("  - 全轨迹预加载到内存")
    print("  - 向量化批量计算")
    print("  - 单位转换 (nm → Å)")
    print("")
    print("使用示例:")
    print("  from LmpPy.tools.ibm_potential.gromacs_loader import TrajectoryCache, TopologyIndex")
    print("")
    print("  # 加载轨迹")
    print("  trj_cache = TrajectoryCache()")
    print("  trj_cache.load_from_gromacs('topol.tpr', 'traj.xtc', stride=10)")
    print("")
    print("  # 加载拓扑")
    print("  topo_idx = TopologyIndex()")
    print("  topo_idx.load_from_gromacs('topol.tpr')")
    print("")
    print("  # 计算分布")
    print("  r, hist = calculate_bond_distribution_vectorized(")
    print("      trj_cache.coords_all, trj_cache.box_all, topo_idx.bond_pairs)")