"""
分布计算模块

计算键长、角度、二面角和RDF分布。

功能:
- 从CG轨迹计算各种分布
- 支持LAMMPS dump和pickle格式轨迹
- 支持排除1-2/1-3/1-4键合对

作者: 整合自 md_base_on_ml/calc_ibm_pot/300DGEBA_150PPD/calculate_all_bonded_dist_dump.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import warnings


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


def pbc_distance(pos1: np.ndarray, pos2: np.ndarray, box: np.ndarray) -> np.ndarray:
    """计算PBC-aware距离"""
    delta = pos2 - pos1
    delta -= box * np.round(delta / box)
    return np.linalg.norm(delta, axis=-1)


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

        bead1 = coords[bond_pairs[0] - 1]
        bead2 = coords[bond_pairs[1] - 1]

        distances = pbc_distance(bead1, bead2, box)
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
                          exclude_dihedrals: bool = True) -> np.ndarray:
    """构建排除对列表（1-2, 1-3, 1-4）"""
    excluded_pairs = set()

    if exclude_bonds and len(bonds_df) > 0:
        for _, row in bonds_df.iterrows():
            pair = tuple(sorted((int(row['atom1_id']), int(row['atom2_id']))))
            excluded_pairs.add(pair)

    if exclude_angles and len(angles_df) > 0:
        for _, row in angles_df.iterrows():
            pair = tuple(sorted((int(row['atom1_id']), int(row['atom3_id']))))
            excluded_pairs.add(pair)

    if exclude_dihedrals and len(dihedrals_df) > 0:
        for _, row in dihedrals_df.iterrows():
            pair = tuple(sorted((int(row['atom1_id']), int(row['atom4_id']))))
            excluded_pairs.add(pair)

    return np.array(sorted(list(excluded_pairs)), dtype=np.int32) if excluded_pairs else np.array([], dtype=np.int32).reshape(0, 2)


def calculate_rdf(cg_data: Dict, bead_info: pd.DataFrame,
                  type1: int, type2: int,
                  bonds_df: pd.DataFrame = None,
                  angles_df: pd.DataFrame = None,
                  dihedrals_df: pd.DataFrame = None,
                  exclude_bonds: bool = True,
                  exclude_angles: bool = True,
                  exclude_dihedrals: bool = True,
                  n_bins: int = 100,
                  custom_range: Tuple[float, float] = None,
                  dr: float = 0.01,
                  output_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算径向分布函数(RDF)。

    Args:
        cg_data: CG轨迹数据
        bead_info: bead信息DataFrame
        type1, type2: bead类型
        bonds_df, angles_df, dihedrals_df: 拓扑DataFrame
        exclude_bonds/angles/dihedrals: 是否排除1-2/1-3/1-4对
        n_bins: bins数
        custom_range: 范围
        dr: RDF bin宽度
        output_file: 输出文件

    Returns:
        (r, g_r): 距离和RDF数组
    """
    # 获取指定类型的bead ID
    beads_type1 = bead_info[bead_info['bead_type'] == type1]['bead_id'].values.astype(np.int32)
    beads_type2 = bead_info[bead_info['bead_type'] == type2]['bead_id'].values.astype(np.int32)

    if len(beads_type1) == 0 or len(beads_type2) == 0:
        return np.array([]), np.array([])

    # 构建排除对
    excluded_pairs = build_exclusion_pairs(
        bonds_df or pd.DataFrame(),
        angles_df or pd.DataFrame(),
        dihedrals_df or pd.DataFrame(),
        exclude_bonds, exclude_angles, exclude_dihedrals
    )

    # 确定r_max
    first_frame = list(cg_data['cell'].keys())[0]
    box_lengths = cg_data['cell'][first_frame]
    r_max = np.min(box_lengths) / 2.0 if custom_range is None else custom_range[1]

    # 设置bins
    if custom_range is not None:
        edges = np.linspace(custom_range[0], custom_range[1], n_bins + 1)
    else:
        edges = np.arange(0.0, r_max + dr, dr)

    # 构建所有对
    n1, n2 = len(beads_type1), len(beads_type2)
    all_pairs = []
    for i in beads_type1:
        for j in beads_type2:
            if i != j:
                all_pairs.append((min(i, j), max(i, j)))

    # 过滤排除对
    excluded_set = set(map(tuple, excluded_pairs))
    valid_pairs = [p for p in all_pairs if p not in excluded_set]

    if not valid_pairs:
        return np.array([]), np.array([])

    # 累积RDF
    rdf_sum = np.zeros(len(edges) - 1)
    num_frames = 0

    for frame_idx in sorted(cg_data['R'].keys()):
        coords = cg_data['R'][frame_idx]
        box = cg_data['cell'][frame_idx]

        distances = []
        for p in valid_pairs:
            idx1, idx2 = p[0] - 1, p[1] - 1
            delta = coords[idx2] - coords[idx1]
            delta -= box * np.round(delta / box)
            distances.append(np.linalg.norm(delta))

        hist, _ = np.histogram(distances, bins=edges)
        rdf_sum += hist
        num_frames += 1

    # 计算g(r)
    r = (edges[:-1] + edges[1:]) / 2
    dr_actual = edges[1] - edges[0]

    # 归一化
    avg_volume = np.mean([np.prod(cg_data['cell'][i]) for i in cg_data['cell'].keys()])
    total_pairs = len(beads_type1) * len(beads_type2) if type1 != type2 else len(beads_type1) * (len(beads_type1) - 1)
    pair_density = total_pairs / avg_volume

    shell_volume = 4 * np.pi * r**2 * dr_actual
    ideal_pairs = shell_volume * pair_density

    g_r = np.zeros_like(r)
    mask = ideal_pairs > 1e-9
    g_r[mask] = (rdf_sum[mask] / num_frames) / ideal_pairs[mask]

    if output_file:
        np.savetxt(output_file, np.column_stack([r, g_r]),
                   header=f"RDF for bead types {type1}-{type2}\nr(A) g(r)")

    return r, g_r


if __name__ == "__main__":
    print("分布计算模块")
    print("使用方法:")
    print("  from LmpPy.tools.ibm_potential import distribution")
    print("  cg_data = distribution.load_cg_trajectory('traj.pkl')")
    print("  r, hist = distribution.calculate_bond_distribution(cg_data, bond_pairs)")