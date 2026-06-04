"""LmpPy 后处理数据加载模块。

提供两个核心功能:
  1. 从 reaction_frames.npz 重建 reaction_details.csv
  2. 从 final_frame.data 解析链拓扑计算链长分布
"""

from __future__ import annotations

import csv
import glob
import os
import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import yaml

import numpy as np
import pandas as pd
from tqdm import tqdm

from LmpPy.utils.graph_utils import find_molecules


# ======================================================================
# 工具函数
# ======================================================================

TMPDIR_PREFIX = "lmppy_loader_"


def _cleanup_stale_tmpdirs() -> None:
    """清理之前异常退出遗留的临时目录。"""
    for pattern in [f"/tmp/{TMPDIR_PREFIX}*", f"/tmp/claude-*/{TMPDIR_PREFIX}*"]:
        for d in glob.glob(pattern):
            if os.path.isdir(d):
                try:
                    shutil.rmtree(d, ignore_errors=True)
                except Exception:
                    pass


# ======================================================================
# reaction_details.csv 重建
# ======================================================================

def _load_reaction_bead_types(config_dir: Path) -> Dict[Tuple[int, int], str]:
    """扫描 reactions 配置目录，提取每个反应模板的 bead_type 对 → reaction_name 映射。

    参数
    ----
    config_dir : Path
        包含 reactions/ 子目录的配置目录

    返回
    ----
    Dict[(bead_type_a, bead_type_b), reaction_name]
        键为排序后的 bead_type 对，值为反应名称
    """
    reactions_dir = config_dir / "reactions"
    bead_pair_to_name: Dict[Tuple[int, int], str] = {}

    if not reactions_dir.exists():
        return bead_pair_to_name

    for rxn_dir in sorted(reactions_dir.iterdir()):
        if not rxn_dir.is_dir():
            continue
        rxn_name = rxn_dir.name

        # 读取 pre_cg_mapping: 所有 *pre_mapping.yaml 文件
        for mapping_file in sorted(rxn_dir.glob("*pre_mapping.yaml")):
            try:
                with open(mapping_file, "r") as f:
                    mapping_data = yaml.safe_load(f)
            except Exception:
                continue

            if not mapping_data or "mapping" not in mapping_data:
                continue

            # 收集所有 bead_type
            bead_types: List[int] = []
            for bead_info in mapping_data["mapping"].values():
                if isinstance(bead_info, dict) and "bead_type" in bead_info:
                    bead_types.append(int(bead_info["bead_type"]))

            # 通常 pre_mapping 包含反应位点的所有 bead，我们需要识别
            # 哪些 bead 会参与反应（形成新键）。
            # 简化策略: 使用所有 bead_type 组合，实际匹配时
            # 检查反应原子的 bead_type 是否都在这个集合中
            # 存储为 (set_of_bead_types, reaction_name)
            # 但为了高效查找，使用 frozenset
            if len(bead_types) >= 2:
                # 取最小的两个不同 bead_type（通常反应涉及两种不同类型的 bead）
                unique_types = sorted(set(bead_types))
                # 对每个可能的 bead_type pair 注册
                for i in range(len(unique_types)):
                    for j in range(i + 1, len(unique_types)):
                        key = (unique_types[i], unique_types[j])
                        if key not in bead_pair_to_name:
                            bead_pair_to_name[key] = rxn_name

    return bead_pair_to_name


def _determine_reaction_type(
    bead_types_reacting: np.ndarray,
    bead_pair_map: Dict[Tuple[int, int], str],
) -> str:
    """根据反应原子的 bead_type 对确定反应类型。

    参数
    ----
    bead_types_reacting : (k, 2) 数组，每行是一对反应原子的 bead_type
    bead_pair_map : bead_type 对 → reaction_name 映射

    返回
    ----
    (k,) 字符串数组，每行是 reaction_name 或 "unknown"
    """
    result = np.array(["unknown"] * len(bead_types_reacting), dtype=object)
    for idx, (bt1, bt2) in enumerate(bead_types_reacting):
        key = (min(bt1, bt2), max(bt1, bt2))
        if key in bead_pair_map:
            result[idx] = bead_pair_map[key]
    return result


def rebuild_reaction_details(
    npz_path: str | Path,
    output_csv: str | Path,
    config_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """从 reaction_frames.npz 重建 reaction_details.csv。

    两阶段处理策略:
      1. 先从 zip 顺序读 bonds → 预计算所有键差（避免 object 数组随机访问）
      2. 再流式顺序读取每个大数组（coords/types/mapping），逐帧填充数据
    确保 HDD 场景下每个文件只顺序读一次。

    参数
    ----
    npz_path : reaction_frames.npz 文件路径
    output_csv : 输出 CSV 路径
    config_dir : reactions 配置目录，用于推断 reaction_type（可选）

    返回
    ----
    pd.DataFrame，与 mlcgsim reaction_details.csv 格式一致
    """
    import zipfile, io as _io, time as _time, re

    _cleanup_stale_tmpdirs()  # 清理上次异常退出可能遗留的临时文件

    npz_path = Path(npz_path)
    output_csv = Path(output_csv)
    t_start = _time.time()

    # 加载 reaction_type 映射
    bead_pair_map: Dict[Tuple[int, int], str] = {}
    if config_dir is not None:
        config_dir = Path(config_dir)
        bead_pair_map = _load_reaction_bead_types(config_dir)
        if bead_pair_map:
            print(f"[loader] 加载反应模板: {list(bead_pair_map.values())}")
        else:
            print("[loader] 未找到反应模板，reaction_type 将设为 'unknown'")

    # ---- 阶段 1: 从 zip 顺序读取 bonds + 预计算所有键差 ----
    print(f"[loader] 阶段 1/3: 加载 bonds 并计算键差...")
    t0 = _time.time()
    with zipfile.ZipFile(npz_path, 'r') as zf:
        # 获取帧数
        with zf.open('aa_bonds_before.npy') as f:
            header = b''
            while True:
                ch = f.read(1)
                if ch == b'\n': break
                header += ch
            shape_m = re.search(rb"'shape':\s*\((\d+),\)", header)
            n_frames = int(shape_m.group(1)) if shape_m else 0

        # 顺序读 bonds
        bb_bytes = zf.read('aa_bonds_before.npy')
        ba_bytes = zf.read('aa_bonds_after.npy')

    aa_bonds_before = np.load(_io.BytesIO(bb_bytes), allow_pickle=True)
    aa_bonds_after = np.load(_io.BytesIO(ba_bytes), allow_pickle=True)
    print(f"[loader]   bonds 加载: {_time.time()-t0:.1f}s, 帧数: {n_frames}")

    # 预计算所有帧的键差，收集索引
    frame_a1_parts = []
    frame_a2_parts = []
    frame_cycle_parts = []
    frame_idx_parts = []
    bond_a1_parts = []
    bond_a2_parts = []

    # 从第一帧推断 n_atoms（需要大数组 shape，从 aa_types 获取）
    with zipfile.ZipFile(npz_path, 'r') as zf:
        with zf.open('aa_types.npy') as f:
            header = b''
            while True:
                ch = f.read(1)
                if ch == b'\n': break
                header += ch
            sh = re.search(rb"'shape':\s*\((\d+),\s*(\d+)\)", header)
            n_atoms = int(sh.group(2)) if sh else 309000
    max_atom_id = n_atoms

    print(f"[loader]   原子数: {n_atoms}, 计算键差中...")

    for i in tqdm(range(n_frames), desc="  键差分析"):
        bb = aa_bonds_before[i]
        ba = aa_bonds_after[i]
        if len(bb) == 0 and len(ba) == 0:
            continue

        bb_atoms = bb[:, 1:].astype(np.int64)
        ba_atoms = ba[:, 1:].astype(np.int64)

        bb_min = np.minimum(bb_atoms[:, 0], bb_atoms[:, 1])
        bb_max = np.maximum(bb_atoms[:, 0], bb_atoms[:, 1])
        ba_min = np.minimum(ba_atoms[:, 0], ba_atoms[:, 1])
        ba_max = np.maximum(ba_atoms[:, 0], ba_atoms[:, 1])

        before_enc = bb_min * max_atom_id + bb_max
        after_enc = ba_min * max_atom_id + ba_max

        new_mask = ~np.isin(after_enc, before_enc)
        n_new = new_mask.sum()
        if n_new == 0:
            continue

        new_bonds = ba_atoms[new_mask]
        cycle = i + 1
        a1_idx = new_bonds[:, 0] - 1
        a2_idx = new_bonds[:, 1] - 1

        frame_a1_parts.append(a1_idx)
        frame_a2_parts.append(a2_idx)
        frame_cycle_parts.append(np.full(n_new, cycle, dtype=np.int32))
        frame_idx_parts.append(np.full(n_new, i, dtype=np.int32))
        bond_a1_parts.append(new_bonds[:, 0].astype(np.int32))
        bond_a2_parts.append(new_bonds[:, 1].astype(np.int32))

    if not frame_a1_parts:
        print("[loader] 无反应事件，写入空 CSV")
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(output_csv, "w") as f:
            f.write("cycle,atom1_id,atom2_id,atom1_type_before,atom2_type_before,atom1_type_after,atom2_type_after,distance,reaction_type,n_angles,n_dihedrals\n")
        return pd.DataFrame()

    frame_a1_arr = np.concatenate(frame_a1_parts)
    frame_a2_arr = np.concatenate(frame_a2_parts)
    frame_cycle_arr = np.concatenate(frame_cycle_parts)
    frame_idx_arr = np.concatenate(frame_idx_parts)
    bond_a1_arr = np.concatenate(bond_a1_parts)
    bond_a2_arr = np.concatenate(bond_a2_parts)
    n_total = len(frame_a1_arr)

    print(f"[loader]   共 {n_total} 个反应事件, 用时 {_time.time()-t0:.1f}s")

    # ---- 阶段 2: 提取大数组到临时目录 (NVMe)，然后 mmap 流式读取 ----
    import tempfile, shutil
    tmpdir = tempfile.mkdtemp(prefix=TMPDIR_PREFIX)
    print(f"[loader] 阶段 2/3: 提取大数组到 {tmpdir} ...")

    distances = np.zeros(n_total, dtype=np.float64)
    types_a1_after = np.zeros(n_total, dtype=np.int32)
    types_a2_after = np.zeros(n_total, dtype=np.int32)
    bead_t1_arr = np.zeros(n_total, dtype=np.int32)
    bead_t2_arr = np.zeros(n_total, dtype=np.int32)

    try:
        with zipfile.ZipFile(npz_path, 'r') as zf:
            # 提取三个大数组
            for member_name in ['aa_coords_before.npy', 'aa_types.npy',
                                'cg_mapping_before.npy']:
                t0 = _time.time()
                zf.extract(member_name, path=tmpdir)
                print(f"[loader]   提取 {member_name}: {_time.time()-t0:.1f}s")

        # mmap 每个文件，逐帧流式处理
        # 2a: 坐标 → 距离
        t0 = _time.time()
        coords_full = np.load(os.path.join(tmpdir, 'aa_coords_before.npy'), mmap_mode='r')
        for i in tqdm(range(n_frames), desc="  距离计算"):
            mask = frame_idx_arr == i
            if mask.sum() == 0: continue
            coords = coords_full[i]
            a1 = frame_a1_arr[mask]
            a2 = frame_a2_arr[mask]
            vec = coords[a1] - coords[a2]
            distances[mask] = np.sqrt(np.sum(vec * vec, axis=1))
        print(f"[loader]   距离: {_time.time()-t0:.1f}s")
        del coords_full

        # 2b: 原子类型
        t0 = _time.time()
        types_full = np.load(os.path.join(tmpdir, 'aa_types.npy'), mmap_mode='r')
        for i in tqdm(range(n_frames), desc="  类型查询"):
            mask = frame_idx_arr == i
            if mask.sum() == 0: continue
            atypes = types_full[i]
            types_a1_after[mask] = atypes[frame_a1_arr[mask]]
            types_a2_after[mask] = atypes[frame_a2_arr[mask]]
        print(f"[loader]   类型: {_time.time()-t0:.1f}s")
        del types_full

        # 2c: CG mapping → bead_type
        t0 = _time.time()
        cg_full = np.load(os.path.join(tmpdir, 'cg_mapping_before.npy'), mmap_mode='r')
        for i in tqdm(range(n_frames), desc="  bead类型"):
            mask = frame_idx_arr == i
            if mask.sum() == 0: continue
            cg_map = cg_full[i]
            bead_types = cg_map[:, 2].astype(np.int32)
            bead_t1_arr[mask] = bead_types[frame_a1_arr[mask]]
            bead_t2_arr[mask] = bead_types[frame_a2_arr[mask]]
        print(f"[loader]   bead: {_time.time()-t0:.1f}s")
        del cg_full
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ---- 阶段 3: 推断 reaction_type + 写 CSV ----
    print("[loader] 阶段 3/3: 推断类型并写 CSV...")
    t0 = _time.time()
    rxn_types = _determine_reaction_type(
        np.column_stack([bead_t1_arr, bead_t2_arr]), bead_pair_map
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cycle", "atom1_id", "atom2_id",
            "atom1_type_before", "atom2_type_before",
            "atom1_type_after", "atom2_type_after",
            "distance", "reaction_type", "pair_type", "n_angles", "n_dihedrals",
        ])
        for j in tqdm(range(n_total), desc="  写CSV"):
            bt1 = int(bead_t1_arr[j])
            bt2 = int(bead_t2_arr[j])
            pair_type = f"{min(bt1, bt2)}-{max(bt1, bt2)}"
            writer.writerow([
                int(frame_cycle_arr[j]),
                int(bond_a1_arr[j]),
                int(bond_a2_arr[j]),
                bt1,
                bt2,
                int(types_a1_after[j]),
                int(types_a2_after[j]),
                round(float(distances[j]), 6),
                str(rxn_types[j]),
                pair_type,
                0, 0,
            ])

    print(f"[loader] 完成: {n_total} 条记录 → {output_csv} (总耗时 {_time.time()-t_start:.1f}s)")

    return pd.read_csv(output_csv)


# ======================================================================
# final_frame.data 解析
# ======================================================================

def load_final_frame_topology(input_dir: str | Path) -> Dict:
    """解析 final_frame.data 文件，提取原子和键信息并计算链长。

    解析 LAMMPS write_data 命令生成的 data 文件，
    通过图搜索 (BFS) 识别连通分子并计算各分子的链长。

    参数
    ----
    input_dir : 包含 final_frame.data 的目录路径

    返回
    ----
    dict with keys:
        atoms : (n_atoms, 6) int 数组 [atom_id, mol_id, type, x, y, z]
        bonds : (n_bonds, 3) int 数组 [bond_type, atom1, atom2]
        chain_lengths : (n_molecules,) int 数组，每个分子的链长
        box_bounds : (3, 2) float 数组 [xlo,xhi; ylo,yhi; zlo,zhi]
    """
    input_dir = Path(input_dir)
    data_file = input_dir / "final_frame.data"
    if not data_file.exists():
        raise FileNotFoundError(f"未找到 final_frame.data: {data_file}")

    with open(data_file, "r") as f:
        lines = f.readlines()

    # 扫描模式：记录各节起始行
    section_starts: Dict[str, int] = {}
    box_line_start = -1

    for idx, line in enumerate(lines):
        ls = line.strip()
        if not ls or ls.startswith("#"):
            continue

        # 检测节头 (使用 startswith 兼容 "Atoms # molecular" 等注释)
        for sec in ["Masses", "Atoms", "Velocities", "Bonds", "Angles", "Dihedrals"]:
            if ls.startswith(sec) and sec not in section_starts:
                section_starts[sec] = idx + 1  # 数据从下一行开始
                break

        if "xlo" in ls and "xhi" in ls and box_line_start < 0:
            box_line_start = idx

    # 确定各节结束位置
    sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])
    section_ends: Dict[str, int] = {}
    for i, (name, start) in enumerate(sorted_sections):
        if i + 1 < len(sorted_sections):
            section_ends[name] = sorted_sections[i + 1][1] - 1
        else:
            section_ends[name] = len(lines)

    def _parse_section(start_line: int, end_line: int, ncols: int) -> list:
        """通用解析：读取行，取前 ncols 列。"""
        rows = []
        for line in lines[start_line:end_line]:
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            parts = ls.split()
            if len(parts) >= ncols:
                rows.append([float(p) if "." in p or "e" in p.lower() else int(p)
                             for p in parts[:ncols]])
        return rows

    # 解析 Atoms
    atoms = []
    if "Atoms" in section_starts:
        atoms = _parse_section(section_starts["Atoms"], section_ends["Atoms"], 6)
    atoms_arr = np.array(atoms, dtype=np.float64) if atoms else np.zeros((0, 6), dtype=np.float64)

    # 解析 Bonds（LAMMPS write_data 输出 4 列: bond_id bond_type atom1 atom2）
    bonds = []
    if "Bonds" in section_starts:
        bonds_raw = _parse_section(section_starts["Bonds"], section_ends["Bonds"], 4)
        # 去掉 bond_id 列，保留 [bond_type, atom1, atom2]
        bonds = [row[1:] for row in bonds_raw]
    bonds_arr = np.array(bonds, dtype=int) if bonds else np.zeros((0, 3), dtype=int)

    # 解析 Box Bounds
    box_bounds = np.zeros((3, 2))
    if box_line_start > 0:
        for i in range(3):
            line = lines[box_line_start + i].strip()
            parts = line.split()
            box_bounds[i] = [float(parts[0]), float(parts[1])]

    # 计算链长
    if len(atoms_arr) > 0 and len(bonds_arr) > 0:
        natoms = len(atoms_arr)
        # find_molecules 需要 bonds (n_bonds, 2) 或 (n_bonds, 3), 和 natoms
        mol_ids = find_molecules(bonds_arr, natoms)
        # 统计每个分子的原子数
        unique_mols, counts = np.unique(mol_ids, return_counts=True)
        chain_lengths = counts.astype(int)
    else:
        chain_lengths = np.array([], dtype=int)

    return {
        "atoms": atoms_arr,
        "bonds": bonds_arr,
        "chain_lengths": chain_lengths,
        "box_bounds": box_bounds,
    }


def load_reaction_details(
    input_dir: str | Path,
    npz_path: Optional[str | Path] = None,
    config_dir: Optional[str | Path] = None,
) -> pd.DataFrame:
    """加载 reaction_details，优先读取已有 CSV，否则从 npz 重建。

    参数
    ----
    input_dir : 包含 reaction_details.csv 或 reaction_frames.npz 的目录
    npz_path : reaction_frames.npz 路径（若不与 input_dir 相同）
    config_dir : reactions 配置目录

    返回
    ----
    pd.DataFrame with columns: cycle, atom1_id, atom2_id, ...,
        reaction_type, distance, pair_type
    """
    input_dir = Path(input_dir)
    csv_path = input_dir / "reaction_details.csv"

    if csv_path.exists():
        print(f"[loader] 加载已有 {csv_path}")
        df = pd.read_csv(csv_path)
        if len(df) > 0 and "pair_type" not in df.columns:
            types_before = df[["atom1_type_before", "atom2_type_before"]].values
            df["pair_type"] = [
                f"{min(a, b)}-{max(a, b)}" for a, b in types_before
            ]
        return df

    # 需要重建
    if npz_path is None:
        npz_candidates = [
            input_dir / "reaction_frames.npz",
            input_dir.parent / "reaction_frames.npz",
        ]
        for candidate in npz_candidates:
            if candidate.exists():
                npz_path = candidate
                break

    if npz_path is None or not Path(npz_path).exists():
        raise FileNotFoundError(
            f"未找到 reaction_details.csv 或 reaction_frames.npz"
        )

    return rebuild_reaction_details(npz_path, csv_path, config_dir=config_dir)


def load_reference_distance(ref_csv: str | Path) -> np.ndarray:
    """加载参考距离分布数据（兼容 mlcgsim 接口）。

    参数
    ----
    ref_csv : CSV 文件路径，单列距离值或 bin_center,count 两列

    返回
    ----
    距离值的一维 numpy 数组
    """
    ref_csv = Path(ref_csv)
    if not ref_csv.exists():
        raise FileNotFoundError(f"参考距离文件未找到: {ref_csv}")

    data = np.loadtxt(ref_csv, delimiter=",", ndmin=1)
    if data.ndim > 1 and data.shape[1] >= 2:
        # bin_center, count 格式 → 展平为原始距离样本
        return np.repeat(data[:, 0], data[:, 1].astype(int))
    return data.flatten()
