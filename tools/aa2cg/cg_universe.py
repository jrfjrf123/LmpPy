"""
CG 体系 → GROMACS 导出（system.gro + topol.top + cg.itp）。

复用 MD_format_transform 仓库 mdfmt 包的内存中间表示（Universe/Frame）与
write_gro；itp/top 由本模块自写薄写出器，原因：

- mdfmt.write_itp 单文件只支持一个 MoleculeType，而本模块允许多种链型共处于一个 cg.itp；
- mdfmt.write_top 把 moleculetype 内联进 top 且 gen-pairs 写死 yes，
  这里需要独立的 cg.itp + comb-rule 3 + gen-pairs no + [nonbond_params]。

mdfmt 不是 PyPI 包，默认通过 --mdfmt-path 或环境变量 MDFMT_PATH 指向
MD_format_transform 仓库根目录后延迟导入（缺省时仅 GROMACS 导出功能不可用，
不影响 LmpPy 其余功能）。

坐标/盒子单位约定：cg_data 中为 Å（LAMMPS real），写出前 ×0.1 转 nm。
力场参数由 cg_ff.load_cg_ff 完成换算，本模块不再处理单位。
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Tuple

import numpy as np

from .cg_ff import CgForceField, load_cg_ff, normalize_combo, validate_ff_completeness

_MDFMT: Optional[SimpleNamespace] = None


def ensure_mdfmt(mdfmt_path: Optional[str] = None) -> SimpleNamespace:
    """
    延迟导入 mdfmt（MD_format_transform 仓库），必要时先把其根目录加入 sys.path。

    Args:
        mdfmt_path: MD_format_transform 仓库根目录；缺省取环境变量 MDFMT_PATH

    Returns:
        SimpleNamespace，含 Universe/MoleculeType/MoleculeInstance/Atom/Bond/Angle/
        Dihedral/Frame/SimulationBox/write_gro

    Raises:
        ImportError: 未提供路径且直接导入失败，或提供路径后仍导入失败
    """
    global _MDFMT
    if _MDFMT is not None:
        return _MDFMT

    def _import():
        from mdfmt.core.universe import (
            Universe, MoleculeType, MoleculeInstance, Atom, Bond, Angle, Dihedral,
        )
        from mdfmt.core.frame import Frame, SimulationBox
        from mdfmt.io.gro import write_gro
        return SimpleNamespace(
            Universe=Universe, MoleculeType=MoleculeType, MoleculeInstance=MoleculeInstance,
            Atom=Atom, Bond=Bond, Angle=Angle, Dihedral=Dihedral,
            Frame=Frame, SimulationBox=SimulationBox, write_gro=write_gro,
        )

    try:
        _MDFMT = _import()
        return _MDFMT
    except ImportError:
        pass

    candidate = mdfmt_path or os.environ.get('MDFMT_PATH')
    if candidate:
        candidate = str(Path(candidate).resolve())
        if candidate not in sys.path:
            sys.path.insert(0, candidate)
        try:
            _MDFMT = _import()
            return _MDFMT
        except ImportError as e:
            raise ImportError(
                f"已将 {candidate} 加入 sys.path，但仍无法导入 mdfmt: {e}"
            ) from e

    raise ImportError(
        "GROMACS 导出依赖 mdfmt（MD_format_transform 仓库），其未安装为 Python 包。\n"
        "请通过 --mdfmt-path 选项或环境变量 MDFMT_PATH 指向该仓库根目录。"
    )


def _split_molecules(cg_data: Dict) -> List[List[int]]:
    """
    按 mol_id 把 bead 位置切分为分子，保持 cg_data 中的出现顺序。

    Returns:
        molecules: 每个分子一个 bead 位置列表（局部顺序 = cg_data 顺序）

    Raises:
        ValueError: 同一 mol_id 的 bead 在体系中不连续（无法在不重排坐标的情况下
                    保持 gro 原子顺序与 instances 展开顺序一致）
    """
    mol_ids = cg_data['mol_ids'].astype(int)
    molecules: List[List[int]] = []
    seen = set()
    cur_mol = None
    for pos, mid in enumerate(mol_ids):
        mid = int(mid)
        if mid != cur_mol:
            if mid in seen:
                raise ValueError(
                    f"mol_id {mid} 的 bead 在体系中不连续，无法导出 GROMACS 拓扑；"
                    "请先按分子重排 CG data 的 bead 顺序"
                )
            seen.add(mid)
            molecules.append([])
            cur_mol = mid
        molecules[-1].append(pos)
    return molecules


def _entry_molecule(pos2mol: List[int], entry_positions, kind: str) -> int:
    """校验拓扑条目所有原子属于同一分子，返回分子序号。"""
    mol_set = {pos2mol[p] for p in entry_positions}
    if len(mol_set) != 1:
        raise ValueError(f"{kind} 条目跨越多个分子（bead 位置 {entry_positions}），无法写入 itp")
    return mol_set.pop()


def _fingerprint(n_types: Tuple[int, ...],
                 bonds: List[Tuple[int, int]],
                 angles: List[Tuple[int, int, int]],
                 dihedrals: List[Tuple[int, int, int, int]]) -> Tuple:
    """
    分子结构指纹：bead 类型序列 + 规范化的局部拓扑连接（不含参数）。

    参数按 bead type 组合键控，结构相同则参数必然相同，故指纹无需包含类型编号。
    """
    canon_bonds = tuple(sorted((min(i, j), max(i, j)) for i, j in bonds))
    canon_angles = tuple(sorted(min((i, j, k), (k, j, i)) for i, j, k in angles))
    canon_dihs = tuple(sorted(min(e, e[::-1]) for e in dihedrals))
    return (tuple(n_types), canon_bonds, canon_angles, canon_dihs)


def build_cg_universe(cg_data: Dict, ff: CgForceField, title: str = "CG system",
                      mdfmt_path: Optional[str] = None):
    """
    把 CG data（Å，全局 1-based bead 编号）构建为 mdfmt Universe + Frame（nm）。

    链归并规则：结构指纹相同的相邻分子合并为同一 MoleculeInstance 的 count；
    指纹相同但不相邻的分子各成 instance（保证 gro 原子顺序与 instances 展开一致）。

    Args:
        cg_data: 含 ids/mol_ids/types/coords/box，及可选 bonds/angles/dihedrals
        ff: 已换算为 GROMACS 单位的 CgForceField
        title: 写入 gro/top 的体系标题
        mdfmt_path: 传给 ensure_mdfmt

    Returns:
        (universe, frame)
    """
    m = ensure_mdfmt(mdfmt_path)

    ids = cg_data['ids'].astype(int)
    types = cg_data['types'].astype(int)
    coords_nm = np.asarray(cg_data['coords'], dtype=np.float64) * 0.1
    box = cg_data['box']

    id2pos = {int(bid): pos for pos, bid in enumerate(ids)}
    molecules = _split_molecules(cg_data)

    # 位置 → (分子序号, 分子内局部序号)
    pos2mol: List[int] = [0] * len(ids)
    pos2local: List[int] = [0] * len(ids)
    for mi, positions in enumerate(molecules):
        for li, p in enumerate(positions):
            pos2mol[p] = mi
            pos2local[p] = li

    def topo_positions(entry):
        return [id2pos[int(b)] for b in entry[1:]]

    # 每个分子的局部拓扑条目
    mol_bonds: List[List[Tuple[int, int]]] = [[] for _ in molecules]
    mol_angles: List[List[Tuple[int, int, int]]] = [[] for _ in molecules]
    mol_dihs: List[List[Tuple[int, int, int, int]]] = [[] for _ in molecules]

    cg_bonds = cg_data.get('bonds')
    if cg_bonds is not None:
        for entry in np.asarray(cg_bonds):
            ps = topo_positions(entry)
            mi = _entry_molecule(pos2mol, ps, 'bond')
            mol_bonds[mi].append(tuple(pos2local[p] for p in ps))

    cg_angles = cg_data.get('angles')
    if cg_angles is not None:
        for entry in np.asarray(cg_angles):
            ps = topo_positions(entry)
            mi = _entry_molecule(pos2mol, ps, 'angle')
            mol_angles[mi].append(tuple(pos2local[p] for p in ps))

    cg_dihedrals = cg_data.get('dihedrals')
    if cg_dihedrals is not None:
        for entry in np.asarray(cg_dihedrals):
            ps = topo_positions(entry)
            mi = _entry_molecule(pos2mol, ps, 'dihedral')
            mol_dihs[mi].append(tuple(pos2local[p] for p in ps))

    def entry_params_bond(li, lj, t_of_local):
        combo = normalize_combo('bond', (t_of_local[li], t_of_local[lj]))
        return ff.bond_params[combo]

    def entry_params_angle(li, lj, lk, t_of_local):
        combo = normalize_combo('angle', (t_of_local[li], t_of_local[lj], t_of_local[lk]))
        return ff.angle_params[combo]

    def entry_params_dih(li, lj, lk, ll, t_of_local):
        combo = normalize_combo('dihedral',
                                (t_of_local[li], t_of_local[lj], t_of_local[lk], t_of_local[ll]))
        return ff.dihedral_params[combo]

    # 游程分组构建 molecule_types / instances
    molecule_types = []
    instances = []
    fp2moltype: Dict[Tuple, int] = {}
    prev_fp = None

    for mi, positions in enumerate(molecules):
        bead_types = tuple(int(types[p]) for p in positions)
        fp = _fingerprint(bead_types, mol_bonds[mi], mol_angles[mi], mol_dihs[mi])

        if fp == prev_fp:
            instances[-1].count += 1
            continue
        prev_fp = fp

        if fp not in fp2moltype:
            # 新链型：以当前分子为代表构建 MoleculeType
            mt_index = len(molecule_types)
            fp2moltype[fp] = mt_index
            mol_name = f"CG{len(fp2moltype)}"
            atoms = []
            for li, p in enumerate(positions):
                t = int(types[p])
                atoms.append(m.Atom(
                    local_index=li,
                    atom_type=ff.atom_names[t],
                    name=ff.atom_names[t],
                    residue_name=mol_name,
                    charge=ff.atom_charges[t],
                    mass=ff.atom_masses[t],
                ))
            bonds = [m.Bond(atom_i=li, atom_j=lj,
                            params=entry_params_bond(li, lj, bead_types))
                     for li, lj in mol_bonds[mi]]
            angles = [m.Angle(atom_i=li, atom_j=lj, atom_k=lk,
                              params=entry_params_angle(li, lj, lk, bead_types))
                      for li, lj, lk in mol_angles[mi]]
            dihs = [m.Dihedral(atom_i=li, atom_j=lj, atom_k=lk, atom_l=ll,
                               params=entry_params_dih(li, lj, lk, ll, bead_types))
                    for li, lj, lk, ll in mol_dihs[mi]]
            molecule_types.append(m.MoleculeType(
                name=mol_name, atoms=atoms, bonds=bonds, angles=angles,
                dihedrals=dihs, nrexcl=ff.nrexcl,
            ))

        instances.append(m.MoleculeInstance(molecule_type_index=fp2moltype[fp], count=1))

    # 单一链型时去掉编号，命名为 CG（更贴合常见 CG 体系习惯）
    if len(molecule_types) == 1:
        mt = molecule_types[0]
        mt.name = "CG"
        for atom in mt.atoms:
            atom.residue_name = "CG"

    lx = float(box[0, 1] - box[0, 0]) * 0.1
    ly = float(box[1, 1] - box[1, 0]) * 0.1
    lz = float(box[2, 1] - box[2, 0]) * 0.1
    universe = m.Universe(
        title=title,
        box=m.SimulationBox.orthogonal(lx, ly, lz),
        molecule_types=molecule_types,
        instances=instances,
    )
    frame = m.Frame(coords=coords_nm)
    universe.frame = frame

    if universe.total_atoms() != len(ids):
        raise RuntimeError(
            f"Universe 原子数 {universe.total_atoms()} 与 CG data bead 数 {len(ids)} 不一致"
        )
    return universe, frame


def collect_system_combos(cg_data: Dict):
    """
    收集体系中出现的 bead type 与规范化拓扑组合（供 validate_ff_completeness 使用）。

    Returns:
        (bead_types, bond_combos, angle_combos, dihedral_combos)
    """
    ids = cg_data['ids'].astype(int)
    types = cg_data['types'].astype(int)
    id2pos = {int(bid): pos for pos, bid in enumerate(ids)}

    def combo_of(entry, kind):
        return normalize_combo(kind, [int(types[id2pos[int(b)]]) for b in entry[1:]])

    bond_combos, angle_combos, dihedral_combos = [], [], []
    if cg_data.get('bonds') is not None:
        bond_combos = [combo_of(e, 'bond') for e in np.asarray(cg_data['bonds'])]
    if cg_data.get('angles') is not None:
        angle_combos = [combo_of(e, 'angle') for e in np.asarray(cg_data['angles'])]
    if cg_data.get('dihedrals') is not None:
        dihedral_combos = [combo_of(e, 'dihedral') for e in np.asarray(cg_data['dihedrals'])]
    return set(int(t) for t in types), bond_combos, angle_combos, dihedral_combos


def write_cg_itp(filepath, universe) -> None:
    """把 Universe 的全部 MoleculeType 写入单个 .itp 文件。"""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("; Created by LmpPy convert_aa2cg\n\n")
        for mol in universe.molecule_types:
            f.write("[ moleculetype ]\n")
            f.write("; name   nrexcl\n")
            f.write(f"{mol.name}   {mol.nrexcl}\n\n")

            f.write("[ atoms ]\n")
            f.write(";   nr  type  resnr  residue  atom  cgnr     charge         mass\n")
            for atom in mol.atoms:
                f.write(f"{atom.local_index+1:>6} {atom.atom_type:<6} {1:>6} "
                        f"{atom.residue_name:<8} {atom.name:<6} {atom.local_index+1:>6} "
                        f"{atom.charge:14.8f} {atom.mass:14.8f}\n")

            if mol.bonds:
                f.write("\n[ bonds ]\n")
                f.write(";   i     j  funct    b0(nm)        k(kJ/mol/nm^2)\n")
                for b in mol.bonds:
                    b0, k = b.params
                    f.write(f"{b.atom_i+1:6}{b.atom_j+1:6}{1:6}{b0:14.6e}{k:14.6e}\n")

            if mol.angles:
                f.write("\n[ angles ]\n")
                f.write(";   i     j     k  funct    theta0(deg)   k(kJ/mol/rad^2)\n")
                for a in mol.angles:
                    theta0, k = a.params
                    f.write(f"{a.atom_i+1:6}{a.atom_j+1:6}{a.atom_k+1:6}{1:6}"
                            f"{theta0:14.6e}{k:14.6e}\n")

            if mol.dihedrals:
                f.write("\n[ dihedrals ]\n")
                f.write(";   i     j     k     l  funct    phi0(deg)     k(kJ/mol)     n\n")
                for d in mol.dihedrals:
                    phi0, k, n = d.params
                    f.write(f"{d.atom_i+1:6}{d.atom_j+1:6}{d.atom_k+1:6}{d.atom_l+1:6}"
                            f"{1:6}{phi0:14.6e}{k:14.6e}{int(n):6d}\n")
            f.write("\n")


def write_cg_top(filepath, universe, ff: CgForceField, itp_name: str = "cg.itp") -> None:
    """
    写出体系级 .top：[defaults] + [atomtypes] + 可选 [nonbond_params] + #include itp
    + [system] + [molecules]。

    固定 nbfunc=1（LJ 12-6）、comb-rule=3（Lorentz-Berthelot：σ 算术平均、ε 几何平均）、
    gen-pairs=no、fudgeLJ=fudgeQQ=1。
    """
    name_of = ff.atom_names
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("; Created by LmpPy convert_aa2cg\n\n")
        f.write("[ defaults ]\n")
        f.write("; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n")
        f.write(f"{1:<8}{3:<11}{'no':<11}{1.0:<9.4f}{1.0:<9.4f}\n\n")

        f.write("[ atomtypes ]\n")
        f.write("; name      mass      charge  ptype    sigma(nm)     epsilon(kJ/mol)\n")
        for btype in sorted(name_of):
            f.write(f"{name_of[btype]:<8}{ff.atom_masses[btype]:>10.4f}"
                    f"{ff.atom_charges[btype]:>10.4f}{'A':>6}"
                    f"{ff.sigma[btype]:14.6e}{ff.epsilon[btype]:14.6e}\n")

        if ff.nb_overrides:
            f.write("\n[ nonbond_params ]\n")
            f.write(";   i    j  funct        c6(kJ/mol·nm^6)   c12(kJ/mol·nm^12)\n")
            for (ti, tj) in sorted(ff.nb_overrides):
                c6, c12 = ff.nb_overrides[(ti, tj)]
                f.write(f"{name_of[ti]:>5}{name_of[tj]:>5}{1:6}{c6:18.6e}{c12:18.6e}\n")

        f.write(f"\n#include \"{itp_name}\"\n")

        f.write("\n[ system ]\n")
        f.write(f"{universe.title or 'CG system'}\n\n")
        f.write("[ molecules ]\n")
        f.write("; name    count\n")
        for inst in universe.instances:
            mol = universe.molecule_types[inst.molecule_type_index]
            f.write(f"{mol.name:<10}{inst.count}\n")


def export_cg_gromacs(cg_data: Dict, ff_yaml, out_dir,
                      auto_masses: Optional[Dict[int, float]] = None,
                      title: str = "CG system",
                      mdfmt_path: Optional[str] = None,
                      gro_name: str = "system.gro",
                      top_name: str = "topol.top",
                      itp_name: str = "cg.itp") -> Dict[str, Path]:
    """
    一站式导出 CG 体系的 GROMACS 三件套：system.gro + topol.top + cg.itp。

    Args:
        cg_data: CG 数据（Å，含可选 bonds/angles/dihedrals）
        ff_yaml: CG 力场 YAML 路径（LAMMPS real 单位，见 cg_ff 模块 docstring）
        out_dir: 输出目录
        auto_masses: {bead_type: 质量}，FF YAML 未给 mass 时的回退（mapping 质量求和）
        title: 体系标题
        mdfmt_path: MD_format_transform 仓库根目录（缺省取环境变量 MDFMT_PATH）
        gro_name/top_name/itp_name: 输出文件名

    Returns:
        {'gro': Path, 'top': Path, 'itp': Path}
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    ff = load_cg_ff(ff_yaml, auto_masses=auto_masses)
    bead_types, bond_combos, angle_combos, dihedral_combos = collect_system_combos(cg_data)
    validate_ff_completeness(ff, bead_types, bond_combos, angle_combos, dihedral_combos)

    universe, frame = build_cg_universe(cg_data, ff, title=title, mdfmt_path=mdfmt_path)

    m = ensure_mdfmt(mdfmt_path)
    gro_path = out / gro_name
    top_path = out / top_name
    itp_path = out / itp_name
    m.write_gro(str(gro_path), universe, frame)
    write_cg_itp(str(itp_path), universe)
    write_cg_top(str(top_path), universe, ff, itp_name=itp_name)

    print(f"GROMACS 文件已导出到: {out}")
    print(f"  GRO: {gro_path.name} ({len(cg_data['ids'])} beads)")
    print(f"  TOP: {top_path.name} (#include \"{itp_name}\")")
    print(f"  ITP: {itp_path.name} ({len(universe.molecule_types)} 种 moleculetype)")
    return {'gro': gro_path, 'top': top_path, 'itp': itp_path}
