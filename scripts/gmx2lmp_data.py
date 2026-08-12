#!/usr/bin/env python3
"""GROMACS top + gro → LAMMPS data（GAFF，real 单位，atom_style full）。

用法:
    python LmpPy/scripts/gmx2lmp_data.py --top system.top --gro conf.gro -o out.data \
        [--type-order type_order.txt]

支持范围（不支持的输入一律报错，不静默转换）:
- top 递归 #include（相对包含文件所在目录解析）
- [ defaults ] comb-rule 2（σ/ε）与 comb-rule 1（C6/C12 自动换算）
- [ atomtypes ] 6 列（name mass charge ptype σ ε）/ 7 列（含 at.num 列）
- bonds/angles funct 1、dihedrals funct 9/1（proper）、funct 4（improper → cvff）
- [ molecules ] 多分子计数展开
- 正交盒；全零盒按坐标范围每边加 1 nm（总长 +2 nm），最小 3 nm
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path

KCAL_PER_KJ = 4.184  # kJ/mol → kcal/mol
NM_TO_ANGSTROM = 10.0  # nm → Å

# GRO 原子行固定列宽切片（3 位小数标准格式）
GRO_X_SLICE = slice(20, 28)
GRO_Y_SLICE = slice(28, 36)
GRO_Z_SLICE = slice(36, 44)

# 兜底分支：按空白切分 gro 原子行时，速度列存在意味着行尾还有 3 个速度分量。
# 此时坐标列之前若包含残基号则有 10 个令牌，某些简写格式无残基号则为 8 个；
# 无速度列时令牌数小于 8，直接取末尾 3 列作为坐标。
GRO_TOKEN_COUNTS_WITH_VELOCITY = frozenset({8, 10})


def parse_gro(gro_path: Path | str) -> tuple[list[tuple[float, float, float]], list[float]]:
    """解析 gro：返回（坐标 Å 列表, 正交盒 [Lx, Ly, Lz] Å）。

    原子行按固定列宽取 x/y/z（第 21-44 列，每列 8 字符、3 位小数；列宽变体兜底
    按空白切分取末尾坐标列）。盒行 3 个数 = 正交盒；9 个数且非对角元非零
    = 三斜盒 → 报错；9 个 0（sobtop 产物）按全零盒返回，由 build_system 兜底。
    """
    lines = Path(gro_path).read_text(encoding="utf-8").splitlines()
    try:
        n = int(lines[1].strip())
    except (IndexError, ValueError):
        raise ValueError(f"gro 文件格式异常（第 2 行应为原子数）: {gro_path}") from None
    if len(lines) < n + 3:
        raise ValueError(f"gro 文件行数不足（声称 {n} 原子）: {gro_path}")

    coords = []
    for line in lines[2:2 + n]:
        try:
            x = float(line[GRO_X_SLICE])
            y = float(line[GRO_Y_SLICE])
            z = float(line[GRO_Z_SLICE])
        except ValueError:
            # 列宽变体兜底：按空白切分，有速度列取倒数 6..4，否则取末 3 列
            t = line.split()
            xyz = t[-6:-3] if len(t) in GRO_TOKEN_COUNTS_WITH_VELOCITY else t[-3:]
            try:
                x, y, z = (float(v) for v in xyz)
            except ValueError:
                raise ValueError(f"gro 原子行解析失败: {line!r}") from None
        coords.append((x * NM_TO_ANGSTROM, y * NM_TO_ANGSTROM, z * NM_TO_ANGSTROM))

    box_tokens = [float(v) for v in lines[2 + n].split()]
    if len(box_tokens) == 3:
        box = [v * NM_TO_ANGSTROM for v in box_tokens]
    elif len(box_tokens) == 9:
        if any(v != 0.0 for v in box_tokens[3:]):
            raise ValueError("三斜盒不支持（仅支持正交盒）")
        box = [v * NM_TO_ANGSTROM for v in box_tokens[:3]]
    else:
        raise ValueError(f"gro 盒行列数异常（需 3 或 9 列）: {lines[2 + n]!r}")
    return coords, box


_INCLUDE_RE = re.compile(r'^\s*#include\s+["<]([^">]+)[">]')
_SECTION_RE = re.compile(r"^\s*\[\s*([A-Za-z_]+)\s*\]")


def _expand_includes(path: Path | str) -> list[str]:
    """递归展开 #include（相对包含文件所在目录解析），返回合并行列表。"""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"include 文件不存在: {path}")
    lines: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        m = _INCLUDE_RE.match(raw)
        if m:
            lines.extend(_expand_includes(path.parent / m.group(1)))
        else:
            lines.append(raw)
    return lines


def _collect_sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """按出现顺序收集 `[ section ]` 段的数据行（去 ; 注释、去空行）。

    节头允许行内注释（`[ dihedrals ] ; propers`——正则只匹配到 `]`）。
    """
    sections: list[tuple[str, list[str]]] = []
    current: tuple[str, list[str]] | None = None
    for raw in lines:
        m = _SECTION_RE.match(raw)
        if m:
            current = (m.group(1).lower(), [])
            sections.append(current)
            continue
        if current is None:
            continue
        stripped = raw.split(";", 1)[0].strip()
        if stripped:
            current[1].append(stripped)
    return sections


@dataclass
class Defaults:
    nbfunc: int = 1
    comb_rule: int = 2
    fudge_lj: float = 1.0
    fudge_qq: float = 1.0


def _parse_defaults(rows: list[str]) -> Defaults:
    if not rows:
        raise ValueError("[ defaults ] 段为空")
    t = rows[0].split()
    if len(t) < 5:
        raise ValueError(f"[ defaults ] 行格式不完整（需 5 列）: {rows[0]!r}")
    return Defaults(nbfunc=int(t[0]), comb_rule=int(t[1]),
                    fudge_lj=float(t[3]), fudge_qq=float(t[4]))


@dataclass
class AtomType:
    name: str
    mass: float        # g/mol
    sigma_nm: float
    epsilon_kj: float  # kJ/mol


def _parse_atomtypes(rows: list[str], comb_rule: int) -> list[AtomType]:
    """解析 [ atomtypes ]；顺序即 LAMMPS 类型号。

    兼容 7 列（name at.num mass charge ptype p1 p2）与 6 列（无 at.num 列）。
    comb-rule 2：p1/p2 = σ(nm)/ε(kJ/mol)；comb-rule 1：p1/p2 = C6/C12，
    换算 σ=(C12/C6)^(1/6)、ε=C6²/(4·C12)（C6 或 C12 ≤ 0 时 σ=ε=0）。
    """
    if comb_rule not in (1, 2):
        raise ValueError(f"不支持的 comb-rule {comb_rule}（仅支持 1/2）")
    types = []
    seen: set[str] = set()
    for r in rows:
        t = r.split()
        if len(t) == 7:
            name, mass, p1, p2 = t[0], float(t[2]), float(t[5]), float(t[6])
        elif len(t) == 6:
            name, mass, p1, p2 = t[0], float(t[1]), float(t[4]), float(t[5])
        else:
            raise ValueError(f"[ atomtypes ] 行列数异常（需 6 或 7 列）: {r!r}")
        if name in seen:
            raise ValueError(f"[ atomtypes ] 中类型 {name!r} 重复定义")
        seen.add(name)
        if comb_rule == 2:
            sigma_nm, epsilon_kj = p1, p2
        else:
            c6, c12 = p1, p2
            if c6 <= 0.0 or c12 <= 0.0:
                sigma_nm, epsilon_kj = 0.0, 0.0
            else:
                sigma_nm = (c12 / c6) ** (1.0 / 6.0)
                epsilon_kj = c6 * c6 / (4.0 * c12)
        types.append(AtomType(name=name, mass=mass,
                              sigma_nm=sigma_nm, epsilon_kj=epsilon_kj))
    return types


@dataclass
class MolAtom:
    nr: int
    type_name: str
    charge: float
    mass: float | None = None  # atoms 行 8 列时的质量覆盖；None 用 atomtype 质量


@dataclass
class BondedRow:
    atoms: tuple          # 分子内 1-based 原子序号
    funct: int
    params: tuple         # 参数列原文（换算时转 float/int）


@dataclass
class MolType:
    name: str
    atoms: list[MolAtom] = field(default_factory=list)
    bonds: list[BondedRow] = field(default_factory=list)
    angles: list[BondedRow] = field(default_factory=list)
    dihedrals: list[BondedRow] = field(default_factory=list)   # funct 9/1（proper）
    impropers: list[BondedRow] = field(default_factory=list)   # funct 4


@dataclass
class Topology:
    defaults: Defaults
    atom_types: list                 # 顺序即 LAMMPS atom type 号
    mol_types: dict
    molecules: list                  # [(分子名, 计数)]，[ molecules ] 顺序


def _parse_atom_row(r: str) -> MolAtom:
    t = r.split()
    if len(t) == 8:
        return MolAtom(nr=int(t[0]), type_name=t[1],
                       charge=float(t[6]), mass=float(t[7]))
    if len(t) == 7:
        return MolAtom(nr=int(t[0]), type_name=t[1], charge=float(t[6]))
    raise ValueError(f"[ atoms ] 行列数异常（需 7 或 8 列）: {r!r}")


def _parse_bonded_row(r: str, n_atoms: int, section: str) -> BondedRow:
    t = r.split()
    if len(t) < n_atoms + 1:
        raise ValueError(f"[ {section} ] 行列数不足: {r!r}")
    try:
        atoms = tuple(int(x) for x in t[:n_atoms])
        funct = int(t[n_atoms])
    except ValueError:
        raise ValueError(f"[ {section} ] 行解析失败: {r!r}") from None
    return BondedRow(atoms=atoms, funct=funct, params=tuple(t[n_atoms + 1:]))


def _check_params(row: BondedRow, need: int, section: str) -> None:
    if len(row.params) < need:
        raise ValueError(
            f"[ {section} ] funct {row.funct} 需 {need} 个参数，"
            f"实际 {len(row.params)}: {row.atoms}")


def _ensure_mol(current_mol: MolType | None, section: str) -> MolType:
    """检查 bonded/atoms 段是否出现在 [ moleculetype ] 之前。"""
    if current_mol is None:
        raise ValueError(f"[ {section} ] 出现在任何 [ moleculetype ] 之前")
    return current_mol


def _add_bonded(mol, rows, n_atoms, section, allowed_funct,
                min_params, target_list):
    """统一解析 bonds/angles/dihedrals 行。

    target_list 为列表时，所有 funct 追加到同一列表；为 dict 时按 funct 分发。
    min_params 可为统一整数或 dict[int, int]。

    allowed_funct 在 target_list 为 dict 且传入 None 时，自动从 dict 的 key
    推导；target_list 为 list 时调用者必须显式传入允许集合，避免 API 误导。
    """
    if allowed_funct is None:
        if isinstance(target_list, dict):
            allowed_funct = set(target_list.keys())
        else:
            raise ValueError(
                f"[ {section} ] 使用列表 target_list 时必须显式提供 allowed_funct")

    for r in rows:
        row = _parse_bonded_row(r, n_atoms, section)
        if row.funct not in allowed_funct:
            raise ValueError(
                f"[ {section} ] 不支持的 functype {row.funct}: {r!r}")
        targets = target_list[row.funct] if isinstance(target_list, dict) else target_list
        need = min_params[row.funct] if isinstance(min_params, dict) else min_params
        _check_params(row, need, section)
        targets.append(row)


def parse_top(top_path: Path | str) -> Topology:
    """解析 top（含 #include 展开）为 Topology。

    functype 边界：bonds/angles 仅 funct 1；dihedrals funct 9/1 → proper、
    funct 4 → improper；其它一律 ValueError。未识别的段（pairs/constraints/
    settles/exclusions 等）跳过并告警。
    """
    sections = _collect_sections(_expand_includes(Path(top_path)))

    defaults = Defaults()
    atomtype_rows: list[str] = []
    mol_types: dict[str, MolType] = {}
    molecules: list[tuple[str, int]] = []
    current_mol: MolType | None = None
    skipped_sections: set[str] = set()

    for name, rows in sections:
        if name == "defaults":
            defaults = _parse_defaults(rows)
        elif name == "atomtypes":
            atomtype_rows.extend(rows)
        elif name == "moleculetype":
            mol_name = rows[0].split()[0]
            if mol_name in mol_types:
                raise ValueError(f"[ moleculetype ] 重复定义: {mol_name}")
            current_mol = MolType(name=mol_name)
            mol_types[mol_name] = current_mol
        elif name == "atoms":
            mol = _ensure_mol(current_mol, name)
            mol.atoms.extend(_parse_atom_row(r) for r in rows)
        elif name == "bonds":
            mol = _ensure_mol(current_mol, name)
            _add_bonded(mol, rows, 2, name, {1}, 2, mol.bonds)
        elif name == "angles":
            mol = _ensure_mol(current_mol, name)
            _add_bonded(mol, rows, 3, name, {1}, 2, mol.angles)
        elif name == "dihedrals":
            mol = _ensure_mol(current_mol, name)
            _add_bonded(
                mol, rows, 4, name,
                allowed_funct={1, 9, 4},
                min_params={1: 3, 9: 3, 4: 3},
                target_list={1: mol.dihedrals, 9: mol.dihedrals, 4: mol.impropers},
            )
        elif name == "molecules":
            for r in rows:
                t = r.split()
                if len(t) != 2:
                    raise ValueError(
                        f"[ molecules ] 行需 2 列（分子名 计数）: {r!r}")
                molecules.append((t[0], int(t[1])))
        elif name == "system":
            continue
        else:
            skipped_sections.add(name)

    for name in sorted(skipped_sections):
        warnings.warn(f"[ {name} ] 段未处理，已跳过"
                      "（pairs/constraints 等由 LAMMPS 端 special_bonds/系综设置覆盖）")

    if not molecules:
        raise ValueError("top 缺少 [ molecules ] 段")
    for mol_name, _count in molecules:
        if mol_name not in mol_types:
            raise ValueError(f"[ molecules ] 引用了未定义的 moleculetype: {mol_name}")

    atom_types = _parse_atomtypes(atomtype_rows, defaults.comb_rule) if atomtype_rows else []
    known = {t.name for t in atom_types}
    for mol in mol_types.values():
        for a in mol.atoms:
            if a.type_name not in known:
                raise ValueError(
                    f"[ atoms ] 引用了未定义的 atomtype: {a.type_name}"
                    f"（moleculetype {mol.name}）")

    return Topology(defaults=defaults, atom_types=atom_types,
                    mol_types=mol_types, molecules=molecules)


@dataclass
class System:
    """展开后的全局体系（原子/拓扑均为全局 1-based id）。

    box 由调用方提供；build_system 内部会执行 ``list(box)`` 拷贝，避免修改外部对象。

    coeffs 表：参数元组 → 类型号（按首次出现顺序，同参数合并同类型）。
      bond: (r0_nm, k)  angle: (a0_deg, k)  dihedral/improper: (phase, kd, pn)
    """
    box: list[float]
    atoms: list = field(default_factory=list)        # (mol_id, type_id, q, x, y, z)
    bonds: list = field(default_factory=list)        # (type_id, i, j)
    angles: list = field(default_factory=list)       # (type_id, i, j, k)
    dihedrals: list = field(default_factory=list)    # (type_id, i, j, k, l)
    impropers: list = field(default_factory=list)    # (type_id, i, j, k, l)
    bond_coeffs: dict = field(default_factory=dict)
    angle_coeffs: dict = field(default_factory=dict)
    dihedral_coeffs: dict = field(default_factory=dict)
    improper_coeffs: dict = field(default_factory=dict)


def build_system(topo: Topology,
                 coords: list[tuple[float, float, float]],
                 box: list[float]) -> System:
    """按 [ molecules ] 展开全局体系；校验原子数、零盒兜底、越界告警。"""
    n_expected = sum(len(topo.mol_types[name].atoms) * count
                     for name, count in topo.molecules)
    if len(coords) != n_expected:
        raise ValueError(
            f"gro 原子数 {len(coords)} 与 top 展开原子数 {n_expected} 不一致")

    box = list(box)
    if all(b == 0.0 for b in box):
        xs, ys, zs = zip(*coords)
        box = [max(max(c) - min(c) + 20.0, 30.0) for c in (xs, ys, zs)]
        warnings.warn("gro 盒尺寸全为 0，按坐标范围 + 每边 1 nm（总长 + 2 nm）"
                      f"边距生成默认正交盒（最小 3 nm）: {box}")

    xs, ys, zs = zip(*coords)
    for lo, hi, L, axis in ((min(xs), max(xs), box[0], "x"),
                            (min(ys), max(ys), box[1], "y"),
                            (min(zs), max(zs), box[2], "z")):
        if lo < 0.0 or hi > L:
            warnings.warn(f"存在原子坐标超出 {axis} 盒范围 [0, {L:.3f}] Å，"
                          "data 盒边界仍写 [0, L]，请自行确认")
            break

    type_id_of = {t.name: i + 1 for i, t in enumerate(topo.atom_types)}
    system = System(box=box)
    offset = 0
    mol_id = 0
    coord_idx = 0
    mass_mismatch = 0
    for name, count in topo.molecules:
        mol = topo.mol_types[name]
        for _ in range(count):
            mol_id += 1
            base = offset           # 当前分子实例的原子索引基准偏移
            offset += len(mol.atoms)  # 与 mol_id 一起自增，维护位置统一
            for a in mol.atoms:
                tid = type_id_of[a.type_name]
                if a.mass is not None and abs(a.mass - topo.atom_types[tid - 1].mass) > 1e-6:
                    mass_mismatch += 1
                x, y, z = coords[coord_idx]
                coord_idx += 1
                system.atoms.append((mol_id, tid, a.charge, x, y, z))
            for row in mol.bonds:
                key = (float(row.params[0]), float(row.params[1]))
                tid = system.bond_coeffs.setdefault(key, len(system.bond_coeffs) + 1)
                system.bonds.append((tid, row.atoms[0] + base, row.atoms[1] + base))
            for row in mol.angles:
                key = (float(row.params[0]), float(row.params[1]))
                tid = system.angle_coeffs.setdefault(key, len(system.angle_coeffs) + 1)
                system.angles.append(
                    (tid, *(a + base for a in row.atoms)))
            for row in mol.dihedrals:
                key = (float(row.params[0]), float(row.params[1]), int(row.params[2]))
                tid = system.dihedral_coeffs.setdefault(
                    key, len(system.dihedral_coeffs) + 1)
                system.dihedrals.append(
                    (tid, *(a + base for a in row.atoms)))
            for row in mol.impropers:
                key = (float(row.params[0]), float(row.params[1]), int(row.params[2]))
                tid = system.improper_coeffs.setdefault(
                    key, len(system.improper_coeffs) + 1)
                system.impropers.append(
                    (tid, *(a + base for a in row.atoms)))

    if mass_mismatch:
        warnings.warn(f"{mass_mismatch} 个原子的 atoms 行质量与 atomtype 质量不一致，"
                      "Masses 段按 atomtype 质量写出")
    return system


def _phase_sign(phase: float, pn: int, label: str) -> int:
    """GAFF 相位 0/180° → LAMMPS 符号；pn<0（cos 翻转）再翻一次。"""
    sign = -1 if phase == 180.0 else 1
    if phase not in (0.0, 180.0):
        warnings.warn(f"{label} 相位 {phase}° 不在 0/180°，按 0°（sign=+1）处理，"
                      "需人工核对")
    if pn < 0:
        sign = -sign
    return sign


def write_lmp_data(system: System, atom_types: list[AtomType], defaults: Defaults,
                   out_path: Path | str,
                   title: str = "GROMACS → LAMMPS data (gmx2lmp_data)") -> None:
    """写出 LAMMPS data（real 单位，atom_style full）。

    换算（与 pre_data_gen/scripts/convert_to_lmp.py 已验证公式一致）：
      bond  K = k/(2·4.184·100) kcal/mol/Å²，r0 Å      → harmonic
      angle K = k/(2·4.184) kcal/mol/rad²，θ0 度        → harmonic
      dihedral K = kd/4.184，phase→sign，n=|pn|         → harmonic
      improper 同 dihedral 形式                          → cvff (K d n)
      pair  ε = ε_kJ/4.184 kcal/mol，σ = σ_nm·10 Å      → lj/cut
    """
    mix = {1: "arithmetic", 2: "geometric"}[defaults.comb_rule]
    L: list[str] = [
        title,
        "# pair_style lj/cut",
        f"# pair_modify mix {mix}",
        "# bond_style harmonic",
        "# angle_style harmonic",
        "# dihedral_style harmonic",
        "# improper_style cvff",
        "# atom_style full",
        f"# special_bonds lj 0.0 0.0 {defaults.fudge_lj} coul 0.0 0.0 {defaults.fudge_qq}",
        "",
        f"{len(system.atoms)} atoms",
        f"{len(system.bonds)} bonds",
        f"{len(system.angles)} angles",
        f"{len(system.dihedrals)} dihedrals",
        f"{len(system.impropers)} impropers",
        "",
        f"{len(atom_types)} atom types",
        f"{len(system.bond_coeffs)} bond types",
        f"{len(system.angle_coeffs)} angle types",
        f"{len(system.dihedral_coeffs)} dihedral types",
        f"{len(system.improper_coeffs)} improper types",
        "",
        f"0.000000 {system.box[0]:.6f} xlo xhi",
        f"0.000000 {system.box[1]:.6f} ylo yhi",
        f"0.000000 {system.box[2]:.6f} zlo zhi",
        "",
        "Masses",
        "",
    ]
    for i, t in enumerate(atom_types, 1):
        L.append(f"{i} {t.mass:g}")

    L += ["", "Pair Coeffs # lj/cut", ""]
    for i, t in enumerate(atom_types, 1):
        L.append(f"{i} {t.epsilon_kj / KCAL_PER_KJ:.6e} {t.sigma_nm * 10.0:.6f}")

    if system.bond_coeffs:
        L += ["", "Bond Coeffs # harmonic", ""]
        for (r0_nm, k), tid in system.bond_coeffs.items():
            L.append(f"{tid} {k / (2.0 * KCAL_PER_KJ * 100.0):.3f} {r0_nm * 10.0:.4f}")

    if system.angle_coeffs:
        L += ["", "Angle Coeffs # harmonic", ""]
        for (a0, k), tid in system.angle_coeffs.items():
            L.append(f"{tid} {k / (2.0 * KCAL_PER_KJ):.3f} {a0:.3f}")

    if system.dihedral_coeffs:
        L += ["", "Dihedral Coeffs # harmonic", ""]
        for (phase, kd, pn), tid in system.dihedral_coeffs.items():
            sign = _phase_sign(phase, pn, f"dihedral 类型 {tid}")
            L.append(f"{tid} {kd / KCAL_PER_KJ:.3f} {sign} {abs(pn)}")

    if system.improper_coeffs:
        L += ["", "Improper Coeffs # cvff", ""]
        for (phase, kd, pn), tid in system.improper_coeffs.items():
            d = _phase_sign(phase, pn, f"improper 类型 {tid}")
            L.append(f"{tid} {kd / KCAL_PER_KJ:.3f} {d} {abs(pn)}")

    L += ["", "Atoms # full", ""]
    for i, (mol_id, tid, q, x, y, z) in enumerate(system.atoms, 1):
        L.append(f"{i} {mol_id} {tid} {q:.8f} {x:.6f} {y:.6f} {z:.6f}")

    if system.bonds:
        L += ["", "Bonds", ""]
        for i, (tid, a, b) in enumerate(system.bonds, 1):
            L.append(f"{i} {tid} {a} {b}")
    if system.angles:
        L += ["", "Angles", ""]
        for i, (tid, a, b, c) in enumerate(system.angles, 1):
            L.append(f"{i} {tid} {a} {b} {c}")
    if system.dihedrals:
        L += ["", "Dihedrals", ""]
        for i, (tid, a, b, c, d) in enumerate(system.dihedrals, 1):
            L.append(f"{i} {tid} {a} {b} {c} {d}")
    if system.impropers:
        L += ["", "Impropers", ""]
        for i, (tid, a, b, c, d) in enumerate(system.impropers, 1):
            L.append(f"{i} {tid} {a} {b} {c} {d}")

    Path(out_path).write_text("\n".join(L) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="GROMACS top+gro → LAMMPS data（GAFF，real 单位，atom_style full）")
    ap.add_argument("--top", required=True, help="GROMACS .top（可含 #include itp）")
    ap.add_argument("--gro", required=True, help="GROMACS .gro（正交盒）")
    ap.add_argument("-o", "--out", required=True, help="输出 LAMMPS data 路径")
    ap.add_argument("--type-order", default=None,
                    help="可选：输出 类型号=GAFF类型名 映射文件")
    args = ap.parse_args(argv)

    try:
        topo = parse_top(args.top)
        coords, box = parse_gro(args.gro)
        system = build_system(topo, coords, box)

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        write_lmp_data(system, topo.atom_types, topo.defaults, out_path,
                       title=f"converted from {Path(args.top).name} + "
                             f"{Path(args.gro).name} (gmx2lmp_data)")

        if args.type_order:
            type_order_path = Path(args.type_order)
            type_order_path.parent.mkdir(parents=True, exist_ok=True)
            order = ",".join(f"{i + 1}={t.name}" for i, t in enumerate(topo.atom_types))
            type_order_path.write_text(
                f"{Path(args.top).stem}: {order}\n", encoding="utf-8")

    except (ValueError, FileNotFoundError) as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2
    except OSError as e:
        print(f"写入失败: {e}", file=sys.stderr)
        return 3

    if all(a[2] == 0.0 for a in system.atoms):
        print("警告: 所有原子电荷为 0，请检查 top/itp 电荷设置", file=sys.stderr)

    print(f"转换完成: {out_path}")
    print(f"  原子 {len(system.atoms)}，键 {len(system.bonds)}，"
          f"角 {len(system.angles)}，二面角 {len(system.dihedrals)}，"
          f"不二面角 {len(system.impropers)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
