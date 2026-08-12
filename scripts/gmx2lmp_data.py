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
- 正交盒；全零盒按坐标范围 + 1 nm 边距兜底（最小 3 nm）
"""
from __future__ import annotations

import argparse  # noqa: F401
import re
import sys  # noqa: F401
import warnings  # noqa: F401
from dataclasses import dataclass
from dataclasses import field  # noqa: F401
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
    for r in rows:
        t = r.split()
        if len(t) == 7:
            name, mass, p1, p2 = t[0], float(t[2]), float(t[5]), float(t[6])
        elif len(t) == 6:
            name, mass, p1, p2 = t[0], float(t[1]), float(t[4]), float(t[5])
        else:
            raise ValueError(f"[ atomtypes ] 行列数异常（需 6 或 7 列）: {r!r}")
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
    atoms: list = field(default_factory=list)
    bonds: list = field(default_factory=list)
    angles: list = field(default_factory=list)
    dihedrals: list = field(default_factory=list)   # funct 9/1（proper）
    impropers: list = field(default_factory=list)   # funct 4


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


def parse_top(top_path) -> Topology:
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
    skipped: set[str] = set()

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
            if current_mol is None:
                raise ValueError("[ atoms ] 出现在任何 [ moleculetype ] 之前")
            current_mol.atoms.extend(_parse_atom_row(r) for r in rows)
        elif name == "bonds":
            if current_mol is None:
                raise ValueError("[ bonds ] 出现在任何 [ moleculetype ] 之前")
            for r in rows:
                row = _parse_bonded_row(r, 2, name)
                if row.funct != 1:
                    raise ValueError(f"[ bonds ] 不支持的 functype {row.funct}: {r!r}")
                _check_params(row, 2, name)
                current_mol.bonds.append(row)
        elif name == "angles":
            if current_mol is None:
                raise ValueError("[ angles ] 出现在任何 [ moleculetype ] 之前")
            for r in rows:
                row = _parse_bonded_row(r, 3, name)
                if row.funct != 1:
                    raise ValueError(f"[ angles ] 不支持的 functype {row.funct}: {r!r}")
                _check_params(row, 2, name)
                current_mol.angles.append(row)
        elif name == "dihedrals":
            if current_mol is None:
                raise ValueError("[ dihedrals ] 出现在任何 [ moleculetype ] 之前")
            for r in rows:
                row = _parse_bonded_row(r, 4, name)
                if row.funct in (1, 9):
                    _check_params(row, 3, name)
                    current_mol.dihedrals.append(row)
                elif row.funct == 4:
                    _check_params(row, 3, name)
                    current_mol.impropers.append(row)
                else:
                    raise ValueError(
                        f"[ dihedrals ] 不支持的 functype {row.funct}: {r!r}")
        elif name == "molecules":
            for r in rows:
                t = r.split()
                molecules.append((t[0], int(t[1])))
        elif name == "system":
            continue
        else:
            skipped.add(name)

    for name in sorted(skipped):
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
