"""
CG 力场 YAML 加载与校验（供 convert_aa2cg 导出 GROMACS 拓扑使用）。

输入 YAML 固定使用 LAMMPS real 单位（Å、kcal/mol、度），本模块在加载时
统一换算为 GROMACS 单位（nm、kJ/mol）：

- 长度：Å → nm，×0.1
- 能量：kcal/mol → kJ/mol，×4.184
- 键/角力常数：LAMMPS harmonic 形式 E = K(x-x0)^2，GROMACS func 1 形式
  E = 1/2·k·(x-x0)^2，因此 k_gmx = 2·K·4.184（键长方向再 ×100，即 Å²→nm²）

非键混合规则固定为 Lorentz-Berthelot（GROMACS comb-rule 3：σ 算术平均、
ε 几何平均）。atomtypes 中每种 bead type 必须给出自身（对角）σ/ε；
非对角对默认由 GROMACS 自动混合，仅当用户在 nonbonded_overrides 中
显式给出时才写入 [nonbond_params]（σ/ε 换算为 c6=4εσ^6、c12=4εσ^12）。

YAML 示例见 convert_aa2cg.py 的 --ff-yaml 帮助或项目文档。
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# 单位换算常数（LAMMPS real → GROMACS）
ANGSTROM_TO_NM = 0.1
KCAL_TO_KJ = 4.184
BOND_K_FACTOR = 2.0 * KCAL_TO_KJ * 100.0   # kcal/mol/Å² → kJ/mol/nm²（含谐振形式 1/2 差异）
ANGLE_K_FACTOR = 2.0 * KCAL_TO_KJ          # kcal/mol/rad² → kJ/mol/rad²（rad 无量纲）
DIHEDRAL_K_FACTOR = KCAL_TO_KJ             # kcal/mol → kJ/mol

SUPPORTED_UNITS = ("lammps-real",)


class CgFFError(ValueError):
    """CG 力场 YAML 格式/完备性错误。"""


class CgForceField:
    """已换算为 GROMACS 单位的 CG 力场参数容器。

    属性:
        nrexcl: [moleculetype] 链内排除数
        atom_names: {bead_type: GROMACS 类型名}
        atom_masses: {bead_type: 质量 u}
        atom_charges: {bead_type: 电荷 e}
        sigma: {bead_type: σ (nm)}，对角 LJ 参数
        epsilon: {bead_type: ε (kJ/mol)}，对角 LJ 参数
        nb_overrides: {(t_i, t_j) 已排序: (c6, c12)}，kJ/mol·nm^6 / kJ/mol·nm^12
        bond_params: {(t1, t2) 规范化: (b0 nm, k kJ/mol/nm²)}
        angle_params: {(t1, t2, t3) 规范化: (theta0 度, k kJ/mol/rad²)}
        dihedral_params: {(t1, t2, t3, t4) 规范化: (phi0 度, k kJ/mol, n)}
    """

    def __init__(self):
        self.nrexcl: int = 3
        self.atom_names: Dict[int, str] = {}
        self.atom_masses: Dict[int, float] = {}
        self.atom_charges: Dict[int, float] = {}
        self.sigma: Dict[int, float] = {}
        self.epsilon: Dict[int, float] = {}
        self.nb_overrides: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.bond_params: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.angle_params: Dict[Tuple[int, int, int], Tuple[float, float]] = {}
        self.dihedral_params: Dict[Tuple[int, int, int, int], Tuple[float, float, int]] = {}
        # 拓扑组合 → 用户钉死的 type_id（兼任 --type-mapping 的作用）
        self.bond_type_ids: Dict[Tuple[int, int], int] = {}
        self.angle_type_ids: Dict[Tuple[int, int, int], int] = {}
        self.dihedral_type_ids: Dict[Tuple[int, int, int, int], int] = {}


def normalize_combo(kind: str, combo) -> Tuple:
    """
    bead type 组合对称规范化（规则与 data_converter._load_type_mapping_yaml 一致）。

    - bond: 两端排序
    - angle: 端点对调使序列正向字典序 ≤ 反向
    - dihedral: 正向与整体反转取字典序较小者
    """
    combo = tuple(int(t) for t in combo)
    if kind == 'bond':
        if len(combo) != 2:
            raise CgFFError(f"bond bead_types 长度必须为 2，实际: {combo}")
        return tuple(sorted(combo))
    if kind == 'angle':
        if len(combo) != 3:
            raise CgFFError(f"angle bead_types 长度必须为 3，实际: {combo}")
        if combo[0] > combo[2]:
            combo = (combo[2], combo[1], combo[0])
        return combo
    if kind == 'dihedral':
        if len(combo) != 4:
            raise CgFFError(f"dihedral bead_types 长度必须为 4，实际: {combo}")
        rev = (combo[3], combo[2], combo[1], combo[0])
        return min(combo, rev)
    raise CgFFError(f"未知拓扑类型: {kind}")


def _require_fields(entry: dict, fields: List[str], where: str):
    for f in fields:
        if f not in entry:
            raise CgFFError(f"{where} 缺少必填字段 '{f}'")


def _positive_int(value, where: str) -> int:
    if not isinstance(value, int) or value <= 0:
        raise CgFFError(f"{where} 必须是正整数，实际: {value!r}")
    return value


def load_cg_ff(yaml_path, auto_masses: Optional[Dict[int, float]] = None) -> CgForceField:
    """
    加载 CG 力场 YAML，返回已换算为 GROMACS 单位的 CgForceField。

    Args:
        yaml_path: YAML 文件路径
        auto_masses: {bead_type: 质量}，atomtypes 条目缺省 mass 时的回退来源
                     （通常由 mapping 质量求和得到，见 convert_aa2cg.build_cg_mass_dict）

    Raises:
        ImportError: PyYAML 未安装
        FileNotFoundError: YAML 不存在
        CgFFError: 格式错误、重复定义、单位不支持
    """
    if not HAS_YAML:
        raise ImportError("需要安装 PyYAML: pip install PyYAML")

    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"CG 力场 YAML 不存在: {yaml_path}")

    with open(yaml_path, 'r') as f:
        raw = yaml.safe_load(f)
    if not isinstance(raw, dict):
        raise CgFFError(f"YAML 根节点必须是 dict（mapping），实际为: {type(raw).__name__}")

    ff = CgForceField()

    # units：当前仅支持 lammps-real（保留键位以便将来扩展）
    units = raw.get('units', 'lammps-real')
    if units not in SUPPORTED_UNITS:
        raise CgFFError(f"units 仅支持 {SUPPORTED_UNITS}，实际: {units!r}")

    ff.nrexcl = _positive_int(raw.get('nrexcl', 3), 'nrexcl')

    # ---- atomtypes（必填；σ/ε 为对角 LJ 参数）----
    atomtypes = raw.get('atomtypes')
    if not isinstance(atomtypes, list) or not atomtypes:
        raise CgFFError("YAML 必须包含非空的 'atomtypes' 列表")
    for i, entry in enumerate(atomtypes):
        where = f"atomtypes 第 {i+1} 项"
        if not isinstance(entry, dict):
            raise CgFFError(f"{where} 必须是 dict")
        _require_fields(entry, ['type', 'sigma', 'epsilon'], where)
        btype = _positive_int(entry['type'], f"{where} 的 type")
        if btype in ff.atom_names:
            raise CgFFError(f"{where} bead type {btype} 重复定义")
        name = str(entry.get('name', f"T{btype}"))
        if not any(ch.isalpha() for ch in name):
            raise CgFFError(f"{where} 的 name '{name}' 必须至少含一个字母（GROMACS 类型名要求）")
        ff.atom_names[btype] = name
        ff.sigma[btype] = float(entry['sigma']) * ANGSTROM_TO_NM
        ff.epsilon[btype] = float(entry['epsilon']) * KCAL_TO_KJ
        ff.atom_charges[btype] = float(entry.get('charge', 0.0))
        if 'mass' in entry:
            ff.atom_masses[btype] = float(entry['mass'])
        elif auto_masses and btype in auto_masses:
            ff.atom_masses[btype] = float(auto_masses[btype])
        else:
            raise CgFFError(
                f"{where} 缺少 mass，且 auto_masses 中没有 bead type {btype} 的质量"
            )

    # ---- nonbonded_overrides（可选；非对角对显式覆盖）----
    overrides = raw.get('nonbonded_overrides') or []
    if not isinstance(overrides, list):
        raise CgFFError("'nonbonded_overrides' 必须是列表")
    for i, entry in enumerate(overrides):
        where = f"nonbonded_overrides 第 {i+1} 项"
        _require_fields(entry, ['types', 'sigma', 'epsilon'], where)
        pair = normalize_combo('bond', entry['types'])  # 非键对同样两端排序
        for t in pair:
            if t not in ff.atom_names:
                raise CgFFError(f"{where} 引用了未在 atomtypes 中定义的 bead type {t}")
        if pair in ff.nb_overrides:
            raise CgFFError(f"{where} 类型对 {pair} 重复定义")
        sig = float(entry['sigma']) * ANGSTROM_TO_NM
        eps = float(entry['epsilon']) * KCAL_TO_KJ
        # GROMACS [nonbond_params] 使用 c6/c12：c6=4εσ^6, c12=4εσ^12
        ff.nb_overrides[pair] = (4.0 * eps * sig ** 6, 4.0 * eps * sig ** 12)

    # ---- bonds / angles / dihedrals（harmonic 唯一形式）----
    _load_bonded(raw, 'bonds', 'bond', ['r0', 'k'], ff.bond_params, ff.bond_type_ids,
                 lambda e: (float(e['r0']) * ANGSTROM_TO_NM,
                            float(e['k']) * BOND_K_FACTOR))
    _load_bonded(raw, 'angles', 'angle', ['theta0', 'k'], ff.angle_params, ff.angle_type_ids,
                 lambda e: (float(e['theta0']),
                            float(e['k']) * ANGLE_K_FACTOR))
    _load_bonded(raw, 'dihedrals', 'dihedral', ['phi0', 'k'], ff.dihedral_params,
                 ff.dihedral_type_ids,
                 lambda e: (float(e['phi0']),
                            float(e['k']) * DIHEDRAL_K_FACTOR,
                            _positive_int(e.get('multiplicity', 1), 'multiplicity')))

    return ff


def _load_bonded(raw: dict, yaml_key: str, kind: str, required: List[str],
                 params_out: Dict, type_ids_out: Dict, convert):
    """解析 bonds/angles/dihedrals 列表，规范化组合键控并换算参数。"""
    entries = raw.get(yaml_key) or []
    if not isinstance(entries, list):
        raise CgFFError(f"'{yaml_key}' 必须是列表")
    for i, entry in enumerate(entries):
        where = f"{yaml_key} 第 {i+1} 项"
        if not isinstance(entry, dict):
            raise CgFFError(f"{where} 必须是 dict")
        _require_fields(entry, ['bead_types'] + required, where)
        combo = normalize_combo(kind, entry['bead_types'])
        for t in combo:
            if not isinstance(t, int) or t <= 0:
                raise CgFFError(f"{where} bead_types 必须是正整数，实际: {combo}")
        if combo in params_out:
            raise CgFFError(f"{where} bead type 组合 {combo}（规范化后）重复定义")
        params_out[combo] = convert(entry)
        if 'type_id' in entry:
            tid = _positive_int(entry['type_id'], f"{where} 的 type_id")
            if tid in type_ids_out.values():
                raise CgFFError(f"{where} type_id={tid} 与其他条目冲突")
            type_ids_out[combo] = tid


def validate_ff_completeness(ff: CgForceField,
                             bead_types,
                             bond_combos,
                             angle_combos,
                             dihedral_combos):
    """
    校验力场对给定体系的完备性：体系中出现的每个 bead type / 拓扑组合都必须有参数。

    所有缺失项汇总后一次性抛出 CgFFError。
    YAML 中定义了但体系中不存在的条目不算错误（允许一个 FF 文件复用于多个体系）。
    """
    missing = []

    for t in sorted(set(int(x) for x in bead_types)):
        if t not in ff.atom_names:
            missing.append(f"atomtypes: bead type {t}")

    for combo in sorted({normalize_combo('bond', c) for c in bond_combos}):
        if combo not in ff.bond_params:
            missing.append(f"bonds: 组合 {combo}")

    for combo in sorted({normalize_combo('angle', c) for c in angle_combos}):
        if combo not in ff.angle_params:
            missing.append(f"angles: 组合 {combo}")

    for combo in sorted({normalize_combo('dihedral', c) for c in dihedral_combos}):
        if combo not in ff.dihedral_params:
            missing.append(f"dihedrals: 组合 {combo}")

    if missing:
        raise CgFFError(
            "CG 力场参数不完备，缺少以下条目：\n  - " + "\n  - ".join(missing)
        )
