#!/usr/bin/env python
"""
build_cg_system.py 的 YAML 配置解析模块（build_cg_config）。

提供:
- MoleculeSpec / BuildCGConfig dataclass 定义
- from_yaml() 类方法：读取和校验 YAML 配置
- 拓扑文件读取函数：read_bonds / read_angles / read_dihedrals
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import yaml


# ============================================================
# 拓扑文件读取
# ============================================================

def read_topology_file(filepath: Optional[str], n_cols_expected: int) -> np.ndarray:
    """
    通用拓扑文件读取。

    支持 n_cols_expected 列（如 bonds=3, angles=4, dihedrals=5）
    或 n_cols_expected+1 列（带 ID 列）。
    支持 # 注释行和空行。

    Args:
        filepath: 文件路径，None 或空字符串返回空数组
        n_cols_expected: 期望的数据列数（不含 ID 列）

    Returns:
        shape (N, n_cols_expected) 的 int32 ndarray，无数据时 shape (0, n_cols_expected)
    """
    if not filepath:
        return np.array([], dtype=np.int32).reshape(0, n_cols_expected)

    fpath = Path(filepath)
    if not fpath.exists():
        raise FileNotFoundError(f"拓扑文件不存在: {fpath}")

    content = fpath.read_text().strip()
    if not content:
        return np.array([], dtype=np.int32).reshape(0, n_cols_expected)

    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        # 判断是否有 ID 列：列数 == n_cols_expected+1 则有 ID 列
        if len(parts) == n_cols_expected:
            rows.append([int(p) for p in parts])
        elif len(parts) == n_cols_expected + 1:
            rows.append([int(p) for p in parts[1:]])  # 跳过第一列 ID
        else:
            import warnings
            warnings.warn(
                f"跳过格式不正确的行 (期望 {n_cols_expected} 或 {n_cols_expected+1} 列，"
                f"实际 {len(parts)} 列): {line}"
            )
            continue

    if not rows:
        return np.array([], dtype=np.int32).reshape(0, n_cols_expected)
    return np.array(rows, dtype=np.int32)


def read_bonds(filepath: Optional[str]) -> np.ndarray:
    """读取 bonds 文件，返回 (N, 3) [bond_type, a1, a2]."""
    return read_topology_file(filepath, 3)


def read_angles(filepath: Optional[str]) -> np.ndarray:
    """读取 angles 文件，返回 (N, 4) [angle_type, a1, a2, a3]."""
    return read_topology_file(filepath, 4)


def read_dihedrals(filepath: Optional[str]) -> np.ndarray:
    """读取 dihedrals 文件，返回 (N, 5) [dihedral_type, a1, a2, a3, a4]."""
    return read_topology_file(filepath, 5)


# ============================================================
# Dataclass 定义
# ============================================================

@dataclass
class MoleculeSpec:
    """单个分子定义，对应 YAML 中 molecules 列表的每一项。"""
    name: str
    count: int
    bonds_file: Optional[str] = None
    angles_file: Optional[str] = None
    dihedrals_file: Optional[str] = None
    beads_per_mol: int = 0  # 从拓扑文件推断，0 表示未设置

    @classmethod
    def from_dict(cls, d: dict) -> "MoleculeSpec":
        return cls(
            name=d["name"],
            count=int(d["count"]),
            bonds_file=d.get("bonds_file") or None,
            angles_file=d.get("angles_file") or None,
            dihedrals_file=d.get("dihedrals_file") or None,
        )


@dataclass
class BuildCGConfig:
    """完整构建配置，对应整个 YAML 文件。"""
    molecules: list[MoleculeSpec]
    masses: dict[int, float]
    types: dict[str, list[int]] = field(default_factory=dict)
    gro: Optional[str] = None
    output: Optional[str] = None

    # ============================================================
    # YAML 解析
    # ============================================================

    @classmethod
    def from_yaml(cls, path: str) -> "BuildCGConfig":
        """
        从 YAML 文件读取并解析配置。

        Args:
            path: YAML 文件路径

        Returns:
            BuildCGConfig 实例

        Raises:
            FileNotFoundError: YAML 文件不存在
            yaml.YAMLError: YAML 格式错误
            ValueError: 参数校验失败
        """
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"YAML 配置文件不存在: {yaml_path}")

        with open(yaml_path, "r") as f:
            raw = yaml.safe_load(f)

        if raw is None:
            raise ValueError(f"YAML 文件为空: {yaml_path}")

        # 解析分子列表
        molecules = []
        mol_list = raw.get("molecules", [])
        for mol_dict in mol_list:
            molecules.append(MoleculeSpec.from_dict(mol_dict))

        # 解析 masses: 确保 key 为 int，并校验质量为正值
        masses_raw = raw.get("masses", {})
        masses = {int(k): float(v) for k, v in masses_raw.items()}
        for atype, mass in masses.items():
            if mass <= 0:
                raise ValueError(f"原子类型 {atype} 的质量必须为正数，当前为 {mass}")

        # 解析 types: 确保 value 为 int 列表
        types_raw = raw.get("types", {})
        types = {str(k): [int(t) for t in v] for k, v in types_raw.items()}

        config = cls(
            molecules=molecules,
            masses=masses,
            types=types,
            gro=raw.get("gro"),
            output=raw.get("output"),
        )

        # 推断每个分子的 beads_per_mol
        for mol in config.molecules:
            if mol.bonds_file and Path(mol.bonds_file).exists():
                bonds_arr = read_bonds(mol.bonds_file)
                if len(bonds_arr) > 0:
                    mol.beads_per_mol = max(
                        int(bonds_arr[:, 1].max()), int(bonds_arr[:, 2].max())
                    )
                else:
                    mol.beads_per_mol = 1
            else:
                mol.beads_per_mol = 1  # 单 bead 分子

        config.validate()
        return config

    # ============================================================
    # 参数校验
    # ============================================================

    def validate(self) -> None:
        """
        校验配置参数，发现问题抛出 ValueError。

        Raises:
            ValueError: 配置参数不合法
        """
        # 1. 至少定义一个分子
        if not self.molecules:
            raise ValueError("molecules 列表不能为空，至少定义一个分子")

        # 2. 检查每个分子
        for i, mol in enumerate(self.molecules):
            if mol.count <= 0:
                raise ValueError(
                    f"molecules[{i}] '{mol.name}' 的 count 必须 > 0，当前为 {mol.count}"
                )
            if mol.bonds_file and not Path(mol.bonds_file).exists():
                raise FileNotFoundError(
                    f"molecules[{i}] '{mol.name}' 的 bonds_file 不存在: {mol.bonds_file}"
                )

        # 3. masses 不能为空
        if not self.masses:
            raise ValueError("masses 不能为空，至少定义一个原子类型质量")

        # 4. 既无 gro 也无 types → 报错
        if not self.gro and not self.types:
            raise ValueError("必须提供 gro 或 types 之一（用于确定 bead 类型和坐标）")

        # 5. 若提供了 types，检查与 beads_per_mol 匹配
        if self.types:
            for mol in self.molecules:
                if mol.name in self.types:
                    expected = mol.beads_per_mol
                    actual = len(self.types[mol.name])
                    if actual != expected:
                        raise ValueError(
                            f"分子 '{mol.name}' 的 types 数量 ({actual}) 与 beads_per_mol ({expected}) 不匹配"
                        )

        # 6. 检查 types 中的分子名都有对应的 molecule 定义
        for t_name in self.types:
            if not any(m.name == t_name for m in self.molecules):
                raise ValueError(f"types 中定义了未知分子 '{t_name}'，molecules 列表中未找到")
