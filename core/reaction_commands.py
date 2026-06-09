#!/usr/bin/env python3
"""
反应命令生成器模块

功能:
- bond/create 和 bond/react 的 LAMMPS fix 命令生成
- bond/create 模式下的 CG 映射更新
- 纯函数设计，不持有状态

作者: Claude Code
日期: 2026-06-10
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


# ============================================================================
# 异常定义
# ============================================================================

class BondCreateConfigError(Exception):
    """bond/create 配置错误"""
    pass


# ============================================================================
# 数据类
# ============================================================================

@dataclass
class BondCreateConfig:
    """bond/create 配置容器"""
    Nevery: int
    itype: int
    jtype: int
    Rmin: float
    bondtype: int
    iparam_maxbond: int = 0
    iparam_newtype: Optional[int] = None
    jparam_maxbond: int = 0
    jparam_newtype: Optional[int] = None
    prob_fraction: float = 1.0
    prob_seed: Optional[int] = None
    cg_type_map: Dict[int, int] = field(default_factory=dict)

    def to_lammps_command(self) -> str:
        """
        生成完整的 fix bond/create LAMMPS 命令字符串

        Returns:
            str: fix bond_create_fix all bond/create Nevery itype jtype Rmin bondtype [keywords...]
        """
        parts = [
            "fix bond_create_fix all bond/create",
            str(self.Nevery),
            str(self.itype),
            str(self.jtype),
            str(self.Rmin),
            str(self.bondtype),
        ]

        # iparam 关键字
        if self.iparam_maxbond > 0 or self.iparam_newtype is not None:
            newtype = self.iparam_newtype if self.iparam_newtype is not None else self.itype
            parts.append(f"iparam {self.iparam_maxbond} {newtype}")

        # jparam 关键字
        if self.jparam_maxbond > 0 or self.jparam_newtype is not None:
            newtype = self.jparam_newtype if self.jparam_newtype is not None else self.jtype
            parts.append(f"jparam {self.jparam_maxbond} {newtype}")

        # prob 关键字
        if self.prob_fraction < 1.0 and self.prob_seed is not None:
            parts.append(f"prob {self.prob_fraction} {self.prob_seed}")

        return " ".join(parts)


# ============================================================================
# 配置加载与验证
# ============================================================================

def load_bond_create_config(data: dict) -> BondCreateConfig:
    """
    从 YAML dict 加载 bond/create 配置并验证

    Args:
        data: YAML 中 bond_create 配置块的 dict

    Returns:
        BondCreateConfig: 验证通过的配置对象

    Raises:
        BondCreateConfigError: 配置验证失败
    """
    # 必填字段验证
    required_fields = ['Nevery', 'itype', 'jtype', 'Rmin', 'bondtype']
    for field_name in required_fields:
        if field_name not in data:
            raise BondCreateConfigError(f"bond_create 缺少必填字段: '{field_name}'")

    # 数值验证
    Nevery = int(data['Nevery'])
    if Nevery <= 0:
        raise BondCreateConfigError(f"Nevery 必须大于 0，当前值: {Nevery}")

    itype = int(data['itype'])
    jtype = int(data['jtype'])
    if itype <= 0 or jtype <= 0:
        raise BondCreateConfigError(f"itype/jtype 必须为正整数，当前值: {itype}/{jtype}")

    Rmin = float(data['Rmin'])
    if Rmin <= 0:
        raise BondCreateConfigError(f"Rmin 必须大于 0，当前值: {Rmin}")

    bondtype = int(data['bondtype'])
    if bondtype <= 0:
        raise BondCreateConfigError(f"bondtype 必须为正整数，当前值: {bondtype}")

    # 可选字段: iparam
    iparam = data.get('iparam', {})
    iparam_maxbond = int(iparam.get('maxbond', 0))
    iparam_newtype = iparam.get('newtype', None)
    if iparam_maxbond < 0:
        raise BondCreateConfigError(f"iparam.maxbond 不能为负数，当前值: {iparam_maxbond}")
    if iparam_newtype is not None:
        iparam_newtype = int(iparam_newtype)
        if iparam_newtype <= 0:
            raise BondCreateConfigError(f"iparam.newtype 必须为正整数，当前值: {iparam_newtype}")

    # 可选字段: jparam
    jparam = data.get('jparam', {})
    jparam_maxbond = int(jparam.get('maxbond', 0))
    jparam_newtype = jparam.get('newtype', None)
    if jparam_maxbond < 0:
        raise BondCreateConfigError(f"jparam.maxbond 不能为负数，当前值: {jparam_maxbond}")
    if jparam_newtype is not None:
        jparam_newtype = int(jparam_newtype)
        if jparam_newtype <= 0:
            raise BondCreateConfigError(f"jparam.newtype 必须为正整数，当前值: {jparam_newtype}")

    # 可选字段: prob
    prob = data.get('prob', {})
    prob_fraction = float(prob.get('fraction', 1.0))
    prob_seed = prob.get('seed', None)
    if prob_fraction <= 0.0 or prob_fraction > 1.0:
        raise BondCreateConfigError(
            f"prob.fraction 必须在 (0.0, 1.0] 范围内，当前值: {prob_fraction}"
        )
    if prob_seed is not None:
        prob_seed = int(prob_seed)
        if prob_seed <= 0:
            raise BondCreateConfigError(f"prob.seed 必须大于 0，当前值: {prob_seed}")

    # 可选字段: cg_update.type_map
    cg_update = data.get('cg_update', {})
    cg_type_map: Dict[int, int] = {}
    raw_type_map = cg_update.get('type_map', {})
    if raw_type_map:
        for k, v in raw_type_map.items():
            old_type = int(k)
            new_type = int(v)
            if old_type <= 0 or new_type <= 0:
                raise BondCreateConfigError(
                    f"cg_update.type_map 的 key/value 必须为正整数，当前值: {old_type} → {new_type}"
                )
            cg_type_map[old_type] = new_type

    return BondCreateConfig(
        Nevery=Nevery,
        itype=itype,
        jtype=jtype,
        Rmin=Rmin,
        bondtype=bondtype,
        iparam_maxbond=iparam_maxbond,
        iparam_newtype=iparam_newtype,
        jparam_maxbond=jparam_maxbond,
        jparam_newtype=jparam_newtype,
        prob_fraction=prob_fraction,
        prob_seed=prob_seed,
        cg_type_map=cg_type_map,
    )


# ============================================================================
# 模式判断
# ============================================================================

def get_reaction_mode(bond_create_enabled: bool) -> str:
    """
    根据配置判断当前反应模式

    Args:
        bond_create_enabled: bond_create.enabled 的值

    Returns:
        str: "bond/create" 或 "bond/react"
    """
    return "bond/create" if bond_create_enabled else "bond/react"


# ============================================================================
# LAMMPS fix 命令生成
# ============================================================================

def generate_fix_bond_create(config: BondCreateConfig) -> str:
    """
    生成 LAMMPS fix bond/create 命令字符串

    Args:
        config: bond/create 配置对象

    Returns:
        str: 完整的 fix 命令字符串
    """
    return config.to_lammps_command()


def generate_fix_bond_react(reactions, stabilization: float) -> str:
    """
    生成 LAMMPS fix bond/react 命令字符串

    从 run_refactored.py:_run_bond_react() 提取，行为完全不变。

    Args:
        reactions: ReactionInfo 列表
        stabilization: stabilization 参数

    Returns:
        str: 完整的 fix rxns all bond/react ... 命令字符串
    """
    # 构建反应命令
    react_cmds = []
    for rxn in reactions:
        react_cmds.append(
            f"react {rxn.name} all 1 0.0 {rxn.cutoff} "
            f"{rxn.pre_mol} {rxn.post_mol} {rxn.map_file}"
        )

    react_cmd = " ".join(react_cmds)

    return (
        f"fix rxns all bond/react stabilization yes npt_grp {stabilization} {react_cmd}"
    )


# ============================================================================
# CG 映射更新
# ============================================================================

def update_cg_mapping_create(
    cg_mapping_data: np.ndarray,
    bonds_before: np.ndarray,
    bonds_after: np.ndarray,
    type_map: Dict[int, int],
) -> bool:
    """
    bond/create 模式下更新 CG 映射中的 bead_type

    流程:
    1. AA 键 → CG 键转换（复用 atom_bonds_to_cg_bonds）
    2. CG 键差集（复用 get_cg_bond_diff）
    3. 对每个新 CG 键两端的 bead，查 type_map
    4. 向量化 mask 原地更新 cg_mapping_data 的 bead_type 列

    Args:
        cg_mapping_data: CG 映射数组 (n_atoms, 5) [bead_id, mol_id, bead_type, AA_id, mass]
        bonds_before: 反应前 AA 键 (n_bonds, 3) [bond_type, atom1, atom2]
        bonds_after: 反应后 AA 键 (n_bonds, 3)
        type_map: {old_bead_type: new_bead_type}

    Returns:
        bool: 是否有 bead_type 被更新
    """
    if not type_map:
        return False

    # 导入依赖（延迟导入避免循环依赖）
    try:
        from .cg_bond_mapper import atom_bonds_to_cg_bonds
        from .cg_reaction_identifier import get_cg_bond_diff
    except ImportError:
        from cg_bond_mapper import atom_bonds_to_cg_bonds
        from cg_reaction_identifier import get_cg_bond_diff

    # 1. AA → CG 键转换
    cg_bonds_before = atom_bonds_to_cg_bonds(bonds_before, cg_mapping_data)
    cg_bonds_after = atom_bonds_to_cg_bonds(bonds_after, cg_mapping_data)

    # 2. CG 键差集
    new_cg_bonds = get_cg_bond_diff(cg_bonds_before, cg_bonds_after)
    if not new_cg_bonds:
        return False

    # 3. 收集需要更新的 bead
    bead_type_updates: Dict[int, int] = {}
    # 构建 bead_id → bead_type 查找表
    bead_type_lut: Dict[int, int] = {}
    for row in cg_mapping_data:
        bid = int(row[0])
        if bid > 0 and bid not in bead_type_lut:
            bead_type_lut[bid] = int(row[2])

    for b1, b2 in new_cg_bonds:
        for bead_id in (b1, b2):
            old_type = bead_type_lut.get(bead_id)
            if old_type is not None and old_type in type_map:
                bead_type_updates[bead_id] = type_map[old_type]

    if not bead_type_updates:
        return False

    # 4. 向量化批量更新
    for bead_id, new_type in bead_type_updates.items():
        mask = cg_mapping_data[:, 0].astype(int) == bead_id
        cg_mapping_data[mask, 2] = float(new_type)

    return True


# ============================================================================
# 单元测试 (python core/reaction_commands.py)
# ============================================================================

if __name__ == "__main__":
    # 测试 1: 最小参数生成 bond/create 命令
    print("=" * 60)
    print("测试 bond/create 命令生成")
    print("=" * 60)

    config = BondCreateConfig(
        Nevery=10, itype=1, jtype=2, Rmin=0.8, bondtype=1
    )
    cmd = generate_fix_bond_create(config)
    expected = "fix bond_create_fix all bond/create 10 1 2 0.8 1"
    assert cmd == expected, f"期望: {expected}\n实际: {cmd}"
    print(f"  最小参数: {cmd}")
    print("  ✅ 通过")

    # 测试 2: 完整参数（iparam + jparam + prob）
    config_full = BondCreateConfig(
        Nevery=10, itype=1, jtype=2, Rmin=0.8, bondtype=1,
        iparam_maxbond=2, iparam_newtype=3,
        jparam_maxbond=0,
        prob_fraction=0.5, prob_seed=4928459
    )
    cmd_full = generate_fix_bond_create(config_full)
    expected_full = (
        "fix bond_create_fix all bond/create 10 1 2 0.8 1 "
        "iparam 2 3 prob 0.5 4928459"
    )
    assert cmd_full == expected_full, f"期望: {expected_full}\n实际: {cmd_full}"
    print(f"  完整参数: {cmd_full}")
    print("  ✅ 通过")

    # 测试 3: 配置验证
    print("\n测试 bond/create 配置验证")
    try:
        load_bond_create_config({})  # 缺少必填字段
        assert False, "应该抛出异常"
    except BondCreateConfigError as e:
        print(f"  缺少必填字段报错: {e}")
        print("  ✅ 通过")

    try:
        load_bond_create_config({
            'Nevery': 10, 'itype': 1, 'jtype': 2, 'Rmin': -0.5, 'bondtype': 1
        })
        assert False, "应该抛出异常"
    except BondCreateConfigError as e:
        print(f"  Rmin<=0 报错: {e}")
        print("  ✅ 通过")

    try:
        load_bond_create_config({
            'Nevery': 10, 'itype': 1, 'jtype': 2, 'Rmin': 0.8, 'bondtype': 1,
            'prob': {'fraction': 1.5, 'seed': 123}
        })
        assert False, "应该抛出异常"
    except BondCreateConfigError as e:
        print(f"  prob.fraction 越界报错: {e}")
        print("  ✅ 通过")

    # 测试 4: update_cg_mapping_create
    print("\n测试 CG 映射更新 (bond/create)")
    cg_mapping = np.array([
        [1, 1, 3, 1, 10.0],   # bead 1, type 3
        [2, 1, 5, 2, 10.0],   # bead 2, type 5
        [3, 2, 3, 3, 10.0],   # bead 3, type 3
        [4, 2, 5, 4, 10.0],   # bead 4, type 5
    ], dtype=np.float32)

    bonds_before = np.array([
        [1, 1, 2],  # bead1-bead2 (type 3-5)
    ], dtype=np.int32)

    bonds_after = np.array([
        [1, 1, 2],
        [1, 3, 4],  # bead3-bead4 新键 (type 3-5)
    ], dtype=np.int32)

    type_map = {3: 1, 5: 3}  # type 3→1, type 5→3

    updated = update_cg_mapping_create(cg_mapping, bonds_before, bonds_after, type_map)
    assert updated, "应该有更新"
    # 验证 bead 3 (type 3 → 1)
    mask3 = cg_mapping[:, 0].astype(int) == 3
    assert cg_mapping[mask3, 2][0] == 1.0, f"bead 3 type 应该变成 1，实际: {cg_mapping[mask3, 2]}"
    # 验证 bead 4 (type 5 → 3)
    mask4 = cg_mapping[:, 0].astype(int) == 4
    assert cg_mapping[mask4, 2][0] == 3.0, f"bead 4 type 应该变成 3，实际: {cg_mapping[mask4, 2]}"
    print("  ✅ 通过")

    # 测试 5: 空 type_map 不更新
    cg_mapping2 = np.array([
        [1, 1, 1, 1, 10.0],
        [2, 1, 2, 2, 10.0],
    ], dtype=np.float32)
    bonds2 = np.array([[1, 1, 2]], dtype=np.int32)
    updated2 = update_cg_mapping_create(cg_mapping2, bonds2, bonds2, {})
    assert not updated2, "空 type_map 应该返回 False"
    print("  ✅ 空 type_map 测试通过")

    print("\n" + "=" * 60)
    print("所有测试通过")
    print("=" * 60)
