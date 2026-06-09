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
from typing import Dict, List, Optional, Tuple
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
class BondCreatePair:
    """单个 bond/create 类型对配置"""
    itype: int
    jtype: int
    Nevery: int
    Rmin: float
    bondtype: int
    iparam_maxbond: int = 0
    iparam_newtype: Optional[int] = None
    jparam_maxbond: int = 0
    jparam_newtype: Optional[int] = None
    prob_fraction: float = 1.0
    prob_seed: Optional[int] = None

    def to_lammps_command(self, fix_id: str) -> str:
        """
        生成单个 fix bond/create LAMMPS 命令字符串

        Args:
            fix_id: fix 标识符

        Returns:
            str: fix {fix_id} all bond/create Nevery itype jtype Rmin bondtype [keywords...]
        """
        parts = [
            f"fix {fix_id} all bond/create",
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


@dataclass
class BondCreateConfig:
    """bond/create 配置容器"""
    pairs: List[BondCreatePair] = field(default_factory=list)
    cg_type_map: Dict[int, int] = field(default_factory=dict)
    sequential: bool = True                     # 顺序执行每对，避免多 fix 冲突
    relax_radius: float = 0.0                   # 松弛反应原子组半径 (Å)


# ============================================================================
# 配置加载与验证
# ============================================================================

def _parse_single_pair(data: dict, index: int) -> BondCreatePair:
    """解析并验证单个 pair 配置"""
    required_fields = ['itype', 'jtype', 'Nevery', 'Rmin', 'bondtype']
    for field_name in required_fields:
        if field_name not in data:
            raise BondCreateConfigError(
                f"bond_create.pairs[{index}] 缺少必填字段: '{field_name}'"
            )

    itype = int(data['itype'])
    jtype = int(data['jtype'])
    if itype <= 0 or jtype <= 0:
        raise BondCreateConfigError(
            f"pairs[{index}]: itype/jtype 必须为正整数，当前值: {itype}/{jtype}"
        )

    Nevery = int(data['Nevery'])
    if Nevery <= 0:
        raise BondCreateConfigError(
            f"pairs[{index}]: Nevery 必须大于 0，当前值: {Nevery}"
        )

    Rmin = float(data['Rmin'])
    if Rmin <= 0:
        raise BondCreateConfigError(
            f"pairs[{index}]: Rmin 必须大于 0，当前值: {Rmin}"
        )

    bondtype = int(data['bondtype'])
    if bondtype <= 0:
        raise BondCreateConfigError(
            f"pairs[{index}]: bondtype 必须为正整数，当前值: {bondtype}"
        )

    # iparam
    iparam = data.get('iparam', {})
    iparam_maxbond = int(iparam.get('maxbond', 0))
    iparam_newtype = iparam.get('newtype', None)
    if iparam_maxbond < 0:
        raise BondCreateConfigError(
            f"pairs[{index}]: iparam.maxbond 不能为负数"
        )
    if iparam_newtype is not None:
        iparam_newtype = int(iparam_newtype)
        if iparam_newtype <= 0:
            raise BondCreateConfigError(
                f"pairs[{index}]: iparam.newtype 必须为正整数"
            )

    # jparam
    jparam = data.get('jparam', {})
    jparam_maxbond = int(jparam.get('maxbond', 0))
    jparam_newtype = jparam.get('newtype', None)
    if jparam_maxbond < 0:
        raise BondCreateConfigError(
            f"pairs[{index}]: jparam.maxbond 不能为负数"
        )
    if jparam_newtype is not None:
        jparam_newtype = int(jparam_newtype)
        if jparam_newtype <= 0:
            raise BondCreateConfigError(
                f"pairs[{index}]: jparam.newtype 必须为正整数"
            )

    # prob
    prob = data.get('prob', {})
    prob_fraction = float(prob.get('fraction', 1.0))
    prob_seed = prob.get('seed', None)
    if prob_fraction <= 0.0 or prob_fraction > 1.0:
        raise BondCreateConfigError(
            f"pairs[{index}]: prob.fraction 必须在 (0.0, 1.0] 范围内"
        )
    if prob_seed is not None:
        prob_seed = int(prob_seed)
        if prob_seed <= 0:
            raise BondCreateConfigError(
                f"pairs[{index}]: prob.seed 必须大于 0"
            )

    return BondCreatePair(
        itype=itype, jtype=jtype,
        Nevery=Nevery, Rmin=Rmin, bondtype=bondtype,
        iparam_maxbond=iparam_maxbond, iparam_newtype=iparam_newtype,
        jparam_maxbond=jparam_maxbond, jparam_newtype=jparam_newtype,
        prob_fraction=prob_fraction, prob_seed=prob_seed,
    )


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
    # 解析 pairs 列表
    pairs_raw = data.get('pairs', [])
    if not pairs_raw:
        raise BondCreateConfigError("bond_create.pairs 不能为空，至少需要一个类型对")

    pairs = []
    for i, pair_data in enumerate(pairs_raw):
        pairs.append(_parse_single_pair(pair_data, i))

    # 解析 cg_update.type_map
    cg_update = data.get('cg_update', {})
    cg_type_map: Dict[int, int] = {}
    raw_type_map = cg_update.get('type_map', {})
    if raw_type_map:
        for k, v in raw_type_map.items():
            old_type = int(k)
            new_type = int(v)
            if old_type <= 0 or new_type <= 0:
                raise BondCreateConfigError(
                    f"cg_update.type_map 的 key/value 必须为正整数，"
                    f"当前值: {old_type} → {new_type}"
                )
            cg_type_map[old_type] = new_type

    # 解析 sequential 和 relax_radius
    sequential = bool(data.get('sequential', True))
    relax_radius = float(data.get('relax_radius', 0.0))
    if relax_radius < 0:
        raise BondCreateConfigError(f"relax_radius 不能为负数，当前值: {relax_radius}")

    return BondCreateConfig(
        pairs=pairs, cg_type_map=cg_type_map,
        sequential=sequential, relax_radius=relax_radius,
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

def generate_fix_bond_create(config: BondCreateConfig) -> List[Tuple[str, str]]:
    """
    生成 LAMMPS fix bond/create 命令列表（每对一个独立 fix）

    Args:
        config: bond/create 配置对象

    Returns:
        List[Tuple[str, str]]: [(fix_id, cmd), ...]
        如: [("bond_create_fix_0", "fix bond_create_fix_0 all bond/create ..."),
              ("bond_create_fix_1", "fix bond_create_fix_1 all bond/create ...")]
    """
    cmds = []
    for i, pair in enumerate(config.pairs):
        fix_id = f"bond_create_fix_{i}"
        cmds.append((fix_id, pair.to_lammps_command(fix_id)))
    return cmds


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

    try:
        from .cg_bond_mapper import atom_bonds_to_cg_bonds
        from .cg_reaction_identifier import get_cg_bond_diff
    except ImportError:
        from cg_bond_mapper import atom_bonds_to_cg_bonds
        from cg_reaction_identifier import get_cg_bond_diff

    cg_bonds_before = atom_bonds_to_cg_bonds(bonds_before, cg_mapping_data)
    cg_bonds_after = atom_bonds_to_cg_bonds(bonds_after, cg_mapping_data)

    new_cg_bonds = get_cg_bond_diff(cg_bonds_before, cg_bonds_after)
    if not new_cg_bonds:
        return False

    bead_type_lut: Dict[int, int] = {}
    for row in cg_mapping_data:
        bid = int(row[0])
        if bid > 0 and bid not in bead_type_lut:
            bead_type_lut[bid] = int(row[2])

    bead_type_updates: Dict[int, int] = {}
    for b1, b2 in new_cg_bonds:
        for bead_id in (b1, b2):
            old_type = bead_type_lut.get(bead_id)
            if old_type is not None and old_type in type_map:
                bead_type_updates[bead_id] = type_map[old_type]

    if not bead_type_updates:
        return False

    for bead_id, new_type in bead_type_updates.items():
        mask = cg_mapping_data[:, 0].astype(int) == bead_id
        cg_mapping_data[mask, 2] = float(new_type)

    return True


# ============================================================================
# 单元测试 (python core/reaction_commands.py)
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("测试 bond/create 多对命令生成")
    print("=" * 60)

    # 测试 1: 单对命令生成
    pair1 = BondCreatePair(itype=3, jtype=5, Nevery=1, Rmin=3.79, bondtype=1)
    cmd1 = pair1.to_lammps_command("bond_create_fix_0")
    expected1 = "fix bond_create_fix_0 all bond/create 1 3 5 3.79 1"
    assert cmd1 == expected1, f"期望: {expected1}\n实际: {cmd1}"
    print(f"  单对命令: {cmd1}")
    print("  ✅ 通过")

    # 测试 2: 完整参数对（iparam + jparam + prob）
    pair2 = BondCreatePair(
        itype=3, jtype=6, Nevery=10, Rmin=4.0, bondtype=2,
        iparam_maxbond=1, iparam_newtype=1,
        jparam_maxbond=1, jparam_newtype=4,
        prob_fraction=0.5, prob_seed=12345
    )
    cmd2 = pair2.to_lammps_command("bond_create_fix_1")
    expected2 = (
        "fix bond_create_fix_1 all bond/create 10 3 6 4.0 2 "
        "iparam 1 1 jparam 1 4 prob 0.5 12345"
    )
    assert cmd2 == expected2, f"期望: {expected2}\n实际: {cmd2}"
    print(f"  完整参数: {cmd2}")
    print("  ✅ 通过")

    # 测试 3: 多对配置加载
    print("\n测试多对配置加载")
    cfg = load_bond_create_config({
        'pairs': [
            {'itype': 3, 'jtype': 5, 'Nevery': 1, 'Rmin': 3.79, 'bondtype': 1,
             'iparam': {'maxbond': 1, 'newtype': 1},
             'jparam': {'maxbond': 1, 'newtype': 3}},
            {'itype': 3, 'jtype': 6, 'Nevery': 1, 'Rmin': 3.79, 'bondtype': 2,
             'iparam': {'maxbond': 1, 'newtype': 1},
             'jparam': {'maxbond': 1, 'newtype': 4}},
            {'itype': 4, 'jtype': 5, 'Nevery': 1, 'Rmin': 3.79, 'bondtype': 2,
             'iparam': {'maxbond': 1, 'newtype': 2},
             'jparam': {'maxbond': 1, 'newtype': 3}},
            {'itype': 4, 'jtype': 6, 'Nevery': 1, 'Rmin': 3.79, 'bondtype': 3,
             'iparam': {'maxbond': 1, 'newtype': 2},
             'jparam': {'maxbond': 1, 'newtype': 4}},
        ],
        'cg_update': {'type_map': {3: 1, 4: 2, 5: 3, 6: 4}}
    })
    assert len(cfg.pairs) == 4, f"期望 4 对，实际 {len(cfg.pairs)}"
    assert cfg.cg_type_map == {3: 1, 4: 2, 5: 3, 6: 4}
    print(f"  pairs 数: {len(cfg.pairs)}")
    print(f"  type_map: {cfg.cg_type_map}")
    print("  ✅ 通过")

    # 测试 4: generate_fix_bond_create 返回多对
    cmds = generate_fix_bond_create(cfg)
    assert len(cmds) == 4, f"期望 4 条命令，实际 {len(cmds)}"
    for fix_id, cmd in cmds:
        print(f"  {fix_id}: {cmd}")
    assert cmds[0][0] == "bond_create_fix_0"
    assert cmds[1][0] == "bond_create_fix_1"
    assert cmds[2][0] == "bond_create_fix_2"
    assert cmds[3][0] == "bond_create_fix_3"
    assert "bond/create 1 3 5 3.79 1 iparam 1 1 jparam 1 3" in cmds[0][1]
    assert "bond/create 1 3 6 3.79 2 iparam 1 1 jparam 1 4" in cmds[1][1]
    assert "bond/create 1 4 5 3.79 2 iparam 1 2 jparam 1 3" in cmds[2][1]
    assert "bond/create 1 4 6 3.79 3 iparam 1 2 jparam 1 4" in cmds[3][1]
    print("  ✅ 4 对命令验证通过")

    # 测试 5: 验证
    print("\n测试配置验证")
    try:
        load_bond_create_config({'pairs': []})
        assert False, "应该抛出异常"
    except BondCreateConfigError as e:
        print(f"  空 pairs 报错: {e}")
        print("  ✅ 通过")

    try:
        load_bond_create_config({'pairs': [
            {'itype': 3, 'jtype': 5, 'Nevery': -1, 'Rmin': 3.79, 'bondtype': 1}
        ]})
        assert False, "应该抛出异常"
    except BondCreateConfigError as e:
        print(f"  Nevery<=0 报错: {e}")
        print("  ✅ 通过")

    print("\n" + "=" * 60)
    print("所有测试通过")
    print("=" * 60)
