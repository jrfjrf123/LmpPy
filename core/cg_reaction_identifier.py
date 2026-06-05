#!/usr/bin/env python3
"""
CG级别反应识别模块

功能:
- 从 reactions/ 目录加载 3-bead 类型签名
- 基于CG键变化识别反应类型
- 作为 ReactionLocator + CGMapper 主流程的独立交叉验证机制

核心设计:
- 完全独立于主流程
- 无 LAMMPS 依赖，纯 Python/NumPy 逻辑
- 动态推断 end/monomer 类型，无硬编码

作者: Claude Code
日期: 2026-06-04
"""

from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import yaml
from dataclasses import dataclass, field


@dataclass
class CGReactionSignature:
    """从模板提取的 CG 级反应签名（生产级，供 _update_cg_mapping 使用）"""
    name: str                                    # "rxn1_EEE"
    signature_3bead: Tuple[int, int, int]        # (interior_type, end_type, monomer_type)
    pre_chains: Dict[int, Tuple[int, ...]]       # initiator_bead_type → 向外的类型链
    type_map: Dict[int, int]                     # {old_bead_type: new_bead_type}
    bead_id_map: Dict[int, int] = field(default_factory=dict)  # {old_local_bead_id: new_local_bead_id}，空=不变


def _trace_bead_chain(start_bead: int,
                      bead_bonds: Set[Tuple[int, int]],
                      edge_bead_ids: Set[int]) -> Tuple[int, ...]:
    """
    从 start_bead 沿 bead_bonds 向外遍历直到 edge 或链端，返回途经 bead ID 序列。

    模板初始化阶段使用。不包含 edge bead 本身。线性链假设：取第一个非 edge 邻居。
    """
    chain = []
    current = start_bead
    visited = {current}
    while True:
        neighbors = []
        for b1, b2 in bead_bonds:
            if b1 == current and b2 not in visited:
                neighbors.append(b2)
            elif b2 == current and b1 not in visited:
                neighbors.append(b1)
        non_edge = [n for n in neighbors if n not in edge_bead_ids]
        if not non_edge:
            break
        next_bead = non_edge[0]
        chain.append(next_bead)
        visited.add(next_bead)
        current = next_bead
    return tuple(chain)


def load_template_signatures(reactions_dir: Path) -> Tuple[
        Dict[Tuple[int, int, int], List[CGReactionSignature]],
        List[CGReactionSignature],
        Set[int], Set[int], Set[int]
]:
    """
    从 reactions/ 目录加载所有模板的 CG 级反应签名（生产级）。

    与 load_reaction_signatures() 并存——后者供 smoke_validator 交叉验证使用。

    Returns:
        signature_index: {(interior_type, end_type, monomer_type): [CGReactionSignature, ...]}
        all_signatures: 所有签名的列表
        end_types: 所有 end bead 类型集合 (运行时区分 end/monomer)
        monomer_types: 所有 monomer bead 类型集合
        interior_types: 所有 interior bead 类型集合
    """
    signature_index: Dict[Tuple[int, int, int], List[CGReactionSignature]] = defaultdict(list)
    all_signatures: List[CGReactionSignature] = []
    end_types: Set[int] = set()
    monomer_types: Set[int] = set()
    interior_types: Set[int] = set()

    reactions_path = Path(reactions_dir)
    if not reactions_path.exists() or not reactions_path.is_dir():
        return dict(signature_index), all_signatures, end_types, monomer_types, interior_types

    for rxn_dir in sorted(reactions_path.iterdir()):
        if not rxn_dir.is_dir():
            continue
        rxn_name = rxn_dir.name

        # 1. 加载 pre/post mapping YAML
        pre_yaml = rxn_dir / f"{rxn_name}_pre_mapping.yaml"
        post_yaml = rxn_dir / f"{rxn_name}_post_mapping.yaml"
        if not pre_yaml.exists() or not post_yaml.exists():
            continue

        try:
            with open(pre_yaml, 'r') as f:
                pre_data = yaml.safe_load(f.read().replace('\t', '  '))['mapping']
            with open(post_yaml, 'r') as f:
                post_data = yaml.safe_load(f.read().replace('\t', '  '))['mapping']
        except (yaml.YAMLError, TypeError, KeyError, ValueError) as e:
            print(f"  警告: 解析 {rxn_name} 的 mapping YAML 失败: {e}")
            continue

        # 2. 解析 .map → InitiatorIDs, EdgeIDs, Equivalences
        map_file = rxn_dir / f"{rxn_name}.map"
        initiator_ids, edge_ids, equivalences = [], [], {}
        if map_file.exists():
            with open(map_file, 'r') as f:
                section = None
                for line in f:
                    line = line.strip()
                    if line == 'InitiatorIDs': section = 'initiator'; continue
                    elif line == 'EdgeIDs': section = 'edge'; continue
                    elif line == 'Equivalences': section = 'equiv'; continue
                    elif line.startswith('Constraints'): section = 'constraint'; continue
                    if not line: continue
                    if section == 'initiator': initiator_ids.append(int(line))
                    elif section == 'edge': edge_ids.append(int(line))
                    elif section == 'equiv':
                        parts = line.split()
                        if len(parts) >= 2: equivalences[int(parts[0])] = int(parts[1])

        # 3. 原子→bead 映射 + pre bead 键连图
        atom_to_bead: Dict[int, int] = {}
        pre_bead_types: Dict[int, int] = {}
        for bead_id_str, info in pre_data.items():
            bid = int(bead_id_str)
            pre_bead_types[bid] = info['bead_type']
            for a in info['atoms']:
                atom_to_bead[a] = bid

        pre_template_file = rxn_dir / f"{rxn_name}_pre.lammpstemplate"
        post_template_file = rxn_dir / f"{rxn_name}_post.lammpstemplate"
        bead_bonds: Set[Tuple[int, int]] = set()
        if pre_template_file.exists():
            with open(pre_template_file, 'r') as f:
                in_bonds = False
                for line in f:
                    line = line.strip()
                    if line.startswith('Bonds'): in_bonds = True; continue
                    if in_bonds:
                        if not line: continue
                        if line.startswith('Angles') or line.startswith('Dihedrals'): break
                        parts = line.split()
                        if len(parts) >= 4:
                            a1, a2 = int(parts[2]), int(parts[3])
                            b1, b2 = atom_to_bead.get(a1), atom_to_bead.get(a2)
                            if b1 is not None and b2 is not None and b1 != b2:
                                bead_bonds.add((min(b1, b2), max(b1, b2)))

        # 4. 找 post 新增 bead 键 → 确定 end/monomer
        post_bead_bonds: Set[Tuple[int, int]] = set()
        if post_template_file.exists():
            with open(post_template_file, 'r') as f:
                in_bonds = False
                for line in f:
                    line = line.strip()
                    if line.startswith('Bonds'): in_bonds = True; continue
                    if in_bonds:
                        if not line: continue
                        if line.startswith('Angles') or line.startswith('Dihedrals'): break
                        parts = line.split()
                        if len(parts) >= 4:
                            a1, a2 = int(parts[2]), int(parts[3])
                            b1, b2 = atom_to_bead.get(a1), atom_to_bead.get(a2)
                            if b1 is not None and b2 is not None and b1 != b2:
                                post_bead_bonds.add((min(b1, b2), max(b1, b2)))

        new_bead_bonds = post_bead_bonds - bead_bonds
        if not new_bead_bonds:
            continue

        # 5. 从新增键推导 3-bead 签名
        degree: Dict[int, int] = defaultdict(int)
        for b1, b2 in bead_bonds:
            degree[b1] += 1; degree[b2] += 1

        new_bead_bonds_list = list(new_bead_bonds)
        if not new_bead_bonds_list:
            continue
        # 对于当前模板，每个反应仅创建一条 bead 间新键
        # 若出现多条新键，仍取第一条推导签名（后续 chain 精筛会处理歧义）
        first_new = new_bead_bonds_list[0]
        b_a, b_b = first_new
        deg_a, deg_b = degree.get(b_a, 0), degree.get(b_b, 0)
        if deg_a == 0 and deg_b > 0:
            monomer_bead, end_bead = b_a, b_b
        elif deg_b == 0 and deg_a > 0:
            monomer_bead, end_bead = b_b, b_a
        else:
            if deg_a <= deg_b:
                monomer_bead, end_bead = b_a, b_b
            else:
                monomer_bead, end_bead = b_b, b_a

        interior_bead = None
        for b1, b2 in bead_bonds:
            if b1 == end_bead: interior_bead = b2; break
            elif b2 == end_bead: interior_bead = b1; break
        if interior_bead is None:
            continue

        interior_type = pre_bead_types[interior_bead]
        end_type = pre_bead_types[end_bead]
        monomer_type = pre_bead_types[monomer_bead]
        sig_3bead = (interior_type, end_type, monomer_type)

        # 6. 构建 pre_chains
        edge_bead_ids = {atom_to_bead[e] for e in edge_ids if e in atom_to_bead}
        pre_chains: Dict[int, Tuple[int, ...]] = {}
        for init_id in initiator_ids:
            bead_id = atom_to_bead.get(init_id)
            if bead_id is None:
                continue
            btype = pre_bead_types[bead_id]
            chain_bead_ids = _trace_bead_chain(bead_id, bead_bonds, edge_bead_ids)
            chain_types = tuple(pre_bead_types[bid] for bid in chain_bead_ids)
            # 以 bead_type 为 key 进行去重——相同类型的 initiator 应有等价的链拓扑
            pre_chains[btype] = chain_types

        # 7. 构建 type_map
        post_bead_types: Dict[int, int] = {}
        for bead_id_str, info in post_data.items():
            post_bead_types[int(bead_id_str)] = info['bead_type']

        type_map: Dict[int, int] = {}
        for bid, pre_type in pre_bead_types.items():
            post_type = post_bead_types.get(bid, pre_type)
            if pre_type != post_type:
                type_map[pre_type] = post_type

        # 8. 构建 bead_id_map
        post_atom_to_bead: Dict[int, int] = {}
        for bead_id_str, info in post_data.items():
            bid = int(bead_id_str)
            for a in info['atoms']:
                post_atom_to_bead[a] = bid
        bead_id_map: Dict[int, int] = {}
        for pre_atom, post_atom in equivalences.items():
            pre_bead = atom_to_bead.get(pre_atom)
            post_bead = post_atom_to_bead.get(post_atom)
            if pre_bead is not None and post_bead is not None and pre_bead != post_bead:
                bead_id_map[pre_bead] = post_bead

        # 9. 索引
        sig = CGReactionSignature(
            name=rxn_name,
            signature_3bead=sig_3bead,
            pre_chains=pre_chains,
            type_map=type_map,
            bead_id_map=bead_id_map,
        )
        signature_index[sig_3bead].append(sig)
        all_signatures.append(sig)
        end_types.add(end_type)
        monomer_types.add(monomer_type)
        interior_types.add(interior_type)

    return (dict(signature_index), all_signatures, end_types, monomer_types, interior_types)


def validate_chain(cg_graph_after: Dict[int, List[int]],
                   start_bead: int,
                   exclude_bead: int,
                   expected_chain: Tuple[int, ...],
                   bead_type_lut: Dict[int, int]) -> bool:
    """
    沿反应后 CG 键图验证类型链。

    从 start_bead 出发，排除 exclude_bead，逐层比对邻居 bead_type 是否与
    expected_chain 一致。expected_chain 为空时直接返回 True（孤立 initiator）。

    Args:
        cg_graph_after: 反应后 CG 键图 {bead_id: [neighbor_bead_ids]}
        start_bead: 起始 bead (initiator)
        exclude_bead: 排除的邻居 (新键对端)
        expected_chain: 期望的 bead_type 序列
        bead_type_lut: {bead_id: bead_type}

    Returns:
        是否匹配
    """
    if not expected_chain:
        return True

    current = start_bead
    for expected_type in expected_chain:
        neighbors = [n for n in cg_graph_after.get(current, [])
                     if n != exclude_bead]
        if not neighbors:
            return False

        found = False
        for nb in neighbors:
            if bead_type_lut.get(nb) == expected_type:
                current = nb
                exclude_bead = None
                found = True
                break
        if not found:
            return False

    return True


def match_reaction(new_bond: Tuple[int, int],
                   bead_type_lut: Dict[int, int],
                   cg_graph_after: Dict[int, List[int]],
                   signature_index: Dict[Tuple[int, int, int], List[CGReactionSignature]],
                   end_types: Set[int],
                   monomer_types: Set[int]) -> Optional[CGReactionSignature]:
    """
    两级匹配：3-bead 粗筛 → [chain 精筛(按需)]。

    阶段1: 从新 CG 键找 end/monomer → interior neighbor → 3-bead 签名查表。
            唯一命中则直接返回。
    阶段2: 多个候选时，用 validate_chain() 沿链验证甄别。
            仍无法唯一确定 → MPI Abort / sys.exit(1)。

    Args:
        new_bond: 新 CG 键 (bead1, bead2)，已规范化 (小在前)
        bead_type_lut: {bead_id: bead_type}
        cg_graph_after: 反应后 CG 键图
        signature_index: load_template_signatures() 返回的索引
        end_types: 所有 end bead 类型
        monomer_types: 所有 monomer bead 类型

    Returns:
        匹配的 CGReactionSignature 或 None

    Raises:
        SystemExit: 多个候选且 chain 精筛后仍无法唯一确定时终止进程
    """
    b1, b2 = new_bond
    t1 = bead_type_lut.get(b1)
    t2 = bead_type_lut.get(b2)
    if t1 is None or t2 is None:
        return None

    # --- 阶段 1: 3-bead 粗筛 ---
    # 区分 end/monomer
    if t1 in end_types and t2 in monomer_types:
        end_bead, monomer_bead = b1, b2
        end_type, monomer_type = t1, t2
    elif t2 in end_types and t1 in monomer_types:
        end_bead, monomer_bead = b2, b1
        end_type, monomer_type = t2, t1
    else:
        return None

    # 找 interior neighbor (end_bead 在图中除 monomer 外的邻居)
    interior_bead = None
    interior_type = None
    interior_types: Set[int] = {k[0] for k in signature_index.keys()}
    for nb in cg_graph_after.get(end_bead, []):
        if nb == monomer_bead:
            continue
        nb_type = bead_type_lut.get(nb)
        if nb_type in interior_types:
            interior_bead = nb
            interior_type = nb_type
            break

    if interior_bead is None:
        return None

    # 查表
    sig_3bead = (interior_type, end_type, monomer_type)
    candidates = signature_index.get(sig_3bead, [])
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]    # 快速路径

    # --- 阶段 2: chain 精筛 ---
    matched = []
    for sig in candidates:
        if not validate_chain(cg_graph_after, end_bead, monomer_bead,
                              sig.pre_chains.get(end_type, ()), bead_type_lut):
            continue
        if not validate_chain(cg_graph_after, monomer_bead, end_bead,
                              sig.pre_chains.get(monomer_type, ()), bead_type_lut):
            continue
        matched.append(sig)

    if len(matched) == 1:
        return matched[0]

    # 无法唯一匹配 → FATAL
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
    except ImportError:
        comm = None

    if comm is not None and comm.Get_size() > 1:
        if comm.Get_rank() == 0:
            print(f"FATAL: 无法唯一确定反应类型")
            print(f"  新键: {new_bond}")
            print(f"  3-bead 签名: {sig_3bead}")
            for s in candidates:
                print(f"  候选: {s.name}, pre_chains={s.pre_chains}")
            if matched:
                print(f"  chain 精筛后: {[s.name for s in matched]}")
        comm.Abort(1)
    else:
        import sys
        print(f"FATAL: 无法唯一确定反应类型")
        print(f"  新键: {new_bond}")
        print(f"  3-bead 签名: {sig_3bead}")
        for s in candidates:
            print(f"  候选: {s.name}, pre_chains={s.pre_chains}")
        sys.exit(1)

    return None  # unreachable


# ============================================================
# 步骤 1: 从模板 YAML 加载 3-bead 类型签名 (交叉验证)
# ============================================================

def load_reaction_signatures(reactions_dir: Path) -> Dict[Tuple[int, ...], Dict]:
    """
    从 reactions/ 目录加载所有反应模板的 3-bead 类型签名。

    通过分析模板的键连图确定 bead 间拓扑关系:
    - 所有模板 pre 图中 bead 1 — bead 2 有键连接 (interior-end 对)
    - monomer bead 在 pre 图中孤立
    - 签名顺序: [interior_bead_type, end_bead_type, monomer_bead_type]

    动态推断 bead 角色 (无硬编码):
    - interior: 在 pre 图中与 end 有键连接，且度数较高
    - end: 在 interior-end 对中，类型发生变化 (pre_type != post_type)
           或含有 initiator 原子
    - monomer: 在 pre 图中孤立 (无 bead 间键)

    Args:
        reactions_dir: reactions/ 目录路径，包含 rxn1_EEE/ 等子目录

    Returns:
        dict: {pre_signature_tuple: reaction_info_dict}
              reaction_info 包含 'name', 'post_sig', 'type_map', 'bead_roles'
    """
    signatures = {}
    reactions_path = Path(reactions_dir)

    for rxn_dir in sorted(reactions_path.iterdir()):
        if not rxn_dir.is_dir():
            continue
        rxn_name = rxn_dir.name

        # 加载 pre 和 post mapping YAML
        pre_yaml = rxn_dir / f"{rxn_name}_pre_mapping.yaml"
        post_yaml = rxn_dir / f"{rxn_name}_post_mapping.yaml"

        if not pre_yaml.exists() or not post_yaml.exists():
            continue

        # YAML 文件可能包含制表符，需要预处理
        with open(pre_yaml, 'r') as f:
            pre_content = f.read().replace('\t', '  ')
            pre_data = yaml.safe_load(pre_content)['mapping']
        with open(post_yaml, 'r') as f:
            post_content = f.read().replace('\t', '  ')
            post_data = yaml.safe_load(post_content)['mapping']

        # 解析 bead groups: {bead_id: (bead_type, atom_ids)}
        pre_groups = {}
        for bead_id_str, info in pre_data.items():
            bid = int(bead_id_str)
            pre_groups[bid] = (info['bead_type'], info['atoms'])

        post_groups = {}
        for bead_id_str, info in post_data.items():
            bid = int(bead_id_str)
            post_groups[bid] = (info['bead_type'], info['atoms'])

        # 加载 .map 文件获取 initiator_ids 和 edge_ids
        map_file = rxn_dir / f"{rxn_name}.map"
        initiator_ids = []
        edge_ids = []
        if map_file.exists():
            with open(map_file, 'r') as f:
                section = None
                for line in f:
                    line = line.strip()
                    if line == 'InitiatorIDs':
                        section = 'initiator'
                        continue
                    elif line == 'EdgeIDs':
                        section = 'edge'
                        continue
                    elif line == 'Equivalences':
                        section = 'equiv'
                        continue
                    elif line.startswith('Constraints'):
                        section = 'constraint'
                        continue
                    if not line:
                        continue
                    if section == 'initiator':
                        initiator_ids.append(int(line))
                    elif section == 'edge':
                        edge_ids.append(int(line))

        # 确定每个模板原子属于哪个 bead
        atom_to_bead = {}
        for bid, (_, atoms) in pre_groups.items():
            for a in atoms:
                atom_to_bead[a] = bid

        # 加载 pre 模板文件确定 bead 间键连
        pre_template_file = rxn_dir / f"{rxn_name}_pre.lammpstemplate"
        bead_bonds = set()
        if pre_template_file.exists():
            with open(pre_template_file, 'r') as f:
                in_bonds = False
                for line in f:
                    line = line.strip()
                    if line.startswith('Bonds'):
                        in_bonds = True
                        continue
                    if in_bonds:
                        if not line:
                            continue
                        if line.startswith('Angles') or line.startswith('Dihedrals'):
                            break
                        parts = line.split()
                        if len(parts) >= 4:
                            a1, a2 = int(parts[2]), int(parts[3])
                            b1 = atom_to_bead.get(a1)
                            b2 = atom_to_bead.get(a2)
                            if b1 is not None and b2 is not None and b1 != b2:
                                bead_bonds.add(tuple(sorted([b1, b2])))

        # 确定 interior, end, monomer bead (动态推断，无硬编码)
        # 计算每个 bead 在 pre 图中的度数
        degree = defaultdict(int)
        for b1, b2 in bead_bonds:
            degree[b1] += 1
            degree[b2] += 1

        all_beads = set(pre_groups.keys())

        # monomer bead: pre 图中度数为 0 (孤立) 的 bead
        monomer_candidates = [b for b in all_beads if degree.get(b, 0) == 0]
        if len(monomer_candidates) != 1:
            continue
        monomer_bead = monomer_candidates[0]

        # end bead: 在 pre->post 中类型发生变化的 bead，且在链上度数较小
        # 先找出所有类型发生变化的 bead
        changing_beads = []
        for bid in all_beads:
            if bid == monomer_bead:
                continue
            if pre_groups[bid][0] != post_groups[bid][0]:
                changing_beads.append(bid)

        if not changing_beads:
            continue

        # end bead 通常是变化的 bead 中度数最小的 (链末端度数为 1)
        # 按度数升序、然后按 bead_id 升序排序以保持确定性
        changing_beads.sort(key=lambda b: (degree.get(b, 0), b))
        end_bead = changing_beads[0]

        # interior bead: 与 end_bead 在 pre 图中有键连接的 bead
        # 取与 end 相连的、度数最大的邻居 (更靠近链中间)
        end_neighbors = []
        for b1, b2 in bead_bonds:
            if b1 == end_bead:
                end_neighbors.append(b2)
            elif b2 == end_bead:
                end_neighbors.append(b1)

        if not end_neighbors:
            continue

        # 选择度数最大的邻居作为 interior (更靠近链中间)
        end_neighbors.sort(key=lambda b: (-degree.get(b, 0), b))
        interior_bead = end_neighbors[0]

        # 构建签名
        pre_sig = (pre_groups[interior_bead][0],
                   pre_groups[end_bead][0],
                   pre_groups[monomer_bead][0])
        post_sig = (post_groups[interior_bead][0],
                    post_groups[end_bead][0],
                    post_groups[monomer_bead][0])

        # type_map: {old_bead_type: new_bead_type} (仅记录有变化的)
        type_map = {}
        for bid, (pre_type, _) in pre_groups.items():
            post_type = post_groups[bid][0]
            if pre_type != post_type:
                type_map[pre_type] = post_type

        signatures[pre_sig] = {
            'name': rxn_name,
            'post_sig': post_sig,
            'type_map': type_map,
            'bead_roles': {
                'interior': (interior_bead, pre_groups[interior_bead][0]),
                'end': (end_bead, pre_groups[end_bead][0]),
                'monomer': (monomer_bead, pre_groups[monomer_bead][0]),
            }
        }

    return signatures


# ============================================================
# 步骤 2: CG 级别反应识别
# ============================================================

def build_cg_bond_graph(cg_bonds: np.ndarray) -> Dict[int, List[int]]:
    """
    从 CG 键连表构建邻接图。

    Args:
        cg_bonds: CG 键数组 (n_bonds, 3) [bond_type, bead1, bead2]

    Returns:
        dict: {bead_id: [neighbor_bead_ids]}
    """
    graph = defaultdict(list)
    for bond in cg_bonds:
        b1, b2 = int(bond[1]), int(bond[2])
        graph[b1].append(b2)
        graph[b2].append(b1)
    return dict(graph)


def get_cg_bond_diff(cg_bonds_before: np.ndarray,
                     cg_bonds_after: np.ndarray) -> List[Tuple[int, int]]:
    """
    获取 CG 键差集 (after - before)。

    Args:
        cg_bonds_before: 反应前 CG 键数组
        cg_bonds_after: 反应后 CG 键数组

    Returns:
        list of (bead1, bead2): 新建的 CG 键
    """
    def encode(b):
        return (int(b[1]), int(b[2]))

    before_set = {encode(b) for b in cg_bonds_before}
    after_set = {encode(b) for b in cg_bonds_after}

    # 规范化 (确保小在前)
    new_bonds = []
    for b1, b2 in (after_set - before_set):
        new_bonds.append((min(b1, b2), max(b1, b2)))

    return new_bonds


def identify_reaction(cg_bonds_before: np.ndarray,
                      cg_bonds_after: np.ndarray,
                      cg_mapping: np.ndarray,
                      signatures: Dict[Tuple[int, ...], Dict]) -> List[Dict]:
    """
    从 CG 级别信息识别反应类型。

    动态推断 end_types 和 monomer_types 从签名表:
    - end_types: 所有签名中 end 角色的类型集合
    - monomer_types: 所有签名中 monomer 角色的类型集合
    - interior_types: 所有签名中 interior 角色的类型集合

    Args:
        cg_bonds_before: 反应前 CG 键数组 (n_bonds, 3)
        cg_bonds_after: 反应后 CG 键数组 (n_bonds, 3)
        cg_mapping: CG 映射数组 (n_atoms, 5) [bead_id, mol_id, bead_type, AA_id, mass]
        signatures: load_reaction_signatures() 的返回值

    Returns:
        List[dict]: 识别结果列表，每个元素包含:
            - 'signature': tuple, 匹配的 3-bead 签名
            - 'name': str, 反应名称
            - 'type_map': dict, 类型变化映射
            - 'involved_beads': dict, 涉及的 bead ID
            - 'bead_types_before': dict, 反应前各 bead 类型
            - 'bead_types_after': dict, 反应后各 bead 类型 (应用 type_map)
        空列表表示无识别结果
    """
    # 找新 CG 键
    new_bonds = get_cg_bond_diff(cg_bonds_before, cg_bonds_after)
    if not new_bonds:
        return []

    if not signatures:
        return []

    # 构建 CG 键图 (反应后，因为新键已存在)
    cg_graph = build_cg_bond_graph(cg_bonds_after)

    # 构建 bead_id -> bead_type 映射
    bead_types = {}
    for row in cg_mapping:
        bid = int(row[0])
        btype = int(row[2])
        if bid > 0:
            bead_types[bid] = btype

    # 从签名表动态推断角色类型集合
    end_types = set()
    monomer_types = set()
    interior_types = set()
    for sig in signatures.keys():
        interior_types.add(sig[0])
        end_types.add(sig[1])
        monomer_types.add(sig[2])

    results = []

    for b1, b2 in new_bonds:
        type1 = bead_types.get(b1)
        type2 = bead_types.get(b2)
        if type1 is None or type2 is None:
            continue

        # 区分链端和单体: 基于签名表动态推断的类型集合
        if type1 in end_types and type2 in monomer_types:
            end_bead, monomer_bead = b1, b2
            end_type, monomer_type = type1, type2
        elif type2 in end_types and type1 in monomer_types:
            end_bead, monomer_bead = b2, b1
            end_type, monomer_type = type2, type1
        else:
            continue  # 不符合预期模式

        # 找链端 bead 的链内邻居 (interior 类型，不是单体)
        interior_bead = None
        interior_type = None
        for nb in cg_graph.get(end_bead, []):
            if nb == monomer_bead:
                continue  # 跳过新键连接的单体
            nb_type = bead_types.get(nb)
            if nb_type in interior_types:
                interior_bead = nb
                interior_type = nb_type
                break

        if interior_bead is None:
            continue

        # 构建 3-bead 签名
        pre_sig = (interior_type, end_type, monomer_type)

        # 查表
        if pre_sig in signatures:
            sig_info = signatures[pre_sig]
            results.append({
                'signature': pre_sig,
                'name': sig_info['name'],
                'type_map': sig_info['type_map'],
                'involved_beads': {
                    'interior': interior_bead,
                    'end': end_bead,
                    'monomer': monomer_bead,
                },
                'bead_types_before': {
                    interior_bead: interior_type,
                    end_bead: end_type,
                    monomer_bead: monomer_type,
                },
                'bead_types_after': {
                    interior_bead: sig_info['type_map'].get(interior_type, interior_type),
                    end_bead: sig_info['type_map'].get(end_type, end_type),
                    monomer_bead: sig_info['type_map'].get(monomer_type, monomer_type),
                },
            })

    return results


# ============================================================
# Self-Test
# ============================================================

if __name__ == "__main__":
    import sys

    # 将项目根目录加入路径以便导入
    # worktree 路径: .../LmpPy/.worktrees/smoke-test/core/
    # 项目根目录: .../lmp_py_react/ (parent^5)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent.parent.parent.parent
    sys.path.insert(0, str(project_root))

    from LmpPy.core.cg_bond_mapper import atom_bonds_to_cg_bonds

    print("=" * 60)
    print("CG 反应识别模块自测")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 测试 1: 从真实 reactions 目录加载签名
    # ------------------------------------------------------------------
    reactions_dir = Path("/home/jrf/PythonProject/lmp_py_react/test_EPR/large_monomers_system/common_CG_map/reactions")

    print("\n[测试 1] 加载反应签名...")
    signatures = load_reaction_signatures(reactions_dir)
    print(f"  加载了 {len(signatures)} 个反应签名:")
    for sig, info in sorted(signatures.items(), key=lambda x: x[1]['name']):
        print(f"    {list(sig)} -> {info['name']}, type_map={info['type_map']}")
        print(f"      roles: interior={info['bead_roles']['interior']}, "
              f"end={info['bead_roles']['end']}, monomer={info['bead_roles']['monomer']}")

    assert len(signatures) == 8, f"期望 8 个签名，实际 {len(signatures)}"

    # 验证 rxn1_EEE 签名
    expected_rxn1_sig = (1, 3, 5)
    assert expected_rxn1_sig in signatures, f"缺少 rxn1_EEE 签名 {expected_rxn1_sig}"
    assert signatures[expected_rxn1_sig]['name'] == 'rxn1_EEE'
    assert signatures[expected_rxn1_sig]['type_map'] == {3: 1, 5: 3}
    print("  rxn1_EEE 签名验证通过")

    # 验证 rxn8_PPP 签名
    expected_rxn8_sig = (2, 4, 6)
    assert expected_rxn8_sig in signatures, f"缺少 rxn8_PPP 签名 {expected_rxn8_sig}"
    assert signatures[expected_rxn8_sig]['name'] == 'rxn8_PPP'
    assert signatures[expected_rxn8_sig]['type_map'] == {4: 2, 6: 4}
    print("  rxn8_PPP 签名验证通过")

    print("\n[测试 1] 通过")

    # ------------------------------------------------------------------
    # 测试 2: 使用合成 CG 键数据测试反应识别 (rxn1_EEE 签名 1,3,5)
    # ------------------------------------------------------------------
    print("\n[测试 2] 反应识别 - rxn1_EEE 场景...")

    # 构建合成 CG 映射: 3 个 bead
    # bead_id, mol_id, bead_type, AA_id, mass
    cg_mapping = np.array([
        [1, 1, 1, 1, 10.0],   # interior bead, type 1
        [2, 1, 3, 2, 10.0],   # end bead, type 3
        [3, 2, 5, 3, 10.0],   # monomer bead, type 5
    ], dtype=np.float32)

    # 反应前: interior-end 有键，monomer 孤立
    cg_bonds_before = np.array([
        [1, 1, 2],  # interior - end
    ], dtype=np.int32)

    # 反应后: 新增 end-monomer 键
    cg_bonds_after = np.array([
        [1, 1, 2],  # interior - end
        [1, 2, 3],  # end - monomer (新键)
    ], dtype=np.int32)

    results = identify_reaction(cg_bonds_before, cg_bonds_after, cg_mapping, signatures)

    assert len(results) == 1, f"期望 1 个识别结果，实际 {len(results)}"
    result = results[0]
    assert result['signature'] == (1, 3, 5), f"签名不匹配: {result['signature']}"
    assert result['name'] == 'rxn1_EEE', f"反应名不匹配: {result['name']}"
    assert result['type_map'] == {3: 1, 5: 3}, f"type_map 不匹配: {result['type_map']}"
    assert result['involved_beads'] == {'interior': 1, 'end': 2, 'monomer': 3}
    assert result['bead_types_before'] == {1: 1, 2: 3, 3: 5}
    assert result['bead_types_after'] == {1: 1, 2: 1, 3: 3}

    print(f"  识别结果: {result['name']}")
    print(f"  签名: {result['signature']}")
    print(f"  涉及 bead: {result['involved_beads']}")
    print(f"  类型变化: {result['bead_types_before']} -> {result['bead_types_after']}")
    print("\n[测试 2] 通过")

    # ------------------------------------------------------------------
    # 测试 3: 无新键时返回空列表
    # ------------------------------------------------------------------
    print("\n[测试 3] 无新键场景...")

    cg_bonds_same = np.array([
        [1, 1, 2],
    ], dtype=np.int32)

    results_empty = identify_reaction(cg_bonds_same, cg_bonds_same, cg_mapping, signatures)
    assert len(results_empty) == 0, f"期望空列表，实际 {len(results_empty)}"
    print("  无新键 -> 空列表")
    print("\n[测试 3] 通过")

    # ------------------------------------------------------------------
    # 测试 4: 使用 atom_bonds_to_cg_bonds 辅助函数验证
    # ------------------------------------------------------------------
    print("\n[测试 4] 使用 atom_bonds_to_cg_bonds 辅助函数...")

    # 构建一个包含多原子 per bead 的 CG 映射
    # bead 1 (interior, type 1): 原子 1
    # bead 2 (end, type 3): 原子 2, 3
    # bead 3 (monomer, type 5): 原子 4
    cg_mapping_multi = np.array([
        [1, 1, 1, 1, 10.0],   # 原子 1 -> bead 1 (interior)
        [2, 1, 3, 2, 10.0],   # 原子 2 -> bead 2 (end)
        [2, 1, 3, 3, 10.0],   # 原子 3 -> bead 2 (end, 同 bead)
        [3, 2, 5, 4, 10.0],   # 原子 4 -> bead 3 (monomer)
    ], dtype=np.float32)

    # 反应前: 原子 1-2 跨 bead1-bead2
    aa_bonds_before = np.array([
        [1, 1, 2],  # bead 1 - bead 2
    ], dtype=np.int32)

    # 反应后: 新增原子 3-4 跨 bead2-bead3 (新 CG 键)
    aa_bonds_after = np.array([
        [1, 1, 2],  # bead 1 - bead 2
        [1, 3, 4],  # bead 2 - bead 3 (新键: 原子 3 在 bead2, 原子 4 在 bead3)
    ], dtype=np.int32)

    cg_bonds_before_from_aa = atom_bonds_to_cg_bonds(aa_bonds_before, cg_mapping_multi)
    cg_bonds_after_from_aa = atom_bonds_to_cg_bonds(aa_bonds_after, cg_mapping_multi)

    print(f"  从原子键推导 CG 键 (反应前): {cg_bonds_before_from_aa.tolist()}")
    print(f"  从原子键推导 CG 键 (反应后): {cg_bonds_after_from_aa.tolist()}")

    results_from_aa = identify_reaction(
        cg_bonds_before_from_aa, cg_bonds_after_from_aa, cg_mapping_multi, signatures
    )

    assert len(results_from_aa) == 1, f"期望 1 个结果，实际 {len(results_from_aa)}"
    assert results_from_aa[0]['name'] == 'rxn1_EEE'
    print(f"  识别结果: {results_from_aa[0]['name']}")
    print("\n[测试 4] 通过")

    # ------------------------------------------------------------------
    # 测试 5: rxn8_PPP 签名 (2, 4, 6)
    # ------------------------------------------------------------------
    print("\n[测试 5] rxn8_PPP 场景...")

    cg_mapping_ppp = np.array([
        [10, 1, 2, 1, 10.0],   # interior bead, type 2
        [20, 1, 4, 2, 10.0],   # end bead, type 4
        [30, 2, 6, 3, 10.0],   # monomer bead, type 6
    ], dtype=np.float32)

    cg_bonds_before_ppp = np.array([
        [1, 10, 20],
    ], dtype=np.int32)

    cg_bonds_after_ppp = np.array([
        [1, 10, 20],
        [1, 20, 30],
    ], dtype=np.int32)

    results_ppp = identify_reaction(cg_bonds_before_ppp, cg_bonds_after_ppp,
                                    cg_mapping_ppp, signatures)

    assert len(results_ppp) == 1, f"期望 1 个结果，实际 {len(results_ppp)}"
    assert results_ppp[0]['signature'] == (2, 4, 6)
    assert results_ppp[0]['name'] == 'rxn8_PPP'
    assert results_ppp[0]['type_map'] == {4: 2, 6: 4}
    print(f"  识别结果: {results_ppp[0]['name']}, 签名: {results_ppp[0]['signature']}")
    print("\n[测试 5] 通过")

    print("\n" + "=" * 60)
    print("所有自测通过")
    print("=" * 60)
