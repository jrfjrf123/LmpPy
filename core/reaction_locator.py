"""
反应位点定位器模块

功能:
- 从键变化定位反应类型和位点
- 使用 created_bonds 聚类同一反应的原子
- k-hop 邻域提取 (BFS)
- 模板匹配 (子图同构)
- pre-before + post-after 双重匹配

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
import time
import numpy as np

# 支持两种导入方式
try:
    from .bond_detector import BondChanges, get_changed_atoms
    from .template_parser import ReactionTemplate, TemplateData
except ImportError:
    from bond_detector import BondChanges, get_changed_atoms
    from template_parser import ReactionTemplate, TemplateData


@dataclass
class ReactionMatch:
    """反应匹配结果"""
    reaction_name: str                                    # 反应名称
    template_to_system: Dict[int, int]                    # 模板原子ID → 体系原子ID
    confidence: float = 1.0                               # 匹配置信度
    matched_bonds: List[Tuple[int, int]] = field(default_factory=list)  # 匹配的键

    def get_system_atoms(self) -> Set[int]:
        """获取所有参与反应的体系原子ID"""
        return set(self.template_to_system.values())


class ReactionLocator:
    """
    反应位点定位器

    匹配策略: pre-before + post-after 双重验证
    - 用 created_bonds 聚类同一反应的原子
    - pre template × before 体系: 验证反应前的原子拓扑
    - post template × after 体系: 验证反应后的原子拓扑
    - 两者都匹配才确认反应
    """

    def __init__(self, reaction_templates: Dict[str, ReactionTemplate]):
        self.reaction_templates = reaction_templates

    def locate(self, bonds_before: np.ndarray, bonds_after: np.ndarray,
               types_before: np.ndarray, types_after: np.ndarray,
               ids_before: np.ndarray, ids_after: np.ndarray,
               n_atoms: int) -> List[ReactionMatch]:
        """
        定位反应位点

        Args:
            bonds_before: 反应前的键 (n_bonds, 3)
            bonds_after: 反应后的键 (n_bonds, 3)
            types_before: 反应前的原子类型 (0-based, gather_atoms 返回的原始顺序)
            types_after: 反应后的原子类型
            ids_before: 反应前的原子ID (与 types_before 同序)
            ids_after: 反应后的原子ID
            n_atoms: 原子总数

        Returns:
            List[ReactionMatch]: 匹配结果列表
        """
        # 构建 atom_id → type 映射 (因为 gather_atoms 不保证按 ID 排序)
        _t0 = time.time()
        # 用 numpy 数组替代 dict: O(N) 批量赋值 vs 3 万次 dict 插入
        max_id = max(int(ids_before.max()), int(ids_after.max()))
        type_before_arr = np.zeros(max_id + 1, dtype=np.int32)
        type_before_arr[ids_before.astype(np.int32)] = types_before
        type_after_arr = np.zeros(max_id + 1, dtype=np.int32)
        type_after_arr[ids_after.astype(np.int32)] = types_after
        _t1 = time.time()
        _t1 = time.time()
        from .bond_detector import BondDetector
        detector = BondDetector(n_atoms)
        bond_changes = detector.detect(bonds_before, bonds_after)
        _t2 = time.time()

        if not bond_changes.has_changes:
            return []

        # 构建键连图
        graph_before = self._build_graph(bonds_before, n_atoms)
        graph_after = self._build_graph(bonds_after, n_atoms)
        _t3 = time.time()

        # 用 created_bonds 聚类同一反应的原子
        components = self._cluster_by_created_bonds(bond_changes, graph_before)
        _t4 = time.time()

        # 对每个反应组件 + 每个反应模板进行双重匹配
        matches = []
        _t_pre_total = 0.0
        _t_post_total = 0.0
        for component in components:
            for name, template in self.reaction_templates.items():
                _tp = time.time()
                pre_match = self._match_pre_template(
                    template, graph_before, type_before_arr, bond_changes, n_atoms, component
                )
                _t_pre_total += time.time() - _tp
                if not pre_match:
                    continue

                _tp2 = time.time()
                post_match = self._match_post_template(
                    template, graph_after, type_after_arr, pre_match, n_atoms
                )
                _t_post_total += time.time() - _tp2
                if not post_match:
                    continue

                post_match.reaction_name = name
                matches.append(post_match)
        _t5 = time.time()

        # print(f"  [locate计时] type_map: {(_t1-_t0)*1000:.1f}ms, "
        #       f"BondDetector: {(_t2-_t1)*1000:.1f}ms, "
        #       f"build_graph: {(_t3-_t2)*1000:.1f}ms, "
        #       f"clustering: {(_t4-_t3)*1000:.1f}ms, "
        #       f"pre_match总计: {_t_pre_total*1000:.1f}ms, "
        #       f"post_match总计: {_t_post_total*1000:.1f}ms, "
        #       f"locate总: {(_t5-_t0)*1000:.1f}ms")

        return matches

    def _cluster_by_created_bonds(self, bond_changes: BondChanges,
                                   graph_before: Dict[int, List[int]]) -> List[Set[int]]:
        """
        使用 created_bonds 聚类同一反应的原子

        策略:
        1. 从 created_bonds 构建图，BFS 找出连通分量
           (同一个 created bond 的两个原子在同一反应中)
        2. 对于不在 created_bonds 中的 changed atoms (如 deleted bonds 的原子),
           如果与某个 component 在 before 图中相邻，合并进去
        """
        # Step 1: 从 created_bonds 构建图
        created_graph = defaultdict(set)
        atoms_in_created = set()

        if len(bond_changes.created_bonds) > 0:
            for bond in bond_changes.created_bonds:
                a1, a2 = int(bond[1]), int(bond[2])
                created_graph[a1].add(a2)
                created_graph[a2].add(a1)
                atoms_in_created.add(a1)
                atoms_in_created.add(a2)

        # BFS 找出 created_bonds 的连通分量
        visited = set()
        components = []

        for start_atom in atoms_in_created:
            if start_atom in visited:
                continue
            component = set()
            queue = deque([start_atom])
            while queue:
                atom = queue.popleft()
                if atom in visited:
                    continue
                visited.add(atom)
                component.add(atom)
                for nb in created_graph.get(atom, []):
                    if nb not in visited:
                        queue.append(nb)
            if component:
                components.append(component)

        # Step 2: 将不在 created_bonds 中的 changed atoms 合并到相邻 component
        all_changed = get_changed_atoms(bond_changes)
        orphaned = all_changed - atoms_in_created

        for orphan in orphaned:
            # 查找 before 图中的邻居属于哪个 component
            merged = False
            for nb in graph_before.get(orphan, []):
                for comp in components:
                    if nb in comp:
                        comp.add(orphan)
                        merged = True
                        break
                if merged:
                    break
            if not merged:
                # 没有找到相邻 component，自成一组
                components.append({orphan})

        return components

    def _build_graph(self, bonds: np.ndarray, n_atoms: int) -> Dict[int, List[int]]:
        """构建邻接表（只包含有键的原子，下游用 graph.get(atom, []) 处理缺失键）"""
        graph = defaultdict(list)
        for bond in bonds:
            a1, a2 = int(bond[1]), int(bond[2])
            graph[a1].append(a2)
            graph[a2].append(a1)
        return dict(graph)

    def _match_pre_template(self, template: ReactionTemplate,
                            graph_before: Dict[int, List[int]],
                            type_before_arr: np.ndarray,
                            bond_changes: BondChanges,
                            n_atoms: int,
                            component: Set[int]) -> Optional[ReactionMatch]:
        """
        用 pre template 匹配反应前的体系
        针对单个反应组件进行匹配
        """
        # 粗筛
        if not template.created_bonds and not template.deleted_bonds:
            return None
        if template.created_bonds and not bond_changes.created_codes:
            return None
        if template.deleted_bonds and not bond_changes.deleted_codes:
            return None

        # 从该组件提取 k-hop 邻域
        max_k = max(template.pre_template.n_atoms, 2)
        neighborhood = self._extract_neighborhood(component, graph_before, k_hop=max_k)

        if len(neighborhood) < template.pre_template.n_atoms:
            return None

        return self._bfs_match(
            template.pre_template,
            template.pre_template.atom_types,
            template.reaction_map.initiator_ids,
            neighborhood, graph_before, type_before_arr,
            handle_isolated=True
        )

    def _match_post_template(self, template: ReactionTemplate,
                             graph_after: Dict[int, List[int]],
                             type_after_arr: np.ndarray,
                             pre_match: ReactionMatch,
                             n_atoms: int) -> Optional[ReactionMatch]:
        """
        用 post template 匹配反应后的体系
        使用 pre 匹配结果中的体系原子作为起点
        """
        # 从 pre 匹配结果获取体系原子，作为后匹配的起点
        system_atoms = pre_match.get_system_atoms()
        max_k = max(template.post_template.n_atoms, 2)
        neighborhood = self._extract_neighborhood(system_atoms, graph_after, k_hop=max_k)

        return self._bfs_match(
            template.post_template,
            template.post_template.atom_types,
            template.reaction_map.initiator_ids,
            neighborhood, graph_after, type_after_arr,
            handle_isolated=False
        )

    def _bfs_match(self, template_data: TemplateData,
                   template_types: np.ndarray,
                   start_ids: List[int],
                   neighborhood: Set[int],
                   graph: Dict[int, List[int]],
                   atom_type_arr: np.ndarray,
                   handle_isolated: bool = False) -> Optional[ReactionMatch]:
        """
        通用 BFS 匹配

        Args:
            template_data: 模板数据（pre 或 post）
            template_types: 模板原子类型
            start_ids: initiator atom ID 列表（匹配起点，替代 edge atoms）
            neighborhood: 候选原子集合
            graph: 体系键连图
            atom_type_arr: numpy 数组，atom_type_arr[atom_id] 直接索引（避免 gather_atoms 顺序依赖）
            handle_isolated: 是否处理孤立原子（pre 需要，post 不需要）
        """
        if not start_ids:
            return None

        template_graph = self._build_template_graph(template_data)
        n_template = template_data.n_atoms

        # 获取 bonded atoms
        bonded_atoms = set()
        for bond in template_data.bonds:
            bonded_atoms.add(int(bond[2]))
            bonded_atoms.add(int(bond[3]))

        # 只遍历与 initiator type 匹配的候选原子
        initiator_types = {template_types[s] for s in start_ids}
        neighborhood_list = [a for a in neighborhood if atom_type_arr[a] in initiator_types]

        for candidate in neighborhood_list:
            for start_atom in start_ids:
                tmpl_type = template_types[start_atom]
                if atom_type_arr[candidate] == tmpl_type:
                    mapping = self._bfs_from_start(
                        start_atom, candidate, template_graph, template_types,
                        neighborhood, graph, atom_type_arr
                    )
                    if mapping is None:
                        continue

                    # 检查未匹配的原子
                    unmatched = set(range(1, n_template + 1)) - set(mapping.keys())
                    matched_sys = set(mapping.values())

                    if unmatched:
                        # === 统一多起点 BFS：处理模板图可能不连通的情况 ===
                        # pre-template: 反应前两个分子未连接，需从两个 initiator 分别匹配
                        # post-template: 通常连通（单起点足够），但统一处理以保持健壮性
                        non_isolated = unmatched & bonded_atoms
                        if non_isolated:
                            # 尝试用其他 start_ids 匹配剩余的不连通组件
                            remaining_starts = [s for s in start_ids if s not in mapping]
                            for rem_start in remaining_starts:
                                rem_type = template_types[rem_start]
                                # 在邻域中找一个未使用且类型匹配的候选原子
                                found_candidate = None
                                for sa in neighborhood:
                                    if sa in matched_sys:
                                        continue
                                    if atom_type_arr[sa] == rem_type:
                                        found_candidate = sa
                                        break

                                if found_candidate is not None:
                                    sub_mapping = self._bfs_from_start(
                                        rem_start, found_candidate, template_graph,
                                        template_types, neighborhood, graph, atom_type_arr
                                    )
                                    if sub_mapping is not None:
                                        # 合并映射（不同组件应无重叠）
                                        for k, v in sub_mapping.items():
                                            if k not in mapping:
                                                mapping[k] = v
                                                matched_sys.add(v)

                            # 重新检查是否所有非孤立原子都已匹配
                            unmatched = set(range(1, n_template + 1)) - set(mapping.keys())
                            non_isolated = unmatched & bonded_atoms
                            if non_isolated:
                                continue  # 仍有未匹配的非孤立原子 → pre/post 都拒绝

                        # === 处理剩余未匹配原子 ===
                        if unmatched:
                            if handle_isolated:
                                # pre-match: 剩余原子必须是孤立原子（无键连接）
                                all_isolated_ok = True
                                for ta in unmatched:
                                    tt = template_types[ta]
                                    found = False
                                    for sa in neighborhood:
                                        if sa in matched_sys:
                                            continue
                                        if atom_type_arr[sa] == tt:
                                            mapping[ta] = sa
                                            matched_sys.add(sa)
                                            found = True
                                            break
                                    if not found:
                                        all_isolated_ok = False
                                        break
                                if not all_isolated_ok:
                                    continue
                            else:
                                # post-match: 不允许有任何未匹配原子
                                continue

                    return ReactionMatch(
                        reaction_name="",
                        template_to_system=mapping,
                        confidence=1.0
                    )

        return None

    def _bfs_from_start(self, start_atom: int, candidate: int,
                       template_graph: Dict[int, List[int]],
                       template_types: np.ndarray,
                       neighborhood: Set[int],
                       graph: Dict[int, List[int]],
                       atom_type_arr: np.ndarray) -> Optional[Dict[int, int]]:
        """
        从 initiator atom 开始 BFS 匹配
        返回 {template_atom_id: system_atom_id} 或 None
        """
        mapping = {start_atom: candidate}
        queue = deque([start_atom])
        visited = {start_atom}

        while queue:
            tmpl_atom = queue.popleft()
            sys_atom = mapping[tmpl_atom]

            for tmpl_nb in template_graph.get(tmpl_atom, []):
                if tmpl_nb in visited:
                    continue

                tmpl_nb_type = template_types[tmpl_nb]
                found = False
                for sys_nb in graph.get(sys_atom, []):
                    if sys_nb in mapping.values():
                        continue
                    if sys_nb not in neighborhood:
                        continue
                    if atom_type_arr[sys_nb] == tmpl_nb_type:
                        mapping[tmpl_nb] = sys_nb
                        visited.add(tmpl_nb)
                        queue.append(tmpl_nb)
                        found = True
                        break

                if not found:
                    return None

        return mapping

    def _extract_neighborhood(self, start_atoms: Set[int],
                              graph: Dict[int, List[int]],
                              k_hop: int = 2) -> Set[int]:
        """提取k-hop邻域"""
        visited = set()
        current = set(start_atoms)

        for _ in range(k_hop):
            next_level = set()
            for atom in current:
                if atom in visited:
                    continue
                visited.add(atom)
                for neighbor in graph.get(atom, []):
                    if neighbor not in visited:
                        next_level.add(neighbor)
            current = next_level

        return visited

    def _build_template_graph(self, template_data: TemplateData) -> Dict[int, List[int]]:
        """从模板数据构建键连图"""
        graph = defaultdict(list)
        for bond in template_data.bonds:
            a1, a2 = int(bond[2]), int(bond[3])
            graph[a1].append(a2)
            graph[a2].append(a1)
        return dict(graph)


def locate_reactions(bonds_before: np.ndarray, bonds_after: np.ndarray,
                     types_before: np.ndarray, types_after: np.ndarray,
                     ids_before: np.ndarray, ids_after: np.ndarray,
                     n_atoms: int,
                     reaction_templates: Dict[str, ReactionTemplate]) -> List[ReactionMatch]:
    """便捷函数：定位反应"""
    locator = ReactionLocator(reaction_templates)
    return locator.locate(bonds_before, bonds_after, types_before, types_after,
                          ids_before, ids_after, n_atoms)


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("测试反应位点定位器")
    print("=" * 60)

    n_atoms = 100

    bonds_before = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [1, 3, 4],
        [1, 10, 11],
        [1, 11, 12],
    ], dtype=np.int32)

    bonds_after = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [1, 3, 4],
        [1, 10, 11],
        [1, 11, 12],
        [2, 5, 6],
    ], dtype=np.int32)

    atom_types = np.zeros(n_atoms + 1, dtype=np.int32)
    atom_types[1:20] = 1
    atom_types[5] = 2
    atom_types[6] = 2

    print("\n测试键变化检测:")
    from bond_detector import BondDetector
    detector = BondDetector(n_atoms)
    changes = detector.detect(bonds_before, bonds_after)
    print(f"  创建的键: {changes.n_created}")
    print(f"  删除的键: {changes.n_deleted}")

    changed_atoms = get_changed_atoms(changes)
    print(f"  参与反应的原子: {changed_atoms}")

    print("\n✅ 测试完成!")