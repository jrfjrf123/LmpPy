"""
反应位点定位器模块

功能:
- 从键变化定位反应类型和位点
- k-hop邻域提取 (BFS)
- 局部键连矩阵构建
- 模板匹配 (子图同构)

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
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

    功能:
    - 从键变化定位反应位点
    - 模板匹配识别反应类型
    """

    def __init__(self, reaction_templates: Dict[str, ReactionTemplate]):
        """
        初始化定位器

        Args:
            reaction_templates: 反应模板字典 {name: ReactionTemplate}
        """
        self.reaction_templates = reaction_templates

    def locate(self, bonds_before: np.ndarray, bonds_after: np.ndarray,
               atom_types: np.ndarray, n_atoms: int) -> List[ReactionMatch]:
        """
        定位反应位点

        Args:
            bonds_before: 反应前的键 (n_bonds, 3)
            bonds_after: 反应后的键 (n_bonds, 3)
            atom_types: 原子类型数组 (n_atoms,) 或 (n_atoms+1,)
            n_atoms: 原子总数

        Returns:
            List[ReactionMatch]: 匹配结果列表
        """
        # 检测键变化
        from .bond_detector import BondDetector
        detector = BondDetector(n_atoms)
        bond_changes = detector.detect(bonds_before, bonds_after)

        if not bond_changes.has_changes:
            return []

        # 获取参与反应的原子
        changed_atoms = get_changed_atoms(bond_changes)

        # 构建键连图
        graph_before = self._build_graph(bonds_before, n_atoms)
        graph_after = self._build_graph(bonds_after, n_atoms)

        # 对每个反应模板进行匹配
        matches = []
        for name, template in self.reaction_templates.items():
            match = self._match_template(
                template,
                changed_atoms,
                graph_before,
                graph_after,
                atom_types,
                bond_changes
            )
            if match:
                matches.append(match)

        return matches

    def _build_graph(self, bonds: np.ndarray, n_atoms: int) -> Dict[int, List[int]]:
        """
        构建邻接图

        Args:
            bonds: 键数组
            n_atoms: 原子总数

        Returns:
            邻接表 {atom_id: [neighbor_ids]}
        """
        graph = defaultdict(list)
        for bond in bonds:
            a1, a2 = int(bond[1]), int(bond[2])
            graph[a1].append(a2)
            graph[a2].append(a1)

        # 确保所有原子在图中
        for i in range(1, n_atoms + 1):
            if i not in graph:
                graph[i] = []

        return dict(graph)

    def _match_template(self, template: ReactionTemplate,
                        changed_atoms: Set[int],
                        graph_before: Dict[int, List[int]],
                        graph_after: Dict[int, List[int]],
                        atom_types: np.ndarray,
                        bond_changes: BondChanges) -> Optional[ReactionMatch]:
        """
        匹配单个模板

        Args:
            template: 反应模板
            changed_atoms: 参与反应的原子
            graph_before: 反应前的键连图
            graph_after: 反应后的键连图
            atom_types: 原子类型
            bond_changes: 键变化

        Returns:
            ReactionMatch 或 None
        """
        # 检查键变化是否匹配模板
        template_created = template.created_bonds
        template_deleted = template.deleted_bonds

        # 如果模板没有键变化，跳过
        if not template_created and not template_deleted:
            return None

        # 检查创建的键是否匹配
        if template_created:
            # 模板中有创建的键，检查体系中是否有对应的新键
            if not bond_changes.created_codes:
                return None

        # 检查删除的键是否匹配
        if template_deleted:
            if not bond_changes.deleted_codes:
                return None

        # 提取反应邻域 (k-hop)
        neighborhood = self._extract_neighborhood(changed_atoms, graph_after, k_hop=2)

        if len(neighborhood) < template.pre_template.n_atoms:
            return None

        # 尝试子图匹配
        match_result = self._subgraph_match(
            template, neighborhood, graph_before, graph_after, atom_types
        )

        return match_result

    def _extract_neighborhood(self, start_atoms: Set[int],
                              graph: Dict[int, List[int]],
                              k_hop: int = 2) -> Set[int]:
        """
        提取k-hop邻域

        Args:
            start_atoms: 起始原子集合
            graph: 键连图
            k_hop: hop数

        Returns:
            邻域原子集合
        """
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

    def _subgraph_match(self, template: ReactionTemplate,
                        neighborhood: Set[int],
                        graph_before: Dict[int, List[int]],
                        graph_after: Dict[int, List[int]],
                        atom_types: np.ndarray) -> Optional[ReactionMatch]:
        """
        子图匹配

        Args:
            template: 反应模板
            neighborhood: 邻域原子集合
            graph_before: 反应前的键连图
            graph_after: 反应后的键连图
            atom_types: 原子类型

        Returns:
            ReactionMatch 或 None
        """
        template_atoms = list(range(1, template.pre_template.n_atoms + 1))
        template_graph = self._build_template_graph(template.pre_template)

        # 获取模板原子类型
        template_types = template.pre_template.atom_types

        # 在邻域中寻找匹配
        neighborhood_list = list(neighborhood)

        # 简化匹配: 基于edge_atoms和initiator_atoms
        edge_atoms = template.reaction_map.edge_ids
        initiator_atoms = template.reaction_map.initiator_ids

        if not edge_atoms and not initiator_atoms:
            return None

        # 遍历邻域中的候选原子对
        for candidate in neighborhood_list:
            # 检查原子类型是否匹配edge atom
            for edge_atom in edge_atoms:
                template_type = template_types[edge_atom]
                if atom_types[candidate] == template_type:
                    # 尝试从该候选原子开始匹配
                    match = self._try_match_from_edge(
                        template, candidate, edge_atom,
                        neighborhood, graph_after, atom_types
                    )
                    if match:
                        return match

        return None

    def _try_match_from_edge(self, template: ReactionTemplate,
                             candidate: int, edge_atom: int,
                             neighborhood: Set[int],
                             graph: Dict[int, List[int]],
                             atom_types: np.ndarray) -> Optional[ReactionMatch]:
        """
        从edge atom尝试匹配

        Args:
            template: 反应模板
            candidate: 候选原子ID
            edge_atom: 模板中的edge atom ID
            neighborhood: 邻域
            graph: 键连图
            atom_types: 原子类型

        Returns:
            ReactionMatch 或 None
        """
        template_types = template.pre_template.atom_types
        template_graph = self._build_template_graph(template.pre_template)

        # BFS匹配
        mapping = {edge_atom: candidate}
        queue = deque([edge_atom])
        visited = {edge_atom}

        while queue:
            template_atom = queue.popleft()
            system_atom = mapping[template_atom]

            # 检查邻居
            for template_neighbor in template_graph.get(template_atom, []):
                if template_neighbor in visited:
                    continue

                # 获取模板邻居的类型
                template_neighbor_type = template_types[template_neighbor]

                # 在体系图中查找匹配的邻居
                found = False
                for system_neighbor in graph.get(system_atom, []):
                    if system_neighbor in mapping.values():
                        continue
                    if system_neighbor not in neighborhood:
                        continue
                    if atom_types[system_neighbor] == template_neighbor_type:
                        mapping[template_neighbor] = system_neighbor
                        visited.add(template_neighbor)
                        queue.append(template_neighbor)
                        found = True
                        break

                if not found:
                    # 匹配失败
                    return None

        # 检查是否所有模板原子都匹配
        if len(mapping) != template.pre_template.n_atoms:
            return None

        return ReactionMatch(
            reaction_name=template.name,
            template_to_system=mapping,
            confidence=1.0
        )

    def _build_template_graph(self, template_data: TemplateData) -> Dict[int, List[int]]:
        """
        从模板数据构建键连图

        Args:
            template_data: 模板数据

        Returns:
            邻接表
        """
        graph = defaultdict(list)
        for bond in template_data.bonds:
            a1, a2 = int(bond[2]), int(bond[3])
            graph[a1].append(a2)
            graph[a2].append(a1)
        return dict(graph)


def locate_reactions(bonds_before: np.ndarray, bonds_after: np.ndarray,
                     atom_types: np.ndarray, n_atoms: int,
                     reaction_templates: Dict[str, ReactionTemplate]) -> List[ReactionMatch]:
    """
    便捷函数：定位反应

    Args:
        bonds_before: 反应前的键
        bonds_after: 反应后的键
        atom_types: 原子类型
        n_atoms: 原子总数
        reaction_templates: 反应模板字典

    Returns:
        匹配结果列表
    """
    locator = ReactionLocator(reaction_templates)
    return locator.locate(bonds_before, bonds_after, atom_types, n_atoms)


if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("测试反应位点定位器")
    print("=" * 60)

    # 创建模拟数据
    n_atoms = 100

    # 模拟键数据
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
        [2, 5, 6],  # 新键
    ], dtype=np.int32)

    # 原子类型
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

    print("\n测试邻域提取:")
    graph_after = defaultdict(list)
    for bond in bonds_after:
        a1, a2 = int(bond[1]), int(bond[2])
        graph_after[a1].append(a2)
        graph_after[a2].append(a1)

    locator = ReactionLocator({})
    neighborhood = locator._extract_neighborhood(changed_atoms, graph_after, k_hop=2)
    print(f"  邻域大小: {len(neighborhood)}")
    print(f"  邻域原子: {neighborhood}")

    print("\n✅ 测试完成!")