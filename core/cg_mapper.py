"""
CG映射更新器模块

功能:
- 反应后更新CG映射
- 基于模板CG映射自动推导CG变化
- 按 bead 整体更新，保证一致性

关键设计:
- 使用 post_cg_mapping 定义反应后的 bead 结构
- 每个 bead 的所有原子一起更新
- 边缘原子不参与更新

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
from copy import deepcopy
import numpy as np

# 支持两种导入方式
try:
    from .reaction_locator import ReactionMatch
    from .template_parser import ReactionTemplate, TemplateData, TemplateCGMapping
except ImportError:
    from reaction_locator import ReactionMatch
    from template_parser import ReactionTemplate, TemplateData, TemplateCGMapping


@dataclass
class CGMapping:
    """
    CG映射数据结构

    存储每个原子的bead_id和bead_type
    """
    data: np.ndarray  # (n_atoms, 2) [bead_id, bead_type]
    n_atoms: int

    @classmethod
    def from_cg_compare_list(cls, cg_compare_list: np.ndarray, n_atoms: int) -> 'CGMapping':
        """
        从cg_compare_list创建CGMapping

        Args:
            cg_compare_list: (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]
            n_atoms: 原子总数

        Returns:
            CGMapping实例
        """
        mapping = np.zeros((n_atoms, 2), dtype=np.int32)

        for row in cg_compare_list:
            bead_id = int(row[0])
            bead_type = int(row[2])
            atom_id = int(row[3])

            if 1 <= atom_id <= n_atoms:
                mapping[atom_id - 1] = [bead_id, bead_type]

        return cls(data=mapping, n_atoms=n_atoms)

    def get_bead_id(self, atom_id: int) -> int:
        """获取原子的bead_id"""
        return self.data[atom_id - 1, 0]

    def get_bead_type(self, atom_id: int) -> int:
        """获取原子的bead_type"""
        return self.data[atom_id - 1, 1]

    def set_bead(self, atom_id: int, bead_id: int, bead_type: int):
        """设置原子的bead_id和bead_type"""
        self.data[atom_id - 1] = [bead_id, bead_type]

    def get_atoms_in_bead(self, bead_id: int) -> np.ndarray:
        """获取属于某个bead的所有原子"""
        return np.where(self.data[:, 0] == bead_id)[0] + 1  # 1-indexed

    def get_max_bead_id(self) -> int:
        """获取最大的bead_id"""
        return int(self.data[:, 0].max())

    def to_cg_compare_list(self, masses: Optional[np.ndarray] = None) -> np.ndarray:
        """
        转换回cg_compare_list格式

        Args:
            masses: 可选的质量数组 (n_atoms,)

        Returns:
            cg_compare_list (n_atoms, 5)
        """
        n_atoms = self.n_atoms
        result = np.zeros((n_atoms, 5), dtype=np.float64)

        for i in range(n_atoms):
            atom_id = i + 1
            bead_id = self.data[i, 0]
            bead_type = self.data[i, 1]
            mol_id = 1  # 需要从外部获取
            mass = masses[i] if masses is not None else 1.0

            result[i] = [bead_id, mol_id, bead_type, atom_id, mass]

        return result


class CGMapper:
    """
    CG映射更新器

    功能:
    - 基于反应模板匹配结果更新CG映射
    - 按 bead 整体更新，保证一致性

    设计原则:
    - 使用 post_cg_mapping 定义反应后的 bead 结构
    - 每个 bead 的所有原子一起分配新的全局 bead_id
    - 边缘原子不参与更新
    - bead_type 从 post_cg_mapping 获取，不推断
    """

    def __init__(self):
        """初始化映射器"""
        pass

    def update(self, cg_mapping: CGMapping,
               reaction_match: ReactionMatch,
               template: ReactionTemplate) -> CGMapping:
        """
        更新CG映射

        算法:
        1. 检查模板是否有 post_cg_mapping，没有则跳过
        2. 获取边缘原子集合
        3. 按 local_bead_id 分组模板原子
        4. 对每个 bead:
           a. 找到对应的体系原子列表 (排除边缘原子)
           b. 分配一个新的全局 bead_id
           c. 使用 post_cg_mapping 中的 bead_type
           d. 更新所有这些原子的 bead_id 和 bead_type

        Args:
            cg_mapping: 当前CG映射
            reaction_match: 反应匹配结果
            template: 反应模板

        Returns:
            更新后的CG映射
        """
        # 检查模板是否有 CG 映射定义
        if template.post_cg_mapping is None:
            # 没有定义 CG 映射，返回原映射
            return CGMapping(
                data=cg_mapping.data.copy(),
                n_atoms=cg_mapping.n_atoms
            )

        # 深拷贝避免修改原数据
        new_mapping = CGMapping(
            data=cg_mapping.data.copy(),
            n_atoms=cg_mapping.n_atoms
        )

        # 获取模板到体系的原子映射
        template_to_system = reaction_match.template_to_system

        # 获取边缘原子 - 这些原子是模板边界，不应更新CG映射
        edge_atoms = set(template.reaction_map.edge_ids)

        # 获取 post_cg_mapping
        post_cg_mapping = template.post_cg_mapping

        # 获取当前最大的 bead_id，用于分配新 ID
        max_bead_id = new_mapping.get_max_bead_id()

        # 按 local_bead_id 分组处理
        # bead_groups: {local_bead_id: (atom_ids, bead_type)}
        bead_groups = post_cg_mapping.bead_groups

        # 向量化更新
        for local_bead_id, (template_atom_ids, bead_type) in bead_groups.items():
            # 找到对应的体系原子，排除边缘原子
            system_atom_ids = []
            for template_atom_id in template_atom_ids:
                if template_atom_id in edge_atoms:
                    continue
                if template_atom_id in template_to_system:
                    system_atom_ids.append(template_to_system[template_atom_id])

            if not system_atom_ids:
                continue

            # 分配新的全局 bead_id
            max_bead_id += 1
            new_bead_id = max_bead_id

            # 更新该 bead 的所有原子
            for system_atom_id in system_atom_ids:
                new_mapping.set_bead(system_atom_id, new_bead_id, bead_type)

        return new_mapping

    def batch_update(self, cg_mapping: CGMapping,
                     reaction_matches: List[ReactionMatch],
                     templates: Dict[str, ReactionTemplate]) -> CGMapping:
        """
        批量更新CG映射

        Args:
            cg_mapping: 当前CG映射
            reaction_matches: 反应匹配结果列表
            templates: 反应模板字典

        Returns:
            更新后的CG映射
        """
        current_mapping = cg_mapping

        for match in reaction_matches:
            template = templates.get(match.reaction_name)
            if template:
                current_mapping = self.update(current_mapping, match, template)

        return current_mapping


def update_cg_mapping(cg_compare_list: np.ndarray,
                      reaction_matches: List[ReactionMatch],
                      templates: Dict[str, ReactionTemplate],
                      n_atoms: int) -> np.ndarray:
    """
    便捷函数：更新CG映射

    Args:
        cg_compare_list: 当前CG映射
        reaction_matches: 反应匹配结果
        templates: 反应模板字典
        n_atoms: 原子总数

    Returns:
        更新后的cg_compare_list
    """
    cg_mapping = CGMapping.from_cg_compare_list(cg_compare_list, n_atoms)
    mapper = CGMapper()
    updated_mapping = mapper.batch_update(cg_mapping, reaction_matches, templates)
    return updated_mapping.data


if __name__ == "__main__":
    print("=" * 60)
    print("测试CG映射更新器")
    print("=" * 60)

    # 创建测试数据
    n_atoms = 50

    # 创建初始CG映射
    cg_compare_list = np.zeros((n_atoms, 5), dtype=np.float64)
    for i in range(n_atoms):
        atom_id = i + 1
        bead_id = (i // 10) + 1  # 每10个原子一个bead
        bead_type = 1
        mol_id = 1
        mass = 12.0

        cg_compare_list[i] = [bead_id, mol_id, bead_type, atom_id, mass]

    print("\n初始CG映射:")
    print(f"  原子数: {n_atoms}")
    print(f"  Bead数: {int(cg_compare_list[:, 0].max())}")

    # 创建CGMapping
    cg_mapping = CGMapping.from_cg_compare_list(cg_compare_list, n_atoms)
    print(f"  CGMapping创建成功")

    # 测试获取方法
    print(f"\n原子1的bead_id: {cg_mapping.get_bead_id(1)}")
    print(f"原子1的bead_type: {cg_mapping.get_bead_type(1)}")
    print(f"最大bead_id: {cg_mapping.get_max_bead_id()}")

    # 测试设置
    cg_mapping.set_bead(1, 100, 5)
    print(f"\n更新后原子1的bead_id: {cg_mapping.get_bead_id(1)}")
    print(f"更新后原子1的bead_type: {cg_mapping.get_bead_type(1)}")

    print("\n✅ 测试完成!")