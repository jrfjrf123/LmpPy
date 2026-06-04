"""
模板解析器模块

功能:
- 解析LAMMPS反应模板文件 (pre/post .lammpstemplate)
- 解析反应映射文件 (.map)
- 在初始化阶段完成解析，模板为固定参数

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from pathlib import Path
import numpy as np

# 支持两种导入方式
try:
    from .config_loader import ConfigError
except ImportError:
    from config_loader import ConfigError


@dataclass
class TemplateData:
    """模板数据结构"""
    n_atoms: int
    n_bonds: int
    n_angles: int
    n_dihedrals: int

    # 原子类型 (1-indexed, shape: (n_atoms+1,))
    atom_types: np.ndarray

    # 坐标 (shape: (n_atoms+1, 3))
    coords: np.ndarray

    # 键连表 (shape: (n_bonds, 4)) [bond_id, bond_type, atom1, atom2]
    bonds: np.ndarray

    # 角度表 (shape: (n_angles, 5)) [angle_id, angle_type, atom1, atom2, atom3]
    angles: np.ndarray

    # 二面角表 (shape: (n_dihedrals, 6)) [dihedral_id, dihedral_type, atom1, atom2, atom3, atom4]
    dihedrals: np.ndarray


@dataclass
class ReactionMapData:
    """反应映射数据结构"""
    n_edge_ids: int
    n_equivalences: int
    n_constraints: int

    initiator_ids: List[int]
    edge_ids: List[int]

    # 等价映射 (pre_atom_id -> post_atom_id)
    equivalences: Dict[int, int]

    # 约束列表 [(type, atom1, atom2, min_dist, max_dist), ...]
    constraints: List[Tuple[str, int, int, float, float]]


@dataclass
class TemplateCGMapping:
    """
    模板的 CG 映射数据结构

    定义模板原子如何映射到粗粒珠子

    Attributes:
        bead_groups: {local_bead_id: (atom_ids, bead_type)}
            - local_bead_id: 模板内部的相对 bead 编号
            - atom_ids: 属于该 bead 的模板原子 ID 列表
            - bead_type: 该 bead 的类型
    """
    bead_groups: Dict[int, Tuple[List[int], int]]  # {local_bead_id: (atom_ids, bead_type)}
    n_beads: int

    def get_atom_bead_info(self, atom_id: int) -> Optional[Tuple[int, int]]:
        """
        获取原子所属的 bead 信息

        Args:
            atom_id: 模板原子 ID

        Returns:
            (local_bead_id, bead_type) 或 None
        """
        for bead_id, (atom_ids, bead_type) in self.bead_groups.items():
            if atom_id in atom_ids:
                return (bead_id, bead_type)
        return None

    def get_atoms_in_bead(self, local_bead_id: int) -> List[int]:
        """获取属于某个 bead 的所有原子"""
        if local_bead_id in self.bead_groups:
            return self.bead_groups[local_bead_id][0]
        return []


@dataclass
class ReactionTemplate:
    """完整反应模板"""
    name: str
    pre_template: TemplateData
    post_template: TemplateData
    reaction_map: ReactionMapData

    # 预计算的键编码集合 (用于快速匹配)
    pre_bond_codes: Set[int] = field(default_factory=set)
    post_bond_codes: Set[int] = field(default_factory=set)

    # 模板 CG 映射 (可选)
    pre_cg_mapping: Optional[TemplateCGMapping] = None
    post_cg_mapping: Optional[TemplateCGMapping] = None

    def __post_init__(self):
        """预计算键编码"""
        self.pre_bond_codes = self._encode_bonds(self.pre_template)
        self.post_bond_codes = self._encode_bonds(self.post_template)

    def _encode_bonds(self, template: TemplateData) -> Set[int]:
        """将键编码为整数集合"""
        codes = set()
        encoder = template.n_atoms + 1
        for bond in template.bonds:
            atom1, atom2 = int(bond[2]), int(bond[3])
            # 规范化: atom1 < atom2
            if atom1 > atom2:
                atom1, atom2 = atom2, atom1
            code = atom1 * encoder + atom2
            codes.add(code)
        return codes

    @property
    def created_bonds(self) -> Set[int]:
        """反应后新创建的键"""
        return self.post_bond_codes - self.pre_bond_codes

    @property
    def deleted_bonds(self) -> Set[int]:
        """反应后删除的键"""
        return self.pre_bond_codes - self.post_bond_codes

    @property
    def changed_atom_types(self) -> Dict[int, Tuple[int, int]]:
        """返回原子类型变化 {atom_id: (pre_type, post_type)}"""
        changed = {}
        for atom_id in range(1, self.pre_template.n_atoms + 1):
            pre_type = int(self.pre_template.atom_types[atom_id])
            post_type = int(self.post_template.atom_types[atom_id])
            if pre_type != post_type:
                changed[atom_id] = (pre_type, post_type)
        return changed


class TemplateParser:
    """
    LAMMPS模板解析器

    功能:
    - 解析.lammpstemplate文件
    - 解析.map文件
    """

    def parse_template(self, template_file: str) -> TemplateData:
        """
        解析模板文件

        Args:
            template_file: 模板文件路径

        Returns:
            TemplateData: 模板数据对象
        """
        path = Path(template_file)
        if not path.exists():
            raise ConfigError(f"模板文件未找到: {template_file}")

        with open(path, 'r') as f:
            lines = f.readlines()

        # 解析计数
        n_atoms = 0
        n_bonds = 0
        n_angles = 0
        n_dihedrals = 0

        # 找到各section的起始位置
        section_starts = {}
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped == 'Types':
                section_starts['types'] = i
            elif stripped == 'Coords':
                section_starts['coords'] = i
            elif stripped == 'Bonds':
                section_starts['bonds'] = i
            elif stripped == 'Angles':
                section_starts['angles'] = i
            elif stripped == 'Dihedrals':
                section_starts['dihedrals'] = i
            # 解析头部计数
            elif 'atoms' in stripped.lower():
                parts = stripped.split()
                if parts[0].isdigit():
                    n_atoms = int(parts[0])
            elif 'bonds' in stripped.lower():
                parts = stripped.split()
                if parts[0].isdigit():
                    n_bonds = int(parts[0])
            elif 'angles' in stripped.lower():
                parts = stripped.split()
                if parts[0].isdigit():
                    n_angles = int(parts[0])
            elif 'dihedrals' in stripped.lower():
                parts = stripped.split()
                if parts[0].isdigit():
                    n_dihedrals = int(parts[0])

        # 解析各section
        atom_types = None
        coords = None
        bonds = None
        angles = None
        dihedrals = None

        # 按照section顺序解析
        section_order = ['types', 'coords', 'bonds', 'angles', 'dihedrals']
        for idx, section in enumerate(section_order):
            if section not in section_starts:
                continue

            start = section_starts[section] + 1  # 跳过section标题行

            # 找到下一个section的起始位置作为结束
            end = len(lines)
            for next_section in section_order[idx + 1:]:
                if next_section in section_starts:
                    end = section_starts[next_section]
                    break

            # 提取数据行
            data_lines = []
            for i in range(start, end):
                line = lines[i].strip()
                if line and line.split()[0].isdigit():
                    data_lines.append(line)

            # 解析数据
            if section == 'types':
                atom_types = self._parse_types(data_lines, n_atoms)
            elif section == 'coords':
                coords = self._parse_coords(data_lines, n_atoms)
            elif section == 'bonds':
                bonds = self._parse_bonds(data_lines, n_bonds)
            elif section == 'angles':
                angles = self._parse_angles(data_lines, n_angles)
            elif section == 'dihedrals':
                dihedrals = self._parse_dihedrals(data_lines, n_dihedrals)

        # 验证必要数据
        if atom_types is None:
            raise ConfigError(f"模板文件缺少Types section: {template_file}")
        if coords is None:
            raise ConfigError(f"模板文件缺少Coords section: {template_file}")

        # 如果没有键，创建空数组
        if bonds is None:
            bonds = np.zeros((0, 4), dtype=np.int32)
        if angles is None:
            angles = np.zeros((0, 5), dtype=np.int32)
        if dihedrals is None:
            dihedrals = np.zeros((0, 6), dtype=np.int32)

        return TemplateData(
            n_atoms=n_atoms,
            n_bonds=n_bonds,
            n_angles=n_angles,
            n_dihedrals=n_dihedrals,
            atom_types=atom_types,
            coords=coords,
            bonds=bonds,
            angles=angles,
            dihedrals=dihedrals
        )

    def _parse_types(self, lines: List[str], n_atoms: int) -> np.ndarray:
        """解析原子类型"""
        atom_types = np.zeros(n_atoms + 1, dtype=np.int32)
        for line in lines:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                atom_id = int(parts[0])
                atom_type = int(parts[1])
                if 1 <= atom_id <= n_atoms:
                    atom_types[atom_id] = atom_type
        return atom_types

    def _parse_coords(self, lines: List[str], n_atoms: int) -> np.ndarray:
        """解析坐标"""
        coords = np.zeros((n_atoms + 1, 3), dtype=np.float64)
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                atom_id = int(parts[0])
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                if 1 <= atom_id <= n_atoms:
                    coords[atom_id] = [x, y, z]
        return coords

    def _parse_bonds(self, lines: List[str], n_bonds: int) -> np.ndarray:
        """解析键"""
        if n_bonds == 0:
            return np.zeros((0, 4), dtype=np.int32)
        bonds = np.zeros((n_bonds, 4), dtype=np.int32)
        for line in lines:
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                bond_id = int(parts[0])
                bond_type = int(parts[1])
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                if 1 <= bond_id <= n_bonds:
                    bonds[bond_id - 1] = [bond_id, bond_type, atom1, atom2]
        return bonds

    def _parse_angles(self, lines: List[str], n_angles: int) -> np.ndarray:
        """解析角度"""
        if n_angles == 0:
            return np.zeros((0, 5), dtype=np.int32)
        angles = np.zeros((n_angles, 5), dtype=np.int32)
        for line in lines:
            parts = line.split()
            if len(parts) >= 5 and parts[0].isdigit():
                angle_id = int(parts[0])
                angle_type = int(parts[1])
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                if 1 <= angle_id <= n_angles:
                    angles[angle_id - 1] = [angle_id, angle_type, atom1, atom2, atom3]
        return angles

    def _parse_dihedrals(self, lines: List[str], n_dihedrals: int) -> np.ndarray:
        """解析二面角"""
        if n_dihedrals == 0:
            return np.zeros((0, 6), dtype=np.int32)
        dihedrals = np.zeros((n_dihedrals, 6), dtype=np.int32)
        for line in lines:
            parts = line.split()
            if len(parts) >= 6 and parts[0].isdigit():
                dihedral_id = int(parts[0])
                dihedral_type = int(parts[1])
                atom1 = int(parts[2])
                atom2 = int(parts[3])
                atom3 = int(parts[4])
                atom4 = int(parts[5])
                if 1 <= dihedral_id <= n_dihedrals:
                    dihedrals[dihedral_id - 1] = [dihedral_id, dihedral_type, atom1, atom2, atom3, atom4]
        return dihedrals

    def parse_map(self, map_file: str) -> ReactionMapData:
        """
        解析反应映射文件

        Args:
            map_file: 映射文件路径

        Returns:
            ReactionMapData: 反应映射数据
        """
        path = Path(map_file)
        if not path.exists():
            raise ConfigError(f"映射文件未找到: {map_file}")

        with open(path, 'r') as f:
            lines = f.readlines()

        n_edge_ids = 0
        n_equivalences = 0
        n_constraints = 0

        initiator_ids = []
        edge_ids = []
        equivalences = {}
        constraints = []

        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 解析计数
            if 'edgeIDs' in line:
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        n_edge_ids = int(p)
                        break
            elif 'equivalences' in line:
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        n_equivalences = int(p)
                        break
            elif 'constraints' in line and 'constraint' not in line.lower().split()[0] if len(line.split()) > 0 else True:
                parts = line.split()
                for p in parts:
                    if p.isdigit():
                        n_constraints = int(p)
                        break

            # 检测section
            if line == 'InitiatorIDs':
                current_section = 'initiators'
                continue
            elif line == 'EdgeIDs':
                current_section = 'edges'
                continue
            elif line == 'Equivalences':
                current_section = 'equivalences'
                continue
            elif line == 'Constraints':
                current_section = 'constraints'
                continue

            # 解析数据
            if current_section == 'initiators' and line.isdigit():
                initiator_ids.append(int(line))
            elif current_section == 'edges' and line.isdigit():
                edge_ids.append(int(line))
            elif current_section == 'equivalences':
                parts = line.split()
                if len(parts) >= 2 and parts[0].isdigit():
                    pre_id = int(parts[0])
                    post_id = int(parts[1])
                    equivalences[pre_id] = post_id
            elif current_section == 'constraints':
                parts = line.split()
                if len(parts) >= 5:
                    constraint_type = parts[0]
                    atom1 = int(parts[1])
                    atom2 = int(parts[2])
                    min_dist = float(parts[3])
                    max_dist = float(parts[4])
                    constraints.append((constraint_type, atom1, atom2, min_dist, max_dist))

        return ReactionMapData(
            n_edge_ids=n_edge_ids,
            n_equivalences=n_equivalences,
            n_constraints=n_constraints,
            initiator_ids=initiator_ids,
            edge_ids=edge_ids,
            equivalences=equivalences,
            constraints=constraints
        )

    def parse_cg_mapping(self, cg_mapping_file: str,
                         n_template_atoms: int) -> TemplateCGMapping:
        """
        解析模板 CG 映射文件 (YAML 格式)

        文件格式示例:
        ```yaml
        # 模板 CG 映射
        mapping:
          1:                    # local_bead_id
            atoms: [1, 2, 3]    # 属于该 bead 的模板原子 ID
            bead_type: 1        # bead 类型
          2:
            atoms: [4, 5, 6]
            bead_type: 2
        ```

        Args:
            cg_mapping_file: CG 映射文件路径
            n_template_atoms: 模板中的原子数 (用于验证)

        Returns:
            TemplateCGMapping: 模板 CG 映射对象

        Raises:
            ConfigError: 文件不存在或格式错误
            ValueError: bead_id 一致性检查失败
        """
        import yaml

        path = Path(cg_mapping_file)
        if not path.exists():
            raise ConfigError(f"模板CG映射文件未找到: {cg_mapping_file}")

        with open(path, 'r') as f:
            data = yaml.safe_load(f)

        if 'mapping' not in data:
            raise ConfigError(f"CG映射文件缺少 'mapping' 字段: {cg_mapping_file}")

        mapping_data = data['mapping']
        bead_groups = {}

        # 用于验证每个原子只属于一个 bead
        atom_to_bead = {}

        for local_bead_id, bead_info in mapping_data.items():
            bead_id = int(local_bead_id)

            if 'atoms' not in bead_info:
                raise ConfigError(f"Bead {bead_id} 缺少 'atoms' 字段")
            if 'bead_type' not in bead_info:
                raise ConfigError(f"Bead {bead_id} 缺少 'bead_type' 字段")

            atom_ids = [int(a) for a in bead_info['atoms']]
            bead_type = int(bead_info['bead_type'])

            # 验证原子 ID 范围
            for atom_id in atom_ids:
                if atom_id < 1 or atom_id > n_template_atoms:
                    raise ConfigError(
                        f"原子 ID {atom_id} 超出模板范围 [1, {n_template_atoms}]"
                    )
                if atom_id in atom_to_bead:
                    raise ConfigError(
                        f"原子 {atom_id} 同时属于 bead {atom_to_bead[atom_id]} 和 {bead_id}"
                    )
                atom_to_bead[atom_id] = bead_id

            bead_groups[bead_id] = (atom_ids, bead_type)

        # 验证边缘原子不应该在 CG 映射中
        # (边缘原子由调用者在更新时排除)

        return TemplateCGMapping(
            bead_groups=bead_groups,
            n_beads=len(bead_groups)
        )

    def load_reaction_template(self, name: str,
                               pre_template_file: str,
                               post_template_file: str,
                               map_file: str,
                               pre_cg_mapping_file: Optional[str] = None,
                               post_cg_mapping_file: Optional[str] = None) -> ReactionTemplate:
        """
        加载完整反应模板

        Args:
            name: 反应名称
            pre_template_file: 反应前模板文件
            post_template_file: 反应后模板文件
            map_file: 映射文件
            pre_cg_mapping_file: 反应前 CG 映射文件 (可选)
            post_cg_mapping_file: 反应后 CG 映射文件 (可选)

        Returns:
            ReactionTemplate: 完整反应模板对象
        """
        pre_template = self.parse_template(pre_template_file)
        post_template = self.parse_template(post_template_file)
        reaction_map = self.parse_map(map_file)

        # 加载 CG 映射 (可选)
        pre_cg_mapping = None
        post_cg_mapping = None

        if pre_cg_mapping_file:
            pre_cg_mapping = self.parse_cg_mapping(
                pre_cg_mapping_file, pre_template.n_atoms
            )
        if post_cg_mapping_file:
            post_cg_mapping = self.parse_cg_mapping(
                post_cg_mapping_file, post_template.n_atoms
            )

        return ReactionTemplate(
            name=name,
            pre_template=pre_template,
            post_template=post_template,
            reaction_map=reaction_map,
            pre_cg_mapping=pre_cg_mapping,
            post_cg_mapping=post_cg_mapping
        )


def load_all_reaction_templates(reaction_dir: Path,
                                load_cg_mapping: bool = True) -> Dict[str, ReactionTemplate]:
    """
    加载目录下所有反应模板

    Args:
        reaction_dir: 反应配置目录
        load_cg_mapping: 是否加载 CG 映射文件

    Returns:
        Dict[str, ReactionTemplate]: 反应名称到模板的映射
    """
    parser = TemplateParser()
    templates = {}

    # 查找所有反应目录
    for rxn_dir in reaction_dir.iterdir():
        if not rxn_dir.is_dir():
            continue

        rxn_name = rxn_dir.name

        # 查找模板文件
        pre_files = list(rxn_dir.glob(f'{rxn_name}_pre.lammpstemplate'))
        post_files = list(rxn_dir.glob(f'{rxn_name}_post.lammpstemplate'))
        map_files = list(rxn_dir.glob(f'{rxn_name}.map'))

        if pre_files and post_files and map_files:
            try:
                # 查找 CG 映射文件 (可选)
                pre_cg_files = list(rxn_dir.glob(f'{rxn_name}_pre_mapping.yaml'))
                post_cg_files = list(rxn_dir.glob(f'{rxn_name}_post_mapping.yaml'))

                template = parser.load_reaction_template(
                    rxn_name,
                    str(pre_files[0]),
                    str(post_files[0]),
                    str(map_files[0]),
                    pre_cg_mapping_file=str(pre_cg_files[0]) if pre_cg_files and load_cg_mapping else None,
                    post_cg_mapping_file=str(post_cg_files[0]) if post_cg_files and load_cg_mapping else None
                )
                templates[rxn_name] = template

                # 打印加载信息
                cg_info = ""
                if template.pre_cg_mapping:
                    cg_info += f" (pre CG: {template.pre_cg_mapping.n_beads} beads)"
                if template.post_cg_mapping:
                    cg_info += f" (post CG: {template.post_cg_mapping.n_beads} beads)"
                # print(f"已加载反应模板: {rxn_name}{cg_info}")

            except Exception as e:
                print(f"警告: 加载反应模板 {rxn_name} 失败: {e}")

    return templates


if __name__ == "__main__":
    import sys

    # 测试模板解析器
    if len(sys.argv) > 1:
        template_file = sys.argv[1]
    else:
        # 使用默认测试文件
        template_file = "/home/ruifengjiang/PythonProject/lmp_py_react/lmp_react_test/MD_data/chunk0/md1/rxn1_pre.lammpstemplate"

    print("=" * 60)
    print("测试模板解析器")
    print("=" * 60)

    parser = TemplateParser()

    try:
        # 测试模板解析
        print(f"\n解析模板: {template_file}")
        template_data = parser.parse_template(template_file)
        print(f"  原子数: {template_data.n_atoms}")
        print(f"  键数: {template_data.n_bonds}")
        print(f"  角度数: {template_data.n_angles}")
        print(f"  二面角数: {template_data.n_dihedrals}")
        print(f"  原子类型: {template_data.atom_types[1:6]}...")
        print(f"  前5个键: {template_data.bonds[:5]}")

        # 测试完整反应模板加载
        if "rxn1_pre" in template_file:
            post_file = template_file.replace("_pre.", "_post.")
            map_file = template_file.replace("_pre.lammpstemplate", ".map")

            print(f"\n加载完整反应模板...")
            rxn_template = parser.load_reaction_template(
                "rxn1",
                template_file,
                post_file,
                map_file
            )

            print(f"  创建的键: {rxn_template.created_bonds}")
            print(f"  删除的键: {rxn_template.deleted_bonds}")
            print(f"  原子类型变化: {rxn_template.changed_atom_types}")
            print(f"  InitiatorIDs: {rxn_template.reaction_map.initiator_ids}")
            print(f"  EdgeIDs: {rxn_template.reaction_map.edge_ids}")

        print("\n✅ 模板解析测试成功!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)