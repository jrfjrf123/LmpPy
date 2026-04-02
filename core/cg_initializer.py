"""
CG 初始化器模块

功能:
- 生成初始 CG 映射文件 (AtomId_BeadId_compare_list.csv)
- 从 data 文件提取 AA bonds
- 生成 CG 拓扑 (bonds, angles, dihedrals)
- 验证 CG 映射一致性

整合功能:
- mapping_generator.py: CG 映射生成
- lammps_data_extractor.py: AA bonds 提取
- cg_bond_mapper.py: AA bonds → CG bonds 转换
- cg_topology.py: CG angles/dihedrals 推导

作者: Claude
日期: 2026-03-27
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from pathlib import Path
import numpy as np

# 支持两种导入方式
try:
    from .config_loader import ConfigLoader, SystemConfig, LAMMPSParams
    from .mapping_generator import MappingGenerator, CGCompareList, generate_cg_compare_list
    from .lammps_data_extractor import parse_bonds_from_data_file, parse_masses_from_data_file
    from .cg_bond_mapper import CGBondMapper, atom_bonds_to_cg_bonds
    from .cg_mapper import CGMapping
    from .cg_topology import (
        CGTopology, derive_cg_topology_from_bonds,
        verify_molecule_ids_consistency
    )
except ImportError:
    from config_loader import ConfigLoader, SystemConfig, LAMMPSParams
    from mapping_generator import MappingGenerator, CGCompareList, generate_cg_compare_list
    from lammps_data_extractor import parse_bonds_from_data_file, parse_masses_from_data_file
    from cg_bond_mapper import CGBondMapper, atom_bonds_to_cg_bonds
    from cg_mapper import CGMapping
    from cg_topology import (
        CGTopology, derive_cg_topology_from_bonds,
        verify_molecule_ids_consistency
    )


@dataclass
class CGSystem:
    """
    CG 体系数据结构

    包含完整的 CG 映射和拓扑信息
    """
    cg_compare_list: CGCompareList
    cg_topology: CGTopology
    n_atoms: int
    n_beads: int
    n_molecules: int

    @property
    def n_bonds(self) -> int:
        return self.cg_topology.n_bonds

    @property
    def n_angles(self) -> int:
        return self.cg_topology.n_angles

    @property
    def n_dihedrals(self) -> int:
        return self.cg_topology.n_dihedrals


class CGInitializer:
    """
    CG 初始化器

    功能:
    - 生成初始 CG 映射
    - 生成 CG 拓扑
    - 验证一致性
    """

    def __init__(self, config_dir: str):
        """
        初始化 CG 初始化器

        Args:
            config_dir: 配置目录路径
        """
        self.config_dir = Path(config_dir)
        self.loader = ConfigLoader(str(config_dir))

    def generate_initial_cg_mapping(self, output_path: Optional[str] = None) -> CGCompareList:
        """
        生成初始 CG 映射

        Args:
            output_path: 可选的输出文件路径

        Returns:
            CGCompareList: CG 映射数据
        """
        system_config = self.loader.load_system_config()

        # 使用 mapping_generator 的功能
        cg_list = generate_cg_compare_list(
            system_config,
            self.config_dir,
            output_path=output_path
        )

        return cg_list

    def extract_aa_bonds(self, data_file: Optional[str] = None) -> np.ndarray:
        """
        从 LAMMPS data 文件提取 AA bonds

        Args:
            data_file: data 文件路径，如果为 None 则从配置读取

        Returns:
            np.ndarray: AA bonds 数组 (n_bonds, 3) [bond_type, atom1, atom2]
        """
        if data_file is None:
            params = self.loader.load_lammps_params()
            data_file = params.data_file

        # 解析路径
        data_path = self.config_dir / data_file

        return parse_bonds_from_data_file(data_path)

    def generate_cg_topology(self,
                             cg_compare_list: CGCompareList,
                             aa_bonds: np.ndarray,
                             default_bond_type: int = 1,
                             default_angle_type: int = 1,
                             default_dihedral_type: int = 1) -> CGTopology:
        """
        从 CG 映射和 AA bonds 生成 CG 拓扑

        Args:
            cg_compare_list: CG 映射数据
            aa_bonds: AA bonds 数组
            default_bond_type: 默认 CG 键类型
            default_angle_type: 默认 CG 角度类型
            default_dihedral_type: 默认 CG 二面角类型

        Returns:
            CGTopology: CG 拓扑数据

        Raises:
            ValueError: 如果 CG 映射不完整或有原子未被映射
        """
        # 获取体系中的原子数
        n_atoms_in_system = int(aa_bonds[:, 1:].max())
        n_atoms_in_mapping = int(cg_compare_list.data[:, 3].max())

        # 验证原子数是否匹配
        if n_atoms_in_mapping < n_atoms_in_system:
            raise ValueError(
                f"CG 映射不完整: 体系中有 {n_atoms_in_system} 个原子，"
                f"但映射只覆盖了 {n_atoms_in_mapping} 个原子。"
                f"请检查 mapping 配置是否与实际体系一致。"
            )

        # 创建 CGMapping 对象
        cg_mapping = CGMapping.from_cg_compare_list(cg_compare_list.data, n_atoms_in_system)

        # 验证所有原子都有有效的 bead_id
        zero_bead_mask = cg_mapping.data[:, 0] == 0
        if zero_bead_mask.any():
            unmapped_atoms = np.where(zero_bead_mask)[0] + 1  # 1-indexed
            raise ValueError(
                f"CG 映射不完整: 发现 {len(unmapped_atoms)} 个原子未被映射（bead_id=0）。"
                f"示例: {list(unmapped_atoms[:10])}... "
                f"请检查 mapping 配置是否与实际体系一致。"
            )

        # AA bonds → CG bonds
        cg_bonds = atom_bonds_to_cg_bonds(
            aa_bonds,
            cg_mapping.data,
            default_bond_type
        )

        # CG bonds → CG angles/dihedrals
        cg_topology = derive_cg_topology_from_bonds(
            cg_bonds,
            default_angle_type,
            default_dihedral_type
        )

        return cg_topology

    def verify_consistency(self,
                           cg_compare_list: CGCompareList,
                           cg_topology: CGTopology) -> Tuple[bool, list]:
        """
        验证 CG 映射和拓扑的一致性

        Args:
            cg_compare_list: CG 映射数据
            cg_topology: CG 拓扑数据

        Returns:
            (is_consistent, issues): 是否一致和问题列表
        """
        issues = []

        # 验证 mol_id 与键连关系一致
        is_consistent, inconsistent_bonds = verify_molecule_ids_consistency(
            cg_compare_list.data,
            cg_topology.bonds
        )

        if not is_consistent:
            for bond, mol1, mol2 in inconsistent_bonds:
                issues.append(f"键 {bond} 连接不同分子: mol_id={mol1} vs {mol2}")

        return len(issues) == 0, issues

    def initialize(self,
                   output_mapping: Optional[str] = None,
                   output_topology_prefix: Optional[str] = None,
                   verify: bool = True) -> CGSystem:
        """
        完整的 CG 初始化流程

        Args:
            output_mapping: CG 映射输出路径
            output_topology_prefix: CG 拓扑输出前缀 (如 "cg" 生成 cg_bonds.txt 等)
            verify: 是否验证一致性

        Returns:
            CGSystem: CG 体系数据
        """
        print("=" * 60)
        print("CG 初始化")
        print("=" * 60)

        # 1. 加载配置
        print("\n1. 加载配置...")
        system_config = self.loader.load_system_config()
        params = self.loader.load_lammps_params()
        print(f"   体系名称: {system_config.name}")
        print(f"   Mapping 文件数: {len(system_config.mapping_files)}")

        # 2. 生成 CG 映射
        print("\n2. 生成 CG 映射...")
        if output_mapping is None:
            output_mapping = params.initial_cg_mapping
        cg_compare_list = self.generate_initial_cg_mapping(output_mapping)
        print(f"   原子映射数: {len(cg_compare_list.data)}")
        print(f"   Bead 数量: {cg_compare_list.n_beads}")
        print(f"   分子数量: {cg_compare_list.n_molecules}")

        # 3. 提取 AA bonds
        print("\n3. 提取 AA bonds...")
        aa_bonds = self.extract_aa_bonds(params.data_file)
        print(f"   AA bonds 数量: {len(aa_bonds)}")

        # 4. 生成 CG 拓扑
        print("\n4. 生成 CG 拓扑...")
        cg_topology = self.generate_cg_topology(cg_compare_list, aa_bonds)
        print(f"   CG bonds: {cg_topology.n_bonds}")
        print(f"   CG angles: {cg_topology.n_angles}")
        print(f"   CG dihedrals: {cg_topology.n_dihedrals}")

        # 5. 验证一致性
        if verify:
            print("\n5. 验证一致性...")
            is_consistent, issues = self.verify_consistency(cg_compare_list, cg_topology)
            if is_consistent:
                print("   ✅ 一致性验证通过")
            else:
                print(f"   ⚠️ 发现 {len(issues)} 个问题:")
                for issue in issues[:5]:  # 只显示前5个问题
                    print(f"      - {issue}")
                if len(issues) > 5:
                    print(f"      ... 还有 {len(issues) - 5} 个问题")

        # 6. 保存 CG 拓扑
        if output_topology_prefix:
            print(f"\n6. 保存 CG 拓扑...")
            cg_topology.to_files(output_topology_prefix)
            print(f"   保存到: {output_topology_prefix}_*.txt")

        # 创建 CGSystem 对象
        n_atoms = int(cg_compare_list.data[:, 3].max())

        cg_system = CGSystem(
            cg_compare_list=cg_compare_list,
            cg_topology=cg_topology,
            n_atoms=n_atoms,
            n_beads=cg_compare_list.n_beads,
            n_molecules=cg_compare_list.n_molecules
        )

        print("\n" + "=" * 60)
        print("✅ CG 初始化完成!")
        print("=" * 60)

        return cg_system


def initialize_cg_system(config_dir: str,
                         output_mapping: Optional[str] = None,
                         output_topology_prefix: Optional[str] = None,
                         verify: bool = True) -> CGSystem:
    """
    便捷函数：初始化 CG 体系

    Args:
        config_dir: 配置目录
        output_mapping: CG 映射输出路径
        output_topology_prefix: CG 拓扑输出前缀
        verify: 是否验证一致性

    Returns:
        CGSystem: CG 体系数据
    """
    initializer = CGInitializer(config_dir)
    return initializer.initialize(output_mapping, output_topology_prefix, verify)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    else:
        config_dir = "docs/examples"

    try:
        cg_system = initialize_cg_system(
            config_dir,
            output_mapping="test_cg_compare_list.csv",
            output_topology_prefix="test_cg",
            verify=True
        )

        print(f"\n最终统计:")
        print(f"  原子数: {cg_system.n_atoms}")
        print(f"  Bead 数: {cg_system.n_beads}")
        print(f"  分子数: {cg_system.n_molecules}")
        print(f"  CG bonds: {cg_system.n_bonds}")
        print(f"  CG angles: {cg_system.n_angles}")
        print(f"  CG dihedrals: {cg_system.n_dihedrals}")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)