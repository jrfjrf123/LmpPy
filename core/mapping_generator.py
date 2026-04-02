"""
CG映射生成器模块

功能:
- 从mapping yaml文件生成初始cg_compare_list
- 复用yaml2csv_mapping.py的逻辑

输出格式:
bead_id,mol_id,bead_type,AA_id,mass

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from pathlib import Path
import numpy as np
import pandas as pd

# 支持两种导入方式
try:
    from .config_loader import MappingConfig, SystemConfig
except ImportError:
    from config_loader import MappingConfig, SystemConfig


@dataclass
class CGCompareList:
    """CG映射数据结构"""
    data: np.ndarray  # (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]

    def to_csv(self, path: str):
        """保存为CSV文件"""
        df = pd.DataFrame(self.data, columns=['bead_id', 'mol_id', 'bead_type', 'AA_id', 'mass'])
        df.to_csv(path, index=False)

    @staticmethod
    def from_csv(path: str) -> 'CGCompareList':
        """从CSV文件加载CG映射"""
        df = pd.read_csv(path)
        data = df.values.astype(np.float64)
        return CGCompareList(data)

    def to_numpy(self) -> np.ndarray:
        """返回numpy数组"""
        return self.data

    @property
    def n_beads(self) -> int:
        """唯一bead数量"""
        return len(np.unique(self.data[:, 0]))

    @property
    def n_molecules(self) -> int:
        """唯一分子数量"""
        return len(np.unique(self.data[:, 1]))


class MappingGenerator:
    """
    CG映射生成器

    功能:
    - 解析mapping yaml配置
    - 生成AA到CG的映射关系
    """

    def __init__(self, bead_type_map: Optional[Dict[str, int]] = None):
        """
        初始化映射生成器

        Args:
            bead_type_map: 可选的自定义bead类型映射
                          如 {'Head': 1, 'Mid': 2, 'End': 3}
                          如果为None，则自动生成连续编号
        """
        self.bead_type_map = bead_type_map

    def generate(self, mapping_config: MappingConfig,
                 global_bead_type_map: Dict[str, int],
                 aa_id_offset: int = 0,
                 mol_id_offset: int = 0,
                 bead_id_offset: int = 0) -> tuple:
        """
        从MappingConfig生成CG映射

        Args:
            mapping_config: CG映射配置对象
            global_bead_type_map: 全局 bead 类型名称映射 (从 system.yaml 提供)
            aa_id_offset: AA ID偏移量
            mol_id_offset: 分子ID偏移量
            bead_id_offset: bead ID偏移量

        Returns:
            (CGCompareList, next_bead_id, next_aa_id, next_mol_id)
        """
        site_types = mapping_config.site_types
        config_list = mapping_config.config

        # 使用全局 bead 类型映射
        bead_type_map = global_bead_type_map

        bead_list = []
        current_bead_id = 1 + bead_id_offset
        current_mol_id = 1 + mol_id_offset

        # 处理每个配置
        for config in config_list:
            anchor = int(config['anchor'])
            repeat = int(config['repeat'])
            offset = int(config['offset'])
            sites = config['sites']

            # 每个repeat代表一个分子
            for rep in range(1, repeat + 1):
                base_aa_id = anchor + (rep - 1) * offset + aa_id_offset

                # 处理每个site
                for site_info in sites:
                    site_name = site_info[0]
                    site_start = site_info[1]

                    # 获取原子索引和质量
                    atom_indices = site_types[site_name]['index']
                    x_weights = site_types[site_name]['x-weight']

                    # 获取bead类型 (从全局映射)
                    bead_type = bead_type_map.get(site_name, 1)

                    # 生成映射
                    for i, atom_idx in enumerate(atom_indices):
                        aa_id = base_aa_id + site_start + atom_idx + 1  # +1 for 1-based
                        aa_mass = x_weights[i]
                        bead_list.append((current_bead_id, current_mol_id, bead_type, aa_id, aa_mass))

                    current_bead_id += 1

                current_mol_id += 1

        # 计算下一个偏移量
        next_aa_id = aa_id_offset + repeat * offset + anchor
        next_mol_id = current_mol_id
        next_bead_id = current_bead_id

        # 转换为numpy数组
        data = np.array(bead_list, dtype=np.float64)

        return CGCompareList(data), next_bead_id, next_aa_id, next_mol_id

    def generate_from_system(self, system_config: SystemConfig,
                            config_dir: Path) -> CGCompareList:
        """
        从SystemConfig生成完整的CG映射

        Args:
            system_config: 体系配置
            config_dir: 配置文件目录

        Returns:
            CGCompareList: 完整的CG映射
        """
        # 使用全局 bead_type_names
        global_bead_type_map = system_config.bead_type_names

        all_bead_list = []
        current_aa_id_offset = 0
        current_mol_id_offset = 0
        current_bead_id_offset = 0

        # 处理每个mapping文件
        for mapping_file_info in system_config.mapping_files:
            mapping_rel_path = mapping_file_info['path']
            num_copies = mapping_file_info['copies']

            # 加载mapping配置
            try:
                from .config_loader import ConfigLoader, ConfigValidationError
            except ImportError:
                from config_loader import ConfigLoader, ConfigValidationError
            loader = ConfigLoader(str(config_dir))
            mapping_config = loader.load_mapping_config(mapping_rel_path)

            # 验证 mapping_config 中的 site_types 名称都在全局映射中
            for site_name in mapping_config.site_types.keys():
                if site_name not in global_bead_type_map:
                    raise ConfigValidationError(
                        f"site类型 '{site_name}' 未在全局 bead_type_names 中定义",
                        mapping_rel_path
                    )

            # 处理每个拷贝
            for copy_idx in range(num_copies):
                cg_list, next_bead_id, next_aa_id, next_mol_id = self.generate(
                    mapping_config,
                    global_bead_type_map=global_bead_type_map,
                    aa_id_offset=current_aa_id_offset,
                    mol_id_offset=current_mol_id_offset,
                    bead_id_offset=current_bead_id_offset
                )

                all_bead_list.append(cg_list.data)

                # 更新偏移量
                current_aa_id_offset = next_aa_id
                current_mol_id_offset = next_mol_id - 1
                current_bead_id_offset = next_bead_id - 1

        # 合并所有数据
        all_data = np.vstack(all_bead_list)

        # 按AA_id排序
        sorted_indices = np.argsort(all_data[:, 3])
        all_data = all_data[sorted_indices]

        return CGCompareList(all_data)


def generate_cg_compare_list(system_config: SystemConfig,
                            config_dir: Path,
                            output_path: Optional[str] = None) -> CGCompareList:
    """
    便捷函数：生成CG映射并可选保存

    Args:
        system_config: 体系配置
        config_dir: 配置目录
        output_path: 可选的输出CSV路径

    Returns:
        CGCompareList: CG映射对象
    """
    generator = MappingGenerator()
    cg_list = generator.generate_from_system(system_config, config_dir)

    if output_path:
        cg_list.to_csv(output_path)
        print(f"CG映射已保存到: {output_path}")
        print(f"  总原子映射数: {len(cg_list.data)}")
        print(f"  Bead数量: {cg_list.n_beads}")
        print(f"  分子数量: {cg_list.n_molecules}")

    return cg_list


if __name__ == "__main__":
    import sys

    # 测试CG映射生成器
    if len(sys.argv) > 1:
        config_dir = Path(sys.argv[1])
    else:
        config_dir = Path("config")

    try:
        from config_loader import ConfigLoader
    except ImportError:
        from .config_loader import ConfigLoader

    try:
        loader = ConfigLoader(str(config_dir))

        print("=" * 60)
        print("测试CG映射生成器")
        print("=" * 60)

        system_config = loader.load_system_config()
        print(f"\n体系名称: {system_config.name}")
        print(f"Mapping文件数: {len(system_config.mapping_files)}")

        # 生成CG映射
        cg_list = generate_cg_compare_list(
            system_config,
            config_dir,
            output_path="test_cg_compare_list.csv"
        )

        print("\n✅ CG映射生成成功!")

    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)