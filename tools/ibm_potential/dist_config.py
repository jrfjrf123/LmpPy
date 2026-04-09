"""
分布计算配置模块

提供 DistributionConfig 配置类，用于统一管理分布计算参数。

使用方法:
    from LmpPy.tools.ibm_potential.dist_config import DistributionConfig

    # 使用默认配置
    config = DistributionConfig()

    # 自定义配置
    config = DistributionConfig(
        n_bins=200,
        bond_range=(0.15, 0.65),
        calc_pairs=True,
        n_jobs=4
    )

    # 从 YAML 文件加载
    config = DistributionConfig.from_yaml('dist_config.yaml')

作者: Claude
日期: 2026-04-05
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional
from pathlib import Path


@dataclass
class DistributionConfig:
    """
    分布计算配置类。

    Attributes:
        n_bins: 直方图bins数
        unit: 距离单位 ('A' 或 'nm')
        bond_range: 键距离范围 (与unit一致)
        angle_range: 角度范围 (deg)
        dihedral_range: 二面角范围 (deg)
        pair_range: Pair距离范围 (与unit一致)
        calc_pairs: 是否计算非键合 pair 分布
        exclude_12: 是否排除1-2键合对
        exclude_13: 是否排除1-3键合对
        exclude_14: 是否排除1-4键合对
        n_jobs: 并行进程数
        show_progress: 是否显示进度条
    """
    # 直方图参数
    n_bins: int = 100

    # 单位 ('A' = 埃, 'nm' = 纳米)
    unit: str = 'A'

    # 范围参数 (默认单位: Å)
    bond_range: Tuple[float, float] = (2.0, 6.0)
    angle_range: Tuple[float, float] = (0, 180)
    dihedral_range: Tuple[float, float] = (-180, 180)
    pair_range: Tuple[float, float] = (3.0, 15.0)

    # 排除选项
    calc_pairs: bool = False
    exclude_12: bool = True
    exclude_13: bool = True
    exclude_14: bool = True

    # 性能选项
    n_jobs: int = 1
    show_progress: bool = True

    @classmethod
    def from_yaml(cls, filepath: str) -> 'DistributionConfig':
        """
        从 YAML 配置文件加载配置。

        Args:
            filepath: YAML 配置文件路径

        Returns:
            DistributionConfig 实例
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装 PyYAML: pip install pyyaml")

        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"配置文件不存在: {filepath}")

        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        # 提取 distribution 配置节
        dist_config = data.get('distribution', {})

        # 处理 tuple 类型的范围参数
        for key in ['bond_range', 'angle_range', 'dihedral_range', 'pair_range']:
            if key in dist_config and isinstance(dist_config[key], list):
                dist_config[key] = tuple(dist_config[key])

        return cls(**dist_config)

    def to_dict(self) -> dict:
        """转换为字典格式"""
        return {
            'n_bins': self.n_bins,
            'unit': self.unit,
            'bond_range': list(self.bond_range),
            'angle_range': list(self.angle_range),
            'dihedral_range': list(self.dihedral_range),
            'pair_range': list(self.pair_range),
            'calc_pairs': self.calc_pairs,
            'exclude_12': self.exclude_12,
            'exclude_13': self.exclude_13,
            'exclude_14': self.exclude_14,
            'n_jobs': self.n_jobs,
            'show_progress': self.show_progress
        }

    def to_yaml(self, filepath: str):
        """
        保存配置到 YAML 文件。

        Args:
            filepath: 输出文件路径
        """
        try:
            import yaml
        except ImportError:
            raise ImportError("需要安装 PyYAML: pip install pyyaml")

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {'distribution': self.to_dict()}

        with open(filepath, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)


# 默认配置实例
DEFAULT_CONFIG = DistributionConfig()


if __name__ == "__main__":
    # 测试配置类
    config = DistributionConfig(
        n_bins=200,
        calc_pairs=True,
        n_jobs=4
    )

    print("DistributionConfig 测试:")
    print(f"  n_bins: {config.n_bins}")
    print(f"  bond_range: {config.bond_range}")
    print(f"  calc_pairs: {config.calc_pairs}")
    print(f"  n_jobs: {config.n_jobs}")

    # 测试 to_dict
    print("\n转换为字典:")
    print(config.to_dict())