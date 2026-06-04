#!/usr/bin/env python
"""
从预计算分布文件计算 IBM 势能

用法:
    python -m LmpPy.scripts.calc_ibm_potential_from_dist -d smoothed_output/
    python -m LmpPy.scripts.calc_ibm_potential_from_dist -d dist_output/ -t 400 --lammps-tables --plot

作者: Claude
日期: 2026-05-22

输入格式支持:
    1. VOTCA 原始格式 (*.dist.tgt): 3列 x, P, 'i'
    2. 平滑格式 (*_dist.txt): 5行元数据头 + 2列数据
"""

import sys
import os
import re
import glob
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.ibm_potential.boltzmann import (
    KB,
    calculate_bond_potential,
    calculate_angle_potential,
    calculate_dihedral_potential,
    calculate_pair_potential,
    extrapolate_and_smooth,
    calculate_force,
)

from LmpPy.tools.ibm_potential.lammps_table import (
    read_potential_file,
    create_lammps_table_files,
    create_reference_file,
    calculate_force as table_calculate_force,
)


# ============================================================================
# 配置类
# ============================================================================

@dataclass
class IBMFromDistConfig:
    """从分布文件计算 IBM 势能的配置"""
    distribution_dir: str = "smoothed_output/"
    output_dir: str = "potentials_output/"
    temperature: float = 400.0
    units: str = "kcal/mol"
    jacobian_correction: bool = True
    smooth_window: int = 21
    smooth_polyorder: int = 3
    extrap_method: str = "linear"
    extrap_points: int = 5
    generate_lammps_tables: bool = False
    generate_plots: bool = False

    # 可选: 手动指定分布文件
    distribution_files: List[str] = field(default_factory=list)


def load_config_from_yaml(config_path: str) -> IBMFromDistConfig:
    """从 YAML 文件加载配置"""
    import yaml

    with open(config_path, 'r') as f:
        data = yaml.safe_load(f)

    return IBMFromDistConfig(
        distribution_dir=data.get('distribution_dir', 'smoothed_output/'),
        output_dir=data.get('output_dir', 'potentials_output/'),
        temperature=data.get('temperature', 400.0),
        units=data.get('units', 'kcal/mol'),
        jacobian_correction=data.get('jacobian_correction', True),
        smooth_window=data.get('smooth_window', 21),
        smooth_polyorder=data.get('smooth_polyorder', 3),
        extrap_method=data.get('extrap_method', 'linear'),
        extrap_points=data.get('extrap_points', 5),
        generate_lammps_tables=data.get('generate_lammps_tables', False),
        distribution_files=data.get('distribution_files', []),
    )


# ============================================================================
# 分布文件加载
# ============================================================================

def parse_smoothed_file_header(filepath: str) -> Dict:
    """
    解析平滑文件的元数据头。

    Args:
        filepath: 平滑分布文件路径

    Returns:
        metadata: 元数据字典 {'temperature', 'method', 'original_file', 'dist_type'}
    """
    metadata = {
        'temperature': None,
        'method': None,
        'original_file': None,
        'dist_type': None,
        'is_smoothed': False,
    }

    with open(filepath, 'r') as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line.startswith('#'):
                # 第5行后是数据
                break

            metadata['is_smoothed'] = True

            # 解析温度
            if 'Temperature' in line:
                match = re.search(r'Temperature:\s*([\d.]+)', line)
                if match:
                    metadata['temperature'] = float(match.group(1))

            # 解析方法
            if 'Method' in line:
                match = re.search(r'Method:\s*(\w+)', line)
                if match:
                    metadata['method'] = match.group(1)

            # 解析原始文件
            if 'Original file' in line:
                match = re.search(r'Original file:\s*(.+)', line)
                if match:
                    metadata['original_file'] = match.group(1).strip()

            # 解析分布类型
            if 'Smoothed' in line:
                match = re.search(r'Smoothed\s+(\w+)\s+distribution', line)
                if match:
                    metadata['dist_type'] = match.group(1).lower()

    return metadata


def load_distribution_file(filepath: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    加载分布文件，自动检测格式。

    Args:
        filepath: 分布文件路径

    Returns:
        x: 坐标数组
        P: 概率密度数组
        metadata: 元数据字典
    """
    metadata = parse_smoothed_file_header(filepath)

    # 加载数据
    data = np.loadtxt(filepath)

    # 检测列数
    if data.shape[1] == 3:
        # VOTCA 格式: x, P, 'i' (第三列忽略)
        x = data[:, 0]
        P = data[:, 1]
    elif data.shape[1] == 2:
        # 平滑格式: x, P_smoothed
        x = data[:, 0]
        P = data[:, 1]
    else:
        raise ValueError(f"不支持的文件格式: {filepath} (列数={data.shape[1]})")

    return x, P, metadata


def detect_distribution_type(filepath: str) -> Tuple[str, str]:
    """
    从文件名识别分布类型。

    Args:
        filepath: 文件路径

    Returns:
        dist_type: 分布类型 ('bond', 'angle', 'dihedral', 'pair')
        type_str: 类型标识 (如 'type1', 'type1_2')
    """
    basename = os.path.basename(filepath)

    # 支持的文件名模式
    patterns = [
        (r'bond_type(\d+)', 'bond'),
        (r'angle_type(\d+)', 'angle'),
        (r'dihedral_type(\d+)', 'dihedral'),
        (r'pair_type(\d+)_(\d+)', 'pair'),
        # 平滑文件可能带有 .dist 前缀
        (r'bond_type(\d+)\.dist', 'bond'),
        (r'angle_type(\d+)\.dist', 'angle'),
        (r'dihedral_type(\d+)\.dist', 'dihedral'),
        (r'pair_type(\d+)_(\d+)\.dist', 'pair'),
    ]

    for pattern, dist_type in patterns:
        match = re.search(pattern, basename)
        if match:
            if dist_type == 'pair':
                type_str = f"type{match.group(1)}_{match.group(2)}"
            else:
                type_str = f"type{match.group(1)}"
            return dist_type, type_str

    # 尝试从平滑文件头信息获取
    metadata = parse_smoothed_file_header(filepath)
    if metadata.get('dist_type'):
        # 尝试从原始文件名提取
        if metadata.get('original_file'):
            return detect_distribution_type(metadata['original_file'])

    raise ValueError(f"无法从文件名识别分布类型: {filepath}")


def scan_distribution_directory(directory: str) -> Dict[str, List[str]]:
    """
    扫描目录中所有分布文件。

    Args:
        directory: 目录路径

    Returns:
        files_by_type: 按类型分组的文件列表
        {
            'bond': ['bond_type1_dist.txt', ...],
            'angle': [...],
            'dihedral': [...],
            'pair': [...]
        }
    """
    files_by_type = {
        'bond': [],
        'angle': [],
        'dihedral': [],
        'pair': [],
    }

    # 查找所有分布文件
    patterns = ['*.dist.tgt', '*_dist.txt']

    for pattern in patterns:
        for filepath in glob.glob(os.path.join(directory, pattern)):
            try:
                dist_type, type_str = detect_distribution_type(filepath)
                files_by_type[dist_type].append(filepath)
            except ValueError as e:
                print(f"  跳过文件: {e}")
                continue

    # 按类型编号排序
    for dist_type in files_by_type:
        files_by_type[dist_type].sort()

    return files_by_type


# ============================================================================
# 势能可视化
# ============================================================================

def plot_single_potential(x: np.ndarray, potential: np.ndarray,
                          dist_x: np.ndarray = None, dist_P: np.ndarray = None,
                          potential_type: str = 'Bond', type_str: str = 'type1',
                          temperature: float = 400.0, units: str = 'kcal/mol',
                          output_file: str = None,
                          figsize: Tuple[float, float] = (10, 8)):
    """
    绘制单个势能图，包含势能曲线和原始分布。

    Args:
        x: 坐标数组
        potential: 势能数组
        dist_x: 分布坐标数组 (可选)
        dist_P: 分布概率数组 (可选)
        potential_type: 势能类型 ('Bond', 'Angle', 'Dihedral', 'Pair')
        type_str: 类型标识
        temperature: 温度
        units: 单位
        output_file: 输出文件路径 (可选，不提供则显示)
        figsize: 图像大小
    """
    import matplotlib.pyplot as plt

    # 创建图形
    if dist_x is not None and dist_P is not None:
        # 双子图: 上方势能，下方分布
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[1.2, 1])

        # 上图: 势能
        ax1 = axes[0]
        finite_mask = np.isfinite(potential)
        ax1.plot(x[finite_mask], potential[finite_mask], 'b-', linewidth=1.5, label='Potential')

        # 设置势能图标题和标签
        xlabel = {
            'Bond': 'Distance (Å)',
            'Angle': 'Angle (deg)',
            'Dihedral': 'Dihedral Angle (deg)',
            'Pair': 'Distance (Å)',
        }.get(potential_type, 'Coordinate')

        ax1.set_xlabel(xlabel)
        ax1.set_ylabel(f'Potential ({units})')
        ax1.set_title(f'{potential_type} Potential ({type_str}) at T={temperature}K')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # 下图: 分布
        ax2 = axes[1]
        finite_dist_mask = dist_P > 0
        ax2.plot(dist_x[finite_dist_mask], dist_P[finite_dist_mask], 'r-', linewidth=1.5, label='Distribution')
        ax2.set_xlabel(xlabel)
        ax2.set_ylabel('Probability Density')
        ax2.set_title('Distribution (Input)')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.tight_layout()

    else:
        # 单图: 仅势能
        fig, ax = plt.subplots(figsize=(figsize[0], figsize[1] * 0.6))

        finite_mask = np.isfinite(potential)
        ax.plot(x[finite_mask], potential[finite_mask], 'b-', linewidth=1.5)

        xlabel = {
            'Bond': 'Distance (Å)',
            'Angle': 'Angle (deg)',
            'Dihedral': 'Dihedral Angle (deg)',
            'Pair': 'Distance (Å)',
        }.get(potential_type, 'Coordinate')

        ax.set_xlabel(xlabel)
        ax.set_ylabel(f'Potential ({units})')
        ax.set_title(f'{potential_type} Potential ({type_str}) at T={temperature}K')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

    # 保存或显示
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close(fig)
        return output_file
    else:
        plt.show()
        return None


def plot_all_potentials(output_dir: str,
                        distribution_dir: str = None,
                        figsize_per_type: Tuple[float, float] = (8, 6),
                        combine_same_type: bool = True,
                        verbose: bool = True):
    """
    绘制目录中所有势能文件的可视化图。

    Args:
        output_dir: 势能文件目录
        distribution_dir: 分布文件目录 (可选，用于叠加原始分布)
        figsize_per_type: 每个类型的图像大小
        combine_same_type: 是否将同类型势能合并到一张图
        verbose: 是否打印详细信息
    """
    import matplotlib.pyplot as plt

    # 查找所有势能文件
    potential_files = glob.glob(os.path.join(output_dir, '*_potential.txt'))

    if not potential_files:
        print(f"在 {output_dir} 中未找到势能文件")
        return []

    # 创建图像输出目录
    plot_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plot_dir, exist_ok=True)

    # 按类型分组
    files_by_type = {'bond': [], 'angle': [], 'dihedral': [], 'pair': []}
    for filepath in potential_files:
        basename = os.path.basename(filepath)
        for dist_type in ['bond', 'angle', 'dihedral', 'pair']:
            if basename.startswith(dist_type):
                files_by_type[dist_type].append(filepath)
                break

    output_plots = []

    # 绘制合并图
    if combine_same_type:
        for dist_type in ['bond', 'angle', 'dihedral', 'pair']:
            files = files_by_type[dist_type]
            if not files:
                continue

            fig, ax = plt.subplots(figsize=(figsize_per_type[0] * 1.2, figsize_per_type[1]))

            colors = plt.cm.tab10(np.linspace(0, 1, len(files)))

            for i, filepath in enumerate(sorted(files)):
                # 读取势能文件
                data = np.loadtxt(filepath)
                x = data[:, 0]
                potential = data[:, 1]

                # 提取类型信息
                basename = os.path.basename(filepath)
                type_str = basename.replace(f'{dist_type}_', '').replace('_potential.txt', '')

                finite_mask = np.isfinite(potential)
                ax.plot(x[finite_mask], potential[finite_mask],
                       color=colors[i], linewidth=1.5, label=type_str)

            xlabel = {
                'bond': 'Distance (Å)',
                'angle': 'Angle (deg)',
                'dihedral': 'Dihedral Angle (deg)',
                'pair': 'Distance (Å)',
            }.get(dist_type, 'Coordinate')

            ax.set_xlabel(xlabel)
            ax.set_ylabel('Potential (kcal/mol)')
            ax.set_title(f'{dist_type.capitalize()} Potentials (Combined)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

            plt.tight_layout()

            output_file = os.path.join(plot_dir, f'{dist_type}_potentials_combined.png')
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

            output_plots.append(output_file)

            if verbose:
                print(f"  {dist_type}: {len(files)} potentials -> {output_file}")

    # 绘制单个势能图
    for filepath in potential_files:
        # 读取势能文件
        data = np.loadtxt(filepath)
        x = data[:, 0]
        potential = data[:, 1]

        # 提取类型信息
        basename = os.path.basename(filepath)
        dist_type = None
        for dt in ['bond', 'angle', 'dihedral', 'pair']:
            if basename.startswith(dt):
                dist_type = dt
                break
        type_str = basename.replace(f'{dist_type}_', '').replace('_potential.txt', '')

        potential_type = dist_type.capitalize() if dist_type else 'Unknown'

        # 尝试加载对应的分布文件
        dist_x, dist_P = None, None
        if distribution_dir:
            # 寻找匹配的分布文件
            dist_patterns = [
                f'{dist_type}_{type_str}_dist.txt',
                f'{dist_type}_{type_str}.dist_dist.txt',
                f'{dist_type}_{type_str}.dist.tgt',
            ]
            for pattern in dist_patterns:
                dist_filepath = os.path.join(distribution_dir, pattern)
                if os.path.exists(dist_filepath):
                    try:
                        dist_data = np.loadtxt(dist_filepath)
                        if dist_data.shape[1] >= 2:
                            dist_x = dist_data[:, 0]
                            dist_P = dist_data[:, 1]
                            break
                    except Exception:
                        pass

        # 绘制
        output_file = os.path.join(plot_dir, f'{dist_type}_{type_str}_potential.png')
        plot_single_potential(
            x, potential, dist_x, dist_P,
            potential_type=potential_type,
            type_str=type_str,
            output_file=output_file,
            figsize=(10, 8)
        )

        output_plots.append(output_file)

    if verbose:
        print(f"\n势能图已保存到: {plot_dir}")
        print(f"  共生成 {len(output_plots)} 个图像文件")

    return output_plots


# ============================================================================
# 势能计算和保存
# ============================================================================

def save_potential_with_force(filename: str, x: np.ndarray, potential: np.ndarray,
                              potential_type: str, type_str: str,
                              temperature: float, units: str,
                              source_file: str, jacobian_correction: bool):
    """
    保存势能文件，包含力信息。

    Args:
        filename: 输出文件路径
        x: 坐标数组
        potential: 势能数组
        potential_type: 势能类型 ('Bond', 'Angle', 'Dihedral', 'Pair')
        type_str: 类型标识
        temperature: 温度
        units: 单位
        source_file: 源分布文件
        jacobian_correction: 是否应用了 Jacobian 校正
    """
    # 计算力
    force = calculate_force(x, potential)

    # 过滤无限值
    finite_mask = np.isfinite(potential) & np.isfinite(force)
    x_finite = x[finite_mask]
    potential_finite = potential[finite_mask]
    force_finite = force[finite_mask]

    with open(filename, 'w') as f:
        f.write(f"# {potential_type} potential ({type_str})\n")
        f.write(f"# Temperature: {temperature} K\n")
        f.write(f"# Units: {units}\n")
        f.write(f"# Jacobian correction: {jacobian_correction}\n")
        f.write(f"# Source: {source_file}\n")
        f.write("# x potential force\n")

        for i in range(len(x_finite)):
            f.write(f"{x_finite[i]:.10e} {potential_finite[i]:.10e} {force_finite[i]:.10e}\n")


def process_distribution_file(filepath: str, config: IBMFromDistConfig,
                              output_dir: str, verbose: bool = True) -> Optional[Tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, float]]:
    """
    处理单个分布文件，计算势能。

    Args:
        filepath: 分布文件路径
        config: 配置对象
        output_dir: 输出目录
        verbose: 是否打印详细信息

    Returns:
        (output_file, x, potential, dist_x, dist_P, dist_type, type_str, temperature):
        输出文件路径、坐标、势能、分布坐标、分布概率、类型、类型标识、温度
        失败返回 None
    """
    try:
        # 加载分布文件
        x_orig, P_orig, metadata = load_distribution_file(filepath)
        x, P = x_orig.copy(), P_orig.copy()  # 保留原始数据用于绘图

        # 检测分布类型
        dist_type, type_str = detect_distribution_type(filepath)

        # 确定温度 (优先级: 命令行 > 配置文件 > 文件头 > 默认值)
        temperature = config.temperature
        if metadata.get('temperature') and temperature == 400.0:
            temperature = metadata['temperature']

        # 是否需要额外平滑
        is_smoothed = metadata.get('is_smoothed', False)

        # 计算势能
        potential = None

        if dist_type == 'bond':
            x, potential = calculate_bond_potential(
                x, P,
                temperature=temperature,
                units=config.units,
                jacobian_correction=config.jacobian_correction
            )
            potential_type = 'Bond'

        elif dist_type == 'angle':
            x, potential = calculate_angle_potential(
                x, P,
                temperature=temperature,
                units=config.units,
                jacobian_correction=config.jacobian_correction
            )
            potential_type = 'Angle'

        elif dist_type == 'dihedral':
            x, potential = calculate_dihedral_potential(
                x, P,
                temperature=temperature,
                units=config.units
            )
            potential_type = 'Dihedral'

        elif dist_type == 'pair':
            x, potential, _ = calculate_pair_potential(
                x, P,
                temperature=temperature,
                units=config.units
            )
            potential_type = 'Pair'

        else:
            print(f"  不支持的分布类型: {dist_type}")
            return None

        # 平滑和外推 (仅对原始分布文件或配置要求时)
        if not is_smoothed or config.smooth_window > 0:
            x, potential = extrapolate_and_smooth(
                x, potential,
                smooth_window=config.smooth_window if not is_smoothed else 0,
                smooth_polyorder=config.smooth_polyorder,
                extrap_points=config.extrap_points,
                extrap_method=config.extrap_method
            )

        # 保存势能文件
        output_filename = os.path.join(output_dir, f"{dist_type}_{type_str}_potential.txt")
        save_potential_with_force(
            output_filename, x, potential,
            potential_type, type_str,
            temperature, config.units,
            filepath, config.jacobian_correction
        )

        if verbose:
            print(f"  {dist_type}_{type_str}: {len(x)} points, T={temperature}K")

        return (output_filename, x, potential, x_orig, P_orig, dist_type, type_str, temperature)

    except Exception as e:
        print(f"  处理失败: {filepath}")
        print(f"    错误: {e}")
        return None


# ============================================================================
# 主流程
# ============================================================================

def run_ibm_from_dist_pipeline(config_path: str = None,
                                distribution_dir: str = None,
                                output_dir: str = None,
                                temperature: float = None,
                                units: str = None,
                                generate_lammps_tables: bool = False,
                                generate_plots: bool = False,
                                verbose: bool = True):
    """
    从分布文件计算 IBM 势能的完整流程。

    Args:
        config_path: YAML 配置文件路径 (可选)
        distribution_dir: 分布文件目录
        output_dir: 输出目录
        temperature: 温度 (K)
        units: 能量单位
        generate_lammps_tables: 是否生成 LAMMPS 表文件
        generate_plots: 是否生成势能可视化图
        verbose: 是否打印详细信息
    """
    # 加载配置
    if config_path:
        config = load_config_from_yaml(config_path)
    else:
        config = IBMFromDistConfig()

    # 命令行参数覆盖配置
    if distribution_dir:
        config.distribution_dir = distribution_dir
    if output_dir:
        config.output_dir = output_dir
    if temperature:
        config.temperature = temperature
    if units:
        config.units = units
    if generate_lammps_tables:
        config.generate_lammps_tables = True
    if generate_plots:
        config.generate_plots = True

    if verbose:
        print("=" * 70)
        print("IBM 势能计算 (从预计算分布文件)")
        print("=" * 70)
        print(f"分布文件目录: {config.distribution_dir}")
        print(f"输出目录: {config.output_dir}")
        print(f"温度: {config.temperature} K")
        print(f"单位: {config.units}")
        print(f"Jacobian 校正: {config.jacobian_correction}")

    # 创建输出目录
    os.makedirs(config.output_dir, exist_ok=True)

    # 扫描分布文件
    if verbose:
        print("\n扫描分布文件...")

    files_by_type = scan_distribution_directory(config.distribution_dir)

    total_files = sum(len(files) for files in files_by_type.values())
    if total_files == 0:
        print(f"  在 {config.distribution_dir} 中未找到分布文件")
        return

    if verbose:
        print(f"  共找到 {total_files} 个分布文件:")
        for dist_type, files in files_by_type.items():
            if files:
                print(f"    {dist_type}: {len(files)} 个")

    # 处理分布文件
    if verbose:
        print("\n计算势能...")

    output_files = []
    potential_data_list = []  # 存储势能数据用于绘图

    for dist_type in ['bond', 'angle', 'dihedral', 'pair']:
        for filepath in files_by_type[dist_type]:
            result = process_distribution_file(
                filepath, config, config.output_dir, verbose
            )
            if result:
                output_filename = result[0]
                output_files.append(output_filename)
                potential_data_list.append(result)

    if verbose:
        print(f"\n势能文件已保存到: {config.output_dir}")
        print(f"  共生成 {len(output_files)} 个势能文件")

    # 生成势能可视化图
    if config.generate_plots:
        if verbose:
            print("\n生成势能可视化图...")

        plot_dir = os.path.join(config.output_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)

        # 绘制合并图 (同类型势能合并到一张图)
        for dist_type in ['bond', 'angle', 'dihedral', 'pair']:
            type_data = [d for d in potential_data_list if d[5] == dist_type]
            if not type_data:
                continue

            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(10, 7))

            colors = plt.cm.tab10(np.linspace(0, 1, len(type_data)))

            for i, (output_file, x, potential, dist_x, dist_P, dt, type_str, temp) in enumerate(sorted(type_data, key=lambda d: d[6])):
                finite_mask = np.isfinite(potential)
                ax.plot(x[finite_mask], potential[finite_mask],
                       color=colors[i], linewidth=1.5, label=type_str)

            xlabel = {
                'bond': 'Distance (Å)',
                'angle': 'Angle (deg)',
                'dihedral': 'Dihedral Angle (deg)',
                'pair': 'Distance (Å)',
            }.get(dist_type, 'Coordinate')

            ax.set_xlabel(xlabel)
            ax.set_ylabel(f'Potential ({config.units})')
            ax.set_title(f'{dist_type.capitalize()} Potentials (Combined)')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=8)

            plt.tight_layout()

            combined_plot_file = os.path.join(plot_dir, f'{dist_type}_potentials_combined.png')
            plt.savefig(combined_plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

            if verbose:
                print(f"  {dist_type}: {len(type_data)} potentials -> {combined_plot_file}")

        # 绘制单个势能图 (含分布)
        for output_file, x, potential, dist_x, dist_P, dist_type, type_str, temp in potential_data_list:
            potential_type = dist_type.capitalize()

            single_plot_file = os.path.join(plot_dir, f'{dist_type}_{type_str}_potential.png')

            # 绘制双子图: 势能 + 分布
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[1.2, 1])

            # 上图: 势能
            ax1 = axes[0]
            finite_mask = np.isfinite(potential)
            ax1.plot(x[finite_mask], potential[finite_mask], 'b-', linewidth=1.5, label='Potential')

            xlabel = {
                'bond': 'Distance (Å)',
                'angle': 'Angle (deg)',
                'dihedral': 'Dihedral Angle (deg)',
                'pair': 'Distance (Å)',
            }.get(dist_type, 'Coordinate')

            ax1.set_xlabel(xlabel)
            ax1.set_ylabel(f'Potential ({config.units})')
            ax1.set_title(f'{potential_type} Potential ({type_str}) at T={temp}K')
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # 下图: 分布
            ax2 = axes[1]
            finite_dist_mask = dist_P > 0
            ax2.plot(dist_x[finite_dist_mask], dist_P[finite_dist_mask], 'r-', linewidth=1.5, label='Distribution')
            ax2.set_xlabel(xlabel)
            ax2.set_ylabel('Probability Density')
            ax2.set_title('Distribution (Input)')
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.tight_layout()
            plt.savefig(single_plot_file, dpi=150, bbox_inches='tight')
            plt.close(fig)

        if verbose:
            print(f"\n势能可视化图已保存到: {plot_dir}")

    # 生成 LAMMPS 表文件
    if config.generate_lammps_tables:
        if verbose:
            print("\n生成 LAMMPS 表文件...")

        table_dir = os.path.join(config.output_dir, 'lammps_tables')
        table_files = create_lammps_table_files(config.output_dir, table_dir)

        if table_files:
            create_reference_file(table_dir)

            if verbose:
                print(f"\nLAMMPS 表文件已保存到: {table_dir}")

    if verbose:
        print("\n" + "=" * 70)
        print("✓ IBM 势能计算完成!")
        print("=" * 70)


# ============================================================================
# CLI 入口
# ============================================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='从预计算分布文件计算 IBM 势能',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从平滑分布文件计算势能
  python -m LmpPy.scripts.calc_ibm_potential_from_dist -d smoothed_output/

  # 指定温度和输出目录
  python -m LmpPy.scripts.calc_ibm_potential_from_dist -d dist_output/ -o potentials/ -t 450

  # 同时生成 LAMMPS 表文件和势能可视化图
  python -m LmpPy.scripts.calc_ibm_potential_from_dist -d smoothed_output/ --lammps-tables --plot

  # 仅生成势能图
  python -m LmpPy.scripts.calc_ibm_potential_from_dist -d smoothed_output/ --plot

  # 使用 YAML 配置文件
  python -m LmpPy.scripts.calc_ibm_potential_from_dist -c ibm_from_dist.yaml

输入文件格式:
  - VOTCA 格式 (*.dist.tgt): 3列 (x, P, 'i')
  - 平滑格式 (*_dist.txt): 5行元数据头 + 2列数据

文件命名规则:
  - bond_type{N}.dist.tgt / bond_type{N}_dist.txt
  - angle_type{N}.dist.tgt / angle_type{N}_dist.txt
  - dihedral_type{N}.dist.tgt / dihedral_type{N}_dist.txt
  - pair_type{t1}_{t2}.dist.tgt / pair_type{t1}_{t2}_dist.txt
"""
    )

    parser.add_argument('-d', '--distribution-dir',
                        help='分布文件目录 (默认: smoothed_output/)')
    parser.add_argument('-o', '--output-dir',
                        help='输出目录 (默认: potentials_output/)')
    parser.add_argument('-t', '--temperature', type=float,
                        help='温度 (K) (默认: 400.0，或从文件头提取)')
    parser.add_argument('-u', '--units', default='kcal/mol',
                        choices=['kcal/mol', 'kJ/mol', 'eV'],
                        help='能量单位 (默认: kcal/mol)')
    parser.add_argument('--lammps-tables', action='store_true',
                        help='生成 LAMMPS 表文件')
    parser.add_argument('--plot', action='store_true',
                        help='生成势能可视化图')
    parser.add_argument('-c', '--config',
                        help='YAML 配置文件路径')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='安静模式')

    # 高级参数
    parser.add_argument('--no-jacobian', action='store_true',
                        help='禁用 Jacobian 校正')
    parser.add_argument('--smooth-window', type=int, default=21,
                        help='Savitzky-Golay 窗口大小 (默认: 21)')
    parser.add_argument('--smooth-polyorder', type=int, default=3,
                        help='Savitzky-Golay 多项式阶数 (默认: 3)')
    parser.add_argument('--extrap-method', default='linear',
                        choices=['linear', 'exponential'],
                        help='边界外推方法 (默认: linear)')

    args = parser.parse_args()

    # 处理 Jacobian 校正参数
    jacobian_correction = not args.no_jacobian

    try:
        run_ibm_from_dist_pipeline(
            config_path=args.config,
            distribution_dir=args.distribution_dir,
            output_dir=args.output_dir,
            temperature=args.temperature,
            units=args.units,
            generate_lammps_tables=args.lammps_tables,
            generate_plots=args.plot,
            verbose=not args.quiet
        )
        return 0

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())