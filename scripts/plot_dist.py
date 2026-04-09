#!/usr/bin/env python
"""
分布绘图脚本 - 绑制分布文件并生成图像

用法:
    python -m LmpPy.scripts.plot_dist -d ./dist_output
    python -m LmpPy.scripts.plot_dist -d ./dist_output -o ./plots
    python -m LmpPy.scripts.plot_dist bond_type1.dist.tgt angle_type1.dist.tgt

输入文件格式:
    支持 VOTCA .dist.tgt 格式或简单的两列数据文件（x, y）

作者: Claude
日期: 2026-04-05
"""

import sys
import argparse
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.ibm_potential.dist_plot import (
    plot_single_distribution,
    plot_all_distributions,
    HAS_MATPLOTLIB
)


def detect_distribution_type(filename: str) -> Tuple[str, str]:
    """
    从文件名检测分布类型。

    Args:
        filename: 文件名

    Returns:
        (dist_type, type_id): 分布类型和类型ID
    """
    name = Path(filename).stem.lower()

    # 移除常见的后缀
    for suffix in ['_dist', '.dist', '_tgt', '']:
        if name.endswith(suffix):
            name = name[:-len(suffix)] if suffix else name
            break

    # 检测类型
    if 'bond' in name:
        dist_type = 'bond'
        # 提取类型编号
        match = re.search(r'type(\d+)', name)
        type_id = match.group(1) if match else 'unknown'
    elif 'angle' in name:
        dist_type = 'angle'
        match = re.search(r'type(\d+)', name)
        type_id = match.group(1) if match else 'unknown'
    elif 'dihedral' in name:
        dist_type = 'dihedral'
        match = re.search(r'type(\d+)', name)
        type_id = match.group(1) if match else 'unknown'
    elif 'pair' in name or 'rdf' in name:
        dist_type = 'pair'
        match = re.search(r'type(\d+)_(\d+)', name)
        if match:
            type_id = f"{match.group(1)}-{match.group(2)}"
        else:
            match = re.search(r'type(\d+)', name)
            type_id = match.group(1) if match else 'unknown'
    else:
        dist_type = 'unknown'
        type_id = name

    return dist_type, type_id


def load_distribution_file(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    加载分布文件。

    Args:
        filepath: 文件路径

    Returns:
        (x, y): x 和 y 数组
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    # 尝试加载数据（跳过注释行和 VOTCA 的 'i' 列）
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    x = float(parts[0])
                    y = float(parts[1])
                    data.append([x, y])
                except ValueError:
                    continue

    if not data:
        raise ValueError(f"无法从文件加载数据: {filepath}")

    data = np.array(data)
    return data[:, 0], data[:, 1]


def get_axis_labels(dist_type: str) -> Tuple[str, str]:
    """
    获取坐标轴标签。

    Args:
        dist_type: 分布类型

    Returns:
        (xlabel, ylabel): x轴和y轴标签
    """
    labels = {
        'bond': ('Distance (nm)', 'Probability'),
        'angle': ('Angle (deg)', 'Probability'),
        'dihedral': ('Dihedral (deg)', 'Probability'),
        'pair': ('Distance (nm)', 'g(r)'),
        'rdf': ('Distance (nm)', 'g(r)'),
        'unknown': ('x', 'y')
    }
    return labels.get(dist_type, ('x', 'y'))


def discover_distribution_files(directory: str,
                                 extensions: List[str] = None) -> Dict[str, List[str]]:
    """
    发现目录下的分布文件。

    Args:
        directory: 目录路径
        extensions: 文件扩展名列表

    Returns:
        {dist_type: [filepaths]}
    """
    if extensions is None:
        extensions = ['.dist.tgt', '.dist', '.txt', '.dat']

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    files_by_type = {
        'bond': [],
        'angle': [],
        'dihedral': [],
        'pair': []
    }

    for ext in extensions:
        for filepath in directory.glob(f'*{ext}'):
            dist_type, _ = detect_distribution_type(filepath.name)
            if dist_type in files_by_type:
                files_by_type[dist_type].append(str(filepath))
            elif dist_type == 'rdf':
                files_by_type['pair'].append(str(filepath))

    # 排序
    for dtype in files_by_type:
        files_by_type[dtype].sort()

    return files_by_type


def plot_distributions_from_files(
    filepaths: List[str],
    output_dir: str,
    show_individual: bool = True,
    show_combined: bool = True
):
    """
    从文件列表绘制分布图。

    Args:
        filepaths: 文件路径列表
        output_dir: 输出目录
        show_individual: 是否绘制单独的图
        show_combined: 是否绘制汇总图
    """
    if not filepaths:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 确定分布类型
    dist_type, _ = detect_distribution_type(filepaths[0])
    xlabel, ylabel = get_axis_labels(dist_type)

    distributions = []

    for filepath in filepaths:
        try:
            x, y = load_distribution_file(filepath)
            _, type_id = detect_distribution_type(filepath)

            # 存储数据
            distributions.append({
                'name': f'Type {type_id}',
                'x': x,
                'hist': y,
                'filepath': filepath
            })

            # 绘制单独的图
            if show_individual:
                output_file = output_dir / f"{Path(filepath).stem}.png"
                plot_single_distribution(
                    x, y, str(output_file),
                    f"{dist_type.capitalize()} Type {type_id} Distribution",
                    xlabel, ylabel
                )
                print(f"  绘制: {output_file.name}")

        except Exception as e:
            print(f"  警告: 无法处理文件 {filepath}: {e}")

    # 绘制汇总图
    if show_combined and len(distributions) > 1:
        plot_all_distributions(
            str(output_dir), dist_type,
            [{'name': d['name'], 'x': d['x'], 'hist': d['hist']} for d in distributions],
            xlabel, dist_type.capitalize()
        )


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description='绘制分布文件图像',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理目录下所有分布文件
  python -m LmpPy.scripts.plot_dist -d ./dist_output

  # 指定输出目录
  python -m LmpPy.scripts.plot_dist -d ./dist_output -o ./plots

  # 只绘制汇总图
  python -m LmpPy.scripts.plot_dist -d ./dist_output --combined-only

  # 处理指定文件
  python -m LmpPy.scripts.plot_dist bond_type1.dist.tgt bond_type2.dist.tgt

  # 只处理特定类型
  python -m LmpPy.scripts.plot_dist -d ./dist_output --types bond angle
"""
    )

    parser.add_argument('files', nargs='*', help='输入分布文件')
    parser.add_argument('-d', '--directory', help='处理目录下所有分布文件')
    parser.add_argument('-o', '--output', default='./plots', help='输出目录 (默认: ./plots)')
    parser.add_argument('--types', nargs='+',
                        choices=['bond', 'angle', 'dihedral', 'pair', 'all'],
                        default=['all'],
                        help='分布类型 (默认: all)')
    parser.add_argument('--individual-only', action='store_true',
                        help='只绘制单独的图')
    parser.add_argument('--combined-only', action='store_true',
                        help='只绘制汇总图')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='安静模式')

    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("错误: 需要安装 matplotlib")
        print("  pip install matplotlib")
        return 1

    show_individual = not args.combined_only
    show_combined = not args.individual_only

    # 收集输入文件
    files_by_type = {
        'bond': [],
        'angle': [],
        'dihedral': [],
        'pair': []
    }

    # 从命令行参数获取文件
    if args.files:
        for filepath in args.files:
            dist_type, _ = detect_distribution_type(filepath)
            if dist_type in files_by_type:
                files_by_type[dist_type].append(filepath)
            elif dist_type == 'rdf':
                files_by_type['pair'].append(filepath)

    # 从目录获取文件
    if args.directory:
        try:
            discovered = discover_distribution_files(args.directory)
            for dtype, files in discovered.items():
                files_by_type[dtype].extend(files)
        except FileNotFoundError as e:
            print(f"错误: {e}")
            return 1

    if not any(files_by_type.values()):
        print("错误: 未找到分布文件")
        print("使用 -h 查看帮助")
        return 1

    # 处理类型过滤
    if 'all' not in args.types:
        for dtype in list(files_by_type.keys()):
            if dtype not in args.types:
                files_by_type[dtype] = []

    # 打印摘要
    if not args.quiet:
        print("=" * 60)
        print("分布绘图脚本")
        print("=" * 60)
        print(f"输出目录: {args.output}")
        for dtype, files in files_by_type.items():
            if files:
                print(f"  {dtype}: {len(files)} 文件")

    # 绘制每种类型的分布
    for dtype in ['bond', 'angle', 'dihedral', 'pair']:
        files = files_by_type[dtype]
        if not files:
            continue

        if not args.quiet:
            print(f"\n绘制 {dtype} 分布...")

        plot_distributions_from_files(
            files, args.output,
            show_individual=show_individual,
            show_combined=show_combined
        )

    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"✓ 完成! 输出目录: {Path(args.output).resolve()}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())