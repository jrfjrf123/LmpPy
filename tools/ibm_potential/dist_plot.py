"""
分布绘图模块

提供分布可视化功能，用于绑制 bond/angle/dihedral/pair 分布图。

使用方法:
    from LmpPy.tools.ibm_potential.dist_plot import (
        plot_single_distribution,
        plot_all_distributions
    )

作者: Claude
日期: 2026-04-05
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

# 可选导入 matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_single_distribution(
    x: np.ndarray,
    hist: np.ndarray,
    output_file: str,
    title: str,
    xlabel: str,
    ylabel: str = "Probability",
    color: str = 'steelblue'
):
    """
    绘制单个分布图并保存。

    Args:
        x: 自变量数组
        hist: 概率分布数组
        output_file: 输出图片文件路径
        title: 图标题
        xlabel: x轴标签
        ylabel: y轴标签
        color: 线条颜色
    """
    if not HAS_MATPLOTLIB:
        print(f"  警告: matplotlib 未安装，跳过绑图")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, hist, color=color, linewidth=1.5)
    ax.fill_between(x, hist, alpha=0.3, color=color)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_all_distributions(
    output_dir: str,
    dist_type: str,
    distributions: List[Dict],
    xlabel: str,
    title_prefix: str
):
    """
    在一张图上绘制同一类型的所有分布。

    Args:
        output_dir: 输出目录
        dist_type: 分布类型 (bond, angle, dihedral, pair)
        distributions: 分布数据列表，每个元素为 {'name': str, 'x': np.ndarray, 'hist': np.ndarray}
        xlabel: x轴标签
        title_prefix: 标题前缀
    """
    if not HAS_MATPLOTLIB or not distributions:
        return

    output_dir = Path(output_dir)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    for i, dist in enumerate(distributions):
        color = colors[i % len(colors)]
        ax.plot(dist['x'], dist['hist'], label=dist['name'],
                color=color, linewidth=1.2, alpha=0.8)

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title(f"{title_prefix} Distributions", fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=10)

    output_file = output_dir / f"all_{dist_type}_distributions.png"
    plt.tight_layout()
    plt.savefig(str(output_file), dpi=150, bbox_inches='tight')
    plt.close(fig)

    print(f"  汇总图: {output_file.name}")


def plot_distribution_results(
    results: Dict[str, Dict],
    output_dir: str
):
    """
    绘制 calculate_all_distributions 返回的所有分布。

    Args:
        results: calculate_all_distributions 的返回结果
        output_dir: 输出目录
    """
    if not HAS_MATPLOTLIB:
        print("  警告: matplotlib 未安装，跳过绑图")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 绘制键分布
    if results.get('bond'):
        distributions = []
        for type_id, (r, hist) in results['bond'].items():
            # 单独图
            output_file = output_dir / f"bond_type{type_id}.png"
            plot_single_distribution(r, hist, str(output_file),
                                    f"Bond Type {type_id} Distribution", "Distance (nm)")
            # 汇总数据
            distributions.append({'name': f'Type {type_id}', 'x': r, 'hist': hist})

        # 汇总图
        if distributions:
            plot_all_distributions(str(output_dir), 'bond', distributions,
                                  "Distance (nm)", "Bond")

    # 绘制角度分布
    if results.get('angle'):
        distributions = []
        for type_id, (theta, hist) in results['angle'].items():
            output_file = output_dir / f"angle_type{type_id}.png"
            plot_single_distribution(theta, hist, str(output_file),
                                    f"Angle Type {type_id} Distribution", "Angle (deg)")
            distributions.append({'name': f'Type {type_id}', 'x': theta, 'hist': hist})

        if distributions:
            plot_all_distributions(str(output_dir), 'angle', distributions,
                                  "Angle (deg)", "Angle")

    # 绘制二面角分布
    if results.get('dihedral'):
        distributions = []
        for type_id, (phi, hist) in results['dihedral'].items():
            output_file = output_dir / f"dihedral_type{type_id}.png"
            plot_single_distribution(phi, hist, str(output_file),
                                    f"Dihedral Type {type_id} Distribution", "Dihedral (deg)")
            distributions.append({'name': f'Type {type_id}', 'x': phi, 'hist': hist})

        if distributions:
            plot_all_distributions(str(output_dir), 'dihedral', distributions,
                                  "Dihedral (deg)", "Dihedral")

    # 绘制 Pair 分布
    if results.get('pair'):
        distributions = []
        for (type1, type2), (r, g_r) in results['pair'].items():
            pair_name = f'Type {type1}-{type2}' if type1 != type2 else f'Type {type1}'
            output_file = output_dir / f"pair_type{type1}_{type2}.png"
            plot_single_distribution(r, g_r, str(output_file),
                                    f"Pair {pair_name} Distribution", "Distance (nm)", "g(r)")
            distributions.append({'name': pair_name, 'x': r, 'hist': g_r})

        if distributions:
            plot_all_distributions(str(output_dir), 'pair', distributions,
                                  "Distance (nm)", "Pair")


if __name__ == "__main__":
    print("分布绘图模块")
    print(f"matplotlib 可用: {HAS_MATPLOTLIB}")
    if not HAS_MATPLOTLIB:
        print("  安装 matplotlib 以启用绘图功能: pip install matplotlib")