#!/usr/bin/env python
"""
Tabulated 势能表绘图脚本 - 绘制 LAMMPS/VOTCA table 格式的势能并生成图像

用法:
    python -m LmpPy.scripts.plot_tabulated -d ./mdrun
    python -m LmpPy.scripts.plot_tabulated -d ./mdrun -o ./plots
    python -m LmpPy.scripts.plot_tabulated nb11.pot.table bond1.pot.table

输入文件格式:
    VOTCA 生成的 LAMMPS table 格式 (*.pot.table 或 *.table)
    数据行: index x energy force  (force = -dU/dx)

势能类型从文件名自动识别:
    nbXX / nonbondXX / pairXX  -> nonbonded
    bondXX                     -> bond
    angleXX                    -> angle

作者: Claude
日期: 2026-08-13
"""

import sys
import argparse
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.ibm_potential.tabulated_potential import (
    collect_tables,
    plot_all_tables,
    AxisLimits,
    PlotLimits,
    HAS_MATPLOTLIB,
)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="绘制 LAMMPS/VOTCA tabulated 势能表图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理目录下所有势能表
  python -m LmpPy.scripts.plot_tabulated -d ./mdrun

  # 指定输出目录
  python -m LmpPy.scripts.plot_tabulated -d ./mdrun -o ./plots

  # 处理指定文件
  python -m LmpPy.scripts.plot_tabulated nb11.pot.table bond1.pot.table

  # 只绘制特定类型
  python -m LmpPy.scripts.plot_tabulated -d ./mdrun --types bond angle

  # 只绘制势能 (不绘制力)
  python -m LmpPy.scripts.plot_tabulated -d ./mdrun --no-force

  # 定制坐标范围: 默认限值 (作用于所有类型)
  python -m LmpPy.scripts.plot_tabulated -d ./mdrun --xlim 4 10 --ylim -0.5 1

  # 按类型定制: nonbonded 与 bond 各自不同的范围
  python -m LmpPy.scripts.plot_tabulated -d ./mdrun \\
      --xlim-nb 4 10 --ylim-nb -0.5 1 \\
      --xlim-bond 1 6 --ylim-bond -0.5 5
""",
    )

    parser.add_argument("files", nargs="*", help="输入势能表文件")
    parser.add_argument("-d", "--directory", help="处理目录下所有势能表文件")
    parser.add_argument("-o", "--output", default="./plots", help="输出目录 (默认: ./plots)")
    parser.add_argument("--types", nargs="+",
                        choices=["bond", "angle", "nonbonded", "dihedral", "all"],
                        default=["all"],
                        help="势能类型 (默认: all)")
    parser.add_argument("--no-force", action="store_true",
                        help="不绘制力曲线")
    parser.add_argument("--energy-unit", default="kcal/mol",
                        help="能量单位标签 (仅用于坐标轴标注, 默认: kcal/mol)")
    parser.add_argument("--y-max", type=float, default=None,
                        help="nonbonded 势绘图能量上限 (kcal/mol)，放大势阱区域")
    parser.add_argument("--xlim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="默认横坐标范围 (所有类型)")
    parser.add_argument("--ylim", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="默认纵坐标范围 (所有类型)")
    parser.add_argument("--xlim-nb", "--xlim-nonbonded", nargs=2, type=float,
                        metavar=("MIN", "MAX"), dest="xlim_nonbonded",
                        help="nonbonded 类型横坐标范围")
    parser.add_argument("--ylim-nb", "--ylim-nonbonded", nargs=2, type=float,
                        metavar=("MIN", "MAX"), dest="ylim_nonbonded",
                        help="nonbonded 类型纵坐标范围")
    parser.add_argument("--xlim-bond", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="bond 类型横坐标范围")
    parser.add_argument("--ylim-bond", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="bond 类型纵坐标范围")
    parser.add_argument("--xlim-angle", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="angle 类型横坐标范围")
    parser.add_argument("--ylim-angle", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="angle 类型纵坐标范围")
    parser.add_argument("--xlim-dihedral", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="dihedral 类型横坐标范围")
    parser.add_argument("--ylim-dihedral", nargs=2, type=float, metavar=("MIN", "MAX"),
                        help="dihedral 类型纵坐标范围")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    if not HAS_MATPLOTLIB:
        print("错误: 需要安装 matplotlib")
        print("  pip install matplotlib")
        return 1

    try:
        tables = collect_tables(args.files, args.directory, args.energy_unit)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    if not tables:
        print("错误: 未找到势能表文件")
        print("使用 -h 查看帮助")
        return 1

    # 类型过滤
    if "all" not in args.types:
        tables = [t for t in tables if t.kind in args.types]

    if not tables:
        print("错误: 过滤后无势能表文件")
        return 1

    if not args.quiet:
        print("=" * 60)
        print("Tabulated 势能表绘图")
        print("=" * 60)
        print(f"输出目录: {args.output}")
        print(f"势能表数量: {len(tables)}")
        for t in tables:
            ids = "".join(str(i) for i in t.type_ids)
            print(f"  {t.kind}: {t.name} (type {ids}, {t.n} points)")

    def _axis_limits(x, y):
        return AxisLimits(
            xlim=tuple(x) if x else None,
            ylim=tuple(y) if y else None,
        )

    limits = PlotLimits(
        default=_axis_limits(args.xlim, args.ylim),
        nonbonded=_axis_limits(args.xlim_nonbonded, args.ylim_nonbonded),
        bond=_axis_limits(args.xlim_bond, args.ylim_bond),
        angle=_axis_limits(args.xlim_angle, args.ylim_angle),
        dihedral=_axis_limits(args.xlim_dihedral, args.ylim_dihedral),
    )

    try:
        outputs = plot_all_tables(
            tables, args.output,
            show_force=not args.no_force,
            energy_cap=args.y_max,
            limits=limits,
        )
    except RuntimeError as e:
        print(f"错误: {e}")
        return 1

    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"✓ 完成! 生成 {len(outputs)} 个图像文件 -> {Path(args.output).resolve()}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
