"""LmpPy 后处理分析 — CLI 统一入口。

与 mlcgsim output_analysis CLI 结构一致，新增 rebuild 子命令。
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """添加所有分析子命令通用的参数。"""
    parser.add_argument(
        "input_dir",
        type=str,
        help="模拟输出目录 (包含 final_frame.data, reaction_frames.npz 等)",
    )
    parser.add_argument(
        "--plot-config",
        type=str,
        default=None,
        help="绘图配置文件路径 (YAML 格式)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录 (默认: <input_dir>/analysis/)",
    )


def cmd_chain_length(args: argparse.Namespace) -> None:
    """链长分布分析。"""
    from .chain_length import analyze_chain_length

    analyze_chain_length(
        input_dir=args.input_dir,
        min_length=args.min_length,
        theory_type=args.theory_type,
        Mn=args.Mn,
        PDI=args.PDI,
        js=args.js,
        bins=args.bins,
        output_dir=args.output_dir,
        plot_config=args.plot_config,
        kde_bw=args.kde_bw,
    )


def cmd_distance(args: argparse.Namespace) -> None:
    """反应距离分布分析。"""
    from .distance import analyze_distance

    analyze_distance(
        input_dir=args.input_dir,
        ref_csv=args.ref_csv,
        bins=args.bins,
        output_dir=args.output_dir,
        plot_config=args.plot_config,
    )


def cmd_reaction_stats(args: argparse.Namespace) -> None:
    """反应统计分析与可视化。"""
    from .reaction_stats import analyze_reaction_stats

    analyze_reaction_stats(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        plot_config=args.plot_config,
    )


def cmd_all(args: argparse.Namespace) -> None:
    """运行全部三个分析。"""
    from .chain_length import analyze_chain_length
    from .distance import analyze_distance
    from .reaction_stats import analyze_reaction_stats

    input_dir = Path(args.input_dir)
    output_dir = args.output_dir or str(input_dir / "analysis")
    plot_config = args.plot_config

    print("=" * 60)
    print("1/3 链长分布分析")
    print("=" * 60)
    analyze_chain_length(
        input_dir=input_dir,
        min_length=args.min_length,
        theory_type=args.theory_type,
        Mn=args.Mn,
        PDI=args.PDI,
        js=args.js,
        bins=args.bins,
        output_dir=output_dir,
        plot_config=plot_config,
        kde_bw=args.kde_bw,
    )

    print("\n" + "=" * 60)
    print("2/3 反应距离分布分析")
    print("=" * 60)
    analyze_distance(
        input_dir=input_dir,
        ref_csv=args.ref_csv,
        bins=args.bins,
        output_dir=output_dir,
        plot_config=plot_config,
    )

    print("\n" + "=" * 60)
    print("3/3 反应统计分析")
    print("=" * 60)
    analyze_reaction_stats(
        input_dir=input_dir,
        output_dir=output_dir,
        plot_config=plot_config,
    )


def cmd_rebuild(args: argparse.Namespace) -> None:
    """从 reaction_frames.npz 预先重建 reaction_details.csv。"""
    from .loader import rebuild_reaction_details

    rebuild_reaction_details(
        npz_path=args.npz_path,
        output_csv=args.output,
        config_dir=args.config_dir,
    )


def main(argv: Optional[list] = None) -> None:
    """CLI 主入口。"""
    parser = argparse.ArgumentParser(
        prog="python -m LmpPy.output_analysis",
        description="LmpPy 粗粒化反应模拟后处理分析工具",
        epilog="示例: python -m LmpPy.output_analysis all ./process1/ --plot-config plot_config.yaml",
    )
    subparsers = parser.add_subparsers(dest="command", help="分析命令")

    # ---- chain-length ----
    p_cl = subparsers.add_parser(
        "chain-length", help="链长分布分析 (含理论曲线对比)"
    )
    _add_common_args(p_cl)
    p_cl.add_argument("--min-length", type=int, default=20, help="最小链长阈值")
    p_cl.add_argument(
        "--theory-type", type=str, default="schulz-zimm",
        choices=["schulz-zimm", "poisson", "lognormal"],
        help="理论曲线类型",
    )
    p_cl.add_argument("--Mn", type=float, default=None, help="理论数均链长")
    p_cl.add_argument("--PDI", type=float, default=None, help="理论多分散指数")
    p_cl.add_argument("--js", action="store_true", help="计算 JS 散度")
    p_cl.add_argument("--bins", type=int, default=50, help="直方图 bin 数")
    p_cl.add_argument("--kde-bw", type=str, default=None, help="KDE 带宽")

    # ---- distance ----
    p_dist = subparsers.add_parser(
        "distance", help="反应距离分布分析"
    )
    _add_common_args(p_dist)
    p_dist.add_argument("--ref-csv", type=str, default=None, help="参考距离 CSV")
    p_dist.add_argument("--bins", type=int, default=50, help="直方图 bin 数")

    # ---- reaction-stats ----
    p_rs = subparsers.add_parser(
        "reaction-stats", help="反应统计分析 (饼图 + 堆叠柱状图)"
    )
    _add_common_args(p_rs)

    # ---- all ----
    p_all = subparsers.add_parser(
        "all", help="运行全部三个分析"
    )
    _add_common_args(p_all)
    p_all.add_argument("--min-length", type=int, default=20, help="最小链长阈值")
    p_all.add_argument(
        "--theory-type", type=str, default="schulz-zimm",
        choices=["schulz-zimm", "poisson", "lognormal"],
        help="理论曲线类型",
    )
    p_all.add_argument("--Mn", type=float, default=None, help="理论数均链长")
    p_all.add_argument("--PDI", type=float, default=None, help="理论多分散指数")
    p_all.add_argument("--js", action="store_true", help="计算 JS 散度")
    p_all.add_argument("--bins", type=int, default=50, help="直方图 bin 数")
    p_all.add_argument("--kde-bw", type=str, default=None, help="KDE 带宽")
    p_all.add_argument("--ref-csv", type=str, default=None, help="参考距离 CSV")

    # ---- rebuild ----
    p_rebuild = subparsers.add_parser(
        "rebuild", help="从 reaction_frames.npz 重建 reaction_details.csv"
    )
    p_rebuild.add_argument(
        "npz_path", type=str, help="reaction_frames.npz 文件路径"
    )
    p_rebuild.add_argument(
        "-o", "--output", type=str, required=True,
        help="输出 CSV 路径",
    )
    p_rebuild.add_argument(
        "--config-dir", type=str, default=None,
        help="reactions 配置目录 (用于推断 reaction_type)",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    handlers = {
        "chain-length": cmd_chain_length,
        "distance": cmd_distance,
        "reaction-stats": cmd_reaction_stats,
        "all": cmd_all,
        "rebuild": cmd_rebuild,
    }

    handler = handlers.get(args.command)
    if handler:
        handler(args)


if __name__ == "__main__":
    main()
