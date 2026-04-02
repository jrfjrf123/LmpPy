#!/usr/bin/env python
"""
分布平滑工具 CLI

用法:
    python -m LmpPy.scripts.smooth_distribution input_file.txt -o output_dir
    python -m LmpPy.scripts.smooth_distribution -d input_directory/ -o output_dir

作者: Claude
日期: 2026-04-02
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.smooth_utils import smooth_distribution, smooth_all_distributions, DEFAULT_TEMPERATURE


def main():
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(
        description='分布平滑工具 - 对bond/angle/dihedral/RDF分布进行平滑处理',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理单个文件
  python -m LmpPy.scripts.smooth_distribution bond_type1_dist.txt -o smoothed_output/

  # 处理目录下所有分布文件
  python -m LmpPy.scripts.smooth_distribution -d distributions/ -o smoothed_output/

  # 指定温度
  python -m LmpPy.scripts.smooth_distribution bond_type1_dist.txt -T 300 -o smoothed_output/
"""
    )

    parser.add_argument('files', nargs='*', help='输入分布文件')
    parser.add_argument('-d', '--directory', help='处理目录下所有 *_dist.txt 文件')
    parser.add_argument('-o', '--output', default='smoothed_output', help='输出目录')
    parser.add_argument('-T', '--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'温度 (K), 默认 {DEFAULT_TEMPERATURE}')
    parser.add_argument('-q', '--quiet', action='store_true', help='安静模式')

    args = parser.parse_args()

    verbose = not args.quiet

    # 收集输入文件
    input_files = []

    if args.files:
        for f in args.files:
            if Path(f).is_file():
                input_files.append(f)
            else:
                print(f"警告: 文件不存在: {f}")

    if args.directory:
        dir_path = Path(args.directory)
        if dir_path.is_dir():
            for f in dir_path.iterdir():
                if f.name.endswith('_dist.txt'):
                    input_files.append(str(f))
        else:
            print(f"错误: 目录不存在: {args.directory}")
            return 1

    if not input_files:
        print("错误: 未指定输入文件")
        print("使用 -h 查看帮助")
        return 1

    if verbose:
        print(f"\n{'#'*70}")
        print(f"# 分布平滑工具")
        print(f"# 输入文件: {len(input_files)}")
        print(f"# 输出目录: {args.output}")
        print(f"# 温度: {args.temperature} K")
        print(f"{'#'*70}")

    # 处理文件
    success_count = 0
    fail_count = 0

    for i, filepath in enumerate(input_files):
        if verbose:
            print(f"\n[{i+1}/{len(input_files)}] 处理: {filepath}")

        try:
            smoothed_path, result = smooth_distribution(
                filepath=filepath,
                output_dir=args.output,
                temperature=args.temperature,
                verbose=verbose
            )
            success_count += 1
        except Exception as e:
            if verbose:
                print(f"错误: {e}")
            fail_count += 1

    # 汇总
    if verbose:
        print(f"\n{'#'*70}")
        print(f"# 处理汇总")
        print(f"#   成功: {success_count}")
        print(f"#   失败: {fail_count}")
        print(f"#   输出目录: {Path(args.output).resolve()}")
        print(f"{'#'*70}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())