#!/usr/bin/env python
"""
分布平滑工具 CLI

用法:
    python -m LmpPy.scripts.smooth_distribution input_file.txt -o output_dir
    python -m LmpPy.scripts.smooth_distribution -d input_directory/ -o output_dir

支持格式:
    - LAMMPS/IBI 格式: *_dist.txt (如 bond_type1_dist.txt)
    - VOTCA 格式: *.dist.tgt (如 bond_type1.dist.tgt)

作者: Claude
日期: 2026-04-02 (更新: 2026-05-22 添加 VOTCA 格式支持)
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
  # 处理单个文件（LAMMPS/IBI 格式）
  python -m LmpPy.scripts.smooth_distribution bond_type1_dist.txt -o smoothed_output/

  # 处理单个文件（VOTCA 格式）
  python -m LmpPy.scripts.smooth_distribution bond_type1.dist.tgt -o smoothed_output/

  # 处理目录下所有分布文件
  python -m LmpPy.scripts.smooth_distribution -d distributions/ -o smoothed_output/

  # 指定温度
  python -m LmpPy.scripts.smooth_distribution bond_type1.dist.tgt -T 300 -o smoothed_output/

支持的文件格式:
  - *_dist.txt: LAMMPS/IBI 格式（如 bond_type1_dist.txt）
  - *.dist.tgt: VOTCA 格式（如 bond_type1.dist.tgt, angle_type1.dist.tgt）
"""
    )

    parser.add_argument('files', nargs='*', help='输入分布文件')
    parser.add_argument('-d', '--directory', help='处理目录下所有 *_dist.txt 和 *.dist.tgt 文件')
    parser.add_argument('-o', '--output', default='smoothed_output', help='输出目录')
    parser.add_argument('-T', '--temperature', type=float, default=DEFAULT_TEMPERATURE,
                       help=f'温度 (K), 默认 {DEFAULT_TEMPERATURE}')
    parser.add_argument('-q', '--quiet', action='store_true', help='安静模式')

    # 单位选项
    parser.add_argument('--input-dist-unit', default='A', choices=['A', 'nm'],
                        help='输入距离单位: A (Angstrom) 或 nm (默认: A)')
    parser.add_argument('--input-ang-unit', default='deg', choices=['deg', 'rad'],
                        help='输入角度单位: deg 或 rad (默认: deg)')
    parser.add_argument('--output-dist-unit', default='A', choices=['A', 'nm'],
                        help='输出距离单位: A (Angstrom) 或 nm (默认: A)')
    parser.add_argument('--output-ang-unit', default='deg', choices=['deg', 'rad'],
                        help='输出角度单位: deg 或 rad (默认: deg)')
    parser.add_argument('-t', '--threshold', type=float, default=1e-8,
                        help='输出阈值，小于该值的 x 和 P 设为 0 (默认: 1e-8)')

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
            # 支持 LAMMPS/IBI 格式 (*_dist.txt)
            for f in dir_path.iterdir():
                if f.name.endswith('_dist.txt'):
                    input_files.append(str(f))
            # 支持 VOTCA 格式 (*.dist.tgt)
            for f in dir_path.iterdir():
                if f.name.endswith('.dist.tgt'):
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
                verbose=verbose,
                input_dist_unit=args.input_dist_unit,
                input_ang_unit=args.input_ang_unit,
                output_dist_unit=args.output_dist_unit,
                output_ang_unit=args.output_ang_unit,
                threshold=args.threshold
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