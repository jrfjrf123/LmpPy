#!/usr/bin/env python3
"""
冒烟测试独立运行脚本

功能:
- 解析命令行参数
- 调用 SmokeTestHarness 执行测试
- 返回 exit code (0=通过, 1=未通过)

用法:
    python LmpPy/scripts/test_smoke_run.py /path/to/config
    python LmpPy/scripts/test_smoke_run.py /path/to/config --loop-num 3
    python LmpPy/scripts/test_smoke_run.py /path/to/config --keep-output

作者: Claude
日期: 2026-06-04
"""

import sys
import argparse
from pathlib import Path

# 确保 LmpPy 在 sys.path 中
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from core.smoke_test_harness import SmokeTestHarness


def main():
    parser = argparse.ArgumentParser(
        description="LmpPy 冒烟测试 - 在少量 loop 下验证输出正确性"
    )
    parser.add_argument("config_dir", help="配置文件目录路径")
    parser.add_argument("--loop-num", type=int, default=5,
                        help="循环次数 (默认: 5)")
    parser.add_argument("--keep-output", action="store_true",
                        help="保留测试输出目录 (默认自动删除)")

    args = parser.parse_args()

    # 检查配置目录存在性
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"错误: 配置目录不存在: {config_dir}")
        sys.exit(1)

    # 运行测试
    harness = SmokeTestHarness(
        str(config_dir.resolve()),
        loop_num=args.loop_num,
        keep_output=args.keep_output
    )

    report = harness.run()
    if report is not None:
        report.print()
        sys.exit(0 if report.passed else 1)


if __name__ == "__main__":
    main()
