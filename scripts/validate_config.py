#!/usr/bin/env python
"""
配置验证 CLI 脚本

功能:
- 验证配置目录的完整性
- 输出详细的验证报告
- 支持多种输出格式

使用:
    python LmpPy/scripts/validate_config.py config_dir/
    python LmpPy/scripts/validate_config.py config_dir/ --format json
    python LmpPy/scripts/validate_config.py config_dir/ --quiet
"""

import argparse
import sys
import json
from pathlib import Path

# 支持两种导入方式
try:
    from LmpPy.core import validate_config, ValidationResult
except ImportError:
    # 直接运行时使用相对导入
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from core import validate_config, ValidationResult


def main():
    parser = argparse.ArgumentParser(
        description="验证 LmpPy 配置目录的完整性",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 验证配置并输出详细报告
    python validate_config.py config/

    # 验证配置并输出 JSON 格式
    python validate_config.py config/ --format json

    # 验证配置，仅输出错误和警告
    python validate_config.py config/ --quiet

    # 验证配置，不输出报告（用于脚本）
    python validate_config.py config/ --no-report && echo "验证通过"
        """
    )

    parser.add_argument(
        "config_dir",
        help="配置文件目录路径"
    )

    parser.add_argument(
        "-f", "--format",
        choices=["text", "json"],
        default="text",
        help="输出格式: text (默认) 或 json"
    )

    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="安静模式，仅输出错误和警告"
    )

    parser.add_argument(
        "--no-report",
        action="store_true",
        help="不输出报告，仅返回退出码"
    )

    parser.add_argument(
        "--exit-on-error",
        action="store_true",
        default=True,
        help="有错误时返回非零退出码 (默认启用)"
    )

    args = parser.parse_args()

    # 验证配置目录
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        print(f"错误: 配置目录不存在: {config_dir}")
        sys.exit(1)

    # 执行验证
    result = validate_config(str(config_dir), print_report=False)

    # 输出结果
    if args.no_report:
        # 不输出报告
        pass
    elif args.format == "json":
        # JSON 格式输出
        output = {
            "config_dir": str(config_dir),
            "passed": result.passed,
            "n_errors": result.n_errors,
            "n_warnings": result.n_warnings,
            "n_infos": result.n_infos,
            "issues": [
                {
                    "level": i.level,
                    "category": i.category,
                    "field": i.field,
                    "message": i.message,
                    "suggestion": i.suggestion,
                    "path": str(i.path) if i.path else None
                }
                for i in result.issues
            ]
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    elif args.quiet:
        # 安静模式，仅输出错误和警告
        for issue in result.issues:
            if issue.level in ["error", "warning"]:
                print(issue)
    else:
        # 详细报告
        result.print_report()

    # 退出码
    if args.exit_on_error and not result.passed:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()