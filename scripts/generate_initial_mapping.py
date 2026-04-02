#!/usr/bin/env python
"""
生成初始 CG 映射 CSV 文件

使用 ConfigLoader 加载配置，调用 generate_cg_compare_list 生成映射。

作者: Claude
日期: 2026-04-01
"""

import sys
from pathlib import Path

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.core.config_loader import ConfigLoader
from LmpPy.core.mapping_generator import generate_cg_compare_list


def main(config_dir: str = None):
    """
    生成初始 CG 映射

    Args:
        config_dir: 配置目录路径，默认使用 test_EPR_LmpPy/config
    """
    # 默认配置目录
    if config_dir is None:
        config_dir = Path(__file__).resolve().parent.parent.parent / "test_EPR_LmpPy" / "config"
    else:
        config_dir = Path(config_dir)

    print("=" * 60)
    print(f"配置目录: {config_dir}")
    print("=" * 60)

    # 加载配置
    loader = ConfigLoader(str(config_dir))
    system_config = loader.load_system_config()

    print(f"\n体系名称: {system_config.name}")
    print(f"Mapping 文件数: {len(system_config.mapping_files)}")

    # 从 lammps_params.yaml 获取输出文件名
    params = loader.load_lammps_params()
    output_path = str(config_dir / params.initial_cg_mapping)

    print(f"输出文件: {output_path}")

    # 生成 CG 映射
    cg_list = generate_cg_compare_list(system_config, config_dir, output_path)

    print("\n✅ CG 映射生成成功!")


if __name__ == "__main__":
    # 支持命令行参数指定配置目录
    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    else:
        config_dir = None

    try:
        main(config_dir)
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)