#!/usr/bin/env python3
"""
使用示例 - 完整的工作流程

展示如何使用重构后的模块进行CG映射和反应处理
"""

import sys
import numpy as np
from pathlib import Path

# 添加父目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 导入核心模块
from core import (
    ConfigLoader,
    generate_cg_compare_list,
    LAMMPSDataExtractor,
    BondDetector,
    CGConverter,
    CGMapping,
    BondsRecorder,
)
from utils import wrap_coordinates, write_lammps_dump_file


def example_basic_usage():
    """基本使用示例"""
    print("=" * 60)
    print("示例 1: 基本使用流程")
    print("=" * 60)

    # 1. 加载配置
    config_dir = Path("config/")
    loader = ConfigLoader(str(config_dir))

    system_config = loader.load_system_config()
    lammps_params = loader.load_lammps_params()
    mass_list = loader.load_mass_list()

    print(f"体系: {system_config.name}")
    print(f"循环次数: {lammps_params.loop_num}")

    # 2. 生成CG映射
    cg_list = generate_cg_compare_list(system_config, config_dir)
    print(f"CG映射原子数: {len(cg_list.data)}")
    print(f"Bead数: {cg_list.n_beads}")

    return cg_list, mass_list


def example_cg_conversion(cg_list, mass_list):
    """CG坐标转换示例"""
    print("\n" + "=" * 60)
    print("示例 2: CG坐标转换")
    print("=" * 60)

    # 创建模拟的原子坐标数据
    n_atoms = 100
    atom_coords = np.zeros((n_atoms, 5), dtype=np.float64)
    atom_coords[:, 0] = np.arange(1, n_atoms + 1)  # ID
    atom_coords[:, 1] = np.random.randint(1, 5, n_atoms)  # type
    atom_coords[:, 2:5] = np.random.rand(n_atoms, 3) * 50  # coords

    # 创建CG映射数据
    cg_mapping_data = np.zeros((n_atoms, 2), dtype=np.int32)
    for i in range(n_atoms):
        bead_id = (i // 10) + 1
        bead_type = bead_id
        cg_mapping_data[i] = [bead_id, bead_type]

    # 使用转换器
    converter = CGConverter()

    # 第一次转换 (构建索引)
    cg_coords = converter.convert(atom_coords, cg_list.data[:n_atoms], mass_list)
    print(f"CG坐标形状: {cg_coords.shape}")

    # 第二次转换 (使用缓存)
    cg_coords2 = converter.convert(atom_coords, cg_list.data[:n_atoms], mass_list)
    print(f"缓存命中，结果一致: {np.allclose(cg_coords, cg_coords2)}")


def example_bond_detection():
    """键变化检测示例"""
    print("\n" + "=" * 60)
    print("示例 3: 键变化检测")
    print("=" * 60)

    # 创建测试键数据
    n_atoms = 100

    bonds_before = np.array([
        [1, 1, 2], [1, 2, 3], [1, 3, 4],
        [1, 10, 11], [1, 11, 12]
    ], dtype=np.int32)

    bonds_after = np.array([
        [1, 1, 2], [1, 2, 3], [1, 3, 4],
        [1, 10, 11], [1, 11, 12],
        [2, 50, 51]  # 新创建的键
    ], dtype=np.int32)

    # 检测键变化
    detector = BondDetector(n_atoms)
    changes = detector.detect(bonds_before, bonds_after)

    print(f"创建的键: {changes.n_created}")
    print(f"删除的键: {changes.n_deleted}")
    print(f"有变化: {changes.has_changes}")

    if changes.n_created > 0:
        print(f"新键: {changes.created_bonds}")


def example_wrap_coordinates():
    """坐标wrap示例"""
    print("\n" + "=" * 60)
    print("示例 4: 坐标Wrap")
    print("=" * 60)

    # 定义盒子
    box = np.array([[0, 50], [0, 50], [0, 50]], dtype=np.float64)

    # 创建测试坐标
    coords = np.array([
        [25, 25, 25],   # 盒子内部
        [55, -5, 60],   # 盒子外部
        [10, 60, 10],   # 部分外部
    ], dtype=np.float64)

    print(f"原始坐标:\n{coords}")

    # Wrap坐标
    wrapped_coords, images = wrap_coordinates(coords, box)

    print(f"\nWrap后坐标:\n{wrapped_coords}")
    print(f"\nImage标记:\n{images}")


def example_save_load():
    """保存和加载示例"""
    print("\n" + "=" * 60)
    print("示例 5: 保存和加载")
    print("=" * 60)

    # 创建键连表记录器
    recorder = BondsRecorder(output_dir="example_records")

    # 记录一次反应
    bonds_before = np.array([[1, 1, 2], [1, 2, 3]], dtype=np.int32)
    bonds_after = np.array([[1, 1, 2], [1, 2, 3], [2, 4, 5]], dtype=np.int32)

    recorder.record(
        timestep=100,
        run_step=1,
        bonds_before=bonds_before,
        bonds_after=bonds_after,
        reaction_type="rxn1"
    )

    print(f"记录数: {recorder.n_records}")

    # 获取摘要
    summary = recorder.get_summary()
    print(f"摘要: {summary}")

    # 清理
    import shutil
    if Path("example_records").exists():
        shutil.rmtree("example_records")


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("LAMMPS 反应模拟后处理 - 使用示例")
    print("=" * 60)

    try:
        # 示例1: 基本使用
        cg_list, mass_list = example_basic_usage()

        # 示例2: CG转换
        example_cg_conversion(cg_list, mass_list)

        # 示例3: 键变化检测
        example_bond_detection()

        # 示例4: 坐标Wrap
        example_wrap_coordinates()

        # 示例5: 保存加载
        example_save_load()

        print("\n" + "=" * 60)
        print("所有示例运行完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()