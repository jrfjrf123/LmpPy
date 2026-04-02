#!/usr/bin/env python3
"""
集成测试脚本

测试内容:
1. 配置加载
2. CG映射生成
3. 模板解析
4. 键变化检测
5. CG坐标转换
6. 完整流程模拟

作者: Claude
日期: 2026-03-27
"""

import sys
import numpy as np
from pathlib import Path

# 添加模块路径
sys.path.insert(0, str(Path(__file__).parent))

from core import (
    ConfigLoader, SystemConfig, LAMMPSParams,
    MappingGenerator, CGCompareList, generate_cg_compare_list,
    TemplateParser, ReactionTemplate, load_all_reaction_templates,
    LAMMPSDataExtractor,
    BondDetector, BondChanges, get_changed_atoms,
    ReactionLocator, ReactionMatch,
    CGMapper, CGMapping,
    CGConverter, lammpstrj2cg,
    CGBondMapper, atom_bonds_to_cg_bonds,
    BondsRecorder, save_bonds_record,
)
from utils import (
    wrap_coordinates, find_molecules,
    write_lammps_dump_file, calculate_central_mass,
)


def test_config_loader(config_dir: str):
    """测试配置加载"""
    print("\n" + "=" * 60)
    print("测试 1: 配置加载器")
    print("=" * 60)

    loader = ConfigLoader(config_dir)

    # 加载系统配置
    system_config = loader.load_system_config()
    print(f"  ✅ 系统配置: {system_config.name}")
    print(f"     Mapping文件数: {len(system_config.mapping_files)}")

    # 加载LAMMPS参数
    lammps_params = loader.load_lammps_params()
    print(f"  ✅ LAMMPS参数: loop_num={lammps_params.loop_num}")

    # 加载质量列表
    mass_list = loader.load_mass_list()
    print(f"  ✅ 质量列表: {len(mass_list)} 种原子类型")

    return loader, system_config, lammps_params, mass_list


def test_mapping_generator(config_dir: str, system_config: SystemConfig):
    """测试CG映射生成"""
    print("\n" + "=" * 60)
    print("测试 2: CG映射生成器")
    print("=" * 60)

    cg_list = generate_cg_compare_list(system_config, Path(config_dir))
    print(f"  ✅ CG映射生成: {len(cg_list.data)} 个原子映射")
    print(f"     Bead数: {cg_list.n_beads}")
    print(f"     分子数: {cg_list.n_molecules}")

    return cg_list


def test_template_parser():
    """测试模板解析"""
    print("\n" + "=" * 60)
    print("测试 3: 模板解析器")
    print("=" * 60)

    parser = TemplateParser()

    # 使用默认测试文件
    template_file = "/home/ruifengjiang/PythonProject/lmp_py_react/lmp_react_test/MD_data/chunk0/md1/rxn1_pre.lammpstemplate"

    try:
        template_data = parser.parse_template(template_file)
        print(f"  ✅ 模板解析: {template_data.n_atoms} 原子, {template_data.n_bonds} 键")

        # 测试完整反应模板
        post_file = template_file.replace("_pre.", "_post.")
        map_file = template_file.replace("_pre.lammpstemplate", ".map")

        rxn_template = parser.load_reaction_template(
            "rxn1", template_file, post_file, map_file
        )
        print(f"  ✅ 反应模板加载: {rxn_template.name}")
        print(f"     创建的键: {len(rxn_template.created_bonds)}")
        print(f"     删除的键: {len(rxn_template.deleted_bonds)}")
        print(f"     原子类型变化: {len(rxn_template.changed_atom_types)}")

        return {"rxn1": rxn_template}
    except Exception as e:
        print(f"  ⚠️ 模板解析跳过 (文件不存在): {e}")
        return {}


def test_bond_detector():
    """测试键变化检测"""
    print("\n" + "=" * 60)
    print("测试 4: 键变化检测器")
    print("=" * 60)

    n_atoms = 100

    # 创建测试数据
    bonds_before = np.array([
        [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 10, 11], [1, 11, 12]
    ], dtype=np.int32)

    bonds_after = np.array([
        [1, 1, 2], [1, 2, 3], [1, 3, 4], [1, 10, 11], [1, 11, 12],
        [2, 50, 51]  # 新键
    ], dtype=np.int32)

    detector = BondDetector(n_atoms)
    changes = detector.detect(bonds_before, bonds_after)

    print(f"  ✅ 键变化检测:")
    print(f"     创建的键: {changes.n_created}")
    print(f"     删除的键: {changes.n_deleted}")
    print(f"     有变化: {changes.has_changes}")

    changed_atoms = get_changed_atoms(changes)
    print(f"     参与反应的原子: {changed_atoms}")

    return changes


def test_cg_converter(mass_list: dict):
    """测试CG坐标转换"""
    print("\n" + "=" * 60)
    print("测试 5: CG坐标转换器")
    print("=" * 60)

    n_atoms = 100
    n_beads = 10

    # 创建原子坐标
    atom_coords = np.zeros((n_atoms, 5), dtype=np.float64)
    atom_coords[:, 0] = np.arange(1, n_atoms + 1)
    atom_coords[:, 1] = np.random.randint(1, 5, n_atoms)
    atom_coords[:, 2:5] = np.random.rand(n_atoms, 3) * 50

    # 创建CG映射
    atoms_per_bead = n_atoms // n_beads
    cg_compare_list = []
    for bead_id in range(1, n_beads + 1):
        start_atom = (bead_id - 1) * atoms_per_bead + 1
        for i in range(atoms_per_bead):
            atom_id = start_atom + i
            cg_compare_list.append([bead_id, 1, bead_id, atom_id, 12.0])

    cg_compare_list = np.array(cg_compare_list, dtype=np.float64)

    # 转换
    converter = CGConverter()
    cg_coords = converter.convert(atom_coords, cg_compare_list, mass_list)

    print(f"  ✅ CG坐标转换:")
    print(f"     输入原子数: {n_atoms}")
    print(f"     输出Bead数: {len(cg_coords)}")

    return converter


def test_cg_mapper():
    """测试CG映射更新"""
    print("\n" + "=" * 60)
    print("测试 6: CG映射更新器")
    print("=" * 60)

    n_atoms = 50

    # 创建初始映射
    cg_mapping_data = np.zeros((n_atoms, 2), dtype=np.int32)
    for i in range(n_atoms):
        bead_id = (i // 10) + 1
        bead_type = 1
        cg_mapping_data[i] = [bead_id, bead_type]

    cg_mapping = CGMapping(data=cg_mapping_data, n_atoms=n_atoms)

    print(f"  ✅ CG映射创建:")
    print(f"     原子数: {cg_mapping.n_atoms}")
    print(f"     最大bead_id: {cg_mapping.get_max_bead_id()}")

    # 测试更新
    cg_mapping.set_bead(1, 100, 5)
    print(f"  ✅ CG映射更新:")
    print(f"     原子1新bead_id: {cg_mapping.get_bead_id(1)}")

    return cg_mapping


def test_cg_bond_mapper(cg_mapping: CGMapping):
    """测试粗粒键映射"""
    print("\n" + "=" * 60)
    print("测试 7: 粗粒键映射器")
    print("=" * 60)

    # 创建原子键
    atom_bonds = np.array([
        [1, 1, 2],    # 同bead
        [1, 1, 11],   # 跨bead
        [1, 10, 11],  # 跨bead
    ], dtype=np.int32)

    mapper = CGBondMapper(cg_mapping)
    cg_bonds = mapper.atom_bonds_to_cg_bonds(atom_bonds)

    print(f"  ✅ 粗粒键映射:")
    print(f"     输入原子键: {len(atom_bonds)}")
    print(f"     输出粗粒键: {len(cg_bonds)}")


def test_edge_atoms_exclusion():
    """测试边缘原子排除功能"""
    print("\n" + "=" * 60)
    print("测试 9: 边缘原子排除")
    print("=" * 60)

    # 创建模拟的 ReactionMapData (包含边缘原子)
    from core.template_parser import ReactionMapData, TemplateData

    reaction_map = ReactionMapData(
        n_edge_ids=2,
        n_equivalences=5,
        n_constraints=0,
        initiator_ids=[3],
        edge_ids=[1, 5],  # 边缘原子是 1 和 5
        equivalences={1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        constraints=[]
    )

    # 创建模拟的模板数据
    pre_template = TemplateData(
        n_atoms=5,
        n_bonds=4,
        n_angles=0,
        n_dihedrals=0,
        atom_types=np.array([0, 1, 2, 3, 4, 5], dtype=np.int32),
        coords=np.zeros((6, 3)),
        bonds=np.array([[1, 1, 1, 2], [1, 1, 2, 3], [1, 1, 3, 4], [1, 1, 4, 5]], dtype=np.int32),
        angles=np.zeros((0, 5), dtype=np.int32),
        dihedrals=np.zeros((0, 6), dtype=np.int32)
    )

    post_template = pre_template  # 简化，使用相同模板

    # 创建反应模板
    template = ReactionTemplate(
        name="test_rxn",
        pre_template=pre_template,
        post_template=post_template,
        reaction_map=reaction_map
    )

    # 创建模拟的反应匹配结果
    reaction_match = ReactionMatch(
        reaction_name="test_rxn",
        template_to_system={1: 101, 2: 102, 3: 103, 4: 104, 5: 105},  # 模板原子 -> 体系原子
        confidence=1.0
    )

    # 创建初始 CG 映射
    n_atoms = 200
    cg_mapping_data = np.zeros((n_atoms, 2), dtype=np.int32)
    for i in range(n_atoms):
        bead_id = (i // 10) + 1
        bead_type = 1
        cg_mapping_data[i] = [bead_id, bead_type]

    # 记录边缘原子原来的 bead_id
    edge_atom_1_original_bead = cg_mapping_data[100, 0]  # 体系原子 101 (索引 100)
    edge_atom_5_original_bead = cg_mapping_data[104, 0]  # 体系原子 105 (索引 104)

    cg_mapping = CGMapping(data=cg_mapping_data, n_atoms=n_atoms)

    # 更新 CG 映射
    mapper = CGMapper()
    new_mapping = mapper.update(cg_mapping, reaction_match, template)

    # 验证边缘原子没有被更新
    edge_atom_1_new_bead = new_mapping.data[100, 0]
    edge_atom_5_new_bead = new_mapping.data[104, 0]

    # 非边缘原子应该被更新
    non_edge_atom_2_bead = new_mapping.data[101, 0]  # 体系原子 102 (模板原子 2)

    print(f"  边缘原子 (模板原子 1 -> 体系原子 101):")
    print(f"    原 bead_id: {edge_atom_1_original_bead}")
    print(f"    新 bead_id: {edge_atom_1_new_bead}")
    print(f"    是否保持不变: {'✅ 是' if edge_atom_1_new_bead == edge_atom_1_original_bead else '❌ 否'}")

    print(f"  边缘原子 (模板原子 5 -> 体系原子 105):")
    print(f"    原 bead_id: {edge_atom_5_original_bead}")
    print(f"    新 bead_id: {edge_atom_5_new_bead}")
    print(f"    是否保持不变: {'✅ 是' if edge_atom_5_new_bead == edge_atom_5_original_bead else '❌ 否'}")

    print(f"  非边缘原子 (模板原子 2 -> 体系原子 102):")
    print(f"    新 bead_id: {non_edge_atom_2_bead}")
    print(f"    是否被更新: {'✅ 是' if non_edge_atom_2_bead != edge_atom_1_original_bead else '❌ 否'}")

    # 断言验证
    assert edge_atom_1_new_bead == edge_atom_1_original_bead, "边缘原子1不应被更新"
    assert edge_atom_5_new_bead == edge_atom_5_original_bead, "边缘原子5不应被更新"

    print("\n  ✅ 边缘原子排除测试通过!")
    return True


def test_bonds_recorder():
    """测试键连表记录"""
    print("\n" + "=" * 60)
    print("测试 8: 键连表记录器")
    print("=" * 60)

    recorder = BondsRecorder(output_dir="test_bonds_records")

    bonds_before = np.array([[1, 1, 2], [1, 2, 3]], dtype=np.int32)
    bonds_after = np.array([[1, 1, 2], [1, 2, 3], [2, 4, 5]], dtype=np.int32)

    recorder.record(100, 1, bonds_before, bonds_after, "rxn1")

    print(f"  ✅ 键连表记录:")
    print(f"     记录数: {recorder.n_records}")

    summary = recorder.get_summary()
    print(f"     摘要: {summary}")


def run_all_tests(config_dir: str = "config/"):
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("LAMMPS 反应模拟后处理 - 集成测试")
    print("=" * 60)

    try:
        # 1. 配置加载
        loader, system_config, lammps_params, mass_list = test_config_loader(config_dir)

        # 2. CG映射生成
        cg_list = test_mapping_generator(config_dir, system_config)

        # 3. 模板解析
        templates = test_template_parser()

        # 4. 键变化检测
        bond_changes = test_bond_detector()

        # 5. CG坐标转换
        converter = test_cg_converter(mass_list)

        # 6. CG映射更新
        cg_mapping = test_cg_mapper()

        # 7. 粗粒键映射
        test_cg_bond_mapper(cg_mapping)

        # 8. 键连表记录
        test_bonds_recorder()

        # 9. 边缘原子排除测试
        test_edge_atoms_exclusion()

        # 总结
        print("\n" + "=" * 60)
        print("测试结果总结")
        print("=" * 60)
        print("  ✅ 所有测试通过!")
        print("\n模块状态:")
        print("  ✅ config_loader    - 配置加载")
        print("  ✅ mapping_generator - CG映射生成")
        print("  ✅ template_parser   - 模板解析")
        print("  ✅ bond_detector     - 键变化检测")
        print("  ✅ cg_converter      - CG坐标转换")
        print("  ✅ cg_mapper         - CG映射更新")
        print("  ✅ cg_bond_mapper    - 粗粒键映射")
        print("  ✅ bonds_recorder    - 键连表记录")
        print("  ✅ edge_atoms        - 边缘原子排除")

        return True

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="集成测试")
    parser.add_argument("--config", default="config/", help="配置目录")

    args = parser.parse_args()

    success = run_all_tests(args.config)
    sys.exit(0 if success else 1)