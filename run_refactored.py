#!/usr/bin/env python3
"""
LAMMPS 反应模拟后处理脚本 - 重构版

功能:
- 多体系支持: 通过配置文件定义不同化学反应体系
- 配置驱动: 所有参数从YAML文件加载
- 高性能: 向量化/Numba优化关键计算
- 模块化: 清晰的模块划分

使用方法:
    python run_refactored.py config/

作者: Claude
日期: 2026-03-27
"""

import os
import sys
import pickle
import argparse
from time import time
from copy import deepcopy
from pathlib import Path
from typing import Dict

import numpy as np

# LAMMPS和MPI导入
try:
    from lammps import lammps, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    me = comm.Get_rank()
except ImportError:
    me = 0
    lammps = None
    LMP_STYLE_GLOBAL = 0
    LMP_TYPE_VECTOR = 0
    print("警告: LAMMPS或MPI未安装，将使用模拟模式")

# 导入重构后的模块
sys.path.insert(0, str(Path(__file__).parent))
from core import (
    # 配置
    ConfigLoader, SystemConfig, LAMMPSParams,
    # 映射
    MappingGenerator, CGCompareList, generate_cg_compare_list,
    # 模板
    TemplateParser, ReactionTemplate, load_all_reaction_templates,
    # 数据提取
    LAMMPSDataExtractor, get_atoms_bonds_info, get_lmp_box_info,
    # 反应处理
    BondDetector, BondChanges, get_changed_atoms,
    CGConverter, lammpstrj2cg,
    CGBondMapper, atom_bonds_to_cg_bonds,
    BondsRecorder, save_bonds_record,
)
from utils import (
    wrap_coordinates,
    find_molecules,
    write_lammps_dump_file,
    calculate_central_mass,
    unwrap_coords_python,
    NUMBA_AVAILABLE,
)


# ============================================================================
# LAMMPS 硬编码设置 - 用户可根据需要修改
# ============================================================================
# 注意: 以下设置在每次初始化LAMMPS时都会应用
# 如需使用外部输入脚本(in.file)，请在lammps_params.yaml中设置input_script字段

LAMMPS_UNITS = "real"                    # 单位系统
LAMMPS_ATOM_STYLE = "full"               # 原子风格

LAMMPS_PAIR_STYLE = "lj/cut 15"          # 对势风格
LAMMPS_BOND_STYLE = "harmonic"           # 键势风格
LAMMPS_ANGLE_STYLE = "harmonic"          # 角势风格
LAMMPS_DIHEDRAL_STYLE = "fourier"        # 二面角势风格

LAMMPS_NEIGH_MODIFY = "every 1 delay 0 check yes"  # 邻居列表更新设置
# ============================================================================


class LAMMPSReactionRunner:
    """
    LAMMPS反应模拟运行器

    功能:
    - 初始化LAMMPS实例
    - 运行主循环
    - 处理反应
    - 输出CG轨迹
    """

    def __init__(self, config_dir: str):
        """
        初始化运行器

        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir).resolve()  # 转换为绝对路径

        # 初始化状态 (必须在 _load_configs 之前)
        self.cg_compare_list = None
        self.reacted_nums = {}
        self.changed_bead_id_list = {}

        # 反应帧缓存（高效：列表动态添加，结束后转 npz）
        self.reaction_frames = {
            'aa_coords_before': [],    # 全原子坐标（反应前）
            'aa_coords_after': [],     # 全原子坐标（反应后）
            'aa_ids': [],              # 原子 ID
            'aa_types': [],            # 原子类型
            'aa_bonds_before': [],     # 原子键（反应前）
            'aa_bonds_after': [],      # 原子键（反应后）
            'cg_mapping_before': [],   # CG mapping（反应前）
            'cg_mapping_after': [],    # CG mapping（反应后）
            'cg_bonds_before': [],     # CG 键（反应前）
            'cg_bonds_after': [],      # CG 键（反应后）
            'timestep': [],            # 时间步
        }

        # 加载配置
        self._load_configs()

        # 初始化组件
        self._init_components()

    def _load_configs(self):
        """加载所有配置"""
        if me == 0:
            print("=" * 60)
            print("加载配置文件...")
            print("=" * 60)

        loader = ConfigLoader(str(self.config_dir))

        # 加载系统配置
        self.system_config = loader.load_system_config()
        if me == 0:
            print(f"  体系名称: {self.system_config.name}")

        # 加载LAMMPS参数
        self.lammps_params = loader.load_lammps_params()
        if me == 0:
            print(f"  循环次数: {self.lammps_params.loop_num}")

        # 加载质量列表 (传入 data_file 用于自动提取)
        self.mass_list = loader.load_mass_list(self.lammps_params.data_file)
        if me == 0:
            print(f"  原子类型数: {len(self.mass_list)}")

        # 加载反应模板
        reaction_dir = self.config_dir / "reactions"
        if reaction_dir.exists():
            self.reaction_templates = load_all_reaction_templates(reaction_dir)
            if me == 0:
                print(f"  反应模板数: {len(self.reaction_templates)}")
        else:
            self.reaction_templates = {}
            if me == 0:
                print("  反应模板数: 0 (未找到reactions目录)")

        # CG映射：优先加载已有文件，否则生成
        cg_mapping_path = self.config_dir / self.lammps_params.initial_cg_mapping
        if cg_mapping_path.exists():
            self.cg_compare_list = CGCompareList.from_csv(str(cg_mapping_path))
            if me == 0:
                print(f"  加载已有CG映射: {self.lammps_params.initial_cg_mapping}")
        else:
            self.cg_compare_list = generate_cg_compare_list(
                self.system_config, self.config_dir
            )
        if me == 0:
            print(f"  CG映射原子数: {len(self.cg_compare_list.data)}")

        # 加载 CG 级反应签名（用于 _update_cg_mapping 生产级匹配）
        self.cg_signature_index: Dict = {}
        self.cg_all_signatures = []
        self.cg_end_types: set = set()
        self.cg_monomer_types: set = set()
        self.cg_interior_types: set = set()
        if reaction_dir.exists():
            from core.cg_reaction_identifier import load_template_signatures
            (self.cg_signature_index,
             self.cg_all_signatures,
             self.cg_end_types,
             self.cg_monomer_types,
             self.cg_interior_types) = load_template_signatures(reaction_dir)
            if me == 0:
                print(f"  CG 反应签名数: {len(self.cg_all_signatures)}")

        # 判断反应模式
        self.bond_create_config = self.lammps_params.bond_create_config
        self.reaction_mode = "bond/create" if self.bond_create_config is not None else "bond/react"
        if me == 0:
            print(f"  反应模式: {self.reaction_mode}")

    def _init_components(self):
        """初始化组件"""
        # 数据提取器
        self.data_extractor = LAMMPSDataExtractor(use_cache=True)

        # CG转换器
        self.cg_converter = CGConverter()

        # 键连表记录器
        self.bonds_recorder = BondsRecorder(
            output_dir=str(self.config_dir / "bonds_records")
        )

        # 初始化反应计数
        if self.reaction_mode == "bond/create":
            for pair in self.bond_create_config.pairs:
                key = f"{pair.itype}-{pair.jtype}"
                self.reacted_nums[key] = 0
        else:
            for rxn in self.lammps_params.reactions:
                self.reacted_nums[rxn.name] = 0

    def run(self):
        """
        运行主循环

        缓存策略:
        - ids, types, bonds: 只在反应时变化，可缓存
        - coords, image_flags: 每次 MD 后都变化，需要更新
        - CG 转换: 只在有反应时进行
        """
        if lammps is None:
            if me == 0:
                print("错误: LAMMPS未安装，无法运行模拟")
            return

        # 切换工作目录到配置目录 (LAMMPS 需要从当前目录读取文件)
        original_dir = Path.cwd()
        os.chdir(self.config_dir)
        if me == 0:
            print(f"工作目录: {self.config_dir}")

        # 记录开始时间
        t1 = time()

        # 初始化LAMMPS
        lmp = self._init_lammps()

        # 准备输出文件 (只有 rank 0 执行文件操作)
        output_cg_traj = self.config_dir / self.lammps_params.output_cg_trajectory
        if me == 0:
            if output_cg_traj.exists():
                output_cg_traj.unlink()
            reaction_count_file = self.config_dir / self.lammps_params.output_reaction_count
            f_react_num = open(reaction_count_file, "w")
            if self.reaction_mode == "bond/create":
                rxn_names = [f"{p.itype}-{p.jtype}" for p in self.bond_create_config.pairs]
            else:
                rxn_names = [rxn.name for rxn in self.lammps_params.reactions]
            f_react_num.write("# timestep " + " ".join(rxn_names) + "\n")
        else:
            f_react_num = None

        # 同步所有进程，确保 rank 0 完成文件操作后再继续
        MPI.COMM_WORLD.Barrier()

        # 主循环
        if me == 0:
            print("\n" + "=" * 60)
            print("开始主循环...")
            print("=" * 60)

        params = self.lammps_params

        # 初始化缓存 (所有进程必须调用 extract_all，因为包含 MPI 集合操作)
        atom_data, bond_data = self.data_extractor.extract_all(lmp)
        if me == 0:
            cached = {
                'ids': atom_data.ids.copy(),
                'types': atom_data.types.copy(),
                'coords': atom_data.coords.copy(),
                'ixyz': atom_data.image_flags.copy(),
                'bonds': bond_data.bonds.copy(),
                'cg_coords': None,  # 延迟计算，只在需要时计算
                'cg_ixyz': None,
            }
            box = get_lmp_box_info(lmp)
            timestep = lmp.extract_global("ntimestep")
        else:
            cached = None

        for i in range(params.loop_num):
            if i % 1000 == 0 and me == 0:
                print(f"  循环进度: {i}/{params.loop_num}")

            # Step 1: 运行反应步骤（按模式分支）
            if self.reaction_mode == "bond/create":
                react_nums = self._run_bond_create(lmp, params)
            else:
                self._run_bond_react(lmp, params)

            # Step 2: 检查反应并处理
            if me == 0:
                # 获取当前时间步和盒子信息
                timestep = lmp.extract_global("ntimestep")
                box = get_lmp_box_info(lmp)

                # 检测反应（bond/react 模式提取计数，bond/create 已在 Step 1 返回）
                if self.reaction_mode != "bond/create":
                    react_nums = self._get_react_num(lmp)

                # 写入反应计数
                react_str = " ".join(str(v) for v in react_nums.values())
                f_react_num.write(f"{timestep} {react_str}\n")

                # 更新反应计数
                for name, num in react_nums.items():
                    self.reacted_nums[name] += num

                has_reaction = any(react_nums.values())
            else:
                has_reaction = False

            # 广播反应结果给所有进程
            has_reaction = MPI.COMM_WORLD.bcast(has_reaction, root=0)

            if has_reaction:
                # === 有反应：完整处理 ===
                # 所有进程都需要调用 extract_all (MPI 集合操作)
                atom_data_after, bond_data_after = self.data_extractor.extract_all(
                    lmp, use_cache=False
                )

                if me == 0:
                    # 帧1: 反应前（使用缓存）
                    if cached['cg_coords'] is None:
                        # 延迟计算 CG 坐标
                        cached['cg_coords'], cached['cg_ixyz'] = self._compute_cg_coords(
                            cached['ids'], cached['types'], cached['coords'],
                            cached['bonds'], box
                        )

                    # 写入反应前帧
                    self._write_cg_frame(
                        output_cg_traj, timestep - params.bond_react_check_step,
                        box, cached['cg_coords'], cached['cg_ixyz']
                    )

                    # 键变化检测
                    bond_changes = self._detect_bond_changes(cached['bonds'], bond_data_after.bonds)

                    # 保存 CG mapping（反应前）
                    cg_mapping_before = self.cg_compare_list.data.copy()

                    # 更新 CG 映射 (按模式选择更新策略)
                    if bond_changes.has_changes:
                        if self.reaction_mode == "bond/create":
                            from core.reaction_commands import update_cg_mapping_create
                            update_cg_mapping_create(
                                self.cg_compare_list.data,
                                cached['bonds'],
                                bond_data_after.bonds,
                                self.bond_create_config.cg_type_map
                            )
                        else:
                            self._update_cg_mapping(
                                cached['bonds'],
                                bond_data_after.bonds,
                                atom_data_after.types,
                                len(atom_data_after.ids)
                            )

                    # 保存 CG mapping（反应后，如果更新成功则与 before 不同）
                    cg_mapping_after = self.cg_compare_list.data.copy()

                    # CG 转换 (反应后)
                    cg_coords_after, cg_ixyz_after = self._compute_cg_coords(
                        atom_data_after.ids, atom_data_after.types,
                        atom_data_after.coords, bond_data_after.bonds, box
                    )

                    # 帧2: 反应后
                    self._write_cg_frame(
                        output_cg_traj, timestep, box, cg_coords_after, cg_ixyz_after
                    )

                    # === 缓存反应帧数据 ===
                    # 1. 全原子数据
                    self.reaction_frames['aa_coords_before'].append(cached['coords'].copy())
                    self.reaction_frames['aa_coords_after'].append(atom_data_after.coords.copy())
                    self.reaction_frames['aa_ids'].append(cached['ids'].copy())
                    self.reaction_frames['aa_types'].append(cached['types'].copy())
                    self.reaction_frames['aa_bonds_before'].append(cached['bonds'].copy())
                    self.reaction_frames['aa_bonds_after'].append(bond_data_after.bonds.copy())

                    # 2. CG mapping
                    self.reaction_frames['cg_mapping_before'].append(cg_mapping_before)
                    self.reaction_frames['cg_mapping_after'].append(cg_mapping_after)

                    # 3. CG 键连信息
                    cg_bonds_before = atom_bonds_to_cg_bonds(cached['bonds'], self.cg_compare_list.data)
                    cg_bonds_after = atom_bonds_to_cg_bonds(bond_data_after.bonds, self.cg_compare_list.data)
                    self.reaction_frames['cg_bonds_before'].append(cg_bonds_before)
                    self.reaction_frames['cg_bonds_after'].append(cg_bonds_after)

                    # 4. 时间步
                    self.reaction_frames['timestep'].append(timestep)

                    # 记录键变化
                    self.bonds_recorder.record(
                        timestep=timestep,
                        run_step=i,
                        bonds_before=cached['bonds'],
                        bonds_after=bond_data_after.bonds,
                        reaction_type="unknown"
                    )

                    # 更新全部缓存
                    cached = {
                        'ids': atom_data_after.ids.copy(),
                        'types': atom_data_after.types.copy(),
                        'coords': atom_data_after.coords.copy(),
                        'ixyz': atom_data_after.image_flags.copy(),
                        'bonds': bond_data_after.bonds.copy(),
                        'cg_coords': cg_coords_after.copy(),
                        'cg_ixyz': cg_ixyz_after.copy(),
                    }

                    # 标记缓存有效
                    self.data_extractor._cache_valid = True

            # 无反应时不需要更新 coords/ixyz，松弛后再更新

            # Step 3: 松弛阶段（按模式分支）
            if self.reaction_mode == "bond/create":
                self._run_relaxation_create(lmp, params)
            else:
                self._run_relaxation(lmp, params)

            # Step 4: 松弛后更新 coords/ixyz (所有进程参与 MPI 集合操作)
            atom_data = self.data_extractor.extract_atoms(lmp, use_cache=True)

            if me == 0:
                cached['coords'] = atom_data.coords.copy()
                cached['ixyz'] = atom_data.image_flags.copy()
                cached['cg_coords'] = None  # 松弛后 CG 坐标需要重新计算

            # 同步 (所有进程都需要调用 Barrier)
            MPI.COMM_WORLD.Barrier()

        # 收尾
        self._finalize(lmp, f_react_num, output_cg_traj, t1)

        # 恢复工作目录
        os.chdir(original_dir)

    def _compute_cg_coords(self, ids, types, coords, bonds, box):
        """
        计算 CG 坐标

        Args:
            ids: 原子 ID 数组
            types: 原子类型数组
            coords: 坐标数组
            bonds: 键连表
            box: 盒子信息

        Returns:
            (cg_coords, cg_ixyz): CG 坐标和 image flags
        """
        # 展开坐标
        molecule_ids = find_molecules(bonds, len(ids))
        unwrapped_coords = self._unwrap_coords(coords, bonds, molecule_ids, box)

        # 构建 dump 格式数据
        dump_data = np.column_stack([
            ids, types, unwrapped_coords,
            np.zeros((len(ids), 3), dtype=np.int32)
        ])

        # CG 转换
        cg_coords = lammpstrj2cg(
            dump_data, self.cg_compare_list.data, self.mass_list, self.cg_converter
        )

        # Wrap 坐标
        wrapped_coords, cg_ixyz = wrap_coordinates(
            cg_coords[:, 2:5], box, return_images=True
        )
        cg_coords[:, 2:5] = wrapped_coords

        return cg_coords, cg_ixyz

    def _write_cg_frame(self, output_path, timestep, box, cg_coords, cg_ixyz):
        """
        写入 CG 帧到轨迹文件

        Args:
            output_path: 输出文件路径
            timestep: 时间步
            box: 盒子信息
            cg_coords: CG 坐标
            cg_ixyz: CG image flags
        """
        write_lammps_dump_file(
            str(output_path), timestep, box,
            cg_coords[:, 0].astype(int),
            cg_coords[:, 1].astype(int),
            cg_coords[:, 2:5],
            cg_ixyz
        )

    def _detect_bond_changes(self, bonds_before, bonds_after):
        """
        检测键变化

        Args:
            bonds_before: 反应前键连表
            bonds_after: 反应后键连表

        Returns:
            BondChanges: 键变化结果
        """
        n_atoms = len(self.mass_list) if hasattr(self, 'mass_list') else max(
            bonds_before[:, 1:3].max(), bonds_after[:, 1:3].max()
        )
        detector = BondDetector(n_atoms)
        return detector.detect(bonds_before, bonds_after)

    def _update_cg_mapping(self, bonds_before, bonds_after, types_after, n_atoms):
        """
        更新 CG 映射（重写版：CG 签名匹配替代 AA BFS）。

        流程:
        1. AA 键 → CG 键转换
        2. CG 键差集 → 新 CG 键
        3. 两级匹配（3-bead 粗筛 → chain 精筛按需）
        4. bead_type + bead_id 原地更新

        Args:
            bonds_before: 反应前键连表 (n_bonds, 3)
            bonds_after: 反应后键连表 (n_bonds, 3)
            types_after: 反应后原子类型 (n_atoms,)  — 保留签名兼容，本次未使用
            n_atoms: 原子总数

        Returns:
            bool: 是否成功更新
        """
        if not self.cg_all_signatures:
            return False

        try:
            from core.cg_reaction_identifier import (
                build_cg_bond_graph, get_cg_bond_diff, match_reaction
            )

            cg_mapping_data = self.cg_compare_list.data

            # 1. AA → CG 键转换
            cg_bonds_before = atom_bonds_to_cg_bonds(bonds_before, cg_mapping_data)
            cg_bonds_after = atom_bonds_to_cg_bonds(bonds_after, cg_mapping_data)

            # 2. CG 键差集
            new_cg_bonds = get_cg_bond_diff(cg_bonds_before, cg_bonds_after)
            if not new_cg_bonds:
                return False

            # 3. 构建运行时查询结构
            cg_graph_after = build_cg_bond_graph(cg_bonds_after)
            bead_type_lut: Dict[int, int] = {}
            for row in cg_mapping_data:
                bid = int(row[0])
                if bid > 0:
                    bead_type_lut[bid] = int(row[2])

            # 4. 逐键匹配
            bead_type_updates: Dict[int, int] = {}
            bead_id_updates: Dict[int, int] = {}
            matched_count = 0

            for new_bond in new_cg_bonds:
                sig = match_reaction(
                    new_bond, bead_type_lut, cg_graph_after,
                    self.cg_signature_index, self.cg_end_types,
                    self.cg_monomer_types, self.cg_interior_types
                )
                if sig is None:
                    continue

                matched_count += 1
                b1, b2 = new_bond
                t1 = bead_type_lut[b1]
                t2 = bead_type_lut[b2]

                # 区分端和单体
                if t1 in self.cg_end_types and t2 in self.cg_monomer_types:
                    end_bead, monomer_bead = b1, b2
                elif t2 in self.cg_end_types and t1 in self.cg_monomer_types:
                    end_bead, monomer_bead = b2, b1
                else:
                    continue

                # 收集 type 更新
                new_type = sig.type_map.get(bead_type_lut[end_bead])
                if new_type is not None:
                    bead_type_updates[end_bead] = new_type
                new_type = sig.type_map.get(bead_type_lut[monomer_bead])
                if new_type is not None:
                    bead_type_updates[monomer_bead] = new_type

            if matched_count == 0:
                return False

            # 5. 向量化批量应用 bead_type 更新
            for bead_id, new_type in bead_type_updates.items():
                mask = cg_mapping_data[:, 0].astype(int) == bead_id
                cg_mapping_data[mask, 2] = float(new_type)

            # 6. bead_id 重分配（按需，当前 EPR 体系为空操作）
            for old_bead_id, new_bead_id in bead_id_updates.items():
                mask = cg_mapping_data[:, 0].astype(int) == old_bead_id
                cg_mapping_data[mask, 0] = float(new_bead_id)

            self.cg_compare_list.data = cg_mapping_data
            return True

        except Exception as e:
            import traceback
            if me == 0:
                print(f"警告: CG 映射更新失败: {e}")
                traceback.print_exc()
            return False

    def _init_lammps(self):
        """初始化LAMMPS实例"""
        params = self.lammps_params
        lmp = lammps(name="py_react")

        # 检查是否使用外部输入脚本
        if params.input_script:
            # 使用外部输入脚本 (传统方式)
            input_script = self.config_dir / params.input_script
            if not input_script.exists():
                raise FileNotFoundError(f"输入脚本不存在: {input_script}")
            lmp.file(str(input_script))
            lmp.command("log none")
        else:
            # 使用内置初始化命令
            if me == 0:
                print("  使用内置LAMMPS初始化...")

            # 1. 基本设置
            lmp.command(f"units {LAMMPS_UNITS}")
            lmp.command(f"atom_style {LAMMPS_ATOM_STYLE}")

            # 2. 势函数设置
            lmp.command(f"pair_style {LAMMPS_PAIR_STYLE}")
            lmp.command(f"bond_style {LAMMPS_BOND_STYLE}")
            lmp.command(f"angle_style {LAMMPS_ANGLE_STYLE}")
            lmp.command(f"dihedral_style {LAMMPS_DIHEDRAL_STYLE}")

            # 3. 读取数据文件
            data_file = self.config_dir / params.data_file
            if not data_file.exists():
                raise FileNotFoundError(f"数据文件不存在: {data_file}")

            extra = params.read_data_extra
            read_data_cmd = (
                f"read_data {data_file} "
                f"extra/special/per/atom {extra.special_per_atom} "
                f"extra/bond/per/atom {extra.bond_per_atom} "
                f"extra/angle/per/atom {extra.angle_per_atom} "
                f"extra/dihedral/per/atom {extra.dihedral_per_atom}"
            )
            if me == 0:
                print(f"    读取数据文件: {params.data_file}")
            lmp.command(read_data_cmd)

            # 4. 邻居列表设置
            lmp.command(f"neigh_modify {LAMMPS_NEIGH_MODIFY}")

            # 5. 时间步设置
            lmp.command(f"timestep {params.timestep}")

            lmp.command("log none")

        # 加载分子模板
        # 策略：从 molecules 配置和 reactions 配置中合并加载，避免重复
        loaded_mols = set()

        # 1. 从 molecules 配置加载 (向后兼容)
        if params.molecules:
            if me == 0:
                print(f"  加载分子模板 (molecules配置): {len(params.molecules)} 个")
            for mol_name, mol_file in params.molecules.items():
                mol_path = self.config_dir / mol_file
                if not mol_path.exists():
                    if me == 0:
                        print(f"    警告: 分子模板文件不存在: {mol_path}")
                    continue
                lmp.command(f"molecule {mol_name} {mol_path}")
                if me == 0:
                    print(f"    加载: {mol_name} <- {mol_file}")
                loaded_mols.add(mol_name)

        # 2. 从 reactions 配置加载 (仅 bond/react 模式)
        if self.reaction_mode == "bond/react" and params.reactions:
            rxn_mols = []
            for rxn in params.reactions:
                if rxn.pre_mol and rxn.pre_mol not in loaded_mols and rxn.pre_template:
                    rxn_mols.append((rxn.pre_mol, rxn.pre_template))
                if rxn.post_mol and rxn.post_mol not in loaded_mols and rxn.post_template:
                    rxn_mols.append((rxn.post_mol, rxn.post_template))

            if rxn_mols:
                if me == 0:
                    print(f"  加载分子模板 (reactions配置): {len(rxn_mols)} 个")
                for mol_name, mol_path_str in rxn_mols:
                    mol_path = self.config_dir / mol_path_str
                    if not mol_path.exists():
                        if me == 0:
                            print(f"    警告: 分子模板文件不存在: {mol_path}")
                        continue
                    lmp.command(f"molecule {mol_name} {mol_path}")
                    if me == 0:
                        print(f"    加载: {mol_name} <- {mol_path}")
                    loaded_mols.add(mol_name)

        # 设置速度
        velocity_seed = np.random.randint(10, 10000)
        lmp.command(f"velocity all create {params.temperature} {velocity_seed}")

        return lmp

    def _get_ensemble_fix(self, params: LAMMPSParams, group: str, fix_name: str) -> str:
        """
        生成系综 fix 命令

        Args:
            params: LAMMPS 参数
            group: 原子组名称
            fix_name: fix 名称

        Returns:
            LAMMPS fix 命令字符串
        """
        temp = params.temperature
        if params.ensemble.lower() == "nvt":
            return f"fix {fix_name} {group} nvt temp {temp} {temp} {params.tcouple}"
        else:  # NPT (默认)
            return f"fix {fix_name} {group} npt temp {temp} {temp} {params.tcouple} iso {params.pressure} {params.pressure} {params.pcouple}"

    def _run_bond_react(self, lmp, params: LAMMPSParams):
        """运行bond/react"""
        temp = params.temperature

        # 构建反应命令
        react_cmds = []
        for i, rxn in enumerate(params.reactions):
            react_cmds.append(
                f"react {rxn.name} all 1 0.0 {rxn.cutoff} "
                f"{rxn.pre_mol} {rxn.post_mol} {rxn.map_file}"
            )

        react_cmd = " ".join(react_cmds)

        lmp.command(
            f"fix rxns all bond/react stabilization yes npt_grp {params.stabilization} {react_cmd}"
        )
        lmp.command(self._get_ensemble_fix(params, "npt_grp_REACT", "1"))
        lmp.command("thermo_style custom step temp press density")
        lmp.command(f"run {params.bond_react_check_step}")

    def _run_bond_create(self, lmp, params):
        """顺序执行每个 bond/create pair，避免多 fix 冲突。返回各对反应计数"""
        from core.reaction_commands import generate_fix_bond_create

        react_nums = {}
        cmds = generate_fix_bond_create(self.bond_create_config)
        for i, (fix_id, cmd) in enumerate(cmds):
            lmp.command(cmd)
            lmp.command(self._get_ensemble_fix(params, "all", "create_ensemble"))
            lmp.command("thermo_style custom step temp press density")
            lmp.command("run 1")
            # 在 unfix 之前提取计数
            pair = self.bond_create_config.pairs[i]
            key = f"{pair.itype}-{pair.jtype}"
            try:
                num = lmp.extract_fix(fix_id, LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, nrow=0)
                react_nums[key] = int(num) if num else 0
            except Exception:
                react_nums[key] = 0
            lmp.command(f"unfix {fix_id}")
            lmp.command("unfix create_ensemble")
        return react_nums

    def _run_relaxation(self, lmp, params: LAMMPSParams):
        """运行松弛阶段"""
        temp = params.temperature

        # 将bond/react创建的动态原子组转为静态，以便后续系综控制
        lmp.command("group npt_grp_REACT static")
        lmp.command("group bond_react_MASTER_group static")
        lmp.command("group reaction_atom union bond_react_MASTER_group")
        lmp.command("group non_reaction_atom union npt_grp_REACT")

        # 关闭反应fix
        lmp.command("unfix 1")
        lmp.command("unfix rxns")
        lmp.command("thermo_style custom step temp press density")

        # NVE/limit + 系综控制
        lmp.command(f"fix react_nve reaction_atom nve/limit {params.stabilization}")
        lmp.command(self._get_ensemble_fix(params, "non_reaction_atom", "react_npt"))
        lmp.command(f"run {params.nve_limit_step}")

        lmp.command("unfix react_nve")
        lmp.command("unfix react_npt")

        # 正常系综控制
        lmp.command(self._get_ensemble_fix(params, "all", "normal_npt"))
        normal_npt_step = params.run_step - params.nve_limit_step - params.bond_react_check_step
        lmp.command(f"run {normal_npt_step}")

        lmp.command("unfix normal_npt")

    def _run_relaxation_create(self, lmp, params, reacted_atom_ids=None):
        """
        bond/create 松弛阶段: 全局 NVE/limit → 全局系综
        """
        lmp.command("thermo_style custom step temp press density")

        # 1. 全局 NVE/limit（吸收键能冲击）
        lmp.command(f"fix relax_nve all nve/limit {params.stabilization}")
        lmp.command(f"run {params.nve_limit_step}")
        lmp.command("unfix relax_nve")

        # 2. 全局系综控制（NVT 或 NPT）
        lmp.command(self._get_ensemble_fix(params, "all", "normal_ensemble"))
        n_pairs = len(self.bond_create_config.pairs)
        normal_step = params.run_step - n_pairs - params.nve_limit_step
        lmp.command(f"run {normal_step}")
        lmp.command("unfix normal_ensemble")

    def _get_react_num(self, lmp) -> dict:
        """获取反应数量"""
        react_nums = {}

        if self.reaction_mode == "bond/create":
            # bond/create: 遍历所有 pairs 提取各自计数
            for i, pair in enumerate(self.bond_create_config.pairs):
                key = f"{pair.itype}-{pair.jtype}"
                try:
                    num = lmp.extract_fix(f"bond_create_fix_{i}", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, nrow=0)
                    react_nums[key] = int(num) if num else 0
                except Exception:
                    react_nums[key] = 0
        else:
            # bond/react 原有逻辑
            for i, rxn in enumerate(self.lammps_params.reactions):
                try:
                    num = lmp.extract_fix("rxns", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR, nrow=i)
                    react_nums[rxn.name] = int(num) if num else 0
                except Exception:
                    react_nums[rxn.name] = 0

        return react_nums

    def _get_reacted_atom_ids(self, created_bonds, coords, box, radius):
        """
        从新建键提取反应原子 + 半径内邻近原子

        策略: 向量化距离矩阵，PBC 最小镜像修正

        Args:
            created_bonds: 新建键列表 (n, 4) [bond_id, type, atom1, atom2]
            coords: 所有原子坐标 (n_atoms, 3)，1-indexed
            box: 盒子边界 (3, 2)
            radius: 搜索半径 (Å)

        Returns:
            list of int: 反应原子 + 邻近原子的 ID 列表
        """
        import numpy as np
        from scipy.spatial.distance import cdist

        # 1. 直接反应原子
        if created_bonds.ndim == 1 or len(created_bonds) == 0:
            return []
        direct_atoms = np.unique(created_bonds[:, 1:3].astype(int).ravel())
        if len(direct_atoms) == 0:
            return []

        # 2. PBC 距离: 反应原子 vs 所有原子（完全向量化）
        center_coords = coords[direct_atoms - 1]     # 0-indexed, (n_center, 3)
        box_size = box[:, 1] - box[:, 0]

        if np.any(box_size > 0):
            # 广播: (n_center, 1, 3) - (1, n_atoms, 3) → (n_center, n_atoms, 3)
            delta = center_coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
            delta -= np.round(delta / box_size) * box_size
            dists = np.sqrt(np.sum(delta ** 2, axis=2))  # (n_center, n_atoms)
        else:
            dists = cdist(center_coords, coords)

        # 3. 半径内所有原子
        mask = np.any(dists < radius, axis=0)

        # 4. 合并直接反应原子和邻近原子
        all_reacted = np.unique(np.concatenate([
            direct_atoms,
            np.where(mask)[0] + 1  # 转回 1-indexed
        ]))

        return all_reacted.tolist()

    def _unwrap_coords(self, coords, bonds, molecule_ids, box):
        """
        展开坐标（处理周期性边界条件）

        Args:
            coords: 原子坐标数组 (n_atoms, 3)
            bonds: 键连表 (n_bonds, 3) [bond_type, atom1, atom2]
            molecule_ids: 分子ID数组 (n_atoms,)
            box: 盒子边界 (3, 2)

        Returns:
            unwrapped_coords: 展开后的坐标数组
        """
        # 提取原子ID部分（bonds格式: [bond_type, atom1, atom2]）
        if bonds.shape[1] == 3:
            bond_atoms = bonds[:, 1:3].astype(np.int32)
        else:
            bond_atoms = bonds.astype(np.int32)

        # 使用Python实现（兼容性更好，Numba版本可在生产环境启用）
        return unwrap_coords_python(coords, bond_atoms, box, molecule_ids)

    def _finalize(self, lmp, f_react_num, output_cg_traj, t1):
        """收尾工作"""
        # 写入最终数据
        lmp.command("write_data final_frame.data pair ij nofix")

        if me == 0:
            # 保存changed_bead_id_list
            with open("changed_bead_id_list.pkl", "wb") as f:
                pickle.dump(self.changed_bead_id_list, f)

            f_react_num.close()

            # 保存最终CG映射
            np.savetxt(
                "final_cg_compare_list.csv",
                self.cg_compare_list.data,
                delimiter=",",
                fmt="%d",
                header="bead_id, mol_id, bead_type, AA_id, mass",
                comments=''
            )

            # 统计
            t2 = time()
            if me == 0:
                print("\n" + "=" * 60)
                print("模拟完成!")
                print("=" * 60)
                print(f"  总运行时间: {t2 - t1:.2f} 秒")
                print(f"  反应统计:")
                for name, num in self.reacted_nums.items():
                    print(f"    {name}: {num}")

                # 保存键连表记录
                if self.bonds_recorder.n_records > 0:
                    paths = self.bonds_recorder.save_all()
                    print(f"  键连表记录: {len(paths)} 个文件")

                # 批量输出反应帧数据
                n_reactions = len(self.reaction_frames['timestep'])
                if n_reactions > 0:
                    print(f"  反应帧数: {n_reactions}")
                    np.savez(
                        'reaction_frames.npz',
                        aa_coords_before=np.array(self.reaction_frames['aa_coords_before']),
                        aa_coords_after=np.array(self.reaction_frames['aa_coords_after']),
                        aa_ids=np.array(self.reaction_frames['aa_ids']),
                        aa_types=np.array(self.reaction_frames['aa_types']),
                        aa_bonds_before=np.array(self.reaction_frames['aa_bonds_before'], dtype=object),
                        aa_bonds_after=np.array(self.reaction_frames['aa_bonds_after'], dtype=object),
                        cg_mapping_before=np.array(self.reaction_frames['cg_mapping_before']),
                        cg_mapping_after=np.array(self.reaction_frames['cg_mapping_after']),
                        cg_bonds_before=np.array(self.reaction_frames['cg_bonds_before'], dtype=object),
                        cg_bonds_after=np.array(self.reaction_frames['cg_bonds_after'], dtype=object),
                        timestep=np.array(self.reaction_frames['timestep']),
                    )
                    print(f"  反应帧数据已保存到: reaction_frames.npz")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="LAMMPS反应模拟后处理 - 重构版")
    parser.add_argument("config_dir", help="配置文件目录")
    parser.add_argument("--test", action="store_true", help="测试模式 (不运行LAMMPS)")
    parser.add_argument("--loop-num", type=int, default=None,
                        help="覆盖循环次数 (用于快速测试)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="冒烟测试模式: 在临时目录中跑少量 loop 并验证输出")

    args = parser.parse_args()

    # 冒烟测试模式: 委托给 SmokeTestHarness
    if args.smoke_test:
        from core.smoke_test_harness import SmokeTestHarness
        harness = SmokeTestHarness(
            args.config_dir,
            loop_num=args.loop_num or 5,
            keep_output=False
        )
        report = harness.run()
        if report is not None:
            report.print()
            sys.exit(0 if report.passed else 1)
        else:
            # 非 rank-0 进程: harness.run() 返回 None
            # 模拟过程中如有错误会通过 MPI 异常传播，此处正常退出
            sys.exit(0)

    # 检查配置目录
    config_dir = Path(args.config_dir)
    if not config_dir.exists():
        if me == 0:
            print(f"错误: 配置目录不存在: {config_dir}")
        sys.exit(1)

    # 创建运行器
    runner = LAMMPSReactionRunner(str(config_dir))

    # 覆盖循环次数
    if args.loop_num is not None:
        runner.lammps_params.loop_num = args.loop_num
        if me == 0:
            print(f"循环次数覆盖为: {args.loop_num}")

    if args.test or lammps is None:
        if me == 0:
            print("\n测试模式: 仅加载配置，不运行模拟")
            print("✅ 配置加载成功!")
        return

    # 运行
    runner.run()


if __name__ == "__main__":
    main()