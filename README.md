# LmpPy - LAMMPS Bond/React 后处理框架

> 版本: 2.4
> 作者: Claude
> 日期: 2026-06-03

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [依赖项与安装](#依赖项与安装)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [主循环执行顺序](#主循环执行顺序)
- [配置文件详解](#配置文件详解)
- [CG映射配置](#cg映射配置)
- [继续计算功能](#继续计算功能)
- [输出文件说明](#输出文件说明)
- [模块API参考](#模块api参考)
- [工具模块API参考](#工具模块-api-参考)
- [工具脚本使用指南](#工具脚本使用指南)
- [IBM势能计算配置文件](#ibm势能计算配置文件)
- [使用教程](#使用教程)
- [常见问题](#常见问题)
- [模块依赖关系](#模块依赖关系)
- [更新日志](#更新日志)

---

## 项目简介

LmpPy 是一个用于 LAMMPS bond/react 模拟的后处理框架，主要功能包括：

### 核心功能

1. **CG轨迹生成** - 实时将全原子轨迹转换为粗粒化轨迹
2. **反应监测** - 检测化学反应事件并统计反应次数
3. **CG映射更新** - 反应后自动更新原子到bead的映射关系
4. **CG拓扑推导** - 从键连信息推导粗粒化键、角、二面角
5. **断点续算** - 支持从上一次运行的最终状态继续计算

### 特性亮点

| 特性 | 说明 |
|------|------|
| **多体系支持** | 通过配置文件定义不同化学反应体系（环氧树脂、聚氨酯、聚丙烯等） |
| **配置驱动** | 所有参数通过 YAML 配置文件定义，无需修改代码 |
| **高性能** | 向量化/Numba 优化关键计算，部分操作加速 30-300 倍 |
| **MPI 并行** | 支持多进程并行运行 |
| **模块化设计** | 清晰的模块划分，便于维护和扩展 |

### 性能优化

| 优化项 | 加速效果 | 说明 |
|--------|----------|------|
| 向量化 image 解码 | **362x** | 8.2ms → 0.023ms |
| Numba find_molecules | **30-80x** | BFS 分子查找 |
| Numba unwrap_coords | **4-9x** | 坐标展开 |
| 惰性索引缓存 | O(n²) → O(n) | CG 坐标转换 |

---

## 项目结构

```
LmpPy/
├── __init__.py                    # 包入口，版本 0.1.0
├── run_refactored.py              # 主入口脚本 (LAMMPSReactionRunner)
├── test_integration.py            # 集成测试
├── core/                          # 核心功能模块
│   ├── __init__.py                # 导出所有核心类和函数
│   ├── config_loader.py           # YAML配置加载器 (ConfigLoader, SystemConfig, LAMMPSParams)
│   ├── mapping_generator.py       # CG映射生成器 (CGCompareList, MappingGenerator)
│   ├── template_parser.py         # LAMMPS模板解析器 (TemplateParser, ReactionTemplate)
│   ├── lammps_data_extractor.py   # LAMMPS数据提取器 (LAMMPSDataExtractor, AtomData, BondData)
│   ├── bond_detector.py           # 键变化检测器 (BondDetector, BondChanges)
│   ├── reaction_locator.py        # 反应位点定位器 (ReactionLocator, ReactionMatch)
│   ├── cg_mapper.py               # CG映射更新器 (CGMapper, CGMapping)
│   ├── cg_converter.py            # CG坐标转换器 (CGConverter)
│   ├── cg_bond_mapper.py          # 粗粒键映射器 (CGBondMapper, CGBond)
│   ├── cg_topology.py             # CG拓扑推导器 (CGTopology)
│   ├── bonds_recorder.py          # 键连表记录器 (BondsRecorder, BondRecord)
│   └── cg_initializer.py          # CG系统初始化器 (CGInitializer, CGSystem)
├── utils/                         # 工具函数
│   ├── __init__.py
│   ├── coordinate_utils.py        # 坐标处理 (wrap_coordinates, pbc_distance, unwrap_coords)
│   ├── file_utils.py              # 文件I/O (write_lammps_dump_file, read_lammps_dump_file)
│   ├── graph_utils.py             # 图论算法 (find_molecules - Numba优化)
│   ├── topology.py                # 拓扑文件读写 (TopologyData)
│   └── units.py                   # 单位转换 (UnitConverter)
├── tools/                         # 工具模块集合
│   ├── __init__.py
│   ├── aa2cg/                     # 全原子到粗粒化转换
│   │   ├── data_converter.py      # LAMMPS data转换
│   │   ├── trj_converter.py       # 轨迹转换
│   │   └── mapping_utils.py       # 映射工具函数
│   ├── ibm_potential/             # IBM势能计算
│   │   ├── config.py              # 配置加载器
│   │   ├── distribution.py        # 分布计算
│   │   ├── boltzmann.py           # 玻尔兹曼反演
│   │   └── lammps_table.py        # LAMMPS表生成
│   └── smooth_utils/              # 分布平滑工具包
│       ├── core.py                # 核心平滑算法
│       ├── cli.py                 # 命令行入口
│       ├── constants.py           # 常量定义
│       ├── io.py                  # 文件I/O
│       ├── preprocess.py          # 预处理
│       ├── peaks.py               # 峰检测
│       ├── zones.py               # 区域分类
│       ├── quality.py             # 质量评估
│       ├── optimize.py            # 参数优化
│       ├── angle_dihedral.py      # 角度/二面角平滑
│       ├── report.py              # 报告生成
│       ├── dist_config.py         # 分布配置
│       └── dist_plot.py           # 分布绘图
├── scripts/                       # CLI脚本
│   ├── build_cg_system.py         # 构建CG体系LAMMPS data文件
│   ├── convert_aa2cg.py           # AA→CG转换CLI
│   ├── generate_initial_mapping.py # 生成初始CG映射
│   ├── smooth_distribution.py     # 分布平滑CLI
│   ├── calc_ibm_potential.py      # IBM势能计算CLI
│   ├── calc_dist.py               # 分布计算CLI (VOTCA格式)
│   ├── plot_dist.py               # 分布绘图CLI
│   ├── yaml2csv_mapping.py        # YAML→CSV映射转换
│   └── data2gro.py                # LAMMPS data转GRO文件
├── config/                        # 配置文件目录
│   ├── mapping/                   # CG映射配置模板
│   └── reactions/                 # 反应模板配置
├── docs/                          # 文档和示例
│   └── examples/                  # 示例配置文件
└── README.md                      # 本文档
```

---

## 依赖项与安装

### 必需依赖

```bash
# Python >= 3.8
pip install numpy pandas pyyaml scipy
```

### 可选依赖

```bash
# Numba JIT加速 (强烈推荐)
pip install numba

# LAMMPS Python接口 (运行模拟必需)
# 需要编译 LAMMPS 并启用 PYTHON 包

# MPI支持 (并行运行必需)
pip install mpi4py
```

### 完整安装

```bash
pip install numpy pandas pyyaml scipy numba mpi4py
```

### LAMMPS 编译

需要编译支持 Python 接口的 LAMMPS：

```bash
# 编译示例
cd lammps/src
make yes-python yes-mpi
make mpi
```

---

## 快速开始

### 1. 准备配置文件

创建一个配置目录，包含以下文件：

```
my_system/
├── system.yaml              # 体系配置
├── lammps_params.yaml       # LAMMPS运行参数
├── molecule1_mapping.yaml   # 分子CG映射配置
├── molecule2_mapping.yaml   # (可选) 多种分子
├── system.data              # LAMMPS初始数据文件
├── rxn1_pre.lammpstemplate  # 反应前模板
├── rxn1_post.lammpstemplate # 反应后模板
├── rxn1.map                 # bond/react map文件
├── rxn1_pre_mapping.yaml    # 反应前CG映射
└── rxn1_post_mapping.yaml   # 反应后CG映射
```

### 2. 运行模拟

```bash
# 单进程运行
python -m LmpPy.run_refactored my_system/

# MPI并行运行 (4进程)
mpirun -np 4 python -m LmpPy.run_refactored my_system/

# 指定循环次数
mpirun -np 4 python -m LmpPy.run_refactored my_system/ --loop-num 100
```

### 3. 测试模式

```bash
# 不运行LAMMPS，仅测试配置加载
python -m LmpPy.run_refactored my_system/ --test
```

---

## 运行方式

### 命令行参数

```bash
python -m LmpPy.run_refactored <config_dir> [options]

参数:
  config_dir          配置文件目录路径

选项:
  --test              测试模式，仅加载配置不运行LAMMPS
  --loop-num N        覆盖配置文件中的循环次数
```

### MPI 并行运行

```bash
# 4进程并行
mpirun -np 4 python -m LmpPy.run_refactored config/

# 8进程并行
mpirun -np 8 python -m LmpPy.run_refactored config/
```

### 环境变量

```bash
# 设置OpenMP线程数 (默认为1)
export OMP_NUM_THREADS=1
```

---

## 主循环执行顺序

LmpPy 模拟的主循环 (`run_refactored.py`) 每个周期按以下顺序执行：

| 步骤 | 描述 | 涉及模块 |
|------|------|----------|
| Step 1 | **运行 bond/react** | LAMMPS `fix bond/react` |
| Step 2 | **检测反应** | `BondDetector` → `ReactionLocator` → `CGMapper` → `CGConverter` → 写入轨迹帧 |
| Step 3 | **弛豫** | NVE/limit + NVT 反应原子弛豫，然后 NPT 全原子 |
| Step 4 | **更新坐标/image flags** | `LAMMPSDataExtractor` |

**Step 2 详细流程**：

```
1. BondDetector: 检测键变化 (created_bonds, deleted_bonds)
2. ReactionLocator: 反应模板匹配 (pre-before + post-after 双重验证)
3. CGMapper: 更新 CG 映射 (基于 ReactionMatch)
4. CGConverter: 转换 CG 坐标 (惰性索引缓存)
5. 写入 CG 轨迹帧 (post-reaction, pre-relaxation)
6. 缓存反应帧数据到 reaction_frames.npz
```

---

## 配置文件详解

### 1. system.yaml - 体系配置

定义体系的名称、bead类型和分子映射文件。

```yaml
system:
  name: "EPR polymerization"    # 体系名称

  # Bead类型名称到编号的映射
  bead_type_names:
    Bead1: 3    # 名称为Bead1，类型编号为3
    Bead2: 3
    Bead3: 5
    Bead4: 3
    Bead5: 4
    Bead6: 5
    Bead7: 1
    Bead8: 2

  # 分子映射文件列表
  mapping_files:
    - path: "dimer_EE_initiator_mapping.yaml"
      copies: 25    # 该类型分子数量
    - path: "dimer_EP_initiator_mapping.yaml"
      copies: 25
    - path: "ethylene_mapping.yaml"
      copies: 1000
    - path: "propylene_mapping.yaml"
      copies: 1000
```

**关键说明：**

- `bead_type_names`: 定义所有可能的 bead 类型名称及其编号
- `mapping_files`: 按照数据文件中分子的出现顺序列出
- `copies`: 该类型分子在初始数据文件中的数量
- 映射文件使用 `anchor=0`，程序会自动累积偏移量

### 2. lammps_params.yaml - LAMMPS运行参数

```yaml
# ============================================
# 模拟基本参数
# ============================================
simulation:
  loop_num: 5              # 主循环次数
  dt: 0.001                # 时间步长 (ps)
  timestep: 1              # LAMMPS timestep 设置 (fs)
  temperature: 400         # 目标温度 (K)
  pressure: 1.0            # 目标压力 (atm)
  ensemble: "nvt"          # 系综类型: "npt" 或 "nvt"

# ============================================
# 步数配置
# ============================================
steps:
  bond_react_check: 1      # 键反应检查步数
  run_per_loop: 100        # 每个loop运行的总步数
  nve_limit: 50            # NVE/limit步数 (反应原子弛豫)

  # 注意: run_per_loop > nve_limit + bond_react_check
  # normal_npt_step = run_per_loop - nve_limit - bond_react_check

# ============================================
# NPT控温控压参数
# ============================================
npt:
  tcouple: 100             # 温度耦合常数 (fs)
  pcouple: 1000            # 压力耦合常数 (fs)

# ============================================
# Bond/React配置
# ============================================
bond_react:
  stabilization: 0.05      # stabilization参数 (Å)

  reactions:
    - name: "rxn1"
      cutoff: 3.0                        # 反应截断半径 (Å)
      map_file: "rxn1.map"               # bond/react map文件
      pre_mol: "mol1"                    # 反应前分子模板名
      post_mol: "mol2"                   # 反应后分子模板名
      pre_template: "rxn1_pre.lammpstemplate"
      post_template: "rxn1_post.lammpstemplate"
      pre_mapping: "rxn1_pre_mapping.yaml"    # 反应前CG映射
      post_mapping: "rxn1_post_mapping.yaml"  # 反应后CG映射

# ============================================
# 分子模板配置
# ============================================
molecules:
  mol1: "rxn1_pre.lammpstemplate"
  mol2: "rxn1_post.lammpstemplate"

# ============================================
# 输入输出文件配置
# ============================================
files:
  # 输入文件
  data_file: "modify_EPR_poly.data"
  initial_cg_mapping: "final_cg_compare_list.csv"  # CG映射文件(支持加载已有映射)

  # read_data 额外参数
  read_data_extra:
    special_per_atom: 6
    bond_per_atom: 5
    angle_per_atom: 9
    dihedral_per_atom: 8

  # 输出文件
  output_cg_trajectory: "cg_trajectory.lammpstrj"
  output_reaction_count: "reaction_num.txt"
  output_final_data: "final_frame.data"
  output_final_mapping: "final_cg_compare_list.csv"

  # CG 拓扑输出
  output_cg_bonds: "cg_bonds.txt"
  output_cg_angles: "cg_angles.txt"
  output_cg_dihedrals: "cg_dihedrals.txt"
```

**参数说明：**

| 参数 | 类型 | 说明 |
|------|------|------|
| `loop_num` | int | 主循环次数 |
| `ensemble` | str | "nvt" 仅控温，"npt" 控温控压 |
| `run_per_loop` | int | 每循环运行的MD步数 |
| `nve_limit` | int | 反应后NVE弛豫步数 |
| `cutoff` | float | bond/react 反应截断半径 |
| `stabilization` | float | bond/react 稳定化参数 |

### 3. read_data_extra 参数

用于 `read_data` 命令的额外参数，预分配内存：

```yaml
read_data_extra:
  special_per_atom: 6    # 每原子特殊邻居数
  bond_per_atom: 5       # 每原子键数
  angle_per_atom: 9      # 每原子角数
  dihedral_per_atom: 8   # 每原子二面角数
```

---

## CG映射配置

### 映射文件格式

每个分子的CG映射配置文件定义原子如何映射到bead：

```yaml
# ============================================
# Site类型定义
# ============================================
site-types:
  Bead1:
    index: [0, 1, 4, 5, 6, 7]              # 相对原子索引 (0-based)
    x-weight: [12, 12, 1, 1, 1, 1]         # 各原子质量权重
  Bead2:
    index: [0, 1, 6, 7, 8, 10]
    x-weight: [12, 12, 1, 1, 1, 1]
  Bead3:
    index: [0, 2, 3, 4]
    x-weight: [12, 1, 1, 1]

# ============================================
# 映射配置
# ============================================
config:
  - anchor: 0            # 起始偏移 (通常为0)
    repeat: 25           # 分子数量
    offset: 16           # 每分子原子数
    sites:
      - [Bead1, 0]       # [site类型名, 起始偏移]
      - [Bead2, 2]
      - [Bead3, 11]
```

**字段说明：**

| 字段 | 说明 |
|------|------|
| `site-types` | 定义所有bead类型及其包含的原子 |
| `index` | 相对于site起始位置的原子索引 (0-based) |
| `x-weight` | 质心计算的权重 (通常为原子质量) |
| `anchor` | **固定为0**，程序自动累积偏移 |
| `repeat` | 该类型分子的数量 |
| `offset` | 每个分子的原子数 |
| `sites` | 分子中的bead列表 |

### 映射逻辑

程序按 `system.yaml` 中 `mapping_files` 的顺序处理映射文件：

```
第1个映射文件: aa_id_offset=0, anchor=0
第2个映射文件: aa_id_offset=第1个文件的原子总数, anchor=0
第3个映射文件: aa_id_offset=前两个文件的原子总数, anchor=0
...
```

**示例：**

```
dimer_EE (25分子×16原子=400): anchor=0, offset=16, repeat=25
dimer_EP (25分子×19原子=475): anchor=0, offset=19, repeat=25
ethylene (1000分子×6原子=6000): anchor=0, offset=6, repeat=1000
propylene (1000分子×9原子=9000): anchor=0, offset=9, repeat=1000

总原子数: 400+475+6000+9000 = 15875
```

### 反应映射文件

反应前后的CG映射文件格式相同，用于反应后更新CG映射：

```yaml
# rxn1_pre_mapping.yaml - 反应前
mapping:
  1:                          # bead编号
    atoms: [2, 3, 6, 7, 8, 9] # 原子ID列表
    bead_type: 3              # bead类型
  2:
    atoms: [4, 5, 10, 11, 12, 13]
    bead_type: 3
  3:
    atoms: [14, 15, 16, 17, 18, 19]
    bead_type: 1
```

---

## 继续计算功能

程序支持从上一次运行的最终状态继续计算。

### 使用方法

1. **修改 `lammps_params.yaml`**：

```yaml
files:
  # 使用上一次的输出作为输入
  data_file: "final_frame.data"
  initial_cg_mapping: "final_cg_compare_list.csv"
```

2. **运行继续计算**：

```bash
mpirun -np 4 python -m LmpPy.run_refactored continue_config/
```

### 工作原理

- 程序检测 `initial_cg_mapping` 指定的文件是否存在
- 如果存在，直接加载已有CG映射（跳过生成步骤）
- 如果不存在，从 `system.yaml` 的 `mapping_files` 生成映射

### 完整示例

```bash
# 第一次运行
cd simulation/
mpirun -np 4 python -m LmpPy.run_refactored . --loop-num 100

# 准备继续计算
mkdir continue/
cd continue/
ln -s ../final_frame.data .
ln -s ../final_cg_compare_list.csv .
ln -s ../*.lammpstemplate .
ln -s ../*.map .
ln -s ../system.yaml .

# 创建新的 lammps_params.yaml
cat > lammps_params.yaml << EOF
simulation:
  loop_num: 50
  ...
files:
  data_file: "final_frame.data"
  initial_cg_mapping: "final_cg_compare_list.csv"
  ...
EOF

# 继续运行
mpirun -np 4 python -m LmpPy.run_refactored .
```

---

## 输出文件说明

输出文件按产生来源分为以下几类。

### 1. 主模拟管线输出文件

由 `run_refactored.py` 直接产生的文件。

| 文件 | 格式 | 说明 | 产生条件 |
|------|------|------|----------|
| `{output_cg_trajectory}`<br>(默认 `cg_trajectory.lammpstrj`) | LAMMPS dump | CG 轨迹文件，可 VMD 可视化 | 每次主循环 |
| `{output_reaction_count}`<br>(默认 `reaction_num.txt`) | 文本 | 每步反应计数统计 | 每次有反应时 |
| `final_frame.data` | LAMMPS data | 最终帧原子数据 (LAMMPS `write_data` 输出) | 模拟结束 |
| `final_cg_compare_list.csv` | CSV (5 列) | 最终 CG 映射关系 | 模拟结束 |
| `reaction_frames.npz` | NPZ (7 个数组) | 反应帧详细数据，供 SOAP 计算使用 | 有反应时 |

### 2. CG 拓扑文件

由 `CGTopology.to_files()` 产生，文件名由 `lammps_params.yaml` 中 `output_cg_bonds` 等配置指定。

| 文件 | 格式 | 说明 |
|------|------|------|
| `{prefix}_bonds.txt` | 空格分隔整数 (3 列) | CG 键：`[bond_type, bead1, bead2]` |
| `{prefix}_angles.txt` | 空格分隔整数 (4 列) | CG 角度：`[angle_type, bead1, bead2, bead3]` |
| `{prefix}_dihedrals.txt` | 空格分隔整数 (5 列) | CG 二面角：`[dihedral_type, bead1, bead2, bead3, bead4]` |

### 3. 关键输出文件详解

#### reaction_num.txt 格式

```
# timestep rxn1 rxn2 rxn3 ...
1 0 0 0
101 1 0 0
201 2 1 0
...
```

#### final_cg_compare_list.csv 格式

```csv
bead_id,mol_id,bead_type,AA_id,mass
1,1,3,1,12
1,1,3,2,12
1,1,3,3,12
...
```

**列说明：**

| 列 | 说明 |
|----|------|
| `bead_id` | Bead 编号 |
| `mol_id` | 分子 ID |
| `bead_type` | Bead 类型编号 |
| `AA_id` | 原子 ID |
| `mass` | 原子质量 |

#### reaction_frames.npz 内容

这是 **SOAP 计算模块的核心输入数据源**，包含每次反应发生时的完整快照。

```python
import numpy as np
data = np.load('reaction_frames.npz', allow_pickle=True)

# 7 个数组（精简格式，v2.4+）
data['aa_coords_before']     # (n_reactions, n_atoms, 5) 反应前 AA 坐标 (含 image_flags)
data['aa_coords_after']      # (n_reactions, n_atoms, 5) 反应后 AA 坐标
data['aa_bonds_before']      # (n_reactions,) object     反应前原子键
data['aa_bonds_after']       # (n_reactions,) object     反应后原子键
data['cg_mapping_before']    # (n_reactions, n_atoms, 5) 反应前 CG 映射
data['cg_mapping_after']     # (n_reactions, n_atoms, 5) 反应后 CG 映射
data['timestep']             # (n_reactions,)            反应时间步

# 向后兼容：旧格式 npz 可能仍包含 aa_ids, aa_types, cg_bonds_before/after
```

**推导规则**（v2.4+ 精简格式下）：
- `aa_ids` → 从 `cg_mapping[:, 3]` 获取 (AA_id 列)
- `aa_types` → 运行时不变，从初始状态获取
- `cg_bonds_before/after` → `atom_bonds_to_cg_bonds(aa_bonds, cg_mapping)`

**CG 键格式**: `[bond_type, bead1_id, bead2_id]`

**使用示例**（SOAP 计算模块中）：
```python
# 反应对 = cg_bonds_after - cg_bonds_before（集合差）
# cg_bonds 可由 aa_bonds + cg_mapping 推导（后处理自动完成）
# 这是 SOAP_calc_and_Feature_select 模块识别正样本的依据
```

#### bonds_record_*.npz 内容（历史兼容）

```python
import numpy as np
data = np.load('bonds_records/bonds_record_100_0.npz', allow_pickle=True)

data['timestep']        # int, 反应时间步
data['run_step']        # int, 循环中的步数
data['bonds_before']    # (n_bonds, 3) [bond_type, atom1, atom2]
data['bonds_after']     # (n_bonds, 3)
data['reaction_type']   # str, 反应类型名称 (如 "rxn1")
```

#### cg_topology_record_*.npz 内容

```python
import numpy as np
data = np.load('bonds_records/cg_topology_record_100_0.npz', allow_pickle=True)

data['timestep']                # int
data['run_step']                # int
data['cg_bonds_before']         # (n_bonds, 3) [bond_type, bead1, bead2]
data['cg_bonds_after']          # (n_bonds, 3)
data['cg_angles_before']        # (n_angles, 4) [angle_type, bead1, bead2, bead3]
data['cg_angles_after']         # (n_angles, 4)
data['cg_dihedrals_before']     # (n_dihedrals, 5) [dihedral_type, bead1, bead2, bead3, bead4]
data['cg_dihedrals_after']      # (n_dihedrals, 5)
data['reaction_type']           # str
```

### 5. 工具模块输出文件

#### AA2CG 工具 (`tools/aa2cg/`, `scripts/convert_aa2cg.py`)

| 文件 | 格式 | 说明 |
|------|------|------|
| `{output}.pkl` | Pickle | CG 轨迹 (帧列表) |
| `{output}.xyz` | XYZ | CG 轨迹 (XYZ 格式) |
| `{output}.data` | LAMMPS data | CG 体系 LAMMPS 数据文件 |
| `{prefix}_bonds.txt` | 文本 | CG 键拓扑 |
| `{prefix}_angles.txt` | 文本 | CG 角度拓扑 |
| `{prefix}_dihedrals.txt` | 文本 | CG 二面角拓扑 |

#### IBM 势能工具 (`tools/ibm_potential/`, `scripts/calc_ibm_potential.py`)

| 文件 | 格式 | 说明 |
|------|------|------|
| `{distributions_dir}/bond_type{N}_dist.txt` | 2 列文本 | 键长分布 P(r) |
| `{distributions_dir}/angle_type{N}_dist.txt` | 2 列文本 | 角度分布 P(θ) |
| `{distributions_dir}/dihedral_type{N}_dist.txt` | 2 列文本 | 二面角分布 P(φ) |
| `{distributions_dir}/rdf_type{A}_{B}.txt` | 2 列文本 | 径向分布函数 g(r) |
| `{potentials_dir}/bond_type{N}_potential.txt` | 2 列文本 | 键长势函数 U(r) |
| `{potentials_dir}/angle_type{N}_potential.txt` | 2 列文本 | 角度势函数 U(θ) |
| `{potentials_dir}/dihedral_type{N}_potential.txt` | 2 列文本 | 二面角势函数 U(φ) |
| `{potentials_dir}/pair_type{A}_{B}_potential.txt` | 2 列文本 | 对势函数 U(r) |
| `{table_dir}/pair_table.txt` | LAMMPS table | LAMMPS 对势表文件 |
| `{table_dir}/bond_table.txt` | LAMMPS table | LAMMPS 键势表文件 |
| `{table_dir}/angle_table.txt` | LAMMPS table | LAMMPS 角度表文件 |
| `{table_dir}/dihedral_table.txt` | LAMMPS table | LAMMPS 二面角表文件 |
| `{table_dir}/table_reference.txt` | 文本 | 势表类型参考 |

#### 分布平滑工具 (`tools/smooth_utils/`, `scripts/smooth_distribution.py`)

| 文件 | 格式 | 说明 |
|------|------|------|
| `{output_dir}/{basename}_dist.txt` | 2 列文本 | 平滑后的分布 |
| `{output_dir}/{basename}_comparison.png` | PNG | 原始/平滑/力曲线三面对比图 |
| `{output_dir}/{basename}_report.txt` | 文本 | 平滑质量评估报告 |

#### 分布计算/绘图脚本 (`scripts/calc_dist.py`, `scripts/plot_dist.py`)

| 文件 | 格式 | 说明 |
|------|------|------|
| `{output_dir}/bond_type{N}.dist.tgt` | VOTCA 格式 | VOTCA 目标分布文件 |
| `{output_dir}/angle_type{N}.dist.tgt` | VOTCA 格式 | VOTCA 角度目标文件 |
| `{output_dir}/dihedral_type{N}.dist.tgt` | VOTCA 格式 | VOTCA 二面角目标文件 |
| `{output_dir}/pair_type{A}_{B}.dist.tgt` | VOTCA 格式 | VOTCA 对目标文件 |
| `{output}/*.png` | PNG | 分布汇总图 |

#### 其他脚本输出

| 脚本 | 输出文件 | 说明 |
|------|----------|------|
| `yaml2csv_mapping.py` | `{output_csv}` (默认 `AtomId_BeadId_compare_list.csv`) | YAML 映射转 CSV |
| `generate_initial_mapping.py` | 配置文件指定的 `initial_cg_mapping` | 初始 CG 映射 |
| `build_cg_system.py` | `{output}.data` | CG 体系 LAMMPS data 文件 |
| `data2gro.py` | `{output}.gro` | LAMMPS data 转 GRO 文件 |

---

## 模块API参考

### 导入模块

```python
from LmpPy.core import (
    # 配置
    ConfigLoader, SystemConfig, LAMMPSParams,
    # 映射
    MappingGenerator, CGCompareList, generate_cg_compare_list,
    # 模板
    TemplateParser, ReactionTemplate, load_all_reaction_templates,
    # 数据提取
    LAMMPSDataExtractor, AtomData, BondData,
    # 反应处理
    BondDetector, BondChanges,
    ReactionLocator, ReactionMatch,
    CGMapper, CGMapping,
    CGConverter, lammpstrj2cg,
    CGBondMapper, atom_bonds_to_cg_bonds,
    BondsRecorder, save_bonds_record,  # 向后兼容：新模拟不再生成 bonds_records
    # CG系统初始化
    CGInitializer, CGSystem, initialize_cg_system,
    # CG拓扑
    CGTopology, derive_cg_topology_from_bonds,
)
```

### ConfigLoader - 配置加载器

```python
from LmpPy.core import ConfigLoader

loader = ConfigLoader("config/")

# 加载各类配置
system_config = loader.load_system_config()
lammps_params = loader.load_lammps_params()
mass_list = loader.load_mass_list()
mapping_config = loader.load_mapping_config("molecule1_mapping.yaml")
```

### CGCompareList - CG映射数据结构

```python
from LmpPy.core import CGCompareList

# 从CSV加载
cg_list = CGCompareList.from_csv("final_cg_compare_list.csv")

# 保存到CSV
cg_list.to_csv("output.csv")

# 访问数据
print(f"原子数: {len(cg_list.data)}")
print(f"Bead数: {cg_list.n_beads}")
print(f"分子数: {cg_list.n_molecules}")
```

### LAMMPSDataExtractor - 数据提取器

```python
from LmpPy.core import LAMMPSDataExtractor

extractor = LAMMPSDataExtractor(use_cache=True)

# 从LAMMPS对象提取数据
atom_data = extractor.extract_atoms(lmp)
bond_data = extractor.extract_bonds(lmp)

# 提取所有数据
atom_data, bond_data = extractor.extract_all(lmp)

# 使用后使缓存失效
extractor.invalidate_cache()
```

### BondDetector - 键变化检测

```python
from LmpPy.core import BondDetector, compare_two_bonds

detector = BondDetector(n_atoms=10000)

# 检测键变化
changes = detector.detect(bonds_before, bonds_after)

if changes.has_changes:
    print(f"新增键: {len(changes.created_bonds)}")
    print(f"断裂键: {len(changes.deleted_bonds)}")

# 或使用便捷函数
# 注意: 返回顺序为 (仅在bonds1中的键, 仅在bonds2中的键)
deleted_bonds, created_bonds = compare_two_bonds(bonds_before, bonds_after)
```

### ReactionLocator - 反应位点定位

```python
from LmpPy.core import ReactionLocator, locate_reactions

locator = ReactionLocator(
    templates=reaction_templates,  # List[ReactionTemplate]
    n_atoms=n_atoms
)

# 定位反应
matches = locator.locate(
    bonds_before=bonds_before,
    bonds_after=bonds_after,
    atom_ids=atom_ids,
    atom_types=atom_types
)

# 或使用便捷函数
matches = locate_reactions(
    templates=reaction_templates,
    bonds_before=bonds_before,
    bonds_after=bonds_after,
    atom_ids=atom_ids,
    atom_types=atom_types,
    n_atoms=n_atoms
)
```

### CGConverter - CG坐标转换

```python
from LmpPy.core import CGConverter, lammpstrj2cg

converter = CGConverter()

# 转换坐标
cg_coords, cg_ixyz = converter.convert(
    atom_coords,       # (n_atoms, 5) 原子坐标
    cg_compare_list,   # CG映射
    mass_list          # 质量列表
)

# 或使用便捷函数
cg_coords = lammpstrj2cg(atom_coords, cg_compare_list, mass_list)

# CG映射变化后使缓存失效
converter.invalidate_cache()
```

### atom_bonds_to_cg_bonds - CG键转换

```python
from LmpPy.core import atom_bonds_to_cg_bonds

# 将原子键转换为CG键
# 注意: cg_mapping_data 是从 CGCompareList 提取的 [bead_id, bead_type] 二维数组
# 不是完整的 5 列 compare_list
cg_mapping_data = cg_list.get_mapping_array()  # (n_atoms, 2) [bead_id, bead_type]
cg_bonds = atom_bonds_to_cg_bonds(
    atom_bonds,        # 原子键列表 [(atom1, atom2), ...]
    cg_mapping_data    # (n_atoms, 2) 数组
)
```

### CGInitializer - CG系统初始化

```python
from LmpPy.core import CGInitializer, initialize_cg_system

# 从完整体系配置一次性初始化CG系统
cg_system = initialize_cg_system(
    system_config,       # SystemConfig
    lammps_data_file,    # .data 文件路径
    mass_list            # 质量列表
)

print(f"Bead数: {cg_system.n_beads}")
print(f"CG键数: {len(cg_system.bonds)}")
```

### CGTopology - CG拓扑推导

```python
from LmpPy.core import CGTopology, derive_cg_topology_from_bonds

# 从 CG 键推导完整拓扑
cg_topology = derive_cg_topology_from_bonds(cg_bonds)

# 拓扑属性
print(f"键数: {len(cg_topology.bonds)}")
print(f"角数: {len(cg_topology.angles)}")
print(f"二面角数: {len(cg_topology.dihedrals)}")

# 写入文件
cg_topology.to_files(
    bonds_file="cg_bonds.txt",
    angles_file="cg_angles.txt",
    dihedrals_file="cg_dihedrals.txt"
)
```

---

## 工具模块 API 参考

### 导入工具模块

```python
# AA→CG转换工具
from LmpPy.tools.aa2cg import (
    load_aa_to_cg_mapping,
    convert_aa_to_cg_frame,
    read_lammps_data,
    write_cg_data_file,
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
)

# IBM势能计算工具
from LmpPy.tools.ibm_potential import (
    load_ibm_config,
    load_cg_trajectory,
    calculate_bond_distribution,
    calculate_bond_potential,
    create_lammps_table_files,
)

# 分布平滑工具
from LmpPy.tools.smooth_utils import (
    smooth_distribution,
    smooth_bond_with_harmonic_boundary,
    smooth_dihedral_periodic,
)

# 工具函数
from LmpPy.utils import (
    read_topology_files,
    convert_energy,
    UnitConverter,
    find_molecules,
    wrap_coordinates,
    pbc_distance,
)
```

### AA→CG转换工具

```python
from LmpPy.tools.aa2cg import (
    load_aa_to_cg_mapping,
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
)

# 加载CG映射
mapping = load_aa_to_cg_mapping("AtomId_BeadId_compare_list.csv")

# 读取GROMACS轨迹
aa_frames = read_gromacs_trr_all_frames("topol.tpr", "traj.trr")

# 转换为CG轨迹
cg_traj = convert_trajectory_to_cg(aa_frames, "mapping.csv")

# 保存为pickle
save_cg_trajectory_pickle(cg_traj, "cg_trajectory.pkl")
```

### IBM势能计算工具

```python
from LmpPy.tools.ibm_potential import (
    load_ibm_config,
    load_cg_trajectory,
    calculate_bond_distribution,
    calculate_bond_potential,
    create_lammps_table_files,
)

# 加载配置
config = load_ibm_config("ibm_potential.yaml")

# 加载CG轨迹
cg_data = load_cg_trajectory(config.trajectory.path)

# 计算键分布
r, hist = calculate_bond_distribution(cg_data, bond_pairs)

# 玻尔兹曼反演
r, U = calculate_bond_potential(r, hist, temperature=400)

# 生成LAMMPS table文件
create_lammps_table_files("potentials_output", "lammps_tables")
```

### 分布平滑工具

```python
from LmpPy.tools.smooth_utils import smooth_distribution

# 平滑单个分布文件
smoothed_path, result = smooth_distribution(
    "bond_type1_dist.txt",
    output_dir="smoothed_output",
    temperature=400
)
```

### 工具函数

```python
from LmpPy.utils import read_topology_files, UnitConverter, find_molecules

# 读取拓扑文件
topo = read_topology_files("cg_bonds.txt", "cg_angles.txt", "cg_dihedrals.txt")

# 单位转换
converter = UnitConverter()
kt = converter.kt(400)  # kT at 400 K
energy_kJ = converter.convert_energy(1.0, 'kcal/mol', 'kJ/mol')

# BFS分子查找 (Numba优化)
molecule_ids = find_molecules(bonds, natoms)
```

---

## 使用教程

### 教程1: 创建新体系

1. **准备LAMMPS数据文件**

确保数据文件包含正确的原子、键、角、二面角定义。

2. **创建 system.yaml**

```yaml
system:
  name: "My Polymer System"
  bead_type_names:
    Head: 1
    Mid: 2
    End: 3
  mapping_files:
    - path: "monomer_mapping.yaml"
      copies: 100
```

3. **创建分子映射文件**

根据分子结构定义原子到bead的映射：

```yaml
site-types:
  Head:
    index: [0, 1, 2, 3]
    x-weight: [12, 12, 1, 1]

config:
  - anchor: 0
    repeat: 100
    offset: 10
    sites:
      - [Head, 0]
```

4. **创建 lammps_params.yaml**

复制模板并根据需要修改参数。

5. **运行测试**

```bash
python -m LmpPy.run_refactored my_system/ --test
```

### 教程2: 添加新反应

1. **准备反应模板文件**

- `rxn_pre.lammpstemplate`: 反应前分子构型
- `rxn_post.lammpstemplate`: 反应后分子构型
- `rxn.map`: bond/react 映射文件

2. **创建反应CG映射文件**

```yaml
# rxn_pre_mapping.yaml
mapping:
  1:
    atoms: [1, 2, 3, 4, 5, 6]
    bead_type: 1
```

3. **在 lammps_params.yaml 中添加反应配置**

```yaml
bond_react:
  reactions:
    - name: "my_rxn"
      cutoff: 3.0
      map_file: "rxn.map"
      pre_mol: "mol_pre"
      post_mol: "mol_post"
      pre_template: "rxn_pre.lammpstemplate"
      post_template: "rxn_post.lammpstemplate"
      pre_mapping: "rxn_pre_mapping.yaml"
      post_mapping: "rxn_post_mapping.yaml"
```

### 教程3: 分析输出结果

```python
import numpy as np
import pandas as pd

# 读取反应计数
reaction_df = pd.read_csv('reaction_num.txt', sep=' ', comment='#',
                          names=['timestep', 'rxn1', 'rxn2'])
print(f"总反应次数: {reaction_df['rxn1'].sum()}")

# 读取CG映射
cg_mapping = pd.read_csv('final_cg_compare_list.csv')
print(f"Bead数量: {cg_mapping['bead_id'].nunique()}")

# 读取反应帧数据
reaction_frames = np.load('reaction_frames.npz', allow_pickle=True)
print(f"记录的反应次数: {len(reaction_frames['timestep'])}")

# 从 aa_bonds 推导 CG 键（v2.4+ 精简格式）
from LmpPy.core import atom_bonds_to_cg_bonds
aa_bonds = reaction_frames['aa_bonds_after'][0]
cg_mapping = reaction_frames['cg_mapping_after'][0]
cg_bonds = atom_bonds_to_cg_bonds(aa_bonds, cg_mapping)
print(f"CG键数: {len(cg_bonds)}")
```

---

## 常见问题

### Q1: IndexError: index X is out of bounds

**原因**: CG映射配置错误，原子ID超出范围。

**解决**: 检查 `mapping_files` 中的 `offset` 和 `repeat` 是否正确，确保所有 `anchor` 为 0。

### Q2: Bond atoms missing on proc

**原因**: 原子跑出盒子边界或键定义问题。

**解决**:
- 检查初始数据文件的键定义
- 减小时间步长
- 增加盒子尺寸

### Q3: 反应不发生

**可能原因**:
- 反应截断半径 `cutoff` 太小
- 分子初始位置距离太远
- 反应模板不匹配

**解决**:
- 增大 `cutoff` 值
- 检查反应模板与实际分子的匹配
- 延长模拟时间

### Q4: Tab character YAML parse error

**原因**: YAML 文件包含 Tab 字符。

**解决**:
```bash
# 将 Tab 替换为空格
sed -i 's/\t/  /g' *.yaml
```

### Q5: MPI 进程通信错误

**原因**: MPI 环境配置问题。

**解决**:
- 确保所有进程可访问相同文件系统
- 检查 MPI 安装和环境变量

---

## 工具脚本使用指南

### 分布平滑工具

```bash
# 处理单个分布文件
python LmpPy/scripts/smooth_distribution.py bond_type1_dist.txt -o smoothed_output/

# 处理目录下所有分布文件
python LmpPy/scripts/smooth_distribution.py -d distributions/ -o smoothed_output/

# 指定温度
python LmpPy/scripts/smooth_distribution.py bond_type1_dist.txt -T 300 -o smoothed_output/
```

### 分布计算工具

```bash
# 计算分布
python LmpPy/scripts/calc_dist.py -c ibm_potential.yaml

# 绘制分布图
python LmpPy/scripts/plot_dist.py dist_file.txt -o output.png
```

### AA→CG转换工具

```bash
# 轨迹转换 (GROMACS TRR)
python LmpPy/scripts/convert_aa2cg.py trj \
    --tpr topol.tpr \
    --trr traj.trr \
    --mapping mapping.csv \
    -o cg_trajectory.pkl

# Data文件转换
python LmpPy/scripts/convert_aa2cg.py data \
    --input system.data \
    --mapping mapping.csv \
    --bonds cg_bonds.txt \
    -o cg.data
```

### IBM势能计算工具

```bash
# 使用配置文件
python LmpPy/scripts/calc_ibm_potential.py -c ibm_potential.yaml

# 使用默认配置（在配置目录下运行）
python LmpPy/scripts/calc_ibm_potential.py

# 安静模式
python LmpPy/scripts/calc_ibm_potential.py -c ibm_potential.yaml -q
```

### 映射转换工具

```bash
# YAML映射转CSV格式
python LmpPy/scripts/yaml2csv_mapping.py mapping.yaml -o mapping.csv
```

### 构建CG体系

```bash
# 根据单分子拓扑和分子数量构建完整多分子体系
python LmpPy/scripts/build_cg_system.py \
    --topology monomer_topo.txt \
    --copies 100 \
    --output system.data
```

### LAMMPS data转GRO

```bash
# LAMMPS data文件转GROMACS GRO格式
python LmpPy/scripts/data2gro.py input.data -o output.gro
```

---

## IBM势能计算配置文件

IBM势能计算流程使用独立的配置文件 `ibm_potential.yaml`：

```yaml
# ============================================
# 模拟参数
# ============================================
simulation:
  temperature: 400          # 温度 (K)
  units: real               # 单位系统: real (kcal/mol), metal (eV), lj

# ============================================
# 轨迹输入
# ============================================
trajectory:
  format: pickle            # 格式: pickle | lammpsdump
  path: cg_trajectory.pkl   # 轨迹文件路径
  stride: 1                 # 帧间隔 (跳帧)

# ============================================
# 分布计算参数
# ============================================
distribution:
  n_bins: 200               # 直方图bins数
  bond_range: [0.5, 6.0]    # 键长范围 (Å)
  angle_range: [0, 180]     # 角度范围 (degrees)
  dihedral_range: [-180, 180]  # 二面角范围 (degrees)
  pair_range: [1.5, 18.0]   # RDF范围 (Å)
  rdf_dr: 0.01              # RDF bin宽度 (Å)
  rdf_exclude_bonds: true   # RDF排除键合对
  rdf_exclude_angles: true  # RDF排除1-3对
  rdf_exclude_dihedrals: true  # RDF排除1-4对

# ============================================
# 平滑参数
# ============================================
smoothing:
  method: auto              # 方法: auto | harmonic | gaussian
  sg_window: 21             # Savitzky-Golay窗口大小
  sg_polyorder: 3           # Savitzky-Golay多项式阶数
  sigma_range: [2.0, 3.0, 5.0, 7.0]  # Gaussian平滑sigma范围

# ============================================
# 输出配置
# ============================================
output:
  table_dir: lammps_tables        # LAMMPS表文件目录
  potentials_dir: potentials_output  # 势能文件目录
  distributions_dir: distributions_output  # 分布文件目录
  generate_plots: true            # 是否生成图表
  generate_reference: true        # 是否生成参考文件
```

**参数说明：**

| 参数 | 说明 |
|------|------|
| `trajectory.format` | 轨迹格式，pickle支持快速加载，lammpsdump为标准LAMMPS dump格式 |
| `stride` | 帧间隔，用于减少计算量 |
| `rdf_exclude_*` | RDF计算时排除直接键合、1-3、1-4相互作用对 |
| `method` | 平滑方法：auto自动选择，harmonic用于bond/angle，gaussian用于dihedral |

---

## 模块依赖关系

### 模块调用图

```
run_refactored.py
    ├── core/config_loader.py
    │   └── ConfigLoader → SystemConfig, LAMMPSParams, ReactionInfo
    ├── core/mapping_generator.py
    │   └── MappingGenerator → CGCompareList
    ├── core/template_parser.py
    │   └── TemplateParser → ReactionTemplate
    ├── core/lammps_data_extractor.py
    │   └── LAMMPSDataExtractor → AtomData, BondData
    │   └── decode_image_flags_vectorized() (362x 加速)
    ├── core/bond_detector.py
    │   └── BondDetector → BondChanges
    │   └── compare_two_bonds()
    ├── core/reaction_locator.py
    │   ├── ReactionLocator → ReactionMatch
    │   └── depends on: bond_detector, template_parser
    ├── core/cg_mapper.py
    │   ├── CGMapper → CGMapping
    │   └── depends on: reaction_locator, template_parser
    ├── core/cg_converter.py
    │   ├── CGConverter (惰性索引缓存)
    │   └── depends on: cg_mapper
    ├── core/cg_bond_mapper.py
    │   ├── CGBondMapper → CGBond
    │   └ atom_bonds_to_cg_bonds()
    ├── core/cg_topology.py
    │   ├── CGTopology
    │   └── derive_cg_topology_from_bonds()
    ├── core/bonds_recorder.py
    │   ├── BondsRecorder → BondRecord, CGTopologyRecord (向后兼容)
    ├── core/cg_initializer.py
    │   ├── CGInitializer → CGSystem
    │   └── depends on: config_loader, mapping_generator, lammps_data_extractor,
    │                    cg_bond_mapper, cg_mapper, cg_topology
    ├── utils/graph_utils.py
    │   └── find_molecules() (Numba 30-80x 加速)
    ├── utils/coordinate_utils.py
    │   ├── wrap_coordinates()
    │   ├── pbc_distance()
    │   └── unwrap_coords_python()
    └── utils/file_utils.py
        ├── write_lammps_dump_file()
        └── write_cg_trajectory()
```

### 数据流

```
1. 配置加载
   ConfigLoader.load_system_config() → SystemConfig
   ConfigLoader.load_lammps_params() → LAMMPSParams

2. 初始化
   MappingGenerator.generate() → CGCompareList (初始CG映射)
   TemplateParser.load_all_reaction_templates() → List[ReactionTemplate]
   CGInitializer.initialize() → CGSystem (初始CG拓扑)

3. 主循环
   ┌─────────────────────────────────────────────────────────────┐
   │ Step 1: bond/react                                          │
   │   LAMMPS fix bond/react                                      │
   │                                                              │
   │ Step 2: 检测反应                                             │
   │   LAMMPSDataExtractor.extract_all() → AtomData, BondData    │
   │   BondDetector.detect() → BondChanges                       │
   │   ReactionLocator.locate() → List[ReactionMatch]            │
   │   CGMapper.batch_update() → CGMapping (更新)                 │
   │   CGConverter.convert() → CG坐标                             │
   │   缓存反应帧 → reaction_frames.npz (v2.4+)                   │
   │                                                              │
   │ Step 3: 弛豫                                                 │
   │   NVE/limit + NVT (反应原子)                                  │
   │   NPT (全原子)                                               │
   │                                                              │
   │ Step 4: 更新                                                 │
   │   LAMMPSDataExtractor.invalidate_cache()                    │
   └─────────────────────────────────────────────────────────────┘

4. 输出
   write_cg_trajectory() → cg_trajectory.lammpstrj
   CGCompareList.to_csv() → final_cg_compare_list.csv
   save_reaction_frames() → reaction_frames.npz
```

---

## 许可证

MIT License

---

## 更新日志

### v2.4 (2026-06-03)
- **输出文件精简重构**
  - 移除 `bonds_records/` 目录输出（功能已由 `reaction_frames.npz` 替代）
  - 移除 `changed_bead_id_list.pkl` 输出（重构遗留的废弃文件）
  - 精简 `reaction_frames.npz` 格式：从 11 个数组减至 7 个数组
    - 移除：`aa_ids`, `aa_types`, `cg_bonds_before`, `cg_bonds_after`
    - `cg_bonds` 由后处理从 `aa_bonds + cg_mapping` 推导
    - `aa_ids` 从 `cg_mapping[:, 3]` (AA_id 列) 获取
  - 后处理代码（`reaction_frames_parser.py`, `reaction_pairs.py`, `non_react_pairs.py`）添加自动格式检测和推导逻辑
  - 向后兼容：旧格式 npz 仍可正常读取
- 更新 README 文档，反映新的输出格式

### v2.3 (2026-05-19)
- 完善 README 文档
  - 新增主循环执行顺序详细说明
  - 新增模块依赖关系图和数据流说明
  - 新增所有 CLI 脚本说明 (build_cg_system.py, data2gro.py)
  - 新增 CGTopology API 参考
  - 新增 ReactionLocator API 参考
- 代码结构完善
  - scripts/ 目录新增 build_cg_system.py, data2gro.py
  - config/ 目录新增 mapping/, reactions/ 子目录

### v2.2 (2026-04-29)
- 新增CLI脚本
  - `calc_dist.py`: 分布计算命令行工具
  - `plot_dist.py`: 分布绘图命令行工具
  - `yaml2csv_mapping.py`: YAML到CSV映射转换工具
- 新增 `CGInitializer` / `CGSystem` / `initialize_cg_system` 模块
- smooth_utils 扩展: 新增 constants, io, preprocess, peaks, zones, quality, optimize, angle_dihedral, report, dist_config, dist_plot 模块
- 修正脚本调用方式 (scripts/ 无 `__init__.py`, 使用直接执行方式)
- 修正 API 文档: `compare_two_bonds` 返回值顺序、`atom_bonds_to_cg_bonds` 参数说明

### v2.1 (2026-04-02)
- 新增工具模块 (`LmpPy/tools/`)
  - `aa2cg`: 全原子→粗粒化转换工具
  - `ibm_potential`: IBM势能计算流程
  - `smooth_utils`: 分布平滑工具包
- 新增CLI脚本 (`LmpPy/scripts/`)
  - `smooth_distribution.py`: 分布平滑命令行工具
  - `convert_aa2cg.py`: AA→CG转换命令行工具
  - `calc_ibm_potential.py`: IBM势能计算命令行工具
- 新增工具函数 (`LmpPy/utils/`)
  - `topology.py`: 拓扑文件读写工具
  - `units.py`: 单位转换工具
- 支持 pickle 和 LAMMPS dump 双轨迹格式

### v2.0 (2026-03-28)
- 添加 `CGCompareList.from_csv()` 方法支持加载已有CG映射
- 添加断点续算功能
- 完善配置文件验证
- 更新文档

### v2.0 (2026-03-27)
- 完全重构，支持多体系
- 配置驱动设计
- 向量化/Numba优化
- 模块化架构