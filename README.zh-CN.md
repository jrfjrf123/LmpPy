# LmpPy - LAMMPS Bond/React 后处理框架 v2.7

> [English](README.md) | 中文

> 版本: 2.7
> 日期: 2026-09-11
> 语言: 中文

LmpPy 是面向高分子反应体系的 LAMMPS `bond/react` 与 `bond/create` 模拟后处理框架。它实时将全原子（AA）模拟转换为粗粒化（CG）轨迹，检测并记录成键与断键事件，自动更新反应后的 CG 映射与拓扑结构，并提供丰富的后处理分析工具。

框架面向高分子材料的多尺度建模：自由基共聚、环氧开环、聚氨酯逐步聚合、交联网络等。除轨迹转换外，它还为迭代玻尔兹曼反演（IBI）势函数开发提供所需的 CG 分布数据，并为在 CG 层面替代 `bond/react` 的机器学习模型提供逐事件训练样本。

---

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [依赖项与安装](#依赖项与安装)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [建键模式](#建键模式)
- [主循环执行顺序](#主循环执行顺序)
- [配置文件快速参考](#配置文件快速参考)
- [CG 映射配置](#cg-映射配置)
- [继续计算功能](#继续计算功能)
- [输出文件概要](#输出文件概要)
- [模块 API 参考](#模块-api-参考)
- [工具模块](#工具模块)
- [后处理分析](#后处理分析)
- [CLI 脚本](#cli-脚本)
- [冒烟测试](#冒烟测试)
- [常见问题](#常见问题)
- [模块依赖关系与数据流](#模块依赖关系与数据流)
- [更新日志](#更新日志)

---

## 项目简介

### 核心功能

1. **CG 轨迹生成** — 实时将反应体系的全原子轨迹转换为粗粒化轨迹
2. **反应监测** — 检测成键与断键事件并按周期统计（支持 bond/react 和 bond/create 两种模式），为单体转化率与竞聚率统计提供原始数据
3. **CG 映射更新** — 反应后自动更新原子到 bead 的映射关系
4. **CG 拓扑推导** — 从键连信息推导粗粒化键、角、二面角
5. **断点续算** — 支持从上一次运行的最终状态继续计算

### 特性亮点

| 特性 | 说明 |
|------|------|
| **多体系支持** | 通过配置文件定义不同高分子反应体系（自由基共聚、环氧开环、聚氨酯逐步聚合、交联网络等） |
| **双建键模式** | bond/react 模板匹配 + bond/create 距离条件，覆盖不同场景 |
| **势函数开发** | IBI 迭代玻尔兹曼反演链路：CG 分布 → 玻尔兹曼反演 → 表格势函数，支持异核 bead 对的交叉混合 |
| **GROMACS 互通** | 将 GROMACS `top` + `gro`（GAFF）体系转换为 LAMMPS `data` 文件，复用已有全原子力场 |
| **配置驱动** | 所有参数通过 YAML 配置文件定义，无需修改代码 |
| **高性能** | 向量化/Numba 优化关键计算，部分操作加速 30–300 倍 |
| **MPI 并行** | 支持多进程并行运行 |
| **模块化设计** | 清晰的模块划分，便于维护和扩展 |
| **完善的文档** | 详细的技术文档（`docs/` 目录），覆盖配置、工作流、API、输出文件等 |

### 性能优化

| 优化项 | 加速效果 | 说明 |
|--------|----------|------|
| 向量化 image 解码 | **362x** | 8.2ms → 0.023ms |
| Numba find_molecules | **30–80x** | BFS 分子查找 |
| Numba unwrap_coords | **4–9x** | 坐标展开 |
| 惰性索引缓存 | O(n^2) → O(n) | CG 坐标转换 |

### 数据流（三仓库管线）

LmpPy 是更大规模多尺度管线的第一环，与以下仓库协作：

```
LmpPy (CG后处理) → SOAP_calc_and_Feature_select (SOAP计算+特征筛选) → mlcgsim (XGBoost驱动)
```

全原子阶段产出反应事件与 CG 映射；SOAP 描述符在反应键附近计算并筛选为紧凑特征集；基于该特征集训练的 XGBoost 分类器随后在 CG 模拟中驱动反应，运行时不再调用 `bond/react`。

详见 [docs/workflow.md](docs/workflow.md) 的完整数据流说明。

---

## 项目结构

```
LmpPy/
├── __init__.py                       # 包入口
├── run_refactored.py                 # 主入口脚本 (LAMMPSReactionRunner)
├── pyproject.toml                    # 包元数据与依赖声明
├── test_integration.py               # 集成测试
├── test_gmx2lmp_data.py              # gmx2lmp_data 测试套件
├── test_reactivity_ratio.py          # 竞聚率统计测试套件
│
├── core/                             # 核心功能模块
│   ├── __init__.py                   # 导出所有核心类和函数
│   ├── config_loader.py              # YAML 配置加载器 (ConfigLoader, SystemConfig, LAMMPSParams)
│   ├── mapping_generator.py          # CG 映射生成器 (CGCompareList, MappingGenerator)
│   ├── template_parser.py            # LAMMPS 模板解析器 (TemplateParser, ReactionTemplate)
│   ├── lammps_data_extractor.py      # LAMMPS 数据提取器 (LAMMPSDataExtractor, AtomData, BondData)
│   ├── bond_detector.py              # 键变化检测器 (BondDetector, BondChanges)
│   ├── reaction_locator.py           # 反应位点定位器 (ReactionLocator, ReactionMatch)
│   ├── cg_reaction_identifier.py     # CG 级反应识别 (v2.6+, 模板签名匹配/交叉验证)
│   ├── reaction_commands.py          # 反应命令生成 (v2.6+, bond/create + bond/react 命令)
│   ├── cg_mapper.py                  # CG 映射更新器 (CGMapper, CGMapping)
│   ├── cg_converter.py               # CG 坐标转换器 (CGConverter)
│   ├── cg_bond_mapper.py             # 粗粒键映射器 (CGBondMapper)
│   ├── cg_topology.py                # CG 拓扑推导器 (CGTopology)
│   ├── bonds_recorder.py             # 键连表记录器 (BondsRecorder, 向后兼容)
│   ├── cg_initializer.py             # CG 系统初始化器 (CGInitializer, CGSystem)
│   ├── smoke_validator.py            # 冒烟测试验证器 (SmokeValidator)
│   └── smoke_test_harness.py         # 冒烟测试编排器 (SmokeTestHarness)
│
├── output_analysis/                  # 后处理分析子包 (v2.6+)
│   ├── __init__.py
│   ├── __main__.py                   # `python -m LmpPy.output_analysis` 入口
│   ├── cli.py                        # 统一 CLI 入口
│   ├── chain_length.py               # 链长分布分析
│   ├── distance.py                   # 反应距离分布分析
│   ├── reaction_stats.py             # 反应统计
│   ├── reactivity_ratio.py           # 竞聚率 r1/r2 估计（CG 与 AA 两条路径）
│   ├── js_divergence.py              # JS 散度计算
│   ├── loader.py                     # 数据加载工具
│   ├── theory.py                     # 理论分布模型 (Schulz-Zimm, Poisson, Log-normal)
│   └── plot.py                       # 统一绘图配置
│
├── utils/                            # 工具函数
│   ├── __init__.py
│   ├── coordinate_utils.py           # 坐标处理 (wrap, PBC distance, unwrap)
│   ├── file_utils.py                 # 文件 I/O (dump 读写)
│   ├── graph_utils.py                # 图论算法 (find_molecules Numba 优化)
│   ├── topology.py                   # 拓扑文件读写
│   └── units.py                      # 单位转换
│
├── tools/                            # 独立工具模块
│   ├── __init__.py
│   ├── aa2cg/                        # 全原子到粗粒化转换
│   │   ├── data_converter.py         # LAMMPS data 转换
│   │   ├── trj_converter.py          # 轨迹转换
│   │   └── mapping_utils.py          # 映射工具函数
│   ├── ibm_potential/                # IBM / IBI 势能计算
│   │   ├── config.py                 # 配置加载
│   │   ├── distribution.py           # 分布计算
│   │   ├── boltzmann.py              # 玻尔兹曼反演
│   │   ├── lammps_table.py           # LAMMPS table 生成
│   │   ├── tabulated_potential.py    # 表格势函数解析/拟合/绘图
│   │   ├── gromacs_loader.py         # GROMACS 轨迹读取
│   │   ├── pickle_loader.py          # pickle 格式 CG 轨迹读取
│   │   ├── dist_config.py            # 分布绘图配置
│   │   └── dist_plot.py              # 分布绘图
│   └── smooth_utils/                 # 分布平滑工具包
│       ├── core.py / cli.py / constants.py / io.py
│       ├── preprocess.py / peaks.py / zones.py / quality.py
│       ├── optimize.py / angle_dihedral.py
│       └── report.py
│
├── scripts/                          # CLI 脚本
│   ├── build_cg_config.py            # 生成 CG 配置
│   ├── build_cg_system.py            # 构建 CG 体系 LAMMPS data 文件
│   ├── gmx2lmp_data.py               # GROMACS top+gro → LAMMPS data 转换
│   ├── convert_aa2cg.py              # AA→CG 转换 CLI
│   ├── generate_initial_mapping.py   # 生成初始 CG 映射
│   ├── yaml2csv_mapping.py           # YAML→CSV 映射转换
│   ├── data2gro.py                   # LAMMPS data 转 GRO
│   ├── smooth_distribution.py        # 分布平滑 CLI
│   ├── calc_dist.py                  # 分布计算 CLI
│   ├── plot_dist.py                  # 分布绘图 CLI
│   ├── calc_ibm_potential.py         # IBM 势能计算 CLI
│   ├── calc_ibm_potential_from_dist.py # 从已有分布计算 IBM 势能
│   ├── fit_tabulated.py              # 表格势函数解析拟合
│   ├── mix_cross_tabulated.py        # 异核表格势函数交叉混合
│   ├── plot_tabulated.py             # 表格势函数绘图
│   ├── extract_reaction_frame.py     # 提取单个反应帧用于可视化
│   ├── validate_config.py            # 配置验证
│   └── test_smoke_run.py             # 冒烟测试独立运行脚本
│
├── config/                           # 配置文件模板
│   ├── system.yaml                   # 体系配置模板
│   ├── lammps_params.yaml            # LAMMPS 运行参数模板
│   ├── mass_list.yaml                # 原子质量表
│   └── mapping/                      # CG 映射配置模板
│
├── docs/                             # 文档和示例
│   ├── bond-modes.md                 # 建键模式 (bond/react vs bond/create)
│   ├── configuration.md              # 完整配置参数参考
│   ├── cg-mapping.md                 # CG 映射配置详解
│   ├── workflow.md                   # 运行流程与数据流
│   ├── output-files.md               # 输出文件格式说明
│   ├── modules.md                    # 模块 API 参考
│   ├── cli-scripts.md                # CLI 脚本使用指南
│   ├── gro_format.md                 # GRO 格式说明
│   ├── lammps_data_format.md         # LAMMPS data 格式说明
│   ├── reaction_locator_fix.md       # ReactionLocator 修复记录
│   └── examples/                     # 示例配置文件
│       ├── bond_react/               # bond/react 模式配置示例
│       ├── bond_create/              # bond/create 模式配置示例
│       └── reactions/                # 反应模板示例
│
├── README.md                         # 英文 README
└── README.zh-CN.md                   # 中文 README（本文档）
```

---

## 依赖项与安装

### 1. 创建虚拟环境（conda）

```bash
conda create -n lmp_py_react python=3.10 -y
conda activate lmp_py_react
```

`pyproject.toml` 要求 Python >= 3.9，推荐 3.10。不用 conda 时等价的 `venv` 做法：

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

后续所有 `pip` 命令都应在该环境下执行（`conda activate` 后 shell 提示符会显示环境名）。

### 2. 安装依赖

必需依赖由 `core/` 与 `utils/` 在导入期无条件加载：

```bash
pip install numpy pandas pyyaml
```

Numba 强烈推荐（`find_molecules` 与 `unwrap_coords` 的 JIT 加速），缺失时回退纯 Python：

```bash
pip install numba
```

其余功能按 `pyproject.toml` 声明的 extras 分组安装。**先 `cd` 到含有 `pyproject.toml` 的仓库根目录**，再按需选择：

```bash
pip install ".[mpi]"        # mpi4py —— MPI 并行运行
pip install ".[analysis]"   # scipy / matplotlib / seaborn / tqdm —— 后处理分析与绘图
pip install ".[tools]"      # MDAnalysis / scikit-optimize —— GROMACS 转换与 IBI 势函数拟合
pip install ".[dev]"        # pytest / black —— 开发与测试
```

一次装齐除 LAMMPS 之外的全部依赖：

```bash
pip install ".[mpi,analysis,tools,dev]"
```

`numba`、`mpi4py` 与 LAMMPS Python 接口均在 `try`/`except` 中导入：缺失时 LmpPy 会回退到纯 Python 实现，或打印警告并以测试模式运行，因此最小环境下包仍可正常导入。

### 3. LAMMPS 编译

需要编译支持 Python 接口的 LAMMPS：

```bash
cd lammps/src
make yes-python yes-mpi
make mpi
```

---

## 快速开始

### 1. 准备配置文件

创建一个配置目录，包含以下文件（示例见 `docs/examples/`）：

```
my_system/
├── system.yaml                    # 体系配置
├── lammps_params.yaml             # LAMMPS 运行参数
├── molecule1_mapping.yaml         # 分子 CG 映射配置
├── system.data                    # LAMMPS 初始数据文件
└── reactions/                     # 反应配置目录（bond/react 模式必需）
    └── rxn1/
        ├── rxn1_pre.lammpstemplate
        ├── rxn1_post.lammpstemplate
        ├── rxn1.map
        ├── rxn1_pre_mapping.yaml
        └── rxn1_post_mapping.yaml
```

**文档导航**：
- `docs/examples/` — 可直接运行的 mini_test 体系（配置 + 映射 + data + 反应模板齐备）
- [docs/examples/bond_react/](docs/examples/bond_react/) — bond/react 字段参考（仅 YAML 骨架，需自备 mapping/data/反应文件）
- [docs/examples/bond_create/](docs/examples/bond_create/) — bond/create 字段参考（仅 YAML 骨架，需自备 mapping/data 文件）
- [docs/configuration.md](docs/configuration.md) — 配置参数字段详解
- [docs/cg-mapping.md](docs/cg-mapping.md) — CG 映射格式说明

若全原子体系已有 GROMACS 形式，`system.data` 可不必手写，直接由 GAFF 的 `top` + `gro` 生成：

```bash
python -m LmpPy.scripts.gmx2lmp_data --top system.top --gro system.gro -o my_system/system.data
```

### 2. 运行模拟

```bash
# 测试模式（仅加载配置，不运行 LAMMPS）
python -m LmpPy.run_refactored my_system/ --test

# 单进程运行
python -m LmpPy.run_refactored my_system/

# MPI 并行运行（4 进程）
mpirun -np 4 python -m LmpPy.run_refactored my_system/ --loop-num 100

# 冒烟测试（5 个 loop，自动验证输出）
python -m LmpPy.run_refactored my_system/ --smoke-test --loop-num 5
```

---

## 运行方式

### 命令行参数

```bash
python -m LmpPy.run_refactored <config_dir> [options]

参数:
  config_dir            配置文件目录路径（位置参数，必需）

选项:
  --test                测试模式，仅加载配置不运行 LAMMPS
  --loop-num N          覆盖配置文件中的循环次数
  --smoke-test          冒烟测试模式（临时目录 + 自动验证输出）
```

### MPI 并行运行

```bash
# 4 进程并行
mpirun -np 4 python -m LmpPy.run_refactored config/

# 8 进程并行
mpirun -np 8 python -m LmpPy.run_refactored config/ --loop-num 100
```

### 环境变量

```bash
# 设置 OpenMP 线程数（默认为 1）
export OMP_NUM_THREADS=1
```

---

## 建键模式

LmpPy 支持两种 LAMMPS 建键模式，通过 `lammps_params.yaml` 配置互斥选择：

| 模式 | 建键策略 | 配置复杂度 | 适用方向 |
|------|---------|-----------|---------|
| **bond/react** | 模板化学匹配（pre/post .lammpstemplate） | 高 | 明确的化学反应（环氧开环、聚氨酯形成等） |
| **bond/create** | 距离 + 原子类型条件（无模板） | 低 | 交联、渗透网络、非特异性键形成 |

核心区别：bond/react 依赖分子模板匹配，产物结构由模板严格定义；bond/create 仅凭原子类型和空间距离判断是否建键，无需模板文件。

**选择指南**：

- 反应化学明确、需要精确控制产物结构 → **bond/react**
- 随机交联、快速原型验证、概率性建键 → **bond/create**
- 从零开始搭建新体系 → **bond/create**（更快速）

**详细文档**：[docs/bond-modes.md](docs/bond-modes.md)

---

## 主循环执行顺序

LmpPy 模拟的主循环每个周期按以下顺序执行：

| 步骤 | 描述 | 涉及模块 |
|------|------|----------|
| **Step 1** | 运行建键命令（bond/react 或 bond/create） | LAMMPS fix |
| **Step 2** | 检测反应：提取数据、检测键变化、定位反应、更新 CG 映射、转换 CG 坐标、写入轨迹 | `LAMMPSDataExtractor` → `BondDetector` → `ReactionLocator` → `CGMapper` → `CGConverter` |
| **Step 3** | 弛豫：NVE/limit + NVT（反应原子）→ NPT（全原子） | LAMMPS 命令序列 |
| **Step 4** | 更新：使数据提取器缓存失效，准备下一轮 | `LAMMPSDataExtractor.invalidate_cache()` |

**Step 2 详细流程**：

```
1. LAMMPSDataExtractor: 提取原子和键数据
2. BondDetector: 检测键变化 (created_bonds, deleted_bonds)
3. ReactionLocator: 反应模板匹配（双重验证）
4. CGMapper: 更新 CG 映射（基于 ReactionMatch）
5. CGConverter: 转换 CG 坐标（惰性索引缓存优化）
6. 写入 CG 轨迹帧（post-reaction, pre-relaxation）
7. 缓存反应帧数据到 reaction_frames.npz
```

**详细文档**：[docs/workflow.md](docs/workflow.md)

---

## 配置文件快速参考

LmpPy 使用 YAML 格式的配置文件驱动运行，主要配置文件有两个：`system.yaml` 和 `lammps_params.yaml`。

### system.yaml 字段总表

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `system.name` | string | 否 | 体系名称，默认 `"unnamed"` |
| `system.bead_type_names` | dict | **是** | bead 类型名称→LAMMPS type 编号映射 |
| `system.mapping_files[].path` | string | **是** | 分子映射 YAML 文件路径 |
| `system.mapping_files[].copies` | int | **是** | 该类型分子在初始 data 中的数量 |
| `system.work_dir` | string | 否 | 工作目录，默认 `"work"` |
| `system.output_dir` | string | 否 | 输出目录，默认 `"output"` |

### lammps_params.yaml 字段总表

| 段 | 关键字段 | 说明 |
|----|---------|------|
| `simulation` | loop_num, dt, temperature, pressure, ensemble | 模拟基本参数 |
| `steps` | bond_react_check, run_per_loop, nve_limit | 步数分配 |
| `npt` | tcouple, pcouple | 控温控压参数 |
| `bond_react` | stabilization, reactions[] | bond/react 模式配置 |
| `bond_create` | enabled, pairs[], cg_update | bond/create 模式配置 |
| `molecules` | {name: template_path} | 分子模板映射 |
| `files` | data_file, initial_cg_mapping, output_* | 输入输出文件路径 |

**详细文档**：[docs/configuration.md](docs/configuration.md)

---

## CG 映射配置

CG 映射定义了全原子模拟中哪些原子构成一个粗粒珠子（bead），以及珠子的质量权重。

### YAML 映射格式

```yaml
# 1. site-types：Bead 类型定义
site-types:
  Bead1:
    index: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]   # 相对原子索引 (0-based)
    x-weight: [12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1] # 质量权重

# 2. config：映射配置
config:
  - anchor: 0                   # 固定为 0（程序自动累积偏移）
    repeat: 1                   # 分子数量
    offset: 23                  # 每个分子的原子数
    sites:
      - [Bead1, 0]              # [bead类型名称, 起始位置]
      - [Bead2, 4]
      - [Bead3, 17]
```

### 核心规则

- `anchor` 必须为 0，程序自动处理多分子和多文件累积偏移
- 原子 ID 计算公式：`atom_id = anchor + site_start_offset + index_value + 1`
- `x-weight` 用于质心计算：`R_bead = sum(m_i * r_i) / sum(m_i)`
- 反应前后需要独立的映射文件（`pre_mapping` / `post_mapping`），反映 bead_type 的变化

**详细文档**：[docs/cg-mapping.md](docs/cg-mapping.md)

---

## 继续计算功能

程序支持从上一次运行的最终状态继续计算，无需重新模拟已完成的步骤。

### 使用方法

1. 将上一轮的 `final_frame.data` 和 `final_cg_compare_list.csv` 复制到新配置目录
2. 更新 `lammps_params.yaml` 中的文件路径：

```yaml
files:
  data_file: "final_frame.data"                # 使用上一轮的最终 data 文件
  initial_cg_mapping: "final_cg_compare_list.csv"  # 使用上一轮的最终 CG 映射
```

3. 设置 `loop_num` 为剩余需要的循环数，然后运行：

```bash
mpirun -np 4 python -m LmpPy.run_refactored continue_config/ --loop-num 50
```

### 工作原理

- 程序检测 `initial_cg_mapping` 指定的文件是否存在
- 如果存在，直接加载已有 CG 映射（跳过生成步骤）
- 如果不存在，从 `system.yaml` 的 `mapping_files` 生成映射

---

## 输出文件概要

主模拟管线输出的核心文件：

| 文件 | 格式 | 说明 |
|------|------|------|
| `cg_trajectory.lammpstrj` | LAMMPS dump | CG 轨迹文件，可 VMD 可视化 |
| `reaction_num.txt` | 文本 | 每步反应计数统计 |
| `final_frame.data` | LAMMPS data | 最终帧原子数据（断点续算用） |
| `final_cg_compare_list.csv` | CSV (5 列) | 最终 CG 映射关系 |
| `reaction_frames.npz` | NPZ (7 数组) | 反应帧详细数据（SOAP 计算输入） |
| `cg_bonds.txt` / `cg_angles.txt` / `cg_dihedrals.txt` | 文本 | CG 拓扑文件 |

**详细文档**：[docs/output-files.md](docs/output-files.md)

### reaction_frames.npz 核心内容

这是 **SOAP 计算模块的核心输入数据源**，包含 7 个数组：

| 数组名 | Shape | 说明 |
|--------|-------|------|
| `aa_coords_before` | (n_reactions, n_atoms, 3) | 反应前 AA 坐标 |
| `aa_coords_after` | (n_reactions, n_atoms, 3) | 反应后 AA 坐标 |
| `aa_bonds_before` | (n_reactions,) object | 反应前原子键 |
| `aa_bonds_after` | (n_reactions,) object | 反应后原子键 |
| `cg_mapping_before` | (n_reactions, n_atoms, 2) | 反应前 CG 映射 [bead_id, bead_type] |
| `cg_mapping_after` | (n_reactions, n_atoms, 2) | 反应后 CG 映射 |
| `timestep` | (n_reactions,) | 反应时间步 |

---

## 模块 API 参考

### 导入核心模块

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
    BondsRecorder,
    # CG 系统
    CGInitializer, CGSystem, initialize_cg_system,
    # CG 拓扑
    CGTopology, derive_cg_topology_from_bonds,
    # v2.6+ 新增
    BondCreateConfig,
    get_reaction_mode, generate_fix_bond_create, generate_fix_bond_react,
    update_cg_mapping_create,
)

# 以下符号未在 core/__init__.py 中重导出，需从子模块导入
from LmpPy.core.reaction_commands import BondCreatePair
from LmpPy.core.cg_reaction_identifier import (
    CGReactionSignature, load_template_signatures, identify_reaction,
)
from LmpPy.core.smoke_test_harness import SmokeTestHarness
from LmpPy.core.smoke_validator import SmokeValidator, SmokeTestReport
```

### 使用示例

```python
# 配置加载
loader = ConfigLoader("config/")
system_config = loader.load_system_config()
lammps_params = loader.load_lammps_params()

# 生成 CG 映射
cg_list = MappingGenerator().generate_from_system(system_config, Path("config/"))
print(f"Bead 数: {cg_list.n_beads}, 分子数: {cg_list.n_molecules}")

# 初始化完整 CG 系统
cg_system = initialize_cg_system("config/")
print(f"CG 键数: {len(cg_system.bonds)}")

# 加载反应模板
templates = load_all_reaction_templates(Path("config/reactions/"))

# 键变化检测
changes = BondDetector(n_atoms=10000).detect(bonds_before, bonds_after)
print(f"新增键: {len(changes.created_bonds)}, 断裂键: {len(changes.deleted_bonds)}")
```

**详细 API 文档**：[docs/modules.md](docs/modules.md)

---

## 工具模块

LmpPy 提供三个独立的工具子包，位于 `LmpPy/tools/` 目录：

### aa2cg — 全原子到粗粒化转换

支持从 GROMACS TRR 轨迹、LAMMPS data 文件、GRO+TPR 结构文件转换为 CG 格式。

```python
from LmpPy.tools.aa2cg import (
    load_aa_to_cg_mapping,
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
)
```

### ibm_potential — IBM / IBI 势能计算

从 CG 轨迹完成分布计算、玻尔兹曼反演到 LAMMPS table 文件生成的全流程，并提供表格势函数的解析、拟合与绘图。

```python
from LmpPy.tools.ibm_potential import (
    # 分布 → 势能 → table
    load_ibm_config,
    calculate_bond_distribution,
    calculate_bond_potential,
    create_lammps_table_files,
    # 表格势函数解析 / 拟合 / 绘图
    read_tabulated_table,
    fit_table,
    fit_lj,
    fit_bond_harmonic,
    plot_single_table,
    plot_all_tables,
)
```

输入轨迹同时支持 GROMACS 轨迹（`gromacs_loader.py`）与 pickle 格式 CG 轨迹（`pickle_loader.py`）。

### smooth_utils — 分布平滑工具包

支持自适应 Savitzky-Golay 平滑、高斯平滑和边界约束平滑。

```python
from LmpPy.tools.smooth_utils import (
    smooth_distribution,
    smooth_bond_with_harmonic_boundary,
)
```

**详细文档**：[docs/modules.md#2-工具模块-tools](docs/modules.md#2-工具模块-tools)

---

## 后处理分析

`output_analysis/` 子包（v2.6+）提供模拟结果的后处理统计分析功能：链长分布、反应距离分布、反应统计、JS 散度，以及共聚竞聚率。

### CLI 入口

```bash
# 链长分布分析
python -m LmpPy.output_analysis chain-length <input_dir> [--min-length N] [--theory-type TYPE]

# 反应距离分布分析
python -m LmpPy.output_analysis distance <input_dir> [--bins N]

# 反应统计
python -m LmpPy.output_analysis reaction-stats <input_dir>

# 从 CG 模拟输出估计竞聚率 r1/r2
python -m LmpPy.output_analysis reactivity-ratio <input_dir>

# 从 AA bond/react 输出估计竞聚率 r1/r2
# （--bond-react-check-step 须与 lammps_params.yaml 中 steps.bond_react_check 一致）
python -m LmpPy.output_analysis reactivity-ratio-aa <aa_dir> [--bond-react-check-step N]

# 执行所有分析
python -m LmpPy.output_analysis all <input_dir>

# 从 npz 重建反应详情
python -m LmpPy.output_analysis rebuild <npz_path> -o <csv_path>
```

两条竞聚率命令均由逐周期反应计数估计单体竞聚率 r1/r2，采用分块 bootstrap 给出置信区间，并输出随转化率演化的结果。二者的区别仅在数据来源：`reactivity-ratio` 读取 CG 模拟输出，`reactivity-ratio-aa` 从 AA 的 `reaction_frames.npz` 配合轨迹重建计数。

### Python API

```python
from LmpPy.output_analysis.chain_length import analyze_chain_length
from LmpPy.output_analysis.distance import analyze_distance
from LmpPy.output_analysis.reaction_stats import analyze_reaction_stats
from LmpPy.output_analysis.js_divergence import js_divergence

analyze_chain_length("output/", theory_type="schulz-zimm", Mn=5000, PDI=2.0)
analyze_distance("output/", bins=50)
stats = analyze_reaction_stats("output/")
```

**详细文档**：[docs/modules.md#4-后处理分析-output_analysis](docs/modules.md#4-后处理分析-output_analysis)

---

## CLI 脚本

LmpPy 提供一系列 CLI 脚本，可通过 `python -m LmpPy.scripts.<name>` 方式运行：

**体系准备**

| 脚本 | 用途 | 详细文档 |
|------|------|----------|
| `gmx2lmp_data.py` | GROMACS `top`+`gro`（GAFF）→ LAMMPS data 文件 | [docs/cli-scripts.md#15-gmx2lmp_datapy](docs/cli-scripts.md#15-gmx2lmp_datapy) |
| `build_cg_config.py` | 生成 CG 配置 | [docs/cli-scripts.md#2-build_cg_configpy](docs/cli-scripts.md#2-build_cg_configpy) |
| `build_cg_system.py` | 构建 CG 体系 LAMMPS data 文件 | [docs/cli-scripts.md#3-build_cg_systempy](docs/cli-scripts.md#3-build_cg_systempy) |
| `convert_aa2cg.py` | AA→CG 转换（data 文件或轨迹） | [docs/cli-scripts.md#7-convert_aa2cgpy](docs/cli-scripts.md#7-convert_aa2cgpy) |
| `generate_initial_mapping.py` | 生成初始 CG 映射 | [docs/cli-scripts.md#9-generate_initial_mappingpy](docs/cli-scripts.md#9-generate_initial_mappingpy) |
| `yaml2csv_mapping.py` | YAML→CSV 映射转换 | [docs/cli-scripts.md#14-yaml2csv_mappingpy](docs/cli-scripts.md#14-yaml2csv_mappingpy) |
| `data2gro.py` | LAMMPS data 转 GRO | [docs/cli-scripts.md#8-data2gropy](docs/cli-scripts.md#8-data2gropy) |

**分布与势函数**

| 脚本 | 用途 | 详细文档 |
|------|------|----------|
| `calc_dist.py` | 分布计算（VOTCA 格式，`--skip-existing` 支持增量重跑） | [docs/cli-scripts.md#4-calc_distpy](docs/cli-scripts.md#4-calc_distpy) |
| `smooth_distribution.py` | 分布平滑 | [docs/cli-scripts.md#11-smooth_distributionpy](docs/cli-scripts.md#11-smooth_distributionpy) |
| `plot_dist.py` | 分布绘图 | [docs/cli-scripts.md#10-plot_distpy](docs/cli-scripts.md#10-plot_distpy) |
| `calc_ibm_potential.py` | IBM 势能计算全流程 | [docs/cli-scripts.md#5-calc_ibm_potentialpy](docs/cli-scripts.md#5-calc_ibm_potentialpy) |
| `calc_ibm_potential_from_dist.py` | 从已有分布计算 IBM 势能 | [docs/cli-scripts.md#6-calc_ibm_potential_from_distpy](docs/cli-scripts.md#6-calc_ibm_potential_from_distpy) |
| `fit_tabulated.py` | 表格势函数解析拟合（LJ well/direct、harmonic、cosine） | — |
| `mix_cross_tabulated.py` | 两个表格势函数交叉混合为异核势 | — |
| `plot_tabulated.py` | 表格势函数与其拟合结果绘图 | — |

**工具**

| 脚本 | 用途 | 详细文档 |
|------|------|----------|
| `extract_reaction_frame.py` | 提取单个反应帧用于可视化 | — |
| `validate_config.py` | 配置验证 | [docs/cli-scripts.md#13-validate_configpy](docs/cli-scripts.md#13-validate_configpy) |
| `test_smoke_run.py` | 冒烟测试独立运行 | [docs/cli-scripts.md#12-test_smoke_runpy](docs/cli-scripts.md#12-test_smoke_runpy) |

**详细文档**：[docs/cli-scripts.md](docs/cli-scripts.md)

---

## 冒烟测试

冒烟测试（Smoke Test）是 v2.5 新增的快速正确性验证工具。它在少量循环下运行完整模拟管线，并自动验证输出文件的完整性和一致性。

### 设计目的

- **快速验证**：配置变更后，用少量 loop（默认 5）确认管线能否正常运行
- **正确性保证**：自动检测输出文件的完整性和数据一致性
- **CI/CD 友好**：返回标准 exit code（0=通过，1=未通过）

### 四项检查

| 检查 | 名称 | 说明 |
|------|------|------|
| A | 文件输出完整性 | 检查所有预期输出文件是否存在且内容合理 |
| B | CG mapping 一致性 | 验证 bead_type 唯一性、AA_id 无重复无缺失 |
| C | 反应后 mapping 交叉验证 | 用独立 CG 级签名算法重构预期 mapping，对比验证 |
| D | 反应计数合理性 | 检查计数非负、行数与 loop_num 匹配 |

### 使用方法

```bash
# 方式 1：通过 run_refactored.py（推荐）
python -m LmpPy.run_refactored my_system/ --smoke-test
python -m LmpPy.run_refactored my_system/ --smoke-test --loop-num 3

# 方式 2：通过独立脚本
python LmpPy/scripts/test_smoke_run.py my_system/ --loop-num 3 --keep-output
```

### 输出示例

```
============================================================
Smoke Test 报告
============================================================
  配置目录: /path/to/my_system
  循环次数: 5
    [✅] A. 文件输出完整性
    [✅] B. CG mapping 一致性
    [✅] C. 反应后 mapping 更新正确
    [✅] D. 反应计数合理性
  结果: 通过
============================================================
```

### 编程接口

```python
from LmpPy.core.smoke_test_harness import SmokeTestHarness

harness = SmokeTestHarness("config/", loop_num=5)
report = harness.run()
if report is not None:
    report.print()
    sys.exit(0 if report.passed else 1)
```

---

## 常见问题

### Q1: IndexError: index X is out of bounds

**原因**: CG 映射配置错误，原子 ID 超出范围。

**解决**: 检查 `mapping_files` 中的 `offset` 和 `repeat` 是否正确，确保所有 `anchor` 为 0。

### Q2: Bond atoms missing on proc

**原因**: 原子跑出盒子边界或键定义问题。

**解决**: 检查初始数据文件的键定义，减小时间步长，增加盒子尺寸。

### Q3: 反应不发生

**可能原因**:
- 反应截断半径 `cutoff` 太小
- 分子初始位置距离太远
- 反应模板不匹配（bond/react 模式）

**解决**: 增大 `cutoff` 值，检查模板与实际分子的匹配，延长模拟时间。

### Q4: YAML 解析报 Tab 字符错误

**原因**: YAML 文件包含 Tab 字符。

**解决**:
```bash
sed -i 's/\t/  /g' *.yaml
```

### Q5: MPI 进程通信错误

**原因**: MPI 环境配置问题。

**解决**: 确保所有进程可访问相同文件系统，检查 MPI 安装和环境变量。

### Q6: bond_create 模式下键未创建

**原因**: bond/create 配置中 `Rmin` 过小，或 `maxbond` 限额已满。

**解决**: 增大 `Rmin` 值，检查 `iparam.maxbond` / `jparam.maxbond` 设置是否合理，确认 `enabled: true` 已设置且 `bond_react.reactions` 未配置。

### Q7: bond_create 与 bond_react 同时启用错误

**原因**: 两种模式互斥，不能同时启用。

**解决**: 确保 `bond_create.enabled=true` 时，`bond_react.reactions` 不存在或为空列表；反之亦然。

### Q8: reactivity-ratio-aa 报 bond_react_check_step 不匹配

**原因**: AA 路径以 `时间步 - bond_react_check_step` 定位反应前帧，该参数默认为 1；若 `steps.bond_react_check` 配置得更大，将找不到对应帧。

**解决**: 用 `--bond-react-check-step` 传入与 `lammps_params.yaml` 中一致的值。程序会显式报错，不会静默取错帧。

---

## 模块依赖关系与数据流

### 模块调用图

```
run_refactored.py
    ├── core/config_loader.py
    │   └── ConfigLoader → SystemConfig, LAMMPSParams, ReactionInfo
    ├── core/mapping_generator.py
    │   └── MappingGenerator → CGCompareList
    ├── core/template_parser.py
    │   └── TemplateParser → ReactionTemplate (bond/react 模式)
    ├── core/reaction_commands.py
    │   └── load_bond_create_config, generate_fix_bond_* (v2.6+)
    ├── core/cg_reaction_identifier.py
    │   └── load_template_signatures, identify_reaction (v2.6+)
    ├── core/lammps_data_extractor.py
    │   └── LAMMPSDataExtractor → AtomData, BondData
    │   └── decode_image_flags_vectorized() (362x 加速)
    ├── core/bond_detector.py
    │   └── BondDetector → BondChanges
    ├── core/reaction_locator.py
    │   └── ReactionLocator → ReactionMatch
    ├── core/cg_mapper.py
    │   └── CGMapper → CGMapping
    ├── core/cg_converter.py
    │   └── CGConverter (惰性索引缓存)
    ├── core/cg_bond_mapper.py
    │   └── atom_bonds_to_cg_bonds()
    ├── core/cg_topology.py
    │   └── CGTopology, derive_cg_topology_from_bonds()
    ├── core/bonds_recorder.py
    │   └── BondsRecorder (向后兼容)
    ├── core/cg_initializer.py
    │   └── CGInitializer → CGSystem
    ├── utils/graph_utils.py
    │   └── find_molecules() (Numba 30-80x)
    ├── utils/coordinate_utils.py
    │   ├── wrap_coordinates(), pbc_distance()
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
   MappingGenerator.generate() → CGCompareList (初始 CG 映射)
   TemplateParser / ReactionInfo → 反应配置
   CGInitializer.initialize() → CGSystem (初始 CG 拓扑)

3. 主循环
   ┌─────────────────────────────────────────────────────────────┐
   │ Step 1: LAMMPS fix 建键                                     │
   │   bond/react 或 bond/create                                 │
   │                                                              │
   │ Step 2: 检测反应                                             │
   │   LAMMPSDataExtractor → BondDetector → ReactionLocator      │
   │   → CGMapper → CGConverter → 写入 CG 轨迹                 │
   │   缓存反应帧 → reaction_frames.npz                          │
   │                                                              │
   │ Step 3: 弛豫 (NVE/limit → NVT → NPT)                        │
   │                                                              │
   │ Step 4: 更新 (invalidate_cache)                              │
   └─────────────────────────────────────────────────────────────┘

4. 输出
   write_cg_trajectory() → cg_trajectory.lammpstrj
   CGCompareList.to_csv() → final_cg_compare_list.csv
   save_reaction_frames() → reaction_frames.npz
```

---

## 更新日志

### v2.7 (2026-09-11)

- **新增 GROMACS → LAMMPS 转换链路**
  - `scripts/gmx2lmp_data.py`: 将 GAFF 的 `top` + `gro` 转换为 LAMMPS `data` 文件（real 单位，`atom_style full`）
  - 多分子展开采用统一偏移基准，补齐 bonded 系数段，增加 atomtypes 重复校验
  - `test_gmx2lmp_data.py`: 端到端交叉验证测试套件
  - 文档: `docs/cli-scripts.md#15`

- **新增表格势函数工具链**
  - `tools/ibm_potential/tabulated_potential.py`: VOTCA/LAMMPS table 解析、类型识别、解析拟合（LJ 12-6、harmonic bond、cosine/harmonic angle）与绘图
  - `scripts/fit_tabulated.py`: 拟合 CLI，LJ 支持 `well` 与 `direct` 两种取参策略
  - `scripts/mix_cross_tabulated.py`: 两个表格势函数交叉混合为异核势
  - `scripts/plot_tabulated.py`: 表格势函数绘图 CLI

- **新增 AA 数据竞聚率分析**
  - `output_analysis/reactivity_ratio.py`: `collect_per_cycle_counts_aa()` 从 `reaction_frames.npz` 配合轨迹重建逐周期计数，反应前帧按 `bond_react_check_step` 回溯
  - `analyze_reactivity_ratio_aa()` 复用既有估计逻辑，新增转化率演化输出与详细报告
  - CLI 新增子命令: `reactivity-ratio`、`reactivity-ratio-aa`
  - `test_reactivity_ratio.py`: 覆盖 CG 与 AA 两条路径的测试套件

- **分布计算支持增量跳过**
  - `scripts/calc_dist.py`: `run_pipeline` 补齐 `--skip-existing`，与两个 pickle 管线对齐

- **正确性修复**
  - `tools/ibm_potential/{gromacs_loader,pickle_loader}.py`: 二面角分布的 `y` 只除了 `|bc|` 而 `x` 除了 `|n1||n2|`，等效把 `tan(phi)` 放大 `|n1||n2|` 倍（CG 键长下可达数百），分布被系统性压向 ±90°
  - `scripts/convert_aa2cg.py`: CG `data` 文件中写入的是 AA 原子质量而非 CG bead 质量；单珠映射下无碍，多珠体系会导致轻珠飞出、IBI 不稳定
  - `tools/aa2cg/data_converter.py`: 头部声明的是 bead 类型**个数**而非**最大类型号**，全局类型编号有空缺的体系会因此报错
  - `core/cg_reaction_identifier.py`: 模板签名支持 interior `bead_type` 为列表（如 `[1, 2]`），按每个候选类型各生成一条签名

- **打包**
  - `pyproject.toml`: 补上遗漏的 `pandas` 依赖，修正 `readme` 与包发现路径，新增 `analysis` / `tools` extras，修正 `config/mapping/` 的 `package-data`

- **文档**
  - `docs/configuration.md`: 补充 AA-AM 自由基共聚推荐生产配置（10480 原子 GAFF 体系实测）

### v2.6 (2026-06-22)

- **新增 bond/create 建键模式**
  - `core/reaction_commands.py`: 全新模块，支持 bond/create 的配置加载、LAMMPS 命令生成和 CG 映射更新
  - `BondCreatePair` / `BondCreateConfig` 数据类，封装 bond/create 类型对参数
  - `generate_fix_bond_create()` / `generate_fix_bond_react()` 纯函数命令生成
  - `update_cg_mapping_create()` 基于 YAML `type_map` 的 CG 映射更新
  - 配置互斥验证：两种模式不能同时启用

- **新增 CG 级反应识别模块**
  - `core/cg_reaction_identifier.py`: 独立于 LAMMPS 的反应类型识别
  - `CGReactionSignature` 数据类，封装 3-bead 类型签名
  - `load_template_signatures()` / `load_reaction_signatures()` 两种签名加载方式
  - 两级匹配策略：3-bead 粗筛 + chain 精筛
  - 双重验证：生产级与 cross-validation 独立验证管道
  - 文档: `docs/bond-modes.md`

- **新增后处理分析子包**
  - `output_analysis/`: 链长分布、反应距离、反应统计、JS 散度分析
  - 统一 CLI 入口 `python -m LmpPy.output_analysis <command>`
  - YAML 驱动的 `PlotConfig` 绘图配置系统
  - Schulz-Zimm/Poisson/Log-normal 理论分布模型

- **文档系统大重构**
  - README 精简为快速入门导航中心，详细内容移至 `docs/`
  - 新增: `docs/bond-modes.md`, `docs/configuration.md` (重写)
  - 重写: `docs/cg-mapping.md`, `docs/output-files.md`, `docs/cli-scripts.md`
  - 新增: `docs/modules.md` (完整 API 参考)
  - 重写: `docs/workflow.md` (完整流程图 + MPI 模型 + 断点续算)
  - 新增示例配置: `docs/examples/bond_react/`, `docs/examples/bond_create/`
  - README 文档导航，每个主题链接到对应 `docs/` 文件

- **配置验证增强**
  - `scripts/validate_config.py`: 全新配置验证脚本
  - `ConfigValidator` 类：文件检查、配置间一致性、语义验证
  - 支持 JSON 和 text 两种输出格式

### v2.5 (2026-06-09)

- **新增冒烟测试系统**
  - `core/smoke_validator.py`: 纯函数验证器，A/B/C/D 四项检查
  - `core/smoke_test_harness.py`: MPI 感知的测试编排器
  - `scripts/test_smoke_run.py`: 独立 CLI 脚本
  - `run_refactored.py` 新增 `--smoke-test` 参数

### v2.4 (2026-06-03)

- **输出文件精简重构**
  - `reaction_frames.npz` 从 11 数组精简至 7 数组
  - 移除 `bonds_records/` 目录输出
  - 移除 `changed_bead_id_list.pkl` 输出
  - 向后兼容旧格式 npz

### v2.3 (2026-05-19)

- **文档完善**
  - 新增模块依赖关系图和数据流说明
  - 新增所有 CLI 脚本说明
  - 新增 CGTopology、ReactionLocator API 参考

### v2.2 (2026-04-29)

- **新增 CLI 脚本**: `calc_dist.py`, `plot_dist.py`, `yaml2csv_mapping.py`
- **新增**: `CGInitializer` / `CGSystem` / `initialize_cg_system` 模块
- **smooth_utils 扩展**: 新增 11 个子模块

### v2.1 (2026-04-02)

- **新增工具模块**: `aa2cg`, `ibm_potential`, `smooth_utils`
- **新增 CLI 脚本**: 对应工具模块的 CLI 入口
- **新增工具函数**: `topology.py`, `units.py`

### v2.0 (2026-03-28)

- **完全重构**: 支持多体系、配置驱动设计、向量化/Numba 优化
- **新增**: 断点续算功能、`CGCompareList.from_csv()`
- **模块化架构**: 清晰的 core/utils/tools/scripts 划分

---

## 许可证

MIT License
