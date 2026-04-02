# LmpPy - LAMMPS Bond/React 后处理框架

> 版本: 2.0
> 作者: Claude
> 日期: 2026-03-28

## 目录

- [项目简介](#项目简介)
- [项目结构](#项目结构)
- [依赖项与安装](#依赖项与安装)
- [快速开始](#快速开始)
- [运行方式](#运行方式)
- [配置文件详解](#配置文件详解)
- [CG映射配置](#cg映射配置)
- [继续计算功能](#继续计算功能)
- [输出文件说明](#输出文件说明)
- [模块API参考](#模块api参考)
- [使用教程](#使用教程)
- [常见问题](#常见问题)

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
├── core/                          # 核心模块
│   ├── __init__.py                # 模块导出
│   ├── config_loader.py           # 配置加载器
│   ├── mapping_generator.py       # CG映射生成器
│   ├── template_parser.py         # 模板解析器
│   ├── lammps_data_extractor.py   # LAMMPS数据提取器
│   ├── bond_detector.py           # 键变化检测器
│   ├── reaction_locator.py        # 反应位点定位器
│   ├── cg_mapper.py               # CG映射更新器
│   ├── cg_converter.py            # CG坐标转换器
│   ├── cg_bond_mapper.py          # 粗粒键映射器
│   ├── cg_topology.py             # CG拓扑推导
│   ├── cg_initializer.py          # CG系统初始化
│   └── bonds_recorder.py          # 键连表输出器
├── utils/                         # 工具函数
│   ├── __init__.py
│   ├── coordinate_utils.py        # 坐标处理
│   ├── file_utils.py              # 文件I/O
│   └── graph_utils.py             # 图论算法
├── docs/                          # 文档和示例
│   └── examples/                  # 示例配置文件
├── run_refactored.py              # 主程序入口
├── test_integration.py            # 集成测试
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
└── rxn1_pre_mapping.yaml    # 反应前CG映射
    rxn1_post_mapping.yaml   # 反应后CG映射
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
  output_bonds_record: "bonds_record.npz"

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

### 主要输出文件

| 文件 | 格式 | 说明 |
|------|------|------|
| `cg_trajectory.lammpstrj` | LAMMPS dump | CG轨迹文件，可VMD可视化 |
| `reaction_num.txt` | 文本 | 每步反应计数统计 |
| `final_frame.data` | LAMMPS data | 最终帧原子数据 |
| `final_cg_compare_list.csv` | CSV | 最终CG映射关系 |
| `bonds_record.npz` | NPZ | 键连表记录（含反应帧数据） |

### reaction_num.txt 格式

```
# timestep rxn1 rxn2 rxn3 ...
1 0 0 0
101 1 0 0
201 2 1 0
...
```

### final_cg_compare_list.csv 格式

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

### bonds_record.npz 内容

```python
import numpy as np
data = np.load('bonds_record.npz')

# 可用的数组
data['aa_bonds_before']    # 反应前原子键 (object数组)
data['aa_bonds_after']     # 反应后原子键
data['cg_bonds_before']    # 反应前CG键
data['cg_bonds_after']     # 反应后CG键
data['timestep']           # 反应发生的时间步
```

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
    BondsRecorder, save_bonds_record,
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
    print(f"新增键: {len(changes.bonds_formed)}")
    print(f"断裂键: {len(changes.bonds_broken)}")

# 或使用便捷函数
bonds_formed, bonds_broken = compare_two_bonds(bonds_before, bonds_after)
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
cg_bonds = atom_bonds_to_cg_bonds(
    atom_bonds,        # 原子键列表 [(atom1, atom2), ...]
    cg_compare_list    # CG映射
)
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

# 读取键记录
bonds_data = np.load('bonds_record.npz', allow_pickle=True)
print(f"记录的反应次数: {len(bonds_data['timestep'])}")
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

## 许可证

MIT License

---

## 更新日志

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