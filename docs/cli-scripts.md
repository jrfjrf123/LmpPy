# CLI 脚本使用指南

> 版本: 2.7
> 更新: 2026-09-11

本文档列出 LmpPy 项目所有 CLI 脚本的用途、参数和使用示例。

> 参数表以脚本 `--help` 输出为准；新增或修改参数后请同步更新本文件。

---

## 目录

| 脚本 | 用途 |
|------|------|
| [run_refactored.py](#1-run_refactoredpy) | 主入口：LAMMPS 反应模拟后处理 |
| [build_cg_config.py](#2-build_cg_configpy) | CG 体系构建配置解析 |
| [build_cg_system.py](#3-build_cg_systempy) | 构建 CG 体系 LAMMPS data 文件 |
| [calc_dist.py](#4-calc_distpy) | 分布计算（VOTCA 格式） |
| [calc_ibm_potential.py](#5-calc_ibm_potentialpy) | IBM 势能计算 |
| [calc_ibm_potential_from_dist.py](#6-calc_ibm_potential_from_distpy) | 从分布文件计算 IBM 势能 |
| [convert_aa2cg.py](#7-convert_aa2cgpy) | AA 到 CG 转换 |
| [data2gro.py](#8-data2gropy) | LAMMPS data 转 GRO |
| [generate_initial_mapping.py](#9-generate_initial_mappingpy) | 生成初始 CG 映射 CSV |
| [plot_dist.py](#10-plot_distpy) | 分布绘图 |
| [smooth_distribution.py](#11-smooth_distributionpy) | 分布平滑 |
| [test_smoke_run.py](#12-test_smoke_runpy) | 冒烟测试独立运行 |
| [validate_config.py](#13-validate_configpy) | 配置验证 |
| [yaml2csv_mapping.py](#14-yaml2csv_mappingpy) | YAML 到 CSV 映射转换 |
| [gmx2lmp_data.py](#15-gmx2lmp_datapy) | GROMACS top+gro 转 LAMMPS data |
| [fit_tabulated.py](#16-fit_tabulatedpy) | tabulated 势能表拟合为解析势 |
| [mix_cross_tabulated.py](#17-mix_cross_tabulatedpy) | 交叉混合生成异核势能表 |
| [plot_tabulated.py](#18-plot_tabulatedpy) | tabulated 势能表绘图 |
| [extract_reaction_frame.py](#19-extract_reaction_framepy) | 从 MD 输出抽取反应帧详情 |

---

## 源码位置

| 路径 | 说明 |
|------|------|
| `/home/large_storage/jrf/lmp_py_react/LmpPy/run_refactored.py` | 主入口脚本 |
| `/home/large_storage/jrf/lmp_py_react/LmpPy/scripts/` | 所有工具脚本 |
| `/home/large_storage/jrf/lmp_py_react/LmpPy/core/cg_bond_mapper.py` | `atom_bonds_to_cg_bonds()` 函数 |

---

## 1. run_refactored.py

**用途**：LAMMPS `bond/react` 模拟后处理主入口。执行完整的模拟管线：CG 映射初始化、主循环（反应检测 + CG 更新 + 弛豫）、输出文件生成。

**位置**：`LmpPy/run_refactored.py`

**参数**：

| 参数 | 类型 | 说明 | 默认值 |
|------|------|------|--------|
| `config_dir` | str | 配置文件目录路径（位置参数） | 必需 |
| `--test` | flag | 测试模式：仅加载配置，不运行 LAMMPS | 关闭 |
| `--loop-num N` | int | 覆盖循环次数（用于快速测试） | 来自配置 |
| `--smoke-test` | flag | 冒烟测试模式：在临时目录跑少量 loop 并验证输出 | 关闭 |

**使用示例**：

```bash
# 单进程运行
python -m LmpPy.run_refactored config/

# MPI 并行运行
mpirun -np 4 python -m LmpPy.run_refactored config/ --loop-num 100

# 测试模式：仅加载配置
python -m LmpPy.run_refactored config/ --test

# 冒烟测试
python -m LmpPy.run_refactored config/ --smoke-test --loop-num 5
```

---

## 2. build_cg_config.py

**用途**：`build_cg_system.py` 的 YAML 配置解析模块。定义 `MoleculeSpec` 和 `BuildCGConfig` dataclass，提供 `from_yaml()` 类方法读取和校验 YAML 配置。也可作为独立模块导入使用。

**位置**：`LmpPy/scripts/build_cg_config.py`

**说明**：此脚本通常不作为独立 CLI 使用，而是被 `build_cg_system.py` 导入。YAML 配置格式如下：

```yaml
molecules:
  - name: "initiator"
    count: 1000
    bonds_file: "initiator_bonds.txt"
    angles_file: "initiator_angles.txt"      # 可选
    dihedrals_file: "initiator_dihedrals.txt" # 可选

masses:
  1: 12.01
  2: 12.01

types:
  initiator: [2, 1, 1, 1, 1, 1]

output: "system.data"
```

**Python API 使用**：

```python
from LmpPy.scripts.build_cg_config import BuildCGConfig

config = BuildCGConfig.from_yaml("build_config.yaml")
for mol in config.molecules:
    print(mol.name, mol.count, mol.bonds_file)
```

---

## 3. build_cg_system.py

**用途**：根据单分子拓扑文件和用户指定的分子数量，构建完整多分子体系的 LAMMPS `.data` 文件。支持 YAML 配置模式和命令行模式。

**位置**：`LmpPy/scripts/build_cg_system.py`

**参数**：

| 参数 | 说明 |
|------|------|
| `--yaml PATH` | YAML 配置文件路径（与 `--mol` 互斥） |
| `--mol SPEC` | 分子规格：`name:bonds_file:count`（可重复） |
| `--types SPEC` | 分子 bead 类型：`name:type1,type2,...` |
| `--masses MAP` | 原子类型质量：`1:12.01,2:12.01,3:12.01` |
| `--gro PATH` | GRO 文件路径（提供坐标和类型，可选） |
| `-o` / `--output PATH` | 输出 LAMMPS .data 文件路径（必需） |

**使用示例**：

```bash
# YAML 模式
python -m LmpPy.scripts.build_cg_system --yaml build_config.yaml -o system.data

# 命令行模式（提供 GRO 坐标）
python -m LmpPy.scripts.build_cg_system \
    --mol initiator:cg_initiator_bonds.txt:1000 \
    --mol monomer::100000 \
    --masses 1:12.01,2:12.01,3:12.01 \
    --gro system.gro \
    -o system.data

# 命令行模式（不提供 GRO，通过 --types 指定）
python -m LmpPy.scripts.build_cg_system \
    --mol initiator:cg_initiator_bonds.txt:1000 \
    --mol monomer::100000 \
    --types initiator:2,1,1,1,1,1 \
    --types monomer:3 \
    --masses 1:12.01,2:12.01,3:12.01 \
    -o system.data
```

**说明**：
- `--mol` 中 bonds 文件可为空（如 `monomer::100000`），表示单 bead 分子
- 自动从 bonds 推导 angles 和 dihedrals
- 输出 `atom_style molecular` 的 LAMMPS data 文件

---

## 4. calc_dist.py

**用途**：计算键/角度/二面角/RDF 分布，输出 VOTCA 格式（`.dist.tgt`）。支持 LAMMPS dump、Pickle 和 GROMACS（XTC/TRR）格式轨迹。

**位置**：`LmpPy/scripts/calc_dist.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--traj PATH` | CG 轨迹文件（`.pkl` 或 `.lammpstrj`） | — |
| `--tpr PATH` | GROMACS TPR 拓扑文件（GROMACS 模式） | — |
| `--xtc` / `--trr PATH` | GROMACS XTC/TRR 轨迹文件（同一参数的别名） | — |
| `--top-dir PATH` | 拓扑文件目录（含 `cg_bonds.txt`, `cg_angles.txt` 等） | — |
| `--output-dir PATH` | 输出目录 | `./dist_output` |
| `--stride N` | 帧间隔（跳帧） | 1 |
| `--n-bins N` | 直方图 bins 数 | 100 |
| `--unit {A,nm}` | 轨迹距离单位 | `A` |
| `--bond-range MIN MAX` | 键距离范围（与轨迹单位一致） | 2.0 6.0 |
| `--angle-range MIN MAX` | 角度范围（度） | 0 180 |
| `--dihedral-range MIN MAX` | 二面角范围（度） | -180 180 |
| `--calc-pairs` | 计算所有 bead type 对的距离分布（排除 1-2/1-3/1-4） | 关闭 |
| `--pair-range MIN MAX` | Pair 距离范围 | 3.0 15.0 |
| `--exclude-12` / `--exclude-13` / `--exclude-14` | 排除 1-2 / 1-3 / 1-4 对 | 关闭 |
| `--n-jobs N` | 并行进程数（仅用于 RDF 计算） | 1（串行） |
| `--skip-existing` | 跳过已存在的输出文件 | 关闭 |
| `--vectorized` | 全向量化计算（仅对 `.pkl` 有效，快但费内存） | 关闭 |
| `--ordered-bond-type` | 测试模式：区分键方向性，不合并 (t1,t2) 与 (t2,t1) | 关闭 |
| `--norm-max` / `--norm-area` | 最大值归一化 / 面积归一化（互斥） | 关闭 |

**输入文件格式**：

| 文件 | 格式 |
|------|------|
| `cg_bonds.txt` | `bond_type atom1 atom2` |
| `cg_angles.txt` | `angle_type atom1 atom2 atom3` |
| `cg_dihedrals.txt` | `dihedral_type atom1 atom2 atom3 atom4` |
| `cg_bead_info.txt` | `bead_id mol_id bead_type mass` |

**输出格式（VOTCA `.dist.tgt`）**：

```
r/theta/phi  probability  i
```

**使用示例**：

```bash
# LAMMPS 格式
python -m LmpPy.scripts.calc_dist --traj cg_trajectory.pkl --top-dir ./
python -m LmpPy.scripts.calc_dist --traj cg_trajectory.lammpstrj --top-dir ./ --output-dir dist_output

# GROMACS 格式
python -m LmpPy.scripts.calc_dist --tpr topol.tpr --xtc traj.xtc --top-dir ./ --stride 10
python -m LmpPy.scripts.calc_dist --tpr topol.tpr --trr traj.trr --stride 5 --unit nm

# 计算 pair 分布并并行加速
python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --calc-pairs --n-jobs 8

# 增量重跑：跳过已算好的分布文件
python -m LmpPy.scripts.calc_dist --traj traj.pkl --top-dir ./ --skip-existing
```

---

## 5. calc_ibm_potential.py

**用途**：完整的 IBM（Iterative Boltzmann Inversion）势能计算管线。从 CG 轨迹加载、分布计算、玻尔兹曼反演到 LAMMPS table 文件生成。

**位置**：`LmpPy/scripts/calc_ibm_potential.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-c` / `--config` | 配置文件路径（`ibm_potential.yaml`） | 可选 |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**使用示例**：

```bash
# 使用配置文件
python -m LmpPy.scripts.calc_ibm_potential -c ibm_potential.yaml

# 安静模式
python -m LmpPy.scripts.calc_ibm_potential -c ibm_potential.yaml -q
```

**配置文件参考**（`ibm_potential.yaml`）：

```yaml
simulation:
  temperature: 400
  units: real

trajectory:
  format: pickle
  path: cg_trajectory.pkl

distribution:
  n_bins: 200
  bond_range: [0.5, 6.0]
  angle_range: [0, 180]
  dihedral_range: [-180, 180]

smoothing:
  method: auto
  sg_window: 21
  sg_polyorder: 3

output:
  table_dir: lammps_tables
  potentials_dir: potentials_output
  distributions_dir: distributions_output
```

---

## 6. calc_ibm_potential_from_dist.py

**用途**：从预计算的分布文件（平滑后的 `_dist.txt` 或 VOTCA `*.dist.tgt`）计算 IBM 势能。适用于需要重新计算势能而不重新跑分布计算的场景。

**位置**：`LmpPy/scripts/calc_ibm_potential_from_dist.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-d` / `--distribution-dir` | 分布文件目录 | `smoothed_output/` |
| `-o` / `--output-dir` | 势能输出目录 | `potentials_output/` |
| `-t` / `--temperature` | 温度（K） | 400.0（或从文件头提取） |
| `-u` / `--units {kcal/mol,kJ/mol,eV}` | 能量单位 | `kcal/mol` |
| `--lammps-tables` | 是否生成 LAMMPS table 文件 | 关闭 |
| `--plot` | 是否生成势能图 | 关闭 |
| `-c` / `--config` | YAML 配置文件路径 | — |
| `--no-jacobian` | 禁用 Jacobian 校正 | 关闭 |
| `--smooth-window N` | Savitzky-Golay 窗口大小 | 21 |
| `--smooth-polyorder N` | Savitzky-Golay 多项式阶数 | 3 |
| `--extrap-method {linear,exponential}` | 边界外推方法 | `linear` |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**输入格式支持**：

1. VOTCA 原始格式（`*.dist.tgt`）：3 列 `x P i`
2. 平滑格式（`*_dist.txt`）：5 行元数据头 + 2 列数据

**使用示例**：

```bash
# 从平滑后的分布计算势能
python -m LmpPy.scripts.calc_ibm_potential_from_dist -d smoothed_output/

# 指定温度并生成 LAMMPS table
python -m LmpPy.scripts.calc_ibm_potential_from_dist -d dist_output/ -t 400 --lammps-tables --plot
```

---

## 7. convert_aa2cg.py

**用途**：全原子（AA）到粗粒化（CG）转换工具。支持三种输入格式：GROMACS TRR 轨迹、LAMMPS data 文件、GRO+TPR 结构文件。

**位置**：`LmpPy/scripts/convert_aa2cg.py`

**子命令**：

| 子命令 | 功能 |
|--------|------|
| `trj` | GROMACS TRR 轨迹转换 |
| `data` | LAMMPS data 文件转换 |
| `gro` | GRO + TPR 结构文件转换 |

### trj 子命令参数

| 参数 | 说明 |
|------|------|
| `--tpr PATH` | GROMACS TPR 拓扑文件 |
| `--trr PATH` | GROMACS TRR 轨迹文件 |
| `--mapping PATH` | CG 映射 CSV 文件 |
| `-o PATH` | 输出 CG 轨迹 pickle 文件 |
| `--stride N` | 帧间隔 |
| `--xyz` | 同时输出 XYZ 格式（最后一帧） |

### data 子命令参数

| 参数 | 说明 |
|------|------|
| `-i` / `--input PATH` | 输入 LAMMPS data 文件 |
| `-m` / `--mapping PATH` | CG 映射 CSV 文件 |
| `-o` / `--output PATH` | 输出 CG data 文件 |
| `--bonds PATH` | CG 键文件（提供时优先，忽略 `--derive-topology`） |
| `--angles PATH` | CG 角度文件 |
| `--dihedrals PATH` | CG 二面角文件 |
| `--derive-topology` | 从 AA 键自动推导 CG 拓扑（bonds/angles/dihedrals） |
| `--output-cg-topology DIR` | 输出 CG 拓扑文件目录 |
| `--type-mapping PATH` | YAML 类型映射文件（仅 `--derive-topology` 下生效） |
| `--xyz` | 同时输出 XYZ 格式 |

### gro 子命令参数

| 参数 | 说明 |
|------|------|
| `-g` / `--gro PATH` | GROMACS GRO 文件 |
| `-t` / `--tpr PATH` | GROMACS TPR 文件 |
| `-m` / `--mapping PATH` | CG 映射 CSV 文件 |
| `-o` / `--output PATH` | 输出 CG data 文件 |
| `--derive-topology` | 从 AA 键自动推导 CG 拓扑（bonds/angles/dihedrals） |
| `--output-cg-topology DIR` | 输出 CG 拓扑文件目录 |
| `--type-mapping PATH` | YAML 类型映射文件（仅 `--derive-topology` 下生效） |
| `--xyz` | 输出 XYZ 文件（`AA_unwrap.xyz` / `CG.xyz`） |

**使用示例**：

```bash
# 转换轨迹（GROMACS TRR）
python -m LmpPy.scripts.convert_aa2cg trj \
    --tpr topol.tpr \
    --trr traj.trr \
    --mapping mapping.csv \
    -o cg_trajectory.pkl

# 转换 data 文件
python -m LmpPy.scripts.convert_aa2cg data \
    --input system.data \
    --mapping mapping.csv \
    --bonds cg_bonds.txt \
    -o cg.data

# Data 文件转换 + 自动推导拓扑 + XYZ 输出
python -m LmpPy.scripts.convert_aa2cg data \
    --input system.data \
    --mapping mapping.csv \
    --derive-topology --output-cg-topology . \
    --xyz

# GRO + TPR 转换
python -m LmpPy.scripts.convert_aa2cg gro \
    --gro system.gro \
    --tpr topol.tpr \
    --mapping mapping.csv \
    -o cg.data
```

---

## 8. data2gro.py

**用途**：将 LAMMPS data 文件（`atom_style molecular` 或 `full`）转换为 GROMACS `.gro` 文件。

**位置**：`LmpPy/scripts/data2gro.py`

**参数**：

| 参数 | 说明 |
|------|------|
| `input.data` | 输入 LAMMPS data 文件（位置参数） |
| `output.gro` | 输出 GRO 文件（位置参数） |
| `--residue-name NAME` | 残基名称（默认按 molecule-ID 命名为 `MOLxx`） |
| `--unit UNIT` | 输入单位：`angstrom`（默认）或 `nm` |

**说明**：
- LAMMPS 坐标单位通常为 angstrom，GRO 文件坐标单位为 nm，脚本默认除以 10 转换
- 使用 `--unit angstrom` 可显式指定输入单位为 angstrom（默认值）

**使用示例**：

```bash
# 基本用法
python -m LmpPy.scripts.data2gro input.data output.gro

# 指定残基名称
python -m LmpPy.scripts.data2gro input.data output.gro --residue-name POLY

# 输入为 nm
python -m LmpPy.scripts.data2gro input.data output.gro --unit nm
```

---

## 9. generate_initial_mapping.py

**用途**：从配置目录加载 system.yaml 和 mapping 文件，生成初始 CG 映射 CSV 文件。使用 `ConfigLoader` 和 `generate_cg_compare_list` 实现。

**位置**：`LmpPy/scripts/generate_initial_mapping.py`

**参数**：

| 参数 | 说明 |
|------|------|
| `config_dir` | 配置目录路径（可选，默认使用 `test_EPR_LmpPy/config`） |

**使用示例**：

```bash
# 使用默认配置目录
python -m LmpPy.scripts.generate_initial_mapping

# 指定配置目录
python -m LmpPy.scripts.generate_initial_mapping my_system/config
```

---

## 10. plot_dist.py

**用途**：绘制分布文件图像，支持 VOTCA `.dist.tgt` 格式和两列数据格式。可生成单个分布图和组合汇总图。

**位置**：`LmpPy/scripts/plot_dist.py`

**参数**：

| 参数 | 说明 |
|------|------|
| `files` | 输入分布文件列表（位置参数，可多个） |
| `-d` / `--directory` | 处理目录下所有分布文件 |
| `-o` / `--output` | 输出目录 | `./plots` |
| `--types` | 分布类型：`bond` `angle` `dihedral` `pair` `all` | `all` |
| `--individual-only` | 只绘制单独的图（不绘制汇总图） | 关闭 |
| `--combined-only` | 只绘制汇总图 | 关闭 |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**使用示例**：

```bash
# 处理目录下所有分布文件
python -m LmpPy.scripts.plot_dist -d ./dist_output

# 指定输出目录
python -m LmpPy.scripts.plot_dist -d ./dist_output -o ./plots

# 只处理特定类型
python -m LmpPy.scripts.plot_dist -d ./dist_output --types bond angle

# 处理指定文件
python -m LmpPy.scripts.plot_dist bond_type1.dist.tgt bond_type2.dist.tgt

# 只绘制汇总图
python -m LmpPy.scripts.plot_dist -d ./dist_output --combined-only
```

---

## 11. smooth_distribution.py

**用途**：对 bond/angle/dihedral/RDF 分布进行平滑处理。支持 LAMMPS/IBI 格式（`*_dist.txt`）和 VOTCA 格式（`*.dist.tgt`）。使用 Savitzky-Golay 滤波或 Gaussian 平滑方法。

**位置**：`LmpPy/scripts/smooth_distribution.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `files` | 输入分布文件列表（位置参数，可多个） | — |
| `-d` / `--directory` | 处理目录下所有分布文件 | — |
| `-o` / `--output` | 输出目录 | `smoothed_output` |
| `-T` / `--temperature` | 温度（K） | 400 |
| `--input-dist-unit {A,nm}` | 输入距离单位 | `A` |
| `--input-ang-unit {deg,rad}` | 输入角度单位 | `deg` |
| `--output-dist-unit {A,nm}` | 输出距离单位 | `A` |
| `--output-ang-unit {deg,rad}` | 输出角度单位 | `deg` |
| `-t` / `--threshold` | 输出阈值，小于该值的 x 和 P 置 0 | `1e-8` |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**支持的文件格式**：

| 格式 | 文件模式 | 说明 |
|------|----------|------|
| LAMMPS/IBI | `*_dist.txt` | 如 `bond_type1_dist.txt` |
| VOTCA | `*.dist.tgt` | 如 `bond_type1.dist.tgt` |

**使用示例**：

```bash
# 处理单个文件（LAMMPS/IBI 格式）
python -m LmpPy.scripts.smooth_distribution bond_type1_dist.txt -o smoothed_output/

# 处理单个文件（VOTCA 格式）
python -m LmpPy.scripts.smooth_distribution bond_type1.dist.tgt -o smoothed_output/

# 处理目录下所有分布文件
python -m LmpPy.scripts.smooth_distribution -d distributions/ -o smoothed_output/

# 指定温度
python -m LmpPy.scripts.smooth_distribution bond_type1.dist.tgt -T 300 -o smoothed_output/
```

---

## 12. test_smoke_run.py

**用途**：独立冒烟测试运行脚本。调用 `SmokeTestHarness` 在临时目录中运行少量 loop，并自动验证输出文件完整性和一致性。

**位置**：`LmpPy/scripts/test_smoke_run.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `config_dir` | 配置文件目录路径 | 必需 |
| `--loop-num N` | 循环次数 | 5 |
| `--keep-output` | 保留测试输出目录（默认自动删除） | 关闭 |

**使用示例**：

```bash
# 基本用法（默认 5 个 loop）
python LmpPy/scripts/test_smoke_run.py my_system/

# 指定循环次数并保留输出
python LmpPy/scripts/test_smoke_run.py my_system/ --loop-num 3 --keep-output
```

**出口码**：
- 0：测试通过
- 1：测试未通过

---

## 13. validate_config.py

**用途**：验证配置目录的完整性。检查所有必需文件是否存在、YAML 格式是否正确、交叉引用是否一致。

**位置**：`LmpPy/scripts/validate_config.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `config_dir` | 配置文件目录路径 | 必需 |
| `-f` / `--format` | 输出格式：`text` 或 `json` | `text` |
| `-q` / `--quiet` | 安静模式，仅输出错误和警告 | 关闭 |
| `--no-report` | 不输出报告，仅返回退出码 | 关闭 |
| `--exit-on-error` | 有错误时返回非零退出码 | 启用 |

**使用示例**：

```bash
# 验证配置并输出详细报告
python -m LmpPy.scripts.validate_config config/

# JSON 格式输出
python -m LmpPy.scripts.validate_config config/ --format json

# 安静模式，仅输出错误和警告
python -m LmpPy.scripts.validate_config config/ --quiet

# 不输出报告，仅返回退出码（用于脚本）
python -m LmpPy.scripts.validate_config config/ --no-report && echo "验证通过"
```

**JSON 输出示例**：

```json
{
  "config_dir": "/path/to/config",
  "passed": true,
  "n_errors": 0,
  "n_warnings": 1,
  "issues": [
    {
      "level": "warning",
      "category": "文件检查",
      "field": "mapping/mapping_epoxy.yaml",
      "message": "未使用全局 bead_type_names 映射",
      "suggestion": "建议在 system.yaml 中定义 bead_type_names"
    }
  ]
}
```

---

## 14. yaml2csv_mapping.py

**用途**：将 YAML 格式的 CG 映射文件（site-types + config）转换为 CSV 格式。

> **⚠️ `-s` 不接收平台的 `system.yaml`。** 脚本读的是 `system.names`（mapping 文件路径
> 列表）与 `system.numbers`（对应 copies 数）（`scripts/yaml2csv_mapping.py:127-128`），
> 而平台 `system.yaml` 用的是 `system.mapping_files: [{path, copies}]`。传入平台
> `system.yaml` 会抛 `KeyError: 'names'`。**新体系请改用
> `python -m LmpPy.scripts.generate_initial_mapping`（见 §9）**，它走 ConfigLoader，
> 支持当前格式。

**位置**：`LmpPy/scripts/yaml2csv_mapping.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-s` / `--system-yaml` | **旧格式** system.yaml 路径（需含 `system.names`/`system.numbers`） | 必需 |
| `-o` / `--output` | 输出 CSV 文件名 | `AtomId_BeadId_compare_list.csv` |
| `--custom-map FILE` | 从 YAML/JSON 文件加载自定义 bead type 映射 `{site_type_name: bead_type_id}` | — |
| `--custom` | 使用硬编码的自定义 bead type 映射（EPR，向后兼容） | 关闭 |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**使用示例**：

```bash
# 基本用法（按 site 出现顺序自动连续编号）
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml -o mapping.csv

# 从文件加载自定义 bead type 映射（推荐）
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml --custom-map bead_type_map.yaml

# 使用内置硬编码映射（EPR，向后兼容）
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml --custom -o custom_mapping.csv
```

**输出列**：`bead_id, mol_id, bead_type, AA_id, mass`

**自定义 bead type 映射示例**（在脚本中修改 `CUSTOM_BEAD_TYPE_MAP` 字典）：

```python
CUSTOM_BEAD_TYPE_MAP = {
    "Bead1": 1,
    "Bead2": 2,
    "Bead3": 1,
    "Bead4": 2,
    "Bead5": 3,
    "Bead6": 4,
    "Bead7": 1,
    "Bead8": 2,
}
```

---

## 15. gmx2lmp_data.py

GROMACS top（可含 `#include` itp）+ gro → LAMMPS data 文件（GAFF 力场、real 单位、`atom_style full`）。纯 Python 标准库实现，无第三方依赖。

```bash
python LmpPy/scripts/gmx2lmp_data.py --top system.top --gro conf.gro -o out.data
python LmpPy/scripts/gmx2lmp_data.py --top system.top --gro conf.gro -o out.data \
    --type-order type_order.txt   # 可选：输出 类型号=GAFF类型名 映射
```

**参数：**

| 参数 | 说明 | 必需 |
|------|------|------|
| `--top PATH` | GROMACS `.top` 文件路径（可含 `#include` itp） | 是 |
| `--gro PATH` | GROMACS `.gro` 文件路径（正交盒） | 是 |
| `-o` / `--out PATH` | 输出 LAMMPS `.data` 文件路径 | 是 |
| `--type-order PATH` | 可选：输出 `类型号=GAFF类型名` 映射文件 | 否 |

**支持范围：**

- 递归 `#include`（相对包含文件所在目录）；`[ molecules ]` 多分子计数展开
- `[ atomtypes ]` 6 列 / 7 列（含 at.num）两种格式；comb-rule 1（C6/C12 自动换算 σ/ε）与 comb-rule 2
- bonds/angles funct 1、dihedrals funct 9/1（proper，多 term 保留叠加）、funct 4（improper → cvff）
- 正交盒；全零盒按坐标范围每边加 1 nm（总长 +2 nm），最小 3 nm
- `Masses` 段使用 `[ atomtypes ]` 质量，`[ atoms ]` 行的质量覆盖仅作校验/告警
- 不支持的 functype（2/3/5/10 等）报错退出；`[ pairs ]`/`[ constraints ]` 等段跳过并告警（1-4 缩放由输出文件头注释建议的 `special_bonds` 覆盖）

输出文件头注释列出了所需的 `pair_style`/`bond_style` 等 LAMMPS 设置与 `special_bonds` 建议值，直接照抄到输入脚本即可。

---

## 16. fit_tabulated.py

**用途**：把 LAMMPS/VOTCA tabulated 势能表拟合成解析势函数（LJ 12-6、harmonic bond、cosine/harmonic angle）。是 IBI 迭代中「从表格势反推解析形式」这一步。

**位置**：`LmpPy/scripts/fit_tabulated.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `files` | 输入势能表文件列表（位置参数，可多个） | — |
| `-d` / `--directory` | 处理目录下所有势能表文件 | — |
| `-o` / `--output` | 输出目录 | `./fit_output` |
| `--energy-unit UNIT` | 能量单位标签（仅标注，不做换算） | `kcal/mol` |
| `--angle-form {auto,cos,harmonic}` | angle 拟合形式 | `auto` |
| `--u-cut U` | bond/angle 势阱区选择阈值，只拟合 `U-Umin <= u_cut` 的点 | 10.0 |
| `--rmin-fit R` / `--rmax-fit R` | nonbonded 拟合区间端点（Å），指定后跳过区间探索 | — |
| `--lj-method {well,direct,explore}` | nonbonded 取参方式 | `well` |
| `--no-plot` | 不生成对比图 | 关闭 |
| `--y-max Y` | nonbonded 对比图能量上限 | — |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**`--lj-method` 三种策略**：

- `well`（默认）：势阱特征取参，以壳层间势垒顶为零点（`eps = E_barr - E_well`，`sigma` 取壁上 `E = E_barr` 的过点，`cutoff = r_barr`）
- `direct`：尾部基线归零后取 `U = 0` 第一个过零点与势阱深度
- `explore`：最小二乘区间探索

**使用示例**：

```bash
# 拟合目录下所有势能表
python -m LmpPy.scripts.fit_tabulated -d ./mdrun -o ./fit_output

# 拟合指定文件
python -m LmpPy.scripts.fit_tabulated nb11.pot.table bond1.pot.table angle1.pot.table

# 指定 nonbonded 拟合区间，跳过区间探索
python -m LmpPy.scripts.fit_tabulated -d ./mdrun --rmin-fit 3.5 --rmax-fit 12.0
```

---

## 17. mix_cross_tabulated.py

**用途**：交叉混合两个 tabulated 势能表，生成异核（cross）势能表（如 `nb11` + `nb33` → `nb13`）。异核势无法从 CG 模拟直接测得，只能由同核对混合得到。

**位置**：`LmpPy/scripts/mix_cross_tabulated.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `table_a` / `table_b` | 两个输入势能表（位置参数） | 必需 |
| `-o` / `--output` | 输出势能表路径 | 必需 |
| `--n-grid N` | 公共网格点数 | `max(N_A, N_B) + 1` |
| `--tail-frac X` | 尾部归零化占比 | 0.1 |
| `--mix-mode {geometric,arithmetic}` | 能量组合方式 | `geometric` |
| `--opposite-sign {force,sign_a}` | 异号区处理（仅 geometric 有效） | `force` |
| `--dump-corrected DIR` | 同时输出修正后的输入表到 DIR | — |

**混合规则**（`--mix-mode geometric`，工作区 `docs/cross-tabulation-mixing.md` v2.0 规范）：

```
sigma_mix = (sigma_A + sigma_B) / 2
eps_mix   = sqrt(eps_A * eps_B)
V*_mix    = sign(V*_A) * sqrt(|V*_A * V*_B|)
```

**使用示例**：

```bash
python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb33.pot.table -o nb13.pot.table
python -m LmpPy.scripts.mix_cross_tabulated nb22.pot.table nb44.pot.table -o nb24.pot.table
```

---

## 18. plot_tabulated.py

**用途**：绘制 LAMMPS/VOTCA tabulated 势能表及其力曲线，用于人工检查势阱深度、零点与截断位置。

**位置**：`LmpPy/scripts/plot_tabulated.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `files` | 输入势能表文件列表（位置参数，可多个） | — |
| `-d` / `--directory` | 处理目录下所有势能表文件 | — |
| `-o` / `--output` | 输出目录 | `./plots` |
| `--types {bond,angle,nonbonded,dihedral,all} [...]` | 势能类型 | `all` |
| `--no-force` | 不绘制力曲线 | 关闭 |
| `--energy-unit UNIT` | 能量单位标签（仅坐标轴标注） | `kcal/mol` |
| `--y-max Y` | nonbonded 势绘图能量上限，放大势阱区域 | — |
| `--xlim MIN MAX` / `--ylim MIN MAX` | 默认横/纵坐标范围（所有类型） | — |
| `--xlim-nb` / `--ylim-nb`（全称 `--xlim-nonbonded` / `--ylim-nonbonded`） | nonbonded 势坐标范围 | — |
| `--xlim-bond` / `--ylim-bond` / `--xlim-angle` / `--ylim-angle` / `--xlim-dihedral` / `--ylim-dihedral` | 按类型分别指定坐标范围 | — |
| `-q` / `--quiet` | 安静模式 | 关闭 |

**使用示例**：

```bash
# 处理目录下所有势能表
python -m LmpPy.scripts.plot_tabulated -d ./fit_output

# 只看 nonbonded 势阱区域
python -m LmpPy.scripts.plot_tabulated -d ./fit_output --types nonbonded --y-max 5

# 不画力曲线
python -m LmpPy.scripts.plot_tabulated bond1.pot.table --no-force
```

---

## 19. extract_reaction_frame.py

**用途**：从 LmpPy 输出目录随机抽取一个反应帧，输出该帧的详细原子/键变化信息，便于用 VMD/OVITO 等做可视化检查。

**位置**：`LmpPy/scripts/extract_reaction_frame.py`

**参数**：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `md_dir` | 含 `cg_trajectory.lammpstrj` 和 `reaction_frames.npz` 的目录 | 必需 |
| `--seed N` | 随机种子（用于可重复的随机选择） | — |
| `--output-dir DIR` | 输出目录 | `md_dir` |
| `--output-prefix PREFIX` | 输出文件前缀 | — |

**使用示例**：

```bash
python -m LmpPy.scripts.extract_reaction_frame output/md1 --seed 42 --output-dir ./frame_dump
```

---

## 附录：脚本执行环境说明

所有脚本需在 `/home/large_storage/jrf/lmp_py_react` 目录下执行，以确保跨仓库导入正确。脚本内部通过以下方式自动添加项目根目录到 `sys.path`：

```python
_project_root = Path(__file__).resolve().parent.parent.parent
if _project_root not in sys.path:
    sys.path.insert(0, str(_project_root))
```

例外：`gmx2lmp_data.py` 是纯 Python 标准库脚本，不依赖跨仓库导入，因此内部无 `sys.path` 处理；直接运行或模块方式均可。

因此两种运行方式均支持：

```bash
# 方式 1：作为模块运行（推荐）
python -m LmpPy.scripts.script_name

# 方式 2：直接运行（适用于没有 __main__ 检查的脚本）
python LmpPy/scripts/script_name.py
```

推荐使用方式 1（`-m` 模块方式），因为路径解析更可靠。
