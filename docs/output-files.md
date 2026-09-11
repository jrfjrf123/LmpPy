# 输出文件完整说明

> 版本: 2.5
> 更新: 2026-06-22

本文档详细说明 LmpPy 模拟管线生成的所有输出文件格式和内容。

---

## 一、主模拟管线输出

由 `run_refactored.py` 的主循环和收尾阶段生成，输出到运行目录。

### 1.1 cg_trajectory.lammpstrj — CG 轨迹文件

**格式**：LAMMPS dump 格式，每帧包含完整盒子信息和 CG 珠子坐标。

**帧结构**：

```
ITEM: TIMESTEP
0                    <-- 时间步
ITEM: NUMBER OF ATOMS
3                    <-- CG 珠子数量
ITEM: BOX BOUNDS pp pp pp
0.0 50.0             <-- xlo xhi
0.0 50.0             <-- ylo yhi
0.0 50.0             <-- zlo zhi
ITEM: ATOMS id type x y z ix iy iz
1 1 12.34 5.67 8.90 0 0 0   <-- bead_id, bead_type, x, y, z, image_flags
2 2 23.45 6.78 9.01 0 0 0
3 3 34.56 7.89 0.12 0 0 0
```

| 列 | 字段 | 说明 |
|----|------|------|
| 1 | id | CG bead ID（1-based） |
| 2 | type | bead 类型编号 |
| 3-5 | x, y, z | CG 坐标（Å） |
| 6-8 | ix, iy, iz | image flags（周期性盒子跨越计数） |

**生成方式**：每轮循环的 CG 弛豫步骤后，通过 `_write_cg_frame()` 写入。

### 1.2 reaction_num.txt — 反应计数文件

**格式**：空格分隔的文本文件，首行为注释行。

```
# timestep rxn1 rxn2
0 0 0
1000 0 0
2000 0 0
3000 0 0
4000 1 0
5000 0 1
...
```

| 列 | 字段 | 说明 |
|----|------|------|
| 1 | timestep | 当前时间步 |
| 2+ | rxn1, rxn2, ... | 各反应在本轮的发生次数 |

- 行数等于 `loop_num`（循环次数）
- 反应名称来自 `lammps_params.yaml` 中 `bond_react.reactions` 配置
- 由 rank 0 进程写入

### 1.3 final_frame.data — 最终构型 LAMMPS data 文件

**格式**：标准 LAMMPS data 文件格式。

```
LAMMPS data file via write_data

10000 atoms
20000 bonds
...

Masses

Pair Coeffs

Atoms # full

1 1 1 0.0 1.0 2.0 3.0
...
```

**生成命令**：`lmp.command("write_data final_frame.data pair ij nofix")`

- 包含完整的原子、键、角、二面角信息
- `pair ij nofix` 选项排除 pair 系数中的 fix 修正
- 可用于断点续算的初始构型

### 1.4 final_cg_compare_list.csv — 最终 CG 映射

**格式**：5 列 CSV 文件，无行号索引。

```csv
bead_id,mol_id,bead_type,AA_id,mass
1,1,1,1,12
1,1,1,2,12
1,1,1,3,12
1,1,1,4,12
1,1,1,7,1
...
```

| 列名 | 说明 |
|------|------|
| bead_id | 粗粒珠子编号（1-based，全局唯一） |
| mol_id | 分子编号（1-based） |
| bead_type | 珠子类型编号（反应后可能更新） |
| AA_id | 全原子 ID（1-based） |
| mass | 该原子的质量权重 |

- 通过 `np.savetxt()` 以 `%d` 格式写入
- 反映经过所有反应后最终的 CG 映射状态（bead_type 和 bead_id 可能已更新）

### 1.5 reaction_frames.npz — 反应帧数据（v2.4+ 精简格式）

**格式**：NumPy NPZ 压缩文件，包含 7 个数组。

| 数组名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| `aa_coords_before` | (n_reactions, n_atoms, 3) | float64 | 反应前全原子坐标 |
| `aa_coords_after` | (n_reactions, n_atoms, 3) | float64 | 反应后全原子坐标 |
| `aa_bonds_before` | (n_reactions,) | object | 反应前原子键列表（每个元素为 (n,3) int32 数组） |
| `aa_bonds_after` | (n_reactions,) | object | 反应后原子键列表 |
| `cg_mapping_before` | (n_reactions, n_atoms, 2) | float64 | 反应前 CG 映射 `[bead_id, bead_type]` |
| `cg_mapping_after` | (n_reactions, n_atoms, 2) | float64 | 反应后 CG 映射 `[bead_id, bead_type]` |
| `timestep` | (n_reactions,) | int64 | 反应发生的时间步 |

#### 1.5.1 推导规则（v2.4 精简）

v2.4 版本移除了 `aa_ids`、`aa_types`、`cg_bonds_before`、`cg_bonds_after` 数组，这些数据可由现有数组推导：

**aa_ids 推导**：

```python
# cg_mapping 的第三列（索引 3）是 AA_id
# 注意：cg_mapping 存储为 [bead_id, bead_type]
# AA_id 通过数组索引推断: cg_mapping_before[frame] 的行索引 + 1
# 或从原始的 CGCompareList 获取
# 实际 aa_ids = np.arange(1, n_atoms + 1)
aa_ids = np.arange(1, aa_coords_before.shape[1] + 1)
```

**cg_bonds 推导**：

```python
from LmpPy.core import atom_bonds_to_cg_bonds

cg_bonds_before = atom_bonds_to_cg_bonds(
    aa_bonds_before,  # 原子键 (n_bonds, 3)
    cg_mapping_before  # CG 映射 (n_atoms, 2)
)
cg_bonds_after = atom_bonds_to_cg_bonds(
    aa_bonds_after,
    cg_mapping_after
)
```

`atom_bonds_to_cg_bonds()` 的推导规则：

1. 获取每个键两端原子的 bead_id
2. 筛选跨 bead 的键（bead1_id != bead2_id）
3. 对有效键排序：bead1 < bead2
4. 去重（同一对 bead 之间的多个原子键只生成一个 CG 键）
5. 输出格式：`[bond_type, bead1, bead2]`

#### 1.5.2 向后兼容旧格式

旧格式（v2.3 及之前）包含 11 个数组：

```
aa_coords_before, aa_coords_after,
aa_ids, aa_types,          # 已移除
aa_bonds_before, aa_bonds_after,
cg_mapping_before, cg_mapping_after,
cg_bonds_before, cg_bonds_after,  # 已移除
timestep
```

后处理代码（`reaction_frames_parser.py`、`reaction_pairs.py`、`non_react_pairs.py`）具有自动格式检测能力：

```python
def _detect_format(data):
    """自动检测 npz 文件格式（v2.3 旧版 vs v2.4+ 新版）"""
    if 'aa_ids' in data:
        # 旧格式：直接读取
        aa_ids = data['aa_ids']
        cg_bonds = data['cg_bonds_after'][idx]
    else:
        # 新格式：推导
        aa_ids = cg_mapping_after[:, :, 3]  # 从 AA_id 列获取
        cg_bonds = atom_bonds_to_cg_bonds(aa_bonds, cg_mapping)
    return aa_ids, cg_bonds
```

### 1.6 changed_bead_id_list.pkl（v2.4 前）

**状态**：v2.4 起已移除。

此文件记录所有发生过变化的 bead_id 列表。v2.4 精简重构后，其功能由 `reaction_frames.npz` 中的 `cg_mapping_before/after` 替代。当前版本仍可能输出此文件（取决于配置），但不再作为主要数据来源。

### 1.7 bonds_records/ 目录（v2.4 前）

**状态**：v2.4 起已移除。

此目录记录每次反应的键连表快照。功能已由 `reaction_frames.npz` 的 `aa_bonds_before/after` 完全替代。

---

## 二、CG 拓扑输出

由 `CGTopology` 模块生成，存储推导的粗粒化拓扑结构。

### 2.1 cg_bonds.txt — CG 键

**格式**：每行 `bond_type bead1 bead2`（可选带 ID 列）。

```
1 1 2
1 2 3
1 3 4
...
```

| 列 | 说明 |
|----|------|
| bond_type | 键类型编号（根据 bead type 组合分配） |
| bead1 | 第一个 bead 的 ID |
| bead2 | 第二个 bead 的 ID |

### 2.2 cg_angles.txt — CG 角

**格式**：每行 `angle_type bead1 bead2 bead3`。

```
1 1 2 3
1 2 3 4
1 3 4 5
...
```

- 从 CG 键拓扑自动推导
- 回环结构（如 (1,1,2) 和 (2,1,1)）合并为同一角度类型

### 2.3 cg_dihedrals.txt — CG 二面角

**格式**：每行 `dihedral_type bead1 bead2 bead3 bead4`。

```
1 1 2 3 4
1 2 3 4 5
1 3 4 5 6
...
```

- 从 CG 键拓扑自动推导
- 回环结构合并为同一二面角类型

### 2.4 拓扑类型分配规则

拓扑类型 ID 根据 bead type 组合分配：

```python
# 规范化规则：
# Angle: (1,1,2) 和 (2,1,1) → 同一类型
# Dihedral: (1,1,2,2) 和 (2,2,1,1) → 同一类型
```

可通过 `type_mapping` YAML 配置预定义类型分配，未列出的组合自动分配新 ID。

---

## 三、工具模块输出

### 3.1 AA2CG 转换输出

由 `convert_aa2cg.py` 脚本生成。

| 格式 | 文件 | 说明 |
|------|------|------|
| Pickle | `*.pkl` | 二进制 CG 轨迹，Python 字典格式：`{'R': coords, 'types': types, 'ids': ids, ...}` |
| XYZ | `*.xyz` | 标准 XYZ 格式，可用于 VMD 等可视化工具 |
| LAMMPS data | `*.data` | 可直接用于 LAMMPS 模拟的 CG data 文件 |

**CG Data 文件内容**：

```
CG data file

3000 atoms
2999 bonds
5998 angles
...

Masses

Atoms

Bonds

Angles

Dihedrals
```

### 3.2 IBM 势能计算输出

由 `calc_ibm_potential.py` 和 `calc_ibm_potential_from_dist.py` 脚本生成。

#### 3.2.1 分布文件

分布计算输出 VOTCA 格式的 `.dist.tgt` 文件：

```
# bond_type1 分布
r/theta/phi  probability  i
0.50  0.0012  0
0.52  0.0015  0
0.54  0.0018  0
...
```

| 文件模式 | 说明 |
|----------|------|
| `bond_type*.dist.tgt` | 键长分布 |
| `angle_type*.dist.tgt` | 角度分布 |
| `dihedral_type*.dist.tgt` | 二面角分布 |
| `pair_type*.dist.tgt` | 对分布（RDF） |

#### 3.2.2 势能文件

玻尔兹曼反演后的势能文件：

| 文件模式 | 说明 |
|----------|------|
| `bond_type*_potential.txt` | 键势能：`r U` 两列 |
| `angle_type*_potential.txt` | 角势能：`theta U` 两列 |
| `dihedral_type*_potential.txt` | 二面角势能：`phi U` 两列 |
| `pair_type*_potential.txt` | 对势能：`r U` 两列 |

#### 3.2.3 LAMMPS Table 文件

可直接用于 LAMMPS `pair_style table` 的势能表文件：

```
# LAMMPS table file for bond_type1
BOND_TYPE1
N 200
...
```

### 3.3 分布平滑输出

由 `smooth_distribution.py` 脚本生成。

| 文件 | 格式 | 说明 |
|------|------|------|
| `*_dist.txt` | 5 行元数据头 + 2 列数据 | 平滑后的分布 |

**平滑文件格式示例**：

```
# Distribution: bond_type1
# Temperature: 400.00 K
# Smooth method: savgol
# Smooth parameters: window=21, polyorder=3
# Columns: x, probability
0.500000   0.001234
0.502000   0.001245
...
```

| 输出文件 | 说明 |
|----------|------|
| `smoothed_output/` | 包含所有平滑后的分布文件 |
| `comparison.png` | 原始 vs 平滑对比图（可选） |
| `report.txt` | 平滑参数和质量评估报告 |

**quality report.txt 示例**：

```
Distribution Smoothing Quality Report
======================================
Total files processed: 5

bond_type1: OK (MSE=0.00012)
bond_type2: OK (MSE=0.00008)
angle_type1: OK (MSE=0.00021)
...
```

---

## 四、后处理分析输出（output_analysis）

由 `LmpPy.output_analysis` 模块生成（直接读取上表的模拟输出目录）。所有文件**平铺**写入 `-o/--output-dir` 指定的目录，不分子目录：

```bash
python -m LmpPy.output_analysis all ./process1/ -o ./analysis/
```

### 4.1 输出文件

| 文件 | 生成命令 | 说明 |
|------|----------|------|
| `chain_length_distribution.png` | `chain-length` | 链长分布直方图 + KDE 曲线 |
| `distance_distribution.png` | `distance` | 各反应类型的反应距离分布图 |
| `reaction_stats.png` | `reaction-stats` | 反应统计（类型占比饼图 + 每循环堆叠柱状图） |
| `reaction_details.csv` | `all` / `distance` / `reaction-stats` | 反应事件明细表（约 15 MB） |
| `reaction_counts.csv` | `reaction-stats` | 每循环各反应类型计数 |
| `reactivity_ratio_summary.json` | `reactivity-ratio` | 竞聚率 r1/r2 双归一化估计 + bootstrap CI |
| `r_vs_conversion.png` | `reactivity-ratio` | r(X) 轨迹 + 窗宽扫描 |
| `composition_mayo_lewis.png` | `reactivity-ratio` | 组成法交叉验证 |
| `channel_rates.png` | `reactivity-ratio` | 四通道细粒度事件率柱状图 |
| `summary_dashboard.png` | `reactivity-ratio` | r1/r2 详细柱状图 + CI 误差棒 |
| `window_estimates.csv` | `reactivity-ratio` | 各转化率窗口的估计值 |

### 4.2 输入依赖

| 文件 | 用途 | 需要的子命令 |
|------|------|--------------|
| `final_frame.data` | 解析键拓扑，BFS 识别连通分子 → 链长 | `chain-length` |
| `reaction_frames.npz` | 反应事件数据（键变化、坐标、CG 映射） | `distance`, `reaction-stats` |
| `reaction_details.csv` | 反应事件表格（已存在则跳过 npz 重建） | `distance`, `reaction-stats` |

`reaction_details.csv` 不存在时模块会从 `reaction_frames.npz` 自动重建；npz 可达 5–100 GB，重建采用两阶段流式读取并临时提取到 `/tmp`。

### 4.3 链长分布图

- 从 `final_frame.data` 解析键拓扑，通过 BFS 识别连通分子后统计链长
- 可用 `--min-length` 过滤短链，`--theory-type`（Schulz-Zimm / Poisson / 对数正态）叠加理论曲线
- `--js` 额外计算 JS 散度

### 4.4 反应距离分布图

- 基于 `reaction_frames.npz` 中的反应前后坐标计算
- 展示反应发生时的原子间距离分布，用于验证反应截断半径 `cutoff` 的合理性
- `--ref-csv` 可叠加参考分布并做 KS 检验

### 4.5 反应统计图

- 饼图展示各反应类型占比；堆叠柱状图展示每循环反应数按类型分解
- 同时导出 `reaction_counts.csv` 供后续统计

### 4.6 竞聚率分析

`reactivity-ratio` 子命令从 mlcgsim ML 驱动 CG 模拟输出统计竞聚率 r1/r2（末端模型，r1=k11/k12、r2=k22/k21），三种用法：

```bash
# 全流程：扫轨迹 + 估计
python -m LmpPy.output_analysis reactivity-ratio ./process1 ./process2 -o ./analysis/
# 只统计计数（可复用）
python -m LmpPy.output_analysis reactivity-ratio ./process1 ./process2 --counts-out per_cycle_counts.csv
# 只做估计（跳过轨迹扫描）
python -m LmpPy.output_analysis reactivity-ratio --from-counts per_cycle_counts.csv -o ./analysis/
```

注意 `--pair-cutoff`（默认 10.0 Å）需与模拟实际使用的 `pair_cutoff` 一致，否则归一化基准不匹配。

---

## 五、输出文件完整列表

| 文件 | 所属模块 | 生成条件 | 用途 |
|------|----------|----------|------|
| `cg_trajectory.lammpstrj` | 主模拟 | 始终生成 | CG 轨迹可视化 |
| `reaction_num.txt` | 主模拟 | 始终生成 | 反应计数统计 |
| `final_frame.data` | 主模拟 | 始终生成 | 断点续算/分析 |
| `final_cg_compare_list.csv` | 主模拟 | 始终生成 | 最终 CG 映射 |
| `reaction_frames.npz` | 主模拟 | 有反应发生时 | 反应帧数据 |
| `changed_bead_id_list.pkl` | 主模拟 | v2.4 前 | 已废弃 |
| `bonds_records/` | 主模拟 | v2.4 前 | 已废弃 |
| `cg_bonds.txt` | CG 拓扑 | 调用 `CGTopology.to_files()` | CG 模拟输入 |
| `cg_angles.txt` | CG 拓扑 | 同上 | CG 模拟输入 |
| `cg_dihedrals.txt` | CG 拓扑 | 同上 | CG 模拟输入 |
| `*.pkl` | AA2CG 转换 | 用户选择 | CG 轨迹 pickle |
| `*.xyz` | AA2CG 转换 | `--xyz` 参数 | 可视化 |
| `*.data` | AA2CG 转换 | `data` 子命令 | CG 模拟 |
| `*_dist.txt` | 分布计算 | 调用 `calc_dist.py` | IBI 势能拟合 |
| `*.dist.tgt` | 分布计算 | VOTCA 格式输出 | IBM 势能计算 |
| `*_potential.txt` | IBM 势能 | 调用 `calc_ibm_potential.py` | CG 势能参数 |
| LAMMPS tables | IBM 势能 | `--lammps-tables` | LAMMPS table 势能 |
| `{name}_comparison.png` | 分布平滑 | 可选 | 平滑效果对比 |
| `{name}_report.txt` | 分布平滑 | 可选 | 平滑质量报告 |
| `chain_length_distribution.png` | 后处理分析 | `chain-length` | 聚合度分析 |
| `distance_distribution.png` | 后处理分析 | `distance` | 反应距离分析 |
| `reaction_stats.png` | 后处理分析 | `reaction-stats` | 反应类型占比与循环演化 |
| `reactivity_ratio_summary.json` 等 | 后处理分析 | `reactivity-ratio` | 竞聚率估计与诊断 |
