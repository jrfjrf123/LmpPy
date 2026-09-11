# LmpPy 完整运行流程详解

> 适用版本: 重构版 (run_refactored.py)
> 最后更新: 2026-06-22

---

## 目录

1. [整体架构](#1-整体架构)
2. [初始化阶段](#2-初始化阶段)
3. [主循环（4步详解）](#3-主循环4步详解)
4. [MPI 并行模型](#4-mpi-并行模型)
5. [断点续算流程](#5-断点续算流程)
6. [Smoke Test 流程](#6-smoke-test-流程)

---

## 1. 整体架构

### 1.1 系统概述

LmpPy 是一个 LAMMPS `bond/react` / `bond/create` 后处理框架，负责：

- 实时粗粒化（CG）映射与更新
- CG 轨迹生成与转换
- 化学反应监测与记录
- CG 拓扑推导
- 断点续算支持

### 1.2 ASCII 流程图

```
┌─────────────────────────────────────────────────────────────────────┐
│                      CLI 入口 (run_refactored.py)                    │
│                                                                     │
│  python -m LmpPy.run_refactored <config_dir> [--test] [--loop-num N]│
│                    [--smoke-test] [--loop-num N]                    │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      1. 配置加载 (_load_configs)                     │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  ConfigLoader                                                 │   │
│  │  ├── system.yaml          → SystemConfig                     │   │
│  │  ├── lammps_params.yaml   → LAMMPSParams                     │   │
│  │  └── data_file (解析)     → mass_list (Dict[int, float])     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CG 映射加载/生成                                              │   │
│  │  ├── initial_cg_mapping 文件存在? → CGCompareList.from_csv() │   │
│  │  └── 不存在? → MappingGenerator.generate() → CGCompareList   │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  反应模板解析                                                  │   │
│  │  └── reactions/ 目录 → TemplateParser / load_all_reaction_   │   │
│  │       templates() → Dict[str, ReactionTemplate]               │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  CG 级反应签名加载 (生产级 CG 映射更新用)                       │   │
│  │  └── cg_reaction_identifier.load_template_signatures()       │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  判断反应模式                                                  │   │
│  │  ├── bond_create_config 存在? → "bond/create" 模式            │   │
│  │  └── 不存在?               → "bond/react" 模式                │   │
│  └──────────────────────────────────────────────────────────────┘   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      2. 组件初始化 (_init_components)                │
│                                                                     │
│  LAMMPSDataExtractor(use_cache=True)   ← 数据提取器（带缓存）       │
│  CGConverter()                         ← CG 坐标转换器             │
│  BondsRecorder()                       ← 键连表记录器              │
│  reacted_nums 字典初始化               ← 反应计数器               │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      3. LAMMPS 初始化 (_init_lammps)                │
│                                                                     │
│  lammps(name="py_react")                                            │
│  ├── 输入脚本模式: lmp.file(input_script)                           │
│  └── 内置初始化:                                                    │
│      ├── units / atom_style / pair_style / bond_style / ...         │
│      ├── read_data <data_file>                                      │
│      ├── neigh_modify                                               │
│      ├── timestep                                                   │
│      ├── molecule 加载 (molecules 配置 + reactions 配置)            │
│      └── velocity all create <temperature> <seed>                   │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      4. 主循环 (for i in range(loop_num))            │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ Step 1   │    │ Step 2   │    │ Step 3   │    │ Step 4   │      │
│  │ 建键反应 │───▶│ 检测反应 │───▶│ 弛豫     │───▶│ 更新坐标 │      │
│  │          │    │ (最复杂) │    │          │    │ image    │      │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      5. 收尾 (_finalize)                             │
│                                                                     │
│  ├── lmp.command("write_data final_frame.data")                     │
│  ├── rank 0: 保存 changed_bead_id_list.pkl                         │
│  ├── rank 0: 关闭 reaction_num.txt                                  │
│  ├── rank 0: 保存 final_cg_compare_list.csv                        │
│  ├── rank 0: 保存 bonds_records                                     │
│  └── rank 0: 保存 reaction_frames.npz                              │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 输出文件清单

| 输出文件 | 格式 | 说明 |
|---|---|---|
| `cg_trajectory.lammpstrj` | LAMMPS dump | CG 轨迹（含反应前/后帧） |
| `reaction_num.txt` | CSV | 每时间步的反应计数 |
| `final_cg_compare_list.csv` | CSV | 最终 CG 映射（bead_id, mol_id, bead_type, AA_id, mass） |
| `reaction_frames.npz` | NPZ | 反应帧详细数据（全原子坐标、键、CG 映射等） |
| `final_frame.data` | LAMMPS data | 最终帧的 data 文件 |
| `changed_bead_id_list.pkl` | Pickle | 变化的 bead ID 列表 |
| `bonds_records/` | CSV | 键连表变化记录 |

---

## 2. 初始化阶段

### 2.1 配置加载 (_load_configs)

配置加载顺序严格遵循以下步骤：

#### 2.1.1 加载 system.yaml

```yaml
# system.yaml 核心字段（全部位于顶层 system: 之下）
system:
  name: "my_system"                  # 体系名称
  bead_type_names:                   # 必需：bead 类型名 → LAMMPS type 编号
    Bead1: 1
    Bead2: 2
  mapping_files:                     # 必需：每项需同时有 path 与 copies
    - path: "mappings/mapping_monomer.yaml"
      copies: 1000
    - path: "mappings/mapping_end.yaml"
      copies: 200
```

`ConfigLoader.load_system_config()` 解析后生成 `SystemConfig` dataclass。校验规则见 `core/config_loader.py:506-536`：顶层必须有 `system`，`system` 内必须有 `mapping_files` 与 `bead_type_names`，且每个 `mapping_files[i]` 必须同时含 `path` 和 `copies`。

#### 2.1.2 加载 lammps_params.yaml

```yaml
# lammps_params.yaml 核心字段（分五段；前缀为 [必需] 的段/键缺失即报错）
simulation:                          # [必需]
  loop_num: 100                      # 主循环次数
  dt: 0.001                          # 时间步长 (ps)
  timestep: 0.5                      # LAMMPS timestep (fs)
  temperature: 400.0                 # 模拟温度 (K)
  pressure: 1.0                      # 模拟压力 (atm)
  ensemble: "nvt"                    # 系综 (nvt / npt)
  pair_style: "lj/cut"               # 可选，默认 lj/cut

steps:                               # [必需]
  bond_react_check: 50               # bond/react 检查步数
  run_per_loop: 200                  # 每个 loop 总步数
  nve_limit: 10                      # NVE/limit 步数

npt:                                 # [必需]
  tcouple: 100                       # 温度耦合常数 (fs)
  pcouple: 1000                      # 压力耦合常数 (fs)

bond_react:                          # [必需]
  stabilization: 0.03                # 稳定化参数 (Å)，无条件要求存在
  reactions:                         # 反应列表；程序只读 name/cutoff/pre_mol/post_mol
    - name: "rxn1"
      cutoff: 6.0
      pre_mol: "monomer"
      post_mol: "dimer"
      # map_file / pre_template / post_template / pre_mapping / post_mapping
      # 一律按 reactions/{name}/ 目录约定自动发现，写入此处不会被读取

molecules:                           # 可选，分子模板
  monomer: "templates/monomer.mol"
  dimer: "templates/dimer.mol"

files:                               # 可选，以下各项均有默认值
  data_file: "data.lmp"
  initial_cg_mapping: "AtomId_BeadId_compare_list.csv"
  read_data_extra:                   # read_data 额外参数
    special_per_atom: 4
    bond_per_atom: 3
    angle_per_atom: 0
    dihedral_per_atom: 0

# bond_create:                       # 可选；enabled: true 时与 bond_react.reactions 互斥
```

`ConfigLoader.load_lammps_params()` 解析后生成 `LAMMPSParams` dataclass。注意 `run_per_loop` 必须严格大于 `nve_limit + bond_react_check`，否则报错（`core/config_loader.py:675-679`）。

#### 2.1.3 加载质量列表

调用 `ConfigLoader.load_mass_list(data_file)`。优先级（`core/config_loader.py:612-625`）：

1. `config_dir/mass_list.yaml` 存在 → 直接用它（作为覆盖配置，**忽略** `data_file`）
2. 否则若传入了 `data_file` → 调用 `parse_masses_from_data_file()` 解析其 Masses 段
3. 两者都没有 → 抛 `ConfigError`

```
Masses

1 12.010736   # c2
2 12.010736   # ce
3 1.007972    # hc
...
```

解析结果：`Dict[int, float]`，如 `{1: 12.010736, 2: 12.010736, 3: 1.007972, ...}`。

#### 2.1.4 加载反应模板

扫描 `config_dir/reactions/` 目录，对每个子目录调用 `load_all_reaction_templates()`，该函数解析以下文件：

| 文件 | 格式 | 内容 |
|---|---|---|
| `<rxn_name>.lammpstemplate` (pre/post) | LAMMPS template | 反应前/后原子拓扑和类型 |
| `<rxn_name>.map` | LAMMPS map | 反应映射（initiator/edge/equivalence/constraint） |
| `<rxn_name>_pre_mapping.yaml` | YAML | 反应前 CG 映射定义 |
| `<rxn_name>_post_mapping.yaml` | YAML | 反应后 CG 映射定义（模板匹配和 CG 更新用） |

解析结果：`Dict[str, ReactionTemplate]`，key 为反应名称（如 `"rxn1"`）。

#### 2.1.5 加载 CG 级反应签名

调用 `cg_reaction_identifier.load_template_signatures()`，从 reactions/ 目录加载 3-bead 类型签名用于生产级 CG 映射更新。

返回：
- `signature_index` — `{(interior_type, end_type, monomer_type): [CGReactionSignature, ...]}`
- `all_signatures` — 所有签名列表
- `end_types`, `monomer_types`, `interior_types` — 类型集合

此签名系统是生产路径下 CG 映射更新的核心依据（取代了早期的 AA 级 `ReactionLocator` + `CGMapper` 方案），完全无 LAMMPS 依赖。

### 2.2 CG 映射生成/加载

断点续算判断逻辑见第 5 节。此处描述首次运行流程：

#### 2.2.1 MappingGenerator

读取 `system.yaml` 中 `mapping_files` 字段指定的 YAML 文件，每个文件描述一种分子的 AA→CG 映射关系：

```yaml
# mapping_monomer.yaml —— 实际格式为 site-types + config 两段
site-types:
  Head:
    index: [0, 1, 2]        # 相对该 bead 起始位置的 0-based 原子偏移
    x-weight: [12, 12, 1]   # 质心权重，长度须与 index 一致
  Mid:
    index: [0, 1, 2]
    x-weight: [12, 12, 1]

config:
  - anchor: 0               # 固定为 0，程序自动累积偏移
    repeat: 1000            # 分子数
    offset: 6               # 分子间原子 ID 偏移
    sites:                  # [site 类型名, 相对于 anchor 的起始位置]
      - [Head, 0]
      - [Mid, 3]
```

原子 ID 计算：`AA_id = anchor + site_offset + index[i] + 1`。详见 `configuration.md` §3。

`MappingGenerator.generate()` 将 YAML 定义转换为 `CGCompareList.data` — 一个 `(n_mapped_atoms, 5)` 的 NumPy 数组（**行数等于被 bead 覆盖的原子数，不是体系总原子数**；mapping 未覆盖的原子不出现在此数组中），列含义：

| 列索引 | 名称 | 说明 |
|---|---|---|
| 0 | bead_id | 粗粒珠子全局 ID |
| 1 | mol_id | 分子 ID |
| 2 | bead_type | 粗粒珠子类型 |
| 3 | AA_id | 全原子原子 ID |
| 4 | mass | 原子质量 |

#### 2.2.2 两种路径代码位置

```python
# LmpPy/run_refactored.py, _load_configs() 第169-179行
cg_mapping_path = self.config_dir / self.lammps_params.initial_cg_mapping
if cg_mapping_path.exists():
    # 断点续算：直接加载已有映射
    self.cg_compare_list = CGCompareList.from_csv(str(cg_mapping_path))
else:
    # 首次运行：从 system.yaml 生成
    self.cg_compare_list = generate_cg_compare_list(
        self.system_config, self.config_dir
    )
```

### 2.3 反应模板解析 (TemplateParser)

`TemplateParser` 负责解析 LAMMPS 反应模板文件和映射文件：

**Pre/Post 模板文件格式** — 标准的 LAMMPS 分子模板格式：

```
# rxn1_pre.lammpstemplate
5 atoms
4 bonds
1 angles
0 dihedrals

Types

1 1
2 2
...

Coords

1 0.000 0.000 0.000
2 1.540 0.000 0.000
...

Bonds

1 1 1 2
2 1 2 3
...
```

> **段顺序不可调换**：`TemplateParser` 按固定的 `section_order = ['types', 'coords', 'bonds', 'angles', 'dihedrals']` 切片（`core/template_parser.py:232`），每个段的结束位置取"下一个段在 `section_order` 中出现的位置"。若把 `Coords` 写在 `Types` 之前，`Types` 段的切片范围为空，`atom_types` 会解析失败。

**映射文件 (.map) 格式**：

```
# rxn1.map
5 edge IDs
2 initiator IDs
0 equivalence
0 constraints

Edge IDs
1
2

Initiator IDs
1
2
```

解析结果 `ReactionTemplate` 包含：
- `pre_template` / `post_template`: `TemplateData` — pre/post 的原子类型、坐标、键连接
- `reaction_map`: `ReactionMapData` — initiator_ids（匹配起点）、edge_ids（不参与 CG 更新的边界原子）、equivalences、constraints
- `pre_cg_mapping` / `post_cg_mapping`: `TemplateCGMapping` — optional，定义模板原子如何映射到 CG 珠子

### 2.4 LAMMPS 初始化 (_init_lammps)

`_init_lammps()` 方法创建并配置 LAMMPS 实例：

**分支 1：外部输入脚本模式**
```
if params.input_script:
    lmp.file(str(input_script))
```

**分支 2：内置初始化模式（默认）**
```
1. units real
2. atom_style full
3. pair_style lj/cut 15
4. bond_style harmonic
5. angle_style harmonic
6. dihedral_style fourier
7. read_data <data_file> extra/special/per/atom N extra/bond/per/atom N ...
8. neigh_modify every 1 delay 0 check yes
9. timestep <value>
```

**分子模板加载（两步策略，避免重复）**：
1. 从 `molecules` 配置加载（向后兼容）
2. 从 `reactions` 配置加载（仅 `bond/react` 模式）

**初始化速度**：
```
lmp.command("velocity all create <temperature> <seed>")
```

### 2.5 CG 系统初始化 (CGInitializer)

`CGInitializer` 是初始化阶段的整合组件：

```
CGInitializer(config_dir)
├── generate_initial_cg_mapping(output_path=None)  → CGCompareList
│   └── 调用 MappingGenerator 生成 AA→CG 映射
├── extract_aa_bonds(data_file=None)               → np.ndarray
│   └── 从 LAMMPS data 文件解析 AA 键
├── generate_cg_topology(data_file=None)           → CGTopology
│   └── AA 键 → CG 键，并推导 CG 角与二面角
├── verify_consistency(cg_compare_list, cg_topology)→ List[str]
│   └── 一致性校验，返回问题列表（空列表表示通过）
└── initialize(output_mapping=None,
               output_topology_prefix=None,
               verify=True)                        → CGSystem
    └── 串联以上各步的完整流程
```

模块级便捷函数 `initialize_cg_system(config_dir, ...)` 等价于 `CGInitializer(config_dir).initialize(...)`。

输出 `CGSystem` 数据类，包含：
- `cg_compare_list` — CG 映射
- `cg_topology` — CG 拓扑（键/角/二面角）
- `n_atoms`, `n_beads`, `n_molecules` — 体系统计

### 2.6 反应命令生成 (ReactionCommands)

根据 `reaction_mode` 选择：

**bond/react 模式** (`_run_bond_react`):
```python
fix rxns all bond/react stabilization yes npt_grp REACT <react_cmds>
```
其中 `react_cmds` 遍历所有 reactions，每个形如：
```
react rxn1 all 1 0.0 6.0 monomer dimer rxn1.map
```

**bond/create 模式** (`generate_fix_bond_create`):
```python
fix bond_create_fix_0 all bond/create Nevery itype jtype Rmin bondtype [iparam ...] [jparam ...] [prob ...]
```

`bond/create` 模式对每对 (itype, jtype) 生成独立 fix，顺序执行避免多 fix 冲突。

---

## 3. 主循环（4步详解）

主循环是 `LAMMPSReactionRunner.run()` 的核心，每次迭代包含 4 个步骤。

### 3.1 Step 1: 建键反应

根据 `reaction_mode` 分支执行：

#### 3.1.1 bond/react 路径

```python
# _run_bond_react()
fix rxns all bond/react stabilization yes npt_grp REACT {
    react rxn1 all 1 0.0 <cutoff> <pre_mol> <post_mol> <map_file>
}
fix npt_grp_REACT nvt temp T T tcouple
run <bond_react_check_step>
```

- LAMMPS `fix bond/react` 在 `bond_react_check_step` 步内自动检测反应条件（距离/取向）
- 反应发生后，LAMMPS 自动删除旧键、创建新键、改变原子类型
- `fix npt_grp_REACT` 提供系综控制

**耗时**: `bond_react_check_step` 步 MD，典型值 50 步。

#### 3.1.2 bond/create 路径

```python
# _run_bond_create() — 顺序执行每对
for each pair:
    fix bond_create_fix_<i> all bond/create Nevery itype jtype Rmin bondtype [iparam/jparam/prob]
    fix ensemble all nvt temp T T tcouple
    run 1
    extract_fix → 提取该对反应计数
    unfix bond_create_fix_<i>
    unfix ensemble
```

- 每对只运行 1 步（`run 1`）
- 顺序执行避免多 fix 冲突
- `extract_fix` 提取该对的反应计数

**耗时**: `n_pairs` 步 MD（每对 1 步）。

### 3.2 Step 2: 检测反应（最复杂步骤）

当 rank 0 检测到反应发生时，完整的处理流水线如下：

```
Step 2 流水线
┌──────────────────────────────────────────────────────────────────────┐
│  2a. LAMMPSDataExtractor.extract_all() → AtomData + BondData        │
│      └─ decode_image_flags_vectorized() (362x 加速)                 │
├──────────────────────────────────────────────────────────────────────┤
│  2b. BondDetector.detect() → BondChanges                            │
│      └─ created_bonds, deleted_bonds (集合运算)                      │
├──────────────────────────────────────────────────────────────────────┤
│  2c. CG 签名匹配（bond/react）OR 距离/类型条件（bond/create）        │
│      └─ load_template_signatures + identify_reaction                 │
├──────────────────────────────────────────────────────────────────────┤
│  2d. _update_cg_mapping() / update_cg_mapping_create() → 更新 CG 映射│
├──────────────────────────────────────────────────────────────────────┤
│  2e. CGConverter.convert() → CG 坐标 (惰性索引缓存)                 │
├──────────────────────────────────────────────────────────────────────┤
│  2f. 写入 CG 轨迹帧（post-reaction, pre-relaxation）                │
├──────────────────────────────────────────────────────────────────────┤
│  2g. 缓存反应帧 → reaction_frames.npz                               │
└──────────────────────────────────────────────────────────────────────┘
```

#### 子步骤 2a: 数据提取

`LAMMPSDataExtractor.extract_all(lmp, use_cache=False)` 调用 `extract_atoms` 和 `extract_bonds`：

```python
# extract_atoms() 内部
natoms = lmp.extract_global("natoms")
coords = np.array(lmp.gather_atoms("x", 1, 3)).reshape(-1, 3)
encode_ixyz = np.array(lmp.gather_atoms("image", 0, 1))

# vectorized decode (362x speedup)
image_flags = decode_image_flags_vectorized(encode_ixyz)

# ids/types use cache when possible, or gather from LAMMPS
ids = np.array(lmp.gather_atoms("id", 0, 1))
types = np.array(lmp.gather_atoms("type", 0, 1))

# extract_bonds() 内部
bonds = np.array(lmp.gather_bonds()[1]).reshape(-1, 3)
bonds[:, 1:3] = np.sort(bonds[:, 1:3], axis=1)  # 确保 atom1 < atom2
```

**缓存策略**:
- `ids`, `types`, `bonds`: 无反应时可复用，只有反应后更新
- `coords`, `image_flags`: 每次 MD 后都变化，需重新收集

**`decode_image_flags_vectorized` 加速细节**:
```python
# 位运算: 0.023ms vs 循环: 8.2ms (362x 加速)
ix = (encode_ixyz & IMGMASK) - IMGMAX
iy = ((encode_ixyz >> IMGBITS) & IMGMASK) - IMGMAX
iz = (encode_ixyz >> IMG2BITS) - IMGMAX
```

#### 子步骤 2b: 键变化检测

`BondDetector.detect(bonds_before, bonds_after)`:

1. **整数编码**: 将键对 `(atom1, atom2)` 编码为单个整数 `atom1 * encoder + atom2`
   - `encoder = n_atoms + 1`
   - 原子 ID 已排序确保唯一性
2. **集合运算**: `created_codes = codes_after - codes_before`, `deleted_codes = codes_before - codes_after`
3. **解码**: 从编码集合恢复键数组（包含键类型）

```python
# 核心算法
detector = BondDetector(n_atoms)
codes_before = detector.encode_bonds(bonds_before)
codes_after  = detector.encode_bonds(bonds_after)
created_codes = codes_after - codes_before
deleted_codes = codes_before - codes_after
```

返回 `BondChanges`，包含：
- `created_bonds`: `(n_created, 3)` — 反应创建的键
- `deleted_bonds`: `(n_deleted, 3)` — 反应删除的键
- `has_changes`: `bool` — 是否有任何键变化

#### 子步骤 2c + 2d: 反应识别与 CG 映射更新

主流程**不做** AA 级模板匹配。`run_refactored.py` 的两个模式都直接在 CG 层处理，通过 `identify_reaction()` 返回的签名来确定反应类型：

**bond/react 模式: CG 签名匹配 (`_update_cg_mapping`)**

1. **AA 键 → CG 键转换**: `atom_bonds_to_cg_bonds()`
2. **CG 键差集**: 新建的 CG 键
3. **两级匹配**（经 `identify_reaction()`）:
   - 3-bead 签名粗筛: 匹配 `(interior_type, end_type, monomer_type)` 三元组
   - chain 精筛: 沿键链向外追踪类型序列，按需启用
4. **向量化批量应用**: 收集所有 bead_type 和 bead_id 更新，一次性写入 `cg_compare_list.data`

**bond/create 模式: 条件更新 (`update_cg_mapping_create`)**

不依赖反应模板，直接按 `bond_create.cg_type_map` 把成键后的 bead 类型改写：

```python
update_cg_mapping_create(
    cg_mapping_data,        # cg_compare_list.data，原地修改
    bonds_before,           # 反应前 AA 键
    bonds_after,            # 反应后 AA 键
    cg_type_map,            # {旧 bead_type: 新 bead_type}
)
```

> **历史说明**：`core/reaction_locator.py` 的 `ReactionLocator` 与 `core/cg_mapper.py` 的 `CGMapper.batch_update()` 实现了 AA 级 pre/post 模板双重验证与 bead 重编号，但**未被 `run_refactored.py` 引用**，属于早期设计遗留。生产路径是上面的 CG 签名匹配。

#### 子步骤 2e: CG 坐标转换

`CGConverter.convert()` 将全原子坐标转换为粗粒坐标：

**惰性索引缓存优化**:
- 无反应时（CG 映射不变）：`O(n_beads × n_atoms)` → `O(n_atoms)`（从原始 O(n²) 优化）
- 有反应时（CG 映射变化）：重建索引 `O(n_atoms log n_atoms)`

**转换算法**:
```python
def _convert_with_cache(self, atom_coords, cg_compare_list, mass_list):
    # 1. 扁平化的原子索引: 所有要聚合的原子坐标
    # 2. 分组质心计算 (scatter-add):
    weighted = mass_values[:, None] * atom_xyz      # (n_atoms, 3)
    np.add.at(sum_weighted, bead_indices, weighted)  # → (n_beads, 3)
    np.add.at(sum_masses, bead_indices, mass_values) # → (n_beads,)
    centroids = sum_weighted / sum_masses[:, None]   # 质心归一化
    # 3. 构建输出 [bead_id, bead_type, x, y, z]
```

**坐标展开**（在 CG 转换之前）:
```python
# _compute_cg_coords() 内部流程:
molecule_ids = find_molecules(bonds, len(ids))     # Numba 加速 BFS
unwrapped = unwrap_coords_python(coords, bonds, box, molecule_ids)
dump_data = [ids, types, unwrapped]                # 构建 dump 格式
cg_coords = lammpstrj2cg(dump_data, cg_compare_list, mass_list, converter)
wrapped_coords, cg_ixyz = wrap_coordinates(cg_coords[:, 2:5], box)
```

#### 子步骤 2f: 写入 CG 轨迹帧

每次反应写入两帧到 CG 轨迹文件：

1. **帧 1（反应前）**: 时间步 = `timestep - bond_react_check_step`，使用缓存坐标
2. **帧 2（反应后）**: 时间步 = `timestep`，使用新提取坐标

```python
def _write_cg_frame(output_path, timestep, box, cg_coords, cg_ixyz):
    write_lammps_dump_file(path, timestep, box,
        cg_coords[:, 0].astype(int),     # bead_id
        cg_coords[:, 1].astype(int),     # bead_type
        cg_coords[:, 2:5],               # x, y, z
        cg_ixyz)                         # image flags
```

#### 子步骤 2g: 缓存反应帧

将当前反应的所有数据追加到 `self.reaction_frames` 字典：

```python
self.reaction_frames = {
    'aa_coords_before': [...],    # 全原子坐标（反应前）
    'aa_coords_after': [...],     # 全原子坐标（反应后）
    'aa_ids': [...],              # 原子 ID
    'aa_types': [...],            # 原子类型
    'aa_bonds_before': [...],     # 原子键（反应前）
    'aa_bonds_after': [...],      # 原子键（反应后）
    'cg_mapping_before': [...],   # CG mapping（反应前）
    'cg_mapping_after': [...],    # CG mapping（反应后）
    'cg_bonds_before': [...],     # CG 键（反应前）
    'cg_bonds_after': [...],      # CG 键（反应后）
    'timestep': [...],            # 时间步
}
```

模拟结束后，所有帧一次性写入 `reaction_frames.npz`。

### 3.3 Step 3: 弛豫

反应发生后，系统需要弛豫以吸收键能冲击：

#### 3.3.1 bond/react 弛豫 (_run_relaxation)

```
Phase 1: NVE/limit + 局部系综
  group reaction_atom = bond_react_MASTER_group
  group non_reaction_atom = npt_grp_REACT
  fix react_nve reaction_atom nve/limit <stabilization>
  fix react_npt non_reaction_atom nvt/npt ...
  run <nve_limit_step>

Phase 2: 全原子系综
  unfix react_nve, unfix react_npt
  fix normal_npt all nvt/npt ...
  run <normal_npt_step>
  unfix normal_npt
```

**弛豫策略**:
1. 反应原子组: `nve/limit` 限制最大位移，防止键能冲击导致飞散
2. 非反应原子组: 正常系综控制，提供热浴/压浴
3. 第二阶段: 全原子统一系综控制，恢复系统平衡

**弛豫无跳过逻辑**: `_run_relaxation()` / `_run_relaxation_create()` 在每轮循环中**无条件执行**（`run_refactored.py:437-440`），无论本轮是否发生反应。无反应时只省去 Step 2 内部的 CG 映射更新与缓存失效。

#### 3.3.2 bond/create 弛豫 (_run_relaxation_create)

```
Phase 1: 全局 NVE/limit
  fix relax_nve all nve/limit <stabilization>
  run <nve_limit_step>

Phase 2: 全局系综
  fix normal_ensemble all nvt/npt ...
  run <normal_step>
```

**差异说明**:
- bond/create 模式没有 LAMMPS 自动标记的反应原子组，因此全部使用全局组
- 顺序执行每对 bond/create 后统一弛豫
- 弛豫步数计算: `normal_step = run_step - n_pairs - nve_limit_step`

### 3.4 Step 4: 更新坐标和 image flags

```python
# 主循环 Step 4 代码
atom_data = self.data_extractor.extract_atoms(lmp, use_cache=True)

# rank 0 更新缓存
cached['coords'] = atom_data.coords.copy()
cached['ixyz']   = atom_data.image_flags.copy()
cached['cg_coords'] = None  # 弛豫后 CG 坐标需要重新计算
```

关键点：
- 所有进程参与 `extract_atoms`（MPI 集合操作 `gather_atoms` 需要所有 rank 同步）
- `use_cache=True` 复用缓存的 ids 和 types（弛豫不改变原子类型和 ID）
- CG 坐标缓存标记为 `None`，下次使用时惰性重建
- 最后执行 `MPI.COMM_WORLD.Barrier()` 同步

---

## 4. MPI 并行模型

### 4.1 数据并行策略

LmpPy 采用 **数据并行**（SPMD, Single Program Multiple Data）模型：

```
Rank 0          Rank 1          Rank 2          Rank 3
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ LAMMPS   │   │ LAMMPS   │   │ LAMMPS   │   │ LAMMPS   │
│ Instance │   │ Instance │   │ Instance │   │ Instance │
│    ↓      │   │    ↓      │   │    ↓      │   │    ↓      │
│ Run MD   │   │ Run MD   │   │ Run MD   │   │ Run MD   │
│ Detect   │   │ Detect   │   │ Detect   │   │ Detect   │
│ Reactions│   │ Reactions│   │ Reactions│   │ Reactions│
└──────────┘   └──────────┘   └──────────┘   └──────────┘
     │              │              │              │
     └──────────────┴──────────────┴──────────────┘
                          │
                    MPI Barrier 同步
                          │
               ┌──────────┴──────────┐
               │   Rank 0: 文件 I/O  │
               │   - 写 CG 轨迹      │
               │   - 写 reaction_num │
               │   - 写 npz 文件     │
               └─────────────────────┘
```

### 4.2 进程角色分配

| 角色 | 职责 |
|---|---|
| **Rank 0** | 配置加载打印、文件 I/O、CG 轨迹写入、反应帧缓存、坐标计算、反应计数累加 |
| **所有 Rank** | LAMMPS 实例独立运行、`extract_atoms` / `extract_bonds` MPI 集合调用、弛豫计算 |

### 4.3 同步点

**Barrier 位置**:
1. 初始化后，rank 0 完成文件操作后: `MPI.COMM_WORLD.Barrier()`
2. 每次主循环 Step 4 结束后: `MPI.COMM_WORLD.Barrier()`
3. Smoke test 清理前: `comm.Barrier()`

**Bcast 位置**:
- `has_reaction` 布尔值从 rank 0 广播到所有 rank（用于控制是否进入反应处理分支）

### 4.4 非 rank-0 进程行为

```python
if me == 0:
    # 仅 rank 0 执行:
    # - 反应检测与计数
    # - CG 坐标转换
    # - 文件写入
    # - 缓存管理
else:
    has_reaction = False  # 从 bcast 接收实际值
```

重要注意事项:
- `LAMMPSDataExtractor.extract_all()` 必须被所有进程调用，因为内部 `gather_atoms` / `gather_bonds` 是 MPI 集合操作
- 非 rank-0 进程不进行 CG 坐标计算和文件写入，但参与数据提取

---

## 5. 断点续算流程

### 5.1 检测机制

断点续算的核心检测点在 `_load_configs()` 方法中：

```python
# 第169-179行
cg_mapping_path = self.config_dir / self.lammps_params.initial_cg_mapping
if cg_mapping_path.exists():
    # 存在 → 直接加载（跳过生成步骤）
    self.cg_compare_list = CGCompareList.from_csv(str(cg_mapping_path))
else:
    # 不存在 → 从 system.yaml 的 mapping_files 生成
    self.cg_compare_list = generate_cg_compare_list(
        self.system_config, self.config_dir
    )
```

### 5.2 完整续算流程

```
首次运行:
  initial_cg_mapping.csv → 不存在
  ↓
  MappingGenerator.generate(system_config)
  ↓
  生成初始 CGCompareList
  ↓
  运行模拟
  ↓
  输出 final_cg_compare_list.csv
  ↓
  用户重命名 final_cg_compare_list.csv → initial_cg_mapping.csv

断点续算:
  initial_cg_mapping.csv → 存在
  ↓
  CGCompareList.from_csv() 直接加载
  ↓
  跳过 MappingGenerator 生成步骤
  ↓
  使用已有映射作为起点继续运行
```

### 5.3 续算时保留的状态

- **CG 映射**: 最后模拟结束时的 bead_id、bead_type、AA_id、mass 全部保留
- **反应计数**: 不保留（从 0 重新计数）
- **CG 轨迹**: 独立文件，续算时覆盖（新轨迹文件覆盖旧文件）

### 5.4 实施建议

用户需手动将 `final_cg_compare_list.csv` 复制/重命名为 `initial_cg_mapping.csv`：

```bash
# 首次运行后
cp final_cg_compare_list.csv initial_cg_mapping.csv

# 再次运行即可断点续算
python -m LmpPy.run_refactored config/
```

---

## 6. Smoke Test 流程

### 6.1 整体编排

`SmokeTestHarness` 是冒烟测试的编排器，负责为所有 rank 协调完整的测试流程：

```python
# 调用方式
python -m LmpPy.run_refactored <config_dir> --smoke-test --loop-num 5
```

### 6.2 SmokeTestHarness 工作流

```
SmokeTestHarness.run()
│
├── Rank 0: 创建临时目录 (tempfile.mkdtemp)
│   └── shutil.copytree: 复制源配置 → 临时目录
│       └── 忽略 __pycache__, .git, .ipynb_checkpoints
│
├── Bcast temp_dir 路径 → 所有 rank
├── Barrier 同步
│
├── 所有 rank: 实例化 LAMMPSReactionRunner
│   ├── runner.lammps_params.loop_num = N (覆盖)
│   └── runner.run()                    (所有 rank 参与)
│
├── Rank 0: SmokeValidator.validate()
│   ├── A. 文件输出完整性检查
│   ├── B. CG mapping 一致性检查
│   ├── C. 反应后 mapping 交叉验证
│   └── D. 反应计数合理性检查
│   └── → SmokeTestReport
│
├── Rank 0: 清理 (默认删除 temp_dir)
│   └── keep_output=True → 保留 temp_dir 供调试
│
├── Barrier 同步
│
└── Rank 0: 返回 report → sys.exit(0/1)
    其他 rank: sys.exit(0)
```

### 6.3 SmokeValidator 四项检查

**检查 A: 文件输出完整性**

验证四个关键输出文件：

| 检查项 | 文件 | 验证内容 |
|---|---|---|
| A1 | `reaction_num.txt` | 存在且非空，有数据行 |
| A2 | `cg_trajectory.lammpstrj` | 至少包含 `loop_num` 个 `ITEM: TIMESTEP` 帧 |
| A3 | `final_cg_compare_list.csv` | 存在且行数 > 0 |
| A4 | `reaction_frames.npz` | 存在则读取 timestep 数量（可选） |

**检查 B: CG mapping 一致性**

| 检查项 | 验证内容 |
|---|---|
| B1 | 每个 bead_id 的所有原子有相同的 bead_type（调用 `validate_cg_mapping_consistency()`） |
| B2 | AA_id 无重复（唯一值数量等于行数） |
| B3 | AA_id 不缺失（如果提供了 `n_atoms`，检查是否覆盖 1..n_atoms 所有原子） |

**检查 C: 反应后 mapping 交叉验证**

使用独立的 CG 级签名算法（`cg_reaction_identifier.identify_reaction()`）重构预期 mapping，与 runner 实际输出对比：

```
1. 从 reaction_frames.npz 加载每帧数据
2. 对每帧:
   a. 用累积 CG 映射将 AA bonds → CG bonds
   b. CG 级签名识别反应
   c. 根据识别结果重构预期 cg_mapping_after
   d. 与 runner 实际 cg_mapping_after 对比 bead_type 列
3. 汇总：匹配帧数 / 未识别帧数 / 不匹配帧数
```

**检查 D: 反应计数合理性**

| 检查项 | 验证内容 |
|---|---|
| D1 | 数据行数等于 `loop_num` |
| D2 | 所有反应计数非负 |
| D3 | 计算各反应类型总计并打印 |

### 6.4 配置文件修改策略

Smoke test 只修改 `loop_num` 一个参数：

```python
runner.lammps_params.loop_num = self.loop_num  # 覆盖为 5（或用户指定值）
```

所有其他配置（势函数、温度、压力、反应模板等）与源配置完全一致。

### 6.5 测试输出

```
============================================================
Smoke Test 报告
============================================================
  配置目录: /path/to/source_config
  循环次数: 5
  临时目录: /tmp/lmpy_smoke_XXXXXX (已删除)
    [✅] A. 文件输出完整性
         reaction_num.txt: 5 行数据
         cg_trajectory.lammpstrj: 10 帧
         final_cg_compare_list.csv: 100 行
         reaction_frames.npz: 3 帧
    [✅] B. CG mapping 一致性
         100 个原子映射
         20 个 bead，每个 bead_type 唯一
    [✅] C. 反应后 mapping 更新正确 (交叉验证)
         共 3 帧
         第 0 帧: ✅ rxn1_EEE 交叉验证通过
         第 1 帧: ✅ rxn1_EEE 交叉验证通过
         第 2 帧: ✅ rxn1_EEE 交叉验证通过
    [✅] D. 反应计数合理性
         数据行数: 5 (预期 5)
         所有反应计数非负 ✅
         rxn1: 3

  结果: 通过
============================================================
```

---

## 附录 A: 关键算法性能指标

| 算法 | 加速比 | 实现方式 |
|---|---|---|
| image flag 解码 | 362x | 向量化位运算 |
| 分子查找 (BFS) | 30-80x | Numba JIT (`njit(cache=True)`) |
| CG 坐标转换（惰性索引缓存） | O(n²) → O(n) | 排序分组 + scatter-add |
| 键编码比较 | 向量化 | 整数编码 + 集合运算 |

## 附录 B: 文件引用

| 模块 | 文件路径 |
|---|---|
| CLI 入口 | `/home/large_storage/jrf/lmp_py_react/LmpPy/run_refactored.py` |
| 配置加载 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/config_loader.py` |
| 映射生成 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/mapping_generator.py` |
| 模板解析 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/template_parser.py` |
| 数据提取 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/lammps_data_extractor.py` |
| 键检测 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/bond_detector.py` |
| 反应定位 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/reaction_locator.py` |
| CG 映射 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/cg_mapper.py` |
| CG 转换 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/cg_converter.py` |
| CG 初始化 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/cg_initializer.py` |
| CG 键映射 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/cg_bond_mapper.py` |
| 反应命令 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/reaction_commands.py` |
| CG 反应识别 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/cg_reaction_identifier.py` |
| 冒烟测试编排 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/smoke_test_harness.py` |
| 冒烟测试验证 | `/home/large_storage/jrf/lmp_py_react/LmpPy/core/smoke_validator.py` |
