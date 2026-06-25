# LmpPy 参数文件解析文档

> 本文档详细说明 LmpPy 框架所有 YAML 配置文件的字段含义、类型、默认值以及使用示例。
> 适用于版本 v2.6+（含 `bond/create` 模式）。

---

## 目录

1. [system.yaml - 体系配置](#1-systemyaml-体系配置)
2. [lammps_params.yaml - LAMMPS 运行参数](#2-lammps_paramsyaml-lammps-运行参数)
   - [2.1 simulation](#21-simulation-模拟基本参数)
   - [2.2 steps](#22-steps-步数配置)
   - [2.3 npt](#23-npt-npt控温控压参数)
   - [2.4 bond_react](#24-bond_react-bondreact配置)
   - [2.5 bond_create (v2.6+)](#25-bond_create-v26-bondcreate配置)
   - [2.6 molecules](#26-molecules-分子模板配置)
   - [2.7 files](#27-files-文件配置)
   - [2.8 mass_list](#28-mass_list-元素质量映射)
3. [映射文件格式 (mapping YAML)](#3-映射文件格式-mapping-yaml)
   - [3.1 site-types](#31-site-types-bead类型定义)
   - [3.2 config](#32-config-映射配置)
4. [反应映射文件格式](#4-反应映射文件格式)
5. [断点续算配置](#5-断点续算配置)
6. [附录: 完整配置示例](#6-附录-完整配置示例)

---

## 1. system.yaml - 体系配置

**文件位置**: `{config_dir}/system.yaml`

**作用**: 定义模拟体系的名称、CG 映射文件列表、bead 类型名称映射。

---

### system.name

| 属性 | 值 |
|------|-----|
| **字段名** | `system.name` |
| **类型** | `string` |
| **默认值** | `"unnamed"` |
| **必需** | 否 |

**说明**: 模拟体系的名称，用于识别和日志输出。建议使用有意义的名称以便于区分不同体系。

**YAML 示例**:
```yaml
system:
  name: "环氧树脂固化体系"
```

---

### system.bead_type_names

| 属性 | 值 |
|------|-----|
| **字段名** | `system.bead_type_names` |
| **类型** | `Dict[string, int]` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: bead 类型名称到 LAMMPS type 编号的映射字典。

**关键约束**:
- 键为 bead 类型名称（字符串），必须与 mapping 文件中 `site-types` 的名称一致。
- 值为 LAMMPS 中的 type 编号（正整数），表示对应 bead 在 LAMMPS 模拟中的原子类型。
- 所有 mapping 文件中定义的 `site-types` 名称必须在此字典中覆盖，否则验证失败。
- 类型 ID 必须唯一，不可重复（重复时程序发出警告）。
- **注意**: bead_type_names 从 LAMMPS 数据文件读取时使用的 type 编号。在 `system.yaml` 中统一管理多个 mapping 文件间的类型映射，确保不同文件中同名的 bead 使用相同的 type ID。

**工作原理**:
当程序读取 LAMMPS data 文件时，每个原子都有一个 type 值（如 1, 2, 3）。`bead_type_names` 将这些 type 值与 CG bead 的名称关联起来，使得程序能够：
1. 将不同分子中相同物理含义的 bead 映射到同一 LAMMPS type
2. 在 CG 轨迹输出中正确标记 bead 类型
3. 为 CG 拓扑（键、角、二面角）生成正确的类型编号

**YAML 示例**:
```yaml
system:
  bead_type_names:
    Head: 1    # 头部珠子 → LAMMPS type 1
    Mid:  2    # 中部珠子 → LAMMPS type 2
    End:  3    # 尾部珠子 → LAMMPS type 3
```

---

### system.mapping_files

| 属性 | 值 |
|------|-----|
| **字段名** | `system.mapping_files` |
| **类型** | `List[Dict]` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 分子映射文件列表。每个元素定义一个 mapping 文件及其在初始数据文件中的分子拷贝数。

#### mapping_files[].path

| 属性 | 值 |
|------|-----|
| **字段名** | `mapping_files[].path` |
| **类型** | `string` |
| **必需** | **是** |

**说明**: mapping YAML 文件的路径。支持相对于 config 目录的相对路径或绝对路径。

**YAML 示例**:
```yaml
system:
  mapping_files:
    - path: "mapping/mapping_epoxy.yaml"
```

#### mapping_files[].copies

| 属性 | 值 |
|------|-----|
| **字段名** | `mapping_files[].copies` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 该类型分子在初始 LAMMPS data 文件中的拷贝数量（分子数）。程序自动累积原子 ID 偏移量。

#### mapping_files[].anchor 固定为 0 的原因

| 属性 | 值 |
|------|-----|
| **字段名** | `mapping_files[].anchor` |
| **类型** | `int` |
| **必需** | 否（外部字段，非 mapping_files 直接字段） |

**说明**: 在 mapping YAML 文件中，每个 config 条目的 `anchor` 固定为 0。

**原因**:
- 当 `system.yaml` 中配置了多个 mapping 文件时，程序会自动为每个 mapping 文件维护各自的 anchor 偏移。
- `copies` 决定了每种分子的数量，每个分子内部的原子 ID 偏移由 mapping 文件的 `offset` 管理。
- 程序内部会在加载所有 mapping 文件后，根据 `copies` 和 `offset` 自动计算每个分子的实际原子 ID 起始位置，因此 **用户层面 anchor 始终设为 0**。
- 这种设计使得 mapping 文件可以独立定义且可复用，无需在文件中硬编码绝对原子位置。

**YAML 示例**:
```yaml
system:
  mapping_files:
    - path: "mapping/mapping_epoxy.yaml"
      copies: 1
    - path: "mapping/mapping_curing_agent.yaml"
      copies: 2
```

---

### system.work_dir / system.output_dir

| 属性 | 值 |
|------|-----|
| **字段名** | `system.work_dir` |
| **类型** | `string` |
| **默认值** | `"work"` |
| **必需** | 否 |

| 属性 | 值 |
|------|-----|
| **字段名** | `system.output_dir` |
| **类型** | `string` |
| **默认值** | `"output"` |
| **必需** | 否 |

**说明**: LAMMPS 运行目录和输出文件目录，相对于运行时的当前工作目录。

**YAML 示例**:
```yaml
system:
  work_dir: "work"
  output_dir: "output"
```

---

## 2. lammps_params.yaml - LAMMPS 运行参数

**文件位置**: `{config_dir}/lammps_params.yaml`

**作用**: 定义 LAMMPS 模拟的全部运行参数，包括模拟基本参数、步数配置、控温控压、反应配置、文件路径等。

---

### 2.1 simulation - 模拟基本参数

```yaml
simulation:
  loop_num: 25000
  dt: 0.001
  timestep: 1
  temperature: 400
  pressure: 1.0
  ensemble: "npt"
  pair_style: "lj/cut"
```

#### simulation.loop_num

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.loop_num` |
| **类型** | `int` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 主循环次数。程序在每个循环中依次执行 NVE/limit 弛豫、bond_react 检查、NPT 平衡。总模拟时间 = `loop_num * run_per_loop * dt`。

**验证**: 必须大于 0。

**YAML 示例**:
```yaml
simulation:
  loop_num: 25000
```

#### simulation.dt

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.dt` |
| **类型** | `float` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 时间步长，单位 ps（皮秒）。这是 LAMMPS 的 `timestep` 命令的数值。

**注意**: LAMMPS 内部使用的 time unit 可能与 dt 的单位不同。对于 `real` unit 体系，dt 单位为 fs（飞秒），但此处的 dt 以 ps 为单位，程序内部自动转换为 LAMMPS 所需的单位。

**验证**: 必须大于 0。合理范围: 0.0001 ~ 0.01 ps。

**YAML 示例**:
```yaml
simulation:
  dt: 0.001
```

#### simulation.timestep

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.timestep` |
| **类型** | `int` |
| **默认值** | `1` |
| **必需** | 否 |

**说明**: LAMMPS `timestep` 设置值，单位为 fs（飞秒）。此值用于 LAMMPS 输入脚本中的 `timestep` 命令。

**注意**: `dt`（ps 单位）和 `timestep`（fs 单位）的关系是: `dt = timestep × 0.001`。在实际使用中，建议保持 `timestep = 1`，通过 `dt` 控制时间步长。

**YAML 示例**:
```yaml
simulation:
  timestep: 1
```

#### simulation.temperature

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.temperature` |
| **类型** | `float` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 目标温度，单位 K（开尔文）。用于 NVT 或 NPT 系综中的温度控制。

**验证**: 必须大于 0。

**YAML 示例**:
```yaml
simulation:
  temperature: 400
```

#### simulation.pressure

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.pressure` |
| **类型** | `float` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 目标压力，单位 atm（标准大气压）。仅在 `ensemble: "npt"` 时有效。当 `ensemble: "nvt"` 时此值被忽略。

**YAML 示例**:
```yaml
simulation:
  pressure: 1.0
```

#### simulation.ensemble

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.ensemble` |
| **类型** | `string` |
| **默认值** | `"npt"` |
| **必需** | 否 |

**说明**: 系综类型，支持 `"npt"` 和 `"nvt"` 两种。

| 选项 | 含义 | 使用场景 |
|------|------|----------|
| `"npt"` | 等温等压系综 | 模拟真实实验条件，盒子体积可变 |
| `"nvt"` | 正则系综（等温等容） | 固定体积下的平衡模拟 |

**注意**: 当 ensemble 为 `"nvt"` 时，`npt.pcouple` 参数被忽略。

**YAML 示例**:
```yaml
simulation:
  ensemble: "npt"
```

#### simulation.pair_style

| 属性 | 值 |
|------|-----|
| **字段名** | `simulation.pair_style` |
| **类型** | `string` |
| **默认值** | `"lj/cut"` |
| **必需** | 否 |

**说明**: LAMMPS `pair_style` 设置。此值不仅用于 LAMMPS 输入脚本生成，还在 SOAP 描述符计算中用于获取正确的 neighbor list。

**YAML 示例**:
```yaml
simulation:
  pair_style: "lj/cut"
```

---

### 2.2 steps - 步数配置

```yaml
steps:
  bond_react_check: 1
  run_per_loop: 50
  nve_limit: 20
```

#### steps.bond_react_check

| 属性 | 值 |
|------|-----|
| **字段名** | `steps.bond_react_check` |
| **类型** | `int` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 每个 loop 中用于执行键反应检查的步数。这是进行 bond/react 或 bond/create 检测并执行反应所分配的 LAMMPS 步数。

**典型值**: 1（每次 loop 执行一次检查即可）。

#### steps.run_per_loop

| 属性 | 值 |
|------|-----|
| **字段名** | `steps.run_per_loop` |
| **类型** | `int` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 每个 loop 运行的总步数。该值被拆分为三个阶段: NVE/limit 弛豫 + bond_react 检查 + 正常 NPT 平衡。

#### steps.nve_limit

| 属性 | 值 |
|------|-----|
| **字段名** | `steps.nve_limit` |
| **类型** | `int` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: NVE/limit 步数，用于新反应原子对的局部弛豫。反应发生后，新形成键的原子附近需要短时间 NVE/limit 弛豫以消除局部应力。

#### 重要约束: run_per_loop > nve_limit + bond_react_check

**说明**: `run_per_loop` 必须大于 `nve_limit + bond_react_check`，否则 NPT 平衡阶段的步数为零或负数。

**正常 NPT 步数计算公式**:
```
normal_npt_step = run_per_loop - nve_limit - bond_react_check
```

**若 normal_npt_step 过小（< 10）**，程序发出警告，提示可能影响体系平衡。

**各阶段执行顺序**:
```
1. NVE/limit 弛豫 (nve_limit 步)
2. bond_react 检查 (bond_react_check 步)
3. 正常 NPT 平衡 (normal_npt_step 步)
```

**YAML 示例**:
```yaml
steps:
  bond_react_check: 1    # 1 步做反应检查
  run_per_loop: 50       # 每 loop 共 50 步
  nve_limit: 20          # 20 步做 NVE/limit 弛豫
  # normal_npt_step = 50 - 20 - 1 = 29 步
```

---

### 2.3 npt - NPT/控温控压参数

```yaml
npt:
  tcouple: 100
  pcouple: 1000
```

#### npt.tcouple

| 属性 | 值 |
|------|-----|
| **字段名** | `npt.tcouple` |
| **类型** | `float` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 温度耦合常数，单位 fs（飞秒）。控制 thermostat 对温度的响应速度。较小的值表示更强的温度耦合（温度波动更小），较大的值表示更弱的耦合。

**合理范围**: 10 ~ 10000 fs。

**YAML 示例**:
```yaml
npt:
  tcouple: 100
```

#### npt.pcouple

| 属性 | 值 |
|------|-----|
| **字段名** | `npt.pcouple` |
| **类型** | `float` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: 压力耦合常数，单位 fs（飞秒）。控制 barostat 对压力的响应速度。仅在 `ensemble: "npt"` 时生效。

**注意**: pcouple 应显著大于 tcouple（通常 5-10 倍），因为压力控制需要更长的时间尺度。

**合理范围**: 100 ~ 100000 fs。

**YAML 示例**:
```yaml
npt:
  pcouple: 1000
```

---

### 2.4 bond_react - bond/react 配置

**说明**: 此配置块定义使用 LAMMPS `fix bond/react` 机制进行反应模拟的相关参数。

**注意**: `bond_react` 与 `bond_create` 两种模式互斥，不能同时启用。

```yaml
bond_react:
  stabilization: 0.03
  reactions:
    - name: "rxn1"
      cutoff: 5.0
      map_file: "rxn1.map"
      pre_mol: "mol1"
      post_mol: "mol2"
      pre_template: "rxn1_pre.lammpstemplate"
      post_template: "rxn1_post.lammpstemplate"
      pre_mapping: "rxn1_pre_mapping.yaml"
      post_mapping: "rxn1_post_mapping.yaml"
```

#### bond_react.stabilization

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_react.stabilization` |
| **类型** | `float` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: LAMMPS bond/react 的 `stabilization` 参数，单位 Angstrom（埃）。此参数控制反应发生时原子位置的稳定化（relaxation）范围。新成键原子在此范围内调整位置以消除局部应力。

**合理范围**: 0.01 ~ 1.0 A（典型值 0.03）。

**YAML 示例**:
```yaml
bond_react:
  stabilization: 0.03
```

#### bond_react.reactions

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_react.reactions` |
| **类型** | `List[Dict]` |
| **默认值** | `[]`（空列表） |
| **必需** | 否 |

**说明**: 化学反应定义列表。每个元素定义一个通过 bond/react 机制进行模拟的化学反应。

##### reactions[].name

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].name` |
| **类型** | `string` |
| **必需** | **是** |

**说明**: 反应名称。必须与 `reactions/{name}/` 子目录同名，程序从该子目录自动查找反应文件。

**YAML 示例**:
```yaml
reactions:
  - name: "rxn1"
```

##### reactions[].cutoff

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].cutoff` |
| **类型** | `float` |
| **必需** | **是** |

**说明**: 反应截断半径，单位 Angstrom（埃）。当反应前分子间的特定原子对距离小于此值时，触发反应。

**YAML 示例**:
```yaml
reactions:
  - cutoff: 5.0
```

##### reactions[].map_file

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].map_file` |
| **类型** | `string` |
| **必需** | **否**（程序自动在 `reactions/{name}/` 下查找 `{name}.map`） |

**说明**: bond/react 的 map 文件。定义反应前后原子的对应关系（equivalences）、边界原子（edge atoms）、引发原子（initiator atoms）。程序自动在 `reactions/{name}/` 目录下查找 `{name}.map` 文件。

**YAML 示例**:
```yaml
reactions:
  - map_file: "rxn1.map"
```

##### reactions[].pre_mol / reactions[].post_mol

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].pre_mol` / `reactions[].post_mol` |
| **类型** | `string` |
| **必需** | **是** |

**说明**: 反应前/反应后的分子模板名称。此名称必须与 `molecules` 配置块中的键一致，用于在 LAMMPS 中通过 `molecule` 命令载入对应模板文件。

**YAML 示例**:
```yaml
reactions:
  - pre_mol: "mol1"
    post_mol: "mol2"
```

##### reactions[].pre_template / reactions[].post_template

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].pre_template` / `reactions[].post_template` |
| **类型** | `string` |
| **必需** | **是**（程序自动在 `reactions/{name}/` 下查找） |

**说明**: 反应前/反应后的 LAMMPS 模板文件（`.lammpstemplate`）。程序自动在 `reactions/{name}/` 目录下查找 `{name}_pre.lammpstemplate` 和 `{name}_post.lammpstemplate`。

**YAML 示例**:
```yaml
reactions:
  - pre_template: "rxn1_pre.lammpstemplate"
    post_template: "rxn1_post.lammpstemplate"
```

##### reactions[].pre_mapping / reactions[].post_mapping

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].pre_mapping` / `reactions[].post_mapping` |
| **类型** | `string` |
| **默认值** | `""`（空字符串，可选） |
| **必需** | 否（推荐提供） |

**说明**: 反应前/反应后的 CG 映射文件路径。这些映射文件定义模板原子在反应前后如何映射到 CG beads。程序自动在 `reactions/{name}/` 目录下查找 `{name}_pre_mapping.yaml` 和 `{name}_post_mapping.yaml`。

**YAML 示例**:
```yaml
reactions:
  - pre_mapping: "rxn1_pre_mapping.yaml"
    post_mapping: "rxn1_post_mapping.yaml"
```

##### reactions[].multi_object_pairs (v2.6+)

| 属性 | 值 |
|------|-----|
| **字段名** | `reactions[].multi_object_pairs` |
| **类型** | `List[List[int]]` |
| **默认值** | `[]`（可选） |
| **必需** | 否 |

**说明**: 多目标反应对列表。用于处理一个反应中涉及多对原子同时成键的复杂场景。每个元素为 `[atom1, atom2, ...]` 形式的列表。

**YAML 示例**:
```yaml
reactions:
  - multi_object_pairs:
      - [1, 2]
      - [3, 4]
```

---

### 2.5 bond_create (v2.6+) - bond/create 配置

**说明**: 此配置块定义使用 LAMMPS `fix bond/create` 机制进行反应模拟的相关参数。此模式下，程序不依赖 bond/react 的分子模板替换，而是通过原子类型对之间的成键条件来驱动反应。

**与 bond_react 的互斥关系**:
- 当 `bond_create.enabled = true` 时，`bond_react.reactions` **不能**存在。
- 两种模式选其一，不能同时启用。
- 通过在 `lammps_params.yaml` 中顶层同时定义 `bond_react` 和 `bond_create` 块来实现配置共存，但 `enabled: true` 标志决定激活哪个模式。

```yaml
bond_create:
  enabled: true
  pairs:
    - itype: 3
      jtype: 5
      Nevery: 1
      Rmin: 3.79
      bondtype: 1
      iparam:
        maxbond: 1
        newtype: 1
      jparam:
        maxbond: 1
        newtype: 3
      prob:
        fraction: 0.5
        seed: 12345
  cg_update:
    type_map:
      3: 1
      4: 2
      5: 3
      6: 4
  sequential: true
  relax_radius: 0.0
```

#### bond_create.enabled

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_create.enabled` |
| **类型** | `bool` |
| **默认值** | `false` |
| **必需** | 否 |

**说明**: 启用 bond/create 模式的开关。当为 `true` 时，程序使用 bond/create 机制代替 bond/react。

#### bond_create.pairs

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_create.pairs` |
| **类型** | `List[Dict]` |
| **默认值** | `[]` |
| **必需** | **是**（当 `enabled: true` 时） |

**说明**: bond/create 类型对列表。每个元素定义一对原子类型之间的成键规则。至少需要一个类型对。

##### bond_create.pairs[].itype / jtype

| 属性 | 值 |
|------|-----|
| **字段名** | `pairs[].itype` / `pairs[].jtype` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 参与成键的两种原子类型（LAMMPS type 编号）。itype 和 jtype 必须为正整数，且通常 itype <= jtype。

**YAML 示例**:
```yaml
pairs:
  - itype: 3
    jtype: 5
```

##### bond_create.pairs[].Nevery

| 属性 | 值 |
|------|-----|
| **字段名** | `pairs[].Nevery` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 反应检查间隔步数。每 Nevery 步进行一次成键条件检查。通常设为 1（每一步都检查）。

**YAML 示例**:
```yaml
pairs:
  - Nevery: 1
```

##### bond_create.pairs[].Rmin

| 属性 | 值 |
|------|-----|
| **字段名** | `pairs[].Rmin` |
| **类型** | `float` |
| **必需** | **是** |

**说明**: 成键距离阈值，单位 Angstrom（埃）。当 itype 与 jtype 原子间距小于 Rmin 时，触发成键。

**YAML 示例**:
```yaml
pairs:
  - Rmin: 3.79
```

##### bond_create.pairs[].bondtype

| 属性 | 值 |
|------|-----|
| **字段名** | `pairs[].bondtype` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 新创建键的类型编号（LAMMPS bond type）。必须为正整数。

**YAML 示例**:
```yaml
pairs:
  - bondtype: 1
```

##### bond_create.pairs[].iparam / jparam

| 属性 | 值 |
|------|-----|
| **字段名** | `pairs[].iparam` / `pairs[].jparam` |
| **类型** | `Dict` |
| **默认值** | `{}`（可选） |
| **必需** | 否 |

**说明**: 控制 itype/jtype 原子成键后的参数。

**子字段**:

| 子字段 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `maxbond` | `int` | `0` | 原子最大允许成键数。0 表示不限制 |
| `newtype` | `int` | 保持原 type | 成键后原子类型的变化值。成键后原子 type 变为 newtype |

**YAML 示例**:
```yaml
pairs:
  - iparam:
      maxbond: 1
      newtype: 1
    jparam:
      maxbond: 1
      newtype: 3
```

##### bond_create.pairs[].prob

| 属性 | 值 |
|------|-----|
| **字段名** | `pairs[].prob` |
| **类型** | `Dict` |
| **默认值** | `{}`（可选） |
| **必需** | 否 |

**说明**: 成键概率控制。用于模拟有限反应概率（非每次满足距离条件都成键）。

**子字段**:

| 子字段 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `fraction` | `float` | `1.0` | 成键概率（0.0 ~ 1.0）。1.0 表示每次满足条件都成键 |
| `seed` | `int` | 无（不启用概率模式） | 随机数种子，用于概率判断 |

**注意**: 当 `fraction < 1.0` 时，`seed` 为必需项。

**YAML 示例**:
```yaml
pairs:
  - prob:
      fraction: 0.5
      seed: 12345
```

#### bond_create.cg_update.type_map

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_create.cg_update.type_map` |
| **类型** | `Dict[int, int]` |
| **默认值** | `{}`（可选） |
| **必需** | 否 |

**说明**: CG 映射更新表。键为旧的 CG bead type，值为新的 CG bead type。当 bond/create 成键后，程序根据此表更新 CG 映射中对应 bead 的类型。

**例如**: `{3: 1, 4: 2, 5: 3, 6: 4}` 表示旧 type 3 变为 type 1，旧 type 4 变为 type 2，依此类推。

**YAML 示例**:
```yaml
bond_create:
  cg_update:
    type_map:
      3: 1
      4: 2
      5: 3
      6: 4
```

#### bond_create.sequential

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_create.sequential` |
| **类型** | `bool` |
| **默认值** | `true` |
| **必需** | 否 |

**说明**: 是否顺序执行多个 bond/create fix。当为 `true` 时，程序依次执行每对 fix，避免多个 fix 同时操作同组原子导致冲突。

#### bond_create.relax_radius

| 属性 | 值 |
|------|-----|
| **字段名** | `bond_create.relax_radius` |
| **类型** | `float` |
| **默认值** | `0.0` |
| **必需** | 否 |

**说明**: 松弛反应原子组半径，单位 Angstrom（埃）。新成键后，对以反应原子为中心、relax_radius 为半径的球形区域内的原子进行位置松弛，消除局部应力。

---

### 2.6 molecules - 分子模板配置

```yaml
molecules:
  mol1: "rxn1_pre.lammpstemplate"
  mol2: "rxn1_post.lammpstemplate"
```

| 属性 | 值 |
|------|-----|
| **字段名** | `molecules` |
| **类型** | `Dict[string, string]` |
| **默认值** | `{}` |
| **必需** | 否（bond/react 模式下必需） |

**说明**: 分子模板名称到模板文件路径的映射。这些模板文件被 LAMMPS 的 `molecule` 命令使用，用于在 bond/react 反应中替换反应前后的分子结构。

**键**: 分子模板名称（如 `"mol1"`, `"mol2"`），与 `bond_react.reactions[].pre_mol` 和 `post_mol` 对应。

**值**: 模板文件路径（相对于配置目录）。

**YAML 示例**:
```yaml
molecules:
  mol1: "rxn1_pre.lammpstemplate"
  mol2: "rxn1_post.lammpstemplate"
```

---

### 2.7 files - 文件配置

```yaml
files:
  data_file: "system.data"
  input_script: "in.epoxy.stabilized"
  initial_cg_mapping: "AtomId_BeadId_compare_list.csv"
  read_data_extra:
    special_per_atom: 3
    bond_per_atom: 3
    angle_per_atom: 3
    dihedral_per_atom: 3
  output_cg_trajectory: "cg_trajectory.lammpstrj"
  output_reaction_count: "reaction_num.txt"
  output_final_data: "final_frame.data"
  output_final_mapping: "final_cg_compare_list.csv"
  output_bonds_record: "bonds_record.npz"
  output_cg_bonds: "cg_bonds.txt"
  output_cg_angles: "cg_angles.txt"
  output_cg_dihedrals: "cg_dihedrals.txt"
```

#### files.data_file

| 属性 | 值 |
|------|-----|
| **字段名** | `files.data_file` |
| **类型** | `string` |
| **默认值** | `"system.data"` |
| **必需** | 否 |

**说明**: LAMMPS 数据文件路径。此文件包含体系的初始原子坐标、键信息、盒子尺寸、原子类型和质量等。

**YAML 示例**:
```yaml
files:
  data_file: "system.data"
```

#### files.input_script

| 属性 | 值 |
|------|-----|
| **字段名** | `files.input_script` |
| **类型** | `string` |
| **默认值** | `""`（空字符串） |
| **必需** | 否 |

**说明**: LAMMPS 输入脚本路径。如果设置此值，程序将使用外部 LAMMPS 输入脚本代替自动生成的脚本。适用于需要精细控制 LAMMPS 命令的高级用户。

**YAML 示例**:
```yaml
files:
  input_script: "in.epoxy.stabilized"
```

#### files.initial_cg_mapping

| 属性 | 值 |
|------|-----|
| **字段名** | `files.initial_cg_mapping` |
| **类型** | `string` |
| **默认值** | `"AtomId_BeadId_compare_list.csv"` |
| **必需** | 否 |

**说明**: 初始 CG 映射文件路径。此文件记录全原子 ID 到 CG bead ID 的对应关系，包含每行: `AA_atom_id, bead_id, mol_id, bead_type`。

**YAML 示例**:
```yaml
files:
  initial_cg_mapping: "AtomId_BeadId_compare_list.csv"
```

#### files.read_data_extra

| 属性 | 值 |
|------|-----|
| **字段名** | `files.read_data_extra` |
| **类型** | `Dict` |
| **默认值** | `{special_per_atom: 3, bond_per_atom: 3, angle_per_atom: 3, dihedral_per_atom: 3}` |
| **必需** | 否 |

**说明**: `read_data` 命令的额外参数，控制 LAMMPS 在读取数据文件时为每个原子分配的数组大小。

**子字段说明**:

| 子字段 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `special_per_atom` | `int` | `3` | 每个原子特殊邻居列表大小。对应 LAMMPS 的 `special bonds` 功能 |
| `bond_per_atom` | `int` | `3` | 每个原子最大键数。若体系中有原子连接超过 3 个键，需增大此值 |
| `angle_per_atom` | `int` | `3` | 每个原子最大角数 |
| `dihedral_per_atom` | `int` | `3` | 每个原子最大二面角数 |

**重要提示**: 对于反应体系，反应后新形成的键可能导致某些原子连接数超过初始值。建议保守设置这些参数（如设为 6 或更高）。

**YAML 示例**:
```yaml
files:
  read_data_extra:
    special_per_atom: 3
    bond_per_atom: 3
    angle_per_atom: 3
    dihedral_per_atom: 3
```

#### 输出文件列表

| 字段名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `output_cg_trajectory` | `string` | `"cg_trajectory.lammpstrj"` | CG 轨迹文件（LAMMPS dump 格式）。每帧包含 CG beads 的位置、类型等信息 |
| `output_reaction_count` | `string` | `"reaction_num.txt"` | 反应计数输出文件。记录每种反应的发生次数 |
| `output_final_data` | `string` | `"final_frame.data"` | 最终帧 LAMMPS 数据文件。可用于断点续算 |
| `output_final_mapping` | `string` | `"final_cg_compare_list.csv"` | 最终 CG 映射文件。记录最终帧的 AA->CG 对应关系 |
| `output_bonds_record` | `string` | `"bonds_record.npz"` | 键记录 NPZ 文件（NumPy 压缩格式）。保存全模拟过程中的键状态快照 |
| `output_cg_bonds` | `string` | `"cg_bonds.txt"` | CG 键拓扑输出文件 |
| `output_cg_angles` | `string` | `"cg_angles.txt"` | CG 角拓扑输出文件 |
| `output_cg_dihedrals` | `string` | `"cg_dihedrals.txt"` | CG 二面角拓扑输出文件 |

**所有输出文件路径均为相对于运行工作目录的路径或绝对路径。**

**YAML 示例**:
```yaml
files:
  output_cg_trajectory: "cg_trajectory.lammpstrj"
  output_reaction_count: "reaction_num.txt"
  output_final_data: "final_frame.data"
  output_final_mapping: "final_cg_compare_list.csv"
  output_bonds_record: "bonds_record.npz"
  output_cg_bonds: "cg_bonds.txt"
  output_cg_angles: "cg_angles.txt"
  output_cg_dihedrals: "cg_dihedrals.txt"
```

---

### 2.8 mass_list - 元素质量映射

| 属性 | 值 |
|------|-----|
| **字段名** | `mass_list`（在 lammps_params.yaml 中） |
| **类型** | `Dict[int, float]` |
| **默认值** | `{}` |
| **必需** | 否 |

**说明**: 原子类型到质量的映射表。键为原子类型编号（LAMMPS type），值为质量（单位与 LAMMPS 单位制一致，`real` 单位下为 g/mol）。

**质量加载优先级**:
1. 如果 `{config_dir}/mass_list.yaml` 存在，则优先使用该文件中的定义。
2. 如果 `lammps_params.yaml` 中存在 `mass_list` 字段，则使用此定义。
3. 如果以上均无，程序从 LAMMPS data 文件中自动提取 Masses 部分。

**YAML 示例**:
```yaml
mass_list:
  1: 12.01
  2: 1.008
  3: 15.999
```

---

## 3. 映射文件格式 (mapping YAML)

**文件位置**: 由 `system.yaml` 中 `mapping_files[].path` 指定

**作用**: 定义全原子到 CG beads 的映射规则。

---

### 3.1 site-types - bead 类型定义

| 属性 | 值 |
|------|-----|
| **字段名** | `site-types` |
| **类型** | `Dict[string, Dict]` |
| **必需** | **是** |

**说明**: 定义每种 CG bead 类型包含的原子和质心权重。每个键为 bead 类型名称，值为包含 `index` 和 `x-weight` 两个字段的字典。

#### site-types.{name}.index

| 属性 | 值 |
|------|-----|
| **字段名** | `site-types.{name}.index` |
| **类型** | `List[int]` |
| **必需** | **是** |

**说明**: 相对原子索引列表（0-based）。这些索引是相对于该 bead 的起始原子位置（由 `config` 中的 `site_offset` 决定）的偏移量。

**索引计算**:
```
实际原子 ID = anchor + site_offset + index[i] + 1
```

**示例**: 若 `index: [0, 1, 2, 3]`，则包含 4 个原子，位置相对于 bead 起始点偏移 0, 1, 2, 3。

**关键概念**: index 可以是不连续的。例如 bead 包含原子 1-4 和 7-15 时:
```yaml
Bead1:
  index: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
```
这表示 bead 包含起始位置的偏移 0,1,2,3 和 6,7,8,9,10,11,12,13,14 处的原子（跳过了偏移 4,5）。

#### site-types.{name}.x-weight

| 属性 | 值 |
|------|-----|
| **字段名** | `site-types.{name}.x-weight` |
| **类型** | `List[float]` |
| **必需** | **是** |

**说明**: 各原子的质量权重，用于计算 CG bead 的质心位置。长度必须与 `index` 完全一致。

**质心计算**:
```
bead_position = sum(weight_i * atom_i_position) / sum(weight_i)
```

通常使用原子质量作为权重，也可以使用自定义权重。

**YAML 示例**:
```yaml
site-types:
  Head:
    index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    x-weight: [12, 12, 1, 1, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1]
  Mid:
    index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    x-weight: [12, 12, 1, 1, 12, 12, 12, 1, 1, 1, 1, 1, 1]
  End:
    index: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    x-weight: [1, 12, 12, 1, 1, 12, 12, 12, 1, 1, 1, 1, 1, 1]
```

---

### 3.2 config - 映射配置

| 属性 | 值 |
|------|-----|
| **字段名** | `config` |
| **类型** | `List[Dict]` |
| **必需** | **是** |

**说明**: 映射配置列表。每个元素定义一个分子的映射规则。对于多分子体系（`copies > 1`），程序自动重复此配置。

#### config[].anchor

| 属性 | 值 |
|------|-----|
| **字段名** | `config[].anchor` |
| **类型** | `int` |
| **默认值** | 无 |
| **必需** | **是** |

**说明**: anchor 始终固定为 0。原因详见 [system.mapping_files[].anchor 固定为 0 的原因](#mapping_filesanchor-固定为-0-的原因)。

**YAML 示例**:
```yaml
config:
  - anchor: 0
```

#### config[].repeat

| 属性 | 值 |
|------|-----|
| **字段名** | `config[].repeat` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 分子数量。通常与 `system.yaml` 中该 mapping 文件的 `copies` 值一致。

**YAML 示例**:
```yaml
config:
  - repeat: 1
```

#### config[].offset

| 属性 | 值 |
|------|-----|
| **字段名** | `config[].offset` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 每个分子的原子数。程序利用此值计算下一个分子的起始原子 ID:
```
下一个分子的 anchor = 当前 anchor + offset
```

**YAML 示例**:
```yaml
config:
  - offset: 132
```

#### config[].sites

| 属性 | 值 |
|------|-----|
| **字段名** | `config[].sites` |
| **类型** | `List[List]` |
| **必需** | **是** |

**说明**: bead 列表。每个元素为 `[site_type_name, start_offset]` 形式的列表。

- `site_type_name`: 引用 `site-types` 中定义的类型名称。
- `start_offset`: 该 bead 的起始原子在分子内的位置（0-based）。

**原子 ID 计算公式**:
```
atom_id = anchor + site_offset + index[i] + 1
```

其中 `anchor` 为分子起始（累积后），`site_offset` 为该 bead 的 `start_offset`，`index[i]` 为 `site-types` 中定义的相对索引，`+1` 是因为 LAMMPS 原子 ID 从 1 开始。

**多重复制时的偏移计算**:
当 `copies > 1` 时，程序对外层循环应用 `copies` 次，每次增加 `offset`:
```
分子 #k: anchor_global = anchor_global_base + k * offset
```

**YAML 示例**:
```yaml
config:
  - anchor: 0
    repeat: 1
    offset: 132
    sites:
      - [Head, 0]    # bead 名为 Head，起始位置 0（原子 1-14）
      - [Mid, 14]    # bead 名为 Mid，起始位置 14（原子 15-27）
      - [End, 118]   # bead 名为 End，起始位置 118（原子 119-132）
```

**完整原子 ID 计算示例**（单分子，anchor=0, offset=132）:
```
Head 原子: anchor(0) + site_offset(0) + index[0..13] + 1 = [1..14]
Mid  原子: anchor(0) + site_offset(14) + index[0..12] + 1 = [15..27]
End  原子: anchor(0) + site_offset(118) + index[0..13] + 1 = [119..132]
```

---

## 4. 反应映射文件格式

**文件位置**: `{config_dir}/reactions/{name}/{name}_pre_mapping.yaml` 和 `_post_mapping.yaml`

**作用**: 定义反应前后模板原子到 CG beads 的映射关系。用于在反应发生时更新 CG 映射，确保 CG 轨迹正确反映反应引起的 bead 结构变化。

### 格式规范

```yaml
mapping:
  bead_id:
    atoms: [模板原子 ID 列表]
    bead_type: bead_type_ID
```

#### mapping.{bead_id}.atoms

| 属性 | 值 |
|------|-----|
| **字段名** | `mapping.{bead_id}.atoms` |
| **类型** | `List[int]` |
| **必需** | **是** |

**说明**: 组成该 bead 的模板原子 ID 列表（1-based，对应模板文件中的原子编号）。这些原子在反应后的新分子中属于同一个 CG bead。

#### mapping.{bead_id}.bead_type

| 属性 | 值 |
|------|-----|
| **字段名** | `mapping.{bead_id}.bead_type` |
| **类型** | `int` |
| **必需** | **是** |

**说明**: 该 bead 在 CG 体系中的类型 ID（对应 `bead_type_names` 中的值）。反应后 bead_type 可能与反应前不同。

**YAML 示例**:
```yaml
# 反应前映射 (rxn1_pre_mapping.yaml)
mapping:
  1:
    atoms: [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    bead_type: 1
  2:
    atoms: [5, 6, 16, 17]
    bead_type: 2
  3:
    atoms: [18, 19, 20, 21, 22, 23]
    bead_type: 3

# 反应后映射 (rxn1_post_mapping.yaml) - bead_type 可能变化
mapping:
  1:
    atoms: [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    bead_type: 1
  2:
    atoms: [5, 6, 16, 17]
    bead_type: 2
  3:
    atoms: [18, 19, 20, 21, 22, 23]
    bead_type: 2  # 反应后 bead 类型从 3 变为 2
```

---

## 5. 断点续算配置

**作用**: LmpPy 支持从上次运行结束的状态继续模拟。断点续算需要两个关键文件。

### 续算文件

| 文件 | 字段配置 | 说明 |
|------|----------|------|
| `final_frame.data` | `files.data_file` | 包含上一轮运行最后一步的完整体系状态（原子坐标、盒子和拓扑信息） |
| `final_cg_compare_list.csv` | `files.initial_cg_mapping` | 上一轮运行结束时的 CG 映射关系 |

### 续算步骤

1. **复制文件**: 将上一轮输出目录中的 `final_frame.data` 和 `final_cg_compare_list.csv` 复制到新配置目录或更新 `lammps_params.yaml` 中的文件路径。

2. **更新 lammps_params.yaml**:
   ```yaml
   files:
     data_file: "final_frame.data"            # 指向上一轮的最终 data 文件
     initial_cg_mapping: "final_cg_compare_list.csv"  # 指向上一轮的最终 CG 映射
   ```

3. **保持其他参数不变**（特别是 `mapping_files`, `bead_type_names`, `reactions` 等）。

### 续算验证要点

- `data_file` 必须指向有效的 LAMMPS data 文件（由上一轮输出的 `final_frame.data`）。
- `initial_cg_mapping` 必须指向有效的 CG 映射 CSV 文件（由上一轮输出的 `final_cg_compare_list.csv`）。
- 确保 `data_file` 中的原子数、分子类型与 `mapping_files` 配置一致。
- 确保 `loop_num` 设置为剩余需要的循环数（而非总数）。

---

## 6. 附录: 完整配置示例

### system.yaml 完整示例

```yaml
# system.yaml - 体系配置
system:
  name: "环氧树脂固化体系"

  # 全局 bead 类型名称到 LAMMPS type ID 的映射
  bead_type_names:
    Head: 1
    Mid: 2
    End: 3

  # CG 映射文件列表
  mapping_files:
    - path: "mapping/mapping_epoxy.yaml"
      copies: 1
    - path: "mapping/mapping_curing_agent.yaml"
      copies: 2

  # 工作目录和输出目录
  work_dir: "work"
  output_dir: "output"

  # 可选: 质量列表文件
  # mass_list: "mass_list.yaml"
```

### lammps_params.yaml 完整示例（bond/react 模式）

```yaml
# lammps_params.yaml - LAMMPS 运行参数配置

simulation:
  loop_num: 25000
  dt: 0.001
  timestep: 1
  temperature: 400
  pressure: 1.0
  ensemble: "npt"
  pair_style: "lj/cut"

steps:
  bond_react_check: 1
  run_per_loop: 50
  nve_limit: 20

npt:
  tcouple: 100
  pcouple: 1000

bond_react:
  stabilization: 0.03
  reactions:
    - name: "rxn1"
      cutoff: 3.0
      pre_mol: "mol1"
      post_mol: "mol2"

molecules:
  mol1: "reactions/rxn1/rxn1_pre.lammpstemplate"
  mol2: "reactions/rxn1/rxn1_post.lammpstemplate"

files:
  data_file: "system.data"
  input_script: ""
  initial_cg_mapping: "AtomId_BeadId_compare_list.csv"
  read_data_extra:
    special_per_atom: 3
    bond_per_atom: 3
    angle_per_atom: 3
    dihedral_per_atom: 3
  output_cg_trajectory: "cg_trajectory.lammpstrj"
  output_reaction_count: "reaction_num.txt"
  output_final_data: "final_frame.data"
  output_final_mapping: "final_cg_compare_list.csv"
  output_bonds_record: "bonds_record.npz"
  output_cg_bonds: "cg_bonds.txt"
  output_cg_angles: "cg_angles.txt"
  output_cg_dihedrals: "cg_dihedrals.txt"
```

### lammps_params.yaml 完整示例（bond/create 模式）

```yaml
# lammps_params.yaml - LAMMPS 运行参数 (bond/create 模式)

simulation:
  loop_num: 25000
  dt: 0.001
  timestep: 1
  temperature: 400
  pressure: 1.0
  ensemble: "npt"

steps:
  bond_react_check: 1
  run_per_loop: 50
  nve_limit: 20

npt:
  tcouple: 100
  pcouple: 1000

# bond/create 模式配置
# 注意: 启用 bond_create 时不能同时配置 bond_react.reactions
bond_create:
  enabled: true
  pairs:
    - itype: 3
      jtype: 5
      Nevery: 1
      Rmin: 3.79
      bondtype: 1
      iparam:
        maxbond: 1
        newtype: 1
      jparam:
        maxbond: 1
        newtype: 3
    - itype: 3
      jtype: 6
      Nevery: 1
      Rmin: 3.79
      bondtype: 2
      iparam:
        maxbond: 1
        newtype: 1
      jparam:
        maxbond: 1
        newtype: 4
  cg_update:
    type_map:
      3: 1
      4: 2
      5: 3
      6: 4
  sequential: true
  relax_radius: 0.0

files:
  data_file: "system.data"
  initial_cg_mapping: "AtomId_BeadId_compare_list.csv"
  read_data_extra:
    special_per_atom: 3
    bond_per_atom: 3
    angle_per_atom: 3
    dihedral_per_atom: 3
  output_cg_trajectory: "cg_trajectory.lammpstrj"
  output_reaction_count: "reaction_num.txt"
  output_final_data: "final_frame.data"
  output_final_mapping: "final_cg_compare_list.csv"
  output_bonds_record: "bonds_record.npz"
  output_cg_bonds: "cg_bonds.txt"
  output_cg_angles: "cg_angles.txt"
  output_cg_dihedrals: "cg_dihedrals.txt"
```

### 映射文件完整示例

```yaml
# mapping.yaml - CG 映射配置

site-types:
  Bead1:
    index: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    x-weight: [12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1]
  Bead2:
    index: [0, 1, 11, 12]
    x-weight: [12, 12, 1, 1]
  Bead3:
    index: [0, 1, 2, 3, 4, 5]
    x-weight: [12, 12, 1, 1, 1, 1]

config:
  - anchor: 0
    repeat: 1
    offset: 23
    sites:
      - [Bead1, 0]
      - [Bead2, 4]
      - [Bead3, 17]
```

### 反应映射文件完整示例

```yaml
# rxn1_post_mapping.yaml - 反应后 CG 映射
mapping:
  1:
    atoms: [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    bead_type: 1
  2:
    atoms: [5, 6, 16, 17]
    bead_type: 2
  3:
    atoms: [18, 19, 20, 21, 22, 23]
    bead_type: 2  # 反应后 bead type 从 3 变为 2
```

---

> **版本历史**:
> - v1.0: 初始版本，覆盖 system.yaml, lammps_params.yaml, mapping YAML
> - v2.0: 新增 bond_create 模式完整配置文档
> - v2.1: 新增断点续算配置章节
