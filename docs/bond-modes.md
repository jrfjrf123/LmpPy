# LmpPy 两种建键模式详解：bond/react vs bond/create

## 1. 概述

LmpPy 在 LAMMPS 层面支持两种建键模式，分别对应 LAMMPS 的 `fix bond/react` 和 `fix bond/create` 命令。两种模式通过 `lammps_params.yaml` 中的配置互斥选择，同一运行时只能启用其中一种。

| 模式 | 建键策略 | 配置复杂度 | 适用方向 |
|------|---------|-----------|---------|
| bond/react | 模板化学匹配 | 高 | 明确的化学反应（环氧开环、聚氨酯形成等） |
| bond/create | 距离+类型条件 | 低 | 交联、渗透网络、非特异性键形成 |

核心区别在于：bond/react 依赖于 pre/post 分子模板来匹配反应前后的分子构型，而 bond/create 仅凭原子类型和空间距离判断是否建键。

---

## 2. bond/react 模式（模板匹配建键）

### 2.1 原理

bond/react 是 LAMMPS 内置的模板匹配建键机制。用户需要提供反应前（pre）和反应后（post）的分子模板，LAMMPS 在模拟过程中搜索与 pre 模板匹配的原子组，当满足截断半径条件时，将其替换为 post 模板的构型并创建/断裂相应键。

### 2.2 需要的文件

每个反应需要在 `config_dir/reactions/` 下有一个独立的子目录（如 `rxn1_EEE/`），包含以下文件：

| 文件 | 必需 | 说明 |
|------|------|------|
| `{name}_pre.lammpstemplate` | 是 | 反应前分子模板，描述反应物构型 |
| `{name}_post.lammpstemplate` | 是 | 反应后分子模板，描述产物构型 |
| `{name}.map` | 是 | bond/react 原子映射文件，包含 InitiatorIDs、EdgeIDs、Equivalences |
| `{name}_pre_mapping.yaml` | 推荐 | 反应前的 CG 映射（bead 分组及类型） |
| `{name}_post_mapping.yaml` | 推荐 | 反应后的 CG 映射（bead 分组及类型） |

### 2.3 配置示例（lammps_params.yaml）

```yaml
# lammps_params.yaml — bond/react 模式
bond_react:
  stabilization: 0.1
  reactions:
    - name: rxn1_EEE
      cutoff: 3.79
      pre_mol: rxn1_EEE_pre
      post_mol: rxn1_EEE_post
    - name: rxn8_PPP
      cutoff: 3.79
      pre_mol: rxn8_PPP_pre
      post_mol: rxn8_PPP_post
```

### 2.4 关键数据结构：ReactionInfo

`config_loader.py` 中定义的 `ReactionInfo` 数据类承载每个反应的信息：

```python
@dataclass
class ReactionInfo:
    name: str                           # 反应名称
    cutoff: float                       # 反应截断半径 (A)
    pre_mol: str                        # 反应前分子模板名
    post_mol: str                       # 反应后分子模板名
    map_file: str = ""                  # .map 文件完整路径
    pre_template: str = ""              # 反应前模板文件完整路径
    post_template: str = ""             # 反应后模板文件完整路径
    pre_mapping: str = ""               # 反应前 CG 映射文件完整路径
    post_mapping: str = ""              # 反应后 CG 映射文件完整路径
    rxn_dir: str = ""                   # reactions/{name}/ 目录完整路径
```

所有路径在 `load_reactions_from_directory()` 中自动由相对路径补全为绝对路径。

### 2.5 CG 映射更新方式

bond/react 模式的 CG 映射更新通过模板签名推断，由 `cg_reaction_identifier.py` 模块完成：

1. 从 reactions 目录加载所有模板的 CG 级反应签名（`load_template_signatures()`）
2. 将 LAMMPS 层面检测到的 AA 键变化转换为 CG 键变化（`atom_bonds_to_cg_bonds()` 和 `get_cg_bond_diff()`）
3. 识别新增 CG 键，通过 3-bead 类型签名匹配反应类型
4. 从匹配结果获取 `type_map`（如 `{3: 1, 5: 3}`），更新 CG 映射中的 bead_type

匹配流程分为两级：
- **阶段 1（3-bead 粗筛）**：从新增 CG 键两端识别 end 和 monomer bead，再找 interior neighbor，组合成 3-bead 签名 `(interior_type, end_type, monomer_type)` 查表。
- **阶段 2（chain 精筛）**：多个候选时，沿反应后 CG 键图验证类型链（`validate_chain()`）以唯一确定反应类型。仍无法唯一确定则 MPI Abort 终止。

### 2.6 适用场景

- 明确的化学反应：环氧开环、聚氨酯形成、酯化反应等
- 反应前后分子构型发生明确变化（原子的连接关系和/或类型发生变化）
- 需要精确控制化学计量比的场景

### 2.7 优点与缺点

**优点：**
- 化学反应匹配准确，模板保证了产物构型的化学合理性
- 支持复杂的多步反应（等价映射机制）
- 反应类型自动识别，无需手动跟踪

**缺点：**
- 配置复杂：每个反应需要 3-5 个文件，包括人工编写 .map 和模板文件
- CG 映射更新依赖模板签名推断逻辑，维护成本高
- 不支持随机建键（无概率参数）
- 运行前需要 pre/post 分子模板在 LAMMPS 中注册（`molecule` 命令）

---

## 3. bond/create 模式（距离条件建键）

### 3.1 原理

bond/create 是 LAMMPS 内置的基于距离条件的建键机制。当两个原子满足指定的原子类型和距离条件时，LAMMPS 自动创建它们之间的键。无需分子模板，所有参数通过 YAML 配置指定。

### 3.2 需要的配置

bond/create 模式只需要在 `lammps_params.yaml` 中配置参数，**不需要**任何模板文件或 reactions 目录结构。

#### 3.2.1 顶层配置结构

```yaml
# lammps_params.yaml — bond/create 模式
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
```

#### 3.2.2 BondCreatePair 字段详解

每个 `pairs` 列表中的元素对应一个 `BondCreatePair` 数据类 `(core/reaction_commands.py)`，字段含义如下：

| 字段 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| `itype` | 是 | - | 供体原子类型 ID（正整数） |
| `jtype` | 是 | - | 受体原子类型 ID（正整数） |
| `Nevery` | 是 | - | 每隔 N 步检查一次建键条件（正整数） |
| `Rmin` | 是 | - | 建键最大距离，单位 A（大于 0，必须 <= pair cutoff） |
| `bondtype` | 是 | - | 新创建键的类型 ID（正整数） |
| `iparam.maxbond` | 否 | 0 | i 原子最大键数上限（0=无限制） |
| `iparam.newtype` | 否 | itype | 达到最大键数后 i 原子变换为的新类型 |
| `jparam.maxbond` | 否 | 0 | j 原子最大键数上限（0=无限制） |
| `jparam.newtype` | 否 | jtype | 达到最大键数后 j 原子变换为的新类型 |
| `prob.fraction` | 否 | 1.0 | 建键概率，范围 (0.0, 1.0] |
| `prob.seed` | 否 | 无 | 随机数种子（正整数，prob.fraction < 1.0 时必填） |

#### 3.2.3 顶层可选字段

| 字段 | 默认值 | 说明 |
|------|--------|------|
| `cg_update.type_map` | {} | CG 映射类型更新表：`{old_bead_type: new_bead_type}`，仅在需要时填写 |
| `sequential` | true | 每对依次执行，避免多 fix 冲突 |
| `relax_radius` | 0.0 | 松弛反应原子组半径（A），0.0 表示不启用 |

### 3.3 CG 映射更新方式

bond/create 模式的 CG 映射更新通过 YAML `type_map` 直接查表，由 `update_cg_mapping_create()` 完成（`core/reaction_commands.py`）：

1. `atom_bonds_to_cg_bonds()`：将 AA 键转换为 CG 键（复用现有函数）
2. `get_cg_bond_diff()`：计算 CG 键差集（after - before），得到新建的 CG 键
3. 对每个新 CG 键两端的 bead，从 YAML 的 `type_map` 中查找旧类型对应的新类型
4. 向量化 mask 原地更新 `cg_mapping_data` 的 bead_type 列

与 bond/react 的重要区别：**bond/create 不依赖模板签名推断**，更新规则完全由用户通过 YAML 显式指定，更直接、可预测。

类型映射为空时，表示反应前后 bead_type 不变，CG 映射不会被更新。

### 3.4 适用于多个类型对

bond/create 模式支持在 `pairs` 列表中配置多个类型对，每对独立生成一个 LAMMPS `fix bond/create` 命令。这对于交联体系中存在多种端基-单体组合（如 E 端基 + E 单体、E 端基 + P 单体等）的场景特别有用。

### 3.5 与 bond/react 的互斥说明

两种模式不能同时启用。`config_loader.py` 在 `load_lammps_params()` 中强制执行以下互斥验证：

```python
if bond_create_data.get('enabled', False):
    if br.get('reactions'):
        raise ConfigValidationError(
            "bond_create.enabled=true 与 bond_react.reactions 不能同时存在，"
            "请选择其中一种模式"
        )
```

配置验证器 `ConfigValidator._validate_reaction_params()` 也包含相同的互斥检查。

---

## 4. 对比表

| 特性 | bond/react | bond/create |
|------|-----------|-------------|
| 建键方式 | 模板化学匹配（pre/post .lammpstemplate） | 距离 + 原子类型条件（无模板） |
| 配置复杂度 | 高：需 reactions 目录 + .map + 模板 + CG mapping | 低：仅 YAML 参数 |
| 必需文件数 | 每反应 3-5 个 | 0 个（纯 YAML） |
| 原子类型变化 | 模板等价映射推断 | iparam/jparam + newtype 显式指定 |
| CG 映射更新 | 模板签名推断（cg_reaction_identifier） | YAML type_map 直接查表 |
| 随机性 | 无 | prob 概率支持 |
| 支持的反应数 | 多个（通过 bond_react.reactions 列表） | 多个（通过 pairs 列表） |
| 每步检查频率 | 由 bond_react_check_step 控制 | 每对独立指定 Nevery |
| 最大键数限制 | 无 | iparam.maxbond / jparam.maxbond |
| 适用场景 | 明确的化学反应 | 交联、渗透网络 |
| 断键支持 | 是（模板自动处理） | 否（仅建键） |
| 松弛策略 | NVE/limit 分治 + `npt_grp_REACT` 动态组 | 全局 NVE/limit + 全局系综 |
| 运行时依赖 | 需在 LAMMPS 中注册 pre/post 分子模板 | 无需分子模板 |

---

## 5. reaction_commands.py 模块

`core/reaction_commands.py` 是两种建键模式的命令生成和 CG 映射更新中心模块，采用纯函数设计，不持有运行状态。

### 5.1 模块结构

```
core/reaction_commands.py
├── 异常类 BondCreateConfigError
├── 数据类 BondCreatePair         # 单个 bond/create 类型对配置
├── 数据类 BondCreateConfig       # bond/create 完整配置容器
├── _parse_single_pair()          # 解析并验证单个 pair 配置
├── load_bond_create_config()     # 从 YAML dict 加载并验证完整配置
├── get_reaction_mode()           # 判断当前反应模式
├── generate_fix_bond_create()    # 生成 bond/create 的 LAMMPS fix 命令
├── generate_fix_bond_react()     # 生成 bond/react 的 LAMMPS fix 命令
└── update_cg_mapping_create()    # bond/create 专属 CG 映射更新
```

### 5.2 BondCreatePair 数据类

`BondCreatePair` 封装了一个类型对的全部参数，并提供 `to_lammps_command()` 方法生成对应的 LAMMPS 命令字符串。核心字段见 3.2.2 节。

命令生成逻辑：

```python
def to_lammps_command(self, fix_id: str) -> str:
    # 基础部分
    "fix {fix_id} all bond/create {Nevery} {itype} {jtype} {Rmin} {bondtype}"
    # 可选部分（按需追加）
    "iparam {maxbond} {newtype}"
    "jparam {maxbond} {newtype}"
    "prob {fraction} {seed}"
```

### 5.3 bond/react 命令生成

`generate_fix_bond_react()` 从 `run_refactored.py` 的 `_run_bond_react()` 方法中提取，行为完全不变。

```python
def generate_fix_bond_react(reactions, stabilization: float) -> str:
    # 为每个反应生成 react 子句
    react_cmds = [
        f"react {rxn.name} all 1 0.0 {rxn.cutoff} "
        f"{rxn.pre_mol} {rxn.post_mol} {rxn.map_file}"
        for rxn in reactions
    ]
    # 组装完整的 fix 命令
    return (
        f"fix rxns all bond/react stabilization yes "
        f"npt_grp {stabilization} {' '.join(react_cmds)}"
    )
```

### 5.4 bond/create 命令生成

`generate_fix_bond_create()` 对 `pairs` 列表中的每个配置对生成一条独立的 fix 命令：

```python
def generate_fix_bond_create(config: BondCreateConfig) -> List[Tuple[str, str]]:
    # 每条命令格式：
    # fix bond_create_fix_{i} all bond/create Nevery itype jtype Rmin bondtype [keywords...]
```

每对使用独立的 fix ID（`bond_create_fix_0`, `bond_create_fix_1`, ...），便于在运行中独立控制。

---

## 6. cg_reaction_identifier.py 模块

`core/cg_reaction_identifier.py` 是 CG 级别反应识别模块，仅 **bond/react 模式** 使用。bond/create 模式不使用此模块的匹配逻辑。

### 6.1 核心功能

- 从 reactions 目录加载 3-bead 类型签名
- 基于 CG 键变化识别反应类型
- 作为 ReactionLocator + CGMapper 主流程的独立交叉验证机制

### 6.2 模块结构

```
cg_reaction_identifier.py
├── CGReactionSignature        # 从模板提取的 CG 级反应签名
├── load_template_signatures() # 加载所有模板的签名（生产级）
├── load_reaction_signatures() # 加载签名（smoke_validator 交叉验证用）
├── validate_chain()           # 沿 CG 键图验证类型链
├── match_reaction()           # 两级匹配：3-bead 粗筛 + chain 精筛
├── build_cg_bond_graph()      # CG 键表 -> 邻接图
├── get_cg_bond_diff()         # CG 键差集 (after - before)
└── identify_reaction()        # 主入口：从 CG 信息识别反应类型
```

### 6.3 3-bead 类型签名

bond/react 模式的核心概念是 **3-bead 类型签名**：`(interior_type, end_type, monomer_type)`。

- **interior bead**：链内部 bead，在 pre 图中与 end bead 有键连接，且度数较高
- **end bead**：链端 bead，位于链末端（pre 中度数为 1），其类型在反应后发生变化
- **monomer bead**：单体 bead，在 pre 图中孤立（无 bead 间键）

签名示例：`(1, 3, 5)` 表示 interior_type=1, end_type=3, monomer_type=5，对应 `rxn1_EEE` 反应。

### 6.4 动态推断 end/monomer 类型

模块不依赖硬编码的角色映射，而是通过以下规则动态推断：

1. **monomer bead**：pre 图中度数为 0（孤立）的 bead
2. **end bead**：pre 到 post 中类型发生变化的 bead，且在链上度数较小（末端）
3. **interior bead**：与 end bead 有键连接且度数最大的邻居 bead

### 6.5 独立交叉验证机制

`load_reaction_signatures()` 和 `load_template_signatures()` 是两个并存的签名加载函数：

- `load_template_signatures()`：生产级使用，返回完整的签名索引、类型集合和 `CGReactionSignature` 列表
- `load_reaction_signatures()`：供 smoke_validator 交叉验证使用，返回简单的 `{signature: info}` 字典

两者功能等价，但后者用于独立的验证管道，确保反应识别逻辑的正确性。

---

## 7. 选择指南

### 何时使用 bond/react

- **反应化学是明确已知的**：如环氧树脂的开环聚合、聚氨酯的异氰酸酯-羟基反应等
- **反应前后分子构型有本质变化**：原子连接方式和/或原子类型发生变化
- **需要精确的化学计量控制**：模板匹配确保了产物结构的化学合理性
- **预算人力投入模板文件的编写**：每个反应需要维护 3-5 个文件

### 何时使用 bond/create

- **交联反应**：聚合物链间随机形成连接，如硫化、辐射交联
- **渗透网络形成**：两种或多组分间随机形成键合
- **非特异性键形成**：不需要精确的化学匹配，只需满足距离条件
- **快速原型验证**：不需要编写模板文件，只需修改 YAML 配置
- **概率性建键需求**：通过 prob 参数控制建键概率，实现随机交联

### 决策流程图

```
需要用什么建键方式？
├── 反应前后分子构型明确已知，需要精确控制产物结构
│   └── 配置模板文件 → bond/react
├── 随机交联或渗透网络，原子类型+距离即可判定
│   └── 仅需 YAML 参数 → bond/create
└── 不确定
    ├── 已有 reactions 目录和模板文件 → bond/react（迁移成本高）
    ├── 从零开始搭建新体系 → bond/create（更快速）
    └── 需要概率性建键 → bond/create（唯一支持 prob 的模式）
```

### 迁移路径

如果项目初期使用 bond/react 模式，后续需要迁移到 bond/create：
1. 保留 reactions 目录中的 CG 映射文件（`*_mapping.yaml`）作为参考
2. 从映射文件中提取 bead_type 对应关系，填入 `cg_update.type_map`
3. 从 .map 文件和模板中提取原子类型对应关系，填入 `pairs` 列表
4. 删除 lammps_params.yaml 中的 `bond_react.reactions` 配置
5. 新增 `bond_create.enabled: true` 和对应的 pairs 配置
