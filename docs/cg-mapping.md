# CG 映射配置详解

> 版本: 2.5
> 更新: 2026-06-22

## 概述

CG（粗粒化）映射定义了全原子（AA）模拟中哪些原子构成一个粗粒珠子（bead），以及这些珠子的质量权重。LmpPy 支持两种映射文件格式：

1. **YAML 格式**：配置文件驱动，定义 site-types + config
2. **CSV 格式**：运行时的 `CGCompareList` 数据结构，5 列格式

---

## 一、YAML 映射文件完整格式

YAML 映射文件由 `site-types` 和 `config` 两部分组成。

### 1.1 site-types：Bead 类型定义

每个 bead 类型通过名称定义，包含 `index`（相对原子索引）和 `x-weight`（质量权重）：

```yaml
site-types:
  Bead1:
    index: [0, 1, 2, 3, 6, 7, 8, 9, 10, 11, 12, 13, 14]
    x-weight: [12, 12, 12, 12, 1, 1, 1, 1, 1, 1, 1, 1, 1]
```

| 字段 | 说明 |
|------|------|
| `index` | 相对原子索引列表（0-based），相对于该 site 的起始位置 |
| `x-weight` | 对应每个原子的质量权重，用于质心（COM）计算 |

**重要**：`index` 是相对于 site 起始位置（见下文 `sites` 中的 `start_offset`）的偏移量，不是相对于分子起始位置，也不是 1-based 原子 ID。

### 1.2 config：映射配置

```yaml
config:
  - anchor: 0
    repeat: 1
    offset: 23
    sites:
      - [Bead1, 0]
      - [Bead2, 4]
      - [Bead3, 17]
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `anchor` | int | 起始原子 ID 偏移，**必须为 0**（程序自动累积偏移） |
| `repeat` | int | 该分子类型重复次数（即分子数量） |
| `offset` | int | 每个分子的原子总数，用于计算下一个分子的起始位置 |
| `sites` | list | `[bead类型名, site起始位置]` 列表 |

### 1.3 原子 ID 计算公式

```text
atom_id = anchor + site_start_offset + index_value + 1
```

其中 `+1` 是因为原子 ID 从 1 开始计数（1-based）。

**计算示例**（基于 mini_test 示例，23 原子分子，3 个 bead）：

```text
Bead1 (site_offset=0):
  atoms = 0 + 0 + [0,1,2,3,6,7,8,9,10,11,12,13,14] + 1
        = [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]

Bead2 (site_offset=4):
  atoms = 0 + 4 + [0,1,11,12] + 1
        = [5, 6, 16, 17]

Bead3 (site_offset=17):
  atoms = 0 + 17 + [0,1,2,3,4,5] + 1
        = [18, 19, 20, 21, 22, 23]
```

---

## 二、anchor=0 的原因

程序在处理多个 mapping 文件和多个分子拷贝时，**自动累积原子 ID 偏移量**。具体逻辑在 `MappingGenerator.generate()` 和 `yaml2csv_mapping.py` 中实现：

```python
# 每个 mapping 文件处理完成后，更新偏移量
current_aa_id_offset = next_aa_id
current_mol_id_offset = next_mol_id - 1
current_bead_id_offset = next_bead_id - 1
```

因此：
- 所有 mapping 文件的 `anchor` 必须为 0
- `next_aa_id = aa_id_offset + repeat * offset + anchor`
- 程序自动保证多分子、多 mapping 文件的原子 ID 不重叠

如果设置 `anchor != 0`，会导致累积偏移计算错误，引起原子 ID 越界或映射错误。

---

## 三、x-weight 质心计算方法

CG 坐标通过**质量加权质心（Center of Mass, COM）**计算：

```text
R_bead = (sum(m_i * r_i)) / (sum(m_i))
```

其中：
- `m_i`：第 i 个原子的质量权重（x-weight）
- `r_i`：第 i 个原子的坐标

**x-weight 设置原则**：

| 原子类型 | 典型 x-weight | 说明 |
|----------|---------------|------|
| 碳原子（C） | 12 | 碳的相对原子质量 ≈ 12 |
| 氢原子（H） | 1 | 氢的相对原子质量 ≈ 1 |
| 氧原子（O） | 16 | 氧的相对原子质量 ≈ 16 |
| 氮原子（N） | 14 | 氮的相对原子质量 ≈ 14 |

**权重不是必须等于精确原子质量**，但越接近实际质量，质心计算越准确。

权重数组长度必须与 `index` 数组长度一致，一一对应。

---

## 四、多分子累积偏移逻辑

### 4.1 单 mapping 文件多拷贝

一个 mapping 文件中 `repeat > 1` 表示多个相同分子：

```yaml
config:
  - anchor: 0
    repeat: 3          # 3 个分子
    offset: 23         # 每个分子 23 个原子
    sites:
      - [Bead1, 0]
      - [Bead2, 4]
      - [Bead3, 17]
```

**原子 ID 计算**（第 2 个分子，rep=2）：

```text
分子 2 的 base_aa_id = anchor + (rep - 1) * offset = 0 + 1 * 23 = 23

Bead1 (site_offset=0):
  atoms = 23 + 0 + [0,1,2,3,6,7,8,9,10,11,12,13,14] + 1
        = [24, 25, 26, 27, 30, 31, 32, 33, 34, 35, 36, 37, 38]

Bead2 (site_offset=4):
  atoms = 23 + 4 + [0,1,11,12] + 1
        = [28, 29, 39, 40]

Bead3 (site_offset=17):
  atoms = 23 + 17 + [0,1,2,3,4,5] + 1
        = [41, 42, 43, 44, 45, 46]
```

### 4.2 多 mapping 文件累积

当 `system.yaml` 配置了多个 mapping 文件时，每个文件处理完后偏移量递增：

```yaml
system:
  mapping_files:
    - path: "monomer_A.yaml"
      copies: 2      # 2 个 A 分子
    - path: "monomer_B.yaml"
      copies: 3      # 3 个 B 分子
```

**处理流程**：

```text
1. 处理 monomer_A.yaml（copies=2）
   - 第 1 份：aa_id_offset=0, mol_id_offset=0, bead_id_offset=0
   - 第 2 份：aa_id_offset=next_aa_id, mol_id_offset=next_mol_id-1, bead_id_offset=next_bead_id-1
   
2. 处理 monomer_B.yaml（copies=3）
   - 第 1 份：aa_id_offset=上一个 next_aa_id, ...
   - 第 2 份：继续递增
   - 第 3 份：继续递增
```

**完整计算示例**（假设 monomer_A 有 23 原子/3 bead，monomer_B 有 10 原子/2 bead）：

```text
monomer_A 第 1 份：  AA_ID 1-23,  Bead_ID 1-3,   Mol_ID 1
monomer_A 第 2 份：  AA_ID 24-46, Bead_ID 4-6,   Mol_ID 2
monomer_B 第 1 份：  AA_ID 47-56, Bead_ID 7-8,   Mol_ID 3
monomer_B 第 2 份：  AA_ID 57-66, Bead_ID 9-10,  Mol_ID 4
monomer_B 第 3 份：  AA_ID 67-76, Bead_ID 11-12, Mol_ID 5
```

---

## 五、反应映射文件格式

反应映射文件用于定义 LAMMPS bond/react 模板中原子到 CG bead 的映射关系。每个反应需要两个映射文件：反应前（pre_mapping）和反应后（post_mapping）。

### 5.1 格式

```yaml
# rxn1_pre_mapping.yaml
mapping:
  1:                            # bead_id（1-based）
    atoms: [1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    bead_type: 1                # bead 类型编号
  2:
    atoms: [5, 6, 16, 17]
    bead_type: 2
  3:
    atoms: [18, 19, 20, 21, 22, 23]
    bead_type: 3
```

| 字段 | 说明 |
|------|------|
| `mapping` 的键 | bead 编号（1-based），在反应模板内唯一 |
| `atoms` | 该 bead 包含的模板原子 ID 列表（1-based，对应 `.lammpstemplate` 中的原子编号） |
| `bead_type` | 该 bead 的类型编号（对应 system.yaml 中的 `bead_type_names` 映射值） |

### 5.2 反应前后映射的区别

反应前和反应后映射的主要区别在于 `bead_type` 可能会变化：

```yaml
# 反应前：Bead3 类型为 3
rxn1_pre_mapping.yaml →
  3:
    atoms: [18, 19, 20, 21, 22, 23]
    bead_type: 3

# 反应后：Bead3 变为类型 2
rxn1_post_mapping.yaml →
  3:
    atoms: [18, 19, 20, 21, 22, 23]
    bead_type: 2
```

这反映了化学反应导致的 bead 类型变化（如端 bead 在反应后变为链内 bead）。

### 5.3 与 site-types 映射的区别

| 特性 | site-types 映射 | 反应映射 |
|------|----------------|----------|
| 作用范围 | 完整体系的所有分子 | 单个反应模板 |
| 原子 ID | 相对索引+偏移计算 | 绝对模板原子 ID（1-based） |
| 格式 | `index`+`x-weight`+`config` | `mapping` 字典 |
| 质量权重 | 有（x-weight） | 无（从主映射继承） |

---

## 六、CGCompareList 数据结构

`CGCompareList` 是 LmpPy 运行时的核心数据结构，以 NumPy 数组存储完整的 AA 到 CG 映射关系。

### 6.1 数据格式

```
数组形状: (n_rows, 5)
列: [bead_id, mol_id, bead_type, AA_id, mass]
```

| 列名 | dtype | 说明 |
|------|-------|------|
| `bead_id` | int | 粗粒珠子编号（1-based，全局唯一） |
| `mol_id` | int | 分子编号（1-based） |
| `bead_type` | int | 粗粒珠子类型编号 |
| `AA_id` | int | 全原子 ID（1-based） |
| `mass` | float | 该原子的质量权重 |

### 6.2 CSV 文件格式

CSV 文件无行号索引，5 列无表头（或带表头取决于导出方式）：

```csv
bead_id,mol_id,bead_type,AA_id,mass
1,1,1,1,12
1,1,1,2,12
1,1,1,3,12
1,1,1,4,12
1,1,1,7,1
...
```

### 6.3 类方法

```python
from LmpPy.core import CGCompareList

# 从 CSV 加载
cg_list = CGCompareList.from_csv("final_cg_compare_list.csv")

# 保存到 CSV
cg_list.to_csv("output.csv")

# 访问属性
print(f"原子数: {len(cg_list.data)}")
print(f"Bead 数: {cg_list.n_beads}")
print(f"分子数: {cg_list.n_molecules}")

# 获取 NumPy 数组
mapping_array = cg_list.to_numpy()
```

---

## 七、YAML 到 CSV 的转换逻辑

YAML 格式的映射（site-types + config）通过以下步骤转换为 `CGCompareList` 的 CSV 格式：

### 7.1 转换流程

```
YAML (site-types + config)
    │
    ├─ 解析 site-types: bead 类型名 → {index, x-weight}
    ├─ 解析 config: anchor, repeat, offset, sites
    │
    ├─ 对每个 repeat（分子拷贝）:
    │   ├─ 计算 base_aa_id = anchor + (rep-1) * offset + aa_id_offset
    │   └─ 对每个 site:
    │       ├─ 计算 AA_id = base_aa_id + site_start + index + 1
    │       ├─ 分配 bead_id（全局递增）
    │       ├─ 分配 mol_id（全局递增）
    │       └─ 分配 bead_type（来自全局 bead_type_names 映射）
    │
    └─ 输出: [bead_id, mol_id, bead_type, AA_id, mass] 的 numpy 数组
```

### 7.2 核心代码逻辑（MappingGenerator.generate）

```python
for config in config_list:
    anchor = int(config['anchor'])    # 必须为 0
    repeat = int(config['repeat'])
    offset = int(config['offset'])
    
    for rep in range(1, repeat + 1):
        base_aa_id = anchor + (rep - 1) * offset + aa_id_offset
        
        for site_info in sites:
            site_name = site_info[0]
            site_start = site_info[1]
            
            atom_indices = site_types[site_name]['index']
            x_weights = site_types[site_name]['x-weight']
            
            for i, atom_idx in enumerate(atom_indices):
                aa_id = base_aa_id + site_start + atom_idx + 1
                aa_mass = x_weights[i]
                bead_list.append((current_bead_id, current_mol_id,
                                  bead_type, aa_id, aa_mass))
            
            current_bead_id += 1
        current_mol_id += 1
```

### 7.3 全局 bead_type_names 映射

`system.yaml` 中的 `bead_type_names` 将 site 类型名映射到全局 bead 类型编号：

```yaml
system:
  bead_type_names:
    Bead1: 1
    Bead2: 2
    Bead3: 3
```

这使得多个 mapping 文件可以共享同一套 bead 类型编号系统。

> **注意 `bead_type_names` 不能省略**：走 `ConfigLoader.load_system_config()` 时它是必需字段，缺失直接抛 `ConfigMissingFieldError`（`core/config_loader.py:533`），不存在"自动按顺序编号"的回退。
>
> 只有在绕过 ConfigLoader、直接调用 `MappingGenerator.generate()` 并传入不含 `bead_type_names` 的 `MappingConfig` 时，才会走 `bead_type_map.get(site_name, 1)` 的回退——即**全部落为 type 1**，而非按顺序递增（`core/mapping_generator.py:129`）。

---

## 八、yaml2csv_mapping.py 脚本使用

该脚本将 YAML 格式的 CG 映射文件转换为 CSV 格式，可直接在体系未运行前生成初始映射。

> **⚠️ 此脚本的 `-s` 参数用的不是平台的 `system.yaml` 格式。** 它读取的是
> `system.names`（mapping 文件路径列表）与 `system.numbers`（对应的 copies 数）两个键
> （`scripts/yaml2csv_mapping.py:127-128`），而平台的 `system.yaml` 用的是
> `system.mapping_files: [{path, copies}]`。直接传平台的 `system.yaml` 会抛
> `KeyError: 'names'`。此脚本尚未跟进配置格式变更，目前只适用于旧格式文件。

### 8.1 命令格式

```bash
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml -o mapping.csv
```

### 8.2 参数列表

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-s` / `--system-yaml` | 旧格式 system.yaml 文件路径（必需，需含 `system.names`/`system.numbers`） | — |
| `-o` / `--output` | 输出 CSV 文件名 | `AtomId_BeadId_compare_list.csv` |
| `--custom-map FILE` | 从 YAML/JSON 文件加载自定义 bead type 映射 `{site_type_name: bead_type_id}` | — |
| `--custom` | 使用硬编码的 EPR 自定义 bead type 映射（向后兼容兜底） | 关闭 |
| `-q` / `--quiet` | 安静模式，不输出详细信息 | 关闭 |

### 8.3 使用示例

```bash
# 基本用法（自动生成 bead type）
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml -o initial_mapping.csv

# 从文件加载自定义 bead type 映射
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml --custom-map types.yaml -o mapping.csv

# 使用硬编码的 EPR 映射
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml --custom -o mapping.csv

# 安静模式
python -m LmpPy.scripts.yaml2csv_mapping -s legacy_system.yaml -q
```

> **推荐替代**：新体系请直接用 `python -m LmpPy.scripts.generate_initial_mapping`（见 `cli-scripts.md` §9），
> 或调用 `MappingGenerator.generate()`，二者均支持当前的 `mapping_files` 格式。

### 8.4 自定义 bead type 映射

映射来源优先级（`scripts/yaml2csv_mapping.py:246-254`）：

1. `--custom-map FILE` — 从 YAML/JSON 文件读取 `{site_type_name: bead_type_id}`
2. `--custom` — 使用脚本内硬编码的 `CUSTOM_BEAD_TYPE_MAP`（EPR 体系，向后兼容兜底）
3. 都不给 — `custom_bead_type_map = None`，按 site 出现顺序自动连续编号

硬编码字典的完整内容：

```python
CUSTOM_BEAD_TYPE_MAP = {
    "Bead1": 1,    # chain E
    "Bead2": 2,    # chain P
    "Bead3": 1,    # chain reactor E
    "Bead4": 2,    # chain reactor P
    "Bead5": 3,    # E
    "Bead6": 4,    # P
    "Bead7": 1,    # chain head E
    "Bead8": 2     # chain head P
}
```

### 8.5 输出示例

```
Processing: mapping/mapping_epoxy.yaml (copies: 1)
  Copy 1: Generated 132 mappings

Total mappings generated: 132
Unique beads: 3
Unique molecules: 1
Unique bead types: 3
AA atoms: 1 to 132
AA mass range: 1.0 to 12.0

Output saved to: mapping.csv
```

---

## 九、完整示例：环氧树脂体系

### system.yaml

```yaml
system:
  name: "环氧树脂固化体系"
  bead_type_names:
    Head: 1
    Mid: 2
    End: 3
  mapping_files:
    - path: "mapping/mapping_epoxy.yaml"
      copies: 1
```

### mapping_epoxy.yaml

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

config:
  - anchor: 0
    repeat: 1
    offset: 132
    sites:
      - [Head, 0]
      - [Mid, 14]
      - [End, 118]
```

### 生成的 CSV（部分）

```csv
bead_id,mol_id,bead_type,AA_id,mass
1,1,1,1,12
1,1,1,2,12
1,1,1,3,1
1,1,1,4,1
...
2,1,2,15,12
2,1,2,16,12
...
3,1,3,119,1
3,1,3,120,12
...
```

| Bead | 类型 | 原子数 | 包含的 AA ID 范围 |
|------|------|--------|-------------------|
| bead_id=1 | Head（类型 1） | 14 | 1-14 |
| bead_id=2 | Mid（类型 2） | 13 | 15-27 |
| bead_id=3 | End（类型 3） | 14 | 119-132 |
