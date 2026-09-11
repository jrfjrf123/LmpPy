# 反应模板文件说明

反应模板用于定义化学反应的结构变化。每个反应需要三个文件：

## 文件结构

```
reactions/
└── rxn1/                              # 反应目录名
    ├── rxn1_pre.lammpstemplate        # 反应前模板
    ├── rxn1_post.lammpstemplate       # 反应后模板
    └── rxn1.map                       # 反应映射文件
```

---

## 1. 模板文件格式 (*.lammpstemplate)

### 文件结构

```
N atoms
N_bonds bonds
N_angles angles
N_dihedrals dihedrals

Types
1 type1
2 type2
...

Coords
1 x y z
2 x y z
...

Bonds
1 bond_type atom1 atom2
2 bond_type atom1 atom2
...

Angles
1 angle_type atom1 atom2 atom3
...

Dihedrals
1 dihedral_type atom1 atom2 atom3 atom4
...
```

### 格式示例（省略中间行，仅示意结构）

> 下面的计数与内容为**格式示意**，不对应本仓库中的任何模板文件。
> 可直接运行的实例见 `reactions/rxn1/rxn1_pre.lammpstemplate`（23 atoms / 21 bonds / 34 angles / 31 dihedrals）。

```
34 atoms
35 bonds
57 angles
61 dihedrals

Types

1 1
2 2
3 2
...
34 7

Coords

1 1.11 0.27 -0.06
2 2.5 -0.32 -0.35
...
34 13.18 -7.01 0.44

Bonds

1 2 1 2
2 4 2 3
...
35 5 28 34

Angles

1 6 1 2 3
...
57 25 29 22 30

Dihedrals

1 1 1 2 3 4
...
61 24 33 27 28 34
```

---

## 2. 映射文件格式 (*.map)

### 文件结构

```
this is a map file

N_edge edgeIDs
N_equiv equivalences
N_constraints constraints

InitiatorIDs
id1
id2
...

EdgeIDs
id1
id2
...

Equivalences
pre_id post_id
pre_id post_id
...

Constraints
distance atom1 atom2 min_dist max_dist
...
```

### 示例: rxn1.map

```
this is a map file

2 edgeIDs
34 equivalences
1 constraints

InitiatorIDs

12
30

EdgeIDs

1
26

Equivalences

1	1
2	2
3	3
...
34	34

Constraints

distance 11 22 1.4 3.5
```

### 字段说明

| 字段 | 说明 |
|------|------|
| `InitiatorIDs` | 发起反应的原子ID，这些原子触发反应 |
| `EdgeIDs` | 边界原子ID，反应位点边界 |
| `Equivalences` | 反应前后原子ID对应关系 (pre → post) |
| `Constraints` | 反应发生条件约束 (距离等) |

---

## 3. 模板匹配流程

```
1. 检测键变化 → 定位参与反应的原子
2. 提取k-hop邻域
3. 构建局部键连矩阵
4. 与模板进行子图匹配
5. 确定反应类型和位点
```

---

## 4. 完整示例

> 本节为**示意性说明**，原子数等数值与仓库中的实际文件无关。
> 可运行的真实实例见 `reactions/rxn1/`。

### 环氧开环反应

**反应前模板** (`rxn1_pre.lammpstemplate`):
- 34个原子（示意值；仓库实际文件为 23 个）
- 包含环氧基团结构

**反应后模板** (`rxn1_post.lammpstemplate`):
- 34个原子 (数量不变)
- 环氧环打开，形成新键

**映射文件** (`rxn1.map`):
- 指定哪些原子发生变化
- 定义反应发生条件