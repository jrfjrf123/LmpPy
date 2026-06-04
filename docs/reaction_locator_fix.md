# ReactionLocator 修复记录

## 问题描述

在纯 CG 体系中，LAMMPS `fix bond/react` 正确执行了 9 次反应，但 Python 后处理的 `ReactionLocator` 返回 0 个匹配。

## 根因分析

存在三层问题，按严重程度排序：

### Bug 1：原子类型索引偏移（1-based vs gather_atoms 顺序）

`gather_atoms("type")` 返回的数组按 LAMMPS 内部内存顺序排列，不保证与 atom ID 对齐。代码中 `types_before[a]`（1-based 索引）和 `types_before[a-1]`（0-based 索引）都假设 `types[i]` 对应 atom ID `i+1`，这在 LAMMPS 内部顺序变化时失效。

**正确做法**：同时获取 `gather_atoms("id")`，构建 `{atom_id: type}` 映射字典，通过 ID 直接查找类型。

### Bug 2：聚类策略错误（18 changed atoms → 18 独立组件）

旧的 `_cluster_by_connectivity` 方法使用 before 图（反应前键连图）来聚类 changed atoms。反应前 initiator 原子分属不同分子，无法通过键连接，导致每个组件只含 1 个原子，BFS 匹配只能随机匹配附近原子，产生假阳性。

**修复**：使用 `_cluster_by_created_bonds`，从模板定义的新建键出发聚类，同一 created bond 的两个端点属于同一反应。

### Bug 3：缺失 post-after 匹配

旧代码只用 pre template 匹配 after 体系（时间线错位），没有用 post template 验证反应后的状态。修复为 pre-before + post-after 双重匹配。

## 修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `core/reaction_locator.py` | 重构 locate/match 方法，引入 dict 类型映射，新增 post 匹配 |
| `core/lammps_data_extractor.py` | 新增 gather_atoms 顺序验证 DEBUG |
| `run_refactored.py` | 传递 ids_before/ids_after，更新 _update_cg_mapping 签名 |

## 详细修改

### 1. `reaction_locator.py` — 核心修复

#### 1.1 `locate()` 方法签名变更

```python
# 修改前
def locate(self, bonds_before, bonds_after, types_before, types_after, n_atoms):

# 修改后
def locate(self, bonds_before, bonds_after, types_before, types_after,
           ids_before, ids_after, n_atoms):
```

新增 `ids_before` 和 `ids_after` 参数，用于构建 atom_id → type 映射：

```python
# 构建 atom_id → type 映射 (因为 gather_atoms 不保证按 ID 排序)
type_before_map = {int(id_): int(t) for id_, t in zip(ids_before, types_before)}
type_after_map = {int(id_): int(t) for id_, t in zip(ids_after, types_after)}
```

所有后续方法统一使用 `Dict[int, int]` 类型映射，不再依赖数组位置索引。

#### 1.2 `_cluster_by_created_bonds()` 替代 `_cluster_by_connectivity()`

```python
def _cluster_by_created_bonds(self, bond_changes, graph_before):
    # Step 1: 从 created_bonds 构建图，BFS 找连通分量
    created_graph = defaultdict(set)
    for bond in bond_changes.created_bonds:
        a1, a2 = int(bond[1]), int(bond[2])
        created_graph[a1].add(a2)
        created_graph[a2].add(a1)
    # BFS 找出连通分量...

    # Step 2: 将 orphaned atoms 合并到相邻 component
    all_changed = get_changed_atoms(bond_changes)
    orphaned = all_changed - atoms_in_created
    for orphan in orphaned:
        for nb in graph_before.get(orphan, []):
            if nb in some_component: merge...
```

#### 1.3 `_bfs_match()` 和 `_bfs_from_start()` 参数变更

```python
# 修改前
def _bfs_match(self, ..., atom_types: np.ndarray, ...):
    # atom_types[candidate], atom_types[sys_nb] 等数组索引

# 修改后
def _bfs_match(self, ..., atom_type_map: Dict[int, int], ...):
    # atom_type_map.get(candidate, -1), atom_type_map.get(sys_nb, -1)
```

所有 `atom_types[x]` / `atom_types[x-1]` 替换为 `atom_type_map.get(x, -1)`。

#### 1.4 新增 `_match_post_template()` 方法

结构与 `_match_pre_template` 对称，使用 post template 的键图和原子类型匹配 after 体系：

```python
def _match_post_template(self, template, graph_after, type_after_map, pre_match, n_atoms):
    system_atoms = pre_match.get_system_atoms()
    neighborhood = self._extract_neighborhood(system_atoms, graph_after, k_hop=max_k)
    return self._bfs_match(..., type_after_map, handle_isolated=False)
```

注意：post template 中所有原子通过新建键相连，不需要孤立原子匹配逻辑。

### 2. `lammps_data_extractor.py` — DEBUG 验证

在 `extract_atoms()` 中新增 DEBUG 输出，验证 `gather_atoms` 返回的数组顺序：

```python
print(f"[DEBUG] gather_atoms: first 5 entries:")
for i in range(min(5, len(ids))):
    print(f"  types[{i}]={types[i]}, ids[{i]}={ids[i]}")
print(f"  前10个atom ID是否按序排列: {is_sorted}")
print(f"  => atom IDs sorted, types[i] 对应 atom ID=i+1，可用 types[a-1] 索引")
```

当前数据集确认：atom IDs 按序排列，`types[i]` 对应 atom ID `i+1`。但 dict 映射方案不依赖此假设，更健壮。

### 3. `run_refactored.py` — 调用链修复

#### 3.1 `_update_cg_mapping()` 方法签名变更

```python
# 修改前
def _update_cg_mapping(self, bonds_before, bonds_after, types_before, types_after, n_atoms):

# 修改后
def _update_cg_mapping(self, bonds_before, bonds_after, types_before, types_after,
                       ids_before, ids_after, n_atoms):
```

DEBUG 代码改为使用 dict 映射打印正确的类型：

```python
type_before_map = {int(id_): int(t) for id_, t in zip(ids_before, types_before)}
type_after_map = {int(id_): int(t) for id_, t in zip(ids_after, types_after)}
print(f"    atom {a}: before_type={type_before_map.get(a, '?')}, after_type={type_after_map.get(a, '?')}")
```

#### 3.2 调用处传递 ids

```python
self._update_cg_mapping(
    cached['bonds'],
    bond_data_after.bonds,
    cached['types'],
    atom_data_after.types,
    cached['ids'],          # 新增
    atom_data_after.ids,    # 新增
    len(atom_data_after.ids)
)
```

## 验证结果

修复后，9 个反应全部正确匹配：

```
[DEBUG] locate: 9 created bonds, 9 reaction components
  component 0: atoms=[97, 25560], types={97: 2, 25560: 3}    # chain-end type=2, attacking type=3
  component 1: atoms=[1, 782],    types={1: 2, 782: 3}
  ...
locate 返回 9 个 match
```

每个反应都通过了 pre-before + post-after 双重验证：
- pre-match: BFS 映射 6/7（chain）+ 孤立原子填充 = 7/7
- post-match: BFS 直接映射 7/7（所有原子通过新键相连）

与 LAMMPS 报告的反应数完全一致。