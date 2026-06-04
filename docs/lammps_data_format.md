






以下是为您编写的 **LAMMPS Data 文件格式说明（Molecular 体系）** 的 Markdown 内容：

---

# LAMMPS Data 文件格式说明（Molecular 体系）

> **适用场景**：`atom_style full`（最常用的分子体系格式），含原子、键、角、二面角。

---

## 1. 文件总览

```
标题行（忽略）
（空行或注释）
数量头信息  （atoms, bonds, angles, dihedrals, atom types, bond types ...）
盒子信息    （xlo xhi, ylo yhi, zlo zhi）
（空行）
Masses      （质量段）
（空行）
Atoms       （原子段）
（空行）
Bonds       （键段）
（空行）
Angles      （角段）
（空行）
Dihedrals   （二面角段）
```

**规则**：

- **第 1 行（标题行）永远被跳过**，可放任意描述文字
- 头信息（数量声明 + 盒子）必须在最前面，其余段落（Masses, Atoms, Bonds, Angles, Dihedrals）顺序可任意
- **每个段落标题后必须紧跟一个空行**（否则报错）
- 段落之间用空行分隔
- `#` 开头的为注释，注释前须至少有一个空格
- 关键词（如 `atoms`、`Masses`、`Atoms`）必须**左对齐**且**大小写正确**

---

## 2. 头信息（数量声明）

紧接标题行之后，声明各类数量。头信息行可以任意顺序出现。

```
100 atoms
95 bonds
50 angles
30 dihedrals

5 atom types
10 bond types
18 angle types
20 dihedral types
```

| 关键词 | 含义 | 是否必需 |
|---|---|---|
| `atoms` | 原子总数 | ✅ 必需 |
| `bonds` | 键总数 | 键数 > 0 时必需 |
| `angles` | 角总数 | 角数 > 0 时必需 |
| `dihedrals` | 二面角总数 | 二面角数 > 0 时必需 |
| `atom types` | 原子类型数 | ✅ 必需 |
| `bond types` | 键类型数 | 键数 > 0 时必需 |
| `angle types` | 角类型数 | 角数 > 0 时必需 |
| `dihedral types` | 二面角类型数 | 二面角数 > 0 时必需 |

> 即使 bonds / angles / dihedrals 数量为 0，对应的 `types` 行也不需要出现。

---

## 3. 盒子信息（Box）

### 正交盒子（最常用）

```
-0.5 0.5 xlo xhi
-0.5 0.5 ylo yhi
-0.5 0.5 zlo zhi
```

- 格式：`<lo值> <hi值> <关键字>`
- 盒子三边向量：**A** = (xhi-xlo, 0, 0)，**B** = (0, yhi-ylo, 0)，**C** = (0, 0, zhi-zlo)
- 原点在左下角 (xlo, ylo, zlo)
- 2D 模拟时，`zlo zhi` 须跨越 0（默认 -0.5 和 0.5 即可），xz、yz 须为 0

### 非正交（三斜）盒子

如需倾斜，使用：

```
-0.5 0.5 xlo xhi
-0.5 0.5 ylo yhi
-0.5 0.5 zlo zhi
0.0 2.0 1.5 xy xz yz
```

| 关键词 | 含义 |
|---|---|
| `xy xz yz` | 倾斜因子（3 个实数） |

- **A** = (xhi-xlo, 0, 0)
- **B** = (xy, yhi-ylo, 0)
- **C** = (xz, yz, zhi-zlo)

> 一般只考虑正交盒子即可。非正交盒子需保证倾斜方向周期性。

---

## 4. Masses 段（原子质量）

```
Masses

  1   12.011
  2    1.008
  3   15.999
  4   14.007
```

| 格式 | 说明 |
|---|---|
| `类型ID 质量值` | 每行一个类型，共 N 行（N = atom types） |
| 质量单位 | 取决于 `units` 命令（如 `real` 为 g/mol，`metal` 为 g/mol，`si` 为 kg/mol 等） |

---

## 5. Atoms 段（原子信息）

`atom_style full` 下，每行格式：

```
atom-ID  molecule-ID  atom-type  q  x  y  z  nx  ny  nz
```

示例：

```
Atoms # full

      1      1       1       0.560   43.99993  58.52678  36.78550   0   0   0
      2      1       2      -0.100   44.10395  58.23499  35.86693   0   0   0
      3      1       3      -0.460   43.81519  59.54928  37.43995   0   0   0
      4      2       1       0.510   20.19985  37.57789  19.12163   0   0   0
      5      2       2      -0.250   20.88641  38.62251  19.70398   0   0   0
      6      2       3      -0.260   21.58030  37.49820  18.45012   0   0   0
```

| 列 | 字段 | 说明 |
|---|---|---|
| 1 | **atom-ID** | 原子编号（1 到 N） |
| 2 | **molecule-ID** | 所属分子的编号（同一分子的所有原子用相同编号） |
| 3 | **atom-type** | 原子类型（正整数，对应 Masses 段和力场参数） |
| 4 | **q** | 电荷（电子单位，如 +1 = 质子电荷） |
| 5 | **x** | x 坐标（单位：由 `units` 决定，通常 Å） |
| 6 | **y** | y 坐标 |
| 7 | **z** | z 坐标（2D 模拟用 0.0） |
| 8 | **nx** | x 方向映像标志（可选，默认 0） |
| 9 | **ny** | y 方向映像标志（可选，默认 0） |
| 10 | **nz** | z 方向映像标志（可选，默认 0） |

> **nx, ny, nz** 为映像标志（image flags），表示原子穿越周期边界的次数。只在周期性模拟中有意义，大多数情况可设为 0。2D 模拟时 nz 须为 0。

---

## 6. Bonds 段（键连接）

每行格式：

```
bond-ID  bond-type  atom-1  atom-2
```

示例：

```
Bonds

      1      1      1      2
      2      1      1      3
      3      1      1      4
      4      1      1      5
      5      2      2      6
      6      2      2      7
```

| 列 | 字段 | 说明 |
|---|---|---|
| 1 | **bond-ID** | 键编号（1 到 M，M = bonds 总数） |
| 2 | **bond-type** | 键类型（对应力场参数中的键类型） |
| 3 | **atom-1** | 参与键的第一个原子的 atom-ID |
| 4 | **atom-2** | 参与键的第二个原子的 atom-ID |

> 共 N 行（N = bonds 头信息中声明的数量）。

---

## 7. Angles 段（角连接）

每行格式：

```
angle-ID  angle-type  atom-1  atom-2  atom-3
```

示例：

```
Angles

      1      1      2      1      3
      2      2      2      1      4
      3      3      2      1      5
      4      4      3      1      4
```

| 列 | 字段 | 说明 |
|---|---|---|
| 1 | **angle-ID** | 角编号（1 到 M） |
| 2 | **angle-type** | 角类型（对应力场参数中的角类型） |
| 3 | **atom-1** | 端原子 1 的 atom-ID |
| 4 | **atom-2** | **中心原子**的 atom-ID |
| 5 | **atom-3** | 端原子 2 的 atom-ID |

> `atom-2` 是角的顶点（中心原子）。

---

## 8. Dihedrals 段（二面角连接）

每行格式：

```
dihedral-ID  dihedral-type  atom-1  atom-2  atom-3  atom-4
```

示例：

```
Dihedrals

      1      1      5      1      2      6
      2      1      4      1      2      7
      3      2      3      1      2      8
      4      3      2      1      3      9
```

| 列 | 字段 | 说明 |
|---|---|---|
| 1 | **dihedral-ID** | 二面角编号（1 到 M） |
| 2 | **dihedral-type** | 二面角类型（对应力场参数中的二面角类型） |
| 3 | **atom-1** | 端原子 1 的 atom-ID |
| 4 | **atom-2** | 中心键原子 1 的 atom-ID |
| 5 | **atom-3** | 中心键原子 2 的 atom-ID |
| 6 | **atom-4** | 端原子 2 的 atom-ID |

> `atom-2` 和 `atom-3` 构成二面角的**中心键**，两端原子分别绕该键扭转。

---

## 9. 完整示例文件

```
LAMMPS data file for a small molecule system

      10  atoms
       8  bonds
      12  angles
       6  dihedrals

       4  atom types
       3  bond types
       4  angle types
       3  dihedral types

 -10.0  10.0  xlo xhi
 -10.0  10.0  ylo yhi
 -10.0  10.0  zlo zhi

Masses

  1   12.011
  2    1.008
  3   15.999
  4   14.007

Atoms # full

      1      1       1       0.560    1.000   0.000   0.000   0   0   0
      2      1       2      -0.100    1.500   1.000   0.000   0   0   0
      3      1       3      -0.460    0.000   0.000   0.000   0   0   0
      4      1       4      -0.250    0.800   1.500   0.000   0   0   0
      5      1       2      -0.100    2.000   1.500   0.000   0   0   0
      6      2       1       0.510    3.000   0.000   0.000   0   0   0
      7      2       2      -0.250    3.500   1.000   0.000   0   0   0
      8      2       3      -0.260    4.000   0.000   0.000   0   0   0
      9      2       4      -0.250    4.500   0.500   0.000   0   0   0
     10      2       2      -0.100    5.000   1.500   0.000   0   0   0

Bonds

      1      1      1      2
      2      2      1      3
      3      1      1      4
      4      2      2      5
      5      1      6      7
      6      2      6      8
      7      1      6      9
      8      2      7     10

Angles

      1      1      2      1      3
      2      2      2      1      4
      3      1      3      1      4
      4      3      1      2      5
      5      1      7      6      8
      6      2      7      6      9
      7      1      8      6      9
      8      4      6      7     10
      9      1      1      4      9
     10      2      3      8      9
     11      1      4      9     10
     12      2      8      7     10

Dihedrals

      1      1      3      1      2      5
      2      1      4      1      2      5
      3      2      8      6      7     10
      4      2      9      6      7     10
      5      1      2      1      4      9
      6      3      5      2      7     10
```

---

## 10. 注意事项

1. **atom_style 必须在 read_data 之前设定**：在 LAMMPS 输入脚本中，`atom_style full` 必须在 `read_data` 命令之前声明。

2. **列顺序不可颠倒**：Atoms 段格式严格遵循 `atom_style` 定义。`full` 和 `molecular` 的列顺序不同：
   - `atom_style full`：atom-ID, molecule-ID, atom-type, q, x, y, z, nx, ny, nz
   - `atom_style molecular`：atom-ID, molecule-ID, atom-type, x, y, z（无电荷字段）

3. **尾随的 nx, ny, nz 可省略**，省略时默认全部为 0。

4. **编号连续性**：atom-ID、bond-ID、angle-ID、dihedral-ID 须从 1 开始连续编号。

5. **分子编号**：molecule-ID 相同的原子属于同一分子。若没有分子概念，可全设为 1 或按原子编号递增。

6. **空格/缩进不重要**：数值之间只需空格分隔即可，无需严格对齐（但建议对齐便于阅读）。

7. **头信息中的类型数要与段落实际数据匹配**：如声明 `4 atom types` 则 Masses 段必须有 4 行；声明 `8 bonds` 则 Bonds 段必须有 8 行。

---

> **参考来源**：LAMMPS 官方文档 — [read_data command](https://docs.lammps.org/read_data.html) / [Data Format](https://docs.lammps.org/2001/data_format.html) / [File Formats](https://docs.lammps.org/Run_formats.html)