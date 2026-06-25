# LmpPy 子模块 API 参考文档

> 本文档覆盖 LmpPy 所有公开子模块的数据类、核心类、函数及其参数、返回值和代码示例。
> 按功能域组织：配置系统、数据提取、反应检测、CG 处理、记录与验证、工具模块、工具函数、后处理分析、入口与集成、CLI 脚本。

---

## 目录

1. [核心模块 (core/)](#1-核心模块-core)
   - 1.1 配置系统
   - 1.2 数据提取
   - 1.3 反应检测
   - 1.4 CG 处理
   - 1.5 记录与验证
2. [工具模块 (tools/)](#2-工具模块-tools)
3. [工具函数 (utils/)](#3-工具函数-utils)
4. [后处理分析 (output_analysis/)](#4-后处理分析-output_analysis)
5. [入口与集成](#5-入口与集成)
6. [CLI 脚本 (scripts/)](#6-cli-脚本-scripts)

---

## 1. 核心模块 (core/)

### 1.1 配置系统

#### config_loader.py — 配置加载与验证

配置加载器负责加载所有 YAML 配置文件，进行严格验证，返回类型化配置对象。

**异常类:**

| 类名 | 说明 |
|---|---|
| `ConfigError` | 配置错误基类 |
| `ConfigFileNotFoundError` | 配置文件未找到 |
| `ConfigValidationError` | 配置验证错误 |
| `ConfigMissingFieldError` | 缺少必需字段 |
| `ConfigInvalidValueError` | 字段值无效 |

**数据类:**

```python
@dataclass
class MappingConfig:
    site_types: Dict[str, Dict[str, List]]  # CG 位点类型定义
    config: List[Dict[str, Any]]            # 映射配置列表
    bead_type_names: Optional[Dict[str, int]] = None  # bead 类型名称→ID 映射

@dataclass
class ReactionInfo:
    name: str                    # 反应名称
    cutoff: float                # 反应截断半径 (A)
    pre_mol: str                 # 反应前分子模板名
    post_mol: str                # 反应后分子模板名
    map_file: str = ""           # bond/react map 文件完整路径
    pre_template: str = ""       # 反应前模板文件完整路径
    post_template: str = ""      # 反应后模板文件完整路径
    pre_mapping: str = ""        # 反应前 CG 映射文件完整路径（可选）
    post_mapping: str = ""       # 反应后 CG 映射文件完整路径（可选）
    rxn_dir: str = ""            # reactions/{name}/ 目录完整路径

@dataclass
class ReadDataExtra:
    special_per_atom: int = 3
    bond_per_atom: int = 3
    angle_per_atom: int = 3
    dihedral_per_atom: int = 3

@dataclass
class LAMMPSParams:
    loop_num: int                          # 循环次数
    dt: float                              # 时间步长 (ps)
    timestep: int                          # LAMMPS timestep (fs)
    temperature: float                     # 温度 (K)
    pressure: float                        # 压力 (atm)
    bond_react_check_step: int             # 反应检测间隔步数
    run_step: int                          # 每循环运行步数
    nve_limit_step: int                    # NVE 限制步数
    tcouple: float                         # 温度耦合常数 (fs)
    pcouple: float                         # 压力耦合常数 (fs)
    stabilization: float                   # 稳定化参数 (A)
    ensemble: str = "npt"                  # 系综类型
    reactions: List[ReactionInfo] = ...    # 反应配置列表
    molecules: Dict[str, str] = ...        # 分子模板路径映射
    mass_list: Dict[int, float] = ...      # 原子类型→质量
    # ... 输出文件路径等字段

@dataclass
class SystemConfig:
    name: str
    mapping_files: List[Dict[str, Any]]
    mass_list: Dict[int, float]
    bead_type_names: Dict[str, int]        # 全局 bead 类型名称→ID
    work_dir: str = "work"
    output_dir: str = "output"

@dataclass
class ReactionConfig:
    name: str
    description: str
    pre_template: str
    post_template: str
    pre_mapping: MappingConfig
    post_mapping: MappingConfig
    equivalences: np.ndarray
    edge_atoms: List[int]
    initiator_atoms: List[int]
    cutoff: float

@dataclass
class ValidationIssue:
    level: str          # "error" | "warning" | "info"
    category: str       # "file" | "config" | "semantic"
    field: str          # 字段名或路径
    message: str        # 详细描述
    suggestion: str = ""  # 建议的修复方案
    path: Optional[Path] = None

@dataclass
class ValidationResult:
    issues: List[ValidationIssue]
    config_dir: Optional[Path] = None
    # 属性: passed, n_errors, n_warnings, n_infos
    # 方法: add_error(), add_warning(), add_info(), print_report(), to_report()
```

**核心类:**

```python
class ConfigLoader:
    """配置加载器：加载所有 YAML 配置文件并返回类型化对象。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(config_dir)` | `config_dir: str` | — | 初始化，验证配置目录存在 |
| `load_system_config()` | — | `SystemConfig` | 加载 `system.yaml` |
| `load_lammps_params()` | — | `LAMMPSParams` | 加载 `lammps_params.yaml` |
| `load_mass_list(data_file=None)` | `data_file: Optional[str]` | `Dict[int, float]` | 加载质量列表，优先从 `mass_list.yaml` |
| `load_mapping_config(mapping_path)` | `mapping_path: str` | `MappingConfig` | 加载 CG 映射配置 |
| `load_reaction_config(reaction_name)` | `reaction_name: str` | `ReactionConfig` | 加载单个反应配置 |
| `load_all_reactions()` | — | `List[ReactionConfig]` | 加载 `reactions/` 下所有反应 |

```python
class ConfigValidator:
    """完整配置验证器：文件一致性、配置间一致性、语义验证。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(config_dir)` | `config_dir: str` | — | 初始化验证器 |
| `validate_all()` | — | `ValidationResult` | 执行完整验证流程 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `load_all_configs(config_dir)` | `config_dir: str` | `(SystemConfig, LAMMPSParams, List[ReactionConfig])` | 一次性加载所有配置 |
| `validate_config(config_dir, print_report=True)` | `config_dir: str, print_report: bool` | `ValidationResult` | 验证配置目录完整性 |
| `load_reactions_from_directory(config_dir, reaction_configs, validate_files=True)` | `config_dir, reaction_configs: List[Dict], validate_files: bool` | `List[ReactionInfo]` | 从 `reactions/` 目录统一加载反应配置 |

**使用示例:**

```python
from LmpPy.core import ConfigLoader, ConfigValidator, load_all_configs, validate_config

# 加载配置
loader = ConfigLoader("config/")
system_config = loader.load_system_config()
params = loader.load_lammps_params()
mass_list = loader.load_mass_list("system.data")

# 完整加载
sys_cfg, lmp_params, reactions = load_all_configs("config/")

# 验证
result = validate_config("config/")
result.print_report()
```

---

#### mapping_generator.py — CG 映射生成

从 mapping YAML 文件生成初始 CG compare list。

**数据类:**

```python
@dataclass
class CGCompareList:
    data: np.ndarray  # (n_rows, 5) [bead_id, mol_id, bead_type, AA_id, mass]
    # 属性: n_beads, n_molecules
    # 方法: to_csv(path), from_csv(path), to_numpy()
```

**核心类:**

```python
class MappingGenerator:
    """CG 映射生成器：解析 YAML 配置，生成 AA 到 CG 映射关系。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(bead_type_map=None)` | `bead_type_map: Optional[Dict[str, int]]` | — | 可选自定义 bead 类型映射 |
| `generate(mapping_config, global_bead_type_map, aa_id_offset=0, mol_id_offset=0, bead_id_offset=0)` | `mapping_config: MappingConfig, global_bead_type_map: Dict[str, int], ...` | `(CGCompareList, next_bead_id, next_aa_id, next_mol_id)` | 从配置生成 CG 映射 |
| `generate_from_system(system_config, config_dir)` | `system_config: SystemConfig, config_dir: Path` | `CGCompareList` | 从 SystemConfig 生成完整 CG 映射 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `generate_cg_compare_list(system_config, config_dir, output_path=None)` | `system_config, config_dir: Path, output_path: Optional[str]` | `CGCompareList` | 生成 CG 映射并可选保存 |

**使用示例:**

```python
from LmpPy.core import MappingGenerator, generate_cg_compare_list

# 生成 CG 映射
cg_list = generate_cg_compare_list(
    system_config, config_dir, output_path="compare_list.csv"
)
print(f"Bead 数: {cg_list.n_beads}, 分子数: {cg_list.n_molecules}")
```

---

#### template_parser.py — 反应模板解析

解析 LAMMPS 反应模板文件 (`.lammpstemplate`) 和反应映射文件 (`.map`)。

**数据类:**

```python
@dataclass
class TemplateData:
    n_atoms: int
    n_bonds: int
    n_angles: int
    n_dihedrals: int
    atom_types: np.ndarray    # (n_atoms+1,)
    coords: np.ndarray        # (n_atoms+1, 3)
    bonds: np.ndarray         # (n_bonds, 4) [bond_id, bond_type, atom1, atom2]
    angles: np.ndarray        # (n_angles, 5) [angle_id, angle_type, atom1, atom2, atom3]
    dihedrals: np.ndarray     # (n_dihedrals, 6) [dihedral_id, dihedral_type, atom1, atom2, atom3, atom4]

@dataclass
class ReactionMapData:
    n_edge_ids: int
    n_equivalences: int
    n_constraints: int
    initiator_ids: List[int]
    edge_ids: List[int]
    equivalences: Dict[int, int]    # {pre_atom_id: post_atom_id}
    constraints: List[Tuple[str, int, int, float, float]]  # [(type, atom1, atom2, min, max)]

@dataclass
class TemplateCGMapping:
    bead_groups: Dict[int, Tuple[List[int], int]]  # {local_bead_id: (atom_ids, bead_type)}
    n_beads: int
    # 方法: get_atom_bead_info(atom_id), get_atoms_in_bead(local_bead_id)

@dataclass
class ReactionTemplate:
    name: str
    pre_template: TemplateData
    post_template: TemplateData
    reaction_map: ReactionMapData
    pre_bond_codes: Set[int] = ...
    post_bond_codes: Set[int] = ...
    pre_cg_mapping: Optional[TemplateCGMapping] = None
    post_cg_mapping: Optional[TemplateCGMapping] = None
    # 属性: created_bonds, deleted_bonds, changed_atom_types
```

**核心类:**

```python
class TemplateParser:
    """LAMMPS 模板解析器"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `parse_template(template_file)` | `template_file: str` | `TemplateData` | 解析 `.lammpstemplate` 文件 |
| `parse_map(map_file)` | `map_file: str` | `ReactionMapData` | 解析 `.map` 文件 |
| `parse_cg_mapping(cg_mapping_file, n_template_atoms)` | `cg_mapping_file: str, n_template_atoms: int` | `TemplateCGMapping` | 解析模板 CG 映射 YAML |
| `load_reaction_template(name, pre_template_file, post_template_file, map_file, pre_cg_mapping_file=None, post_cg_mapping_file=None)` | `name, ...` | `ReactionTemplate` | 加载完整反应模板 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `load_all_reaction_templates(reaction_dir, load_cg_mapping=True)` | `reaction_dir: Path, load_cg_mapping: bool` | `Dict[str, ReactionTemplate]` | 加载目录下所有反应模板 |

**使用示例:**

```python
from LmpPy.core import TemplateParser, load_all_reaction_templates

parser = TemplateParser()
template = parser.load_reaction_template(
    "rxn1",
    "reactions/rxn1/rxn1_pre.lammpstemplate",
    "reactions/rxn1/rxn1_post.lammpstemplate",
    "reactions/rxn1/rxn1.map"
)
print(f"创建键: {template.created_bonds}")
print(f"删除键: {template.deleted_bonds}")
print(f"类型变化: {template.changed_atom_types}")

# 批量加载
templates = load_all_reaction_templates(Path("config/reactions/"))
```

---

### 1.2 数据提取

#### lammps_data_extractor.py — LAMMPS 数据提取

从 LAMMPS 实例提取原子和键信息，向量化 image flag 解码，条件性缓存优化。

**数据类:**

```python
@dataclass
class AtomData:
    ids: np.ndarray           # (n_atoms,)
    types: np.ndarray         # (n_atoms,)
    coords: np.ndarray        # (n_atoms, 3)
    image_flags: np.ndarray   # (n_atoms, 3)
    # 属性: n_atoms

@dataclass
class BondData:
    bonds: np.ndarray         # (n_bonds, 3) [bond_type, atom1, atom2]
    # 属性: n_bonds
```

**核心类:**

```python
class LAMMPSDataExtractor:
    """LAMMPS 数据提取器，支持缓存优化。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(use_cache=True)` | `use_cache: bool` | — | 初始化，可选启用缓存 |
| `extract_atoms(lmp, use_cache=True)` | `lmp: lammps, use_cache: bool` | `AtomData` | 提取原子数据（坐标、类型、image flags） |
| `extract_bonds(lmp, use_cache=True)` | `lmp: lammps, use_cache: bool` | `BondData` | 提取键数据 |
| `extract_all(lmp, use_cache=True)` | `lmp: lammps, use_cache: bool` | `(AtomData, BondData)` | 提取所有数据并标记缓存有效 |
| `invalidate_cache()` | — | — | 使缓存失效（反应发生后调用） |
| `extract_box_info(lmp)` | `lmp: lammps` | `(box_bounds, box_origin)` | 提取盒子信息 |

**独立函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `decode_image_flags_vectorized(encode_ixyz)` | `encode_ixyz: np.ndarray` | `np.ndarray (n, 3)` | 向量化解码 LAMMPS image flags（362x 加速） |
| `get_atoms_bonds_info(lmp, extractor=None, use_cache=True)` | `lmp, extractor: Optional[LAMMPSDataExtractor], use_cache: bool` | `(coords, ids, bonds, types, image_flags)` | 兼容原接口的便捷函数 |
| `get_lmp_box_info(lmp)` | `lmp: lammps` | `np.ndarray (3, 2)` | 获取盒子 [min, max] 定义 |
| `parse_masses_from_data_file(data_file_path)` | `data_file_path: str | Path` | `Dict[int, float]` | 从 LAMMPS data 文件解析 Masses |
| `parse_bonds_from_data_file(data_file_path)` | `data_file_path: str | Path` | `np.ndarray (n_bonds, 3)` | 从 LAMMPS data 文件解析 Bonds |

**使用示例:**

```python
from LmpPy.core import LAMMPSDataExtractor, decode_image_flags_vectorized

extractor = LAMMPSDataExtractor(use_cache=True)
atom_data, bond_data = extractor.extract_all(lmp)  # lmp: LAMMPS 实例

# 无反应：缓存 ids/types/bonds
# 反应后：使缓存失效
extractor.invalidate_cache()

# 向量化 image flag 解码
flags = decode_image_flags_vectorized(encoded_flags)
```

---

### 1.3 反应检测

#### bond_detector.py — 键变化检测

检测反应前后的键变化，向量化键编码实现高效比较。

**数据类:**

```python
@dataclass
class BondChanges:
    created_bonds: np.ndarray   # (n_created, 3) [bond_type, atom1, atom2]
    deleted_bonds: np.ndarray   # (n_deleted, 3)
    created_codes: Set[int]     # 创建键的编码集合
    deleted_codes: Set[int]     # 删除键的编码集合
    # 属性: n_created, n_deleted, has_changes
```

**核心类:**

```python
class BondDetector:
    """键变化检测器。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(n_atoms)` | `n_atoms: int` | — | 初始化，指定原子总数 |
| `encode_bonds(bonds)` | `bonds: np.ndarray (n_bonds, 3) 或 (n_bonds, 2)` | `Set[int]` | 将键编码为整数集合 |
| `encode_bond(atom1, atom2)` | `atom1: int, atom2: int` | `int` | 编码单个键 |
| `decode_code(code)` | `code: int` | `(atom1, atom2)` | 解码单个编码 |
| `detect(bonds_before, bonds_after)` | `bonds_before, bonds_after: np.ndarray` | `BondChanges` | 检测键变化 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `compare_two_bonds(bonds1, bonds2)` | `bonds1, bonds2: np.ndarray (n_bonds, 3)` | `(bonds1_only, bonds2_only)` | 比较两组键差异 |
| `get_changed_atoms(bond_changes)` | `bond_changes: BondChanges` | `Set[int]` | 提取参与反应的原子 ID |

**使用示例:**

```python
from LmpPy.core import BondDetector, compare_two_bonds

detector = BondDetector(n_atoms=10000)
changes = detector.detect(bonds_before, bonds_after)
if changes.has_changes:
    print(f"创建 {changes.n_created} 键, 删除 {changes.n_deleted} 键")
```

---

#### reaction_locator.py — 反应位点定位

从键变化定位反应类型和位点，使用 pre-before + post-after 双重匹配策略。

**数据类:**

```python
@dataclass
class ReactionMatch:
    reaction_name: str                              # 反应名称
    template_to_system: Dict[int, int]              # 模板原子 ID → 体系原子 ID
    confidence: float = 1.0                         # 匹配置信度
    matched_bonds: List[Tuple[int, int]] = ...      # 匹配的键
    # 方法: get_system_atoms() -> Set[int]
```

**核心类:**

```python
class ReactionLocator:
    """反应位点定位器：双重匹配策略。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(reaction_templates)` | `reaction_templates: Dict[str, ReactionTemplate]` | — | 初始化 |
| `locate(bonds_before, bonds_after, types_before, types_after, ids_before, ids_after, n_atoms)` | `所有 np.ndarray, n_atoms: int` | `List[ReactionMatch]` | 定位反应位点 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `locate_reactions(bonds_before, bonds_after, types_before, types_after, ids_before, ids_after, n_atoms, reaction_templates)` | 同上 | `List[ReactionMatch]` | 便捷函数 |

---

#### cg_reaction_identifier.py — CG 级反应识别（v2.6+）

从 CG 键变化独立识别反应类型，无 LAMMPS 依赖，作为主流程的交叉验证机制。

**数据类:**

```python
@dataclass
class CGReactionSignature:
    name: str                                    # "rxn1_EEE"
    signature_3bead: Tuple[int, int, int]        # (interior_type, end_type, monomer_type)
    pre_chains: Dict[int, Tuple[int, ...]]       # initiator_bead_type → 向外的类型链
    type_map: Dict[int, int]                     # {old_bead_type: new_bead_type}
    bead_id_map: Dict[int, int] = ...            # {old_local_bead_id: new_local_bead_id}
```

**核心函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `load_template_signatures(reactions_dir)` | `reactions_dir: Path` | `(signature_index, all_signatures, end_types, monomer_types, interior_types)` | 从 reactions/ 目录加载 CG 级反应签名（生产级） |
| `load_reaction_signatures(reactions_dir)` | `reactions_dir: Path` | `Dict[Tuple, Dict]` | 从 reactions/ 目录加载 3-bead 签名（交叉验证用） |
| `build_cg_bond_graph(cg_bonds)` | `cg_bonds: np.ndarray` | `Dict[int, List[int]]` | 从 CG 键构建邻接图 |
| `get_cg_bond_diff(cg_bonds_before, cg_bonds_after)` | 两个 `np.ndarray` | `List[Tuple[int, int]]` | CG 键差集 |
| `identify_reaction(cg_bonds_before, cg_bonds_after, cg_mapping, signatures)` | `cg_bonds_before, cg_bonds_after, cg_mapping: np.ndarray, signatures: Dict` | `List[Dict]` | 从 CG 级别信息识别反应类型 |
| `match_reaction(new_bond, bead_type_lut, cg_graph_after, signature_index, end_types, monomer_types, interior_types)` | `new_bond: Tuple, ...` | `Optional[CGReactionSignature]` | 两级匹配：3-bead 粗筛 + chain 精筛 |

---

#### reaction_commands.py — 反应命令生成（v2.6+）

bond/create 和 bond/react 的 LAMMPS fix 命令生成，CG 映射更新。

**数据类:**

```python
@dataclass
class BondCreatePair:
    itype: int               # i 原子类型
    jtype: int               # j 原子类型
    Nevery: int              # 检测频率
    Rmin: float              # 最小成键距离
    bondtype: int            # 创建的键类型
    iparam_maxbond: int = 0
    iparam_newtype: Optional[int] = None
    jparam_maxbond: int = 0
    jparam_newtype: Optional[int] = None
    prob_fraction: float = 1.0
    prob_seed: Optional[int] = None
    # 方法: to_lammps_command(fix_id) -> str

@dataclass
class BondCreateConfig:
    pairs: List[BondCreatePair] = ...
    cg_type_map: Dict[int, int] = ...
    sequential: bool = True
    relax_radius: float = 0.0
```

**核心函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `load_bond_create_config(data)` | `data: dict` | `BondCreateConfig` | 从 YAML dict 加载并验证 bond/create 配置 |
| `get_reaction_mode(bond_create_enabled)` | `bond_create_enabled: bool` | `str` | 判断反应模式："bond/create" 或 "bond/react" |
| `generate_fix_bond_create(config)` | `config: BondCreateConfig` | `List[Tuple[str, str]]` | 生成 LAMMPS fix bond/create 命令列表 |
| `generate_fix_bond_react(reactions, stabilization)` | `reactions: List[ReactionInfo], stabilization: float` | `str` | 生成 LAMMPS fix bond/react 命令字符串 |
| `update_cg_mapping_create(cg_mapping_data, bonds_before, bonds_after, type_map)` | `cg_mapping_data, bonds_before, bonds_after: np.ndarray, type_map: Dict[int, int]` | `bool` | bond/create 模式下更新 CG 映射 |

---

### 1.4 CG 处理

#### cg_mapper.py — CG 映射更新

反应后基于模板 CG 映射更新 CG 映射，按 bead 整体更新保证一致性。

**数据类:**

```python
@dataclass
class CGMapping:
    data: np.ndarray          # (n_atoms, 2) [bead_id, bead_type]
    n_atoms: int
    # 类方法: from_cg_compare_list(cg_compare_list, n_atoms) -> CGMapping
    # 方法: get_bead_id(atom_id), get_bead_type(atom_id), set_bead(atom_id, bead_id, bead_type)
    #       get_atoms_in_bead(bead_id), get_max_bead_id(), to_cg_compare_list(masses=None)
```

**核心类:**

```python
class CGMapper:
    """CG 映射更新器：基于反应模板匹配结果更新 CG 映射。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `update(cg_mapping, reaction_match, template)` | `cg_mapping: CGMapping, reaction_match: ReactionMatch, template: ReactionTemplate` | `CGMapping` | 更新单个反应的 CG 映射 |
| `batch_update(cg_mapping, reaction_matches, templates)` | `cg_mapping: CGMapping, reaction_matches: List[ReactionMatch], templates: Dict[str, ReactionTemplate]` | `CGMapping` | 批量更新多个反应的 CG 映射 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `update_cg_mapping(cg_compare_list, reaction_matches, templates, n_atoms)` | `cg_compare_list: np.ndarray, ...` | `np.ndarray` | 更新 CG 映射并返回原格式 |

---

#### cg_converter.py — CG 坐标转换

全原子坐标到粗粒化坐标的转换，惰性索引缓存优化。

**数据类:**

```python
@dataclass
class CGConverter:
    # 惰性索引缓存
    bead_to_atoms: Dict[int, np.ndarray] = ...
    bead_to_type: Dict[int, int] = ...
    cache_valid: bool = False
    # 扁平化索引（预计算，全向量化转换）
    _flat_atom_indices: Optional[np.ndarray] = None
    _flat_bead_indices: Optional[np.ndarray] = None
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__()` | — | — | 初始化转换器 |
| `convert(atom_coords, cg_compare_list, mass_list, validate=True)` | `atom_coords: np.ndarray, cg_compare_list: np.ndarray, mass_list: Dict[int, float], validate: bool` | `np.ndarray (n_beads, 5)` | 全原子→CG 坐标转换 |
| `invalidate_cache()` | — | — | 使缓存失效 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `lammpstrj2cg(atom_coords, cg_compare_list, mass_list, converter=None)` | 同上（converter 可选） | `np.ndarray (n_beads, 5)` | 兼容原接口 |
| `validate_cg_mapping_consistency(cg_compare_list, raise_error=True)` | `cg_compare_list: np.ndarray, raise_error: bool` | `Dict[int, int]` | 验证 CG 映射一致性（同一 bead 的 type 一致） |

**使用示例:**

```python
from LmpPy.core import CGConverter, lammpstrj2cg

converter = CGConverter()
cg_coords = converter.convert(atom_coords, cg_compare_list, mass_list)
# 第二次调用利用缓存加速
cg_coords2 = converter.convert(atom_coords, cg_compare_list, mass_list)
converter.invalidate_cache()  # CG 映射变化时
```

---

#### cg_bond_mapper.py — 粗粒键映射

原子键到粗粒键的转换，忽略同一 bead 内的键。

**数据类:**

```python
@dataclass
class CGBond:
    bead1: int
    bead2: int
    bond_type: int
```

**核心类:**

```python
class CGBondMapper:
    """粗粒键映射器"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(cg_mapping)` | `cg_mapping: CGMapping` | — | 初始化 |
| `atom_bonds_to_cg_bonds(atom_bonds, default_bond_type=1)` | `atom_bonds: np.ndarray (n_bonds, 3), default_bond_type: int` | `np.ndarray (n_cg_bonds, 3)` | 原子键→粗粒键 |
| `update_cg_bonds(existing_cg_bonds, new_atom_bonds, deleted_atom_bonds, default_bond_type=1)` | `existing, new, deleted: np.ndarray, default_bond_type: int` | `np.ndarray (n_cg_bonds, 3)` | 更新粗粒键 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `atom_bonds_to_cg_bonds(atom_bonds, cg_mapping_data, default_bond_type=1)` | `atom_bonds: np.ndarray, cg_mapping_data: np.ndarray (n_atoms, 2), default_bond_type: int` | `np.ndarray (n_cg_bonds, 3)` | 原子键→粗粒键（独立函数） |

---

#### cg_topology.py — CG 拓扑推导

从 CG 键推导角度和二面角拓扑，根据 bead type 组合分配拓扑类型 ID。

**数据类:**

```python
@dataclass
class CGTopology:
    bonds: np.ndarray       # (n_bonds, 3) [bond_type, bead1, bead2]
    angles: np.ndarray      # (n_angles, 4) [angle_type, bead1, bead2, bead3]
    dihedrals: np.ndarray   # (n_dihedrals, 5) [dihedral_type, bead1, bead2, bead3, bead4]
    # 属性: n_bonds, n_angles, n_dihedrals
    # 方法: to_files(prefix) -> 保存到 {prefix}_bonds.txt 等
```

**核心函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `derive_angles_from_bonds(cg_bonds, bead_types=None, default_angle_type=1, type_mapping=None)` | `cg_bonds: np.ndarray, bead_types: Optional[Dict[int, int]], ...` | `np.ndarray (n_angles, 4)` | 从键列表推导角度 |
| `derive_dihedrals_from_bonds(cg_bonds, bead_types=None, default_dihedral_type=1, type_mapping=None)` | 同类型参数 | `np.ndarray (n_dihedrals, 5)` | 从键列表推导二面角 |
| `derive_cg_topology_from_bonds(cg_bonds, bead_types=None, default_angle_type=1, default_dihedral_type=1, angle_type_mapping=None, dihedral_type_mapping=None)` | 同上 | `CGTopology` | 从 CG 键推导完整 CG 拓扑 |
| `create_cg_bead_info(cg_compare_list)` | `cg_compare_list: np.ndarray (n_rows, 5)` | `np.ndarray (n_beads, 4)` | 创建 CG 珠子信息 [bead_id, mol_id, bead_type, total_mass] |
| `verify_molecule_ids_consistency(cg_compare_list, cg_bonds)` | `cg_compare_list, cg_bonds: np.ndarray` | `(bool, List)` | 验证分子 ID 与键连关系一致 |

---

#### cg_initializer.py — CG 初始化器

整合映射生成、AA bonds 提取、CG 拓扑推导的完整初始化流程。

**数据类:**

```python
@dataclass
class CGSystem:
    cg_compare_list: CGCompareList
    cg_topology: CGTopology
    n_atoms: int
    n_beads: int
    n_molecules: int
    # 属性: n_bonds, n_angles, n_dihedrals
```

**核心类:**

```python
class CGInitializer:
    """CG 初始化器：生成初始 CG 映射和拓扑。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(config_dir)` | `config_dir: str` | — | 初始化 |
| `generate_initial_cg_mapping(output_path=None)` | `output_path: Optional[str]` | `CGCompareList` | 生成初始 CG 映射 |
| `extract_aa_bonds(data_file=None)` | `data_file: Optional[str]` | `np.ndarray (n_bonds, 3)` | 从 data 文件提取 AA bonds |
| `generate_cg_topology(cg_compare_list, aa_bonds, default_bond_type=1, default_angle_type=1, default_dihedral_type=1)` | `cg_compare_list: CGCompareList, aa_bonds: np.ndarray, ...` | `CGTopology` | 生成 CG 拓扑 |
| `verify_consistency(cg_compare_list, cg_topology)` | `cg_compare_list: CGCompareList, cg_topology: CGTopology` | `(bool, list)` | 验证一致性 |
| `initialize(output_mapping=None, output_topology_prefix=None, verify=True)` | `...` | `CGSystem` | 完整 CG 初始化流程 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `initialize_cg_system(config_dir, output_mapping=None, output_topology_prefix=None, verify=True)` | 同上 | `CGSystem` | 便捷函数 |

**使用示例:**

```python
from LmpPy.core import initialize_cg_system

cg_system = initialize_cg_system(
    "config/",
    output_mapping="initial_cg.csv",
    output_topology_prefix="cg"
)
print(f"Beads: {cg_system.n_beads}, Bonds: {cg_system.n_bonds}")
```

---

### 1.5 记录与验证

#### bonds_recorder.py — 键连表记录

记录反应前后的键连表和 CG 拓扑变化，保存为 `.npz` 文件。

**数据类:**

```python
@dataclass
class BondRecord:
    timestep: int
    run_step: int
    bonds_before: np.ndarray   # (n_bonds, 3)
    bonds_after: np.ndarray    # (n_bonds, 3)
    reaction_type: str
    # 方法: save(output_dir, index=0) -> str

@dataclass
class CGTopologyRecord:
    timestep: int
    run_step: int
    cg_bonds_before: np.ndarray      # (n_bonds, 3)
    cg_bonds_after: np.ndarray       # (n_bonds, 3)
    cg_angles_before: np.ndarray     # (n_angles, 4)
    cg_angles_after: np.ndarray      # (n_angles, 4)
    cg_dihedrals_before: np.ndarray  # (n_dihedrals, 5)
    cg_dihedrals_after: np.ndarray   # (n_dihedrals, 5)
    reaction_type: str
    # 方法: save(output_dir, index=0) -> str
```

**核心类:**

```python
class BondsRecorder:
    """键连表记录器"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(output_dir)` | `output_dir: str` | — | 初始化 |
| `record(timestep, run_step, bonds_before, bonds_after, reaction_type)` | `...` | `BondRecord` | 记录键变化 |
| `save_all()` | — | `List[str]` | 保存所有记录 |
| `save_incremental(record)` | `record: BondRecord` | `str` | 增量保存 |
| `record_cg_topology(timestep, run_step, cg_bonds_before, cg_bonds_after, cg_angles_before, cg_angles_after, cg_dihedrals_before, cg_dihedrals_after, reaction_type)` | `...` | `CGTopologyRecord` | 记录 CG 拓扑变化 |
| `clear()` | — | — | 清空记录 |
| `get_summary()` | — | `dict` | 统计摘要 |

**便捷函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `save_bonds_record(output_path, timestep, run_step, bonds_before, bonds_after, reaction_type)` | `...` | `str` | 保存单次键记录 |
| `load_bonds_record(filepath)` | `filepath: str` | `BondRecord` | 加载键记录 |

---

#### smoke_validator.py — 冒烟测试验证

四项检查：文件输出完整性、CG mapping 一致性、反应后 mapping 交叉验证、反应计数合理性。

**数据类:**

```python
@dataclass
class SmokeTestReport:
    passed: bool = True
    messages: List[str] = ...
    artifacts: Dict[str, Path] = ...
    # 方法: add(check_name, status, detail), print()
```

**核心类:**

```python
class SmokeValidator:
    """冒烟测试验证器：四项检查。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `validate(temp_dir, loop_num, n_reaction_types=0, n_atoms=None)` | `temp_dir: Path, loop_num: int, n_reaction_types: int, n_atoms: Optional[int]` | `SmokeTestReport` | 执行所有验证检查 |

---

#### smoke_test_harness.py — 冒烟测试运行编排

创建临时目录、复制配置、覆盖 loop_num 运行、调用 SmokeValidator 验证。

**核心类:**

```python
class SmokeTestHarness:
    """冒烟测试编排器（MPI 感知版）。"""
```

| 方法 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(source_config_dir, loop_num=5, keep_output=False)` | `source_config_dir: str, loop_num: int, keep_output: bool` | — | 初始化 |
| `run()` | — | `Optional[SmokeTestReport]` | 执行冒烟测试（rank 0 返回报告，其他 rank 返回 None） |

**使用示例:**

```python
from LmpPy.core import SmokeTestHarness

harness = SmokeTestHarness("config/", loop_num=5)
report = harness.run()
if report is not None:
    report.print()
    sys.exit(0 if report.passed else 1)
```

---

## 2. 工具模块 (tools/)

### aa2cg/ — 全原子到粗粒化转换

#### 顶层导出

通过 `LmpPy.tools.aa2cg`（以及 `LmpPy.tools`）直接使用：

```python
from LmpPy.tools.aa2cg import (
    load_aa_to_cg_mapping,        # 加载 AA→CG 映射 CSV
    convert_aa_to_cg_frame,       # 单帧坐标转换
    read_lammps_data,             # 读取 LAMMPS data 文件
    write_cg_data_file,           # 写入 CG data 文件
    convert_data_to_cg,           # AA data → CG data 完整转换
    read_gromacs_trr_all_frames,  # 读取 GROMACS TRR 轨迹
    convert_trajectory_to_cg,     # AA 轨迹 → CG 轨迹
    convert_trajectory_to_cg_optimized,  # 优化版（Numba JIT）
    save_cg_trajectory_pickle,    # 保存 CG 轨迹为 pickle
    load_cg_trajectory_pickle,    # 加载 CG 轨迹 pickle
    write_trajectory_to_xyz,      # 写入 XYZ 轨迹
)
```

主要函数：

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `read_lammps_data(data_file)` | `data_file: str` | `Dict` | 读取 LAMMPS data 文件（含 ids, mol_ids, types, coords, bonds, angles, dihedrals, mass_list, box） |
| `write_cg_data_file(filename, cg_data, cg_bonds=None, cg_angles=None, cg_dihedrals=None, mass_list=None)` | `filename: str, cg_data: Dict, ...` | — | 写入 CG LAMMPS data 文件 |
| `convert_data_to_cg(aa_data, mapping_csv, cg_bonds_file=None, cg_angles_file=None, cg_dihedrals_file=None, derive_topology=False, output_cg_topology_dir=None, export_bead_info=True, type_mapping_yaml=None)` | `aa_data: Dict, mapping_csv: str, ...` | `(cg_data, mapping_dict)` | AA data→CG data 完整转换 |
| `read_gromacs_trr_all_frames(tpr_file, trr_file, make_whole=True, stride=1)` | `tpr_file, trr_file: str, make_whole: bool, stride: int` | `List[Dict]` | 读取 GROMACS TRR 轨迹所有帧 |
| `convert_trajectory_to_cg(aa_frames, mapping_csv)` | `aa_frames: List[Dict], mapping_csv: str` | `List[Dict]` | AA 轨迹→CG 轨迹 |
| `convert_trajectory_to_cg_optimized(aa_frames, mapping_csv, use_numba=True, verbose=True)` | 同上 + `use_numba, verbose` | `List[Dict]` | 优化版（30-100x 加速） |
| `save_cg_trajectory_pickle(cg_trajectory, output_file)` | `cg_trajectory: List[Dict], output_file: str` | — | 保存为 pickle |
| `load_cg_trajectory_pickle(pickle_file)` | `pickle_file: str` | `List[Dict]` | 加载 pickle |

---

### ibm_potential/ — IBM 势能计算

通过 IBM（Inverse Boltzmann Method）从 CG 轨迹计算势能表。

**配置数据类:**

```python
@dataclass
class IBMConfig:
    simulation: SimulationConfig       # temperature, units
    trajectory: TrajectoryConfig       # format, path, stride
    distribution: DistributionConfig   # n_bins, bond_range, angle_range, dihedral_range, pair_range, rdf_dr, rdf_exclude_*
    smoothing: SmoothingConfig         # method, sg_window, sg_polyorder, sigma_range
    output: OutputConfig               # table_dir, potentials_dir, distributions_dir, generate_plots
    config_dir: Path = ...
    # 方法: resolve_path(path) -> Path
```

**核心函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `load_ibm_config(config_path=None)` | `config_path: Optional[str]` | `IBMConfig` | 加载 IBM 配置 |
| `load_cg_trajectory(path)` | `path: str` | — | 加载 CG 轨迹 |
| `load_topology(bonds_file, angles_file, dihedrals_file)` | 文件路径 | — | 加载拓扑 |
| `calculate_bond_distribution(cg_data, bond_pairs)` | — | `(r, hist)` | 计算键长分布 |
| `calculate_angle_distribution(cg_data, angle_pairs)` | — | `(theta, hist)` | 计算角度分布 |
| `calculate_dihedral_distribution(cg_data, dihedral_pairs)` | — | `(phi, hist)` | 计算二面角分布 |
| `calculate_rdf(cg_data, pair_types, dr)` | — | `(r, g_r)` | 计算 RDF |
| `calculate_bond_potential(r, hist, temperature=400, units='kcal/mol', jacobian_correction=True)` | `r, hist: np.ndarray, temperature: float, ...` | `(r, potential)` | 玻尔兹曼反演得到键势能 |
| `calculate_angle_potential(theta, hist, temperature=400, units='kcal/mol', jacobian_correction=True)` | 同上 | `(theta, potential)` | 角度势能 |
| `calculate_dihedral_potential(phi, hist, temperature=400, units='kcal/mol')` | 同上 | `(phi, potential)` | 二面角势能 |
| `calculate_pair_potential(r, g_r, temperature=400, units='kcal/mol')` | 同上 | `(r, potential, g_r)` | 非键合势能 |
| `extrapolate_and_smooth(x, potential, smooth_window=21, smooth_polyorder=3, extrap_points=5, extrap_method='linear')` | `x, potential: np.ndarray, ...` | `(x, potential_processed)` | 平滑外推 |
| `create_lammps_table_files(input_dir, output_dir, units='real', contributor='...', date_str=None)` | `input_dir, output_dir: str, ...` | `Dict[str, str]` | 生成 LAMMPS table 文件 |
| `create_reference_file(output_dir)` | `output_dir: str` | — | 创建 table 参考文件 |

---

### smooth_utils/ — 分布平滑工具

分布平滑工具包，支持自适应 Savitzky-Golay 平滑、高斯平滑和边界约束平滑。

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `smooth_distribution(x, P, method='auto', ...)` | 坐标和概率密度 | `(P_smooth, info)` | 自动选择平滑策略 |
| `smooth_all_distributions(distributions, **kwargs)` | `distributions: Dict` | `Dict` | 批量平滑所有分布 |
| `adaptive_smooth(x, P, zone_mask, window_small=11, window_large=31, polyorder=3)` | `x, P, zone_mask: np.ndarray` | `P_smooth` | 自适应平滑（峰区小窗口，谷区大窗口） |
| `gaussian_smooth_with_constraint(x, P, original_n_peaks, sigma_range=None, verbose=False)` | 同上 + `original_n_peaks: int` | `(P_smooth, sigma_used, n_peaks_smooth)` | 峰数约束的高斯平滑 |
| `gaussian_smooth_for_rdf(x, P, idx_valid_start, idx_valid_end, ..., temperature=400)` | 同上 + `idx_valid_start, idx_valid_end` | `(P_smooth, sigma_used, metrics)` | RDF 优化的高斯平滑（边界保护） |
| `smooth_bond_with_harmonic_boundary(x, P, idx_valid_start, idx_valid_end, ...)` | 同上 | `(P_smooth, metrics)` | 键分布的谐波边界平滑 |
| `smooth_rdf_with_harmonic_left_boundary(x, g, idx_valid_start, idx_valid_end, ...)` | 同上 | `(g_smooth, metrics)` | RDF 的左侧谐波边界平滑 |

---

## 3. 工具函数 (utils/)

### coordinate_utils.py — 坐标处理

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `wrap_coordinates(coords, box, return_images=True)` | `coords: np.ndarray (n, 3), box: np.ndarray (3, 2), return_images: bool` | `(wrapped_coords, images)` | 将坐标映射回盒子内 |
| `pbc_distance(coord1, coord2, box)` | `coord1, coord2: (3,) 或 (n, 3), box: (3, 2)` | `np.ndarray` | 最小镜像距离 |
| `unwrap_molecule_bfs(start_atom, coords, graph, box, visited)` | `start_atom: int, coords: dict, graph: dict, box: (3, 2), visited: set` | `(unwrapped, visited)` | BFS 展开单分子坐标 |
| `unwrap_coords_numba(coords, bonds, box, molecule_ids)` | `coords: (n_atoms, 3), bonds: (n_bonds, 2), box: (3, 2), molecule_ids: (n_atoms,)` | `np.ndarray` | Numba 优化坐标展开 |
| `unwrap_coords_python(coords, bonds, box, molecule_ids)` | 同上 | `np.ndarray` | Python 实现坐标展开 |
| `calculate_central_mass(coords, masses)` | `coords: (n, 3), masses: (n,)` | `np.ndarray (3,)` | 计算质心 |

---

### graph_utils.py — 图论算法

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `find_molecules(bonds, natoms, use_numba=True)` | `bonds: (n_bonds, 2) 或 (n_bonds, 3), natoms: int, use_numba: bool` | `np.ndarray (natoms,)` | BFS 查找分子（30-80x Numba 加速） |
| `build_bond_graph(bonds, natoms)` | 同上 | `Dict[int, List[int]]` | 构建键连邻接表 |
| `get_molecule_sizes(molecule_ids)` | `molecule_ids: np.ndarray` | `np.ndarray (n_molecules,)` | 获取每个分子的大小 |
| `get_molecule_atoms(molecule_ids, mol_id)` | `molecule_ids: np.ndarray, mol_id: int` | `np.ndarray` | 获取指定分子的所有原子 |

常量: `NUMBA_AVAILABLE: bool`

---

### file_utils.py — 文件 I/O

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `write_lammps_dump_file(path, timestep, box, id_arr, atype_arr, coord_arr, ixyz=None, append=True)` | `path: str, box: (3, 2), id_arr, atype_arr, coord_arr: np.ndarray, ixyz: Optional[ndarray], append: bool` | — | 写入 LAMMPS dump 格式轨迹 |
| `write_cg_trajectory(path, timestep, box, bead_ids, bead_types, bead_coords, append=True)` | 同上（bead 版） | — | 写入 CG 轨迹 |
| `read_lammps_dump_file(path, max_frames=None)` | `path: str, max_frames: Optional[int]` | `List[Dict]` | 读取 LAMMPS dump 轨迹 |

---

### topology.py — 拓扑文件读写

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `read_topology_files(bonds_file, angles_file=None, dihedrals_file=None)` | `bonds_file, angles_file, dihedrals_file: str` | `TopologyData` | 读取拓扑文件 |
| `read_type_dict(file_path, n_beads)` | `file_path: str, n_beads: int` | `Dict[int, Tuple]` | 读取类型字典 |
| `write_topology_file(output_file, topology, topology_type, comment='')` | `output_file: str, topology: np.ndarray, topology_type: str, comment: str` | — | 写入拓扑文件 |
| `derive_cg_bonds_from_aa(aa_bonds, aa_to_cg_mapping, bead_types=None, type_mapping=None)` | `aa_bonds: np.ndarray, aa_to_cg_mapping: Dict, ...` | `np.ndarray` | 从 AA 键推导 CG 键 |
| `assign_topology_types(topology_array, bead_types, topology_kind, type_mapping=None)` | `topology_array: np.ndarray, bead_types: Dict, topology_kind: str, type_mapping: Optional[Dict]` | `np.ndarray` | 根据 bead type 组合分配拓扑类型 ID |

**数据类:**

```python
@dataclass
class TopologyData:
    bonds: np.ndarray       # (nbonds, 3) [type, atom1, atom2]
    angles: np.ndarray      # (nangles, 4)
    dihedrals: np.ndarray   # (ndihedrals, 5)
```

---

### units.py — 单位转换

**常量:**

- `ENERGY_CONVERSION`: 能量转换因子（相对于 kcal/mol）
- `LENGTH_CONVERSION`: 长度转换因子（相对于 Angstrom）
- `KB`: 玻尔兹曼常数（按能量单位索引）

**核心函数:**

| 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `convert_energy(value, from_unit, to_unit)` | `value: float, from_unit: str, to_unit: str` | `float` | 能量单位转换 |
| `convert_length(value, from_unit, to_unit)` | 同上 | `float` | 长度单位转换 |
| `get_kb(unit)` | `unit: str` | `float` | 获取指定单位的玻尔兹曼常数 |
| `thermal_energy(temperature, unit='kcal/mol')` | `temperature: float, unit: str` | `float` | 计算 kT |

**核心类:**

```python
class UnitConverter:
    def __init__(self, energy_unit='kcal/mol', length_unit='A'): ...
    def convert_energy(self, value, from_unit=None, to_unit=None): ...
    def convert_length(self, value, from_unit=None, to_unit=None): ...
    @property
    def kb(self) -> float: ...
    def kt(self, temperature) -> float: ...
```

---

## 4. 后处理分析 (output_analysis/)

### 模块概述

提供三种核心分析和统一的绘图配置系统，CLI 入口统一管理。

**数据类配置系统（plot.py）：**

```python
@dataclass
class PlotConfig:
    # 从 YAML 加载，包含各 chart 样式
    charts: Dict[str, ChartStyle] = ...
    # 类方法: from_yaml(path) -> PlotConfig

@dataclass
class ChartStyle:
    title: Optional[str] = None
    figsize: Tuple[float, float] = (8, 5)
    dpi: int = 150
    xlabel: Optional[str] = None
    ylabel: Optional[str] = None
    histogram: Optional[HistogramStyle] = None
    kde: Optional[KDEStyle] = None
    line: Optional[LineStyle] = None
    # 方法: fill_defaults() -> ChartStyle
```

### 各模块函数

| 模块 | 函数 | 参数 | 返回值 | 说明 |
|---|---|---|---|---|
| `chain_length` | `analyze_chain_length(input_dir, min_length=1, theory_type=None, Mn=None, PDI=None, js=False, bins=None, output_dir=None, plot_config=None, kde_bw='scott')` | `input_dir: str | Path, ...` | — | 链长分布分析，生成直方图 + KDE + 理论曲线对比 |
| `distance` | `analyze_distance(df=None, input_dir=None, ref_csv=None, bins=None, output_dir=None, plot_config=None)` | `df: Optional[pd.DataFrame], input_dir: Optional[str | Path], ref_csv: Optional[str | Path], ...` | — | 反应距离分布分析 |
| `reaction_stats` | `analyze_reaction_stats(df=None, input_dir=None, output_dir=None, plot_config=None)` | `df: Optional[pd.DataFrame], ...` | `Dict` | 反应统计（总数、按类型、按循环） |
| `js_divergence` | `js_divergence(p, q)` | `p, q: np.ndarray` | `float` | JS 散度 [0,1] |
| `js_divergence` | `safe_kl_divergence(p, q, eps=1e-12)` | 同上 + `eps` | `float` | 安全 KL 散度 |
| `theory` | `schulz_zimm(N, Mn, PDI)` | `N: np.ndarray, Mn: float, PDI: float` | `np.ndarray` | Schulz-Zimm (Gamma) 分布 |
| `theory` | `poisson(N, Mn)` | `N: np.ndarray, Mn: float` | `np.ndarray` | Poisson 分布 |
| `theory` | `log_normal(N, Mn, PDI)` | 同上 | `np.ndarray` | 对数正态分布 |
| `loader` | `load_reaction_details(input_dir)` | `input_dir: str | Path` | `pd.DataFrame` | 重建 `reaction_details.csv` |
| `loader` | `load_final_frame_topology(input_dir)` | `input_dir: str | Path` | `pd.DataFrame` | 解析最终帧链拓扑 |

**CLI 入口（cli.py）：**

```bash
python -m LmpPy.output_analysis chain-length <input_dir> [--min-length N] [--theory-type TYPE] [--Mn F] [--PDI F] [--js]
python -m LmpPy.output_analysis distance <input_dir> [--ref-csv PATH] [--bins N]
python -m LmpPy.output_analysis reaction-stats <input_dir>
python -m LmpPy.output_analysis all <input_dir>
python -m LmpPy.output_analysis rebuild <npz_path> -o <csv_path>
```

---

## 5. 入口与集成

### run_refactored.py — LAMMPS 反应模拟运行器

主入口脚本，初始化 LAMMPS、运行主循环、处理反应、输出 CG 轨迹。

**核心类:**

```python
class LAMMPSReactionRunner:
    """LAMMPS 反应模拟运行器。"""
```

| 方法/属性 | 参数 | 返回值 | 说明 |
|---|---|---|---|
| `__init__(config_dir)` | `config_dir: str` | — | 初始化：加载配置、初始化组件、加载/生成 CG 映射 |
| `run()` | — | — | 运行主循环（MPI 感知） |
| `system_config` | — | `SystemConfig` | 加载的系统配置 |
| `lammps_params` | — | `LAMMPSParams` | 加载的 LAMMPS 参数 |
| `cg_compare_list` | — | `CGCompareList` | 当前 CG 映射 |
| `reaction_templates` | — | `Dict[str, ReactionTemplate]` | 加载的反应模板 |
| `mass_list` | — | `Dict[int, float]` | 质量列表 |

**运行方式：**

```bash
# 单进程运行
python -m LmpPy.run_refactored config/

# MPI 并行
mpirun -np 4 python -m LmpPy.run_refactored config/ --loop-num 100

# 测试模式（不运行 LAMMPS）
python -m LmpPy.run_refactored config/ --test
```

---

### test_integration.py — 集成测试

不依赖 LAMMPS 运行的模块接口测试。

**测试函数:**

| 函数 | 说明 |
|---|---|
| `test_config_loader(config_dir)` | 测试配置加载器 |
| `test_mapping_generator(config_dir, system_config)` | 测试 CG 映射生成 |
| `test_template_parser()` | 测试模板解析 |
| `test_bond_detector()` | 测试键变化检测 |
| `test_cg_converter(mass_list)` | 测试 CG 坐标转换 |
| `test_cg_mapper()` | 测试 CG 映射更新 |
| `test_cg_bond_mapper(cg_mapping)` | 测试粗粒键映射 |
| `test_edge_atoms_exclusion()` | 测试边缘原子排除 |

**运行方式：**

```bash
python -m LmpPy.test_integration --config config/
```

---

## 6. CLI 脚本 (scripts/)

| 脚本 | 用途 |
|---|---|
| `build_cg_config.py` | 构建 CG 配置文件模板 |
| `build_cg_system.py` | 构建 CG 体系（生成初始映射和拓扑） |
| `calc_dist.py` | 计算分布（键长、角度、二面角） |
| `calc_ibm_potential.py` | IBM 势能计算流程（从轨迹到 LAMMPS table） |
| `calc_ibm_potential_from_dist.py` | 从已有分布文件计算 IBM 势能 |
| `convert_aa2cg.py` | AA→CG 转换（data 文件或轨迹） |
| `data2gro.py` | LAMMPS data → GROMACS gro 格式 |
| `generate_initial_mapping.py` | 生成初始 CG 映射（从 YAML 配置） |
| `plot_dist.py` | 绘制分布图 |
| `smooth_distribution.py` | 平滑分布 |
| `test_smoke_run.py` | 冒烟测试运行脚本 |
| `validate_config.py` | 验证配置目录完整性 |
| `yaml2csv_mapping.py` | YAML 映射配置 → CSV 映射文件 |

---

> 文档版本: 2026-06-22
> 对应 LmpPy 模块版本: 2.6+
