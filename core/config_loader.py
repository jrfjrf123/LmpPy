"""
配置加载器模块

功能:
- 加载所有YAML配置文件
- 严格验证配置字段
- 提供配置数据对象

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path
import yaml
import numpy as np

# 支持两种导入方式
try:
    from .lammps_data_extractor import parse_masses_from_data_file, DataExtractorError
except ImportError:
    from lammps_data_extractor import parse_masses_from_data_file, DataExtractorError


# ============================================================================
# 异常定义
# ============================================================================

class ConfigError(Exception):
    """配置错误基类"""
    pass


class ConfigFileNotFoundError(ConfigError):
    """配置文件未找到"""
    def __init__(self, path: Path):
        super().__init__(f"配置文件未找到: {path}")
        self.path = path


class ConfigValidationError(ConfigError):
    """配置验证错误"""
    def __init__(self, message: str, path: Path = None):
        if path:
            message = f"{path}: {message}"
        super().__init__(message)


class ConfigMissingFieldError(ConfigValidationError):
    """缺少必需字段"""
    def __init__(self, field_name: str, path: Path = None):
        super().__init__(f"缺少必需字段: '{field_name}'", path)
        self.field_name = field_name


class ConfigInvalidValueError(ConfigValidationError):
    """字段值无效"""
    def __init__(self, field_name: str, value: Any, reason: str, path: Path = None):
        super().__init__(f"字段 '{field_name}' 值无效: {value} ({reason})", path)
        self.field_name = field_name
        self.value = value


# ============================================================================
# 验证结果数据类
# ============================================================================

@dataclass
class ValidationIssue:
    """验证问题"""
    level: str              # "error" | "warning" | "info"
    category: str           # "file" | "config" | "semantic"
    field: str              # 字段名或路径
    message: str            # 详细描述
    suggestion: str = ""    # 建议的修复方案
    path: Optional[Path] = None  # 相关文件路径

    def __str__(self) -> str:
        """格式化输出"""
        symbols = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ "}
        symbol = symbols.get(self.level, "?")
        base = f"{symbol} {self.field} - {self.message}"
        if self.suggestion:
            base += f"\n      建议: {self.suggestion}"
        return base


@dataclass
class ValidationResult:
    """验证结果"""
    issues: List[ValidationIssue] = field(default_factory=list)
    config_dir: Optional[Path] = None

    @property
    def passed(self) -> bool:
        """是否通过验证 (无 error)"""
        return self.n_errors == 0

    @property
    def n_errors(self) -> int:
        """错误数量"""
        return sum(1 for i in self.issues if i.level == "error")

    @property
    def n_warnings(self) -> int:
        """警告数量"""
        return sum(1 for i in self.issues if i.level == "warning")

    @property
    def n_infos(self) -> int:
        """信息数量"""
        return sum(1 for i in self.issues if i.level == "info")

    def add_error(self, category: str, field: str, message: str,
                  suggestion: str = "", path: Optional[Path] = None):
        """添加错误"""
        self.issues.append(ValidationIssue(
            level="error", category=category, field=field,
            message=message, suggestion=suggestion, path=path
        ))

    def add_warning(self, category: str, field: str, message: str,
                    suggestion: str = "", path: Optional[Path] = None):
        """添加警告"""
        self.issues.append(ValidationIssue(
            level="warning", category=category, field=field,
            message=message, suggestion=suggestion, path=path
        ))

    def add_info(self, category: str, field: str, message: str,
                 suggestion: str = "", path: Optional[Path] = None):
        """添加信息"""
        self.issues.append(ValidationIssue(
            level="info", category=category, field=field,
            message=message, suggestion=suggestion, path=path
        ))

    def add_success(self, category: str, field: str, message: str = "正常"):
        """添加成功信息"""
        self.issues.append(ValidationIssue(
            level="success", category=category, field=field, message=message
        ))

    def get_errors(self) -> List[ValidationIssue]:
        """获取所有错误"""
        return [i for i in self.issues if i.level == "error"]

    def get_warnings(self) -> List[ValidationIssue]:
        """获取所有警告"""
        return [i for i in self.issues if i.level == "warning"]

    def to_report(self) -> str:
        """生成详细报告文本"""
        lines = []
        lines.append("=" * 80)
        lines.append(f"配置验证报告 - {self.config_dir}")
        lines.append("=" * 80)

        # 按类别分组
        categories = {"file": "文件一致性检查", "config": "配置间一致性检查", "semantic": "语义验证"}

        for cat, cat_name in categories.items():
            cat_issues = [i for i in self.issues if i.category == cat]
            if not cat_issues:
                continue

            lines.append(f"\n【{cat_name}】")
            for issue in cat_issues:
                symbols = {"error": "❌", "warning": "⚠️ ", "info": "ℹ️ ", "success": "✅"}
                symbol = symbols.get(issue.level, "?")
                line = f"  {symbol} {issue.field} - {issue.message}"
                lines.append(line)
                if issue.suggestion:
                    lines.append(f"      建议: {issue.suggestion}")

        # 统计
        lines.append("\n" + "=" * 80)
        lines.append("验证结果统计")
        lines.append("=" * 80)
        lines.append(f"  错误 (Error):   {self.n_errors} 个")
        lines.append(f"  警告 (Warning): {self.n_warnings} 个")
        lines.append(f"  信息 (Info):    {self.n_infos} 个")

        if self.passed:
            lines.append("\n  ✅ 验证通过 - 配置可以运行")
            if self.n_warnings > 0:
                lines.append("  ⚠️  存在警告，建议检查后运行")
        else:
            lines.append("\n  ❌ 验证未通过 - 存在错误，请修复后重新运行")

        # 修复建议
        errors = self.get_errors()
        if errors:
            lines.append("\n需要修复的错误:")
            for i, err in enumerate(errors, 1):
                lines.append(f"  {i}. [{err.category}] {err.field}: {err.message}")
                if err.suggestion:
                    lines.append(f"     -> {err.suggestion}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def print_report(self):
        """打印报告"""
        print(self.to_report())


# ============================================================================
# 数据类定义
# ============================================================================

@dataclass
class MappingConfig:
    """CG映射配置"""
    site_types: Dict[str, Dict[str, List]]
    config: List[Dict[str, Any]]
    # 可选: bead类型名称到数值ID的映射
    # 如 {'Head': 1, 'Mid': 2, 'End': 3}
    bead_type_names: Optional[Dict[str, int]] = None


@dataclass
class ReactionInfo:
    """单个反应信息（用于LAMMPS bond/react）

    注意：所有文件路径均为完整绝对路径，由 load_reactions_from_directory() 自动填充
    """
    name: str                           # 反应名称
    cutoff: float                       # 反应截断半径 (Å)
    pre_mol: str                        # 反应前分子模板名
    post_mol: str                       # 反应后分子模板名

    # 完整路径（验证后填充）
    map_file: str = ""                  # bond/react map 文件完整路径
    pre_template: str = ""              # 反应前模板文件完整路径
    post_template: str = ""             # 反应后模板文件完整路径
    pre_mapping: str = ""               # 反应前 CG 映射文件完整路径（可选）
    post_mapping: str = ""              # 反应后 CG 映射文件完整路径（可选）

    # 反应目录路径
    rxn_dir: str = ""                   # reactions/{name}/ 目录完整路径


def load_reactions_from_directory(
    config_dir: Path,
    reaction_configs: List[Dict[str, Any]],
    validate_files: bool = True
) -> List[ReactionInfo]:
    """
    从 reactions/ 目录统一加载反应配置

    统一的读取流程：
    1. 检查 reactions/ 目录存在 → 否则 error
    2. 检查每个 reaction_configs 中的 name 对应子目录存在 → 否则 error
    3. 检查每个子目录包含必需文件 → 否则 error
    4. 构建完整路径，供 LAMMPS 命令和 CG 映射更新使用

    Args:
        config_dir: 配置目录路径
        reaction_configs: YAML 中的 reactions 配置列表，每个元素包含:
            - name: 反应名称
            - cutoff: 反应截断半径
            - pre_mol: 反应前分子模板名
            - post_mol: 反应后分子模板名
        validate_files: 是否验证文件完整性

    Returns:
        List[ReactionInfo]: 带完整路径的反应信息列表

    Raises:
        ConfigFileNotFoundError: reactions 目录不存在
        ConfigValidationError: 反应子目录或必需文件缺失
    """
    reactions_dir = config_dir / "reactions"

    # 1. 检查 reactions 目录存在
    if not reactions_dir.exists():
        raise ConfigFileNotFoundError(reactions_dir)

    reactions = []
    missing_dirs = []
    missing_files = []

    for rxn_cfg in reaction_configs:
        rxn_name = rxn_cfg['name']
        rxn_dir = reactions_dir / rxn_name

        # 2. 检查子目录存在
        if not rxn_dir.exists():
            missing_dirs.append(rxn_name)
            continue

        # 3. 检查必需文件
        map_file = rxn_dir / f"{rxn_name}.map"
        pre_template = rxn_dir / f"{rxn_name}_pre.lammpstemplate"
        post_template = rxn_dir / f"{rxn_name}_post.lammpstemplate"

        if validate_files:
            if not map_file.exists():
                missing_files.append(str(map_file))
            if not pre_template.exists():
                missing_files.append(str(pre_template))
            if not post_template.exists():
                missing_files.append(str(post_template))

        # 4. 可选文件
        pre_mapping = rxn_dir / f"{rxn_name}_pre_mapping.yaml"
        post_mapping = rxn_dir / f"{rxn_name}_post_mapping.yaml"

        # 构建 ReactionInfo（完整路径）
        reactions.append(ReactionInfo(
            name=rxn_name,
            cutoff=float(rxn_cfg['cutoff']),
            pre_mol=rxn_cfg['pre_mol'],
            post_mol=rxn_cfg['post_mol'],
            map_file=str(map_file),
            pre_template=str(pre_template),
            post_template=str(post_template),
            pre_mapping=str(pre_mapping) if pre_mapping.exists() else "",
            post_mapping=str(post_mapping) if post_mapping.exists() else "",
            rxn_dir=str(rxn_dir)
        ))

    # 报告错误
    if missing_dirs:
        raise ConfigValidationError(
            f"reactions 子目录缺失: {', '.join(missing_dirs)}\n"
            f"请在 reactions/ 目录下创建对应的反应子目录"
        )
    if missing_files:
        raise ConfigValidationError(
            f"反应必需文件缺失:\n  " + "\n  ".join(missing_files)
        )

    return reactions


@dataclass
class ReadDataExtra:
    """read_data 额外参数"""
    special_per_atom: int = 3
    bond_per_atom: int = 3
    angle_per_atom: int = 3
    dihedral_per_atom: int = 3


@dataclass
class LAMMPSParams:
    """LAMMPS运行参数"""
    # 模拟参数 (无默认值，必须在前面)
    loop_num: int
    dt: float
    timestep: int  # LAMMPS timestep 设置 (fs)
    temperature: float
    pressure: float

    # 步数配置 (无默认值)
    bond_react_check_step: int
    run_step: int
    nve_limit_step: int

    # NPT参数 (无默认值)
    tcouple: float
    pcouple: float

    # bond/react参数 (无默认值)
    stabilization: float

    # 以下是有默认值的字段
    ensemble: str = "npt"  # 系综类型: "npt" 或 "nvt"
    reactions: List[ReactionInfo] = field(default_factory=list)

    # 分子模板配置 (模板名 -> 文件路径)
    molecules: Dict[str, str] = field(default_factory=dict)

    # 原子类型质量配置 (原子类型 -> 质量)
    mass_list: Dict[int, float] = field(default_factory=dict)

    # 文件配置
    data_file: str = "system.data"  # LAMMPS数据文件
    input_script: str = ""  # 可选，若设置则使用外部输入脚本
    initial_cg_mapping: str = "AtomId_BeadId_compare_list.csv"
    output_cg_trajectory: str = "cg_trajectory.lammpstrj"
    output_reaction_count: str = "reaction_num.txt"
    output_final_data: str = "final_frame.data"
    output_final_mapping: str = "final_cg_compare_list.csv"
    output_bonds_record: str = "bonds_record.npz"

    # CG 拓扑输出文件
    output_cg_bonds: str = "cg_bonds.txt"
    output_cg_angles: str = "cg_angles.txt"
    output_cg_dihedrals: str = "cg_dihedrals.txt"

    # read_data 额外参数
    read_data_extra: ReadDataExtra = field(default_factory=ReadDataExtra)

    # pair style (用于 SOAP 计算时获取 neighbor list)
    pair_style: str = "lj/cut"  # 默认值，可在 lammps_params.yaml 中覆盖


@dataclass
class ReactionConfig:
    """反应配置（包含模板和映射）"""
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
class SystemConfig:
    """体系配置"""
    name: str
    mapping_files: List[Dict[str, Any]]
    mass_list: Dict[int, float]
    bead_type_names: Dict[str, int]  # 全局 bead 类型名称映射 (必需)

    # 目录配置（相对于 config_dir 或绝对路径）
    work_dir: str = "work"      # LAMMPS 运行目录
    output_dir: str = "output"  # 输出文件目录


# ============================================================================
# 配置加载器
# ============================================================================

class ConfigLoader:
    """
    配置加载器

    功能:
    - 加载所有YAML配置文件
    - 严格验证必需字段
    - 返回类型化的配置对象
    """

    def __init__(self, config_dir: str):
        """
        初始化配置加载器

        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self._validate_config_dir()

    def _validate_config_dir(self):
        """验证配置目录结构"""
        if not self.config_dir.exists():
            raise ConfigFileNotFoundError(self.config_dir)

        # 检查必需文件 (mass_list.yaml 现在是可选的)
        required_files = ['system.yaml', 'lammps_params.yaml']
        for f in required_files:
            path = self.config_dir / f
            if not path.exists():
                raise ConfigFileNotFoundError(path)

    def _load_yaml(self, path: Path) -> Dict:
        """加载YAML文件"""
        if not path.exists():
            raise ConfigFileNotFoundError(path)

        with open(path, 'r', encoding='utf-8') as f:
            try:
                return yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise ConfigError(f"YAML解析错误: {path}: {e}")

    def _validate_required(self, data: Dict, keys: List[str], path: Path):
        """验证必需字段存在"""
        for key in keys:
            if key not in data:
                raise ConfigMissingFieldError(key, path)

    def _resolve_path(self, path: str | Path) -> Path:
        """解析路径（支持相对路径）"""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.config_dir / path

    # ==================== 加载方法 ====================

    def load_system_config(self) -> SystemConfig:
        """
        加载体系配置

        Returns:
            SystemConfig: 体系配置对象
        """
        path = self.config_dir / 'system.yaml'
        data = self._load_yaml(path)

        # 验证必需字段
        self._validate_required(data, ['system'], path)
        system = data['system']
        self._validate_required(system, ['mapping_files'], path)

        # 加载质量列表
        # 优先级：1. mass_list.yaml  2. 外部传入的 data_file
        mass_list_path = self.config_dir / 'mass_list.yaml'
        if mass_list_path.exists():
            mass_list = self._load_mass_list_from_yaml(mass_list_path)
        else:
            # 不再自动从数据文件提取，返回空字典
            # 调用者应该从 lammps_params.mass_list 获取
            mass_list = {}

        # 验证mapping_files格式
        mapping_files = system['mapping_files']
        for i, mf in enumerate(mapping_files):
            if 'path' not in mf:
                raise ConfigValidationError(
                    f"mapping_files[{i}] 缺少 'path' 字段", path
                )
            if 'copies' not in mf:
                raise ConfigValidationError(
                    f"mapping_files[{i}] 缺少 'copies' 字段", path
                )

        # 验证 bead_type_names 存在 (必需字段)
        if 'bead_type_names' not in system:
            raise ConfigMissingFieldError('bead_type_names', path)

        bead_type_names = system['bead_type_names']

        # 验证 bead_type_names 格式
        if not isinstance(bead_type_names, dict):
            raise ConfigValidationError(
                'bead_type_names', bead_type_names, "必须是字典类型", path
            )

        return SystemConfig(
            name=system.get('name', 'unnamed'),
            mapping_files=mapping_files,
            mass_list=mass_list,
            bead_type_names=bead_type_names,
            work_dir=system.get('work_dir', 'work'),
            output_dir=system.get('output_dir', 'output')
        )

    def _get_data_file_from_params(self) -> Optional[str]:
        """
        从 lammps_params.yaml 获取 data_file 路径

        Returns:
            data_file 路径，如果不存在则返回 None
        """
        params_path = self.config_dir / 'lammps_params.yaml'
        if not params_path.exists():
            return None

        data = self._load_yaml(params_path)
        files = data.get('files', {})
        return files.get('data_file')

    def _load_mass_list_from_yaml(self, mass_list_path: Path) -> Dict[int, float]:
        """
        从 YAML 文件加载质量列表

        Args:
            mass_list_path: mass_list.yaml 文件路径

        Returns:
            Dict[int, float]: 原子类型 -> 质量
        """
        data = self._load_yaml(mass_list_path)
        self._validate_required(data, ['mass_list'], mass_list_path)

        mass_list = {}
        for k, v in data['mass_list'].items():
            try:
                atom_type = int(k)
                mass = float(v)
                if mass <= 0:
                    raise ConfigInvalidValueError(
                        f"mass_list[{k}]", mass, "质量必须为正数", mass_list_path
                    )
                mass_list[atom_type] = mass
            except (ValueError, TypeError) as e:
                raise ConfigInvalidValueError(
                    f"mass_list[{k}]", v, str(e), mass_list_path
                )

        return mass_list

    def load_mass_list(self, data_file: Optional[str] = None) -> Dict[int, float]:
        """
        加载质量列表

        优先级:
        1. 如果存在 mass_list.yaml，使用它作为覆盖配置
        2. 否则，从 data_file 自动提取 Masses 部分

        Args:
            data_file: 可选的LAMMPS data文件路径，用于自动提取质量

        Returns:
            Dict[int, float]: 原子类型 -> 质量
        """
        mass_list_path = self.config_dir / 'mass_list.yaml'

        # 优先使用 mass_list.yaml (作为覆盖配置)
        if mass_list_path.exists():
            return self._load_mass_list_from_yaml(mass_list_path)

        # 从 data_file 自动提取
        if data_file:
            # 解析路径（支持相对路径）
            data_path = self._resolve_path(data_file)
            try:
                return parse_masses_from_data_file(data_path)
            except DataExtractorError as e:
                raise ConfigError(str(e))

        # 没有可用的质量数据源
        raise ConfigError(
            "无法获取质量数据: mass_list.yaml 不存在且未指定 data_file"
        )

    def load_lammps_params(self) -> LAMMPSParams:
        """
        加载LAMMPS运行参数

        Returns:
            LAMMPSParams: LAMMPS参数对象
        """
        path = self.config_dir / 'lammps_params.yaml'
        data = self._load_yaml(path)

        # 验证必需字段
        self._validate_required(data, ['simulation', 'steps', 'npt', 'bond_react'], path)

        sim = data['simulation']
        steps = data['steps']
        npt = data['npt']
        br = data['bond_react']

        # 验证simulation字段
        self._validate_required(sim, ['loop_num', 'dt', 'temperature', 'pressure'], path)

        # 验证steps字段
        self._validate_required(steps, ['bond_react_check', 'run_per_loop', 'nve_limit'], path)

        # 验证npt字段
        self._validate_required(npt, ['tcouple', 'pcouple'], path)

        # 验证bond_react字段
        self._validate_required(br, ['stabilization'], path)

        # 验证数值范围
        if sim['loop_num'] <= 0:
            raise ConfigInvalidValueError('loop_num', sim['loop_num'], '必须大于0', path)
        if sim['dt'] <= 0:
            raise ConfigInvalidValueError('dt', sim['dt'], '必须大于0', path)
        if sim['temperature'] <= 0:
            raise ConfigInvalidValueError('temperature', sim['temperature'], '必须大于0', path)

        # 计算run_step和normal_npt_step
        bond_react_check_step = steps['bond_react_check']
        run_step = steps['run_per_loop']
        nve_limit_step = steps['nve_limit']

        if run_step <= nve_limit_step + bond_react_check_step:
            raise ConfigValidationError(
                f"run_per_loop ({run_step}) 必须 > nve_limit ({nve_limit_step}) + bond_react_check ({bond_react_check_step})",
                path
            )

        # 解析reactions - 从 reactions/ 目录统一加载
        reaction_configs = []
        if 'reactions' in br:
            for i, rxn in enumerate(br['reactions']):
                # 只需要验证核心字段，文件路径由 load_reactions_from_directory 自动发现
                self._validate_required(rxn, ['name', 'cutoff', 'pre_mol', 'post_mol'], path)
                reaction_configs.append({
                    'name': rxn['name'],
                    'cutoff': rxn['cutoff'],
                    'pre_mol': rxn['pre_mol'],
                    'post_mol': rxn['post_mol'],
                })

        # 从目录统一加载，获取完整路径
        reactions = load_reactions_from_directory(
            self.config_dir,
            reaction_configs,
            validate_files=True
        )

        # 解析files配置
        files = data.get('files', {})

        # 解析read_data_extra配置
        read_data_extra_raw = files.get('read_data_extra', {})
        read_data_extra = ReadDataExtra(
            special_per_atom=int(read_data_extra_raw.get('special_per_atom', 3)),
            bond_per_atom=int(read_data_extra_raw.get('bond_per_atom', 3)),
            angle_per_atom=int(read_data_extra_raw.get('angle_per_atom', 3)),
            dihedral_per_atom=int(read_data_extra_raw.get('dihedral_per_atom', 3))
        )

        # 解析molecules配置
        molecules = data.get('molecules', {})

        # 解析mass_list配置
        mass_list_raw = data.get('mass_list', {})
        mass_list = {}
        for k, v in mass_list_raw.items():
            try:
                atom_type = int(k)
                mass = float(v)
                mass_list[atom_type] = mass
            except (ValueError, TypeError):
                pass  # 忽略无效条目

        return LAMMPSParams(
            loop_num=int(sim['loop_num']),
            dt=float(sim['dt']),
            timestep=int(sim.get('timestep', 1)),
            temperature=float(sim['temperature']),
            pressure=float(sim['pressure']),
            ensemble=str(sim.get('ensemble', 'npt')).lower(),
            bond_react_check_step=int(bond_react_check_step),
            run_step=int(run_step),
            nve_limit_step=int(nve_limit_step),
            tcouple=float(npt['tcouple']),
            pcouple=float(npt['pcouple']),
            stabilization=float(br['stabilization']),
            reactions=reactions,
            molecules=molecules,
            mass_list=mass_list,
            data_file=files.get('data_file', 'system.data'),
            input_script=files.get('input_script', ''),
            initial_cg_mapping=files.get('initial_cg_mapping', 'AtomId_BeadId_compare_list.csv'),
            output_cg_trajectory=files.get('output_cg_trajectory', 'cg_trajectory.lammpstrj'),
            output_reaction_count=files.get('output_reaction_count', 'reaction_num.txt'),
            output_final_data=files.get('output_final_data', 'final_frame.data'),
            output_final_mapping=files.get('output_final_mapping', 'final_cg_compare_list.csv'),
            output_bonds_record=files.get('output_bonds_record', 'bonds_record.npz'),
            output_cg_bonds=files.get('output_cg_bonds', 'cg_bonds.txt'),
            output_cg_angles=files.get('output_cg_angles', 'cg_angles.txt'),
            output_cg_dihedrals=files.get('output_cg_dihedrals', 'cg_dihedrals.txt'),
            read_data_extra=read_data_extra,
            pair_style=sim.get('pair_style', 'lj/cut')
        )

    def load_mapping_config(self, mapping_path: str) -> MappingConfig:
        """
        加载CG映射配置

        Args:
            mapping_path: mapping文件路径（相对或绝对）

        Returns:
            MappingConfig: CG映射配置对象
        """
        path = self._resolve_path(mapping_path)
        data = self._load_yaml(path)

        self._validate_required(data, ['site-types', 'config'], path)

        # 验证site-types格式
        for site_name, site_data in data['site-types'].items():
            if 'index' not in site_data:
                raise ConfigMissingFieldError(f"site-types.{site_name}.index", path)
            if 'x-weight' not in site_data:
                raise ConfigMissingFieldError(f"site-types.{site_name}.x-weight", path)

        # 验证config格式
        for i, cfg in enumerate(data['config']):
            if 'anchor' not in cfg:
                raise ConfigMissingFieldError(f"config[{i}].anchor", path)
            if 'repeat' not in cfg:
                raise ConfigMissingFieldError(f"config[{i}].repeat", path)
            if 'offset' not in cfg:
                raise ConfigMissingFieldError(f"config[{i}].offset", path)
            if 'sites' not in cfg:
                raise ConfigMissingFieldError(f"config[{i}].sites", path)

        return MappingConfig(
            site_types=data['site-types'],
            config=data['config'],
            bead_type_names=data.get('bead_type_names')
        )

    def load_reaction_config(self, reaction_name: str) -> ReactionConfig:
        """
        加载反应配置

        Args:
            reaction_name: 反应名称（如 "rxn1"）

        Returns:
            ReactionConfig: 反应配置对象
        """
        rxn_dir = self.config_dir / 'reactions' / reaction_name

        if not rxn_dir.exists():
            raise ConfigFileNotFoundError(rxn_dir)

        # 加载映射文件
        map_path = rxn_dir / f"{reaction_name}_map.yaml"
        if map_path.exists():
            map_data = self._load_yaml(map_path)
        else:
            # 兼容原有 .map 格式
            legacy_map_path = rxn_dir / f"{reaction_name}.map"
            if legacy_map_path.exists():
                map_data = self._parse_legacy_map(legacy_map_path)
            else:
                raise ConfigFileNotFoundError(map_path)

        # 验证模板文件存在
        pre_template = rxn_dir / f"{reaction_name}_pre.lammpstemplate"
        post_template = rxn_dir / f"{reaction_name}_post.lammpstemplate"

        if not pre_template.exists():
            raise ConfigFileNotFoundError(pre_template)
        if not post_template.exists():
            raise ConfigFileNotFoundError(post_template)

        # 加载模板CG映射
        pre_mapping_path = rxn_dir / f"{reaction_name}_pre_mapping.yaml"
        post_mapping_path = rxn_dir / f"{reaction_name}_post_mapping.yaml"

        if not pre_mapping_path.exists():
            raise ConfigFileNotFoundError(pre_mapping_path)
        if not post_mapping_path.exists():
            raise ConfigFileNotFoundError(post_mapping_path)

        pre_mapping = self.load_mapping_config(pre_mapping_path)
        post_mapping = self.load_mapping_config(post_mapping_path)

        # 解析equivalences
        equivalences = self._parse_equivalences(map_data)

        # 从reaction配置获取cutoff（如果有的话）
        reaction_info = map_data.get('reaction', {})

        return ReactionConfig(
            name=reaction_info.get('name', reaction_name),
            description=reaction_info.get('description', ''),
            pre_template=str(pre_template),
            post_template=str(post_template),
            pre_mapping=pre_mapping,
            post_mapping=post_mapping,
            equivalences=equivalences,
            edge_atoms=map_data.get('edge_atoms', []),
            initiator_atoms=map_data.get('initiator_atoms', []),
            cutoff=float(reaction_info.get('cutoff', 3.0))
        )

    def load_all_reactions(self) -> List[ReactionConfig]:
        """
        加载所有反应配置

        Returns:
            List[ReactionConfig]: 所有反应配置列表
        """
        reactions_dir = self.config_dir / 'reactions'
        if not reactions_dir.exists():
            return []

        reactions = []
        for rxn_dir in reactions_dir.iterdir():
            if rxn_dir.is_dir():
                reactions.append(self.load_reaction_config(rxn_dir.name))

        return reactions

    # ==================== 辅助解析方法 ====================

    def _parse_legacy_map(self, path: Path) -> Dict:
        """
        解析原有.map格式

        示例格式 (IP_rxn1.map):
        this is a map file

        1 edgeIDs
        27 equivalences

        InitiatorIDs

        4
        15

        EdgeIDs

        13

        Equivalences

        1    1
        2    2
        ...
        """
        with open(path, 'r') as f:
            lines = f.readlines()

        result = {
            'reaction': {'name': path.stem},
            'edge_atoms': [],
            'initiator_atoms': [],
            'equivalences': []
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            if 'edgeIDs' in line.lower():
                # 下一行是edgeIDs数量
                i += 1
                n_edge = int(lines[i].strip())

                # 找到EdgeIDs部分
                i += 1
                while i < len(lines) and 'EdgeIDs' not in lines[i]:
                    i += 1
                i += 1  # 跳过EdgeIDs标题

                for _ in range(n_edge):
                    if i < len(lines):
                        edge_id = int(lines[i].strip())
                        result['edge_atoms'].append(edge_id)
                        i += 1

            elif 'InitiatorIDs' in line:
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('Edge'):
                    try:
                        init_id = int(lines[i].strip())
                        result['initiator_atoms'].append(init_id)
                    except ValueError:
                        pass
                    i += 1

            elif 'Equivalences' in line:
                i += 1
                while i < len(lines):
                    line = lines[i].strip()
                    if not line or line.startswith('#'):
                        i += 1
                        continue
                    try:
                        parts = line.split()
                        if len(parts) >= 2:
                            pre_id = int(parts[0])
                            post_id = int(parts[1])
                            result['equivalences'].append({'pre': pre_id, 'post': post_id})
                    except ValueError:
                        pass
                    i += 1

            i += 1

        return result

    def _parse_equivalences(self, map_data: Dict) -> np.ndarray:
        """解析原子对应关系"""
        equiv_list = map_data.get('equivalences', [])
        if not equiv_list:
            return np.array([]).reshape(0, 2)

        return np.array([[e['pre'], e['post']] for e in equiv_list], dtype=np.int32)


# ============================================================================
# 配置验证器
# ============================================================================

class ConfigValidator:
    """
    完整配置验证器

    功能:
    - 文件一致性检查: 验证所有引用的文件是否存在
    - 配置间一致性检查: 验证不同配置之间的关联是否正确
    - 语义验证: 验证配置值的合理性和边界情况
    - 详细报告: 收集所有验证结果并输出

    使用方式:
        validator = ConfigValidator("config_dir/")
        result = validator.validate_all()
        result.print_report()
    """

    # 语义验证的参数范围
    RANGES = {
        'cutoff': (0.5, 10.0, "反应截断半径 (Å)"),
        'temperature': (1, 10000, "温度 (K)"),
        'dt': (0.0001, 0.01, "时间步长 (ps)"),
        'stabilization': (0.01, 1.0, "稳定化参数 (Å)"),
        'tcouple': (10, 10000, "温度耦合常数 (fs)"),
        'pcouple': (100, 100000, "压力耦合常数 (fs)"),
    }

    def __init__(self, config_dir: str):
        """
        初始化验证器

        Args:
            config_dir: 配置文件目录路径
        """
        self.config_dir = Path(config_dir)
        self.result = ValidationResult(config_dir=self.config_dir)
        self.loader: Optional[ConfigLoader] = None
        self.system_config: Optional[SystemConfig] = None
        self.lammps_params: Optional[LAMMPSParams] = None
        self._mapping_configs: Dict[str, MappingConfig] = {}

    def _resolve_path(self, path: str | Path) -> Path:
        """解析路径（支持相对路径）"""
        path = Path(path)
        if path.is_absolute():
            return path
        return self.config_dir / path

    def _add_error(self, category: str, field: str, message: str,
                   suggestion: str = "", path: Optional[Path] = None):
        """添加错误"""
        self.result.add_error(category, field, message, suggestion, path)

    def _add_warning(self, category: str, field: str, message: str,
                     suggestion: str = "", path: Optional[Path] = None):
        """添加警告"""
        self.result.add_warning(category, field, message, suggestion, path)

    def _add_info(self, category: str, field: str, message: str,
                  suggestion: str = "", path: Optional[Path] = None):
        """添加信息"""
        self.result.add_info(category, field, message, suggestion, path)

    def _add_success(self, category: str, field: str, message: str = "正常"):
        """添加成功信息"""
        self.result.add_success(category, field, message)

    # ==================== 主验证方法 ====================

    def validate_all(self) -> ValidationResult:
        """
        执行完整验证

        Returns:
            ValidationResult: 验证结果对象
        """
        # Phase 1: 基础文件检查
        self._validate_required_config_files()
        if self.result.n_errors > 0:
            return self.result  # 基础文件缺失无法继续

        # Phase 2: 初始化加载器并加载配置
        self.loader = ConfigLoader(str(self.config_dir))
        try:
            self.system_config = self.loader.load_system_config()
            self.lammps_params = self.loader.load_lammps_params()
        except ConfigError as e:
            self._add_error("config", "配置加载", f"无法加载配置: {e}")
            return self.result

        # Phase 3: 文件一致性验证
        self._validate_mapping_files()
        self._validate_data_file()
        self._validate_reaction_files()

        # Phase 4: 配置间一致性验证
        self._validate_molecules_reactions_consistency()
        self._validate_bead_type_coverage()
        self._validate_mass_coverage()

        # Phase 5: 语义验证
        self._validate_simulation_params()
        self._validate_step_params()
        self._validate_npt_params()
        self._validate_reaction_params()

        return self.result

    # ==================== Phase 1: 基础文件检查 ====================

    def _validate_required_config_files(self):
        """验证必需配置文件存在"""
        required = ['system.yaml', 'lammps_params.yaml']

        for f in required:
            path = self.config_dir / f
            if path.exists():
                self._add_success("file", f, "文件存在")
            else:
                self._add_error("file", f, "文件未找到",
                               f"在 {self.config_dir} 目录中创建 {f}",
                               path)

    # ==================== Phase 3: 文件一致性验证 ====================

    def _validate_mapping_files(self):
        """验证 mapping_files 中引用的文件"""
        if not self.system_config:
            return

        for i, mf in enumerate(self.system_config.mapping_files):
            path_str = mf.get('path')

            # 检查 path 字段存在且非空
            if not path_str:
                self._add_error("file", f"mapping_files[{i}]",
                               "缺少 'path' 字段或路径为空",
                               "在 system.yaml 的 mapping_files 中添加有效的 path")
                continue

            path = self._resolve_path(path_str)

            if path.exists():
                self._add_success("file", path_str, "映射文件存在")
                # 尝试加载并验证
                try:
                    mapping_config = self.loader.load_mapping_config(path_str)
                    self._mapping_configs[path_str] = mapping_config
                except ConfigError as e:
                    self._add_error("file", path_str, f"映射文件解析失败: {e}",
                                   "检查 YAML 格式和必需字段", path)
            else:
                self._add_error("file", path_str, "映射文件未找到",
                               f"创建映射配置文件 {path_str}", path)

    def _validate_data_file(self):
        """验证 LAMMPS 数据文件"""
        if not self.lammps_params:
            return

        data_file = self.lammps_params.data_file
        path = self._resolve_path(data_file)

        if path.exists():
            self._add_success("file", data_file, "LAMMPS 数据文件存在")
        else:
            self._add_error("file", data_file, "LAMMPS 数据文件未找到",
                           f"创建 LAMMPS 数据文件 {data_file}", path)

    def _validate_reaction_files(self):
        """验证反应相关文件 - 使用完整路径验证"""
        if not self.lammps_params:
            return

        # 1. 检查 reactions 目录存在
        reactions_dir = self.config_dir / "reactions"
        if not reactions_dir.exists():
            self._add_error("file", "reactions/", "reactions 目录不存在",
                           f"在 {self.config_dir} 下创建 reactions/ 目录", reactions_dir)
            return

        self._add_success("file", "reactions/", "reactions 目录存在")

        reactions = self.lammps_params.reactions
        if not reactions:
            self._add_info("file", "reactions", "未配置反应")
            return

        for rxn in reactions:
            # 2. 检查子目录存在
            rxn_dir = Path(rxn.rxn_dir) if rxn.rxn_dir else reactions_dir / rxn.name
            if rxn_dir.exists():
                self._add_success("file", f"reactions/{rxn.name}/", f"{rxn.name}: 反应子目录存在")
            else:
                self._add_error("file", f"reactions/{rxn.name}/", f"{rxn.name}: 反应子目录不存在",
                               f"创建反应子目录 reactions/{rxn.name}/", rxn_dir)
                continue

            # 3. 检查必需文件 - 使用完整路径
            # map_file (完整路径)
            map_path = Path(rxn.map_file) if rxn.map_file else rxn_dir / f"{rxn.name}.map"
            if map_path.exists():
                self._add_success("file", f"reactions/{rxn.name}/{rxn.name}.map", f"{rxn.name}: map 文件存在")
            else:
                self._add_error("file", f"reactions/{rxn.name}/{rxn.name}.map",
                               f"{rxn.name}: map 文件未找到",
                               f"创建 bond/react map 文件", map_path)

            # pre_template (完整路径)
            pre_path = Path(rxn.pre_template) if rxn.pre_template else rxn_dir / f"{rxn.name}_pre.lammpstemplate"
            if pre_path.exists():
                self._add_success("file", f"reactions/{rxn.name}/{rxn.name}_pre.lammpstemplate",
                                 f"{rxn.name}: 反应前模板存在")
            else:
                self._add_error("file", f"reactions/{rxn.name}/{rxn.name}_pre.lammpstemplate",
                               f"{rxn.name}: 反应前模板未找到",
                               f"创建模板文件", pre_path)

            # post_template (完整路径)
            post_path = Path(rxn.post_template) if rxn.post_template else rxn_dir / f"{rxn.name}_post.lammpstemplate"
            if post_path.exists():
                self._add_success("file", f"reactions/{rxn.name}/{rxn.name}_post.lammpstemplate",
                                 f"{rxn.name}: 反应后模板存在")
            else:
                self._add_error("file", f"reactions/{rxn.name}/{rxn.name}_post.lammpstemplate",
                               f"{rxn.name}: 反应后模板未找到",
                               f"创建模板文件", post_path)

            # pre_mapping (可选，推荐)
            pre_map_path = Path(rxn.pre_mapping) if rxn.pre_mapping else rxn_dir / f"{rxn.name}_pre_mapping.yaml"
            if rxn.pre_mapping or pre_map_path.exists():
                if pre_map_path.exists():
                    self._add_success("file", f"reactions/{rxn.name}/{rxn.name}_pre_mapping.yaml",
                                     f"{rxn.name}: 反应前CG映射存在")
                else:
                    self._add_warning("file", f"reactions/{rxn.name}/{rxn.name}_pre_mapping.yaml",
                                     f"{rxn.name}: 反应前CG映射未找到",
                                     f"创建 CG 映射文件 (可选)", pre_map_path)

            # post_mapping (可选，推荐)
            post_map_path = Path(rxn.post_mapping) if rxn.post_mapping else rxn_dir / f"{rxn.name}_post_mapping.yaml"
            if rxn.post_mapping or post_map_path.exists():
                if post_map_path.exists():
                    self._add_success("file", f"reactions/{rxn.name}/{rxn.name}_post_mapping.yaml",
                                     f"{rxn.name}: 反应后CG映射存在")
                else:
                    self._add_warning("file", f"reactions/{rxn.name}/{rxn.name}_post_mapping.yaml",
                                     f"{rxn.name}: 反应后CG映射未找到",
                                     f"创建 CG 映射文件 (可选)", post_map_path)

    # ==================== Phase 4: 配置间一致性验证 ====================

    def _validate_molecules_reactions_consistency(self):
        """验证 reactions 配置中的模板文件完整性"""
        if not self.lammps_params:
            return

        reactions = self.lammps_params.reactions

        if not reactions:
            self._add_success("config", "molecules-reactions", "无反应配置")
            return

        # 直接验证 ReactionInfo 中存储的完整路径
        all_valid = True
        for rxn in reactions:
            # 检查 pre_template
            if rxn.pre_template:
                pre_path = Path(rxn.pre_template)
                if pre_path.exists():
                    self._add_success("config", f"{rxn.name}.pre_template",
                                     f"{rxn.name}: 反应前模板已验证")
                else:
                    self._add_error("config", f"{rxn.name}.pre_template",
                                   f"{rxn.name}: 反应前模板文件不存在",
                                   f"确保 reactions/{rxn.name}/{rxn.name}_pre.lammpstemplate 存在")
                    all_valid = False

            # 检查 post_template
            if rxn.post_template:
                post_path = Path(rxn.post_template)
                if post_path.exists():
                    self._add_success("config", f"{rxn.name}.post_template",
                                     f"{rxn.name}: 反应后模板已验证")
                else:
                    self._add_error("config", f"{rxn.name}.post_template",
                                   f"{rxn.name}: 反应后模板文件不存在",
                                   f"确保 reactions/{rxn.name}/{rxn.name}_post.lammpstemplate 存在")
                    all_valid = False

            # 检查 map_file
            if rxn.map_file:
                map_path = Path(rxn.map_file)
                if map_path.exists():
                    self._add_success("config", f"{rxn.name}.map_file",
                                     f"{rxn.name}: map 文件已验证")
                else:
                    self._add_error("config", f"{rxn.name}.map_file",
                                   f"{rxn.name}: map 文件不存在",
                                   f"确保 reactions/{rxn.name}/{rxn.name}.map 存在")
                    all_valid = False

        if all_valid:
            self._add_success("config", "molecules-reactions", "所有反应模板已正确验证")
        else:
            self._add_error("config", "molecules-reactions", "部分反应模板文件缺失",
                           "检查 reactions/ 目录下每个反应子目录的必需文件")

    def _validate_bead_type_coverage(self):
        """验证 bead_type_names 覆盖所有 site_types 中定义的类型"""
        if not self.system_config:
            return

        bead_type_names = self.system_config.bead_type_names

        # 收集所有 mapping 文件中定义的 site_types
        all_site_types = set()
        for path_str, mapping_config in self._mapping_configs.items():
            all_site_types.update(mapping_config.site_types.keys())

        # 检查覆盖情况
        missing_types = []
        for site_type in all_site_types:
            if site_type in bead_type_names:
                self._add_success("config", f"bead_type_names.{site_type}",
                                 f"site 类型 '{site_type}' 已映射到 {bead_type_names[site_type]}")
            else:
                missing_types.append(site_type)

        if missing_types:
            self._add_error("config", "bead_type_names",
                           f"未定义 site 类型: {', '.join(missing_types)}",
                           f"在 system.yaml 的 bead_type_names 中添加这些类型定义")
        else:
            self._add_success("config", "bead_type_names", "覆盖所有 site 类型")

        # 检查 bead_type ID 是否连续且唯一
        type_ids = list(bead_type_names.values())
        if len(type_ids) != len(set(type_ids)):
            self._add_warning("config", "bead_type_names",
                             "存在重复的 bead_type ID",
                             "确保每个 bead_type ID 唯一")

    def _validate_mass_coverage(self):
        """验证 mass_list 是否覆盖所有原子类型"""
        if not self.lammps_params:
            return

        mass_list = self.lammps_params.mass_list

        if not mass_list:
            self._add_warning("config", "mass_list",
                             "mass_list 为空，将从 data_file 自动提取",
                             "可以在 lammps_params.yaml 或 mass_list.yaml 中手动指定")
            return

        self._add_success("config", "mass_list",
                         f"已定义 {len(mass_list)} 种原子类型的质量")

    # ==================== Phase 5: 语义验证 ====================

    def _check_range(self, value: float, name: str, category: str = "semantic"):
        """检查数值是否在合理范围内"""
        min_val, max_val, desc = self.RANGES[name]

        if value < min_val:
            self._add_warning(category, name,
                             f"{desc} 值 {value} 偏小 (建议范围: {min_val}-{max_val})",
                             f"增大 {name} 参数值")
        elif value > max_val:
            self._add_warning(category, name,
                             f"{desc} 值 {value} 偏大 (建议范围: {min_val}-{max_val})",
                             f"减小 {name} 参数值")
        else:
            self._add_success(category, name, f"{desc} = {value} (正常范围)")

    def _validate_simulation_params(self):
        """验证模拟参数的合理性"""
        if not self.lammps_params:
            return

        params = self.lammps_params

        # 检查 loop_num
        if params.loop_num <= 0:
            self._add_error("semantic", "loop_num", f"循环次数 {params.loop_num} 必须大于0")
        elif params.loop_num > 100000:
            self._add_info("semantic", "loop_num",
                          f"循环次数 {params.loop_num} 较大，运行时间可能很长",
                          "根据实际需求调整")
        else:
            self._add_success("semantic", "loop_num", f"循环次数 = {params.loop_num}")

        # 检查 temperature
        self._check_range(params.temperature, 'temperature')

        # 检查 dt
        self._check_range(params.dt, 'dt')

        # 检查 ensemble
        if params.ensemble not in ['npt', 'nvt']:
            self._add_warning("semantic", "ensemble",
                             f"系综类型 '{params.ensemble}' 不是标准选项",
                             "使用 'npt' 或 'nvt'")
        else:
            self._add_success("semantic", "ensemble", f"系综 = {params.ensemble}")

    def _validate_step_params(self):
        """验证步数参数的合理性"""
        if not self.lammps_params:
            return

        params = self.lammps_params

        # 检查 run_step > nve_limit + bond_react_check
        bond_check = params.bond_react_check_step
        run_step = params.run_step
        nve_limit = params.nve_limit_step

        if run_step <= nve_limit + bond_check:
            self._add_error("semantic", "run_per_loop",
                           f"run_per_loop ({run_step}) 必须 > nve_limit ({nve_limit}) + bond_react_check ({bond_check})",
                           f"增大 run_per_loop 或减小 nve_limit")
        else:
            normal_npt = run_step - nve_limit - bond_check
            if normal_npt < 10:
                self._add_warning("semantic", "run_per_loop",
                                 f"正常 NPT 步数 ({normal_npt}) 较少，可能影响平衡",
                                 f"增大 run_per_loop 或减小 nve_limit")
            else:
                self._add_success("semantic", "run_per_loop",
                                 f"run_step={run_step}, nve_limit={nve_limit}, normal_npt={normal_npt}")

    def _validate_npt_params(self):
        """验证 NPT 参数的合理性"""
        if not self.lammps_params:
            return

        params = self.lammps_params

        # 检查耦合常数
        self._check_range(params.tcouple, 'tcouple')
        self._check_range(params.pcouple, 'pcouple')

    def _validate_reaction_params(self):
        """验证反应参数的合理性"""
        if not self.lammps_params:
            return

        params = self.lammps_params

        # 检查 stabilization
        self._check_range(params.stabilization, 'stabilization')

        # 检查每个反应的 cutoff
        for rxn in params.reactions:
            self._check_range(rxn.cutoff, 'cutoff', "semantic")


# ============================================================================
# 便捷函数
# ============================================================================

def load_all_configs(config_dir: str) -> tuple:
    """
    一次性加载所有配置

    Args:
        config_dir: 配置目录路径

    Returns:
        (SystemConfig, LAMMPSParams, List[ReactionConfig])
    """
    loader = ConfigLoader(config_dir)

    system_config = loader.load_system_config()
    lammps_params = loader.load_lammps_params()
    reactions = loader.load_all_reactions()

    return system_config, lammps_params, reactions


def validate_config(config_dir: str, print_report: bool = True) -> ValidationResult:
    """
    验证配置目录的完整性

    Args:
        config_dir: 配置目录路径
        print_report: 是否打印报告

    Returns:
        ValidationResult: 验证结果对象
    """
    validator = ConfigValidator(config_dir)
    result = validator.validate_all()

    if print_report:
        result.print_report()

    return result


if __name__ == "__main__":
    # 测试配置加载器
    import sys

    if len(sys.argv) > 1:
        config_dir = sys.argv[1]
    else:
        config_dir = "config"

    try:
        loader = ConfigLoader(config_dir)

        print("=" * 60)
        print("加载体系配置...")
        system_config = loader.load_system_config()
        print(f"  名称: {system_config.name}")
        print(f"  Mapping文件数: {len(system_config.mapping_files)}")
        print(f"  原子类型数: {len(system_config.mass_list)}")

        print("\n加载LAMMPS参数...")
        params = loader.load_lammps_params()
        print(f"  循环次数: {params.loop_num}")
        print(f"  温度: {params.temperature} K")
        print(f"  反应数: {len(params.reactions)}")

        print("\n加载反应配置...")
        reactions = loader.load_all_reactions()
        for rxn in reactions:
            print(f"  - {rxn.name}: {rxn.description}")

        print("\n✅ 配置加载成功!")

    except ConfigError as e:
        print(f"\n❌ 配置错误: {e}")
        sys.exit(1)