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
    """单个反应信息（用于LAMMPS bond/react）"""
    name: str
    cutoff: float
    map_file: str
    pre_mol: str
    post_mol: str
    pre_template: str = ""
    post_template: str = ""
    pre_mapping: str = ""
    post_mapping: str = ""


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

        # 获取 data_file 路径 (用于自动提取 mass_list)
        # 先从 lammps_params.yaml 读取 files.data_file
        data_file = self._get_data_file_from_params()

        # 加载质量列表 (优先使用 mass_list.yaml，否则从 data_file 自动提取)
        mass_list = self.load_mass_list(data_file)

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
            bead_type_names=bead_type_names
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

        # 解析reactions
        reactions = []
        if 'reactions' in br:
            for i, rxn in enumerate(br['reactions']):
                self._validate_required(rxn, ['name', 'cutoff', 'map_file', 'pre_mol', 'post_mol'], path)
                reactions.append(ReactionInfo(
                    name=rxn['name'],
                    cutoff=float(rxn['cutoff']),
                    map_file=rxn['map_file'],
                    pre_mol=rxn['pre_mol'],
                    post_mol=rxn['post_mol'],
                    pre_template=rxn.get('pre_template', ''),
                    post_template=rxn.get('post_template', ''),
                    pre_mapping=rxn.get('pre_mapping', ''),
                    post_mapping=rxn.get('post_mapping', '')
                ))

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
            read_data_extra=read_data_extra
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