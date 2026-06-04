"""
Core模块

包含配置加载、映射生成、模板解析、数据提取、键检测、反应定位、CG映射等核心功能
"""

from .config_loader import (
    ConfigLoader,
    SystemConfig,
    LAMMPSParams,
    ReactionInfo,
    ReadDataExtra,
    ReactionConfig,
    MappingConfig,
    ConfigError,
    ValidationIssue,
    ValidationResult,
    ConfigValidator,
    load_all_configs,
    validate_config,
    load_reactions_from_directory,
)

from .mapping_generator import (
    MappingGenerator,
    CGCompareList,
    generate_cg_compare_list
)

from .template_parser import (
    TemplateParser,
    TemplateData,
    ReactionMapData,
    ReactionTemplate,
    TemplateCGMapping,
    load_all_reaction_templates
)

from .lammps_data_extractor import (
    LAMMPSDataExtractor,
    AtomData,
    BondData,
    decode_image_flags_vectorized,
    get_atoms_bonds_info,
    get_lmp_box_info,
    parse_masses_from_data_file,
    parse_bonds_from_data_file,
    DataExtractorError
)

from .bond_detector import (
    BondDetector,
    BondChanges,
    compare_two_bonds,
    get_changed_atoms
)

from .reaction_locator import (
    ReactionLocator,
    ReactionMatch,
    locate_reactions
)

from .cg_mapper import (
    CGMapper,
    CGMapping,
    update_cg_mapping
)

from .cg_converter import (
    CGConverter,
    CGMappingValidationError,
    validate_cg_mapping_consistency,
    lammpstrj2cg
)

from .cg_bond_mapper import (
    CGBondMapper,
    CGBond,
    atom_bonds_to_cg_bonds
)

from .cg_topology import (
    CGTopology,
    derive_angles_from_bonds,
    derive_dihedrals_from_bonds,
    create_cg_bead_info,
    verify_molecule_ids_consistency,
    derive_cg_topology_from_bonds
)

from .bonds_recorder import (
    BondsRecorder,
    BondRecord,
    CGTopologyRecord,
    save_bonds_record,
    load_bonds_record
)

from .cg_initializer import (
    CGInitializer,
    CGSystem,
    initialize_cg_system
)

__all__ = [
    # config_loader
    'ConfigLoader',
    'SystemConfig',
    'LAMMPSParams',
    'ReactionInfo',
    'ReadDataExtra',
    'ReactionConfig',
    'MappingConfig',
    'ConfigError',
    'ValidationIssue',
    'ValidationResult',
    'ConfigValidator',
    'load_all_configs',
    'validate_config',
    'load_reactions_from_directory',
    # mapping_generator
    'MappingGenerator',
    'CGCompareList',
    'generate_cg_compare_list',
    # template_parser
    'TemplateParser',
    'TemplateData',
    'ReactionMapData',
    'ReactionTemplate',
    'TemplateCGMapping',
    'load_all_reaction_templates',
    # lammps_data_extractor
    'LAMMPSDataExtractor',
    'AtomData',
    'BondData',
    'decode_image_flags_vectorized',
    'get_atoms_bonds_info',
    'get_lmp_box_info',
    'parse_masses_from_data_file',
    'parse_bonds_from_data_file',
    'DataExtractorError',
    # bond_detector
    'BondDetector',
    'BondChanges',
    'compare_two_bonds',
    'get_changed_atoms',
    # reaction_locator
    'ReactionLocator',
    'ReactionMatch',
    'locate_reactions',
    # cg_mapper
    'CGMapper',
    'CGMapping',
    'update_cg_mapping',
    # cg_converter
    'CGConverter',
    'CGMappingValidationError',
    'validate_cg_mapping_consistency',
    'lammpstrj2cg',
    # cg_bond_mapper
    'CGBondMapper',
    'CGBond',
    'atom_bonds_to_cg_bonds',
    # cg_topology
    'CGTopology',
    'derive_angles_from_bonds',
    'derive_dihedrals_from_bonds',
    'create_cg_bead_info',
    'verify_molecule_ids_consistency',
    'derive_cg_topology_from_bonds',
    # bonds_recorder
    'BondsRecorder',
    'BondRecord',
    'CGTopologyRecord',
    'save_bonds_record',
    'load_bonds_record',
    # cg_initializer
    'CGInitializer',
    'CGSystem',
    'initialize_cg_system',
]