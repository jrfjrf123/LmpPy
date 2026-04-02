"""
LmpPy Tools - 工具模块集合

包含:
- aa2cg: 全原子→粗粒化转换工具
- ibm_potential: IBM势能计算流程
- smooth_utils: 分布平滑工具包
"""

# 导入子模块
from . import smooth_utils
from . import aa2cg
from . import ibm_potential

# 从 smooth_utils 导出主要函数
from .smooth_utils import (
    smooth_distribution,
    smooth_all_distributions,
    DEFAULT_TEMPERATURE,
)

# 从 aa2cg 导出主要函数
from .aa2cg import (
    load_aa_to_cg_mapping,
    convert_aa_to_cg_frame,
    read_lammps_data,
    write_cg_data_file,
    convert_data_to_cg,
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
    load_cg_trajectory_pickle,
    write_trajectory_to_xyz,
)

# 从 ibm_potential 导出主要函数
from .ibm_potential import (
    load_ibm_config,
    load_cg_trajectory,
    load_topology,
    load_type_dict,
    calculate_bond_distribution,
    calculate_angle_distribution,
    calculate_dihedral_distribution,
    calculate_rdf,
    calculate_bond_potential,
    calculate_angle_potential,
    calculate_dihedral_potential,
    calculate_pair_potential,
    extrapolate_and_smooth,
    save_potential,
    create_lammps_table_files,
    create_reference_file,
)

__all__ = [
    # 子模块
    'smooth_utils',
    'aa2cg',
    'ibm_potential',
    # smooth_utils 函数
    'smooth_distribution',
    'smooth_all_distributions',
    'DEFAULT_TEMPERATURE',
    # aa2cg 函数
    'load_aa_to_cg_mapping',
    'convert_aa_to_cg_frame',
    'read_lammps_data',
    'write_cg_data_file',
    'convert_data_to_cg',
    'read_gromacs_trr_all_frames',
    'convert_trajectory_to_cg',
    'save_cg_trajectory_pickle',
    'load_cg_trajectory_pickle',
    'write_trajectory_to_xyz',
    # ibm_potential 函数
    'load_ibm_config',
    'load_cg_trajectory',
    'load_topology',
    'load_type_dict',
    'calculate_bond_distribution',
    'calculate_angle_distribution',
    'calculate_dihedral_distribution',
    'calculate_rdf',
    'calculate_bond_potential',
    'calculate_angle_potential',
    'calculate_dihedral_potential',
    'calculate_pair_potential',
    'extrapolate_and_smooth',
    'save_potential',
    'create_lammps_table_files',
    'create_reference_file',
]