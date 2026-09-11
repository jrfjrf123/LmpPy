"""
IBM势能计算模块

通过IBM (Inverse Boltzmann Method) 从CG轨迹计算势能表。

工作流程:
1. 加载CG轨迹和拓扑
2. 计算分布 (bond/angle/dihedral/RDF)
3. 玻尔兹曼反演得到势能
4. 平滑和外推
5. 生成LAMMPS table文件

使用示例:
    from LmpPy.tools.ibm_potential import (
        load_ibm_config,
        load_cg_trajectory,
        calculate_bond_distribution,
        calculate_bond_potential,
        create_lammps_table_files
    )

    # 加载配置
    config = load_ibm_config('ibm_potential.yaml')

    # 加载轨迹
    cg_data = load_cg_trajectory(config.trajectory.path)

    # 计算分布和势能
    r, hist = calculate_bond_distribution(cg_data, bond_pairs)
    r, U = calculate_bond_potential(r, hist, config.simulation.temperature)

    # 生成table
    create_lammps_table_files('potentials_output', 'lammps_tables')
"""

# 配置
from .config import (
    IBMConfig,
    IBMConfigLoader,
    load_ibm_config
)

# 分布计算
from .distribution import (
    load_cg_trajectory,
    load_topology,
    load_type_dict,
    calculate_bond_distribution,
    calculate_angle_distribution,
    calculate_dihedral_distribution,
    calculate_rdf,
    calculate_all_distributions,
    build_exclusion_pairs
)

# 分布配置
from .dist_config import (
    DistributionConfig,
    DEFAULT_CONFIG
)

# 玻尔兹曼反演
from .boltzmann import (
    KB,
    calculate_bond_potential,
    calculate_angle_potential,
    calculate_dihedral_potential,
    calculate_pair_potential,
    extrapolate_and_smooth,
    calculate_force,
    save_potential
)

# LAMMPS表生成
from .lammps_table import (
    read_potential_file,
    create_lammps_table_files,
    create_reference_file
)

# Tabulated 势能表读取/绘图/拟合
from .tabulated_potential import (
    TabulatedPotential,
    FitResult,
    read_tabulated_table,
    detect_kind,
    extract_type_ids,
    detect_angle_unit,
    discover_tables,
    lj_12_6,
    harmonic_bond,
    cos_angle,
    harmonic_angle,
    fit_lj,
    fit_bond_harmonic,
    fit_angle,
    fit_table,
    AxisLimits,
    PlotLimits,
    plot_single_table,
    plot_all_tables,
    plot_fit_comparison,
)

# 可选：绘图模块
try:
    from .dist_plot import (
        plot_single_distribution,
        plot_all_distributions,
        plot_distribution_results
    )
except ImportError:
    pass

# 可选：GROMACS 加载模块
try:
    from .gromacs_loader import (
        TrajectoryCache,
        TopologyIndex,
        create_exclusion_mask,
        calculate_bond_distribution_vectorized,
        calculate_angle_distribution_vectorized,
        calculate_dihedral_distribution_vectorized,
        calculate_rdf_vectorized,
        calculate_all_distributions_from_gromacs
    )
    HAS_GROMACS_SUPPORT = True
except ImportError:
    HAS_GROMACS_SUPPORT = False

__all__ = [
    # 配置
    'IBMConfig',
    'IBMConfigLoader',
    'load_ibm_config',
    'DistributionConfig',
    'DEFAULT_CONFIG',

    # 分布
    'load_cg_trajectory',
    'load_topology',
    'load_type_dict',
    'calculate_bond_distribution',
    'calculate_angle_distribution',
    'calculate_dihedral_distribution',
    'calculate_rdf',
    'calculate_all_distributions',
    'build_exclusion_pairs',

    # 玻尔兹曼反演
    'KB',
    'calculate_bond_potential',
    'calculate_angle_potential',
    'calculate_dihedral_potential',
    'calculate_pair_potential',
    'extrapolate_and_smooth',
    'calculate_force',
    'save_potential',

    # LAMMPS表
    'read_potential_file',
    'create_lammps_table_files',
    'create_reference_file',

    # Tabulated 势能表
    'TabulatedPotential',
    'FitResult',
    'read_tabulated_table',
    'detect_kind',
    'extract_type_ids',
    'detect_angle_unit',
    'discover_tables',
    'lj_12_6',
    'harmonic_bond',
    'cos_angle',
    'harmonic_angle',
    'fit_lj',
    'fit_bond_harmonic',
    'fit_angle',
    'fit_table',
    'AxisLimits',
    'PlotLimits',
    'plot_single_table',
    'plot_all_tables',
    'plot_fit_comparison',

    # 绘图（可选）
    'plot_single_distribution',
    'plot_all_distributions',
    'plot_distribution_results',

    # GROMACS 支持（可选）
    'TrajectoryCache',
    'TopologyIndex',
    'create_exclusion_mask',
    'calculate_bond_distribution_vectorized',
    'calculate_angle_distribution_vectorized',
    'calculate_dihedral_distribution_vectorized',
    'calculate_rdf_vectorized',
    'calculate_all_distributions_from_gromacs',
    'HAS_GROMACS_SUPPORT',
]