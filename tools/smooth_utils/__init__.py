"""
Smooth Utils - 分布平滑工具包

用于IBM势能计算中的分布曲线平滑处理。

功能:
- bond/angle/dihedral/RDF分布平滑
- 谐波边界处理
- 峰值保护
- 质量评估
- 自动参数优化

使用方法:
    from LmpPy.tools.smooth_utils import (
        smooth_distribution,
        smooth_bond_with_harmonic_boundary,
        smooth_angle_with_harmonic_boundary,
        smooth_dihedral_periodic,
        ...
    )

作者: 原始代码来自 md_base_on_ml/calc_ibm_pot/dist_smooth
整合时间: 2026-04-02
"""

# Constants
from .constants import (
    DEFAULT_THRESHOLD,
    DEFAULT_CDF_PERCENT,
    MAX_PEAKS,
    BOUNDARY_RATIO,
    EPSILON,
    BOUNDARY_DECAY_LENGTH,
    WEIGHT_PEAK_POS,
    WEIGHT_FORCE_SMOOTH,
    WEIGHT_PEAK_HEIGHT,
    KB,
    DEFAULT_TEMPERATURE,
    FORCE_CONSISTENCY_THRESHOLD,
    DEFAULT_POLYORDER,
    DEFAULT_PEAK_WIDTH_FACTOR,
    DEFAULT_WINDOW_SMALL_FACTOR,
    DEFAULT_WINDOW_LARGE_FACTOR,
    PARAM_GRID,
    USE_BAYESIAN_OPTIMIZATION,
    BAYESIAN_N_CALLS
)

# I/O functions
from .io import (
    load_distribution,
    get_adaptive_window_base
)

# Preprocessing functions
from .preprocess import (
    find_valid_range,
    interpolate_zeros
)

# Peak detection functions
from .peaks import (
    detect_peaks,
    estimate_peak_width,
    detect_significant_peaks
)

# Zone definition functions
from .zones import (
    define_zones,
    should_use_gaussian
)

# Core smoothing functions
from .core import (
    adaptive_smooth,
    gaussian_smooth_with_constraint,
    gaussian_smooth_for_rdf,
    smooth_bond_with_harmonic_boundary,
    smooth_rdf_with_harmonic_left_boundary,
    apply_rdf_boundary,
    smooth_zone_transitions
)

# Angle and Dihedral smoothing functions
from .angle_dihedral import (
    smooth_angle_with_harmonic_boundary,
    smooth_dihedral_periodic
)

# Quality assessment functions
from .quality import (
    calculate_potential_and_force,
    calculate_quality_metrics,
    check_force_consistency,
    match_peaks
)

# Optimization functions
from .optimize import (
    grid_search_optimize,
    bayesian_optimize,
    auto_optimize
)

# Report generation functions
from .report import (
    generate_comparison_plot,
    generate_text_report,
    generate_report
)

# Main CLI function
from .cli import (
    smooth_distribution,
    smooth_all_distributions
)

__all__ = [
    # Constants
    'DEFAULT_THRESHOLD',
    'DEFAULT_CDF_PERCENT',
    'MAX_PEAKS',
    'BOUNDARY_RATIO',
    'EPSILON',
    'BOUNDARY_DECAY_LENGTH',
    'WEIGHT_PEAK_POS',
    'WEIGHT_FORCE_SMOOTH',
    'WEIGHT_PEAK_HEIGHT',
    'KB',
    'DEFAULT_TEMPERATURE',
    'FORCE_CONSISTENCY_THRESHOLD',
    'DEFAULT_POLYORDER',
    'DEFAULT_PEAK_WIDTH_FACTOR',
    'DEFAULT_WINDOW_SMALL_FACTOR',
    'DEFAULT_WINDOW_LARGE_FACTOR',
    'PARAM_GRID',
    'USE_BAYESIAN_OPTIMIZATION',
    'BAYESIAN_N_CALLS',

    # I/O
    'load_distribution',
    'get_adaptive_window_base',

    # Preprocessing
    'find_valid_range',
    'interpolate_zeros',

    # Peak detection
    'detect_peaks',
    'estimate_peak_width',
    'detect_significant_peaks',

    # Zone definition
    'define_zones',
    'should_use_gaussian',

    # Core smoothing
    'adaptive_smooth',
    'gaussian_smooth_with_constraint',
    'gaussian_smooth_for_rdf',
    'smooth_bond_with_harmonic_boundary',
    'smooth_rdf_with_harmonic_left_boundary',
    'apply_rdf_boundary',
    'smooth_zone_transitions',

    # Angle/Dihedral smoothing
    'smooth_angle_with_harmonic_boundary',
    'smooth_dihedral_periodic',

    # Quality assessment
    'calculate_potential_and_force',
    'calculate_quality_metrics',
    'check_force_consistency',
    'match_peaks',

    # Optimization
    'grid_search_optimize',
    'bayesian_optimize',
    'auto_optimize',

    # Report generation
    'generate_comparison_plot',
    'generate_text_report',
    'generate_report',

    # Main functions
    'smooth_distribution',
    'smooth_all_distributions'
]