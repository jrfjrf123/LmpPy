"""
Constants and default parameters for distribution smoothing.
分布平滑模块的常量和默认参数定义
"""

# Threshold for zero value detection
DEFAULT_THRESHOLD = 1e-4

# Cumulative distribution percentage for valid range
DEFAULT_CDF_PERCENT = 0.99

# Maximum number of peaks to detect and protect
MAX_PEAKS = 6

# Boundary constraint ratio for RDF (start applying constraint at this fraction of r_max)
BOUNDARY_RATIO = 0.8

# Epsilon for RDF boundary constraint (g(r) → 1 - epsilon, approaching from below)
EPSILON = 0.001

# Decay length for RDF boundary constraint (Angstrom)
BOUNDARY_DECAY_LENGTH = 2.0

# Quality metric weights
WEIGHT_PEAK_POS = 0.5      # Peak position fidelity
WEIGHT_FORCE_SMOOTH = 0.35  # Force curve smoothness
WEIGHT_PEAK_HEIGHT = 0.15   # Peak height fidelity

# Physical constants
KB = 0.0019872041  # Boltzmann constant in kcal/mol/K
DEFAULT_TEMPERATURE = 400  # Temperature in K

# Force consistency check threshold
FORCE_CONSISTENCY_THRESHOLD = 1.0

# Smoothing parameter defaults
DEFAULT_POLYORDER = 3
DEFAULT_PEAK_WIDTH_FACTOR = 1.0
DEFAULT_WINDOW_SMALL_FACTOR = 0.9  # Multiplied by base window for peak regions
DEFAULT_WINDOW_LARGE_FACTOR = 3.1  # Multiplied by base window for valley regions

# Parameter search space for auto-optimization
PARAM_GRID = {
    'window_small_factors': [0.5, 0.7, 0.9, 1.1, 1.3, 1.5],  # Multiplied by base window
    'window_large_factors': [1.5, 2.1, 3.1, 4.1, 5.1],
    'polyorder': [2, 3, 4],
    'peak_width_factor': [0.5, 1.0, 1.5, 2.0]
}

# Bayesian optimization settings
USE_BAYESIAN_OPTIMIZATION = True
BAYESIAN_N_CALLS = 30  # Number of Bayesian optimization iterations