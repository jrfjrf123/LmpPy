"""
LAMMPS Bond/React 后处理框架

功能模块:
- core: 核心功能 (配置加载、CG映射、数据提取、反应处理)
- utils: 工具函数
"""

from . import core
from . import utils

__version__ = "0.1.0"
__all__ = ["core", "utils"]