"""
工具函数模块

包含坐标处理、文件I/O、图论算法等工具函数
"""

from .coordinate_utils import (
    wrap_coordinates,
    pbc_distance,
    unwrap_molecule_bfs,
    unwrap_coords_numba,
    unwrap_coords_python,
    calculate_central_mass
)

from .file_utils import (
    write_lammps_dump_file,
    write_cg_trajectory,
    read_lammps_dump_file
)

from .graph_utils import (
    find_molecules,
    build_bond_graph,
    get_molecule_sizes,
    get_molecule_atoms,
    NUMBA_AVAILABLE
)

__all__ = [
    # coordinate_utils
    'wrap_coordinates',
    'pbc_distance',
    'unwrap_molecule_bfs',
    'unwrap_coords_numba',
    'unwrap_coords_python',
    'calculate_central_mass',
    # file_utils
    'write_lammps_dump_file',
    'write_cg_trajectory',
    'read_lammps_dump_file',
    # graph_utils
    'find_molecules',
    'build_bond_graph',
    'get_molecule_sizes',
    'get_molecule_atoms',
    'NUMBA_AVAILABLE',
]