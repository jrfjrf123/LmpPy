"""
工具函数模块

包含坐标处理、文件I/O、图论算法、拓扑文件、单位转换等工具函数
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

from .topology import (
    TopologyData,
    read_topology_files,
    read_type_dict,
    write_topology_file,
    derive_cg_bonds_from_aa
)

from .units import (
    ENERGY_CONVERSION,
    LENGTH_CONVERSION,
    KB,
    convert_energy,
    convert_length,
    get_kb,
    thermal_energy,
    UnitConverter
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
    # topology
    'TopologyData',
    'read_topology_files',
    'read_type_dict',
    'write_topology_file',
    'derive_cg_bonds_from_aa',
    # units
    'ENERGY_CONVERSION',
    'LENGTH_CONVERSION',
    'KB',
    'convert_energy',
    'convert_length',
    'get_kb',
    'thermal_energy',
    'UnitConverter',
]