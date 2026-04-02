"""
AA2CG - 全原子到粗粒化转换工具

提供AA（全原子）到CG（粗粒化）的转换功能：
- LAMMPS data文件转换
- 轨迹文件转换（支持GROMACS TRR和LAMMPS dump）
- CG映射工具函数

使用示例:
    from LmpPy.tools.aa2cg import (
        load_aa_to_cg_mapping,
        convert_aa_to_cg_frame,
        read_lammps_data,
        write_cg_data_file,
        read_gromacs_trr_all_frames,
        convert_trajectory_to_cg
    )

    # 转换data文件
    aa_data = read_lammps_data('system.data')
    cg_data, mapping = convert_data_to_cg(aa_data, 'mapping.csv', 'cg_bonds.txt')
    write_cg_data_file('cg.data', cg_data, cg_data['bonds'])

    # 转换轨迹
    aa_frames = read_gromacs_trr_all_frames('topol.tpr', 'traj.trr')
    cg_traj = convert_trajectory_to_cg(aa_frames, 'mapping.csv')
    save_cg_trajectory_pickle(cg_traj, 'cg_trajectory.pkl')
"""

# 映射工具函数
from .mapping_utils import (
    load_aa_to_cg_mapping,
    convert_aa_to_cg_frame,
    pbc_distance,
    wrap_coords,
    calculate_com
)

# Data文件转换
from .data_converter import (
    read_lammps_data,
    write_cg_data_file,
    convert_data_to_cg,
    unwrap_coords
)

# 轨迹转换
from .trj_converter import (
    read_lammps_dump,
    read_gromacs_trr,
    read_gromacs_trr_all_frames,
    convert_trajectory_to_cg,
    save_cg_trajectory_pickle,
    load_cg_trajectory_pickle,
    write_frame_to_xyz,
    write_trajectory_to_xyz
)

__all__ = [
    # 映射工具
    'load_aa_to_cg_mapping',
    'convert_aa_to_cg_frame',
    'pbc_distance',
    'wrap_coords',
    'calculate_com',

    # Data转换
    'read_lammps_data',
    'write_cg_data_file',
    'convert_data_to_cg',
    'unwrap_coords',

    # 轨迹转换
    'read_lammps_dump',
    'read_gromacs_trr',
    'read_gromacs_trr_all_frames',
    'convert_trajectory_to_cg',
    'save_cg_trajectory_pickle',
    'load_cg_trajectory_pickle',
    'write_frame_to_xyz',
    'write_trajectory_to_xyz'
]