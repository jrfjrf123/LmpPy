"""
LAMMPS势能表生成模块

从势能文件创建LAMMPS table格式的势能表文件。

输出格式:
- pair_table.txt
- bond_table.txt
- angle_table.txt
- dihedral_table.txt

作者: 整合自 md_base_on_ml/calc_ibm_pot/create_lammps_pot.py
"""

import numpy as np
import os
import glob
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def read_potential_file(filename: str) -> Tuple[np.ndarray, np.ndarray, str]:
    """
    读取势能文件。

    Args:
        filename: 势能文件路径

    Returns:
        (x, potential, header): 坐标、势能和头部信息
    """
    with open(filename, 'r') as f:
        header = f.readline().strip()
        if header.startswith('#'):
            header = header[1:].strip()

    data = np.loadtxt(filename)
    x = data[:, 0]
    potential = data[:, 1]

    return x, potential, header


def calculate_force(x: np.ndarray, potential: np.ndarray) -> np.ndarray:
    """计算力 (-dU/dx)"""
    return -np.gradient(potential, x)


def extract_type_info(filename: str) -> Tuple[str, str, List[int]]:
    """
    从文件名提取势能类型和类型编号。

    Args:
        filename: 文件名，如 'bond_type1_potential.txt'

    Returns:
        (pot_type, type_str, bead_types): 势能类型、类型字符串、bead类型列表
    """
    basename = os.path.basename(filename)

    if basename.startswith('bond_type'):
        pot_type = 'bond'
        type_str = basename.split('_potential')[0].replace('bond_', '')
    elif basename.startswith('angle_type'):
        pot_type = 'angle'
        type_str = basename.split('_potential')[0].replace('angle_', '')
    elif basename.startswith('dihedral_type'):
        pot_type = 'dihedral'
        type_str = basename.split('_potential')[0].replace('dihedral_', '')
    elif basename.startswith('pair_type'):
        pot_type = 'pair'
        type_str = basename.split('_potential')[0].replace('pair_', '')
    else:
        raise ValueError(f"无法从文件名确定势能类型: {filename}")

    # 提取bead类型 (e.g., 'type1_2' -> [1, 2])
    import re
    match = re.search(r'type([0-9_]+)', type_str)
    if match:
        type_numbers = match.group(1).split('_')
        bead_types = [int(t) for t in type_numbers if t.isdigit()]
    else:
        bead_types = []

    return pot_type, type_str, bead_types


def write_lammps_table_section(f, keyword: str, x: np.ndarray,
                                potential: np.ndarray, force: np.ndarray,
                                pot_type: str = 'pair',
                                n_points: int = None):
    """
    写入单个table section。

    Args:
        f: 文件对象
        keyword: section关键字
        x: 坐标数组
        potential: 势能数组
        force: 力数组
        pot_type: 势能类型
        n_points: N参数值
    """
    N = n_points if n_points is not None else len(x)
    r_min = x[0]
    r_max = x[-1]

    f.write(f"{keyword}\n")

    if pot_type == 'bond':
        f.write(f"N {N}\n")
    elif pot_type == 'angle':
        f.write(f"N {N}\n")
    elif pot_type == 'dihedral':
        f.write(f"N {N} DEGREES\n")
    else:  # pair
        f.write(f"N {N} R {r_min:.10e} {r_max:.10e}\n")

    f.write("\n")

    for i in range(N):
        f.write(f"{i+1} {x[i]:.10e} {potential[i]:.10e} {force[i]:.10e}\n")

    f.write("\n")


def create_lammps_table_files(input_dir: str, output_dir: str,
                               units: str = 'real',
                               contributor: str = 'IBM Potential Generator',
                               date_str: str = None) -> Dict[str, str]:
    """
    从势能文件创建LAMMPS table文件。

    Args:
        input_dir: 势能文件目录
        output_dir: 输出目录
        units: 单位系统
        contributor: 贡献者信息
        date_str: 日期字符串

    Returns:
        output_files: 输出文件字典 {pot_type: filepath}
    """
    os.makedirs(output_dir, exist_ok=True)

    if date_str is None:
        date_str = datetime.now().strftime('%Y-%m-%d')

    # 查找势能文件
    potential_files = glob.glob(os.path.join(input_dir, '*_potential*.txt'))

    if not potential_files:
        print(f"在 {input_dir} 中未找到势能文件")
        return {}

    # 按类型分组
    potentials_by_type = {'pair': [], 'bond': [], 'angle': [], 'dihedral': []}

    for filename in sorted(potential_files):
        try:
            pot_type, type_str, bead_types = extract_type_info(filename)
            potentials_by_type[pot_type].append((filename, type_str, bead_types))
        except ValueError as e:
            print(f"跳过文件: {e}")
            continue

    output_files = {}

    print(f"\n创建LAMMPS table势能文件...")
    print(f"  输入目录: {input_dir}")
    print(f"  输出目录: {output_dir}")

    for pot_type in ['pair', 'bond', 'angle', 'dihedral']:
        if not potentials_by_type[pot_type]:
            continue

        output_filename = os.path.join(output_dir, f'{pot_type}_table.txt')
        output_files[pot_type] = output_filename

        print(f"\n  创建 {pot_type} table: {output_filename}")
        print(f"    {pot_type}类型数: {len(potentials_by_type[pot_type])}")

        with open(output_filename, 'w') as f:
            f.write(f"# DATE: {date_str} UNITS: {units} CONTRIBUTOR: {contributor}\n")
            f.write(f"# LAMMPS table potential for {pot_type} interactions\n")
            f.write(f"# Generated from IBM potential calculation\n")
            f.write("#\n")
            f.write(f"# This file contains {len(potentials_by_type[pot_type])} {pot_type} type(s)\n")
            f.write("#\n\n")

            for filename, type_str, bead_types in potentials_by_type[pot_type]:
                x, potential, header = read_potential_file(filename)

                # 过滤inf值
                finite_mask = np.isfinite(potential)
                if not np.any(finite_mask):
                    print(f"    警告: {type_str} 无有限值，跳过")
                    continue

                x_finite = x[finite_mask]
                potential_finite = potential[finite_mask]
                force_finite = calculate_force(x_finite, potential_finite)

                keyword = f"{pot_type.upper()}_TYPE{type_str.replace('type', '').upper()}"

                f.write(f"# {header}\n")
                write_lammps_table_section(f, keyword, x_finite, potential_finite,
                                           force_finite, pot_type=pot_type)

                print(f"    - {keyword}: {len(x_finite)} points, r=[{x_finite[0]:.3f}, {x_finite[-1]:.3f}]")

    print(f"\n✓ LAMMPS table文件创建成功!")

    # 打印使用示例
    print("\nLAMMPS使用示例:")
    print("-" * 50)
    if 'pair' in output_files:
        print("pair_style table linear 1000")
        print(f"pair_coeff 1 1 {os.path.basename(output_files['pair'])} PAIR_TYPE1_1")
    if 'bond' in output_files:
        print("bond_style table linear 1000")
        print(f"bond_coeff 1 {os.path.basename(output_files['bond'])} BOND_TYPE1")

    return output_files


def create_reference_file(output_dir: str):
    """创建参考文件，列出所有可用的table关键字"""
    table_files = glob.glob(os.path.join(output_dir, '*_table.txt'))

    if not table_files:
        print("未找到table文件用于生成参考")
        return

    ref_file = os.path.join(output_dir, 'table_reference.txt')

    with open(ref_file, 'w') as f:
        f.write("# LAMMPS Table Potential Reference\n")
        f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("#\n\n")

        for table_file in sorted(table_files):
            pot_type = os.path.basename(table_file).replace('_table.txt', '')
            f.write(f"\n{'='*60}\n")
            f.write(f"{pot_type.upper()} POTENTIALS\n")
            f.write(f"File: {os.path.basename(table_file)}\n")
            f.write(f"{'='*60}\n\n")

            with open(table_file, 'r') as tf:
                for line in tf:
                    line_stripped = line.strip()
                    # 跳过注释和参数行
                    if (line_stripped and
                        not line_stripped.startswith('#') and
                        not line_stripped.startswith('N ') and
                        not line_stripped[0].isdigit()):
                        f.write(f"  {line_stripped}\n")

    print(f"\n参考文件创建: {ref_file}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
    else:
        input_dir = 'potentials_output'

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = 'lammps_tables'

    output_files = create_lammps_table_files(input_dir, output_dir)

    if output_files:
        create_reference_file(output_dir)