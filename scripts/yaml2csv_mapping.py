import yaml
import numpy as np
import pandas as pd
from pathlib import Path


def load_yaml(yaml_file):
    """Load YAML file."""
    with open(yaml_file, 'r') as f:
        return yaml.safe_load(f)


def parse_mapping_file(mapping_file, bead_type_offset=0, aa_id_offset=0, mol_id_offset=0, bead_id_offset=0, bead_type_map=None):
    """
    Parse a single mapping YAML file and generate bead information.
    
    Args:
        mapping_file: str or Path, path to mapping YAML file
        bead_type_offset: int, offset for bead type numbering
        aa_id_offset: int, offset for AA atom ID numbering
        mol_id_offset: int, offset for molecule ID numbering
        bead_id_offset: int, offset for bead ID numbering
    
    Returns:
        bead_list: list of tuples [(bead_id, mol_id, bead_type, aa_id, mass), ...]
        next_bead_id: int, next available bead ID
        next_bead_type: int, next available bead type
        next_aa_id: int, next available AA atom ID
        next_mol_id: int, next available molecule ID
    """
    data = load_yaml(mapping_file)
    site_types = data['site-types']
    config_list = data['config']
    if bead_type_map is None:
        print("Creating bead type mapping...")
        # Create bead type name to ID mapping
        bead_type_names = list(site_types.keys())
        bead_type_map = {name: i + 1 + bead_type_offset for i, name in enumerate(bead_type_names)}        
    
    bead_list = []
    current_bead_id = 1 + bead_id_offset  # Start from 1 + offset
    current_mol_id = 1 + mol_id_offset  # Start from 1 + offset
    
    # Process each configuration
    for config in config_list:
        anchor = int(config['anchor'])
        repeat = int(config['repeat'])
        offset = int(config['offset'])
        sites = config['sites']
        
        # For each repeat (each repeat = one molecule)
        for rep in range(1, repeat + 1):
            base_aa_id = anchor + (rep - 1) * offset + aa_id_offset
            
            # For each site in this repeat
            for site_info in sites:
                site_name = site_info[0]
                site_start = site_info[1]
                
                # Get atom indices and masses for this bead type
                atom_indices = site_types[site_name]['index']
                x_weights = site_types[site_name]['x-weight']
                
                # Get bead type ID
                bead_type = bead_type_map[site_name]
                
                # Calculate actual AA IDs and corresponding masses for this bead
                for i, atom_idx in enumerate(atom_indices):
                    aa_id = base_aa_id + site_start + atom_idx + 1  # +1 for 1-based indexing
                    aa_mass = x_weights[i]  # Get mass for this specific atom
                    bead_list.append((current_bead_id, current_mol_id, bead_type, aa_id, aa_mass))
                
                current_bead_id += 1
            
            # Increment molecule ID after each repeat
            current_mol_id += 1

    next_aa_id = aa_id_offset + repeat * offset + anchor
    next_mol_id = current_mol_id  # current_mol_id已经在循环中递增了

    if bead_type_map is None:
        next_bead_type = len(list(site_types.keys())) + 1 + bead_type_offset
    else:
        next_bead_type = None

    return bead_list, current_bead_id, next_bead_type, next_aa_id, next_mol_id


def load_custom_bead_type_map(path):
    """从 YAML/JSON 文件加载自定义 bead 类型映射 {site_type_name: bead_type_id}。

    YAML 是 JSON 超集，safe_load 同时兼容 .yaml / .json。
    映射须覆盖 system.yaml 引用的所有 mapping 文件里的 site-types 名称，
    否则缺失名称会走 parse_mapping_file 的自动编号（bead_type_map 非 None 时不回退）。
    """
    data = load_yaml(path)
    if not isinstance(data, dict):
        raise ValueError(f"自定义映射文件必须是键值映射: {path}")
    return {str(k): int(v) for k, v in data.items()}


def yaml_to_csv(system_yaml_file, output_csv='AtomId_BeadId_compare_list.csv', custom_bead_type_map=None, output_flag=True):
    """
    Convert YAML mapping files to CSV format.
    
    Args:
        system_yaml_file: str or Path, path to system.yaml file
        output_csv: str, output CSV filename
        custom_bead_type_map: dict, optional custom mapping of bead names to type IDs
                             Example: {'epoxy': 1, 'ether1': 3, 'ca1': 4, ...}
                             If None, auto-generate sequential IDs starting from 1
    
    Returns:
        df: pandas DataFrame with columns [bead_id, mol_id, bead_type, AA_id]
    
    Example:
        >>> # Auto-generate bead types
        >>> df = yaml_to_csv('system.yaml')
        
        >>> # Use custom bead type mapping
        >>> custom_map = {'epoxy': 1, 'ether1': 3, 'ca1': 4, 'RNH2': 2}
        >>> df = yaml_to_csv('system.yaml', custom_bead_type_map=custom_map)
    """
    # Load system configuration
    system_data = load_yaml(system_yaml_file)
    
    mapping_files = system_data['system']['names']
    numbers = system_data['system']['numbers']
    
    all_bead_list = []
    current_bead_type_offset = 0
    current_aa_id_offset = 0
    current_mol_id_offset = 0
    current_bead_id_offset = 0
    
    # If custom bead type map provided, print it
    if custom_bead_type_map is not None:
        print("Using custom bead type mapping:")
        for name, type_id in sorted(custom_bead_type_map.items(), key=lambda x: x[1]):
            print(f"  {name}: {type_id}")
        print()
    
    # Process each mapping file
    for i, (mapping_file, num_copies) in enumerate(zip(mapping_files, numbers)):
        print(f"Processing: {mapping_file} (copies: {num_copies})")
        
        # Get the actual file path (handle both absolute and relative paths)
        if not Path(mapping_file).exists():
            # Try relative to system.yaml directory
            system_dir = Path(system_yaml_file).parent
            mapping_file = system_dir / mapping_file  # 使用完整相对路径，而非只取文件名
        
        # Process each copy of this mapping file
        for copy_idx in range(num_copies):
            bead_list, next_bead_id, next_bead_type, next_aa_id, next_mol_id = parse_mapping_file(
                mapping_file,
                bead_type_offset=current_bead_type_offset,
                aa_id_offset=current_aa_id_offset,
                mol_id_offset=current_mol_id_offset,
                bead_id_offset=current_bead_id_offset,
                bead_type_map=custom_bead_type_map
            )
            
            all_bead_list.extend(bead_list)
            
            # Update offsets for next copy
            current_aa_id_offset = next_aa_id
            current_mol_id_offset = next_mol_id - 1  # -1 because next_mol_id is already incremented
            current_bead_id_offset = next_bead_id - 1  # -1 because next_bead_id is already incremented
            
            print(f"  Copy {copy_idx + 1}: Generated {len(bead_list)} mappings")
        
        # Update bead type offset for next mapping file (only if auto-generating)
        if custom_bead_type_map is None and next_bead_type is not None:
            current_bead_type_offset = next_bead_type - 1
    
    # Convert to DataFrame
    df = pd.DataFrame(all_bead_list, columns=['bead_id', 'mol_id', 'bead_type', 'AA_id', 'mass'])
    
    # Sort by AA_id for better readability
    df = df.sort_values('AA_id').reset_index(drop=True)
    
    if output_flag:
        # Save to CSV
        df.to_csv(output_csv, index=False)
        
        print(f"\nTotal mappings generated: {len(df)}")
        print(f"Unique beads: {df['bead_id'].nunique()}")
        print(f"Unique molecules: {df['mol_id'].nunique()}")
        print(f"Unique bead types: {df['bead_type'].nunique()}")
        print(f"AA atoms: {df['AA_id'].min()} to {df['AA_id'].max()}")
        print(f"AA mass range: {df['mass'].min():.1f} to {df['mass'].max():.1f}")
        print(f"\nOutput saved to: {output_csv}")
    
    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert YAML mapping files to CSV format for CG mapping.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 (自动生成 bead type)
  python -m LmpPy.scripts.yaml2csv_mapping -s system.yaml -o mapping.csv

  # 从 YAML/JSON 文件加载自定义 bead type 映射（推荐）
  python -m LmpPy.scripts.yaml2csv_mapping -s system.yaml --custom-map bead_type_map.yaml

  # 使用内置硬编码的自定义 bead type 映射（EPR，向后兼容）
  python -m LmpPy.scripts.yaml2csv_mapping -s system.yaml --custom
"""
    )

    parser.add_argument('-s', '--system-yaml', required=True,
                        help='Path to system.yaml file')
    parser.add_argument('-o', '--output', default='AtomId_BeadId_compare_list.csv',
                        help='Output CSV filename (default: AtomId_BeadId_compare_list.csv)')
    parser.add_argument('--custom-map', metavar='FILE',
                        help='从 YAML/JSON 文件加载自定义 bead type 映射 '
                             '{site_type_name: bead_type_id}')
    parser.add_argument('--custom', action='store_true',
                        help='Use hardcoded custom bead type mapping (EPR)')
    parser.add_argument('-q', '--quiet', action='store_true',
                        help='Suppress detailed output')

    args = parser.parse_args()

    # ========================================
    # 硬编码的自定义 bead type 映射（EPR，向后兼容兜底）
    # 新增体系建议改用 --custom-map 文件输入，避免改脚本
    # ========================================
    CUSTOM_BEAD_TYPE_MAP = {
    "Bead1": 1,    # chain E
    "Bead2": 2,    # chain P
    "Bead3": 1,    # chain reactor E
    "Bead4": 2,    # chain reactor P
    "Bead5": 3,    # E
    "Bead6": 4,    # P
    "Bead7": 1,    # chain head E
    "Bead8": 2     # chain head P
    }

    # 解析 bead type 映射：--custom-map 文件 > --custom 硬编码 > 自动编号(None)
    if args.custom_map and args.custom:
        print("警告: --custom-map 与 --custom 同时指定，优先使用 --custom-map")
    if args.custom_map:
        custom_bead_type_map = load_custom_bead_type_map(args.custom_map)
    elif args.custom:
        custom_bead_type_map = CUSTOM_BEAD_TYPE_MAP
    else:
        custom_bead_type_map = None

    if not args.quiet and custom_bead_type_map is not None:
        print("Using custom bead type mapping:")
        for name, type_id in sorted(custom_bead_type_map.items(), key=lambda x: x[1]):
            print(f"  {name}: {type_id}")
        print()

    # Run conversion
    df = yaml_to_csv(
        args.system_yaml,
        output_csv=args.output,
        custom_bead_type_map=custom_bead_type_map,
        output_flag=not args.quiet
    )

    if not args.quiet:
        print("\nFirst 10 rows:")
        print(df.head(10))
        print("\nBead type distribution:")
        print(df['bead_type'].value_counts().sort_index())
