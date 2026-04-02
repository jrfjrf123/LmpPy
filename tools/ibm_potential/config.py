"""
IBM势能计算配置加载器

加载 ibm_potential.yaml 配置文件，用于IBM势能计算流程。

配置文件示例:
    simulation:
      temperature: 400
      units: real

    trajectory:
      format: pickle
      path: cg_trajectory.pkl
      stride: 1

    distribution:
      n_bins: 200
      bond_range: [0.5, 6.0]
      angle_range: [0, 180]
      dihedral_range: [-180, 180]
      pair_range: [1.5, 18.0]

    smoothing:
      method: auto
      sg_window: 21

    output:
      table_dir: lammps_tables
      potentials_dir: potentials_output
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import yaml


@dataclass
class SimulationConfig:
    """模拟参数配置"""
    temperature: float = 400.0  # K
    units: str = 'real'  # real, metal, lj


@dataclass
class TrajectoryConfig:
    """轨迹输入配置"""
    format: str = 'pickle'  # pickle, lammpsdump
    path: str = 'cg_trajectory.pkl'
    stride: int = 1


@dataclass
class DistributionConfig:
    """分布计算配置"""
    n_bins: int = 200
    bond_range: Tuple[float, float] = (0.5, 6.0)
    angle_range: Tuple[float, float] = (0.0, 180.0)
    dihedral_range: Tuple[float, float] = (-180.0, 180.0)
    pair_range: Tuple[float, float] = (1.5, 18.0)
    rdf_dr: float = 0.01
    rdf_exclude_bonds: bool = True
    rdf_exclude_angles: bool = True
    rdf_exclude_dihedrals: bool = True


@dataclass
class SmoothingConfig:
    """平滑参数配置"""
    method: str = 'auto'  # auto, harmonic, gaussian
    sg_window: int = 21
    sg_polyorder: int = 3
    sigma_range: List[float] = field(default_factory=lambda: [2.0, 3.0, 5.0, 7.0])
    temperature: float = 400.0


@dataclass
class OutputConfig:
    """输出配置"""
    table_dir: str = 'lammps_tables'
    potentials_dir: str = 'potentials_output'
    distributions_dir: str = 'distributions_output'
    generate_plots: bool = True
    generate_reference: bool = True


@dataclass
class IBMConfig:
    """IBM势能计算完整配置"""
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    trajectory: TrajectoryConfig = field(default_factory=TrajectoryConfig)
    distribution: DistributionConfig = field(default_factory=DistributionConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # 配置文件目录（用于解析相对路径）
    config_dir: Path = field(default_factory=lambda: Path('.'))

    def resolve_path(self, path: str) -> Path:
        """解析相对路径"""
        p = Path(path)
        if p.is_absolute():
            return p
        return self.config_dir / p


class IBMConfigLoader:
    """IBM配置加载器"""

    def __init__(self, config_path: str = None):
        """
        初始化配置加载器。

        Args:
            config_path: 配置文件路径（ibm_potential.yaml）
        """
        self.config_path = Path(config_path) if config_path else None
        self.config_dir = self.config_path.parent if self.config_path else Path('.')

    def load(self) -> IBMConfig:
        """加载配置文件"""
        if self.config_path is None or not self.config_path.exists():
            print("警告: 配置文件不存在，使用默认配置")
            return IBMConfig(config_dir=self.config_dir)

        with open(self.config_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        # 解析各部分配置
        sim_data = data.get('simulation', {})
        traj_data = data.get('trajectory', {})
        dist_data = data.get('distribution', {})
        smooth_data = data.get('smoothing', {})
        out_data = data.get('output', {})

        # 构建配置对象
        config = IBMConfig(
            simulation=SimulationConfig(
                temperature=float(sim_data.get('temperature', 400.0)),
                units=sim_data.get('units', 'real')
            ),
            trajectory=TrajectoryConfig(
                format=traj_data.get('format', 'pickle'),
                path=traj_data.get('path', 'cg_trajectory.pkl'),
                stride=int(traj_data.get('stride', 1))
            ),
            distribution=DistributionConfig(
                n_bins=int(dist_data.get('n_bins', 200)),
                bond_range=tuple(dist_data.get('bond_range', [0.5, 6.0])),
                angle_range=tuple(dist_data.get('angle_range', [0.0, 180.0])),
                dihedral_range=tuple(dist_data.get('dihedral_range', [-180.0, 180.0])),
                pair_range=tuple(dist_data.get('pair_range', [1.5, 18.0])),
                rdf_dr=float(dist_data.get('rdf_dr', 0.01)),
                rdf_exclude_bonds=dist_data.get('rdf_exclude_bonds', True),
                rdf_exclude_angles=dist_data.get('rdf_exclude_angles', True),
                rdf_exclude_dihedrals=dist_data.get('rdf_exclude_dihedrals', True)
            ),
            smoothing=SmoothingConfig(
                method=smooth_data.get('method', 'auto'),
                sg_window=int(smooth_data.get('sg_window', 21)),
                sg_polyorder=int(smooth_data.get('sg_polyorder', 3)),
                sigma_range=smooth_data.get('sigma_range', [2.0, 3.0, 5.0, 7.0]),
                temperature=float(smooth_data.get('temperature', 400.0))
            ),
            output=OutputConfig(
                table_dir=out_data.get('table_dir', 'lammps_tables'),
                potentials_dir=out_data.get('potentials_dir', 'potentials_output'),
                distributions_dir=out_data.get('distributions_dir', 'distributions_output'),
                generate_plots=out_data.get('generate_plots', True),
                generate_reference=out_data.get('generate_reference', True)
            ),
            config_dir=self.config_dir
        )

        return config

    def save(self, config: IBMConfig, output_path: str = None):
        """保存配置到YAML文件"""
        path = Path(output_path) if output_path else self.config_path

        data = {
            'simulation': {
                'temperature': config.simulation.temperature,
                'units': config.simulation.units
            },
            'trajectory': {
                'format': config.trajectory.format,
                'path': config.trajectory.path,
                'stride': config.trajectory.stride
            },
            'distribution': {
                'n_bins': config.distribution.n_bins,
                'bond_range': list(config.distribution.bond_range),
                'angle_range': list(config.distribution.angle_range),
                'dihedral_range': list(config.distribution.dihedral_range),
                'pair_range': list(config.distribution.pair_range),
                'rdf_dr': config.distribution.rdf_dr,
                'rdf_exclude_bonds': config.distribution.rdf_exclude_bonds,
                'rdf_exclude_angles': config.distribution.rdf_exclude_angles,
                'rdf_exclude_dihedrals': config.distribution.rdf_exclude_dihedrals
            },
            'smoothing': {
                'method': config.smoothing.method,
                'sg_window': config.smoothing.sg_window,
                'sg_polyorder': config.smoothing.sg_polyorder,
                'sigma_range': config.smoothing.sigma_range,
                'temperature': config.smoothing.temperature
            },
            'output': {
                'table_dir': config.output.table_dir,
                'potentials_dir': config.output.potentials_dir,
                'distributions_dir': config.output.distributions_dir,
                'generate_plots': config.output.generate_plots,
                'generate_reference': config.output.generate_reference
            }
        }

        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(data, f, default_flow_style=False, allow_unicode=True)

        print(f"配置已保存到: {path}")


def load_ibm_config(config_path: str = None) -> IBMConfig:
    """便捷函数：加载IBM配置"""
    loader = IBMConfigLoader(config_path)
    return loader.load()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    else:
        config_path = None

    config = load_ibm_config(config_path)

    print("IBM势能计算配置:")
    print(f"  温度: {config.simulation.temperature} K")
    print(f"  轨迹格式: {config.trajectory.format}")
    print(f"  轨迹路径: {config.trajectory.path}")
    print(f"  分布bins: {config.distribution.n_bins}")
    print(f"  输出目录: {config.output.potentials_dir}")