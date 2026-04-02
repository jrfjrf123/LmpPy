"""
单位转换工具

提供能量、长度等物理量的单位转换。

支持的单位:
- 能量: kcal/mol, kJ/mol, eV
- 长度: Angstrom, nm

作者: Claude
日期: 2026-04-02
"""

from typing import Dict

# 能量转换因子 (相对于 kcal/mol)
ENERGY_CONVERSION = {
    'kcal/mol': 1.0,
    'kJ/mol': 4.184,          # 1 kcal/mol = 4.184 kJ/mol
    'eV': 0.0433641,          # 1 kcal/mol = 0.0433641 eV
    'J': 6.9477e-21,          # 1 kcal/mol = 6.9477e-21 J
    'K': 503.221              # 1 kcal/mol = 503.221 K (能量单位)
}

# 长度转换因子 (相对于 Angstrom)
LENGTH_CONVERSION = {
    'A': 1.0,
    'Angstrom': 1.0,
    'angstrom': 1.0,
    'nm': 0.1,                # 1 A = 0.1 nm
    'nm': 0.1,
    'pm': 100.0,              # 1 A = 100 pm
    'm': 1e-10                # 1 A = 1e-10 m
}

# 玻尔兹曼常数
KB = {
    'kcal/mol': 0.0019872041,   # kcal/mol/K
    'kJ/mol': 0.0083144626,     # kJ/mol/K
    'eV': 8.617333262e-5,       # eV/K
    'J': 1.380649e-23           # J/K
}


def convert_energy(value: float, from_unit: str, to_unit: str) -> float:
    """
    转换能量单位。

    Args:
        value: 能量值
        from_unit: 原单位
        to_unit: 目标单位

    Returns:
        转换后的值
    """
    if from_unit not in ENERGY_CONVERSION:
        raise ValueError(f"未知能量单位: {from_unit}")
    if to_unit not in ENERGY_CONVERSION:
        raise ValueError(f"未知能量单位: {to_unit}")

    # 转换为 kcal/mol 作为中间单位
    value_kcal = value / ENERGY_CONVERSION[from_unit]
    # 转换到目标单位
    return value_kcal * ENERGY_CONVERSION[to_unit]


def convert_length(value: float, from_unit: str, to_unit: str) -> float:
    """
    转换长度单位。

    Args:
        value: 长度值
        from_unit: 原单位
        to_unit: 目标单位

    Returns:
        转换后的值
    """
    if from_unit not in LENGTH_CONVERSION:
        raise ValueError(f"未知长度单位: {from_unit}")
    if to_unit not in LENGTH_CONVERSION:
        raise ValueError(f"未知长度单位: {to_unit}")

    # 转换为 Angstrom 作为中间单位
    value_A = value / LENGTH_CONVERSION[from_unit]
    # 转换到目标单位
    return value_A * LENGTH_CONVERSION[to_unit]


def get_kb(unit: str) -> float:
    """
    获取指定单位的玻尔兹曼常数。

    Args:
        unit: 能量单位

    Returns:
        玻尔兹曼常数值
    """
    if unit not in KB:
        raise ValueError(f"未知单位: {unit}")
    return KB[unit]


def thermal_energy(temperature: float, unit: str = 'kcal/mol') -> float:
    """
    计算热能 kT。

    Args:
        temperature: 温度 (K)
        unit: 能量单位

    Returns:
        kT 值
    """
    return get_kb(unit) * temperature


class UnitConverter:
    """单位转换器类"""

    def __init__(self, energy_unit: str = 'kcal/mol', length_unit: str = 'A'):
        """
        初始化转换器。

        Args:
            energy_unit: 能量单位
            length_unit: 长度单位
        """
        self.energy_unit = energy_unit
        self.length_unit = length_unit

    def convert_energy(self, value: float, from_unit: str = None, to_unit: str = None) -> float:
        """转换能量"""
        from_unit = from_unit or self.energy_unit
        to_unit = to_unit or self.energy_unit
        return convert_energy(value, from_unit, to_unit)

    def convert_length(self, value: float, from_unit: str = None, to_unit: str = None) -> float:
        """转换长度"""
        from_unit = from_unit or self.length_unit
        to_unit = to_unit or self.length_unit
        return convert_length(value, from_unit, to_unit)

    @property
    def kb(self) -> float:
        """获取当前单位的玻尔兹曼常数"""
        return get_kb(self.energy_unit)

    def kt(self, temperature: float) -> float:
        """计算热能 kT"""
        return self.kb * temperature


if __name__ == "__main__":
    print("单位转换工具")
    print("\n能量转换:")
    print(f"  1 kcal/mol = {convert_energy(1.0, 'kcal/mol', 'kJ/mol'):.4f} kJ/mol")
    print(f"  1 kcal/mol = {convert_energy(1.0, 'kcal/mol', 'eV'):.6f} eV")

    print("\n长度转换:")
    print(f"  1 A = {convert_length(1.0, 'A', 'nm'):.2f} nm")

    print("\n热能 (400 K):")
    print(f"  kT = {thermal_energy(400, 'kcal/mol'):.6f} kcal/mol")
    print(f"  kT = {thermal_energy(400, 'kJ/mol'):.6f} kJ/mol")