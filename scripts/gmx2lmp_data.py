#!/usr/bin/env python3
"""GROMACS top + gro → LAMMPS data（GAFF，real 单位，atom_style full）。

用法:
    python LmpPy/scripts/gmx2lmp_data.py --top system.top --gro conf.gro -o out.data \
        [--type-order type_order.txt]

支持范围（不支持的输入一律报错，不静默转换）:
- top 递归 #include（相对包含文件所在目录解析）
- [ defaults ] comb-rule 2（σ/ε）与 comb-rule 1（C6/C12 自动换算）
- [ atomtypes ] 6 列（name mass charge ptype σ ε）/ 7 列（含 at.num 列）
- bonds/angles funct 1、dihedrals funct 9/1（proper）、funct 4（improper → cvff）
- [ molecules ] 多分子计数展开
- 正交盒；全零盒按坐标范围 + 1 nm 边距兜底（最小 3 nm）
"""
from __future__ import annotations

import argparse  # noqa: F401
import re  # noqa: F401
import sys  # noqa: F401
import warnings  # noqa: F401
from dataclasses import dataclass  # noqa: F401
from dataclasses import field  # noqa: F401
from pathlib import Path

KCAL_PER_KJ = 4.184  # kJ/mol → kcal/mol
NM_TO_ANGSTROM = 10.0  # nm → Å

# GRO 原子行固定列宽切片（3 位小数标准格式）
GRO_X_SLICE = slice(20, 28)
GRO_Y_SLICE = slice(28, 36)
GRO_Z_SLICE = slice(36, 44)

# 兜底分支：按空白切分时，含速度列的典型令牌数（8=无残基号？10=含速度）
GRO_TOKEN_COUNTS_WITH_VELOCITY = frozenset({8, 10})


def parse_gro(gro_path: Path | str) -> tuple[list[tuple[float, float, float]], list[float]]:
    """解析 gro：返回（坐标 Å 列表, 正交盒 [Lx, Ly, Lz] Å）。

    原子行按固定列宽取 x/y/z（第 21-44 列，每列 8 字符、3 位小数；列宽变体兜底
    按空白切分取末尾坐标列）。盒行 3 个数 = 正交盒；9 个数且非对角元非零
    = 三斜盒 → 报错；9 个 0（sobtop 产物）按全零盒返回，由 build_system 兜底。
    """
    lines = Path(gro_path).read_text(encoding="utf-8").splitlines()
    try:
        n = int(lines[1].strip())
    except (IndexError, ValueError):
        raise ValueError(f"gro 文件格式异常（第 2 行应为原子数）: {gro_path}") from None
    if len(lines) < n + 3:
        raise ValueError(f"gro 文件行数不足（声称 {n} 原子）: {gro_path}")

    coords = []
    for line in lines[2:2 + n]:
        try:
            x = float(line[GRO_X_SLICE])
            y = float(line[GRO_Y_SLICE])
            z = float(line[GRO_Z_SLICE])
        except ValueError:
            # 列宽变体兜底：按空白切分，有速度列取倒数 6..4，否则取末 3 列
            t = line.split()
            xyz = t[-6:-3] if len(t) in GRO_TOKEN_COUNTS_WITH_VELOCITY else t[-3:]
            try:
                x, y, z = (float(v) for v in xyz)
            except ValueError:
                raise ValueError(f"gro 原子行解析失败: {line!r}") from None
        coords.append((x * NM_TO_ANGSTROM, y * NM_TO_ANGSTROM, z * NM_TO_ANGSTROM))

    box_tokens = [float(v) for v in lines[2 + n].split()]
    if len(box_tokens) == 3:
        box = [v * NM_TO_ANGSTROM for v in box_tokens]
    elif len(box_tokens) == 9:
        if any(v != 0.0 for v in box_tokens[3:]):
            raise ValueError("三斜盒不支持（仅支持正交盒）")
        box = [v * NM_TO_ANGSTROM for v in box_tokens[:3]]
    else:
        raise ValueError(f"gro 盒行列数异常（需 3 或 9 列）: {lines[2 + n]!r}")
    return coords, box
