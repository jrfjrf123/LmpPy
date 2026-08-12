#!/usr/bin/env python3
"""gmx2lmp_data 的 pytest 测试。"""
import re
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import gmx2lmp_data as g2l

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestParseGro:
    def test_basic_orthogonal_box(self, tmp_path):
        gro = tmp_path / "a.gro"
        gro.write_text(
            "title\n"
            "2\n"
            "    1UNL     C1    1   1.000   2.000   3.000\n"
            "    1UNL     C2    2   1.100   2.100   3.100\n"
            "   5.00000   5.00000   5.00000\n",
            encoding="utf-8",
        )
        coords, box = g2l.parse_gro(gro)
        assert coords[0] == pytest.approx((10.0, 20.0, 30.0))
        assert coords[1] == pytest.approx((11.0, 21.0, 31.0))
        assert box == pytest.approx([50.0, 50.0, 50.0])

    def test_stuck_name_and_number_columns(self, tmp_path):
        """EPR system.gro 的真实行：atomname 与 atomnr 之间无空格（固定列宽）。"""
        gro = tmp_path / "s.gro"
        gro.write_text(
            "EPR 357 chains\n"
            "2\n"
            "    1UNL     C1    1   9.052   6.655   9.483\n"
            "    1UNL    H9449980   6.607   6.353   2.717\n"
            "  10.35573  10.35573  10.35573\n",
            encoding="utf-8",
        )
        coords, box = g2l.parse_gro(gro)
        assert coords[0] == pytest.approx((90.52, 66.55, 94.83))
        assert coords[1] == pytest.approx((66.07, 63.53, 27.17))
        assert box == pytest.approx([103.5573] * 3)

    def test_triclinic_box_raises(self, tmp_path):
        gro = tmp_path / "t.gro"
        gro.write_text(
            "t\n1\n"
            "    1UNL     C1    1   1.000   1.000   1.000\n"
            "   5.00000   5.00000   5.00000   0.00000   0.00000   1.00000"
            "   0.00000   0.00000   0.00000\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="三斜"):
            g2l.parse_gro(gro)

    def test_zero_box_9val_accepted_as_zero(self, tmp_path):
        """sobtop 的 gro 盒行为 9 个 0：按全零盒处理（兜底在 build_system）。"""
        gro = tmp_path / "z.gro"
        gro.write_text(
            "z\n1\n"
            "    1UNL     C1    1   1.000   1.000   1.000\n"
            "   0.00000   0.00000   0.00000   0.00000   0.00000   0.00000"
            "   0.00000   0.00000   0.00000\n",
            encoding="utf-8",
        )
        coords, box = g2l.parse_gro(gro)
        assert box == [0.0, 0.0, 0.0]
