#!/usr/bin/env python3
"""gmx2lmp_data 的 pytest 测试。"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
import gmx2lmp_data as g2l


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

    def test_fallback_5_decimal_precision(self, tmp_path):
        """5 位小数列宽变体触发兜底分支，必须保留完整精度。"""
        gro = tmp_path / "5dec.gro"
        gro.write_text(
            "5dec\n"
            "2\n"
            "    1UNL    H9449980  6.60700  6.35300  2.71700\n"
            "    1UNL    H9449981  1.00001  2.00002  3.00003\n"
            "  10.35573  10.35573  10.35573\n",
            encoding="utf-8",
        )
        coords, box = g2l.parse_gro(gro)
        # 第二行固定列宽切片会截断，兜底分支读取完整 5 位小数
        assert coords[1] == pytest.approx(
            (1.00001 * 10.0, 2.00002 * 10.0, 3.00003 * 10.0)
        )
        # 第一行 likewise 验证兜底把 6.60700 等完整读入
        assert coords[0] == pytest.approx(
            (6.60700 * 10.0, 6.35300 * 10.0, 2.71700 * 10.0)
        )
        assert box == pytest.approx([103.5573] * 3)

    def test_invalid_atom_count_line(self, tmp_path):
        gro = tmp_path / "bad_count.gro"
        gro.write_text("title\nnot_a_number\n", encoding="utf-8")
        with pytest.raises(ValueError, match="原子数"):
            g2l.parse_gro(gro)

    def test_insufficient_lines(self, tmp_path):
        gro = tmp_path / "short.gro"
        gro.write_text(
            "title\n2\n    1UNL     C1    1   1.000   1.000   1.000\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="行数不足"):
            g2l.parse_gro(gro)

    def test_bad_box_line_column_count(self, tmp_path):
        gro = tmp_path / "bad_box.gro"
        gro.write_text(
            "title\n1\n    1UNL     C1    1   1.000   1.000   1.000\n"
            "   5.00000   5.00000\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="盒行列数异常"):
            g2l.parse_gro(gro)


class TestTopInfrastructure:
    def test_expand_includes_recursive(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "inner.itp").write_text("[ atoms ]\n1 c3 1 MOL C1 1 0.0\n", encoding="utf-8")
        (tmp_path / "outer.itp").write_text(
            '#include "sub/inner.itp"\n', encoding="utf-8")
        top = tmp_path / "s.top"
        top.write_text('[ defaults ]\n1 2 yes 0.5 0.8333\n#include "outer.itp"\n',
                       encoding="utf-8")
        lines = g2l._expand_includes(top)
        assert any("[ atoms ]" in l for l in lines)
        assert any(l.startswith("1 c3") for l in lines)

    def test_expand_includes_missing_raises(self, tmp_path):
        top = tmp_path / "s.top"
        top.write_text('#include "nope.itp"\n', encoding="utf-8")
        with pytest.raises(FileNotFoundError, match="nope.itp"):
            g2l._expand_includes(top)

    def test_collect_sections_header_inline_comment(self):
        """sobtop 风格节头行内注释 `[ dihedrals ] ; propers` 正常识别。"""
        lines = [
            "[ bonds ]",
            "1 2 1 0.15 83680.0  ; C1-C2 注释",
            "",
            "[ dihedrals ] ; propers",
            "1 2 3 4 9 0.0 2.0 3",
        ]
        sections = g2l._collect_sections(lines)
        assert [name for name, _ in sections] == ["bonds", "dihedrals"]
        assert sections[0][1] == ["1 2 1 0.15 83680.0"]
        assert sections[1][1] == ["1 2 3 4 9 0.0 2.0 3"]

    def test_parse_defaults(self):
        d = g2l._parse_defaults(["1 2 yes 0.5 0.8333"])
        assert d.comb_rule == 2
        assert d.fudge_lj == 0.5
        assert d.fudge_qq == pytest.approx(0.8333)

    def test_parse_defaults_empty_raises(self):
        with pytest.raises(ValueError, match="defaults ] 段为空"):
            g2l._parse_defaults([])

    def test_atomtypes_7col_with_atnum(self):
        types = g2l._parse_atomtypes(
            ["c3 6 12.010736 0.000000 A 3.397710E-01 4.510352E-01"], comb_rule=2)
        assert types[0].name == "c3"
        assert types[0].mass == pytest.approx(12.010736)
        assert types[0].sigma_nm == pytest.approx(0.3397710)
        assert types[0].epsilon_kj == pytest.approx(0.4510352)

    def test_atomtypes_6col_without_atnum(self):
        types = g2l._parse_atomtypes(
            ["hc 1.007941 0.000000 A 2.600177E-01 8.702720E-02"], comb_rule=2)
        assert types[0].name == "hc"
        assert types[0].mass == pytest.approx(1.007941)

    def test_atomtypes_comb_rule_1_c6_c12(self):
        """comb-rule 1：σ=(C12/C6)^(1/6)，ε=C6²/(4·C12)。

        构造值：c6=6.4e-05、c12=4.096e-09 → σ=0.2 nm，ε=0.25 kJ/mol。
        """
        types = g2l._parse_atomtypes(
            ["c3 6 12.011 0.0 A 6.4e-05 4.096e-09"], comb_rule=1)
        assert types[0].sigma_nm == pytest.approx(0.2)
        assert types[0].epsilon_kj == pytest.approx(0.25)

    def test_atomtypes_bad_column_count_raises(self):
        with pytest.raises(ValueError, match="atomtypes"):
            g2l._parse_atomtypes(["c3 12.011 0.0"], comb_rule=2)
