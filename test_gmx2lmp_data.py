#!/usr/bin/env python3
"""gmx2lmp_data 的 pytest 测试。"""
import re
import subprocess
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


MINI_TOP = """\
[ defaults ]
1 2 yes 0.5 0.8333

[ atomtypes ]
c3   6   12.011   0.0   A   3.400000E-01   4.184000E-01
hc   1   1.008    0.0   A   2.600000E-01   8.368000E-02

[ moleculetype ]
mol   3

[ atoms ]
1  c3  1  MOL  C1  1  0.10000000  12.011
2  c3  1  MOL  C2  2  -0.20000000  12.011
3  c3  1  MOL  C3  3  0.05000000  12.011
4  hc  1  MOL  H4  4  0.05000000  1.008

[ bonds ]
1  2  1  0.150000  83680.0
2  3  1  0.150000  83680.0
3  4  1  0.109000  83680.0

[ angles ]
1  2  3  1  109.500  836.8
2  3  4  1  109.500  836.8

[ dihedrals ]
1  2  3  4  9  180.0  4.184  -3
1  2  3  4  9  0.0  2.092  1
1  2  3  4  4  0.0  8.368  2

[ moleculetype ]
single  3

[ atoms ]
1  hc  1  SOL  H1  1  0.00000000  1.008

[ molecules ]
mol     2
single  2
"""

MINI_GRO = """\
mini
10
    1MOL     C1    1   1.000   1.000   1.000
    1MOL     C2    2   1.150   1.000   1.000
    1MOL     C3    3   1.300   1.000   1.000
    1MOL     H4    4   1.400   1.000   1.000
    2MOL     C1    5   2.000   2.000   2.000
    2MOL     C2    6   2.150   2.000   2.000
    2MOL     C3    7   2.300   2.000   2.000
    2MOL     H4    8   2.400   2.000   2.000
    3SOL     H1    9   3.000   3.000   3.000
    4SOL     H1   10   3.500   3.500   3.500
   5.00000   5.00000   5.00000
"""


def _write_mini(tmp_path: Path):
    top = tmp_path / "mini.top"
    gro = tmp_path / "mini.gro"
    top.write_text(MINI_TOP, encoding="utf-8")
    gro.write_text(MINI_GRO, encoding="utf-8")
    return top, gro


class TestParseTop:
    def test_mini_system(self, tmp_path):
        top, _ = _write_mini(tmp_path)
        topo = g2l.parse_top(top)
        assert topo.defaults.comb_rule == 2
        assert [t.name for t in topo.atom_types] == ["c3", "hc"]
        assert topo.molecules == [("mol", 2), ("single", 2)]
        mol = topo.mol_types["mol"]
        assert len(mol.atoms) == 4
        assert mol.atoms[1].charge == pytest.approx(-0.2)
        assert len(mol.bonds) == 3
        assert len(mol.angles) == 2
        # funct 9 两行 → propers（多 term 同四元组保留），funct 4 → improper
        assert len(mol.dihedrals) == 2
        assert len(mol.impropers) == 1
        assert mol.dihedrals[0].params == ("180.0", "4.184", "-3")
        assert len(topo.mol_types["single"].atoms) == 1

    def test_atoms_7col_no_mass(self, tmp_path):
        text = MINI_TOP.replace("1  hc  1  SOL  H1  1  0.00000000  1.008",
                                "1  hc  1  SOL  H1  1  0.00000000")
        top = tmp_path / "t.top"
        top.write_text(text, encoding="utf-8")
        topo = g2l.parse_top(top)
        assert topo.mol_types["single"].atoms[0].mass is None
        assert topo.mol_types["mol"].atoms[0].mass == pytest.approx(12.011)

    def test_bond_funct_unsupported_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.replace("1  2  1  0.150000  83680.0",
                                        "1  2  2  0.150000  83680.0"),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="functype"):
            g2l.parse_top(top)

    def test_dihedral_funct_3_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.replace("1  2  3  4  9  180.0  4.184  -3",
                                        "1  2  3  4  3  0.0  0.0  0.0  0.0  0.0  0.0"),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="functype"):
            g2l.parse_top(top)

    def test_undefined_moleculetype_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.replace("single  2", "ghost  2"), encoding="utf-8")
        with pytest.raises(ValueError, match="ghost"):
            g2l.parse_top(top)

    def test_undefined_atomtype_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.replace("1  c3  1  MOL  C1", "1  xx  1  MOL  C1"),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="xx"):
            g2l.parse_top(top)

    def test_missing_molecules_section_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.split("[ molecules ]")[0], encoding="utf-8")
        with pytest.raises(ValueError, match="molecules"):
            g2l.parse_top(top)

    def test_molecules_bad_column_count_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.replace("mol     2", "mol"), encoding="utf-8")
        with pytest.raises(ValueError, match="molecules"):
            g2l.parse_top(top)

    def test_angle_funct_unsupported_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(MINI_TOP.replace("1  2  3  1  109.500  836.8",
                                        "1  2  3  2  109.500  836.8"),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="functype"):
            g2l.parse_top(top)

    def test_duplicate_moleculetype_raises(self, tmp_path):
        top = tmp_path / "t.top"
        # 把第二个 moleculetype 也改名为 mol，触发重复定义
        top.write_text(MINI_TOP.replace("single  3", "mol  3"),
                       encoding="utf-8")
        with pytest.raises(ValueError, match="重复定义"):
            g2l.parse_top(top)

    def test_atoms_before_moleculetype_raises(self, tmp_path):
        top = tmp_path / "t.top"
        top.write_text(
            "[ atoms ]\n"
            "1  c3  1  MOL  C1  1  0.00000000  12.011\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="出现在任何"):
            g2l.parse_top(top)


class TestBuildSystem:
    def test_expansion_offsets_and_mol_ids(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        topo = g2l.parse_top(top)
        coords, box = g2l.parse_gro(gro)
        system = g2l.build_system(topo, coords, box)
        # 2 mol × 4 原子 + 2 single × 1 原子 = 10
        assert len(system.atoms) == 10
        # atoms: (mol_id, type_id, charge, x, y, z)
        assert system.atoms[0][0] == 1 and system.atoms[3][0] == 1
        assert system.atoms[4][0] == 2 and system.atoms[7][0] == 2
        assert system.atoms[8][0] == 3 and system.atoms[9][0] == 4
        assert system.atoms[0][1] == 1  # c3
        assert system.atoms[3][1] == 2  # hc
        assert system.atoms[1][2] == pytest.approx(-0.2)
        assert system.atoms[4][3] == pytest.approx(20.0)  # x Å
        # 第二个 mol 实例的键指向全局 id 5-8
        assert system.bonds[3] == (1, 5, 6)
        assert system.dihedrals[2][1:] == (5, 6, 7, 8)
        # 同参数合并同类型：3 条键两种参数 → 2 个 bond type
        assert len(system.bond_coeffs) == 2
        assert len(system.bonds) == 6
        assert len(system.angles) == 4
        assert len(system.dihedrals) == 4   # 2 term × 2 实例
        assert len(system.impropers) == 2

    def test_atom_count_mismatch_raises(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        gro.write_text(MINI_GRO.replace("10\n", "9\n", 1).replace(
            "    4SOL     H1   10   3.500   3.500   3.500\n", ""), encoding="utf-8")
        topo = g2l.parse_top(top)
        coords, box = g2l.parse_gro(gro)
        with pytest.raises(ValueError, match="不一致"):
            g2l.build_system(topo, coords, box)

    def test_zero_box_fallback(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        gro.write_text(MINI_GRO.replace(
            "   5.00000   5.00000   5.00000\n",
            "   0.00000   0.00000   0.00000\n"), encoding="utf-8")
        topo = g2l.parse_top(top)
        coords, box = g2l.parse_gro(gro)
        with pytest.warns(UserWarning, match="全为 0"):
            system = g2l.build_system(topo, coords, box)
        # x 范围 10..35 Å → 25 + 20 = 45 Å；y/z 同理
        assert system.box == pytest.approx([45.0, 45.0, 45.0])

    def test_atom_mass_override_warns(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        top.write_text(MINI_TOP.replace("1  c3  1  MOL  C1  1  0.10000000  12.011",
                                        "1  c3  1  MOL  C1  1  0.10000000  13.500"),
                       encoding="utf-8")
        topo = g2l.parse_top(top)
        coords, box = g2l.parse_gro(gro)
        with pytest.warns(UserWarning, match="质量"):
            g2l.build_system(topo, coords, box)


def _build_mini_system(tmp_path):
    top, gro = _write_mini(tmp_path)
    topo = g2l.parse_top(top)
    coords, box = g2l.parse_gro(gro)
    return topo, g2l.build_system(topo, coords, box)


class TestWriteNonbonded:
    def test_header_counts_and_box(self, tmp_path):
        topo, system = _build_mini_system(tmp_path)
        out = tmp_path / "out.data"
        g2l.write_lmp_data(system, topo.atom_types, topo.defaults, out)
        text = out.read_text(encoding="utf-8")
        for line in ("10 atoms", "6 bonds", "4 angles", "4 dihedrals",
                     "2 impropers", "2 atom types", "2 bond types",
                     "1 angle types", "2 dihedral types", "1 improper types",
                     "0.000000 50.000000 xlo xhi"):
            assert line in text, line

    def test_header_comment_block(self, tmp_path):
        topo, system = _build_mini_system(tmp_path)
        out = tmp_path / "out.data"
        g2l.write_lmp_data(system, topo.atom_types, topo.defaults, out)
        text = out.read_text(encoding="utf-8")
        assert "# pair_modify mix geometric" in text
        assert "# special_bonds lj 0.0 0.0 0.5 coul 0.0 0.0 0.8333" in text
        assert "# improper_style cvff" in text

    def test_masses_and_pair_coeffs(self, tmp_path):
        topo, system = _build_mini_system(tmp_path)
        out = tmp_path / "out.data"
        g2l.write_lmp_data(system, topo.atom_types, topo.defaults, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        mi = lines.index("Masses")
        assert lines[mi + 1] == ""  # 节头后必须空行（read_data 会吃掉下一行）
        assert lines[mi + 2].split() == ["1", "12.011"]
        assert lines[mi + 3].split() == ["2", "1.008"]
        pi = next(i for i, l in enumerate(lines) if l.startswith("Pair Coeffs"))
        # c3: ε=0.4184 kJ/mol → 0.1 kcal/mol；σ=0.34 nm → 3.4 Å
        assert lines[pi + 2].split() == ["1", "1.000000e-01", "3.400000"]
        # hc: ε=0.08368 kJ/mol → 0.02 kcal/mol；σ=0.26 nm → 2.6 Å
        assert lines[pi + 3].split() == ["2", "2.000000e-02", "2.600000"]

    def test_atoms_section_full_style(self, tmp_path):
        topo, system = _build_mini_system(tmp_path)
        out = tmp_path / "out.data"
        g2l.write_lmp_data(system, topo.atom_types, topo.defaults, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        ai = next(i for i, l in enumerate(lines) if l.startswith("Atoms"))
        rows = [l.split() for l in lines[ai + 2:ai + 12]]
        assert len(rows) == 10
        # id mol type q x y z
        assert rows[0][0] == "1" and rows[0][1] == "1" and rows[0][2] == "1"
        assert float(rows[0][3]) == pytest.approx(0.1)
        assert float(rows[0][4]) == pytest.approx(10.0)
        assert rows[8][1] == "3" and rows[9][1] == "4"  # single 分子的 mol_id


class TestBondedSections:
    def test_coeff_values(self, tmp_path):
        """MINI 体系的取整参数 → 精确换算值。

        bond k=83680 → K=83680/836.8=100.0, r0=0.15nm→1.5Å / 0.109nm→1.09Å
        angle k=836.8 → K=836.8/(2·4.184)=100.0, θ0=109.5
        dihedral phase=180 kd=4.184 pn=-3 → K=1.0, sign=+1（180→-1，pn<0 再翻）, n=3
        improper phase=0 kd=8.368 pn=2 → K=2.0, d=+1, n=2
        """
        topo, system = _build_mini_system(tmp_path)
        out = tmp_path / "out.data"
        g2l.write_lmp_data(system, topo.atom_types, topo.defaults, out)
        lines = out.read_text(encoding="utf-8").splitlines()

        bi = next(i for i, l in enumerate(lines) if l.startswith("Bond Coeffs"))
        assert lines[bi + 2].split() == ["1", "100.000", "1.5000"]
        assert lines[bi + 3].split() == ["2", "100.000", "1.0900"]

        ai = next(i for i, l in enumerate(lines) if l.startswith("Angle Coeffs"))
        assert lines[ai + 2].split() == ["1", "100.000", "109.500"]

        di = next(i for i, l in enumerate(lines) if l.startswith("Dihedral Coeffs"))
        assert lines[di + 2].split() == ["1", "1.000", "1", "3"]
        assert lines[di + 3].split() == ["2", "0.500", "1", "1"]  # kd=2.092 → 0.5

        ii = next(i for i, l in enumerate(lines) if l.startswith("Improper Coeffs"))
        assert lines[ii + 2].split() == ["1", "2.000", "1", "2"]

    def test_topology_sections(self, tmp_path):
        topo, system = _build_mini_system(tmp_path)
        out = tmp_path / "out.data"
        g2l.write_lmp_data(system, topo.atom_types, topo.defaults, out)
        lines = out.read_text(encoding="utf-8").splitlines()

        bi = next(i for i, l in enumerate(lines) if l == "Bonds")
        rows = [l.split() for l in lines[bi + 2:bi + 8]]
        # id type i j；第一实例键 1-2/2-3(type 1)、3-4(type 2)，第二实例 +4
        assert rows[0] == ["1", "1", "1", "2"]
        assert rows[2] == ["3", "2", "3", "4"]
        assert rows[3] == ["4", "1", "5", "6"]
        assert rows[5] == ["6", "2", "7", "8"]

        di = next(i for i, l in enumerate(lines) if l == "Dihedrals")
        rows = [l.split() for l in lines[di + 2:di + 6]]
        # 多 term 同四元组保留为多条目（type 1、2 各一条 × 2 实例）
        assert rows[0] == ["1", "1", "1", "2", "3", "4"]
        assert rows[1] == ["2", "2", "1", "2", "3", "4"]
        assert rows[2] == ["3", "1", "5", "6", "7", "8"]

        ii = next(i for i, l in enumerate(lines) if l == "Impropers")
        rows = [l.split() for l in lines[ii + 2:ii + 4]]
        assert rows[0] == ["1", "1", "1", "2", "3", "4"]
        assert rows[1] == ["2", "1", "5", "6", "7", "8"]

    def test_nonstandard_phase_warns(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        top.write_text(MINI_TOP.replace("9  180.0  4.184  -3", "9  90.0  4.184  3"),
                       encoding="utf-8")
        topo = g2l.parse_top(top)
        coords, box = g2l.parse_gro(gro)
        system = g2l.build_system(topo, coords, box)
        with pytest.warns(UserWarning, match="相位"):
            g2l.write_lmp_data(system, topo.atom_types, topo.defaults,
                               tmp_path / "out.data")


SCRIPT = Path(__file__).resolve().parent / "scripts" / "gmx2lmp_data.py"


class TestCli:
    def _run(self, *argv):
        return subprocess.run([sys.executable, str(SCRIPT), *argv],
                              capture_output=True, text=True)

    def test_smoke(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        out = tmp_path / "out.data"
        r = self._run("--top", str(top), "--gro", str(gro), "-o", str(out))
        assert r.returncode == 0, r.stderr
        assert out.exists()
        assert "10 atoms" in out.read_text(encoding="utf-8")

    def test_type_order_file(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        order = tmp_path / "type_order.txt"
        r = self._run("--top", str(top), "--gro", str(gro),
                      "-o", str(tmp_path / "out.data"),
                      "--type-order", str(order))
        assert r.returncode == 0, r.stderr
        assert order.read_text(encoding="utf-8") == "mini: 1=c3,2=hc\n"

    def test_bad_input_exit_2(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        top.write_text(MINI_TOP.split("[ molecules ]")[0], encoding="utf-8")
        r = self._run("--top", str(top), "--gro", str(gro),
                      "-o", str(tmp_path / "out.data"))
        assert r.returncode == 2
        assert "molecules" in r.stderr

    def test_zero_charge_warning(self, tmp_path):
        top, gro = _write_mini(tmp_path)
        zeroed = re.sub(r"(-?\d+\.\d{8})(  \d+\.\d+)?$", "0.00000000\\2",
                        MINI_TOP, flags=re.MULTILINE)
        top.write_text(zeroed, encoding="utf-8")
        r = self._run("--top", str(top), "--gro", str(gro),
                      "-o", str(tmp_path / "out.data"))
        assert r.returncode == 0, r.stderr
        assert "电荷为 0" in r.stderr
