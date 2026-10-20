#!/usr/bin/env python3
"""CG → GROMACS 导出（cg_ff / cg_universe / convert_aa2cg --ff-yaml）的 pytest 测试。"""
import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from LmpPy.tools.aa2cg.cg_ff import (
    CgFFError, load_cg_ff, validate_ff_completeness,
    ANGSTROM_TO_NM, KCAL_TO_KJ, BOND_K_FACTOR, ANGLE_K_FACTOR, DIHEDRAL_K_FACTOR,
)
from LmpPy.tools.aa2cg.cg_universe import ensure_mdfmt, export_cg_gromacs

MDFMT_DEFAULT = Path("/home/jrf/PythonProject/MD_format_transform")


@pytest.fixture()
def mdfmt_ready():
    """mdfmt 不可用时跳过（非 PyPI 包，需指向 MD_format_transform 仓库）。"""
    try:
        return ensure_mdfmt(str(MDFMT_DEFAULT) if MDFMT_DEFAULT.exists() else None)
    except ImportError as e:
        pytest.skip(f"mdfmt 不可用: {e}")


def _two_chain_cg_data():
    """两条相同的 4-bead 链（类型 1-2-2-1），坐标/盒子单位为 Å。"""
    coords = np.array([
        [1.0, 1.0, 1.0],
        [2.0, 1.0, 1.0],
        [3.0, 1.0, 1.0],
        [4.0, 1.0, 1.0],
        [1.0, 5.0, 1.0],
        [2.0, 5.0, 1.0],
        [3.0, 5.0, 1.0],
        [4.0, 5.0, 1.0],
    ])
    return {
        'ids': np.arange(1, 9, dtype=np.int32),
        'mol_ids': np.array([1, 1, 1, 1, 2, 2, 2, 2], dtype=np.int32),
        'types': np.array([1, 2, 2, 1, 1, 2, 2, 1], dtype=np.int32),
        'coords': coords,
        'box': np.array([[0.0, 50.0], [0.0, 50.0], [0.0, 50.0]]),
        'bonds': np.array([
            [1, 1, 2], [2, 2, 3], [1, 3, 4],
            [1, 5, 6], [2, 6, 7], [1, 7, 8],
        ]),
        'angles': np.array([
            [1, 1, 2, 3], [1, 2, 3, 4],
            [1, 5, 6, 7], [1, 6, 7, 8],
        ]),
        'dihedrals': np.array([
            [1, 1, 2, 3, 4],
            [1, 5, 6, 7, 8],
        ]),
        'natoms': 8,
    }


FF_YAML = """
units: lammps-real
nrexcl: 3

atomtypes:
  - type: 1
    name: A
    mass: 54.09
    sigma: 4.42
    epsilon: 0.233
  - type: 2
    name: B
    sigma: 4.38
    epsilon: 0.200

nonbonded_overrides:
  - types: [1, 2]
    sigma: 4.40
    epsilon: 0.215

bonds:
  - bead_types: [1, 2]
    r0: 3.80
    k: 150.0
  - bead_types: [2, 2]
    r0: 4.00
    k: 120.0

angles:
  - bead_types: [1, 2, 2]
    theta0: 112.0
    k: 40.0

dihedrals:
  - bead_types: [1, 2, 2, 1]
    phi0: 180.0
    k: 3.0
    multiplicity: 1
"""


def _write_ff(tmp_path, text=FF_YAML):
    ff_path = tmp_path / "cg_ff.yaml"
    ff_path.write_text(text, encoding="utf-8")
    return ff_path


class TestLoadCgFF:
    def test_unit_conversion(self, tmp_path):
        ff = load_cg_ff(_write_ff(tmp_path), auto_masses={2: 13.0})
        assert ff.sigma[1] == pytest.approx(4.42 * ANGSTROM_TO_NM)
        assert ff.epsilon[1] == pytest.approx(0.233 * KCAL_TO_KJ)
        assert ff.atom_masses[1] == pytest.approx(54.09)
        # mass 缺省回退 auto_masses
        assert ff.atom_masses[2] == pytest.approx(13.0)
        # 键 k：LAMMPS E=K(r-r0)^2 → GROMACS E=1/2 k(r-b0)^2，含 ×2
        assert ff.bond_params[(1, 2)] == pytest.approx((3.80 * ANGSTROM_TO_NM,
                                                        150.0 * BOND_K_FACTOR))
        assert ff.angle_params[(1, 2, 2)] == pytest.approx((112.0, 40.0 * ANGLE_K_FACTOR))
        assert ff.dihedral_params[(1, 2, 2, 1)] == pytest.approx(
            (180.0, 3.0 * DIHEDRAL_K_FACTOR, 1))

    def test_combo_symmetry_normalization(self, tmp_path):
        """反向书写的组合应规范化到同一键。"""
        ff = load_cg_ff(_write_ff(tmp_path), auto_masses={2: 13.0})
        assert (2, 1) not in ff.bond_params
        assert (1, 2) in ff.bond_params

    def test_nb_override_c6_c12(self, tmp_path):
        ff = load_cg_ff(_write_ff(tmp_path), auto_masses={2: 13.0})
        sig = 4.40 * ANGSTROM_TO_NM
        eps = 0.215 * KCAL_TO_KJ
        c6, c12 = ff.nb_overrides[(1, 2)]
        assert c6 == pytest.approx(4.0 * eps * sig ** 6)
        assert c12 == pytest.approx(4.0 * eps * sig ** 12)

    def test_duplicate_atomtype_rejected(self, tmp_path):
        bad = FF_YAML.replace("  - type: 2\n    name: B", "  - type: 1\n    name: C")
        with pytest.raises(CgFFError, match="重复"):
            load_cg_ff(_write_ff(tmp_path, bad), auto_masses={1: 1.0})

    def test_missing_mass_without_auto_rejected(self, tmp_path):
        with pytest.raises(CgFFError, match="mass"):
            load_cg_ff(_write_ff(tmp_path))

    def test_completeness_reports_all_missing(self, tmp_path):
        ff = load_cg_ff(_write_ff(tmp_path), auto_masses={2: 13.0})
        # 四类各给一项确实缺失的，校验应一次性全部列出
        with pytest.raises(CgFFError, match="dihedrals") as exc_info:
            validate_ff_completeness(ff, {1, 2, 3}, [(1, 1)], [(1, 1, 1)],
                                     [(1, 1, 1, 1)])
        msg = str(exc_info.value)
        assert "bead type 3" in msg
        assert "bonds: 组合 (1, 1)" in msg
        assert "angles: 组合 (1, 1, 1)" in msg
        assert "dihedrals: 组合 (1, 1, 1, 1)" in msg


class TestExport:
    def test_roundtrip(self, tmp_path, mdfmt_ready):
        cg_data = _two_chain_cg_data()
        ff_path = _write_ff(tmp_path)
        out = tmp_path / "gmx"
        paths = export_cg_gromacs(
            cg_data, ff_path, out,
            auto_masses={2: 13.0}, title="test_sys",
            mdfmt_path=str(MDFMT_DEFAULT),
        )
        assert paths['gro'].exists() and paths['top'].exists() and paths['itp'].exists()

        # --- gro 读回：原子数、坐标 Å→nm、盒子 ---
        from mdfmt.io.gro import read_gro
        _, frame = read_gro(str(paths['gro']))
        assert frame.coords.shape == (8, 3)
        assert frame.coords[0] == pytest.approx(cg_data['coords'][0] * 0.1)
        assert frame.coords[7] == pytest.approx(cg_data['coords'][7] * 0.1)

        gro_text = paths['gro'].read_text(encoding="utf-8")
        box_vals = [float(v) for v in gro_text.strip().splitlines()[-1].split()]
        assert box_vals[:3] == pytest.approx([5.0, 5.0, 5.0])

        # --- top：comb-rule 3、gen-pairs no、atomtypes 换算、nonbond_params、[molecules] ---
        top_text = paths['top'].read_text(encoding="utf-8")
        assert '#include "cg.itp"' in top_text
        defaults_line = top_text.split("[ defaults ]")[1].splitlines()[2]
        assert defaults_line.split()[:3] == ['1', '3', 'no']
        assert f"{54.09:10.4f}" in top_text
        assert f"{4.42 * ANGSTROM_TO_NM:14.6e}" in top_text
        assert f"{0.233 * KCAL_TO_KJ:14.6e}" in top_text
        sig, eps = 4.40 * ANGSTROM_TO_NM, 0.215 * KCAL_TO_KJ
        assert "[ nonbond_params ]" in top_text
        assert f"{4.0 * eps * sig ** 6:18.6e}" in top_text
        assert f"{4.0 * eps * sig ** 12:18.6e}" in top_text
        # 两条相同链归并为一个 moleculetype，count=2
        mol_section = top_text.split("[ molecules ]")[1]
        assert "CG" in mol_section and "2" in mol_section.split()[-1]

        # --- itp：单 moleculetype、funct=1、参数换算（键 k 含 ×2）---
        itp_text = paths['itp'].read_text(encoding="utf-8")
        assert itp_text.count("[ moleculetype ]") == 1
        assert "CG   3" in itp_text  # nrexcl
        bond_lines = itp_text.split("[ bonds ]")[1].split("[ angles ]")[0]
        bond_rows = [l for l in bond_lines.splitlines()
                     if l.strip() and not l.strip().startswith(";")]
        assert len(bond_rows) == 3
        first = bond_rows[0].split()
        assert first[:3] == ['1', '2', '1']
        assert float(first[3]) == pytest.approx(3.80 * ANGSTROM_TO_NM)
        assert float(first[4]) == pytest.approx(150.0 * BOND_K_FACTOR)
        dih_lines = itp_text.split("[ dihedrals ]")[1]
        dih_rows = [l for l in dih_lines.splitlines()
                    if l.strip() and not l.strip().startswith(";")]
        assert len(dih_rows) == 1
        dih = dih_rows[0].split()
        assert dih[:5] == ['1', '2', '3', '4', '1']
        assert float(dih[5]) == pytest.approx(180.0)
        assert float(dih[6]) == pytest.approx(3.0 * DIHEDRAL_K_FACTOR)
        assert int(dih[7]) == 1

    def test_incomplete_ff_rejected(self, tmp_path, mdfmt_ready):
        no_dih = FF_YAML.split("dihedrals:")[0]
        with pytest.raises(CgFFError, match="dihedrals"):
            export_cg_gromacs(_two_chain_cg_data(), _write_ff(tmp_path, no_dih),
                              tmp_path / "gmx", auto_masses={2: 13.0},
                              mdfmt_path=str(MDFMT_DEFAULT))

    def test_non_contiguous_mol_ids_rejected(self, tmp_path, mdfmt_ready):
        cg_data = _two_chain_cg_data()
        # mol_id 1 的 bead 被 mol_id 2 隔断
        cg_data['mol_ids'] = np.array([1, 1, 2, 2, 1, 1, 2, 2], dtype=np.int32)
        with pytest.raises(ValueError, match="不连续"):
            export_cg_gromacs(cg_data, _write_ff(tmp_path), tmp_path / "gmx",
                              auto_masses={2: 13.0},
                              mdfmt_path=str(MDFMT_DEFAULT))
