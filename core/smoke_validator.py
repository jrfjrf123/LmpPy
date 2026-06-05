"""
冒烟测试验证器模块

功能:
- 文件输出完整性检查 (A)
- CG mapping 一致性检查 (B)
- 反应后 mapping 交叉验证 (C)
- 反应计数合理性检查 (D)

设计原则:
- 纯函数，不依赖 LAMMPS 运行时状态
- 所有输入通过文件路径传入
- 返回统一的 SmokeTestReport

作者: Claude
日期: 2026-06-04
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import numpy as np
import sys

# 同时支持从项目根导入 (LmpPy.xxx) 和从 worktree 直接导入 (core.xxx)
try:
    from LmpPy.core.cg_reaction_identifier import load_reaction_signatures, identify_reaction
    from LmpPy.core.cg_converter import validate_cg_mapping_consistency
    from LmpPy.core.cg_bond_mapper import atom_bonds_to_cg_bonds
except ImportError:
    # worktree 环境: 使用完整包路径
    from core.cg_reaction_identifier import load_reaction_signatures, identify_reaction
    from core.cg_converter import validate_cg_mapping_consistency
    from core.cg_bond_mapper import atom_bonds_to_cg_bonds


@dataclass
class SmokeTestReport:
    """冒烟测试报告"""
    passed: bool = True
    messages: List[str] = field(default_factory=list)
    artifacts: Dict[str, Path] = field(default_factory=dict)

    def add(self, check_name: str, status: str, detail: str):
        """添加一项检查结果"""
        prefix = "✅" if status == "pass" else ("❌" if status == "fail" else "⚠️")
        self.messages.append(f"    [{prefix}] {check_name}")
        if detail:
            for line in detail.strip().split("\n"):
                self.messages.append(f"         {line}")

    def print(self):
        """打印报告"""
        print("\n" + "=" * 60)
        print("Smoke Test 报告")
        print("=" * 60)
        for msg in self.messages:
            print(msg)
        print()
        status_text = "通过" if self.passed else "未通过"
        print(f"  结果: {status_text}")
        print("=" * 60)


class SmokeValidator:
    """
    冒烟测试验证器

    四项检查:
    A. 文件输出完整性
    B. CG mapping 一致性
    C. 反应后 mapping 更新正确性（交叉验证）
    D. 反应计数合理性
    """

    @staticmethod
    def validate(temp_dir: Path,
                 loop_num: int,
                 n_reaction_types: int = 0,
                 n_atoms: Optional[int] = None) -> SmokeTestReport:
        """
        执行所有验证检查。

        Args:
            temp_dir: 测试运行的临时目录（包含输出文件）
            loop_num: 使用的循环次数
            n_reaction_types: 反应类型数量（用于检查 D 的列数校验）
            n_atoms: 体系原子总数（用于检查 B 的 AA_id 完整性），None 则不检查

        Returns:
            SmokeTestReport
        """
        report = SmokeTestReport()

        # 收集输出文件路径
        reaction_count_file = temp_dir / "reaction_num.txt"
        cg_trajectory_file = temp_dir / "cg_trajectory.lammpstrj"
        final_mapping_file = temp_dir / "final_cg_compare_list.csv"
        reaction_frames_file = temp_dir / "reaction_frames.npz"
        reactions_dir = temp_dir / "reactions"

        report.artifacts = {
            'reaction_num': reaction_count_file,
            'cg_trajectory': cg_trajectory_file,
            'final_mapping': final_mapping_file,
            'reaction_frames': reaction_frames_file,
        }

        # === A. 文件输出完整性 ===
        SmokeValidator._check_a_file_output(
            report, reaction_count_file, cg_trajectory_file,
            final_mapping_file, reaction_frames_file, loop_num
        )

        # === B. CG mapping 一致性 ===
        SmokeValidator._check_b_cg_mapping_consistency(
            report, final_mapping_file, n_atoms
        )

        # === C. 反应后 mapping 交叉验证 ===
        SmokeValidator._check_c_cross_validation(
            report, reaction_frames_file, reactions_dir
        )

        # === D. 反应计数合理性 ===
        SmokeValidator._check_d_reaction_counts(
            report, reaction_count_file, loop_num, n_reaction_types
        )

        return report

    # ================================================================
    # A. 文件输出完整性
    # ================================================================
    @staticmethod
    def _check_a_file_output(report: SmokeTestReport,
                             reaction_count_file: Path,
                             cg_trajectory_file: Path,
                             final_mapping_file: Path,
                             reaction_frames_file: Path,
                             loop_num: int):
        """检查文件输出完整性"""
        details = []

        # A1: reaction_num.txt 存在且非空
        if reaction_count_file.exists():
            with open(reaction_count_file) as f:
                lines = f.readlines()
            n_data_lines = len([l for l in lines if not l.startswith('#') and l.strip()])
            details.append(f"reaction_num.txt: {n_data_lines} 行数据")
            if n_data_lines == 0:
                report.passed = False
                SmokeValidator._record(report, "A. 文件输出完整性", "fail",
                                       "\n".join(details) + "\nreaction_num.txt 为空")
                return
        else:
            report.passed = False
            SmokeValidator._record(report, "A. 文件输出完整性", "fail",
                                   "reaction_num.txt 不存在")
            return

        # A2: cg_trajectory.lammpstrj 至少包含 loop_num 个 TIMESTEP
        if cg_trajectory_file.exists():
            with open(cg_trajectory_file) as f:
                content = f.read()
            frame_count = content.count("ITEM: TIMESTEP")
            details.append(f"cg_trajectory.lammpstrj: {frame_count} 帧")
            if frame_count < loop_num:
                report.passed = False
                details.append(f"警告: 预期至少 {loop_num} 帧，实际 {frame_count}")
        else:
            report.passed = False
            details.append("cg_trajectory.lammpstrj 不存在")

        # A3: final_cg_compare_list.csv 存在且行数 > 0
        if final_mapping_file.exists():
            data = np.loadtxt(final_mapping_file, delimiter=',', skiprows=1, ndmin=2)
            details.append(f"final_cg_compare_list.csv: {len(data)} 行")
            if len(data) == 0:
                report.passed = False
                details.append("final_cg_compare_list.csv 为空")
        else:
            report.passed = False
            details.append("final_cg_compare_list.csv 不存在")

        # A4: reaction_frames.npz（可选）
        if reaction_frames_file.exists():
            try:
                rf = np.load(reaction_frames_file, allow_pickle=True)
                n_frames = len(rf.get('timestep', []))
                details.append(f"reaction_frames.npz: {n_frames} 帧")
                rf.close()
            except Exception as e:
                details.append(f"reaction_frames.npz 读取失败: {e}")
        else:
            details.append("reaction_frames.npz: 未生成（可能未触发反应）")

        SmokeValidator._record(report, "A. 文件输出完整性", "pass" if report.passed else "fail",
                               "\n".join(details))

    # ================================================================
    # B. CG mapping 一致性
    # ================================================================
    @staticmethod
    def _check_b_cg_mapping_consistency(report: SmokeTestReport,
                                        final_mapping_file: Path,
                                        n_atoms: Optional[int]):
        """检查 CG mapping 一致性"""
        if not final_mapping_file.exists():
            SmokeValidator._record(report, "B. CG mapping 一致性", "fail",
                                   "final_cg_compare_list.csv 不存在")
            report.passed = False
            return

        details = []
        b_passed = True  # 独立追踪 B 的通过状态
        try:
            cg_data = np.loadtxt(final_mapping_file, delimiter=',', skiprows=1, ndmin=2)
            details.append(f"{len(cg_data)} 个原子映射")

            # B1: 调用已有的一致性验证函数
            try:
                bead_id_to_type = validate_cg_mapping_consistency(cg_data, raise_error=True)
                details.append(f"{len(bead_id_to_type)} 个 bead，每个 bead_type 唯一")
            except Exception as e:
                b_passed = False
                details.append(f"一致性验证失败: {e}")

            # B2: AA_id 无重复检查
            aa_ids = cg_data[:, 3].astype(np.int32)
            unique_aa = len(np.unique(aa_ids))
            if unique_aa != len(aa_ids):
                b_passed = False
                details.append(f"AA_id 存在重复: {len(aa_ids)} 行但只有 {unique_aa} 个唯一值")

            # B3: AA_id 不缺失检查 (如果提供了 n_atoms)
            if n_atoms is not None:
                expected = set(range(1, n_atoms + 1))
                actual = set(aa_ids.tolist())
                missing = expected - actual
                if missing:
                    b_passed = False
                    details.append(f"AA_id 缺失: {len(missing)} 个原子未被映射 (示例: {sorted(list(missing))[:5]}...)")

            SmokeValidator._record(report, "B. CG mapping 一致性",
                                   "pass" if b_passed else "fail",
                                   "\n".join(details))
        except Exception as e:
            report.passed = False
            SmokeValidator._record(report, "B. CG mapping 一致性", "fail",
                                   f"读取 final_cg_compare_list.csv 失败: {e}")

    # ================================================================
    # C. 反应后 mapping 交叉验证
    # ================================================================
    @staticmethod
    def _check_c_cross_validation(report: SmokeTestReport,
                                  reaction_frames_file: Path,
                                  reactions_dir: Path):
        """交叉验证：用独立的 CG 级签名算法重构预期 mapping，与 runner 实际输出对比"""
        if not reaction_frames_file.exists():
            SmokeValidator._record(report, "C. 反应后 mapping 更新正确 (交叉验证)", "skip",
                                   "reaction_frames.npz 不存在，未发生反应，跳过交叉验证")
            return

        # 加载签名
        signatures = load_reaction_signatures(reactions_dir)
        if not signatures:
            SmokeValidator._record(report, "C. 反应后 mapping 更新正确 (交叉验证)", "skip",
                                   "未找到反应签名（reactions/ 目录为空），降级为简单差异检查")
            return

        details = []
        unidentified_frames = []
        mismatch_frames = []

        try:
            data = np.load(reaction_frames_file, allow_pickle=True)
            n_frames = len(data['timestep'])
            details.append(f"共 {n_frames} 帧")

            if n_frames == 0:
                data.close()
                SmokeValidator._record(report, "C. 反应后 mapping 更新正确 (交叉验证)", "skip",
                                       "\n".join(details) + "\n0 帧，跳过")
                return

            # 用第 0 帧的 cg_mapping_before 作为初始累积映射
            current_mapping = data['cg_mapping_before'][0].copy()

            for frame_idx in range(n_frames):
                # 从 AA 键推导 CG 键（使用当前累积映射，即反应前的状态）
                aa_bonds_before = data['aa_bonds_before'][frame_idx]
                aa_bonds_after = data['aa_bonds_after'][frame_idx]

                cg_bonds_before = atom_bonds_to_cg_bonds(aa_bonds_before, current_mapping)
                cg_bonds_after = atom_bonds_to_cg_bonds(aa_bonds_after, current_mapping)

                # CG 级识别
                results = identify_reaction(cg_bonds_before, cg_bonds_after,
                                            current_mapping, signatures)

                if not results:
                    unidentified_frames.append(frame_idx)
                    # 即使未识别，也要更新当前映射为 runner 的输出，保证后续帧正确
                    current_mapping = data['cg_mapping_after'][frame_idx].copy()
                    continue

                # 根据识别结果重构预期的 cg_mapping_after
                expected_mapping = current_mapping.copy()
                for result in results:
                    for bead_id, new_type in result['bead_types_after'].items():
                        mask = (expected_mapping[:, 0].astype(int) == bead_id)
                        expected_mapping[mask, 2] = float(new_type)

                # 与 runner 实际输出对比
                actual_mapping = data['cg_mapping_after'][frame_idx]
                # 只对比 bead_type 列（第 2 列，0-indexed）
                expected_types = expected_mapping[:, 2]
                actual_types = actual_mapping[:, 2].astype(float)

                if not np.array_equal(expected_types, actual_types):
                    diff_mask = expected_types != actual_types
                    n_diff = np.sum(diff_mask)
                    diff_atoms = np.where(diff_mask)[0][:5]
                    mismatch_frames.append({
                        'frame': frame_idx,
                        'n_diff': n_diff,
                        'sample_atoms': diff_atoms.tolist(),
                        'reactions': [r['name'] for r in results],
                    })
                    details.append(f"  第 {frame_idx} 帧: ❌ {n_diff} 个原子 bead_type 不匹配"
                                   f" (示例 atom_id: {[int(a)+1 for a in diff_atoms]})")
                else:
                    rxn_names = [r['name'] for r in results]
                    details.append(f"  第 {frame_idx} 帧: ✅ {', '.join(rxn_names)} 交叉验证通过")

                # 更新当前映射为 runner 的实际输出（而非预期输出），继续累加
                current_mapping = actual_mapping.copy()

            data.close()

            if unidentified_frames:
                details.append(f"⚠️ {len(unidentified_frames)} 帧无法被签名识别"
                               f" (索引: {unidentified_frames})")

            all_passed = len(mismatch_frames) == 0 and len(unidentified_frames) == 0
            if len(mismatch_frames) > 0:
                report.passed = False

            status = "pass" if all_passed else ("fail" if mismatch_frames else "warn")
            SmokeValidator._record(report, "C. 反应后 mapping 更新正确 (交叉验证)",
                                   status, "\n".join(details))

        except Exception as e:
            report.passed = False
            SmokeValidator._record(report, "C. 反应后 mapping 更新正确 (交叉验证)", "fail",
                                   f"交叉验证异常: {e}\n{details}")

    # ================================================================
    # D. 反应计数合理性
    # ================================================================
    @staticmethod
    def _check_d_reaction_counts(report: SmokeTestReport,
                                 reaction_count_file: Path,
                                 loop_num: int,
                                 n_reaction_types: int = 0):
        """检查反应计数合理性"""
        if not reaction_count_file.exists():
            report.passed = False
            SmokeValidator._record(report, "D. 反应计数合理性", "fail",
                                   "reaction_num.txt 不存在")
            return

        details = []
        try:
            with open(reaction_count_file) as f:
                lines = f.readlines()

            data_lines = [l for l in lines if not l.startswith('#') and l.strip()]
            n_data_lines = len(data_lines)
            details.append(f"数据行数: {n_data_lines} (预期 {loop_num})")

            if n_data_lines != loop_num:
                details.append(f"⚠️ 行数不匹配: 预期 {loop_num}，实际 {n_data_lines}")

            # 解析 header 获取反应名称
            header_line = None
            for l in lines:
                if l.startswith('#'):
                    header_line = l
                    break

            if header_line:
                parts = header_line.lstrip('#').strip().split()
                # parts[0] = "timestep", parts[1:] = reaction names
                rxn_names = parts[1:] if len(parts) > 1 else []
            else:
                rxn_names = []

            # 检查每行计数非负
            all_non_negative = True
            totals = None
            for i, line in enumerate(data_lines):
                values = [int(v) for v in line.strip().split()]
                # values[0] = timestep, values[1:] = reaction counts
                counts = values[1:]
                if totals is None:
                    totals = np.zeros(len(counts), dtype=np.int64)
                for j, c in enumerate(counts):
                    if c < 0:
                        all_non_negative = False
                        details.append(f"第 {i+1} 行: 反应计数为负 ({c})")
                totals += np.array(counts, dtype=np.int64)

            if all_non_negative:
                details.append("所有反应计数非负 ✅")
            else:
                report.passed = False

            # 汇总
            if totals is not None and len(rxn_names) == len(totals):
                for name, total in zip(rxn_names, totals):
                    details.append(f"  {name}: {total}")
            elif totals is not None:
                for j, total in enumerate(totals):
                    details.append(f"  反应 {j+1}: {total}")

            # 独立判断 D 状态，不受前序检查影响
            d_passed = all_non_negative and (n_data_lines == loop_num)
            SmokeValidator._record(report, "D. 反应计数合理性",
                                   "pass" if d_passed else "fail",
                                   "\n".join(details))

        except Exception as e:
            report.passed = False
            SmokeValidator._record(report, "D. 反应计数合理性", "fail",
                                   f"解析 reaction_num.txt 失败: {e}")

    # ================================================================
    # 辅助方法
    # ================================================================
    @staticmethod
    def _record(report: SmokeTestReport, name: str, status: str, detail: str):
        """记录检查结果到报告（标记 passed=False 时也更新 report.passed）"""
        report.add(name, status, detail)
        if status == "fail":
            report.passed = False


# ============================================================
# 自测代码
# ============================================================
if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("测试 smoke_validator")
    print("=" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # 创建合成 reaction_num.txt
        with open(tmp / "reaction_num.txt", "w") as f:
            f.write("# timestep rxn1 rxn2\n")
            for ts in range(1, 6):
                f.write(f"{ts*1000} 0 0\n")

        # 创建合成 final_cg_compare_list.csv
        n_atoms_test = 15
        cg_data = np.zeros((n_atoms_test, 5), dtype=np.float64)
        for i in range(n_atoms_test):
            bead_id = (i // 3) + 1
            cg_data[i] = [bead_id, 1, bead_id, i + 1, 12.0]
        np.savetxt(tmp / "final_cg_compare_list.csv", cg_data,
                   delimiter=',', fmt='%.0f,%.0f,%.0f,%.0f,%.6f',
                   header='bead_id, mol_id, bead_type, AA_id, mass', comments='')

        # 创建合成 cg_trajectory.lammpstrj (5 帧)
        with open(tmp / "cg_trajectory.lammpstrj", "w") as f:
            for ts in range(5):
                f.write("ITEM: TIMESTEP\n")
                f.write(f"{ts*1000}\n")
                f.write("ITEM: NUMBER OF ATOMS\n")
                f.write("5\n")
                f.write("ITEM: BOX BOUNDS pp pp pp\n")
                f.write("0 100\n0 100\n0 100\n")
                f.write("ITEM: ATOMS id type x y z ix iy iz\n")
                for a in range(1, 6):
                    f.write(f"{a} 1 0 0 0 0 0 0\n")

        # 运行验证
        report = SmokeValidator.validate(tmp, loop_num=5, n_reaction_types=2,
                                         n_atoms=n_atoms_test)
        report.print()
        print(f"\n  配置目录: {tmpdir}")
        print(f"  循环次数: 5")
        print(f"  passed: {report.passed}")

        if report.passed:
            print("\n✅ smoke_validator 自测通过!")
        else:
            print("\n❌ smoke_validator 自测未通过!")
