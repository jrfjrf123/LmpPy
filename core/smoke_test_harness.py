"""
冒烟测试运行编排模块

功能:
- 创建临时目录并复制配置
- 覆盖 loop_num 运行 LAMMPSReactionRunner
- 调用 SmokeValidator 验证输出
- 清理临时目录（默认）或保留（--keep-output）

作者: Claude
日期: 2026-06-04
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

# 导入路径处理
try:
    from LmpPy.core.smoke_validator import SmokeValidator, SmokeTestReport
except ImportError:
    from core.smoke_validator import SmokeValidator, SmokeTestReport


class SmokeTestHarness:
    """
    冒烟测试编排器

    使用示例:
        harness = SmokeTestHarness("/path/to/config", loop_num=5)
        report = harness.run()
        report.print()
        sys.exit(0 if report.passed else 1)
    """

    def __init__(self,
                 source_config_dir: str,
                 loop_num: int = 5,
                 keep_output: bool = False):
        """
        Args:
            source_config_dir: 源配置目录路径（包含 system.yaml, lammps_params.yaml 等）
            loop_num: 覆盖的循环次数
            keep_output: 是否保留临时目录（True 则不删除，方便调试）
        """
        self.source_config_dir = Path(source_config_dir).resolve()
        self.loop_num = loop_num
        self.keep_output = keep_output
        self.temp_dir: Optional[Path] = None

        # 验证源目录存在
        if not self.source_config_dir.exists():
            raise FileNotFoundError(f"配置目录不存在: {self.source_config_dir}")
        if not self.source_config_dir.is_dir():
            raise NotADirectoryError(f"不是目录: {self.source_config_dir}")

    def run(self) -> SmokeTestReport:
        """
        执行冒烟测试的完整流程。

        Returns:
            SmokeTestReport
        """
        # 1. 创建临时目录
        self.temp_dir = Path(tempfile.mkdtemp(prefix="lmpy_smoke_"))
        print(f"\n临时目录: {self.temp_dir}")

        # 2. 复制配置目录到临时目录
        print(f"复制配置: {self.source_config_dir} -> {self.temp_dir}")

        def _ignore_patterns(directory, contents):
            return [c for c in contents if c in ('__pycache__', '.git', '.ipynb_checkpoints')]

        shutil.copytree(self.source_config_dir, self.temp_dir,
                        dirs_exist_ok=True, ignore=_ignore_patterns)

        original_cwd = Path.cwd()
        report = SmokeTestReport()

        try:
            # 3. 实例化 runner 并覆盖 loop_num
            try:
                from LmpPy.run_refactored import LAMMPSReactionRunner
            except ImportError:
                import sys
                _worktree_root = Path(__file__).resolve().parent.parent
                sys.path.insert(0, str(_worktree_root))
                from run_refactored import LAMMPSReactionRunner

            runner = LAMMPSReactionRunner(str(self.temp_dir))
            runner.lammps_params.loop_num = self.loop_num
            print(f"循环次数: {self.loop_num}")

            # 获取体系原子数和反应类型数（用于验证）
            n_atoms = len(runner.cg_compare_list.data) if runner.cg_compare_list is not None else None
            n_reaction_types = len(runner.lammps_params.reactions)

            # 4. 运行模拟
            print(f"\n开始运行 {self.loop_num} 个 loop...")
            runner.run()

            # 5. 验证
            print("\n开始验证...")
            report = SmokeValidator.validate(
                self.temp_dir,
                loop_num=self.loop_num,
                n_reaction_types=n_reaction_types,
                n_atoms=n_atoms,
            )

            # 补充报告上下文
            report.messages.insert(0, f"  配置目录: {self.source_config_dir}")
            report.messages.insert(1, f"  循环次数: {self.loop_num}")

        except Exception as e:
            report.passed = False
            report.messages = [
                f"  配置目录: {self.source_config_dir}",
                f"  循环次数: {self.loop_num}",
                f"  临时目录: {self.temp_dir}",
                "",
                f"    [❌] 运行异常",
                f"         {type(e).__name__}: {e}",
            ]
            import traceback
            traceback.print_exc()

        finally:
            # 6. 确保恢复工作目录
            try:
                os.chdir(original_cwd)
            except OSError:
                pass

            # 7. 清理临时目录
            if self.keep_output and self.temp_dir and self.temp_dir.exists():
                report.messages.insert(2, f"  临时目录: {self.temp_dir} (已保留)")
                print(f"\n测试输出保留在: {self.temp_dir}")
            elif self.temp_dir and self.temp_dir.exists():
                report.messages.insert(2, f"  临时目录: {self.temp_dir} (已删除)")
                shutil.rmtree(self.temp_dir, ignore_errors=True)

        return report
