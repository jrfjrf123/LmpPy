"""
键连表输出器模块

功能:
- 记录反应前后的键连表信息
- 记录 CG 拓扑变化 (bonds, angles, dihedrals)
- 仅在有反应发生时触发
- 保存为.npz文件格式

输出格式:
{
    'timestep': int,
    'run_step': int,
    'bonds_before': np.ndarray,
    'bonds_after': np.ndarray,
    'reaction_type': str,
}

作者: Claude
日期: 2026-03-26
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import numpy as np


@dataclass
class BondRecord:
    """单次反应的键记录"""
    timestep: int
    run_step: int
    bonds_before: np.ndarray  # (n_bonds, 3)
    bonds_after: np.ndarray   # (n_bonds, 3)
    reaction_type: str

    def save(self, output_dir: str, index: int = 0):
        """
        保存到.npz文件

        Args:
            output_dir: 输出目录
            index: 文件索引
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"bonds_record_{self.timestep}_{index}.npz"
        filepath = output_path / filename

        np.savez(
            filepath,
            timestep=self.timestep,
            run_step=self.run_step,
            bonds_before=self.bonds_before,
            bonds_after=self.bonds_after,
            reaction_type=self.reaction_type
        )

        return str(filepath)


@dataclass
class CGTopologyRecord:
    """
    单次反应的 CG 拓扑记录

    包含完整的 CG 拓扑变化: bonds, angles, dihedrals
    """
    timestep: int
    run_step: int
    cg_bonds_before: np.ndarray      # (n_bonds, 3)
    cg_bonds_after: np.ndarray       # (n_bonds, 3)
    cg_angles_before: np.ndarray     # (n_angles, 4)
    cg_angles_after: np.ndarray      # (n_angles, 4)
    cg_dihedrals_before: np.ndarray  # (n_dihedrals, 5)
    cg_dihedrals_after: np.ndarray   # (n_dihedrals, 5)
    reaction_type: str

    def save(self, output_dir: str, index: int = 0):
        """
        保存到.npz文件

        Args:
            output_dir: 输出目录
            index: 文件索引
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        filename = f"cg_topology_record_{self.timestep}_{index}.npz"
        filepath = output_path / filename

        np.savez(
            filepath,
            timestep=self.timestep,
            run_step=self.run_step,
            cg_bonds_before=self.cg_bonds_before,
            cg_bonds_after=self.cg_bonds_after,
            cg_angles_before=self.cg_angles_before,
            cg_angles_after=self.cg_angles_after,
            cg_dihedrals_before=self.cg_dihedrals_before,
            cg_dihedrals_after=self.cg_dihedrals_after,
            reaction_type=self.reaction_type
        )

        return str(filepath)


class BondsRecorder:
    """
    键连表记录器

    功能:
    - 记录反应前后的键连表
    - 记录 CG 拓扑变化
    - 批量保存到文件
    """

    def __init__(self, output_dir: str = "bonds_records"):
        """
        初始化记录器

        Args:
            output_dir: 输出目录
        """
        self.output_dir = output_dir
        self.records: List[BondRecord] = []
        self.cg_topology_records: List[CGTopologyRecord] = []  # 新增

    def record(self, timestep: int, run_step: int,
               bonds_before: np.ndarray, bonds_after: np.ndarray,
               reaction_type: str) -> BondRecord:
        """
        记录一次反应的键变化

        Args:
            timestep: 时间步
            run_step: 循环中的步数
            bonds_before: 反应前的键
            bonds_after: 反应后的键
            reaction_type: 反应类型

        Returns:
            BondRecord: 创建的记录
        """
        record = BondRecord(
            timestep=timestep,
            run_step=run_step,
            bonds_before=bonds_before.copy(),
            bonds_after=bonds_after.copy(),
            reaction_type=reaction_type
        )

        self.records.append(record)
        return record

    def save_all(self) -> List[str]:
        """
        保存所有记录

        Returns:
            保存的文件路径列表
        """
        paths = []
        for i, record in enumerate(self.records):
            path = record.save(self.output_dir, i)
            paths.append(path)

        return paths

    def save_incremental(self, record: BondRecord) -> str:
        """
        增量保存单个记录

        Args:
            record: 要保存的记录

        Returns:
            保存的文件路径
        """
        index = len(self.records)
        return record.save(self.output_dir, index)

    def clear(self):
        """清空记录"""
        self.records.clear()

    @property
    def n_records(self) -> int:
        """记录数量"""
        return len(self.records)

    def get_summary(self) -> dict:
        """
        获取统计摘要

        Returns:
            摘要字典
        """
        if not self.records:
            return {
                'n_records': 0,
                'timestep_range': (0, 0),
                'reaction_types': []
            }

        timesteps = [r.timestep for r in self.records]
        reaction_types = list(set(r.reaction_type for r in self.records))

        return {
            'n_records': len(self.records),
            'timestep_range': (min(timesteps), max(timesteps)),
            'reaction_types': reaction_types
        }

    def record_cg_topology(self, timestep: int, run_step: int,
                           cg_bonds_before: np.ndarray, cg_bonds_after: np.ndarray,
                           cg_angles_before: np.ndarray, cg_angles_after: np.ndarray,
                           cg_dihedrals_before: np.ndarray, cg_dihedrals_after: np.ndarray,
                           reaction_type: str) -> CGTopologyRecord:
        """
        记录一次反应的 CG 拓扑变化

        Args:
            timestep: 时间步
            run_step: 循环中的步数
            cg_bonds_before: 反应前的 CG 键
            cg_bonds_after: 反应后的 CG 键
            cg_angles_before: 反应前的 CG 角度
            cg_angles_after: 反应后的 CG 角度
            cg_dihedrals_before: 反应前的 CG 二面角
            cg_dihedrals_after: 反应后的 CG 二面角
            reaction_type: 反应类型

        Returns:
            CGTopologyRecord: 创建的记录
        """
        record = CGTopologyRecord(
            timestep=timestep,
            run_step=run_step,
            cg_bonds_before=cg_bonds_before.copy() if len(cg_bonds_before) > 0 else cg_bonds_before,
            cg_bonds_after=cg_bonds_after.copy() if len(cg_bonds_after) > 0 else cg_bonds_after,
            cg_angles_before=cg_angles_before.copy() if len(cg_angles_before) > 0 else cg_angles_before,
            cg_angles_after=cg_angles_after.copy() if len(cg_angles_after) > 0 else cg_angles_after,
            cg_dihedrals_before=cg_dihedrals_before.copy() if len(cg_dihedrals_before) > 0 else cg_dihedrals_before,
            cg_dihedrals_after=cg_dihedrals_after.copy() if len(cg_dihedrals_after) > 0 else cg_dihedrals_after,
            reaction_type=reaction_type
        )

        self.cg_topology_records.append(record)
        return record

    def save_all_cg_topology(self) -> List[str]:
        """
        保存所有 CG 拓扑记录

        Returns:
            保存的文件路径列表
        """
        paths = []
        for i, record in enumerate(self.cg_topology_records):
            path = record.save(self.output_dir, i)
            paths.append(path)

        return paths

    def clear_all(self):
        """清空所有记录"""
        self.records.clear()
        self.cg_topology_records.clear()

    @property
    def n_cg_topology_records(self) -> int:
        """CG 拓扑记录数量"""
        return len(self.cg_topology_records)


def save_bonds_record(output_path: str, timestep: int, run_step: int,
                      bonds_before: np.ndarray, bonds_after: np.ndarray,
                      reaction_type: str) -> str:
    """
    便捷函数：保存单次键记录

    Args:
        output_path: 输出文件路径 (.npz)
        timestep: 时间步
        run_step: 循环中的步数
        bonds_before: 反应前的键
        bonds_after: 反应后的键
        reaction_type: 反应类型

    Returns:
        保存的文件路径
    """
    np.savez(
        output_path,
        timestep=timestep,
        run_step=run_step,
        bonds_before=bonds_before,
        bonds_after=bonds_after,
        reaction_type=reaction_type
    )

    return output_path


def load_bonds_record(filepath: str) -> BondRecord:
    """
    加载键记录

    Args:
        filepath: 文件路径

    Returns:
        BondRecord对象
    """
    data = np.load(filepath, allow_pickle=True)

    return BondRecord(
        timestep=int(data['timestep']),
        run_step=int(data['run_step']),
        bonds_before=data['bonds_before'],
        bonds_after=data['bonds_after'],
        reaction_type=str(data['reaction_type'])
    )


if __name__ == "__main__":
    import tempfile
    import os

    print("=" * 60)
    print("测试键连表输出器")
    print("=" * 60)

    # 创建测试数据
    bonds_before = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [1, 3, 4],
    ], dtype=np.int32)

    bonds_after = np.array([
        [1, 1, 2],
        [1, 2, 3],
        [1, 3, 4],
        [2, 5, 6],  # 新键
    ], dtype=np.int32)

    # 测试记录器
    print("\n测试BondsRecorder:")
    recorder = BondsRecorder(output_dir="test_bonds_records")

    # 记录两次反应
    recorder.record(timestep=100, run_step=1,
                    bonds_before=bonds_before, bonds_after=bonds_after,
                    reaction_type="rxn1")

    recorder.record(timestep=200, run_step=2,
                    bonds_before=bonds_after, bonds_after=bonds_before,
                    reaction_type="rxn2")

    print(f"  记录数: {recorder.n_records}")

    summary = recorder.get_summary()
    print(f"  摘要: {summary}")

    # 保存
    paths = recorder.save_all()
    print(f"  保存路径: {paths[0] if paths else 'None'}")

    # 测试加载
    if paths:
        print("\n测试加载记录:")
        loaded = load_bonds_record(paths[0])
        print(f"  时间步: {loaded.timestep}")
        print(f"  反应类型: {loaded.reaction_type}")
        print(f"  反应前键数: {len(loaded.bonds_before)}")
        print(f"  反应后键数: {len(loaded.bonds_after)}")

        # 清理测试文件
        for p in paths:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists("test_bonds_records"):
            os.rmdir("test_bonds_records")

    print("\n✅ 测试完成!")