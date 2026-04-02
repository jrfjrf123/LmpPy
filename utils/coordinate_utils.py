"""
坐标处理工具函数

功能:
- wrap_coordinates: 将坐标映射回盒子内
- unwrap_molecules: 展开分子坐标
- pbc_distance: 计算周期性边界条件下的距离

作者: Claude
日期: 2026-03-26
"""

import numpy as np
from typing import Tuple, Optional


def wrap_coordinates(coords: np.ndarray, box: np.ndarray,
                     return_images: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    根据周期性边界条件将原子坐标映射回盒子内

    参数:
        coords: 原子坐标数组，shape=(n, 3)
        box: 盒子定义，shape=(3, 2)，每行表示一个维度的[min, max]
        return_images: 是否返回image标记

    返回:
        wrapped_coords: 映射后的坐标，shape=(n, 3)
        images: image标记数组，shape=(n, 3) (如果return_images=True)
    """
    # 计算盒子尺寸
    box_size = box[:, 1] - box[:, 0]  # [Lx, Ly, Lz]

    # 将坐标平移到以盒子左下角为原点
    shifted_coords = coords - box[:, 0]

    # 计算image标记（跨越盒子的次数）
    images = np.floor(shifted_coords / box_size).astype(int)

    # 将坐标映射回盒子内
    wrapped_coords = shifted_coords - images * box_size

    # 将坐标平移回原始坐标系
    wrapped_coords = wrapped_coords + box[:, 0]

    if return_images:
        return wrapped_coords, images
    else:
        return wrapped_coords, None


def pbc_distance(coord1: np.ndarray, coord2: np.ndarray,
                 box: np.ndarray) -> np.ndarray:
    """
    计算周期性边界条件下的最小镜像距离

    参数:
        coord1: 第一个坐标，shape=(3,) 或 (n, 3)
        coord2: 第二个坐标，shape=(3,) 或 (n, 3)
        box: 盒子定义，shape=(3, 2)

    返回:
        delta: 从coord1到coord2的位移向量
    """
    delta = coord2 - coord1
    box_size = box[:, 1] - box[:, 0]

    # 应用最小镜像约定
    delta = delta - box_size * np.round(delta / box_size)

    return delta


def unwrap_molecule_bfs(start_atom: int, coords: np.ndarray,
                        graph: dict, box: np.ndarray,
                        visited: set) -> Tuple[np.ndarray, set]:
    """
    使用BFS展开单个分子的坐标

    参数:
        start_atom: 起始原子ID
        coords: 坐标字典 {atom_id: [x, y, z]}
        graph: 键连图 {atom_id: [neighbor_ids]}
        box: 盒子定义，shape=(3, 2)
        visited: 已访问原子集合

    返回:
        unwrapped_coords: 展开后的坐标字典
        visited: 更新后的已访问集合
    """
    from collections import deque

    unwrapped = {}
    queue = deque([start_atom])
    visited.add(start_atom)
    unwrapped[start_atom] = coords[start_atom].copy()

    box_size = box[:, 1] - box[:, 0]

    while queue:
        current = queue.popleft()

        for neighbor in graph[current]:
            if neighbor not in visited:
                visited.add(neighbor)

                # 计算最小镜像位移
                delta = coords[neighbor] - unwrapped[current]
                delta = delta - box_size * np.round(delta / box_size)

                # 展开后的坐标
                unwrapped[neighbor] = unwrapped[current] + delta

                queue.append(neighbor)

    return unwrapped, visited


def unwrap_coords_numba(coords: np.ndarray, bonds: np.ndarray,
                        box: np.ndarray, molecule_ids: np.ndarray) -> np.ndarray:
    """
    使用Numba优化的坐标展开 (需要外部JIT编译)

    参数:
        coords: 坐标数组，shape=(n_atoms, 3)
        bonds: 键连数组，shape=(n_bonds, 2)
        box: 盒子定义，shape=(3, 2)
        molecule_ids: 分子ID数组，shape=(n_atoms,)

    返回:
        unwrapped_coords: 展开后的坐标数组
    """
    try:
        from numba import njit

        @njit(cache=True)
        def _unwrap_core(coords, bonds, box, molecule_ids, box_size):
            n_atoms = len(coords)
            unwrapped = coords.copy()

            # 按分子展开
            max_mol_id = molecule_ids.max()
            for mol_id in range(1, max_mol_id + 1):
                # 找到该分子所有原子
                mol_atoms = []
                for i in range(n_atoms):
                    if molecule_ids[i] == mol_id:
                        mol_atoms.append(i)

                if len(mol_atoms) <= 1:
                    continue

                # BFS展开
                visited = [False] * n_atoms
                queue = [0] * n_atoms
                head = 0
                tail = 0

                start = mol_atoms[0]
                queue[tail] = start
                tail += 1
                visited[start] = True

                while head < tail:
                    current = queue[head]
                    head += 1

                    # 查找邻居
                    for bond_idx in range(len(bonds)):
                        a1, a2 = bonds[bond_idx]
                        neighbor = -1
                        if a1 == current + 1 and not visited[a2 - 1]:
                            neighbor = a2 - 1
                        elif a2 == current + 1 and not visited[a1 - 1]:
                            neighbor = a1 - 1

                        if neighbor >= 0 and molecule_ids[neighbor] == mol_id:
                            visited[neighbor] = True
                            queue[tail] = neighbor
                            tail += 1

                            # 计算最小镜像位移
                            delta = coords[neighbor] - unwrapped[current]
                            for d in range(3):
                                delta[d] = delta[d] - box_size[d] * round(delta[d] / box_size[d])

                            unwrapped[neighbor] = unwrapped[current] + delta

            return unwrapped

        box_size = box[:, 1] - box[:, 0]
        return _unwrap_core(coords, bonds, box, molecule_ids, box_size)

    except ImportError:
        # Numba不可用时回退到Python实现
        return unwrap_coords_python(coords, bonds, box, molecule_ids)


def unwrap_coords_python(coords: np.ndarray, bonds: np.ndarray,
                         box: np.ndarray, molecule_ids: np.ndarray) -> np.ndarray:
    """
    Python实现的坐标展开

    参数:
        coords: 坐标数组，shape=(n_atoms, 3)
        bonds: 键连数组，shape=(n_bonds, 2)，原子ID从1开始
        box: 盒子定义，shape=(3, 2)
        molecule_ids: 分子ID数组，shape=(n_atoms,)

    返回:
        unwrapped_coords: 展开后的坐标数组
    """
    from collections import defaultdict, deque

    box_size = box[:, 1] - box[:, 0]
    n_atoms = len(coords)
    unwrapped = coords.copy()

    # 构建邻接图
    graph = defaultdict(list)
    for bond in bonds:
        a1, a2 = int(bond[0]), int(bond[1])
        graph[a1].append(a2)
        graph[a2].append(a1)

    # 按分子展开
    unique_molecules = np.unique(molecule_ids)

    for mol_id in unique_molecules:
        mol_atoms = np.where(molecule_ids == mol_id)[0]

        if len(mol_atoms) <= 1:
            continue

        # 转换为1-indexed
        mol_atom_ids = mol_atoms + 1

        # BFS展开
        visited = set()
        start_atom = mol_atom_ids[0]
        queue = deque([start_atom])
        visited.add(start_atom)

        while queue:
            current = queue.popleft()
            current_idx = current - 1  # 转换为0-indexed

            for neighbor in graph[current]:
                if neighbor in visited:
                    continue
                if molecule_ids[neighbor - 1] != mol_id:
                    continue

                visited.add(neighbor)
                neighbor_idx = neighbor - 1

                # 计算最小镜像位移
                delta = coords[neighbor_idx] - unwrapped[current_idx]
                delta = delta - box_size * np.round(delta / box_size)

                unwrapped[neighbor_idx] = unwrapped[current_idx] + delta
                queue.append(neighbor)

    return unwrapped


def calculate_central_mass(coords: np.ndarray, masses: np.ndarray) -> np.ndarray:
    """
    计算点集的质心

    参数:
        coords: 形状为(n, 3)的坐标数组
        masses: 形状为(n,)的质量数组

    返回:
        质心坐标 (3,)
    """
    total_mass = np.sum(masses)
    weighted_coords = masses[:, np.newaxis] * coords
    central_mass = np.sum(weighted_coords, axis=0) / total_mass
    return central_mass


if __name__ == "__main__":
    print("=" * 60)
    print("测试坐标处理工具函数")
    print("=" * 60)

    # 测试wrap_coordinates
    print("\n测试 wrap_coordinates:")
    box = np.array([[0, 10], [0, 10], [0, 10]], dtype=np.float64)
    test_coords = np.array([[11, -2, 5], [3, 15, 8]], dtype=np.float64)

    wrapped, images = wrap_coordinates(test_coords, box)
    print(f"  原始坐标: {test_coords}")
    print(f"  映射后: {wrapped}")
    print(f"  Image标记: {images}")

    # 测试pbc_distance
    print("\n测试 pbc_distance:")
    coord1 = np.array([1, 1, 1], dtype=np.float64)
    coord2 = np.array([9, 9, 9], dtype=np.float64)
    delta = pbc_distance(coord1, coord2, box)
    print(f"  coord1: {coord1}")
    print(f"  coord2: {coord2}")
    print(f"  位移: {delta} (期望: [-2, -2, -2])")

    print("\n✅ 测试完成!")