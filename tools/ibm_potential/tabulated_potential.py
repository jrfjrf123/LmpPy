"""
LAMMPS / VOTCA tabulated 势能表读取、绘图与解析拟合模块

支持读取 VOTCA 生成的 LAMMPS table 格式 (``*.pot.table``) 势能表：

    VOTCA
    N 1800 R 0.000010 18.000000
    <空行>
    1  1.0000e-05  5.5235e+12  4.8377e+13
    ...

数据行共 4 列: ``index  x  energy  force``，其中 ``force = -dU/dx``。

势能类型由文件名自动识别:
- ``nbXX`` / ``nonbondXX`` / ``pairXX``  -> nonbonded (默认拟合 12-6 Lennard-Jones)
- ``bondXX``                            -> bond (默认拟合 harmonic)
- ``angleXX``                           -> angle (默认拟合 cos 形式, 失败回退 harmonic)

角度横坐标既可能是弧度也可能是度，模块会自动识别并统一转换为度用于拟合/绘图。

作者: Claude
日期: 2026-08-13
"""

import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# 可选依赖: scipy 用于拟合
try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False

# 可选依赖: matplotlib 用于绘图
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:  # pragma: no cover
    HAS_MATPLOTLIB = False

# 2^(1/6)，LJ 势能最低点与 sigma 的关系
LJ_RMIN_FACTOR = 2.0 ** (1.0 / 6.0)


# ============================================================================
# 数据结构
# ============================================================================

class TabulatedPotential:
    """
    单个 tabulated 势能表。

    Attributes:
        path: 文件路径
        kind: 势能类型 ('bond' / 'angle' / 'nonbonded' / 'unknown')
        type_ids: 类型编号元组 (如 nb12 -> (1, 2), bond1 -> (1,))
        x: 原始横坐标 (距离 Å 或角度 弧度/度)
        energy: 势能数组
        force: 力数组 (-dU/dx)
        n: 表格点数
        rmin: pair 表格的 R 起始值 (可选)
        rmax: pair 表格的 R 结束值 (可选)
        header_keyword: 首行关键词 (如 'VOTCA')
        header_angle_unit: 头文件显式声明的角度单位 (可选)
        energy_unit: 能量单位标签 (默认 'kcal/mol')
    """

    def __init__(
        self,
        path: Path,
        kind: str,
        type_ids: Tuple[int, ...],
        x: np.ndarray,
        energy: np.ndarray,
        force: np.ndarray,
        n: int,
        rmin: Optional[float] = None,
        rmax: Optional[float] = None,
        header_keyword: str = "",
        header_angle_unit: Optional[str] = None,
        energy_unit: str = "kcal/mol",
    ):
        self.path = Path(path)
        self.kind = kind
        self.type_ids = type_ids
        self.x = x
        self.energy = energy
        self.force = force
        self.n = n
        self.rmin = rmin
        self.rmax = rmax
        self.header_keyword = header_keyword
        self.header_angle_unit = header_angle_unit
        self.energy_unit = energy_unit

    @property
    def name(self) -> str:
        """文件名（去掉 .pot.table / .table / .pot 等后缀）"""
        stem = self.path.stem
        if stem.endswith(".pot"):
            stem = stem[:-4]
        return stem

    def angle_unit(self) -> str:
        """返回角度横坐标单位 ('deg' 或 'rad')，仅对 angle 类型有意义。"""
        return detect_angle_unit(self.x, self.header_angle_unit)

    def theta_deg(self) -> np.ndarray:
        """返回以度为单位的横坐标（angle 类型），其余类型原样返回 x。"""
        if self.kind != "angle":
            return self.x
        unit = self.angle_unit()
        if unit == "rad":
            return np.rad2deg(self.x)
        return self.x

    def x_label(self) -> str:
        """返回绘图/拟合用的横坐标标签。"""
        if self.kind == "angle":
            return "Angle (deg)"
        return "Distance (A)"


# ============================================================================
# 文件读取
# ============================================================================

def read_tabulated_table(filepath: str, energy_unit: str = "kcal/mol") -> TabulatedPotential:
    """
    读取一个 LAMMPS / VOTCA tabulated 势能表文件。

    Args:
        filepath: 势能表文件路径
        energy_unit: 能量单位标签 (仅用于元信息，不参与数值)

    Returns:
        TabulatedPotential 对象
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")

    header: Dict[str, object] = {}
    rows: List[Tuple[float, float, float]] = []

    with open(filepath, "r") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue

            # 解析头行: N <n> [R rmin rmax] [DEGREES|RADIANS]
            m = re.match(r"^N\s+(\d+)\b(.*)$", s, re.IGNORECASE)
            if m:
                header["n"] = int(m.group(1))
                rest = m.group(2)
                rm = re.search(r"\bR\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)", rest, re.IGNORECASE)
                if rm:
                    header["rmin"] = float(rm.group(1))
                    header["rmax"] = float(rm.group(2))
                if re.search(r"\bDEGREES\b", rest, re.IGNORECASE):
                    header["angle_unit"] = "deg"
                elif re.search(r"\bRADIANS\b", rest, re.IGNORECASE):
                    header["angle_unit"] = "rad"
                continue

            # 数据行: index x energy force
            parts = s.split()
            if len(parts) >= 4:
                try:
                    x = float(parts[1])
                    e = float(parts[2])
                    fr = float(parts[3])
                except (ValueError, IndexError):
                    continue
                rows.append((x, e, fr))
                continue

            # 首个无法解析为 N 行/数据行的内容行 -> 头关键词 (如 'VOTCA')
            if "keyword" not in header:
                header["keyword"] = parts[0] if parts else ""

    if not rows:
        raise ValueError(f"无法从文件读取任何数据行: {filepath}")

    if "n" in header and header["n"] != len(rows):
        warnings.warn(
            f"{filepath.name}: 头部声明 N={header['n']}，实际读取 {len(rows)} 行数据"
        )

    data = np.array(rows, dtype=float)
    # 按横坐标排序（VOTCA 通常已升序，此处保证）
    order = np.argsort(data[:, 0])
    data = data[order]

    kind = detect_kind(filepath.name)
    type_ids = extract_type_ids(filepath.name, kind)

    return TabulatedPotential(
        path=filepath,
        kind=kind,
        type_ids=type_ids,
        x=data[:, 0],
        energy=data[:, 1],
        force=data[:, 2],
        n=int(header.get("n", len(rows))),
        rmin=header.get("rmin"),
        rmax=header.get("rmax"),
        header_keyword=str(header.get("keyword", "")),
        header_angle_unit=header.get("angle_unit"),
        energy_unit=energy_unit,
    )


def detect_kind(filename: str) -> str:
    """
    从文件名识别势能类型。

    Args:
        filename: 文件名

    Returns:
        'bond' / 'angle' / 'nonbonded' / 'dihedral' / 'unknown'
    """
    stem = Path(filename).stem.lower()
    if stem.endswith(".pot"):
        stem = stem[:-4]
    if stem.startswith("nb") or stem.startswith("nonbond") or stem.startswith("pair"):
        return "nonbonded"
    if stem.startswith("bond"):
        return "bond"
    if stem.startswith("angle"):
        return "angle"
    if stem.startswith("dih") or stem.startswith("dihed"):
        return "dihedral"
    return "unknown"


def extract_type_ids(filename: str, kind: str) -> Tuple[int, ...]:
    """
    从文件名提取类型编号。

    Args:
        filename: 文件名
        kind: 势能类型

    Returns:
        类型编号元组
    """
    stem = Path(filename).stem.lower()
    if stem.endswith(".pot"):
        stem = stem[:-4]

    if kind == "nonbonded":
        m = re.search(r"(?:nb|nonbond|nonbonded|pair)[_\-]?(\d+)[_\-]?(\d+)", stem)
        if m:
            return (int(m.group(1)), int(m.group(2)))
        digits = re.findall(r"\d", stem)
        if len(digits) >= 2:
            return (int(digits[0]), int(digits[1]))
        return ()

    m = re.search(r"(?:bond|angle|dihedral|dihed)[_\-]?(\d+)", stem)
    if m:
        return (int(m.group(1)),)
    return ()


def detect_angle_unit(x: np.ndarray, header_unit: Optional[str] = None) -> str:
    """
    自动识别角度横坐标单位。

    Args:
        x: 横坐标数组
        header_unit: 头文件显式声明的单位 (可选)

    Returns:
        'deg' 或 'rad'
    """
    if header_unit in ("deg", "rad"):
        return header_unit

    xmax = float(np.nanmax(x))
    if abs(xmax - 180.0) < 10.0:
        return "deg"
    if abs(xmax - np.pi) < 0.5:
        return "rad"
    if xmax > 10.0:
        return "deg"
    return "rad"


def discover_tables(directory: str, extensions: Optional[List[str]] = None) -> Dict[str, List[str]]:
    """
    发现目录下的 tabulated 势能表文件。

    Args:
        directory: 目录路径
        extensions: 扩展名列表

    Returns:
        {kind: [filepaths]}，按类型分组的文件列表
    """
    if extensions is None:
        extensions = [".pot.table", ".table"]

    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")

    files_by_kind: Dict[str, List[str]] = {
        "bond": [],
        "angle": [],
        "nonbonded": [],
        "dihedral": [],
        "unknown": [],
    }

    seen = set()
    for ext in extensions:
        for filepath in directory.glob(f"*{ext}"):
            key = str(filepath.resolve())
            if key in seen:
                continue
            seen.add(key)
            kind = detect_kind(filepath.name)
            files_by_kind[kind].append(str(filepath))

    for kind in files_by_kind:
        files_by_kind[kind].sort()

    return files_by_kind


def collect_tables(files: Optional[List[str]] = None,
                   directory: Optional[str] = None,
                   energy_unit: str = "kcal/mol") -> List[TabulatedPotential]:
    """
    从文件列表与可选目录收集势能表（按绝对路径去重）。

    读取失败的文件打印警告并跳过；目录不存在时抛出 FileNotFoundError。

    Args:
        files: 势能表文件路径列表
        directory: 额外搜索的目录 (discover_tables)
        energy_unit: 能量单位标签

    Returns:
        TabulatedPotential 列表
    """
    tables: List[TabulatedPotential] = []
    seen = set()

    def _add(fp: str):
        p = Path(fp)
        if not p.exists():
            print(f"警告: 文件不存在，跳过: {fp}")
            return
        key = str(p.resolve())
        if key in seen:
            return
        seen.add(key)
        try:
            tables.append(read_tabulated_table(str(p), energy_unit=energy_unit))
        except Exception as e:
            print(f"警告: 无法读取文件 {fp}: {e}")

    for fp in files or []:
        _add(fp)

    if directory:
        discovered = discover_tables(directory)
        for kind in ["nonbonded", "bond", "angle", "dihedral", "unknown"]:
            for fp in discovered.get(kind, []):
                _add(fp)

    return tables


# ============================================================================
# 解析函数模型
# ============================================================================

def lj_12_6(r: np.ndarray, eps: float, sigma: float) -> np.ndarray:
    """12-6 Lennard-Jones 势能: U(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]。"""
    inv = np.divide(sigma, r, out=np.zeros_like(np.asarray(r, dtype=float)), where=np.asarray(r) != 0)
    inv2 = inv * inv
    inv6 = inv2 * inv2 * inv2
    return 4.0 * eps * (inv6 * inv6 - inv6)


def harmonic_bond(r: np.ndarray, k: float, r0: float, e0: float) -> np.ndarray:
    """harmonic 键势能: U(r) = k*(r - r0)^2 + e0。"""
    dr = np.asarray(r, dtype=float) - r0
    return k * dr * dr + e0


def cos_angle(theta_deg: np.ndarray, k: float, theta0_deg: float, e0: float) -> np.ndarray:
    """cos 形式角势能: U(theta) = k*[1 - cos(theta - theta0)] + e0 (角度单位为度)。"""
    dth = np.deg2rad(np.asarray(theta_deg, dtype=float) - theta0_deg)
    return k * (1.0 - np.cos(dth)) + e0


def harmonic_angle(theta_deg: np.ndarray, k: float, theta0_deg: float, e0: float) -> np.ndarray:
    """harmonic 角势能: U(theta) = k*(theta - theta0)^2 + e0，k 单位为 energy/rad^2。"""
    dth = np.deg2rad(np.asarray(theta_deg, dtype=float) - theta0_deg)
    return k * dth * dth + e0


# ============================================================================
# 拟合结果
# ============================================================================

class FitResult:
    """
    单个势能表的解析拟合结果。

    Attributes:
        name: 势能表文件名
        kind: 势能类型
        type_ids: 类型编号元组
        form: 拟合形式 ('lj_12_6' / 'harmonic_bond' / 'cos_angle' / 'harmonic_angle')
        params: 拟合参数字典
        rmse: 均方根误差
        r_squared: 决定系数 R^2
        data_x: 参与拟合评估的横坐标
        data_y: 参与拟合评估的势能
        fit_x: 拟合曲线横坐标
        fit_y: 拟合曲线势能
        note: 备注
    """

    def __init__(
        self,
        name: str,
        kind: str,
        type_ids: Tuple[int, ...],
        form: str,
        params: Dict[str, float],
        rmse: float,
        r_squared: float = 0.0,
        data_x: Optional[np.ndarray] = None,
        data_y: Optional[np.ndarray] = None,
        fit_x: Optional[np.ndarray] = None,
        fit_y: Optional[np.ndarray] = None,
        note: str = "",
    ):
        self.name = name
        self.kind = kind
        self.type_ids = type_ids
        self.form = form
        self.params = params
        self.rmse = rmse
        self.r_squared = r_squared
        self.data_x = data_x
        self.data_y = data_y
        self.fit_x = fit_x
        self.fit_y = fit_y
        self.note = note


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot <= 0:
        return 1.0 if ss_res <= 0 else 0.0
    return 1.0 - ss_res / ss_tot


# ============================================================================
# 拟合函数
# ============================================================================

def _lj_initial_region(x: np.ndarray, e: np.ndarray,
                       u_wall: float = 2.0) -> Tuple[float, float, float, float]:
    """
    确定 LJ 拟合的初始区间与建议 cutoff。

    左端: 排斥壁下降沿上 U = u_wall 的点 (最靠近势阱者)；
    右端: 势阱右侧 U 回升到 ≈0 的点 (即"第二个 y=0"点，从表尾向前搜索，
          避免尾部数值振荡导致提前截断)。

    Returns:
        (x_left, x_right, r0, e_min)  r0/e_min 为势能最低点位置/能量
    """
    imin = int(np.argmin(e))
    r0 = float(x[imin])
    e_min = float(e[imin])

    wall = np.where((x < r0) & (e >= u_wall))[0]
    x_left = float(x[wall[-1]]) if len(wall) > 0 else float(x[0])

    tol = max(1e-3, abs(e_min) * 0.01)
    x_right = float(x[-1])
    tail = np.where((x > r0) & (e < -tol))[0]
    if len(tail) > 0:
        x_right = float(x[min(int(tail[-1]) + 1, len(x) - 1)])

    return x_left, x_right, r0, e_min


def _lj_fit_once(xf: np.ndarray, ef: np.ndarray,
                 eps0: float, sigma0: float) -> Tuple[float, float]:
    """在给定数据上做一次 12-6 LJ 两参数拟合。"""
    popt, _ = curve_fit(
        lj_12_6, xf, ef,
        p0=[eps0, sigma0],
        bounds=([1e-12, 1e-3], [np.inf, np.inf]),
        maxfev=20000,
    )
    return float(popt[0]), float(popt[1])


def fit_lj(table: TabulatedPotential,
           rmin_fit: Optional[float] = None,
           rmax_fit: Optional[float] = None,
           u_wall: float = 2.0,
           u_eval_max: float = 3.0,
           n_left: int = 6,
           n_right: int = 12) -> FitResult:
    """
    将 nonbonded 势能表拟合为 12-6 Lennard-Jones 势 (拟合区间循环探索)。

    未手动指定区间时，在初始区间内网格化尝试不同的 [左端, 右端] 组合，
    每个区间做一次拟合，按固定评估集上的 R^2 (等价于残差和最小) 选优:

    - 初始区间: 左端 = 排斥壁下降沿 U = u_wall 处，右端 = 势阱右侧 U≈0 处，
      所有候选区间均不超出初始区间；
    - 评估集: 拟合区间内的点 + 初始区间右侧的所有点 + 左侧 U <= u_eval_max
      的点 (陡壁区 U > u_eval_max 不参与评估)。

    Args:
        table: 势能表
        rmin_fit/rmax_fit: 手动指定拟合区间 (指定任一即跳过探索)
        u_wall: 初始区间左端的能量阈值
        u_eval_max: 评估集左侧能量上限
        n_left/n_right: 左/右端候选点网格数

    Returns:
        FitResult
    """
    if not HAS_SCIPY:
        raise RuntimeError("拟合需要 scipy，请安装: pip install scipy")

    finite = np.isfinite(table.energy) & (table.x > 0)
    x = table.x[finite]
    e = table.energy[finite]
    if len(x) < 5:
        raise ValueError(f"{table.name}: 有效数据点不足")

    x_left0, x_right0, r0, e_min = _lj_initial_region(x, e, u_wall=u_wall)

    # 固定评估集: 左侧只取 U<=u_eval_max 的点，初始区间右侧取所有点
    eval_mask = (e <= u_eval_max) | (x > x_right0)
    xe, ee = x[eval_mask], e[eval_mask]
    if len(xe) < 5:
        raise ValueError(f"{table.name}: 评估集数据点不足")

    eps0 = max(abs(e_min), 0.01)
    sigma0 = r0 / LJ_RMIN_FACTOR

    if rmin_fit is not None or rmax_fit is not None:
        # 手动区间: 不探索
        left_cands = [rmin_fit if rmin_fit is not None else x_left0]
        right_cands = [rmax_fit if rmax_fit is not None else x_right0]
    else:
        left_cands = np.linspace(x_left0, r0, max(n_left, 2))
        right_cands = np.linspace(r0, x_right0, max(n_right, 2))[1:]

    best = None  # (r2, rmse, L, R, eps, sigma, xf)
    for L in left_cands:
        for R in right_cands:
            if R <= L:
                continue
            m = (x >= L) & (x <= R)
            xf, ef = x[m], e[m]
            if len(xf) < 5:
                continue
            try:
                eps, sigma = _lj_fit_once(xf, ef, eps0, sigma0)
            except Exception:
                continue
            ypred = lj_12_6(xe, eps, sigma)
            r2 = _r_squared(ee, ypred)
            rmse = _rmse(ee, ypred)
            if best is None or r2 > best[0]:
                best = (r2, rmse, float(L), float(R), eps, sigma, xf)

    if best is None:
        raise ValueError(f"{table.name}: 所有候选拟合区间均拟合失败")

    r2, rmse, L, R, eps, sigma, xf = best
    yfit = lj_12_6(xf, eps, sigma)

    if rmin_fit is not None or rmax_fit is not None:
        note = f"拟合区间 r=[{xf[0]:.3f}, {xf[-1]:.3f}] A (手动指定)"
    else:
        note = (f"区间探索: 最佳拟合区间 r=[{xf[0]:.3f}, {xf[-1]:.3f}] A "
                f"(初始 [{x_left0:.3f}, {x_right0:.3f}]); "
                f"评估集 = 拟合区间 + 右侧全部点 + 左侧 U<={u_eval_max:g} 的点")
    note += f"; 建议 cutoff={x_right0:.3f} A"

    return FitResult(
        name=table.name,
        kind=table.kind,
        type_ids=table.type_ids,
        form="lj_12_6",
        params={"eps": eps, "sigma": sigma, "rmin": r0, "emin": e_min,
                "cutoff": x_right0},
        rmse=rmse,
        r_squared=r2,
        data_x=x,
        data_y=e,
        fit_x=xf,
        fit_y=yfit,
        note=note,
    )


def fit_lj_direct(table: TabulatedPotential,
                  u_eval_max: float = 3.0) -> FitResult:
    """
    不做最小二乘，直接从 table 的物理特征点读取 12-6 LJ 参数。

    步骤:
        1. 归一化纵坐标: 减去右侧尾部基线 (最后 10% 点的中位数)，使 U(∞)=0；
        2. sigma   = 归一化后势能为 0 的第一个点 (排斥壁下降沿过零点，线性插值)；
        3. epsilon = 归一化后势能最小值的绝对值 (势阱深度)。

    RMSE/R^2 仍在与 fit_lj 相同的评估集 (左侧 U<=u_eval_max + 右侧全部点)
    上计算，便于两种方法对比。

    Args:
        table: 势能表
        u_eval_max: 评估集左侧能量上限

    Returns:
        FitResult (fit_x/fit_y 为 None，无拟合区间)
    """
    finite = np.isfinite(table.energy) & (table.x > 0)
    x = table.x[finite]
    e_raw = table.energy[finite]
    if len(x) < 5:
        raise ValueError(f"{table.name}: 有效数据点不足")

    # 1. 归一化纵坐标: 减去右侧尾部基线
    n_tail = max(len(x) // 10, 5)
    baseline = float(np.median(e_raw[-n_tail:]))
    e = e_raw - baseline

    imin = int(np.argmin(e))
    r0 = float(x[imin])
    e_min = float(e[imin])
    if e_min >= 0:
        raise ValueError(f"{table.name}: 势能最小值非负 ({e_min:.4g})，无势阱可取 epsilon")
    eps = abs(e_min)

    # 2. sigma: 第一个从正到负的 U=0 过零点 (线性插值)
    crossings = np.where((e[:-1] > 0) & (e[1:] <= 0))[0]
    if len(crossings) == 0:
        raise ValueError(f"{table.name}: 未找到 U=0 过零点 (纯排斥或恒负势能?)")
    i = int(crossings[0])
    frac = e[i] / (e[i] - e[i + 1])
    sigma = float(x[i] + frac * (x[i + 1] - x[i]))

    # 建议 cutoff 与 fit_lj 一致: 势阱右侧 U≈0 的点
    _, x_right0, _, _ = _lj_initial_region(x, e)

    # 3. 评估 (与区间探索法相同的固定评估集)
    eval_mask = (e <= u_eval_max) | (x > x_right0)
    xe, ee = x[eval_mask], e[eval_mask]
    ypred = lj_12_6(xe, eps, sigma)
    rmse = _rmse(ee, ypred)
    r2 = _r_squared(ee, ypred)

    note = (f"直接取值: sigma=U=0 第一个过零点 ({sigma:.4f} A), "
            f"eps=|Umin| ({eps:.6g}); 纵坐标已归一化 (尾部基线={baseline:.4g}); "
            f"评估集 = 左侧 U<={u_eval_max:g} + 右侧全部点; "
            f"建议 cutoff={x_right0:.3f} A")

    return FitResult(
        name=table.name,
        kind=table.kind,
        type_ids=table.type_ids,
        form="lj_12_6",
        params={"eps": eps, "sigma": sigma, "rmin": r0, "emin": e_min,
                "cutoff": x_right0},
        rmse=rmse,
        r_squared=r2,
        data_x=x,
        data_y=e,
        fit_x=None,
        fit_y=None,
        note=note,
    )


def fit_lj_well(table: TabulatedPotential,
                u_wall: float = 1.5,
                u_eval_max: float = 3.0) -> FitResult:
    """
    势阱特征取参 (不做最小二乘): 以壳层间势垒顶为能量零点读取 12-6 LJ 参数。

    适用于带第一/第二壳层结构的 IBI/PMF 类有效对势: 第一壳层势阱整体
    略高于零、远程还有浅层振荡时, "第一个 U=0 过零点" 会滑到远程区,
    导致 fit_lj_direct 把主导的第一壳层误判为排斥区。本方法改为:

        1. 尾部基线归一 (仅用于区域判定, eps/sigma 对该平移不变);
        2. 左边界 = 排斥壁下降沿 E 首次低于 u_wall 的位置,
           避开壁上 E 较高的噪声抖动, 拐点只在左边界之后取;
        3. 左边界后第一个局部极小 = 第一壳层势阱 (r_min, E_well),
           其后第一个局部极大 = 壳层间势垒 (r_barr, E_barr);
        4. 以势垒顶为能量零点: eps = E_barr - E_well,
           sigma = 壁上 E=E_barr 的下降沿过点 (线性插值);
        5. cutoff = r_barr, 右侧区域整体截除 (参数确定后 LJ 尾部随之确定)。

    注意: 返回结果中 data_y 已平移为势垒顶参考 (E - E_barr), 与 LJ 的
    零点定义一致, 使 RMSE/R^2 与对比图在同一参考面下计算。

    Args:
        table: 势能表
        u_wall: 左边界能量阈值 (拐点只在该等值线之后取)
        u_eval_max: 评估集左侧能量上限

    Returns:
        FitResult
    """
    finite = np.isfinite(table.energy) & (table.x > 0)
    x = table.x[finite]
    e_raw = table.energy[finite]
    if len(x) < 5:
        raise ValueError(f"{table.name}: 有效数据点不足")

    # 1. 尾部基线归一 (仅用于区域判定)
    n_tail = max(len(x) // 10, 5)
    baseline = float(np.median(e_raw[-n_tail:]))
    e = e_raw - baseline

    # 2. 左边界: 下降沿首次低于 u_wall
    i_start = int(np.argmax(e < u_wall))
    if not e[i_start] < u_wall:
        raise ValueError(f"{table.name}: 未找到 E<{u_wall:g} 的区域 (纯排斥势能?)")

    # 3. 第一个局部极小 (势阱) 与其后的第一个局部极大 (势垒)
    de = np.diff(e)
    i_well = i_barr = None
    for i in range(i_start + 1, len(de)):
        if de[i - 1] < 0 < de[i]:
            i_well = i
            break
    if i_well is None:
        raise ValueError(f"{table.name}: E<{u_wall:g} 区域后未找到局部极小 (势阱), "
                         f"可改用 --lj-method direct/explore")
    for i in range(i_well + 1, len(de)):
        if de[i - 1] > 0 > de[i]:
            i_barr = i
            break
    if i_barr is None:
        raise ValueError(f"{table.name}: 势阱后未找到局部极大 (壳层间势垒), "
                         f"曲线可能无壳层结构, 可改用 --lj-method direct/explore")

    r_min, e_well = float(x[i_well]), float(e[i_well])
    r_barr, e_barr = float(x[i_barr]), float(e[i_barr])

    # 4. eps 与 sigma (势垒顶为零点)
    eps = e_barr - e_well
    if eps <= 0:
        raise ValueError(f"{table.name}: 势阱深度非正 (E_barr-E_well={eps:.4g})")
    below = np.where(e[i_start:i_well + 1] <= e_barr)[0]
    if len(below) == 0:
        raise ValueError(f"{table.name}: 壁上未找到 E=E_barr 的过点")
    j = i_start + int(below[0])  # 第一个 <= e_barr 的点, 前一个点 > e_barr
    frac = (e[j - 1] - e_barr) / (e[j - 1] - e[j])
    sigma = float(x[j - 1] + frac * (x[j] - x[j - 1]))

    # 5. 评估: 势垒顶参考面, 范围 [左端 U<=u_eval_max, r_barr] (右侧已截除)
    e_ref = e - e_barr
    eval_mask = (e <= u_eval_max) & (x <= r_barr)
    xe, ee = x[eval_mask], e_ref[eval_mask]
    if len(xe) < 5:
        raise ValueError(f"{table.name}: 评估集数据点不足")
    ypred = lj_12_6(xe, eps, sigma)
    rmse = _rmse(ee, ypred)
    r2 = _r_squared(ee, ypred)

    note = (f"势阱特征取参: 势阱 r_min={r_min:.3f} A, 壳层间势垒 r_barr={r_barr:.3f} A; "
            f"以势垒顶为零点, eps=E_barr-E_well={eps:.6g}, "
            f"sigma=壁上 E=E_barr 过点 ({sigma:.4f} A); "
            f"cutoff=r_barr={r_barr:.3f} A (右侧区域截除); "
            f"评估集 = U<={u_eval_max:g} 且 r<=r_barr 的点 (势垒顶参考面)")

    return FitResult(
        name=table.name,
        kind=table.kind,
        type_ids=table.type_ids,
        form="lj_12_6",
        params={"eps": eps, "sigma": sigma, "rmin": r_min, "emin": -eps,
                "cutoff": r_barr},
        rmse=rmse,
        r_squared=r2,
        data_x=x,
        data_y=e_ref,
        fit_x=None,
        fit_y=None,
        note=note,
    )


def fit_bond_harmonic(table: TabulatedPotential,
                      u_cut: float = 10.0,
                      rmin: Optional[float] = None,
                      rmax: Optional[float] = None) -> FitResult:
    """
    将 bond 势能表拟合为 harmonic 势: U(r) = k*(r - r0)^2 + e0。

    Args:
        table: 势能表
        u_cut: 势阱区选择阈值 (kcal/mol)，只拟合 U - Umin <= u_cut 的点
        rmin/rmax: 拟合区间覆盖

    Returns:
        FitResult
    """
    if not HAS_SCIPY:
        raise RuntimeError("拟合需要 scipy，请安装: pip install scipy")

    finite = np.isfinite(table.energy)
    x = table.x[finite]
    e = table.energy[finite]
    if len(x) < 5:
        raise ValueError(f"{table.name}: 有效数据点不足")

    imin = int(np.argmin(e))
    r0 = float(x[imin])
    e_min = float(e[imin])

    mask = (e - e_min) <= u_cut
    if rmin is not None:
        mask &= x >= rmin
    if rmax is not None:
        mask &= x <= rmax

    xf, ef = x[mask], e[mask]
    if len(xf) < 5:
        raise ValueError(f"{table.name}: 拟合区间内数据点不足 (u_cut={u_cut} 过大或过小)")

    k0 = 10.0
    popt, _ = curve_fit(
        harmonic_bond, xf, ef,
        p0=[k0, r0, e_min],
        bounds=([1e-9, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
        maxfev=20000,
    )
    k, r0f, e0 = float(popt[0]), float(popt[1]), float(popt[2])

    yfit = harmonic_bond(xf, k, r0f, e0)
    rmse = _rmse(ef, yfit)
    r2 = _r_squared(ef, yfit)

    return FitResult(
        name=table.name,
        kind=table.kind,
        type_ids=table.type_ids,
        form="harmonic_bond",
        params={"k": k, "r0": r0f, "e0": e0},
        rmse=rmse,
        r_squared=r2,
        data_x=x,
        data_y=e,
        fit_x=xf,
        fit_y=yfit,
        note=f"拟合区间 r=[{xf[0]:.3f}, {xf[-1]:.3f}] A (U-Umin<={u_cut:.2f} {table.energy_unit})",
    )


def fit_angle(table: TabulatedPotential,
              form: str = "auto",
              u_cut: float = 10.0) -> FitResult:
    """
    将 angle 势能表拟合为 cos 形式 (默认) 或 harmonic 形式。

    cos 形式: U = k*[1 - cos(theta - theta0)] + e0
    harmonic 形式: U = k*(theta - theta0)^2 + e0 (k 单位 energy/rad^2)

    Args:
        table: 势能表
        form: 'auto' (先尝试 cos，失败或明显更差时回退 harmonic)、
              'cos' 或 'harmonic'
        u_cut: 势阱区选择阈值，只拟合 U - Umin <= u_cut 的点，
               避免势壁高能量区主导最小二乘

    Returns:
        FitResult
    """
    if not HAS_SCIPY:
        raise RuntimeError("拟合需要 scipy，请安装: pip install scipy")

    finite = np.isfinite(table.energy)
    t_all = table.theta_deg()[finite]
    e_all = table.energy[finite]
    if len(t_all) < 5:
        raise ValueError(f"{table.name}: 有效数据点不足")

    imin = int(np.argmin(e_all))
    e_min_all = float(e_all[imin])

    mask = (e_all - e_min_all) <= u_cut
    t, e = t_all[mask], e_all[mask]
    if len(t) < 5:
        raise ValueError(f"{table.name}: 拟合区间内数据点不足 (u_cut={u_cut} 过大或过小)")

    imin = int(np.argmin(e))
    t0 = float(t[imin])
    e_min = float(e[imin])

    k0 = max(float(np.ptp(e)) * 0.5, 0.01)

    # 尝试 cos 拟合
    popt_cos: Optional[np.ndarray] = None
    rmse_cos = np.inf
    if form in ("auto", "cos"):
        try:
            popt_cos, _ = curve_fit(
                cos_angle, t, e,
                p0=[k0, t0, e_min],
                bounds=([1e-12, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=50000,
            )
            rmse_cos = _rmse(e, cos_angle(t, *popt_cos))
        except Exception:
            popt_cos = None
            rmse_cos = np.inf

    # 尝试 harmonic 拟合
    popt_harm: Optional[np.ndarray] = None
    rmse_harm = np.inf
    if form in ("auto", "harmonic"):
        try:
            popt_harm, _ = curve_fit(
                harmonic_angle, t, e,
                p0=[k0, t0, e_min],
                bounds=([1e-12, -np.inf, -np.inf], [np.inf, np.inf, np.inf]),
                maxfev=50000,
            )
            rmse_harm = _rmse(e, harmonic_angle(t, *popt_harm))
        except Exception:
            popt_harm = None
            rmse_harm = np.inf

    # 决定最终形式
    if form == "cos":
        chosen = "cos_angle" if popt_cos is not None else "harmonic_angle"
    elif form == "harmonic":
        chosen = "harmonic_angle"
    else:  # auto: 优先 cos，明显更差时回退 harmonic
        if popt_cos is not None and (popt_harm is None or rmse_cos <= 2.0 * rmse_harm):
            chosen = "cos_angle"
        else:
            chosen = "harmonic_angle"

    if chosen == "cos_angle":
        k, theta0, e0 = float(popt_cos[0]), float(popt_cos[1]), float(popt_cos[2])
        yfit = cos_angle(t, k, theta0, e0)
        rmse = rmse_cos
        note = "cos 形式"
        if abs(theta0 - 180.0) < 1.0:
            note += " (theta0≈180°，等价于 LAMMPS angle_style cosine)"
    else:
        if popt_harm is None:
            raise RuntimeError(f"{table.name}: cos 与 harmonic 拟合均失败")
        k, theta0, e0 = float(popt_harm[0]), float(popt_harm[1]), float(popt_harm[2])
        yfit = harmonic_angle(t, k, theta0, e0)
        rmse = rmse_harm
        note = "harmonic 形式 (cos 拟合不佳，回退)"

    r2 = _r_squared(e, yfit)

    return FitResult(
        name=table.name,
        kind=table.kind,
        type_ids=table.type_ids,
        form=chosen,
        params={"k": k, "theta0": theta0, "e0": e0},
        rmse=rmse,
        r_squared=r2,
        data_x=t,
        data_y=e,
        fit_x=t,
        fit_y=yfit,
        note=note + f"; 拟合区间 (U-Umin<={u_cut:.2f} {table.energy_unit})",
    )


def fit_table(table: TabulatedPotential,
              angle_form: str = "auto",
              u_cut: float = 10.0,
              rmin_fit: Optional[float] = None,
              rmax_fit: Optional[float] = None,
              lj_method: str = "well") -> FitResult:
    """
    按势能类型分发到对应的拟合函数。

    Args:
        table: 势能表
        angle_form: angle 拟合形式 ('auto' / 'cos' / 'harmonic')
        u_cut: bond/angle 势阱区阈值，只拟合 U - Umin <= u_cut 的点
        rmin_fit/rmax_fit: nonbonded 拟合区间覆盖 (lj_method='explore' 时有效)
        lj_method: nonbonded 取参方式 ('well' 势阱特征取参, 以壳层间势垒顶为
                   零点 (默认) / 'direct' 尾部基线归零后取第一个 U=0 过零点 /
                   'explore' 最小二乘区间探索)

    Returns:
        FitResult
    """
    if table.kind == "nonbonded":
        if lj_method == "well":
            return fit_lj_well(table)
        if lj_method == "direct":
            return fit_lj_direct(table)
        return fit_lj(table, rmin_fit=rmin_fit, rmax_fit=rmax_fit)
    if table.kind == "bond":
        return fit_bond_harmonic(table, u_cut=u_cut, rmin=rmin_fit, rmax=rmax_fit)
    if table.kind == "angle":
        return fit_angle(table, form=angle_form, u_cut=u_cut)
    raise ValueError(f"不支持的势能类型: {table.kind} (无法拟合)")


def eval_fit(result: FitResult, x: np.ndarray) -> np.ndarray:
    """在任意横坐标上评估拟合得到的解析势函数。"""
    p = result.params
    if result.form == "lj_12_6":
        return lj_12_6(x, p["eps"], p["sigma"])
    if result.form == "harmonic_bond":
        return harmonic_bond(x, p["k"], p["r0"], p["e0"])
    if result.form == "cos_angle":
        return cos_angle(x, p["k"], p["theta0"], p["e0"])
    if result.form == "harmonic_angle":
        return harmonic_angle(x, p["k"], p["theta0"], p["e0"])
    raise ValueError(f"未知的拟合形式: {result.form}")


# ============================================================================
# 绘图函数
# ============================================================================

@dataclass
class AxisLimits:
    """单个坐标轴的范围限值。"""
    xlim: Optional[Tuple[float, float]] = None
    ylim: Optional[Tuple[float, float]] = None


@dataclass
class PlotLimits:
    """
    按势能类型定制的绘图坐标范围。

    每种类型的 xlim/ylim 逐分量回退: 类型未指定 → default → None (自动)。
    """
    default: AxisLimits = field(default_factory=AxisLimits)
    nonbonded: AxisLimits = field(default_factory=AxisLimits)
    bond: AxisLimits = field(default_factory=AxisLimits)
    angle: AxisLimits = field(default_factory=AxisLimits)
    dihedral: AxisLimits = field(default_factory=AxisLimits)

    def resolve(self, kind: str) -> Tuple[Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
        """返回给定类型应使用的 (xlim, ylim)，逐分量回退到 default。"""
        k = getattr(self, kind, None)
        xlim = self.default.xlim
        ylim = self.default.ylim
        if k is not None:
            if k.xlim is not None:
                xlim = k.xlim
            if k.ylim is not None:
                ylim = k.ylim
        return xlim, ylim


def default_plot_mask(table: TabulatedPotential, energy_cap: Optional[float] = None) -> np.ndarray:
    """
    返回默认绘图掩码，用于放大显示物理相关区域。

    对 nonbonded 势，屏蔽能量远高于势阱的巨大排斥值，使势阱清晰可见。

    Args:
        table: 势能表
        energy_cap: 能量上限 (覆盖默认)

    Returns:
        布尔掩码
    """
    e = table.energy
    finite = np.isfinite(e)
    if table.kind == "nonbonded":
        if energy_cap is None:
            e_min = float(e[finite].min()) if np.any(finite) else 0.0
            energy_cap = max(abs(e_min) * 5.0, 10.0)
        return finite & (e <= energy_cap)
    return finite


def _x_for_plot(table: TabulatedPotential) -> np.ndarray:
    return table.theta_deg() if table.kind == "angle" else table.x


def plot_single_table(table: TabulatedPotential,
                      output_file: str,
                      show_force: bool = True,
                      energy_cap: Optional[float] = None,
                      xlim: Optional[Tuple[float, float]] = None,
                      ylim: Optional[Tuple[float, float]] = None):
    """
    绘制单个势能表 (势能 + 可选力) 并保存。

    Args:
        table: 势能表
        output_file: 输出 PNG 路径
        show_force: 是否绘制力曲线
        energy_cap: nonbonded 势的能量上限
        xlim: 手动指定横坐标范围 (xmin, xmax)，默认 None 由 matplotlib 自动确定
        ylim: 手动指定势能纵坐标范围 (ymin, ymax)，仅作用于势能子图，
              默认 None 由 matplotlib 自动确定
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("绘图需要 matplotlib，请安装: pip install matplotlib")

    mask = default_plot_mask(table, energy_cap)
    x = _x_for_plot(table)[mask]
    e = table.energy[mask]
    f = table.force[mask]

    xlabel = table.x_label()
    eunit = table.energy_unit

    nrows = 2 if show_force else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(8, 6), sharex=True)
    if nrows == 1:
        axes = [axes]

    axes[0].plot(x, e, "b-", linewidth=1.2)
    if xlim is not None:
        axes[0].set_xlim(*xlim)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_ylabel(f"Energy ({eunit})")
    axes[0].set_title(f"{table.kind.capitalize()} potential: {table.name}")
    axes[0].grid(True, alpha=0.3)

    if show_force:
        axes[1].plot(x, f, "r-", linewidth=1.2)
        axes[1].set_ylabel("Force (-dU/dx)")
        axes[1].set_xlabel(xlabel)
        axes[1].grid(True, alpha=0.3)
    else:
        axes[0].set_xlabel(xlabel)

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_tables(tables: List[TabulatedPotential],
                    output_dir: str,
                    show_force: bool = True,
                    energy_cap: Optional[float] = None,
                    limits: Optional[PlotLimits] = None) -> List[str]:
    """
    绘制一批势能表：每个单独一张图 + 同类型合并一张图。

    Args:
        tables: 势能表列表
        output_dir: 输出目录
        show_force: 是否绘制力曲线
        energy_cap: nonbonded 势的能量上限
        limits: 按势能类型定制的坐标范围 (PlotLimits)，逐分量回退到
            limits.default，未指定则为 None (由 matplotlib 自动确定)

    Returns:
        输出文件列表
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("绘图需要 matplotlib，请安装: pip install matplotlib")

    if limits is None:
        limits = PlotLimits()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs: List[str] = []

    # 单独绘图
    for table in tables:
        out = output_dir / f"{table.name}.png"
        xlim, ylim = limits.resolve(table.kind)
        plot_single_table(table, str(out), show_force=show_force, energy_cap=energy_cap,
                          xlim=xlim, ylim=ylim)
        outputs.append(str(out))

    # 同类型合并图
    kinds = {}
    for table in tables:
        kinds.setdefault(table.kind, []).append(table)

    for kind, group in kinds.items():
        if len(group) <= 1:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10(np.linspace(0, 1, len(group)))
        for i, table in enumerate(group):
            mask = default_plot_mask(table, energy_cap)
            x = _x_for_plot(table)[mask]
            e = table.energy[mask]
            ax.plot(x, e, color=colors[i], linewidth=1.2, label=table.name)
        ax.set_xlabel(group[0].x_label())
        ax.set_ylabel(f"Energy ({group[0].energy_unit})")
        xlim, ylim = limits.resolve(kind)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_title(f"{kind.capitalize()} potentials (combined)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
        out = output_dir / f"{kind}_combined.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        outputs.append(str(out))

    return outputs


def plot_fit_comparison(result: FitResult, output_file: str,
                        table: Optional[TabulatedPotential] = None,
                        energy_cap: Optional[float] = None):
    """
    绘制 tabulated 数据 vs 解析拟合曲线的对比图。

    拟合曲线在与 tabulated 数据相同的横坐标范围内绘制，使二者显示的最大值一致，
    便于直接对比；实际参与拟合的区间用红色阴影标出。

    Args:
        result: 拟合结果
        output_file: 输出 PNG 路径
        table: 原始势能表 (可选，未使用，保留兼容)
        energy_cap: nonbonded 势的能量上限
    """
    if not HAS_MATPLOTLIB:
        raise RuntimeError("绘图需要 matplotlib，请安装: pip install matplotlib")

    fig, ax = plt.subplots(figsize=(8, 6))

    xlabel = "Angle (deg)" if result.kind == "angle" else "Distance (A)"

    # 原始数据
    x_data = np.asarray(result.data_x, dtype=float)
    y_data = np.asarray(result.data_y, dtype=float)
    if result.kind == "nonbonded":
        finite = np.isfinite(y_data)
        cap = energy_cap if energy_cap is not None \
            else max(abs(float(y_data[finite].min())) * 5.0, 10.0)
        m = finite & (y_data <= cap)
        if np.any(m):
            x_data, y_data = x_data[m], y_data[m]

    ax.plot(x_data, y_data, "b-", linewidth=1.2, label="Tabulated")

    # 拟合曲线: 在整个数据显示范围内绘制，与 tabulated 的最大值对齐
    x_grid = np.linspace(float(x_data[0]), float(x_data[-1]), 1000)
    y_grid = eval_fit(result, x_grid)
    ax.plot(x_grid, y_grid, "r--", linewidth=1.4, label=f"Fit ({result.form})")

    # 拟合区间标识 (红色阴影)，direct 方法无拟合区间则跳过
    if result.fit_x is not None and len(result.fit_x) > 0:
        lo = max(float(result.fit_x[0]), float(x_data[0]))
        hi = min(float(result.fit_x[-1]), float(x_data[-1]))
        if hi > lo:
            ax.axvspan(lo, hi, color="r", alpha=0.08, label="Fit region")

    # y 轴范围以 tabulated 数据为准，保证两条曲线显示的最大值一致
    ymin = float(np.min(y_data))
    ymax = float(np.max(y_data))
    span = ymax - ymin
    if span <= 0:
        span = max(abs(ymax), 1.0)
    ax.set_ylim(ymin - 0.05 * span, ymax + 0.05 * span)

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Energy")
    ax.set_title(f"{result.name}: fit RMSE={result.rmse:.4g}, R2={result.r_squared:.4f}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
