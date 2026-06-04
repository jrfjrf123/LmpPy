"""理论链长分布模块。

提供三种理论分布：
- Schulz-Zimm (Gamma) 分布 — 适用于自由基聚合
- Poisson 分布 — 适用于活性阴离子聚合
- 对数正态分布 — 通用经验分布
"""

import numpy as np
from scipy.special import gamma, factorial
from typing import Tuple


# ---------------------------------------------------------------------------
# 公共参数验证
# ---------------------------------------------------------------------------

def _validate_mn_pdi(Mn: float, PDI: float) -> Tuple[float, float]:
    """验证 Mn 和 PDI 参数的合法性。"""
    if Mn <= 0:
        raise ValueError(f"Mn 必须 > 0，当前值: {Mn}")
    if PDI <= 1.0:
        raise ValueError(f"PDI 必须 > 1.0（对于 Schulz-Zimm 和对数正态），当前值: {PDI}")
    return float(Mn), float(PDI)


def _normalize_pdf(result: np.ndarray, N: np.ndarray) -> np.ndarray:
    """归一化使曲线下积分面积为 1（梯形法则积分）。

    不同于 result.sum() 的离散求和归一化，本函数使用梯形积分，
    正确考虑网格间距 Δx，确保连续分布的概率密度归一化条件：
        ∫ f(x) dx ≈ 1

    参数
    ----
    result : 概率密度数组
    N : 对应的横轴网格

    返回
    ----
    归一化后的概率密度数组
    """
    integral = float(np.trapezoid(result, N))
    if integral > 0:
        result = result / integral
    return result


# ---------------------------------------------------------------------------
# 理论分布函数
# ---------------------------------------------------------------------------

def schulz_zimm(N: np.ndarray, Mn: float, PDI: float) -> np.ndarray:
    """Schulz-Zimm (Gamma) 分布。

    参数
    ----
    N : 链长横轴网格 (1D array)
    Mn : 数均链长 N_n
    PDI : 多分散性指数 = M_w / M_n

    返回
    ----
    每个 N 对应的概率密度 (归一化)
    """
    Mn, PDI = _validate_mn_pdi(Mn, PDI)
    k = 1.0 / (PDI - 1.0)  # 形状参数
    theta = Mn / k          # 尺度参数

    # Gamma 分布 PDF: f(x) = x^(k-1) * exp(-x/theta) / (theta^k * Gamma(k))
    result = N ** (k - 1.0) * np.exp(-N / theta) / (theta ** k * gamma(k))

    # 归一化：使曲线下积分面积为 1（而非离散求和）
    result = _normalize_pdf(result, N)

    return result


def poisson(N: np.ndarray, Mn: float, PDI: float = None) -> tuple[np.ndarray, float]:
    """Poisson 分布（适用于活性阴离子聚合）。

    P(N) = nu^(N-1) * exp(-nu) / (N-1)!
    其中 nu = Mn - 1

    参数
    ----
    N : 链长横轴网格 (1D array)
    Mn : 数均链长 N_n
    PDI : 不使用（保留参数一致性）

    返回
    ----
    (概率质量函数, 实际PDI值)
    """
    if Mn <= 1.0:
        raise ValueError(f"Poisson 分布要求 Mn > 1，当前值: {Mn}")

    nu = Mn - 1.0  # 平均聚合度减 1 (即期望单体数)

    N_int = np.asarray(N, dtype=np.float64)
    # 使用 gamma 函数避免大数溢出：log P = (N-1)*log(nu) - nu - log(Γ(N))
    log_p = (N_int - 1.0) * np.log(max(nu, 1e-12)) - nu - np.log(gamma(N_int))

    # 截断 log_p 避免溢出
    max_log = np.max(log_p)
    result = np.exp(log_p - max_log)

    # 归一化：使曲线下积分面积为 1（而非离散求和）
    result = _normalize_pdf(result, N)

    actual_pdi = 1.0 + 1.0 / nu
    return result, actual_pdi


def lognormal(N: np.ndarray, Mn: float, PDI: float) -> np.ndarray:
    """对数正态分布。

    参数
    ----
    N : 链长横轴网格 (1D array)
    Mn : 数均链长 N_n
    PDI : 多分散性指数

    返回
    ----
    每个 N 对应的概率密度 (归一化)
    """
    Mn, PDI = _validate_mn_pdi(Mn, PDI)

    sigma = np.sqrt(np.log(PDI))                # sigma = sqrt(ln(PDI))
    mu = np.log(Mn) - sigma ** 2 / 2.0          # mu = ln(Mn) - sigma^2/2

    # 对数正态 PDF: f(N) = 1/(N * sigma * sqrt(2*pi)) * exp(-(ln(N)-mu)^2 / (2*sigma^2))
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.exp(-(np.log(N) - mu) ** 2 / (2.0 * sigma ** 2))
        result = np.where(N > 0, result / (N * sigma * np.sqrt(2.0 * np.pi)), 0.0)

    # 归一化：使曲线下积分面积为 1（而非离散求和）
    result = _normalize_pdf(result, N)

    return result


# ---------------------------------------------------------------------------
# 理论曲线工厂函数
# ---------------------------------------------------------------------------

THEORY_FUNCTIONS = {
    "schulz-zimm": schulz_zimm,
    "schulz_zimm": schulz_zimm,
    "poisson": poisson,
    "lognormal": lognormal,
    "log-normal": lognormal,
}


def get_theory_curve(theory_type: str, N: np.ndarray, Mn: float, PDI: float):
    """获取理论曲线。

    参数
    ----
    theory_type : 分布类型名称 ("schulz-zimm", "poisson", "lognormal")
    N : 链长横轴网格
    Mn : 数均链长
    PDI : 多分散性指数 (Poisson 分布时忽略)

    返回
    ----
    (概率密度数组, 元信息字典)
    """
    theory_type = theory_type.lower().strip()
    if theory_type not in THEORY_FUNCTIONS:
        raise ValueError(
            f"未知的分布类型: '{theory_type}'。可选: {list(THEORY_FUNCTIONS.keys())}"
        )

    func = THEORY_FUNCTIONS[theory_type]
    meta = {"type": theory_type, "Mn": Mn}

    if theory_type == "poisson":
        result, actual_pdi = func(N, Mn)
        meta["PDI"] = actual_pdi
        meta["note"] = f"Poisson 分布 PDI 由 Mn 唯一确定: PDI = {actual_pdi:.4f}"
    else:
        result = func(N, Mn, PDI)
        meta["PDI"] = PDI

    return result, meta
