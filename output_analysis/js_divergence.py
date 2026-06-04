"""Jensen-Shannon 散度计算模块。

提供 JS 散度和 KL 散度的安全计算，包含拉普拉斯平滑避免 log(0)。
"""

import numpy as np


def safe_kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    """安全的 KL 散度计算，加拉普拉斯平滑避免 log(0)。

    参数
    ----
    p, q : 归一化概率分布数组
    eps : 平滑小量

    返回
    ----
    KL(p || q) 值
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # 归一化
    p = p / (p.sum() + eps)
    q = q / (q.sum() + eps)

    # 拉普拉斯平滑
    p = np.clip(p, eps, 1.0)
    q = np.clip(q, eps, 1.0)

    return float(np.sum(p * np.log2(p / q)))


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon 散度。

    JSD(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M)
    其中 M = (P + Q) / 2

    参数
    ----
    p, q : 归一化概率分布数组

    返回
    ----
    JS 散度值，范围 [0, 1]（以 2 为底的对数）
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)

    # 归一化
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum > 0:
        p = p / p_sum
    if q_sum > 0:
        q = q / q_sum

    m = (p + q) / 2.0

    return 0.5 * safe_kl_divergence(p, m) + 0.5 * safe_kl_divergence(q, m)
