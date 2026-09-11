#!/usr/bin/env python
"""
交叉混合两个 tabulated 势能表，生成异核（cross）势能表。

依据规范: docs/cross-tabulation-mixing.md (v2.0 离散势能表交叉混合操作规范)

流程（严格按规范）:
    1. 强制尾部归零化: 取末尾 10% 数据点平均值 V_tail, 全局平移 V' = V - V_tail
    2. 势阱诊断:
       情形A (V'min < 0): sigma = 首次穿越零点的横坐标 r0 (r0 < rmin), eps = |V'min|
       情形B (V'min >= 0): 定位第二个拐点 r_infl2, 平移 V''new = V' - V_infl2,
                           再重新提取 r0 / eps (强制引入负势阱)
    3. 无量纲化: r* = r/sigma, V* = V/eps
    4. 公共网格: [max(r*_min), min(r*_max)] 交集区间, 点数 > max(N_A, N_B), 等间距
    5. 三次样条插值 (Cubic Spline), 保证纵坐标与一阶导数光滑
    6. 混合: V*_mix = sign(V*_A) * sqrt(|V*_A * V*_B|);
       两表异号时强制 V*_mix = -sqrt(|V*_A * V*_B|) (保持吸引趋势连续)
    7. 去归一化: sigma_mix = (sigma_A + sigma_B)/2, eps_mix = sqrt(eps_A * eps_B)
       r_real = r*_grid * sigma_mix, V_real = V*_mix * eps_mix
    8. 质量验收: 零点位置 / 阱深范围 / 排斥区单调性 三项硬性检查
    9. 输出 VOTCA 格式 table (N / R 头 + index x energy force)

用法:
    python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb33.pot.table -o nb13.pot.table
    python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb44.pot.table -o nb14.pot.table
    python -m LmpPy.scripts.mix_cross_tabulated nb22.pot.table nb33.pot.table -o nb23.pot.table
    python -m LmpPy.scripts.mix_cross_tabulated nb22.pot.table nb44.pot.table -o nb24.pot.table

    同时输出修正后的输入势能表 (尾部归零化, 情形B再做拐点平移):
    python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb33.pot.table \
        -o nb13.pot.table --dump-corrected ./corrected
    # -> ./corrected/nb11.corrected.pot.table, ./corrected/nb33.corrected.pot.table

    异号区处理方式 (规范默认 force):
    python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb33.pot.table \
        -o nb13.pot.table --opposite-sign sign_a
    # sign_a: 不强制取负, 全程跟随 sign(V*_A) (第二壳层区可能翻正且力不连续)

    E 算术平均 (非规范, 无符号问题):
    python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb33.pot.table \
        -o nb13.pot.table --mix-mode arithmetic
    # E_mix = (E_A + E_B)/2, 采样点真实能量平均, 力连续

作者: Claude
日期: 2026-08-24
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

# 可选依赖: scipy 三次样条
try:
    from scipy.interpolate import CubicSpline
    HAS_SCIPY = True
except ImportError:  # pragma: no cover
    HAS_SCIPY = False


# ============================================================================
# VOTCA table 读写
# ============================================================================

def read_votca_table(filepath: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    读取 VOTCA 格式势能表 (index x energy force)。

    Returns:
        (x, energy, force)
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"文件不存在: {filepath}")
    data = np.loadtxt(filepath, skiprows=3)
    if data.ndim != 2 or data.shape[1] < 4:
        raise ValueError(f"无法解析势能表: {filepath}")
    order = np.argsort(data[:, 1])
    data = data[order]
    return data[:, 1].astype(float), data[:, 2].astype(float), data[:, 3].astype(float)


def write_votca_table(filepath: str, x: np.ndarray, energy: np.ndarray, force: np.ndarray):
    """
    写入 VOTCA 格式势能表 (index x energy force)。

    Args:
        filepath: 输出路径
        x: 横坐标
        energy: 势能
        force: 力 (-dU/dx)
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    n = len(x)
    with open(filepath, "w") as f:
        f.write("VOTCA\n")
        f.write(f"N {n} R {x[0]:.10e} {x[-1]:.10e}\n")
        f.write("\n")
        for i in range(n):
            f.write(f"{i + 1} {x[i]:.10e} {energy[i]:.10e} {force[i]:.10e}\n")
    print(f"已写入: {filepath.resolve()}  (N={n}, R=[{x[0]:.6e}, {x[-1]:.6e}])")


# ============================================================================
# 势能曲线特征提取 (规范 §1-3)
# ============================================================================

def analyze_potential(x: np.ndarray, e: np.ndarray,
                      tail_frac: float = 0.1,
                      force: Optional[np.ndarray] = None) -> Dict[str, object]:
    """
    提取单个势能表的长度/能量标尺 (规范 §1-3)。

    - 尾部归零化: 末尾 tail_frac 数据点均值平移;
    - 情形A (负势阱): sigma = 首次过零点, eps = |Vmin|;
    - 情形B (正/零势阱): 第二个拐点平移强制引入负阱后重新提取。

    Returns:
        dict(sigma, eps, rmin, emin, tail, corrected, x_corr, e_corr, f_corr)
        corrected=True 表示经历了情形B的强制修正;
        x_corr/e_corr/f_corr 为修正后 (尾部归零, 情形B再拐点平移) 的曲线数组
    """
    if not HAS_SCIPY:
        raise RuntimeError("需要 scipy，请安装: pip install scipy")

    finite = np.isfinite(e)
    x = x[finite]
    e = e[finite]
    if force is not None:
        force = force[finite]
    if len(x) < 5:
        raise ValueError("有效数据点不足")

    n_tail = max(int(len(x) * tail_frac), 5)
    tail = float(np.mean(e[-n_tail:]))
    e0 = e - tail
    e_corr = e0
    f_corr = force

    imin = int(np.argmin(e0))
    rmin = float(x[imin])
    emin = float(e0[imin])
    corrected = False

    def _zero_crossing(xa: np.ndarray, ea: np.ndarray) -> Optional[float]:
        """首个从正到负的过零点 (线性插值)。"""
        idx = np.where((xa[:-1] < rmin) & (ea[:-1] > 0) & (ea[1:] <= 0))[0]
        if len(idx) == 0:
            return None
        i = int(idx[0])
        frac = ea[i] / (ea[i] - ea[i + 1])
        return float(xa[i] + frac * (xa[i + 1] - xa[i]))

    sigma = _zero_crossing(x, e0)
    if emin < 0.0 and sigma is not None:
        return dict(sigma=sigma, eps=abs(emin), rmin=rmin, emin=emin,
                    tail=tail, corrected=False,
                    x_corr=x, e_corr=e_corr, f_corr=f_corr)

    # ---- 情形B: 正势阱/零势阱强制修正 (规范 §3) ----
    # 定位第二个二阶导数为零的点 (忽略靠近排斥壁的第一个拐点)
    cs = CubicSpline(x, e0)
    x2 = np.linspace(x[0], x[-1], max(len(x) * 4, 2000))
    d2 = cs(x2, 2)
    d2sign = np.sign(d2)
    # 二阶导数符号变化 => 拐点
    flips = np.where(np.diff(d2sign) != 0)[0]
    if len(flips) < 2:
        raise ValueError("未找到第二个拐点，此势表不适用于异核混合 (规范 §9)")
    i_infl2 = int(flips[1])
    r_infl2 = float(x2[i_infl2])
    v_infl2 = float(cs(r_infl2))

    e1 = e0 - v_infl2
    e_corr = e1
    imin1 = int(np.argmin(e1))
    rmin = float(x[imin1])
    emin = float(e1[imin1])
    sigma = _zero_crossing(x, e1)
    if sigma is None or emin >= 0.0:
        raise ValueError("情形B修正后仍无负势阱，此势表不适用于异核混合")
    corrected = True

    return dict(sigma=sigma, eps=abs(emin), rmin=rmin, emin=emin,
                tail=tail, corrected=corrected,
                x_corr=x, e_corr=e_corr, f_corr=f_corr)


# ============================================================================
# 核心混合算法
# ============================================================================

def mix_two_tables(path_a: str, path_b: str,
                   n_grid: Optional[int] = None,
                   tail_frac: float = 0.1,
                   opposite_sign: str = "force",
                   mix_mode: str = "geometric") -> Dict[str, object]:
    """
    按规范混合两张势能表。

    Args:
        path_a/path_b: 输入势能表路径
        n_grid: 公共网格点数 (默认取 max(N_A, N_B) + 1)
        tail_frac: 尾部归零化占比
        opposite_sign: 异号区处理方式 (仅 mix_mode='geometric' 有效)
            - 'force':  异号时强制取负 (规范默认, 保持吸引趋势连续, 力连续)
            - 'sign_a': 不强制, 全程跟随 sign(V*_A) (第二壳层区可能翻正且力不连续)
        mix_mode: 能量组合方式
            - 'geometric': 带符号几何平均 V*_mix = sign(V*_A)·√(|V*_A·V*_B|) (规范 §6)
            - 'arithmetic': E 算术平均 E_mix = (E_A + E_B)/2 (采样点真实能量平均,
                无符号问题、力连续, 偏离规范 §6/§7)

    Returns:
        dict(x, energy, force, sigma_mix, eps_mix, a, b, checks,
             corrected_a, corrected_b)
        corrected_a/b: 修正后的源表 (x_corr, e_corr, f_corr, 源文件名)
    """
    if not HAS_SCIPY:
        raise RuntimeError("需要 scipy，请安装: pip install scipy")

    xa, ea, fa = read_votca_table(path_a)
    xb, eb, fb = read_votca_table(path_b)

    ca = analyze_potential(xa, ea, tail_frac=tail_frac, force=fa)
    cb = analyze_potential(xb, eb, tail_frac=tail_frac, force=fb)

    # 归一化
    ea_c = ea - ca["tail"]
    eb_c = eb - cb["tail"]
    rsa = xa / ca["sigma"]
    rsb = xb / cb["sigma"]
    vsa = ea_c / ca["eps"]
    vsb = eb_c / cb["eps"]

    # 公共网格: 交集区间 [max(r*_min), min(r*_max)], 点数 > max(N_A, N_B)
    lo = max(rsa[0], rsb[0])
    hi = min(rsa[-1], rsb[-1])
    if hi <= lo:
        raise ValueError("两张表的无量纲横坐标无交集")
    n_max = max(len(rsa), len(rsb))
    if n_grid is None:
        n_grid = n_max + 1
    if n_grid <= n_max:
        raise ValueError(f"公共网格点数 {n_grid} 需大于原表最大点数 {n_max}")
    rs_grid = np.linspace(lo, hi, n_grid)

    # 三次样条插值 (规范 §5)
    spl_a = CubicSpline(rsa, vsa)
    spl_b = CubicSpline(rsb, vsb)
    vsa_g = spl_a(rs_grid)
    vsb_g = spl_b(rs_grid)

    # 去归一化标尺 (规范 §7)
    sigma_mix = (ca["sigma"] + cb["sigma"]) / 2.0
    eps_mix = np.sqrt(ca["eps"] * cb["eps"])

    # 力: 解析求导 (规范 §5 光滑性)  F = -dU/dr
    dvs_a = spl_a(rs_grid, 1)
    dvs_b = spl_b(rs_grid, 1)

    if mix_mode == "arithmetic":
        # E 算术平均: 采样点处真实修正能量的平均
        #   E_mix = (E_A + E_B)/2,  E_A = eps_A·V*_A,  E_B = eps_B·V*_B
        # 天然保留符号, 无需 sign/异号处理, 力连续
        eA = ca["eps"] * vsa_g
        eB = cb["eps"] * vsb_g
        e_real = 0.5 * (eA + eB)
        dE_dr = 0.5 * (ca["eps"] * dvs_a + cb["eps"] * dvs_b)
        force = -dE_dr / sigma_mix
    else:
        # 混合 (规范 §6): 同号取带符号几何平均; 异号区按 opposite_sign 策略
        prod = vsa_g * vsb_g
        diff_sign = np.sign(vsa_g) != np.sign(vsb_g)
        num = dvs_a * vsb_g + vsa_g * dvs_b   # d(vsA*vsB)/dr*
        denom = np.sqrt(np.abs(prod))
        if opposite_sign == "force":
            # 规范原版: 异号强制取负, 保持吸引趋势连续
            vmix = np.sign(vsa_g) * np.sqrt(np.abs(prod))
            vmix[diff_sign] = -np.sqrt(np.abs(prod[diff_sign]))
            with np.errstate(divide="ignore", invalid="ignore"):
                dvmix_dr = np.where(denom > 0,
                                    np.sign(vsa_g) * num / (2.0 * denom), 0.0)
                neg_dr = np.where(denom > 0, -num / (2.0 * denom), 0.0)
            dvmix_dr[diff_sign] = neg_dr[diff_sign]
        else:
            # sign_a: 不强制, 全程跟随 sign(V*_A)
            vmix = np.sign(vsa_g) * np.sqrt(np.abs(prod))
            with np.errstate(divide="ignore", invalid="ignore"):
                dvmix_dr = np.where(denom > 0,
                                    np.sign(vsa_g) * num / (2.0 * denom), 0.0)
        force = -eps_mix * dvmix_dr / sigma_mix
        e_real = vmix * eps_mix

    r_real = rs_grid * sigma_mix

    checks = verify_checks(ca, cb, sigma_mix, eps_mix, r_real, e_real)

    def _stem(p: str) -> str:
        stem = Path(p).stem
        if stem.endswith(".pot"):
            stem = stem[:-4]
        return stem

    return dict(x=r_real, energy=e_real, force=force,
                sigma_mix=sigma_mix, eps_mix=eps_mix,
                a=ca, b=cb, checks=checks,
                corrected_a=dict(x=ca["x_corr"], e=ca["e_corr"], f=ca["f_corr"],
                                 stem=_stem(path_a)),
                corrected_b=dict(x=cb["x_corr"], e=cb["e_corr"], f=cb["f_corr"],
                                 stem=_stem(path_b)))


def verify_checks(ca: Dict[str, float], cb: Dict[str, float],
                  sigma_mix: float, eps_mix: float,
                  r_real: np.ndarray, e_real: np.ndarray) -> Dict[str, bool]:
    """
    质量验收 (规范 §8): 零点位置 / 阱深范围 / 排斥区单调性。

    注: 两表各自以自身 sigma 归一化后，其过零点都落在公共约化坐标 r*=1 处，
    因此混合表的首次过零点位于 r_real = sigma_mix (算术平均)，天然介于
    sigma_A 与 sigma_B 之间。

    阱深检查用混合表实际主阱深度 |min(E)|，对几何/算术平均两种模式均适用。

    排斥区单调性以"避免非物理软塌陷"为准则: 随 r 增大，排斥壁能量必须严格
    单调递减 (即朝更小 r 方向严格递增)。
    """
    lo = min(ca["sigma"], cb["sigma"])
    hi = max(ca["sigma"], cb["sigma"])

    # 1. 零点位置: 混合表首次过零点应介于 sigma_A 与 sigma_B 之间
    zidx = np.where((r_real[:-1] > 0) & (e_real[:-1] > 0) & (e_real[1:] <= 0))[0]
    if len(zidx) > 0:
        i = int(zidx[0])
        frac = e_real[i] / (e_real[i] - e_real[i + 1])
        sigma_new = r_real[i] + frac * (r_real[i + 1] - r_real[i])
        check1 = lo <= sigma_new <= hi
    else:
        sigma_new = float("nan")
        check1 = False

    # 2. 阱深范围: 混合表实际主阱深度 |min(E)| 介于 min/max(eps_A, eps_B)
    well = float(abs(e_real[np.argmin(e_real)]))
    check2 = min(ca["eps"], cb["eps"]) <= well <= max(ca["eps"], cb["eps"])

    # 3. 排斥区 (r < sigma_mix) 单调性: 随 r 增大严格单调递减 (硬壁, 无软塌陷)
    rep = r_real < sigma_mix
    de = np.diff(e_real[rep])
    check3 = bool(np.all(de < 0)) if np.sum(rep) > 2 else False

    return dict(zero_crossing=float(sigma_new), between_sigma=check1,
                eps_between=check2, repulsive_monotonic=check3)


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="交叉混合两个 tabulated 势能表，生成异核 (cross) 势能表",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb33.pot.table -o nb13.pot.table
  python -m LmpPy.scripts.mix_cross_tabulated nb11.pot.table nb44.pot.table -o nb14.pot.table
  python -m LmpPy.scripts.mix_cross_tabulated nb22.pot.table nb33.pot.table -o nb23.pot.table
  python -m LmpPy.scripts.mix_cross_tabulated nb22.pot.table nb44.pot.table -o nb24.pot.table

规范: docs/cross-tabulation-mixing.md (v2.0)
  sigma_mix = (sigma_A + sigma_B) / 2,  eps_mix = sqrt(eps_A * eps_B)
  V*_mix = sign(V*_A) * sqrt(|V*_A * V*_B|),  异号时取负以保持吸引趋势
""",
    )
    parser.add_argument("table_a", help="第一个势能表 (如 nb11.pot.table)")
    parser.add_argument("table_b", help="第二个势能表 (如 nb33.pot.table)")
    parser.add_argument("-o", "--output", required=True, help="输出势能表路径 (如 nb13.pot.table)")
    parser.add_argument("--n-grid", type=int, default=None,
                        help="公共网格点数 (默认 max(N_A,N_B)+1)")
    parser.add_argument("--tail-frac", type=float, default=0.1,
                        help="尾部归零化占比 (0.1~0.2, 默认 0.1)")
    parser.add_argument("--mix-mode", choices=["geometric", "arithmetic"],
                        default="geometric",
                        help="能量组合方式: geometric=带符号几何平均 (规范 §6, 默认) / "
                             "arithmetic=E 算术平均 (E_mix=(E_A+E_B)/2, 无符号问题)")
    parser.add_argument("--opposite-sign", choices=["force", "sign_a"], default="force",
                        help="异号区处理 (仅 --mix-mode geometric 有效): "
                             "force=强制取负 (规范默认, 吸引趋势连续/力连续) / "
                             "sign_a=不强制, 跟随 sign(V*_A) (第二壳层区可能翻正且力不连续)")
    parser.add_argument("--dump-corrected", default=None, metavar="DIR",
                        help="同时输出修正后的输入势能表到 DIR"
                             " (命名 <源文件名>.corrected.pot.table)")

    args = parser.parse_args()

    if not HAS_SCIPY:
        print("错误: 需要 scipy (pip install scipy)")
        return 1

    try:
        result = mix_two_tables(args.table_a, args.table_b,
                                n_grid=args.n_grid, tail_frac=args.tail_frac,
                                opposite_sign=args.opposite_sign,
                                mix_mode=args.mix_mode)
    except Exception as e:
        print(f"错误: {e}")
        return 1

    ca, cb = result["a"], result["b"]
    if args.mix_mode == "arithmetic":
        mode_desc = "arithmetic (E 算术平均: E_mix=(E_A+E_B)/2)"
    else:
        mode_desc = ("geometric (带符号几何平均, 规范 §6)" +
                     (" + force (异号强制取负)" if args.opposite_sign == "force"
                      else " + sign_a (异号跟随 sign(V*_A))"))
    print("=" * 60)
    print("交叉混合势能表")
    print("=" * 60)
    print(f"能量组合: {mode_desc}")
    print(f"表A: sigma={ca['sigma']:.4f} A, eps={ca['eps']:.4f} kcal/mol"
          f"{' (情形B修正)' if ca['corrected'] else ''}")
    print(f"表B: sigma={cb['sigma']:.4f} A, eps={cb['eps']:.4f} kcal/mol"
          f"{' (情形B修正)' if cb['corrected'] else ''}")
    print(f"sigma_mix = {result['sigma_mix']:.4f} A (算术平均)")
    print(f"eps_mix   = {result['eps_mix']:.4f} kcal/mol (几何平均)")

    print("\n质量验收 (规范 §8):")
    chk = result["checks"]
    print(f"  [1] 零点位置: 混合表过零点 r0={chk['zero_crossing']:.4f} A"
          f" 介于 sigma_A/B 之间: {'通过' if chk['between_sigma'] else '失败'}")
    print(f"  [2] 阱深范围: eps_mix 介于 min/max(eps_A,eps_B): "
          f"{'通过' if chk['eps_between'] else '失败'}")
    print(f"  [3] 排斥区单调性 (r<sigma_mix, 随 r 增大能量递减): "
          f"{'通过' if chk['repulsive_monotonic'] else '失败'}")

    ok = all([chk["between_sigma"], chk["eps_between"], chk["repulsive_monotonic"]])
    if not ok:
        print("\n警告: 存在未通过的质量验收项，请检查输入势能表")

    write_votca_table(args.output, result["x"], result["energy"], result["force"])

    if args.dump_corrected:
        cdir = Path(args.dump_corrected)
        cdir.mkdir(parents=True, exist_ok=True)
        for corr in (result["corrected_a"], result["corrected_b"]):
            if corr["f"] is None:
                continue
            out_path = cdir / f"{corr['stem']}.corrected.pot.table"
            write_votca_table(str(out_path), corr["x"], corr["e"], corr["f"])
        print("已输出修正后的输入势能表 (尾部归零, 情形B再做拐点平移)")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
