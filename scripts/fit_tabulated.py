#!/usr/bin/env python
"""
Tabulated 势能表解析拟合脚本 - 将 VOTCA/LAMMPS table 势拟合为解析函数

用法:
    python -m LmpPy.scripts.fit_tabulated -d ./mdrun
    python -m LmpPy.scripts.fit_tabulated -d ./mdrun -o ./fit_output
    python -m LmpPy.scripts.fit_tabulated nb11.pot.table bond1.pot.table angle1.pot.table

拟合形式:
    nonbonded  -> 12-6 Lennard-Jones:   U(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]
                  --lj-method well:    势阱特征取参, 以壳层间势垒顶为零点:
                                       eps=E_barr-E_well, sigma=壁上 E=E_barr 过点,
                                       cutoff=r_barr (默认, 适合带壳层结构的 IBI 势)
                  --lj-method direct:  尾部基线归零后取 sigma=U=0 第一个过零点、
                                       eps=势阱深度
                  --lj-method explore: 最小二乘 + 拟合区间循环探索
    bond       -> harmonic:             U(r) = k*(r - r0)^2
    angle      -> cos:  U(theta) = k*[1 - cos(theta - theta0)]  (失败或明显更差时回退 harmonic)

输出:
    <output>/fit_summary.txt   拟合参数汇总 + LAMMPS 就绪命令
    <output>/plots/*.png       tabulated vs 拟合对比图

作者: Claude
日期: 2026-08-13
"""

import sys
import math
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

# 添加项目根目录到 sys.path
_project_root = Path(__file__).resolve().parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from LmpPy.tools.ibm_potential.tabulated_potential import (
    collect_tables,
    fit_table,
    plot_fit_comparison,
    FitResult,
    HAS_SCIPY,
    HAS_MATPLOTLIB,
)


def _fmt_ids(ids) -> str:
    return "".join(str(i) for i in ids)


def unified_angle_style(angle_results: List[FitResult]) -> str:
    """
    为所有 angle 结果选择统一的 LAMMPS angle_style。

    LAMMPS 一次模拟只允许一种 angle_style，因此所有 angle 必须共用同一形式，
    系数由 convert_angle_coeff 做近似换算。
    """
    if all(r.form == "harmonic_angle" for r in angle_results):
        return "harmonic"
    if all(abs(r.params["theta0"] - 180.0) < 1.0 for r in angle_results):
        return "cosine"
    return "cosine/squared"


def convert_angle_coeff(result: FitResult, style: str) -> Tuple[float, Optional[float], str]:
    """
    将拟合参数近似换算为指定 angle_style 的系数。

    换算关系（Δθ 以弧度计）:
        cos 形式:           U = k*(1 - cos(Δθ))      ≈ k*Δθ²/2
        harmonic 形式:      U = K*Δθ²
        cosine (θ0=180°):   U = K*(1 + cos θ)        ≈ K*Δθ²/2
        cosine/squared:     U = K*(cosθ - cosθ0)²    ≈ K*sin²θ0*Δθ²

    Returns:
        (K, theta0 或 None, 换算说明)
    """
    k = result.params["k"]
    theta0 = result.params["theta0"]
    s = math.sin(math.radians(theta0))

    if style == "harmonic":
        if result.form == "harmonic_angle":
            return k, theta0, ""
        return k / 2.0, theta0, (f"cos 形式 k={k:.4f} 换算: 1-cos(Δθ)≈Δθ²/2 -> K=k/2")

    if style == "cosine":
        # theta0≈180°: k*(1-cos(θ-180°)) = k*(1+cosθ)，与 cosine 形式 K*(1+cosθ) 一致
        if result.form == "cos_angle":
            return k, None, "theta0≈180°，k*(1+cosθ) 与 cosine 形式等价"
        return 2.0 * k, None, f"harmonic k={k:.4f} 换算: K*(1+cosθ)≈K*Δθ²/2 -> K=2k"

    # cosine/squared
    warn = ""
    if abs(s) < 0.1:
        warn = ("警告: theta0 接近 0/180°，sin²θ0≈0，cosine/squared 在平衡点附近退化，"
                "建议改用 harmonic; ")
    if result.form == "cos_angle":
        return k / (2.0 * s * s), theta0, warn + f"K=k/(2*sin²θ0)，cos 形式 k={k:.4f}"
    return k / (s * s), theta0, warn + f"K=k/sin²θ0，harmonic 形式 k={k:.4f}"


def lj_coeff_line(result: FitResult) -> str:
    eps = result.params["eps"]
    sigma = result.params["sigma"]
    ids = result.type_ids
    return f"pair_coeff {ids[0]} {ids[1]} {eps:.6g} {sigma:.6g}"


def bond_coeff_line(result: FitResult) -> str:
    k = result.params["k"]
    r0 = result.params["r0"]
    return f"bond_coeff {_fmt_ids(result.type_ids)} {k:.6f} {r0:.6f}"


def describe_params(result: FitResult) -> List[str]:
    """生成参数描述行。"""
    lines = []
    p = result.params
    if result.form == "lj_12_6":
        lines.append(f"    形式: U(r) = 4*eps*[(sigma/r)^12 - (sigma/r)^6]")
        lines.append(f"    eps   = {p['eps']:.6g} kcal/mol")
        lines.append(f"    sigma = {p['sigma']:.6g} A")
        lines.append(f"    r_min = {p['rmin']:.6f} A  (势能最低点)")
        lines.append(f"    cutoff= {p['cutoff']:.4f} A")
    elif result.form == "harmonic_bond":
        lines.append(f"    形式: U(r) = k*(r - r0)^2  (+e0={p['e0']:.6f})")
        lines.append(f"    k  = {p['k']:.6f} kcal/mol/A^2")
        lines.append(f"    r0 = {p['r0']:.6f} A")
    elif result.form in ("cos_angle", "harmonic_angle"):
        if result.form == "cos_angle":
            lines.append(f"    形式: U(theta) = k*[1 - cos(theta - theta0)]  (+e0={p['e0']:.6f})")
            lines.append(f"    k      = {p['k']:.6f} kcal/mol")
        else:
            lines.append(f"    形式: U(theta) = k*(theta - theta0)^2  (+e0={p['e0']:.6f})")
            lines.append(f"    k      = {p['k']:.6f} kcal/mol/rad^2")
        lines.append(f"    theta0 = {p['theta0']:.4f} deg")
    return lines


def write_summary(results: List[FitResult], output_dir: Path, energy_unit: str) -> str:
    """写入拟合汇总文件。"""
    summary_path = output_dir / "fit_summary.txt"

    pair_results = [r for r in results if r.form == "lj_12_6"]
    bond_results = [r for r in results if r.form == "harmonic_bond"]
    angle_results = [r for r in results if r.form in ("cos_angle", "harmonic_angle")]

    lines = []
    lines.append("=" * 70)
    lines.append("Tabulated 势能表解析拟合结果")
    lines.append("=" * 70)
    lines.append(f"能量单位: {energy_unit}")
    lines.append("注意: 能量单位仅为标注，脚本未做任何单位换算。")
    lines.append("      VOTCA 输出通常为 kJ/mol，用于 LAMMPS (real 单位为 kcal/mol)")
    lines.append("      前请确认单位一致或自行换算。")
    lines.append("")

    # 逐项结果
    for r in results:
        lines.append("-" * 70)
        lines.append(f"{r.kind} {r.name} (type {_fmt_ids(r.type_ids)})")
        lines.append(f"    {r.note}")
        lines.extend(describe_params(r))
        lines.append(f"    RMSE = {r.rmse:.6e}   R^2 = {r.r_squared:.6f}")

    # LAMMPS 就绪命令
    lines.append("")
    lines.append("=" * 70)
    lines.append("LAMMPS 就绪命令")
    lines.append("=" * 70)
    lines.append("")

    if pair_results:
        max_cutoff = max(r.params["cutoff"] for r in pair_results)
        lines.append("# --- nonbonded (12-6 LJ) ---")
        lines.append(f"pair_style lj/cut {max_cutoff:.4f}")
        for r in pair_results:
            lines.append(lj_coeff_line(r) + f"   # cutoff={r.params['cutoff']:.3f} A")
        lines.append("")

    if bond_results:
        lines.append("# --- bond (harmonic) ---")
        lines.append("bond_style harmonic")
        for r in bond_results:
            lines.append(bond_coeff_line(r))
        lines.append("")

    if angle_results:
        lines.append("# --- angle ---")
        style = unified_angle_style(angle_results)
        lines.append(f"angle_style {style}")
        forms = {r.form for r in angle_results}
        if len(forms) > 1:
            lines.append("# 注: 各 angle 的拟合形式不同，已统一为单一 angle_style")
            lines.append("#     (LAMMPS 一次模拟只允许一种 angle_style)，系数做了近似换算，")
            lines.append("#     请结合 plots/ 中的对比图确认精度。")
        for r in angle_results:
            K, theta0, comment = convert_angle_coeff(r, style)
            if comment:
                lines.append(f"# {r.name}: {comment}")
            if theta0 is None:
                coeff = f"angle_coeff {_fmt_ids(r.type_ids)} {K:.6f}"
            else:
                coeff = f"angle_coeff {_fmt_ids(r.type_ids)} {K:.6f} {theta0:.6f}"
            if comment.startswith("警告"):
                # 退化情形 (sin²θ0≈0)，换算系数无意义，注释掉防止误用
                lines.append(f"# {coeff}   <-- 已禁用，见上方警告")
            else:
                lines.append(coeff)
        lines.append("")

    lines.append("# 注: e0 为拟合的势能零点偏移，不影响力，LAMMPS 中可忽略。")

    text = "\n".join(lines) + "\n"
    summary_path.write_text(text)
    return str(summary_path)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="将 LAMMPS/VOTCA tabulated 势能表拟合为解析势函数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 拟合目录下所有势能表
  python -m LmpPy.scripts.fit_tabulated -d ./mdrun

  # 指定输出目录
  python -m LmpPy.scripts.fit_tabulated -d ./mdrun -o ./fit_output

  # 拟合指定文件
  python -m LmpPy.scripts.fit_tabulated nb11.pot.table bond1.pot.table angle1.pot.table

  # angle 强制使用 harmonic
  python -m LmpPy.scripts.fit_tabulated -d ./mdrun --angle-form harmonic

  # 不生成对比图
  python -m LmpPy.scripts.fit_tabulated -d ./mdrun --no-plot
""",
    )

    parser.add_argument("files", nargs="*", help="输入势能表文件")
    parser.add_argument("-d", "--directory", help="处理目录下所有势能表文件")
    parser.add_argument("-o", "--output", default="./fit_output", help="输出目录 (默认: ./fit_output)")
    parser.add_argument("--energy-unit", default="kcal/mol",
                        help="能量单位标签 (仅用于标注，不做换算; VOTCA 通常为 kJ/mol, 默认: kcal/mol)")
    parser.add_argument("--angle-form", choices=["auto", "cos", "harmonic"], default="auto",
                        help="angle 拟合形式 (默认: auto, 先尝试 cos，失败或明显更差时回退 harmonic)")
    parser.add_argument("--u-cut", type=float, default=10.0,
                        help="bond/angle 势阱区选择阈值，只拟合 U-Umin<=u_cut 的点 (默认: 10.0)")
    parser.add_argument("--rmin-fit", type=float, default=None,
                        help="nonbonded 拟合区间左端点 (A)，指定后跳过区间探索")
    parser.add_argument("--rmax-fit", type=float, default=None,
                        help="nonbonded 拟合区间右端点 (A)，指定后跳过区间探索")
    parser.add_argument("--lj-method", choices=["well", "direct", "explore"], default="well",
                        help="nonbonded 取参方式: well=势阱特征取参, 以壳层间势垒顶为零点 "
                             "(eps=E_barr-E_well, sigma=壁上 E=E_barr 过点, cutoff=r_barr) "
                             "(默认); direct=尾部基线归零后取 U=0 第一个过零点+势阱深度; "
                             "explore=最小二乘区间探索")
    parser.add_argument("--no-plot", action="store_true", help="不生成对比图")
    parser.add_argument("--y-max", type=float, default=None,
                        help="nonbonded 对比图能量上限 (kcal/mol)")
    parser.add_argument("-q", "--quiet", action="store_true", help="安静模式")

    args = parser.parse_args()

    if not HAS_SCIPY:
        print("错误: 拟合需要 scipy，请安装: pip install scipy")
        return 1

    try:
        tables = collect_tables(args.files, args.directory, args.energy_unit)
    except FileNotFoundError as e:
        print(f"错误: {e}")
        return 1

    # 过滤掉无法拟合的类型 (dihedral 暂不支持)
    fit_tables = [t for t in tables if t.kind in ("nonbonded", "bond", "angle")]
    skipped = [t for t in tables if t.kind not in ("nonbonded", "bond", "angle")]

    if not fit_tables:
        print("错误: 未找到可拟合的势能表 (支持 nonbonded/bond/angle)")
        return 1

    if not args.quiet:
        print("=" * 60)
        print("Tabulated 势能表解析拟合")
        print("=" * 60)
        print(f"能量单位: {args.energy_unit}")
        print(f"待拟合势能表: {len(fit_tables)}")
        for t in fit_tables:
            ids = "".join(str(i) for i in t.type_ids)
            print(f"  {t.kind}: {t.name} (type {ids}, {t.n} points)")
        if skipped:
            print(f"跳过 (暂不支持): {', '.join(t.name for t in skipped)}")

    results = []
    for t in fit_tables:
        try:
            r = fit_table(
                t,
                angle_form=args.angle_form,
                u_cut=args.u_cut,
                rmin_fit=args.rmin_fit,
                rmax_fit=args.rmax_fit,
                lj_method=args.lj_method,
            )
            results.append(r)
            if not args.quiet:
                ids = "".join(str(i) for i in t.type_ids)
                print(f"\n  {t.kind} {t.name}:")
                for ln in describe_params(r):
                    print(f"    {ln.strip()}")
                print(f"    RMSE = {r.rmse:.6e}   R^2 = {r.r_squared:.6f}")
        except Exception as e:
            print(f"  拟合失败 {t.name}: {e}")

    if not results:
        print("错误: 所有势能表拟合均失败")
        return 1

    # 输出目录
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 写汇总
    summary_path = write_summary(results, output_dir, args.energy_unit)

    # 对比图
    if not args.no_plot:
        if not HAS_MATPLOTLIB:
            print("警告: 需要 matplotlib 才能生成对比图 (pip install matplotlib)")
        else:
            plot_dir = output_dir / "plots"
            plot_dir.mkdir(parents=True, exist_ok=True)
            for r in results:
                out = plot_dir / f"{r.name}_fit.png"
                try:
                    plot_fit_comparison(r, str(out), energy_cap=args.y_max)
                except Exception as e:
                    print(f"  对比图失败 {r.name}: {e}")

    if not args.quiet:
        print("\n" + "=" * 60)
        print(f"✓ 完成! 拟合汇总: {summary_path}")
        if not args.no_plot and HAS_MATPLOTLIB:
            print(f"  对比图目录: {output_dir / 'plots'}")
        print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
