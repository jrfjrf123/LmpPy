"""Chain length distribution analysis.

Computes chain length distribution from final_frame.data bond topology,
compares with theoretical Schulz-Zimm / Poisson / Log-normal curves,
and optionally calculates Jensen-Shannon divergence.

Plots histogram + KDE overlay with theoretical curve comparison.
"""

from pathlib import Path
from typing import Optional, Dict
import numpy as np
from scipy.stats import gaussian_kde

from .loader import load_final_frame_topology
from .theory import get_theory_curve
from .js_divergence import js_divergence
from .plot import (
    setup_style, save_figure, new_figure,
    PlotConfig, ChartStyle, HistogramStyle, KDEStyle,
    get_config, set_config,
)


def _compute_kde(data: np.ndarray, x_grid: np.ndarray, bw: str = "scott") -> np.ndarray:
    """计算 KDE 曲线。

    参数
    ----
    data : 原始数据
    x_grid : 评估 KDE 的横轴网格
    bw : bandwidth 方法 ("scott", "silverman") 或 float 因子字符串

    返回
    ----
    KDE 估计的概率密度值
    """
    if len(data) < 2:
        return np.zeros_like(x_grid)

    kde = gaussian_kde(data)

    # 带宽调整
    if bw == "scott":
        # scott's rule 是 gaussian_kde 的默认行为
        pass
    elif bw == "silverman":
        n = len(data)
        kde.set_bandwidth(kde.factor * (n ** (-1.0 / 5.0)) / (n ** (-1.0 / (4.0 + data.ndim))))
    else:
        try:
            factor = float(bw)
            kde.set_bandwidth(kde.factor * factor)
        except ValueError:
            pass  # 保持 scott 默认

    return kde.evaluate(x_grid)


def analyze_chain_length(
    chain_lengths: Optional[np.ndarray] = None,
    input_dir: Optional[str | Path] = None,
    min_length: int = 20,
    theory_type: str = "schulz-zimm",
    Mn: Optional[float] = None,
    PDI: Optional[float] = None,
    js: bool = False,
    bins: int = 50,
    output_dir: Optional[str | Path] = None,
    plot_config: Optional[str | Path] = None,
    kde_bw: Optional[str] = None,
) -> Dict:
    """Analyze chain length distribution with theoretical curve comparison.

    Parameters
    ----------
    chain_lengths : array of chain lengths (computed from final_frame.data if None)
    input_dir : simulation output directory
    min_length : minimum chain length threshold
    theory_type : "schulz-zimm", "poisson", or "lognormal"
    Mn : theoretical number-average chain length (uses simulated value if None)
    PDI : theoretical polydispersity index (ignored for Poisson)
    js : compute JS divergence if True
    bins : histogram bin count (CLI override)
    output_dir : output directory for plots
    plot_config : path to plot_config.yaml
    kde_bw : KDE bandwidth override (CLI, highest priority)

    Returns
    -------
    dict with simulated_Mn, simulated_PDI, js_divergence, etc.
    """
    # --- 加载配置 (三层优先级: CLI > YAML > 默认) ---
    config = PlotConfig.from_yaml(plot_config)
    set_config(config)
    chart = config.get_chart("chain_length")

    # CLI 覆盖
    if bins is not None:
        if chart.histogram is None:
            chart.histogram = HistogramStyle()
        chart.histogram.bins = bins
    if kde_bw is not None:
        if chart.kde is None:
            chart.kde = KDEStyle()
        chart.kde.bandwidth = kde_bw
    chart = chart.fill_defaults()
    setup_style(chart)

    # --- 数据加载 ---
    if chain_lengths is None:
        if input_dir is None:
            raise ValueError("Must provide either chain_lengths or input_dir")
        topo = load_final_frame_topology(input_dir)
        chain_lengths = topo["chain_lengths"]

    if output_dir is None and input_dir is not None:
        output_dir = Path(input_dir) / "analysis"
    elif output_dir is None:
        output_dir = Path(".")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- 筛选 ---
    long_chains = chain_lengths[chain_lengths >= min_length]
    if len(long_chains) == 0:
        raise ValueError(
            f"No chains with length >= {min_length}. "
            f"Max chain length: {chain_lengths.max() if len(chain_lengths) > 0 else 'N/A'}. "
            f"Try lowering --min-length."
        )

    # --- 模拟统计量 ---
    simulated_Mn = float(np.mean(long_chains))
    simulated_Mw = float(np.sum(long_chains.astype(np.float64) ** 2) / long_chains.sum())
    simulated_PDI = simulated_Mw / simulated_Mn

    # --- 理论曲线 ---
    if Mn is None:
        Mn = simulated_Mn
        print(f"[chain-length] Using simulated Mn={Mn:.1f} for theoretical curve")
    if PDI is None:
        if theory_type != "poisson":
            PDI = simulated_PDI
            print(f"[chain-length] Using simulated PDI={PDI:.3f} for theoretical curve")

    N_min = max(1, int(min_length * 0.8))
    N_max = max(int(simulated_Mn * 3), int(long_chains.max() * 1.1), int(Mn * 3))
    N_grid = np.linspace(N_min, N_max, 500)

    try:
        theory_curve, theory_meta = get_theory_curve(theory_type, N_grid, Mn, PDI)
    except ValueError as e:
        print(f"[chain-length] Theory curve error: {e}")
        theory_curve = None
        theory_meta = {}

    # --- JS 散度 ---
    js_value = None
    if js and theory_curve is not None:
        hist, bin_edges = np.histogram(long_chains, bins=chart.histogram.bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        theory_at_bins = np.interp(bin_centers, N_grid, theory_curve)
        hist_norm = hist / (hist.sum() + 1e-12)
        theory_norm = theory_at_bins / (theory_at_bins.sum() + 1e-12)
        js_value = js_divergence(hist_norm, theory_norm)

    # ================================================================
    # 绘图
    # ================================================================

    fig, ax = new_figure(figsize=chart.figure.figsize)

    # --- 直方图 (低透明度) ---
    hist_style = chart.histogram
    ax.hist(long_chains, bins=hist_style.bins, density=True,
            alpha=hist_style.alpha, color=hist_style.color,
            edgecolor=hist_style.edge_color,
            label=f"Simulation (Mn={simulated_Mn:.1f}, PDI={simulated_PDI:.3f}, n={len(long_chains)})")

    # --- KDE 曲线 (仅线条，无填充) ---
    kde_style = chart.kde
    x_kde = np.linspace(N_min, N_max, 300)
    kde_y = _compute_kde(long_chains, x_kde, bw=kde_style.bandwidth)
    ax.plot(x_kde, kde_y, color=kde_style.color, linewidth=kde_style.linewidth,
            linestyle=kde_style.linestyle, alpha=kde_style.alpha,
            label=f"KDE (bw={kde_style.bandwidth})")

    # --- 理论曲线 ---
    if theory_curve is not None:
        tl = chart.theory_line
        label = f"{theory_type} theory (Mn={Mn:.1f}"
        if theory_type == "poisson":
            label += f", PDI={theory_meta.get('PDI', 0):.4f})"
        else:
            label += f", PDI={PDI:.3f})"
        if js_value is not None:
            label += f" [JSD={js_value:.4f}]"
        ax.plot(N_grid, theory_curve, color=tl.color, linewidth=tl.linewidth,
                linestyle=tl.linestyle, alpha=tl.alpha, label=label)

    # --- 均值竖线 ---
    ml = chart.mean_line
    ax.axvline(simulated_Mn, color="blue", linestyle="--", alpha=ml.alpha,
               label=f"Sim. Mn={simulated_Mn:.0f}")
    ax.axvline(Mn, color=ml.color, linestyle=ml.linestyle, alpha=ml.alpha,
               label=f"Theory Mn={Mn:.0f}")

    # --- 轴标签 ---
    axis_cfg = chart.axis
    xlabel = axis_cfg.xlabel or "Chain Length (beads)"
    ylabel = axis_cfg.ylabel or "Probability Density"
    title = axis_cfg.title or f"Chain Length Distribution (min_length={min_length})"
    ax.set_xlabel(xlabel, fontsize=axis_cfg.label_fontsize, labelpad=axis_cfg.label_pad)
    ax.set_ylabel(ylabel, fontsize=axis_cfg.label_fontsize, labelpad=axis_cfg.label_pad)
    ax.set_title(title, fontsize=axis_cfg.title_fontsize, pad=axis_cfg.title_pad)

    # --- 图例 ---
    lg = chart.legend
    ax.legend(loc=lg.loc, fontsize=lg.font_size, framealpha=lg.frame_alpha)

    fig.tight_layout()

    output_path = output_dir / "chain_length_distribution.png"
    saved_path = save_figure(fig, output_path, dpi=chart.figure.dpi,
                             save_format=chart.figure.save_format)
    print(f"[chain-length] Saved to: {saved_path}")

    # --- 摘要 ---
    print(f"\n=== Chain Length Distribution ===")
    print(f"  Total molecules: {len(chain_lengths)}")
    print(f"  Long chains (bead>={min_length}): {len(long_chains)}")
    print(f"  Simulated Mn={simulated_Mn:.1f}, Mw={simulated_Mw:.1f}, PDI={simulated_PDI:.4f}")
    if theory_type == "poisson":
        print(f"  Theory Poisson: Mn={Mn:.1f}, actual PDI={theory_meta.get('PDI', 0):.4f}")
    else:
        print(f"  Theory {theory_type}: Mn={Mn:.1f}, PDI={PDI:.3f}")
    if js_value is not None:
        print(f"  JS Divergence: {js_value:.6f}")
    print(f"  Chain length range: {long_chains.min()} - {long_chains.max()}")
    print(f"  Top 10 longest: {long_chains[:10]}")

    return {
        "simulated_Mn": simulated_Mn,
        "simulated_Mw": simulated_Mw,
        "simulated_PDI": simulated_PDI,
        "n_chains": len(chain_lengths),
        "n_long_chains": len(long_chains),
        "js_divergence": js_value,
        "theory_meta": theory_meta,
    }
