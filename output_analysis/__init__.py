"""LmpPy 后处理分析模块。

提供三种核心分析:
  - 链长分布 (chain_length)
  - 反应距离分布 (distance)
  - 反应统计 (reaction_stats)

以及统一的绘图配置系统 (plot.py + plot_config.yaml)。

CLI 入口:
  python -m LmpPy.output_analysis chain-length <dir>
  python -m LmpPy.output_analysis distance <dir>
  python -m LmpPy.output_analysis reaction-stats <dir>
  python -m LmpPy.output_analysis all <dir>
  python -m LmpPy.output_analysis rebuild <npz_path> -o <csv_path>
"""
