"""统一可视化配置模块。

提供 matplotlib + seaborn 公共样式设置和图片保存函数。
包含三层优先级的嵌套 dataclass 配置系统:
  1. dataclass 默认值 (最低)
  2. 用户 YAML 文件 (中间)
  3. CLI 参数覆盖 (最高)
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import Optional, Dict, Any, Tuple
import copy

import matplotlib
import matplotlib.font_manager as fm
matplotlib.use("Agg")  # 无 GUI 后端，适合服务器环境
import matplotlib.pyplot as plt
import seaborn as sns
import yaml

# --- 模块级字体配置 ---
_AVAILABLE_FONTS = {f.name for f in fm.fontManager.ttflist}
_CJK_CANDIDATES = [
    "Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Serif CJK SC",
    "AR PL UMing CN", "AR PL UKai CN",
    "SimHei", "WenQuanYi Micro Hei",
]
for _font in _CJK_CANDIDATES:
    if _font in _AVAILABLE_FONTS:
        plt.rcParams["font.family"] = _font
        break


# ======================================================================
# 嵌套 dataclass 配置
# ======================================================================

@dataclass
class LineStyle:
    """线条样式 (理论曲线、均值线等)。"""
    linewidth: float = 2.0
    linestyle: str = "-"          # "-" "--" ":" "-."
    alpha: float = 0.85
    color: str = "#DD4444"        # 默认红色

    @classmethod
    def _merge(cls, base: Optional[LineStyle], override: Optional[LineStyle]) -> LineStyle:
        """合并两个 LineStyle，override 中非 None 的字段覆盖 base。"""
        return _merge_dataclass(cls(), base, override)


@dataclass
class KDEStyle:
    """KDE 曲线样式。"""
    bandwidth: str = "scott"       # "scott" | "silverman" | float因子字符串
    linewidth: float = 2.0
    linestyle: str = "-"
    alpha: float = 0.85
    color: str = "#FF7F0E"         # 橙色

    @classmethod
    def _merge(cls, base: Optional[KDEStyle], override: Optional[KDEStyle]) -> KDEStyle:
        return _merge_dataclass(cls(), base, override)


@dataclass
class HistogramStyle:
    """直方图样式。"""
    alpha: float = 0.4             # 30-50% 透明度
    edge_color: str = "white"
    bins: int = 50
    color: str = "#4C72B0"         # 蓝色

    @classmethod
    def _merge(cls, base: Optional[HistogramStyle], override: Optional[HistogramStyle]) -> HistogramStyle:
        return _merge_dataclass(cls(), base, override)


@dataclass
class FigureStyle:
    """画布样式。"""
    dpi: int = 150
    figsize: Tuple[float, float] = (10, 6)
    save_format: str = "png"       # png | pdf | svg

    @classmethod
    def _merge(cls, base: Optional[FigureStyle], override: Optional[FigureStyle]) -> FigureStyle:
        return _merge_dataclass(cls(), base, override)


@dataclass
class FontStyle:
    """字体样式。"""
    family: str = "DejaVu Sans"
    title_size: int = 14
    label_size: int = 12
    tick_size: int = 10
    legend_size: int = 9

    @classmethod
    def _merge(cls, base: Optional[FontStyle], override: Optional[FontStyle]) -> FontStyle:
        return _merge_dataclass(cls(), base, override)


@dataclass
class AxisText:
    """坐标轴文本内容及样式。"""
    title: str = ""                # 空 = 使用内置默认标题
    xlabel: str = ""
    ylabel: str = ""
    title_fontsize: int = 14
    label_fontsize: int = 12
    title_pad: float = 10.0
    label_pad: float = 8.0

    @classmethod
    def _merge(cls, base: Optional[AxisText], override: Optional[AxisText]) -> AxisText:
        return _merge_dataclass(cls(), base, override)


@dataclass
class LegendStyle:
    """图例样式。"""
    loc: str = "best"
    frame_alpha: float = 0.8
    font_size: int = 9

    @classmethod
    def _merge(cls, base: Optional[LegendStyle], override: Optional[LegendStyle]) -> LegendStyle:
        return _merge_dataclass(cls(), base, override)


# ======================================================================
# ChartStyle — 单个图表的完整样式
# ======================================================================

@dataclass
class ChartStyle:
    """单个图表的完整样式。

    所有子字段 Optional: None 表示沿用上层 (default) 配置。
    """
    figure: Optional[FigureStyle] = None
    font: Optional[FontStyle] = None
    histogram: Optional[HistogramStyle] = None
    kde: Optional[KDEStyle] = None
    theory_line: Optional[LineStyle] = None       # 理论曲线
    reference: Optional[HistogramStyle] = None      # 参考分布直方图
    legend: Optional[LegendStyle] = None
    axis: Optional[AxisText] = None
    mean_line: Optional[LineStyle] = None           # 均值竖线

    @classmethod
    def _merge(cls, base: Optional[ChartStyle], override: Optional[ChartStyle]) -> ChartStyle:
        """合并两个 ChartStyle。

        base 的 None 子字段被 override 的非 None 同名字段替换。
        如果 override 的某个子字段也为 None，保持 base 的值 (可能仍为 None)。
        """
        result = ChartStyle()
        for f in fields(ChartStyle):
            base_val = getattr(base, f.name) if base is not None else None
            ovr_val = getattr(override, f.name) if override is not None else None
            # 子 dataclass 的合并
            if ovr_val is not None:
                setattr(result, f.name, ovr_val)
            elif base_val is not None:
                setattr(result, f.name, base_val)
            else:
                setattr(result, f.name, None)
        return result

    def fill_defaults(self) -> ChartStyle:
        """将 None 字段填充为硬编码默认值，返回新的 ChartStyle (全填充)。"""
        filled = ChartStyle()
        filled.figure = self.figure or FigureStyle()
        filled.font = self.font or FontStyle()
        filled.histogram = self.histogram or HistogramStyle()
        filled.kde = self.kde or KDEStyle()
        filled.theory_line = self.theory_line or LineStyle(color="#DD4444")
        filled.reference = self.reference or HistogramStyle(color="#DD8452")
        filled.legend = self.legend or LegendStyle()
        filled.axis = self.axis or AxisText()
        filled.mean_line = self.mean_line or LineStyle(color="red", linestyle="--", alpha=0.6)
        return filled


# ======================================================================
# PlotConfig — 全局配置，三层优先级
# ======================================================================

@dataclass
class PlotConfig:
    """全局绘图配置。

    三层优先级: CLI 参数 > 用户 YAML > dataclass 默认值
    """
    default: ChartStyle = field(default_factory=ChartStyle)
    chain_length: ChartStyle = field(default_factory=ChartStyle)
    distance: ChartStyle = field(default_factory=ChartStyle)
    reaction_stats: ChartStyle = field(default_factory=ChartStyle)

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str | Path] = None) -> PlotConfig:
        """从 YAML 文件加载配置，返回 PlotConfig。

        YAML 中的字段覆盖 dataclass 默认值，未出现的字段保持默认。
        """
        config = cls()  # 全默认

        if yaml_path is not None:
            yaml_path = Path(yaml_path)
            if yaml_path.exists():
                with open(yaml_path, "r") as f:
                    raw: Dict[str, Any] = yaml.safe_load(f) or {}
                config = _dict_to_plot_config(raw)

        return config

    def get_chart(self, chart_name: str) -> ChartStyle:
        """获取指定图表的最终样式 (合并 default + chart 两层)。

        参数
        ----
        chart_name : "chain_length" | "distance" | "reaction_stats"

        返回
        ----
        合并后的 ChartStyle (子字段仍可能为 None，需 fill_defaults() 填充)
        """
        chart_override = getattr(self, chart_name, ChartStyle())
        return ChartStyle._merge(self.default, chart_override)


# ======================================================================
# 合并辅助函数
# ======================================================================

def _merge_dataclass(result, base, override):
    """通用 dataclass 字段级合并: override > base > result 默认。

    将 base 和 override 中的非 None 字段逐一覆盖到 result 上。
    """
    if base is None and override is None:
        return result
    for f in fields(result):
        # override 优先
        if override is not None and hasattr(override, f.name):
            ovr_val = getattr(override, f.name)
            if ovr_val is not None:
                setattr(result, f.name, ovr_val)
                continue
        # 其次 base
        if base is not None and hasattr(base, f.name):
            base_val = getattr(base, f.name)
            if base_val is not None:
                setattr(result, f.name, base_val)
    return result


def _dict_to_plot_config(raw: Dict[str, Any]) -> PlotConfig:
    """递归将 YAML 字典转换为 PlotConfig + ChartStyle 层次结构。"""
    config = PlotConfig()

    for chart_key in ["default", "chain_length", "distance", "reaction_stats"]:
        if chart_key in raw:
            chart_dict = raw[chart_key]
            if isinstance(chart_dict, dict):
                chart_style = _dict_to_chart_style(chart_dict)
                setattr(config, chart_key, chart_style)

    return config


def _dict_to_chart_style(d: Dict[str, Any]) -> ChartStyle:
    """将字典转换为 ChartStyle (仅填充 dict 中存在的键)。"""
    cs = ChartStyle()
    sub_mapping = {
        "figure": (FigureStyle, lambda v: FigureStyle(**v) if isinstance(v, dict) else None),
        "font": (FontStyle, lambda v: FontStyle(**v) if isinstance(v, dict) else None),
        "histogram": (HistogramStyle, lambda v: HistogramStyle(**v) if isinstance(v, dict) else None),
        "kde": (KDEStyle, lambda v: KDEStyle(**v) if isinstance(v, dict) else None),
        "theory_line": (LineStyle, lambda v: LineStyle(**v) if isinstance(v, dict) else None),
        "reference": (HistogramStyle, lambda v: HistogramStyle(**v) if isinstance(v, dict) else None),
        "legend": (LegendStyle, lambda v: LegendStyle(**v) if isinstance(v, dict) else None),
        "axis": (AxisText, lambda v: AxisText(**v) if isinstance(v, dict) else None),
        "mean_line": (LineStyle, lambda v: LineStyle(**v) if isinstance(v, dict) else None),
    }
    for key, (_, factory) in sub_mapping.items():
        if key in d and isinstance(d[key], dict):
            val = factory(d[key])
            if val is not None:
                setattr(cs, key, val)
        elif key in d and d[key] is None:
            # YAML 中显式设为 null，保持 None (不填充默认值)
            pass
    return cs


# ======================================================================
# 全局配置实例 & 设置函数
# ======================================================================

_current_config: Optional[PlotConfig] = None


def get_config() -> PlotConfig:
    """获取当前全局配置实例。"""
    global _current_config
    if _current_config is None:
        _current_config = PlotConfig()
    return _current_config


def set_config(config: PlotConfig) -> None:
    """设置全局配置实例。"""
    global _current_config
    _current_config = config


def setup_style(chart_style: Optional[ChartStyle] = None) -> None:
    """设置全局 seaborn/matplotlib 样式。

    参数
    ----
    chart_style : 图表样式 (已 fill_defaults 的 ChartStyle), 为 None 时用全局默认
    """
    if chart_style is None:
        config = get_config()
        chart_style = config.default.fill_defaults()

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)

    # 字体
    if chart_style.font:
        plt.rcParams.update({
            "font.family": chart_style.font.family,
            "font.size": 12,
            "axes.titlesize": chart_style.font.title_size,
            "axes.labelsize": chart_style.font.label_size,
            "xtick.labelsize": chart_style.font.tick_size,
            "ytick.labelsize": chart_style.font.tick_size,
            "legend.fontsize": chart_style.font.legend_size,
        })

    # 画布
    if chart_style.figure:
        plt.rcParams.update({
            "figure.dpi": chart_style.figure.dpi,
            "savefig.dpi": chart_style.figure.dpi,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
        })


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150,
                save_format: Optional[str] = None) -> Path:
    """保存并关闭 figure，返回实际保存路径。

    参数
    ----
    fig : matplotlib Figure
    path : 输出文件路径
    dpi : 分辨率 (会被 save_format 透传覆盖)
    save_format : 输出格式 ("png" | "pdf" | "svg" 等), 为 None 时沿用 path 后缀

    返回
    ----
    实际保存的文件路径 (受 save_format 影响可能与 path 后缀不同)
    """
    path = Path(path)
    if save_format:
        path = path.with_suffix(f".{save_format}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", format=save_format)
    plt.close(fig)
    return path


def new_figure(figsize=(10, 6)):
    """创建新的 figure + axes 对象，返回 (fig, ax)。"""
    return plt.subplots(figsize=figsize)
