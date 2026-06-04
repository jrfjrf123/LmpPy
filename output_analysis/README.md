# LmpPy Output Analysis

LmpPy 粗粒化反应模拟后处理分析工具。基于 LAMMPS bond/react 模拟输出，生成链长分布、反应距离分布、反应统计三种分析图表。

## 快速开始

```bash
# 运行全部三种分析
python -m LmpPy.output_analysis all ./process1/

# 指定绘图配置文件
python -m LmpPy.output_analysis all ./process1/ --plot-config plot_config.yaml

# 单独运行某个分析
python -m LmpPy.output_analysis chain-length ./process1/
python -m LmpPy.output_analysis distance ./process1/
python -m LmpPy.output_analysis reaction-stats ./process1/

# 从 reaction_frames.npz 预先重建 reaction_details.csv
python -m LmpPy.output_analysis rebuild ./process1/reaction_frames.npz -o ./process1/reaction_details.csv --config-dir ./process1/
```

## 输入文件

模块需要模拟输出目录中包含以下文件：

| 文件 | 用途 | 必需？ |
|------|------|:---:|
| `final_frame.data` | 链长分布分析（LAMMPS data 格式） | chain-length |
| `reaction_frames.npz` | 反应事件数据（键变化、坐标、CG映射） | distance, reaction-stats |
| `reaction_details.csv` | 反应事件表格（若已存在则跳过 npz 重建） | distance, reaction-stats |

如果 `reaction_details.csv` 不存在，模块会从 `reaction_frames.npz` 自动重建。

## 三种分析

### 1. 链长分布 (`chain-length`)

- 从 `final_frame.data` 解析键拓扑，通过 BFS 识别连通分子
- 计算每个分子的链长（珠子数）
- 绘制直方图 + KDE 曲线
- 可选：叠加 Schulz-Zimm / Poisson / 对数正态理论曲线
- 可选：计算 JS 散度

```bash
python -m LmpPy.output_analysis chain-length ./process1/ --min-length 10 --theory-type lognormal --js
```

### 2. 反应距离分布 (`distance`)

- 绘制每个反应类型的反应距离直方图
- 可选：叠加参考分布并计算 KS 检验
- 标注均值和标准差

```bash
python -m LmpPy.output_analysis distance ./process1/ --ref-csv reference.csv
```

### 3. 反应统计 (`reaction-stats`)

- 饼图：各反应类型占比
- 堆叠柱状图：每循环的反应数量按类型堆叠
- 输出 `reaction_counts.csv`（每循环各类型计数）

```bash
python -m LmpPy.output_analysis reaction-stats ./process1/
```

## 绘图配置

通过 `plot_config.yaml` 控制所有图表样式，支持三层优先级：

1. **CLI 参数**（最高）：如 `--bins 100`
2. **YAML 配置文件**（中间）：`--plot-config plot_config.yaml`
3. **dataclass 默认值**（最低）：代码内置默认值

```yaml
# plot_config.yaml 示例
default:
  figure:
    dpi: 300
    figsize: [10, 6]
    save_format: "png"   # png | pdf | svg
  font:
    family: "DejaVu Sans"
    title_size: 14
  histogram:
    alpha: 0.4
    bins: 50
    color: "#4C72B0"
  # ... 更多配置见模板文件

chain_length:
  figure:
    figsize: [8, 6]
  axis:
    title: "链长分布"

distance:
  figure:
    figsize: [6, 4]

reaction_stats:
  figure:
    figsize: [14, 6]
```

完整模板见 `plot_config.yaml`。

## 性能说明

`reaction_frames.npz` 文件通常很大（5-100 GB），重建 `reaction_details.csv` 时采用以下策略：

- **两阶段处理**：先预计算所有键差（内存），再流式读取大数组
- **临时提取到 NVMe**：将大数组临时提取到 `/tmp`，利用 NVMe 高速顺序读取
- **自动清理**：处理完成后立即删除临时文件，启动时也会清理上次异常退出遗留的临时目录
- **进度条**：所有耗时步骤均有 tqdm 进度条

以 process1 为例（96 GB npz，1879 帧，309k 原子/帧），重建耗时约 90 秒。

## 输出文件

| 文件 | 内容 |
|------|------|
| `chain_length_distribution.png` | 链长分布图 |
| `distance_distribution.png` | 反应距离分布图 |
| `reaction_stats.png` | 反应统计图（饼图+堆叠柱状图） |
| `reaction_details.csv` | 反应事件明细表（约 15 MB） |
| `reaction_counts.csv` | 每循环各反应类型计数 |

## reaction_details.csv 列说明

| 列名 | 说明 |
|------|------|
| `cycle` | 循环编号 |
| `atom1_id` / `atom2_id` | 反应原子 ID |
| `atom1_type_before` / `atom2_type_before` | 反应前原子类型（= 珠子类型） |
| `atom1_type_after` / `atom2_type_after` | 反应后原子类型 |
| `distance` | 反应距离 (Å) |
| `reaction_type` | 反应名称（从模板匹配） |
| `pair_type` | 珠子类型对（如 `2-3`） |
| `n_angles` / `n_dihedrals` | 新增角/二面角数（LmpPy 填 0） |
