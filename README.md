# 风电场线路-场站分层预测系统

## 项目简介

本项目实现了一套风电场功率**分层预测**框架，通过 LightGBM 模型对风电场的线路级（子场站）和全站级功率进行多步长预测，并对比**分层聚合预测**（各线路预测值求和）与**直接预测**（全站直接预测）两种策略的精度差异。

数据来源：三个风电场（XYA、XYB、XS）的 1 分钟分辨率历史功率及限电数据。

---

## 目录结构

```
├── CODE/
│   ├── #1开始训练.py        # 步骤一：模型训练与预测结果输出
│   ├── #2预测结果汇总.py    # 步骤二：按场站和预测步长合并结果文件
│   └── #3汇总结果分析.py    # 步骤三：对比分析分层聚合预测 vs 直接预测
└── RAW_DATA/
    └── all_stations.csv     # 原始风电场数据（含各场站功率及限电信息）
```

---

## 脚本功能详解

### `#1开始训练.py` — 模型训练与预测

#### 功能概述
针对三个风电场（XYA、XYB、XS）的各线路和全站目标列，在 16 个预测步长（15、30、…、240 分钟）下分别训练 LightGBM 回归模型，输出预测结果和训练曲线图。

#### 关键配置
| 参数 | 值 | 说明 |
|------|------|------|
| `M` | 240（分钟） | 历史窗口长度（4 小时） |
| `predict_steps` | [15, 30, …, 240] | 16 个预测步长（单位：分钟） |
| 含/不含限电 | `include_limit` | 对每个场站分别做含限电和不含限电两组实验 |

#### 场站配置

| 场站 | 目标列 | 装机容量 |
|------|--------|----------|
| XYA | 甲线路、乙线路、全站 | 400 MW |
| XYB | 丙线路、丁线路、戊线路、全站 | 900 MW |
| XS | 甲线路、乙线路、全站 | 400 MW |

#### 数据处理流程
1. 读取 CSV 原始数据（1 分钟分辨率，含 `timestamp` 列）。
2. 以滑动窗口（长度 `M+1`）构造特征：将历史功率序列（可选拼接限电序列）展平为特征向量。
3. 按 8:2 划分训练集与测试集；训练集再按 8:2 划分训练子集与验证子集，用于早停。

#### 模型训练
- 算法：LightGBM 回归（`objective: regression`，`metric: rmse`）
- 超参：`learning_rate=0.01`，`num_leaves=31`，`max_depth=6`，`min_data_in_leaf=50`，特征/样本采样比 0.8
- 迭代上限：2000 轮，早停轮数：100（基于验证集 RMSE）

#### 偏差率计算
$$
\text{bias\_rate}_i = \begin{cases}
\dfrac{|y_{\text{pred},i} - y_{\text{true},i}|}{y_{\text{true},i}} & \text{若 } y_{\text{true},i} \geq 0.2 \times P_{\text{capacity}} \\[6pt]
\dfrac{|y_{\text{pred},i} - y_{\text{true},i}|}{0.2 \times P_{\text{capacity}}} & \text{否则}
\end{cases}
$$

#### 输出
- 预测结果 CSV：`对比实验/{场站}/{目标列}_t+{步长}.csv`（含：`timestamp`、`y_true`、`y_pred`、`bias_rate`、`limit_value`）
- RMSE 曲线图（PNG）：保存于同目录
- 汇总指标文件：`所有实验汇总_metrics.csv`（含 RMSE、MAE、MAPE、R²）

---

### `#2预测结果汇总.py` — 结果文件合并

#### 功能概述
将步骤一输出的、按（目标列 × 预测步长）组织的碎片化 CSV 文件，**按场站 × 预测步长**横向合并，便于后续对比分析。

#### 处理逻辑
1. 遍历 `对比实验/` 和 `对比实验_无限电/` 两个目录。
2. 对每个场站，按文件名（`{目标列}_t+{步长}.csv`）解析目标列简称和步长。
3. 对各目标列的结果列重命名（如 `y_true` → `JIA_true`），以 `timestamp` 为键外连接合并。
4. 合并结果按时间戳排序后保存。

#### 输出
```
合并结果/
├── with_limit/
│   ├── XYA/XYA_t+{步长}_merged.csv
│   ├── XYB/XYB_t+{步长}_merged.csv
│   └── XS/XS_t+{步长}_merged.csv
└── no_limit/
    └── ...（结构相同）
```

每个合并文件包含 `timestamp` 及该步长下所有目标列的 `{名称}_true`、`{名称}_pred`、`{名称}_bias_rate`、`{名称}_limit_value` 列。

---

### `#3汇总结果分析.py` — 分层对比分析

#### 功能概述
对每个场站在每个预测步长下，比较两种全站功率预测策略：
- **直接预测**：直接使用全站预测值（`STATION_pred`）
- **分层聚合预测**（加和）：将各线路预测值求和（`sum_pred`）

#### 对比指标
| 指标 | 说明 |
|------|------|
| RMSE | 均方根误差（越小越好） |
| MAE | 平均绝对误差（越小越好） |
| R² | 决定系数（越接近 1 越好） |

#### 输出
汇总对比结果保存为：`比较分析结果_全站预测加和_vs_直接预测.csv`

字段包括：`station`、`step`、`limit_mode`、`RMSE_sum`、`RMSE_direct`、`MAE_sum`、`MAE_direct`、`R2_sum`、`R2_direct`。

---

## 运行顺序

```bash
# 1. 训练模型并生成预测结果
python "CODE/#1开始训练.py"

# 2. 合并各场站预测结果文件
python "CODE/#2预测结果汇总.py"

# 3. 对比分层聚合预测与直接预测的精度
python "CODE/#3汇总结果分析.py"
```

> **注意**：运行前请修改 `#1开始训练.py` 中的 `DATA_PATH` 为实际数据文件路径。

---

## 依赖环境

```
pandas
numpy
lightgbm
scikit-learn
matplotlib
```

安装：
```bash
pip install pandas numpy lightgbm scikit-learn matplotlib
```
