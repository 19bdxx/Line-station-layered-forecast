# 输电线路分层预测 · 全流程运行指南

## 一、仓库简介

本项目针对三个风电输电线站（XYA、XYB、XS）进行**多步长功率预测**实验，并对比两种预测策略：

| 策略 | 说明 |
|------|------|
| 直接预测 | 直接预测全站有功功率 |
| 加和预测 | 分别预测各子线路功率后求和 |

同时支持**含限电 / 无限电**两种实验条件，以及 6 种可切换的预测模型：

| MODEL_TYPE | 模型 |
|---|---|
| `lightgbm` | LightGBM 梯度提升树（默认） |
| `xgboost` | XGBoost 梯度提升树 |
| `random_forest` | 随机森林（scikit-learn） |
| `ridge` | 岭回归（scikit-learn，含标准化） |
| `mlp` | 多层感知机（scikit-learn，含标准化 + 早停） |
| `lstm` | LSTM 神经网络（PyTorch，含早停，自动使用 GPU） |

---

## 二、目录结构

```
Line-station-layered-forecast/
├── CODE/
│   ├── #1开始训练.py        ← 步骤1：模型训练与预测
│   ├── #2预测结果汇总.py    ← 步骤2：按预测步长合并结果文件
│   └── #3汇总结果分析.py    ← 步骤3：直接预测 vs 加和预测对比分析
├── requirements.txt
└── README.md
```

脚本运行后会自动生成以下输出目录（无需手动创建）：

```
对比实验/              ← 含限电预测结果（步骤1 输出）
对比实验_无限电/        ← 无限电预测结果（步骤1 输出）
合并结果/              ← 合并后文件（步骤2 输出）
├── with_limit/
│   ├── XYA/
│   ├── XYB/
│   └── XS/
└── no_limit/
    ├── XYA/
    ├── XYB/
    └── XS/
所有实验汇总_metrics_{MODEL_TYPE}.csv   ← 所有实验指标汇总（步骤1 输出）
比较分析结果_全站预测加和_vs_直接预测.csv  ← 对比分析结果（步骤3 输出）
```

---

## 三、环境安装

> 推荐 Python 3.9 ~ 3.11

### 3.1 安装依赖

```bash
pip install -r requirements.txt
```

如需使用 GPU 加速 LSTM，请先参照 [PyTorch 官网](https://pytorch.org/get-started/locally/) 安装匹配 CUDA 版本的 `torch`，再运行上述命令。

### 3.2 验证安装

```bash
python -c "import lightgbm, xgboost, sklearn, torch, pandas, numpy, matplotlib; print('依赖安装成功')"
```

---

## 四、数据格式要求

数据文件为 **CSV 格式**，须包含以下字段（列名须完全一致）：

| 字段名 | 说明 |
|---|---|
| `timestamp` | 时间戳，15 分钟间隔，可被 `parse_dates` 解析 |
| `XYA_ACTIVE_POWER_JIA` | XYA 场站甲线路有功功率 (MW) |
| `XYA_ACTIVE_POWER_YI` | XYA 场站乙线路有功功率 (MW) |
| `XYA_ACTIVE_POWER_STATION` | XYA 全站有功功率 (MW) |
| `XYA_LIMIT_POWER` | XYA 限电功率 (MW) |
| `XYB_ACTIVE_POWER_BING` | XYB 场站丙线路有功功率 (MW) |
| `XYB_ACTIVE_POWER_DING` | XYB 场站丁线路有功功率 (MW) |
| `XYB_ACTIVE_POWER_WU` | XYB 场站戊线路有功功率 (MW) |
| `XYB_ACTIVE_POWER_STATION` | XYB 全站有功功率 (MW) |
| `XYB_LIMIT_POWER` | XYB 限电功率 (MW) |
| `XS_ACTIVE_POWER_JIA` | XS 场站甲线路有功功率 (MW) |
| `XS_ACTIVE_POWER_YI` | XS 场站乙线路有功功率 (MW) |
| `XS_ACTIVE_POWER_STATION` | XS 全站有功功率 (MW) |
| `XS_LIMIT_POWER` | XS 限电功率 (MW) |

---

## 五、配置说明

在运行步骤1之前，打开 `CODE/#1开始训练.py`，按需修改第 **29–32 行**的配置：

```python
M = 32                  # 历史窗口长度（时间步数）
MODEL_TYPE = 'lightgbm' # 选择预测模型，见下方可选值

# Windows 路径示例（使用原始字符串 r"..."）：
DATA_PATH = r"G:\WindPowerForecast\data\all_stations_15min.csv"
# Linux / macOS 路径示例（使用正斜杠）：
# DATA_PATH = "/home/user/data/all_stations_15min.csv"

predict_steps = [i for i in range(1, 96)]  # 预测步长范围：1~95（每步=15分钟，即15分钟~24小时）
```

**`MODEL_TYPE` 可选值：**

```python
MODEL_TYPE = 'lightgbm'      # LightGBM（默认，速度最快）
MODEL_TYPE = 'xgboost'       # XGBoost
MODEL_TYPE = 'random_forest' # 随机森林
MODEL_TYPE = 'ridge'         # 岭回归
MODEL_TYPE = 'mlp'           # 多层感知机
MODEL_TYPE = 'lstm'          # LSTM（需要 PyTorch，支持 GPU）
```

---

## 六、全流程运行指令

> 以下命令均在项目根目录（`Line-station-layered-forecast/`）下执行。
> 三个脚本必须**按顺序依次运行**，后一个脚本依赖前一个脚本的输出。

### 步骤 1：模型训练与预测

```bash
python "CODE/#1开始训练.py"
```

**功能：**
- 对三个场站（XYA、XYB、XS）、每个目标列、1~95 预测步长、含/无限电共计大量组合分别训练模型
- 每组实验结果保存为独立 CSV 文件
- 打印各组的 RMSE、MAE、MAPE、R² 指标
- LightGBM 模型额外保存 RMSE 训练曲线图（.png）

**预期输出目录：**
```
对比实验/XYA/    对比实验/XYB/    对比实验/XS/
对比实验_无限电/XYA/    对比实验_无限电/XYB/    对比实验_无限电/XS/
所有实验汇总_metrics_lightgbm.csv   （或对应 MODEL_TYPE 名称）
```

**切换模型后重新运行示例（以 XGBoost 为例）：**

```bash
# 1. 修改 CODE/#1开始训练.py 第30行：MODEL_TYPE = 'xgboost'
# 2. 再次运行：
python "CODE/#1开始训练.py"
# 输出文件：所有实验汇总_metrics_xgboost.csv
```

---

### 步骤 2：预测结果合并

```bash
python "CODE/#2预测结果汇总.py"
```

**功能：**
- 读取步骤1生成的所有预测结果 CSV 文件
- 按场站 + 预测步长将不同目标列的预测结果横向合并
- 将合并后文件保存至 `合并结果/` 目录

**预期输出目录：**
```
合并结果/
├── with_limit/
│   ├── XYA/XYA_t+1_merged.csv, XYA_t+2_merged.csv, ..., XYA_t+95_merged.csv
│   ├── XYB/XYB_t+1_merged.csv, XYB_t+2_merged.csv, ..., XYB_t+95_merged.csv
│   └── XS/ XS_t+1_merged.csv,  XS_t+2_merged.csv,  ..., XS_t+95_merged.csv
└── no_limit/
    ├── XYA/ （同上，无限电条件）
    ├── XYB/ （同上，无限电条件）
    └── XS/  （同上，无限电条件）
```

---

### 步骤 3：对比分析（直接预测 vs 加和预测）

```bash
python "CODE/#3汇总结果分析.py"
```

**功能：**
- 读取步骤2生成的合并 CSV 文件
- 对每个场站、每个预测步长，计算：
  - **直接预测**全站功率的 RMSE / MAE / R²
  - **子线路加和**预测全站功率的 RMSE / MAE / R²
- 将对比结果汇总为一个 CSV 文件

**预期输出文件：**
```
比较分析结果_全站预测加和_vs_直接预测.csv
```

输出文件字段说明：

| 字段 | 说明 |
|---|---|
| `station` | 场站名称 |
| `step` | 预测步长 |
| `limit_mode` | `with_limit` / `no_limit` |
| `RMSE_sum` | 加和预测的 RMSE |
| `RMSE_direct` | 直接预测的 RMSE |
| `MAE_sum` | 加和预测的 MAE |
| `MAE_direct` | 直接预测的 MAE |
| `R2_sum` | 加和预测的 R² |
| `R2_direct` | 直接预测的 R² |

---

## 七、完整流程一键执行

将以下内容保存为 `run_all.sh`（Linux/macOS）或 `run_all.bat`（Windows），在项目根目录执行：

**Linux / macOS：**

```bash
#!/bin/bash
set -e

echo "========== 步骤1：模型训练与预测 =========="
python "CODE/#1开始训练.py"

echo "========== 步骤2：预测结果合并 =========="
python "CODE/#2预测结果汇总.py"

echo "========== 步骤3：对比分析 =========="
python "CODE/#3汇总结果分析.py"

echo "========== 全流程完成 =========="
```

**Windows CMD：**

```bat
@echo off
echo ========== 步骤1：模型训练与预测 ==========
python "CODE/#1开始训练.py" || exit /b 1

echo ========== 步骤2：预测结果合并 ==========
python "CODE/#2预测结果汇总.py" || exit /b 1

echo ========== 步骤3：对比分析 ==========
python "CODE/#3汇总结果分析.py" || exit /b 1

echo ========== 全流程完成 ==========
```

---

## 八、常见问题

| 问题 | 原因 | 解决方法 |
|---|---|---|
| `FileNotFoundError: DATA_PATH` | 数据路径配置有误 | 修改 `#1开始训练.py` 中的 `DATA_PATH` 为实际绝对路径 |
| `ValueError: 未找到目标字段` | 数据列名与配置不符 | 检查 CSV 列名是否与[第四节](#四数据格式要求)一致 |
| `⚠️ 目录不存在` | 步骤2在步骤1之前运行 | 先运行步骤1，再运行步骤2 |
| `❌ 没有成功收集任何结果` | 步骤3在步骤2之前运行 | 先完成步骤1和步骤2，再运行步骤3 |
| LSTM 训练慢 | 无 GPU 或 CUDA 未配置 | 参照 PyTorch 官网安装 GPU 版本，或切换为其他模型 |
