import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)


# -------------------- 代码说明 --------------------
# 本代码旨在通过使用 LightGBM 模型预测风电场的功率，并计算多个评估指标（RMSE, MAE, MAPE, R² 和偏差率）。
# 具体步骤如下：
# 1. 加载数据并进行预处理；
# 2. 针对不同的风电场、目标列和预测步长进行多次实验；
# 3. 训练 LightGBM 模型，计算预测值；
# 4. 计算评估指标，包括 RMSE, MAE, MAPE, R² 和偏差率；
# 5. 将每次实验的结果保存为 CSV 文件，并输出到控制台。

# -------------------- 配置 --------------------
M = 4 * 60
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'RAW_DATA', 'all_stations.csv')
predict_steps = [i*15 for i in range(1, 17)]

# 优化选项
USE_STAT_FEATURES = True   # True: 使用统计摘要特征+时间特征（维度低，训练快）; False: 使用原始滑动窗口展平
SAVE_MODEL = True          # True: 将训练好的模型保存为 .txt 文件

stations = {
    "XYA": {
        "target_cols": ['XYA_ACTIVE_POWER_JIA','XYA_ACTIVE_POWER_YI','XYA_ACTIVE_POWER_STATION'],
        "limit_col": 'XYA_LIMIT_POWER',
        "P_capacity": 400
    },
    "XYB": {
        "target_cols": ['XYB_ACTIVE_POWER_BING', 'XYB_ACTIVE_POWER_DING', 'XYB_ACTIVE_POWER_WU', 'XYB_ACTIVE_POWER_STATION'],
        "limit_col": 'XYB_LIMIT_POWER',
        "P_capacity": 900
    },
    "XS": {
        "target_cols": ['XS_ACTIVE_POWER_JIA', 'XS_ACTIVE_POWER_YI', 'XS_ACTIVE_POWER_STATION'],
        "limit_col": 'XS_LIMIT_POWER',
        "P_capacity": 400
    }
}

# -------------------- 计算偏差率 --------------------
def calculate_bias_rate(y_true, y_pred, P_capacity):
    bias_rate = np.zeros_like(y_true)
    condition = y_true >= 0.2 * P_capacity
    bias_rate[condition] = abs(y_pred[condition] - y_true[condition]) / y_true[condition]
    bias_rate[~condition] = abs(y_pred[~condition] - y_true[~condition]) / (0.2 * P_capacity)
    return bias_rate

# -------------------- 绘图函数 --------------------
def plot_rmse_curve(evals_result, station_name, target_col, predict_step, save_dir):
    train_rmse = evals_result['train']['rmse']
    valid_rmse = evals_result['valid']['rmse']
    plt.figure(figsize=(8, 5))
    plt.plot(train_rmse, label='Train RMSE')
    plt.plot(valid_rmse, label='Valid RMSE')
    plt.xlabel('Iteration')
    plt.ylabel('RMSE')
    plt.title(f'{station_name} | {target_col} | t+{predict_step} RMSE Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    # 保存图像
    plot_path = os.path.join(save_dir, f"{target_col}_t+{predict_step}_RMSE_curve.png")
    plt.savefig(plot_path)
    plt.close()

# -------------------- 统计特征提取 --------------------
def extract_features(df, hist_features, i, M):
    """
    提取统计摘要特征 + 时间特征，替代原始滑动窗口展平。
    在多个时间窗口（15/30/60/120/M分钟）内分别计算均值、标准差、最小值、最大值，
    并附加当前时刻值、近15步趋势斜率及时间特征（小时、分钟、星期几）。
    相比原始展平特征（241×N维），维度大幅降低（约25~50维），训练速度更快。
    """
    feats = []
    windows = [15, 30, 60, 120, M]
    for w in windows:
        start = max(i - w, i - M)
        seg = df[hist_features].iloc[start:i + 1]
        feats.extend(seg.mean().tolist())
        feats.extend(seg.std().fillna(0).tolist())
        feats.extend(seg.min().tolist())
        feats.extend(seg.max().tolist())
    # 当前时刻值
    feats.extend(df[hist_features].iloc[i].tolist())
    # 近15步线性趋势斜率
    for col in hist_features:
        recent = df[col].iloc[max(i - 14, 0):i + 1].values
        slope = np.polyfit(range(len(recent)), recent, 1)[0] if len(recent) > 1 else 0.0
        feats.append(slope)
    # 时间特征
    ts = df['timestamp'].iloc[i]
    feats += [ts.hour, ts.minute, ts.dayofweek]
    return feats

# -------------------- 主实验函数 --------------------
def run_experiment(df, station_name, target_col, predict_step, limit_col, include_limit, save_dir):
    if target_col not in df.columns:
        raise ValueError(f"未找到目标字段：{target_col}")
    hist_features = [target_col]
    if include_limit:
        hist_features.append(limit_col)

    X_list, y_list, ts_list, lv_list = [], [], [], []

    total_rows = len(df) - predict_step - M
    print_every = max(1, total_rows // 10)   # 每完成 10% 打印一次
    print(f"    提取特征中... 共 {total_rows} 行", flush=True)
    feat_t0 = time.time()

    for idx, i in enumerate(range(M, len(df) - predict_step)):
        if USE_STAT_FEATURES:
            x_input = extract_features(df, hist_features, i, M)
        else:
            x_input = df[hist_features].iloc[i - M:i + 1].values.flatten()
        y_target = df.iloc[i + predict_step][target_col]
        ts_target = df.iloc[i + predict_step]['timestamp']
        lv_target = df.iloc[i + predict_step][limit_col]  # 修复：在循环中同步收集限电值
        P_capacity = stations[station_name]['P_capacity']

        X_list.append(x_input)
        y_list.append(y_target)
        ts_list.append(ts_target)
        lv_list.append(lv_target)

        if (idx + 1) % print_every == 0 or (idx + 1) == total_rows:
            pct = (idx + 1) / total_rows * 100
            elapsed = time.time() - feat_t0
            eta = elapsed / (idx + 1) * (total_rows - idx - 1)
            print(f"      {pct:5.1f}%  ({idx+1}/{total_rows})  已用 {elapsed:.0f}s  预计剩余 {eta:.0f}s", flush=True)

    print(f"    特征提取完成，耗时 {time.time() - feat_t0:.1f}s", flush=True)

    X = np.array(X_list)
    y = np.array(y_list)
    ts = np.array(ts_list)
    lv = np.array(lv_list)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test = ts[split_idx:]
    lv_test = lv[split_idx:]  # 修复：使用与测试集对齐的限电值

    split_idx_train = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx_train], X_train[split_idx_train:]
    y_tr, y_val = y_train[:split_idx_train], y_train[split_idx_train:]

    train_data = lgb.Dataset(X_tr, label=y_tr)
    valid_data = lgb.Dataset(X_val, label=y_val)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 6,
        'min_data_in_leaf': 50,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'verbose': -1
    }

    print(f"    训练模型中... 训练集 {len(X_tr)} 行，验证集 {len(X_val)} 行", flush=True)
    train_t0 = time.time()
    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=2000,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=[
            early_stopping(stopping_rounds=100),
            log_evaluation(50),
            record_evaluation(evals_result)
        ]
    )
    print(f"    模型训练完成，耗时 {time.time() - train_t0:.1f}s  最佳迭代: {model.best_iteration}", flush=True)

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    bias_rate = calculate_bias_rate(y_test, y_pred, P_capacity)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # 修复：避免使用已弃用的 squared=False
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)

    results_df = pd.DataFrame({
        'timestamp': ts_test,
        'y_true': y_test,
        'y_pred': y_pred,
        'bias_rate': bias_rate,
        'limit_value': lv_test  # 修复：使用正确对齐的测试集限电值
    })

    file_prefix = f"{target_col}_t+{predict_step}"
    results_df.to_csv(os.path.join(save_dir, f"{file_prefix}.csv"), index=False)

    # 保存RMSE曲线图
    plot_rmse_curve(evals_result, station_name, target_col, predict_step, save_dir)

    # 保存训练好的模型
    if SAVE_MODEL:
        model.save_model(os.path.join(save_dir, f"{file_prefix}_model.txt"))

    # 保存特征重要性
    if USE_STAT_FEATURES:
        importance = model.feature_importance(importance_type='gain')
        imp_path = os.path.join(save_dir, f"{file_prefix}_feature_importance.csv")
        pd.DataFrame({'importance': importance}).to_csv(imp_path, index=False)

    print(f"\n✅ {station_name} {'含限电' if include_limit else '无限电'} | {file_prefix}")
    print(f"RMSE : {rmse:.2f}")
    print(f"MAE  : {mae:.2f}")
    print(f"MAPE : {mape:.2f}%")
    print(f"R²   : {r2:.4f}")

    return {
        'station': station_name,
        'limit_mode': 'with_limit' if include_limit else 'no_limit',
        'target': target_col,
        'step': predict_step,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }

# -------------------- 执行所有实验 --------------------
df_all = pd.read_csv(DATA_PATH, parse_dates=['timestamp'])
all_results = []

# 计算总实验数
total_experiments = sum(
    len(config['target_cols']) * len(predict_steps)
    for config in stations.values()
) * 2  # ×2: with_limit + no_limit
exp_no = 0

print(f"\n{'='*60}")
print(f"  共 {total_experiments} 个实验待运行")
print(f"  场站: {list(stations.keys())}")
print(f"  预测步长: {predict_steps}")
print(f"{'='*60}\n")
pipeline_t0 = time.time()

for station_name, config in stations.items():
    for include_limit in [True, False]:
        sub_dir = '对比实验' if include_limit else '对比实验_无限电'
        save_dir = os.path.join(sub_dir, station_name)
        os.makedirs(save_dir, exist_ok=True)

        for target_col in config['target_cols']:
            for step in predict_steps:
                exp_no += 1
                mode_label = '含限电' if include_limit else '无限电'
                print(f"\n[{exp_no}/{total_experiments}] 开始: {station_name} | {target_col} | t+{step}min | {mode_label}", flush=True)
                exp_t0 = time.time()
                result = run_experiment(
                    df_all, station_name, target_col, step,
                    config['limit_col'], include_limit, save_dir
                )
                print(f"[{exp_no}/{total_experiments}] 完成，本次耗时 {time.time() - exp_t0:.1f}s  累计耗时 {time.time() - pipeline_t0:.0f}s", flush=True)
                all_results.append(result)

summary_df = pd.DataFrame(all_results)
summary_df.to_csv('所有实验汇总_metrics.csv', index=False)
print(f"\n{'='*60}")
print(f"  所有 {total_experiments} 个实验完成！总耗时 {time.time() - pipeline_t0:.0f}s")
print(f"  汇总结果已保存为 所有实验汇总_metrics.csv")
print(f"{'='*60}")