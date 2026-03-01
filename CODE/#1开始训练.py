import pandas as pd
import numpy as np
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt  # ✅ 新增：用于绘图
import os
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message=".*squared.*deprecated.*")


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
DATA_PATH = r"G:\WindPowerForecast\集群_场站预测\data\all_stations_输电线未发电置零_1min.csv"
predict_steps = [i*15 for i in range(1, 17)]

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

# -------------------- 主实验函数 --------------------
def run_experiment(df, station_name, target_col, predict_step, limit_col, include_limit, save_dir):
    if target_col not in df.columns:
        raise ValueError(f"未找到目标字段：{target_col}")
    hist_features = [target_col]
    if include_limit:
        hist_features.append(limit_col)

    X_list, y_list, ts_list = [], [], []

    for i in range(M, len(df) - predict_step):
        x_hist = df[hist_features].iloc[i - M:i + 1].values.flatten()
        x_input = x_hist
        y_target = df.loc[i + predict_step, target_col]
        ts_target = df.loc[i + predict_step, 'timestamp']
        P_capacity = stations[station_name]['P_capacity']

        X_list.append(x_input)
        y_list.append(y_target)
        ts_list.append(ts_target)

    X = np.array(X_list)
    y = np.array(y_list)
    ts = np.array(ts_list)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test = ts[split_idx:]

    split_idx_train = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx_train], X_train[split_idx_train:]
    y_tr, y_val = y_train[:split_idx_train], y_train[split_idx_train:]
    ts_tr, ts_val = ts[:split_idx_train], ts[split_idx_train:]

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

    y_pred = model.predict(X_test, num_iteration=model.best_iteration)
    bias_rate = calculate_bias_rate(y_test, y_pred, P_capacity)

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)

    results_df = pd.DataFrame({
        'timestamp': ts_test,
        'y_true': y_test,
        'y_pred': y_pred,
        'bias_rate': bias_rate,
        'limit_value': [df.loc[i + predict_step, limit_col] for i in range(len(y_test))]
    })

    file_prefix = f"{target_col}_t+{predict_step}"
    results_df.to_csv(os.path.join(save_dir, f"{file_prefix}.csv"), index=False)

    # ✅ 新增：保存RMSE曲线图
    plot_rmse_curve(evals_result, station_name, target_col, predict_step, save_dir)

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

for station_name, config in stations.items():
    for include_limit in [True, False]:
        sub_dir = '对比实验' if include_limit else '对比实验_无限电'
        save_dir = os.path.join(sub_dir, station_name)
        os.makedirs(save_dir, exist_ok=True)

        for target_col in config['target_cols']:
            for step in predict_steps:
                result = run_experiment(
                    df_all, station_name, target_col, step,
                    config['limit_col'], include_limit, save_dir
                )
                all_results.append(result)

summary_df = pd.DataFrame(all_results)
summary_df.to_csv('所有实验汇总_metrics.csv', index=False)
print("\n📊 所有实验完成，汇总结果已保存为 所有实验汇总_metrics.csv")