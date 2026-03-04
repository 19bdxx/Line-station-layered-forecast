import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation, record_evaluation
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb


# -------------------- 代码说明 --------------------
# 本脚本在与 #1开始训练.py 完全相同的数据流程和评估体系下，
# 支持同时运行多种预测模型，以验证"分层预测优于直接预测"这一结论
# 是否在不同算法下具有普遍性。
#
# 支持模型：
#   lgb   - LightGBM（与脚本#1保持一致，作为基线）
#   rf    - 随机森林 (Random Forest)
#   xgb   - XGBoost
#   ridge - 岭回归 (Ridge Regression)
#   et    - 极端随机树 (Extra Trees)
#
# 运行方式：
#   - 修改下方 MODEL_LIST 以指定需要运行的模型子集
#   - 结果保存在 多模型实验/{model_type}/ 目录下，格式与脚本#1完全兼容
#   - 运行完成后，执行 #5多模型结果分析.py 进行跨模型对比


# -------------------- 配置 --------------------
M = 4 * 60  # 历史窗口长度（分钟），与脚本#1保持一致
DATA_PATH = r"G:\WindPowerForecast\集群_场站预测\data\all_stations_输电线未发电置零_1min.csv"
predict_steps = [i * 15 for i in range(1, 17)]  # 16个预测步长：15~240分钟

# 指定要运行的模型列表（可按需注释掉不需要的模型）
MODEL_LIST = ['lgb', 'rf', 'xgb', 'ridge', 'et']

stations = {
    "XYA": {
        "target_cols": ['XYA_ACTIVE_POWER_JIA', 'XYA_ACTIVE_POWER_YI', 'XYA_ACTIVE_POWER_STATION'],
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


# -------------------- 模型超参配置 --------------------
# 各模型默认超参，可按需调整
MODEL_PARAMS = {
    'lgb': {
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
    },
    'rf': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 20,
        'n_jobs': -1,
        'random_state': 42
    },
    'xgb': {
        'n_estimators': 500,
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1
    },
    'ridge': {
        'alpha': 1.0
    },
    'et': {
        'n_estimators': 200,
        'max_depth': 10,
        'min_samples_leaf': 20,
        'n_jobs': -1,
        'random_state': 42
    }
}


# -------------------- 模型构建 --------------------
def build_sklearn_model(model_type):
    """根据模型类型构建 sklearn 兼容的模型实例。"""
    p = MODEL_PARAMS[model_type]
    if model_type == 'rf':
        return RandomForestRegressor(**p)
    elif model_type == 'et':
        return ExtraTreesRegressor(**p)
    elif model_type == 'ridge':
        return Ridge(**p)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# -------------------- 计算偏差率 --------------------
def calculate_bias_rate(y_true, y_pred, P_capacity):
    bias_rate = np.zeros_like(y_true, dtype=float)
    condition = y_true >= 0.2 * P_capacity
    bias_rate[condition] = np.abs(y_pred[condition] - y_true[condition]) / y_true[condition]
    bias_rate[~condition] = np.abs(y_pred[~condition] - y_true[~condition]) / (0.2 * P_capacity)
    return bias_rate


# -------------------- 主实验函数 --------------------
def run_experiment(df, station_name, target_col, predict_step, limit_col, include_limit, save_dir, model_type):
    """
    对单个（场站, 目标列, 预测步长, 模型）组合完成特征构建、模型训练和结果保存。

    参数
    ----
    df           : 完整时序数据 DataFrame，含 timestamp 列
    station_name : 场站名称（XYA / XYB / XS）
    target_col   : 预测目标列名
    predict_step : 预测步长（分钟）
    limit_col    : 限电列名
    include_limit: 是否将限电特征纳入输入
    save_dir     : 预测结果 CSV 的保存目录
    model_type   : 模型类型字符串（lgb / rf / xgb / ridge / et）
    """
    if target_col not in df.columns:
        raise ValueError(f"未找到目标字段：{target_col}")

    P_capacity = stations[station_name]['P_capacity']
    hist_features = [target_col] + ([limit_col] if include_limit else [])

    X_list, y_list, ts_list = [], [], []
    for i in range(M, len(df) - predict_step):
        x_hist = df[hist_features].iloc[i - M:i + 1].values.flatten()
        X_list.append(x_hist)
        y_list.append(df.loc[i + predict_step, target_col])
        ts_list.append(df.loc[i + predict_step, 'timestamp'])

    X = np.array(X_list)
    y = np.array(y_list)
    ts = np.array(ts_list)

    # 划分训练集（80%）与测试集（20%）
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    ts_test = ts[split_idx:]

    # 训练集内再划分 80% 训练子集 + 20% 验证子集（供 lgb/xgb 早停使用）
    split_idx_train = int(len(X_train) * 0.8)
    X_tr, X_val = X_train[:split_idx_train], X_train[split_idx_train:]
    y_tr, y_val = y_train[:split_idx_train], y_train[split_idx_train:]

    # 根据模型类型选择训练方式
    if model_type == 'lgb':
        train_data = lgb.Dataset(X_tr, label=y_tr)
        valid_data = lgb.Dataset(X_val, label=y_val)
        evals_result = {}
        model = lgb.train(
            MODEL_PARAMS['lgb'],
            train_data,
            num_boost_round=2000,
            valid_sets=[train_data, valid_data],
            valid_names=['train', 'valid'],
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(100),
                record_evaluation(evals_result)
            ]
        )
        y_pred = model.predict(X_test, num_iteration=model.best_iteration)

    elif model_type == 'xgb':
        p = MODEL_PARAMS['xgb'].copy()
        model = xgb.XGBRegressor(**p)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        y_pred = model.predict(X_test)

    else:
        # rf / ridge / et：使用全部训练集（无需验证子集早停）
        model = build_sklearn_model(model_type)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    # 计算评估指标
    bias_rate = calculate_bias_rate(y_test, y_pred, P_capacity)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    r2 = r2_score(y_test, y_pred)

    # 保存预测结果 CSV（格式与脚本#1完全一致）
    results_df = pd.DataFrame({
        'timestamp': ts_test,
        'y_true': y_test,
        'y_pred': y_pred,
        'bias_rate': bias_rate,
        'limit_value': [df.loc[i + predict_step, limit_col] for i in range(len(y_test))]
    })
    file_prefix = f"{target_col}_t+{predict_step}"
    results_df.to_csv(os.path.join(save_dir, f"{file_prefix}.csv"), index=False)

    print(f"\n✅ [{model_type}] {station_name} {'含限电' if include_limit else '无限电'} | {file_prefix}")
    print(f"   RMSE : {rmse:.2f}  MAE : {mae:.2f}  MAPE : {mape:.2f}%  R² : {r2:.4f}")

    return {
        'model': model_type,
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

for model_type in MODEL_LIST:
    print(f"\n{'='*60}")
    print(f"🚀 开始训练模型：{model_type.upper()}")
    print(f"{'='*60}")

    for station_name, config in stations.items():
        for include_limit in [True, False]:
            sub_dir = '对比实验' if include_limit else '对比实验_无限电'
            save_dir = os.path.join('多模型实验', model_type, sub_dir, station_name)
            os.makedirs(save_dir, exist_ok=True)

            for target_col in config['target_cols']:
                for step in predict_steps:
                    try:
                        result = run_experiment(
                            df_all, station_name, target_col, step,
                            config['limit_col'], include_limit, save_dir, model_type
                        )
                        all_results.append(result)
                    except Exception as e:
                        print(f"❌ 错误 [{model_type}] {station_name} {target_col} t+{step}: {e}")

summary_df = pd.DataFrame(all_results)
summary_df.to_csv('多模型实验_汇总metrics.csv', index=False)
print("\n📊 所有模型训练完成，汇总结果已保存为 多模型实验_汇总metrics.csv")
